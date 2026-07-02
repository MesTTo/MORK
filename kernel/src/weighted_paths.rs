use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::Mutex;

use mork_expr::{Tag, maybe_byte_item};
use pathmap::PathMap;
use pathmap::morphisms::Catamorphism;
use pathmap::zipper::{Zipper, ZipperAbsolutePath, ZipperIteration, ZipperValues, ZipperWriting};

/// Derived weighted index over encoded MORK paths.
///
/// This keeps weights outside the authoritative `PathMap<()>` atom store. It is
/// intended as the safe version of the `ws` experiment from the iCog fork: a
/// future sink can maintain this sidecar without changing byte-path semantics.
#[derive(Clone, Debug, Default)]
pub struct WeightedPathIndex {
    weights: PathMap<i64>,
    total_positive_weight: u64,
    updates: usize,
}

/// Read-only counters for a [`WeightedPathIndex`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WeightedPathStats {
    /// Number of retained non-zero weighted paths.
    pub entries: usize,
    /// Number of retained paths with positive sampling weight.
    pub positive_entries: usize,
    /// Number of retained paths with zero-or-negative signed weight.
    pub non_positive_entries: usize,
    /// Sum of positive weights visible to weighted selection.
    pub total_positive_weight: u64,
    /// Number of explicit set/delta operations applied to this sidecar.
    pub updates: usize,
}

/// Errors from weighted sidecar maintenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightedPathError {
    /// Signed path weight arithmetic overflowed while applying a delta.
    WeightOverflow { current: i64, delta: i64 },
    /// Positive sampling-weight aggregation overflowed.
    TotalPositiveWeightOverflow { left: u64, right: u64 },
    /// Positive sampling-weight aggregation underflowed, which indicates a
    /// broken sidecar invariant.
    TotalPositiveWeightUnderflow { current: u64, decrement: u64 },
}

/// Aggregate positive-weight snapshot for structural descent.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WeightedSelectionTree {
    total_positive_weight: u64,
    nodes: BTreeMap<Vec<u8>, WeightedSelectionNode>,
}

/// Read-only counters for a [`WeightedSelectionTree`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WeightedSelectionTreeStats {
    /// Positive-total structural trie positions retained in the aggregate snapshot.
    pub nodes: usize,
    /// Positive-total child edges retained across all aggregate nodes.
    pub child_edges: usize,
    /// Nodes with a positive value at the exact node path.
    pub positive_value_nodes: usize,
    /// Sum of positive weights visible to weighted selection.
    pub total_positive_weight: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct WeightedSelectionNode {
    self_weight: u64,
    total_weight: u64,
    children: Box<[(u8, u64)]>,
}

impl WeightedPathIndex {
    /// Creates an empty weighted sidecar.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a compression-gain weight index over an atom store `PathMap<()>`.
    ///
    /// For every prefix shared by `count >= 2` stored atoms, the weight is the bytes
    /// saved by factoring that prefix into one definition plus `count` references:
    /// `(count - 1) * len - count * ref_cost`. Only positive gains are kept, so
    /// [`iter_any_topk`](Self::iter_any_topk) / [`iter_prefix_topk`](Self::iter_prefix_topk)
    /// then surface the heaviest compressible subpatterns (WILLIAM's "compression-gain
    /// sums" over the trie, whitepaper 5.12).
    ///
    /// A single bottom-up catamorphism computes each subtrie's occurrence count, so the
    /// whole index is built in one pass over the store.
    pub fn from_compression_gain(atoms: &PathMap<()>, ref_cost: u64) -> Self {
        Self::build_compression_gain(atoms, ref_cost, false)
    }

    /// Boundary-restricted [`from_compression_gain`](Self::from_compression_gain): a
    /// prefix is weighted only when its byte offset falls on a MORK term boundary (the
    /// start of an item, never inside a symbol's raw payload). Every retained pattern is
    /// then a whole MeTTa subexpression spine, so [`decode_pattern`] renders it and no
    /// candidate is a mid-symbol byte cut. This is the whitepaper's intent: factor
    /// subexpressions, not arbitrary byte prefixes.
    pub fn from_compression_gain_on_boundaries(atoms: &PathMap<()>, ref_cost: u64) -> Self {
        Self::build_compression_gain(atoms, ref_cost, true)
    }

    fn build_compression_gain(atoms: &PathMap<()>, ref_cost: u64, boundaries_only: bool) -> Self {
        let mut index = Self::new();
        atoms.read_zipper().into_cata_side_effect(
            |_mask, children: &mut [usize], value: Option<&()>, path: &[u8]| -> usize {
                let count = value.is_some() as usize + children.iter().copied().sum::<usize>();
                let admit = count >= 2
                    && !path.is_empty()
                    && (!boundaries_only || path_ends_on_term_boundary(path));
                if admit {
                    let gain = boundary_gain(count as i64, path.len(), ref_cost);
                    if gain > 0 {
                        // Overflow only at extreme scales; a dropped weight just omits one
                        // compressible prefix from the derived index, never corrupts the store.
                        let _ = index.set_weight(path, gain);
                    }
                }
                count
            },
        );
        index
    }

    /// Returns the signed weight stored for `path`, or zero when absent.
    pub fn weight(&self, path: &[u8]) -> i64 {
        self.weights.get_val_at(path).copied().unwrap_or(0)
    }

    /// Returns the total positive weight used by [`select_by_offset`](Self::select_by_offset).
    pub fn total_positive_weight(&self) -> u64 {
        self.total_positive_weight
    }

    /// Sets the signed weight for `path`.
    ///
    /// Zero removes the sidecar entry. Negative values are retained as signed
    /// maintenance state, but are ignored by weighted selection.
    pub fn set_weight(&mut self, path: &[u8], weight: i64) -> Result<(), WeightedPathError> {
        let current_total = self.total_positive_weight;
        let mut zipper = self.weights.write_zipper_at_path(path);
        let previous = zipper.val().copied().unwrap_or(0);
        let total_positive_weight = updated_total(current_total, previous, weight)?;

        if weight == 0 {
            zipper.remove_val(true);
        } else {
            zipper.set_val(weight);
        }

        self.total_positive_weight = total_positive_weight;
        self.updates += 1;
        Ok(())
    }

    /// Adds `delta` to the signed weight for `path`.
    ///
    /// The addition is checked so malformed or adversarial updates cannot
    /// silently saturate, wrap, or publish an incorrect selection total.
    pub fn apply_delta(&mut self, path: &[u8], delta: i64) -> Result<(), WeightedPathError> {
        let current_total = self.total_positive_weight;
        let mut zipper = self.weights.write_zipper_at_path(path);
        let previous = zipper.val().copied().unwrap_or(0);
        let next = previous
            .checked_add(delta)
            .ok_or(WeightedPathError::WeightOverflow {
                current: previous,
                delta,
            })?;
        let total_positive_weight = updated_total(current_total, previous, next)?;

        if next == 0 {
            zipper.remove_val(true);
        } else {
            zipper.set_val(next);
        }

        self.total_positive_weight = total_positive_weight;
        self.updates += 1;
        Ok(())
    }

    /// Selects the path containing `offset` in cumulative positive-weight order.
    ///
    /// `offset` is zero-based and must be smaller than
    /// [`total_positive_weight`](Self::total_positive_weight). Paths are visited
    /// in the `PathMap` value iteration order, which is deterministic for a
    /// fixed set of encoded paths.
    pub fn select_by_offset(&self, offset: u64) -> Option<Vec<u8>> {
        if offset >= self.total_positive_weight {
            return None;
        }

        let mut remaining = offset;
        let mut zipper = self.weights.read_zipper();

        if let Some(path) = select_here(&zipper, &mut remaining) {
            return Some(path);
        }

        while zipper.to_next_val() {
            if let Some(path) = select_here(&zipper, &mut remaining) {
                return Some(path);
            }
        }

        None
    }

    /// Builds a subtree-aggregate snapshot for repeated weighted selections.
    ///
    /// This is the sidecar-safe version of the iCog `btm_i32_ws_test` branch's
    /// weighted traversal idea: aggregate weights live outside the authoritative
    /// atom `PathMap<()>`, and selection can descend by child totals rather than
    /// scanning every weighted value for every sample.
    pub fn selection_tree(&self) -> Result<WeightedSelectionTree, WeightedPathError> {
        WeightedSelectionTree::from_weights(&self.weights)
    }

    /// Selects through a freshly built aggregate snapshot.
    ///
    /// Prefer [`selection_tree`](Self::selection_tree) when drawing several
    /// samples from the same weights.
    pub fn select_by_offset_tree(&self, offset: u64) -> Result<Option<Vec<u8>>, WeightedPathError> {
        Ok(self.selection_tree()?.select_by_offset(offset))
    }

    /// WILLIAM `iter_any_topk(k)`: the `k` globally highest-positive-weight paths,
    /// each as `(path, weight)`, sorted by weight descending then path ascending.
    ///
    /// Builds a fresh aggregate snapshot; prefer [`selection_tree`](Self::selection_tree)
    /// then [`WeightedSelectionTree::top_k`] when issuing several top-k queries.
    pub fn iter_any_topk(&self, k: usize) -> Result<Vec<(Vec<u8>, u64)>, WeightedPathError> {
        Ok(self.selection_tree()?.top_k(k))
    }

    /// WILLIAM `iter_prefix_topk(prefix, k)`: the `k` highest-positive-weight paths
    /// at or below `prefix`, best-first with subtree-total pruning (no full scan).
    pub fn iter_prefix_topk(
        &self,
        prefix: &[u8],
        k: usize,
    ) -> Result<Vec<(Vec<u8>, u64)>, WeightedPathError> {
        Ok(self.selection_tree()?.top_k_under(prefix, k))
    }

    /// WILLIAM maximal top-k: the `k` heaviest patterns forming a prefix-free antichain,
    /// so nested prefixes of one hot chain collapse to a single representative. See
    /// [`WeightedSelectionTree::top_k_maximal`].
    pub fn iter_any_topk_maximal(&self, k: usize) -> Result<Vec<(Vec<u8>, u64)>, WeightedPathError> {
        Ok(self.selection_tree()?.top_k_maximal(k))
    }

    /// Builds a fixed-`k_max` precomputed top-k index for O(k) repeated prefix queries.
    /// See [`WeightedTopKIndex`]. Prefer this over repeated [`iter_prefix_topk`](Self::iter_prefix_topk)
    /// when the snapshot is static and many prefixes are queried.
    pub fn topk_index(&self, k_max: usize) -> Result<WeightedTopKIndex, WeightedPathError> {
        Ok(self.selection_tree()?.topk_index(k_max))
    }

    /// Baseline top-k by a full scan of every positive entry (O(entries) per call), sorted
    /// weight descending then path ascending. This is the work [`iter_any_topk`](Self::iter_any_topk)
    /// and [`WeightedSelectionTree::top_k`] avoid by descending the aggregate tree and pruning;
    /// it is kept as the reference oracle and the benchmark baseline.
    pub fn top_k_by_scan(&self, k: usize) -> Vec<(Vec<u8>, u64)> {
        let mut all: Vec<(Vec<u8>, u64)> = Vec::new();
        self.weights.for_each_value(|path, &w| {
            if w > 0 {
                all.push((path.to_vec(), w as u64));
            }
        });
        all.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }

    /// Returns sidecar counters without exposing the retained path data.
    pub fn stats(&self) -> WeightedPathStats {
        let mut stats = WeightedPathStats {
            total_positive_weight: self.total_positive_weight,
            updates: self.updates,
            ..WeightedPathStats::default()
        };

        self.weights.for_each_value(|_, &weight| {
            stats.entries += 1;
            if weight > 0 {
                stats.positive_entries += 1;
            } else {
                stats.non_positive_entries += 1;
            }
        });

        stats
    }
}

impl WeightedSelectionTree {
    fn from_weights(weights: &PathMap<i64>) -> Result<Self, WeightedPathError> {
        let mut nodes = BTreeMap::new();
        let total_positive_weight = weights.read_zipper().into_cata_side_effect(
            |mask, children: &mut [Result<u64, WeightedPathError>], value, path| {
                let self_weight = value.copied().map(positive_weight).unwrap_or(0);
                let mut total_weight = self_weight;
                let mut retained_children = Vec::new();

                for (byte, child_total) in mask.iter().zip(children.iter().copied()) {
                    let child_total = child_total?;
                    total_weight = checked_add_positive_weight(total_weight, child_total)?;
                    if child_total > 0 {
                        retained_children.push((byte, child_total));
                    }
                }

                if total_weight > 0 {
                    nodes.insert(
                        path.to_vec(),
                        WeightedSelectionNode {
                            self_weight,
                            total_weight,
                            children: retained_children.into_boxed_slice(),
                        },
                    );
                }

                Ok(total_weight)
            },
        );
        let total_positive_weight = total_positive_weight?;

        Ok(Self {
            total_positive_weight,
            nodes,
        })
    }

    /// Returns the total positive weight represented by this snapshot.
    pub fn total_positive_weight(&self) -> u64 {
        self.total_positive_weight
    }

    /// Selects the path containing `offset` in cumulative positive-weight order
    /// by descending subtree aggregates.
    pub fn select_by_offset(&self, offset: u64) -> Option<Vec<u8>> {
        if offset >= self.total_positive_weight {
            return None;
        }

        let mut remaining = offset;
        let mut path = Vec::new();

        loop {
            let node = self.nodes.get(path.as_slice())?;
            if remaining < node.self_weight {
                return Some(path);
            }
            remaining -= node.self_weight;

            let mut descended = false;
            for &(byte, child_total) in node.children.iter() {
                if child_total == 0 {
                    continue;
                }
                if remaining < child_total {
                    path.push(byte);
                    descended = true;
                    break;
                }
                remaining -= child_total;
            }

            if !descended {
                return None;
            }
        }
    }

    /// WILLIAM `iter_prefix_topk`: the `k` highest individual positive weights at or
    /// below `prefix`, best-first by subtree total. Each subtree's `total_weight` is an
    /// admissible upper bound on any single weight inside it, so a subtree whose total
    /// cannot strictly beat the current k-th best is pruned (output-sensitive, no full
    /// scan). Sorted by weight descending, then path ascending (deterministic).
    pub fn top_k_under(&self, prefix: &[u8], k: usize) -> Vec<(Vec<u8>, u64)> {
        if k == 0 {
            return Vec::new();
        }
        // Kept results as a max-heap whose top is the most-evictable entry: smallest
        // weight, and on a weight tie the larger path (so ties prefer smaller paths).
        let mut best: BinaryHeap<(Reverse<u64>, Vec<u8>)> = BinaryHeap::new();
        // Frontier keyed by subtree total_weight (the admissible upper bound).
        let mut frontier: BinaryHeap<(u64, Vec<u8>)> = BinaryHeap::new();
        if let Some(node) = self.nodes.get(prefix) {
            if node.total_weight > 0 {
                frontier.push((node.total_weight, prefix.to_vec()));
            }
        }
        while let Some((bound, path)) = frontier.pop() {
            // The frontier pops the largest bound first; once it cannot strictly beat
            // the k-th best weight, no remaining subtree can contribute a better entry.
            if best.len() >= k {
                if let Some((Reverse(worst), _)) = best.peek() {
                    if bound < *worst {
                        break;
                    }
                }
            }
            let Some(node) = self.nodes.get(path.as_slice()) else {
                continue;
            };
            if node.self_weight > 0 {
                offer_top_k(&mut best, k, node.self_weight, &path);
            }
            for &(byte, child_total) in node.children.iter() {
                if child_total == 0 {
                    continue;
                }
                if best.len() >= k {
                    if let Some((Reverse(worst), _)) = best.peek() {
                        if child_total < *worst {
                            continue;
                        }
                    }
                }
                let mut child_path = path.clone();
                child_path.push(byte);
                frontier.push((child_total, child_path));
            }
        }
        let mut out: Vec<(Vec<u8>, u64)> =
            best.into_iter().map(|(Reverse(w), p)| (p, w)).collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// WILLIAM `iter_any_topk`: the `k` globally highest individual positive weights.
    pub fn top_k(&self, k: usize) -> Vec<(Vec<u8>, u64)> {
        self.top_k_under(&[], k)
    }

    /// WILLIAM maximal top-k: the `k` highest-weight patterns forming a prefix-free
    /// antichain (no chosen pattern is a byte-prefix of another). Greedy by descending
    /// weight, so along any root-to-leaf chain only its single heaviest node survives.
    /// This dedups the nested-prefix chains that [`top_k`](Self::top_k) returns (e.g.
    /// `(rule (when …`, `(rule (when`, `(rule (whe…`), leaving one representative per
    /// branch. Tie-break: path ascending.
    ///
    /// Unlike [`top_k`], this considers every weighted node (the antichain constraint
    /// couples a pick to its whole chain, so subtree pruning cannot bound it); it is the
    /// report-time query, not the inner-loop sampler.
    pub fn top_k_maximal(&self, k: usize) -> Vec<(Vec<u8>, u64)> {
        if k == 0 {
            return Vec::new();
        }
        let mut cands: Vec<(u64, &Vec<u8>)> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.self_weight > 0)
            .map(|(path, node)| (node.self_weight, path))
            .collect();
        // Heaviest first; smaller path wins a weight tie (matches top_k's ordering).
        cands.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));

        let mut chosen: Vec<Vec<u8>> = Vec::new();
        let mut out: Vec<(Vec<u8>, u64)> = Vec::new();
        for (weight, path) in cands {
            if out.len() >= k {
                break;
            }
            let overlaps = chosen
                .iter()
                .any(|c| path.starts_with(c) || c.starts_with(path.as_slice()));
            if !overlaps {
                chosen.push(path.clone());
                out.push((path.clone(), weight));
            }
        }
        out
    }

    /// Returns aggregate snapshot counters.
    pub fn stats(&self) -> WeightedSelectionTreeStats {
        WeightedSelectionTreeStats {
            nodes: self.nodes.len(),
            child_edges: self.nodes.values().map(|node| node.children.len()).sum(),
            positive_value_nodes: self
                .nodes
                .values()
                .filter(|node| node.self_weight > 0)
                .count(),
            total_positive_weight: self.total_positive_weight,
        }
    }

    /// Materializes a fixed-`k_max` per-node top-k index for O(k) prefix queries.
    /// See [`WeightedTopKIndex`].
    pub fn topk_index(&self, k_max: usize) -> WeightedTopKIndex {
        WeightedTopKIndex::from_selection_tree(self, k_max)
    }
}

/// Precomputed per-node top-k lists: each node stores its `k_max` heaviest descendants,
/// so [`top_k_under`](Self::top_k_under) is an O(k) slice lookup instead of the
/// best-first descent in [`WeightedSelectionTree::top_k_under`]. This is the fixed-k
/// range-top-k precomputation (Hon, Shah & Vitter 2009): trade space and one build pass
/// for constant-time repeated queries, the right shape when many prefixes are queried
/// against a static snapshot.
///
/// Built bottom-up: processing nodes in reverse-lexicographic order visits every
/// descendant before its ancestor (a node path is a strict prefix of, hence lexicographically
/// less than, all its descendants), so each node merges its own weight with its children's
/// already-materialized lists and keeps the heaviest `k_max`. Keeping `k_max` at every node
/// suffices to answer any `k <= k_max` query at any ancestor.
#[derive(Clone, Debug, Default)]
pub struct WeightedTopKIndex {
    k_max: usize,
    per_node: BTreeMap<Vec<u8>, Vec<(Vec<u8>, u64)>>,
}

impl WeightedTopKIndex {
    fn from_selection_tree(tree: &WeightedSelectionTree, k_max: usize) -> Self {
        let mut per_node: BTreeMap<Vec<u8>, Vec<(Vec<u8>, u64)>> = BTreeMap::new();
        if k_max == 0 {
            return Self { k_max, per_node };
        }
        // Reverse-lexicographic: every descendant (a lexicographically greater path)
        // is finalized before its ancestor, so child lists are ready at merge time.
        for (path, node) in tree.nodes.iter().rev() {
            let mut candidates: Vec<(u64, Vec<u8>)> = Vec::new();
            if node.self_weight > 0 {
                candidates.push((node.self_weight, path.clone()));
            }
            for &(byte, child_total) in node.children.iter() {
                if child_total == 0 {
                    continue;
                }
                let mut child_path = path.clone();
                child_path.push(byte);
                if let Some(child_list) = per_node.get(&child_path) {
                    for (p, w) in child_list {
                        candidates.push((*w, p.clone()));
                    }
                }
            }
            // Heaviest first, smaller path on a weight tie: identical ordering to
            // WeightedSelectionTree::top_k_under so the two agree byte for byte.
            candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            candidates.truncate(k_max);
            per_node.insert(
                path.clone(),
                candidates.into_iter().map(|(w, p)| (p, w)).collect(),
            );
        }
        Self { k_max, per_node }
    }

    /// The fixed cap this index was built for. Queries with `k > k_max` are clamped;
    /// answering a larger `k` needs a rebuild with a larger cap.
    pub fn k_max(&self) -> usize {
        self.k_max
    }

    /// The `k` heaviest patterns at or below `prefix`, as an O(k) lookup. `k` is clamped
    /// to [`k_max`](Self::k_max). Byte-for-byte equal to
    /// [`WeightedSelectionTree::top_k_under`] for `k <= k_max`.
    pub fn top_k_under(&self, prefix: &[u8], k: usize) -> Vec<(Vec<u8>, u64)> {
        let k = k.min(self.k_max);
        if k == 0 {
            return Vec::new();
        }
        match self.per_node.get(prefix) {
            None => Vec::new(),
            Some(list) => list[..list.len().min(k)].to_vec(),
        }
    }

    /// The `k` globally heaviest patterns, as an O(k) lookup.
    pub fn top_k(&self, k: usize) -> Vec<(Vec<u8>, u64)> {
        self.top_k_under(&[], k)
    }
}

/// One store mutation reported to a [`LiveCompressionGainIndex`].
#[derive(Clone, Debug)]
enum LiveGainEvent {
    Inserted(Vec<u8>),
    Removed(Vec<u8>),
}

/// A registered producer handle for one writer (thread) of a
/// [`LiveCompressionGainIndex`]. Only its owner pushes into the cell, so recording is
/// uncontended on the hot path; [`LiveCompressionGainIndex::refresh`] briefly locks each
/// cell to drain it. This is the per-core-buffer half of the snapshot/RCU design, the
/// same shape as the engine's thread-local metrics cells, but instance-scoped so two
/// live indexes never couple through a global.
#[derive(Clone)]
pub struct LiveGainWriter {
    cell: std::sync::Arc<Mutex<Vec<LiveGainEvent>>>,
}

impl LiveGainWriter {
    /// Records that `atom` was genuinely added to the store (it was not present).
    /// PathMap inserts are idempotent, so the caller must report only real additions;
    /// the engine's write path already knows (`any_new`).
    pub fn record_insert(&self, atom: &[u8]) {
        self.cell.lock().unwrap().push(LiveGainEvent::Inserted(atom.to_vec()));
    }

    /// Records that `atom` was genuinely deleted from the store (it was present).
    pub fn record_remove(&self, atom: &[u8]) {
        self.cell.lock().unwrap().push(LiveGainEvent::Removed(atom.to_vec()));
    }
}

/// Compression-gain index maintained live under concurrent store writes (WILLIAM
/// S2-LIVE): writers record genuine store mutations through per-writer buffers, a
/// maintainer folds them in with [`refresh`](Self::refresh), and readers take immutable
/// [`WeightedSelectionTree`] snapshots wait-free via [`current`](Self::current).
///
/// The maintained state is `counts`: for every term-boundary prefix `p`, the number of
/// stored atoms extending `p`. One atom of `T` items touches exactly its `T` boundary
/// prefixes, so recording a mutation costs O(atom items), the same order as writing the
/// atom itself, and never rescans the store. Gains derive from counts pointwise
/// (`boundary_gain`), so a refresh recomputes only the prefixes the drained events
/// touched. Count deltas commute, so any drain interleaving of the same event multiset
/// (transient negatives included) lands on the same final counts.
pub struct LiveCompressionGainIndex {
    ref_cost: u64,
    counts: PathMap<i64>,
    index: WeightedPathIndex,
    published: std::sync::RwLock<std::sync::Arc<WeightedSelectionTree>>,
    writers: Mutex<Vec<std::sync::Arc<Mutex<Vec<LiveGainEvent>>>>>,
}

impl LiveCompressionGainIndex {
    /// An empty live index (no atoms yet).
    pub fn new(ref_cost: u64) -> Self {
        Self {
            ref_cost,
            counts: PathMap::new(),
            index: WeightedPathIndex::new(),
            published: std::sync::RwLock::new(std::sync::Arc::new(
                WeightedSelectionTree::default(),
            )),
            writers: Mutex::new(Vec::new()),
        }
    }

    /// Seeds the live index from an existing store, equal to
    /// [`WeightedPathIndex::from_compression_gain_on_boundaries`] over the same atoms.
    pub fn from_store(atoms: &PathMap<()>, ref_cost: u64) -> Self {
        let mut live = Self::new(ref_cost);
        let mut seeded: Vec<Vec<u8>> = Vec::new();
        atoms.for_each_value(|path, _| seeded.push(path.to_vec()));
        let mut touched: Vec<Vec<u8>> = Vec::new();
        for atom in &seeded {
            live.bump_boundary_counts(atom, 1, &mut touched);
        }
        live.recompute_gains(&mut touched);
        live.publish();
        live
    }

    /// Registers a producer handle. Each writer (typically one per thread) records its
    /// mutations through its own handle.
    pub fn writer(&self) -> LiveGainWriter {
        let cell = std::sync::Arc::new(Mutex::new(Vec::new()));
        self.writers.lock().unwrap().push(std::sync::Arc::clone(&cell));
        LiveGainWriter { cell }
    }

    /// The most recently published snapshot. Readers clone an `Arc`, so a snapshot is
    /// immutable and never torn by an in-flight refresh.
    pub fn current(&self) -> std::sync::Arc<WeightedSelectionTree> {
        std::sync::Arc::clone(&self.published.read().unwrap())
    }

    /// Drains every writer buffer, folds the events into counts and gains, and publishes
    /// a fresh snapshot. Events recorded during the drain land in the next refresh.
    pub fn refresh(&mut self) {
        let drained: Vec<Vec<LiveGainEvent>> = {
            let writers = self.writers.lock().unwrap();
            writers.iter().map(|cell| std::mem::take(&mut *cell.lock().unwrap())).collect()
        };
        let mut touched: Vec<Vec<u8>> = Vec::new();
        for events in drained {
            for event in events {
                match event {
                    LiveGainEvent::Inserted(atom) => {
                        self.bump_boundary_counts(&atom, 1, &mut touched)
                    }
                    LiveGainEvent::Removed(atom) => {
                        self.bump_boundary_counts(&atom, -1, &mut touched)
                    }
                }
            }
        }
        self.recompute_gains(&mut touched);
        self.publish();
    }

    /// The maintained index (for gain lookups against the latest refresh).
    pub fn index(&self) -> &WeightedPathIndex {
        &self.index
    }

    fn publish(&self) {
        let tree = self
            .index
            .selection_tree()
            .unwrap_or_else(|_| WeightedSelectionTree::default());
        *self.published.write().unwrap() = std::sync::Arc::new(tree);
    }

    /// Adds `delta` to the count of every term-boundary prefix of `atom`, recording each
    /// touched prefix. One forward item scan yields all boundary offsets, O(atom items).
    fn bump_boundary_counts(&mut self, atom: &[u8], delta: i64, touched: &mut Vec<Vec<u8>>) {
        let mut pos = 0usize;
        while pos < atom.len() {
            let step = match maybe_byte_item(atom[pos]) {
                Ok(Tag::NewVar) | Ok(Tag::VarRef(_)) | Ok(Tag::Arity(_)) => 1,
                Ok(Tag::SymbolSize(s)) => 1 + s as usize,
                Err(_) => return, // foreign path: no boundary prefixes.
            };
            if pos + step > atom.len() {
                return; // truncated symbol payload: not a MORK atom.
            }
            pos += step;
            let prefix = &atom[..pos];
            let mut zipper = self.counts.write_zipper_at_path(prefix);
            let next = zipper.val().copied().unwrap_or(0) + delta;
            if next == 0 {
                zipper.remove_val(true);
            } else {
                zipper.set_val(next);
            }
            touched.push(prefix.to_vec());
        }
    }

    /// Recomputes the gain entry for each touched prefix from its final count.
    fn recompute_gains(&mut self, touched: &mut Vec<Vec<u8>>) {
        touched.sort();
        touched.dedup();
        for prefix in touched.drain(..) {
            let count = self.counts.get_val_at(&prefix).copied().unwrap_or(0);
            let gain = boundary_gain(count, prefix.len(), self.ref_cost);
            let weight = if count >= 2 && gain > 0 { gain } else { 0 };
            // Overflow is only reachable at extreme scales; dropping the update omits
            // one entry from the derived index, never corrupts counts.
            let _ = self.index.set_weight(&prefix, weight);
        }
    }
}

/// Bytes saved by factoring a `len`-byte pattern shared by `count` atoms into one
/// definition plus `count` references: `(count - 1) * len - count * ref_cost`.
fn boundary_gain(count: i64, len: usize, ref_cost: u64) -> i64 {
    (count - 1) * len as i64 - count * ref_cost as i64
}

fn positive_weight(weight: i64) -> u64 {
    if weight > 0 { weight as u64 } else { 0 }
}

/// Offer `(weight, path)` to the kept top-k set. The heap's top is the most-evictable
/// entry (smallest weight, larger path on a tie), so a new entry is kept when it has a
/// larger weight, or an equal weight with a smaller path.
fn offer_top_k(best: &mut BinaryHeap<(Reverse<u64>, Vec<u8>)>, k: usize, weight: u64, path: &[u8]) {
    if best.len() < k {
        best.push((Reverse(weight), path.to_vec()));
        return;
    }
    if let Some((Reverse(worst_weight), worst_path)) = best.peek() {
        let better = weight > *worst_weight
            || (weight == *worst_weight && path < worst_path.as_slice());
        if better {
            best.pop();
            best.push((Reverse(weight), path.to_vec()));
        }
    }
}

fn updated_total(current_total: u64, previous: i64, next: i64) -> Result<u64, WeightedPathError> {
    let previous_positive = positive_weight(previous);
    let next_positive = positive_weight(next);

    if next_positive >= previous_positive {
        checked_add_positive_weight(current_total, next_positive - previous_positive)
    } else {
        let decrement = previous_positive - next_positive;
        current_total.checked_sub(decrement).ok_or(
            WeightedPathError::TotalPositiveWeightUnderflow {
                current: current_total,
                decrement,
            },
        )
    }
}

fn checked_add_positive_weight(left: u64, right: u64) -> Result<u64, WeightedPathError> {
    left.checked_add(right)
        .ok_or(WeightedPathError::TotalPositiveWeightOverflow { left, right })
}

fn select_here<Z>(zipper: &Z, remaining: &mut u64) -> Option<Vec<u8>>
where
    Z: Zipper + ZipperAbsolutePath + ZipperValues<i64>,
{
    let weight = positive_weight(*zipper.val()?);
    if weight == 0 {
        return None;
    }

    if *remaining < weight {
        return Some(zipper.path().to_vec());
    }

    *remaining -= weight;
    None
}

/// Whether `path` (a prefix of a MORK-encoded atom) ends exactly between items.
///
/// MORK encodes each item as a single tag byte; `SymbolSize(s)` is followed by `s` raw
/// payload bytes and `Arity(a)` opens `a` complete subterms. A prefix that ends inside a
/// symbol's payload is a mid-symbol cut, not a subexpression boundary. Scanning item by
/// item, `path.len()` is a boundary iff the scan lands on it exactly (every symbol's full
/// payload consumed). O(items) in the prefix.
///
/// Total on arbitrary stores: a byte in the reserved tag range (a path that is not a
/// MORK-encoded expression) makes the whole prefix a non-boundary rather than panicking,
/// so the boundary-restricted index simply skips foreign paths.
fn path_ends_on_term_boundary(path: &[u8]) -> bool {
    let mut pos = 0usize;
    while pos < path.len() {
        match maybe_byte_item(path[pos]) {
            Ok(Tag::NewVar) | Ok(Tag::VarRef(_)) | Ok(Tag::Arity(_)) => pos += 1,
            Ok(Tag::SymbolSize(s)) => {
                let next = pos + 1 + s as usize;
                if next > path.len() {
                    // path ends inside this symbol's payload: not a boundary.
                    return false;
                }
                pos = next;
            }
            Err(_) => return false,
        }
    }
    pos == path.len()
}

/// Frame for one open compound while decoding: how many argument slots remain, and
/// whether the compound has emitted its first element yet (for s-expression spacing).
struct DecodeFrame {
    remaining: usize,
    first: bool,
}

/// Renders a MORK-encoded path prefix as readable MeTTa.
///
/// The prefix must end on a term boundary (as produced by
/// [`WeightedPathIndex::from_compression_gain_on_boundaries`]). A compound whose trailing
/// arguments the prefix cut off shows the missing slots as `…`, so the pattern reads as
/// the whole subexpression it factors: the encoding of `(rule (when $x))` truncated after
/// `rule` decodes to `(rule …)`, and an arity opened with no head yet decodes to `(…)`.
/// Non-UTF-8 symbol bytes are hex-escaped so the rendering never panics.
pub fn decode_pattern(path: &[u8]) -> String {
    let mut out = String::new();
    let mut stack: Vec<DecodeFrame> = Vec::new();
    let mut pos = 0usize;

    while pos < path.len() {
        let Ok(item) = maybe_byte_item(path[pos]) else {
            break; // foreign byte: render what decoded so far.
        };
        match item {
            Tag::Arity(a) => {
                pos += 1;
                before_element(&mut out, &mut stack);
                out.push('(');
                if a == 0 {
                    out.push(')');
                    complete_element(&mut out, &mut stack);
                } else {
                    stack.push(DecodeFrame { remaining: a as usize, first: true });
                }
            }
            Tag::SymbolSize(s) => {
                let start = pos + 1;
                let end = start + s as usize;
                if end > path.len() {
                    break; // defensive: a non-boundary prefix slipped in.
                }
                before_element(&mut out, &mut stack);
                emit_symbol(&mut out, &path[start..end]);
                pos = end;
                complete_element(&mut out, &mut stack);
            }
            Tag::NewVar => {
                pos += 1;
                before_element(&mut out, &mut stack);
                out.push('$');
                complete_element(&mut out, &mut stack);
            }
            Tag::VarRef(i) => {
                pos += 1;
                before_element(&mut out, &mut stack);
                out.push('_');
                out.push_str(&i.to_string());
                complete_element(&mut out, &mut stack);
            }
        }
    }

    // Truncation: close every still-open compound, filling its unfilled argument slots
    // with `…`. A closed compound occupies one slot of its parent, so decrement the
    // parent before flushing it (that slot is the in-progress child, not a fresh `…`).
    while let Some(frame) = stack.pop() {
        for slot in 0..frame.remaining {
            if !(frame.first && slot == 0) {
                out.push(' ');
            }
            out.push('…');
        }
        out.push(')');
        if let Some(parent) = stack.last_mut() {
            parent.remaining = parent.remaining.saturating_sub(1);
            parent.first = false;
        }
    }

    out
}

/// Emit s-expression spacing before an element: a space when the current compound has
/// already emitted an element, nothing when this is its head (or at the top level).
fn before_element(out: &mut String, stack: &mut [DecodeFrame]) {
    if let Some(top) = stack.last_mut() {
        if top.first {
            top.first = false;
        } else {
            out.push(' ');
        }
    }
}

/// Register that one element finished: decrement the enclosing compound's remaining
/// slots and cascade-close every compound that reaches zero.
fn complete_element(out: &mut String, stack: &mut Vec<DecodeFrame>) {
    loop {
        match stack.last_mut() {
            None => break,
            Some(top) => {
                top.remaining -= 1;
                if top.remaining == 0 {
                    out.push(')');
                    stack.pop();
                } else {
                    break;
                }
            }
        }
    }
}

/// Append a symbol's raw bytes as text, hex-escaping when they are not valid UTF-8.
fn emit_symbol(out: &mut String, sym: &[u8]) {
    match std::str::from_utf8(sym) {
        Ok(s) => out.push_str(s),
        Err(_) => {
            out.push_str("0x");
            for b in sym {
                out.push_str(&format!("{b:02x}"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_by_offset_returns_paths_by_positive_weight_ranges() -> Result<(), WeightedPathError> {
        let mut index = WeightedPathIndex::new();
        index.set_weight(b"foo", 2)?;
        index.set_weight(b"bar", 1)?;
        index.set_weight(b"zap", 3)?;

        assert_eq!(index.total_positive_weight(), 6);
        assert_eq!(index.select_by_offset(0).as_deref(), Some(&b"bar"[..]));
        assert_eq!(index.select_by_offset(1).as_deref(), Some(&b"foo"[..]));
        assert_eq!(index.select_by_offset(2).as_deref(), Some(&b"foo"[..]));
        assert_eq!(index.select_by_offset(3).as_deref(), Some(&b"zap"[..]));
        assert_eq!(index.select_by_offset(5).as_deref(), Some(&b"zap"[..]));
        assert_eq!(index.select_by_offset(6), None);
        Ok(())
    }

    #[test]
    fn apply_delta_removes_zero_weight_entries_and_updates_total() -> Result<(), WeightedPathError>
    {
        let mut index = WeightedPathIndex::new();
        index.apply_delta(b"foo", 5)?;
        index.apply_delta(b"foo", -2)?;
        index.apply_delta(b"foo", -3)?;

        assert_eq!(index.weight(b"foo"), 0);
        assert_eq!(index.total_positive_weight(), 0);
        assert_eq!(index.stats().entries, 0);
        Ok(())
    }

    #[test]
    fn negative_weights_are_retained_but_not_selected() -> Result<(), WeightedPathError> {
        let mut index = WeightedPathIndex::new();
        index.set_weight(b"cold", -4)?;
        index.set_weight(b"hot", 2)?;

        assert_eq!(index.weight(b"cold"), -4);
        assert_eq!(index.select_by_offset(0).as_deref(), Some(&b"hot"[..]));

        let stats = index.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.positive_entries, 1);
        assert_eq!(stats.non_positive_entries, 1);
        assert_eq!(stats.total_positive_weight, 2);
        Ok(())
    }

    #[test]
    fn selection_tree_matches_linear_selection_with_prefix_values() -> Result<(), WeightedPathError>
    {
        let mut index = WeightedPathIndex::new();
        index.set_weight(b"a", 2)?;
        index.set_weight(b"ab", 3)?;
        index.set_weight(b"ac", 1)?;
        index.set_weight(b"b", -10)?;
        index.set_weight(b"bd", 4)?;

        let tree = index.selection_tree()?;

        assert_eq!(tree.total_positive_weight(), index.total_positive_weight());
        for offset in 0..index.total_positive_weight() {
            assert_eq!(
                tree.select_by_offset(offset),
                index.select_by_offset(offset),
                "offset {offset}",
            );
        }
        assert_eq!(tree.select_by_offset(index.total_positive_weight()), None);

        let stats = tree.stats();
        assert_eq!(stats.positive_value_nodes, 4);
        assert_eq!(stats.total_positive_weight, 10);
        assert!(stats.nodes >= stats.positive_value_nodes);
        assert!(stats.child_edges >= 4);
        Ok(())
    }

    #[test]
    fn selection_tree_prunes_zero_positive_subtrees() -> Result<(), WeightedPathError> {
        let mut index = WeightedPathIndex::new();
        index.set_weight(b"cold", -10)?;

        let tree = index.selection_tree()?;
        let stats = tree.stats();

        assert_eq!(tree.total_positive_weight(), 0);
        assert_eq!(tree.select_by_offset(0), None);
        assert_eq!(stats.nodes, 0);
        assert_eq!(stats.child_edges, 0);
        assert_eq!(stats.positive_value_nodes, 0);
        Ok(())
    }

    #[test]
    fn apply_delta_rejects_signed_weight_overflow_without_mutation() -> Result<(), WeightedPathError>
    {
        let mut index = WeightedPathIndex::new();
        index.set_weight(b"huge", i64::MAX)?;

        assert_eq!(
            index.apply_delta(b"huge", 1),
            Err(WeightedPathError::WeightOverflow {
                current: i64::MAX,
                delta: 1
            })
        );
        assert_eq!(index.weight(b"huge"), i64::MAX);
        assert_eq!(index.total_positive_weight(), i64::MAX as u64);
        assert_eq!(index.stats().updates, 1);
        Ok(())
    }

    #[test]
    fn set_weight_rejects_total_positive_overflow_without_mutation() -> Result<(), WeightedPathError>
    {
        let mut index = WeightedPathIndex::new();
        index.set_weight(b"a", i64::MAX)?;
        index.set_weight(b"b", i64::MAX)?;

        assert_eq!(
            index.set_weight(b"c", 2),
            Err(WeightedPathError::TotalPositiveWeightOverflow {
                left: (i64::MAX as u64) * 2,
                right: 2
            })
        );
        assert_eq!(index.weight(b"c"), 0);
        assert_eq!(index.total_positive_weight(), (i64::MAX as u64) * 2);
        assert_eq!(index.stats().updates, 2);
        Ok(())
    }

    #[test]
    fn top_k_matches_brute_force_including_ties_and_prefixes() -> Result<(), WeightedPathError> {
        let mut index = WeightedPathIndex::new();
        let data: &[(&[u8], i64)] = &[
            (b"a", 5),
            (b"ab", 5),
            (b"abc", 2),
            (b"ad", 9),
            (b"b", -3),
            (b"bd", 7),
            (b"bde", 7),
            (b"c", 1),
            (b"cc", 4),
        ];
        for (p, w) in data {
            index.set_weight(p, *w)?;
        }
        let tree = index.selection_tree()?;

        // Brute force: every positive-weight path, ranked weight desc then path asc.
        let mut all: Vec<(Vec<u8>, u64)> = Vec::new();
        index.weights.for_each_value(|path, &w| {
            if w > 0 {
                all.push((path.to_vec(), w as u64));
            }
        });
        all.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        for k in 0..=all.len() + 2 {
            let mut brute = all.clone();
            brute.truncate(k);
            assert_eq!(tree.top_k(k), brute, "top_k({k})");
            assert_eq!(index.iter_any_topk(k)?, brute, "iter_any_topk({k})");
        }

        for prefix in [
            &b""[..],
            &b"a"[..],
            &b"ab"[..],
            &b"b"[..],
            &b"bd"[..],
            &b"c"[..],
            &b"z"[..],
        ] {
            let mut scoped: Vec<(Vec<u8>, u64)> = all
                .iter()
                .filter(|(p, _)| p.starts_with(prefix))
                .cloned()
                .collect();
            for k in 0..=scoped.len() + 1 {
                let mut want = scoped.clone();
                want.truncate(k);
                assert_eq!(tree.top_k_under(prefix, k), want, "top_k_under({prefix:?}, {k})");
                assert_eq!(index.iter_prefix_topk(prefix, k)?, want, "iter_prefix_topk");
            }
            let _ = &mut scoped;
        }
        Ok(())
    }

    #[test]
    fn compression_gain_index_is_sound_and_complete_on_nodes() {
        use std::collections::BTreeSet;

        // Representation-independent oracle. PathMap path-compresses unary chains, so which
        // exact prefixes become trie nodes is an implementation detail; two properties hold
        // for any representation and pin the semantics:
        //   soundness:    every stored entry has count >= 2 and weight == the true gain > 0;
        //   completeness: every canonical node (a stored value, or a >= 2-way branch -- these
        //                 exist in any representation) with count >= 2 and positive gain is stored.
        fn check(atoms_list: &[&[u8]], ref_cost: u64) {
            let mut atoms: PathMap<()> = PathMap::new();
            for a in atoms_list {
                atoms.insert(*a, ());
            }
            let index = WeightedPathIndex::from_compression_gain(&atoms, ref_cost);

            let count_of = |p: &[u8]| atoms_list.iter().filter(|k| k.starts_with(p)).count();
            let gain_of = |p: &[u8]| {
                let c = count_of(p) as i64;
                (c - 1) * p.len() as i64 - c * ref_cost as i64
            };

            index.weights.for_each_value(|path, &w| {
                assert_eq!(w, gain_of(path), "wrong gain at {path:?}");
                assert!(count_of(path) >= 2 && w > 0, "spurious entry {path:?}");
            });

            let mut prefixes: BTreeSet<Vec<u8>> = BTreeSet::new();
            for a in atoms_list {
                for l in 1..=a.len() {
                    prefixes.insert(a[..l].to_vec());
                }
            }
            for p in &prefixes {
                let is_value = atoms_list.iter().any(|k| *k == &p[..]);
                let nexts: BTreeSet<u8> = atoms_list
                    .iter()
                    .filter(|k| k.len() > p.len() && k.starts_with(&p[..]))
                    .map(|k| k[p.len()])
                    .collect();
                let is_node = is_value || nexts.len() >= 2;
                if is_node && count_of(p) >= 2 && gain_of(p) > 0 {
                    assert_eq!(index.weight(p), gain_of(p), "missing canonical node {p:?}");
                }
            }
        }

        check(&[b"cat", b"car", b"card", b"care", b"dog", b"do"], 1);
        check(&[b"aaaax", b"aaaay"], 1); // unary chain; only branch node "aaaa" matters
        check(&[b"aaab", b"aaac", b"aaad"], 1); // "aaa" count 3, gain (3-1)*3 - 3 = 3
        check(&[b"x"], 1); // singleton: no shared prefixes, empty index

        // Ordering: "car" (gain 3) is the single most compressible prefix in the branchy set.
        let mut atoms: PathMap<()> = PathMap::new();
        for a in [&b"cat"[..], &b"car"[..], &b"card"[..], &b"care"[..], &b"dog"[..], &b"do"[..]] {
            atoms.insert(a, ());
        }
        let index = WeightedPathIndex::from_compression_gain(&atoms, 1);
        assert_eq!(index.iter_any_topk(1).unwrap(), vec![(b"car".to_vec(), 3u64)]);
        assert_eq!(index.weight(b"ca"), 2);
    }

    // ---- W1: term-boundary gain and MeTTa decode ----

    use crate::encoded_test_helpers::{arity, cat, var as newvar, var_ref as varref};

    /// Encode a symbol item from text (the shared helper takes raw bytes).
    fn sym(s: &str) -> Vec<u8> {
        crate::encoded_test_helpers::sym(s.as_bytes())
    }

    #[test]
    fn term_boundary_holds_only_between_items() {
        // (f a b) = Arity(3) Sym(1)f Sym(1)a Sym(1)b, offsets 0..=7.
        let e = cat(&[arity(3), sym("f"), sym("a"), sym("b")]);
        assert_eq!(e.len(), 7);
        // Item starts (0,1,3,5) and the full length (7) are boundaries; the raw
        // symbol payload bytes (2,4,6) are not.
        let want = [true, true, false, true, false, true, false, true];
        for off in 0..=e.len() {
            assert_eq!(
                path_ends_on_term_boundary(&e[..off]),
                want[off],
                "offset {off}"
            );
        }
    }

    #[test]
    fn decode_renders_boundary_prefixes_as_whole_subexpressions() {
        // Full expressions round-trip.
        assert_eq!(decode_pattern(&cat(&[arity(3), sym("f"), sym("a"), sym("b")])), "(f a b)");
        assert_eq!(decode_pattern(&sym("abc")), "abc");
        assert_eq!(decode_pattern(&newvar()), "$");
        assert_eq!(decode_pattern(&varref(2)), "_2");
        assert_eq!(decode_pattern(&arity(0)), "()");
        assert_eq!(
            decode_pattern(&cat(&[arity(2), sym("f"), cat(&[arity(2), sym("g"), sym("x")])])),
            "(f (g x))"
        );

        // Truncated-on-a-boundary prefixes show the cut-off argument slots as `…`.
        assert_eq!(decode_pattern(&cat(&[arity(3), sym("f")])), "(f … …)");
        assert_eq!(decode_pattern(&arity(3)), "(… … …)");
        assert_eq!(decode_pattern(&cat(&[arity(2), arity(3), sym("f")])), "((f … …) …)");
        // A complete 2-ary expression is not truncated; a 3-ary with one arg still open is.
        assert_eq!(decode_pattern(&cat(&[arity(2), sym("cons"), sym("head")])), "(cons head)");
        assert_eq!(decode_pattern(&cat(&[arity(3), sym("cons"), sym("head")])), "(cons head …)");
    }

    #[test]
    fn boundary_gain_is_the_boundary_restriction_of_full_gain() {
        use std::collections::BTreeSet;

        // Two 3-ary rules sharing the `(rule (when a) …)` spine and then diverging at that
        // boundary by tag type (a symbol vs a compound), so the shared spine is a real
        // branch node, not an interior unary prefix that path compression would drop.
        let atoms_list: Vec<Vec<u8>> = vec![
            cat(&[arity(3), sym("rule"), cat(&[arity(2), sym("when"), sym("a")]), sym("A")]),
            cat(&[
                arity(3),
                sym("rule"),
                cat(&[arity(2), sym("when"), sym("a")]),
                cat(&[arity(1), sym("B")]),
            ]),
            cat(&[arity(2), sym("fact"), sym("z")]),
        ];
        let ref_cost = 2u64;

        let mut atoms: PathMap<()> = PathMap::new();
        for a in &atoms_list {
            atoms.insert(&a[..], ());
        }
        let full = WeightedPathIndex::from_compression_gain(&atoms, ref_cost);
        let bounded = WeightedPathIndex::from_compression_gain_on_boundaries(&atoms, ref_cost);

        let count_of = |p: &[u8]| atoms_list.iter().filter(|k| k.starts_with(p)).count() as i64;
        let gain_of = |p: &[u8]| (count_of(p) - 1) * p.len() as i64 - count_of(p) * ref_cost as i64;

        // Brute-force oracle: exactly the boundary prefixes with count>=2 and gain>0.
        let mut want: BTreeSet<Vec<u8>> = BTreeSet::new();
        let mut all_prefixes: BTreeSet<Vec<u8>> = BTreeSet::new();
        for a in &atoms_list {
            for l in 1..=a.len() {
                all_prefixes.insert(a[..l].to_vec());
            }
        }
        for p in &all_prefixes {
            if path_ends_on_term_boundary(p) && count_of(p) >= 2 && gain_of(p) > 0 {
                want.insert(p.clone());
            }
        }

        // Every bounded entry ends on a boundary, carries the true gain, and is present in
        // the unrestricted index (a subset). Completeness is checked against `want` below.
        let mut got: BTreeSet<Vec<u8>> = BTreeSet::new();
        bounded.weights.for_each_value(|path, &w| {
            assert!(path_ends_on_term_boundary(path), "non-boundary entry {path:?}");
            assert_eq!(w as i64, gain_of(path), "wrong gain at {path:?}");
            assert!(full.weight(path) >= w as i64, "bounded entry not in full index {path:?}");
            assert!(decode_pattern(path).len() > 0);
            got.insert(path.to_vec());
        });

        // Canonical completeness: every boundary node that is a stored value or a >=2-way
        // branch and clears the gain bar must be present (path compression can drop only
        // interior unary boundary prefixes, never a branch or value node).
        for p in &want {
            let is_value = atoms_list.iter().any(|k| k == p);
            let nexts: BTreeSet<u8> = atoms_list
                .iter()
                .filter(|k| k.len() > p.len() && k.starts_with(&p[..]))
                .map(|k| k[p.len()])
                .collect();
            if is_value || nexts.len() >= 2 {
                assert!(got.contains(p), "missing canonical boundary node {}", decode_pattern(p));
            }
        }

        // The shared rule spine is the heaviest boundary pattern and reads as MeTTa.
        let top = bounded.iter_any_topk(1).unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(decode_pattern(&top[0].0), "(rule (when a) …)");
    }

    // ---- W2: maximal / non-overlapping top-k ----

    /// Independent brute-force greedy antichain: heaviest first, keep a pattern only if it
    /// neither contains nor is contained by an already-kept pattern.
    fn brute_maximal(entries: &[(Vec<u8>, u64)], k: usize) -> Vec<(Vec<u8>, u64)> {
        let mut sorted = entries.to_vec();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut chosen: Vec<Vec<u8>> = Vec::new();
        let mut out = Vec::new();
        for (p, w) in sorted {
            if out.len() >= k {
                break;
            }
            if !chosen.iter().any(|c| p.starts_with(c) || c.starts_with(&p[..])) {
                chosen.push(p.clone());
                out.push((p, w));
            }
        }
        out
    }

    #[test]
    fn maximal_top_k_matches_brute_force_and_is_an_antichain() -> Result<(), WeightedPathError> {
        // A hot chain of nested prefixes (a>b>c>d, decreasing gain along the chain) plus two
        // independent lighter patterns off to the side.
        let mut index = WeightedPathIndex::new();
        index.set_weight(b"chain", 40)?;
        index.set_weight(b"chai", 35)?;
        index.set_weight(b"cha", 30)?;
        index.set_weight(b"ch", 20)?;
        index.set_weight(b"other", 25)?;
        index.set_weight(b"zeta", 15)?;

        let tree = index.selection_tree()?;
        let entries: Vec<(Vec<u8>, u64)> = {
            let mut v = Vec::new();
            index.weights.for_each_value(|p, &w| {
                if w > 0 {
                    v.push((p.to_vec(), w as u64));
                }
            });
            v
        };

        for k in 0..=6 {
            let got = tree.top_k_maximal(k);
            assert_eq!(got, brute_maximal(&entries, k), "k={k}");
            // Output is a prefix-free antichain.
            for i in 0..got.len() {
                for j in 0..got.len() {
                    if i != j {
                        assert!(
                            !got[i].0.starts_with(&got[j].0[..]),
                            "not an antichain: {:?} under {:?}",
                            got[i].0,
                            got[j].0
                        );
                    }
                }
            }
        }

        // The whole `chain` chain collapses to its single heaviest node ("chain", 40); the
        // side patterns survive. Plain top-k would instead return the four nested prefixes.
        assert_eq!(
            tree.top_k_maximal(3),
            vec![(b"chain".to_vec(), 40), (b"other".to_vec(), 25), (b"zeta".to_vec(), 15)]
        );
        assert_eq!(
            tree.top_k(3),
            vec![(b"chain".to_vec(), 40), (b"chai".to_vec(), 35), (b"cha".to_vec(), 30)]
        );
        Ok(())
    }

    // ---- W3: precomputed per-node top-k (S1b) ----

    #[test]
    fn boundary_builder_is_total_on_non_mork_stores() {
        // Raw-text paths sit in the reserved tag range (0x40..0x7F); the boundary builder
        // must skip them without panicking and admit nothing.
        let mut atoms: PathMap<()> = PathMap::new();
        for key in ["(rule (c C01) (a A0f))", "(rule (c C01) (b B0f))", "plain text"] {
            atoms.insert(key.as_bytes(), ());
        }
        let bounded = WeightedPathIndex::from_compression_gain_on_boundaries(&atoms, 1);
        assert_eq!(bounded.stats().entries, 0);
        // The unrestricted builder still weights the shared byte prefixes.
        let full = WeightedPathIndex::from_compression_gain(&atoms, 1);
        assert!(full.stats().positive_entries > 0);
    }

    #[test]
    fn precomputed_topk_matches_on_the_fly_descent() {
        // Deterministic pseudo-random compression stores (LCG-driven) exercise branchy
        // encoded tries with shared spines and ties; the precomputed O(k) index must
        // agree byte-for-byte with the best-first descent for every prefix and k <= k_max.
        let k_max = 6;
        for seed in [1u64, 7, 13, 29, 101] {
            let mut atoms: PathMap<()> = PathMap::new();
            let mut x = seed;
            for _ in 0..400 {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // Two shared rule templates + a unique-fact tail, MORK-encoded, so
                // boundary prefixes collide, branch, and tie.
                let key = match x % 3 {
                    0 => cat(&[
                        arity(3),
                        sym("rule"),
                        cat(&[arity(2), sym("c"), sym(&format!("C{:02}", x % 8))]),
                        cat(&[arity(2), sym("a"), sym(&format!("A{:04x}", (x >> 8) % 4096))]),
                    ]),
                    1 => cat(&[
                        arity(3),
                        sym("rule"),
                        cat(&[arity(2), sym("c"), sym(&format!("C{:02}", x % 8))]),
                        cat(&[arity(2), sym("b"), sym(&format!("B{:04x}", (x >> 8) % 4096))]),
                    ]),
                    _ => cat(&[
                        arity(3),
                        sym("fact"),
                        sym(&format!("F{:04x}", (x >> 4) % 4096)),
                        sym(&format!("V{:04x}", (x >> 16) % 4096)),
                    ]),
                };
                atoms.insert(&key[..], ());
            }
            let index = WeightedPathIndex::from_compression_gain_on_boundaries(&atoms, 3);
            let tree = index.selection_tree().unwrap();
            let pre = index.topk_index(k_max).unwrap();

            // Global and every-node prefix, all k in 0..=k_max, must match exactly.
            let mut prefixes: Vec<Vec<u8>> = vec![Vec::new()];
            tree.nodes.keys().for_each(|p| prefixes.push(p.clone()));
            // A few prefixes that are not nodes must return empty from both.
            prefixes.push(b"\xff\xff not-a-node".to_vec());
            for prefix in &prefixes {
                for k in 0..=k_max {
                    assert_eq!(
                        pre.top_k_under(prefix, k),
                        tree.top_k_under(prefix, k),
                        "seed={seed} k={k} prefix={prefix:?}"
                    );
                }
            }
            // Clamping: a query beyond k_max returns the k_max-capped list.
            assert_eq!(pre.top_k(k_max + 5), pre.top_k(k_max));
            assert_eq!(pre.top_k(k_max), tree.top_k(k_max));
        }
    }

    // ---- W4: live maintenance under writes (S2-LIVE) ----

    fn weights_of(index: &WeightedPathIndex) -> std::collections::BTreeMap<Vec<u8>, i64> {
        let mut m = std::collections::BTreeMap::new();
        index.weights.for_each_value(|p, &w| {
            m.insert(p.to_vec(), w);
        });
        m
    }

    /// The live index must equal the batch boundary builder over the same store: same
    /// positive entries (zero-weight live leftovers aside), same snapshot top-k.
    fn assert_live_matches_batch(live: &LiveCompressionGainIndex, atoms: &PathMap<()>, ref_cost: u64) {
        let batch = WeightedPathIndex::from_compression_gain_on_boundaries(atoms, ref_cost);
        let live_w: std::collections::BTreeMap<Vec<u8>, i64> = weights_of(live.index())
            .into_iter()
            .filter(|(_, w)| *w > 0)
            .collect();
        assert_eq!(live_w, weights_of(&batch), "live weights diverge from batch");
        assert_eq!(live.current().top_k(16), batch.iter_any_topk(16).unwrap());
    }

    fn lcg_atom(x: u64) -> Vec<u8> {
        match x % 3 {
            0 => cat(&[
                arity(3),
                sym("rule"),
                cat(&[arity(2), sym("c"), sym(&format!("C{:02}", x % 8))]),
                sym(&format!("A{:03x}", (x >> 8) % 64)),
            ]),
            1 => cat(&[
                arity(3),
                sym("rule"),
                cat(&[arity(2), sym("c"), sym(&format!("C{:02}", x % 8))]),
                cat(&[arity(1), sym(&format!("B{:03x}", (x >> 8) % 64))]),
            ]),
            _ => cat(&[arity(2), sym("fact"), sym(&format!("F{:03x}", (x >> 4) % 512))]),
        }
    }

    #[test]
    fn live_index_seeded_from_store_matches_batch_builder() {
        let mut atoms: PathMap<()> = PathMap::new();
        let mut x = 42u64;
        for _ in 0..300 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            atoms.insert(&lcg_atom(x)[..], ());
        }
        let live = LiveCompressionGainIndex::from_store(&atoms, 3);
        assert_live_matches_batch(&live, &atoms, 3);
    }

    #[test]
    fn live_index_converges_to_batch_under_interleaved_mutations() {
        use std::collections::BTreeSet;
        for seed in [3u64, 17, 59] {
            let mut live = LiveCompressionGainIndex::new(2);
            let writer = live.writer();
            let mut reference: BTreeSet<Vec<u8>> = BTreeSet::new();
            let mut x = seed;
            for step in 0..800 {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let atom = lcg_atom(x);
                // Honor the genuine-mutation contract: report an insert only when the
                // atom was absent, a remove only when present.
                if x % 4 == 0 {
                    if reference.remove(&atom) {
                        writer.record_remove(&atom);
                    }
                } else if reference.insert(atom.clone()) {
                    writer.record_insert(&atom);
                }
                if step % 97 == 0 {
                    live.refresh(); // mid-stream refreshes must not disturb convergence
                }
            }
            live.refresh();
            let mut atoms: PathMap<()> = PathMap::new();
            for a in &reference {
                atoms.insert(&a[..], ());
            }
            assert_live_matches_batch(&live, &atoms, 2);
        }
    }

    #[test]
    fn live_index_is_safe_under_concurrent_writers_and_readers() {
        use std::collections::BTreeSet;
        use std::sync::{Arc, RwLock};

        let live = Arc::new(RwLock::new(LiveCompressionGainIndex::new(2)));
        // Handles are created up front; recording never touches the index lock.
        let handles: Vec<LiveGainWriter> =
            (0..4).map(|_| live.read().unwrap().writer()).collect();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // A reader hammers immutable snapshots while writers and the maintainer run.
        let reader = {
            let live = Arc::clone(&live);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut last_len = 0usize;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let snap = live.read().unwrap().current();
                    let top = snap.top_k(8);
                    last_len = top.len();
                    for w in top.windows(2) {
                        assert!(w[0].1 >= w[1].1, "snapshot top-k out of order");
                    }
                }
                last_len
            })
        };

        let mut expected: Vec<BTreeSet<Vec<u8>>> = Vec::new();
        let writers: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(t, handle)| {
                // Disjoint per-thread atom families keep every report a genuine mutation
                // without cross-thread coordination.
                let mut family: BTreeSet<Vec<u8>> = BTreeSet::new();
                let mut x = 1000 + t as u64;
                let mut ops: Vec<(bool, Vec<u8>)> = Vec::new();
                for _ in 0..300 {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let atom = cat(&[
                        arity(3),
                        sym(&format!("t{t}")),
                        sym(&format!("K{:03x}", (x >> 4) % 128)),
                        sym(&format!("V{:03x}", (x >> 16) % 128)),
                    ]);
                    if x % 4 == 0 {
                        if family.remove(&atom) {
                            ops.push((false, atom));
                        }
                    } else if family.insert(atom.clone()) {
                        ops.push((true, atom));
                    }
                }
                expected.push(family);
                std::thread::spawn(move || {
                    for (is_insert, atom) in ops {
                        if is_insert {
                            handle.record_insert(&atom);
                        } else {
                            handle.record_remove(&atom);
                        }
                    }
                })
            })
            .collect();

        // Maintainer: refresh while writers are in flight, then once after they finish.
        for _ in 0..5 {
            live.write().unwrap().refresh();
        }
        for w in writers {
            w.join().unwrap();
        }
        live.write().unwrap().refresh();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        reader.join().unwrap();

        let mut atoms: PathMap<()> = PathMap::new();
        for family in &expected {
            for a in family {
                atoms.insert(&a[..], ());
            }
        }
        assert_live_matches_batch(&live.read().unwrap(), &atoms, 2);
    }
}
