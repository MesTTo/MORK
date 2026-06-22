use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

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
        let mut index = Self::new();
        atoms.read_zipper().into_cata_side_effect(
            |_mask, children: &mut [usize], value: Option<&()>, path: &[u8]| -> usize {
                let count = value.is_some() as usize + children.iter().copied().sum::<usize>();
                if count >= 2 && !path.is_empty() {
                    let gain = (count as i64 - 1) * path.len() as i64
                        - count as i64 * ref_cost as i64;
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
}
