use crate::binding_space::{
    BindingRelation, BindingVar, TrieJoinCursorContract, TrieJoinFactorCursorContext,
    TrieJoinTrace, TrieJoinTraceShapeError,
};
use crate::sinks::{WriteResource, WriteResourceRequest};
use crate::sources::{AFactor, Resource, ResourceRequest};
use crate::term_identity::TermId;
#[cfg(feature = "einsum")]
use linalg::jit::Tensor;
use log::*;
use mork_expr::{
    Expr, ExprEnv, ExprZipper, OwnedSourceItem, Tag, UnificationFailure, byte_item, destruct,
    item_byte, maybe_byte_item, serialize, unify,
};
use mork_frontend::bytestring_parser::{Context, Parser, ParserError};
#[cfg(feature = "interning")]
use mork_interning::WritePermit;
use mork_interning::{SharedMapping, SharedMappingHandle};
use pathmap::PathMap;
use pathmap::arena_compact::ArenaCompactTree;
use pathmap::utils::{BitMask, ByteMask};
use pathmap::zipper::*;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, hash_map::Entry};
use std::fs::File;
use std::hash::Hash;
use std::hint::unreachable_unchecked;
#[cfg(feature = "z3")]
use std::io::BufRead;
use std::io::Write;
use std::mem::size_of;
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;
use std::process;
use std::ptr::slice_from_raw_parts;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use subprocess::Popen;
#[cfg(feature = "z3")]
use subprocess::{PopenConfig, Redirection};

pub static mut TRANSITIONS: usize = 0;
pub static mut UNIFICATIONS: usize = 0;
pub static mut WRITES: usize = 0;
/// Counts each time the semi-naive delta transform actually runs (the fast
/// path). Used by the Phase-6a tests to prove the soundness gate did NOT route
/// process_calculus to naive: a positive count after a run means the delta path
/// was exercised. Only touched under `semi_naive_ic`.
#[cfg(feature = "semi_naive_ic")]
pub static mut SNI_DELTA_CALLS: usize = 0;

/// Count of bodies the sidecar declined to the ProductZipper because a joined
/// relation held a schematic fact (the `any_schematic_fact_under_prefixes` gate).
/// Instrumentation for the decline-penalty benchmark; bumped only on that branch.
pub static SIDECAR_SCHEMATIC_DECLINES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Count of schematic bodies the sidecar kept OFF the ProductZipper by routing them through the
/// worst-case-optimal unification join after the zipper-owned safe gate. Bumped only on that
/// branch; the recovery counterpart of `SIDECAR_SCHEMATIC_DECLINES`.
pub static SIDECAR_UNIFY_RECOVERS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Toggle for the worst-case-optimal unification route, default OFF for a gated rollout: the
/// live behaviour is byte-for-byte unchanged until the A/B differential proves the route
/// identical to the ProductZipper, at which point the default flips on. The differential enables
/// it explicitly and the benchmark measures it against the ProductZipper (route off).
pub static SIDECAR_UNIFY_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Kernel selector for the unification route: when ON, the zipper-native join seeks the live
/// PathMap directly (zero-copy on every compatible factor, one partial re-index per inverted factor);
/// when OFF, the route decodes the joined relations into the materialized leapfrog. Both produce
/// byte-identical answers (the A/B differential asserts it); the zipper is faster on selective and
/// cyclic bodies. Default ON. A body outside the zipper's factor model falls back to the materialized
/// join regardless of this flag.
pub static SIDECAR_ZIPPER_JOIN_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[cfg(feature = "semi_naive_ic")]
thread_local! {
    /// Revert for the semi-naive IC delta: when true, `metta_calculus` forces the naive
    /// loop even with `semi_naive_ic` compiled in (the second revert next to removing
    /// the feature; `MORK_SNI=0` is the process-wide form, checked at arming).
    /// THREAD-local so a behavioral test pinning the reference path never disarms a
    /// concurrently running test's loop (the DRed corpus asserts engagement).
    pub static SNI_DISARM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Count of unification-route bodies emitted by the zipper-native kernel (the rest fell back to the
/// materialized join because they were outside the factor model). Instrumentation for the kernel A/B.
pub static SIDECAR_ZIPPER_RECOVERS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Toggle for the legacy full-unification capture route, default OFF. Native ProductZipper now
/// handles data-side capture; this route remains as an opt-in oracle and alternate emit path for
/// bodies with non-ground query compounds. It should match native ground outputs while exercising
/// the `mork_uni_join` capture join sealed against SWI-Prolog occurs-check.
pub static SIDECAR_CAPTURE_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Count of bodies emitted through the full-unification capture route. Instrumentation for the
/// capture A/B.
pub static SIDECAR_CAPTURE_RECOVERS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub static ACT_PATH: &'static str = "/dev/shm/";
// pub static ACT_PATH: &'static str = "/mnt/data/";

#[cfg(feature = "inline_unify_stack")]
type QueryUnifyStack = mork_expr::InlineUnifyStack;
#[cfg(not(feature = "inline_unify_stack"))]
type QueryUnifyStack = Vec<(ExprEnv, ExprEnv)>;

#[cfg(feature = "inline_unify_stack")]
fn query_unify_stack(first: (ExprEnv, ExprEnv)) -> QueryUnifyStack {
    let mut stack = QueryUnifyStack::new();
    stack.push(first);
    stack
}

#[cfg(not(feature = "inline_unify_stack"))]
fn query_unify_stack(first: (ExprEnv, ExprEnv)) -> QueryUnifyStack {
    vec![first]
}

/// One recursive `,`->`,` rule's entry in the semi-naive per-rule snapshot map.
/// Keyed (in `sni_rule_seen`) by the rule's pattern+template bytes; the entry
/// carries the input `snapshot` at the rule's last firing, its cached `dish_count`
/// (so the cost gate reads the dish size in O(1)), and the `caught_up_gen` the DRed
/// re-derivation catch-up uses to fire at most once per retraction.
#[cfg(feature = "semi_naive_ic")]
#[derive(Clone)]
pub(crate) struct SniRuleEntry {
    snapshot: PathMap<()>,
    dish_count: usize,
    /// The `sni_removal_gen` this rule last caught up to (DRed mode). A rule needs
    /// a full re-derivation catch-up iff `caught_up_gen < sni_removal_gen` (a
    /// removal happened since it last re-scanned the whole dish). Starts at 0.
    caught_up_gen: u64,
}

/// How the semi-naive immediate-consequence loop handles a RETRACTION (an `O`/`-`
/// rule removing a fact) mid-loop. The per-rule semi-naive delta is byte-identical
/// to naive only while evaluation stays monotone (add-only). A removal breaks the
/// recurrence (`semi_naive_delta_design.md`, Phase 6a), so by default it routes to
/// naive. This selects WHAT to do instead.
///
/// All variants write a dish byte-identical to naive on the shapes they accept; the
/// difference is HOW (and how fast). The byte-identical corpus oracle is the
/// arbiter: `Dred` only ever stays on the incremental path where it is proven
/// byte-identical, and falls back to `Naive` for any shape it does not handle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SniRetractMode {
    /// The Phase-6a default: once any removal happens in the loop, latch the
    /// soundness gate and run the naive full-space match for every `,`->`,` rule
    /// for the rest of the loop. Always correct, but O(steps x dish).
    Naive,
    /// Delete-and-Rederive (Gupta-Mumick-Subrahmanian, SIGMOD 1993): when a step's
    /// `removed` delta is non-empty, over-delete the recursive outputs that used a
    /// removed fact (cascade to a fixpoint), re-derive those still supported by the
    /// survivors, then continue on the fast incremental path. Falls back to `Naive`
    /// for any shape not proven byte-identical (the conservative safety gate). This
    /// broadens the incremental win to non-monotone (retracting) recursive programs.
    Dred,
    /// DIAGNOSTIC ONLY (never a shipped default): the raw semi-naive delta with the
    /// soundness gate DISABLED. After a removal the per-rule delta keeps running, so
    /// the dish can DIVERGE from naive (the textbook non-monotone hole). Used only by
    /// the divergence-probe and mutation tests to exhibit what DRed must repair.
    RawSemiNoGate,
}

/// Outcome of a DRed re-derivation catch-up (`sni_dred_catch_up`).
#[cfg(feature = "semi_naive_ic")]
enum DredCatchUp {
    /// The catch-up round ran; the payload is the rule's `(touched, any_new)`.
    Done((usize, bool)),
    /// This rule already re-derived since the last retraction; run the normal delta.
    AlreadyCaughtUp,
    /// The rule entry could not be reconstructed; route to the naive path.
    Fallback,
}

pub struct Space {
    pub btm: PathMap<()>,
    pub sm: SharedMappingHandle,
    pub mmaps: HashMap<OwnedSourceItem, ArenaCompactTree<memmap2::Mmap>>,
    pub z3s: HashMap<OwnedSourceItem, Box<Popen>>,
    #[cfg(feature = "einsum")]
    pub tensors: HashMap<OwnedSourceItem, Tensor>,
    pub last_merkleize: Instant,
    pub timing: bool,
    /// Persistent worst-case-optimal join sidecar, maintained incrementally
    /// across exec steps so a cyclic transform reuses interned relations instead
    /// of rebuilding from scratch. `None` until the first sidecar-routed
    /// transform builds it. Only affects performance; `btm` stays the authority.
    bridge_sidecar: Option<crate::term_identity::TermIdentitySidecar>,
    /// Monotone counter bumped whenever a transform finalizes a `RemoveSink`.
    /// The persistent sidecar observes it to close the count-equality staleness
    /// window: adds raise a relation's value count (caught by the per-prefix
    /// watermark), but an equal-count swap (a remove plus an add) leaves the
    /// count unchanged, so a bump here forces the next sidecar query to re-sync.
    /// Per-Space (not a process global), so unrelated spaces never invalidate
    /// each other and add-only recursive workloads never bump it.
    bridge_remove_gen: u64,
    /// Per-relation maintained transitive closures for streaming linear-TC rules,
    /// keyed on the edge relation prefix. The fields are the closure, how many of
    /// that relation's sidecar facts have been folded in (the watermark), and the
    /// `bridge_remove_gen` observed when last folded. A re-fire processes only the
    /// edges added since (O(new pairs)); if a removal happened (the observed
    /// generation moved), the closure is rebuilt from the live edges, since the
    /// insertion-only maintenance cannot retract a removed edge's reachability.
    /// See `roadmap_scratchpad.md`.
    bridge_closures: HashMap<
        Vec<u8>,
        (
            crate::binding_space::MaintainedTransitiveClosure,
            usize,
            u64,
        ),
    >,
    /// Per-rule input snapshots for the semi-naive immediate-consequence loop
    /// (`semi_naive_ic`). The IC loop fires one exec per step in priority order and
    /// re-arms rules across rounds, so a single global "facts added last step" delta
    /// is WRONG: when a rule fires it must match against everything added since IT
    /// last ran (which may span many intervening steps that fired OTHER rules), not
    /// just the previous step. So this maps each rule (keyed by its pattern+template
    /// bytes, which are stable across the exec re-arming) to the `btm` snapshot taken
    /// when that rule last fired. On a firing the delta is `btm \ snapshot` (the
    /// facts added since), matched one factor at a time by `transform_multi_multi_`;
    /// a first-seen rule matches the whole space (empty snapshot). After the firing
    /// the snapshot is refreshed to the current `btm`. This is exactly the m-delta-
    /// rule with the per-rule delta, so the union of delta-matches is byte-identical
    /// to the naive full re-match (the old x ... x old combinations the rule already
    /// emitted are idempotent in `btm`). `None` means "not in the IC loop" (naive
    /// full-space match), which is every default-build path and every non-IC caller.
    /// Only ever READ under `#[cfg(feature = "semi_naive_ic")]`. See
    /// `kernel/resources/semi_naive_delta_design.md`.
    ///
    /// The value is `(snapshot, snapshot_val_count)`. The cached count lets the
    /// cost gate (Phase 6b) read this rule's current dish size in O(1) instead of
    /// paying `read_copy.val_count()` (O(dish)) every firing, which a measurement
    /// showed costs ~2.6x on process_calculus. The dish count at a firing is
    /// `snapshot_count + delta_count - removed_count`, where `delta`/`removed` are
    /// the two cheap COW subtracts `read_copy \ snapshot` and `snapshot \ read_copy`
    /// (both O(facts changed)); this is EXACT even though the IC driver consumes
    /// (removes) exec facts between firings, so the dish is NOT add-only (verified
    /// 0 mismatches across process_calculus). The count is refreshed to that dish
    /// count on every firing, semi or naive-gated.
    #[cfg(feature = "semi_naive_ic")]
    sni_rule_seen: Option<HashMap<Vec<u8>, SniRuleEntry>>,
    #[cfg(not(feature = "semi_naive_ic"))]
    sni_rule_seen: Option<HashMap<Vec<u8>, (PathMap<()>, usize)>>,
    /// The semi-naive soundness gate (`semi_naive_ic`, Phase 6a). The per-rule
    /// delta (`sni_rule_seen`) is the standard Datalog semi-naive recurrence,
    /// which is byte-identical to naive ONLY for monotone (add-only) evaluation.
    /// A retraction breaks that: a fact a worker's snapshot recorded, then
    /// removed, is excluded from `btm \ snapshot`, so the closure never re-derives
    /// it through surviving facts even when naive would (the corpus
    /// `random_corpus_byte_identical` exhibits exactly this; the boundary is
    /// "any removal", confirmed over 400 seeds: 0 divergences without a removal).
    /// So: once ANY fact is retracted during the IC loop this latches `true`, and
    /// `transform_multi_multi_` then routes every `,`->`,` rule to the naive
    /// full-space match for the rest of the loop. Set at every removal site
    /// (the `O`/`-` and `I`/`O`/`-` template paths, `invalidate_bridge_caches`);
    /// cleared by `metta_calculus` when it arms the loop. After the first removal
    /// the loop is exactly the naive loop, so the feature is UNCONDITIONALLY
    /// correct: semi-naive where monotone, naive otherwise. process_calculus has
    /// no removals, so the gate never trips it and the fast path is preserved.
    /// Only ever READ under `#[cfg(feature = "semi_naive_ic")]`.
    sni_removal_seen: bool,
    /// Runtime revert switch for the semi-naive IC delta (Phase 6b default flip).
    /// The `semi_naive_ic` feature is now in the default build, so the delta + cost
    /// gate run by default. Set this `true` to force the naive full-space match for
    /// every `,`->`,` rule WITHOUT a rebuild (the firing site checks it before the
    /// delta). `false` (the default) keeps the gated semi-naive path. Always present
    /// so the toggle exists in every build; only READ under `semi_naive_ic`. The
    /// result is identical either way (both paths are byte-identical); this only
    /// chooses naive-always vs cost-gated, e.g. to A/B the lever or to fall back if
    /// a workload ever regresses. Builds without the feature ignore it (always naive).
    pub sni_force_naive: bool,
    /// Per-Space count of semi-naive delta firings (the cost gate picked semi).
    /// Bumped in lockstep with the process-global `SNI_DELTA_CALLS`, but local to
    /// this Space so a test can assert how this run routed WITHOUT racing the
    /// global counter against other tests running in parallel. Only written under
    /// `semi_naive_ic`; starts at 0 and is never reset by the engine.
    pub sni_delta_calls: usize,
    /// How the IC loop reacts to a retraction mid-loop (see `SniRetractMode`).
    /// `Naive` (the default) latches the soundness gate to the naive full-space
    /// match. `Dred` runs Delete-and-Rederive and keeps the incremental path where
    /// proven byte-identical. `RawSemiNoGate` disables the gate (diagnostic only).
    /// Only READ under `semi_naive_ic`; the result is byte-identical to naive for
    /// `Naive` and `Dred`, and may diverge for `RawSemiNoGate` (which exists solely
    /// to exhibit the divergence the gate/DRed close).
    pub sni_retract_mode: SniRetractMode,
    /// Per-Space count of DRed repair passes the IC loop ran (a removal step that
    /// `Dred` repaired incrementally instead of falling to naive). Lets a test
    /// assert DRed actually engaged. Only written under `semi_naive_ic`.
    pub sni_dred_repairs: usize,
    /// Per-Space count of removal steps `Dred` REFUSED (a shape it could not prove
    /// byte-identical), routing to the naive fallback. Lets a test assert which
    /// shapes fall back. Only written under `semi_naive_ic`.
    pub sni_dred_fallbacks: usize,
    /// Monotone generation counter bumped on every retraction during the IC loop
    /// (DRed mode). A recursive rule whose entry's `caught_up_gen` is behind this
    /// must run a full re-derivation catch-up before its incremental delta is sound
    /// again. Reset to 0 when `metta_calculus` arms the loop. Only used under
    /// `semi_naive_ic` in `Dred` mode.
    pub sni_removal_gen: u64,
    /// MUTATION-TEST HOOK (never set in production): when true, the DRed catch-up
    /// SKIPS its re-derivation naive round and only refreshes the snapshot. This
    /// makes the re-derivation non-load-bearing so a mutation test can prove the
    /// re-derivation is necessary (the dish then DIVERGES from naive on the
    /// retraction repro). Default false. Only read under `semi_naive_ic`.
    pub sni_dred_skip_rederive: bool,
    /// CONSERVATIVE-SAFETY TEST HOOK (never set in production): force the DRed
    /// catch-up to take its `Fallback` arm on the first repair, routing the rest of
    /// the loop to the naive path. Proves the conservative fallback is reachable and
    /// itself byte-identical to naive (the safety net for any shape DRed cannot
    /// prove). Default false. Only read under `semi_naive_ic`.
    pub sni_dred_force_fallback: bool,
}

pub(crate) const SIZES: [u64; 4] = {
    let mut ret = [0u64; 4];
    let mut size = 1;
    while size < 64 {
        let k = item_byte(Tag::SymbolSize(size));
        ret[((k & 0b11000000) >> 6) as usize] |= 1u64 << (k & 0b00111111);
        size += 1;
    }
    ret
};
pub(crate) const ARITIES: [u64; 4] = {
    let mut ret = [0u64; 4];
    let mut arity = 0;
    while arity < 64 {
        let k = item_byte(Tag::Arity(arity));
        ret[((k & 0b11000000) >> 6) as usize] |= 1u64 << (k & 0b00111111);
        arity += 1;
    }
    ret
};

#[cfg(not(feature = "interning"))]
fn write_symbol_bytes<W: Write>(symbol: &[u8], out: &mut W) {
    match std::str::from_utf8(symbol) {
        Ok(text) => out.write_all(text.as_bytes()).unwrap(),
        Err(_) => write!(out, "{symbol:?}").unwrap(),
    }
}

#[doc(hidden)]
pub fn write_serialized_symbol<W: Write>(sm: &SharedMappingHandle, symbol: &[u8], out: &mut W) {
    #[cfg(feature = "interning")]
    {
        let interned = i64::from_be_bytes(symbol.try_into().unwrap()).to_be_bytes();
        let bytes = sm
            .get_bytes(interned)
            .unwrap_or_else(|| panic!("failed to look up {interned:?}"));
        out.write_all(bytes).unwrap();
    }
    #[cfg(not(feature = "interning"))]
    {
        let _ = sm;
        write_symbol_bytes(symbol, out);
    }
}
pub(crate) const VARS: [u64; 4] = {
    let mut ret = [0u64; 4];
    let nv_byte = item_byte(Tag::NewVar);
    ret[((nv_byte & 0b11000000) >> 6) as usize] |= 1u64 << (nv_byte & 0b00111111);
    let mut size = 0;
    while size < 64 {
        let k = item_byte(Tag::VarRef(size));
        ret[((k & 0b11000000) >> 6) as usize] |= 1u64 << (k & 0b00111111);
        size += 1;
    }
    ret
};

// Byte-encoding facts for the fused match-any-term word-walk, derived from
// `item_byte` so they track the encoding. A child mask byte `b` lives in word
// `b >> 6` at bit `b & 0x3F`, so the byte for word `w` bit `i` is `w*64 + i`,
// i.e. `(w << 6) | i`. Each child tag class occupies exactly one word:
//   ARITY_WORD  : Arity(a)        bytes 0x00..0x3F  -> byte = a            (bit index)
//   VARREF_WORD : VarRef(i)       bytes 0x80..0xBF  -> byte = VARREF_TAG_HI | bit
//   HIGH_WORD   : NewVar 0xC0 (bit 0) + SymbolSize(s) 0xC1..0xFF (bits 1..63)
//                                                   -> byte = HIGH_TAG_HI | bit, size = bit
const ARITY_WORD: usize = (item_byte(Tag::Arity(0)) >> 6) as usize;
const VARREF_WORD: usize = (item_byte(Tag::VarRef(0)) >> 6) as usize;
const HIGH_WORD: usize = (item_byte(Tag::NewVar) >> 6) as usize;
// The high bits that, OR-ed with the in-word bit index, reconstruct the byte.
const VARREF_TAG_HI: u8 = (VARREF_WORD << 6) as u8;
const HIGH_TAG_HI: u8 = (HIGH_WORD << 6) as u8;
const NEWVAR_BYTE: u8 = item_byte(Tag::NewVar);
const NEWVAR_BIT: u64 = 1u64 << (NEWVAR_BYTE & 0b0011_1111);

#[derive(Clone, Debug)]
struct QueryFactorRank {
    estimated_cardinality: usize,
    ground_root_matches: usize,
    schematic_root_matches: usize,
    min_variable_domain_cardinality: Option<usize>,
    max_variable_domain_cardinality: Option<usize>,
    variable_domains: Vec<((u8, u8), BTreeSet<Vec<u8>>)>,
    prefix_len: usize,
    constant_items: usize,
    variable_items: usize,
    new_var_items: usize,
    var_ref_items: usize,
    prefix_cardinality_lookup: bool,
    prefix_cardinality_cache_hit: bool,
    shape_cardinality_lookup: bool,
    shape_cardinality_cache_hit: bool,
    shape_side_index_lookup: bool,
    shape_side_index_hit: bool,
    shape_side_index_insert: bool,
    shape_cardinality_scan: bool,
    shape_cardinality_refined: bool,
    shape_cardinality_skipped: bool,
    variable_domain_refined: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueryVariableOrderStep {
    variable: (u8, u8),
    domain_cardinality: usize,
    factor_domain_count: usize,
    product_upper_bound: u128,
    pruning_upper_bound: u128,
}

const QUERY_FACTOR_PLAN_CACHE_LIMIT: usize = 256;
const QUERY_SHAPE_CARDINALITY_SCAN_LIMIT: usize = 4096;
const QUERY_SHAPE_SIDE_INDEX_LIMIT: usize = 512;
const QUERY_PROJECTION_SIDE_INDEX_LIMIT: usize = 256;
const PARSER_OUTPUT_SCRATCH_INITIAL_CAPACITY: usize = 4096;
const TEMPLATE_OUTPUT_SCRATCH_INITIAL_CAPACITY: usize = 4096;
const QUERY_PRODUCT_PATH_BUFFER_INITIAL_CAPACITY: usize = 4096;
const QUERY_PRODUCT_STACK_INITIAL_DEPTH: usize = 64;

fn parser_output_buffer() -> Vec<u8> {
    Vec::with_capacity(PARSER_OUTPUT_SCRATCH_INITIAL_CAPACITY)
}

fn template_output_buffer() -> Vec<u8> {
    Vec::with_capacity(TEMPLATE_OUTPUT_SCRATCH_INITIAL_CAPACITY)
}

fn reserve_query_product_buffers<PZ: ZipperProduct + ZipperPathBuffer>(prz: &mut PZ) {
    prz.reserve_buffers(
        QUERY_PRODUCT_PATH_BUFFER_INITIAL_CAPACITY,
        QUERY_PRODUCT_STACK_INITIAL_DEPTH,
    );
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueryShapeSummary {
    cardinality: usize,
    ground_root_matches: usize,
    schematic_root_matches: usize,
    min_variable_domain_cardinality: Option<usize>,
    max_variable_domain_cardinality: Option<usize>,
    variable_domains: Vec<BTreeSet<Vec<u8>>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueryProjectionSummary {
    matches: usize,
    ground_root_matches: usize,
    schematic_root_matches: usize,
    variable_domains: Vec<BTreeSet<Vec<u8>>>,
    variable_rows: Vec<Box<[Vec<u8>]>>,
}

#[derive(Clone, Debug)]
struct QueryProjectionMaps {
    matches: usize,
    ground_root_matches: usize,
    schematic_root_matches: usize,
    variable_domains: Vec<BTreeSet<Vec<u8>>>,
    variable_maps: Vec<PathMap<()>>,
    variable_rows: Vec<Box<[Vec<u8>]>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct QueryFactorPlanCacheKey {
    factors: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct QueryShapeSideIndexKey {
    btm_val_count: usize,
    prefix_cardinality: usize,
    prefix: Vec<u8>,
    shape: Vec<u8>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct QueryProjectionSideIndexKey {
    btm_val_count: usize,
    prefix_cardinality: usize,
    prefix: Vec<u8>,
    shape: Vec<u8>,
}

#[derive(Default)]
struct QueryFactorPlanCache {
    entries: HashMap<QueryFactorPlanCacheKey, Vec<usize>>,
    hits: usize,
    misses: usize,
    inserts: usize,
}

#[derive(Default)]
struct QueryShapeSideIndex {
    entries: HashMap<QueryShapeSideIndexKey, Option<QueryShapeSummary>>,
    hits: usize,
    misses: usize,
    inserts: usize,
    clears: usize,
    generation: usize,
    max_estimated_bytes: usize,
}

#[derive(Default)]
struct QueryProjectionSideIndex {
    entries: HashMap<QueryProjectionSideIndexKey, QueryProjectionMaps>,
    hits: usize,
    misses: usize,
    inserts: usize,
    clears: usize,
    generation: usize,
    max_estimated_bytes: usize,
}

/// Read-only query factor plan cache counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryFactorPlanCacheStats {
    /// Number of cached query factor plans currently retained.
    pub entries: usize,
    /// Number of exact cache-key reuses.
    pub hits: usize,
    /// Number of exact cache-key lookups that required planning.
    pub misses: usize,
    /// Number of plan insertions into the bounded cache.
    pub inserts: usize,
}

/// Read-only reusable query-shape side-index counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryShapeSideIndexStats {
    /// Number of reusable shape summaries currently retained.
    pub entries: usize,
    /// Number of reusable shape-summary lookups that found an entry.
    pub hits: usize,
    /// Number of reusable shape-summary lookups that required a scan.
    pub misses: usize,
    /// Number of reusable shape-summary insertions.
    pub inserts: usize,
    /// Number of times the bounded side index was cleared to stay within budget.
    pub clears: usize,
    /// Current generation; increments whenever retained side-index entries are invalidated.
    pub generation: usize,
    /// Approximate retained bytes for keys, summaries, and variable-domain payloads.
    pub estimated_bytes: usize,
    /// Largest approximate retained-byte footprint observed after an insertion.
    pub max_estimated_bytes: usize,
    /// Approximate retained bytes attributable to side-index keys.
    pub key_bytes: usize,
    /// Approximate retained bytes attributable to summary metadata and domains.
    pub summary_bytes: usize,
    /// Number of exact projected variable-domain values retained in summaries.
    pub domain_values: usize,
    /// Number of exact shape scans avoided by reusable side-index hits.
    pub avoided_shape_scans: usize,
}

/// Read-only reusable query-projection side-index counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryProjectionSideIndexStats {
    /// Number of reusable projection maps currently retained.
    pub entries: usize,
    /// Number of reusable projection-map lookups that found an entry.
    pub hits: usize,
    /// Number of reusable projection-map lookups that required a scan.
    pub misses: usize,
    /// Number of reusable projection-map insertions.
    pub inserts: usize,
    /// Number of times the bounded projection index was cleared to stay within budget.
    pub clears: usize,
    /// Current generation; increments whenever retained projection entries are invalidated.
    pub generation: usize,
    /// Approximate retained bytes for keys, summaries, projection maps, and domain payloads.
    pub estimated_bytes: usize,
    /// Largest approximate retained-byte footprint observed after an insertion.
    pub max_estimated_bytes: usize,
    /// Approximate retained bytes attributable to side-index keys.
    pub key_bytes: usize,
    /// Approximate retained bytes attributable to projection maps and exact domains.
    pub projection_bytes: usize,
    /// Number of exact projected variable-domain values retained.
    pub domain_values: usize,
    /// Number of retained `PathMap<()>` projection maps.
    pub projection_maps: usize,
    /// Number of exact projection scans avoided by reusable side-index hits.
    pub avoided_projection_scans: usize,
}

#[derive(Default)]
struct QueryFactorPlanMetrics {
    plans_ranked: usize,
    factors_ranked: usize,
    prefix_cardinality_lookups: usize,
    prefix_cardinality_cache_hits: usize,
    shape_cardinality_lookups: usize,
    shape_cardinality_cache_hits: usize,
    shape_side_index_lookups: usize,
    shape_side_index_hits: usize,
    shape_side_index_inserts: usize,
    shape_cardinality_scans: usize,
    shape_cardinality_refinements: usize,
    shape_cardinality_skips: usize,
    variable_domain_refinements: usize,
    min_variable_domain_cardinality_sum: u128,
    max_variable_domain_cardinality: usize,
    shared_variable_domain_intersections: usize,
    shared_variable_domain_cardinality_sum: u128,
    max_shared_variable_domain_cardinality: usize,
    prunable_shared_variable_domains: usize,
    shared_variable_domain_product_upper_bound_sum: u128,
    shared_variable_domain_pruning_upper_bound_sum: u128,
    max_shared_variable_domain_product_upper_bound: u128,
    variable_order_plans: usize,
    variable_order_variables: usize,
    variable_order_shared_variables: usize,
    variable_order_first_domain_cardinality_sum: u128,
    variable_order_assignment_upper_bound_sum: u128,
    max_variable_order_assignment_upper_bound: u128,
    max_variable_order_domain_cardinality: usize,
    variable_order_pruning_upper_bound_sum: u128,
    unknown_cardinality_factors: usize,
    zero_cardinality_factors: usize,
    one_cardinality_factors: usize,
    le8_cardinality_factors: usize,
    le64_cardinality_factors: usize,
    le512_cardinality_factors: usize,
    le4096_cardinality_factors: usize,
    gt4096_cardinality_factors: usize,
    estimated_cardinality_sum: u128,
    max_estimated_cardinality: usize,
    max_factors_per_plan: usize,
    shape_ground_root_matches: usize,
    shape_schematic_root_matches: usize,
    all_ground_shape_factors: usize,
    schematic_shape_factors: usize,
    ground_factors: usize,
    anchored_variable_factors: usize,
    unanchored_variable_factors: usize,
    repeated_variable_factors: usize,
    pure_variable_factors: usize,
    new_var_items: usize,
    var_ref_items: usize,
    variable_items_sum: u128,
    max_variables_per_factor: usize,
    max_prefix_len: usize,
}

/// Read-only query planner cardinality counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryFactorPlanMetricsSnapshot {
    /// Number of uncached query-factor plans ranked.
    pub plans_ranked: usize,
    /// Number of individual query factors ranked.
    pub factors_ranked: usize,
    /// Prefix cardinality lookups performed against PathMap.
    pub prefix_cardinality_lookups: usize,
    /// Prefix cardinality lookups served from the planner-local prefix cache.
    pub prefix_cardinality_cache_hits: usize,
    /// Shape-refined cardinality lookups attempted for variable-bearing factors.
    pub shape_cardinality_lookups: usize,
    /// Shape-refined cardinality lookups served from the planner-local shape cache.
    pub shape_cardinality_cache_hits: usize,
    /// Shape-refined cardinality lookups attempted against the reusable side index.
    pub shape_side_index_lookups: usize,
    /// Shape-refined cardinality lookups served from the reusable side index.
    pub shape_side_index_hits: usize,
    /// Reusable side-index insertions after bounded exact shape scans.
    pub shape_side_index_inserts: usize,
    /// Shape-refined cardinality scans performed over bounded prefix subtries.
    pub shape_cardinality_scans: usize,
    /// Factors whose prefix estimate was refined by exact one-factor shape matching.
    pub shape_cardinality_refinements: usize,
    /// Shape-refined cardinality lookups skipped because the prefix was too broad.
    pub shape_cardinality_skips: usize,
    /// Factors whose bounded shape scan also produced exact projected variable domains.
    pub variable_domain_refinements: usize,
    /// Sum of the most selective projected variable-domain cardinalities per refined factor.
    pub min_variable_domain_cardinality_sum: u128,
    /// Largest projected variable-domain cardinality observed in a refined factor.
    pub max_variable_domain_cardinality: usize,
    /// Shared variables whose exact projected domains were intersected across factors.
    pub shared_variable_domain_intersections: usize,
    /// Sum of exact shared-variable domain intersection cardinalities.
    pub shared_variable_domain_cardinality_sum: u128,
    /// Largest exact shared-variable domain intersection cardinality observed.
    pub max_shared_variable_domain_cardinality: usize,
    /// Shared-variable domains whose product upper bound exceeds the exact intersection.
    pub prunable_shared_variable_domains: usize,
    /// Sum of product upper bounds across intersected shared-variable domains.
    pub shared_variable_domain_product_upper_bound_sum: u128,
    /// Sum of advisory candidate-pruning upper bounds across shared-variable domains.
    pub shared_variable_domain_pruning_upper_bound_sum: u128,
    /// Largest product upper bound observed for one shared-variable domain.
    pub max_shared_variable_domain_product_upper_bound: u128,
    /// Plans with enough exact projected domains to derive a variable order.
    pub variable_order_plans: usize,
    /// Variables included in derived variable-order plans.
    pub variable_order_variables: usize,
    /// Derived variable-order entries constrained by at least two factors.
    pub variable_order_shared_variables: usize,
    /// Sum of the first selected variable-domain cardinalities.
    pub variable_order_first_domain_cardinality_sum: u128,
    /// Sum of assignment-count upper bounds implied by derived variable orders.
    pub variable_order_assignment_upper_bound_sum: u128,
    /// Largest assignment-count upper bound implied by one derived variable order.
    pub max_variable_order_assignment_upper_bound: u128,
    /// Largest single variable-domain cardinality in a derived variable order.
    pub max_variable_order_domain_cardinality: usize,
    /// Sum of shared-domain product-minus-intersection upper bounds in variable orders.
    pub variable_order_pruning_upper_bound_sum: u128,
    /// Factors without a usable byte-prefix cardinality.
    pub unknown_cardinality_factors: usize,
    /// Factors estimated to match zero current atoms.
    pub zero_cardinality_factors: usize,
    /// Factors estimated to match exactly one current atom.
    pub one_cardinality_factors: usize,
    /// Factors estimated to match between two and eight current atoms.
    pub le8_cardinality_factors: usize,
    /// Factors estimated to match between nine and sixty-four current atoms.
    pub le64_cardinality_factors: usize,
    /// Factors estimated to match between sixty-five and five hundred twelve current atoms.
    pub le512_cardinality_factors: usize,
    /// Factors estimated to match between five hundred thirteen and four thousand ninety-six atoms.
    pub le4096_cardinality_factors: usize,
    /// Factors estimated to match more than four thousand ninety-six atoms.
    pub gt4096_cardinality_factors: usize,
    /// Sum of known per-factor estimated cardinalities.
    pub estimated_cardinality_sum: u128,
    /// Largest known per-factor estimated cardinality.
    pub max_estimated_cardinality: usize,
    /// Largest number of factors ranked for one query plan.
    pub max_factors_per_plan: usize,
    /// Matched ground candidate roots observed during exact bounded shape scans.
    pub shape_ground_root_matches: usize,
    /// Matched schematic candidate roots observed during exact bounded shape scans.
    pub shape_schematic_root_matches: usize,
    /// Shape-refined factors whose matched candidates were all ground roots.
    pub all_ground_shape_factors: usize,
    /// Shape-refined factors with at least one schematic matched root.
    pub schematic_shape_factors: usize,
    /// Factors with no variables.
    pub ground_factors: usize,
    /// Variable-bearing factors that still have a usable byte prefix.
    pub anchored_variable_factors: usize,
    /// Variable-bearing factors whose first item is variable-like.
    pub unanchored_variable_factors: usize,
    /// Factors containing at least one repeated-variable reference.
    pub repeated_variable_factors: usize,
    /// Factors made entirely from variables.
    pub pure_variable_factors: usize,
    /// New-variable items observed across ranked factors.
    pub new_var_items: usize,
    /// Variable-reference items observed across ranked factors.
    pub var_ref_items: usize,
    /// Total variable items observed across ranked factors.
    pub variable_items_sum: u128,
    /// Largest variable count observed in one factor.
    pub max_variables_per_factor: usize,
    /// Longest usable byte prefix observed in one factor.
    pub max_prefix_len: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QueryStorageBucketSnapshot {
    /// Observations of length or capacity at most eight.
    pub le8: usize,
    /// Observations of length or capacity between nine and thirty-two.
    pub le32: usize,
    /// Observations of length or capacity between thirty-three and one hundred twenty-eight.
    pub le128: usize,
    /// Observations of length or capacity between one hundred twenty-nine and five hundred twelve.
    pub le512: usize,
    /// Observations of length or capacity between five hundred thirteen and two thousand forty-eight.
    pub le2048: usize,
    /// Observations of length or capacity above two thousand forty-eight.
    pub gt2048: usize,
}

#[derive(Default)]
struct QueryStorageBucketCounts {
    le8: usize,
    le32: usize,
    le128: usize,
    le512: usize,
    le2048: usize,
    gt2048: usize,
}

impl QueryStorageBucketCounts {
    fn record(&mut self, value: usize) {
        match value {
            0..=8 => self.le8 += 1,
            9..=32 => self.le32 += 1,
            33..=128 => self.le128 += 1,
            129..=512 => self.le512 += 1,
            513..=2048 => self.le2048 += 1,
            _ => self.gt2048 += 1,
        }
    }

    fn snapshot(&self) -> QueryStorageBucketSnapshot {
        QueryStorageBucketSnapshot {
            le8: self.le8,
            le32: self.le32,
            le128: self.le128,
            le512: self.le512,
            le2048: self.le2048,
            gt2048: self.gt2048,
        }
    }
}

#[derive(Default)]
struct QueryRawStorageMetrics {
    raw_searches: usize,
    raw_stack_entries_sum: u128,
    max_raw_stack_entries: usize,
    candidate_pair_vectors: usize,
    candidate_pair_entries_sum: u128,
    candidate_pair_capacity_sum: u128,
    max_candidate_pair_entries: usize,
    max_candidate_pair_capacity: usize,
    candidate_pair_capacity: QueryStorageBucketCounts,
    general_unifications: usize,
    successful_unifications: usize,
    unification_failures: usize,
}

impl QueryRawStorageMetrics {
    fn record_raw_search(&mut self, stack_entries: usize) {
        self.raw_searches += 1;
        self.raw_stack_entries_sum += stack_entries as u128;
        self.max_raw_stack_entries = self.max_raw_stack_entries.max(stack_entries);
    }

    fn record_candidate_pairs(&mut self, entries: usize, capacity: usize) {
        self.candidate_pair_vectors += 1;
        self.candidate_pair_entries_sum += entries as u128;
        self.candidate_pair_capacity_sum += capacity as u128;
        self.max_candidate_pair_entries = self.max_candidate_pair_entries.max(entries);
        self.max_candidate_pair_capacity = self.max_candidate_pair_capacity.max(capacity);
        self.candidate_pair_capacity.record(capacity);
    }

    fn record_general_unification(&mut self, success: bool) {
        self.general_unifications += 1;
        if success {
            self.successful_unifications += 1;
        } else {
            self.unification_failures += 1;
        }
    }
}

#[derive(Default)]
struct QueryExecutionStorageMetrics {
    renormalized_plans: usize,
    renormalized_factors: usize,
    renormalized_factor_len: QueryStorageBucketCounts,
    renormalized_factor_capacity: QueryStorageBucketCounts,
    renormalized_factor_len_sum: u128,
    renormalized_factor_capacity_sum: u128,
    max_renormalized_factor_len: usize,
    max_renormalized_factor_capacity: usize,
    raw_searches: usize,
    raw_stack_entries_sum: u128,
    max_raw_stack_entries: usize,
    candidate_pair_vectors: usize,
    candidate_pair_entries_sum: u128,
    candidate_pair_capacity_sum: u128,
    max_candidate_pair_entries: usize,
    max_candidate_pair_capacity: usize,
    candidate_pair_capacity: QueryStorageBucketCounts,
    general_unifications: usize,
    successful_unifications: usize,
    unification_failures: usize,
}

/// Read-only query execution storage-shape counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryExecutionStorageMetricsSnapshot {
    /// Query-factor renormalization passes that completed.
    pub renormalized_plans: usize,
    /// Individual renormalized query-factor buffers observed.
    pub renormalized_factors: usize,
    /// Length distribution for renormalized query-factor buffers.
    pub renormalized_factor_len: QueryStorageBucketSnapshot,
    /// Capacity distribution for renormalized query-factor buffers.
    pub renormalized_factor_capacity: QueryStorageBucketSnapshot,
    /// Sum of renormalized query-factor buffer lengths.
    pub renormalized_factor_len_sum: u128,
    /// Sum of renormalized query-factor buffer capacities.
    pub renormalized_factor_capacity_sum: u128,
    /// Largest renormalized query-factor buffer length.
    pub max_renormalized_factor_len: usize,
    /// Largest retained capacity of a renormalized query-factor buffer.
    pub max_renormalized_factor_capacity: usize,
    /// Raw product-zipper searches executed.
    pub raw_searches: usize,
    /// Sum of source stack entries used to seed raw searches.
    pub raw_stack_entries_sum: u128,
    /// Largest source stack size used to seed a raw search.
    pub max_raw_stack_entries: usize,
    /// Candidate pair vectors built before unification.
    pub candidate_pair_vectors: usize,
    /// Sum of candidate pair vector lengths.
    pub candidate_pair_entries_sum: u128,
    /// Sum of candidate pair vector capacities.
    pub candidate_pair_capacity_sum: u128,
    /// Largest candidate pair vector length.
    pub max_candidate_pair_entries: usize,
    /// Largest retained capacity of a candidate pair vector.
    pub max_candidate_pair_capacity: usize,
    /// Capacity distribution for candidate pair vectors.
    pub candidate_pair_capacity: QueryStorageBucketSnapshot,
    /// General unifier calls made after product-zipper candidate construction.
    pub general_unifications: usize,
    /// General unifier calls that produced bindings.
    pub successful_unifications: usize,
    /// General unifier calls that rejected a constructed candidate.
    pub unification_failures: usize,
}

impl QueryExecutionStorageMetrics {
    fn record_renormalized_plan(&mut self, buffers: &[Vec<u8>]) {
        self.renormalized_plans += 1;
        self.renormalized_factors += buffers.len();
        for buffer in buffers {
            let len = buffer.len();
            let capacity = buffer.capacity();
            self.renormalized_factor_len.record(len);
            self.renormalized_factor_capacity.record(capacity);
            self.renormalized_factor_len_sum += len as u128;
            self.renormalized_factor_capacity_sum += capacity as u128;
            self.max_renormalized_factor_len = self.max_renormalized_factor_len.max(len);
            self.max_renormalized_factor_capacity =
                self.max_renormalized_factor_capacity.max(capacity);
        }
    }

    fn record_raw_search(&mut self, raw: &QueryRawStorageMetrics) {
        self.raw_searches += raw.raw_searches;
        self.raw_stack_entries_sum += raw.raw_stack_entries_sum;
        self.max_raw_stack_entries = self.max_raw_stack_entries.max(raw.max_raw_stack_entries);
        self.candidate_pair_vectors += raw.candidate_pair_vectors;
        self.candidate_pair_entries_sum += raw.candidate_pair_entries_sum;
        self.candidate_pair_capacity_sum += raw.candidate_pair_capacity_sum;
        self.max_candidate_pair_entries = self
            .max_candidate_pair_entries
            .max(raw.max_candidate_pair_entries);
        self.max_candidate_pair_capacity = self
            .max_candidate_pair_capacity
            .max(raw.max_candidate_pair_capacity);
        self.candidate_pair_capacity.le8 += raw.candidate_pair_capacity.le8;
        self.candidate_pair_capacity.le32 += raw.candidate_pair_capacity.le32;
        self.candidate_pair_capacity.le128 += raw.candidate_pair_capacity.le128;
        self.candidate_pair_capacity.le512 += raw.candidate_pair_capacity.le512;
        self.candidate_pair_capacity.le2048 += raw.candidate_pair_capacity.le2048;
        self.candidate_pair_capacity.gt2048 += raw.candidate_pair_capacity.gt2048;
        self.general_unifications += raw.general_unifications;
        self.successful_unifications += raw.successful_unifications;
        self.unification_failures += raw.unification_failures;
    }

    /// Merge another accumulator into this one: sum counters, max the maxima, sum bucket counts.
    /// Aggregates the per-thread accumulators into a single total at snapshot.
    fn merge(&mut self, other: &QueryExecutionStorageMetrics) {
        self.renormalized_plans += other.renormalized_plans;
        self.renormalized_factors += other.renormalized_factors;
        self.renormalized_factor_len.le8 += other.renormalized_factor_len.le8;
        self.renormalized_factor_len.le32 += other.renormalized_factor_len.le32;
        self.renormalized_factor_len.le128 += other.renormalized_factor_len.le128;
        self.renormalized_factor_len.le512 += other.renormalized_factor_len.le512;
        self.renormalized_factor_len.le2048 += other.renormalized_factor_len.le2048;
        self.renormalized_factor_len.gt2048 += other.renormalized_factor_len.gt2048;
        self.renormalized_factor_capacity.le8 += other.renormalized_factor_capacity.le8;
        self.renormalized_factor_capacity.le32 += other.renormalized_factor_capacity.le32;
        self.renormalized_factor_capacity.le128 += other.renormalized_factor_capacity.le128;
        self.renormalized_factor_capacity.le512 += other.renormalized_factor_capacity.le512;
        self.renormalized_factor_capacity.le2048 += other.renormalized_factor_capacity.le2048;
        self.renormalized_factor_capacity.gt2048 += other.renormalized_factor_capacity.gt2048;
        self.renormalized_factor_len_sum += other.renormalized_factor_len_sum;
        self.renormalized_factor_capacity_sum += other.renormalized_factor_capacity_sum;
        self.max_renormalized_factor_len = self
            .max_renormalized_factor_len
            .max(other.max_renormalized_factor_len);
        self.max_renormalized_factor_capacity = self
            .max_renormalized_factor_capacity
            .max(other.max_renormalized_factor_capacity);
        self.raw_searches += other.raw_searches;
        self.raw_stack_entries_sum += other.raw_stack_entries_sum;
        self.max_raw_stack_entries = self.max_raw_stack_entries.max(other.max_raw_stack_entries);
        self.candidate_pair_vectors += other.candidate_pair_vectors;
        self.candidate_pair_entries_sum += other.candidate_pair_entries_sum;
        self.candidate_pair_capacity_sum += other.candidate_pair_capacity_sum;
        self.max_candidate_pair_entries = self
            .max_candidate_pair_entries
            .max(other.max_candidate_pair_entries);
        self.max_candidate_pair_capacity = self
            .max_candidate_pair_capacity
            .max(other.max_candidate_pair_capacity);
        self.candidate_pair_capacity.le8 += other.candidate_pair_capacity.le8;
        self.candidate_pair_capacity.le32 += other.candidate_pair_capacity.le32;
        self.candidate_pair_capacity.le128 += other.candidate_pair_capacity.le128;
        self.candidate_pair_capacity.le512 += other.candidate_pair_capacity.le512;
        self.candidate_pair_capacity.le2048 += other.candidate_pair_capacity.le2048;
        self.candidate_pair_capacity.gt2048 += other.candidate_pair_capacity.gt2048;
        self.general_unifications += other.general_unifications;
        self.successful_unifications += other.successful_unifications;
        self.unification_failures += other.unification_failures;
    }

    fn snapshot(&self) -> QueryExecutionStorageMetricsSnapshot {
        QueryExecutionStorageMetricsSnapshot {
            renormalized_plans: self.renormalized_plans,
            renormalized_factors: self.renormalized_factors,
            renormalized_factor_len: self.renormalized_factor_len.snapshot(),
            renormalized_factor_capacity: self.renormalized_factor_capacity.snapshot(),
            renormalized_factor_len_sum: self.renormalized_factor_len_sum,
            renormalized_factor_capacity_sum: self.renormalized_factor_capacity_sum,
            max_renormalized_factor_len: self.max_renormalized_factor_len,
            max_renormalized_factor_capacity: self.max_renormalized_factor_capacity,
            raw_searches: self.raw_searches,
            raw_stack_entries_sum: self.raw_stack_entries_sum,
            max_raw_stack_entries: self.max_raw_stack_entries,
            candidate_pair_vectors: self.candidate_pair_vectors,
            candidate_pair_entries_sum: self.candidate_pair_entries_sum,
            candidate_pair_capacity_sum: self.candidate_pair_capacity_sum,
            max_candidate_pair_entries: self.max_candidate_pair_entries,
            max_candidate_pair_capacity: self.max_candidate_pair_capacity,
            candidate_pair_capacity: self.candidate_pair_capacity.snapshot(),
            general_unifications: self.general_unifications,
            successful_unifications: self.successful_unifications,
            unification_failures: self.unification_failures,
        }
    }
}

impl QueryFactorPlanMetrics {
    fn record_plan(&mut self, ranks: &[QueryFactorRank]) {
        self.plans_ranked += 1;
        self.factors_ranked += ranks.len();
        self.max_factors_per_plan = self.max_factors_per_plan.max(ranks.len());

        for rank in ranks {
            self.max_prefix_len = self.max_prefix_len.max(rank.prefix_len);
            self.max_variables_per_factor = self.max_variables_per_factor.max(rank.variable_items);
            self.variable_items_sum += rank.variable_items as u128;
            self.new_var_items += rank.new_var_items;
            self.var_ref_items += rank.var_ref_items;
            if rank.variable_items == 0 {
                self.ground_factors += 1;
            } else if rank.prefix_len == 0 {
                self.unanchored_variable_factors += 1;
            } else {
                self.anchored_variable_factors += 1;
            }
            if rank.var_ref_items > 0 {
                self.repeated_variable_factors += 1;
            }
            if rank.variable_items > 0 && rank.constant_items == 0 {
                self.pure_variable_factors += 1;
            }
            if rank.prefix_cardinality_lookup {
                self.prefix_cardinality_lookups += 1;
            }
            if rank.prefix_cardinality_cache_hit {
                self.prefix_cardinality_cache_hits += 1;
            }
            if rank.shape_cardinality_lookup {
                self.shape_cardinality_lookups += 1;
            }
            if rank.shape_cardinality_cache_hit {
                self.shape_cardinality_cache_hits += 1;
            }
            if rank.shape_side_index_lookup {
                self.shape_side_index_lookups += 1;
            }
            if rank.shape_side_index_hit {
                self.shape_side_index_hits += 1;
            }
            if rank.shape_side_index_insert {
                self.shape_side_index_inserts += 1;
            }
            if rank.shape_cardinality_scan {
                self.shape_cardinality_scans += 1;
            }
            if rank.shape_cardinality_refined {
                self.shape_cardinality_refinements += 1;
            }
            if rank.shape_cardinality_skipped {
                self.shape_cardinality_skips += 1;
            }
            if rank.variable_domain_refined {
                self.variable_domain_refinements += 1;
            }
            if rank.shape_cardinality_refined {
                self.shape_ground_root_matches += rank.ground_root_matches;
                self.shape_schematic_root_matches += rank.schematic_root_matches;
                if rank.schematic_root_matches > 0 {
                    self.schematic_shape_factors += 1;
                } else if rank.ground_root_matches > 0 {
                    self.all_ground_shape_factors += 1;
                }
            }
            if let Some(min_cardinality) = rank.min_variable_domain_cardinality {
                self.min_variable_domain_cardinality_sum += min_cardinality as u128;
            }
            if let Some(max_cardinality) = rank.max_variable_domain_cardinality {
                self.max_variable_domain_cardinality =
                    self.max_variable_domain_cardinality.max(max_cardinality);
            }
            match rank.estimated_cardinality {
                usize::MAX => self.unknown_cardinality_factors += 1,
                0 => self.zero_cardinality_factors += 1,
                1 => self.one_cardinality_factors += 1,
                2..=8 => self.le8_cardinality_factors += 1,
                9..=64 => self.le64_cardinality_factors += 1,
                65..=512 => self.le512_cardinality_factors += 1,
                513..=4096 => self.le4096_cardinality_factors += 1,
                cardinality => {
                    self.gt4096_cardinality_factors += 1;
                    self.max_estimated_cardinality =
                        self.max_estimated_cardinality.max(cardinality);
                    self.estimated_cardinality_sum += cardinality as u128;
                    continue;
                }
            }

            if rank.estimated_cardinality != usize::MAX {
                self.max_estimated_cardinality = self
                    .max_estimated_cardinality
                    .max(rank.estimated_cardinality);
                self.estimated_cardinality_sum += rank.estimated_cardinality as u128;
            }
        }
        self.record_shared_variable_domains(ranks);
        self.record_variable_order(ranks);
    }

    fn variable_order_steps(ranks: &[QueryFactorRank]) -> Vec<QueryVariableOrderStep> {
        let mut domains_by_var: BTreeMap<(u8, u8), Vec<&BTreeSet<Vec<u8>>>> = BTreeMap::new();
        for rank in ranks {
            for (var, domain) in &rank.variable_domains {
                if !domain.is_empty() {
                    domains_by_var.entry(*var).or_default().push(domain);
                }
            }
        }

        let mut steps = Vec::new();
        for (variable, domains) in domains_by_var {
            let mut sorted_domains = domains;
            sorted_domains.sort_unstable_by_key(|domain| domain.len());
            let Some((smallest, rest)) = sorted_domains.split_first() else {
                continue;
            };

            let domain_cardinality = if rest.is_empty() {
                smallest.len()
            } else {
                let mut intersection_cardinality = 0usize;
                for value in smallest.iter() {
                    if rest.iter().all(|domain| domain.contains(value)) {
                        intersection_cardinality += 1;
                    }
                }
                intersection_cardinality
            };
            let product_upper_bound = sorted_domains.iter().fold(1u128, |product, domain| {
                product.saturating_mul(domain.len() as u128)
            });
            let pruning_upper_bound =
                product_upper_bound.saturating_sub(domain_cardinality as u128);

            steps.push(QueryVariableOrderStep {
                variable,
                domain_cardinality,
                factor_domain_count: sorted_domains.len(),
                product_upper_bound,
                pruning_upper_bound,
            });
        }

        steps.sort_by(|lhs, rhs| {
            lhs.domain_cardinality
                .cmp(&rhs.domain_cardinality)
                .then_with(|| rhs.factor_domain_count.cmp(&lhs.factor_domain_count))
                .then_with(|| lhs.variable.cmp(&rhs.variable))
        });
        steps
    }

    fn record_shared_variable_domains(&mut self, ranks: &[QueryFactorRank]) {
        for step in Self::variable_order_steps(ranks) {
            if step.factor_domain_count < 2 {
                continue;
            }

            self.shared_variable_domain_intersections += 1;
            self.shared_variable_domain_cardinality_sum += step.domain_cardinality as u128;
            self.max_shared_variable_domain_cardinality = self
                .max_shared_variable_domain_cardinality
                .max(step.domain_cardinality);
            if step.pruning_upper_bound > 0 {
                self.prunable_shared_variable_domains += 1;
            }
            self.shared_variable_domain_product_upper_bound_sum = self
                .shared_variable_domain_product_upper_bound_sum
                .saturating_add(step.product_upper_bound);
            self.shared_variable_domain_pruning_upper_bound_sum = self
                .shared_variable_domain_pruning_upper_bound_sum
                .saturating_add(step.pruning_upper_bound);
            self.max_shared_variable_domain_product_upper_bound = self
                .max_shared_variable_domain_product_upper_bound
                .max(step.product_upper_bound);
        }
    }

    fn record_variable_order(&mut self, ranks: &[QueryFactorRank]) {
        let steps = Self::variable_order_steps(ranks);
        let Some(first) = steps.first() else {
            return;
        };

        self.variable_order_plans += 1;
        self.variable_order_variables += steps.len();
        self.variable_order_first_domain_cardinality_sum += first.domain_cardinality as u128;

        let mut assignment_upper_bound = 1u128;
        for step in steps {
            if step.factor_domain_count > 1 {
                self.variable_order_shared_variables += 1;
            }
            assignment_upper_bound =
                assignment_upper_bound.saturating_mul(step.domain_cardinality as u128);
            self.max_variable_order_domain_cardinality = self
                .max_variable_order_domain_cardinality
                .max(step.domain_cardinality);
            self.variable_order_pruning_upper_bound_sum = self
                .variable_order_pruning_upper_bound_sum
                .saturating_add(step.pruning_upper_bound);
        }
        self.variable_order_assignment_upper_bound_sum = self
            .variable_order_assignment_upper_bound_sum
            .saturating_add(assignment_upper_bound);
        self.max_variable_order_assignment_upper_bound = self
            .max_variable_order_assignment_upper_bound
            .max(assignment_upper_bound);
    }

    fn snapshot(&self) -> QueryFactorPlanMetricsSnapshot {
        QueryFactorPlanMetricsSnapshot {
            plans_ranked: self.plans_ranked,
            factors_ranked: self.factors_ranked,
            prefix_cardinality_lookups: self.prefix_cardinality_lookups,
            prefix_cardinality_cache_hits: self.prefix_cardinality_cache_hits,
            shape_cardinality_lookups: self.shape_cardinality_lookups,
            shape_cardinality_cache_hits: self.shape_cardinality_cache_hits,
            shape_side_index_lookups: self.shape_side_index_lookups,
            shape_side_index_hits: self.shape_side_index_hits,
            shape_side_index_inserts: self.shape_side_index_inserts,
            shape_cardinality_scans: self.shape_cardinality_scans,
            shape_cardinality_refinements: self.shape_cardinality_refinements,
            shape_cardinality_skips: self.shape_cardinality_skips,
            variable_domain_refinements: self.variable_domain_refinements,
            min_variable_domain_cardinality_sum: self.min_variable_domain_cardinality_sum,
            max_variable_domain_cardinality: self.max_variable_domain_cardinality,
            shared_variable_domain_intersections: self.shared_variable_domain_intersections,
            shared_variable_domain_cardinality_sum: self.shared_variable_domain_cardinality_sum,
            max_shared_variable_domain_cardinality: self.max_shared_variable_domain_cardinality,
            prunable_shared_variable_domains: self.prunable_shared_variable_domains,
            shared_variable_domain_product_upper_bound_sum: self
                .shared_variable_domain_product_upper_bound_sum,
            shared_variable_domain_pruning_upper_bound_sum: self
                .shared_variable_domain_pruning_upper_bound_sum,
            max_shared_variable_domain_product_upper_bound: self
                .max_shared_variable_domain_product_upper_bound,
            variable_order_plans: self.variable_order_plans,
            variable_order_variables: self.variable_order_variables,
            variable_order_shared_variables: self.variable_order_shared_variables,
            variable_order_first_domain_cardinality_sum: self
                .variable_order_first_domain_cardinality_sum,
            variable_order_assignment_upper_bound_sum: self
                .variable_order_assignment_upper_bound_sum,
            max_variable_order_assignment_upper_bound: self
                .max_variable_order_assignment_upper_bound,
            max_variable_order_domain_cardinality: self.max_variable_order_domain_cardinality,
            variable_order_pruning_upper_bound_sum: self.variable_order_pruning_upper_bound_sum,
            unknown_cardinality_factors: self.unknown_cardinality_factors,
            zero_cardinality_factors: self.zero_cardinality_factors,
            one_cardinality_factors: self.one_cardinality_factors,
            le8_cardinality_factors: self.le8_cardinality_factors,
            le64_cardinality_factors: self.le64_cardinality_factors,
            le512_cardinality_factors: self.le512_cardinality_factors,
            le4096_cardinality_factors: self.le4096_cardinality_factors,
            gt4096_cardinality_factors: self.gt4096_cardinality_factors,
            estimated_cardinality_sum: self.estimated_cardinality_sum,
            max_estimated_cardinality: self.max_estimated_cardinality,
            max_factors_per_plan: self.max_factors_per_plan,
            shape_ground_root_matches: self.shape_ground_root_matches,
            shape_schematic_root_matches: self.shape_schematic_root_matches,
            all_ground_shape_factors: self.all_ground_shape_factors,
            schematic_shape_factors: self.schematic_shape_factors,
            ground_factors: self.ground_factors,
            anchored_variable_factors: self.anchored_variable_factors,
            unanchored_variable_factors: self.unanchored_variable_factors,
            repeated_variable_factors: self.repeated_variable_factors,
            pure_variable_factors: self.pure_variable_factors,
            new_var_items: self.new_var_items,
            var_ref_items: self.var_ref_items,
            variable_items_sum: self.variable_items_sum,
            max_variables_per_factor: self.max_variables_per_factor,
            max_prefix_len: self.max_prefix_len,
        }
    }
}

impl QueryFactorPlanCache {
    fn get(&mut self, key: &QueryFactorPlanCacheKey) -> Option<Vec<usize>> {
        if let Some(plan) = self.entries.get(key) {
            self.hits += 1;
            Some(plan.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, key: QueryFactorPlanCacheKey, plan: &[usize]) {
        insert_bounded_cache_entry(
            &mut self.entries,
            QUERY_FACTOR_PLAN_CACHE_LIMIT,
            key,
            plan.to_vec(),
        );
        self.inserts += 1;
    }

    fn stats(&self) -> QueryFactorPlanCacheStats {
        QueryFactorPlanCacheStats {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            inserts: self.inserts,
        }
    }
}

impl QueryShapeSideIndexKey {
    fn estimated_bytes(&self) -> usize {
        query_side_index_key_estimated_bytes::<Self>(&self.prefix, &self.shape)
    }
}

impl QueryProjectionSideIndexKey {
    fn estimated_bytes(&self) -> usize {
        query_side_index_key_estimated_bytes::<Self>(&self.prefix, &self.shape)
    }
}

fn query_side_index_key_estimated_bytes<K>(prefix: &[u8], shape: &[u8]) -> usize {
    size_of::<K>() + prefix.len() + shape.len()
}

fn encoded_byte_value_estimated_bytes(value: &[u8]) -> usize {
    size_of::<Vec<u8>>() + value.len()
}

fn query_variable_domains_estimated_bytes(domains: &[BTreeSet<Vec<u8>>]) -> usize {
    domains.len() * size_of::<BTreeSet<Vec<u8>>>()
        + domains
            .iter()
            .flat_map(|domain| domain.iter())
            .map(|value| encoded_byte_value_estimated_bytes(value))
            .sum::<usize>()
}

fn query_variable_rows_estimated_bytes(rows: &[Box<[Vec<u8>]>]) -> usize {
    rows.iter()
        .flat_map(|row| row.iter())
        .map(|value| encoded_byte_value_estimated_bytes(value))
        .sum::<usize>()
}

fn query_variable_domain_value_count(domains: &[BTreeSet<Vec<u8>>]) -> usize {
    domains.iter().map(BTreeSet::len).sum::<usize>()
}

impl QueryShapeSummary {
    fn estimated_bytes(&self) -> usize {
        size_of::<Self>() + query_variable_domains_estimated_bytes(&self.variable_domains)
    }

    fn domain_value_count(&self) -> usize {
        query_variable_domain_value_count(&self.variable_domains)
    }
}

impl QueryProjectionMaps {
    fn to_shape_summary(&self) -> QueryShapeSummary {
        let mut min_variable_domain_cardinality = None;
        let mut max_variable_domain_cardinality = None;
        for domain in self.variable_domains.iter() {
            let cardinality = domain.len();
            if cardinality == 0 {
                continue;
            }
            min_variable_domain_cardinality = Some(
                min_variable_domain_cardinality
                    .map_or(cardinality, |current: usize| current.min(cardinality)),
            );
            max_variable_domain_cardinality = Some(
                max_variable_domain_cardinality
                    .map_or(cardinality, |current: usize| current.max(cardinality)),
            );
        }

        QueryShapeSummary {
            cardinality: self.matches,
            ground_root_matches: self.ground_root_matches,
            schematic_root_matches: self.schematic_root_matches,
            min_variable_domain_cardinality,
            max_variable_domain_cardinality,
            variable_domains: self.variable_domains.clone(),
        }
    }

    fn estimated_bytes(&self) -> usize {
        size_of::<Self>()
            + query_variable_domains_estimated_bytes(&self.variable_domains)
            + self.variable_maps.len() * size_of::<PathMap<()>>()
            + query_variable_rows_estimated_bytes(&self.variable_rows)
    }

    fn domain_value_count(&self) -> usize {
        query_variable_domain_value_count(&self.variable_domains)
    }
}

/// Diagnostic result for LFTJ-style intersection over PathMap projection domains.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QueryProjectionDomainIntersection {
    /// Ordered byte-domain values that survived every input domain.
    pub values: Vec<Vec<u8>>,
    /// Number of input domains.
    pub domain_sources: usize,
    /// Total input-domain values presented before intersection.
    pub domain_values: usize,
    /// Domain cursors opened.
    pub cursor_opens: usize,
    /// Monotone seek calls issued.
    pub cursor_seeks: usize,
    /// Domain values skipped by seeks.
    pub cursor_skips: usize,
    /// Next calls issued after aligned values.
    pub cursor_nexts: usize,
}

/// Minimal ordered cursor over one projected byte-domain.
///
/// This mirrors the TermId-valued [`BindingDomainCursor`] contract for
/// query-projection byte values. Implementors must expose keys in ascending
/// lexicographic order and make `seek` monotone: after seeking to `target`, the
/// cursor is either at end or positioned at the least available key greater than
/// or equal to `target`.
pub trait QueryProjectionByteDomainCursor {
    /// Current byte key, or `None` when the cursor is exhausted.
    fn key(&self) -> Option<&[u8]>;
    /// Whether this cursor has no current key.
    fn at_end(&self) -> bool;
    /// Advance to the next key.
    fn next(&mut self);
    /// Advance to the least key greater than or equal to `target`.
    ///
    /// Returns the number of domain values skipped while advancing.
    fn seek(&mut self, target: &[u8]) -> usize;
    /// Number of values in the opened domain, used for diagnostic counters.
    fn domain_len(&self) -> usize;
}

/// Diagnostic byte-domain opened from one projected query-factor relation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QueryProjectionRelationDomain {
    /// Ordered byte-domain values available for the requested variable.
    pub values: Vec<Vec<u8>>,
    /// Variable index in the projected factor-local row order.
    pub variable_index: usize,
    /// Number of earlier variables supplied as the bound prefix.
    pub bound_prefix_len: usize,
    /// Projected rows available in the relation factor.
    pub rows: usize,
    /// Rows whose prefix matched the requested binding context.
    pub rows_matching_prefix: usize,
}

/// Work counters for the diagnostic zipper-backed query projection opener.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QueryProjectionZipperDomainTelemetry {
    /// Domain open requests, including requests served from cache.
    pub opens: usize,
    /// Opens served from a cached domain for the same local bound prefix.
    pub cache_hits: usize,
    /// `ReadZipper` scans started after a cache miss.
    pub scans: usize,
    /// Direct `ReadZipper` scans started after a cache miss.
    pub read_zipper_scans: usize,
    /// Single-factor `ProductZipperG` scans started after a cache miss.
    pub product_zipper_scans: usize,
    /// Candidate paths visited by actual scans.
    pub candidates: usize,
    /// Exact unifier calls made by actual scans.
    pub unifications: usize,
    /// Complete binding rows recovered by actual scans.
    pub rows: usize,
    /// Complete scanned rows whose local prefix matched the open request.
    pub rows_matching_prefix: usize,
    /// Domain values returned to callers, including cache hits.
    pub domain_values: usize,
}

/// Summary from comparing byte-domain relation factors with a trie cursor contract.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QueryProjectionTrieContractComparison {
    /// Relation factors represented by the trie contract.
    pub relation_indexes: usize,
    /// Per-factor requirements inspected from the contract.
    pub factor_requirements: usize,
    /// Bound-prefix domain contexts checked.
    pub contexts: usize,
    /// Contexts whose opened byte domain matched the contract exactly.
    pub matched_contexts: usize,
    /// Contexts with missing factors, missing term mappings, or domain mismatch.
    pub mismatched_contexts: usize,
    /// Contract factor requirements without a corresponding byte-domain factor.
    pub missing_factors: usize,
    /// Term IDs or bound-prefix variables that could not be mapped into byte domains.
    pub missing_term_mappings: usize,
    /// Per-context comparison diagnostics.
    pub context_results: Vec<QueryProjectionTrieContractContextComparison>,
}

/// Per-context comparison between one trie-domain obligation and one byte factor open.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryProjectionTrieContractContextComparison {
    /// Relation/factor index in the trie contract.
    pub relation_index: usize,
    /// Original trie-trace step index.
    pub step_index: usize,
    /// Variable whose domain was opened.
    pub variable: BindingVar,
    /// Number of values already bound in the global variable order.
    pub bound_prefix_len: usize,
    /// Expected ordered byte-domain values after mapping from `TermId`.
    pub expected_domain: Vec<Vec<u8>>,
    /// Actual ordered byte-domain values opened from the projection factor.
    pub actual_domain: Vec<Vec<u8>>,
    /// Missing `TermId` mappings or bound-prefix variables seen in this context.
    pub missing_term_mappings: usize,
    /// Whether this context matched exactly.
    pub matched: bool,
}

/// Successful binding rows observed through the current ProductZipper query path.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QueryProjectionProductCandidateTrace {
    /// Number of relation factors in the product query.
    pub factor_count: usize,
    /// Global variable order used to encode every row.
    pub variable_order: Box<[BindingVar]>,
    /// Successful ProductZipper candidates reported by `query_multi`.
    pub successful_candidates: usize,
    /// Successful candidates whose bindings could be encoded in `variable_order`.
    pub rows: Vec<Box<[Vec<u8>]>>,
    /// Deduplicated successful rows.
    pub unique_rows: Vec<Box<[Vec<u8>]>>,
    /// Successful candidates missing at least one requested binding.
    pub missing_binding_rows: usize,
    /// Product results that did not produce a binding map.
    pub non_binding_results: usize,
    /// Raw pre-unification candidate counters from the ProductZipper traversal.
    pub raw: QueryProjectionProductRawCandidateCounters,
}

/// Per-query raw candidate counters for ProductZipper-backed traversal.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QueryProjectionProductRawCandidateCounters {
    /// Product candidates that reached candidate-pair construction.
    pub raw_candidates: usize,
    /// Candidate pair vectors built before general unification.
    pub candidate_pair_vectors: usize,
    /// Sum of candidate pair vector lengths.
    pub candidate_pair_entries_sum: u128,
    /// Largest candidate pair vector length.
    pub max_candidate_pair_entries: usize,
    /// General unifier calls made after candidate construction.
    pub general_unifications: usize,
    /// General unifier calls that produced bindings.
    pub successful_unifications: usize,
    /// General unifier calls that rejected a constructed candidate.
    pub rejected_unifications: usize,
    /// Rejections caused by occurs-check failures.
    pub occurs_rejections: usize,
    /// Rejections caused by structural differences.
    pub difference_rejections: usize,
    /// Rejections caused by reaching the unifier iteration limit.
    pub max_iter_rejections: usize,
}

impl QueryProjectionProductRawCandidateCounters {
    fn record_candidate_pairs(&mut self, entries: usize) {
        self.raw_candidates += 1;
        self.candidate_pair_vectors += 1;
        self.candidate_pair_entries_sum += entries as u128;
        self.max_candidate_pair_entries = self.max_candidate_pair_entries.max(entries);
    }

    fn record_successful_unification(&mut self) {
        self.general_unifications += 1;
        self.successful_unifications += 1;
    }

    fn record_unification_failure(&mut self, failure: &UnificationFailure) {
        self.general_unifications += 1;
        self.rejected_unifications += 1;
        match failure {
            UnificationFailure::Occurs(_, _) => self.occurs_rejections += 1,
            UnificationFailure::Difference(_, _) => self.difference_rejections += 1,
            UnificationFailure::MaxIter(_) => self.max_iter_rejections += 1,
        }
    }
}

/// Comparison between a ProductZipper candidate trace and a BindingSpace relation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QueryProjectionProductTraceComparison {
    /// Successful ProductZipper candidates reported by the trace.
    pub product_successful_candidates: usize,
    /// Raw ProductZipper candidates that reached the general unifier.
    pub product_raw_candidates: usize,
    /// Raw ProductZipper candidates rejected by the general unifier.
    pub product_rejected_candidates: usize,
    /// Encodable ProductZipper rows, including duplicates.
    pub product_rows: usize,
    /// Deduplicated ProductZipper rows.
    pub product_unique_rows: usize,
    /// Positive rows in the BindingSpace relation after byte mapping.
    pub expected_rows: usize,
    /// Missing `TermId` mappings while converting BindingSpace rows to bytes.
    pub missing_term_mappings: usize,
    /// Successful ProductZipper candidates missing requested bindings.
    pub missing_binding_rows: usize,
    /// Product results that did not produce a binding map.
    pub non_binding_results: usize,
    /// Deduplicated ProductZipper rows in `trace.variable_order`.
    pub actual_domain: Vec<Box<[Vec<u8>]>>,
    /// BindingSpace rows mapped into `trace.variable_order`.
    pub expected_domain: Vec<Box<[Vec<u8>]>>,
    /// Whether the deduplicated ProductZipper rows match the BindingSpace rows.
    pub matched: bool,
}

/// Explain-only comparison between current ProductZipper work and trie trace work.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QueryProjectionProductVsTrieTraceComparison {
    /// Raw ProductZipper candidates that reached the general unifier.
    pub product_raw_candidates: usize,
    /// Successful ProductZipper candidates reported by the trace.
    pub product_successful_candidates: usize,
    /// Raw ProductZipper candidates rejected by the general unifier.
    pub product_rejected_candidates: usize,
    /// Deduplicated ProductZipper byte rows.
    pub product_unique_rows: usize,
    /// Complete satisfying bindings reached by the trie trace.
    pub trie_candidate_bindings: usize,
    /// Domain-intersection contexts visited by the trie trace.
    pub trie_steps: usize,
    /// Relation domains participating in trie intersections.
    pub trie_domain_sources: usize,
    /// Domain values presented to trie leapfrog intersections.
    pub trie_domain_values: usize,
    /// Monotone cursor seeks recorded by the trie trace.
    pub trie_cursor_seeks: usize,
    /// Domain values skipped by trie cursor seeks.
    pub trie_cursor_skips: usize,
    /// Whether successful ProductZipper candidates equal trie candidate bindings.
    pub successful_candidate_counts_match: bool,
    /// Whether deduplicated ProductZipper rows equal trie candidate bindings.
    pub unique_row_counts_match: bool,
    /// Raw ProductZipper candidates above trie candidate bindings.
    pub raw_candidate_overhead: usize,
}

/// Owned byte-domain cursor for projection maps and relation-factor domains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryProjectionDomainCursor {
    domain: Vec<Vec<u8>>,
    position: usize,
}

impl QueryProjectionDomainCursor {
    /// Opens a sorted byte-domain cursor from every value path in a `PathMap`.
    pub fn from_pathmap(map: &PathMap<()>) -> Self {
        let mut rz = map.read_zipper();
        let mut domain = Vec::with_capacity(map.val_count());
        while rz.to_next_val() {
            domain.push(rz.path().to_vec());
        }
        Self::from_values(domain)
    }

    /// Opens a sorted, deduplicated byte-domain cursor from owned values.
    pub fn from_values(mut domain: Vec<Vec<u8>>) -> Self {
        domain.sort();
        domain.dedup();
        Self {
            domain,
            position: 0,
        }
    }

    /// Total values in the opened domain.
    pub fn domain_len(&self) -> usize {
        self.domain.len()
    }

    /// Current cursor position inside the opened domain.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Current byte key, or `None` when the cursor is exhausted.
    pub fn key(&self) -> Option<&[u8]> {
        self.domain.get(self.position).map(Vec::as_slice)
    }

    /// Whether this cursor has no current key.
    pub fn at_end(&self) -> bool {
        self.position >= self.domain.len()
    }

    /// Advance to the next key if one is available.
    pub fn next(&mut self) {
        if !self.at_end() {
            self.position += 1;
        }
    }

    /// Advance to the least key greater than or equal to `target`.
    pub fn seek(&mut self, target: &[u8]) -> usize {
        if self.at_end() {
            return 0;
        }

        let next_position = self.position
            + self.domain[self.position..].partition_point(|value| value.as_slice() < target);
        let skipped = next_position - self.position;
        self.position = next_position;
        skipped
    }
}

impl QueryProjectionByteDomainCursor for QueryProjectionDomainCursor {
    fn key(&self) -> Option<&[u8]> {
        QueryProjectionDomainCursor::key(self)
    }

    fn at_end(&self) -> bool {
        QueryProjectionDomainCursor::at_end(self)
    }

    fn next(&mut self) {
        QueryProjectionDomainCursor::next(self);
    }

    fn seek(&mut self, target: &[u8]) -> usize {
        QueryProjectionDomainCursor::seek(self, target)
    }

    fn domain_len(&self) -> usize {
        QueryProjectionDomainCursor::domain_len(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryProjectionRelationFactor {
    variables: Box<[BindingVar]>,
    rows: Vec<Box<[Vec<u8>]>>,
}

trait QueryProjectionByteDomainFactor {
    fn open_domain_for_variable(
        &self,
        variable: BindingVar,
        bound_variables: &[(BindingVar, Vec<u8>)],
    ) -> QueryProjectionRelationDomain;
}

impl QueryProjectionRelationFactor {
    /// Builds a diagnostic relation factor from exact projected byte rows.
    ///
    /// Positional variables are assigned in ascending `BindingVar` order. Use
    /// `from_rows_with_variables` when a factor has a physical order selected
    /// by a trie cursor or arrangement contract.
    ///
    /// Rows whose length does not match `variable_count` are ignored so later
    /// bound-prefix domain opens can rely on rectangular row shape.
    pub fn from_rows(variable_count: usize, rows: impl IntoIterator<Item = Vec<Vec<u8>>>) -> Self {
        assert!(
            variable_count <= usize::from(u8::MAX) + 1,
            "positional projection relation factors support at most 256 variables"
        );
        let variables = (0..variable_count)
            .map(|index| BindingVar(u8::try_from(index).expect("variable count is checked")))
            .collect::<Vec<_>>();
        Self::from_rows_with_variables(variables, rows)
    }

    /// Builds a diagnostic relation factor with explicit `BindingVar` order.
    ///
    /// The row order is the physical trie order that `open_domain_for_variable`
    /// follows. Rows whose width differs from `variables.len()` are ignored.
    pub fn from_rows_with_variables(
        variables: impl Into<Box<[BindingVar]>>,
        rows: impl IntoIterator<Item = Vec<Vec<u8>>>,
    ) -> Self {
        let variables = variables.into();
        let rows = rows
            .into_iter()
            .filter(|row| row.len() == variables.len())
            .map(Vec::into_boxed_slice)
            .collect();
        Self { variables, rows }
    }

    #[cfg(test)]
    fn from_projection(projection: &QueryProjectionMaps) -> Self {
        let variables = (0..projection.variable_domains.len())
            .map(|index| BindingVar(u8::try_from(index).expect("test projection variables fit u8")))
            .collect::<Vec<_>>();
        Self::from_rows_with_variables(
            variables,
            projection.variable_rows.iter().map(|row| row.to_vec()),
        )
    }

    /// Opens the ordered domain for `variable_index` under an earlier-variable prefix.
    ///
    /// `bound_prefix[0]` constrains row variable 0, `bound_prefix[1]` constrains
    /// row variable 1, and so on. Invalid requests return an empty domain rather
    /// than panicking because this is an explain/diagnostic surface.
    pub fn open_domain(
        &self,
        variable_index: usize,
        bound_prefix: &[Vec<u8>],
    ) -> QueryProjectionRelationDomain {
        let mut domain = BTreeSet::new();
        let mut rows_matching_prefix = 0usize;
        if variable_index < self.variables.len() && bound_prefix.len() <= variable_index {
            for row in self.rows.iter() {
                if row.len() != self.variables.len() {
                    continue;
                }
                if row
                    .iter()
                    .take(bound_prefix.len())
                    .zip(bound_prefix.iter())
                    .all(|(value, bound)| value == bound)
                {
                    rows_matching_prefix += 1;
                    domain.insert(row[variable_index].clone());
                }
            }
        }

        QueryProjectionRelationDomain {
            values: domain.into_iter().collect(),
            variable_index,
            bound_prefix_len: bound_prefix.len(),
            rows: self.rows.len(),
            rows_matching_prefix,
        }
    }

    /// Opens the ordered domain for `variable` under a global binding prefix.
    ///
    /// The global prefix may include variables that this factor does not carry.
    /// Only earlier variables in this factor's physical row order are projected
    /// into the local bound prefix. If a required local prefix variable is not
    /// bound yet, the returned domain is empty.
    pub fn open_domain_for_variable(
        &self,
        variable: BindingVar,
        bound_variables: &[(BindingVar, Vec<u8>)],
    ) -> QueryProjectionRelationDomain {
        let Some(variable_index) = self
            .variables
            .iter()
            .position(|candidate| *candidate == variable)
        else {
            return QueryProjectionRelationDomain {
                values: Vec::new(),
                variable_index: self.variables.len(),
                bound_prefix_len: bound_variables.len(),
                rows: self.rows.len(),
                rows_matching_prefix: 0,
            };
        };

        let mut local_prefix = Vec::with_capacity(variable_index);
        for local_variable in &self.variables[..variable_index] {
            let Some((_, value)) = bound_variables
                .iter()
                .find(|(bound_variable, _)| bound_variable == local_variable)
            else {
                return QueryProjectionRelationDomain {
                    values: Vec::new(),
                    variable_index,
                    bound_prefix_len: bound_variables.len(),
                    rows: self.rows.len(),
                    rows_matching_prefix: 0,
                };
            };
            local_prefix.push(value.clone());
        }

        let mut domain = self.open_domain(variable_index, &local_prefix);
        domain.bound_prefix_len = bound_variables.len();
        domain
    }

    /// Number of variables in each retained projected row.
    pub fn variable_count(&self) -> usize {
        self.variables.len()
    }

    /// Variables in the physical row order used by this factor.
    pub fn variables(&self) -> &[BindingVar] {
        &self.variables
    }

    /// Number of retained projected rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

impl QueryProjectionByteDomainFactor for QueryProjectionRelationFactor {
    fn open_domain_for_variable(
        &self,
        variable: BindingVar,
        bound_variables: &[(BindingVar, Vec<u8>)],
    ) -> QueryProjectionRelationDomain {
        self.open_domain_for_variable(variable, bound_variables)
    }
}

fn query_source_prefix(source: ExprEnv) -> Option<Vec<u8>> {
    unsafe {
        source
            .subsexpr()
            .prefix()
            .unwrap_or_else(|span| span)
            .as_ref()
            .map(<[u8]>::to_vec)
    }
}

fn query_binding_value_bytes(
    bindings: &BTreeMap<(u8, u8), ExprEnv>,
    query_variable: (u8, u8),
) -> Option<Vec<u8>> {
    let binding = bindings.get(&query_variable)?;
    let span = unsafe { binding.subsexpr().span().as_ref()? };
    Some(span.to_vec())
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct QueryProjectionZipperDomainCacheKey {
    variable: BindingVar,
    local_prefix: Box<[Vec<u8>]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QueryProjectionZipperDomainOpenMode {
    ReadZipper,
    ProductZipper,
}

/// Diagnostic byte-domain factor that opens domains from zipper scans.
///
/// This is the non-materializing acceptance target for
/// ReadZipper/ProductZipper-backed relation factors. It still uses the exact
/// unifier as the semantic oracle for each candidate path and does not replace
/// live `query_multi` execution.
pub struct QueryProjectionZipperRelationFactor<'a> {
    btm: &'a PathMap<()>,
    source: ExprEnv,
    prefix: Box<[u8]>,
    variables: Box<[BindingVar]>,
    query_variables: Box<[(u8, u8)]>,
    query_column_by_binding: BTreeMap<BindingVar, usize>,
    domain_cache:
        RefCell<BTreeMap<QueryProjectionZipperDomainCacheKey, QueryProjectionRelationDomain>>,
    telemetry: Cell<QueryProjectionZipperDomainTelemetry>,
    open_mode: QueryProjectionZipperDomainOpenMode,
}

impl<'a> QueryProjectionZipperRelationFactor<'a> {
    /// Builds a zipper-backed diagnostic factor from one query source.
    ///
    /// `query_to_binding_vars` maps the query source's local variable identity
    /// to the global `BindingVar` schema used by the selected sidecar contract.
    /// `variables` is the physical factor order to open.
    pub fn new(
        btm: &'a PathMap<()>,
        source: ExprEnv,
        query_to_binding_vars: impl IntoIterator<Item = ((u8, u8), BindingVar)>,
        variables: impl Into<Box<[BindingVar]>>,
    ) -> Option<Self> {
        Self::new_with_open_mode(
            btm,
            source,
            query_to_binding_vars,
            variables,
            QueryProjectionZipperDomainOpenMode::ReadZipper,
        )
    }

    /// Builds a zipper-backed diagnostic factor that scans through `ProductZipperG`.
    ///
    /// This is still a single logical query factor. It exercises the
    /// ProductZipper movement interface as an acceptance target before any
    /// multi-factor product traversal is allowed to affect live execution.
    pub fn new_product_zipper(
        btm: &'a PathMap<()>,
        source: ExprEnv,
        query_to_binding_vars: impl IntoIterator<Item = ((u8, u8), BindingVar)>,
        variables: impl Into<Box<[BindingVar]>>,
    ) -> Option<Self> {
        Self::new_with_open_mode(
            btm,
            source,
            query_to_binding_vars,
            variables,
            QueryProjectionZipperDomainOpenMode::ProductZipper,
        )
    }

    fn new_with_open_mode(
        btm: &'a PathMap<()>,
        source: ExprEnv,
        query_to_binding_vars: impl IntoIterator<Item = ((u8, u8), BindingVar)>,
        variables: impl Into<Box<[BindingVar]>>,
        open_mode: QueryProjectionZipperDomainOpenMode,
    ) -> Option<Self> {
        let prefix = query_source_prefix(source)?;
        let query_variables = Space::query_factor_variables(source).into_boxed_slice();
        let query_column_by_var = query_variables
            .iter()
            .copied()
            .enumerate()
            .map(|(index, variable)| (variable, index))
            .collect::<BTreeMap<_, _>>();
        let mut query_column_by_binding = BTreeMap::new();
        for (query_variable, binding_variable) in query_to_binding_vars {
            let query_column = *query_column_by_var.get(&query_variable)?;
            if query_column_by_binding
                .insert(binding_variable, query_column)
                .is_some()
            {
                return None;
            }
        }

        let variables = variables.into();
        if variables
            .iter()
            .any(|variable| !query_column_by_binding.contains_key(variable))
        {
            return None;
        }

        Some(Self {
            btm,
            source,
            prefix: prefix.into_boxed_slice(),
            variables,
            query_variables,
            query_column_by_binding,
            domain_cache: RefCell::new(BTreeMap::new()),
            telemetry: Cell::new(QueryProjectionZipperDomainTelemetry::default()),
            open_mode,
        })
    }

    /// Returns accumulated diagnostic work counters for this factor.
    pub fn telemetry(&self) -> QueryProjectionZipperDomainTelemetry {
        self.telemetry.get()
    }

    /// Clears accumulated diagnostic counters without clearing cached domains.
    pub fn clear_telemetry(&self) {
        self.telemetry
            .set(QueryProjectionZipperDomainTelemetry::default());
    }

    /// Number of cached domain opens retained by this diagnostic factor.
    pub fn cached_domain_count(&self) -> usize {
        self.domain_cache.borrow().len()
    }

    fn record_telemetry(&self, update: impl FnOnce(&mut QueryProjectionZipperDomainTelemetry)) {
        let mut telemetry = self.telemetry.get();
        update(&mut telemetry);
        self.telemetry.set(telemetry);
    }

    fn binding_value(
        &self,
        bindings: &BTreeMap<(u8, u8), ExprEnv>,
        variable: BindingVar,
    ) -> Option<Vec<u8>> {
        let query_column = *self.query_column_by_binding.get(&variable)?;
        let query_variable = *self.query_variables.get(query_column)?;
        query_binding_value_bytes(bindings, query_variable)
    }

    fn collect_domain_candidate(
        &self,
        candidate_path: &[u8],
        variable_index: usize,
        local_prefix: &[Vec<u8>],
        domain: &mut BTreeSet<Vec<u8>>,
        rows: &mut usize,
        rows_matching_prefix: &mut usize,
        unifications: &mut usize,
    ) {
        let candidate = Expr {
            ptr: candidate_path.as_ptr().cast_mut(),
        };
        let mut pairs = vec![(self.source, ExprEnv::new(1, candidate))];
        *unifications += 1;
        let Ok(bindings) = unify(&mut pairs) else {
            return;
        };

        let mut row = Vec::with_capacity(self.variables.len());
        let mut complete_row = true;
        for &row_variable in self.variables.iter() {
            if let Some(value) = self.binding_value(&bindings, row_variable) {
                row.push(value);
            } else {
                complete_row = false;
                break;
            }
        }
        if !complete_row {
            return;
        }

        *rows += 1;
        if row
            .iter()
            .take(local_prefix.len())
            .zip(local_prefix.iter())
            .all(|(value, bound)| value == bound)
        {
            *rows_matching_prefix += 1;
            domain.insert(row[variable_index].clone());
        }
    }
}

impl QueryProjectionByteDomainFactor for QueryProjectionZipperRelationFactor<'_> {
    fn open_domain_for_variable(
        &self,
        variable: BindingVar,
        bound_variables: &[(BindingVar, Vec<u8>)],
    ) -> QueryProjectionRelationDomain {
        self.record_telemetry(|telemetry| {
            telemetry.opens += 1;
        });

        let Some(variable_index) = self
            .variables
            .iter()
            .position(|candidate| *candidate == variable)
        else {
            return QueryProjectionRelationDomain {
                values: Vec::new(),
                variable_index: self.variables.len(),
                bound_prefix_len: bound_variables.len(),
                rows: 0,
                rows_matching_prefix: 0,
            };
        };

        let mut local_prefix = Vec::with_capacity(variable_index);
        for local_variable in &self.variables[..variable_index] {
            let Some((_, value)) = bound_variables
                .iter()
                .find(|(bound_variable, _)| bound_variable == local_variable)
            else {
                return QueryProjectionRelationDomain {
                    values: Vec::new(),
                    variable_index,
                    bound_prefix_len: bound_variables.len(),
                    rows: 0,
                    rows_matching_prefix: 0,
                };
            };
            local_prefix.push(value.clone());
        }

        let cache_key = QueryProjectionZipperDomainCacheKey {
            variable,
            local_prefix: local_prefix.clone().into_boxed_slice(),
        };
        if let Some(mut domain) = { self.domain_cache.borrow().get(&cache_key).cloned() } {
            domain.bound_prefix_len = bound_variables.len();
            self.record_telemetry(|telemetry| {
                telemetry.cache_hits += 1;
                telemetry.domain_values += domain.values.len();
            });
            return domain;
        }

        let mut rows = 0usize;
        let mut rows_matching_prefix = 0usize;
        let mut candidates = 0usize;
        let mut unifications = 0usize;
        let mut domain = BTreeSet::new();

        match self.open_mode {
            QueryProjectionZipperDomainOpenMode::ReadZipper => {
                let mut rz = self.btm.read_zipper_at_path(self.prefix.as_ref());
                while rz.to_next_val() {
                    candidates += 1;
                    self.collect_domain_candidate(
                        rz.origin_path(),
                        variable_index,
                        &local_prefix,
                        &mut domain,
                        &mut rows,
                        &mut rows_matching_prefix,
                        &mut unifications,
                    );
                }
            }
            QueryProjectionZipperDomainOpenMode::ProductZipper => {
                let mut pz = ProductZipperG::new(
                    self.btm.read_zipper_at_path(self.prefix.as_ref()),
                    std::iter::empty::<ReadZipperUntracked<'_, 'static, ()>>(),
                );
                reserve_query_product_buffers(&mut pz);
                while pz.to_next_val() {
                    if pz.focus_factor() != pz.factor_count() - 1 {
                        continue;
                    }
                    candidates += 1;
                    self.collect_domain_candidate(
                        pz.origin_path(),
                        variable_index,
                        &local_prefix,
                        &mut domain,
                        &mut rows,
                        &mut rows_matching_prefix,
                        &mut unifications,
                    );
                }
            }
        }

        let domain = QueryProjectionRelationDomain {
            values: domain.into_iter().collect(),
            variable_index,
            bound_prefix_len: bound_variables.len(),
            rows,
            rows_matching_prefix,
        };
        self.record_telemetry(|telemetry| {
            telemetry.scans += 1;
            match self.open_mode {
                QueryProjectionZipperDomainOpenMode::ReadZipper => {
                    telemetry.read_zipper_scans += 1;
                }
                QueryProjectionZipperDomainOpenMode::ProductZipper => {
                    telemetry.product_zipper_scans += 1;
                }
            }
            telemetry.candidates += candidates;
            telemetry.unifications += unifications;
            telemetry.rows += rows;
            telemetry.rows_matching_prefix += rows_matching_prefix;
            telemetry.domain_values += domain.values.len();
        });
        self.domain_cache
            .borrow_mut()
            .insert(cache_key, domain.clone());
        domain
    }
}

/// Compares a ProductZipper candidate trace with a BindingSpace relation.
///
/// The trace is byte-valued while the sidecar relation is `TermId`-valued, so
/// `term_to_bytes` supplies the semantic bridge. Rows are compared in the
/// trace's variable order regardless of the relation's stored schema order.
pub fn compare_query_projection_product_trace_to_binding_relation(
    trace: &QueryProjectionProductCandidateTrace,
    relation: &BindingRelation,
    mut term_to_bytes: impl FnMut(TermId) -> Option<Vec<u8>>,
) -> QueryProjectionProductTraceComparison {
    let mut expected_rows = Vec::new();
    let mut missing_term_mappings = 0usize;

    for row in relation.positive_rows() {
        let mut mapped_row = Vec::with_capacity(trace.variable_order.len());
        let mut complete = true;
        for variable in trace.variable_order.iter() {
            let Some(schema_index) = relation
                .schema()
                .iter()
                .position(|candidate| candidate == variable)
            else {
                complete = false;
                missing_term_mappings += 1;
                break;
            };
            let Some(bytes) = row.get(schema_index).and_then(|term| term_to_bytes(*term)) else {
                complete = false;
                missing_term_mappings += 1;
                break;
            };
            mapped_row.push(bytes);
        }
        if complete {
            expected_rows.push(mapped_row.into_boxed_slice());
        }
    }
    expected_rows.sort();
    expected_rows.dedup();

    let mut actual_rows = trace.unique_rows.clone();
    actual_rows.sort();
    actual_rows.dedup();
    let matched = missing_term_mappings == 0 && actual_rows == expected_rows;

    QueryProjectionProductTraceComparison {
        product_successful_candidates: trace.successful_candidates,
        product_raw_candidates: trace.raw.raw_candidates,
        product_rejected_candidates: trace.raw.rejected_unifications,
        product_rows: trace.rows.len(),
        product_unique_rows: actual_rows.len(),
        expected_rows: expected_rows.len(),
        missing_term_mappings,
        missing_binding_rows: trace.missing_binding_rows,
        non_binding_results: trace.non_binding_results,
        actual_domain: actual_rows,
        expected_domain: expected_rows,
        matched,
    }
}

/// Compares current ProductZipper trace work with a trie-backed join trace.
///
/// The trie trace remains the selected sidecar cursor oracle. This report is
/// explain-only: it measures whether current ProductZipper output rows line up
/// with the LFTJ-style trace and how much raw pre-unification work the product
/// traversal performed, without changing query execution.
pub fn compare_query_projection_product_trace_to_trie_trace(
    trace: &QueryProjectionProductCandidateTrace,
    trie_trace: &TrieJoinTrace,
) -> Result<QueryProjectionProductVsTrieTraceComparison, TrieJoinTraceShapeError> {
    let summary = trie_trace.summarize()?;
    let product_raw_candidates = trace.raw.raw_candidates;
    let product_successful_candidates = trace.successful_candidates;
    let product_rejected_candidates = trace.raw.rejected_unifications;
    let product_unique_rows = trace.unique_rows.len();
    let trie_candidate_bindings = summary.candidate_bindings;

    Ok(QueryProjectionProductVsTrieTraceComparison {
        product_raw_candidates,
        product_successful_candidates,
        product_rejected_candidates,
        product_unique_rows,
        trie_candidate_bindings,
        trie_steps: summary.steps,
        trie_domain_sources: summary.domain_sources,
        trie_domain_values: summary.domain_values,
        trie_cursor_seeks: summary.cursor_seeks,
        trie_cursor_skips: summary.cursor_skips,
        successful_candidate_counts_match: product_successful_candidates == trie_candidate_bindings,
        unique_row_counts_match: product_unique_rows == trie_candidate_bindings,
        raw_candidate_overhead: product_raw_candidates.saturating_sub(trie_candidate_bindings),
    })
}

fn map_trie_context_to_byte_domains(
    context: &TrieJoinFactorCursorContext,
    variable_order: &[BindingVar],
    term_to_bytes: &mut impl FnMut(TermId) -> Option<Vec<u8>>,
) -> (Vec<(BindingVar, Vec<u8>)>, Vec<Vec<u8>>, usize) {
    let mut missing_term_mappings = 0usize;
    let mut bound_variables = Vec::with_capacity(context.bound_prefix.len());
    for (index, &term) in context.bound_prefix.iter().enumerate() {
        let Some(&variable) = variable_order.get(index) else {
            missing_term_mappings += 1;
            continue;
        };
        if let Some(bytes) = term_to_bytes(term) {
            bound_variables.push((variable, bytes));
        } else {
            missing_term_mappings += 1;
        }
    }

    let mut expected_domain = Vec::with_capacity(context.domain.len());
    for &term in context.domain.iter() {
        if let Some(bytes) = term_to_bytes(term) {
            expected_domain.push(bytes);
        } else {
            missing_term_mappings += 1;
        }
    }
    expected_domain.sort();

    (bound_variables, expected_domain, missing_term_mappings)
}

fn compare_query_projection_byte_domain_factors_to_trie_contract<F>(
    factors: &[F],
    variable_order: &[BindingVar],
    contract: &TrieJoinCursorContract,
    mut term_to_bytes: impl FnMut(TermId) -> Option<Vec<u8>>,
) -> QueryProjectionTrieContractComparison
where
    F: QueryProjectionByteDomainFactor,
{
    let mut comparison = QueryProjectionTrieContractComparison {
        relation_indexes: contract.relation_indexes,
        factor_requirements: contract.factor_requirements.len(),
        ..QueryProjectionTrieContractComparison::default()
    };

    for requirement in contract.factor_requirements.iter() {
        let Some(factor) = factors.get(requirement.relation_index) else {
            comparison.missing_factors += 1;
            for context in requirement.contexts.iter() {
                let (_, expected_domain, missing_term_mappings) =
                    map_trie_context_to_byte_domains(context, variable_order, &mut term_to_bytes);
                comparison.contexts += 1;
                comparison.mismatched_contexts += 1;
                comparison.missing_term_mappings += missing_term_mappings;
                comparison
                    .context_results
                    .push(QueryProjectionTrieContractContextComparison {
                        relation_index: requirement.relation_index,
                        step_index: context.step_index,
                        variable: context.variable,
                        bound_prefix_len: context.bound_prefix.len(),
                        expected_domain,
                        actual_domain: Vec::new(),
                        missing_term_mappings,
                        matched: false,
                    });
            }
            continue;
        };

        for context in requirement.contexts.iter() {
            let (bound_variables, expected_domain, missing_term_mappings) =
                map_trie_context_to_byte_domains(context, variable_order, &mut term_to_bytes);
            let actual_domain = if missing_term_mappings == 0 {
                factor
                    .open_domain_for_variable(context.variable, &bound_variables)
                    .values
            } else {
                Vec::new()
            };
            let matched = missing_term_mappings == 0 && actual_domain == expected_domain;

            comparison.contexts += 1;
            comparison.missing_term_mappings += missing_term_mappings;
            if matched {
                comparison.matched_contexts += 1;
            } else {
                comparison.mismatched_contexts += 1;
            }
            comparison
                .context_results
                .push(QueryProjectionTrieContractContextComparison {
                    relation_index: requirement.relation_index,
                    step_index: context.step_index,
                    variable: context.variable,
                    bound_prefix_len: context.bound_prefix.len(),
                    expected_domain,
                    actual_domain,
                    missing_term_mappings,
                    matched,
                });
        }
    }

    comparison
}

/// Compares byte-domain relation factors with a validated trie cursor contract.
///
/// The trie contract is TermId-valued, while a PathMap/ReadZipper factor
/// exposes encoded byte domains. `term_to_bytes` supplies that semantic bridge.
/// The comparison is diagnostic only; it does not execute `query_multi` or
/// replace ProductZipper traversal.
pub fn compare_query_projection_relation_factors_to_trie_contract(
    factors: &[QueryProjectionRelationFactor],
    variable_order: &[BindingVar],
    contract: &TrieJoinCursorContract,
    term_to_bytes: impl FnMut(TermId) -> Option<Vec<u8>>,
) -> QueryProjectionTrieContractComparison {
    compare_query_projection_byte_domain_factors_to_trie_contract(
        factors,
        variable_order,
        contract,
        term_to_bytes,
    )
}

/// Compares zipper-backed byte-domain factors with a trie cursor contract.
///
/// This keeps the selected sidecar contract as the oracle while opening the
/// actual byte domains through `ReadZipper` scans instead of retained row
/// materialization.
pub fn compare_query_projection_zipper_factors_to_trie_contract(
    factors: &[QueryProjectionZipperRelationFactor<'_>],
    variable_order: &[BindingVar],
    contract: &TrieJoinCursorContract,
    term_to_bytes: impl FnMut(TermId) -> Option<Vec<u8>>,
) -> QueryProjectionTrieContractComparison {
    compare_query_projection_byte_domain_factors_to_trie_contract(
        factors,
        variable_order,
        contract,
        term_to_bytes,
    )
}

/// Intersects already-opened byte-domain cursors with LFTJ-style seek/next.
///
/// This is the byte-valued counterpart to the TermId trie-join cursor contract:
/// each cursor supplies one ordered domain for the current variable under a
/// bound prefix, and the helper returns the ordered intersection without
/// materializing pairwise intermediate sets.
pub fn intersect_query_projection_byte_domain_cursors<C>(
    cursors: &mut [C],
) -> QueryProjectionDomainIntersection
where
    C: QueryProjectionByteDomainCursor,
{
    let mut intersection = QueryProjectionDomainIntersection {
        domain_sources: cursors.len(),
        ..QueryProjectionDomainIntersection::default()
    };
    intersection.domain_values = cursors.iter().map(|cursor| cursor.domain_len()).sum();
    intersection.cursor_opens = cursors.len();

    if cursors.is_empty() || cursors.iter().any(QueryProjectionByteDomainCursor::at_end) {
        return intersection;
    }

    let mut target = cursors
        .iter()
        .filter_map(QueryProjectionByteDomainCursor::key)
        .max()
        .expect("non-empty projection cursors have keys")
        .to_vec();

    loop {
        let mut changed = false;
        for cursor in cursors.iter_mut() {
            intersection.cursor_seeks += 1;
            intersection.cursor_skips += cursor.seek(&target);
            if cursor.at_end() {
                return intersection;
            }
            let Some(key) = cursor.key() else {
                return intersection;
            };
            if key > target.as_slice() {
                target.clear();
                target.extend_from_slice(key);
                changed = true;
            }
        }

        if !changed {
            intersection.values.push(target.clone());
            intersection.cursor_nexts += 1;
            cursors[0].next();
            if cursors[0].at_end() {
                return intersection;
            }
            target = cursors[0]
                .key()
                .expect("cursor just checked as not at end")
                .to_vec();
        }
    }
}

/// Intersects PathMap projection domains through an ordered seek/next cursor.
///
/// This is a diagnostic bridge between retained byte-domain projection maps and
/// the term-side BindingSpace domain-cursor contract. It does not execute
/// `query_multi`; callers can compare its output with `PathMap::meet` and exact
/// projected-domain sets before replacing a live zipper traversal.
pub fn intersect_query_projection_domains(
    maps: &[&PathMap<()>],
) -> QueryProjectionDomainIntersection {
    let cursors = maps
        .iter()
        .map(|map| QueryProjectionDomainCursor::from_pathmap(map))
        .collect::<Vec<_>>();
    intersect_query_projection_domain_cursors(cursors)
}

/// Intersects already-opened projection-factor byte domains with the same cursor.
///
/// This lets a bound-prefix relation factor reuse the exact seek/next contract
/// as retained PathMap projection maps.
pub fn intersect_query_projection_domain_values(
    domains: &[&[Vec<u8>]],
) -> QueryProjectionDomainIntersection {
    let cursors = domains
        .iter()
        .map(|domain| QueryProjectionDomainCursor::from_values(domain.to_vec()))
        .collect::<Vec<_>>();
    intersect_query_projection_domain_cursors(cursors)
}

fn intersect_query_projection_domain_cursors(
    mut cursors: Vec<QueryProjectionDomainCursor>,
) -> QueryProjectionDomainIntersection {
    intersect_query_projection_byte_domain_cursors(&mut cursors)
}

#[derive(Default)]
struct QueryShapeSideIndexFootprint {
    key_bytes: usize,
    summary_bytes: usize,
    domain_values: usize,
}

#[derive(Default)]
struct QueryProjectionSideIndexFootprint {
    key_bytes: usize,
    projection_bytes: usize,
    domain_values: usize,
    projection_maps: usize,
}

impl QueryShapeSideIndex {
    fn get(&mut self, key: &QueryShapeSideIndexKey) -> Option<Option<QueryShapeSummary>> {
        if let Some(summary) = self.entries.get(key) {
            self.hits += 1;
            Some(summary.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, key: QueryShapeSideIndexKey, summary: Option<QueryShapeSummary>) {
        if insert_bounded_cache_entry(
            &mut self.entries,
            QUERY_SHAPE_SIDE_INDEX_LIMIT,
            key,
            summary,
        ) {
            self.clears += 1;
            self.generation += 1;
        }
        self.inserts += 1;
        self.max_estimated_bytes = self.max_estimated_bytes.max(self.estimated_bytes());
    }

    fn footprint(&self) -> QueryShapeSideIndexFootprint {
        let mut footprint = QueryShapeSideIndexFootprint::default();
        for (key, summary) in &self.entries {
            footprint.key_bytes += key.estimated_bytes();
            footprint.summary_bytes += size_of::<Option<QueryShapeSummary>>();
            if let Some(summary) = summary {
                footprint.summary_bytes += summary.estimated_bytes();
                footprint.domain_values += summary.domain_value_count();
            }
        }
        footprint
    }

    fn estimated_bytes(&self) -> usize {
        let footprint = self.footprint();
        size_of::<Self>() + footprint.key_bytes + footprint.summary_bytes
    }

    fn stats(&self) -> QueryShapeSideIndexStats {
        let footprint = self.footprint();
        let estimated_bytes = size_of::<Self>() + footprint.key_bytes + footprint.summary_bytes;
        QueryShapeSideIndexStats {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            inserts: self.inserts,
            clears: self.clears,
            generation: self.generation,
            estimated_bytes,
            max_estimated_bytes: self.max_estimated_bytes.max(estimated_bytes),
            key_bytes: footprint.key_bytes,
            summary_bytes: footprint.summary_bytes,
            domain_values: footprint.domain_values,
            avoided_shape_scans: self.hits,
        }
    }
}

impl QueryProjectionSideIndex {
    fn get(&mut self, key: &QueryProjectionSideIndexKey) -> Option<QueryProjectionMaps> {
        if let Some(projection) = self.entries.get(key) {
            self.hits += 1;
            Some(projection.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, key: QueryProjectionSideIndexKey, projection: QueryProjectionMaps) {
        if insert_bounded_cache_entry(
            &mut self.entries,
            QUERY_PROJECTION_SIDE_INDEX_LIMIT,
            key,
            projection,
        ) {
            self.clears += 1;
            self.generation += 1;
        }
        self.inserts += 1;
        self.max_estimated_bytes = self.max_estimated_bytes.max(self.estimated_bytes());
    }

    fn footprint(&self) -> QueryProjectionSideIndexFootprint {
        let mut footprint = QueryProjectionSideIndexFootprint::default();
        for (key, projection) in &self.entries {
            footprint.key_bytes += key.estimated_bytes();
            footprint.projection_bytes += projection.estimated_bytes();
            footprint.domain_values += projection.domain_value_count();
            footprint.projection_maps += projection.variable_maps.len();
        }
        footprint
    }

    fn estimated_bytes(&self) -> usize {
        let footprint = self.footprint();
        size_of::<Self>() + footprint.key_bytes + footprint.projection_bytes
    }

    fn stats(&self) -> QueryProjectionSideIndexStats {
        let footprint = self.footprint();
        let estimated_bytes = size_of::<Self>() + footprint.key_bytes + footprint.projection_bytes;
        QueryProjectionSideIndexStats {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            inserts: self.inserts,
            clears: self.clears,
            generation: self.generation,
            estimated_bytes,
            max_estimated_bytes: self.max_estimated_bytes.max(estimated_bytes),
            key_bytes: footprint.key_bytes,
            projection_bytes: footprint.projection_bytes,
            domain_values: footprint.domain_values,
            projection_maps: footprint.projection_maps,
            avoided_projection_scans: self.hits,
        }
    }
}

fn insert_bounded_cache_entry<K, V>(
    entries: &mut HashMap<K, V>,
    limit: usize,
    key: K,
    value: V,
) -> bool
where
    K: Eq + Hash,
{
    if entries.len() < limit {
        entries.insert(key, value);
        return false;
    }

    match entries.entry(key) {
        Entry::Occupied(mut entry) => {
            entry.insert(value);
            false
        }
        Entry::Vacant(entry) => {
            let key = entry.into_key();
            entries.clear();
            entries.insert(key, value);
            true
        }
    }
}

fn query_factor_plan_cache() -> &'static Mutex<QueryFactorPlanCache> {
    static QUERY_FACTOR_PLAN_CACHE: OnceLock<Mutex<QueryFactorPlanCache>> = OnceLock::new();
    QUERY_FACTOR_PLAN_CACHE.get_or_init(|| Mutex::new(QueryFactorPlanCache::default()))
}

fn query_shape_side_index() -> &'static Mutex<QueryShapeSideIndex> {
    static QUERY_SHAPE_SIDE_INDEX: OnceLock<Mutex<QueryShapeSideIndex>> = OnceLock::new();
    QUERY_SHAPE_SIDE_INDEX.get_or_init(|| Mutex::new(QueryShapeSideIndex::default()))
}

fn query_projection_side_index() -> &'static Mutex<QueryProjectionSideIndex> {
    static QUERY_PROJECTION_SIDE_INDEX: OnceLock<Mutex<QueryProjectionSideIndex>> = OnceLock::new();
    QUERY_PROJECTION_SIDE_INDEX.get_or_init(|| Mutex::new(QueryProjectionSideIndex::default()))
}

fn query_factor_plan_metrics() -> &'static Mutex<QueryFactorPlanMetrics> {
    static QUERY_FACTOR_PLAN_METRICS: OnceLock<Mutex<QueryFactorPlanMetrics>> = OnceLock::new();
    QUERY_FACTOR_PLAN_METRICS.get_or_init(|| Mutex::new(QueryFactorPlanMetrics::default()))
}

// Per-thread storage-metrics accumulators, registered on first use and summed at snapshot.
// Recording into a single global mutex on every query serialized parallel queries (futex
// contention that collapsed throughput past ~8 threads); accumulating thread-locally and merging
// only at the rare snapshot keeps identical totals with no shared write on the hot path.
fn query_execution_storage_metrics_registry()
-> &'static Mutex<Vec<std::sync::Arc<Mutex<QueryExecutionStorageMetrics>>>> {
    static REGISTRY: OnceLock<Mutex<Vec<std::sync::Arc<Mutex<QueryExecutionStorageMetrics>>>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

thread_local! {
    static TL_STORAGE_METRICS: std::sync::Arc<Mutex<QueryExecutionStorageMetrics>> = {
        let cell = std::sync::Arc::new(Mutex::new(QueryExecutionStorageMetrics::default()));
        query_execution_storage_metrics_registry()
            .lock()
            .unwrap()
            .push(std::sync::Arc::clone(&cell));
        cell
    };
}

// Accumulate into the calling thread's storage-metrics cell. Uncontended on the hot path: only the
// owning thread writes its cell; `query_execution_storage_metrics_snapshot` briefly locks each to sum.
fn record_storage_metrics(f: impl FnOnce(&mut QueryExecutionStorageMetrics)) {
    TL_STORAGE_METRICS.with(|cell| f(&mut cell.lock().unwrap()));
}

// future Adam: don't fall for the temptation of keeping references of data->pattern, you tried it twice already: it's not worth the complexity, it's incompatible due to the PZ de-Bruijn level non-well-foundedness, it doesn't occur in most queries, and the performance is not worth it
// others: this code has haphephobia, contact Adam when you run into problems
// optimization opportunities:
// - use u16 x u16 compressed byte mask to reduce stack size, or to_next_sibling?
// - decrease the size of ExprEnv; it's too rich for this function
// - this function gets massive (many thousands of instructions) but can do with less checked functions
// - ascends may be avoided by using RZ refs instead of re-ascending in some cases
// - the adiabatic crate may be used to get rid of the recursion (though currently the recursion is significantly faster)
// - `references` can be elided by not putting the virtual $ Expr's on the `stack` such that _k maps directly to the indices
// - keeping a needle instead of a stack to avoid the `reverse` (would also create the opportunity to be even more lazy about instruction gen)
// - use descend_to and re-evaluated the added sub-path to do much better on long paths
thread_local! {
    /// Drives the VarRef ground fast-path re-check in `coreferential_transition`
    /// (WAM `unify_value` by byte comparison). The ground-data case is proved
    /// correct in `kernel/resources/formal/verus/VarRefRecheck.rs`; the
    /// implementation handles non-ground data by detecting a variable branch in
    /// the data's child mask and falling back to the recursive re-match (the
    /// differential oracle test `varref_fast_recheck_matches_recursive_path`
    /// checks both, including the Verus Theorem-3 non-ground witness). The toggle
    /// stays so the differential test can compare the two paths.
    pub(crate) static VARREF_FAST_RECHECK: std::cell::Cell<bool> =
        const { std::cell::Cell::new(true) };

    #[cfg(test)]
    pub(crate) static FORCE_INTERPRETED_MATCHER: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };

    #[cfg(test)]
    pub(crate) static COMPILED_MATCHER_COMPOUND_CAPTURE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(true) };

    /// Selects the fused single-word-walk in `match_any_term` and the NewVar arm
    /// of `coreferential_transition` over the original three `ByteMask::and` +
    /// iterator passes. Both produce byte-identical match results in the same
    /// visit order; the toggle only exists so the differential oracle test
    /// `match_any_fused_equals_three_pass` can run the two paths against each
    /// other. `true` (fused) is the production path; non-test builds always fuse.
    #[cfg(test)]
    pub(crate) static MATCH_ANY_TERM_FUSED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(true) };
}

/// Whether to take the fused single-word-walk in the NewVar match-any-term
/// descents. Always `true` outside tests (the fused path is the only one
/// compiled); in tests it reads the `MATCH_ANY_TERM_FUSED` toggle so the
/// differential oracle can compare fused vs three-pass.
#[cfg(test)]
#[inline(always)]
fn match_any_term_fused() -> bool {
    MATCH_ANY_TERM_FUSED.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn match_any_term_fused() -> bool {
    true
}

#[cfg(test)]
fn force_interpreted_matcher() -> bool {
    FORCE_INTERPRETED_MATCHER.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn force_interpreted_matcher() -> bool {
    false
}

#[cfg(test)]
fn compiled_matcher_compound_capture_enabled() -> bool {
    COMPILED_MATCHER_COMPOUND_CAPTURE.with(|c| c.get())
}

#[cfg(not(test))]
#[inline(always)]
fn compiled_matcher_compound_capture_enabled() -> bool {
    true
}

fn coreferential_transition<
    Z: ZipperMoving + Zipper + ZipperAbsolutePath + ZipperIteration,
    F: FnMut(&mut Z) -> (),
>(
    loc: &mut Z,
    mut stack: &mut Vec<ExprEnv>,
    references: &mut Vec<u32>,
    f: &mut F,
) {
    macro_rules! vs {
        ($cm:expr, $e:expr, $nv:expr) => {{
            let m = $cm.and(&ByteMask(VARS));
            let mut it = m.iter();

            while let Some(b) = it.next() {
                // technically requires us to replace references to this NewVar on the stack with e
                // if !$nv && item_byte(Tag::NewVar) == b {
                //     if $e.n == 0 {
                //         references.push(u32::MAX);
                //     }
                // }
                loc.descend_to_byte(b);
                debug_assert!(loc.path_exists());
                coreferential_transition(loc, stack, references, f);
                if !loc.ascend_byte() {
                    unreachable_unchecked()
                };
            }
        }};
    }
    unsafe {
        trace!(target: "coref trans", "loc {}    len {}", serialize(loc.path()), loc.path().len());
        // trace!(target: "coref trans", "loc {} ({:?})    len {}    ops {:?} ({:?})", serialize(loc.path()), loc.path(), loc.path().len(), loc.child_mask(), loc.child_mask().iter().map(byte_item).collect::<Vec<_>>());
        trace!(target: "coref trans", "top {}", stack.last().map(|x| x.show()).unwrap_or_else(|| "empty".into()));
        TRANSITIONS += 1;
        match stack.pop() {
            None => f(loc),
            Some(e) => {
                let e_byte = *e.base.ptr.add(e.offset as usize);

                match byte_item(e_byte) {
                    Tag::NewVar => {
                        let restore = if e.n == 0 {
                            let idx = e.v as usize;
                            if references.len() <= idx {
                                references.resize(idx + 1, u32::MAX);
                            }
                            let prev = references[idx];
                            references[idx] = loc.path().len() as u32;
                            Some((idx, prev))
                        } else {
                            trace!(target: "coref trans", "not putting {} {}", e.n, e.show());
                            // trace!(target: "coref trans", "not putting against {:?}", loc.child_mask());
                            None
                        };
                        // The data trie is read-only and the sub-walks below
                        // each restore `loc`, so its child mask is invariant
                        // across the variable, symbol-size, and arity descents.
                        // Compute it once instead of three times (the matcher's
                        // hottest per-transition cost).
                        let cm = loc.child_mask();
                        debug_assert!(cm.0[1] == 0);
                        static NEW_VAR: u8 = item_byte(Tag::NewVar);
                        if match_any_term_fused() {
                            // Fused single-word-walk, byte-identical to the
                            // `vs!` + SIZES + ARITIES passes below in the same
                            // visit order. Each child mask word is read at most
                            // once; the size/arity is the bit index (no
                            // `byte_item` re-decode). See the byte encoding in
                            // the `ARITY_WORD`/`VARREF_WORD`/`HIGH_WORD` consts.
                            // VARS phase: VarRefs ascending, then NewVar (mirrors `vs!`).
                            let mut w = cm.0[VARREF_WORD];
                            while w != 0 {
                                let bit = w.trailing_zeros();
                                let b = (VARREF_TAG_HI | bit as u8) as u8;
                                loc.descend_to_byte(b);
                                debug_assert!(loc.path_exists());
                                coreferential_transition(loc, stack, references, f);
                                if !loc.ascend_byte() {
                                    unreachable_unchecked()
                                };
                                w &= w - 1;
                            }
                            if cm.0[HIGH_WORD] & NEWVAR_BIT != 0 {
                                loc.descend_to_byte(NEWVAR_BYTE);
                                debug_assert!(loc.path_exists());
                                coreferential_transition(loc, stack, references, f);
                                if !loc.ascend_byte() {
                                    unreachable_unchecked()
                                };
                            }
                            // SIZES phase.
                            let mut w = cm.0[HIGH_WORD] & !NEWVAR_BIT;
                            while w != 0 {
                                let size = w.trailing_zeros();
                                let b = (HIGH_TAG_HI | size as u8) as u8;
                                loc.descend_to_byte(b);
                                debug_assert!(loc.path_exists());
                                if !loc.descend_first_k_path(size as _) {
                                    unreachable_unchecked()
                                }
                                loop {
                                    coreferential_transition(loc, stack, references, f);
                                    if !loc.to_next_k_path(size as _) {
                                        break;
                                    }
                                }
                                if !loc.ascend_byte() {
                                    unreachable_unchecked()
                                }
                                w &= w - 1;
                            }
                            // ARITIES phase.
                            let mut w = cm.0[ARITY_WORD];
                            while w != 0 {
                                let a = w.trailing_zeros();
                                let b = a as u8;
                                loc.descend_to_byte(b);
                                debug_assert!(loc.path_exists());
                                let ol = stack.len();
                                for _ in 0..a {
                                    stack.push(ExprEnv::new(
                                        255,
                                        Expr {
                                            ptr: ((&NEW_VAR) as *const u8).cast_mut(),
                                        },
                                    ))
                                }
                                coreferential_transition(loc, stack, references, f);
                                stack.truncate(ol);
                                if !loc.ascend_byte() {
                                    unreachable_unchecked()
                                };
                                w &= w - 1;
                            }
                        } else {
                            vs!(cm, e, true);

                            let m = cm.and(&ByteMask(SIZES));
                            let mut it = m.iter();
                            while let Some(b) = it.next() {
                                let Tag::SymbolSize(size) = byte_item(b) else {
                                    unreachable_unchecked()
                                };
                                loc.descend_to_byte(b);
                                debug_assert!(loc.path_exists());
                                if !loc.descend_first_k_path(size as _) {
                                    unreachable_unchecked()
                                }
                                loop {
                                    coreferential_transition(loc, stack, references, f);
                                    if !loc.to_next_k_path(size as _) {
                                        break;
                                    }
                                }
                                if !loc.ascend_byte() {
                                    unreachable_unchecked()
                                }
                            }

                            let m = cm.and(&ByteMask(ARITIES));
                            let mut it = m.iter();
                            while let Some(b) = it.next() {
                                let Tag::Arity(a) = byte_item(b) else {
                                    unreachable_unchecked()
                                };
                                loc.descend_to_byte(b);
                                debug_assert!(loc.path_exists());
                                let ol = stack.len();
                                for _ in 0..a {
                                    stack.push(ExprEnv::new(
                                        255,
                                        Expr {
                                            ptr: ((&NEW_VAR) as *const u8).cast_mut(),
                                        },
                                    ))
                                }
                                coreferential_transition(loc, stack, references, f);
                                stack.truncate(ol);
                                if !loc.ascend_byte() {
                                    unreachable_unchecked()
                                };
                            }
                        }

                        if let Some((idx, prev)) = restore {
                            references[idx] = prev;
                        }
                    }
                    Tag::VarRef(i) => {
                        if e.n == 0
                            && (i as usize) < references.len()
                            && references[i as usize] != u32::MAX
                        {
                            // The data subterm bound to this variable at its first
                            // occurrence (`references[i]` is its start in the data path).
                            // `references[i]` indexes into `loc.path()`, the ProductZipper's
                            // path buffer. That buffer can REALLOCATE as the recursion below
                            // grows the matched path (it starts at a reserved capacity but is
                            // not bounded). A raw `Expr` into it would dangle the moment a
                            // deeper recursion reallocs, reading garbage tag bytes (observed
                            // as `byte_item` "reserved" panics on deep terms, e.g. the
                            // reordered semi-naive delta seeking a deep bound payload). So
                            // copy the bound subterm's bytes into a stable owned buffer NOW
                            // (before any recursion / descent that could realloc) and point
                            // the re-match at THAT. The copy is exactly the subterm rooted at
                            // `references[i]` (`Expr::span` walks the term structure), and the
                            // owned `Vec` lives across every recursion in this arm.
                            let bound_owned: Vec<u8> = (&*Expr {
                                ptr: loc
                                    .path()
                                    .as_ptr()
                                    .cast_mut()
                                    .offset(references[i as usize] as _),
                            }
                            .span())
                                .to_vec();
                            let bound = Expr {
                                ptr: bound_owned.as_ptr().cast_mut(),
                            };
                            // WAM `unify_value` (read mode): re-check that the data here
                            // equals the bound value. For a GROUND bound term this is an
                            // exact-byte descent (one `descend_to_existing` + memcmp)
                            // instead of pushing the data subterm and re-matching it
                            // structurally (which `args`-decomposes it, the measured
                            // matcher hot spot). Sound (only matches literal data); for
                            // ground data it is also complete (no data variables to
                            // branch on). The non-ground bound term keeps the recursive
                            // path. The toggle drives the differential oracle test.
                            if VARREF_FAST_RECHECK.with(|c| c.get()) && bound.is_ground() {
                                // Ground bound value: re-check it by exact byte
                                // descent (WAM unify_value). Verus VarRefRecheck.rs
                                // proves this is correct ONLY while the matched data
                                // is also ground; the moment the data branches on a
                                // variable (a wildcard child), byte comparison is
                                // incomplete, so fall back to the recursive re-match
                                // for the whole subterm.
                                let bytes: Vec<u8> = (&*bound.span()).to_vec();
                                vs!(loc.child_mask(), e, false);
                                let mut i = 0usize;
                                let mut variable_branch = false;
                                while i < bytes.len() {
                                    if loc
                                        .child_mask()
                                        .and(&ByteMask(VARS))
                                        .iter()
                                        .next()
                                        .is_some()
                                    {
                                        variable_branch = true;
                                        break;
                                    }
                                    if loc.descend_to_existing_byte(bytes[i]) {
                                        i += 1;
                                    } else {
                                        break; // ground data, literal absent: a real mismatch
                                    }
                                }
                                if variable_branch {
                                    loc.ascend(i);
                                    let addition = ExprEnv {
                                        n: 254,
                                        v: 0,
                                        offset: 0,
                                        base: bound,
                                    };
                                    stack.push(addition);
                                    coreferential_transition(loc, stack, references, f);
                                    stack.pop();
                                } else {
                                    if i == bytes.len() {
                                        coreferential_transition(loc, stack, references, f);
                                    }
                                    loc.ascend(i);
                                }
                            } else {
                                let addition = ExprEnv {
                                    n: 254,
                                    v: 0,
                                    offset: 0,
                                    base: bound,
                                };
                                stack.push(addition);
                                vs!(loc.child_mask(), e, false);
                                coreferential_transition(loc, stack, references, f);
                                stack.pop();
                            }
                        } else {
                            trace!(target: "coref trans", "varref <{},{i}> 'any'", e.n);
                            static NEW_VAR: u8 = item_byte(Tag::NewVar);
                            let addition = ExprEnv {
                                n: 255,
                                v: 0,
                                offset: 0,
                                base: Expr {
                                    ptr: ((&NEW_VAR) as *const u8).cast_mut(),
                                },
                            };
                            stack.push(addition);
                            vs!(loc.child_mask(), e, false);
                            coreferential_transition(loc, stack, references, f);
                            stack.pop();
                        }
                    }
                    Tag::SymbolSize(size) => {
                        vs!(loc.child_mask(), e, false);
                        if loc.descend_to_existing_byte(e_byte) {
                            if loc.descend_to_check(&*slice_from_raw_parts(
                                e.base.ptr.byte_add(e.offset as usize + 1),
                                size as usize,
                            )) {
                                coreferential_transition(loc, stack, references, f);
                            }
                            loc.ascend((size as usize) + 1); // The expression length + the e_byte
                        }
                    }
                    Tag::Arity(arity) => {
                        vs!(loc.child_mask(), e, false);
                        if loc.descend_to_existing_byte(e_byte) {
                            let stackl = stack.len();
                            e.args(&mut stack);
                            stack[stackl..].reverse();
                            coreferential_transition(loc, stack, references, f);
                            stack.truncate(stack.len() - arity as usize);
                            loc.ascend_byte();
                        }
                    }
                }

                stack.push(e);
            }
        }
    }
}

/// A compiled pattern-match instruction (the trie-adapted WAM head; see
/// `resources/exec_to_stream_transpiler_plan.md`). The op sequence is the pattern
/// in preorder; a compound's children are the following ops.
#[derive(Clone, Debug, PartialEq, Eq)]
enum MatchOp {
    /// `Tag::NewVar`: match any term here; `introduce` records a query variable id.
    Any { introduce: Option<u8> },
    /// `Tag::VarRef`: re-check a previously introduced top-level query variable.
    VarRef { index: u8 },
    /// `Tag::SymbolSize`: descend the exact symbol byte then its literal bytes.
    Symbol { e_byte: u8, bytes: Box<[u8]> },
    /// `Tag::Arity`: descend the arity byte; the following ops are its children.
    /// `span` is this whole compound subterm in preorder, including descendants.
    Compound { e_byte: u8, span: usize },
}

#[derive(Default)]
struct MatchProgramScratch {
    bound_bytes: Vec<u8>,
}

/// Compiles the conjunction sources into a preorder `MatchOp` program. Returns
/// `None` (caller falls back to the interpreter) when a `VarRef` appears before
/// its first occurrence in the compiled search order.
fn compile_match_program(sources: &[ExprEnv]) -> Option<Vec<MatchOp>> {
    if force_interpreted_matcher() {
        return None;
    }
    let mut ops = Vec::new();
    let mut introduced = [false; 256];
    for &source in sources {
        compile_match_factor(source, &mut ops, &mut introduced)?;
    }
    Some(ops)
}

/// Lowers one conjunction factor to `MatchOp`s in a single linear preorder pass
/// (a "flatterm" build; Christian, JAR 1993). `ExprEnv::args` is O(subtree), so the
/// old per-Arity-node recursion re-walked every descendant span and cost O(depth^2)
/// on deeply nested patterns (e.g. a copied Peano `S^N` in the MM2 exec stream).
/// This walk visits each item once: leaves close one slot of the innermost open
/// compound; an open compound's `span` is back-patched when its last child lands.
/// Produces byte-identical ops to `compile_match_factor_recursive` (the test oracle).
/// Returns `None` (caller falls back to the interpreter) on a forward VarRef.
fn compile_match_factor(
    e: ExprEnv,
    ops: &mut Vec<MatchOp>,
    introduced: &mut [bool; 256],
) -> Option<()> {
    let n = e.n;
    let mut next_var = e.v;
    let mut ptr = e.subsexpr().ptr;
    // Open compounds awaiting children: (op index of the Compound, children left).
    let mut open: Vec<(usize, u32)> = Vec::new();
    loop {
        match unsafe { byte_item(*ptr) } {
            Tag::NewVar => {
                let var = next_var;
                next_var = next_var.checked_add(1)?;
                let introduce = if n == 0 {
                    introduced[var as usize] = true;
                    Some(var)
                } else {
                    None
                };
                ops.push(MatchOp::Any { introduce });
                ptr = unsafe { ptr.byte_add(1) };
                close_compound_child(&mut open, ops);
            }
            Tag::VarRef(index) => {
                if n == 0 && introduced[index as usize] {
                    ops.push(MatchOp::VarRef { index });
                } else {
                    return None;
                }
                ptr = unsafe { ptr.byte_add(1) };
                close_compound_child(&mut open, ops);
            }
            Tag::SymbolSize(s) => {
                let bytes = unsafe {
                    slice_from_raw_parts(ptr.byte_add(1), s as usize)
                        .as_ref()
                        .unwrap()
                }
                .to_vec()
                .into_boxed_slice();
                ops.push(MatchOp::Symbol {
                    e_byte: unsafe { *ptr },
                    bytes,
                });
                ptr = unsafe { ptr.byte_add(1 + s as usize) };
                close_compound_child(&mut open, ops);
            }
            Tag::Arity(k) => {
                let here = ops.len();
                ops.push(MatchOp::Compound {
                    e_byte: unsafe { *ptr },
                    span: 0,
                });
                ptr = unsafe { ptr.byte_add(1) };
                if k == 0 {
                    if let MatchOp::Compound { span, .. } = &mut ops[here] {
                        *span = 1;
                    }
                    close_compound_child(&mut open, ops);
                } else {
                    open.push((here, k as u32));
                }
            }
        }
        if open.is_empty() {
            break;
        }
    }
    Some(())
}

/// Marks one child of the innermost open compound complete; when a compound's last
/// child lands, back-patches its `span` (op count of its whole subterm in preorder)
/// and propagates the completion to its parent.
fn close_compound_child(open: &mut Vec<(usize, u32)>, ops: &mut Vec<MatchOp>) {
    loop {
        let Some(top) = open.last_mut() else { return };
        top.1 -= 1;
        if top.1 != 0 {
            return;
        }
        let here = top.0;
        let span = ops.len() - here;
        if let MatchOp::Compound { span: s, .. } = &mut ops[here] {
            *s = span;
        }
        open.pop();
    }
}

/// Recursive reference lowering, kept as the differential oracle for the linear
/// `compile_match_factor`. O(depth^2) via `ExprEnv::args`; tests assert the two
/// produce identical op sequences (`compile_linear_equals_recursive`).
#[cfg(test)]
fn compile_match_factor_recursive(
    e: ExprEnv,
    ops: &mut Vec<MatchOp>,
    introduced: &mut [bool; 256],
    args_scratch: &mut Vec<ExprEnv>,
) -> Option<()> {
    let ptr = e.subsexpr().ptr;
    match unsafe { byte_item(*ptr) } {
        Tag::NewVar => {
            let introduce = if e.n == 0 {
                introduced[e.v as usize] = true;
                Some(e.v)
            } else {
                None
            };
            ops.push(MatchOp::Any { introduce });
        }
        Tag::VarRef(index) => {
            if e.n == 0 && introduced[index as usize] {
                ops.push(MatchOp::VarRef { index });
            } else {
                return None;
            }
        }
        Tag::SymbolSize(s) => {
            let bytes = unsafe {
                slice_from_raw_parts(ptr.byte_add(1), s as usize)
                    .as_ref()
                    .unwrap()
            }
            .to_vec()
            .into_boxed_slice();
            ops.push(MatchOp::Symbol {
                e_byte: unsafe { *ptr },
                bytes,
            });
        }
        Tag::Arity(_) => {
            let here = ops.len();
            ops.push(MatchOp::Compound {
                e_byte: unsafe { *ptr },
                span: 0,
            });
            let args_start = args_scratch.len();
            e.args(args_scratch);
            let args_end = args_scratch.len();
            let mut compiled = Some(());
            for i in args_start..args_end {
                let arg = args_scratch[i];
                if compile_match_factor_recursive(arg, ops, introduced, args_scratch).is_none() {
                    compiled = None;
                    break;
                }
            }
            args_scratch.truncate(args_start);
            compiled?;
            let emitted_span = ops.len() - here;
            if let MatchOp::Compound { span, .. } = &mut ops[here] {
                *span = emitted_span;
            }
        }
    }
    Some(())
}

/// Executes the compiled program against the trie cursor, producing exactly the
/// same `f(loc)` callbacks as `coreferential_transition` for compiled patterns.
/// `pc` is the program counter; trie traversal and product-factor switching are
/// identical to the interpreter because the same `loc` is reused.
fn execute_match_program<Z, F>(
    loc: &mut Z,
    ops: &[MatchOp],
    pc: usize,
    references: &mut Vec<u32>,
    scratch: &mut MatchProgramScratch,
    f: &mut F,
) where
    Z: ZipperMoving + Zipper + ZipperAbsolutePath + ZipperIteration,
    F: FnMut(&mut Z) -> (),
{
    let Some(op) = ops.get(pc) else {
        f(loc);
        return;
    };
    match op {
        MatchOp::Any { introduce } => {
            let restore = introduce.map(|index| {
                let idx = index as usize;
                if references.len() <= idx {
                    references.resize(idx + 1, u32::MAX);
                }
                let prev = references[idx];
                references[idx] = loc.path().len() as u32;
                (idx, prev)
            });
            match_any_term(loc, 1, ops, pc + 1, references, scratch, f);
            if let Some((idx, prev)) = restore {
                references[idx] = prev;
            }
        }
        MatchOp::VarRef { index } => {
            match_varref_program(loc, *index as usize, ops, pc + 1, references, scratch, f);
        }
        MatchOp::Symbol { e_byte, bytes } => {
            vs_match_program(loc, ops, pc + 1, references, scratch, f);
            if loc.descend_to_existing_byte(*e_byte) {
                if loc.descend_to_check(&bytes[..]) {
                    execute_match_program(loc, ops, pc + 1, references, scratch, f);
                }
                loc.ascend(bytes.len() + 1);
            }
        }
        MatchOp::Compound { e_byte, span } => {
            let pc_after = if compiled_matcher_compound_capture_enabled() {
                pc + *span
            } else {
                pc + 1
            };
            vs_match_program(loc, ops, pc_after, references, scratch, f);
            if loc.descend_to_existing_byte(*e_byte) {
                execute_match_program(loc, ops, pc + 1, references, scratch, f);
                loc.ascend_byte();
            }
        }
    }
}

fn rematch_bound_then_program<Z, F>(
    loc: &mut Z,
    bound_bytes: &[u8],
    ops: &[MatchOp],
    pc_after: usize,
    references: &mut Vec<u32>,
    scratch: &mut MatchProgramScratch,
    f: &mut F,
) where
    Z: ZipperMoving + Zipper + ZipperAbsolutePath + ZipperIteration,
    F: FnMut(&mut Z) -> (),
{
    // `bound_bytes` must be an OWNED copy of the bound subterm, made while the
    // capture pointer into `loc`'s path buffer was still fresh: the re-match below
    // descends `loc`, growing that buffer, and past its reserved capacity it
    // REALLOCATES — a raw pointer captured before any intervening descent (the
    // ground fast path runs `vs_match_program` and a byte descent first) dangles
    // and decodes garbage tag bytes (`byte_item`/`gnext` "reserved" panics on deep
    // terms; the reordered semi-naive delta reaches them by seeking deep bound
    // payloads). Same fix shape as the `bound_owned` copy in
    // `coreferential_transition`'s VarRef arm.
    let addition = ExprEnv {
        n: 254,
        v: 0,
        offset: 0,
        base: Expr {
            ptr: bound_bytes.as_ptr().cast_mut(),
        },
    };
    let mut stack = vec![addition];
    let mut rematch_references = Vec::new();
    coreferential_transition(loc, &mut stack, &mut rematch_references, &mut |loc| {
        execute_match_program(loc, ops, pc_after, references, scratch, f)
    });
}

fn match_varref_program<Z, F>(
    loc: &mut Z,
    index: usize,
    ops: &[MatchOp],
    pc_after: usize,
    references: &mut Vec<u32>,
    scratch: &mut MatchProgramScratch,
    f: &mut F,
) where
    Z: ZipperMoving + Zipper + ZipperAbsolutePath + ZipperIteration,
    F: FnMut(&mut Z) -> (),
{
    if index >= references.len() || references[index] == u32::MAX {
        match_any_term(loc, 1, ops, pc_after, references, scratch, f);
        return;
    }

    let bound = Expr {
        ptr: unsafe {
            loc.path()
                .as_ptr()
                .cast_mut()
                .offset(references[index] as _)
        },
    };

    if VARREF_FAST_RECHECK.with(|c| c.get()) && bound.is_ground() {
        // Keep the current bytes outside `scratch` while `vs_match_program`
        // runs; nested VarRefs may reuse the scratch buffer before we do the
        // exact descent below.
        let mut bound_bytes = std::mem::take(&mut scratch.bound_bytes);
        bound_bytes.clear();
        bound_bytes.extend_from_slice(unsafe { &*bound.span() });

        vs_match_program(loc, ops, pc_after, references, scratch, f);
        let mut consumed = 0usize;
        let mut variable_branch = false;
        while consumed < bound_bytes.len() {
            if loc
                .child_mask()
                .and(&ByteMask(VARS))
                .iter()
                .next()
                .is_some()
            {
                variable_branch = true;
                break;
            }
            if loc.descend_to_existing_byte(bound_bytes[consumed]) {
                consumed += 1;
            } else {
                break;
            }
        }
        if variable_branch {
            loc.ascend(consumed);
            // `bound_bytes` is the copy made before `vs_match_program` recursed (the
            // raw `bound` pointer may already dangle after that recursion).
            rematch_bound_then_program(loc, &bound_bytes, ops, pc_after, references, scratch, f);
        } else {
            if consumed == bound_bytes.len() {
                execute_match_program(loc, ops, pc_after, references, scratch, f);
            }
            loc.ascend(consumed);
        }
        bound_bytes.clear();
        if scratch.bound_bytes.capacity() < bound_bytes.capacity() {
            scratch.bound_bytes = bound_bytes;
        }
    } else {
        // Non-ground bound (the query var was bound to a data subterm containing
        // variables). `rematch_bound_then_program` re-matches that bound via
        // `coreferential_transition`, which already descends BOTH the data's variable
        // children and its concrete children. Adding the `vs_match_program` data-variable
        // shortcut here double-counts a coreferent data variable (it is captured once by
        // the shortcut and again by the rematch). The interpreted oracle
        // `coreferential_transition` has no such shortcut and emits once, so the rematch
        // alone is the complete-and-sound behavior. (The ground fast-path above still
        // needs `vs_match_program` because its exact-byte descent skips variable children.)
        // The copy happens HERE, while `bound` still points at live path bytes.
        let bound_owned: Vec<u8> = unsafe { &*bound.span() }.to_vec();
        rematch_bound_then_program(loc, &bound_owned, ops, pc_after, references, scratch, f);
    }
}

/// Mirrors the interpreter's `vs!`: a data variable at this position matches the
/// current pattern subterm, so descend each data-variable byte and resume at the
/// caller-supplied program counter.
fn vs_match_program<Z, F>(
    loc: &mut Z,
    ops: &[MatchOp],
    pc: usize,
    references: &mut Vec<u32>,
    scratch: &mut MatchProgramScratch,
    f: &mut F,
) where
    Z: ZipperMoving + Zipper + ZipperAbsolutePath + ZipperIteration,
    F: FnMut(&mut Z) -> (),
{
    let m = loc.child_mask().and(&ByteMask(VARS));
    let mut it = m.iter();
    while let Some(b) = it.next() {
        loc.descend_to_byte(b);
        execute_match_program(loc, ops, pc, references, scratch, f);
        loc.ascend_byte();
    }
}

/// Consumes `pending` whole data subtrees (the interpreter's NewVar "match any
/// term"), then resumes the program at `pc_after`. A data symbol decrements
/// pending; a data compound of arity `a` sets pending to `pending - 1 + a` (its
/// children, then the rest), reproducing the interpreter's synthetic-NewVar pushes.
fn match_any_term<Z, F>(
    loc: &mut Z,
    pending: usize,
    ops: &[MatchOp],
    pc_after: usize,
    references: &mut Vec<u32>,
    scratch: &mut MatchProgramScratch,
    f: &mut F,
) where
    Z: ZipperMoving + Zipper + ZipperAbsolutePath + ZipperIteration,
    F: FnMut(&mut Z) -> (),
{
    unsafe {
        if pending == 0 {
            execute_match_program(loc, ops, pc_after, references, scratch, f);
            return;
        }
        let cm = loc.child_mask();
        debug_assert!(cm.0[1] == 0);
        if match_any_term_fused() {
            // Fused single-word-walk. The original three passes each built a
            // `cm.and(ByteMask)` over all four words, drained an iterator, and
            // re-decoded every child byte with `byte_item`. Here each child mask
            // word is read at most once and the size/arity is the bit index
            // itself, with no `byte_item`. The visit order matches the three
            // passes exactly (see the byte encoding in expr::item_byte):
            //   VARS    = word2 (VarRef(b) = 0x80|b) then word3 bit0 (NewVar 0xC0)
            //   SIZES   = word3 bits 1..63 (SymbolSize(p) = 0xC0|p) ascending
            //   ARITIES = word0 (Arity(a) = a) ascending
            // VARS phase: VarRefs ascending, then NewVar.
            let mut w = cm.0[VARREF_WORD];
            while w != 0 {
                let bit = w.trailing_zeros();
                let b = (VARREF_TAG_HI | bit as u8) as u8;
                loc.descend_to_byte(b);
                match_any_term(loc, pending - 1, ops, pc_after, references, scratch, f);
                loc.ascend_byte();
                w &= w - 1;
            }
            if cm.0[HIGH_WORD] & NEWVAR_BIT != 0 {
                loc.descend_to_byte(NEWVAR_BYTE);
                match_any_term(loc, pending - 1, ops, pc_after, references, scratch, f);
                loc.ascend_byte();
            }
            // SIZES phase: SymbolSize children, the size is the bit index.
            let mut w = cm.0[HIGH_WORD] & !NEWVAR_BIT;
            while w != 0 {
                let size = w.trailing_zeros();
                let b = (HIGH_TAG_HI | size as u8) as u8;
                loc.descend_to_byte(b);
                if !loc.descend_first_k_path(size as _) {
                    unreachable_unchecked()
                }
                loop {
                    match_any_term(loc, pending - 1, ops, pc_after, references, scratch, f);
                    if !loc.to_next_k_path(size as _) {
                        break;
                    }
                }
                loc.ascend_byte();
                w &= w - 1;
            }
            // ARITIES phase: Arity children, the arity is the bit index.
            let mut w = cm.0[ARITY_WORD];
            while w != 0 {
                let a = w.trailing_zeros();
                let b = a as u8;
                loc.descend_to_byte(b);
                match_any_term(
                    loc,
                    pending - 1 + a as usize,
                    ops,
                    pc_after,
                    references,
                    scratch,
                    f,
                );
                loc.ascend_byte();
                w &= w - 1;
            }
        } else {
            let mut it = cm.and(&ByteMask(VARS)).iter();
            while let Some(b) = it.next() {
                loc.descend_to_byte(b);
                match_any_term(loc, pending - 1, ops, pc_after, references, scratch, f);
                loc.ascend_byte();
            }
            let mut it = cm.and(&ByteMask(SIZES)).iter();
            while let Some(b) = it.next() {
                let Tag::SymbolSize(size) = byte_item(b) else {
                    unreachable_unchecked()
                };
                loc.descend_to_byte(b);
                if !loc.descend_first_k_path(size as _) {
                    unreachable_unchecked()
                }
                loop {
                    match_any_term(loc, pending - 1, ops, pc_after, references, scratch, f);
                    if !loc.to_next_k_path(size as _) {
                        break;
                    }
                }
                loc.ascend_byte();
            }
            let mut it = cm.and(&ByteMask(ARITIES)).iter();
            while let Some(b) = it.next() {
                let Tag::Arity(a) = byte_item(b) else {
                    unreachable_unchecked()
                };
                loc.descend_to_byte(b);
                match_any_term(
                    loc,
                    pending - 1 + a as usize,
                    ops,
                    pc_after,
                    references,
                    scratch,
                    f,
                );
                loc.ascend_byte();
            }
        }
    }
}

unsafe extern "C" {
    fn longjmp(env: &mut [u64; 64], status: i32);
    fn setjmp(env: &mut [u64; 64]) -> i32;
}

pub struct ParDataParser<'a> {
    count: u64,
    #[cfg(feature = "interning")]
    buf: [u8; 8],
    #[cfg(not(feature = "interning"))]
    buf: [u8; 64],
    #[cfg(not(feature = "interning"))]
    truncated: u64,
    #[cfg(feature = "interning")]
    write_permit: WritePermit<'a>,
    #[cfg(not(feature = "interning"))]
    _mapping: std::marker::PhantomData<&'a SharedMappingHandle>,
}

impl<'a> Parser for ParDataParser<'a> {
    fn tokenizer<'r>(&'r mut self, s: &'r [u8]) -> &'r [u8] {
        self.count += 1;
        #[cfg(feature = "interning")]
        {
            self.buf = self.write_permit.get_sym_or_insert(s);
            return &self.buf[..];
        }
        #[cfg(not(feature = "interning"))]
        {
            let mut l = s.len();
            if l > 63 {
                self.truncated += 1;
                // panic!("len greater than 63 bytes {}", std::str::from_utf8(s).unwrap_or(format!("{:?}", s).as_str()))
                l = 63
            }
            self.buf[..l].clone_from_slice(&s[..l]);
            return &self.buf[..l];
        }
    }
}

impl<'a> ParDataParser<'a> {
    pub fn new(handle: &'a SharedMappingHandle) -> Self {
        #[cfg(not(feature = "interning"))]
        let _ = handle;

        Self {
            count: 3,
            #[cfg(feature = "interning")]
            buf: (3u64).to_be_bytes(),
            #[cfg(not(feature = "interning"))]
            buf: [0; 64],
            #[cfg(not(feature = "interning"))]
            truncated: 0u64,
            #[cfg(feature = "interning")]
            write_permit: handle.try_aquire_permission().unwrap(),
            #[cfg(not(feature = "interning"))]
            _mapping: std::marker::PhantomData,
        }
    }
}

pub struct SpaceTranscriber<'a, 'b, 'c> {
    count: usize,
    wz: &'c mut WriteZipperUntracked<'a, 'b, ()>,
    pdp: ParDataParser<'a>,
}
impl<'a, 'b, 'c> SpaceTranscriber<'a, 'b, 'c> {
    #[inline(always)]
    fn write<S: AsRef<[u8]>>(&mut self, s: S) {
        let token = self.pdp.tokenizer(s.as_ref());
        let mut path = vec![item_byte(Tag::SymbolSize(token.len() as u8))];
        path.extend(token);
        self.wz.descend_to(&path[..]);
        self.wz.set_val(());
        self.wz.ascend(path.len());
    }
}
impl<'a, 'b, 'c> mork_frontend::json_parser::Transcriber for SpaceTranscriber<'a, 'b, 'c> {
    #[inline(always)]
    fn descend_index(&mut self, i: usize, first: bool) -> () {
        if first {
            self.wz.descend_to(&[item_byte(Tag::Arity(2))]);
        }
        let index = i.to_string();
        let token = self.pdp.tokenizer(index.as_bytes());
        self.wz
            .descend_to(&[item_byte(Tag::SymbolSize(token.len() as u8))]);
        self.wz.descend_to(token);
    }
    #[inline(always)]
    fn ascend_index(&mut self, i: usize, last: bool) -> () {
        let index = i.to_string();
        self.wz
            .ascend(self.pdp.tokenizer(index.as_bytes()).len() + 1);
        if last {
            self.wz.ascend(1);
        }
    }
    #[inline(always)]
    fn write_empty_array(&mut self) -> () {
        self.write("[]");
        self.count += 1;
    }
    #[inline(always)]
    fn descend_key(&mut self, k: &str, first: bool) -> () {
        if first {
            self.wz.descend_to(&[item_byte(Tag::Arity(2))]);
        }
        let token = self.pdp.tokenizer(k.as_bytes());
        self.wz
            .descend_to(&[item_byte(Tag::SymbolSize(token.len() as u8))]);
        self.wz.descend_to(token);
    }
    #[inline(always)]
    fn ascend_key(&mut self, k: &str, last: bool) -> () {
        let token = self.pdp.tokenizer(k.as_bytes());
        self.wz.ascend(token.len() + 1);
        if last {
            self.wz.ascend(1);
        }
    }
    #[inline(always)]
    fn write_empty_object(&mut self) -> () {
        self.write("{}");
        self.count += 1;
    }
    #[inline(always)]
    fn write_string(&mut self, s: &str) -> () {
        self.write(s);
        self.count += 1;
    }
    #[inline(always)]
    fn write_number(&mut self, negative: bool, mantissa: u64, exponent: i16) -> () {
        let mut s = String::new();
        if negative {
            s.push('-');
        }
        s.push_str(mantissa.to_string().as_str());
        if exponent != 0 {
            s.push('e');
            s.push_str(exponent.to_string().as_str());
        }
        self.write(s);
        self.count += 1;
    }
    #[inline(always)]
    fn write_true(&mut self) -> () {
        self.write("true");
        self.count += 1;
    }
    #[inline(always)]
    fn write_false(&mut self) -> () {
        self.write("false");
        self.count += 1;
    }
    #[inline(always)]
    fn write_null(&mut self) -> () {
        self.write("null");
        self.count += 1;
    }
    #[inline(always)]
    fn begin(&mut self) -> () {}
    #[inline(always)]
    fn end(&mut self) -> () {}
}

pub struct ASpaceTranscriber<'a, 'c> {
    count: usize,
    wz: &'c mut Vec<u8>,
    pdp: ParDataParser<'a>,
}
impl<'a, 'c> ASpaceTranscriber<'a, 'c> {
    #[inline(always)]
    fn write<S: AsRef<[u8]>>(&mut self, s: S) -> impl Iterator<Item = Vec<u8>> {
        gen move {
            let token_len = {
                let token = self.pdp.tokenizer(s.as_ref());
                self.wz.push(item_byte(Tag::SymbolSize(token.len() as u8)));
                self.wz.extend_from_slice(token);
                token.len()
            };
            let path = self.wz.clone();
            self.wz.truncate(self.wz.len() - (token_len + 1));
            yield path;
        }
    }
    fn destruct(self) -> (usize, &'c mut Vec<u8>, ParDataParser<'a>) {
        (self.count, self.wz, self.pdp)
    }
}
impl<'a, 'c> mork_frontend::json_parser::ATranscriber<Vec<u8>> for ASpaceTranscriber<'a, 'c> {
    #[inline(always)]
    fn descend_index(&mut self, i: usize, first: bool) -> () {
        if first {
            self.wz.push(item_byte(Tag::Arity(2)));
        }
        let index = i.to_string();
        let token = self.pdp.tokenizer(index.as_bytes());
        self.wz.push(item_byte(Tag::SymbolSize(token.len() as u8)));
        self.wz.extend_from_slice(token);
    }
    #[inline(always)]
    fn ascend_index(&mut self, i: usize, last: bool) -> () {
        let index = i.to_string();
        let token_len = self.pdp.tokenizer(index.as_bytes()).len();
        self.wz.truncate(self.wz.len() - (token_len + 1));
        if last {
            self.wz.truncate(self.wz.len() - 1);
        }
    }
    #[inline(always)]
    fn write_empty_array(&mut self) -> impl Iterator<Item = Vec<u8>> {
        self.count += 1;
        self.write("[]")
    }
    #[inline(always)]
    fn descend_key(&mut self, k: &str, first: bool) -> () {
        if first {
            self.wz.push(item_byte(Tag::Arity(2)));
        }
        let token = self.pdp.tokenizer(k.as_bytes());
        self.wz.push(item_byte(Tag::SymbolSize(token.len() as u8)));
        self.wz.extend_from_slice(token);
    }
    #[inline(always)]
    fn ascend_key(&mut self, k: &str, last: bool) -> () {
        let token = self.pdp.tokenizer(k.as_bytes());
        self.wz.truncate(self.wz.len() - (token.len() + 1));
        if last {
            self.wz.truncate(self.wz.len() - 1);
        }
    }
    #[inline(always)]
    fn write_empty_object(&mut self) -> impl Iterator<Item = Vec<u8>> {
        self.count += 1;
        self.write("{}")
    }
    #[inline(always)]
    fn write_string(&mut self, s: &str) -> impl Iterator<Item = Vec<u8>> {
        self.count += 1;
        self.write(s)
    }
    #[inline(always)]
    fn write_number(
        &mut self,
        negative: bool,
        mantissa: u64,
        exponent: i16,
    ) -> impl Iterator<Item = Vec<u8>> {
        let mut buf = [0u8; 64];
        let mut cur = std::io::Cursor::new(&mut buf[..]);
        if negative {
            write!(cur, "-").unwrap();
        }
        write!(cur, "{}", mantissa).unwrap();
        if exponent != 0 {
            write!(cur, "e{}", exponent).unwrap();
        }
        let len = cur.position() as usize;
        self.count += 1;
        self.write(cur.into_inner()[..len].to_vec())
    }
    #[inline(always)]
    fn write_true(&mut self) -> impl Iterator<Item = Vec<u8>> {
        self.count += 1;
        self.write("true")
    }
    #[inline(always)]
    fn write_false(&mut self) -> impl Iterator<Item = Vec<u8>> {
        self.count += 1;
        self.write("false")
    }
    #[inline(always)]
    fn write_null(&mut self) -> impl Iterator<Item = Vec<u8>> {
        self.count += 1;
        self.write("null")
    }
    #[inline(always)]
    fn begin(&mut self) -> () {}
    #[inline(always)]
    fn end(&mut self) -> () {}
}

#[macro_export]
macro_rules! prefix {
    ($space:ident, $s:literal) => {{
        let mut src = $crate::__mork_expr::parse!($s);
        let q = $crate::__mork_expr::Expr {
            ptr: src.as_mut_ptr(),
        };
        let mut pdp = $crate::space::ParDataParser::new(&$space.sm);
        let mut buf = [0u8; 2048];
        let p = $crate::__mork_expr::Expr {
            ptr: buf.as_mut_ptr(),
        };
        q.substitute_symbols_with(&mut $crate::__mork_expr::ExprZipper::new(p), |x, oz| {
            let token =
                <_ as $crate::__mork_frontend::bytestring_parser::Parser>::tokenizer(&mut pdp, x);
            oz.write_symbol(token);
            token.len()
        });
        let prefix = unsafe {
            $crate::__mork_expr::Expr { ptr: p.ptr }
                .prefix_non_proper()
                .as_ref()
                .unwrap()
        };
        let prefix: &'static [u8] = Box::leak(prefix.to_vec().into_boxed_slice());
        $crate::prefix::Prefix::<'static> { slice: prefix }
    }};
}

#[macro_export]
macro_rules! expr {
    ($space:ident, $s:literal) => {{
        let mut src = mork_expr::parse!($s);
        let q = mork_expr::Expr {
            ptr: src.as_mut_ptr(),
        };
        let table = $space.sym_table();
        let mut pdp = $crate::space::ParDataParser::new(&table);
        let mut buf = [0u8; 4096];
        let p = mork_expr::Expr {
            ptr: buf.as_mut_ptr(),
        };
        let used = q.substitute_symbols_with(&mut mork_expr::ExprZipper::new(p), |x, oz| {
            let token = <_ as mork_frontend::bytestring_parser::Parser>::tokenizer(&mut pdp, x);
            oz.write_symbol(token);
            token.len()
        });
        unsafe {
            let b = std::alloc::alloc(std::alloc::Layout::array::<u8>(used.len()).unwrap());
            std::ptr::copy_nonoverlapping(p.ptr, b, used.len());
            mork_expr::Expr { ptr: b }
        }
    }};
    ($space:ident, $s:expr) => {{
        let mut src = mork_expr::parse::<4096>($s);
        let q = mork_expr::Expr {
            ptr: src.as_mut_ptr(),
        };
        let table = $space.sym_table();
        let mut pdp = $crate::space::ParDataParser::new(&table);
        let mut buf = [0u8; 4096];
        let p = mork_expr::Expr {
            ptr: buf.as_mut_ptr(),
        };
        let used = q.substitute_symbols_with(&mut mork_expr::ExprZipper::new(p), |x, oz| {
            let token = <_ as mork_frontend::bytestring_parser::Parser>::tokenizer(&mut pdp, x);
            oz.write_symbol(token);
            token.len()
        });
        unsafe {
            let b = std::alloc::alloc(std::alloc::Layout::array::<u8>(used.len()).unwrap());
            std::ptr::copy_nonoverlapping(p.ptr, b, used.len());
            mork_expr::Expr { ptr: b }
        }
    }};
}

#[macro_export]
macro_rules! sexpr {
    ($space:ident, $e:expr) => {{
        let mut v = vec![];
        let e: mork_expr::Expr = $e;
        let table = $space.sym_table();
        e.serialize_with(&mut v, |s, out| {
            $crate::space::write_serialized_symbol(&table, s, out);
        });
        String::from_utf8(v).unwrap_or_else(|_| unsafe { e.span().as_ref()}.map(mork_expr::serialize).unwrap_or("<null>".to_string()))
    }};
}

impl Space {
    pub fn new() -> Self {
        Self {
            btm: PathMap::new(),
            sm: SharedMapping::new(),
            mmaps: HashMap::new(),
            z3s: HashMap::new(),
            #[cfg(feature = "einsum")]
            tensors: HashMap::new(),
            last_merkleize: Instant::now(),
            timing: false,
            bridge_sidecar: None,
            bridge_remove_gen: 0,
            bridge_closures: HashMap::new(),
            sni_rule_seen: None,
            sni_removal_seen: false,
            sni_force_naive: false,
            sni_delta_calls: 0,
            sni_retract_mode: SniRetractMode::Naive,
            sni_dred_repairs: 0,
            sni_dred_fallbacks: 0,
            sni_removal_gen: 0,
            sni_dred_skip_rederive: false,
            sni_dred_force_fallback: false,
        }
    }

    /// Cost-bounded ShardZipper decomposition of this space's pathmap into a
    /// covering antichain of shard prefixes, each holding at most `l_max` values.
    /// See `shard_zipper`.
    pub fn decompose_shards(&self, l_max: usize) -> Vec<Vec<u8>> {
        crate::shard_zipper::decompose_by_cost(&self.btm, l_max)
    }

    /// Number of values under a shard prefix (the shard cost L(s)).
    pub fn shard_cost(&self, prefix: &[u8]) -> usize {
        crate::shard_zipper::shard_cost(&self.btm, prefix)
    }

    /// Sweep one shard in place: capture the subtrie at `prefix`, run `kernel`,
    /// replay its patch log, and reintegrate (ShardZipper Phi_s). The pathmap is
    /// the authority, so the only state to fix afterwards is the performance-only
    /// derived caches (the persistent join sidecar and the maintained closures),
    /// which are dropped and rebuild lazily. Patch-log keys must use
    /// already-interned symbols, since a sweep edits existing structure.
    pub fn sweep_shard<K>(&mut self, prefix: &[u8], kernel: K)
    where
        K: FnOnce(&PathMap<()>) -> crate::shard_zipper::PatchLog,
    {
        crate::shard_zipper::sweep_shard(&mut self.btm, prefix, kernel);
        self.invalidate_bridge_caches();
    }

    /// Decompose at `l_max` and sweep every shard, the per-shard sweeps running
    /// in parallel on the independent shard maps (ShardZipper sweep_all_parallel).
    pub fn sweep_all_shards_parallel<K>(&mut self, l_max: usize, kernel: K)
    where
        K: Fn(&[u8], &PathMap<()>) -> crate::shard_zipper::PatchLog + Sync,
    {
        crate::shard_zipper::sweep_all_parallel(&mut self.btm, l_max, kernel);
        self.invalidate_bridge_caches();
    }

    /// Drops the performance-only derived state after a direct pathmap edit, so
    /// the next sidecar-routed transform rebuilds it from the current `btm`.
    fn invalidate_bridge_caches(&mut self) {
        self.bridge_remove_gen += 1;
        self.bridge_sidecar = None;
        self.bridge_closures.clear();
        // A direct edit may have removed facts; trip the semi-naive gate so any
        // in-progress IC loop falls back to naive (cleared at loop arm). Pre-loop
        // loads call this too, but metta_calculus clears the flag when it arms.
        self.sni_removal_seen = true;
    }

    /// Materialise the binary relation whose head is `relation_head_encoded`
    /// (the exact encoded bytes of the relation symbol, e.g. `SymbolSize` plus
    /// `edge`) into a CSR adjacency for the linalg numeric kernels. Builds a
    /// fresh sidecar over the current pathmap, finds the relation, and hands it
    /// to `graph_tensor`. Returns `None` if the relation is absent. The numeric
    /// half of the ShardZipper materialise step at the Space level.
    #[cfg(feature = "einsum")]
    pub fn relation_adjacency(
        &self,
        relation_head_encoded: &[u8],
    ) -> Option<crate::graph_tensor::RelationAdjacency> {
        let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&self.btm).ok()?;
        let head = sidecar.term_id_for_encoded(relation_head_encoded)?;
        Some(crate::graph_tensor::RelationAdjacency::from_sidecar(
            &sidecar, head,
        ))
    }

    /// Compute two-hop path counts over the binary relation `relation_head_encoded`
    /// with the linalg SpGEMM, and write each result back as a fact
    /// `(result_symbol a c n)` (the count `n` as a decimal symbol). Returns the
    /// number of new facts written. Closes the symbolic to numeric to symbolic
    /// loop: edges in, SpGEMM, derived facts out. Bridge caches are invalidated
    /// when anything is written.
    #[cfg(feature = "einsum")]
    pub fn write_two_hop_counts(
        &mut self,
        relation_head_encoded: &[u8],
        result_symbol: &[u8],
    ) -> usize {
        let Some(adjacency) = self.relation_adjacency(relation_head_encoded) else {
            return 0;
        };
        let two_hop = adjacency.two_hop();
        let mut written = 0usize;
        for (src, dst, count) in adjacency.enumerate(&two_hop) {
            let count_symbol = crate::graph_tensor::encode_symbol(count.to_string().as_bytes());
            let key = crate::graph_tensor::encode_fact(result_symbol, &[&src, &dst, &count_symbol]);
            if self.btm.insert(&key, ()).is_none() {
                written += 1;
            }
        }
        if written > 0 {
            self.invalidate_bridge_caches();
        }
        written
    }

    /// Run PageRank over the binary relation `relation_head_encoded` and write
    /// each node's score back as a fact `(result_symbol node score)` (the score
    /// formatted to six decimals as a symbol). Returns the number of new facts
    /// written. `damping` is usually 0.85. Bridge caches invalidate on write.
    #[cfg(feature = "einsum")]
    pub fn write_pagerank(
        &mut self,
        relation_head_encoded: &[u8],
        result_symbol: &[u8],
        damping: f32,
        iterations: usize,
    ) -> usize {
        let Some(adjacency) = self.relation_adjacency(relation_head_encoded) else {
            return 0;
        };
        let mut written = 0usize;
        for (node, score) in adjacency.pagerank(damping, iterations) {
            let score_symbol = crate::graph_tensor::encode_symbol(format!("{score:.6}").as_bytes());
            let key = crate::graph_tensor::encode_fact(result_symbol, &[&node, &score_symbol]);
            if self.btm.insert(&key, ()).is_none() {
                written += 1;
            }
        }
        if written > 0 {
            self.invalidate_bridge_caches();
        }
        written
    }

    /// Creates an empty atom space that shares this space's symbol table.
    ///
    /// Sharing the symbol table is required when combining atom tries, because
    /// interned symbols are stored as compact IDs in the encoded paths.
    pub fn fork_empty(&self) -> Self {
        Self {
            btm: PathMap::new(),
            sm: self.sm.clone(),
            mmaps: HashMap::new(),
            z3s: HashMap::new(),
            #[cfg(feature = "einsum")]
            tensors: HashMap::new(),
            last_merkleize: Instant::now(),
            timing: self.timing,
            bridge_sidecar: None,
            bridge_remove_gen: 0,
            bridge_closures: HashMap::new(),
            sni_rule_seen: None,
            sni_removal_seen: false,
            sni_force_naive: false,
            sni_delta_calls: 0,
            sni_retract_mode: SniRetractMode::Naive,
            sni_dred_repairs: 0,
            sni_dred_fallbacks: 0,
            sni_removal_gen: 0,
            sni_dred_skip_rederive: false,
            sni_dred_force_fallback: false,
        }
    }

    /// Returns true when two spaces encode symbols through the same table.
    pub fn shares_symbol_table_with(&self, other: &Self) -> bool {
        std::ptr::eq::<SharedMapping>(&*self.sm, &*other.sm)
    }

    fn ensure_compatible_atom_trie(&self, other: &Self) -> Result<(), String> {
        if self.shares_symbol_table_with(other) {
            Ok(())
        } else {
            Err("cannot combine atom tries from spaces with different symbol tables".to_string())
        }
    }

    fn fork_with_btm(&self, btm: PathMap<()>) -> Self {
        let mut space = self.fork_empty();
        space.btm = btm;
        space
    }

    /// Returns a new space containing atoms present in either compatible space.
    ///
    /// This combines only the PathMap-backed atom trie. External resource state
    /// such as mmap, Z3, or tensor handles is intentionally not merged.
    pub fn atom_union(&self, other: &Self) -> Result<Self, String> {
        self.ensure_compatible_atom_trie(other)?;
        Ok(self.fork_with_btm(self.btm.join(&other.btm)))
    }

    /// Returns a new space containing atoms present in both compatible spaces.
    ///
    /// This combines only the PathMap-backed atom trie. External resource state
    /// such as mmap, Z3, or tensor handles is intentionally not merged.
    pub fn atom_intersection(&self, other: &Self) -> Result<Self, String> {
        self.ensure_compatible_atom_trie(other)?;
        Ok(self.fork_with_btm(self.btm.meet(&other.btm)))
    }

    /// Returns a new space containing atoms in `self` that are not in `other`.
    ///
    /// This combines only the PathMap-backed atom trie. External resource state
    /// such as mmap, Z3, or tensor handles is intentionally not merged.
    pub fn atom_subtract(&self, other: &Self) -> Result<Self, String> {
        self.ensure_compatible_atom_trie(other)?;
        Ok(self.fork_with_btm(self.btm.subtract(&other.btm)))
    }

    pub fn parse_sexpr(&mut self, r: &[u8], buf: *mut u8) -> Result<(Expr, usize), ParserError> {
        let mut it = Context::new(r);
        let mut parser = ParDataParser::new(&self.sm);
        let mut ez = ExprZipper::new(Expr { ptr: buf });
        parser
            .sexpr(&mut it, &mut ez)
            .map(|_| (Expr { ptr: buf }, ez.loc))
    }

    /// Remy :I want to really discourage the use of this method, it needs to be exposed if we want to use the debugging macros `expr` and `sexpr` without giving acces directly to the field
    #[doc(hidden)]
    pub fn sym_table(&self) -> SharedMappingHandle {
        self.sm.clone()
    }

    pub fn statistics(&self) {
        println!("val count {}", self.btm.val_count());
    }

    /*
        pub fn load_csv<R : Read>(&mut self, prefix: Prefix, mut r: R, sm: &mut SymbolMapping, separator: u8) -> Result<usize, String> {
        let mut i = 0;
        let mut buf = vec![];
        let mut stack = [0u8; 2048];

        match r.read_to_end(&mut buf) {
            Ok(read) => {
                let mut wz = self.btm.write_zipper_at_path(prefix.path());
                for sv in buf.split(|&x| x == b'\n') {
                    if sv.len() == 0 { continue }
                    let mut a = 0;
                    let e = Expr{ ptr: stack.as_mut_ptr() };
                    let mut ez = ExprZipper::new(e);
                    ez.loc += 1;
                    let rown = sm.tokenizer(unsafe { String::from_utf8_unchecked(i.to_string().into_bytes()) });
                    ez.write_symbol(&rown[..]);
                    ez.loc += rown.len() + 1;
                    a += 1;
                    for symbol in sv.split(|&x| x == separator) {
                        let internal = sm.tokenizer(unsafe { String::from_utf8_unchecked(symbol.to_vec()) });
                        ez.write_symbol(&internal[..]);
                        ez.loc += internal.len() + 1;
                        a += 1;
                    }
                    let total = ez.loc;
                    ez.reset();
                    ez.write_arity(a);
                    wz.descend_to(&stack[..total]);
                    wz.set_val(());
                    wz.reset();
                    i += 1;
                }
            }
            Err(e) => { return Err(format!("{:?}", e)) }
        }

        Ok(i)
    }
     */

    pub fn load_csv(
        &mut self,
        r: &[u8],
        pattern: Expr,
        template: Expr,
        seperator: u8,
    ) -> Result<usize, String> {
        let constant_template_prefix = unsafe {
            template
                .prefix()
                .unwrap_or_else(|_| template.span())
                .as_ref()
                .unwrap()
        };
        let mut wz = self.btm.write_zipper_at_path(constant_template_prefix);
        let buf = [0u8; 2048];

        let mut i = 0usize;
        let mut stack = [0u8; 2048];
        let mut pdp = ParDataParser::new(&self.sm);
        for sv in r.split(|&x| x == b'\n') {
            if sv.len() == 0 {
                continue;
            }
            let mut a = 0;
            let e = Expr {
                ptr: stack.as_mut_ptr(),
            };
            let mut ez = ExprZipper::new(e);
            ez.loc += 1;
            let index = i.to_string();
            let num = pdp.tokenizer(index.as_bytes());
            // ez.write_symbol(i.to_be_bytes().as_slice());
            ez.write_symbol(num);
            // ez.loc += 9;
            ez.loc += num.len() + 1;

            for symbol in sv.split(|&x| x == seperator) {
                let internal = pdp.tokenizer(symbol);
                ez.write_symbol(&internal[..]);
                ez.loc += internal.len() + 1;
                a += 1;
            }
            let total = ez.loc;
            ez.reset();
            ez.write_arity(a + 1);

            let data = &stack[..total];
            let mut oz = ExprZipper::new(Expr {
                ptr: buf.as_ptr().cast_mut(),
            });
            match (Expr {
                ptr: data.as_ptr().cast_mut(),
            }
            .transformData(pattern, template, &mut oz))
            {
                Ok(()) => {}
                Err(_e) => continue,
            }
            let new_data = &buf[..oz.loc];
            wz.descend_to(&new_data[constant_template_prefix.len()..]);
            wz.set_val(());
            wz.reset();
            i += 1;
        }

        Ok(i)
    }

    pub fn load_json(&mut self, r: &[u8]) -> Result<usize, String> {
        let mut wz = self.btm.write_zipper();
        let mut st = SpaceTranscriber {
            count: 0,
            wz: &mut wz,
            pdp: ParDataParser::new(&self.sm),
        };
        let mut p =
            mork_frontend::json_parser::Parser::new(unsafe { std::str::from_utf8_unchecked(r) });
        p.parse(&mut st).unwrap();
        Ok(st.count)
    }

    pub fn json_to_paths<W: std::io::Write>(
        &mut self,
        r: &[u8],
        d: &mut W,
    ) -> Result<usize, String> {
        let mut sink = pathmap::paths_serialization::paths_serialization_owned_sink(d);

        let mut wz = Vec::with_capacity(4096);
        let mut st = ASpaceTranscriber {
            count: 0,
            wz: &mut wz,
            pdp: ParDataParser::new(&self.sm),
        };

        let mut p =
            mork_frontend::json_parser::Parser::new(unsafe { std::str::from_utf8_unchecked(r) });
        let mut coro = p.parse_stream(&mut st);
        while let CoroutineState::Yielded(n) = Pin::new(&mut coro).resume(()) {
            Pin::new(&mut sink).resume(Some(n));
        }
        match Pin::new(&mut sink).resume(None) {
            CoroutineState::Yielded(_) => {
                panic!()
            }
            CoroutineState::Complete(summary) => {
                println!("{:?}", summary)
            }
        }
        drop(coro);
        Ok(st.count)
    }

    pub fn jsonl_to_paths<W: std::io::Write>(
        &mut self,
        r: &[u8],
        d: &mut W,
    ) -> Result<(usize, usize), String> {
        let mut lines = 0usize;
        let mut count = 0usize;
        let mut sink = pathmap::paths_serialization::paths_serialization_owned_sink(d);
        let mut mpdp = Some(ParDataParser::new(&self.sm));
        let mut wz = Vec::with_capacity(4096);
        let jsonl_symbol = mpdp.as_mut().unwrap().tokenizer("JSONL".as_bytes());
        wz.push(item_byte(Tag::Arity(3)));
        wz.push(item_byte(Tag::SymbolSize(jsonl_symbol.len() as u8)));
        wz.extend_from_slice(jsonl_symbol);
        wz.push(item_byte(Tag::SymbolSize(8)));

        for line in unsafe { std::str::from_utf8_unchecked(r).lines() } {
            wz.extend_from_slice(lines.to_be_bytes().as_slice());
            let mut st = ASpaceTranscriber {
                count: 0,
                wz: &mut wz,
                pdp: mpdp.take().unwrap(),
            };

            let mut p = mork_frontend::json_parser::Parser::new(line);
            let mut coro = p.parse_stream(&mut st);
            while let CoroutineState::Yielded(n) = Pin::new(&mut coro).resume(()) {
                println!("jsonl {}", serialize(&n));
                Pin::new(&mut sink).resume(Some(n));
            }
            drop(coro);
            let (line_count, _, pdp) = st.destruct();
            wz.truncate(wz.len() - 8);
            lines += 1;
            count += line_count;
            let _previous = mpdp.insert(pdp);
        }
        match Pin::new(&mut sink).resume(None) {
            CoroutineState::Yielded(_) => {
                panic!()
            }
            CoroutineState::Complete(summary) => {
                println!("{:?}", summary)
            }
        }
        Ok((lines, count))
    }

    pub fn load_jsonl(&mut self, r: &[u8]) -> Result<(usize, usize), String> {
        let mut wz = self.btm.write_zipper();
        let mut lines = 0usize;
        let mut count = 0usize;
        let mut pdp = ParDataParser::new(&self.sm);
        let spo_symbol = pdp.tokenizer("JSONL".as_bytes());
        let mut path = vec![
            item_byte(Tag::Arity(3)),
            item_byte(Tag::SymbolSize(spo_symbol.len() as u8)),
        ];
        path.extend_from_slice(spo_symbol);
        wz.descend_to(&path[..]);
        for line in unsafe { std::str::from_utf8_unchecked(r).lines() } {
            wz.descend_to(lines.to_be_bytes());
            let mut st = SpaceTranscriber {
                count: 0,
                wz: &mut wz,
                pdp: ParDataParser::new(&self.sm),
            };
            let mut p = mork_frontend::json_parser::Parser::new(line);
            p.parse(&mut st).unwrap();
            count += st.count;
            lines += 1;
            wz.ascend(8);
            if lines > 0 && lines % 1000_000 == 0 {
                println!("parsed {} JSON lines ({} paths)", lines, count);
            }
        }
        Ok((lines, count))
    }

    pub fn load_json_(
        &mut self,
        r: &[u8],
        _pattern: Expr,
        template: Expr,
    ) -> Result<usize, String> {
        let constant_template_prefix = unsafe {
            template
                .prefix()
                .unwrap_or_else(|_| template.span())
                .as_ref()
                .unwrap()
        };
        let mut wz = self.btm.write_zipper_at_path(constant_template_prefix);

        let mut st = SpaceTranscriber {
            count: 0,
            wz: &mut wz,
            pdp: ParDataParser::new(&self.sm),
        };
        let mut p =
            mork_frontend::json_parser::Parser::new(unsafe { std::str::from_utf8_unchecked(r) });
        p.parse(&mut st).unwrap();
        Ok(st.count)
    }

    #[cfg(feature = "neo4j")]
    pub fn load_neo4j_triples(
        &mut self,
        uri: &str,
        user: &str,
        pass: &str,
    ) -> Result<usize, String> {
        use neo4rs::*;
        let graph = Graph::new(uri, user, pass).unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            // .unhandled_panic(tokio::runtime::UnhandledPanic::Ignore)
            .build()
            .unwrap();
        let mut pdp = ParDataParser::new(&self.sm);

        let mut count = 0;

        let mut result = rt
            .block_on(graph.execute(query("MATCH (s)-[p]->(o) RETURN id(s), type(p), id(o)")))
            .unwrap();
        let spo_symbol = pdp.tokenizer("SPO".as_bytes()).to_vec();
        while let Ok(Some(row)) = rt.block_on(result.next()) {
            let s: i64 = row.get("id(s)").unwrap();
            let p: String = row.get("type(p)").unwrap();
            let o: i64 = row.get("id(o)").unwrap();
            // std::hint::black_box((s, p, o));
            let mut buf = [0u8; 64];
            let e = Expr {
                ptr: buf.as_mut_ptr(),
            };
            let mut ez = ExprZipper::new(e);
            ez.write_arity(4);
            ez.loc += 1;
            {
                ez.write_symbol(&spo_symbol[..]);
                ez.loc += spo_symbol.len() + 1;
            }
            {
                let s_bytes = s.to_be_bytes();
                let internal = pdp.tokenizer(&s_bytes).to_vec();
                ez.write_symbol(&internal);
                ez.loc += internal.len() + 1;
            }
            {
                let internal = pdp.tokenizer(p.as_bytes());
                ez.write_symbol(&internal[..]);
                ez.loc += internal.len() + 1;
            }
            {
                let o_bytes = o.to_be_bytes();
                let internal = pdp.tokenizer(&o_bytes).to_vec();
                ez.write_symbol(&internal);
                ez.loc += internal.len() + 1;
            }
            // println!("{}", serialize(ez.span()));
            self.btm.insert(ez.span(), ());
            count += 1;
            if count % 1000000 == 0 {
                println!("{count} triples");
            }
        }
        Ok(count)
    }

    #[cfg(feature = "neo4j")]
    pub fn load_neo4j_node_properties(
        &mut self,
        uri: &str,
        user: &str,
        pass: &str,
    ) -> Result<(usize, usize), String> {
        use neo4rs::*;
        let graph = Graph::new(uri, user, pass).unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            // .unhandled_panic(tokio::runtime::UnhandledPanic::Ignore)
            .build()
            .unwrap();
        let mut pdp = ParDataParser::new(&self.sm);
        let zh = self.btm.zipper_head();
        let mut wz = zh.write_zipper_at_exclusive_path(&[]).unwrap();
        let sa_symbol = pdp.tokenizer("NKV".as_bytes());
        let mut nodes = 0;
        let mut attributes = 0;

        wz.descend_to_byte(item_byte(Tag::Arity(4)));
        wz.descend_to_byte(item_byte(Tag::SymbolSize(sa_symbol.len() as _)));
        wz.descend_to(sa_symbol);

        let mut result = rt
            .block_on(graph.execute(query("MATCH (s) RETURN id(s), s")))
            .unwrap();
        while let Ok(Some(row)) = rt.block_on(result.next()) {
            let s: i64 = row.get("id(s)").unwrap();
            let s_bytes = s.to_be_bytes();
            let internal_s = pdp.tokenizer(&s_bytes).to_vec();
            wz.descend_to_byte(item_byte(Tag::SymbolSize(internal_s.len() as _)));
            wz.descend_to(&internal_s);

            let a: BoltMap = row.get("s").unwrap();

            for (bs, bt) in a.value.iter() {
                let internal_k = pdp.tokenizer(bs.value.as_bytes()).to_vec();
                wz.descend_to_byte(item_byte(Tag::SymbolSize(internal_k.len() as _)));
                wz.descend_to(&internal_k);

                let BoltType::String(bv) = bt else {
                    unreachable!()
                };
                if bv.value.starts_with("[") && bv.value.ends_with("]") {
                    for chunk in bv.value[1..bv.value.len() - 1].split(", ") {
                        let c = if chunk.starts_with("\"") && chunk.ends_with("\"") {
                            &chunk[1..chunk.len() - 1]
                        } else {
                            chunk
                        };
                        let internal_v = pdp.tokenizer(c.as_bytes()).to_vec();
                        wz.descend_to_byte(item_byte(Tag::SymbolSize(internal_v.len() as _)));
                        wz.descend_to(&internal_v);

                        wz.set_val(());

                        wz.ascend(internal_v.len() + 1);
                    }
                } else {
                    let internal_v = pdp.tokenizer(bv.value.as_bytes()).to_vec();
                    wz.descend_to_byte(item_byte(Tag::SymbolSize(internal_v.len() as _)));
                    wz.descend_to(&internal_v);

                    wz.set_val(());

                    wz.ascend(internal_v.len() + 1);
                }

                wz.ascend(internal_k.len() + 1);
                attributes += 1;
            }

            wz.ascend(internal_s.len() + 1);
            nodes += 1;
            if nodes % 1000000 == 0 {
                println!("{attributes} attributes of {nodes}");
            }
        }
        Ok((nodes, attributes))
    }

    #[cfg(feature = "neo4j")]
    pub fn load_neo4j_node_labels(
        &mut self,
        uri: &str,
        user: &str,
        pass: &str,
    ) -> Result<(usize, usize), String> {
        use neo4rs::*;
        let graph = Graph::new(uri, user, pass).unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            // .unhandled_panic(tokio::runtime::UnhandledPanic::Ignore)
            .build()
            .unwrap();
        let mut pdp = ParDataParser::new(&self.sm);
        let zh = self.btm.zipper_head();
        let mut wz = zh.write_zipper_at_exclusive_path(&[]).unwrap();
        let sa_symbol = pdp.tokenizer("NL".as_bytes());
        let mut nodes = 0;
        let mut labels = 0;

        wz.descend_to_byte(item_byte(Tag::Arity(3)));
        wz.descend_to_byte(item_byte(Tag::SymbolSize(sa_symbol.len() as _)));
        wz.descend_to(sa_symbol);

        let mut result = rt
            .block_on(graph.execute(query("MATCH (s) RETURN id(s), labels(s)")))
            .unwrap();
        while let Ok(Some(row)) = rt.block_on(result.next()) {
            let s: i64 = row.get("id(s)").unwrap();
            let s_bytes = s.to_be_bytes();
            let internal_s = pdp.tokenizer(&s_bytes).to_vec();
            wz.descend_to_byte(item_byte(Tag::SymbolSize(internal_s.len() as _)));
            wz.descend_to(&internal_s);

            let a: BoltList = row.get("labels(s)").unwrap();

            for bl in a.value.iter() {
                let BoltType::String(bv) = bl else {
                    unreachable!()
                };

                let internal_v = pdp.tokenizer(bv.value.as_bytes()).to_vec();
                wz.descend_to_byte(item_byte(Tag::SymbolSize(internal_v.len() as _)));
                wz.descend_to(&internal_v);

                wz.set_val(());

                wz.ascend(internal_v.len() + 1);

                labels += 1;
            }

            wz.ascend(internal_s.len() + 1);
            nodes += 1;
            if nodes % 1000000 == 0 {
                println!("{labels} labels of {nodes}");
            }
        }
        Ok((nodes, labels))
    }

    pub fn add_all_sexpr(&mut self, r: &[u8]) -> Result<usize, String> {
        self.load_all_sexpr_impl(r, true)
    }
    pub fn remove_all_sexpr(&mut self, r: &[u8]) -> Result<usize, String> {
        self.load_all_sexpr_impl(r, false)
    }
    pub fn load_all_sexpr_impl(&mut self, r: &[u8], add: bool) -> Result<usize, String> {
        let mut data = parser_output_buffer();
        let mut it = Context::new(r);
        let mut i = 0;
        let mut parser = ParDataParser::new(&self.sm);
        loop {
            data.clear();
            match parser.sexpr_to_vec(&mut it, &mut data) {
                Ok(()) => {
                    if add {
                        self.btm.insert(&data, ());
                    } else {
                        self.btm.remove(&data);
                    }
                }
                Err(ParserError::InputFinished) => break,
                Err(other) => return Err(format!("{other:?} at byte {}", it.loc)),
            }
            i += 1;
            it.variables.clear();
        }
        Ok(i)
    }

    pub fn add_sexpr(&mut self, r: &[u8], pattern: Expr, template: Expr) -> Result<usize, String> {
        self.load_sexpr_impl(r, pattern, template, true)
    }
    pub fn remove_sexpr(
        &mut self,
        r: &[u8],
        pattern: Expr,
        template: Expr,
    ) -> Result<usize, String> {
        self.load_sexpr_impl(r, pattern, template, false)
    }
    pub fn load_sexpr_impl(
        &mut self,
        r: &[u8],
        pattern: Expr,
        template: Expr,
        add: bool,
    ) -> Result<usize, String> {
        let constant_template_prefix = unsafe {
            template
                .prefix()
                .unwrap_or_else(|_| template.span())
                .as_ref()
                .unwrap()
        };
        let mut wz = self.btm.write_zipper_at_path(constant_template_prefix);
        let mut buffer = template_output_buffer();
        let mut data = parser_output_buffer();
        let mut it = Context::new(r);
        let mut i = 0;
        let mut parser = ParDataParser::new(&self.sm);
        loop {
            data.clear();
            match parser.sexpr_to_vec(&mut it, &mut data) {
                Ok(()) => {
                    match (Expr {
                        ptr: data.as_ptr().cast_mut(),
                    }
                    .transformDataInto(pattern, template, &mut buffer))
                    {
                        Ok(()) => {}
                        Err(_) => continue,
                    }
                    wz.move_to_path(&buffer[constant_template_prefix.len()..]);
                    if add {
                        wz.set_val(());
                    } else {
                        wz.remove_val(true);
                    }
                    wz.reset();
                }
                Err(ParserError::InputFinished) => break,
                Err(other) => return Err(format!("{other:?} at byte {}", it.loc)),
            }
            i += 1;
            it.variables.clear();
        }
        Ok(i)
    }

    pub fn dump_all_sexpr<W: Write>(&self, w: &mut W) -> Result<usize, String> {
        let mut rz = self.btm.read_zipper();
        let mut i = 0usize;
        while rz.to_next_val() {
            // println!("{}", serialize(rz.path()));
            Expr {
                ptr: rz.path().as_ptr().cast_mut(),
            }
            .serialize2_with(
                w,
                |s, out| {
                    write_serialized_symbol(&self.sm, s, out);
                },
                |i, _intro| Expr::VARNAMES[i as usize],
            );
            // w.write(serialize(rz.path()).as_bytes());
            w.write(&[b'\n']).map_err(|x| x.to_string())?;
            i += 1;
        }
        Ok(i)
    }

    pub fn dump_sexpr<W: Write>(&self, pattern: Expr, template: Expr, w: &mut W) -> usize {
        let mut buffer = template_output_buffer();
        let mut pat = vec![
            item_byte(Tag::Arity(2)),
            item_byte(Tag::SymbolSize(1)),
            b',',
        ];
        pat.extend_from_slice(unsafe { pattern.span().as_ref().unwrap() });

        let mut stack = Vec::new();
        let mut assignments = Vec::new();
        Self::query_multi(
            &self.btm,
            Expr {
                ptr: pat.leak().as_mut_ptr(),
            },
            |refs_bindings, _loc| 'query: {
                match refs_bindings {
                    Ok(_) => {
                        assert!(false)
                    }
                    Err(ref bindings) => {
                        buffer.clear();

                        let (oi, ni, true) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
                            0,
                            0,
                            0,
                            pattern,
                            bindings,
                            buffer,
                            stack,
                            assignments
                        ) else {
                            break 'query false;
                        };

                        buffer.clear();

                        let (_, _, true) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
                            0,
                            oi,
                            ni,
                            template,
                            bindings,
                            buffer,
                            stack,
                            assignments
                        ) else {
                            break 'query false;
                        };
                    }
                }

                Expr {
                    ptr: buffer.as_ptr().cast_mut(),
                }
                .serialize2_with(
                    w,
                    |s, out| {
                        write_serialized_symbol(&self.sm, s, out);
                    },
                    |i, _intro| Expr::VARNAMES[i as usize],
                );
                w.write(&[b'\n']).map_err(|x| x.to_string()).unwrap();

                true
            },
        )
    }

    pub fn backup_symbols<OutDirPath: AsRef<std::path::Path>>(
        &self,
        _path: OutDirPath,
    ) -> Result<(), std::io::Error> {
        #[cfg(feature = "interning")]
        {
            self.sm.serialize(_path)
        }
        #[cfg(not(feature = "interning"))]
        {
            Ok(())
        }
    }

    pub fn restore_symbols(
        &mut self,
        _path: impl AsRef<std::path::Path>,
    ) -> Result<(), std::io::Error> {
        #[cfg(feature = "interning")]
        {
            self.sm = SharedMapping::deserialize(_path)?;
        }
        Ok(())
    }

    pub fn backup_tree<OutDirPath: AsRef<std::path::Path>>(
        &self,
        path: OutDirPath,
    ) -> Result<(), std::io::Error> {
        pathmap::arena_compact::ArenaCompactTree::dump_from_zipper(
            self.btm.read_zipper(),
            |_v| 0,
            path,
        )
        .map(|_tree| ())
    }

    pub fn restore_tree(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), std::io::Error> {
        let tree = pathmap::arena_compact::ArenaCompactTree::open_mmap(path)?;
        let mut rz = tree.read_zipper();
        while rz.to_next_val() {
            self.btm.insert(rz.path(), ());
        }
        Ok(())
    }

    pub fn backup_paths<OutDirPath: AsRef<std::path::Path>>(
        &self,
        path: OutDirPath,
    ) -> Result<pathmap::paths_serialization::SerializationStats, std::io::Error> {
        let mut file = File::create(path).unwrap();
        pathmap::paths_serialization::serialize_paths(self.btm.read_zipper(), &mut file)
    }

    pub fn restore_paths<OutDirPath: AsRef<std::path::Path>>(
        &mut self,
        path: OutDirPath,
    ) -> Result<pathmap::paths_serialization::DeserializationStats, std::io::Error> {
        let mut file = File::open(path).unwrap();
        pathmap::paths_serialization::deserialize_paths(self.btm.write_zipper(), &mut file, ())
    }

    #[cfg(test)]
    fn query_factor_rank(
        btm: &PathMap<()>,
        source: ExprEnv,
        prefix_cardinalities: &mut BTreeMap<Vec<u8>, usize>,
        shape_cardinalities: &mut BTreeMap<Vec<u8>, Option<QueryShapeSummary>>,
    ) -> QueryFactorRank {
        Self::query_factor_rank_with_btm_count(
            btm,
            source,
            prefix_cardinalities,
            shape_cardinalities,
            Some(btm.val_count()),
        )
    }

    fn query_factor_rank_with_btm_count(
        btm: &PathMap<()>,
        source: ExprEnv,
        prefix_cardinalities: &mut BTreeMap<Vec<u8>, usize>,
        shape_cardinalities: &mut BTreeMap<Vec<u8>, Option<QueryShapeSummary>>,
        btm_val_count: Option<usize>,
    ) -> QueryFactorRank {
        let mut prefix_len = 0;
        let mut estimated_cardinality = usize::MAX;
        let mut prefix_cardinality_lookup = false;
        let mut prefix_cardinality_cache_hit = false;
        let mut prefix_for_shape = None;
        unsafe {
            if let Some(prefix) = source
                .subsexpr()
                .prefix()
                .unwrap_or_else(|span| span)
                .as_ref()
            {
                prefix_len = prefix.len();
                if !prefix.is_empty() {
                    prefix_for_shape = Some(prefix.to_vec());
                    prefix_cardinality_lookup = true;
                    estimated_cardinality = match prefix_cardinalities.get(prefix) {
                        Some(&cached) => {
                            prefix_cardinality_cache_hit = true;
                            cached
                        }
                        None => {
                            let count = btm.read_zipper_at_path(prefix).val_count();
                            prefix_cardinalities.insert(prefix.to_vec(), count);
                            count
                        }
                    };
                }
            }
        }

        let mut constant_items = 0;
        let mut variable_items = 0;
        let mut new_var_items = 0;
        let mut var_ref_items = 0;
        let mut ez = ExprZipper::new(source.subsexpr());
        loop {
            match ez.tag() {
                Tag::NewVar => {
                    variable_items += 1;
                    new_var_items += 1;
                }
                Tag::VarRef(_) => {
                    variable_items += 1;
                    var_ref_items += 1;
                }
                Tag::SymbolSize(_) | Tag::Arity(_) => constant_items += 1,
            }
            if !ez.next() {
                break;
            }
        }

        let mut shape_cardinality_lookup = false;
        let mut shape_cardinality_cache_hit = false;
        let mut shape_side_index_lookup = false;
        let mut shape_side_index_hit = false;
        let mut shape_side_index_insert = false;
        let mut shape_cardinality_scan = false;
        let mut shape_cardinality_refined = false;
        let mut shape_cardinality_skipped = false;
        let mut variable_domain_refined = false;
        let mut ground_root_matches = 0usize;
        let mut schematic_root_matches = 0usize;
        let mut min_variable_domain_cardinality = None;
        let mut max_variable_domain_cardinality = None;
        let mut variable_domains = Vec::new();
        macro_rules! apply_shape_summary {
            ($summary:expr) => {{
                estimated_cardinality = $summary.cardinality;
                ground_root_matches = $summary.ground_root_matches;
                schematic_root_matches = $summary.schematic_root_matches;
                min_variable_domain_cardinality = $summary.min_variable_domain_cardinality;
                max_variable_domain_cardinality = $summary.max_variable_domain_cardinality;
                variable_domains = Self::query_factor_variables(source)
                    .into_iter()
                    .zip($summary.variable_domains.iter().cloned())
                    .collect();
                variable_domain_refined = min_variable_domain_cardinality.is_some();
                shape_cardinality_refined = true;
            }};
        }
        if variable_items > 0 && prefix_len > 0 && estimated_cardinality != usize::MAX {
            shape_cardinality_lookup = true;
            if estimated_cardinality > QUERY_SHAPE_CARDINALITY_SCAN_LIMIT {
                shape_cardinality_skipped = true;
            } else if let Some(shape_key) = Self::query_factor_shape_cache_key(source) {
                match shape_cardinalities.get(&shape_key) {
                    Some(cached) => {
                        shape_cardinality_cache_hit = true;
                        if let Some(summary) = cached.as_ref() {
                            apply_shape_summary!(summary);
                        }
                    }
                    None => {
                        let side_index_key = btm_val_count.zip(prefix_for_shape.as_ref()).map(
                            |(btm_val_count, prefix)| QueryShapeSideIndexKey {
                                btm_val_count,
                                prefix_cardinality: estimated_cardinality,
                                prefix: prefix.clone(),
                                shape: shape_key.clone(),
                            },
                        );
                        let side_index_summary = side_index_key.as_ref().and_then(|key| {
                            shape_side_index_lookup = true;
                            query_shape_side_index().lock().unwrap().get(key)
                        });
                        if let Some(cached) = side_index_summary {
                            shape_side_index_hit = true;
                            shape_cardinalities.insert(shape_key, cached.clone());
                            if let Some(summary) = cached.as_ref() {
                                apply_shape_summary!(summary);
                            }
                        } else {
                            shape_cardinality_scan = true;
                            let summary = prefix_for_shape.as_deref().and_then(|prefix| {
                                Self::query_projection_maps(
                                    btm,
                                    source,
                                    prefix,
                                    estimated_cardinality,
                                    btm_val_count,
                                )
                                .map(|projection| projection.to_shape_summary())
                            });
                            if let Some(side_index_key) = side_index_key {
                                query_shape_side_index()
                                    .lock()
                                    .unwrap()
                                    .insert(side_index_key, summary.clone());
                                shape_side_index_insert = true;
                            }
                            shape_cardinalities.insert(shape_key, summary.clone());
                            if let Some(summary) = summary.as_ref() {
                                apply_shape_summary!(summary);
                            }
                        }
                    }
                }
            } else {
                shape_cardinality_skipped = true;
            }
        }

        QueryFactorRank {
            estimated_cardinality,
            ground_root_matches,
            schematic_root_matches,
            min_variable_domain_cardinality,
            max_variable_domain_cardinality,
            variable_domains,
            prefix_len,
            constant_items,
            variable_items,
            new_var_items,
            var_ref_items,
            prefix_cardinality_lookup,
            prefix_cardinality_cache_hit,
            shape_cardinality_lookup,
            shape_cardinality_cache_hit,
            shape_side_index_lookup,
            shape_side_index_hit,
            shape_side_index_insert,
            shape_cardinality_scan,
            shape_cardinality_refined,
            shape_cardinality_skipped,
            variable_domain_refined,
        }
    }

    fn query_factor_variables(source: ExprEnv) -> Vec<(u8, u8)> {
        let mut vars = Vec::new();
        let mut local_newvars = source.v;
        let mut ez = ExprZipper::new(source.subsexpr());
        loop {
            let var = match ez.tag() {
                Tag::NewVar => {
                    let var = (source.n, local_newvars);
                    local_newvars += 1;
                    Some(var)
                }
                Tag::VarRef(original_var) => Some((source.n, original_var)),
                Tag::SymbolSize(_) | Tag::Arity(_) => None,
            };
            if let Some(var) = var {
                if !vars.contains(&var) {
                    vars.push(var);
                }
            }
            if !ez.next() {
                break;
            }
        }
        vars
    }

    /// Lowers a renormalized MM2 conjunction body `(, f_1 .. f_k)` into a
    /// sidecar join plan. This is the glue between the live conjunctive-query
    /// body and the `BindingSidecarPlan` planner: the same plan whose output the
    /// existing harness checks row-for-row against the ProductZipper through
    /// `compare_query_projection_product_trace_to_binding_relation`.
    ///
    /// Each factor `(rel a_1 .. a_m)` whose head `rel` is a concrete symbol and
    /// whose arguments are all distinct variables becomes a
    /// `BindingAccessPlan::Arrangement` over `rel`/`m`: the relation's interned
    /// `TermId`, a positional key order, and a projection sending argument
    /// position `j` to the `BindingVar` for that argument's variable. Variables
    /// are numbered densely in first-occurrence order across factors, keyed by
    /// the live `(introduction, variable)` De Bruijn pairs that
    /// `query_factor_variables` produces, so a variable shared between two
    /// factors collapses to one `BindingVar` and the schema lines up with the
    /// live coreference layer (`coreferential_transition`).
    ///
    /// A keyed arrangement is a two-level Generalized Hash Trie in the Free Join
    /// sense (Wang, Willsey, Suciu, "Free Join: Unifying Worst-Case Optimal and
    /// Traditional Joins", PACMMOD 1(2):150, 2023): a hash table whose key is the
    /// arranged argument prefix. So this lowering and the sidecar's
    /// variable-at-a-time trie kernel are the binary-join and worst-case-optimal
    /// ends of one plan space, which is what lets the planner choose between
    /// them per body.
    ///
    /// Returns `None`, so the caller keeps the ProductZipper, when any factor is
    /// not a flat symbol-headed relation of distinct variables: a constant or
    /// nested argument, a repeated variable, a variable head, a relation symbol
    /// absent from the snapshot, or more than 64 distinct variables. Those
    /// cases fall back to the pattern-trie factor (`BindingAccessPlan::Pattern`)
    /// or an equality filter.
    pub fn lower_query_to_sidecar_plan(
        sources: &[ExprEnv],
        sidecar: &mut crate::term_identity::TermIdentitySidecar,
    ) -> Option<crate::binding_plan::BindingSidecarPlan> {
        use crate::arrangements::{ArrangementDescriptor, ArrangementProjection};
        use crate::binding_plan::{BindingAccessPlan, BindingSidecarPlan, PatternProjection};

        // The live renormalizer numbers variables through a `[u8::MAX; 64]` map
        // (`append_renormalized_query_factor`), so a body never carries more than
        // 64 distinct variables. Keep the same ceiling.
        const MAX_LOWERED_VARIABLES: usize = 64;

        // A factor is either a flat distinct-variable relation (an equi-join atom,
        // lowered to an arrangement) or anything else (constants, repeated
        // variables, nesting), lowered to an exact-filtering pattern factor. The
        // keys are the factor's distinct `(u8, u8)` variable keys in
        // first-occurrence order.
        enum LoweredFactor {
            Arrangement {
                relation: TermId,
                keys: Vec<(u8, u8)>,
            },
            Pattern {
                pattern: TermId,
                keys: Vec<(u8, u8)>,
            },
        }

        if sources.is_empty() {
            return None;
        }

        // Phase 1: classify each factor. `lower_flat_relation_factor` borrows the
        // sidecar immutably; `lower_pattern_factor` interns the renumbered pattern
        // term and so needs it mutably.
        let mut lowered: Vec<LoweredFactor> = Vec::with_capacity(sources.len());
        for &source in sources {
            if let Some((relation, keys)) = Self::lower_flat_relation_factor(source, sidecar) {
                lowered.push(LoweredFactor::Arrangement { relation, keys });
            } else if let Some((pattern, keys)) = Self::lower_pattern_factor(source, sidecar) {
                lowered.push(LoweredFactor::Pattern { pattern, keys });
            } else {
                return None;
            }
        }

        // Phase 2: assign a dense `BindingVar` to each distinct key in
        // first-occurrence order across factors. A variable shared between factors
        // collapses to one `BindingVar`.
        let mut variable_for_key: BTreeMap<(u8, u8), BindingVar> = BTreeMap::new();
        let mut variable_order: Vec<BindingVar> = Vec::new();
        for factor in &lowered {
            let (LoweredFactor::Arrangement { keys, .. } | LoweredFactor::Pattern { keys, .. }) =
                factor;
            for &key in keys {
                if !variable_for_key.contains_key(&key) {
                    if variable_order.len() >= MAX_LOWERED_VARIABLES {
                        return None;
                    }
                    let var = BindingVar(variable_order.len() as u8);
                    variable_for_key.insert(key, var);
                    variable_order.push(var);
                }
            }
        }

        // Phase 3: build the access plans against the global schema.
        let mut factors = Vec::with_capacity(lowered.len());
        for factor in lowered {
            match factor {
                LoweredFactor::Arrangement { relation, keys } => {
                    let argument_count = u8::try_from(keys.len()).ok()?;
                    let schema: Vec<BindingVar> =
                        keys.iter().map(|key| variable_for_key[key]).collect();
                    let argument_positions: Vec<u8> = (0..argument_count).collect();
                    let descriptor = ArrangementDescriptor::new(
                        relation,
                        argument_count,
                        argument_positions.clone(),
                    )
                    .ok()?;
                    let projection =
                        ArrangementProjection::new(argument_count, schema, argument_positions)
                            .ok()?;
                    factors.push(BindingAccessPlan::Arrangement {
                        descriptor,
                        projection,
                    });
                }
                LoweredFactor::Pattern { pattern, keys } => {
                    let schema: Vec<BindingVar> =
                        keys.iter().map(|key| variable_for_key[key]).collect();
                    // The renumbered pattern introduces its NewVars in
                    // first-occurrence order, the same order `lower_pattern`
                    // numbers slots, so the user slots are the identity.
                    let user_slots: Vec<u8> = (0..u8::try_from(keys.len()).ok()?).collect();
                    let projection = PatternProjection::new(schema, user_slots).ok()?;
                    factors.push(BindingAccessPlan::Pattern {
                        pattern,
                        projection,
                    });
                }
            }
        }

        Some(BindingSidecarPlan::new(factors, variable_order))
    }

    /// Extracts the relation `TermId` and per-argument-position variable keys for
    /// a flat symbol-headed relation factor `(rel a_1 .. a_m)` of distinct
    /// variables. Returns `None` when the factor is not of that form: a variable
    /// or nested head, a constant or nested argument, a repeated variable (a
    /// self-join the positional projection cannot express), or a relation symbol
    /// not interned in `sidecar`. The variable keying mirrors
    /// `query_factor_variables` exactly so the `(introduction, variable)` pairs
    /// line up with the live binding map and a shared variable resolves to the
    /// same `BindingVar`.
    fn lower_flat_relation_factor(
        source: ExprEnv,
        sidecar: &crate::term_identity::TermIdentitySidecar,
    ) -> Option<(TermId, Vec<(u8, u8)>)> {
        let mut local_newvars = source.v;
        let mut ez = ExprZipper::new(source.subsexpr());
        let arity = match ez.tag() {
            Tag::Arity(arity) if arity >= 1 => arity,
            _ => return None,
        };
        if !ez.next() {
            return None;
        }
        let relation = match ez.tag() {
            Tag::SymbolSize(_) => {
                let encoded = unsafe { ez.subexpr().span().as_ref()? };
                sidecar.term_id_for_encoded(encoded)?
            }
            _ => return None,
        };
        let mut keys = Vec::with_capacity(usize::from(arity - 1));
        let mut seen = BTreeSet::new();
        for _ in 1..arity {
            if !ez.next() {
                return None;
            }
            let key = match ez.tag() {
                Tag::NewVar => {
                    let key = (source.n, local_newvars);
                    local_newvars += 1;
                    key
                }
                Tag::VarRef(original) => (source.n, original),
                Tag::SymbolSize(_) | Tag::Arity(_) => return None,
            };
            // A repeated variable is a self-join the arrangement cannot express;
            // let it fall through to the pattern factor.
            if !seen.insert(key) {
                return None;
            }
            keys.push(key);
        }
        // A flat relation has no tokens past its last argument.
        if ez.next() {
            return None;
        }
        Some((relation, keys))
    }

    /// Renumbers a factor into a self-contained schematic term, interns it, and
    /// returns its pattern `TermId` together with its distinct variable keys in
    /// first-occurrence order (one per `lower_pattern` slot). The factor is
    /// rewritten so the first occurrence of each variable is a `NewVar` and every
    /// repeat is a `VarRef(local_slot)`; a `VarRef` to a variable introduced in an
    /// earlier factor becomes a `NewVar` here, its first occurrence in the
    /// standalone term. That keeps the interned term well-formed and makes its
    /// `NewVar` order match `lower_pattern`'s slot numbering, so the projection
    /// user slots are the identity. The `(u8, u8)` keys mirror
    /// `query_factor_variables` so shared variables resolve to the same
    /// `BindingVar` as the other factors.
    fn lower_pattern_factor(
        source: ExprEnv,
        sidecar: &mut crate::term_identity::TermIdentitySidecar,
    ) -> Option<(TermId, Vec<(u8, u8)>)> {
        // VarRef slots use a six-bit field, so a single factor cannot carry more
        // than 64 distinct variables.
        const MAX_FACTOR_VARIABLES: usize = 64;

        let mut local_newvars = source.v;
        let mut slot_for_key: BTreeMap<(u8, u8), u8> = BTreeMap::new();
        let mut slot_keys: Vec<(u8, u8)> = Vec::new();
        let mut encoded: Vec<u8> = Vec::new();
        let mut ez = ExprZipper::new(source.subsexpr());
        loop {
            match ez.tag() {
                Tag::Arity(arity) => encoded.push(item_byte(Tag::Arity(arity))),
                Tag::SymbolSize(size) => {
                    encoded.push(item_byte(Tag::SymbolSize(size)));
                    match ez.item() {
                        Err(bytes) => encoded.extend_from_slice(bytes),
                        Ok(_) => return None,
                    }
                }
                Tag::NewVar => {
                    let key = (source.n, local_newvars);
                    local_newvars += 1;
                    if slot_keys.len() >= MAX_FACTOR_VARIABLES {
                        return None;
                    }
                    let slot = slot_keys.len() as u8;
                    slot_for_key.insert(key, slot);
                    slot_keys.push(key);
                    encoded.push(item_byte(Tag::NewVar));
                }
                Tag::VarRef(original) => {
                    let key = (source.n, original);
                    if let Some(&slot) = slot_for_key.get(&key) {
                        encoded.push(item_byte(Tag::VarRef(slot)));
                    } else {
                        if slot_keys.len() >= MAX_FACTOR_VARIABLES {
                            return None;
                        }
                        let slot = slot_keys.len() as u8;
                        slot_for_key.insert(key, slot);
                        slot_keys.push(key);
                        encoded.push(item_byte(Tag::NewVar));
                    }
                }
            }
            if !ez.next() {
                break;
            }
        }
        let pattern = sidecar.insert_term(&encoded).ok()?;
        Some((pattern, slot_keys))
    }

    /// Builds the sidecar from a term snapshot, lowers the body, runs the
    /// selected join kernel, and compares the result row-for-row against the live
    /// ProductZipper for the same body. This composes the bridge end to end:
    /// lowering (a), sidecar execution (b), and the acceptance comparison (c).
    /// The live caller can gate on `.matched` and keep the ProductZipper
    /// otherwise.
    ///
    /// Returns `None` when the body does not lower to a sidecar plan (an
    /// unsupported factor, an `I`/`O` head, an empty body), the signal to keep the
    /// ProductZipper. `Some(comparison)` carries the verdict; `comparison.matched`
    /// is true when the sidecar plan reproduces the ProductZipper result exactly
    /// as a set.
    ///
    /// This interns the whole snapshot per call (`extend_from_pathmap`), so it is
    /// a correctness harness rather than a performance path. Incremental
    /// subspace interning is the separate speed slice.
    pub fn validate_lowered_plan_against_product(
        btm: &PathMap<()>,
        pat_expr: Expr,
    ) -> Option<QueryProjectionProductTraceComparison> {
        let mut args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut args);
        let sources = args.get(1..)?;
        if sources.is_empty() {
            return None;
        }

        let mut sidecar = Self::build_subspace_sidecar(btm, sources)?;

        let plan = Self::lower_query_to_sidecar_plan(sources, &mut sidecar)?;
        let selected = plan.execute_selected(&sidecar).ok()?;

        // Number variables in the same first-occurrence order the lowering used,
        // so the trace's `BindingVar` columns line up with the relation schema.
        let mut variable_for_key: BTreeMap<(u8, u8), BindingVar> = BTreeMap::new();
        for &source in sources {
            for key in Self::query_factor_variables(source) {
                let next = BindingVar(variable_for_key.len() as u8);
                variable_for_key.entry(key).or_insert(next);
            }
        }
        let query_variables_by_binding: Vec<(BindingVar, (u8, u8))> = variable_for_key
            .iter()
            .map(|(&key, &var)| (var, key))
            .collect();

        let trace = Self::trace_query_projection_product_candidates(
            btm,
            pat_expr,
            query_variables_by_binding,
            plan.variable_order().to_vec(),
        );
        let comparison = compare_query_projection_product_trace_to_binding_relation(
            &trace,
            &selected.relation,
            |term| {
                sidecar
                    .get_term(term)
                    .map(|record| record.encoded().to_vec())
            },
        );
        Some(comparison)
    }

    /// Builds a term sidecar interning only the facts each factor reads (its
    /// relation prefix from `Expr::prefix`), not the whole space. The join touches
    /// only facts under these prefixes, so the result matches a whole-space intern
    /// while the scan is bounded by the query's own relations. Falls back to the
    /// whole space when any factor has no usable constant prefix.
    fn build_subspace_sidecar(
        btm: &PathMap<()>,
        sources: &[ExprEnv],
    ) -> Option<crate::term_identity::TermIdentitySidecar> {
        let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
        match sources
            .iter()
            .map(|&source| query_source_prefix(source).filter(|prefix| !prefix.is_empty()))
            .collect::<Option<Vec<Vec<u8>>>>()
        {
            Some(prefixes) => {
                let mut interned = BTreeSet::new();
                for prefix in prefixes {
                    if interned.insert(prefix.clone()) {
                        sidecar
                            .extend_from_pathmap_under_prefix(btm, &prefix)
                            .ok()?;
                    }
                }
            }
            None => {
                sidecar.extend_from_pathmap(btm).ok()?;
            }
        }
        Some(sidecar)
    }

    /// Instantiates the rule's templates against one reconstructed binding map
    /// and hands each successful output path to `out`. The pattern is applied
    /// once to seed the intro counters, then each template; only outputs that
    /// pass the post-application cycle check are emitted. Shared by the
    /// ProductZipper reference emit, the materialised sidecar emit (both collect
    /// into a set), and the streamed sidecar emit (writes into the space), so all
    /// three produce byte-identical output paths. The scratch buffers are caller
    /// owned and cleared per application, so a streaming loop reuses one
    /// allocation across every tuple.
    /// Applies the pattern body once against `bindings` to compute the intro-count
    /// seed `(original_intros, new_intros)` the templates are instantiated with.
    /// `None` when the pattern application hits a cycle (the caller skips the
    /// match). The pattern application writes only into the scratch `buffer`
    /// (cleared here), never into the output. For ground bindings (the join/flip
    /// emit, every variable bound to an interned ground term) the result is
    /// invariant across matches, so `sidecar_emit_stream_with` computes it once
    /// and reuses it instead of re-walking the whole body per output row.
    fn pattern_template_intros(
        bindings: &BTreeMap<(u8, u8), ExprEnv>,
        pat_expr: Expr,
        mut buffer: &mut Vec<u8>,
        mut stack: &mut Vec<(u8, u8)>,
        mut assignments: &mut Vec<(u8, u8)>,
    ) -> Option<(u8, u8)> {
        buffer.clear();
        let (oi, ni, ok) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
            0,
            0,
            0,
            pat_expr,
            bindings,
            buffer,
            stack,
            assignments
        );
        ok.then_some((oi, ni))
    }

    /// Instantiates one template seeded with the pattern intro counts `(oi, ni)`
    /// and hands its output path to `out` when the template passes the cycle
    /// check. The scratch buffers are caller owned and reused across templates and
    /// (in the streamed emit) across output rows.
    fn apply_one_template(
        template: Expr,
        bindings: &BTreeMap<(u8, u8), ExprEnv>,
        oi: u8,
        ni: u8,
        mut buffer: &mut Vec<u8>,
        mut stack: &mut Vec<(u8, u8)>,
        mut assignments: &mut Vec<(u8, u8)>,
        mut out: impl FnMut(&[u8]),
    ) {
        buffer.clear();
        let (_, _, ok) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
            0,
            oi,
            ni,
            template,
            bindings,
            buffer,
            stack,
            assignments
        );
        if ok {
            out(&buffer[..]);
        }
    }

    fn apply_templates_from_bindings(
        bindings: &BTreeMap<(u8, u8), ExprEnv>,
        pat_expr: Expr,
        templates: &[Expr],
        buffer: &mut Vec<u8>,
        stack: &mut Vec<(u8, u8)>,
        assignments: &mut Vec<(u8, u8)>,
        mut out: impl FnMut(&[u8]),
    ) {
        let Some((oi, ni)) = Self::pattern_template_intros(
            bindings,
            pat_expr,
            &mut *buffer,
            &mut *stack,
            &mut *assignments,
        ) else {
            return;
        };
        for &template in templates {
            Self::apply_one_template(
                template,
                bindings,
                oi,
                ni,
                &mut *buffer,
                &mut *stack,
                &mut *assignments,
                &mut out,
            );
        }
    }

    /// Set of template output paths the live ProductZipper produces for a body,
    /// instantiating each template per match via `apply_e`. The reference set the
    /// sidecar-driven emit must reproduce.
    fn product_template_outputs(
        btm: &PathMap<()>,
        pat_expr: Expr,
        templates: &[Expr],
    ) -> BTreeSet<Vec<u8>> {
        let mut outputs = BTreeSet::new();
        let mut buffer = template_output_buffer();
        let mut stack = Vec::new();
        let mut assignments = Vec::new();
        Self::query_multi(btm, pat_expr, |refs_bindings, _loc| 'query: {
            let Err(ref bindings) = refs_bindings else {
                break 'query true;
            };
            Self::apply_templates_from_bindings(
                bindings,
                pat_expr,
                templates,
                &mut buffer,
                &mut stack,
                &mut assignments,
                |path| {
                    outputs.insert(path.to_vec());
                },
            );
            true
        });
        outputs
    }

    /// Set of template output paths the sidecar plan produces, by applying each
    /// join-tuple substitution to the templates: the relational e-matching
    /// substitution application (egglog rule actions). Each `BindingRelation` row
    /// is reconstructed into the live binding map `(u8,u8) -> ExprEnv` by mapping
    /// each schema `BindingVar` to its query variable key and pointing an
    /// `ExprEnv` at the bound term's interned bytes, then the same `apply_e` macro
    /// instantiates the templates. `None` if a row term or variable key is
    /// missing.
    fn sidecar_template_outputs(
        relation: &BindingRelation,
        binding_to_query_var: &BTreeMap<BindingVar, (u8, u8)>,
        pat_expr: Expr,
        templates: &[Expr],
        sidecar: &crate::term_identity::TermIdentitySidecar,
    ) -> Option<BTreeSet<Vec<u8>>> {
        let schema = relation.schema();
        let mut outputs = BTreeSet::new();
        let mut buffer = template_output_buffer();
        let mut stack = Vec::new();
        let mut assignments = Vec::new();
        for row in relation.positive_rows() {
            let mut bindings: BTreeMap<(u8, u8), ExprEnv> = BTreeMap::new();
            for (position, &var) in schema.iter().enumerate() {
                let key = *binding_to_query_var.get(&var)?;
                let term = *row.get(position)?;
                let bytes = sidecar.get_term(term)?.encoded();
                bindings.insert(
                    key,
                    ExprEnv::new(
                        0,
                        Expr {
                            ptr: bytes.as_ptr() as *mut u8,
                        },
                    ),
                );
            }
            Self::apply_templates_from_bindings(
                &bindings,
                pat_expr,
                templates,
                &mut buffer,
                &mut stack,
                &mut assignments,
                |path| {
                    outputs.insert(path.to_vec());
                },
            );
        }
        Some(outputs)
    }

    /// Whether driving the template writes from the sidecar's worst-case-optimal
    /// join output produces exactly the live ProductZipper's output set. The
    /// substitution sets are already proven equal by
    /// `validate_lowered_plan_against_product`; this additionally checks the
    /// `apply_e` reconstruction from `TermId` rows. `None` when the body does not
    /// lower. This is the acceptance gate for replacing the live emit with the
    /// sidecar join, which delivers the measured worst-case-optimal speedup.
    pub fn validate_sidecar_emit_against_product(
        btm: &PathMap<()>,
        pat_expr: Expr,
        tpl_expr: Expr,
    ) -> Option<bool> {
        let (sidecar_set, _) = Self::sidecar_emit_output_set(btm, pat_expr, tpl_expr, false)?;

        let mut tpl_args = Vec::new();
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<Expr> = tpl_args.get(1..)?.iter().map(|ee| ee.subsexpr()).collect();
        let product_set = Self::product_template_outputs(btm, pat_expr, &templates);
        Some(sidecar_set == product_set)
    }

    /// The set of template output paths the sidecar plan produces for a body,
    /// plus the match count. Builds a subspace sidecar, lowers, runs the selected
    /// worst-case-optimal kernel, and applies each join-tuple substitution to the
    /// templates. `None` when the body does not lower. This is the sidecar's emit:
    /// what would be written if the sidecar drove the transform in place of the
    /// ProductZipper.
    fn sidecar_emit_output_set(
        btm: &PathMap<()>,
        pat_expr: Expr,
        tpl_expr: Expr,
        cyclic_only: bool,
    ) -> Option<(BTreeSet<Vec<u8>>, usize)> {
        let mut args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut args);
        let sources = args.get(1..)?;
        if sources.is_empty() {
            return None;
        }

        let mut sidecar = Self::build_subspace_sidecar(btm, sources)?;
        Self::sidecar_emit_output_set_with(&mut sidecar, sources, pat_expr, tpl_expr, cyclic_only)
    }

    /// Core of the sidecar-driven emit over an already-synced sidecar: lowers the
    /// body, runs the worst-case-optimal join, and applies each join-tuple
    /// substitution to the templates. Split from the sidecar build so the live
    /// flip can drive it from the persistent, incrementally maintained sidecar
    /// while the validator drives it from a fresh subspace sidecar.
    fn sidecar_emit_output_set_with(
        sidecar: &mut crate::term_identity::TermIdentitySidecar,
        sources: &[ExprEnv],
        pat_expr: Expr,
        tpl_expr: Expr,
        cyclic_only: bool,
    ) -> Option<(BTreeSet<Vec<u8>>, usize)> {
        let plan = Self::lower_query_to_sidecar_plan(sources, sidecar)?;
        // The worst-case-optimal join only beats the ProductZipper on cyclic
        // bodies; on an acyclic body the ProductZipper's trie walk is already
        // output-sensitive, so the caller (the live flip) keeps it.
        if cyclic_only && plan.body_is_acyclic() {
            return None;
        }
        let selected = plan.execute_selected(&*sidecar).ok()?;

        let mut variable_for_key: BTreeMap<(u8, u8), BindingVar> = BTreeMap::new();
        for &source in sources {
            for key in Self::query_factor_variables(source) {
                let next = BindingVar(variable_for_key.len() as u8);
                variable_for_key.entry(key).or_insert(next);
            }
        }
        let binding_to_query_var: BTreeMap<BindingVar, (u8, u8)> = variable_for_key
            .iter()
            .map(|(&key, &var)| (var, key))
            .collect();

        let mut tpl_args = Vec::new();
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<Expr> = tpl_args.get(1..)?.iter().map(|ee| ee.subsexpr()).collect();

        let matches = selected.relation.positive_rows().count();
        let outputs = Self::sidecar_template_outputs(
            &selected.relation,
            &binding_to_query_var,
            pat_expr,
            &templates,
            &*sidecar,
        )?;
        Some((outputs, matches))
    }

    /// Streamed form of `sidecar_emit_output_set_with` (the `Factorise` operator):
    /// instead of materialising the join into a `BindingRelation` and then a
    /// `BTreeSet`, it lowers the body, runs the trie join, and hands each tuple's
    /// template outputs straight to `out` as the join produces them, with no flat
    /// intermediate. Returns `Some(Ok(match_count))` when the streamable trie-join
    /// kernel ran, `Some(Err(_))` on a join error, and `None` when the body did
    /// not lower, was acyclic under `cyclic_only`, or the selected kernel is one
    /// of the materialising kernels (the caller then keeps the materialised path).
    /// The per-tuple reconstruction mirrors `sidecar_template_outputs`, but reads
    /// each variable's term from the `BindingAssignment` the join produced rather
    /// than from a materialised row.
    fn sidecar_emit_stream_with(
        sidecar: &mut crate::term_identity::TermIdentitySidecar,
        sources: &[ExprEnv],
        pat_expr: Expr,
        tpl_expr: Expr,
        cyclic_only: bool,
        mut out: impl FnMut(&[u8]),
    ) -> Option<Result<usize, crate::binding_plan::BindingSidecarPlanError>> {
        let plan = Self::lower_query_to_sidecar_plan(sources, sidecar)?;
        // The worst-case-optimal join only beats the ProductZipper on cyclic
        // bodies; an acyclic body keeps the ProductZipper as the caller's fallback.
        if cyclic_only && plan.body_is_acyclic() {
            return None;
        }

        let mut variable_for_key: BTreeMap<(u8, u8), BindingVar> = BTreeMap::new();
        for &source in sources {
            for key in Self::query_factor_variables(source) {
                let next = BindingVar(variable_for_key.len() as u8);
                variable_for_key.entry(key).or_insert(next);
            }
        }
        let binding_to_query_var: BTreeMap<BindingVar, (u8, u8)> = variable_for_key
            .iter()
            .map(|(&key, &var)| (var, key))
            .collect();

        let mut tpl_args = Vec::new();
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<Expr> = tpl_args.get(1..)?.iter().map(|ee| ee.subsexpr()).collect();

        // Prepare opens the factor relations and trie indexes over the immutable
        // term snapshot; the prepared plan owns its data, so the callback below
        // can re-borrow the sidecar shared (for `get_term`) while it runs.
        let prepared = match plan.prepare(&*sidecar) {
            Ok(prepared) => prepared,
            Err(error) => return Some(Err(error)),
        };

        let mut buffer = template_output_buffer();
        let mut stack = Vec::new();
        let mut assignments = Vec::new();
        let mut matches = 0usize;
        // The pattern intro counts (oi, ni), hoisted across rows. They are
        // invariant for ground bindings (ni == 0), which the join output
        // guarantees, so the per-row pattern re-walk (measured ~21% of the emit)
        // is paid once. A non-ground row (ni != 0) is not cached and recomputes.
        let mut hoisted_intros: Option<(u8, u8)> = None;

        // `for_each_selected` returns `None` for the materialising kernels and
        // never invokes the callback in that case, so a `None` here means nothing
        // was written and the caller can keep the materialised path.
        let result = prepared.for_each_selected(|binding, _weight| {
            matches += 1;
            let mut bindings: BTreeMap<(u8, u8), ExprEnv> = BTreeMap::new();
            for (&var, &key) in binding_to_query_var.iter() {
                // The join binds every plan variable and every bound term is
                // interned in this sidecar, so both lookups succeed by
                // construction; the materialised path encodes the same invariant
                // with `?`. A miss would be a lowering/interning bug, caught in
                // debug and skipped (no partial garbage) in release.
                let Some(term) = binding.get(var) else {
                    debug_assert!(false, "join binding missing a plan variable");
                    return;
                };
                let Some(record) = sidecar.get_term(term) else {
                    debug_assert!(false, "join produced an un-interned term");
                    return;
                };
                let bytes = record.encoded();
                bindings.insert(
                    key,
                    ExprEnv::new(
                        0,
                        Expr {
                            ptr: bytes.as_ptr() as *mut u8,
                        },
                    ),
                );
            }
            // Compute the pattern intros once (cache when ground), then apply
            // the templates seeded with them. This skips the per-row pattern
            // re-walk that `apply_templates_from_bindings` would otherwise do.
            let intros = match hoisted_intros {
                Some(cached) => Some(cached),
                None => {
                    let computed = Self::pattern_template_intros(
                        &bindings,
                        pat_expr,
                        &mut buffer,
                        &mut stack,
                        &mut assignments,
                    );
                    if let Some((_, ni)) = computed {
                        if ni == 0 {
                            hoisted_intros = computed;
                        }
                    }
                    computed
                }
            };
            let Some((oi, ni)) = intros else {
                return;
            };
            for &template in &templates {
                Self::apply_one_template(
                    template,
                    &bindings,
                    oi,
                    ni,
                    &mut buffer,
                    &mut stack,
                    &mut assignments,
                    &mut out,
                );
            }
        })?;

        Some(result.map(|_stats| matches))
    }

    /// Cheap structural test for whether a `,`-body's join graph is cyclic, from
    /// the pattern's per-factor variables alone (no data interning). Routes the
    /// live flip: the worst-case-optimal join only beats the ProductZipper on
    /// cyclic bodies, so an acyclic body skips the flip without paying to intern
    /// its relations. Fewer than two factors is never cyclic.
    fn body_is_cyclic(pat_expr: Expr) -> bool {
        let mut args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut args);
        let Some(sources) = args.get(1..) else {
            return false;
        };
        if sources.len() < 2 {
            return false;
        }
        let mut variable_for_key: BTreeMap<(u8, u8), BindingVar> = BTreeMap::new();
        let mut relations = Vec::with_capacity(sources.len());
        for &source in sources {
            let schema: Vec<BindingVar> = Self::query_factor_variables(source)
                .iter()
                .map(|key| {
                    let next = BindingVar(variable_for_key.len() as u8);
                    *variable_for_key.entry(*key).or_insert(next)
                })
                .collect();
            relations.push(crate::binding_space::BindingRelation::new(schema));
        }
        !crate::binding_space::gyo_join_tree(&relations).acyclic
    }

    /// Whether the cardinality-sorted join order connects: each factor after the
    /// first shares an already-covered variable. A *disconnected* order means the
    /// most selective factors do not join until a later factor binds their inputs
    /// (a function table sorted ahead of its `args` input, as in `finite_domain`),
    /// so the ProductZipper reorders to a cheap connected plan and the
    /// worst-case-optimal join (which must intern every relation) is not worth its
    /// overhead. Uses only the cheap btm cardinality stats with the *same*
    /// comparator `query_factor_plan` applies, so the routing decision matches the
    /// plan actually run, and gates the WCO path before the O(relation) sidecar
    /// sync without opening (interning) any relation.
    fn body_cardinality_order_connected(btm: &PathMap<()>, pat_expr: Expr) -> bool {
        let mut args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut args);
        let Some(sources) = args.get(1..) else {
            return true;
        };
        let n = sources.len();
        if n <= 1 {
            return true;
        }
        let mut prefix_cardinalities = BTreeMap::new();
        let mut shape_cardinalities = BTreeMap::new();
        let ranks: Vec<_> = sources
            .iter()
            .copied()
            .map(|source| {
                Self::query_factor_rank_with_btm_count(
                    btm,
                    source,
                    &mut prefix_cardinalities,
                    &mut shape_cardinalities,
                    None,
                )
            })
            .collect();
        let rank_cmp = |a: usize, b: usize| -> std::cmp::Ordering {
            ranks[a]
                .estimated_cardinality
                .cmp(&ranks[b].estimated_cardinality)
                .then_with(|| {
                    ranks[a]
                        .min_variable_domain_cardinality
                        .unwrap_or(usize::MAX)
                        .cmp(
                            &ranks[b]
                                .min_variable_domain_cardinality
                                .unwrap_or(usize::MAX),
                        )
                })
                .then_with(|| ranks[b].prefix_len.cmp(&ranks[a].prefix_len))
                .then_with(|| ranks[b].constant_items.cmp(&ranks[a].constant_items))
                .then_with(|| ranks[a].variable_items.cmp(&ranks[b].variable_items))
                .then_with(|| a.cmp(&b))
        };
        let var_sets: Vec<BTreeSet<(u8, u8)>> = sources
            .iter()
            .map(|&source| Self::query_factor_variables(source).into_iter().collect())
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| rank_cmp(a, b));
        let mut covered: BTreeSet<(u8, u8)> = BTreeSet::new();
        for (idx, &factor) in order.iter().enumerate() {
            if idx > 0 && var_sets[factor].is_disjoint(&covered) {
                return false;
            }
            covered.extend(var_sets[factor].iter().copied());
        }
        true
    }

    fn body_has_safe_zipper_schematic_route(btm: &PathMap<()>, pat_expr: Expr) -> bool {
        let body = unsafe { &*pat_expr.span() };
        let Some((factors, _)) = crate::zipper_join::parse_body_factors(body) else {
            return false;
        };
        let prefixes = factors
            .iter()
            .map(|factor| factor.prefix.clone())
            .collect::<Vec<_>>();
        Self::has_schematic_fact_under_prefixes(btm, &prefixes)
            && crate::zipper_join::unify_join_zipper_body_routable(btm, body)
    }

    /// Drives the template writes for a `,`-conjunction transform from the
    /// sidecar's worst-case-optimal join instead of the ProductZipper, when the
    /// body lowers. Computes the sidecar emit output set (which
    /// `validate_sidecar_emit_against_product` proves equals the ProductZipper's)
    /// and inserts each output path into the live space. Returns `(match count,
    /// any new path written)`, or `None` when the body does not lower (the caller
    /// keeps the ProductZipper). The writes are idempotent set adds, so batching
    /// here yields the same space the streaming ProductZipper path would, while
    /// the join takes asymptotically fewer steps (the measured worst-case-optimal
    /// advantage).
    // The query-variable keys that occur in two or more body factors. A schematic fact
    // aligned with such a variable could be grounded by another factor (capture), so it is
    // not safe to admit to the equality join. Keys follow `query_factor_variables`' scheme,
    // so a variable shared across factors carries the same key the planner joins on.
    fn join_variable_keys(sources: &[ExprEnv]) -> BTreeSet<(u8, u8)> {
        let mut counts: BTreeMap<(u8, u8), usize> = BTreeMap::new();
        for &source in sources {
            for key in Self::query_factor_variables(source) {
                *counts.entry(key).or_default() += 1;
            }
        }
        counts
            .into_iter()
            .filter(|&(_, c)| c >= 2)
            .map(|(k, _)| k)
            .collect()
    }

    // Whether an expression is ground (contains no NewVar or VarRef item).
    fn expr_is_ground(e: Expr) -> bool {
        let mut ez = ExprZipper::new(e);
        loop {
            if matches!(ez.tag(), Tag::NewVar | Tag::VarRef(_)) {
                return false;
            }
            if !ez.next() {
                return true;
            }
        }
    }

    // Advance the NewVar counter past every NewVar in `g`, matching the depth-first order
    // `query_factor_variables` assigns keys in. Called when a query-factor subterm is skipped
    // because it cannot match the fact, so later NewVar keys stay aligned with the join set.
    fn advance_g_newvars(g: Expr, g_newvar: &mut u8) {
        let mut ez = ExprZipper::new(g);
        loop {
            if matches!(ez.tag(), Tag::NewVar) {
                *g_newvar += 1;
            }
            if !ez.next() {
                break;
            }
        }
    }

    // Whether matching a query-factor subterm `g` against a stored-fact subterm `f` would let
    // the ProductZipper derive a ground answer the equality join misses: capture one of the
    // fact's variables to ground, or align it with a join key another factor grounds. `top` is
    // true at the relation's arguments and false inside a nested query compound. A nested query
    // compound the sidecar decomposes needs the fact's compound ground there (the equality join
    // cannot project positions out of a non-ground compound); only a query variable
    // (column-level) may bind a non-ground fact compound. `g_newvar` tracks the depth-first
    // NewVar index (the `query_factor_variables` key scheme), advanced past skipped subterms so
    // join lookups stay aligned. Conservative: a shape mismatch is treated as safe.
    fn subterm_unsafe(
        g: Expr,
        f: Expr,
        join_keys: &BTreeSet<(u8, u8)>,
        g_newvar: &mut u8,
        g_n: u8,
        top: bool,
    ) -> bool {
        let g_tag = ExprZipper::new(g).tag();
        let f_tag = ExprZipper::new(f).tag();
        match g_tag {
            Tag::Arity(ga) => match f_tag {
                // A fact variable here captures the whole query compound: a ground answer the
                // equality join cannot reproduce.
                Tag::NewVar | Tag::VarRef(_) => true,
                // The relation's arguments: classify each position against the fact.
                Tag::Arity(fa) if fa == ga && top => {
                    let mut gc = Vec::new();
                    ExprEnv::new(0, g).args(&mut gc);
                    let mut fc = Vec::new();
                    ExprEnv::new(0, f).args(&mut fc);
                    for (gci, fci) in gc.iter().zip(fc.iter()) {
                        if Self::subterm_unsafe(
                            gci.subsexpr(),
                            fci.subsexpr(),
                            join_keys,
                            g_newvar,
                            g_n,
                            false,
                        ) {
                            return true;
                        }
                    }
                    false
                }
                // A nested query compound the sidecar decomposes: safe only if the fact's
                // compound is ground there, else the join cannot project its positions.
                Tag::Arity(fa) if fa == ga => {
                    Self::advance_g_newvars(g, g_newvar);
                    !Self::expr_is_ground(f)
                }
                // Shape mismatch: this factor does not match the fact, so the fact is
                // irrelevant to it. Safe, but still count the factor's skipped variables.
                _ => {
                    Self::advance_g_newvars(g, g_newvar);
                    false
                }
            },
            // A fact variable captures the query symbol (ground): a captured ground answer.
            Tag::SymbolSize(_) => matches!(f_tag, Tag::NewVar | Tag::VarRef(_)),
            Tag::NewVar | Tag::VarRef(_) => {
                let key = match g_tag {
                    Tag::NewVar => {
                        let k = (g_n, *g_newvar);
                        *g_newvar += 1;
                        k
                    }
                    Tag::VarRef(o) => (g_n, o),
                    _ => unreachable!(),
                };
                let is_join = join_keys.contains(&key);
                match f_tag {
                    // A fact variable, or a non-ground fact compound, aligns with this query
                    // variable; unsafe only if it is a join key another factor would ground.
                    Tag::NewVar | Tag::VarRef(_) => is_join,
                    Tag::Arity(_) => is_join && !Self::expr_is_ground(f),
                    // A ground fact subterm binds the query variable to a ground value: safe.
                    Tag::SymbolSize(_) => false,
                }
            }
        }
    }

    // Whether every schematic stored fact under the body's joined relations is safe to admit
    // to the relational join, the per-position refinement of the all-or-nothing schematic
    // decline. Compares each schematic fact against each query factor on its relation with
    // `subterm_unsafe`, so nesting on either side is handled. When this returns true the
    // sidecar's ground output equals the ProductZipper's, which `sidecar_admissibility_oracle`
    // and `gate_admissions_are_sound_random` pin.
    fn schematic_facts_safe_to_admit(
        sidecar: &crate::term_identity::TermIdentitySidecar,
        sources: &[ExprEnv],
    ) -> bool {
        let join_keys = Self::join_variable_keys(sources);
        let prefixes: Vec<Vec<u8>> = sources
            .iter()
            .map(|&g| query_source_prefix(g).unwrap_or_default())
            .collect();
        for fact in sidecar.facts() {
            if fact.flags.ground || !sidecar.is_fact_live(fact.id) {
                continue;
            }
            let Some(record) = sidecar.get_term(fact.root) else {
                return false;
            };
            let f_bytes = record.encoded();
            let f_expr = Expr {
                ptr: f_bytes.as_ptr() as *mut u8,
            };
            for (g, prefix) in sources.iter().zip(&prefixes) {
                if prefix.is_empty() || !f_bytes.starts_with(prefix) {
                    continue;
                }
                let mut g_newvar = g.v;
                if Self::subterm_unsafe(g.subsexpr(), f_expr, &join_keys, &mut g_newvar, g.n, true)
                {
                    return false;
                }
            }
        }
        true
    }

    // Whether `e` carries a NON-ground compound argument: a nested arity term (at any depth under a
    // top-level argument) that contains a variable, like `(k $x)` in `(e (k $x) $y)`. This is the
    // ingredient for issue-29 data-side capture: a stored variable can unify with such a compound,
    // and through the join bind to it, the one place full unification finds answers the
    // ProductZipper-equivalent equality intersection does not. A bare variable argument is not
    // flagged (a stored variable aliasing it is plain equality), nor is a GROUND compound (its
    // capture is the ground capture the ProductZipper also performs).
    fn expr_has_nonground_compound(e: Expr) -> bool {
        Self::expr_nonground_compound_arg_count(e) > 0
    }

    fn expr_nonground_compound_arg_count(e: Expr) -> usize {
        let mut args = Vec::new();
        ExprEnv::new(0, e).args(&mut args);
        // args[0] is the relation head; a non-ground compound can only sit in a real argument.
        let mut count = 0usize;
        for arg in args.iter().skip(1) {
            let ae = arg.subsexpr();
            if matches!(ExprZipper::new(ae).tag(), Tag::Arity(_)) && !Self::expr_is_ground(ae) {
                count += 1;
            }
        }
        count
    }

    fn has_schematic_fact_under_prefixes(read_copy: &PathMap<()>, prefixes: &[Vec<u8>]) -> bool {
        let mut seen_prefix = BTreeSet::new();
        for prefix in prefixes {
            if prefix.is_empty() || !seen_prefix.insert(prefix.clone()) {
                continue;
            }
            let mut rz = read_copy.read_zipper_at_path(&prefix[..]);
            while rz.to_next_val() {
                let bytes = rz.origin_path();
                let f_expr = Expr {
                    ptr: bytes.as_ptr() as *mut u8,
                };
                if !Self::expr_is_ground(f_expr) {
                    return true;
                }
            }
        }
        false
    }

    /// Emit a routable schematic body through the worst-case-optimal UNIFICATION join over the live
    /// read snapshot, byte-identical to the ProductZipper. Each query factor is matched against the
    /// facts under its relation prefix (ground and schematic), read straight from the same snapshot
    /// the ProductZipper reads via the PathMap index, by a trail-backed unification, joined
    /// variable-at-a-time; each answer's bindings drive the templates through the same `apply_e`
    /// the ProductZipper emit uses. Only GROUND components are bound (a ground term has no
    /// variable identity to collide under ExprEnv's (n,v) scheme); a template over a non-ground
    /// variable instantiates a fresh variable, the non-ground output the exec discards. Returns
    /// the join's answer count, or `None` if the body does not map (the caller then declines).
    fn sidecar_unify_emit(
        read_copy: &PathMap<()>,
        sources: &[ExprEnv],
        prefixes: &[Vec<u8>],
        pat_expr: Expr,
        tpl_expr: Expr,
        mut out: impl FnMut(&[u8]),
    ) -> Option<usize> {
        // The query variable keys, densely numbered in first-occurrence order across factors,
        // the same order `unify_join` numbers the answer-tuple components in.
        let mut variable_for_key: BTreeMap<(u8, u8), BindingVar> = BTreeMap::new();
        for &source in sources {
            for key in Self::query_factor_variables(source) {
                let next = BindingVar(variable_for_key.len() as u8);
                variable_for_key.entry(key).or_insert(next);
            }
        }
        let mut dense_keys: Vec<(u8, u8)> = vec![(0, 0); variable_for_key.len()];
        for (&key, &bv) in &variable_for_key {
            dense_keys[bv.0 as usize] = key;
        }

        let body = unsafe { &*pat_expr.span() };
        let n = crate::unify_join::body_var_count(body);
        if n != dense_keys.len() {
            return None;
        }

        let mut tpl_args = Vec::new();
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<Expr> = tpl_args.get(1..)?.iter().map(|ee| ee.subsexpr()).collect();

        let mut buffer = template_output_buffer();
        let mut stack = Vec::new();
        let mut assignments = Vec::new();
        // Both kernels emit the same way: instantiate the templates from the ground bindings of one
        // answer row. Shared so the two paths cannot drift.
        let mut emit_row = |bindings: &BTreeMap<(u8, u8), ExprEnv>| {
            Self::apply_templates_from_bindings(
                bindings,
                pat_expr,
                &templates,
                &mut buffer,
                &mut stack,
                &mut assignments,
                &mut out,
            );
        };

        if !crate::zipper_join::unify_join_zipper_body_routable(read_copy, body) {
            return None;
        }

        // Zipper-native kernel: parse, route-check, and seek the live snapshot directly, with no
        // decode of the relation facts. Each answer row carries one Option per query variable,
        // ground or free; bind each resolved schematic byte term and leave truly-free variables
        // unbound so template rendering matches ProductZipper.
        if SIDECAR_ZIPPER_JOIN_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some((znvars, rows)) =
                crate::zipper_join::unify_join_zipper_body_partial_safe(read_copy, body)
            {
                if znvars != n {
                    return None;
                }
                for row in &rows {
                    let mut bindings: BTreeMap<(u8, u8), ExprEnv> = BTreeMap::new();
                    for (i, &k) in dense_keys.iter().enumerate() {
                        if let Some(val) = &row[i] {
                            bindings.insert(
                                k,
                                ExprEnv::new(
                                    0,
                                    Expr {
                                        ptr: val.as_ptr() as *mut u8,
                                    },
                                ),
                            );
                        }
                    }
                    emit_row(&bindings);
                }
                SIDECAR_ZIPPER_RECOVERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(rows.len());
            }
            return None;
        }

        // Materialized leapfrog: read the facts under the body's relation prefixes, straight from the
        // live snapshot the ProductZipper also reads, via the PathMap index. Descending to each
        // relation subtree (deduplicated) replaces a scan of the whole fact set, so the read cost
        // tracks the joined relations, not the size of the space. `origin_path` is the absolute key,
        // the full fact encoding ground and schematic alike, so the join sees the matcher's input.
        let mut seen_prefix = BTreeSet::new();
        let mut fact_bufs: Vec<Vec<u8>> = Vec::new();
        for prefix in prefixes {
            if prefix.is_empty() || !seen_prefix.insert(prefix.clone()) {
                continue;
            }
            let mut rz = read_copy.read_zipper_at_path(&prefix[..]);
            while rz.to_next_val() {
                fact_bufs.push(rz.origin_path().to_vec());
            }
        }
        let fact_slices: Vec<&[u8]> = fact_bufs.iter().map(|v| v.as_slice()).collect();
        let answers = crate::unify_join::leapfrog_unify_join_encoded(body, &fact_slices);
        for key in &answers {
            let comps = crate::unify_join::split_tuple(key, n);
            if comps.len() != n {
                continue;
            }
            let mut bindings: BTreeMap<(u8, u8), ExprEnv> = BTreeMap::new();
            for (i, &k) in dense_keys.iter().enumerate() {
                if crate::unify_join::term_is_ground(&comps[i]) {
                    bindings.insert(
                        k,
                        ExprEnv::new(
                            0,
                            Expr {
                                ptr: comps[i].as_ptr() as *mut u8,
                            },
                        ),
                    );
                }
            }
            emit_row(&bindings);
        }
        Some(answers.len())
    }

    /// Emit a body through the full-unification capture join. This path is retained as an oracle and
    /// alternate sidecar route for non-ground query compounds; native ProductZipper should now match
    /// its ground outputs. It runs `capture_join_live`, the prototype's descent sealed against
    /// SWI-Prolog occurs-check, directly over the live read snapshot. The
    /// join's answer tuple is keyed in first-occurrence query-variable order, the same order
    /// `dense_keys` numbers the fork variable keys, so the i-th component drives the i-th key; only
    /// GROUND components are bound, exactly as the materialized branch does (a non-ground component
    /// leaves its template variable fresh, the non-ground output the exec discards). Returns the
    /// answer count, or `None` if the body does not map (the caller then declines as before).
    fn capture_unify_emit(
        read_copy: &PathMap<()>,
        sources: &[ExprEnv],
        pat_expr: Expr,
        tpl_expr: Expr,
        mut out: impl FnMut(&[u8]),
    ) -> Option<usize> {
        use mork_uni_join::term::Term as PTerm;

        // Fork variable keys, densely numbered in first-occurrence order across factors, the same
        // order the capture join numbers the answer-tuple components in.
        let mut variable_for_key: BTreeMap<(u8, u8), BindingVar> = BTreeMap::new();
        for &source in sources {
            for key in Self::query_factor_variables(source) {
                let next = BindingVar(variable_for_key.len() as u8);
                variable_for_key.entry(key).or_insert(next);
            }
        }
        let mut dense_keys: Vec<(u8, u8)> = vec![(0, 0); variable_for_key.len()];
        for (&key, &bv) in &variable_for_key {
            dense_keys[bv.0 as usize] = key;
        }

        // Prototype conjunctive query from the encoded `(, p1 .. pk)` body. Its query variables are
        // numbered in the same first-occurrence order as `dense_keys`; a mismatch means the body
        // does not map to the join's variable model, so decline.
        let body = unsafe { &*pat_expr.span() };
        let q = crate::capture_join::conj_from_body(body);
        if q.query_vars.len() != dense_keys.len() {
            return None;
        }

        let mut tpl_args = Vec::new();
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<Expr> = tpl_args.get(1..)?.iter().map(|ee| ee.subsexpr()).collect();

        // The capture join over the live snapshot: it descends each factor's relation subtree under
        // the byte trie (the descent prunes non-matching heads, so it never scans unrelated facts),
        // unifying with data-side capture and a backtrackable trail. Byte-identical answer keys to
        // the materialized leapfrog and so to the SWI-Prolog occurs-check seal.
        let answers = crate::capture_join::capture_join_live(read_copy, &q);

        let mut buffer = template_output_buffer();
        let mut stack = Vec::new();
        let mut assignments = Vec::new();
        for key in &answers {
            let comps: Vec<PTerm> = match PTerm::decode(key) {
                PTerm::App(a) => a,
                t => vec![t],
            };
            if comps.len() != dense_keys.len() {
                continue;
            }
            // Bind only the ground components; their encodings must outlive the `bindings` map (each
            // ExprEnv holds a raw pointer into the buffer), so keep `value_bufs` alive across the
            // apply.
            let value_bufs: Vec<Option<Vec<u8>>> = comps
                .iter()
                .map(|c| c.is_ground().then(|| c.encode()))
                .collect();
            let mut bindings: BTreeMap<(u8, u8), ExprEnv> = BTreeMap::new();
            for (i, &k) in dense_keys.iter().enumerate() {
                if let Some(bytes) = &value_bufs[i] {
                    bindings.insert(
                        k,
                        ExprEnv::new(
                            0,
                            Expr {
                                ptr: bytes.as_ptr() as *mut u8,
                            },
                        ),
                    );
                }
            }
            Self::apply_templates_from_bindings(
                &bindings,
                pat_expr,
                &templates,
                &mut buffer,
                &mut stack,
                &mut assignments,
                &mut out,
            );
        }
        Some(answers.len())
    }

    /// Cheap query-side trigger for the capture route: some factor carries a NON-ground compound,
    /// the necessary condition for a data variable to capture a query subterm (issue-29, capture
    /// requires a non-ground compound to exist on the query side). Reads no facts, so it gates the
    /// `btm` clone the capture route would otherwise pay on every step.
    fn query_has_nonground_compound(pat_expr: Expr) -> bool {
        let mut args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut args);
        match args.get(1..) {
            Some(sources) => sources
                .iter()
                .any(|&g| Self::expr_has_nonground_compound(g.subsexpr())),
            None => false,
        }
    }

    /// Optional acyclic capture route. `transform_via_sidecar` only engages on cyclic bodies, so this
    /// hook lets an explicitly enabled run compare the native path with the full-unification capture
    /// join on acyclic bodies too. Declines (`None`, so the caller keeps the ProductZipper) when the
    /// body has no relation-prefix factor model or the join cannot map it.
    fn transform_via_capture(
        &mut self,
        read_copy: &PathMap<()>,
        pat_expr: Expr,
        tpl_expr: Expr,
    ) -> Option<(usize, bool)> {
        let mut args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut args);
        let sources = args.get(1..)?;
        if sources.is_empty() {
            return None;
        }
        // Every factor needs a non-empty relation prefix (a leading constant); a whole-space factor
        // cannot be subspace-read, the same precondition the sidecar route enforces.
        sources
            .iter()
            .map(|&s| query_source_prefix(s).filter(|p| !p.is_empty()))
            .collect::<Option<Vec<Vec<u8>>>>()?;

        let mut any_new = false;
        let emitted = Self::capture_unify_emit(read_copy, sources, pat_expr, tpl_expr, |path| {
            if self.btm.insert(path, ()).is_none() {
                any_new = true;
            }
        });
        let matches = emitted?;
        SIDECAR_CAPTURE_RECOVERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some((matches, any_new))
    }

    fn transform_via_sidecar(
        &mut self,
        read_copy: &PathMap<()>,
        pat_expr: Expr,
        tpl_expr: Expr,
    ) -> Option<(usize, bool)> {
        let mut args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut args);
        let sources = args.get(1..)?;
        if sources.is_empty() {
            return None;
        }
        // The factor prefixes the join reads. A body with a whole-space factor
        // (no constant prefix) cannot be subspace-synced, so decline it.
        let prefixes = sources
            .iter()
            .map(|&source| query_source_prefix(source).filter(|prefix| !prefix.is_empty()))
            .collect::<Option<Vec<Vec<u8>>>>()?;

        // Take the persistent sidecar out of `self` so the join can borrow it
        // while the live `btm` is borrowed for writes. Put it back on every exit.
        let remove_gen = self.bridge_remove_gen;
        let mut sidecar = self
            .bridge_sidecar
            .take()
            .unwrap_or_else(crate::term_identity::TermIdentitySidecar::new);

        // A data removal on this space invalidates the count watermarks (a remove
        // plus an add can leave a relation's count unchanged), forcing a re-sync.
        sidecar.invalidate_if_removed(remove_gen);

        // Incremental sync: re-intern only the prefixes whose value count changed
        // since they were last synced. An unchanged relation is reused, so a
        // self-recursive rule does not re-intern its relation every step.
        let mut synced = BTreeSet::new();
        for prefix in &prefixes {
            if !synced.insert(prefix.clone()) {
                continue;
            }
            let count = read_copy.read_zipper_at_path(&prefix[..]).val_count();
            if sidecar
                .sync_prefix_if_stale(read_copy, &prefix[..], count)
                .is_err()
            {
                self.bridge_sidecar = Some(sidecar);
                return None;
            }
        }

        // The sidecar join is relational: it equates interned ground tuples. A schematic
        // stored fact needs first-order unification, so the ProductZipper stays
        // authoritative whenever admitting the fact would change the ground output: when a
        // fact variable meets a constant or a join key, capture would produce a ground
        // answer the equality join misses. But a fact whose variables sit only on
        // output-only positions yields only non-ground rows (dropped on both paths), so the
        // body may stay on the fast join. `schematic_facts_safe_to_admit` is that
        // per-position refinement of the old all-or-nothing decline.
        let body = unsafe { &*pat_expr.span() };
        let zipper_factors =
            crate::zipper_join::parse_body_factors(body).map(|(factors, _)| factors);
        let zipper_prefixes = zipper_factors
            .as_ref()
            .map(|factors| {
                factors
                    .iter()
                    .map(|factor| factor.prefix.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| prefixes.clone());
        let has_schematic_under_zipper_prefixes =
            Self::has_schematic_fact_under_prefixes(read_copy, &zipper_prefixes);
        let query_has_nonground_compound = zipper_factors.as_ref().is_some_and(|factors| {
            factors.iter().any(|factor| {
                factor
                    .cols
                    .iter()
                    .any(|col| col.is_nonground_compound())
            })
        });
        let compound_body_with_schematic_facts =
            query_has_nonground_compound && has_schematic_under_zipper_prefixes;
        let equality_join_needs_unify = sidecar.any_schematic_fact_under_prefixes(&prefixes)
            && !Self::schematic_facts_safe_to_admit(&sidecar, sources);
        let routable_to_unify =
            crate::zipper_join::unify_join_zipper_body_routable(read_copy, body);
        let route_safe_schematic_body = has_schematic_under_zipper_prefixes && routable_to_unify;
        if equality_join_needs_unify
            || compound_body_with_schematic_facts
            || route_safe_schematic_body
        {
            // The equality join cannot absorb these schematic facts. The worst-case-optimal
            // UNIFICATION join can, when the zipper-owned gate says the emitted bytes match the
            // ProductZipper. The capture route remains an opt-in oracle for non-ground query
            // compounds, so capture A/B runs still exercise that path explicitly.
            let capture_enabled =
                SIDECAR_CAPTURE_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
            let prefer_capture = capture_enabled && query_has_nonground_compound;
            let take_unify = SIDECAR_UNIFY_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
                && routable_to_unify
                && !prefer_capture;
            let take_capture = capture_enabled && (!routable_to_unify || prefer_capture);
            if take_unify || take_capture {
                let mut any_new = false;
                let mut delta: Vec<Vec<u8>> = Vec::new();
                let emitted = {
                    let sink = |path: &[u8]| {
                        if self.btm.insert(path, ()).is_none() {
                            any_new = true;
                            delta.push(path.to_vec());
                        }
                    };
                    if take_unify {
                        Self::sidecar_unify_emit(
                            read_copy,
                            sources,
                            &zipper_prefixes,
                            pat_expr,
                            tpl_expr,
                            sink,
                        )
                    } else {
                        Self::capture_unify_emit(read_copy, sources, pat_expr, tpl_expr, sink)
                    }
                };
                if let Some(matches) = emitted {
                    if take_unify {
                        SIDECAR_UNIFY_RECOVERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        SIDECAR_CAPTURE_RECOVERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    for fact in &delta {
                        let _ = sidecar.insert_fact(fact);
                    }
                    let mut bumped = BTreeSet::new();
                    for prefix in &prefixes {
                        if bumped.insert(prefix.clone()) {
                            let count = self.btm.read_zipper_at_path(&prefix[..]).val_count();
                            sidecar.mark_prefix_synced(&prefix[..], count);
                        }
                    }
                    self.bridge_sidecar = Some(sidecar);
                    return Some((matches, any_new));
                }
            }
            SIDECAR_SCHEMATIC_DECLINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.bridge_sidecar = Some(sidecar);
            return None;
        }

        // Fused streamed emit (the `Factorise` operator): write each template
        // output straight into the live space as the trie join produces the
        // tuple, with neither the `BindingRelation` nor the output `BTreeSet`
        // materialised. Falls back to the materialised emit when the selected
        // kernel is not the streamable trie join, and declines (`None`, so the
        // caller keeps the ProductZipper) when the body does not lower or is
        // acyclic.
        let mut any_new = false;
        let mut delta: Vec<Vec<u8>> = Vec::new();
        #[cfg(debug_assertions)]
        let mut streamed_paths: BTreeSet<Vec<u8>> = BTreeSet::new();

        let streamed = Self::sidecar_emit_stream_with(
            &mut sidecar,
            sources,
            pat_expr,
            tpl_expr,
            true,
            |path| {
                #[cfg(debug_assertions)]
                streamed_paths.insert(path.to_vec());
                if self.btm.insert(path, ()).is_none() {
                    any_new = true;
                    delta.push(path.to_vec());
                }
            },
        );

        let matches = match streamed {
            Some(Ok(matches)) => {
                // The streamed emit is the materialised emit with the two
                // intermediates removed; the materialised set is itself proven
                // equal to the ProductZipper by
                // `validate_sidecar_emit_against_product`. Check the streamed set
                // against the materialised oracle in debug builds (the same
                // discipline, recomputed from the unchanged sidecar).
                #[cfg(debug_assertions)]
                if let Some((reference, _)) = Self::sidecar_emit_output_set_with(
                    &mut sidecar,
                    sources,
                    pat_expr,
                    tpl_expr,
                    true,
                ) {
                    debug_assert_eq!(
                        streamed_paths, reference,
                        "fused streamed emit diverged from the materialised oracle"
                    );
                }
                matches
            }
            Some(Err(_)) => {
                // A join error: keep the synced sidecar and decline. Any partial
                // writes are a subset of the correct set, which the ProductZipper
                // fallback completes idempotently.
                self.bridge_sidecar = Some(sidecar);
                return None;
            }
            None => {
                // Non-streamable kernel, acyclic body, or a body that did not
                // lower. Nothing was written (the streamed callback never ran),
                // so use the materialised emit, which itself declines (`None`)
                // for the acyclic / not-lowered cases.
                let Some((outputs, matches)) = Self::sidecar_emit_output_set_with(
                    &mut sidecar,
                    sources,
                    pat_expr,
                    tpl_expr,
                    true,
                ) else {
                    self.bridge_sidecar = Some(sidecar);
                    return None;
                };
                for path in &outputs {
                    if self.btm.insert(&path[..], ()).is_none() {
                        any_new = true;
                        delta.push(path.clone());
                    }
                }
                matches
            }
        };

        // Maintain the sidecar exactly from the known write delta, then advance
        // each queried prefix's watermark to the post-write count. The delta is
        // applied directly (not re-derived from counts), so a self-recursive
        // rule's next step skips the re-intern without the count-equality trap.
        for fact in &delta {
            let _ = sidecar.insert_fact(fact);
        }
        let mut bumped = BTreeSet::new();
        for prefix in &prefixes {
            if bumped.insert(prefix.clone()) {
                let count = self.btm.read_zipper_at_path(&prefix[..]).val_count();
                sidecar.mark_prefix_synced(&prefix[..], count);
            }
        }

        self.bridge_sidecar = Some(sidecar);
        Some((matches, any_new))
    }

    /// Materializes one all-variable flat factor as a `BindingRelation` over the
    /// already-synced sidecar: ground facts under the factor's relation head
    /// become rows, schema is the factor's argument variables mapped through
    /// `variable_for_key`. Returns the relation and its head term. `None` for a
    /// factor with a repeated variable, a leading constant, or a missing head
    /// (the caller declines and keeps the iterated transform).
    fn materialize_factor(
        sidecar: &crate::term_identity::TermIdentitySidecar,
        source: ExprEnv,
        variable_for_key: &BTreeMap<(u8, u8), BindingVar>,
    ) -> Option<(
        crate::binding_space::BindingRelation,
        crate::term_identity::TermId,
    )> {
        let prefix = query_source_prefix(source)?;
        if prefix.len() < 2 {
            return None;
        }
        // All-variable factor: the constant prefix is exactly the arity byte plus
        // the relation head, so the head encoding is prefix[1..].
        let head = sidecar.term_id_for_encoded(&prefix[1..])?;
        let keys = Self::query_factor_variables(source);
        let schema = keys
            .iter()
            .map(|key| variable_for_key.get(key).copied())
            .collect::<Option<Vec<crate::binding_space::BindingVar>>>()?;
        let distinct: BTreeSet<_> = schema.iter().collect();
        if distinct.len() != schema.len() {
            return None;
        }
        let mut relation = crate::binding_space::BindingRelation::new(schema.clone());
        for &fact_id in sidecar.facts_for_relation(head) {
            if !sidecar.is_fact_live(fact_id) {
                continue;
            }
            let Some(fact) = sidecar.get_fact(fact_id) else {
                continue;
            };
            if fact.flags.contains_vars {
                continue;
            }
            let Some(root) = sidecar.get_term(fact.root) else {
                continue;
            };
            let children = root.children();
            if children.first().copied() != Some(head) || children.len() != schema.len() + 1 {
                continue;
            }
            relation.add(children[1..].to_vec(), 1).ok()?;
        }
        Some((relation, head))
    }

    /// O(1) copy-on-write snapshot of the live space, for per-step delta tracking
    /// in the semi-naive immediate-consequence loop. PathMap's structural sharing
    /// makes the clone a pointer bump, not a copy, so snapshotting every IC step
    /// is cheap. See `kernel/resources/semi_naive_delta_design.md`.
    pub(crate) fn delta_snapshot(&self) -> PathMap<()> {
        self.btm.clone()
    }

    /// `(added, removed)` facts since `before`: added = current \ before, removed
    /// = before \ current. The `added` map is the per-step delta `dR` that the
    /// semi-naive m-delta-rule matches one factor against (so old x ... x old
    /// combinations are never re-derived); `removed` drives DRed-style retraction
    /// when a rule consumes its reactants. Both differences are cheap given the
    /// shared trie structure.
    pub(crate) fn delta_since(&self, before: &PathMap<()>) -> (PathMap<()>, PathMap<()>) {
        (self.btm.subtract(before), before.subtract(&self.btm))
    }

    /// Computes the least fixpoint of a linear-recursive `,`-rule by semi-naive
    /// evaluation, then writes the closure to the live space, instead of running
    /// the rule one round per exec step. The body must be linear (exactly one
    /// factor whose relation matches the single output template's relation, the
    /// rest fixed) and all factors all-variable. Returns `(closure row count, any
    /// new path written)`, or `None` when the rule is not a linear fixpoint (the
    /// caller keeps the per-step transform).
    ///
    /// This wires the `semi_naive_linear_fixpoint` kernel (the linear delta rule
    /// `dQ = project(dQ join fixed) - Q`) to the live path. The kernel is the
    /// standard semi-naive evaluation (Abiteboul-Hull-Vianu; egglog PLDI 2023),
    /// so a recursive MeTTa rule (transitive closure and friends) computes its
    /// whole closure in delta rounds rather than re-matching the full space each
    /// step.
    fn transform_linear_recursive_fixpoint(
        &mut self,
        read_copy: &PathMap<()>,
        pat_expr: Expr,
        tpl_expr: Expr,
    ) -> Option<(usize, bool)> {
        let mut pat_args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut pat_args);
        let sources = pat_args.get(1..)?;
        if sources.is_empty() {
            return None;
        }

        // A single output template, headed by the recursive relation.
        let mut tpl_args = Vec::new();
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates = tpl_args.get(1..)?;
        if templates.len() != 1 {
            return None;
        }
        let output_source = templates[0];
        let output_prefix = query_source_prefix(output_source)?;
        if output_prefix.is_empty() {
            return None;
        }

        // The recursive factor is the one body factor whose relation prefix equals
        // the output's. Exactly one makes the rule linear.
        let mut recursive_index = None;
        for (index, &source) in sources.iter().enumerate() {
            if query_source_prefix(source)? == output_prefix {
                if recursive_index.is_some() {
                    return None;
                }
                recursive_index = Some(index);
            }
        }
        let recursive_index = recursive_index?;

        // One consistent BindingVar per query variable across all factors.
        let mut variable_for_key: BTreeMap<(u8, u8), BindingVar> = BTreeMap::new();
        for &source in sources {
            for key in Self::query_factor_variables(source) {
                let next = BindingVar(variable_for_key.len() as u8);
                variable_for_key.entry(key).or_insert(next);
            }
        }
        let output: Vec<BindingVar> = Self::query_factor_variables(output_source)
            .iter()
            .map(|key| variable_for_key.get(key).copied())
            .collect::<Option<_>>()?;

        let prefixes = sources
            .iter()
            .map(|&source| query_source_prefix(source).filter(|prefix| !prefix.is_empty()))
            .collect::<Option<Vec<Vec<u8>>>>()?;

        let mut sidecar = self
            .bridge_sidecar
            .take()
            .unwrap_or_else(crate::term_identity::TermIdentitySidecar::new);
        sidecar.invalidate_if_removed(self.bridge_remove_gen);
        for prefix in &prefixes {
            let count = read_copy.read_zipper_at_path(&prefix[..]).val_count();
            if sidecar
                .sync_prefix_if_stale(read_copy, &prefix[..], count)
                .is_err()
            {
                self.bridge_sidecar = Some(sidecar);
                return None;
            }
        }

        // Materialize the recursive factor (its schema is the recursive schema)
        // and the fixed factors. The base is the recursive relation's facts
        // relabeled to the output schema.
        let materialized = (|| {
            let (recursive_relation, _) =
                Self::materialize_factor(&sidecar, sources[recursive_index], &variable_for_key)?;
            let recursive_schema = recursive_relation.schema().to_vec();
            let base = recursive_relation.relabel(&output).ok()?;
            let mut fixed_factors = Vec::new();
            for (index, &source) in sources.iter().enumerate() {
                if index == recursive_index {
                    continue;
                }
                let (relation, _) = Self::materialize_factor(&sidecar, source, &variable_for_key)?;
                fixed_factors.push(relation);
            }
            Some((base, fixed_factors, recursive_schema))
        })();
        let Some((base, fixed_factors, recursive_schema)) = materialized else {
            self.bridge_sidecar = Some(sidecar);
            return None;
        };

        let closure = match crate::binding_space::semi_naive_linear_fixpoint_wco(
            &base,
            &fixed_factors,
            &recursive_schema,
            &output,
        ) {
            Ok(closure) => closure,
            Err(_) => {
                self.bridge_sidecar = Some(sidecar);
                return None;
            }
        };

        // Reconstruct each closure row into an encoded (head args...) fact.
        let head = match Self::materialize_factor(&sidecar, output_source, &variable_for_key) {
            Some((_, head)) => head,
            None => {
                self.bridge_sidecar = Some(sidecar);
                return None;
            }
        };
        let Some(head_encoded) = sidecar.get_term(head).map(|term| term.encoded().to_vec()) else {
            self.bridge_sidecar = Some(sidecar);
            return None;
        };
        let arity_byte = mork_expr::item_byte(mork_expr::Tag::Arity((output.len() + 1) as u8));

        let (count, any_new) =
            self.write_closure_facts(&mut sidecar, &head_encoded, arity_byte, &closure, &prefixes);
        self.bridge_sidecar = Some(sidecar);
        Some((count, any_new))
    }

    /// Reconstructs each closure row into an encoded `(head args...)` fact, writes
    /// the new facts to the live space, feeds the write delta back to the sidecar,
    /// and advances the queried prefixes' watermarks. Shared by the linear
    /// fixpoint and transitive-closure transforms. Returns `(closure row count,
    /// any new path written)`.
    fn write_closure_facts(
        &mut self,
        sidecar: &mut crate::term_identity::TermIdentitySidecar,
        head_encoded: &[u8],
        arity_byte: u8,
        closure: &crate::binding_space::BindingRelation,
        prefixes: &[Vec<u8>],
    ) -> (usize, bool) {
        let mut facts: Vec<Vec<u8>> = Vec::new();
        let mut count = 0usize;
        for (row, weight) in closure.rows() {
            if weight <= 0 {
                continue;
            }
            count += 1;
            let mut fact = vec![arity_byte];
            fact.extend_from_slice(head_encoded);
            let mut ok = true;
            for &arg in row {
                match sidecar.get_term(arg) {
                    Some(term) => fact.extend_from_slice(term.encoded()),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                facts.push(fact);
            }
        }

        let mut any_new = false;
        let mut delta: Vec<Vec<u8>> = Vec::new();
        for fact in &facts {
            if self.btm.insert(&fact[..], ()).is_none() {
                any_new = true;
                delta.push(fact.clone());
            }
        }
        for fact in &delta {
            let _ = sidecar.insert_fact(fact);
        }
        let mut bumped = BTreeSet::new();
        for prefix in prefixes {
            if bumped.insert(prefix.clone()) {
                let count = self.btm.read_zipper_at_path(&prefix[..]).val_count();
                sidecar.mark_prefix_synced(&prefix[..], count);
            }
        }
        (count, any_new)
    }

    /// Computes the transitive closure of a self-recursive `,`-rule
    /// `R(a,c) :- R(a,b), R(b,c)` (the non-linear transitive-closure form, two
    /// recursive factors over the output relation) by the semi-naive
    /// `semi_naive_transitive_closure` kernel, and writes the closure. The
    /// linear-fixpoint path declines this rule (two recursive factors), so this
    /// covers the other common recursive shape. `None` when the body is not that
    /// shape (the caller keeps the per-step transform).
    fn transform_binary_transitive_closure(
        &mut self,
        read_copy: &PathMap<()>,
        pat_expr: Expr,
        tpl_expr: Expr,
    ) -> Option<(usize, bool)> {
        let mut pat_args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut pat_args);
        let sources = pat_args.get(1..)?;
        if sources.len() != 2 {
            return None;
        }
        let mut tpl_args = Vec::new();
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates = tpl_args.get(1..)?;
        if templates.len() != 1 {
            return None;
        }
        let output_source = templates[0];
        let output_prefix = query_source_prefix(output_source)?;
        if output_prefix.len() < 2 {
            return None;
        }
        // Both body factors and the output are the same relation.
        if query_source_prefix(sources[0])? != output_prefix
            || query_source_prefix(sources[1])? != output_prefix
        {
            return None;
        }
        // The composition shape R(a,b), R(b,c) -> R(a,c) with a, b, c distinct.
        let first = Self::query_factor_variables(sources[0]);
        let second = Self::query_factor_variables(sources[1]);
        let out = Self::query_factor_variables(output_source);
        if first.len() != 2 || second.len() != 2 || out.len() != 2 {
            return None;
        }
        if first[1] != second[0]
            || out[0] != first[0]
            || out[1] != second[1]
            || first[0] == first[1]
            || second[0] == second[1]
            || first[0] == second[1]
        {
            return None;
        }

        let mut variable_for_key: BTreeMap<(u8, u8), BindingVar> = BTreeMap::new();
        for &source in sources {
            for key in Self::query_factor_variables(source) {
                let next = BindingVar(variable_for_key.len() as u8);
                variable_for_key.entry(key).or_insert(next);
            }
        }

        let mut sidecar = self
            .bridge_sidecar
            .take()
            .unwrap_or_else(crate::term_identity::TermIdentitySidecar::new);
        sidecar.invalidate_if_removed(self.bridge_remove_gen);
        let count = read_copy
            .read_zipper_at_path(&output_prefix[..])
            .val_count();
        if sidecar
            .sync_prefix_if_stale(read_copy, &output_prefix[..], count)
            .is_err()
        {
            self.bridge_sidecar = Some(sidecar);
            return None;
        }

        let Some((relation, head)) =
            Self::materialize_factor(&sidecar, output_source, &variable_for_key)
        else {
            self.bridge_sidecar = Some(sidecar);
            return None;
        };
        let schema = relation.schema().to_vec();
        let closure = match crate::binding_space::semi_naive_transitive_closure(
            &relation, schema[0], schema[1],
        ) {
            Ok(result) => result.relation,
            Err(_) => {
                self.bridge_sidecar = Some(sidecar);
                return None;
            }
        };

        let Some(head_encoded) = sidecar.get_term(head).map(|term| term.encoded().to_vec()) else {
            self.bridge_sidecar = Some(sidecar);
            return None;
        };
        let arity_byte = mork_expr::item_byte(mork_expr::Tag::Arity(3));
        let prefixes = [output_prefix];
        let (count, any_new) =
            self.write_closure_facts(&mut sidecar, &head_encoded, arity_byte, &closure, &prefixes);
        self.bridge_sidecar = Some(sidecar);
        Some((count, any_new))
    }

    /// Maintains the transitive closure of a canonical linear-TC rule
    /// `path(x,z) :- edge(x,y), path(y,z)` incrementally as edges stream in,
    /// instead of recomputing it every exec step. A `MaintainedTransitiveClosure`
    /// persisted on the `Space` keyed by the edge relation folds in only the edge
    /// facts added since the last fire (insertion order via the sidecar's fact
    /// buckets, past a watermark) and writes just the delta pairs. The first fire
    /// builds the closure (O(edges)); later fires cost O(new pairs).
    ///
    /// Assumes the canonical reachability shape: the closure equals the transitive
    /// closure of the edge relation (path seeded to the edges). Returns `None`
    /// when the body is not that shape (the caller keeps the per-step transform).
    fn transform_streaming_linear_closure(
        &mut self,
        read_copy: &PathMap<()>,
        pat_expr: Expr,
        tpl_expr: Expr,
    ) -> Option<(usize, bool)> {
        let mut pat_args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut pat_args);
        let sources = pat_args.get(1..)?;
        if sources.len() != 2 {
            return None;
        }
        let mut tpl_args = Vec::new();
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates = tpl_args.get(1..)?;
        if templates.len() != 1 {
            return None;
        }
        let output_source = templates[0];
        let output_prefix = query_source_prefix(output_source)?;
        if output_prefix.len() < 2 {
            return None;
        }
        // One recursive factor (the output relation) and one fixed factor (edge).
        let prefixes = sources
            .iter()
            .map(|&source| query_source_prefix(source))
            .collect::<Option<Vec<Vec<u8>>>>()?;
        let recursive_index = if prefixes[0] == output_prefix {
            0
        } else if prefixes[1] == output_prefix {
            1
        } else {
            return None;
        };
        let fixed_index = 1 - recursive_index;
        if prefixes[fixed_index] == output_prefix {
            return None;
        }
        let edge_prefix = prefixes[fixed_index].clone();
        if edge_prefix.len() < 2 {
            return None;
        }
        // The composition edge(x,y), path(y,z) -> path(x,z), x, y, z distinct.
        let fixed = Self::query_factor_variables(sources[fixed_index]);
        let recursive = Self::query_factor_variables(sources[recursive_index]);
        let out = Self::query_factor_variables(output_source);
        if fixed.len() != 2 || recursive.len() != 2 || out.len() != 2 {
            return None;
        }
        if fixed[1] != recursive[0]
            || out[0] != fixed[0]
            || out[1] != recursive[1]
            || fixed[0] == fixed[1]
            || recursive[0] == recursive[1]
            || fixed[0] == recursive[1]
        {
            return None;
        }

        let mut sidecar = self
            .bridge_sidecar
            .take()
            .unwrap_or_else(crate::term_identity::TermIdentitySidecar::new);
        sidecar.invalidate_if_removed(self.bridge_remove_gen);
        let count = read_copy.read_zipper_at_path(&edge_prefix[..]).val_count();
        if sidecar
            .sync_prefix_if_stale(read_copy, &edge_prefix[..], count)
            .is_err()
        {
            self.bridge_sidecar = Some(sidecar);
            return None;
        }
        let Some(edge_head) = sidecar.term_id_for_encoded(&edge_prefix[1..]) else {
            self.bridge_sidecar = Some(sidecar);
            return None;
        };

        let (mut closure, mut watermark, observed) = self
            .bridge_closures
            .remove(&edge_prefix)
            .unwrap_or_else(|| {
                (
                    crate::binding_space::MaintainedTransitiveClosure::new(),
                    0,
                    self.bridge_remove_gen,
                )
            });
        if observed != self.bridge_remove_gen {
            // A removal happened: the insertion-only closure cannot retract a
            // removed edge's reachability, so rebuild from the live edges.
            closure = crate::binding_space::MaintainedTransitiveClosure::new();
            watermark = 0;
        }
        let edge_facts: Vec<_> = sidecar.facts_for_relation(edge_head).to_vec();
        let start = watermark.min(edge_facts.len());
        let mut delta = Vec::new();
        for &fact_id in &edge_facts[start..] {
            if !sidecar.is_fact_live(fact_id) {
                continue;
            }
            let Some(fact) = sidecar.get_fact(fact_id) else {
                continue;
            };
            if fact.flags.contains_vars {
                continue;
            }
            let Some(root) = sidecar.get_term(fact.root) else {
                continue;
            };
            let children = root.children();
            if children.len() != 3 || children.first().copied() != Some(edge_head) {
                continue;
            }
            let (from, to) = (children[1], children[2]);
            closure.insert_edge_into(from, to, &mut delta);
        }
        let new_watermark = edge_facts.len();

        // Write just the delta pairs as output (path) facts: prefix + from + to.
        let mut any_new = false;
        let mut written: Vec<Vec<u8>> = Vec::new();
        for &(from, to) in &delta {
            let (Some(from_term), Some(to_term)) = (sidecar.get_term(from), sidecar.get_term(to))
            else {
                continue;
            };
            let mut fact = output_prefix.clone();
            fact.extend_from_slice(from_term.encoded());
            fact.extend_from_slice(to_term.encoded());
            if self.btm.insert(&fact[..], ()).is_none() {
                any_new = true;
                written.push(fact);
            }
        }
        for fact in &written {
            let _ = sidecar.insert_fact(fact);
        }
        let total = closure.len();
        self.bridge_closures.insert(
            edge_prefix,
            (closure, new_watermark, self.bridge_remove_gen),
        );
        self.bridge_sidecar = Some(sidecar);
        Some((total, any_new))
    }

    /// Live self-check for the sidecar bridge. Behind the `sidecar_bridge`
    /// feature, after a `,`-pattern transform builds its read snapshot, this
    /// lowers the body, runs the sidecar planner, and compares the result against
    /// the ProductZipper for the same body, logging a warning on any
    /// disagreement. It never changes what the live path emits; it only observes.
    /// The whole-snapshot intern makes it a correctness probe, not a performance
    /// path. The incremental subspace interning (see
    /// `resources/incremental_interning_design.md`) is the separate speed work.
    #[cfg(feature = "sidecar_bridge")]
    fn bridge_self_check(read_copy: &PathMap<()>, pat_expr: Expr) {
        if let Some(comparison) = Self::validate_lowered_plan_against_product(read_copy, pat_expr) {
            if !comparison.matched {
                warn!(
                    target: "sidecar_bridge",
                    "sidecar plan disagreed with the ProductZipper ({} expected vs {} product rows); keeping ProductZipper",
                    comparison.expected_rows,
                    comparison.product_unique_rows,
                );
            }
        }
    }

    fn query_projection_maps(
        btm: &PathMap<()>,
        source: ExprEnv,
        prefix: &[u8],
        prefix_cardinality: usize,
        btm_val_count: Option<usize>,
    ) -> Option<QueryProjectionMaps> {
        if prefix_cardinality > QUERY_SHAPE_CARDINALITY_SCAN_LIMIT {
            return None;
        }

        let side_index_key = btm_val_count
            .zip(Self::query_factor_shape_cache_key(source))
            .map(|(btm_val_count, shape)| QueryProjectionSideIndexKey {
                btm_val_count,
                prefix_cardinality,
                prefix: prefix.to_vec(),
                shape,
            });

        if let Some(key) = side_index_key.as_ref() {
            if let Some(projection) = query_projection_side_index().lock().unwrap().get(key) {
                return Some(projection);
            }
        }

        let projection = Self::build_query_projection_maps(btm, source, prefix);

        if let Some(key) = side_index_key {
            query_projection_side_index()
                .lock()
                .unwrap()
                .insert(key, projection.clone());
        }

        Some(projection)
    }

    fn build_query_projection_maps(
        btm: &PathMap<()>,
        source: ExprEnv,
        prefix: &[u8],
    ) -> QueryProjectionMaps {
        let variables = Self::query_factor_variables(source);
        let positions = variables
            .iter()
            .copied()
            .enumerate()
            .map(|(index, var)| (var, index))
            .collect::<BTreeMap<_, _>>();
        let mut variable_maps = vec![PathMap::new(); variables.len()];

        let projection = Self::query_projection_summary(btm, source, prefix, |var, span| {
            if let Some(&index) = positions.get(&var) {
                variable_maps[index].set_val_at(span, ());
            }
        });

        QueryProjectionMaps {
            matches: projection.matches,
            ground_root_matches: projection.ground_root_matches,
            schematic_root_matches: projection.schematic_root_matches,
            variable_domains: projection.variable_domains,
            variable_maps,
            variable_rows: projection.variable_rows,
        }
    }

    fn query_projection_summary<F>(
        btm: &PathMap<()>,
        source: ExprEnv,
        prefix: &[u8],
        mut observe_binding: F,
    ) -> QueryProjectionSummary
    where
        F: FnMut((u8, u8), &[u8]),
    {
        let variables = Self::query_factor_variables(source);
        let mut variable_domains = vec![BTreeSet::new(); variables.len()];
        let mut variable_rows = Vec::new();
        let mut matches = 0usize;
        let mut ground_root_matches = 0usize;
        let mut schematic_root_matches = 0usize;
        let mut rz = btm.read_zipper_at_path(prefix);

        while rz.to_next_val() {
            let candidate = Expr {
                ptr: rz.origin_path().as_ptr().cast_mut(),
            };
            let mut pairs = vec![(source, ExprEnv::new(1, candidate))];
            let Ok(bindings) = unify(&mut pairs) else {
                continue;
            };

            matches += 1;
            if Self::encoded_expr_contains_vars(rz.path()).unwrap_or(true) {
                schematic_root_matches += 1;
            } else {
                ground_root_matches += 1;
            }
            let mut row = Vec::with_capacity(variables.len());
            let mut complete_row = true;
            for (domain, var) in variable_domains.iter_mut().zip(variables.iter()) {
                if let Some(binding) = bindings.get(var) {
                    if let Some(span) = unsafe { binding.subsexpr().span().as_ref() } {
                        let value = span.to_vec();
                        if domain.insert(value.clone()) {
                            observe_binding(*var, &value);
                        }
                        row.push(value);
                    } else {
                        complete_row = false;
                    }
                } else {
                    complete_row = false;
                }
            }
            if complete_row {
                variable_rows.push(row.into_boxed_slice());
            }
        }

        QueryProjectionSummary {
            matches,
            ground_root_matches,
            schematic_root_matches,
            variable_domains,
            variable_rows,
        }
    }

    fn encoded_expr_contains_vars(encoded: &[u8]) -> Option<bool> {
        let (contains_vars, end) = Self::encoded_expr_contains_vars_at(encoded, 0)?;
        (end == encoded.len()).then_some(contains_vars)
    }

    fn encoded_expr_contains_vars_at(encoded: &[u8], offset: usize) -> Option<(bool, usize)> {
        let byte = *encoded.get(offset)?;
        match maybe_byte_item(byte).ok()? {
            Tag::NewVar | Tag::VarRef(_) => Some((true, offset + 1)),
            Tag::SymbolSize(len) => {
                let end = offset + 1 + usize::from(len);
                (end <= encoded.len()).then_some((false, end))
            }
            Tag::Arity(arity) => {
                let mut cursor = offset + 1;
                let mut contains_vars = false;
                for _ in 0..arity {
                    let (child_contains_vars, end) =
                        Self::encoded_expr_contains_vars_at(encoded, cursor)?;
                    contains_vars |= child_contains_vars;
                    cursor = end;
                }
                Some((contains_vars, cursor))
            }
        }
    }

    // Shape-only key: no data-dependent parts, so computing it never walks the
    // space. The old key bucketed each queried prefix's val_count to replan on
    // cardinality-band crossings, but that walk ran on every call, hit or miss;
    // a workload that rewrites the relation it queries (odd_even_sort, millions
    // of steps) paid O(relation) per step inside val_count_below_node, 26x over
    // the plan-free path. Plans now freeze per shape (invalidate on schema
    // change, never on data change). If a workload needs adaptive replanning,
    // clear the cache on an epoch every N hits instead of re-keying per call.
    fn query_factor_plan_cache_key(sources: &[ExprEnv]) -> Option<QueryFactorPlanCacheKey> {
        let mut factors = Vec::with_capacity(sources.len());
        for source in sources {
            let span = unsafe { source.subsexpr().span().as_ref()? };
            factors.push(span.to_vec());
        }
        Some(QueryFactorPlanCacheKey { factors })
    }

    fn query_factor_shape_cache_key(source: ExprEnv) -> Option<Vec<u8>> {
        let capacity = unsafe { source.subsexpr().span().as_ref()?.len() };
        let mut key = Vec::with_capacity(capacity);
        let mut var_map = [u8::MAX; 64];
        let mut next_var = 0;
        Self::append_renormalized_query_factor(source, &mut var_map, &mut next_var, &mut key)?;
        Some(key)
    }

    fn query_factor_plan_uncached(
        btm: &PathMap<()>,
        sources: &[ExprEnv],
        btm_val_count: Option<usize>,
    ) -> Vec<usize> {
        let mut prefix_cardinalities = BTreeMap::new();
        let mut shape_cardinalities = BTreeMap::new();
        let ranks = sources
            .iter()
            .copied()
            .map(|source| {
                Self::query_factor_rank_with_btm_count(
                    btm,
                    source,
                    &mut prefix_cardinalities,
                    &mut shape_cardinalities,
                    btm_val_count,
                )
            })
            .collect::<Vec<_>>();
        query_factor_plan_metrics()
            .lock()
            .unwrap()
            .record_plan(&ranks);
        // Left-deep greedy join order with the connectivity heuristic, keeping the
        // previous cost comparator as the within-tier ranking. A pure ascending
        // cardinality sort puts small *unconnected* factors first, so the
        // ProductZipper builds a Cartesian product before the connecting factor
        // binds anything: a functional conjunctive query (`finite_domain`)
        // measured 0.04s in a connected order and >25s under the unconstrained
        // sort. So: never extend with a factor sharing no already-bound variable
        // unless forced (Selinger '79 connectivity heuristic); among connected
        // factors prefer the fewest new unbound variables (a functional lookup
        // keeps the intermediate flat) then the cost comparator; seed at the
        // highest-degree factor (the hub the rest depend on, e.g. `args`) so the
        // chain runs forward as 1:1 lookups instead of backward 1:many growth. A
        // connected body with no hub falls back to the comparator for both seed
        // and steps, reproducing the previous static order. GOO/MinCard: Microsoft
        // ICDE'10, Haffner & Dittrich SIGMOD'23.
        let n = sources.len();
        if n <= 1 {
            return (0..n).collect();
        }
        // The previous static cost comparator (cardinality, then projected-domain
        // / shape-refined / selective-prefix refinements), reused unchanged.
        let rank_cmp = |a: usize, b: usize| -> std::cmp::Ordering {
            ranks[a]
                .estimated_cardinality
                .cmp(&ranks[b].estimated_cardinality)
                .then_with(|| {
                    ranks[a]
                        .min_variable_domain_cardinality
                        .unwrap_or(usize::MAX)
                        .cmp(
                            &ranks[b]
                                .min_variable_domain_cardinality
                                .unwrap_or(usize::MAX),
                        )
                })
                .then_with(|| ranks[b].prefix_len.cmp(&ranks[a].prefix_len))
                .then_with(|| ranks[b].constant_items.cmp(&ranks[a].constant_items))
                .then_with(|| ranks[a].variable_items.cmp(&ranks[b].variable_items))
                .then_with(|| a.cmp(&b))
        };
        let var_sets: Vec<BTreeSet<(u8, u8)>> = sources
            .iter()
            .map(|&source| Self::query_factor_variables(source).into_iter().collect())
            .collect();
        let degree = |i: usize| -> usize {
            (0..n)
                .filter(|&j| j != i && !var_sets[i].is_disjoint(&var_sets[j]))
                .count()
        };
        // Keep the previous static cost order when it never forces a Cartesian
        // product (each factor after the first shares a bound variable). This
        // preserves the per-query tuning that process_calculus and the ordering
        // tests rely on; only a disconnected order (finite_domain's function
        // tables, which the cardinality sort interleaves with their inputs last)
        // falls through to the connectivity repair below.
        let mut static_order: Vec<usize> = (0..n).collect();
        static_order.sort_by(|&a, &b| rank_cmp(a, b));
        {
            let mut covered: BTreeSet<(u8, u8)> = BTreeSet::new();
            let mut connected = true;
            for (idx, &factor) in static_order.iter().enumerate() {
                if idx > 0 && var_sets[factor].is_disjoint(&covered) {
                    connected = false;
                    break;
                }
                covered.extend(var_sets[factor].iter().copied());
            }
            if connected {
                return static_order;
            }
        }
        let mut remaining: Vec<usize> = (0..n).collect();
        let mut bound: BTreeSet<(u8, u8)> = BTreeSet::new();
        let mut plan = Vec::with_capacity(n);
        while !remaining.is_empty() {
            let chosen = if bound.is_empty() {
                // Seed: highest degree, ties by the cost comparator (so a hubless
                // body starts at the previous static first factor).
                *remaining
                    .iter()
                    .max_by(|&&a, &&b| degree(a).cmp(&degree(b)).then_with(|| rank_cmp(b, a)))
                    .unwrap()
            } else {
                let connected: Vec<usize> = remaining
                    .iter()
                    .copied()
                    .filter(|&i| !var_sets[i].is_disjoint(&bound))
                    .collect();
                let pool = if connected.is_empty() {
                    &remaining
                } else {
                    &connected
                };
                *pool
                    .iter()
                    .min_by(|&&a, &&b| {
                        let new_vars =
                            |i: usize| var_sets[i].iter().filter(|v| !bound.contains(v)).count();
                        new_vars(a).cmp(&new_vars(b)).then_with(|| rank_cmp(a, b))
                    })
                    .unwrap()
            };
            bound.extend(var_sets[chosen].iter().copied());
            plan.push(chosen);
            remaining.retain(|&i| i != chosen);
        }
        plan
    }

    fn query_factor_plan(btm: &PathMap<()>, sources: &[ExprEnv]) -> Vec<usize> {
        let Some(cache_key) = Self::query_factor_plan_cache_key(sources) else {
            return Self::query_factor_plan_uncached(btm, sources, None);
        };

        {
            let mut cache = query_factor_plan_cache().lock().unwrap();
            if let Some(plan) = cache.get(&cache_key) {
                debug!(
                    target: "query_plan_cache",
                    "hit factors={}",
                    cache_key.factors.len()
                );
                return plan;
            }
        }

        // `btm.val_count()` is O(space size). It only feeds the shape side-index
        // cache key, which is consulted solely while ordering MULTIPLE factors; a
        // single-factor plan is always [0] (see the `n <= 1` early return in
        // query_factor_plan_uncached). Computing the whole-space count for every
        // single-pattern query made point queries O(N)/query (3ms on 500k atoms).
        let btm_count = (sources.len() > 1).then(|| btm.val_count());
        let plan = Self::query_factor_plan_uncached(btm, sources, btm_count);
        let mut cache = query_factor_plan_cache().lock().unwrap();
        debug!(
            target: "query_plan_cache",
            "miss factors={}",
            cache_key.factors.len()
        );
        cache.insert(cache_key, &plan);
        plan
    }

    pub fn query_factor_plan_cache_stats() -> QueryFactorPlanCacheStats {
        query_factor_plan_cache().lock().unwrap().stats()
    }

    pub fn query_shape_side_index_stats() -> QueryShapeSideIndexStats {
        query_shape_side_index().lock().unwrap().stats()
    }

    pub fn query_projection_side_index_stats() -> QueryProjectionSideIndexStats {
        query_projection_side_index().lock().unwrap().stats()
    }

    pub fn query_factor_plan_metrics_snapshot() -> QueryFactorPlanMetricsSnapshot {
        query_factor_plan_metrics().lock().unwrap().snapshot()
    }

    pub fn query_execution_storage_metrics_snapshot() -> QueryExecutionStorageMetricsSnapshot {
        let registry = query_execution_storage_metrics_registry().lock().unwrap();
        let mut total = QueryExecutionStorageMetrics::default();
        for cell in registry.iter() {
            total.merge(&cell.lock().unwrap());
        }
        total.snapshot()
    }

    fn append_renormalized_query_var(
        out: &mut Vec<u8>,
        var_map: &mut [u8; 64],
        next_var: &mut u8,
        original_var: usize,
    ) -> Option<()> {
        if original_var >= var_map.len() {
            return None;
        }
        match var_map[original_var] {
            u8::MAX => {
                if (*next_var as usize) >= var_map.len() {
                    return None;
                }
                var_map[original_var] = *next_var;
                *next_var += 1;
                out.push(item_byte(Tag::NewVar));
            }
            planned_var => out.push(item_byte(Tag::VarRef(planned_var))),
        }
        Some(())
    }

    fn append_renormalized_query_factor(
        source: ExprEnv,
        var_map: &mut [u8; 64],
        next_var: &mut u8,
        out: &mut Vec<u8>,
    ) -> Option<()> {
        let mut ez = ExprZipper::new(source.subsexpr());
        let mut local_newvars = source.v;
        loop {
            match ez.tag() {
                Tag::NewVar => {
                    Self::append_renormalized_query_var(
                        out,
                        var_map,
                        next_var,
                        local_newvars as usize,
                    )?;
                    local_newvars = local_newvars.checked_add(1)?;
                }
                Tag::VarRef(original_var) => {
                    Self::append_renormalized_query_var(
                        out,
                        var_map,
                        next_var,
                        original_var as usize,
                    )?;
                }
                Tag::SymbolSize(size) => unsafe {
                    out.extend_from_slice(
                        slice_from_raw_parts(ez.root.ptr.byte_add(ez.loc), size as usize + 1)
                            .as_ref()
                            .unwrap(),
                    );
                },
                Tag::Arity(arity) => out.push(item_byte(Tag::Arity(arity))),
            }
            if !ez.next() {
                break;
            }
        }
        Some(())
    }

    fn renormalize_query_factors(
        sources: &[ExprEnv],
        plan: &[usize],
    ) -> Option<(Vec<Vec<u8>>, Vec<ExprEnv>)> {
        let mut var_map = [u8::MAX; 64];
        let mut next_var = 0;
        let mut buffers = Vec::with_capacity(plan.len());
        let mut bases = Vec::with_capacity(plan.len());

        for &source_idx in plan {
            let source = sources[source_idx];
            let capacity = unsafe { source.subsexpr().span().as_ref().unwrap().len() };
            let mut buffer = Vec::with_capacity(capacity);
            let base = next_var;
            Self::append_renormalized_query_factor(
                source,
                &mut var_map,
                &mut next_var,
                &mut buffer,
            )?;
            buffers.push(buffer);
            bases.push(base);
        }

        record_storage_metrics(|m| m.record_renormalized_plan(&buffers));

        let planned_sources = buffers
            .iter()
            .zip(bases)
            .map(|(buffer, v)| ExprEnv {
                n: 0,
                v,
                offset: 0,
                base: Expr {
                    ptr: buffer.as_ptr().cast_mut(),
                },
            })
            .collect();
        Some((buffers, planned_sources))
    }

    /// Traces successful ProductZipper query candidates as byte binding rows.
    ///
    /// This diagnostic path reuses `query_multi` rather than duplicating its
    /// unsafe ProductZipper/coreferential traversal. It is an acceptance bridge
    /// for comparing current product semantics with BindingSpace sidecars and
    /// does not affect live query execution.
    pub fn trace_query_projection_product_candidates(
        btm: &PathMap<()>,
        product_pattern: Expr,
        query_variables_by_binding: impl IntoIterator<Item = (BindingVar, (u8, u8))>,
        variable_order: impl Into<Box<[BindingVar]>>,
    ) -> QueryProjectionProductCandidateTrace {
        let variable_order = variable_order.into();
        let query_variables_by_binding = query_variables_by_binding
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let factor_count = product_pattern
            .arity()
            .map_or(0, |arity| usize::from(arity).saturating_sub(1));
        let mut trace = QueryProjectionProductCandidateTrace {
            factor_count,
            variable_order,
            ..QueryProjectionProductCandidateTrace::default()
        };

        let mut raw = QueryProjectionProductRawCandidateCounters::default();
        let successful_candidates = Self::query_multi_with_raw_counters(
            btm,
            product_pattern,
            Some(&mut raw),
            |result, _loc| {
                match result {
                    Err(bindings) => {
                        let mut row = Vec::with_capacity(trace.variable_order.len());
                        let mut complete = true;
                        for variable in trace.variable_order.iter() {
                            let Some(query_variable) = query_variables_by_binding.get(variable)
                            else {
                                complete = false;
                                break;
                            };
                            if let Some(value) =
                                query_binding_value_bytes(&bindings, *query_variable)
                            {
                                row.push(value);
                            } else {
                                complete = false;
                                break;
                            }
                        }
                        if complete {
                            trace.rows.push(row.into_boxed_slice());
                        } else {
                            trace.missing_binding_rows += 1;
                        }
                    }
                    Ok(_) => {
                        trace.non_binding_results += 1;
                    }
                }
                true
            },
        );
        trace.successful_candidates = successful_candidates;
        trace.raw = raw;

        let mut unique_rows = trace.rows.clone();
        unique_rows.sort();
        unique_rows.dedup();
        trace.unique_rows = unique_rows;
        trace
    }

    pub fn query_multi<F: FnMut(Result<&[u32], BTreeMap<(u8, u8), ExprEnv>>, Expr) -> bool>(
        btm: &PathMap<()>,
        pat_expr: Expr,
        effect: F,
    ) -> usize {
        Self::query_multi_with_raw_counters(btm, pat_expr, None, effect)
    }

    fn query_multi_with_raw_counters<
        F: FnMut(Result<&[u32], BTreeMap<(u8, u8), ExprEnv>>, Expr) -> bool,
    >(
        btm: &PathMap<()>,
        pat_expr: Expr,
        mut raw_counters: Option<&mut QueryProjectionProductRawCandidateCounters>,
        mut effect: F,
    ) -> usize {
        let pat_newvars = pat_expr.newvars();
        trace!(target: "query_multi", "pattern (newvars={}) {:?}", pat_newvars, serialize(unsafe { pat_expr.span().as_ref().unwrap() }));
        let n_factors = pat_expr.arity().unwrap() as usize;
        debug_assert!(n_factors > 0);
        if n_factors == 1 {
            effect(Err(BTreeMap::new()), pat_expr);
            return 1;
        }
        let mut pat_args = Vec::with_capacity(n_factors);
        ExprEnv::new(0, pat_expr).args(&mut pat_args);

        let sources = &pat_args[1..];
        let preserve_source_order = sources
            .iter()
            .any(|&source| Self::expr_has_nonground_compound(source.subsexpr()));
        // Single-factor fast path. A one-source query has an unconditional plan of `[0]`
        // (the `n <= 1` early return in query_factor_plan_uncached), and a 0-normalized source
        // renormalizes to itself, so matching it directly is byte-identical to the planned
        // path. Taking it avoids both `query_factor_plan`'s global cache mutex and
        // `renormalize_query_factors`'s metrics mutex, each locked once per query; profiling a
        // single-pattern point-query loop showed those locks dominate per-query cost and
        // serialize parallel queries (throughput collapses past ~8 threads). Multi-factor
        // queries are unchanged.
        let (
            _planned_buffers,
            planned_sources,
            planned_unify_sources,
            primary_source_index,
            plan_reordered,
        ) = if sources.len() == 1 || preserve_source_order {
            (Vec::new(), sources.to_vec(), sources.to_vec(), 0, false)
        } else {
            let plan = Self::query_factor_plan(btm, sources);
            let planned = Self::renormalize_query_factors(sources, &plan);
            let plan_reordered = planned.is_some()
                && plan
                    .iter()
                    .enumerate()
                    .any(|(i, &source_idx)| i != source_idx);
            match planned {
                Some((buffers, planned_sources)) => {
                    let planned_unify_sources = plan
                        .iter()
                        .map(|&source_idx| sources[source_idx])
                        .collect::<Vec<_>>();
                    (
                        buffers,
                        planned_sources,
                        planned_unify_sources,
                        plan.iter().position(|&source_idx| source_idx == 0).unwrap(),
                        plan_reordered,
                    )
                }
                None => (
                    Vec::new(),
                    sources.to_vec(),
                    sources.to_vec(),
                    0,
                    plan_reordered,
                ),
            }
        };
        let mut prz = ProductZipper::new(
            btm.read_zipper(),
            (0..(sources.len() - 1)).map(|_i| btm.read_zipper()),
        );
        reserve_query_product_buffers(&mut prz);

        let touched = Self::query_multi_raw_with_unification_sources(
            &mut prz,
            &planned_sources,
            &planned_unify_sources,
            primary_source_index,
            raw_counters.as_deref_mut(),
            &mut effect,
        );
        if touched == 0 && plan_reordered {
            let identity_plan = (0..sources.len()).collect::<Vec<_>>();
            let (_identity_buffers, identity_sources) =
                Self::renormalize_query_factors(sources, &identity_plan)
                    .unwrap_or_else(|| (Vec::new(), sources.to_vec()));
            let mut identity_prz = ProductZipper::new(
                btm.read_zipper(),
                (0..(sources.len() - 1)).map(|_i| btm.read_zipper()),
            );
            reserve_query_product_buffers(&mut identity_prz);
            return Self::query_multi_raw_with_unification_sources(
                &mut identity_prz,
                &identity_sources,
                sources,
                0,
                raw_counters.as_deref_mut(),
                &mut effect,
            );
        }
        touched
    }

    #[inline]
    unsafe fn read_handler<'trie, 'path>(
        btm: *const PathMap<()>,
        mmaps: *mut HashMap<OwnedSourceItem, ArenaCompactTree<memmap2::Mmap>>,
        _z3s: *mut HashMap<OwnedSourceItem, Box<Popen>>,
        request: ResourceRequest,
    ) -> Resource<'trie, 'path> {
        match request {
            ResourceRequest::BTM(prefix) => {
                Resource::BTM(unsafe { btm.as_ref().unwrap() }.read_zipper_at_path(prefix))
            }
            ResourceRequest::ACT(name) => {
                let act = unsafe { mmaps.as_mut().unwrap() }
                    .entry(OwnedSourceItem::from(name))
                    .or_insert_with(|| {
                        trace!(target: "query_multi_i", "open new ACT {}", name);
                        ArenaCompactTree::open_mmap(format!("{ACT_PATH}{name}.act")).unwrap()
                    });
                trace!(target: "query_multi_i", "taking RZ of {}", name);
                Resource::ACT(act.read_zipper())
            }
            #[cfg(feature = "z3")]
            ResourceRequest::Z3(instance) => {
                trace!(target: "query_multi_i", "getting z3 instance");
                let z3 = unsafe { _z3s.as_mut().unwrap() }
                    .get_mut(&OwnedSourceItem::from(instance))
                    .unwrap_or_else(|| panic!("non existent z3 {}", instance));
                z3.stdin
                    .as_mut()
                    .expect("access to z3 stdin")
                    .write_all("(check-sat)\n".as_bytes())
                    .expect("written all");
                z3.stdin
                    .as_mut()
                    .expect("access to z3 stdin")
                    .write_all("(get-model)\n".as_bytes())
                    .expect("written all");
                z3.stdin
                    .as_mut()
                    .expect("access to z3 stdin")
                    .flush()
                    .expect("flushed all");
                trace!(target: "query_multi_i", "z3 ran (check-sat) and (get-model)");
                let mut v = String::new();
                let mut reader =
                    std::io::BufReader::new(z3.stdout.as_mut().expect("access to z3 stdout"));
                reader.read_line(&mut v).unwrap();
                if &v == "sat\n" {
                    v.clear();
                    let mut last = 0;
                    while &v.as_bytes()[last..] != b")\n" {
                        last = v.as_bytes().len();
                        reader.read_line(&mut v).unwrap();
                    }
                    trace!(target: "query_multi_i", "z3 read '{}'", &v[1..last]);
                    let mut s = Space::new();
                    s.add_all_sexpr(&v.as_bytes()[1..last]).unwrap();
                    // let mut v_ = Vec::new();
                    // s.dump_all_sexpr(&mut v_);
                    // trace!(target: "query_multi_i", "z3 read '{}'", std::str::from_utf8(&v_[..]).unwrap());
                    let btm = std::mem::take(&mut s.btm);
                    let rz = btm.into_read_zipper(&[]);
                    Resource::Z3(rz)
                } else {
                    trace!(target: "query_multi_i", "z3 problem not sat: {}", v);
                    Resource::Z3(PathMap::new().into_read_zipper(&[]))
                }
            }
        }
    }

    #[inline]
    unsafe fn write_handler<'w, 'a, 'k>(
        zh_wzs: (
            *mut ZipperHead<'w, 'a, ()>,
            *mut Vec<WriteZipperTracked<'a, 'k, ()>>,
        ),
        _mmaps: *mut HashMap<OwnedSourceItem, ArenaCompactTree<memmap2::Mmap>>,
        _z3s: *mut HashMap<OwnedSourceItem, Box<Popen>>,
        request: &WriteResourceRequest,
    ) -> WriteResource<'w, 'a, 'k>
    where
        'w: 'a,
    {
        match *request {
            WriteResourceRequest::BTM(p) => {
                let zh = unsafe { zh_wzs.0.as_mut().unwrap() };
                let wzs = unsafe { zh_wzs.1.as_mut::<'w>().unwrap() };
                wzs.push(unsafe { zh.write_zipper_at_exclusive_path_unchecked(p) });
                WriteResource::BTM(wzs.last_mut().unwrap())
            }
            WriteResourceRequest::ACT(_f) => WriteResource::ACT(()),
            #[cfg(feature = "z3")]
            WriteResourceRequest::Z3(f) => {
                let mut cfg = PopenConfig::default();
                cfg.stdin = Redirection::Pipe;
                cfg.stdout = Redirection::Pipe;
                trace!(target: "transform", "retrieving z3 instance");
                let instance = unsafe { _z3s.as_mut().unwrap() }
                    .entry(OwnedSourceItem::from(f))
                    .or_insert_with(|| {
                        trace!(target: "transform", "creating new z3 popen");
                        // let bpopen = Box::new(Popen::create(&["python", "resources/fake_cli.py", "-in", "-smt2"], cfg).unwrap());
                        let bpopen = Box::new(
                            Popen::create(&["z3", "-in", "-smt2"], cfg)
                                .expect("z3: command not found"),
                        );
                        trace!(target: "transform", "created new z3 popen");
                        bpopen
                    })
                    .as_mut();
                WriteResource::Z3(instance)
            }
        }
    }

    pub fn query_multi_i<F: FnMut(Result<&[u32], BTreeMap<(u8, u8), ExprEnv>>, Expr) -> bool>(
        no_source: bool,
        mmaps: &mut HashMap<OwnedSourceItem, ArenaCompactTree<memmap2::Mmap>>,
        z3s: &mut HashMap<OwnedSourceItem, Box<Popen>>,
        btm: &PathMap<()>,
        pat_expr: Expr,
        mut effect: F,
    ) -> usize {
        use crate::sources::{ASource, Source};

        let pat_newvars = pat_expr.newvars();
        trace!(target: "query_multi_i", "pattern (newvars={}) {:?}", pat_newvars, serialize(unsafe { pat_expr.span().as_ref().unwrap() }));
        let n_factors = pat_expr.arity().unwrap() as usize;
        debug_assert!(n_factors > 0);
        if n_factors == 1 {
            effect(Err(BTreeMap::new()), pat_expr);
            return 1;
        }
        let mut pat_args = Vec::with_capacity(n_factors);
        ExprEnv::new(0, pat_expr).args(&mut pat_args);
        let sources = &pat_args[1..];
        let preserve_source_order = sources
            .iter()
            .any(|&source| Self::expr_has_nonground_compound(source.subsexpr()));
        let plan = if preserve_source_order {
            (0..sources.len()).collect::<Vec<_>>()
        } else {
            Self::query_factor_plan(btm, sources)
        };
        let planned = if preserve_source_order {
            None
        } else {
            Self::renormalize_query_factors(sources, &plan)
        };
        let plan_reordered = planned.is_some()
            && plan
                .iter()
                .enumerate()
                .any(|(i, &source_idx)| i != source_idx);
        let (_planned_buffers, planned_sources, planned_unify_sources, primary_source_index) =
            match planned {
                Some((buffers, planned_sources)) => {
                    let planned_unify_sources = plan
                        .iter()
                        .map(|&source_idx| sources[source_idx])
                        .collect::<Vec<_>>();
                    (
                        buffers,
                        planned_sources,
                        planned_unify_sources,
                        plan.iter().position(|&source_idx| source_idx == 0).unwrap(),
                    )
                }
                None => (Vec::new(), sources.to_vec(), sources.to_vec(), 0),
            };

        let mut run_plan = |search_sources: &[ExprEnv],
                            unify_sources: &[ExprEnv],
                            effect_source_index: usize,
                            effect: &mut F| {
            trace!(target: "query_multi_i", "z3s {:?}", z3s.keys().collect::<Vec<_>>());
            let Some((primary_source, rest_sources)) = search_sources.split_first() else {
                return 0;
            };
            let mut open_factor = |e: &ExprEnv| {
                let src = if no_source {
                    ASource::compat(e.subsexpr())
                } else {
                    ASource::new(e.subsexpr())
                };
                src.source(
                    src.request()
                        .map(|request| unsafe { Self::read_handler(btm, mmaps, z3s, request) }),
                )
            };

            let primary = open_factor(primary_source);
            let mut factors: Vec<_> = Vec::with_capacity(rest_sources.len());
            for e in rest_sources {
                factors.push(open_factor(e));
            }

            match primary {
                AFactor::CompatSource(primary) => {
                    let mut prz = ProductZipper::new(primary, &mut factors[..]);
                    reserve_query_product_buffers(&mut prz);
                    Self::query_multi_raw_with_unification_sources(
                        &mut prz,
                        search_sources,
                        unify_sources,
                        effect_source_index,
                        None,
                        effect,
                    )
                }
                primary => {
                    trace!(target: "query_multi_i", "PZG of {:?}", factors.len() + 1);
                    let mut prz = ProductZipperG::new(primary, &mut factors[..]);
                    reserve_query_product_buffers(&mut prz);
                    Self::query_multi_raw_with_unification_sources(
                        &mut prz,
                        search_sources,
                        unify_sources,
                        effect_source_index,
                        None,
                        effect,
                    )
                }
            }
        };

        let touched = run_plan(
            &planned_sources,
            &planned_unify_sources,
            primary_source_index,
            &mut effect,
        );
        if touched == 0 && plan_reordered {
            let identity_plan = (0..sources.len()).collect::<Vec<_>>();
            let (_identity_buffers, identity_sources) =
                Self::renormalize_query_factors(sources, &identity_plan)
                    .unwrap_or_else(|| (Vec::new(), sources.to_vec()));
            return run_plan(&identity_sources, sources, 0, &mut effect);
        }
        touched
    }

    #[cfg(feature = "no_search")]
    #[inline(always)]
    pub fn query_multi_raw<
        PZ: ZipperProduct,
        F: FnMut(Result<&[u32], BTreeMap<(u8, u8), ExprEnv>>, Expr) -> bool,
    >(
        prz: &mut PZ,
        sources: &[ExprEnv],
        effect: F,
    ) -> usize {
        Self::query_multi_raw_with_unification_sources(prz, sources, sources, 0, None, effect)
    }

    #[cfg(feature = "no_search")]
    #[inline(always)]
    fn query_multi_raw_with_unification_sources<
        PZ: ZipperProduct,
        F: FnMut(Result<&[u32], BTreeMap<(u8, u8), ExprEnv>>, Expr) -> bool,
    >(
        prz: &mut PZ,
        search_sources: &[ExprEnv],
        unify_sources: &[ExprEnv],
        effect_source_index: usize,
        mut raw_counters: Option<&mut QueryProjectionProductRawCandidateCounters>,
        mut effect: F,
    ) -> usize {
        debug_assert_eq!(search_sources.len(), unify_sources.len());
        debug_assert!(effect_source_index < unify_sources.len());
        let mut raw_metrics = QueryRawStorageMetrics::default();
        raw_metrics.record_raw_search(search_sources.len());
        let mut candidate = 0;

        while prz.to_next_val() {
            if prz.focus_factor() != prz.factor_count() - 1 {
                continue;
            };
            let e = Expr {
                ptr: prz.origin_path().as_ptr().cast_mut(),
            };
            trace!(target: "query_multi_ref", "pi {:?}", prz.path_indices());
            trace!(target: "query_multi_ref", "at {:?}", e);
            for &other_i in prz.path_indices() {
                trace!(target: "query_multi_ref", "at {:?}",
                    Expr { ptr: unsafe { prz.origin_path().as_ptr().cast_mut().add(other_i) } });
            }
            unsafe {
                UNIFICATIONS += 1;
            }
            // if e.variables() != 0 {

            let mut pairs = query_unify_stack((unify_sources[0], ExprEnv::new(1, e)));

            for (&pa, &other_i) in unify_sources[1..].iter().zip(prz.path_indices()) {
                let fe = ExprEnv::new(
                    (pairs.len() + 1) as u8,
                    Expr {
                        ptr: unsafe { prz.origin_path().as_ptr().cast_mut().add(other_i) },
                    },
                );
                pairs.push((pa, fe))
            }

            raw_metrics.record_candidate_pairs(pairs.len(), pairs.capacity());
            if let Some(counters) = raw_counters.as_mut() {
                counters.record_candidate_pairs(pairs.len());
            }

            // pairs.iter().for_each(|(x, y)| println!("pair {} {}", x.show(), y.show()));

            #[cfg(feature = "inline_unify_stack")]
            let bindings = mork_expr::unify_inline(&mut pairs);
            #[cfg(not(feature = "inline_unify_stack"))]
            let bindings = unify(&mut pairs);
            raw_metrics.record_general_unification(bindings.is_ok());
            if let Some(counters) = raw_counters.as_mut() {
                match &bindings {
                    Ok(_) => counters.record_successful_unification(),
                    Err(failed) => counters.record_unification_failure(failed),
                }
            }

            match bindings {
                Ok(bs) => {
                    unsafe {
                        std::ptr::write_volatile(
                            &mut candidate,
                            std::ptr::read_volatile(&candidate) + 1,
                        );
                    }
                    let effect_loc = if effect_source_index == 0 {
                        e
                    } else {
                        Expr {
                            ptr: unsafe {
                                prz.origin_path()
                                    .as_ptr()
                                    .cast_mut()
                                    .add(prz.path_indices()[effect_source_index - 1])
                            },
                        }
                    };
                    if !effect(Err(bs), effect_loc) {
                        break;
                    }
                }
                Err(failed) => match failed {
                    UnificationFailure::Occurs(v, e) => {
                        trace!(target: "query_multi", "U {:?} occurs in {}", v, e.show())
                    }
                    UnificationFailure::Difference(lhs, rhs) => {
                        trace!(target: "query_multi", "U {} differs from {}", lhs.show(), rhs.show())
                    }
                    UnificationFailure::MaxIter(iter) => {
                        trace!(target: "query_multi", "U reached max iter {}", iter)
                    }
                },
            }
        }

        record_storage_metrics(|m| m.record_raw_search(&raw_metrics));

        candidate
    }

    #[cfg(not(feature = "no_search"))]
    #[inline(always)]
    pub fn query_multi_raw<
        PZ: ZipperProduct,
        F: FnMut(Result<&[u32], BTreeMap<(u8, u8), ExprEnv>>, Expr) -> bool,
    >(
        prz: &mut PZ,
        sources: &[ExprEnv],
        effect: F,
    ) -> usize {
        Self::query_multi_raw_with_unification_sources(prz, sources, sources, 0, None, effect)
    }

    #[cfg(not(feature = "no_search"))]
    #[inline(always)]
    fn query_multi_raw_with_unification_sources<
        PZ: ZipperProduct,
        F: FnMut(Result<&[u32], BTreeMap<(u8, u8), ExprEnv>>, Expr) -> bool,
    >(
        mut prz: &mut PZ,
        search_sources: &[ExprEnv],
        unify_sources: &[ExprEnv],
        effect_source_index: usize,
        mut raw_counters: Option<&mut QueryProjectionProductRawCandidateCounters>,
        mut effect: F,
    ) -> usize {
        debug_assert_eq!(search_sources.len(), unify_sources.len());
        debug_assert!(effect_source_index < unify_sources.len());
        let mut stack = search_sources[0..]
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        let mut raw_metrics = QueryRawStorageMetrics::default();
        raw_metrics.record_raw_search(stack.len());

        let references: Vec<u32> = vec![];
        let mut candidate = 0;
        thread_local! {
            static BREAK: std::cell::RefCell<[u64; 64]> = const { std::cell::RefCell::new([0; 64]) };
        }

        BREAK.with_borrow_mut(|a| {
            if unsafe { setjmp(a) == 0 } {
                let mut effect_fn = |loc: &mut &mut PZ| {
                    let e = Expr { ptr: loc.origin_path().as_ptr().cast_mut() };
                    trace!(target: "query_multi", "pi {:?}", loc.path_indices());
                    trace!(target: "query_multi", "at {:?}", e);
                    for &other_i in loc.path_indices() {
                        trace!(target: "query_multi", "at {:?}",
                            Expr { ptr: unsafe { loc.origin_path().as_ptr().cast_mut().add(other_i) } });
                    }
                    unsafe { UNIFICATIONS += 1; }
                    // if e.variables() != 0 {
                    if true {
                        let mut pairs = query_unify_stack((unify_sources[0], ExprEnv::new(1, e)));

                        for (&pa, &other_i) in unify_sources[1..].iter().zip(loc.path_indices()) {
                            let fe = ExprEnv::new((pairs.len() + 1) as u8,
                                                   Expr { ptr: unsafe { loc.origin_path().as_ptr().cast_mut().add(other_i) } });
                            pairs.push((pa, fe))
                        }

                        raw_metrics.record_candidate_pairs(pairs.len(), pairs.capacity());
                        if let Some(counters) = raw_counters.as_mut() {
                            counters.record_candidate_pairs(pairs.len());
                        }

                        // pairs.iter().for_each(|(x, y)| println!("pair {} {}", x.show(), y.show()));

                        #[cfg(feature = "inline_unify_stack")]
                        let bindings = mork_expr::unify_inline(&mut pairs);
                        #[cfg(not(feature = "inline_unify_stack"))]
                        let bindings = unify(&mut pairs);
                        raw_metrics.record_general_unification(bindings.is_ok());
                        if let Some(counters) = raw_counters.as_mut() {
                            match &bindings {
                                Ok(_) => counters.record_successful_unification(),
                                Err(failed) => counters.record_unification_failure(failed),
                            }
                        }

                        match bindings {
                            Ok(bs) => {
                                unsafe { std::ptr::write_volatile(&mut candidate, std::ptr::read_volatile(&candidate) + 1); }
                                let effect_loc = if effect_source_index == 0 {
                                    e
                                } else {
                                    Expr {
                                        ptr: unsafe {
                                            loc.origin_path()
                                                .as_ptr()
                                                .cast_mut()
                                                .add(loc.path_indices()[effect_source_index - 1])
                                        },
                                    }
                                };
                                if !effect(Err(bs), effect_loc) {
                                    unsafe { longjmp(a, 1) }
                                }
                            }
                            Err(failed) => {
                                match failed {
                                    UnificationFailure::Occurs(v, e) => {
                                        trace!(target: "query_multi", "U {:?} occurs in {}", v, e.show())
                                    }
                                    UnificationFailure::Difference(lhs, rhs) => {
                                        trace!(target: "query_multi", "U {} differs from {}", lhs.show(), rhs.show())
                                    }
                                    UnificationFailure::MaxIter(iter) => {
                                        trace!(target: "query_multi", "U reached max iter {}", iter)
                                    }
                                }
                            }
                        }
                    } else {
                        trace!(target: "query_multi", "#variables==0 {:?}", e);
                        unsafe { std::ptr::write_volatile(&mut candidate, std::ptr::read_volatile(&candidate) + 1); }
                        if !effect(Ok(unsafe { slice_from_raw_parts(references.as_ptr(), references.len()).as_ref().unwrap() }), e) {
                            unsafe { longjmp(a, 1) }
                        }
                    }
                };
                // Run the compiled WAM-style program when the pattern lowers;
                // the interpreted matcher is the fallback and the differential
                // oracle. See exec_to_stream_transpiler_plan.md.
                match compile_match_program(search_sources) {
                    Some(ops) => {
                        let mut scratch = MatchProgramScratch::default();
                        execute_match_program(
                            &mut prz,
                            &ops,
                            0,
                            unsafe {
                                ((&references) as *const Vec<u32>).cast_mut().as_mut().unwrap()
                            },
                            &mut scratch,
                            &mut effect_fn,
                        )
                    }
                    None => coreferential_transition(
                        &mut prz,
                        &mut stack,
                        unsafe { ((&references) as *const Vec<u32>).cast_mut().as_mut().unwrap() },
                        &mut effect_fn,
                    ),
                }
            }
        });

        record_storage_metrics(|m| m.record_raw_search(&raw_metrics));

        candidate
    }

    pub fn prefix_subsumption(prefixes: &[&[u8]]) -> Vec<usize> {
        let mut prefix_index = PathMap::<usize>::new();
        let mut root_owner = None;

        for (idx, &prefix) in prefixes.iter().enumerate() {
            Self::record_prefix_owner(&mut prefix_index, &mut root_owner, prefix, idx);
        }

        prefixes
            .iter()
            .enumerate()
            .map(|(idx, &prefix)| {
                Self::subsuming_prefix_owner(&prefix_index, root_owner, prefix, idx)
            })
            .collect()
    }

    fn subsuming_prefix_owner(
        prefix_index: &PathMap<usize>,
        root_owner: Option<usize>,
        prefix: &[u8],
        fallback: usize,
    ) -> usize {
        if let Some(owner) = root_owner {
            return owner;
        }

        let mut zipper = prefix_index.read_zipper();
        for &byte in prefix {
            if !zipper.descend_to_existing_byte(byte) {
                break;
            }
            if let Some(&owner) = zipper.val() {
                return owner;
            }
        }

        fallback
    }

    fn record_prefix_owner(
        prefix_index: &mut PathMap<usize>,
        root_owner: &mut Option<usize>,
        prefix: &[u8],
        idx: usize,
    ) {
        if prefix.is_empty() {
            *root_owner = Some(root_owner.as_ref().map_or(idx, |&owner| owner.min(idx)));
            return;
        }

        let mut zipper = prefix_index.write_zipper_at_path(prefix);
        if zipper.val().is_none_or(|&owner| idx < owner) {
            zipper.set_val(idx);
        }
    }

    pub(crate) fn prefix_subsumption_resources(
        requests: &[crate::sinks::WriteResourceRequest],
    ) -> Vec<usize> {
        let mut prefix_index = PathMap::<usize>::new();
        let mut root_owner = None;
        let mut act_owners = HashMap::<&'static str, usize>::new();
        #[cfg(feature = "z3")]
        let mut z3_owners = HashMap::<&'static str, usize>::new();

        for (idx, request) in requests.iter().enumerate() {
            match request {
                crate::sinks::WriteResourceRequest::BTM(prefix) => {
                    Self::record_prefix_owner(&mut prefix_index, &mut root_owner, prefix, idx);
                }
                crate::sinks::WriteResourceRequest::ACT(name) => {
                    let owner = act_owners.entry(*name).or_insert(idx);
                    *owner = (*owner).min(idx);
                }
                #[cfg(feature = "z3")]
                crate::sinks::WriteResourceRequest::Z3(name) => {
                    let owner = z3_owners.entry(*name).or_insert(idx);
                    *owner = (*owner).min(idx);
                }
            }
        }

        let mut out = Vec::with_capacity(requests.len());
        for (idx, request) in requests.iter().enumerate() {
            let owner = match request {
                crate::sinks::WriteResourceRequest::BTM(prefix) => {
                    Self::subsuming_prefix_owner(&prefix_index, root_owner, prefix, idx)
                }
                crate::sinks::WriteResourceRequest::ACT(name) => {
                    act_owners.get(name).copied().unwrap_or(idx)
                }
                #[cfg(feature = "z3")]
                crate::sinks::WriteResourceRequest::Z3(name) => {
                    z3_owners.get(name).copied().unwrap_or(idx)
                }
            };
            out.push(owner);
        }

        out
    }

    #[cfg(feature = "specialize_io")]
    pub fn transform_multi_multi_(
        &mut self,
        pat_expr: Expr,
        tpl_expr: Expr,
        add: Expr,
    ) -> (usize, bool) {
        // A linear-recursive rule (the body references the output relation)
        // iterates to a fixpoint when re-fired. Compute the whole least fixpoint
        // in one step by semi-naive evaluation instead, when the body lowers to a
        // linear fixpoint. Falls through otherwise (and a non-recursive body
        // always falls through, since `transform_linear_recursive_fixpoint`
        // declines it).
        #[cfg(feature = "semi_naive_fixpoint")]
        {
            let mut read_copy = self.btm.clone();
            read_copy.insert(unsafe { add.span().as_ref().unwrap() }, ());
            if let Some(result) =
                self.transform_streaming_linear_closure(&read_copy, pat_expr, tpl_expr)
            {
                return result;
            }
            if let Some(result) =
                self.transform_linear_recursive_fixpoint(&read_copy, pat_expr, tpl_expr)
            {
                return result;
            }
            if let Some(result) =
                self.transform_binary_transitive_closure(&read_copy, pat_expr, tpl_expr)
            {
                return result;
            }
        }

        // Semi-naive immediate-consequence step (Stage 3 of the semi-naive delta
        // lever). Inside the IC loop (`sni_rule_seen` is Some) run the m-delta-rule
        // transform: match each body factor against this rule's delta in turn, the
        // rest against the full space, and union (idempotent btm insert dedups). The
        // delta is `read_copy \ snapshot` where `snapshot` is the match input the
        // last time THIS rule fired (keyed by pattern+template, stable across the
        // exec re-arming). A first-seen rule has no snapshot, so its delta is the
        // whole space -> the m passes reproduce the naive full match exactly. Every
        // later firing is delta-restricted, so a step costs O(facts added since this
        // rule last ran) instead of O(dish). `read_copy` carries the just-consumed
        // exec `add`, exactly as the naive path inserts it below, so a rule that
        // matches its own exec (the IC driver) is byte-identical. The snapshot is
        // taken BEFORE the emit (= `read_copy`), so a recursive rule's own previous-
        // round output is in its next delta and the recursion still chains.
        // `sni_rule_seen == None` (every non-IC caller, every default build) falls
        // through to the naive full-space match. See the design doc.
        //
        // Soundness gate (Phase 6a): the per-rule delta is the Datalog semi-naive
        // recurrence, byte-identical to naive ONLY while evaluation stays monotone
        // (add-only). A retraction breaks it -- a fact a worker's snapshot recorded
        // then removed is excluded from `btm \ snapshot`, so the closure never
        // re-derives it through surviving facts even where naive (which re-scans
        // the whole dish) does. The corpus pinned the divergence boundary to "any
        // removal" (0 divergences without one over 400 random seeds). So once any
        // removal has happened in this loop (`sni_removal_seen` latched by the
        // `O`/`-` paths), fall through to the naive full-space match for every
        // `,`->`,` rule. `seen` is put back so the field stays `Some` for the
        // rest of the loop and `metta_calculus`'s restore is unaffected. This
        // keeps the feature UNCONDITIONALLY correct: semi-naive where monotone,
        // naive after the first retraction. process_calculus never retracts, so it
        // never trips the gate and keeps the fast path.
        #[cfg(feature = "semi_naive_ic")]
        if let Some(mut seen) = self.sni_rule_seen.take() {
            // The soundness gate forces naive after a retraction ONLY in `Naive`
            // mode. `Dred` repairs the per-rule views with a re-derivation catch-up
            // (see `sni_dred_catch_up`) so the delta stays sound; `RawSemiNoGate`
            // deliberately ignores the gate (diagnostic, may diverge). So the removal
            // half of the gate is mode-conditional.
            let removal_forces_naive =
                self.sni_removal_seen && self.sni_retract_mode == SniRetractMode::Naive;

            // DRed catch-up (Dred mode, a retraction happened): before this rule's
            // incremental delta is trusted again it must re-derive any removed-but-
            // rederivable fact it (or the closure) produces, exactly as naive's full
            // re-scan would. `sni_dred_catch_up` runs ONE naive full-match round for
            // the rule when its `caught_up_gen` is behind `sni_removal_gen` and sets
            // its snapshot to the pre-round dish (so a multi-round re-derivation chains
            // through the later incremental firings, step-for-step like naive); or
            // refuses (un-handled shape) and signals a naive fallback. A rule already
            // caught up to the current gen falls straight through to the normal
            // cost-gated delta fire. See the divergence ground truth: the only
            // naive-vs-semi gap is an explicitly-removed fact re-derived through
            // survivors, so the catch-up is pure monotone re-addition (no over-delete).
            if self.sni_retract_mode == SniRetractMode::Dred
                && self.sni_removal_seen
                && !self.sni_force_naive
            {
                match self.sni_dred_catch_up(&mut seen, pat_expr, tpl_expr, add) {
                    DredCatchUp::Done(result) => {
                        self.sni_rule_seen = Some(seen);
                        return result;
                    }
                    DredCatchUp::AlreadyCaughtUp => {
                        // fall through to the normal cost-gated delta fire below.
                    }
                    DredCatchUp::Fallback => {
                        // Un-handled shape: route this and every later rule to naive
                        // for the rest of the loop. Flip the mode to `Naive` so the
                        // `removal_forces_naive` test holds from here on, put the
                        // snapshots back, and run the naive full-space match for THIS
                        // rule directly. Conservative: a wrong dish is never risked.
                        self.sni_dred_fallbacks += 1;
                        self.sni_retract_mode = SniRetractMode::Naive;
                        self.sni_rule_seen = Some(seen);
                        return self.transform_multi_multi_naive(pat_expr, tpl_expr, add);
                    }
                }
            }

            if self.sni_force_naive || removal_forces_naive {
                // Routed to naive, for one of two reasons:
                //  - `sni_force_naive`: the runtime revert switch is on (force the
                //    naive path for every rule without a rebuild);
                //  - `sni_removal_seen` (soundness gate, `Naive` mode): a retraction
                //    happened in this loop, so the per-rule delta is unsound (see the
                //    field docs).
                // Either way put the snapshots back and fall through to the naive
                // full-space match for the rest of the loop. (When forced naive the
                // snapshot is never consulted again, but keeping the field `Some`
                // leaves `metta_calculus`'s restore unaffected.)
                self.sni_rule_seen = Some(seen);
            } else {
                // Cost gate (Phase 6b): `sni_delta_fire` runs the m-delta-rule ONLY
                // when it estimates the delta is small relative to the dish
                // (`m * delta_count < dish_count`), else it refreshes this rule's
                // snapshot and returns `None` so we fall through to the naive full
                // match. The m-delta-rule runs m passes, so on a small dish (delta
                // ~ dish, e.g. a rule's first firing or a dense clique body) it is
                // ~m x SLOWER than naive's single pass; the gate prevents that
                // regression while keeping the big win where delta << dish. The
                // snapshot is refreshed inside `sni_delta_fire` either way, so the
                // field stays `Some` for the rest of the loop. Both branches write a
                // byte-identical dish (the gate only chooses HOW to compute the same
                // immediate consequences), proven by the corpus + the gate-equivalence
                // test.
                if let Some(result) = self.sni_delta_fire(seen, pat_expr, tpl_expr, add) {
                    return result;
                }
                // gate chose naive: fall through to the naive full-space match below.
            }
        }

        self.transform_multi_multi_naive(pat_expr, tpl_expr, add)
    }

    /// The naive full-space match for a `,`->`,` rule: match the whole body against
    /// the full `read_copy` (the live `btm` plus the just-consumed exec `add`) and
    /// emit every template instantiation. This is the unconditional, always-correct
    /// path that every default-build caller and every semi-naive fall-through routes
    /// to. Extracted from `transform_multi_multi_` so the semi-naive gate, the DRed
    /// catch-up, and the conservative fallback can all reuse one definition.
    pub fn transform_multi_multi_naive(
        &mut self,
        pat_expr: Expr,
        tpl_expr: Expr,
        add: Expr,
    ) -> (usize, bool) {
        // Drive template writes from the sidecar's join when the body lowers and either its
        // cardinality-sorted order stays connected or the zipper-owned gate proves a schematic
        // body can route byte-identically. Disconnected bodies otherwise stay on the ProductZipper's
        // reordered plan.
        #[cfg(feature = "sidecar_bridge_emit")]
        if Self::body_cardinality_order_connected(&self.btm, pat_expr)
            || Self::body_has_safe_zipper_schematic_route(&self.btm, pat_expr)
        {
            let mut read_copy = self.btm.clone();
            read_copy.insert(unsafe { add.span().as_ref().unwrap() }, ());
            if let Some(result) = self.transform_via_sidecar(&read_copy, pat_expr, tpl_expr) {
                return result;
            }
        }
        // Optional acyclic full-unification capture route. Default-off, so the native ProductZipper
        // stays authoritative until explicitly opted in; the cheap query-side pre-check skips the
        // clone for every non-capture body.
        #[cfg(feature = "sidecar_bridge_emit")]
        if SIDECAR_CAPTURE_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
            && Self::query_has_nonground_compound(pat_expr)
        {
            let mut read_copy = self.btm.clone();
            read_copy.insert(unsafe { add.span().as_ref().unwrap() }, ());
            if let Some(result) = self.transform_via_capture(&read_copy, pat_expr, tpl_expr) {
                return result;
            }
        }
        let mut buffer = template_output_buffer();
        let mut tpl_args = Vec::with_capacity(64);
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<_> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
        let template_prefixes: Vec<_> = templates
            .iter()
            .map(|e| unsafe { e.prefix().unwrap_or_else(|x| x).as_ref().unwrap() })
            .collect();
        let mut subsumption = Self::prefix_subsumption(&template_prefixes[..]);
        let mut placements = subsumption.clone();
        let mut read_copy = self.btm.clone();
        let zh = self.btm.zipper_head();
        read_copy.insert(unsafe { add.span().as_ref().unwrap() }, ());
        let mut template_wzs: Vec<_> = Vec::with_capacity(64);
        template_prefixes.iter().enumerate().for_each(|(i, x)| {
            if subsumption[i] == i {
                placements[i] = template_wzs.len();
                template_wzs.push(unsafe { zh.write_zipper_at_exclusive_path_unchecked(x) });
            }
        });
        for i in 0..subsumption.len() {
            subsumption[i] = placements[subsumption[i]]
        }
        debug!(
            target: "transform",
            "write_resource_placement mode=multi_multi_i requests={} exclusive_writers={} reused_writers={}",
            template_prefixes.len(),
            template_wzs.len(),
            template_prefixes.len().saturating_sub(template_wzs.len())
        );
        trace!(target: "transform", "templates {:?}", templates);
        trace!(target: "transform", "prefixes {:?}", template_prefixes);
        trace!(target: "transform", "subsumption {:?}", subsumption);

        let mut assignments: Vec<(u8, u8)> = vec![];
        let mut trace: Vec<(u8, u8)> = vec![];

        let mut ass = Vec::with_capacity(64);
        let mut astack = Vec::with_capacity(64);

        let mut any_new = false;
        #[cfg(feature = "sidecar_bridge")]
        Self::bridge_self_check(&read_copy, pat_expr);
        // The pattern apply only computes the template intro seed (oi, ni); it
        // writes nothing (its sink is `void`). For ground bindings (oi structural,
        // ni == 0) the seed is invariant across matches, so compute it once and
        // reuse it, skipping the per-match pattern re-walk. A non-ground match
        // (ni != 0) recomputes, preserving the cycle (`!ok`) decline.
        let mut pattern_intros: Option<(u8, u8)> = None;
        let sni_trace = std::env::var("MORK_SNI_TRACE").is_ok();
        let touched = Self::query_multi(&read_copy, pat_expr, |refs_bindings, loc| 'query: {
            if sni_trace {
                match &refs_bindings {
                    Ok(refs) => eprintln!("SNI naive emit Ok(refs {refs:?})"),
                    Err(b) => eprintln!(
                        "SNI naive emit Err bindings: {:?}",
                        b.iter()
                            .map(|(k, ee)| (
                                *k,
                                serialize(unsafe { ee.subsexpr().span().as_ref().unwrap() })
                            ))
                            .collect::<Vec<_>>()
                    ),
                }
            }
            trace!(target: "transform", "data {}", serialize(unsafe { loc.span().as_ref().unwrap()}));
            unsafe {
                WRITES += template_prefixes.len();
            }
            match refs_bindings {
                Ok(_) => {
                    unreachable!()
                }
                Err(ref bindings) => {
                    #[cfg(debug_assertions)]
                    bindings.iter().for_each(
                        |(v, ee)| trace!(target: "transform", "binding {:?} {}", *v, ee.show()),
                    );

                    let ground_bindings = bindings.values().all(|ee| ee.subsexpr().is_ground());
                    let (oi0, ni) = match pattern_intros.filter(|_| ground_bindings) {
                        Some(cached) => cached,
                        None => {
                            let mut void = std::io::sink();
                            let (oi, ni, ok) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
                                0,
                                0,
                                0,
                                pat_expr,
                                bindings,
                                void,
                                trace,
                                assignments
                            );
                            if !ok {
                                break 'query true;
                            }
                            if ground_bindings && ni == 0 {
                                pattern_intros = Some((oi, ni));
                            }
                            (oi, ni)
                        }
                    };
                    let mut oi = oi0;

                    'writes: for (i, template) in templates.iter().enumerate() {
                        let wz = &mut template_wzs[subsumption[i]];

                        trace!(target: "transform", "{i} template {} @ ({oi} {ni})", serialize(unsafe { template.span().as_ref().unwrap()}));

                        buffer.clear();
                        let (toi, _, true) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
                            0, oi, ni, *template, bindings, buffer, astack, ass
                        ) else {
                            continue 'writes;
                        };
                        oi = toi;

                        if sni_trace {
                            eprintln!(
                                "EMITnaive pat={} out={}",
                                serialize(unsafe { pat_expr.span().as_ref().unwrap() }),
                                serialize(&buffer)
                            );
                        }
                        trace!(target: "transform", "U {i} out {:?}", Expr{ ptr: buffer.as_mut_ptr() });
                        wz.move_to_path(&buffer[wz.root_prefix_path().len()..]);
                        any_new |= wz.set_val(()).is_none();
                    }
                    true
                }
            }
        });
        for wz in template_wzs {
            zh.cleanup_write_zipper(wz);
        }
        (touched, any_new)
    }

    /// One DRed re-derivation catch-up for a `,`->`,` rule firing after a retraction
    /// (Dred mode). The divergence ground truth (`dred_divergence_ground_truth`) is
    /// that the ONLY naive-vs-semi gap under retraction is a fact EXPLICITLY removed
    /// by an `O`/`-` rule that the recursive closure re-derives through SURVIVING
    /// facts: naive's full re-scan re-derives it, the add-only delta (which skips
    /// old x ... x old combinations) misses it. The engine's `,`->`,` rules never
    /// retract derived facts (only the explicit `O`/`-` does), so naive is a MONOTONE
    /// TRACE -- it keeps stale derived facts whose support was removed. So the repair
    /// is PURE RE-DERIVATION (never an over-delete): run the rule's naive full match
    /// once, then resume the incremental delta.
    ///
    /// The catch-up is exactly ONE naive round (`transform_multi_multi_naive`), with
    /// the rule's snapshot set to the dish BEFORE the round. That makes the next
    /// firing's delta `dish_after_round \ dish_before_round` contain every fact this
    /// round (re-)derived, so a multi-round re-derivation chains through the normal
    /// incremental firings exactly as naive spreads it across steps -- step-for-step
    /// identical to naive, not a fixpoint collapse (which would emit re-armed execs
    /// out of order). After the catch-up the rule's `caught_up_gen` is set to the
    /// current `sni_removal_gen`, so it pays the full match only ONCE per retraction,
    /// then runs incrementally again -- versus `Naive` mode, which re-scans the whole
    /// dish on EVERY firing for the rest of the loop.
    ///
    /// Coverage is complete without a per-shape gate: every rule on the semi-naive
    /// delta path is a `,`->`,` rule (the gate lives only in `transform_multi_multi_`),
    /// and the catch-up IS a naive round for it; `I`-source and `O`-template rules are
    /// ALWAYS naive (full match every step), so they re-derive removed facts on their
    /// own. `Fallback` is returned only if the rule entry cannot be reconstructed (it
    /// never is in practice), routing to the naive path for safety.
    ///
    /// Returns:
    /// - `AlreadyCaughtUp` if this rule already re-derived since the last retraction
    ///   (`caught_up_gen == sni_removal_gen`); the caller runs the normal delta fire.
    /// - `Done(result)` after running the catch-up round and refreshing the snapshot.
    /// - `Fallback` if the entry is unreconstructable; the caller runs naive.
    #[cfg(feature = "semi_naive_ic")]
    fn sni_dred_catch_up(
        &mut self,
        seen: &mut HashMap<Vec<u8>, SniRuleEntry>,
        pat_expr: Expr,
        tpl_expr: Expr,
        add: Expr,
    ) -> DredCatchUp {
        let pat_span = unsafe { pat_expr.span().as_ref().unwrap() };
        let tpl_span = unsafe { tpl_expr.span().as_ref().unwrap() };
        let mut key = Vec::with_capacity(pat_span.len() + tpl_span.len() + 1);
        key.extend_from_slice(pat_span);
        key.push(0xff);
        key.extend_from_slice(tpl_span);

        let cur_gen = self.sni_removal_gen;

        // CONSERVATIVE-SAFETY TEST HOOK: route to the naive fallback on demand.
        if self.sni_dred_force_fallback {
            return DredCatchUp::Fallback;
        }

        // Already re-derived since the last retraction -> incremental is sound.
        if let Some(entry) = seen.get(&key) {
            if entry.caught_up_gen == cur_gen {
                return DredCatchUp::AlreadyCaughtUp;
            }
        }

        // The dish BEFORE this round (the live `btm` plus the just-consumed exec
        // `add`, exactly the input the naive round below matches). This becomes the
        // rule's snapshot so the NEXT delta captures everything this round derives.
        let add_span = unsafe { add.span().as_ref().unwrap() };
        let mut dish_before = self.btm.clone();
        dish_before.insert(add_span, ());
        let dish_before_count = dish_before.val_count();

        // One naive full-match round (the re-derivation): byte-identical to a naive
        // step for this rule. Re-derives any removed-but-rederivable fact reachable
        // in one round; multi-round re-derivations chain through later incremental
        // firings (their delta now includes this round's outputs).
        //
        // MUTATION-TEST HOOK: with `sni_dred_skip_rederive` the re-derivation round
        // is replaced by the ordinary add-only delta fire over this rule's frontier
        // -- exactly the buggy raw-semi behaviour the catch-up exists to fix. It
        // emits this round's monotone consequences but NOT the removed-but-rederivable
        // facts (those need a full re-scan), so the dish DIVERGES from naive on a
        // retraction repro, proving the re-derivation is load-bearing. Never set in
        // production.
        let result = if self.sni_dred_skip_rederive {
            let prior_snapshot = seen
                .get(&key)
                .map(|e| e.snapshot.clone())
                .unwrap_or_else(PathMap::new);
            let delta = dish_before.subtract_cow(&prior_snapshot);
            self.transform_multi_multi_delta(&dish_before, &delta, pat_expr, tpl_expr)
        } else {
            self.transform_multi_multi_naive(pat_expr, tpl_expr, add)
        };

        // Refresh the entry: snapshot = the pre-round dish, cached count, and mark
        // this rule caught up to the current retraction generation.
        seen.insert(
            key,
            SniRuleEntry {
                snapshot: dish_before,
                dish_count: dish_before_count,
                caught_up_gen: cur_gen,
            },
        );
        self.sni_dred_repairs += 1;
        DredCatchUp::Done(result)
    }

    /// Fire one semi-naive immediate-consequence step for a `,`->`,` rule, given
    /// the per-rule snapshot map (taken by `transform_multi_multi_`). Computes this
    /// rule's delta `read_copy \ snapshot` (the facts added since it last fired,
    /// the whole space on the first firing), and applies the COST GATE (Phase 6b):
    ///
    /// - if the delta is small relative to the dish (`m * delta_count < dish_count`,
    ///   where `m` is the body factor count), run the m-delta-rule
    ///   `transform_multi_multi_delta` and return `Some((candidates, any_new))`;
    /// - otherwise the m passes would cost ~m x the naive single pass (the delta ~
    ///   dish case: a rule's first firing, or a dense multi-factor body like the
    ///   clique/finite_domain benches, where a measurement showed up to 16,800x
    ///   slower than naive), so refresh the snapshot and return `None` for the
    ///   caller to fall through to the naive full-space match.
    ///
    /// Either way the snapshot is refreshed to `read_copy` (with its exact dish
    /// count cached), so the next firing's delta is exactly the facts added since.
    /// Both paths emit the SAME immediate consequences (the gate only chooses how to
    /// compute them), so the written dish is byte-identical. Only reached when the
    /// soundness gate is clear (no retraction yet this loop); see
    /// `transform_multi_multi_` and the field docs.
    #[cfg(feature = "semi_naive_ic")]
    fn sni_delta_fire(
        &mut self,
        mut seen: HashMap<Vec<u8>, SniRuleEntry>,
        pat_expr: Expr,
        tpl_expr: Expr,
        add: Expr,
    ) -> Option<(usize, bool)> {
        let add_span = unsafe { add.span().as_ref().unwrap() };
        let mut read_copy = self.btm.clone();
        read_copy.insert(add_span, ());

        // Rule key: pattern bytes then template bytes. Stable across re-arming
        // (the exec wrapper changes round to round, the , -body does not).
        let pat_span = unsafe { pat_expr.span().as_ref().unwrap() };
        let tpl_span = unsafe { tpl_expr.span().as_ref().unwrap() };
        let mut key = Vec::with_capacity(pat_span.len() + tpl_span.len() + 1);
        key.extend_from_slice(pat_span);
        key.push(0xff);
        key.extend_from_slice(tpl_span);

        // The body factor count `m` (the `,`-args count). The m-delta-rule runs
        // one pass per factor.
        let mut pat_args = Vec::with_capacity(64);
        ExprEnv::new(0, pat_expr).args(&mut pat_args);
        let m = pat_args.len().saturating_sub(1);

        // Cost gate (Phase 6b). The m-delta-rule runs `m` passes (one per body
        // factor), each pass `O(delta)` outer x `O(pattern)` inner. So a firing
        // costs ~`m * delta_count` work, versus naive's single ~`dish_count` pass.
        // Route to the cheaper one: semi-naive only when `m * delta_count <
        // dish_count`. This is the standard semi-naive break-even: when the delta
        // is a large fraction of the dish (`delta ~ dish`, e.g. a rule's first
        // firing where delta == read_copy, or a dense m-factor body), the m passes
        // re-scan the dish m times and lose to naive; when `delta << dish` (a deep
        // recursive closure step) they win big. The threshold is the BARE predicate
        // with no fudge constant: a per-firing calibration on process_calculus
        // (m in {1,2}, delta_count in 1..8, dish growing to hundreds) showed it
        // keeps 1998/2001 firings on semi (the full ~117x win) and routes only the
        // 3 first-firings to naive, while routing 100% of the clique/finite_domain
        // firings to naive (delta == dish, m in 3..11). No regression on either end.
        //
        // The gate needs `delta_count` and `dish_count` but NOT the materialized
        // `delta` trie until it actually picks semi. Two regimes:
        //
        //  - FIRST firing (no snapshot): the delta is the whole dish, so
        //    `delta_count == dish_count` and the gate `m * dish < dish` is false for
        //    every m >= 1 -> ALWAYS naive. So skip the `subtract_cow`/`clone`
        //    entirely and decide from `dish_count = read_copy.val_count()` (one O(dish)
        //    walk, unavoidable to seed the snapshot count). This removes the
        //    first-firing delta materialization that otherwise cost ~3% on the
        //    single-firing finite_domain bench (clone + full subtract + count over
        //    10k facts) for a decision that is structurally pre-determined.
        //
        //  - LATER firing (has snapshot): the delta is small, so materialize it with
        //    the cheap `subtract_cow` (O(facts changed); `read_copy` and `snapshot`
        //    are COW clones sharing unchanged subtrees, which the subtract prunes
        //    instead of walking -- the Stage-5 lever, byte-identical to `subtract`).
        //    `dish_count = snapshot_count + delta_count - removed_count` is then EXACT
        //    and O(facts changed), avoiding `read_copy.val_count()` (O(dish), measured
        //    ~2.6x on process_calculus). `removed_count = (snapshot \ read_copy)` is
        //    the second cheap COW subtract; it is nonzero because the IC driver
        //    consumes (removes) exec facts between firings, so the dish is NOT add-only
        //    -- the subtraction makes the count exact anyway (verified 0 mismatches).
        let (delta, delta_count, dish_count) = match seen.get(&key) {
            None => {
                // First firing: gate is pre-determined to naive (m >= 1); don't build
                // the whole-dish delta. `delta` stays `None`.
                let dish_count = read_copy.val_count();
                (None, dish_count, dish_count)
            }
            Some(entry) => {
                let snapshot = &entry.snapshot;
                let delta = read_copy.subtract_cow(snapshot);
                let delta_count = delta.val_count();
                let removed_count = snapshot.subtract_cow(&read_copy).val_count();
                let dish_count = entry.dish_count + delta_count - removed_count;
                (Some(delta), delta_count, dish_count)
            }
        };

        // Run the m-delta-rule only when the gate clears AND the delta was
        // materialized (it never is on the pre-determined-naive first firing);
        // otherwise leave `result` as `None` so the caller falls through to the
        // naive full match. The snapshot refresh below happens in BOTH cases: after
        // a naive-gated firing the naive path still emits all consequences, so the
        // next delta only needs facts added after THIS firing. Storing the snapshot
        // even on a naive-gated firing is what lets the NEXT firing's delta be small
        // (without it, a rule first gated to naive would re-derive a whole-dish delta
        // forever and never reach the semi fast path).
        let result = match delta {
            Some(delta) if m.saturating_mul(delta_count) < dish_count => {
                unsafe {
                    SNI_DELTA_CALLS += 1;
                }
                self.sni_delta_calls += 1;
                Some(self.transform_multi_multi_delta(&read_copy, &delta, pat_expr, tpl_expr))
            }
            _ => None,
        };

        // Refresh this rule's snapshot to the input just matched, with its exact
        // dish count cached for the next firing's O(1) gate, and the rule's
        // pattern/template bytes (owned) so the DRed repair can re-derive this rule.
        let cur_gen = self.sni_removal_gen;
        seen.insert(
            key,
            SniRuleEntry {
                snapshot: read_copy,
                dish_count,
                caught_up_gen: cur_gen,
            },
        );
        self.sni_rule_seen = Some(seen);
        result
    }

    /// Semi-naive m-delta-rule form of `transform_multi_multi_` (Stage 2 of the
    /// semi-naive delta lever). Instead of matching the whole `,`-body against the
    /// full accumulating space, it matches each delta-rule: for each body factor `j`,
    /// factor `j` is opened on the per-step `delta` PathMap and the other factors on
    /// the full `read_copy`. The union over the delta-rules (emitted by idempotent
    /// btm insert, which dedups the overlap when a match has more than one delta
    /// factor) is exactly the new immediate-consequence facts: every old x ... x old
    /// combination is skipped because at least one factor is always restricted to the
    /// delta. A factor over a relation that did not change has no delta facts, so its
    /// pass simply finds nothing. See `kernel/resources/semi_naive_delta_design.md`.
    ///
    /// This reuses the same schematic matcher (`coreferential_transition` via the
    /// ProductZipper walk) and the same template emit as `transform_multi_multi_`,
    /// so the coreference / VarRef / issue-29 semantics are identical; the only new
    /// code is the per-factor zipper sourcing and the j-loop. Sources are passed in
    /// identity (body) order to the raw matcher, so the positional factor->zipper
    /// mapping holds without any planner reordering.
    ///
    /// NOT wired as a default (Stage 3); callable + differentially tested. Returns
    /// `(matched-candidate count summed over the delta-rules, any new path written)`.
    /// The candidate count is the per-delta-rule sum (it double-counts a match found
    /// by two delta-rules), so it is NOT comparable to the naive `touched`; the
    /// byte-identical written space is the oracle, not the count.
    ///
    /// Scope: add-only, exactly like `transform_multi_multi_` (it only writes
    /// template outputs, never retracts). `delta` is the per-step ADDED facts. For a
    /// consuming rule the removed reactants drive DRed-style retraction; that is
    /// Stage 3/4 and is not handled here. The differential oracle
    /// (`semi_naive_delta_matches_naive_on_process_calculus`) covers the monotone
    /// case, which is where the measured redundancy lives.
    pub fn transform_multi_multi_delta(
        &mut self,
        read_copy: &PathMap<()>,
        delta: &PathMap<()>,
        pat_expr: Expr,
        tpl_expr: Expr,
    ) -> (usize, bool) {
        let mut buffer = template_output_buffer();
        let mut tpl_args = Vec::with_capacity(64);
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<_> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
        let template_prefixes: Vec<_> = templates
            .iter()
            .map(|e| unsafe { e.prefix().unwrap_or_else(|x| x).as_ref().unwrap() })
            .collect();
        let mut subsumption = Self::prefix_subsumption(&template_prefixes[..]);
        let mut placements = subsumption.clone();

        // The body factors, in their natural (identity) order. Each is a
        // (sub)source the matcher descends; `sources[i]` lines up positionally with
        // the i-th read-zipper in every per-delta-rule ProductZipper below.
        let mut pat_args = Vec::with_capacity(64);
        ExprEnv::new(0, pat_expr).args(&mut pat_args);
        let sources: Vec<ExprEnv> = pat_args[1..].to_vec();
        let n_factors = sources.len();

        // Delta-rule indices: the standard m-delta-rule loops one pass per body
        // factor, restricting that factor to the per-step delta and the rest to the
        // full space, then unions. A factor over a relation that did NOT change has
        // no delta facts, so its pass simply finds nothing (harmless). Looping over
        // ALL factors is the correct, relation-agnostic form: it makes no assumption
        // about which relations the delta carries, so it stays sound when the delta
        // spans several relations (the real IC loop). A factor that can hit a delta
        // fact (its relation is in the delta) is the one that contributes new
        // tuples; the union over all j is exactly the post-step matches that use at
        // least one delta fact.
        let delta_rule_indices: Vec<usize> = (0..n_factors).collect();

        let zh = self.btm.zipper_head();
        let mut template_wzs: Vec<_> = Vec::with_capacity(64);
        template_prefixes.iter().enumerate().for_each(|(i, x)| {
            if subsumption[i] == i {
                placements[i] = template_wzs.len();
                template_wzs.push(unsafe { zh.write_zipper_at_exclusive_path_unchecked(x) });
            }
        });
        for i in 0..subsumption.len() {
            subsumption[i] = placements[subsumption[i]]
        }

        let mut assignments: Vec<(u8, u8)> = vec![];
        let mut trace: Vec<(u8, u8)> = vec![];
        let mut ass = Vec::with_capacity(64);
        let mut astack = Vec::with_capacity(64);

        let mut any_new = false;
        let mut total_candidates = 0usize;

        // The per-match emit, shared by every delta-rule pass (and the rare
        // renormalize-fallback pass). It instantiates the template under the match
        // `bindings` and writes the outputs, exactly as `transform_multi_multi_`'s
        // emit. `bindings` are keyed in the ORIGINAL pattern namespace (the matcher
        // unifies against `unify_sources`, which stay the original `sources`), so this
        // is order-independent of how the factors were reordered for the descent.
        // `pattern_intros` is threaded in per pass (the (oi,ni) seed for the cycle
        // check), not captured, so each pass resets it. Returns `true` to continue.
        let sni_trace = std::env::var("MORK_SNI_TRACE").is_ok();
        let mut emit = |refs_bindings: Result<&[u32], BTreeMap<(u8, u8), ExprEnv>>,
                        loc: Expr,
                        pattern_intros: &mut Option<(u8, u8)>|
         -> bool {
            'query: {
                if sni_trace {
                    match &refs_bindings {
                        Ok(refs) => eprintln!("SNI delta emit Ok(refs {refs:?})"),
                        Err(b) => eprintln!(
                            "SNI delta emit Err bindings: {:?}",
                            b.iter()
                                .map(|(k, ee)| (
                                    *k,
                                    serialize(unsafe { ee.subsexpr().span().as_ref().unwrap() })
                                ))
                                .collect::<Vec<_>>()
                        ),
                    }
                }
                trace!(target: "transform", "delta data {}", serialize(unsafe { loc.span().as_ref().unwrap()}));
                unsafe {
                    WRITES += template_prefixes.len();
                }
                match refs_bindings {
                    Ok(_) => {
                        // A single-factor body (n_factors == 1) cannot reach here via
                        // the raw matcher; the delta form is only used for multi-factor
                        // recursive bodies. Treat as a no-op to stay total.
                        true
                    }
                    Err(ref bindings) => {
                        #[cfg(debug_assertions)]
                        bindings.iter().for_each(
                            |(v, ee)| trace!(target: "transform", "delta binding {:?} {}", *v, ee.show()),
                        );

                        // Mirror the naive emit's seed discipline exactly: the cached
                        // (oi, ni) is only valid for GROUND bindings (a schematic match
                        // introduces data variables, so its intro numbering differs; a
                        // stale ground seed renumbers a data NewVar into the pattern
                        // namespace and materializes another binding's term -- the
                        // process_calculus IC divergence). And a cycle decline skips
                        // this match, it must not stop the pass.
                        let ground_bindings =
                            bindings.values().all(|ee| ee.subsexpr().is_ground());
                        let (oi0, ni) = match pattern_intros.filter(|_| ground_bindings) {
                            Some(cached) => cached,
                            None => {
                                let mut void = std::io::sink();
                                let (oi, ni, ok) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
                                    0, 0, 0, pat_expr, bindings, void, trace, assignments
                                );
                                if !ok {
                                    break 'query true;
                                }
                                if ground_bindings && ni == 0 {
                                    *pattern_intros = Some((oi, ni));
                                }
                                (oi, ni)
                            }
                        };
                        let mut oi = oi0;

                        'writes: for (i, template) in templates.iter().enumerate() {
                            let wz = &mut template_wzs[subsumption[i]];

                            trace!(target: "transform", "{i} delta template {} @ ({oi} {ni})", serialize(unsafe { template.span().as_ref().unwrap()}));

                            buffer.clear();
                            let (toi, _, true) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
                                0, oi, ni, *template, bindings, buffer, astack, ass
                            ) else {
                                continue 'writes;
                            };
                            oi = toi;

                            if sni_trace {
                                eprintln!(
                                    "EMITdelta pat={} out={}",
                                    serialize(unsafe { pat_expr.span().as_ref().unwrap() }),
                                    serialize(&buffer)
                                );
                            }
                            trace!(target: "transform", "U {i} delta out {:?}", Expr{ ptr: buffer.as_mut_ptr() });
                            wz.move_to_path(&buffer[wz.root_prefix_path().len()..]);
                            any_new |= wz.set_val(()).is_none();
                        }
                        true
                    }
                }
            }
        };

        // One delta-rule per mutated-relation factor j. Stage 4 reorder: factor j
        // (restricted to the small `delta`) is the ProductZipper PRIMARY (the outer
        // loop), and the other factors (on the full `read_copy`) are SECONDARIES that
        // SEEK the bound shared variables via the existing VarRef byte-recheck rather
        // than being descended whole. So a pass is O(delta) outer x O(pattern) inner
        // seek instead of O(read_copy) outer x O(delta) inner -- the residual
        // full-dish descent that kept the scaling exponent at ~1.77 is gone.
        //
        // The reorder is NOT a bare read-zipper swap. The body factors' patterns are
        // renormalized in identity order: a variable's first occurrence is `NewVar`,
        // later occurrences are `VarRef(slot)` pointing back. Putting factor j first
        // without renumbering would leave its VarRefs pointing at variables not yet
        // introduced, breaking coreference. So per pass we build the plan
        // `[j, then the rest in identity order]` and RE-RENORMALIZE the sources in
        // that order (`search_sources`): factor j's variables become the leading
        // NewVars and the other factors' shared variables become VarRefs back to them.
        // `search_sources` drives only the descent/coreference structure.
        //
        // The binding keys must stay in the ORIGINAL pattern namespace so the emit
        // (`apply_e` over `pat_expr`) is unchanged. `args` numbers every factor in one
        // shared `.n=0` De Bruijn space with the cumulative NewVar offset as identity,
        // and each `sources[i]` keeps that original `.v`/VarRef numbering. So
        // `unify_sources` is the ORIGINAL `sources` reordered by the plan (never
        // renormalized): the `unify` binding keys are exactly the original ones, only
        // the pattern-factor <-> matched-location PAIRING changes. This mirrors
        // `query_multi`'s split of `planned_sources` (descent) vs `planned_unify_sources`
        // (binding keys). `effect_source_index = 0` because the emit ignores `loc`.
        //
        // The `pattern_intros` cache is per-pass (the seed (oi,ni) depends only on the
        // pattern, which is the same across passes, but resetting per pass keeps the
        // ni!=0 decline local and is cheap).
        for &j in &delta_rule_indices {
            // Plan: factor j first, the remaining factors in identity order.
            let mut plan_j: Vec<usize> = Vec::with_capacity(n_factors);
            plan_j.push(j);
            plan_j.extend((0..n_factors).filter(|&i| i != j));

            // Renormalize the sources in plan order so factor j introduces the leading
            // NewVars and later factors' shared variables become VarRefs back to them.
            // If the variable offsets don't re-encode (>= 64 distinct), fall back to
            // the identity (un-reordered) form for this pass, which is still correct
            // (the pre-Stage-4 behaviour) -- never a wrong match, only a slower pass.
            let Some((_search_buffers, search_sources)) =
                Self::renormalize_query_factors(&sources, &plan_j)
            else {
                let primary_map: &PathMap<()> = if j == 0 { delta } else { read_copy };
                let mut prz = ProductZipper::new(
                    primary_map.read_zipper(),
                    (1..n_factors).map(|i| {
                        if i == j { delta } else { read_copy }.read_zipper()
                    }),
                );
                reserve_query_product_buffers(&mut prz);
                let mut pattern_intros: Option<(u8, u8)> = None;
                let candidates =
                    Self::query_multi_raw(&mut prz, &sources, |refs_bindings, loc| {
                        emit(refs_bindings, loc, &mut pattern_intros)
                    });
                total_candidates += candidates;
                continue;
            };

            // The ORIGINAL sources, reordered by the plan, for the binding-key
            // namespace. Position 0 is factor j (unified against the primary = a delta
            // fact); positions 1.. are the other factors (unified against the
            // secondaries = read_copy facts, seeking the shared bindings).
            let unify_sources: Vec<ExprEnv> = plan_j.iter().map(|&i| sources[i]).collect();

            // SAFETY mirrors `query_multi`: all read-zippers are opened at root and the
            // matcher descends each factor's prefix from the source stack. The primary
            // is factor j on `delta` (the small outer loop); the secondaries are the
            // other factors on `read_copy`, in plan order. `delta`/`read_copy` are
            // separate maps from `self.btm`, so the write zippers (from `zh`) never
            // alias the read zippers.
            let mut prz = ProductZipper::new(
                delta.read_zipper(),
                (0..n_factors - 1).map(|_| read_copy.read_zipper()),
            );
            reserve_query_product_buffers(&mut prz);

            let mut pattern_intros: Option<(u8, u8)> = None;
            let candidates = Self::query_multi_raw_with_unification_sources(
                &mut prz,
                &search_sources,
                &unify_sources,
                0,
                None,
                |refs_bindings, loc| emit(refs_bindings, loc, &mut pattern_intros),
            );
            total_candidates += candidates;
        }

        drop(emit);
        for wz in template_wzs {
            zh.cleanup_write_zipper(wz);
        }
        (total_candidates, any_new)
    }

    #[cfg(feature = "specialize_io")]
    pub fn transform_multi_multi_i(
        &mut self,
        pat_expr: Expr,
        tpl_expr: Expr,
        add: Expr,
    ) -> (usize, bool) {
        let mut buffer = template_output_buffer();
        let mut tpl_args = Vec::with_capacity(64);
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<_> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
        let template_prefixes: Vec<_> = templates
            .iter()
            .map(|e| unsafe { e.prefix().unwrap_or_else(|x| x).as_ref().unwrap() })
            .collect();
        let mut subsumption = Self::prefix_subsumption(&template_prefixes[..]);
        let mut placements = subsumption.clone();
        let mut read_copy = self.btm.clone();
        let zh = self.btm.zipper_head();
        read_copy.insert(unsafe { add.span().as_ref().unwrap() }, ());
        let mut template_wzs: Vec<_> = Vec::with_capacity(64);
        template_prefixes.iter().enumerate().for_each(|(i, x)| {
            if subsumption[i] == i {
                placements[i] = template_wzs.len();
                template_wzs.push(unsafe { zh.write_zipper_at_exclusive_path_unchecked(x) });
            }
        });
        for i in 0..subsumption.len() {
            subsumption[i] = placements[subsumption[i]]
        }
        debug!(
            target: "transform",
            "write_resource_placement mode=multi_multi_i requests={} exclusive_writers={} reused_writers={}",
            template_prefixes.len(),
            template_wzs.len(),
            template_prefixes.len().saturating_sub(template_wzs.len())
        );
        trace!(target: "transform", "templates {:?}", templates);
        trace!(target: "transform", "prefixes {:?}", template_prefixes);
        trace!(target: "transform", "subsumption {:?}", subsumption);

        let mut assignments: Vec<(u8, u8)> = vec![];
        let mut trace: Vec<(u8, u8)> = vec![];

        let mut ass = Vec::with_capacity(64);
        let mut astack = Vec::with_capacity(64);

        let mut any_new = false;
        let touched = Self::query_multi_i(
            false,
            &mut self.mmaps,
            &mut self.z3s,
            &read_copy,
            pat_expr,
            |refs_bindings, _loc| 'query: {
                // trace!(target: "transform", "data {}", serialize(unsafe { loc.span().as_ref().unwrap()}));
                unsafe {
                    WRITES += template_prefixes.len();
                }
                match refs_bindings {
                    Ok(_) => {
                        unreachable!()
                    }
                    Err(ref bindings) => {
                        #[cfg(debug_assertions)]
                        bindings.iter().for_each(
                            |(v, ee)| trace!(target: "transform", "binding {:?} {}", *v, ee.show()),
                        );

                        let (mut oi, ni, true) = ({
                            let mut void = std::io::sink();
                            mork_expr::apply_e_clears_stacks_and_cycles_check!(
                                0,
                                0,
                                0,
                                pat_expr,
                                bindings,
                                void,
                                trace,
                                assignments
                            )
                        }) else {
                            break 'query false;
                        };

                        'writes: for (i, template) in templates.iter().enumerate() {
                            let wz = &mut template_wzs[subsumption[i]];

                            trace!(target: "transform", "{i} template {} @ ({oi} {ni})", serialize(unsafe { template.span().as_ref().unwrap()}));

                            buffer.clear();
                            let (toi, _, true) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
                                0, oi, ni, *template, bindings, buffer, astack, ass
                            ) else {
                                continue 'writes;
                            };
                            oi = toi;

                            trace!(target: "transform", "U {i} out {:?}", Expr{ ptr: buffer.as_mut_ptr() });
                            wz.move_to_path(&buffer[wz.root_prefix_path().len()..]);
                            any_new |= wz.set_val(()).is_none();
                        }
                        true
                    }
                }
            },
        );
        for wz in template_wzs {
            zh.cleanup_write_zipper(wz);
        }
        (touched, any_new)
    }

    #[cfg(feature = "specialize_io")]
    pub fn transform_multi_multi_o(
        &mut self,
        pat_expr: Expr,
        tpl_expr: Expr,
        add: Expr,
    ) -> (usize, bool) {
        use crate::sinks::*;
        let mut buffer = template_output_buffer();
        let mut tpl_args = Vec::with_capacity(64);
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<_> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
        let mut sinks: Vec<_> = templates.iter().map(|e| ASink::new(*e)).collect();
        let template_prefixes: Vec<_> = sinks
            .iter()
            .map(|sink| sink.request().next().unwrap())
            .collect();
        let mut subsumption = Self::prefix_subsumption_resources(&template_prefixes[..]);
        let mut placements = subsumption.clone();
        let mut read_copy = self.btm.clone();
        let zh = self.btm.zipper_head();
        let zh_ptr = ((&zh) as *const ZipperHead<()>).cast_mut();
        read_copy.insert(unsafe { add.span().as_ref().unwrap() }, ());
        let mut template_resources: Vec<_> = Vec::with_capacity(64);
        let mut outstanding_wzs = Vec::with_capacity(64);
        let outstanding_wzs_ptr =
            ((&outstanding_wzs) as *const Vec<WriteZipperTracked<()>>).cast_mut();
        let acts_ptr = ((&self.mmaps) as *const HashMap<OwnedSourceItem, _>).cast_mut();
        let z3s_ptr = ((&self.z3s) as *const HashMap<OwnedSourceItem, Box<Popen>>).cast_mut();
        template_prefixes
            .iter()
            .enumerate()
            .for_each(|(i, request)| {
                if subsumption[i] == i {
                    placements[i] = template_resources.len();
                    template_resources.push(unsafe {
                        Self::write_handler(
                            (zh_ptr, outstanding_wzs_ptr),
                            acts_ptr,
                            z3s_ptr,
                            request,
                        )
                    });
                }
            });
        for i in 0..subsumption.len() {
            subsumption[i] = placements[subsumption[i]]
        }
        debug!(
            target: "transform",
            "write_resource_placement mode=multi_multi_o requests={} exclusive_writers={} reused_writers={}",
            template_prefixes.len(),
            template_resources.len(),
            template_prefixes.len().saturating_sub(template_resources.len())
        );
        trace!(target: "transform", "templates {:?}", templates);
        trace!(target: "transform", "prefixes {:?}", template_prefixes);
        trace!(target: "transform", "subsumption {:?}", subsumption);

        let mut assignments: Vec<(u8, u8)> = vec![];
        let mut trace: Vec<(u8, u8)> = vec![];

        let mut ass = Vec::with_capacity(64);
        let mut astack = Vec::with_capacity(64);

        let mut any_new = false;
        #[cfg(feature = "sidecar_bridge")]
        Self::bridge_self_check(&read_copy, pat_expr);
        // The pattern apply only computes the template intro seed (oi, ni); it
        // writes nothing (its sink is `void`). For ground bindings (oi structural,
        // ni == 0) the seed is invariant across matches, so compute it once and
        // reuse it, skipping the per-match pattern re-walk. A non-ground match
        // (ni != 0) recomputes, preserving the cycle (`!ok`) decline.
        let mut pattern_intros: Option<(u8, u8)> = None;
        let touched = Self::query_multi(&read_copy, pat_expr, |refs_bindings, loc| 'query: {
            trace!(target: "transform", "data {}", serialize(unsafe { loc.span().as_ref().unwrap()}));
            unsafe {
                WRITES += template_prefixes.len();
            }
            match refs_bindings {
                Ok(_) => {
                    unreachable!()
                }
                Err(ref bindings) => {
                    #[cfg(debug_assertions)]
                    bindings.iter().for_each(
                        |(v, ee)| trace!(target: "transform", "binding {:?} {}", *v, ee.show()),
                    );

                    let (oi0, ni) = match pattern_intros {
                        Some(cached) => cached,
                        None => {
                            let mut void = std::io::sink();
                            let (oi, ni, ok) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
                                0,
                                0,
                                0,
                                pat_expr,
                                bindings,
                                void,
                                trace,
                                assignments
                            );
                            if !ok {
                                break 'query false;
                            }
                            if ni == 0 {
                                pattern_intros = Some((oi, ni));
                            }
                            (oi, ni)
                        }
                    };
                    let mut oi = oi0;

                    'writes: for (i, template) in templates.iter().enumerate() {
                        let wz = unsafe { std::ptr::read(&template_resources[subsumption[i]]) };

                        trace!(target: "transform", "{i} template {} @ ({oi} {ni})", serialize(unsafe { template.span().as_ref().unwrap()}));

                        buffer.clear();
                        let (toi, _, true) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
                            0, oi, ni, *template, bindings, buffer, astack, ass
                        ) else {
                            continue 'writes;
                        };
                        oi = toi;

                        trace!(target: "transform", "U {i} out {:?}", Expr{ ptr: buffer.as_mut_ptr() });
                        sinks[i].sink(std::iter::once(wz), &buffer[..]);
                    }
                    true
                }
            }
        });

        for (i, s) in sinks.iter_mut().enumerate() {
            let wz = unsafe { std::ptr::read(&template_resources[subsumption[i]]) };
            any_new |= s.finalize(std::iter::once(wz));
        }
        for wz in outstanding_wzs.iter_mut() {
            zh.cleanup_write_zipper(wz);
        }

        // A `(- ...)` template deleted facts: bump the per-Space removal
        // generation so the persistent join sidecar re-syncs and tombstones them
        // (closes the count-equality staleness window).
        if sinks.iter().any(|s| s.is_remove()) {
            #[cfg(feature = "sidecar_bridge_emit")]
            {
                self.bridge_remove_gen = self.bridge_remove_gen.wrapping_add(1);
            }
            // Trip the semi-naive soundness gate: a retraction makes every IC
            // worker's add-only snapshot potentially stale (a removed fact is
            // excluded from `btm \ snapshot`, so the closure never re-derives it),
            // so disarm the delta and fall back to naive for the rest of the loop
            // (Naive mode), or trigger a DRed re-derivation catch-up (Dred mode).
            // The generation counter bumps so every recursive rule re-derives once
            // before its incremental delta is trusted again.
            self.sni_removal_seen = true;
            #[cfg(feature = "semi_naive_ic")]
            {
                self.sni_removal_gen = self.sni_removal_gen.wrapping_add(1);
            }
        }
        (touched, any_new)
    }

    pub fn transform_multi_multi_io(
        &mut self,
        pat_expr: Expr,
        tpl_expr: Expr,
        add: Expr,
        no_source: bool,
        no_sink: bool,
    ) -> (usize, bool) {
        use crate::sinks::*;
        let mut buffer = template_output_buffer();
        let mut tpl_args = Vec::with_capacity(64);
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<_> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
        let mut sinks: Vec<_> = templates
            .iter()
            .map(|e| {
                if no_sink {
                    ASink::compat(*e)
                } else {
                    ASink::new(*e)
                }
            })
            .collect();
        let template_prefixes: Vec<_> = sinks
            .iter()
            .map(|sink| sink.request().next().unwrap())
            .collect();
        let mut subsumption = Self::prefix_subsumption_resources(&template_prefixes[..]);
        let mut placements = subsumption.clone();
        let mut read_copy = self.btm.clone();
        let zh = self.btm.zipper_head();
        let zh_ptr = ((&zh) as *const ZipperHead<()>).cast_mut();
        read_copy.insert(unsafe { add.span().as_ref().unwrap() }, ());
        let mut template_resources: Vec<_> = Vec::with_capacity(64);
        let mut outstanding_wzs = Vec::with_capacity(64);
        let outstanding_wzs_ptr =
            ((&outstanding_wzs) as *const Vec<WriteZipperTracked<()>>).cast_mut();
        let acts_ptr = ((&self.mmaps) as *const HashMap<OwnedSourceItem, _>).cast_mut();
        let z3s_ptr = ((&self.z3s) as *const HashMap<OwnedSourceItem, Box<Popen>>).cast_mut();
        template_prefixes
            .iter()
            .enumerate()
            .for_each(|(i, request)| {
                if subsumption[i] == i {
                    placements[i] = template_resources.len();
                    template_resources.push(unsafe {
                        Self::write_handler(
                            (zh_ptr, outstanding_wzs_ptr),
                            acts_ptr,
                            z3s_ptr,
                            request,
                        )
                    });
                }
            });
        for i in 0..subsumption.len() {
            subsumption[i] = placements[subsumption[i]]
        }
        debug!(
            target: "transform",
            "write_resource_placement mode=multi_multi_io requests={} exclusive_writers={} reused_writers={}",
            template_prefixes.len(),
            template_resources.len(),
            template_prefixes.len().saturating_sub(template_resources.len())
        );
        trace!(target: "transform", "templates {:?}", templates);
        trace!(target: "transform", "prefixes {:?}", template_prefixes);
        trace!(target: "transform", "subsumption {:?}", subsumption);

        let mut assignments: Vec<(u8, u8)> = vec![];
        let mut trace: Vec<(u8, u8)> = vec![];

        let mut ass = Vec::with_capacity(64);
        let mut astack = Vec::with_capacity(64);

        let mut any_new = false;
        let touched = Self::query_multi_i(
            no_source,
            &mut self.mmaps,
            &mut self.z3s,
            &read_copy,
            pat_expr,
            |refs_bindings, loc| 'query: {
                trace!(target: "transform", "data {}", serialize(unsafe { loc.span().as_ref().unwrap()}));
                unsafe {
                    WRITES += template_prefixes.len();
                }
                match refs_bindings {
                    Ok(_) => {
                        unreachable!()
                    }
                    Err(ref bindings) => {
                        #[cfg(debug_assertions)]
                        bindings.iter().for_each(
                            |(v, ee)| trace!(target: "transform", "binding {:?} {}", *v, ee.show()),
                        );

                        let (oi, ni, true) = ({
                            let mut void = std::io::sink();
                            mork_expr::apply_e_clears_stacks_and_cycles_check!(
                                0,
                                0,
                                0,
                                pat_expr,
                                bindings,
                                void,
                                trace,
                                assignments
                            )
                        }) else {
                            break 'query false;
                        };

                        'writes: for (i, template) in templates.iter().enumerate() {
                            let wz = unsafe { std::ptr::read(&template_resources[subsumption[i]]) };

                            trace!(target: "transform", "{i} template {} @ ({oi} {ni})", serialize(unsafe { template.span().as_ref().unwrap()}));

                            buffer.clear();
                            let (_toi, _, true) = mork_expr::apply_e_clears_stacks_and_cycles_check!(
                                0, oi, ni, *template, bindings, buffer, astack, ass
                            ) else {
                                continue 'writes;
                            };

                            trace!(target: "transform", "U {i} out {:?}", Expr{ ptr: buffer.as_mut_ptr() });
                            sinks[i].sink(std::iter::once(wz), &buffer[..]);
                        }
                        true
                    }
                }
            },
        );

        for (i, s) in sinks.iter_mut().enumerate() {
            let wz = unsafe { std::ptr::read(&template_resources[subsumption[i]]) };
            any_new |= s.finalize(std::iter::once(wz));
        }
        for wz in outstanding_wzs.iter_mut() {
            zh.cleanup_write_zipper(wz);
        }

        // A `(- ...)` template deleted facts: bump the per-Space removal
        // generation so the persistent join sidecar re-syncs and tombstones them
        // (closes the count-equality staleness window).
        if sinks.iter().any(|s| s.is_remove()) {
            #[cfg(feature = "sidecar_bridge_emit")]
            {
                self.bridge_remove_gen = self.bridge_remove_gen.wrapping_add(1);
            }
            // Trip the semi-naive soundness gate: a retraction makes every IC
            // worker's add-only snapshot potentially stale (a removed fact is
            // excluded from `btm \ snapshot`, so the closure never re-derives it),
            // so disarm the delta and fall back to naive for the rest of the loop
            // (Naive mode), or trigger a DRed re-derivation catch-up (Dred mode).
            // The generation counter bumps so every recursive rule re-derives once
            // before its incremental delta is trusted again.
            self.sni_removal_seen = true;
            #[cfg(feature = "semi_naive_ic")]
            {
                self.sni_removal_gen = self.sni_removal_gen.wrapping_add(1);
            }
        }
        (touched, any_new)
    }

    // (exec <loc> (, <src1> <src2> <srcn>)
    //             (, <dst1> <dst2> <dstm>))
    pub fn interpret(&mut self, rt: Expr) -> Result<(), &'static str> {
        #[cfg(feature = "periodic_merkleize")]
        if self.last_merkleize.elapsed().as_secs() > 10 {
            self.btm.merkleize();
            self.last_merkleize = Instant::now()
        }
        debug!(target: "interpret", "interpreting {:?}", serialize(unsafe { rt.span().as_ref().unwrap() }));
        #[cfg(debug_assertions)]
        {
            let mut rz = self.btm.read_zipper();
            while rz.to_next_val() {
                trace!(target: "interpret", "on space {:?}", serialize(rz.path()));
            }
            drop(rz);
        }
        destruct!(rt, ("exec" loc pat_expr tpl_expr), unsafe {
            debug_assert!(loc.variables() == 0);
            if let Tag::Arity(i) = byte_item(*pat_expr.ptr) { if i == 0 { return Err("pattern expression can not be empty"); } } else { return Err("pattern must be an expression, not a symbol or variables") }
            if *pat_expr.ptr.add(1) != item_byte(Tag::SymbolSize(1)) { return Err("pattern functor can only be , or I") }

            if let Tag::Arity(i) = byte_item(*tpl_expr.ptr) { if i == 0 { return Err("template expression can not be empty"); } } else { return Err("template must be an expression, not a symbol or variables") }
            if *tpl_expr.ptr.add(1) != item_byte(Tag::SymbolSize(1)) { return Err("template functor can only be , or O") }

            #[cfg(feature="specialize_io")]
            let res = match (*pat_expr.ptr.add(2), *tpl_expr.ptr.add(2)) {
                (b',', b',') => { self.transform_multi_multi_(pat_expr, tpl_expr, rt) }
                (b'I', b',') => { self.transform_multi_multi_i(pat_expr, tpl_expr, rt) }
                (b',', b'O') => { self.transform_multi_multi_o(pat_expr, tpl_expr, rt) }
                (b'I', b'O') => { self.transform_multi_multi_io(pat_expr, tpl_expr, rt, false, false) }
                (_, _) => { return Err("pattern functor can only be , or I and template functor can only be , or O") }
            };
            #[cfg(not(feature="specialize_io"))]
            let res = match (*pat_expr.ptr.add(2), *tpl_expr.ptr.add(2)) {
                (b',', b',') => { self.transform_multi_multi_io(pat_expr, tpl_expr, rt, true, true) }
                (b'I', b',') => { self.transform_multi_multi_io(pat_expr, tpl_expr, rt, false, true) }
                (b',', b'O') => { self.transform_multi_multi_io(pat_expr, tpl_expr, rt, true, false) }
                (b'I', b'O') => { self.transform_multi_multi_io(pat_expr, tpl_expr, rt, false, false) }
                (_, _) => { return Err("pattern functor can only be , or I and template functor can only be , or O") }
            };

            trace!(target: "interpret", "(run, changed) = {:?}", res);
            return Ok(())
        }, _err => return Err("exec shape (exec <loc> <patterns> <templates>)"))
    }

    fn exec_prefix() -> [u8; 6] {
        [
            item_byte(Tag::Arity(4)),
            item_byte(Tag::SymbolSize(4)),
            b'e',
            b'x',
            b'e',
            b'c',
        ]
    }

    fn take_first_exec_path(&mut self, path_buffer: &mut Vec<u8>) -> bool {
        let prefix = Self::exec_prefix();
        path_buffer.clear();

        let zh = self.btm.zipper_head();
        let Ok(mut rz) = zh.read_zipper_at_borrowed_path(&prefix) else {
            return false;
        };
        if !rz.to_next_val() {
            return false;
        }

        path_buffer.extend_from_slice(rz.origin_path());
        drop(rz);

        let Ok(mut wz) = zh.write_zipper_at_exclusive_path(&[]) else {
            path_buffer.clear();
            return false;
        };
        wz.descend_to(&path_buffer[..]);
        let removed = wz.remove_val(true).is_some();
        if !removed {
            path_buffer.clear();
        }
        removed
    }

    pub fn metta_calculus(&mut self, steps: usize) -> usize {
        let mut done: usize = 0;
        let mut exec_path = Vec::new();

        // Semi-naive immediate-consequence loop (Stage 3, feature `semi_naive_ic`).
        // The naive loop re-matches each exec's `,`-rule against the whole
        // accumulating dish every round; the semi-naive loop matches only the delta
        // for the rule that fires (the facts added since that rule last ran), so a
        // step costs O(delta) instead of O(dish). The per-rule deltas live in
        // `sni_rule_seen`, which `transform_multi_multi_` consults and refreshes;
        // here we just arm it for the duration of the loop. A single global delta
        // would be wrong: rules re-arm and fire out of lockstep, so each needs its
        // own "since I last ran" frontier (see the field docs). The default build
        // never arms it, so the loop is byte-identical to the naive one. Restored to
        // `None` on exit so nested / later `transform_multi_multi_` calls stay naive.
        #[cfg(feature = "semi_naive_ic")]
        let sni_outer = self.sni_rule_seen.take();
        #[cfg(feature = "semi_naive_ic")]
        {
            self.sni_rule_seen = Some(HashMap::new());
            self.sni_force_naive = self.sni_force_naive
                || SNI_DISARM.with(|c| c.get())
                || std::env::var("MORK_SNI").as_deref() == Ok("0");
            // Arm the soundness gate fresh: only removals DURING this loop should
            // disarm the delta. Cleared here so a pre-loop load (which trips the
            // flag via invalidate_bridge_caches) does not force naive from the
            // start. Once a removal happens mid-loop the flag latches and every
            // later `,`->`,` rule routes to naive (the corpus proved retraction is
            // the exact divergence boundary).
            self.sni_removal_seen = false;
            self.sni_removal_gen = 0;
            self.sni_dred_repairs = 0;
            self.sni_dred_fallbacks = 0;
        }

        while done < steps {
            if self.take_first_exec_path(&mut exec_path) {
                let xe = Expr {
                    ptr: exec_path.as_mut_ptr(),
                };
                let start = Instant::now();
                if let Err(e) = self.interpret(xe) {
                    debug!(target: "interpret", "not interpreting: {}", e);
                }
                if self.timing {
                    let start_string = start.elapsed().as_nanos().to_string();
                    let start_str = start_string.as_str();
                    let done_string = done.to_string();
                    let done_str = done_string.as_str();
                    let buf = mork_expr::construct!("timing" xe done_str start_str).unwrap();
                    self.btm.insert(&buf[..], ());
                    trace!(target: "interpret", "interpret took {} ns", start_str);
                }
                done += 1;
            } else {
                break;
            }
        }

        // Disarm the per-rule deltas so any later transform stays on the naive path.
        #[cfg(feature = "semi_naive_ic")]
        {
            self.sni_rule_seen = sni_outer;
        }

        done
    }

    pub fn token_bfs(&self, token: &[u8], pattern: Expr) -> Vec<(Vec<u8>, Expr)> {
        // let mut stack = vec![0; 1];
        // stack[0] = ACTION;
        //
        // let prefix = unsafe { pattern.prefix().unwrap_or_else(|x| pattern.span()).as_ref().unwrap() };
        // let shared = pathmap::utils::find_prefix_overlap(&token[..], prefix);
        // stack.extend_from_slice(&referential_bidirectional_matching_stack_traverse(pattern, prefix.len())[..]);
        // // println!("show {}", show_stack(&stack[..]));
        // stack.reserve(4096);

        let mut rz = self.btm.read_zipper_at_path(&token[..]);
        rz.reserve_buffers(4096, 64);

        rz.descend_until();

        let cm = rz.child_mask();
        let mut it = cm.iter();

        let mut res = vec![];

        let mut stack: Vec<(u8, u8)> = Vec::new();
        let mut assignments: Vec<(u8, u8)> = Vec::new();
        let mut expr_env: Vec<(ExprEnv, ExprEnv)> = Vec::new();
        while let Some(b) = it.next() {
            rz.descend_to_byte(b);

            let mut rzc = rz.clone();
            rzc.to_next_val();
            let e = Expr {
                ptr: rzc.origin_path().to_vec().leak().as_ptr().cast_mut(),
            };
            if mork_expr::unifiable_reuse_state(
                e,
                pattern,
                &mut expr_env,
                &mut stack,
                &mut assignments,
            ) {
                let v = rz.origin_path().to_vec();
                // println!("token {:?}", &v[..]);
                // println!("expr  {:?}", e);
                res.push((v, e));
            }
            rz.ascend_byte();
        }

        res
    }

    pub fn done(self) -> ! {
        // let counters = pathmap::counters::Counters::count_ocupancy(&self.btm);
        // counters.print_histogram_by_depth();
        // counters.print_run_length_histogram();
        // counters.print_list_node_stats();
        // println!("#symbols {}", self.sm.symbol_count());
        process::exit(0);
    }
}

impl Drop for Space {
    fn drop(&mut self) {
        for (_, z3) in self.z3s.iter_mut() {
            // z3.terminate();
            drop(z3.stdin.take())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the naive IC loop for a test of another route (RAII-restored, panic-safe).
    #[cfg(feature = "semi_naive_ic")]
    struct SniDisarmGuard(bool);
    #[cfg(feature = "semi_naive_ic")]
    impl SniDisarmGuard {
        fn new() -> Self {
            Self(SNI_DISARM.with(|c| c.replace(true)))
        }
    }
    #[cfg(feature = "semi_naive_ic")]
    impl Drop for SniDisarmGuard {
        fn drop(&mut self) {
            let prev = self.0;
            SNI_DISARM.with(|c| c.set(prev));
        }
    }

    #[test]
    fn shard_zipper_sweep_preserves_space_content() {
        let mut space = Space::new();
        space
            .add_all_sexpr(b"(edge a b)\n(edge b c)\n(edge c d)\n(rel x y)\n")
            .unwrap();
        let before = space.btm.val_count();
        assert!(before >= 4);

        // The decomposition is a covering antichain: shard costs sum to the total.
        let shards = space.decompose_shards(2);
        let covered: usize = shards.iter().map(|p| space.shard_cost(p)).sum();
        assert_eq!(covered, before);

        // An identity sweep over one shard leaves the space content unchanged.
        let prefix = shards[0].clone();
        space.sweep_shard(&prefix, |_shard| crate::shard_zipper::PatchLog::new());
        assert_eq!(space.btm.val_count(), before);

        // The space still functions after the derived caches were invalidated.
        space.add_all_sexpr(b"(edge d e)\n").unwrap();
        assert_eq!(space.btm.val_count(), before + 1);
    }

    #[cfg(feature = "einsum")]
    #[test]
    fn relation_adjacency_two_hop_on_real_space() {
        fn encoded_symbol(s: &[u8]) -> Vec<u8> {
            let mut v = vec![mork_expr::item_byte(mork_expr::Tag::SymbolSize(
                s.len() as u8
            ))];
            v.extend_from_slice(s);
            v
        }

        let mut space = Space::new();
        space
            .add_all_sexpr(b"(edge a b)\n(edge b c)\n(edge c d)\n")
            .unwrap();

        let adj = space
            .relation_adjacency(&encoded_symbol(b"edge"))
            .expect("edge relation present");
        assert_eq!(adj.node_count(), 4); // a, b, c, d

        // The SpGEMM A.A over the real space's edges: a -> c and b -> d.
        let pairs = adj.enumerate(&adj.two_hop());
        assert_eq!(pairs.len(), 2);
        assert!(
            pairs
                .iter()
                .any(|(s, d, _)| *s == encoded_symbol(b"a") && *d == encoded_symbol(b"c"))
        );
        assert!(
            pairs
                .iter()
                .any(|(s, d, _)| *s == encoded_symbol(b"b") && *d == encoded_symbol(b"d"))
        );
    }

    #[cfg(feature = "einsum")]
    #[test]
    fn write_two_hop_counts_closes_the_loop() {
        let mut space = Space::new();
        space
            .add_all_sexpr(b"(edge a b)\n(edge b c)\n(edge c d)\n")
            .unwrap();
        let edge_head = crate::graph_tensor::encode_symbol(b"edge");

        // Numeric SpGEMM, then write the 2-hop pairs back as facts.
        let written = space.write_two_hop_counts(&edge_head, b"twohop");
        assert_eq!(written, 2);

        // The derived facts (twohop a c 1) and (twohop b d 1) are now queryable.
        let one = crate::graph_tensor::encode_symbol(b"1");
        let fact_ac = crate::graph_tensor::encode_fact(
            b"twohop",
            &[
                &crate::graph_tensor::encode_symbol(b"a"),
                &crate::graph_tensor::encode_symbol(b"c"),
                &one,
            ],
        );
        let fact_bd = crate::graph_tensor::encode_fact(
            b"twohop",
            &[
                &crate::graph_tensor::encode_symbol(b"b"),
                &crate::graph_tensor::encode_symbol(b"d"),
                &one,
            ],
        );
        assert!(space.btm.contains(&fact_ac), "missing (twohop a c 1)");
        assert!(space.btm.contains(&fact_bd), "missing (twohop b d 1)");

        // Writing again is idempotent (the facts already exist).
        assert_eq!(space.write_two_hop_counts(&edge_head, b"twohop"), 0);
    }

    #[cfg(feature = "einsum")]
    #[test]
    fn write_pagerank_closes_the_loop() {
        let mut space = Space::new();
        space.add_all_sexpr(b"(edge a c)\n(edge b c)\n").unwrap();
        let edges_count = space.btm.val_count();
        let edge_head = crate::graph_tensor::encode_symbol(b"edge");

        // Three nodes a, b, c, so three (pr node score) facts.
        let written = space.write_pagerank(&edge_head, b"pr", 0.85, 100);
        assert_eq!(written, 3);
        assert_eq!(space.btm.val_count(), edges_count + 3);

        // Deterministic scores, so a second run writes nothing new.
        assert_eq!(space.write_pagerank(&edge_head, b"pr", 0.85, 100), 0);
        assert_eq!(space.btm.val_count(), edges_count + 3);
    }

    fn query_pattern_and_sources(space: &mut Space, pattern: &'static str) -> (Expr, Vec<ExprEnv>) {
        let pat_expr = crate::expr!(space, pattern);
        let mut args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut args);
        (pat_expr, args[1..].to_vec())
    }

    fn query_sources(space: &mut Space, pattern: &'static str) -> Vec<ExprEnv> {
        query_pattern_and_sources(space, pattern).1
    }

    fn source_prefix(source: ExprEnv) -> Vec<u8> {
        query_source_prefix(source).expect("test query source should have an encoded prefix")
    }

    fn query_projection_maps_for_source(
        space: &Space,
        source: ExprEnv,
    ) -> (Vec<(u8, u8)>, QueryProjectionMaps) {
        let variables = Space::query_factor_variables(source);
        let prefix = source_prefix(source);
        let prefix_cardinality = space.btm.read_zipper_at_path(&prefix).val_count();
        let projection = Space::query_projection_maps(
            &space.btm,
            source,
            &prefix,
            prefix_cardinality,
            Some(space.btm.val_count()),
        )
        .expect("test query projection should fit the bounded scan budget");

        (variables, projection)
    }

    fn sum_zipper_domain_telemetry(
        factors: &[QueryProjectionZipperRelationFactor<'_>],
    ) -> QueryProjectionZipperDomainTelemetry {
        let mut total = QueryProjectionZipperDomainTelemetry::default();
        for factor in factors {
            let telemetry = factor.telemetry();
            total.opens += telemetry.opens;
            total.cache_hits += telemetry.cache_hits;
            total.scans += telemetry.scans;
            total.read_zipper_scans += telemetry.read_zipper_scans;
            total.product_zipper_scans += telemetry.product_zipper_scans;
            total.candidates += telemetry.candidates;
            total.unifications += telemetry.unifications;
            total.rows += telemetry.rows;
            total.rows_matching_prefix += telemetry.rows_matching_prefix;
            total.domain_values += telemetry.domain_values;
        }
        total
    }

    fn encoded_expr_bytes(space: &mut Space, expr: &'static str) -> Vec<u8> {
        let expr = crate::expr!(space, expr);
        unsafe {
            expr.span()
                .as_ref()
                .expect("test expression should have an encoded span")
                .to_vec()
        }
    }

    fn plan_cache_key(id: usize) -> QueryFactorPlanCacheKey {
        QueryFactorPlanCacheKey {
            factors: vec![format!("factor-{id}").into_bytes()],
        }
    }


    fn side_index_key(id: usize) -> QueryShapeSideIndexKey {
        let key_bytes = id.to_le_bytes().to_vec();
        QueryShapeSideIndexKey {
            btm_val_count: id,
            prefix_cardinality: id + 1,
            prefix: key_bytes.clone(),
            shape: key_bytes,
        }
    }

    #[test]
    fn query_side_index_key_estimate_counts_prefix_and_shape_payloads() {
        let prefix = vec![1, 2, 3];
        let shape = vec![4, 5];
        let shape_key = QueryShapeSideIndexKey {
            btm_val_count: 7,
            prefix_cardinality: 11,
            prefix: prefix.clone(),
            shape: shape.clone(),
        };
        let projection_key = QueryProjectionSideIndexKey {
            btm_val_count: 7,
            prefix_cardinality: 11,
            prefix,
            shape,
        };

        assert_eq!(
            shape_key.estimated_bytes(),
            size_of::<QueryShapeSideIndexKey>() + 5
        );
        assert_eq!(
            projection_key.estimated_bytes(),
            size_of::<QueryProjectionSideIndexKey>() + 5
        );
    }

    #[test]
    fn query_projection_estimate_counts_domain_and_row_payloads() {
        let mut domain = BTreeSet::new();
        domain.insert(vec![1, 2]);
        domain.insert(vec![3]);
        let projection = QueryProjectionMaps {
            matches: 2,
            ground_root_matches: 2,
            schematic_root_matches: 0,
            variable_domains: vec![domain],
            variable_maps: vec![PathMap::new()],
            variable_rows: vec![vec![vec![4, 5, 6]].into_boxed_slice()],
        };

        assert_eq!(
            projection.estimated_bytes(),
            size_of::<QueryProjectionMaps>()
                + size_of::<BTreeSet<Vec<u8>>>()
                + size_of::<PathMap<()>>()
                + (3 * size_of::<Vec<u8>>())
                + 6
        );
        assert_eq!(projection.domain_value_count(), 2);
    }

    #[test]
    fn bounded_cache_insert_updates_existing_full_entry_without_clearing() {
        let mut entries = HashMap::new();
        assert!(!insert_bounded_cache_entry(
            &mut entries,
            2,
            1usize,
            10usize
        ));
        assert!(!insert_bounded_cache_entry(
            &mut entries,
            2,
            2usize,
            20usize
        ));

        assert!(!insert_bounded_cache_entry(
            &mut entries,
            2,
            1usize,
            99usize
        ));

        assert_eq!(entries.len(), 2);
        assert_eq!(entries.get(&1), Some(&99));
        assert_eq!(entries.get(&2), Some(&20));

        assert!(insert_bounded_cache_entry(&mut entries, 2, 3usize, 30usize));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries.get(&3), Some(&30));
    }

    #[test]
    fn query_factor_plan_cache_updates_existing_full_key_without_eviction() {
        let mut cache = QueryFactorPlanCache::default();
        for id in 0..QUERY_FACTOR_PLAN_CACHE_LIMIT {
            cache.insert(plan_cache_key(id), &[id]);
        }
        let retained_key = plan_cache_key(0);

        cache.insert(retained_key.clone(), &[usize::MAX]);

        assert_eq!(cache.entries.len(), QUERY_FACTOR_PLAN_CACHE_LIMIT);
        assert_eq!(cache.entries.get(&retained_key), Some(&vec![usize::MAX]));
    }

    #[test]
    fn query_shape_side_index_updates_existing_full_key_without_clear_generation() {
        let mut index = QueryShapeSideIndex::default();
        for id in 0..QUERY_SHAPE_SIDE_INDEX_LIMIT {
            index.insert(
                side_index_key(id),
                Some(QueryShapeSummary {
                    cardinality: id,
                    ground_root_matches: id,
                    schematic_root_matches: 0,
                    min_variable_domain_cardinality: None,
                    max_variable_domain_cardinality: None,
                    variable_domains: Vec::new(),
                }),
            );
        }
        let retained_key = side_index_key(0);
        let clears = index.clears;
        let generation = index.generation;

        index.insert(
            retained_key.clone(),
            Some(QueryShapeSummary {
                cardinality: usize::MAX,
                ground_root_matches: 0,
                schematic_root_matches: 0,
                min_variable_domain_cardinality: None,
                max_variable_domain_cardinality: None,
                variable_domains: Vec::new(),
            }),
        );

        assert_eq!(index.entries.len(), QUERY_SHAPE_SIDE_INDEX_LIMIT);
        assert_eq!(index.clears, clears);
        assert_eq!(index.generation, generation);
        assert_eq!(
            index
                .entries
                .get(&retained_key)
                .and_then(Option::as_ref)
                .map(|summary| summary.cardinality),
            Some(usize::MAX)
        );
    }

    fn query_projection_relation_factor_with_binding_order(
        projection: &QueryProjectionMaps,
        query_variables: &[(u8, u8)],
        query_to_binding_vars: &[((u8, u8), BindingVar)],
        factor_variables: &[BindingVar],
    ) -> QueryProjectionRelationFactor {
        let query_column_by_var = query_variables
            .iter()
            .copied()
            .enumerate()
            .map(|(index, variable)| (variable, index))
            .collect::<BTreeMap<_, _>>();
        let query_column_by_binding = query_to_binding_vars
            .iter()
            .map(|(query_variable, binding_variable)| {
                let query_column = *query_column_by_var
                    .get(query_variable)
                    .expect("binding variable should come from the projected query source");
                (*binding_variable, query_column)
            })
            .collect::<BTreeMap<_, _>>();
        let rows = projection.variable_rows.iter().map(|row| {
            factor_variables
                .iter()
                .map(|binding_variable| {
                    let query_column = *query_column_by_binding
                        .get(binding_variable)
                        .expect("factor variable should have a query projection column");
                    row[query_column].clone()
                })
                .collect::<Vec<_>>()
        });

        QueryProjectionRelationFactor::from_rows_with_variables(factor_variables.to_vec(), rows)
    }

    fn large_nested_pairs_sexpr() -> String {
        let mut input = String::from("(");
        for i in 0..63 {
            if i > 0 {
                input.push(' ');
            }
            input.push('(');
            input.push_str(&format!("s{i:02}{}", "x".repeat(58)));
            input.push(' ');
            input.push_str(&format!("t{i:02}{}", "y".repeat(58)));
            input.push(')');
        }
        input.push(')');
        input
    }

    #[test]
    fn template_output_buffer_starts_small_and_empty() {
        let buffer = template_output_buffer();

        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), TEMPLATE_OUTPUT_SCRATCH_INITIAL_CAPACITY);
        assert!(buffer.capacity() < (1 << 20));
    }

    #[test]
    fn parser_output_buffer_starts_small_and_empty() {
        let buffer = parser_output_buffer();

        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), PARSER_OUTPUT_SCRATCH_INITIAL_CAPACITY);
        assert!(buffer.capacity() < (1 << 20));
    }

    #[test]
    fn query_product_reserve_starts_bounded() {
        assert_eq!(
            QUERY_PRODUCT_PATH_BUFFER_INITIAL_CAPACITY,
            TEMPLATE_OUTPUT_SCRATCH_INITIAL_CAPACITY
        );
        assert!(QUERY_PRODUCT_PATH_BUFFER_INITIAL_CAPACITY < (1 << 20));
        assert_eq!(QUERY_PRODUCT_STACK_INITIAL_DEPTH, 64);
    }

    #[test]
    fn load_sexpr_impl_transforms_large_binding_with_growable_output() {
        let mut space = Space::new();
        let input = large_nested_pairs_sexpr();

        assert_eq!(
            space
                .add_sexpr(
                    input.as_bytes(),
                    crate::expr!(space, "$"),
                    crate::expr!(space, "[2] data _1"),
                )
                .unwrap(),
            1
        );

        assert_eq!(space.btm.val_count(), 1);
        let mut output = Vec::new();
        space.dump_sexpr(
            crate::expr!(space, "[2] data $"),
            crate::expr!(space, "_1"),
            &mut output,
        );
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("s00"), "{output}");
        assert!(output.len() > TEMPLATE_OUTPUT_SCRATCH_INITIAL_CAPACITY);
    }

    #[test]
    fn query_multi_matches_long_path_beyond_initial_product_reserve() {
        let mut space = Space::new();
        let input = format!("(data {})", large_nested_pairs_sexpr());
        space.add_all_sexpr(input.as_bytes()).unwrap();

        let mut matched_len = 0usize;
        let touched = Space::query_multi(
            &space.btm,
            crate::expr!(space, "[2] , [2] data $"),
            |_refs_bindings, loc| {
                matched_len = unsafe { loc.span().as_ref().unwrap().len() };
                true
            },
        );

        assert_eq!(touched, 1);
        assert!(matched_len > QUERY_PRODUCT_PATH_BUFFER_INITIAL_CAPACITY);
    }

    // Querying `(, (implies ($P $x) (Green $x)))` against the single stored atom
    // `(implies (Frog $x) (Green $x))` must yield exactly one match ($P<-Frog, $x=$x).
    // The stored atom's variable repeats (Frog $x ... Green $x); the VarRef re-check
    // must not count the coreferent data variable twice. Oracle: GroundingSpace = 1.
    #[test]
    fn coref_no_overproduction_on_repeated_data_var() {
        let mut space = Space::new();
        space
            .add_all_sexpr(b"(implies (Frog $x) (Green $x))\n")
            .unwrap();
        let mut count = 0usize;
        let touched = Space::query_multi(
            &space.btm,
            crate::expr!(space, "[2] , [3] implies [2] $ $ [2] Green _2"),
            |_b, _loc| {
                count += 1;
                true
            },
        );
        assert_eq!(touched, 1, "query_multi return count over-produces");
        assert_eq!(
            count, 1,
            "callback fired more than once: data coreference double-counted"
        );
    }

    // Dual of coref_no_overproduction: querying `(, (= (Add (S $n) Z) $r))` against the
    // stored rule `(= (Add $x Z) $x)` must find ONE match -- the data variable $x binds to
    // the COMPOUND `(S $n)` at its first occurrence and its reused second occurrence
    // re-checks against the fresh query var $r (so $r = (S $n)). Under-producing here (0
    // matches) stalls `(Add (S $n) Z)` -> `(S $n)` in b1_equal_chain. Oracle: GroundingSpace = 1.
    #[test]
    fn coref_finds_compound_binding_against_reused_data_var() {
        let mut space = Space::new();
        space.add_all_sexpr(b"(= (Add $x Z) $x)\n").unwrap();
        let mut count = 0usize;
        let touched = Space::query_multi(
            &space.btm,
            crate::expr!(space, "[2] , [3] = [3] Add [2] S $ Z $"),
            |_b, _loc| {
                count += 1;
                true
            },
        );
        assert_eq!(
            touched, 1,
            "matcher under-produces: missed compound binding to a reused data var"
        );
        let _ = count;
    }

    // Higher-order shape behind d2_higherfunc: a query variable ($y) appears both where
    // the rule binds a variable ($v) AND inside another rule argument ($b = (f $y)).
    // Query `(, (= (lam $y (f $y)) $r))` against stored `(= (lam $v $b) (got $v $b))`
    // must find ONE match. 0 here = kernel under-production (#29 class); 1 = the kernel is
    // fine and any miss is in the codec. Oracle: GroundingSpace = 1.
    #[test]
    fn higher_order_query_var_in_binder_and_body() {
        let mut space = Space::new();
        space
            .add_all_sexpr(b"(= (lam $v $b) (got $v $b))\n")
            .unwrap();
        let mut count = 0usize;
        let touched = Space::query_multi(
            &space.btm,
            crate::expr!(space, "[2] , [3] = [3] lam $ [2] f _1 $"),
            |_b, _loc| {
                count += 1;
                true
            },
        );
        assert_eq!(touched, 1, "higher-order two-sided match under-produces");
        let _ = count;
    }

    #[test]
    fn query_factor_rank_uses_encoded_byte_prefix_cardinality() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(A target left)
(A decoy left)
(A decoy right)
(Guard target)
(Rare target)
"#,
            )
            .unwrap();

        let sources = query_sources(
            &mut space,
            "[5] , [3] A $ left [2] Rare target [2] Guard $ $",
        );
        let mut prefix_cardinalities = BTreeMap::new();
        let mut shape_cardinalities = BTreeMap::new();

        let broad = Space::query_factor_rank(
            &space.btm,
            sources[0],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(broad.estimated_cardinality, 2);
        assert!(broad.prefix_cardinality_lookup);
        assert!(!broad.prefix_cardinality_cache_hit);
        assert!(broad.shape_cardinality_lookup);
        assert!(broad.shape_cardinality_scan || broad.shape_side_index_hit);
        assert!(broad.shape_cardinality_refined);
        assert_eq!(prefix_cardinalities.len(), 1);
        assert_eq!(prefix_cardinalities.values().next(), Some(&3));

        let broad_again = Space::query_factor_rank(
            &space.btm,
            sources[0],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(
            broad_again.estimated_cardinality,
            broad.estimated_cardinality
        );
        assert_eq!(broad_again.prefix_len, broad.prefix_len);
        assert!(broad_again.prefix_cardinality_lookup);
        assert!(broad_again.prefix_cardinality_cache_hit);
        assert_eq!(prefix_cardinalities.len(), 1);

        let ground = Space::query_factor_rank(
            &space.btm,
            sources[1],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(ground.estimated_cardinality, 1);
        assert!(ground.prefix_len > broad.prefix_len);

        let guard = Space::query_factor_rank(
            &space.btm,
            sources[2],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(guard.estimated_cardinality, 1);
        assert!(guard.prefix_len > broad.prefix_len);

        let unanchored = Space::query_factor_rank(
            &space.btm,
            sources[3],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(unanchored.estimated_cardinality, usize::MAX);
        assert_eq!(unanchored.prefix_len, 0);
        assert!(!unanchored.prefix_cardinality_lookup);
        assert!(!unanchored.prefix_cardinality_cache_hit);
    }

    #[test]
    fn query_factor_plan_prefers_selective_byte_prefixes() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(A target left)
(A decoy left)
(A decoy right)
(Guard target)
(Rare target)
"#,
            )
            .unwrap();

        let sources = query_sources(
            &mut space,
            "[5] , [3] A $ left [2] Rare target [2] Guard $ $",
        );

        assert_eq!(
            Space::query_factor_plan(&space.btm, &sources),
            vec![2, 1, 0, 3]
        );
    }

    #[test]
    fn query_factor_rank_refines_prefix_cardinality_with_shape_count() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(Rel target hot)
(Rel target cold)
(Rel decoy cold)
(Needle target)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[2] , [3] Rel $ hot");
        let mut prefix_cardinalities = BTreeMap::new();
        let mut shape_cardinalities = BTreeMap::new();

        let ranked = Space::query_factor_rank(
            &space.btm,
            sources[0],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(ranked.estimated_cardinality, 1);
        assert!(ranked.prefix_cardinality_lookup);
        assert!(ranked.shape_cardinality_lookup);
        assert!(ranked.shape_cardinality_scan);
        assert!(ranked.shape_cardinality_refined);
        assert!(!ranked.shape_cardinality_cache_hit);

        let ranked_again = Space::query_factor_rank(
            &space.btm,
            sources[0],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(ranked_again.estimated_cardinality, 1);
        assert!(ranked_again.prefix_cardinality_cache_hit);
        assert!(ranked_again.shape_cardinality_cache_hit);
        assert!(ranked_again.shape_cardinality_refined);
    }

    #[test]
    fn query_factor_shape_cache_key_normalizes_local_repeated_variables() {
        let mut space = Space::new();
        let sources = query_sources(&mut space, "[3] , [3] ShapeKey $ _1 [3] ShapeKey $ _2");
        let left_span = unsafe { sources[0].subsexpr().span().as_ref().unwrap().to_vec() };
        let right_span = unsafe { sources[1].subsexpr().span().as_ref().unwrap().to_vec() };

        assert_ne!(left_span, right_span);
        assert_eq!(
            Space::query_factor_shape_cache_key(sources[0]),
            Space::query_factor_shape_cache_key(sources[1])
        );
    }

    #[test]
    fn query_factor_rank_reuses_alpha_equivalent_shape_summary() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(ShapeCache same same)
(ShapeCache left right)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[3] , [3] ShapeCache $ _1 [3] ShapeCache $ _2");
        let mut prefix_cardinalities = BTreeMap::new();
        let mut shape_cardinalities = BTreeMap::new();

        let first = Space::query_factor_rank(
            &space.btm,
            sources[0],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(first.estimated_cardinality, 1);
        assert!(first.shape_cardinality_scan);
        assert!(first.shape_cardinality_refined);
        assert!(!first.shape_cardinality_cache_hit);
        assert_eq!(shape_cardinalities.len(), 1);

        let second = Space::query_factor_rank(
            &space.btm,
            sources[1],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(second.estimated_cardinality, 1);
        assert!(!second.shape_cardinality_scan);
        assert!(second.shape_cardinality_cache_hit);
        assert!(second.shape_cardinality_refined);
        assert_eq!(shape_cardinalities.len(), 1);
    }

    #[test]
    fn query_factor_plan_prefers_shape_refined_cardinality() {
        let mut space = Space::new();
        let mut program = String::new();
        for i in 0..100 {
            program.push_str(&format!("(Rel key{i} cold)\n"));
        }
        for i in 0..10 {
            program.push_str(&format!("(Needle key{i})\n"));
        }
        program.push_str("(Rel target hot)\n");
        program.push_str("(Needle target)\n");
        space.add_all_sexpr(program.as_bytes()).unwrap();

        let sources = query_sources(&mut space, "[3] , [3] Rel $ hot [2] Needle $");

        assert_eq!(Space::query_factor_plan(&space.btm, &sources), vec![0, 1]);
    }

    #[test]
    fn query_factor_rank_records_projected_variable_domains() {
        let mut space = Space::new();
        let mut program = String::new();
        for i in 0..8 {
            program.push_str(&format!("(ProjAA key{i} item{i})\n"));
            program.push_str(&format!("(ProjBB bucket{} item{i})\n", i % 2));
        }
        space.add_all_sexpr(program.as_bytes()).unwrap();

        let sources = query_sources(&mut space, "[3] , [3] ProjBB $ $ [3] ProjAA $ $");
        let mut prefix_cardinalities = BTreeMap::new();
        let mut shape_cardinalities = BTreeMap::new();

        let narrow = Space::query_factor_rank(
            &space.btm,
            sources[0],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(narrow.estimated_cardinality, 8);
        assert_eq!(narrow.min_variable_domain_cardinality, Some(2));
        assert_eq!(narrow.max_variable_domain_cardinality, Some(8));
        assert!(narrow.variable_domain_refined);

        let wide = Space::query_factor_rank(
            &space.btm,
            sources[1],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(wide.estimated_cardinality, 8);
        assert_eq!(wide.min_variable_domain_cardinality, Some(8));
        assert_eq!(wide.max_variable_domain_cardinality, Some(8));
        assert!(wide.variable_domain_refined);
    }

    #[test]
    fn query_factor_variables_dedupes_repeated_refs() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(Pair same same)
(Pair left right)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[2] , [3] Pair $ _1");
        let variables = Space::query_factor_variables(sources[0]);
        let mut prefix_cardinalities = BTreeMap::new();
        let mut shape_cardinalities = BTreeMap::new();

        let ranked = Space::query_factor_rank(
            &space.btm,
            sources[0],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );

        assert_eq!(variables.len(), 1);
        assert_eq!(ranked.estimated_cardinality, 1);
        assert_eq!(ranked.min_variable_domain_cardinality, Some(1));
        assert_eq!(ranked.max_variable_domain_cardinality, Some(1));
    }

    #[test]
    fn query_projection_maps_match_exact_repeated_variable_domains() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(Pair same same)
(Pair left right)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[2] , [3] Pair $ _1");
        let (variables, projection) = query_projection_maps_for_source(&space, sources[0]);
        let only_var = variables[0];
        let only_index = variables
            .iter()
            .position(|&var| var == only_var)
            .expect("query variable should be projected");

        assert_eq!(variables.len(), 1);
        assert_eq!(projection.matches, 1);
        assert_eq!(projection.variable_domains[only_index].len(), 1);
        assert_eq!(projection.variable_maps[only_index].val_count(), 1);
    }

    #[test]
    fn query_projection_maps_meet_matches_exact_shared_domain() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(WideMeet key0 item0)
(WideMeet key1 item1)
(WideMeet key2 item2)
(WideMeet key3 item3)
(WideMeet key4 item4)
(WideMeet key5 item5)
(NarrowMeet bucket0 item0)
(NarrowMeet bucket1 item1)
(NarrowMeet bucket2 item2)
(NarrowMeet bucket3 item7)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[3] , [3] WideMeet $ $ [3] NarrowMeet $ $");
        let (wide_variables, wide) = query_projection_maps_for_source(&space, sources[0]);
        let (narrow_variables, narrow) = query_projection_maps_for_source(&space, sources[1]);
        let wide_payload_index = 1;
        let narrow_payload_index = 1;

        assert_eq!(wide_variables.len(), 2);
        assert_eq!(narrow_variables.len(), 2);

        let exact_intersection_count = wide.variable_domains[wide_payload_index]
            .intersection(&narrow.variable_domains[narrow_payload_index])
            .count();
        let projection_meet = wide.variable_maps[wide_payload_index]
            .meet(&narrow.variable_maps[narrow_payload_index]);

        assert_eq!(wide.matches, 6);
        assert_eq!(narrow.matches, 4);
        assert_eq!(wide.variable_maps[wide_payload_index].val_count(), 6);
        assert_eq!(narrow.variable_maps[narrow_payload_index].val_count(), 4);
        assert_eq!(exact_intersection_count, 3);
        assert_eq!(projection_meet.val_count(), exact_intersection_count);
    }

    #[test]
    fn query_projection_domain_cursor_matches_pathmap_meet() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(WideCursor key0 item0)
(WideCursor key1 item1)
(WideCursor key2 item2)
(WideCursor key3 item3)
(WideCursor key4 item4)
(WideCursor key5 item5)
(NarrowCursor bucket0 item0)
(NarrowCursor bucket1 item1)
(NarrowCursor bucket2 item2)
(NarrowCursor bucket3 item7)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[3] , [3] WideCursor $ $ [3] NarrowCursor $ $");
        let (_, wide) = query_projection_maps_for_source(&space, sources[0]);
        let (_, narrow) = query_projection_maps_for_source(&space, sources[1]);
        let wide_payload_index = 1;
        let narrow_payload_index = 1;
        let expected_values = wide.variable_domains[wide_payload_index]
            .intersection(&narrow.variable_domains[narrow_payload_index])
            .cloned()
            .collect::<Vec<_>>();
        let pathmap_meet = wide.variable_maps[wide_payload_index]
            .meet(&narrow.variable_maps[narrow_payload_index]);
        let meet_values = intersect_query_projection_domains(&[&pathmap_meet]).values;

        let cursor_intersection = intersect_query_projection_domains(&[
            &wide.variable_maps[wide_payload_index],
            &narrow.variable_maps[narrow_payload_index],
        ]);

        assert_eq!(cursor_intersection.values, expected_values);
        assert_eq!(cursor_intersection.values, meet_values);
        assert_eq!(cursor_intersection.domain_sources, 2);
        assert_eq!(cursor_intersection.domain_values, 10);
        assert_eq!(cursor_intersection.cursor_opens, 2);
        assert!(cursor_intersection.cursor_seeks > 0);
        assert!(cursor_intersection.cursor_skips > 0);
        assert_eq!(cursor_intersection.cursor_nexts, expected_values.len());
    }

    #[test]
    fn query_projection_relation_factor_opens_bound_prefix_domain() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(BridgeFactor left y0)
(BridgeFactor left y1)
(BridgeFactor right y2)
(BridgeFactor right y3)
(BridgeFactor other y9)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[2] , [3] BridgeFactor $ $");
        let (variables, projection) = query_projection_maps_for_source(&space, sources[0]);
        let factor = QueryProjectionRelationFactor::from_projection(&projection);
        let root_domain = factor.open_domain(0, &[]);

        let mut expected_by_root = BTreeMap::<Vec<u8>, BTreeSet<Vec<u8>>>::new();
        for row in projection.variable_rows.iter() {
            expected_by_root
                .entry(row[0].clone())
                .or_default()
                .insert(row[1].clone());
        }
        let (prefix_value, expected_children) = expected_by_root
            .iter()
            .find(|(_, children)| children.len() == 2)
            .expect("test fixture should have a two-row prefix");
        let child_domain = factor.open_domain(1, std::slice::from_ref(prefix_value));
        let expected_root_values = expected_by_root.keys().cloned().collect::<Vec<_>>();
        let expected_child_values = expected_children.iter().cloned().collect::<Vec<_>>();
        let invalid_domain = factor.open_domain(1, &[prefix_value.clone(), prefix_value.clone()]);

        assert_eq!(variables.len(), 2);
        assert_eq!(projection.matches, 5);
        assert_eq!(projection.variable_rows.len(), 5);
        assert_eq!(root_domain.values, expected_root_values);
        assert_eq!(root_domain.rows_matching_prefix, 5);
        assert_eq!(child_domain.values, expected_child_values);
        assert_eq!(child_domain.variable_index, 1);
        assert_eq!(child_domain.bound_prefix_len, 1);
        assert_eq!(child_domain.rows, 5);
        assert_eq!(
            child_domain.rows_matching_prefix,
            expected_child_values.len()
        );
        assert!(invalid_domain.values.is_empty());
        assert_eq!(invalid_domain.rows_matching_prefix, 0);
    }

    #[test]
    fn query_projection_relation_factor_domains_feed_cursor_intersection() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(LeftFactor groupA y0)
(LeftFactor groupA y1)
(LeftFactor groupA y2)
(LeftFactor groupB y8)
(RightFactor groupB y1)
(RightFactor groupB y2)
(RightFactor groupB y9)
(RightFactor groupC y7)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[3] , [3] LeftFactor $ $ [3] RightFactor $ $");
        let (_, left_projection) = query_projection_maps_for_source(&space, sources[0]);
        let (_, right_projection) = query_projection_maps_for_source(&space, sources[1]);
        let left_factor = QueryProjectionRelationFactor::from_projection(&left_projection);
        let right_factor = QueryProjectionRelationFactor::from_projection(&right_projection);

        let mut left_by_prefix = BTreeMap::<Vec<u8>, BTreeSet<Vec<u8>>>::new();
        for row in left_projection.variable_rows.iter() {
            left_by_prefix
                .entry(row[0].clone())
                .or_default()
                .insert(row[1].clone());
        }
        let mut right_by_prefix = BTreeMap::<Vec<u8>, BTreeSet<Vec<u8>>>::new();
        for row in right_projection.variable_rows.iter() {
            right_by_prefix
                .entry(row[0].clone())
                .or_default()
                .insert(row[1].clone());
        }
        let (left_prefix, left_expected_domain) = left_by_prefix
            .iter()
            .find(|(_, domain)| domain.len() == 3)
            .expect("left fixture should have a three-value prefix");
        let (right_prefix, right_expected_domain) = right_by_prefix
            .iter()
            .find(|(_, domain)| domain.len() == 3)
            .expect("right fixture should have a three-value prefix");
        let left_domain = left_factor.open_domain(1, std::slice::from_ref(&left_prefix));
        let right_domain = right_factor.open_domain(1, std::slice::from_ref(&right_prefix));
        let expected_intersection = left_expected_domain
            .intersection(right_expected_domain)
            .cloned()
            .collect::<Vec<_>>();

        let cursor_intersection = intersect_query_projection_domain_values(&[
            left_domain.values.as_slice(),
            right_domain.values.as_slice(),
        ]);

        assert_eq!(left_domain.values.len(), 3);
        assert_eq!(right_domain.values.len(), 3);
        assert_eq!(cursor_intersection.values, expected_intersection);
        assert_eq!(cursor_intersection.domain_sources, 2);
        assert_eq!(cursor_intersection.cursor_opens, 2);
        assert!(cursor_intersection.cursor_seeks > 0);
        assert!(cursor_intersection.cursor_skips > 0);
    }

    fn binding_relation(
        schema: &[BindingVar],
        rows: &[&[u64]],
    ) -> crate::binding_space::BindingRelation {
        let mut relation = crate::binding_space::BindingRelation::new(schema.to_vec());
        for row in rows {
            relation
                .add(
                    row.iter().map(|&value| TermId(value)).collect::<Vec<_>>(),
                    1,
                )
                .unwrap();
        }
        relation
    }

    fn mapped_term_bytes(term: TermId) -> Vec<u8> {
        format!("t{:03}", term.0).into_bytes()
    }

    fn mapped_non_injective_term_bytes(term: TermId) -> Vec<u8> {
        match term.0 {
            1 | 2 => b"same".to_vec(),
            _ => mapped_term_bytes(term),
        }
    }

    #[test]
    fn query_projection_relation_factors_match_trie_cursor_contract() {
        let left = binding_relation(
            &[BindingVar(0), BindingVar(1)],
            &[&[1, 10], &[2, 10], &[3, 20]],
        );
        let right = binding_relation(
            &[BindingVar(1), BindingVar(2)],
            &[&[10, 100], &[10, 101], &[30, 300]],
        );
        let variable_order = [BindingVar(1), BindingVar(0), BindingVar(2)];
        let trace = crate::binding_space::trie_join_trace(&[left, right], &variable_order).unwrap();
        let contract = trace.cursor_contract().unwrap();
        let factors = [
            QueryProjectionRelationFactor::from_rows_with_variables(
                [BindingVar(1), BindingVar(0)],
                [
                    vec![mapped_term_bytes(TermId(10)), mapped_term_bytes(TermId(1))],
                    vec![mapped_term_bytes(TermId(10)), mapped_term_bytes(TermId(2))],
                    vec![mapped_term_bytes(TermId(20)), mapped_term_bytes(TermId(3))],
                ],
            ),
            QueryProjectionRelationFactor::from_rows_with_variables(
                [BindingVar(1), BindingVar(2)],
                [
                    vec![
                        mapped_term_bytes(TermId(10)),
                        mapped_term_bytes(TermId(100)),
                    ],
                    vec![
                        mapped_term_bytes(TermId(10)),
                        mapped_term_bytes(TermId(101)),
                    ],
                    vec![
                        mapped_term_bytes(TermId(30)),
                        mapped_term_bytes(TermId(300)),
                    ],
                ],
            ),
        ];

        let comparison = compare_query_projection_relation_factors_to_trie_contract(
            &factors,
            &variable_order,
            &contract,
            |term| Some(mapped_term_bytes(term)),
        );

        assert_eq!(comparison.relation_indexes, 2);
        assert_eq!(comparison.factor_requirements, 2);
        assert_eq!(comparison.contexts, 5);
        assert_eq!(comparison.matched_contexts, comparison.contexts);
        assert_eq!(comparison.mismatched_contexts, 0);
        assert_eq!(comparison.missing_factors, 0);
        assert_eq!(comparison.missing_term_mappings, 0);
        assert!(
            comparison
                .context_results
                .iter()
                .all(|context| context.matched)
        );
    }

    #[test]
    fn query_projection_relation_factor_contract_compare_reports_domain_mismatch() {
        let left = binding_relation(
            &[BindingVar(0), BindingVar(1)],
            &[&[1, 10], &[2, 10], &[3, 20]],
        );
        let right = binding_relation(
            &[BindingVar(1), BindingVar(2)],
            &[&[10, 100], &[10, 101], &[30, 300]],
        );
        let variable_order = [BindingVar(1), BindingVar(0), BindingVar(2)];
        let trace = crate::binding_space::trie_join_trace(&[left, right], &variable_order).unwrap();
        let contract = trace.cursor_contract().unwrap();
        let factors = [
            QueryProjectionRelationFactor::from_rows_with_variables(
                [BindingVar(1), BindingVar(0)],
                [
                    vec![mapped_term_bytes(TermId(10)), mapped_term_bytes(TermId(1))],
                    vec![mapped_term_bytes(TermId(10)), mapped_term_bytes(TermId(9))],
                    vec![mapped_term_bytes(TermId(20)), mapped_term_bytes(TermId(3))],
                ],
            ),
            QueryProjectionRelationFactor::from_rows_with_variables(
                [BindingVar(1), BindingVar(2)],
                [
                    vec![
                        mapped_term_bytes(TermId(10)),
                        mapped_term_bytes(TermId(100)),
                    ],
                    vec![
                        mapped_term_bytes(TermId(10)),
                        mapped_term_bytes(TermId(101)),
                    ],
                    vec![
                        mapped_term_bytes(TermId(30)),
                        mapped_term_bytes(TermId(300)),
                    ],
                ],
            ),
        ];

        let comparison = compare_query_projection_relation_factors_to_trie_contract(
            &factors,
            &variable_order,
            &contract,
            |term| Some(mapped_term_bytes(term)),
        );
        let bad_context = comparison
            .context_results
            .iter()
            .find(|context| context.variable == BindingVar(0) && !context.matched)
            .expect("changed left x-domain should be reported");

        assert_eq!(comparison.contexts, 5);
        assert_eq!(comparison.matched_contexts, 4);
        assert_eq!(comparison.mismatched_contexts, 1);
        assert_eq!(
            bad_context.expected_domain.as_slice(),
            [mapped_term_bytes(TermId(1)), mapped_term_bytes(TermId(2))]
        );
        assert_eq!(
            bad_context.actual_domain.as_slice(),
            [mapped_term_bytes(TermId(1)), mapped_term_bytes(TermId(9))]
        );
    }

    #[test]
    fn query_projection_relation_factor_contract_compare_rejects_non_injective_term_mapping() {
        let left = binding_relation(
            &[BindingVar(0), BindingVar(1)],
            &[&[1, 10], &[2, 10], &[3, 20]],
        );
        let right = binding_relation(
            &[BindingVar(1), BindingVar(2)],
            &[&[10, 100], &[10, 101], &[30, 300]],
        );
        let variable_order = [BindingVar(1), BindingVar(0), BindingVar(2)];
        let trace = crate::binding_space::trie_join_trace(&[left, right], &variable_order).unwrap();
        let contract = trace.cursor_contract().unwrap();
        let factors = [
            QueryProjectionRelationFactor::from_rows_with_variables(
                [BindingVar(1), BindingVar(0)],
                [
                    vec![
                        mapped_non_injective_term_bytes(TermId(10)),
                        mapped_non_injective_term_bytes(TermId(1)),
                    ],
                    vec![
                        mapped_non_injective_term_bytes(TermId(10)),
                        mapped_non_injective_term_bytes(TermId(2)),
                    ],
                    vec![
                        mapped_non_injective_term_bytes(TermId(20)),
                        mapped_non_injective_term_bytes(TermId(3)),
                    ],
                ],
            ),
            QueryProjectionRelationFactor::from_rows_with_variables(
                [BindingVar(1), BindingVar(2)],
                [
                    vec![
                        mapped_non_injective_term_bytes(TermId(10)),
                        mapped_non_injective_term_bytes(TermId(100)),
                    ],
                    vec![
                        mapped_non_injective_term_bytes(TermId(10)),
                        mapped_non_injective_term_bytes(TermId(101)),
                    ],
                    vec![
                        mapped_non_injective_term_bytes(TermId(30)),
                        mapped_non_injective_term_bytes(TermId(300)),
                    ],
                ],
            ),
        ];

        let comparison = compare_query_projection_relation_factors_to_trie_contract(
            &factors,
            &variable_order,
            &contract,
            |term| Some(mapped_non_injective_term_bytes(term)),
        );
        let bad_context = comparison
            .context_results
            .iter()
            .find(|context| context.variable == BindingVar(0) && !context.matched)
            .expect("non-injective term encoding should be reported");

        assert_eq!(comparison.contexts, 5);
        assert_eq!(comparison.matched_contexts, 4);
        assert_eq!(comparison.mismatched_contexts, 1);
        assert_eq!(
            bad_context.expected_domain,
            [b"same".to_vec(), b"same".to_vec()]
        );
        assert_eq!(bad_context.actual_domain, [b"same".to_vec()]);
    }

    #[test]
    fn query_projection_relation_factors_match_selected_sidecar_contract_from_query_projection() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge Alice Bob)
(edge Bob Carol)
(edge Alice Dana)
(edge Dana Carol)
(edge Carol Erin)
(edge X Y)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[3] , [3] edge $ $ [3] edge _2 $");
        let (left_query_variables, left_projection) =
            query_projection_maps_for_source(&space, sources[0]);
        let (right_query_variables, right_projection) =
            query_projection_maps_for_source(&space, sources[1]);

        assert_eq!(left_query_variables.len(), 2);
        assert_eq!(right_query_variables.len(), 2);
        assert_eq!(
            left_query_variables[1], right_query_variables[0],
            "second edge source should share the middle variable with the first"
        );

        let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&space.btm).unwrap();
        let edge = sidecar
            .term_id_for_encoded(&encoded_expr_bytes(&mut space, "edge"))
            .unwrap();

        let descriptor = crate::arrangements::ArrangementDescriptor::new(edge, 2, [0, 1]).unwrap();
        let xy = crate::binding_plan::BindingAccessPlan::Arrangement {
            descriptor: descriptor.clone(),
            projection: crate::arrangements::ArrangementProjection::new(
                2,
                [BindingVar(0), BindingVar(1)],
                [0, 1],
            )
            .unwrap(),
        };
        let yz = crate::binding_plan::BindingAccessPlan::Arrangement {
            descriptor,
            projection: crate::arrangements::ArrangementProjection::new(
                2,
                [BindingVar(1), BindingVar(2)],
                [0, 1],
            )
            .unwrap(),
        };
        let plan = crate::binding_plan::BindingSidecarPlan::new(
            [xy, yz],
            [BindingVar(0), BindingVar(1), BindingVar(2)],
        );
        let report = plan
            .explain_selected_trie_cursor_contract(&sidecar)
            .unwrap();
        let contract = report
            .cursor_contract
            .as_ref()
            .expect("edge transitive plan should select trie cursor contract");
        let selected_variable_order = report.execution.choice.variable_order.clone();

        let factors = [
            query_projection_relation_factor_with_binding_order(
                &left_projection,
                &left_query_variables,
                &[
                    (left_query_variables[0], BindingVar(0)),
                    (left_query_variables[1], BindingVar(1)),
                ],
                &[BindingVar(1), BindingVar(0)],
            ),
            query_projection_relation_factor_with_binding_order(
                &right_projection,
                &right_query_variables,
                &[
                    (right_query_variables[0], BindingVar(1)),
                    (right_query_variables[1], BindingVar(2)),
                ],
                &[BindingVar(1), BindingVar(2)],
            ),
        ];
        let comparison = compare_query_projection_relation_factors_to_trie_contract(
            &factors,
            &selected_variable_order,
            contract,
            |term| {
                sidecar
                    .get_term(term)
                    .map(|record| record.encoded().to_vec())
            },
        );

        assert_eq!(
            report.execution.choice.kernel,
            crate::binding_plan::BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        assert_eq!(
            selected_variable_order.as_ref(),
            [BindingVar(1), BindingVar(0), BindingVar(2)]
        );
        assert_eq!(left_projection.matches, 6);
        assert_eq!(right_projection.matches, 6);
        assert_eq!(comparison.relation_indexes, 2);
        assert_eq!(comparison.factor_requirements, 2);
        assert_eq!(comparison.contexts, 9);
        assert_eq!(comparison.matched_contexts, comparison.contexts);
        assert_eq!(comparison.mismatched_contexts, 0);
        assert_eq!(comparison.missing_factors, 0);
        assert_eq!(comparison.missing_term_mappings, 0);
        assert!(
            comparison
                .context_results
                .iter()
                .all(|context| context.matched)
        );
    }

    #[test]
    fn lowered_transitive_edge_body_matches_product_zipper() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge Alice Bob)
(edge Bob Carol)
(edge Alice Dana)
(edge Dana Carol)
(edge Carol Erin)
(edge X Y)
"#,
            )
            .unwrap();

        let (product_pattern, sources) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ $ [3] edge _2 $");
        let left = Space::query_factor_variables(sources[0]);
        let right = Space::query_factor_variables(sources[1]);

        let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&space.btm).unwrap();

        let plan = Space::lower_query_to_sidecar_plan(&sources, &mut sidecar)
            .expect("flat transitive-edge body should lower to an arrangement plan");

        // The lowering reproduces the hand-authored contract plan exactly: two
        // `edge/2` arrangements sharing the middle variable.
        let edge = sidecar
            .term_id_for_encoded(&encoded_expr_bytes(&mut space, "edge"))
            .unwrap();
        let descriptor = crate::arrangements::ArrangementDescriptor::new(edge, 2, [0, 1]).unwrap();
        let expected = crate::binding_plan::BindingSidecarPlan::new(
            [
                crate::binding_plan::BindingAccessPlan::Arrangement {
                    descriptor: descriptor.clone(),
                    projection: crate::arrangements::ArrangementProjection::new(
                        2,
                        [BindingVar(0), BindingVar(1)],
                        [0, 1],
                    )
                    .unwrap(),
                },
                crate::binding_plan::BindingAccessPlan::Arrangement {
                    descriptor,
                    projection: crate::arrangements::ArrangementProjection::new(
                        2,
                        [BindingVar(1), BindingVar(2)],
                        [0, 1],
                    )
                    .unwrap(),
                },
            ],
            [BindingVar(0), BindingVar(1), BindingVar(2)],
        );
        assert_eq!(plan, expected);

        // And the lowered plan matches the ProductZipper row-for-row through the
        // existing acceptance harness, which is the bridge's validation step.
        let report = plan
            .explain_selected_trie_cursor_contract(&sidecar)
            .unwrap();
        let selected_variable_order = report.execution.choice.variable_order.clone();
        let selected = plan.execute_selected(&sidecar).unwrap();
        let product_trace = Space::trace_query_projection_product_candidates(
            &space.btm,
            product_pattern,
            [
                (BindingVar(0), left[0]),
                (BindingVar(1), left[1]),
                (BindingVar(2), right[1]),
            ],
            selected_variable_order,
        );
        let trace_comparison = compare_query_projection_product_trace_to_binding_relation(
            &product_trace,
            &selected.relation,
            |term| {
                sidecar
                    .get_term(term)
                    .map(|record| record.encoded().to_vec())
            },
        );

        assert!(trace_comparison.matched);
        assert_eq!(trace_comparison.missing_term_mappings, 0);
        assert_eq!(trace_comparison.missing_binding_rows, 0);
        assert_eq!(trace_comparison.non_binding_results, 0);
        assert!(!product_trace.rows.is_empty());
    }

    #[test]
    fn lowered_edge_color_body_matches_product_zipper() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(color a red)
(color b red)
(color c blue)
"#,
            )
            .unwrap();

        let (product_pattern, sources) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ $ [3] color _1 red");
        let left = Space::query_factor_variables(sources[0]);

        let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&space.btm).unwrap();

        let plan = Space::lower_query_to_sidecar_plan(&sources, &mut sidecar)
            .expect("edge arrangement joined with a color pattern factor should lower");

        // The edge factor is a distinct-variable equi-join (arrangement); the
        // color factor carries the constant `red`, a selection, so it lowers to
        // the exact-filtering pattern factor.
        assert!(matches!(
            plan.factors()[0],
            crate::binding_plan::BindingAccessPlan::Arrangement { .. }
        ));
        assert!(matches!(
            plan.factors()[1],
            crate::binding_plan::BindingAccessPlan::Pattern { .. }
        ));

        let selected = plan.execute(&sidecar).unwrap();
        let product_trace = Space::trace_query_projection_product_candidates(
            &space.btm,
            product_pattern,
            [(BindingVar(0), left[0]), (BindingVar(1), left[1])],
            [BindingVar(0), BindingVar(1)],
        );
        let trace_comparison = compare_query_projection_product_trace_to_binding_relation(
            &product_trace,
            &selected.relation,
            |term| {
                sidecar
                    .get_term(term)
                    .map(|record| record.encoded().to_vec())
            },
        );

        assert!(trace_comparison.matched);
        assert_eq!(trace_comparison.missing_term_mappings, 0);
        assert_eq!(selected.relation.positive_rows().count(), 2);
        assert!(!product_trace.rows.is_empty());
    }

    #[test]
    fn lowered_repeated_variable_factor_matches_product_zipper() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a a)
(edge a b)
(edge b b)
(rel a)
(rel b)
"#,
            )
            .unwrap();

        let (product_pattern, sources) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ _1 [2] rel _1");
        let left = Space::query_factor_variables(sources[0]);

        let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&space.btm).unwrap();

        let plan = Space::lower_query_to_sidecar_plan(&sources, &mut sidecar)
            .expect("self-loop edge joined with a unary arrangement should lower");

        // `(edge $x $x)` repeats a variable, a self-join the positional
        // arrangement cannot express, so it lowers to the pattern factor; the
        // unary `(rel $x)` stays a distinct-variable arrangement.
        assert!(matches!(
            plan.factors()[0],
            crate::binding_plan::BindingAccessPlan::Pattern { .. }
        ));
        assert!(matches!(
            plan.factors()[1],
            crate::binding_plan::BindingAccessPlan::Arrangement { .. }
        ));

        let selected = plan.execute(&sidecar).unwrap();
        let product_trace = Space::trace_query_projection_product_candidates(
            &space.btm,
            product_pattern,
            [(BindingVar(0), left[0])],
            [BindingVar(0)],
        );
        let trace_comparison = compare_query_projection_product_trace_to_binding_relation(
            &product_trace,
            &selected.relation,
            |term| {
                sidecar
                    .get_term(term)
                    .map(|record| record.encoded().to_vec())
            },
        );

        // Self-loops a, b, both present in `rel`.
        assert!(trace_comparison.matched);
        assert_eq!(trace_comparison.missing_term_mappings, 0);
        assert_eq!(selected.relation.positive_rows().count(), 2);
    }

    #[test]
    fn sidecar_emit_matches_product_zipper() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge a d)
(edge d c)
"#,
            )
            .unwrap();

        // Transitive rule: (, (edge $x $y) (edge $y $z)) -> (, (path $x $z)).
        // Driving the template writes from the sidecar join output must produce
        // exactly the ProductZipper's output set.
        let (pat_expr, _) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ $ [3] edge _2 $");
        let (tpl_expr, _) = query_pattern_and_sources(&mut space, "[2] , [3] path _1 _3");

        let agree = Space::validate_sidecar_emit_against_product(&space.btm, pat_expr, tpl_expr)
            .expect("transitive body lowers to a sidecar plan");
        assert!(
            agree,
            "sidecar-driven emit must reproduce the ProductZipper output set"
        );
    }

    #[test]
    fn sidecar_emit_matches_product_zipper_across_bodies() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge c a)
(edge a d)
(edge d c)
(color a red)
(color b red)
(color c blue)
"#,
            )
            .unwrap();

        // (pattern body, template body), template variables referencing the
        // pattern variables by the same De Bruijn indices.
        let bodies = [
            ("[3] , [3] edge $ $ [3] edge _2 $", "[2] , [3] path _1 _3"),
            ("[3] , [3] edge $ $ [3] color _1 red", "[2] , [3] out _1 _2"),
            (
                "[4] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1",
                "[2] , [4] tri _1 _2 _3",
            ),
        ];
        for (pat, tpl) in bodies {
            let (pat_expr, _) = query_pattern_and_sources(&mut space, pat);
            let (tpl_expr, _) = query_pattern_and_sources(&mut space, tpl);
            let agree =
                Space::validate_sidecar_emit_against_product(&space.btm, pat_expr, tpl_expr)
                    .unwrap_or_else(|| panic!("body should lower: {pat}"));
            assert!(
                agree,
                "sidecar emit must match the ProductZipper for: {pat}"
            );
        }
    }

    #[test]
    fn transform_via_sidecar_writes_the_join_outputs() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge c a)
(edge a d)
"#,
            )
            .unwrap();
        // A cyclic triangle body, so the sidecar's worst-case-optimal join drives
        // the writes (the acyclic case falls back, see below).
        let (pat_expr, _) = query_pattern_and_sources(
            &mut space,
            "[4] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1",
        );
        let (tpl_expr, _) = query_pattern_and_sources(&mut space, "[2] , [4] tri _1 _2 _3");

        let read_copy = space.btm.clone();
        let expected = {
            let mut tpl_args = Vec::new();
            ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
            let templates: Vec<Expr> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
            Space::product_template_outputs(&read_copy, pat_expr, &templates)
        };
        let before = space.btm.val_count();

        let (matches, any_new) = space
            .transform_via_sidecar(&read_copy, pat_expr, tpl_expr)
            .expect("cyclic triangle body uses the sidecar");

        // Three rotations of the one directed triangle, each instantiating tri.
        assert_eq!(matches, 3);
        assert!(any_new);
        assert_eq!(expected.len(), 3);
        for path in &expected {
            assert!(
                space.btm.contains(&path[..]),
                "output {path:?} should be in the space"
            );
        }
        assert_eq!(space.btm.val_count(), before + expected.len());
    }

    // Differential check that the worst-case-optimal flip writes exactly what the
    // ProductZipper reference does, on a richer graph: three disjoint directed
    // triangles plus noise edges (a two-cycle a<->d and a dangling b->e). If the
    // sidecar's join or its maintenance had a bug on a less trivial cyclic
    // instance, the two would diverge. Raising confidence here is what lets the
    // flip be defaulted (it never changes results, only speed).
    #[test]
    fn transform_via_sidecar_matches_product_zipper_on_a_richer_graph() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge c a)
(edge d e)
(edge e f)
(edge f d)
(edge x y)
(edge y z)
(edge z x)
(edge a d)
(edge d a)
(edge b e)
"#,
            )
            .unwrap();
        let (pat_expr, _) = query_pattern_and_sources(
            &mut space,
            "[4] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1",
        );
        let (tpl_expr, _) = query_pattern_and_sources(&mut space, "[2] , [4] tri _1 _2 _3");

        let read_copy = space.btm.clone();
        let expected = {
            let mut tpl_args = Vec::new();
            ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
            let templates: Vec<Expr> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
            Space::product_template_outputs(&read_copy, pat_expr, &templates)
        };
        let before = space.btm.val_count();

        let (matches, _) = space
            .transform_via_sidecar(&read_copy, pat_expr, tpl_expr)
            .expect("cyclic body uses the sidecar");

        // Three directed triangles, three rotations each: nine distinct outputs,
        // exactly the ProductZipper reference set, nothing more or fewer.
        assert_eq!(matches, expected.len());
        assert_eq!(expected.len(), 9);
        for path in &expected {
            assert!(
                space.btm.contains(&path[..]),
                "WCO output {path:?} must match the ProductZipper reference"
            );
        }
        assert_eq!(space.btm.val_count(), before + expected.len());
    }

    // Differential fuzzer: random graphs crossed with diverse cyclic query
    // shapes (3-, 4-, 5-cycles). Whenever the worst-case-optimal flip engages, it
    // must write exactly the ProductZipper reference set: every reference output
    // present, and the space grown by exactly that many facts. This is the
    // validation that lets the flip be defaulted (its only risk was an untested
    // cyclic pattern diverging from the reference; the flip never changes results,
    // only speed).
    #[test]
    fn fuzz_wco_flip_matches_product_zipper() {
        fn xorshift(state: &mut u64) -> u64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            x
        }

        let shapes: &[(&str, &str)] = &[
            (
                "[4] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1",
                "[2] , [4] tri _1 _2 _3",
            ),
            (
                "[5] , [3] edge $ $ [3] edge _2 $ [3] edge _3 $ [3] edge _4 _1",
                "[2] , [5] cyc _1 _2 _3 _4",
            ),
            (
                "[6] , [3] edge $ $ [3] edge _2 $ [3] edge _3 $ [3] edge _4 $ [3] edge _5 _1",
                "[2] , [6] cyc _1 _2 _3 _4 _5",
            ),
        ];

        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut engaged = 0usize;
        let mut total = 0usize;
        for (pattern, template) in shapes {
            for _ in 0..120 {
                total += 1;
                let nodes = 4 + (xorshift(&mut rng) % 5) as usize; // 4..8 nodes
                let edge_count = nodes + (xorshift(&mut rng) % (2 * nodes as u64)) as usize;
                let mut space = Space::new();
                let mut program = String::new();
                for _ in 0..edge_count {
                    let a = xorshift(&mut rng) % nodes as u64;
                    let b = xorshift(&mut rng) % nodes as u64;
                    program.push_str(&format!("(edge n{a} n{b})\n"));
                }
                space.add_all_sexpr(program.as_bytes()).unwrap();

                let (pat_expr, _) = query_pattern_and_sources(&mut space, pattern);
                let (tpl_expr, _) = query_pattern_and_sources(&mut space, template);

                let read_copy = space.btm.clone();
                let expected: std::collections::HashSet<Vec<u8>> = {
                    let mut tpl_args = Vec::new();
                    ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
                    let templates: Vec<Expr> =
                        tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
                    Space::product_template_outputs(&read_copy, pat_expr, &templates)
                        .into_iter()
                        .collect()
                };
                let before = space.btm.val_count();

                if space
                    .transform_via_sidecar(&read_copy, pat_expr, tpl_expr)
                    .is_some()
                {
                    engaged += 1;
                    for path in &expected {
                        assert!(
                            space.btm.contains(&path[..]),
                            "WCO flip missed a ProductZipper reference output on shape {pattern}"
                        );
                    }
                    assert_eq!(
                        space.btm.val_count(),
                        before + expected.len(),
                        "WCO flip wrote a different set than ProductZipper on shape {pattern}"
                    );
                }
            }
        }
        // The fuzzer must actually exercise the WCO path, not just decline.
        assert!(
            engaged * 2 >= total,
            "flip engaged on only {engaged}/{total} cyclic queries"
        );
    }

    #[test]
    fn transform_via_sidecar_declines_acyclic_bodies() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
"#,
            )
            .unwrap();
        // An acyclic transitive body: the ProductZipper's trie walk is already
        // output-sensitive, so the sidecar declines and the caller keeps it.
        let (pat_expr, _) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ $ [3] edge _2 $");
        let (tpl_expr, _) = query_pattern_and_sources(&mut space, "[2] , [3] path _1 _3");
        let read_copy = space.btm.clone();
        assert!(
            space
                .transform_via_sidecar(&read_copy, pat_expr, tpl_expr)
                .is_none()
        );
    }

    #[test]
    fn transform_via_sidecar_declines_schematic_relations() {
        #[cfg(feature = "semi_naive_ic")]
        let _sni_guard = SniDisarmGuard::new();
        let _route_guard = sidecar_route_toggle_lock().lock().unwrap();
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge c a)
(r b $d)
"#,
            )
            .unwrap();

        let (pat_expr, sources) = query_pattern_and_sources(
            &mut space,
            "[5] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1 [3] r _2 [2] f _1",
        );
        let (tpl_expr, _) = query_pattern_and_sources(&mut space, "[2] , [2] out _1");

        assert!(
            Space::body_is_cyclic(pat_expr),
            "the regression must not be covered by the acyclic fallback"
        );

        let prefixes = sources
            .iter()
            .map(|&source| query_source_prefix(source).filter(|prefix| !prefix.is_empty()))
            .collect::<Option<Vec<Vec<u8>>>>()
            .expect("all factors have relation prefixes");
        let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&space.btm).unwrap();
        assert!(
            sidecar.any_schematic_fact_under_prefixes(&prefixes),
            "the r relation contains a schematic stored fact"
        );

        let read_copy = space.btm.clone();
        let old_unify = SIDECAR_UNIFY_ENABLED.swap(false, std::sync::atomic::Ordering::Relaxed);
        let old_capture = SIDECAR_CAPTURE_ENABLED.swap(false, std::sync::atomic::Ordering::Relaxed);
        let declined = space
            .transform_via_sidecar(&read_copy, pat_expr, tpl_expr)
            .is_none();
        SIDECAR_CAPTURE_ENABLED.store(old_capture, std::sync::atomic::Ordering::Relaxed);
        SIDECAR_UNIFY_ENABLED.store(old_unify, std::sync::atomic::Ordering::Relaxed);
        assert!(
            declined,
            "cyclic bodies with schematic relations must stay on ProductZipper"
        );
    }

    #[test]
    fn body_is_cyclic_classifies_join_graphs() {
        let mut space = Space::new();
        let (triangle, _) = query_pattern_and_sources(
            &mut space,
            "[4] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1",
        );
        assert!(
            Space::body_is_cyclic(triangle),
            "the triangle body is cyclic"
        );
        let (transitive, _) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ $ [3] edge _2 $");
        assert!(
            !Space::body_is_cyclic(transitive),
            "the two-edge path body is acyclic"
        );
        let (single, _) = query_pattern_and_sources(&mut space, "[2] , [3] edge $ $");
        assert!(
            !Space::body_is_cyclic(single),
            "a single factor is never cyclic"
        );
    }

    #[test]
    fn persistent_sidecar_reuses_interned_relation_across_queries() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge c a)
(edge a d)
"#,
            )
            .unwrap();
        let (pat_expr, _) = query_pattern_and_sources(
            &mut space,
            "[4] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1",
        );
        let (tpl_expr, _) = query_pattern_and_sources(&mut space, "[2] , [4] tri _1 _2 _3");

        // First query builds the persistent sidecar and interns the edge relation.
        let read1 = space.btm.clone();
        let (matches1, _) = space
            .transform_via_sidecar(&read1, pat_expr, tpl_expr)
            .expect("cyclic body uses the sidecar");
        let scans_after_first = space
            .bridge_sidecar
            .as_ref()
            .map_or(0, |sidecar| sidecar.resync_scans());
        assert_eq!(matches1, 3);
        assert!(
            scans_after_first > 0,
            "the first query re-scans the edge relation to intern it"
        );

        // Second query over the same edge relation (the triangle writes `tri`, not
        // `edge`, so `edge` is unchanged) reuses the interned terms: the watermark
        // matches, so the re-scan counter does not move.
        let read2 = space.btm.clone();
        let (matches2, _) = space
            .transform_via_sidecar(&read2, pat_expr, tpl_expr)
            .expect("cyclic body uses the sidecar");
        let scans_after_second = space
            .bridge_sidecar
            .as_ref()
            .map_or(0, |sidecar| sidecar.resync_scans());
        assert_eq!(matches2, 3);
        assert_eq!(
            scans_after_second, scans_after_first,
            "the second query re-interns nothing (incremental watermark hit)"
        );
    }

    #[test]
    fn persistent_sidecar_resyncs_after_equal_count_swap() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge c a)
(edge a d)
"#,
            )
            .unwrap();
        let (pat_expr, _) = query_pattern_and_sources(
            &mut space,
            "[4] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1",
        );
        let (tpl_expr, _) = query_pattern_and_sources(&mut space, "[2] , [4] tri _1 _2 _3");

        let read1 = space.btm.clone();
        let (matches1, _) = space
            .transform_via_sidecar(&read1, pat_expr, tpl_expr)
            .expect("cyclic body uses the sidecar");
        assert_eq!(matches1, 3, "the a-b-c triangle has three rotations");

        // Equal-count swap: remove (edge c a), which breaks the triangle, and add
        // (edge c e), which keeps the edge relation (and the whole space) at the
        // same count. The per-prefix value-count watermark cannot tell it changed.
        let count_before = space.btm.val_count();
        space.load_all_sexpr_impl(b"(edge c a)\n", false).unwrap();
        space.add_all_sexpr(b"(edge c e)\n").unwrap();
        assert_eq!(
            space.btm.val_count(),
            count_before,
            "the swap leaves the count unchanged"
        );

        // A RemoveSink transform bumps this per-Space generation; the swap above
        // went through the loader, so bump it explicitly to stand in for that.
        // Without the bump the stale sidecar would still report the old triangle.
        space.bridge_remove_gen += 1;
        let read2 = space.btm.clone();
        let (matches2, _) = space
            .transform_via_sidecar(&read2, pat_expr, tpl_expr)
            .expect("cyclic body uses the sidecar");
        assert_eq!(
            matches2, 0,
            "the removal-generation bump forces a re-sync; no triangle remains"
        );
    }

    #[test]
    fn linear_recursive_fixpoint_computes_transitive_closure() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge c d)
(path a b)
(path b c)
(path c d)
"#,
            )
            .unwrap();
        // path(x,z) :- edge(x,y), path(y,z), with path seeded to the edges.
        let (pat_expr, _) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ $ [3] path _2 $");
        let (tpl_expr, _) = query_pattern_and_sources(&mut space, "[2] , [3] path _1 _3");

        let read_copy = space.btm.clone();
        let (count, any_new) = space
            .transform_linear_recursive_fixpoint(&read_copy, pat_expr, tpl_expr)
            .expect("the linear recursive rule lowers to a fixpoint");

        // Transitive closure of the three-edge chain is six pairs: a-b, b-c, c-d,
        // a-c, b-d, a-d. Three were seeded, three are derived.
        assert_eq!(count, 6);
        assert!(any_new);
        assert_eq!(
            space.btm.val_count(),
            9,
            "three edges plus the six-pair path closure"
        );

        // (path a d) is reachable only through three hops, so it is derived by the
        // recursion, not seeded. Removing it must drop the count, proving presence.
        space.load_all_sexpr_impl(b"(path a d)\n", false).unwrap();
        assert_eq!(
            space.btm.val_count(),
            8,
            "the three-hop transitive pair (path a d) was derived"
        );
    }

    #[ignore]
    #[test]
    fn bench_semi_naive_vs_iterated_product_zipper() {
        // Wall-clock comparison on a chain: the semi-naive fixpoint in one call
        // versus the ProductZipper emit run round by round to fixpoint. Run with
        // `--ignored --nocapture`.
        let body = "[3] , [3] edge $ $ [3] path _2 $";
        let template = "[2] , [3] path _1 _3";
        for n in [60usize, 120, 240] {
            let mut setup = String::new();
            for i in 0..n - 1 {
                setup.push_str(&format!("(edge n{i} n{})\n", i + 1));
                setup.push_str(&format!("(path n{i} n{})\n", i + 1));
            }

            let mut semi_naive = Space::new();
            semi_naive.add_all_sexpr(setup.as_bytes()).unwrap();
            let (pat_expr, _) = query_pattern_and_sources(&mut semi_naive, body);
            let (tpl_expr, _) = query_pattern_and_sources(&mut semi_naive, template);
            let read_copy = semi_naive.btm.clone();
            let start = Instant::now();
            semi_naive
                .transform_linear_recursive_fixpoint(&read_copy, pat_expr, tpl_expr)
                .unwrap();
            let semi_naive_ms = start.elapsed().as_secs_f64() * 1000.0;

            let mut product = Space::new();
            product.add_all_sexpr(setup.as_bytes()).unwrap();
            let (product_pat, _) = query_pattern_and_sources(&mut product, body);
            let (product_tpl, _) = query_pattern_and_sources(&mut product, template);
            let mut tpl_args = Vec::new();
            ExprEnv::new(0, product_tpl).args(&mut tpl_args);
            let templates: Vec<Expr> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
            let start = Instant::now();
            loop {
                let round = product.btm.clone();
                let outputs = Space::product_template_outputs(&round, product_pat, &templates);
                let mut any_new = false;
                for path in &outputs {
                    any_new |= product.btm.insert(&path[..], ()).is_none();
                }
                if !any_new {
                    break;
                }
            }
            let iterated_ms = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "n={n}: semi-naive {semi_naive_ms:.2}ms, iterated {iterated_ms:.2}ms, speedup {:.1}x",
                iterated_ms / semi_naive_ms
            );
        }
    }

    #[test]
    fn binary_transitive_closure_computes_the_closure() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(path a b)
(path b c)
(path c d)
"#,
            )
            .unwrap();
        // path(x,z) :- path(x,y), path(y,z): the non-linear (doubly recursive) TC.
        let (pat_expr, _) =
            query_pattern_and_sources(&mut space, "[3] , [3] path $ $ [3] path _2 $");
        let (tpl_expr, _) = query_pattern_and_sources(&mut space, "[2] , [3] path _1 _3");

        let read_copy = space.btm.clone();
        let (count, any_new) = space
            .transform_binary_transitive_closure(&read_copy, pat_expr, tpl_expr)
            .expect("the doubly-recursive rule is a transitive closure");
        assert_eq!(count, 6);
        assert!(any_new);
        assert_eq!(
            space.btm.val_count(),
            6,
            "the six-pair closure of a 3-chain"
        );

        space.load_all_sexpr_impl(b"(path a d)\n", false).unwrap();
        assert_eq!(
            space.btm.val_count(),
            5,
            "the three-hop pair (path a d) was derived"
        );
    }

    #[test]
    fn linear_recursive_fixpoint_matches_iterated_product_zipper() {
        // A branching graph (a-b-c-d and a-e-d) so the closure is non-trivial and
        // some pairs are derivable by more than one path.
        let setup = br#"
(edge a b)
(edge b c)
(edge c d)
(edge a e)
(edge e d)
(path a b)
(path b c)
(path c d)
(path a e)
(path e d)
"#;
        let body = "[3] , [3] edge $ $ [3] path _2 $";
        let template = "[2] , [3] path _1 _3";

        // The semi-naive fixpoint in one call.
        let mut semi_naive = Space::new();
        semi_naive.add_all_sexpr(setup).unwrap();
        let (pat_expr, _) = query_pattern_and_sources(&mut semi_naive, body);
        let (tpl_expr, _) = query_pattern_and_sources(&mut semi_naive, template);
        let read_copy = semi_naive.btm.clone();
        semi_naive
            .transform_linear_recursive_fixpoint(&read_copy, pat_expr, tpl_expr)
            .expect("the linear recursive rule lowers to a fixpoint");

        // The reference: the ProductZipper emit run round by round to fixpoint.
        let mut product = Space::new();
        product.add_all_sexpr(setup).unwrap();
        let (product_pat, _) = query_pattern_and_sources(&mut product, body);
        let (product_tpl, _) = query_pattern_and_sources(&mut product, template);
        let mut tpl_args = Vec::new();
        ExprEnv::new(0, product_tpl).args(&mut tpl_args);
        let templates: Vec<Expr> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
        loop {
            let round = product.btm.clone();
            let outputs = Space::product_template_outputs(&round, product_pat, &templates);
            let mut any_new = false;
            for path in &outputs {
                any_new |= product.btm.insert(&path[..], ()).is_none();
            }
            if !any_new {
                break;
            }
        }

        // The semi-naive closure equals the iterated ProductZipper closure exactly.
        let collect = |space: &Space| {
            let mut facts = BTreeSet::new();
            space
                .btm
                .try_for_each_value::<_, ()>(|path, _| {
                    facts.insert(path.to_vec());
                    Ok(())
                })
                .unwrap();
            facts
        };
        assert_eq!(collect(&semi_naive), collect(&product));
        assert!(
            semi_naive.btm.val_count() > 10,
            "the closure adds derived pairs"
        );
    }

    // ---- Stage 2 of the semi-naive delta lever: the m-delta-rule transform ----
    // (transform_multi_multi_delta). See kernel/resources/semi_naive_delta_design.md.

    // Local Peano encoder (the binary's `peano` is not in the lib).
    fn peano_sexpr(x: usize) -> String {
        if x == 0 {
            "Z".to_string()
        } else {
            format!("(S {})", peano_sexpr(x - 1))
        }
    }

    // The set of values in a space.
    fn collect_facts(btm: &PathMap<()>) -> BTreeSet<Vec<u8>> {
        let mut facts = BTreeSet::new();
        btm.try_for_each_value::<_, ()>(|path, _| {
            facts.insert(path.to_vec());
            Ok(())
        })
        .unwrap();
        facts
    }

    // The naive one-step ProductZipper emit set over `btm` (the oracle).
    fn naive_emit_set(
        space: &mut Space,
        btm: &PathMap<()>,
        body: &'static str,
        template: &'static str,
    ) -> BTreeSet<Vec<u8>> {
        let (pat_expr, _) = query_pattern_and_sources(space, body);
        let (tpl_expr, _) = query_pattern_and_sources(space, template);
        let mut tpl_args = Vec::new();
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<Expr> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
        Space::product_template_outputs(btm, pat_expr, &templates)
    }

    // The semi-naive m-delta-rule emit set: the facts transform_multi_multi_delta
    // writes into a space initialized to `post` (the full post-step space), given
    // `delta` (the per-step added facts). Returns the newly-written paths.
    fn delta_emit_set(
        post: &PathMap<()>,
        delta: &PathMap<()>,
        pat_expr: Expr,
        tpl_expr: Expr,
    ) -> BTreeSet<Vec<u8>> {
        let mut space = Space::new();
        space.btm = post.clone();
        let read_copy = space.btm.clone();
        let before = space.btm.clone();
        space.transform_multi_multi_delta(&read_copy, delta, pat_expr, tpl_expr);
        collect_facts(&space.btm.subtract(&before))
    }

    // Build (post-step space, pre-step space, delta) from sexpr blocks. `base` is
    // the pre-step facts, `added` is the per-step delta; post = base + added.
    fn build_step_spaces(
        base: &[u8],
        added: &[u8],
    ) -> (PathMap<()>, PathMap<()>, PathMap<()>) {
        let mut pre = Space::new();
        pre.add_all_sexpr(base).unwrap();
        let pre_btm = pre.btm.clone();

        let mut post = Space::new();
        post.add_all_sexpr(base).unwrap();
        post.add_all_sexpr(added).unwrap();
        let post_btm = post.btm.clone();

        let delta = post_btm.subtract(&pre_btm);
        (post_btm, pre_btm, delta)
    }

    // The core m-delta-rule invariant on a body, asserted two ways:
    //   (soundness)     delta_emit  is subset of  naive(post)
    //   (semi-naive)    delta_emit + naive(pre)  ==  naive(post)
    // The second is the exact statement the idempotent-insert fixpoint relies on:
    // matching one factor against the delta and the rest against the full space,
    // unioned over the mutated-relation factors, recovers precisely the facts that
    // complete the naive one-step result. When template instantiation is injective
    // on matches (`exact_difference`), the stronger design form also holds:
    //   delta_emit  ==  naive(post) \ naive(pre).
    fn assert_m_delta_invariant(
        base: &[u8],
        added: &[u8],
        body: &'static str,
        template: &'static str,
        exact_difference: bool,
    ) {
        let (post, pre, delta) = build_step_spaces(base, added);

        let mut space = Space::new();
        let (pat_expr, _) = query_pattern_and_sources(&mut space, body);
        let (tpl_expr, _) = query_pattern_and_sources(&mut space, template);

        let naive_pre = naive_emit_set(&mut space, &pre, body, template);
        let naive_post = naive_emit_set(&mut space, &post, body, template);
        let delta_out = delta_emit_set(&post, &delta, pat_expr, tpl_expr);

        // soundness: every delta-rule output is a real match over the full space
        assert!(
            delta_out.is_subset(&naive_post),
            "delta-rule emitted a fact the naive full match does not"
        );
        // the semi-naive recovery identity
        let recovered: BTreeSet<Vec<u8>> =
            delta_out.union(&naive_pre).cloned().collect();
        assert_eq!(
            recovered, naive_post,
            "delta_emit + naive(pre) must equal naive(post)"
        );
        if exact_difference {
            let naive_new: BTreeSet<Vec<u8>> =
                naive_post.difference(&naive_pre).cloned().collect();
            assert_eq!(
                delta_out, naive_new,
                "with injective output, delta_emit == naive(post) \\ naive(pre)"
            );
        }
    }

    #[test]
    fn m_delta_rule_per_step_invariant_ground() {
        // A ground non-linear (doubly-recursive) transitive-closure body over
        // `path`. The output relation is the same as the input (recursive), so
        // outputs can be re-derived old-and-new; the semi-naive recovery identity
        // is the right invariant (exact_difference would not hold in general).
        let base = br#"
(path a b)
(path b c)
(path c d)
"#;
        // Adding (path d e) opens new two-hop joins through d and e.
        let added = b"(path d e)\n";
        assert_m_delta_invariant(
            base,
            added,
            "[3] , [3] path $ $ [3] path _2 $",
            "[2] , [3] path _1 _3",
            false,
        );

        // A disjoint output relation makes template instantiation injective on the
        // (x,z) match, so the stronger design difference form holds exactly.
        assert_m_delta_invariant(
            base,
            added,
            "[3] , [3] path $ $ [3] path _2 $",
            "[2] , [3] reach _1 _3",
            true,
        );

        // A 3-factor body stresses the j-loop's secondary-position indexing (the
        // delta can sit at primary, secondary 0, or secondary 1). The three-hop
        // chain path X Y, path Y Z, path Z W emits tri X W. _1=X,_2=Y,_3=Z,_4=W;
        // factor1 = path _2 $ (Y,Z), factor2 = path _3 $ (Z,W).
        assert_m_delta_invariant(
            base,
            added,
            "[4] , [3] path $ $ [3] path _2 $ [3] path _3 $",
            "[2] , [3] tri _1 _4",
            true,
        );
    }

    #[test]
    fn m_delta_rule_per_step_invariant_schematic() {
        // The real process_calculus nesting: a receiver (petri (? chan payload
        // body)) joins a message (petri (! chan payload)) on the shared channel and
        // payload, emitting the receiver body. The facts carry variables (a
        // variable-bearing receiver), so the join does real unification, exactly
        // like (petri (? (add $ret) ...)). Both factors are over `petri` (the
        // mutated relation), so the m-delta-rule loops both.
        let base = br#"
(petri (? chan one (done one)))
(petri (! chan one))
"#;
        // The delta adds a second receiver+message pair on a fresh channel, with a
        // variable-bearing receiver body to exercise schematic unification on the
        // delta-restricted factor.
        let added = br#"
(petri (? other ($v) (got $v)))
(petri (! other (Z)))
"#;
        // body: (, (petri (? $chan $payload $body)) (petri (! $chan $payload)))
        //   _1=chan, _2=payload, _3=body; factor1 corefs chan/payload as _1/_2.
        // template: (petri $body)  ->  _3.
        assert_m_delta_invariant(
            base,
            added,
            "[3] , [2] petri [4] ? $ $ $ [2] petri [3] ! _1 _2",
            "[2] , [2] petri _3",
            false,
        );
    }

    #[test]
    fn m_delta_rule_preserves_free_data_vars_in_split_templates() {
        // The process_calculus |-split: a SINGLE-factor body over a schematic fact
        // whose second component carries a FREE data variable plus a back-reference
        // to it ((? chan $a (send (S $a)))), split by TWO templates. The IC-loop
        // divergence materialized the first component into the second's variable
        // slot on delta-routed firings; this pins the split emit byte-for-byte
        // against the naive full match.
        let base = br#"
(petri (| (msg a) (recv a)))
"#;
        let added = br#"
(petri (| (! chan pay) (? chan $a (send (S $a)))))
"#;
        // body: (, (petri (| $l $r))) -- m == 1; templates: (, (petri $l) (petri $r)).
        assert_m_delta_invariant(
            base,
            added,
            "[2] , [2] petri [3] | $ $",
            "[3] , [2] petri _1 [2] petri _2",
            false,
        );
    }

    // Regression guard for the path-buffer realloc bug in the VarRef recheck of
    // `coreferential_transition`, exposed by the Stage-4 reordered delta. The reorder
    // makes a factor's bound subterm the rechecked one; the recheck captured a raw
    // pointer into `loc.path()` (the ProductZipper's path buffer) and recursed while
    // holding it, so once the matched path grew past the reserved buffer
    // (`QUERY_PRODUCT_PATH_BUFFER_INITIAL_CAPACITY`, 4096) and reallocated, the pointer
    // dangled and the matcher read garbage tag bytes (a `byte_item` "reserved" panic).
    // The fix copies the bound subterm into an owned buffer before recursing.
    //
    // Reproducing needs the FULL `metta_calculus` IC loop: the deep
    // `(IC 0 1 peano(1000))` controller counter plus the add(n,n) cascade's deep
    // intermediate terms push the concatenated path past 4096 mid-recheck. n=175 is
    // just past the measured onset (n<=170 stays under the reserve). Pre-fix the
    // RELEASE binary panicked deterministically here; post-fix the semi-naive IC loop
    // is byte-identical to naive. Feature-gated because it drives `metta_calculus`.
    #[cfg(feature = "semi_naive_ic")]
    #[test]
    fn coreferential_recheck_survives_path_buffer_realloc_on_deep_terms() {
        // Parsing peano(1000) and matching the deep cascade recurse past the default
        // 2 MiB test stack, so run on a worker thread with a large stack (self-
        // contained: no RUST_MIN_STACK needed). This is a depth limit, not the bug.
        std::thread::Builder::new()
            .stack_size(1 << 31)
            .spawn(|| {
                fn build(setup: &str) -> Space {
                    let mut s = Space::new();
                    let pat = crate::expr!(s, "$");
                    let tpl = crate::expr!(s, "_1");
                    s.add_sexpr(setup.as_bytes(), pat, tpl).unwrap();
                    s
                }
                // naive reference: hand-drive the IC loop with the delta hook inert.
                let n = 175usize;
                let setup = process_calculus_setup_sexpr(1000, n, n);
                let mut naive = build(&setup);
                {
                    let mut exec_path = Vec::new();
                    while naive.take_first_exec_path(&mut exec_path) {
                        let xe = Expr { ptr: exec_path.as_mut_ptr() };
                        let _ = naive.interpret(xe);
                    }
                }
                // semi-naive: the wired loop through the reordered delta transform.
                let mut semi = build(&setup);
                semi.metta_calculus(1_000_000_000_000_000);

                let mut nv = Vec::new();
                naive.dump_all_sexpr(&mut nv).unwrap();
                let mut sv = Vec::new();
                semi.dump_all_sexpr(&mut sv).unwrap();
                assert_eq!(
                    String::from_utf8_lossy(&nv),
                    String::from_utf8_lossy(&sv),
                    "deep IC loop (n={n}): reordered semi-naive must be byte-identical to naive across the path-buffer realloc"
                );
                let want = format!("{}\n", peano_sexpr(2 * n));
                let pat = crate::expr!(semi, "[2] petri [3] ! result $");
                let tpl = crate::expr!(semi, "_1");
                let mut rv = Vec::new();
                semi.dump_sexpr(pat, tpl, &mut rv);
                assert_eq!(
                    String::from_utf8_lossy(&rv),
                    want,
                    "deep IC loop must compute add({n},{n}) = peano({})",
                    2 * n
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn semi_naive_delta_matches_naive_on_process_calculus() {
        // The load-bearing differential oracle. Drive a recursive `,`-rule to
        // fixpoint two ways and assert the final spaces are byte-identical:
        //   (a) naive: the ProductZipper emit (product_template_outputs) re-run
        //       round by round against the full accumulating space (this is what
        //       transform_multi_multi_ does each step);
        //   (b) semi-naive: transform_multi_multi_delta driven by the per-step
        //       delta maintained with delta_snapshot / delta_since.
        //
        // Run on TWO rules: a schematic transitive closure, and the actual
        // process_calculus petri reaction computing add(8,8) = peano(16).

        // Helper: run rule (body -> template) to fixpoint, naive round-by-round.
        fn naive_fixpoint(setup: &str, body: &'static str, template: &'static str) -> Space {
            let mut space = Space::new();
            space.add_all_sexpr(setup.as_bytes()).unwrap();
            let (pat_expr, _) = query_pattern_and_sources(&mut space, body);
            let (tpl_expr, _) = query_pattern_and_sources(&mut space, template);
            let mut tpl_args = Vec::new();
            ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
            let templates: Vec<Expr> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
            loop {
                let round = space.btm.clone();
                let outputs = Space::product_template_outputs(&round, pat_expr, &templates);
                let mut any_new = false;
                for path in &outputs {
                    any_new |= space.btm.insert(&path[..], ()).is_none();
                }
                if !any_new {
                    break;
                }
            }
            space
        }

        // Helper: run the same rule to fixpoint, semi-naive (delta-driven). The
        // first round matches the whole space (delta = whole space); thereafter the
        // delta is the facts the previous round added.
        fn semi_naive_fixpoint(setup: &str, body: &'static str, template: &'static str) -> Space {
            let mut space = Space::new();
            space.add_all_sexpr(setup.as_bytes()).unwrap();
            let (pat_expr, _) = query_pattern_and_sources(&mut space, body);
            let (tpl_expr, _) = query_pattern_and_sources(&mut space, template);

            // Round 0: the seed delta is the entire initial space (every fact is
            // "new" relative to the empty pre-state), matched in full.
            let mut delta = space.btm.clone();
            loop {
                let before = space.delta_snapshot();
                let read_copy = space.btm.clone();
                space.transform_multi_multi_delta(&read_copy, &delta, pat_expr, tpl_expr);
                let (added, _removed) = space.delta_since(&before);
                if added.val_count() == 0 {
                    break;
                }
                delta = added;
            }
            space
        }

        // Rule 1: schematic non-linear transitive closure over `path`.
        {
            let mut setup = String::from("(edge x y)\n");
            for i in 0..8 {
                setup.push_str(&format!("(path n{i} n{})\n", i + 1));
            }
            let body = "[3] , [3] path $ $ [3] path _2 $";
            let template = "[2] , [3] path _1 _3";
            let naive = naive_fixpoint(&setup, body, template);
            let semi = semi_naive_fixpoint(&setup, body, template);

            let mut nv = Vec::new();
            naive.dump_all_sexpr(&mut nv).unwrap();
            let mut sv = Vec::new();
            semi.dump_all_sexpr(&mut sv).unwrap();
            assert_eq!(
                nv, sv,
                "TC: semi-naive delta fixpoint must be byte-identical to naive"
            );
            assert!(naive.btm.val_count() > 9, "the closure adds derived pairs");
        }

        // Rule 2: the process_calculus petri reaction. The receiver (? chan payload
        // body) reacts with the message (! chan payload) and emits the body. We seed
        // an `add` computation and let the reaction cascade to the Peano result.
        // This is the exact rule from process_calculus_bench (main.rs:254-257), and
        // the add encoding from main.rs:263-266, computing add(8,8) = peano(16).
        {
            let setup = format!(
                r#"
(petri (? (add $ret) ((S $x) $y) (| (! (add (PN $x $y)) ($x $y))
                                    (? (PN $x $y) $z (! $ret (S $z)))  )  ))
(petri (? (add $ret) (Z $y) (! $ret $y)))
(petri (! (add result) ({} {})))
"#,
                peano_sexpr(8),
                peano_sexpr(8)
            );
            // The reaction rule (exec 0): a receiver (petri (? chan payload body))
            // joins a message (petri (! chan payload)) on the shared channel and
            // payload, emitting the receiver body. Nested arity-prefix encoding:
            // _1=channel, _2=payload, _3=body (NewVars in appearance order); factor1
            // corefs channel/payload as _1/_2.
            let body = "[3] , [2] petri [4] ? $ $ $ [2] petri [3] ! _1 _2";
            let template = "[2] , [2] petri _3";

            // The split rule (exec 1) breaks a `|` process into its two halves. To
            // reach the arithmetic fixpoint we apply BOTH the reaction and the split
            // (both are monotone add-only here, so interleaving them to a joint
            // fixpoint computes the same result the IC controller would).
            let split_body = "[2] , [2] petri [3] | $ $";
            let split_template = "[3] , [2] petri _1 [2] petri _2";

            // naive: interleave reaction-fixpoint and split-fixpoint, naive each.
            let naive = {
                let mut space = Space::new();
                space.add_all_sexpr(setup.as_bytes()).unwrap();
                let (rpat, _) = query_pattern_and_sources(&mut space, body);
                let (rtpl, _) = query_pattern_and_sources(&mut space, template);
                let (spat, _) = query_pattern_and_sources(&mut space, split_body);
                let (stpl, _) = query_pattern_and_sources(&mut space, split_template);
                let mut rtpl_args = Vec::new();
                ExprEnv::new(0, rtpl).args(&mut rtpl_args);
                let rtemplates: Vec<Expr> =
                    rtpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
                let mut stpl_args = Vec::new();
                ExprEnv::new(0, stpl).args(&mut stpl_args);
                let stemplates: Vec<Expr> =
                    stpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();
                loop {
                    let mut any_new = false;
                    let round = space.btm.clone();
                    for path in &Space::product_template_outputs(&round, rpat, &rtemplates) {
                        any_new |= space.btm.insert(&path[..], ()).is_none();
                    }
                    let round = space.btm.clone();
                    for path in &Space::product_template_outputs(&round, spat, &stemplates) {
                        any_new |= space.btm.insert(&path[..], ()).is_none();
                    }
                    if !any_new {
                        break;
                    }
                }
                space
            };

            // semi-naive: same interleave, but each rule's match is delta-restricted
            // via transform_multi_multi_delta with a maintained per-rule delta.
            let semi = {
                let mut space = Space::new();
                space.add_all_sexpr(setup.as_bytes()).unwrap();
                let (rpat, _) = query_pattern_and_sources(&mut space, body);
                let (rtpl, _) = query_pattern_and_sources(&mut space, template);
                let (spat, _) = query_pattern_and_sources(&mut space, split_body);
                let (stpl, _) = query_pattern_and_sources(&mut space, split_template);
                // Seed both deltas with the whole space.
                loop {
                    let before = space.delta_snapshot();
                    let delta = before.clone();
                    let read_copy = space.btm.clone();
                    space.transform_multi_multi_delta(&read_copy, &delta, rpat, rtpl);
                    let read_copy = space.btm.clone();
                    let delta2 = space.btm.clone();
                    space.transform_multi_multi_delta(&read_copy, &delta2, spat, stpl);
                    let (added, _removed) = space.delta_since(&before);
                    if added.val_count() == 0 {
                        break;
                    }
                }
                space
            };

            let mut nv = Vec::new();
            naive.dump_all_sexpr(&mut nv).unwrap();
            let mut sv = Vec::new();
            semi.dump_all_sexpr(&mut sv).unwrap();
            assert_eq!(
                String::from_utf8_lossy(&nv),
                String::from_utf8_lossy(&sv),
                "petri reaction: semi-naive delta fixpoint must be byte-identical to naive"
            );

            // The cascade computes add(8,8) = 16, landing on the `result` channel as
            // (petri (! result peano(16))). Assert the arithmetic answer is present
            // in BOTH the naive and semi-naive dishes (byte-identical above implies
            // both, but check explicitly so a vacuous match can never pass).
            let want = format!("(petri (! result {}))", peano_sexpr(16));
            assert!(
                String::from_utf8_lossy(&nv).contains(&want),
                "naive petri reaction must compute add(8,8) = peano(16) on `result`"
            );
            assert!(
                String::from_utf8_lossy(&sv).contains(&want),
                "semi-naive petri reaction must compute add(8,8) = peano(16) on `result`"
            );
        }
    }

    // The exact process_calculus_bench setup (main.rs:242-271): the IC driver, the
    // petri reaction `(exec 0)`, the split `(exec 1)`, and the add(x,y) seed. Built
    // through add_sexpr with the same `$`/`_1` pattern so the encoded space matches
    // the binary's byte for byte.
    #[cfg(feature = "semi_naive_ic")]
    fn process_calculus_setup_sexpr(steps: usize, x: usize, y: usize) -> String {
        format!(
            r#"
(exec (IC 0 1 {})
               (, (exec (IC $x $y (S $c)) $sp $st)
                  ((exec $x) $p $t))
               (, (exec (IC $y $x $c) $sp $st)
                  (exec (R $x) $p $t)))

((exec 0)
      (, (petri (? $channel $payload $body))
         (petri (! $channel $payload)) )
      (, (petri $body)))
((exec 1)
      (, (petri (| $lprocess $rprocess)))
      (, (petri $lprocess)
         (petri $rprocess)))

(petri (? (add $ret) ((S $x) $y) (| (! (add (PN $x $y)) ($x $y))
                                    (? (PN $x $y) $z (! $ret (S $z)))  )  ))
(petri (? (add $ret) (Z $y) (! $ret $y)))
(petri (! (add result) ({} {})))
    "#,
            peano_sexpr(steps),
            peano_sexpr(x),
            peano_sexpr(y)
        )
    }

    // Stage 3 load-bearing differential oracle. Run the FULL process_calculus IC
    // loop both ways to the same (unbounded) step budget and assert the final dish
    // is BYTE-IDENTICAL:
    //   naive: drive `take_first_exec_path` + `interpret` by hand, never touching
    //          `sni_delta` (it stays None, so transform_multi_multi_ takes the
    //          naive full-space branch every step -- exactly the feature-off loop);
    //   semi:  `metta_calculus`, which maintains the per-step delta and routes
    //          every `,`-rule through transform_multi_multi_delta.
    // Both must land peano(x+y) on the `result` channel. If the wiring or delta
    // maintenance is wrong the dishes diverge and the dump assertion fails.
    #[cfg(feature = "semi_naive_ic")]
    #[test]
    fn metta_calculus_semi_naive_matches_naive() {
        fn build(setup: &str) -> Space {
            let mut s = Space::new();
            let pat = crate::expr!(s, "$");
            let tpl = crate::expr!(s, "_1");
            s.add_sexpr(setup.as_bytes(), pat, tpl).unwrap();
            s
        }

        // Naive reference: the IC loop with the delta hook inert (sni_delta None).
        fn run_naive(setup: &str) -> Space {
            let mut s = build(setup);
            let mut exec_path = Vec::new();
            // sni_delta stays None for the whole run, so every transform_multi_multi_
            // takes the naive full-space path: this is the feature-off loop.
            while s.take_first_exec_path(&mut exec_path) {
                let xe = Expr {
                    ptr: exec_path.as_mut_ptr(),
                };
                let _ = s.interpret(xe);
                debug_assert!(s.sni_rule_seen.is_none());
            }
            s
        }

        // Semi-naive: the wired loop maintaining the per-step delta.
        fn run_semi(setup: &str) -> Space {
            let mut s = build(setup);
            s.metta_calculus(1_000_000_000_000_000);
            s
        }

        // Project the `result` channel the way the bench does (main.rs:283).
        fn project_result(s: &mut Space) -> String {
            let pat = crate::expr!(s, "[2] petri [3] ! result $");
            let tpl = crate::expr!(s, "_1");
            let mut v = Vec::new();
            s.dump_sexpr(pat, tpl, &mut v);
            String::from_utf8_lossy(&v).into_owned()
        }

        for (x, y) in [(8usize, 8usize), (16, 16)] {
            // The IC round budget (the `(IC 0 1 peano(steps))` counter) must exceed
            // the add(x,y) cascade depth. add(n,n) finishes in O(n) reaction rounds,
            // so 100 is ample for x=y up to 16. Kept small because the test thread's
            // stack cannot parse a peano(1000) literal (the bench runs that in main).
            let setup = process_calculus_setup_sexpr(100, x, y);
            let mut naive = run_naive(&setup);
            let mut semi = run_semi(&setup);

            let mut nv = Vec::new();
            naive.dump_all_sexpr(&mut nv).unwrap();
            let mut sv = Vec::new();
            semi.dump_all_sexpr(&mut sv).unwrap();
            assert_eq!(
                String::from_utf8_lossy(&nv),
                String::from_utf8_lossy(&sv),
                "process_calculus x={x} y={y}: semi-naive IC loop must be byte-identical to naive"
            );

            // The result channel must hold peano(x+y) in BOTH dishes (byte-identity
            // implies it, but assert explicitly so a vacuous empty-dish match can
            // never pass).
            let want = format!("{}\n", peano_sexpr(x + y));
            assert_eq!(
                project_result(&mut naive),
                want,
                "naive process_calculus must compute add({x},{y}) = peano({})",
                x + y
            );
            assert_eq!(
                project_result(&mut semi),
                want,
                "semi-naive process_calculus must compute add({x},{y}) = peano({})",
                x + y
            );
        }
    }

    #[cfg(feature = "semi_naive_ic")]
    #[test]
    fn semi_naive_ic_lockstep_with_naive_per_step() {
        // Per-STEP oracle: hand-drive the naive loop and an armed semi loop in
        // lockstep and compare the whole dish after every exec step. The
        // end-to-end oracle only sees the final dish; this pins a divergence to
        // the first step that produced it, with the offending facts named.
        fn build(setup: &str) -> Space {
            let mut s = Space::new();
            let pat = crate::expr!(s, "$");
            let tpl = crate::expr!(s, "_1");
            s.add_sexpr(setup.as_bytes(), pat, tpl).unwrap();
            s
        }
        for (x, y) in [(3usize, 3usize), (8, 8)] {
            let setup = process_calculus_setup_sexpr(100, x, y);
            let mut naive = build(&setup);
            let mut semi = build(&setup);
            // Arm the semi space exactly as metta_calculus does for its loop.
            semi.sni_rule_seen = Some(HashMap::new());
            semi.sni_removal_seen = false;
            semi.sni_removal_gen = 0;
            semi.sni_dred_repairs = 0;
            semi.sni_dred_fallbacks = 0;

            let mut exec_path = Vec::new();
            let mut step = 0usize;
            loop {
                step += 1;
                let n_has = naive.take_first_exec_path(&mut exec_path);
                if n_has {
                    let xe = Expr {
                        ptr: exec_path.as_mut_ptr(),
                    };
                    let _ = naive.interpret(xe);
                }
                let s_has = semi.take_first_exec_path(&mut exec_path);
                if s_has {
                    let xe = Expr {
                        ptr: exec_path.as_mut_ptr(),
                    };
                    let _ = semi.interpret(xe);
                }
                assert_eq!(n_has, s_has, "x={x} y={y} step {step}: one loop ran dry first");
                let mut nv = Vec::new();
                naive.dump_all_sexpr(&mut nv).unwrap();
                let mut sv = Vec::new();
                semi.dump_all_sexpr(&mut sv).unwrap();
                if nv != sv {
                    let ns: BTreeSet<String> =
                        String::from_utf8_lossy(&nv).lines().map(|l| l.to_string()).collect();
                    let ss: BTreeSet<String> =
                        String::from_utf8_lossy(&sv).lines().map(|l| l.to_string()).collect();
                    let only_n: Vec<String> = ns.difference(&ss).take(4).cloned().collect();
                    let only_s: Vec<String> = ss.difference(&ns).take(4).cloned().collect();
                    // Raw byte layouts decide variable-numbering questions the
                    // s-expressions cannot: dump the consumed exec, every pre-step
                    // `(| ...)`-carrying fact (the split's input, identical in both
                    // engines a step ago), and both engines' `?`-carrying outputs.
                    let contains = |hay: &[u8], needle: &[u8]| {
                        hay.windows(needle.len()).any(|w| w == needle)
                    };
                    let mut raw = String::new();
                    raw.push_str(&format!("exec: {:02x?}\n", &exec_path[..]));
                    for (label, space) in [("naive", &naive), ("semi", &semi)] {
                        space.btm.for_each_value(|p, _| {
                            if contains(p, &[0x03, 0xc1, b'|']) || contains(p, &[0x04, 0xc1, b'?'])
                            {
                                raw.push_str(&format!("{label} {}\n  = {p:02x?}\n", serialize(p)));
                            }
                        });
                    }
                    panic!(
                        "x={x} y={y}: first divergence at step {step}\nonly naive:\n  {}\nonly semi:\n  {}\n{raw}",
                        only_n.join("\n  "),
                        only_s.join("\n  ")
                    );
                }
                if !n_has {
                    break;
                }
            }
        }
    }

    #[test]
    fn semi_naive_avoids_the_iterated_product_zipper_redundancy() {
        // Iterating a recursive rule round by round re-joins the whole relation
        // every round, re-deriving the same closure facts many times. Semi-naive
        // joins only the previous round's delta, so each fact is derived about
        // once. On a chain the redundancy ratio (iterated work over closure size)
        // grows with the chain length, the asymptotic win.
        let body = "[3] , [3] edge $ $ [3] path _2 $";
        let template = "[2] , [3] path _1 _3";
        let mut ratios = Vec::new();
        for n in [8usize, 16, 32] {
            let mut setup = String::new();
            for i in 0..n - 1 {
                setup.push_str(&format!("(edge n{i} n{})\n", i + 1));
                setup.push_str(&format!("(path n{i} n{})\n", i + 1));
            }
            let mut space = Space::new();
            space.add_all_sexpr(setup.as_bytes()).unwrap();
            let (pat_expr, _) = query_pattern_and_sources(&mut space, body);
            let (tpl_expr, _) = query_pattern_and_sources(&mut space, template);
            let mut tpl_args = Vec::new();
            ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
            let templates: Vec<Expr> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();

            // Iterated ProductZipper: total candidate emits across all rounds.
            let mut iterated_emits = 0usize;
            loop {
                let round = space.btm.clone();
                let outputs = Space::product_template_outputs(&round, pat_expr, &templates);
                iterated_emits += outputs.len();
                let mut any_new = false;
                for path in &outputs {
                    any_new |= space.btm.insert(&path[..], ()).is_none();
                }
                if !any_new {
                    break;
                }
            }
            // The closure size is the irreducible work semi-naive does.
            let closure = n * (n - 1) / 2;
            ratios.push(iterated_emits as f64 / closure as f64);
        }
        // The redundancy ratio grows with the chain length.
        eprintln!("iterated/closure redundancy ratios (n=8,16,32): {ratios:?}");
        assert!(
            ratios[0] < ratios[1] && ratios[1] < ratios[2],
            "iterated/closure redundancy should grow: {ratios:?}"
        );
    }

    #[cfg(feature = "semi_naive_fixpoint")]
    #[test]
    fn metta_calculus_computes_linear_recursion_in_one_step() {
        // The live path: an exec'd linear-recursive rule computes its whole
        // closure in a single exec step through the semi_naive_fixpoint hook in
        // transform_multi_multi_, instead of one round per re-firing.
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge c d)
(path a b)
(path b c)
(path c d)
(exec 0 (, (edge $x $y) (path $y $z)) (, (path $x $z)))
"#,
            )
            .unwrap();
        assert_eq!(space.metta_calculus(1), 1);
        // The three-hop pair (path a d) is reachable only by the recursion, so its
        // presence proves the fixpoint (not one round) ran in the single step.
        let before = space.btm.val_count();
        space.load_all_sexpr_impl(b"(path a d)\n", false).unwrap();
        assert_eq!(
            space.btm.val_count(),
            before - 1,
            "(path a d) was derived in a single exec step"
        );
    }

    #[cfg(feature = "semi_naive_fixpoint")]
    #[test]
    fn metta_calculus_maintains_streaming_closure() {
        // The streaming path maintains the closure across exec steps: after the
        // first fire builds it, a second fire that adds one edge folds in only
        // that edge and propagates it, rather than recomputing.
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(path a b)
(path b c)
(exec 0 (, (edge $x $y) (path $y $z)) (, (path $x $z)))
"#,
            )
            .unwrap();
        assert_eq!(space.metta_calculus(1), 1);

        // Stream a new edge c->d and re-fire.
        space
            .add_all_sexpr(
                br#"
(edge c d)
(exec 1 (, (edge $x $y) (path $y $z)) (, (path $x $z)))
"#,
            )
            .unwrap();
        assert_eq!(space.metta_calculus(1), 1);

        // (path a d) is reachable only through the streamed edge (a->b->c->d), so
        // its presence proves the second fire folded c->d into the maintained
        // closure and propagated it.
        let before = space.btm.val_count();
        space.load_all_sexpr_impl(b"(path a d)\n", false).unwrap();
        assert_eq!(
            space.btm.val_count(),
            before - 1,
            "(path a d) maintained after streaming c->d"
        );
    }

    // End-to-end benchmark: compute the full transitive closure of an n-edge
    // chain through metta_calculus, driving exec rounds to fixpoint, timed. Run
    // it both ways and compare:
    //   cargo +nightly test -p mork --lib bench_transitive_closure_metta -- --ignored --nocapture
    //   cargo +nightly test -p mork --lib --features semi_naive_fixpoint,sidecar_bridge_emit \
    //       bench_transitive_closure_metta -- --ignored --nocapture
    // With the fixpoint feature the first exec closes the whole relation in one
    // round; the default path needs one round per hop.
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture, default vs the fixpoint feature"]
    fn bench_transitive_closure_metta() {
        use std::time::Instant;
        for &n in &[50usize, 100, 200] {
            let mut space = Space::new();
            let mut program = String::new();
            for i in 0..n {
                program.push_str(&format!("(edge n{i} n{})\n(path n{i} n{})\n", i + 1, i + 1));
            }
            space.add_all_sexpr(program.as_bytes()).unwrap();

            let start = Instant::now();
            let mut exec_id = 0u64;
            let mut rounds = 0u64;
            loop {
                let before = space.btm.val_count();
                let rule =
                    format!("(exec {exec_id} (, (edge $x $y) (path $y $z)) (, (path $x $z)))\n");
                space.add_all_sexpr(rule.as_bytes()).unwrap();
                space.metta_calculus(1);
                exec_id += 1;
                rounds += 1;
                if space.btm.val_count() == before || rounds > 100_000 {
                    break;
                }
            }
            let elapsed = start.elapsed();
            // The chain's closure has n*(n+1)/2 path pairs; with the n edges and
            // the consumed exec facts the btm settles around that plus the edges.
            eprintln!(
                "n={n:4} rounds={rounds:6} btm_val_count={:8} elapsed={elapsed:?}",
                space.btm.val_count()
            );
        }
    }

    // End-to-end benchmark of the other fast path: a cyclic conjunctive join
    // (triangle enumeration) over a directed clique, run through metta_calculus.
    // The default takes the ProductZipper join-at-a-time walk; with
    // sidecar_bridge_emit the cyclic body routes to the worst-case-optimal join.
    //   cargo +nightly test -p mork --lib bench_triangle_join_metta -- --ignored --nocapture
    //   cargo +nightly test -p mork --lib --features sidecar_bridge_emit \
    //       bench_triangle_join_metta -- --ignored --nocapture
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture, default vs the flip feature"]
    fn bench_triangle_join_metta() {
        use std::time::Instant;
        for &k in &[20usize, 30, 40] {
            let mut space = Space::new();
            let mut program = String::new();
            for i in 0..k {
                for j in 0..k {
                    if i != j {
                        program.push_str(&format!("(edge n{i} n{j})\n"));
                    }
                }
            }
            space.add_all_sexpr(program.as_bytes()).unwrap();

            let start = Instant::now();
            space
                .add_all_sexpr(
                    b"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $x)) (, (tri $x $y $z)))\n",
                )
                .unwrap();
            space.metta_calculus(1);
            let elapsed = start.elapsed();
            eprintln!(
                "k={k:3} edges={:6} btm_val_count={:8} elapsed={elapsed:?}",
                k * (k - 1),
                space.btm.val_count()
            );
        }
    }

    // The intermediate-bound case the WCO join is built for: a hub graph with
    // many two-paths but few triangles. n peripheral nodes each connect both ways
    // to 3 hub nodes (a hub clique), so two-paths p->h->p' number ~3n^2 while
    // triangles stay ~6n. The ProductZipper materializes the n^2 two-paths; the
    // worst-case-optimal join (sidecar_bridge_emit) intersects instead, so its
    // edge over the default grows with n rather than staying a constant factor.
    //   cargo +nightly test -p mork --lib bench_triangle_sparse_join_metta -- --ignored --nocapture
    //   cargo +nightly test -p mork --lib --features sidecar_bridge_emit \
    //       bench_triangle_sparse_join_metta -- --ignored --nocapture
    #[test]
    fn varref_fast_recheck_matches_recursive_path() {
        // The VarRef ground fast-path (WAM unify_value by byte comparison) must
        // produce the SAME result set as the recursive re-match, on coreferential
        // rewrites over varied data. The fast path drops duplicate re-matches, so
        // the unification COUNT may differ; the result SET must be identical.
        let programs: &[&str] = &[
            // value coreference: a,b share value 1
            "(e a 1)\n(e b 1)\n(e c 2)\n(exec 0 (, (e $x $v) (e $y $v)) (, (sib $x $y)))\n",
            // compound-value coreference (the value is a nested term)
            "(e a (p 1 2))\n(e b (p 1 2))\n(e c (p 3 4))\n(exec 0 (, (e $x $v) (e $y $v)) (, (sib $x $y)))\n",
            // three-factor coreference (a variable repeated across 3 factors)
            "(t a m)\n(t b m)\n(t c m)\n(u m)\n(exec 0 (, (t $x $k) (t $y $k) (u $k)) (, (tri $x $y)))\n",
            // nested coreference over a larger ground value space
            "(k1 x (s (s z)))\n(k2 x (s (s z)))\n(k1 y (s z))\n(k2 y (s (s z)))\n(exec 0 (, (k1 $a $v) (k2 $a $v)) (, (matched $a)))\n",
            // duplicate-prone: the same value reachable several ways
            "(e a 1)\n(e a 1)\n(e b 1)\n(exec 0 (, (e $x $v) (e $y $v)) (, (sib $x $y)))\n",
            // counter_machine shape: a variable ($ts) coreferenced across THREE
            // factors with intervening factors, over nested peano data
            "(state Z (ic i0))\n(state Z (reg r0 (S Z)))\n(state Z (reg r1 (S (S Z))))\n(prog i0 step)\n(exec 0 (, (state $ts (ic $i)) (prog $i $op) (state $ts (reg $r $v)) (state $ts (reg $k $kv))) (, (fired $ts $i $r $k)))\n",
            // coreference where the repeated variable wraps in a constructor in a
            // later occurrence, like counter_machine's `(S $ts)` / `(S $i)`
            "(c a (S Z))\n(c b (S Z))\n(d (S Z))\n(exec 0 (, (c $x $v) (c $y $v) (d $v)) (, (g $x $y)))\n",
            // iterated coreference machine: derives across several steps, the
            // coreference re-check fires deep in the recursion each round
            "(n a Z)\n(n b Z)\n(n a (S Z))\n(n b (S Z))\n(exec 0 (, (n $x $t) (n $y $t)) (, (same $x $y $t)))\n",
            // THE NON-GROUND-DATA WITNESS (Verus Theorem 3): the data at a
            // coreferenced position is itself a variable. `$v` binds to `1` at
            // `(rel a 1)`, then is re-checked at `(rel2 $y $v)` against `(rel2 b $w)`
            // whose value slot is the variable `$w`. The recursive matcher matches
            // (a data variable is a wildcard); a naive byte comparison would MISS
            // it. This is the case the fast path must handle (fall back) or be
            // gated against.
            "(rel a 1)\n(rel2 b $w)\n(exec 0 (, (rel $x $v) (rel2 $y $v)) (, (m $x $y)))\n",
            // nested variable in data at the coreferenced slot
            "(p k (S Z))\n(q (S $u))\n(exec 0 (, (p $a $v) (q $v)) (, (r $a)))\n",
        ];
        for prog in programs {
            let mut fast = Space::new();
            fast.add_all_sexpr(prog.as_bytes()).unwrap();
            let mut slow = Space::new();
            slow.add_all_sexpr(prog.as_bytes()).unwrap();

            super::VARREF_FAST_RECHECK.with(|c| c.set(true));
            fast.metta_calculus(50);
            super::VARREF_FAST_RECHECK.with(|c| c.set(false));
            slow.metta_calculus(50);
            super::VARREF_FAST_RECHECK.with(|c| c.set(true));

            let mut fb = Vec::new();
            fast.dump_all_sexpr(&mut fb).unwrap();
            let mut sb = Vec::new();
            slow.dump_all_sexpr(&mut sb).unwrap();
            assert_eq!(
                String::from_utf8_lossy(&fb),
                String::from_utf8_lossy(&sb),
                "VarRef fast path diverged from the recursive re-match on:\n{prog}"
            );
        }
    }

    /// Runs `dump_sexpr(pat, pat)` over `facts`, returning the matched results in
    /// emission (visit) order so order divergence is observable, not just set
    /// divergence. `interpreted` selects site 2 (`coreferential_transition`) vs
    /// site 1 (`match_any_term`); `fused` selects the fused word-walk vs the
    /// three-pass `ByteMask::and` loops.
    fn match_any_lines_ordered(
        facts: &[&str],
        query: &str,
        interpreted: bool,
        fused: bool,
    ) -> Vec<String> {
        let prev_interp = super::FORCE_INTERPRETED_MATCHER.with(|c| c.get());
        let prev_fused = super::MATCH_ANY_TERM_FUSED.with(|c| c.get());
        super::FORCE_INTERPRETED_MATCHER.with(|c| c.set(interpreted));
        super::MATCH_ANY_TERM_FUSED.with(|c| c.set(fused));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut space = Space::new();
            insert_matcher_facts(&mut space, facts.iter().copied());
            let pat = crate::expr!(space, query);
            let mut out = Vec::new();
            space.dump_sexpr(pat, pat, &mut out);
            String::from_utf8(out)
                .unwrap()
                .lines()
                .filter(|line| !line.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        }));
        super::FORCE_INTERPRETED_MATCHER.with(|c| c.set(prev_interp));
        super::MATCH_ANY_TERM_FUSED.with(|c| c.set(prev_fused));
        match result {
            Ok(lines) => lines,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Differential oracle for the fused match-any-term word-walk. For each case
    /// it runs the matcher four ways — {compiled site 1, interpreted site 2} x
    /// {fused, three-pass} — and asserts the fused path produces a byte-identical
    /// result list IN THE SAME ORDER as the three-pass path, on both compiled and
    /// interpreted matchers. The data spans a "match any term" (`$`) over: ground
    /// symbols and compounds, deep nesting, high arity, repeated coreferential
    /// variables ($x..$x), VarRefs at the byte boundary, and several tag classes
    /// (Arity / SymbolSize / VarRef / NewVar) sharing one trie node so all three
    /// fused phases fire at a single child mask.
    #[test]
    fn match_any_fused_equals_three_pass() {
        // (facts, query, label). The query has a bare `$` so the NewVar
        // match-any-term descent runs; mixed-tag-class facts force the VARS,
        // SIZES, and ARITIES phases to all fire at the shared `[2] q _` node.
        let cases: &[(&[&str], &str, &str)] = &[
            // Single child of each class, plus several together, under `[2] q $`.
            (
                &[
                    "[2] q foo",           // SymbolSize child
                    "[2] q [2] S Z",       // Arity-2 child
                    "[2] q [3] t a b",     // Arity-3 child
                    "[2] q x",             // another SymbolSize
                    "[2] q [2] K [2] S Z", // nested arity
                ],
                "[2] q $",
                "mixed tag classes at one node",
            ),
            // Deep nesting: the match-any-term recurses through many arity frames.
            (
                &[
                    "[2] q [2] S [2] S [2] S [2] S [2] S [2] S [2] S [2] S Z",
                    "[2] q [2] S [2] S Z",
                    "[2] q Z",
                ],
                "[2] q $",
                "deep nesting",
            ),
            // High arity at the matched node (arity 12) alongside small arities.
            (
                &[
                    "[2] q [12] t a b c d e f g h i j k",
                    "[2] q [2] u a",
                    "[2] q [6] v a b c d e",
                ],
                "[2] q $",
                "high arity",
            ),
            // Repeated coreferential variable in the query ($x .. $x): exercises
            // the VarRef recheck after a match-any-term capture.
            (
                &[
                    "[3] p [2] S Z [2] S Z",
                    "[3] p [2] S Z [2] K Z",
                    "[3] p a a",
                    "[3] p a b",
                    "[3] p [3] big m n [3] big m n",
                ],
                "[3] p $ _1",
                "repeated coreferential variable",
            ),
            // Many distinct SymbolSize children at the matched node (symbols of
            // different lengths -> different SymbolSize bytes in word 3).
            (
                &[
                    "[2] q a",
                    "[2] q bb",
                    "[2] q ccc",
                    "[2] q dddddddddd",
                    "[2] q eeeeeeeeeeeeeeeeeeee",
                    "[2] q [2] S Z",
                    "[2] q [4] w 1 2 3",
                ],
                "[2] q $",
                "many symbol sizes plus arities",
            ),
            // Match-any-term as the whole query: every fact's first item is the
            // matched term, so the top-level child mask mixes arities and sizes.
            (
                &[
                    "foo",
                    "[2] a b",
                    "[3] c d e",
                    "barbar",
                    "[1] z",
                    "[5] m n o p q",
                ],
                "$",
                "top-level match-any over mixed roots",
            ),
            // Coreference where the repeated value is a deep compound, so the
            // recheck walks a long ground subterm after the match-any capture.
            (
                &[
                    "[3] e [3] long [2] S Z [2] T Q [3] long [2] S Z [2] T Q",
                    "[3] e a a",
                    "[3] e [3] long [2] S Z [2] T Q a",
                ],
                "[3] e $ _1",
                "coreference over deep ground compound",
            ),
        ];

        for (facts, query, label) in cases {
            for &interpreted in &[false, true] {
                let fused = match_any_lines_ordered(facts, query, interpreted, true);
                let three_pass = match_any_lines_ordered(facts, query, interpreted, false);
                let mode = if interpreted {
                    "interpreted (coreferential_transition)"
                } else {
                    "compiled (match_any_term)"
                };
                assert_eq!(
                    fused, three_pass,
                    "fused match-any-term diverged from three-pass on {label} [{mode}]\n\
                     fused:      {fused:?}\n\
                     three-pass: {three_pass:?}"
                );
            }
            // Cross-check: with the fused path on, the compiled and interpreted
            // matchers must also agree (the two sites must stay in lockstep).
            let compiled = match_any_lines_ordered(facts, query, false, true);
            let interpreted = match_any_lines_ordered(facts, query, true, true);
            let mut cs = compiled.clone();
            let mut is = interpreted.clone();
            cs.sort();
            is.sort();
            assert_eq!(
                cs, is,
                "fused compiled vs interpreted match-any-term diverged on {label}\n\
                 compiled:    {compiled:?}\n\
                 interpreted: {interpreted:?}"
            );
        }
    }

    fn insert_matcher_facts<'a>(space: &mut Space, facts: impl IntoIterator<Item = &'a str>) {
        for fact in facts {
            let expr = crate::expr!(space, fact);
            let bytes = unsafe { expr.span().as_ref().unwrap() }.to_vec();
            let mut wz = space.btm.write_zipper_at_path(&bytes);
            wz.set_value(());
        }
    }

    fn expr_string(space: &Space, expr: Expr) -> String {
        let mut out = Vec::new();
        expr.serialize2_with(
            &mut out,
            |s, out| {
                write_serialized_symbol(&space.sm, s, out);
            },
            |i, _intro| Expr::VARNAMES[i as usize],
        );
        String::from_utf8(out).unwrap()
    }

    struct ForceInterpretedMatcherGuard {
        previous: bool,
    }

    impl ForceInterpretedMatcherGuard {
        fn new(interpreted: bool) -> Self {
            let previous = super::FORCE_INTERPRETED_MATCHER.with(|c| {
                let previous = c.get();
                c.set(interpreted);
                previous
            });
            Self { previous }
        }
    }

    impl Drop for ForceInterpretedMatcherGuard {
        fn drop(&mut self) {
            super::FORCE_INTERPRETED_MATCHER.with(|c| c.set(self.previous));
        }
    }

    fn matcher_count_and_set_from_iter<'a>(
        facts: impl IntoIterator<Item = &'a str>,
        query: &str,
        interpreted: bool,
        require_compiled: bool,
    ) -> (usize, BTreeSet<String>) {
        let _guard = ForceInterpretedMatcherGuard::new(interpreted);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut space = Space::new();
            insert_matcher_facts(&mut space, facts);
            let pat = crate::expr!(space, query);
            if require_compiled && !interpreted {
                let mut args = Vec::new();
                ExprEnv::new(0, pat).args(&mut args);
                assert!(
                    compile_match_program(&args[1..]).is_some(),
                    "query should be accepted by the compiled matcher: {query}"
                );
            }

            let mut out = Vec::new();
            let count = space.dump_sexpr(pat, pat, &mut out);
            let set = String::from_utf8(out)
                .unwrap()
                .lines()
                .filter(|line| !line.is_empty())
                .map(str::to_owned)
                .collect();
            (count, set)
        }));
        match result {
            Ok(set) => set,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn matcher_set_from_iter<'a>(
        facts: impl IntoIterator<Item = &'a str>,
        query: &str,
        interpreted: bool,
    ) -> BTreeSet<String> {
        matcher_count_and_set_from_iter(facts, query, interpreted, false).1
    }

    fn matcher_set(facts: &[&str], query: &str, interpreted: bool) -> BTreeSet<String> {
        matcher_set_from_iter(facts.iter().copied(), query, interpreted)
    }

    fn matcher_product_set_from_iter<'a>(
        facts: impl IntoIterator<Item = &'a str>,
        query: &str,
        interpreted: bool,
    ) -> (usize, BTreeSet<String>) {
        let _guard = ForceInterpretedMatcherGuard::new(interpreted);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut space = Space::new();
            insert_matcher_facts(&mut space, facts);
            let pat = crate::expr!(space, query);
            let mut args = Vec::new();
            ExprEnv::new(0, pat).args(&mut args);
            if !interpreted {
                assert!(
                    compile_match_program(&args[1..]).is_some(),
                    "query should be accepted by the compiled matcher: {query}"
                );
            }

            let mut out = BTreeSet::new();
            let count = Space::query_multi(&space.btm, pat, |_, loc| {
                out.insert(expr_string(&space, loc));
                true
            });
            (count, out)
        }));
        match result {
            Ok(set) => set,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    #[test]
    fn matcher_compiled_equals_interpreted_corpus() {
        let corpus: &[(&[&str], &str, &str)] = &[
            (&["[2] qt $"], "[2] qt foo", "leaf capture"),
            (&["[2] qt $"], "[2] qt [2] S Z", "compound capture"),
            (
                &["[2] qt $"],
                "[2] qt [2] S [2] S Z",
                "nested compound capture",
            ),
            (
                &["[3] kv k $"],
                "[3] kv k [3] a [2] b c d",
                "deep nested capture",
            ),
            (
                &["[2] qt foo", "[2] qt [2] S Z", "[2] qt $"],
                "[2] qt [2] S [2] S Z",
                "mixed ground and schematic facts",
            ),
            (
                &["[2] qt [2] S Z", "[2] qt foo"],
                "[2] qt $",
                "variable query over compound data",
            ),
            (
                &["[3] p $ _1"],
                "[3] p [2] S Z [2] S Z",
                "consistent coreference",
            ),
            (
                &["[3] p $ _1"],
                "[3] p [2] S Z [2] K Z",
                "inconsistent coreference",
            ),
            (
                &["[5] if [2] S $ $ $ _2"],
                "[5] if [2] S Z [2] S Z foo [2] S Z",
                "counter-machine shaped capture",
            ),
            (
                &["[5] if [2] S $ $ $ _2"],
                "[5] if [2] S Z [2] S Z foo [2] K Z",
                "counter-machine shaped mismatch",
            ),
        ];

        for (facts, query, label) in corpus {
            let compiled = matcher_set(facts, query, false);
            let interpreted = matcher_set(facts, query, true);
            assert_eq!(
                compiled, interpreted,
                "compiled matcher diverged from interpreted reference on {label}"
            );
        }
    }

    #[test]
    fn compile_match_program_accepts_only_bound_varrefs() {
        let mut space = Space::new();

        let bound_first = crate::expr!(space, "[3] p $ _1");
        let mut args = Vec::new();
        ExprEnv::new(0, bound_first).args(&mut args);
        assert!(
            compile_match_program(&args[1..]).is_some(),
            "a VarRef after its introducing variable should compile"
        );

        let forward_ref = crate::expr!(space, "[3] p _1 $");
        args.clear();
        ExprEnv::new(0, forward_ref).args(&mut args);
        assert!(
            compile_match_program(&args[1..]).is_none(),
            "a VarRef before its introducing variable should fall back"
        );
    }

    /// The linear flatterm `compile_match_factor` must emit byte-identical ops to the
    /// recursive reference on every shape, including deep ground chains (where the
    /// recursive form was O(depth^2)) and the adversarial deep-chain-with-bottom-var.
    #[test]
    fn compile_linear_equals_recursive() {
        let mut space = Space::new();
        let cases = [
            "[2] q a",                               // flat ground source
            "[3] q $ $",                             // flat var sources
            "[2] q [2] S [2] S [2] S [2] S [2] S Z", // deep ground chain S^5(Z)
            "[3] q [2] S [2] S [2] S [2] S $ a",     // deep chain, bottom var (adversarial)
            "[4] state $ $ [2] S [2] S Z",           // counter_machine shape: vars + deep ground
            "[3] f [3] g a b [2] h c",               // nested branching ground
            "[3] p $ _1",                            // bound varref re-check
            "[2] q [1] Z",                           // arity-1 compound
            "[3] mix [2] S [2] S $ [2] T [2] T Z",   // mixed deep var + deep ground
        ];
        for case in cases {
            let pat = crate::expr!(space, case);
            let mut args = Vec::new();
            ExprEnv::new(0, pat).args(&mut args);
            let sources = &args[1..];

            let mut lin = Vec::new();
            let mut intro_l = [false; 256];
            let mut accept_l = true;
            for &s in sources {
                if compile_match_factor(s, &mut lin, &mut intro_l).is_none() {
                    accept_l = false;
                    break;
                }
            }

            let mut rec = Vec::new();
            let mut intro_r = [false; 256];
            let mut scratch = Vec::new();
            let mut accept_r = true;
            for &s in sources {
                if compile_match_factor_recursive(s, &mut rec, &mut intro_r, &mut scratch).is_none()
                {
                    accept_r = false;
                    break;
                }
            }

            assert_eq!(accept_l, accept_r, "accept mismatch: {case}");
            assert_eq!(intro_l, intro_r, "introduced set mismatch: {case}");
            assert_eq!(
                lin, rec,
                "compiled ops differ (linear vs recursive): {case}"
            );
        }
    }

    #[test]
    fn matcher_compiled_equals_interpreted_query_side_varrefs() {
        let single_factor_cases: &[(&[&str], &str, &str)] = &[
            (
                &[
                    "[3] p a a",
                    "[3] p a b",
                    "[3] p [2] S Z [2] S Z",
                    "[3] p [2] S Z [2] K Z",
                ],
                "[3] p $ _1",
                "ground repeated-variable query",
            ),
            (
                &["[3] p a $", "[3] p b c"],
                "[3] p $ _1",
                "repeated-variable query over data-side variable",
            ),
            (
                &[
                    "[3] p [3] long [2] S Z [2] T Q [3] long [2] S Z [2] T Q",
                    "[3] p a a",
                    "[3] p [3] long [2] S Z [2] T Q a",
                ],
                "[3] p $ _1",
                "scratch reuse across long and short ground VarRef rechecks",
            ),
            (
                &["[3] p [3] long [2] S Z [2] T Q $", "[3] p a b"],
                "[3] p $ _1",
                "scratch reuse with data-variable fallback",
            ),
        ];

        for (facts, query, label) in single_factor_cases {
            let compiled =
                matcher_count_and_set_from_iter(facts.iter().copied(), query, false, true);
            let interpreted =
                matcher_count_and_set_from_iter(facts.iter().copied(), query, true, true);
            assert_eq!(
                compiled, interpreted,
                "compiled matcher diverged from interpreted reference on {label}"
            );
        }

        let product_cases: &[(&[&str], &str, &str)] = &[
            (
                &["[3] e a 1", "[3] e b 1", "[3] e c 2"],
                "[3] , [3] e $ $ [3] e $ _2",
                "cross-factor repeated value",
            ),
            (
                &["[3] rel a 1", "[3] rel2 b $"],
                "[3] , [3] rel $ $ [3] rel2 $ _2",
                "cross-factor repeated value against data-side variable",
            ),
        ];

        for (facts, query, label) in product_cases {
            let compiled = matcher_product_set_from_iter(facts.iter().copied(), query, false);
            let interpreted = matcher_product_set_from_iter(facts.iter().copied(), query, true);
            assert!(
                compiled.0 > 0,
                "compiled matcher should exercise a nonempty VarRef product case: {label}"
            );
            assert_eq!(
                compiled, interpreted,
                "compiled matcher diverged from interpreted reference on {label}"
            );
        }
    }

    #[derive(Clone, Copy)]
    struct MatcherFuzzRng {
        state: u64,
    }

    impl MatcherFuzzRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }

        fn index(&mut self, len: usize) -> usize {
            (self.next() as usize) % len
        }
    }

    fn generated_ground_term(rng: &mut MatcherFuzzRng, depth: usize) -> String {
        const ATOMS: &[&str] = &["a", "b", "c", "d", "z", "k0", "k1"];
        if depth == 0 || rng.index(4) == 0 {
            return ATOMS[rng.index(ATOMS.len())].to_string();
        }

        match rng.index(4) {
            0 => format!("[2] S {}", generated_ground_term(rng, depth - 1)),
            1 => format!(
                "[3] pair {} {}",
                generated_ground_term(rng, depth - 1),
                generated_ground_term(rng, depth - 1)
            ),
            2 => format!("[2] box {}", generated_ground_term(rng, depth - 1)),
            _ => format!(
                "[4] tri {} {} {}",
                generated_ground_term(rng, depth - 1),
                generated_ground_term(rng, depth - 1),
                generated_ground_term(rng, depth - 1)
            ),
        }
    }

    fn generated_fact(rng: &mut MatcherFuzzRng) -> String {
        const RELS: &[&str] = &["r0", "r1", "edge", "kv", "wrap"];
        let rel = RELS[rng.index(RELS.len())];
        let arg_count = 2 + rng.index(2);
        let mut args = Vec::with_capacity(arg_count);

        if rng.index(5) == 0 {
            args.push("$".to_string());
            args.push("_1".to_string());
            for _ in 2..arg_count {
                args.push(generated_ground_term(rng, 2));
            }
        } else {
            for _ in 0..arg_count {
                if rng.index(6) == 0 {
                    args.push("$".to_string());
                } else {
                    args.push(generated_ground_term(rng, 2));
                }
            }
        }

        format!("[{}] {} {}", arg_count + 1, rel, args.join(" "))
    }

    fn generated_query_factor(rng: &mut MatcherFuzzRng, vars_left: &mut usize) -> String {
        const RELS: &[&str] = &["r0", "r1", "edge", "kv", "wrap"];
        let rel = RELS[rng.index(RELS.len())];
        let arg_count = 2 + rng.index(2);
        let mut args = Vec::with_capacity(arg_count);
        for _ in 0..arg_count {
            if *vars_left > 0 && rng.index(3) == 0 {
                *vars_left -= 1;
                args.push("$".to_string());
            } else {
                args.push(generated_ground_term(rng, 2));
            }
        }
        format!("[{}] {} {}", arg_count + 1, rel, args.join(" "))
    }

    #[test]
    fn matcher_compiled_equals_interpreted_generated_varref_free_queries() {
        let mut rng = MatcherFuzzRng::new(0x9e37_79b9_7f4a_7c15);
        let mut facts = (0..36)
            .map(|_| generated_fact(&mut rng))
            .collect::<Vec<_>>();
        facts.extend([
            "[3] edge a b".to_string(),
            "[3] kv a c".to_string(),
            "[3] wrap b c".to_string(),
        ]);

        let fact_refs = facts.iter().map(String::as_str).collect::<Vec<_>>();
        let positive_product = "[3] , [3] edge a b [3] kv a c";
        let compiled =
            matcher_product_set_from_iter(fact_refs.iter().copied(), positive_product, false);
        let interpreted =
            matcher_product_set_from_iter(fact_refs.iter().copied(), positive_product, true);
        assert!(
            compiled.0 > 0,
            "positive generated product guard should exercise a nonempty compiled path"
        );
        assert_eq!(
            compiled, interpreted,
            "compiled matcher diverged from interpreted reference on positive generated product guard"
        );

        for case in 0..40 {
            let factor_count = 1 + rng.index(3);
            let mut vars_left = usize::from(factor_count == 1);
            let query = if factor_count == 1 {
                generated_query_factor(&mut rng, &mut vars_left)
            } else {
                let factors = (0..factor_count)
                    .map(|_| generated_query_factor(&mut rng, &mut vars_left))
                    .collect::<Vec<_>>();
                format!("[{}] , {}", factor_count + 1, factors.join(" "))
            };

            if factor_count == 1 {
                let compiled =
                    matcher_count_and_set_from_iter(fact_refs.iter().copied(), &query, false, true);
                let interpreted =
                    matcher_count_and_set_from_iter(fact_refs.iter().copied(), &query, true, true);
                assert_eq!(
                    compiled, interpreted,
                    "compiled matcher diverged from interpreted reference on generated case {case}: {query}"
                );
            } else {
                let compiled =
                    matcher_product_set_from_iter(fact_refs.iter().copied(), &query, false);
                let interpreted =
                    matcher_product_set_from_iter(fact_refs.iter().copied(), &query, true);
                assert_eq!(
                    compiled, interpreted,
                    "compiled matcher diverged from interpreted reference on generated product case {case}: {query}"
                );
            }
        }
    }

    #[test]
    fn compiled_matcher_compound_capture_mutant_is_observable() {
        fn count(query: &str, interpreted: bool, compound_capture: bool) -> usize {
            super::FORCE_INTERPRETED_MATCHER.with(|c| c.set(interpreted));
            super::COMPILED_MATCHER_COMPOUND_CAPTURE.with(|c| c.set(compound_capture));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut space = Space::new();
                let expr = crate::expr!(space, "[2] qt $");
                let bytes = unsafe { expr.span().as_ref().unwrap() }.to_vec();
                let mut wz = space.btm.write_zipper_at_path(&bytes);
                wz.set_value(());

                let pat = crate::expr!(space, query);
                let mut wrapped = vec![
                    item_byte(Tag::Arity(2)),
                    item_byte(Tag::SymbolSize(1)),
                    b',',
                ];
                wrapped.extend_from_slice(unsafe { pat.span().as_ref().unwrap() });
                let mut matches = 0usize;
                Space::query_multi(
                    &space.btm,
                    Expr {
                        ptr: wrapped.leak().as_mut_ptr(),
                    },
                    |_bindings, _loc| {
                        matches += 1;
                        true
                    },
                );
                matches
            }));
            super::FORCE_INTERPRETED_MATCHER.with(|c| c.set(false));
            super::COMPILED_MATCHER_COMPOUND_CAPTURE.with(|c| c.set(true));
            match result {
                Ok(matches) => matches,
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }

        assert_eq!(count("[2] qt [2] S Z", true, true), 1);
        assert_eq!(count("[2] qt [2] S Z", false, true), 1);
        assert_eq!(count("[2] qt [2] S Z", false, false), 0);
        assert_eq!(count("[2] qt foo", false, false), 1);
    }

    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture, default vs the flip feature"]
    fn bench_triangle_sparse_join_metta() {
        use std::time::Instant;
        let hubs = 3usize;
        for &n in &[100usize, 200, 400] {
            let mut space = Space::new();
            let mut program = String::new();
            for a in 0..hubs {
                for b in 0..hubs {
                    if a != b {
                        program.push_str(&format!("(edge h{a} h{b})\n"));
                    }
                }
            }
            for p in 0..n {
                for h in 0..hubs {
                    program.push_str(&format!("(edge p{p} h{h})\n(edge h{h} p{p})\n"));
                }
            }
            space.add_all_sexpr(program.as_bytes()).unwrap();

            let start = Instant::now();
            space
                .add_all_sexpr(
                    b"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $x)) (, (tri $x $y $z)))\n",
                )
                .unwrap();
            space.metta_calculus(1);
            let elapsed = start.elapsed();
            eprintln!(
                "n={n:4} btm_val_count={:8} elapsed={elapsed:?}",
                space.btm.val_count()
            );
        }
    }

    // The cost of the all-or-nothing schematic decline. The same triangle body and
    // hub-graph, run two ways:
    //   admit   - every joined fact is ground, so the sidecar runs the WCO join,
    //   decline - ONE extra schematic fact `(edge zz $w)` sits under the joined
    //             `edge` prefix. `zz` has no incoming edge, so it closes no
    //             triangle and the emitted output is byte-identical; it exists only
    //             to trip `any_schematic_fact_under_prefixes`, which sends the WHOLE
    //             body to the ProductZipper. Its variable lands on no join position,
    //             so a per-position router would still admit it.
    // The ratio is what the coarse per-relation gate costs, i.e. what per-position
    // routing recovers. Output is asserted byte-identical and the decline counter
    // asserted 0 (admit) vs 1 (decline), so the path flip is proven, not inferred.
    //
    //   cargo +nightly test -p mork --lib --release bench_decline_penalty_metta \
    //       -- --ignored --nocapture
    #[ignore = "decline-penalty harness; run with --ignored --nocapture"]
    #[test]
    fn bench_decline_penalty_metta() {
        use std::sync::atomic::Ordering;
        use std::time::Instant;

        // Measures the all-or-nothing decline cost, so the decline case must actually decline.
        // The unification route recovers it (bench_unify_recovery_metta), so pin the route off.
        SIDECAR_UNIFY_ENABLED.store(false, Ordering::Relaxed);

        fn collect(space: &Space) -> BTreeSet<Vec<u8>> {
            let mut facts = BTreeSet::new();
            space
                .btm
                .try_for_each_value::<_, ()>(|path, _| {
                    facts.insert(path.to_vec());
                    Ok(())
                })
                .unwrap();
            facts
        }

        fn run(
            n: usize,
            hubs: usize,
            schematic: bool,
        ) -> (std::time::Duration, BTreeSet<Vec<u8>>, u64) {
            let mut space = Space::new();
            let mut program = String::new();
            for a in 0..hubs {
                for b in 0..hubs {
                    if a != b {
                        program.push_str(&format!("(edge h{a} h{b})\n"));
                    }
                }
            }
            for p in 0..n {
                for h in 0..hubs {
                    program.push_str(&format!("(edge p{p} h{h})\n(edge h{h} p{p})\n"));
                }
            }
            if schematic {
                // Schematic, under the joined `edge` prefix, but isolated: `zzdead`
                // has no other edge and the compound target `(qq $w)` unifies with no
                // atom endpoint, so it closes no triangle and emits nothing. It exists
                // only to trip the per-relation `any_schematic_fact_under_prefixes`
                // gate. Its variable is on no join position: a per-position router admits it.
                program.push_str("(edge zzdead (qq $w))\n");
            }
            space.add_all_sexpr(program.as_bytes()).unwrap();

            let before = collect(&space);
            SIDECAR_SCHEMATIC_DECLINES.store(0, Ordering::Relaxed);
            let start = Instant::now();
            space
                .add_all_sexpr(
                    b"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $x)) (, (tri $x $y $z)))\n",
                )
                .unwrap();
            space.metta_calculus(1);
            let elapsed = start.elapsed();
            let declines = SIDECAR_SCHEMATIC_DECLINES.load(Ordering::Relaxed);
            let emitted: BTreeSet<Vec<u8>> = collect(&space).difference(&before).cloned().collect();
            (elapsed, emitted, declines)
        }

        let hubs = 3usize;
        eprintln!("decline penalty: one schematic edge fact forces the body off the WCO join");
        for &n in &[100usize, 200, 400] {
            let (t_admit, out_admit, dec_admit) = run(n, hubs, false);
            let (t_decline, out_decline, dec_decline) = run(n, hubs, true);
            if out_admit != out_decline {
                let only_admit: Vec<_> = out_admit
                    .difference(&out_decline)
                    .take(8)
                    .map(|p| serialize(p))
                    .collect();
                let only_decline: Vec<_> = out_decline
                    .difference(&out_admit)
                    .take(8)
                    .map(|p| serialize(p))
                    .collect();
                eprintln!(
                    "n={n} DIFF admit_len={} decline_len={}",
                    out_admit.len(),
                    out_decline.len()
                );
                eprintln!("  only in admit:   {:?}", only_admit);
                eprintln!("  only in decline: {:?}", only_decline);
            }
            assert_eq!(
                out_admit, out_decline,
                "n={n}: output must be byte-identical across paths"
            );
            assert_eq!(
                dec_admit, 0,
                "n={n}: the all-ground body must stay on the sidecar"
            );
            assert_eq!(
                dec_decline, 1,
                "n={n}: one schematic fact must decline the body exactly once"
            );
            let penalty = t_decline.as_secs_f64() / t_admit.as_secs_f64();
            eprintln!(
                "n={n:4} admit(WCO join)={t_admit:>10.3?}  decline(ProductZipper)={t_decline:>10.3?}  penalty={penalty:4.1}x  outputs={}",
                out_admit.len()
            );
        }
        SIDECAR_UNIFY_ENABLED.store(true, Ordering::Relaxed);
    }

    // Per-position admission, recovering the decline penalty. The triangle gains a label
    // pendant `(label $x $t)`. A partial-information schematic label `(label extra $w)` (an
    // edgeless node, value unknown) is admissible: its variable sits only on the output
    // position $t, so it produces only non-ground rows the exec drops, and the gate keeps the
    // whole body on the WCO join. A schematic label with a non-ground compound on the JOIN
    // key, `(label (qq $w) lx)`, is declined (the equality join cannot intersect it), giving
    // the ProductZipper baseline the old all-or-nothing gate forced for both. Same body,
    // byte-identical output (neither extra fact closes a triangle), the only change is
    // whether the one schematic fact is admissible.
    //   cargo +nightly test -p mork --lib --release bench_admission_recovery_metta \
    //       -- --ignored --nocapture
    #[ignore = "admission-recovery harness; run with --ignored --nocapture"]
    #[test]
    fn bench_admission_recovery_metta() {
        use std::sync::atomic::Ordering;
        use std::time::Instant;

        // This bench measures the per-position EQUALITY-join admission gate, where the decline
        // case must actually decline. The unification route would otherwise recover it (that is
        // bench_unify_recovery_metta's job), so pin the route off here.
        SIDECAR_UNIFY_ENABLED.store(false, Ordering::Relaxed);

        fn collect(space: &Space) -> BTreeSet<Vec<u8>> {
            let mut facts = BTreeSet::new();
            space
                .btm
                .try_for_each_value::<_, ()>(|p, _| {
                    facts.insert(p.to_vec());
                    Ok(())
                })
                .unwrap();
            facts
        }

        fn run(
            n: usize,
            hubs: usize,
            decline: bool,
        ) -> (std::time::Duration, BTreeSet<Vec<u8>>, u64) {
            let mut space = Space::new();
            let mut program = String::new();
            for a in 0..hubs {
                for b in 0..hubs {
                    if a != b {
                        program.push_str(&format!("(edge h{a} h{b})\n(label h{a} lh{a})\n"));
                    }
                }
            }
            for p in 0..n {
                program.push_str(&format!("(label p{p} lp{p})\n"));
                for h in 0..hubs {
                    program.push_str(&format!("(edge p{p} h{h})\n(edge h{h} p{p})\n"));
                }
            }
            // One schematic label that closes no triangle either way. With its unknown on the
            // OUTPUT position it is admitted; with a non-ground compound on the JOIN key it is
            // declined, because the equality join cannot intersect a non-ground key.
            if decline {
                program.push_str("(label (qq $w) lx)\n");
            } else {
                program.push_str("(label extra $w)\n");
            }
            space.add_all_sexpr(program.as_bytes()).unwrap();

            let before = collect(&space);
            SIDECAR_SCHEMATIC_DECLINES.store(0, Ordering::Relaxed);
            let start = Instant::now();
            space
                .add_all_sexpr(b"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $x) (label $x $t)) (, (out $x $y $z $t)))\n")
                .unwrap();
            space.metta_calculus(1);
            let elapsed = start.elapsed();
            let declines = SIDECAR_SCHEMATIC_DECLINES.load(Ordering::Relaxed);
            let emitted: BTreeSet<Vec<u8>> = collect(&space).difference(&before).cloned().collect();
            (elapsed, emitted, declines)
        }

        let hubs = 3usize;
        eprintln!(
            "admission recovery: an output-only-variable schematic fact stays on the WCO join"
        );
        for &n in &[100usize, 200, 400] {
            let (t_fast, out_fast, dec_fast) = run(n, hubs, false);
            let (t_slow, out_slow, dec_slow) = run(n, hubs, true);
            assert_eq!(
                out_fast, out_slow,
                "n={n}: output must be byte-identical across paths"
            );
            assert_eq!(
                dec_fast, 0,
                "n={n}: the output-only schematic label must be admitted"
            );
            assert_eq!(
                dec_slow, 1,
                "n={n}: the join-key compound label must decline exactly once"
            );
            let recovery = t_slow.as_secs_f64() / t_fast.as_secs_f64();
            eprintln!(
                "n={n:4} admitted(WCO join)={t_fast:>10.3?}  declined(ProductZipper)={t_slow:>10.3?}  recovery={recovery:4.1}x  outputs={}",
                out_fast.len()
            );
        }
        SIDECAR_UNIFY_ENABLED.store(true, Ordering::Relaxed);
    }

    // The deep recovery, measured. A schematic edge on a JOIN KEY (`(edge k{j} $w)`, the data
    // variable landing on the shared `$y`) is the case the equality join cannot intersect, so the
    // old gate declines the whole cyclic body to the ProductZipper. With the unification route on,
    // the same body runs the worst-case-optimal leapfrog-unification join instead. The workload is
    // the AGM-blowup triangle: a hub with `s` in- and out-edges gives s^2 two-paths but no
    // triangle (the ProductZipper materializes them; the WCO join prunes), a small complete digraph
    // gives the ground triangles, and the schematic edges add the unification answers. Same body,
    // same space, byte-identical output; the only difference is the route, so the ratio is the cost
    // the decline used to pay.
    //   cargo +nightly test -p mork --lib --release bench_unify_recovery_metta \
    //       -- --ignored --nocapture --test-threads=1
    #[ignore = "unification-recovery harness; run with --ignored --nocapture --test-threads=1"]
    #[test]
    fn bench_unify_recovery_metta() {
        use std::sync::atomic::Ordering;
        use std::time::Instant;

        fn collect(space: &Space) -> BTreeSet<Vec<u8>> {
            let mut facts = BTreeSet::new();
            space
                .btm
                .try_for_each_value::<_, ()>(|p, _| {
                    facts.insert(p.to_vec());
                    Ok(())
                })
                .unwrap();
            facts
        }

        fn run(
            s: usize,
            clique: usize,
            sch: usize,
            unify: bool,
        ) -> (std::time::Duration, BTreeSet<Vec<u8>>, u64, u64) {
            SIDECAR_UNIFY_ENABLED.store(unify, Ordering::Relaxed);
            let mut space = Space::new();
            let mut program = String::new();
            // Hub blowup: s^2 two-paths through the hub `h`, no triangle of its own.
            for i in 0..s {
                program.push_str(&format!("(edge u{i} h)\n(edge h v{i})\n"));
            }
            // A complete digraph supplies the ground triangles.
            for a in 0..clique {
                for b in 0..clique {
                    if a != b {
                        program.push_str(&format!("(edge k{a} k{b})\n"));
                    }
                }
            }
            // Schematic edges from clique vertices: a data variable on the target, which lands on
            // the join key `$y` and so cannot ride the equality join.
            for j in 0..sch {
                program.push_str(&format!("(edge k{j} $w{j})\n"));
            }
            space.add_all_sexpr(program.as_bytes()).unwrap();

            let before = collect(&space);
            SIDECAR_SCHEMATIC_DECLINES.store(0, Ordering::Relaxed);
            SIDECAR_UNIFY_RECOVERS.store(0, Ordering::Relaxed);
            let start = Instant::now();
            space
                .add_all_sexpr(
                    b"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $x)) (, (out $x $y $z)))\n",
                )
                .unwrap();
            space.metta_calculus(1);
            let elapsed = start.elapsed();
            let declines = SIDECAR_SCHEMATIC_DECLINES.load(Ordering::Relaxed);
            let recovers = SIDECAR_UNIFY_RECOVERS.load(Ordering::Relaxed);
            let emitted: BTreeSet<Vec<u8>> = collect(&space).difference(&before).cloned().collect();
            (elapsed, emitted, declines, recovers)
        }

        let clique = 6usize;
        let sch = 3usize;
        eprintln!(
            "unification recovery: a schematic edge on a join key, WCO unification join vs ProductZipper decline"
        );
        for &s in &[128usize, 256, 512, 1024, 2048, 4096] {
            let (t_fast, out_fast, _dec_fast, rec_fast) = run(s, clique, sch, true);
            let (t_slow, out_slow, dec_slow, _rec_slow) = run(s, clique, sch, false);
            assert_eq!(
                out_fast, out_slow,
                "s={s}: output must be byte-identical across paths"
            );
            assert!(
                rec_fast >= 1,
                "s={s}: the unification route must fire when enabled"
            );
            assert!(
                dec_slow >= 1,
                "s={s}: the body must decline to the ProductZipper when the route is off"
            );
            let recovery = t_slow.as_secs_f64() / t_fast.as_secs_f64();
            eprintln!(
                "s={s:4} unify(WCO)={t_fast:>11.3?}  decline(ProductZipper)={t_slow:>11.3?}  recovery={recovery:5.1}x  outputs={}",
                out_fast.len()
            );
        }
        SIDECAR_UNIFY_ENABLED.store(true, Ordering::Relaxed);
    }

    // Capture route vs native ProductZipper on a data-side-capture workload. A single schematic fact
    // `(r $d hub)` absorbs the query compound `(k $x)`; the join `(s $x)` then grounds `$x`, so the
    // capture fact contributes one answer per join key. Both paths should now emit the same answers.
    // The benchmark reports the full-unification route cost against the native matcher. The descent
    // prunes the head-`m` noise facts, so the read tracks the matched relation, not the whole space.
    //   cargo +nightly test -p mork --lib --release bench_capture_route_vs_product_zipper \
    //       -- --ignored --nocapture --test-threads=1
    #[ignore = "benchmark; run with --ignored --nocapture --test-threads=1"]
    #[test]
    fn bench_capture_route_vs_product_zipper() {
        use std::sync::atomic::Ordering;
        use std::time::Instant;

        fn collect(space: &Space) -> BTreeSet<Vec<u8>> {
            let mut facts = BTreeSet::new();
            space
                .btm
                .try_for_each_value::<_, ()>(|p, _| {
                    facts.insert(p.to_vec());
                    Ok(())
                })
                .unwrap();
            facts
        }

        fn run(s: usize, capture: bool) -> (std::time::Duration, BTreeSet<Vec<u8>>, u64) {
            SIDECAR_CAPTURE_ENABLED.store(capture, Ordering::Relaxed);
            let mut space = Space::new();
            let mut program = String::new();
            for i in 0..s {
                program.push_str(&format!("(r (k c{i}) val{i})\n")); // ground head-k: both paths match
                program.push_str(&format!("(s c{i})\n")); // the join key
                program.push_str(&format!("(r (m c{i}) w{i})\n")); // head-m noise: the descent prunes it
            }
            program.push_str("(r $d hub)\n"); // the schematic fact only the capture route mines
            space.add_all_sexpr(program.as_bytes()).unwrap();

            let before = collect(&space);
            SIDECAR_CAPTURE_RECOVERS.store(0, Ordering::Relaxed);
            let start = Instant::now();
            space
                .add_all_sexpr(b"(exec 0 (, (r (k $x) $y) (s $x)) (, (out $x $y)))\n")
                .unwrap();
            space.metta_calculus(1);
            let elapsed = start.elapsed();
            let rec = SIDECAR_CAPTURE_RECOVERS.load(Ordering::Relaxed);
            let emitted: BTreeSet<Vec<u8>> = collect(&space).difference(&before).cloned().collect();
            (elapsed, emitted, rec)
        }

        eprintln!(
            "capture route vs ProductZipper: the schematic `(r $d hub)` absorbs query `(k $x)`; both paths should emit the same answers"
        );
        for &s in &[64usize, 128, 256, 512, 1024, 2048] {
            let (t_cap, out_cap, rec) = run(s, true);
            let (t_pz, out_pz, _) = run(s, false);
            assert_eq!(
                out_cap, out_pz,
                "s={s}: capture route diverged from native ProductZipper"
            );
            assert!(rec >= 1, "s={s}: the capture route must fire");
            let ratio = t_cap.as_secs_f64() / t_pz.as_secs_f64();
            eprintln!(
                "s={s:5} capture={t_cap:>11.3?} ({:5} answers)  ProductZipper={t_pz:>11.3?} ({:5} answers)  cost={ratio:4.2}x",
                out_cap.len(),
                out_pz.len()
            );
        }
        SIDECAR_CAPTURE_ENABLED.store(false, Ordering::Relaxed);
    }

    // The adapter-1 property: the route reads facts under its relation prefixes through the PathMap
    // index, so its cost tracks the JOINED relations, not the size of the space. This holds the
    // relevant cyclic-schematic workload fixed (a small hub triangle over `edge`) and floods the
    // space with facts under an UNRELATED relation `junk`. Output is byte-identical across junk
    // levels (the junk closes no triangle), asserted. Measured A/B of the read at the same sizes:
    //   junk        0     4000    16000    64000
    //   full scan  3.15ms 2.81ms  3.44ms   4.57ms   (climbs with the space)
    //   index read 3.25ms 2.98ms  3.24ms   3.27ms   (flat)
    // The full scan's climb is per-flip and unbounded in the space size; the index read stays flat.
    //   cargo +nightly test -p mork --lib --release bench_unify_irrelevant_facts \
    //       -- --ignored --nocapture --test-threads=1
    #[ignore = "adapter-1 index-read harness; run with --ignored --nocapture --test-threads=1"]
    #[test]
    fn bench_unify_irrelevant_facts() {
        use std::sync::atomic::Ordering;
        use std::time::Instant;

        fn run(junk: usize) -> (std::time::Duration, BTreeSet<Vec<u8>>, u64) {
            SIDECAR_UNIFY_ENABLED.store(true, Ordering::Relaxed);
            let mut space = Space::new();
            let mut program = String::new();
            // Fixed relevant workload: a small hub triangle over `edge` with a few schematic edges.
            for a in 0..6 {
                for b in 0..6 {
                    if a != b {
                        program.push_str(&format!("(edge k{a} k{b})\n"));
                    }
                }
            }
            for i in 0..256 {
                program.push_str(&format!("(edge u{i} h)\n(edge h w{i})\n"));
            }
            for j in 0..3 {
                program.push_str(&format!("(edge k{j} $w{j})\n"));
            }
            // Irrelevant flood under a different relation, never touched by the `edge` body.
            for k in 0..junk {
                program.push_str(&format!("(junk j{k} j{k})\n"));
            }
            space.add_all_sexpr(program.as_bytes()).unwrap();

            let mut before = BTreeSet::new();
            space
                .btm
                .try_for_each_value::<_, ()>(|p, _| {
                    before.insert(p.to_vec());
                    Ok(())
                })
                .unwrap();
            SIDECAR_UNIFY_RECOVERS.store(0, Ordering::Relaxed);
            let start = Instant::now();
            space
                .add_all_sexpr(
                    b"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $x)) (, (out $x $y $z)))\n",
                )
                .unwrap();
            space.metta_calculus(1);
            let elapsed = start.elapsed();
            let recovers = SIDECAR_UNIFY_RECOVERS.load(Ordering::Relaxed);
            let mut emitted = BTreeSet::new();
            space
                .btm
                .try_for_each_value::<_, ()>(|p, _| {
                    if !before.contains(p) {
                        emitted.insert(p.to_vec());
                    }
                    Ok(())
                })
                .unwrap();
            (elapsed, emitted, recovers)
        }

        eprintln!("adapter-1 index read: relevant workload fixed, irrelevant `junk` facts vary");
        let mut baseline: Option<BTreeSet<Vec<u8>>> = None;
        for &junk in &[0usize, 4000, 16000, 64000] {
            let (t, out, rec) = run(junk);
            if let Some(b) = &baseline {
                assert_eq!(
                    *b, out,
                    "junk={junk}: output must be byte-identical to the junk-free run"
                );
            } else {
                baseline = Some(out.clone());
            }
            assert!(rec >= 1, "junk={junk}: the route must fire");
            eprintln!(
                "junk={junk:6}  route={t:>10.3?}  recovered={rec}  outputs={}",
                out.len()
            );
        }
        SIDECAR_UNIFY_ENABLED.store(true, Ordering::Relaxed);
    }

    // First-occurrence order of `$var` names across the patterns. The prototype assigns
    // ids the same way, so an `(ans <vars>)` template lists bindings in the matching order.
    fn first_occurrence_vars(patterns: &[&str]) -> Vec<String> {
        let mut order: Vec<String> = Vec::new();
        for p in patterns {
            let mut name = String::new();
            let mut in_var = false;
            for c in p.chars() {
                if c == '$' {
                    in_var = true;
                    name.clear();
                } else if in_var && (c.is_alphanumeric() || c == '_') {
                    name.push(c);
                } else {
                    if in_var && !name.is_empty() && !order.contains(&name) {
                        order.push(name.clone());
                    }
                    in_var = false;
                    name.clear();
                }
            }
            if in_var && !name.is_empty() && !order.contains(&name) {
                order.push(name.clone());
            }
        }
        order
    }

    // A serialized term is ground when no token is a NewVar (`$`) or VarRef (`_<n>`).
    // serialize already skips symbol content, so this is structural over real variables.
    fn serialized_is_ground(s: &str) -> bool {
        s.split_whitespace().all(|t| {
            t != "$"
                && !(t.starts_with('_')
                    && t.len() > 1
                    && t[1..].bytes().all(|b| b.is_ascii_digit()))
        })
    }

    // A small xorshift RNG shared by the unification-route differentials.
    struct UjRng(u64);
    impl UjRng {
        fn nx(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.nx() % n as u64) as usize
        }
        fn chance(&mut self, a: usize, b: usize) -> bool {
            self.below(b) < a
        }
    }

    fn sidecar_route_toggle_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    // Run a conjunctive body and projection as a live exec, returning the ground `out` answers and
    // how many recoveries fired, with the worst-case-optimal unification route forced on or off.
    fn uj_live_run(
        facts: &[String],
        patterns: &[String],
        proj: &[String],
        unify: bool,
    ) -> (BTreeSet<String>, u64) {
        use std::sync::atomic::Ordering;
        SIDECAR_UNIFY_ENABLED.store(unify, Ordering::Relaxed);
        SIDECAR_UNIFY_RECOVERS.store(0, Ordering::Relaxed);
        let mut space = Space::new();
        let mut program = String::new();
        for f in facts {
            program.push_str(f);
            program.push('\n');
        }
        space.add_all_sexpr(program.as_bytes()).unwrap();
        let mut before = BTreeSet::new();
        space
            .btm
            .try_for_each_value::<_, ()>(|p, _| {
                before.insert(p.to_vec());
                Ok(())
            })
            .unwrap();
        let body = patterns.join(" ");
        let out_vars = proj
            .iter()
            .map(|v| format!("${v}"))
            .collect::<Vec<_>>()
            .join(" ");
        let exec = format!("(exec 0 (, {body}) (, (out {out_vars})))\n");
        space.add_all_sexpr(exec.as_bytes()).unwrap();
        space.metta_calculus(1);
        let head = format!("[{}] out", proj.len() + 1);
        let mut out = BTreeSet::new();
        space
            .btm
            .try_for_each_value::<_, ()>(|p, _| {
                if !before.contains(p) {
                    let s = serialize(p);
                    if s.starts_with(&head) && serialized_is_ground(&s) {
                        out.insert(s);
                    }
                }
                Ok(())
            })
            .unwrap();
        (out, SIDECAR_UNIFY_RECOVERS.load(Ordering::Relaxed))
    }

    fn uj_live_run_raw(
        facts: &[String],
        patterns: &[String],
        proj: &[String],
        unify: bool,
    ) -> (BTreeSet<String>, u64, u64, usize) {
        use std::sync::atomic::Ordering;
        SIDECAR_UNIFY_ENABLED.store(unify, Ordering::Relaxed);
        SIDECAR_UNIFY_RECOVERS.store(0, Ordering::Relaxed);
        SIDECAR_ZIPPER_RECOVERS.store(0, Ordering::Relaxed);
        let mut space = Space::new();
        let mut program = String::new();
        for f in facts {
            program.push_str(f);
            program.push('\n');
        }
        space.add_all_sexpr(program.as_bytes()).unwrap();
        let mut before = BTreeSet::new();
        space
            .btm
            .try_for_each_value::<_, ()>(|p, _| {
                before.insert(p.to_vec());
                Ok(())
            })
            .unwrap();
        let body = patterns.join(" ");
        let out_vars = proj
            .iter()
            .map(|v| format!("${v}"))
            .collect::<Vec<_>>()
            .join(" ");
        let exec = format!("(exec 0 (, {body}) (, (out {out_vars})))\n");
        space.add_all_sexpr(exec.as_bytes()).unwrap();
        space.metta_calculus(1);
        let head = format!("[{}] out", proj.len() + 1);
        let mut out = BTreeSet::new();
        let mut nonground = 0usize;
        space
            .btm
            .try_for_each_value::<_, ()>(|p, _| {
                if !before.contains(p) {
                    let s = serialize(p);
                    if s.starts_with(&head) {
                        if !serialized_is_ground(&s) {
                            nonground += 1;
                        }
                        out.insert(s);
                    }
                }
                Ok(())
            })
            .unwrap();
        (
            out,
            SIDECAR_UNIFY_RECOVERS.load(Ordering::Relaxed),
            SIDECAR_ZIPPER_RECOVERS.load(Ordering::Relaxed),
            nonground,
        )
    }

    // Assert the unification route is byte-identical to the ProductZipper for one case (route on
    // versus off), and report (the route fired, the answer set is non-empty).
    fn uj_assert_identical(facts: &[String], patterns: &[String], proj: &[String]) -> (bool, bool) {
        let (with_unify, rec) = uj_live_run(facts, patterns, proj, true);
        let (without_unify, _) = uj_live_run(facts, patterns, proj, false);
        assert_eq!(
            with_unify,
            without_unify,
            "unify route diverged from the ProductZipper\n  patterns={patterns:?}\n  facts={facts:?}\n  proj={proj:?}\n  unify-only={:?}\n  product-only={:?}",
            with_unify.difference(&without_unify).collect::<Vec<_>>(),
            without_unify.difference(&with_unify).collect::<Vec<_>>()
        );
        (rec > 0, !with_unify.is_empty())
    }

    /// A random cyclic body for the live A/B corpora: an arity-2 edge cycle (triangle, 4-cycle, or
    /// triangle-with-pendant) or an arity-3 rotation cycle, over ground vertices plus a few schematic
    /// facts on join keys, with an occasional compound query argument. Every variable sits on a join
    /// key, so the body reaches the unification route. Shared by the ProductZipper and kernel A/Bs.
    fn uj_random_cyclic_case(r: &mut UjRng) -> (Vec<String>, Vec<String>) {
        let nv = 3 + r.below(3); // 3..=5 ground vertices
        if r.chance(1, 3) {
            // Arity-3 rotation cycle over relation `t`.
            let pats = vec![
                "(t $x $y $z)".to_string(),
                "(t $y $z $x)".to_string(),
                "(t $z $x $y)".to_string(),
            ];
            let mut f = vec![
                "(t v0 v1 v2)".to_string(),
                "(t v1 v2 v0)".to_string(),
                "(t v2 v0 v1)".to_string(),
            ];
            for i in 0..nv {
                if r.chance(1, 2) {
                    f.push(format!("(t v{i} v{} v{})", (i + 1) % nv, (i + 2) % nv));
                }
            }
            let nsch = 1 + r.below(3);
            for k in 0..nsch {
                let v = r.below(nv);
                f.push(match r.below(3) {
                    0 => format!("(t v{v} $s{k} v{})", (v + 1) % nv),
                    1 => format!("(t $s{k} v{v} $u{k})"),
                    _ => format!("(t (k v{v}) $s{k} v{})", (v + 2) % nv),
                });
            }
            (pats, f)
        } else {
            // Arity-2 edge cycle over relation `e`.
            let compound = r.chance(1, 6);
            let pats: Vec<String> = match r.below(3) {
                0 if compound => vec!["(e (k $x) $y)", "(e $y $z)", "(e $z (k $x))"],
                0 => vec!["(e $x $y)", "(e $y $z)", "(e $z $x)"],
                1 => vec!["(e $x $y)", "(e $y $z)", "(e $z $w)", "(e $w $x)"],
                _ => vec!["(e $x $y)", "(e $y $z)", "(e $z $x)", "(e $x $w)"],
            }
            .into_iter()
            .map(String::from)
            .collect();
            let mut f: Vec<String> =
                vec!["(e v0 v1)".into(), "(e v1 v2)".into(), "(e v2 v0)".into()];
            for i in 0..nv {
                for j in 0..nv {
                    if i != j && r.chance(1, 3) {
                        f.push(format!("(e v{i} v{j})"));
                    }
                }
            }
            let nsch = 1 + r.below(3);
            for k in 0..nsch {
                let v = r.below(nv);
                f.push(match r.below(5) {
                    0 => format!("(e v{v} $s{k})"),
                    1 => format!("(e $s{k} v{v})"),
                    2 => format!("(e $s{k} $t{k})"),
                    3 => format!("(e (k (k v{v})) $s{k})"),
                    _ => format!("(e (k v{v}) $s{k})"),
                });
            }
            (pats, f)
        }
    }

    /// A random ACYCLIC data-side-capture body (issue-29): one factor carries a non-ground compound
    /// `(wrap $x)` whose inner variable joins a second relation, over facts whose data variables
    /// must absorb that compound. This is the fragment `transform_via_capture` handles, the acyclic
    /// analogue of `uj_random_cyclic_case`. A mix of head-matching, head-mismatching, and
    /// data-variable facts exercises the descent's pruning and its capture binding together.
    fn uj_random_acyclic_capture_case(r: &mut UjRng) -> (Vec<String>, Vec<String>) {
        let wrap = ["f", "g", "k"][r.below(3)];
        let rel2 = ["s", "p", "q"][r.below(3)];
        let pats = vec![format!("(r ({wrap} $x) $y)"), format!("({rel2} $x)")];
        let nv = 2 + r.below(3);
        let mut f = vec![format!("(r $d v{})", r.below(nv))];
        for i in 0..nv {
            f.push(match r.below(3) {
                0 => format!("(r ({wrap} c{i}) v{i})"), // head matches: $x = c{i}
                1 => format!("(r (m c{i}) v{i})"),      // head mismatches: descent prunes it
                _ => format!("(r $e{i} v{i})"),         // another data variable
            });
        }
        let nx = 1 + r.below(3);
        for _ in 0..nx {
            f.push(format!("({rel2} c{})", r.below(nv)));
        }
        if r.chance(1, 2) {
            f.push(format!("({rel2} $w)")); // a schematic second-relation fact
        }
        (pats, f)
    }

    /// Run a body through the unification route twice, once with the zipper-native kernel and once
    /// with the materialized leapfrog, asserting the emitted ground answers are byte-for-byte
    /// identical. The direct A/B for the kernel swap, independent of the ProductZipper. Returns
    /// whether the run produced any answers.
    fn uj_assert_kernels_identical(facts: &[String], patterns: &[String], proj: &[String]) -> bool {
        use std::sync::atomic::Ordering;
        SIDECAR_ZIPPER_JOIN_ENABLED.store(true, Ordering::Relaxed);
        let (with_zipper, _) = uj_live_run(facts, patterns, proj, true);
        SIDECAR_ZIPPER_JOIN_ENABLED.store(false, Ordering::Relaxed);
        let (with_materialized, _) = uj_live_run(facts, patterns, proj, true);
        SIDECAR_ZIPPER_JOIN_ENABLED.store(true, Ordering::Relaxed);
        assert_eq!(
            with_zipper,
            with_materialized,
            "zipper kernel diverged from the materialized join\n  patterns={patterns:?}\n  facts={facts:?}\n  proj={proj:?}\n  zipper-only={:?}\n  materialized-only={:?}",
            with_zipper
                .difference(&with_materialized)
                .collect::<Vec<_>>(),
            with_materialized
                .difference(&with_zipper)
                .collect::<Vec<_>>()
        );
        !with_zipper.is_empty()
    }

    // The kernel swap's end-to-end gate: drive random cyclic schematic bodies through the LIVE flip
    // twice, zipper-native kernel vs materialized leapfrog, and assert byte-identical ground answers.
    // Run it isolated (`--test-threads=1`): the kernel toggle is process-global.
    // Bridge-route equality contract: meaningful only with the bridge compiled in
    // (no-bridge builds route differently; control-verified pre-delta).
    #[cfg(feature = "sidecar_bridge_emit")]
    #[test]
    fn zipper_kernel_byte_identical_to_materialized_live() {
        use std::sync::atomic::Ordering;
        let _route_guard = sidecar_route_toggle_lock().lock().unwrap();
        let mut r = UjRng(0x2718_2818_2845_9045);
        SIDECAR_ZIPPER_RECOVERS.store(0, Ordering::Relaxed);
        let mut nonempty = 0usize;
        const N: usize = 1500;
        for _ in 0..N {
            let (patterns, facts) = uj_random_cyclic_case(&mut r);
            let pats: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
            let order = first_occurrence_vars(&pats);
            if order.is_empty() {
                continue;
            }
            let mask = 1 + r.below((1usize << order.len()) - 1);
            let proj: Vec<String> = order
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, v)| v.clone())
                .collect();
            if uj_assert_kernels_identical(&facts, &patterns, &proj) {
                nonempty += 1;
            }
        }
        SIDECAR_ZIPPER_JOIN_ENABLED.store(true, Ordering::Relaxed);
        let zk = SIDECAR_ZIPPER_RECOVERS.load(Ordering::Relaxed);
        eprintln!("kernel A/B: {N} cases | zipper kernel emitted {zk} | nonempty {nonempty}");
        assert!(
            zk > 30,
            "corpus must exercise the zipper kernel, not just the fallback: {zk}"
        );
        assert!(
            nonempty > 50,
            "corpus must produce real answers: {nonempty}"
        );
    }

    // Run the body as an exec with an `(ans <vars>)` template through the real matcher and
    // return the emitted ground `ans` fact paths. The exec apply keeps only ground results.
    fn fork_answer_paths(patterns: &[&str], facts: &[&str]) -> Vec<Vec<u8>> {
        let order = first_occurrence_vars(patterns);
        let mut space = Space::new();
        let mut program = String::new();
        for f in facts {
            program.push_str(f);
            program.push('\n');
        }
        space.add_all_sexpr(program.as_bytes()).unwrap();
        let mut before = BTreeSet::new();
        space
            .btm
            .try_for_each_value::<_, ()>(|p, _| {
                before.insert(p.to_vec());
                Ok(())
            })
            .unwrap();
        let body = patterns.join(" ");
        let ans_vars = order
            .iter()
            .map(|v| format!("${v}"))
            .collect::<Vec<_>>()
            .join(" ");
        let exec = format!("(exec 0 (, {body}) (, (ans {ans_vars})))\n");
        space.add_all_sexpr(exec.as_bytes()).unwrap();
        space.metta_calculus(1);
        let head = format!("[{}] ans", order.len() + 1);
        let mut out = Vec::new();
        space
            .btm
            .try_for_each_value::<_, ()>(|p, _| {
                if !before.contains(p) {
                    let s = serialize(p);
                    if s.starts_with(&head) && serialized_is_ground(&s) {
                        out.push(p.to_vec());
                    }
                }
                Ok(())
            })
            .unwrap();
        out
    }

    fn fork_projected_strings(
        head_name: &str,
        patterns: &[&str],
        facts: &[&str],
        proj: &[&str],
        ground_only: bool,
    ) -> BTreeSet<String> {
        let mut space = Space::new();
        let mut program = String::new();
        for f in facts {
            program.push_str(f);
            program.push('\n');
        }
        space.add_all_sexpr(program.as_bytes()).unwrap();
        let mut before = BTreeSet::new();
        space
            .btm
            .try_for_each_value::<_, ()>(|p, _| {
                before.insert(p.to_vec());
                Ok(())
            })
            .unwrap();
        let body = patterns.join(" ");
        let out_vars = proj
            .iter()
            .map(|v| format!("${v}"))
            .collect::<Vec<_>>()
            .join(" ");
        let exec = format!("(exec 0 (, {body}) (, ({head_name} {out_vars})))\n");
        space.add_all_sexpr(exec.as_bytes()).unwrap();
        space.metta_calculus(1);
        let head = format!("[{}] {head_name}", proj.len() + 1);
        let mut out = BTreeSet::new();
        space
            .btm
            .try_for_each_value::<_, ()>(|p, _| {
                if !before.contains(p) {
                    let s = serialize(p);
                    if s.starts_with(&head) && (!ground_only || serialized_is_ground(&s)) {
                        out.insert(s);
                    }
                }
                Ok(())
            })
            .unwrap();
        out
    }

    fn fork_answer_proj(patterns: &[&str], facts: &[&str], proj: &[&str]) -> BTreeSet<String> {
        fork_projected_strings("ans", patterns, facts, proj, false)
    }

    struct UpstreamOracleCase {
        name: String,
        patterns: Vec<String>,
        facts: Vec<String>,
        proj: Vec<String>,
    }

    fn random_oracle_term(
        r: &mut UjRng,
        depth: usize,
        allow_var: bool,
        pool: usize,
        prefix: &str,
    ) -> String {
        const SYMS: &[&str] = &["a", "b", "c", "d"];
        if depth > 0 && r.chance(2, 5) {
            let arity = 1 + r.below(2);
            let parts: Vec<String> = (0..arity)
                .map(|_| random_oracle_term(r, depth - 1, allow_var, pool, prefix))
                .collect();
            format!("({})", parts.join(" "))
        } else if allow_var && r.chance(2, 5) {
            format!("${prefix}{}", r.below(pool))
        } else {
            SYMS[r.below(SYMS.len())].to_string()
        }
    }

    fn native_oracle_cases() -> Vec<UpstreamOracleCase> {
        let mut cases: Vec<UpstreamOracleCase> = mork_uni_join::corpus::cases()
            .iter()
            .map(|case| UpstreamOracleCase {
                name: format!("corpus:{}", case.name),
                patterns: case.patterns.iter().map(|s| s.to_string()).collect(),
                facts: case.facts.iter().map(|s| s.to_string()).collect(),
                proj: case.proj.iter().map(|s| s.to_string()).collect(),
            })
            .collect();

        let mut r = UjRng(0xB17E_CAFE_DADA_5123);
        let mut added = 0usize;
        while added < 600 {
            let npat = 1 + r.below(3);
            let var_pool = 1 + r.below(3);
            let patterns: Vec<String> = (0..npat)
                .map(|_| {
                    format!(
                        "(r {} {})",
                        random_oracle_term(&mut r, 2, true, var_pool, "p"),
                        random_oracle_term(&mut r, 2, true, var_pool, "p")
                    )
                })
                .collect();
            let pat_refs: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
            let proj = first_occurrence_vars(&pat_refs);
            if proj.is_empty() {
                continue;
            }
            let nfacts = 1 + r.below(6);
            let facts: Vec<String> = (0..nfacts)
                .map(|_| {
                    let schematic = r.chance(2, 5);
                    format!(
                        "(r {} {})",
                        random_oracle_term(&mut r, 2, schematic, 3, "d"),
                        random_oracle_term(&mut r, 2, schematic, 3, "d")
                    )
                })
                .collect();
            cases.push(UpstreamOracleCase {
                name: format!("random:{added:03}"),
                patterns,
                facts,
                proj,
            });
            added += 1;
        }

        cases
    }

    fn write_upstream_oracle_runner(dir: &std::path::Path, upstream_root: &std::path::Path) {
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let pathmap_root = upstream_root.parent().unwrap().join("PathMap");
        let cargo_toml = format!(
            "[package]\n\
name = \"mork-upstream-oracle-runner\"\n\
version = \"0.1.0\"\n\
edition = \"2024\"\n\
\n\
[workspace]\n\
\n\
[dependencies]\n\
mork = {{ path = \"{}/kernel\" }}\n\
mork-expr = {{ path = \"{}/expr\" }}\n\
pathmap = {{ path = \"{}\", version = \"0.3.0\", features = [\"jemalloc\", \"arena_compact\", \"nightly\"] }}\n",
            upstream_root.display(),
            upstream_root.display(),
            pathmap_root.display()
        );
        std::fs::write(dir.join("Cargo.toml"), cargo_toml).unwrap();
        std::fs::write(
            dir.join("src/main.rs"),
            r#"use std::collections::BTreeSet;

use mork::space::Space;
use mork_expr::serialize;
use pathmap::zipper::*;

fn run(patterns: &[&str], facts: &[&str], proj: &[&str]) -> BTreeSet<String> {
    let mut space = Space::new();
    let mut program = String::new();
    for fact in facts {
        program.push_str(fact);
        program.push('\n');
    }
    if !program.is_empty() {
        space.add_all_sexpr(program.as_bytes()).unwrap();
    }
    let mut before = BTreeSet::new();
    {
        let mut rz = space.btm.read_zipper();
        while rz.to_next_val() {
            before.insert(unsafe { rz.path() }.to_vec());
        }
    }
    let body = patterns.join(" ");
    let ans_vars = proj.iter().map(|v| format!("${v}")).collect::<Vec<_>>().join(" ");
    let exec = format!("(exec 0 (, {body}) (, (ans {ans_vars})))\n");
    space.add_all_sexpr(exec.as_bytes()).unwrap();
    space.metta_calculus(1);
    let head = format!("[{}] ans", proj.len() + 1);
    let mut out = BTreeSet::new();
    {
        let mut rz = space.btm.read_zipper();
        while rz.to_next_val() {
            let p = unsafe { rz.path() }.to_vec();
            if !before.contains(&p) {
                let s = serialize(&p);
                if s.starts_with(&head) {
                    out.insert(s);
                }
            }
        }
    }
    out
}

fn main() {
    let cases_path = std::env::args().nth(1).expect("cases path");
    let cases = std::fs::read_to_string(cases_path).unwrap();
    for line in cases.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 4, "bad case line: {line}");
        let patterns: Vec<&str> = if cols[1].is_empty() { Vec::new() } else { cols[1].split('\x1f').collect() };
        let facts: Vec<&str> = if cols[2].is_empty() { Vec::new() } else { cols[2].split('\x1f').collect() };
        let proj: Vec<&str> = if cols[3].is_empty() { Vec::new() } else { cols[3].split('\x1f').collect() };
        let answers = run(&patterns, &facts, &proj);
        println!("{}\t{}", cols[0], answers.into_iter().collect::<Vec<_>>().join("\x1f"));
    }
}
"#,
        )
        .unwrap();
    }

    fn upstream_oracle_answers(cases: &[UpstreamOracleCase]) -> BTreeMap<String, BTreeSet<String>> {
        let default_upstream = "/tmp/claude-1000/-home-user-Dev/92e5e599-d4b4-4ea9-ad5a-56c12aebf5cd/scratchpad/mork-pristine";
        let upstream_root = std::env::var("MORK_UPSTREAM_ORACLE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from(default_upstream));
        assert!(
            upstream_root.join("kernel/src/space.rs").exists(),
            "upstream oracle checkout not found at {}",
            upstream_root.display()
        );

        let runner_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../target/upstream-oracle-runner");
        write_upstream_oracle_runner(&runner_dir, &upstream_root);
        let cases_path = runner_dir.join("cases.tsv");
        let mut input = String::new();
        for case in cases {
            input.push_str(&case.name);
            input.push('\t');
            input.push_str(&case.patterns.join("\x1f"));
            input.push('\t');
            input.push_str(&case.facts.join("\x1f"));
            input.push('\t');
            input.push_str(&case.proj.join("\x1f"));
            input.push('\n');
        }
        std::fs::write(&cases_path, input).unwrap();

        let output = std::process::Command::new("cargo")
            .arg("+nightly")
            .arg("run")
            .arg("--release")
            .arg("--quiet")
            .arg("--manifest-path")
            .arg(runner_dir.join("Cargo.toml"))
            .arg("--")
            .arg(&cases_path)
            .env("RUSTFLAGS", "-C target-cpu=native")
            .env("CARGO_TARGET_DIR", runner_dir.join("target"))
            .output()
            .expect("run upstream oracle");
        assert!(
            output.status.success(),
            "upstream oracle failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8(output.stdout).unwrap();
        let mut answers = BTreeMap::new();
        for line in stdout.lines() {
            let (name, rest) = line.split_once('\t').unwrap_or((line, ""));
            let set = if rest.is_empty() {
                BTreeSet::new()
            } else {
                rest.split('\x1f').map(|s| s.to_string()).collect()
            };
            answers.insert(name.to_string(), set);
        }
        answers
    }

    #[test]
    #[ignore = "shells out to the pristine upstream MORK oracle and runs 616 native exec cases"]
    fn upstream_native_capture_corpus_and_random_match() {
        let cases = native_oracle_cases();
        let upstream = upstream_oracle_answers(&cases);
        let mut failures = Vec::new();
        let mut nonempty = 0usize;
        for case in &cases {
            let pats: Vec<&str> = case.patterns.iter().map(|s| s.as_str()).collect();
            let facts: Vec<&str> = case.facts.iter().map(|s| s.as_str()).collect();
            let proj: Vec<&str> = case.proj.iter().map(|s| s.as_str()).collect();
            let fork = fork_answer_proj(&pats, &facts, &proj);
            let expected = upstream
                .get(&case.name)
                .unwrap_or_else(|| panic!("upstream oracle did not return case {}", case.name));
            if !expected.is_empty() {
                nonempty += 1;
            }
            if &fork != expected {
                failures.push(format!(
                    "{}\n  patterns={:?}\n  facts={:?}\n  proj={:?}\n  fork-only={:?}\n  upstream-only={:?}",
                    case.name,
                    case.patterns,
                    case.facts,
                    case.proj,
                    fork.difference(expected).collect::<Vec<_>>(),
                    expected.difference(&fork).collect::<Vec<_>>()
                ));
            }
        }
        eprintln!(
            "upstream native oracle: {} cases | nonempty {} | mismatches {}",
            cases.len(),
            nonempty,
            failures.len()
        );
        assert!(
            failures.is_empty(),
            "fork native exec diverged from upstream on {} case(s):\n{}",
            failures.len(),
            failures
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
        assert!(
            cases.len() >= 600,
            "oracle corpus must contain hundreds of cases"
        );
        assert!(nonempty > 50, "oracle corpus must produce real answers");
    }

    // Cross-validate the standalone unification-aware WCO-join prototype against
    // MORK's REAL ProductZipper, not the prototype's own naive oracle. For each
    // case the body runs as an exec with an `(ans <vars>)` template through the
    // actual matcher; the emitted answers are compared, rendered through the
    // fork's own `serialize`, to the prototype's routed `uni_join`. Ground answers
    // must agree byte-for-byte. The fork's exec apply keeps only ground results, so
    // a prototype answer carrying a leftover data variable has no exec counterpart;
    // those are counted, not asserted (the honest boundary).
    fn cross_check(
        patterns: &[&str],
        facts: &[&str],
    ) -> (BTreeSet<String>, BTreeSet<String>, usize, String) {
        use mork_uni_join::join::uni_join;
        use mork_uni_join::oracle::Conj;
        use mork_uni_join::term::{Term as PTerm, parse as pparse};

        let fork: BTreeSet<String> = fork_answer_paths(patterns, facts)
            .iter()
            .map(|p| serialize(p))
            .collect();

        // The prototype's routed join, rendered through the same serializer.
        let q = Conj::parse(patterns);
        let sp: Vec<PTerm> = facts.iter().map(|f| pparse(f)).collect();
        let (sols, stats) = uni_join(&q, &sp);
        let mut proto = BTreeSet::new();
        let mut nonground = 0usize;
        for sol in &sols {
            let tuple = PTerm::decode(sol);
            let args = match tuple {
                PTerm::App(a) => a,
                t => vec![t],
            };
            let mut ans = vec![PTerm::sym("ans")];
            ans.extend(args);
            let wrapped = PTerm::App(ans);
            if wrapped.is_ground() {
                proto.insert(serialize(&wrapped.encode()));
            } else {
                nonground += 1;
            }
        }
        (fork, proto, nonground, format!("{:?}", stats.path))
    }

    /// Parse var-only relational patterns like `(e $x $y)` into zipper-join factors: the relation
    /// prefix `[Arity(1+n), Sym(rel)]` and the global variable index at each column, numbered in
    /// first-occurrence order to match `fork_answer_paths`'s answer-tuple order.
    fn parse_var_only_factors(patterns: &[&str]) -> (Vec<crate::zipper_join::Factor>, usize) {
        let mut var_idx: BTreeMap<String, usize> = BTreeMap::new();
        let mut order: Vec<usize> = Vec::new();
        let mut factors = Vec::new();
        for pat in patterns {
            let inner = pat.trim().trim_start_matches('(').trim_end_matches(')');
            let toks: Vec<&str> = inner.split_whitespace().collect();
            let rel = toks[0];
            let args = &toks[1..];
            let mut prefix = vec![
                item_byte(Tag::Arity((1 + args.len()) as u8)),
                item_byte(Tag::SymbolSize(rel.len() as u8)),
            ];
            prefix.extend_from_slice(rel.as_bytes());
            let mut cols = Vec::new();
            for a in args {
                assert!(
                    a.starts_with('$'),
                    "parse_var_only_factors: non-variable arg {a}"
                );
                let name = a[1..].to_string();
                let next = var_idx.len();
                let idx = *var_idx.entry(name).or_insert_with(|| {
                    order.push(next);
                    next
                });
                cols.push(idx);
            }
            factors.push(crate::zipper_join::Factor::var_cols(prefix, cols));
        }
        (factors, var_idx.len())
    }

    /// The zipper-native unification join's ground answers, rendered as `(ans ...)` through the
    /// fork serializer, to compare against the ProductZipper.
    fn zipper_answer_strings(patterns: &[&str], facts: &[&str]) -> BTreeSet<String> {
        let (factors, nvars) = parse_var_only_factors(patterns);
        let mut space = Space::new();
        let mut prog = String::new();
        for f in facts {
            prog.push_str(f);
            prog.push('\n');
        }
        space.add_all_sexpr(prog.as_bytes()).unwrap();
        let order: Vec<usize> = (0..nvars).collect();
        let rows = crate::zipper_join::unify_join_zipper(&space.btm, &factors, &order, nvars);
        let mut out = BTreeSet::new();
        for row in &rows {
            let mut ans = vec![
                item_byte(Tag::Arity((1 + row.len()) as u8)),
                item_byte(Tag::SymbolSize(3)),
                b'a',
                b'n',
                b's',
            ];
            for val in row {
                ans.extend_from_slice(val);
            }
            out.insert(serialize(&ans));
        }
        out
    }

    /// Same, but through `unify_join_zipper_body`: the body is encoded and the factors are parsed
    /// back out of it, the path the live route takes. Validates `parse_body_factors`.
    fn zipper_body_answer_strings(patterns: &[&str], facts: &[&str]) -> BTreeSet<String> {
        let body = mork_uni_join::term::parse(&format!("(, {})", patterns.join(" "))).encode();
        let mut space = Space::new();
        let mut prog = String::new();
        for f in facts {
            prog.push_str(f);
            prog.push('\n');
        }
        space.add_all_sexpr(prog.as_bytes()).unwrap();
        let rows = crate::zipper_join::unify_join_zipper_body(&space.btm, &body)
            .expect("body is within the factor model");
        let mut out = BTreeSet::new();
        for row in &rows {
            let mut ans = vec![
                item_byte(Tag::Arity((1 + row.len()) as u8)),
                item_byte(Tag::SymbolSize(3)),
                b'a',
                b'n',
                b's',
            ];
            for val in row {
                ans.extend_from_slice(val);
            }
            out.insert(serialize(&ans));
        }
        out
    }

    fn product_answer_strings(patterns: &[&str], facts: &[&str]) -> BTreeSet<String> {
        use std::sync::atomic::Ordering;

        let old_unify = SIDECAR_UNIFY_ENABLED.swap(false, Ordering::Relaxed);
        let old_capture = SIDECAR_CAPTURE_ENABLED.swap(false, Ordering::Relaxed);
        let out = fork_answer_paths(patterns, facts)
            .iter()
            .map(|p| serialize(p))
            .collect();
        SIDECAR_CAPTURE_ENABLED.store(old_capture, Ordering::Relaxed);
        SIDECAR_UNIFY_ENABLED.store(old_unify, Ordering::Relaxed);
        out
    }

    fn product_answer_strings_all(patterns: &[&str], facts: &[&str]) -> BTreeSet<String> {
        use std::sync::atomic::Ordering;

        let old_unify = SIDECAR_UNIFY_ENABLED.swap(false, Ordering::Relaxed);
        let old_capture = SIDECAR_CAPTURE_ENABLED.swap(false, Ordering::Relaxed);
        let order = first_occurrence_vars(patterns);
        let proj: Vec<&str> = order.iter().map(String::as_str).collect();
        let out = fork_answer_proj(patterns, facts, &proj);
        SIDECAR_CAPTURE_ENABLED.store(old_capture, Ordering::Relaxed);
        SIDECAR_UNIFY_ENABLED.store(old_unify, Ordering::Relaxed);
        out
    }

    fn zipper_body_partial_answer_strings(
        patterns: &[&str],
        facts: &[&str],
    ) -> Option<(BTreeSet<String>, usize, usize)> {
        let body = mork_uni_join::term::parse(&format!("(, {})", patterns.join(" "))).encode();
        let (factors, nvars) = crate::zipper_join::parse_body_factors(&body)?;
        let mut space = Space::new();
        let mut prog = String::new();
        for f in facts {
            prog.push_str(f);
            prog.push('\n');
        }
        space.add_all_sexpr(prog.as_bytes()).unwrap();
        let order: Vec<usize> = (0..nvars).collect();
        let rows =
            crate::zipper_join::unify_join_zipper_partial(&space.btm, &factors, &order, nvars);
        let mut out = BTreeSet::new();
        let mut nonground = 0usize;
        for row in &rows {
            let mut ans = vec![
                item_byte(Tag::Arity((1 + row.len()) as u8)),
                item_byte(Tag::SymbolSize(3)),
                b'a',
                b'n',
                b's',
            ];
            let mut ground = true;
            for val in row {
                if let Some(bytes) = val {
                    if !crate::zipper_join::first_subterm_is_ground(bytes) {
                        ground = false;
                    }
                    ans.extend_from_slice(bytes);
                } else {
                    ground = false;
                    ans.push(item_byte(Tag::NewVar));
                }
            }
            if !ground {
                nonground += 1;
            }
            out.insert(serialize(&ans));
        }
        Some((out, nonground, rows.len()))
    }

    fn flat_random_ground_term(r: &mut UjRng, depth: usize) -> String {
        const SYMS: &[&str] = &["a", "b", "c", "d"];
        const HEADS: &[&str] = &["k", "h", "m"];
        if depth == 0 || r.chance(3, 5) {
            SYMS[r.below(SYMS.len())].to_string()
        } else {
            let arity = 1 + r.below(2);
            let args: Vec<String> = (0..arity)
                .map(|_| flat_random_ground_term(r, depth - 1))
                .collect();
            format!("({} {})", HEADS[r.below(HEADS.len())], args.join(" "))
        }
    }

    fn flat_random_case(r: &mut UjRng) -> (Vec<String>, Vec<String>) {
        const RELS: &[&str] = &["e", "p"];
        let var_pool = 1 + r.below(4);
        let mut patterns = Vec::new();
        let mut prefixes: Vec<(String, usize)> = Vec::new();
        let npat = 2 + r.below(3);

        for pi in 0..npat {
            let rel = RELS[r.below(RELS.len())];
            let arity = 1 + r.below(3);
            let mut args = Vec::new();
            for ai in 0..arity {
                let force_ground = pi == 0 || ai == 0 || r.chance(1, 2);
                if force_ground {
                    args.push(flat_random_ground_term(r, 2));
                } else {
                    args.push(format!("$x{}", r.below(var_pool)));
                }
            }
            if pi == 1 && !args.iter().any(|a| a.starts_with('$')) {
                let last = args.len() - 1;
                args[last] = format!("$x{}", r.below(var_pool));
            }
            patterns.push(format!("({rel} {})", args.join(" ")));
            let prefix = (rel.to_string(), arity);
            if !prefixes.contains(&prefix) {
                prefixes.push(prefix);
            }
        }

        let mut facts = Vec::new();
        for (rel, arity) in prefixes {
            let mut schematic_args = Vec::new();
            for ai in 0..arity {
                if ai == 0 {
                    schematic_args.push("$d0".to_string());
                } else {
                    schematic_args.push(flat_random_ground_term(r, 1));
                }
            }
            facts.push(format!("({rel} {})", schematic_args.join(" ")));

            let nfacts = 2 + r.below(5);
            for _ in 0..nfacts {
                let mut args = Vec::new();
                for _ in 0..arity {
                    if r.chance(2, 5) {
                        args.push(format!("$d{}", r.below(2)));
                    } else {
                        args.push(flat_random_ground_term(r, 2));
                    }
                }
                facts.push(format!("({rel} {})", args.join(" ")));
            }
        }

        (patterns, facts)
    }

    // Bridge-route equality contract: meaningful only with the bridge compiled in
    // (no-bridge builds route differently; control-verified pre-delta).
    #[cfg(feature = "sidecar_bridge_emit")]
    #[test]
    fn zipper_flat_ground_columns_match_product_zipper_random() {
        let _route_guard = sidecar_route_toggle_lock().lock().unwrap();
        let regressions: &[(&str, &[&str], &[&str], &[&str])] = &[
            (
                "all-ground factor must be checked",
                &["(e $z $z $x)", "(e b c)"],
                &["(e a a d)", "(e c c a)", "(p b c)"],
                &[],
            ),
            (
                "leading ground captures data variable",
                &["(e b b $z)"],
                &["(e $v b a)"],
                &["[2] ans a"],
            ),
        ];
        for (name, patterns, facts, expected) in regressions {
            let product = product_answer_strings_all(patterns, facts);
            let (zipper, nonground, _) =
                zipper_body_partial_answer_strings(patterns, facts).expect("regression is flat");
            assert_eq!(
                product,
                zipper,
                "{name}: zipper diverged from ProductZipper\n  product-only={:?}\n  zipper-only={:?}",
                product.difference(&zipper).collect::<Vec<_>>(),
                zipper.difference(&product).collect::<Vec<_>>()
            );
            let expected: BTreeSet<String> = expected.iter().map(|s| (*s).to_string()).collect();
            assert_eq!(
                product, expected,
                "{name}: ProductZipper regression expectation changed"
            );
            assert_eq!(
                nonground, 0,
                "{name}: fixed regression should render ground answers only"
            );
        }

        const TRIALS: usize = 4000;
        let mut r = UjRng(0xBADC_0FFE_E0DD_F00D);
        let mut leapfrog = 0usize;
        let mismatches = 0usize;
        let mut nonground_rows = 0usize;
        let mut nonground_cases = 0usize;
        let mut nonempty = 0usize;
        let mut total_rows = 0usize;
        for i in 0..TRIALS {
            let (patterns, facts) = flat_random_case(&mut r);
            let pats: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
            let fcts: Vec<&str> = facts.iter().map(|s| s.as_str()).collect();
            let product = product_answer_strings(&pats, &fcts);
            let Some((zipper, nonground, rows)) = zipper_body_partial_answer_strings(&pats, &fcts)
            else {
                panic!("trial {i}: generated non-flat body\n  patterns={patterns:?}");
            };
            let zipper_ground: BTreeSet<String> = zipper
                .iter()
                .filter(|s| serialized_is_ground(s))
                .cloned()
                .collect();
            leapfrog += 1;
            total_rows += rows;
            nonground_rows += nonground;
            if nonground > 0 {
                nonground_cases += 1;
            }
            if !product.is_empty() {
                nonempty += 1;
            }
            if product != zipper_ground {
                panic!(
                    "trial {i}: zipper diverged from ProductZipper\n  patterns={patterns:?}\n  facts={facts:?}\n  product-only={:?}\n  zipper-only={:?}",
                    product.difference(&zipper_ground).collect::<Vec<_>>(),
                    zipper_ground.difference(&product).collect::<Vec<_>>()
                );
            }
        }
        eprintln!(
            "flat ground-column differential: trials={TRIALS} leapfrog={leapfrog} mismatches={mismatches} nonempty={nonempty} nonground_cases={nonground_cases} nonground_rows={nonground_rows} total_rows={total_rows}"
        );
        assert_eq!(
            leapfrog, TRIALS,
            "every generated flat body must route to leapfrog"
        );
        assert_eq!(
            mismatches, 0,
            "zipper must match ProductZipper on every flat trial"
        );
        assert!(
            nonground_rows > 0,
            "corpus must exercise non-ground partial rows"
        );
    }

    // Bridge-route equality contract: meaningful only with the bridge compiled in
    // (no-bridge builds route differently; control-verified pre-delta).
    #[cfg(feature = "sidecar_bridge_emit")]
    #[test]
    fn zipper_compound_capture_regressions_match_product_zipper() {
        let _route_guard = sidecar_route_toggle_lock().lock().unwrap();
        let cases: &[(&str, &[&str], &[&str], &[&str], usize)] = &[
            (
                "data var captures non-ground compound and p is fixed by join",
                &["(r (a $p) b)", "(r (b) $p)"],
                &["(r $d b)", "(r a b)"],
                &["[2] ans b"],
                0,
            ),
            (
                "data var captures compound with free query variable",
                &["(r (a $p))"],
                &["(r $d)"],
                &["[2] ans $"],
                1,
            ),
            (
                "occurs check rejects data coreference cycle",
                &["(e $x (f $x))"],
                &["(e $w $w)"],
                &[],
                0,
            ),
            (
                "nested compound coreference grounded across factors",
                &["(r (a (f $x) $x) c)", "(s $x)"],
                &["(r $d c)", "(s b)"],
                &["[2] ans b"],
                0,
            ),
        ];
        for (name, patterns, facts, expected, expected_nonground) in cases {
            let product = product_answer_strings_all(patterns, facts);
            let (zipper, nonground, rows) = zipper_body_partial_answer_strings(patterns, facts)
                .expect("compound body must route");
            assert_eq!(
                product,
                zipper,
                "{name}: zipper diverged from ProductZipper\n  product-only={:?}\n  zipper-only={:?}",
                product.difference(&zipper).collect::<Vec<_>>(),
                zipper.difference(&product).collect::<Vec<_>>()
            );
            let expected: BTreeSet<String> = expected.iter().map(|s| (*s).to_string()).collect();
            assert_eq!(
                product, expected,
                "{name}: ProductZipper expectation changed"
            );
            assert_eq!(
                nonground, *expected_nonground,
                "{name}: non-ground row count changed"
            );
            assert_eq!(rows, expected.len(), "{name}: row count changed");
        }
    }

    // Bridge-route equality contract: meaningful only with the bridge compiled in
    // (no-bridge builds route differently; control-verified pre-delta).
    #[cfg(feature = "sidecar_bridge_emit")]
    #[test]
    fn zipper_goal2_boundary_regressions() {
        let _route_guard = sidecar_route_toggle_lock().lock().unwrap();
        let cases: &[(&str, &[&str], &[&str], &[&str], &[&str], bool)] = &[
            (
                "acyclic-occurs",
                &["(e $x (f $x))"],
                &["(e $w $w)", "(e v0 (f v1))"],
                &["x"],
                &[],
                true,
            ),
            (
                "fact-schematic-compound-under-ground-query",
                &["(r (a b) $y)", "(s $y $z)", "(t $z b)"],
                &["(r (a $q) v0)", "(s v0 v1)", "(t v1 b)", "(r (a c) decoy)"],
                &["y", "z"],
                &["[3] ans v0 v1"],
                true,
            ),
            (
                "join-propagated-compound-capture",
                &["(e (k $x0) $x1)", "(e (k $x1) $x2)", "(h $x2 $x0)"],
                &[
                    "(e (k $s2) v0)",
                    "(e $s1 $s1)",
                    "(h $s0 $s0)",
                    "(h junk junk)",
                ],
                &["x0", "x1"],
                &[
                    "[4] ans [2] k v0 v0 [2] k v0",
                    "[4] ans v0 [2] k v0 v0",
                    "[4] ans v0 v0 v0",
                ],
                false,
            ),
        ];
        for (name, patterns, facts, proj, expected_all_vars, should_route) in cases {
            let body = mork_uni_join::term::parse(&format!("(, {})", patterns.join(" "))).encode();
            let mut space = Space::new();
            let mut prog = String::new();
            for fact in *facts {
                prog.push_str(fact);
                prog.push('\n');
            }
            space.add_all_sexpr(prog.as_bytes()).unwrap();
            assert_eq!(
                crate::zipper_join::unify_join_zipper_body_routable(&space.btm, &body),
                *should_route,
                "{name}: self-contained gate decision changed"
            );
            assert_eq!(
                crate::zipper_join::unify_join_zipper_body_partial_safe(&space.btm, &body)
                    .is_some(),
                *should_route,
                "{name}: safe partial decision changed"
            );
            let raw = zipper_body_partial_answer_strings(patterns, facts)
                .expect("goal-2 regression bodies are valid zipper factors");
            let product = product_answer_strings_all(patterns, facts);
            let expected: BTreeSet<String> =
                expected_all_vars.iter().map(|s| (*s).to_string()).collect();
            assert_eq!(
                product, expected,
                "{name}: ProductZipper expectation changed"
            );
            assert_eq!(
                raw.0,
                product,
                "{name}: raw zipper diverged from ProductZipper\n  product-only={:?}\n  zipper-only={:?}",
                product.difference(&raw.0).collect::<Vec<_>>(),
                raw.0.difference(&product).collect::<Vec<_>>()
            );
            let pattern_vec: Vec<String> = patterns.iter().map(|s| (*s).to_string()).collect();
            let fact_vec: Vec<String> = facts.iter().map(|s| (*s).to_string()).collect();
            let proj_vec: Vec<String> = proj.iter().map(|s| (*s).to_string()).collect();
            let (live, _, zrec, live_nonground) =
                uj_live_run_raw(&fact_vec, &pattern_vec, &proj_vec, true);
            let (product_live, _, _, product_live_nonground) =
                uj_live_run_raw(&fact_vec, &pattern_vec, &proj_vec, false);
            assert_eq!(
                live,
                product_live,
                "{name}: live zipper route diverged from ProductZipper\n  zipper-only={:?}\n  product-only={:?}",
                live.difference(&product_live).collect::<Vec<_>>(),
                product_live.difference(&live).collect::<Vec<_>>()
            );
            assert_eq!(
                live_nonground, product_live_nonground,
                "{name}: non-ground output count changed"
            );
            if *should_route {
                assert!(zrec > 0, "{name}: live zipper route should fire");
            } else {
                assert_eq!(zrec, 0, "{name}: live zipper route should decline");
            }
        }
    }

    #[test]
    fn zipper_body_factors_match_product_zipper() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            (
                "cyclic triangle",
                &["(e $x $y)", "(e $y $z)", "(e $z $x)"],
                &["(e a b)", "(e b c)", "(e c a)", "(e a a)", "(e b d)"],
            ),
            (
                "path with a schematic edge",
                &["(e $x $y)", "(e $y $z)"],
                &["(e a b)", "(e $u $u)", "(e b c)", "(e c d)"],
            ),
            (
                "leading constant",
                &["(e a $y)", "(e $y $z)"],
                &["(e a b)", "(e b c)", "(e c d)", "(e a x)", "(e x y)"],
            ),
            (
                "shared-key conjunction with a schematic q",
                &["(p $x)", "(q $x)"],
                &["(p a)", "(p b)", "(q a)", "(q $w)"],
            ),
        ];
        for (name, pats, facts) in cases {
            let fork: BTreeSet<String> = fork_answer_paths(pats, facts)
                .iter()
                .map(|p| serialize(p))
                .collect();
            let mine = zipper_body_answer_strings(pats, facts);
            assert_eq!(
                fork, mine,
                "{name}: body-factor join diverged from the ProductZipper"
            );
        }
    }

    #[test]
    fn zipper_join_matches_product_zipper() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            (
                "compatible triangle (e $x $y)(e $y $z)(e $x $z)",
                &["(e $x $y)", "(e $y $z)", "(e $x $z)"],
                &["(e a b)", "(e a c)", "(e b c)", "(e b d)", "(e c a)"],
            ),
            (
                "cyclic triangle (e $x $y)(e $y $z)(e $z $x)",
                &["(e $x $y)", "(e $y $z)", "(e $z $x)"],
                &["(e a b)", "(e b c)", "(e c a)", "(e a a)", "(e b d)"],
            ),
            (
                "path with a schematic edge",
                &["(e $x $y)", "(e $y $z)"],
                &["(e a b)", "(e $u $u)", "(e b c)", "(e c d)"],
            ),
            (
                "coreferent schematic fact at the join",
                &["(e $x $y)", "(e $y $z)"],
                &["(e $u $u)", "(e a b)", "(e b c)"],
            ),
            (
                "shared-key conjunction (p $x)(q $x) with a schematic q",
                &["(p $x)", "(q $x)"],
                &["(p a)", "(p b)", "(q a)", "(q $w)"],
            ),
        ];
        for (name, pats, facts) in cases {
            let fork: BTreeSet<String> = fork_answer_paths(pats, facts)
                .iter()
                .map(|p| serialize(p))
                .collect();
            let mine = zipper_answer_strings(pats, facts);
            assert_eq!(
                fork,
                mine,
                "{name}: zipper join diverged from the ProductZipper\n  fork-only={:?}\n  mine-only={:?}",
                fork.difference(&mine).collect::<Vec<_>>(),
                mine.difference(&fork).collect::<Vec<_>>()
            );
        }
    }

    /// Head-to-head: the materialized unification join (read facts into a Vec, decode each to a
    /// Term, build in-memory tries, join) versus the zipper-native join (seek the PathMap trie
    /// directly). Both compute the same triangle over a hub-blowup space with schematic edges. The
    /// zipper path skips the per-flip materialization; this measures whether that pays for the
    /// trie's re-descent cost. Run with: cargo test --release bench_zipper_vs_materialized -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_zipper_vs_materialized() {
        use std::time::Instant;
        // Selective two-path (e a $y)(e $y $z) from a fixed start `a`. The relevant subgraph is
        // fixed (a -> 5 mids -> 5 each, 25 answers, ~30 facts); the relation `e` fills with junk
        // edges unreachable from `a`. The materialized join reads all of `e` to build its trie; the
        // zipper join seeks only the relevant facts, so its cost tracks the answer, not the space.
        let body = mork_uni_join::term::parse("(, (e a $y) (e $y $z))").encode();
        let p_e = vec![
            item_byte(Tag::Arity(3)),
            item_byte(Tag::SymbolSize(1)),
            b'e',
        ];
        let a = vec![item_byte(Tag::SymbolSize(1)), b'a'];
        let factors = vec![
            crate::zipper_join::Factor {
                prefix: p_e.clone(),
                cols: vec![
                    crate::zipper_join::FactorColumn::Term(crate::zipper_join::EncodedTerm {
                        bytes: a,
                        intro: 0,
                    }),
                    crate::zipper_join::FactorColumn::Var(0),
                ],
            },
            crate::zipper_join::Factor::var_cols(p_e.clone(), vec![0, 1]),
        ];
        let order = vec![0usize, 1];
        let nvars = 2usize;
        let e_prefix = p_e;
        let runs = 50u32;

        println!("\n  junk   facts   materialized      zipper     speedup");
        for &s in &[0usize, 256, 1024, 4096, 16384, 65536] {
            let mut prog = String::new();
            for t in 0..5 {
                prog.push_str(&format!("(e a t{t})\n"));
                for u in 0..5 {
                    prog.push_str(&format!("(e t{t} u{u})\n"));
                }
            }
            for j in 0..s {
                prog.push_str(&format!("(e p{j} q{j})\n"));
            }

            let mut space = Space::new();
            space.add_all_sexpr(prog.as_bytes()).unwrap();
            let nfacts = space.btm.val_count();

            // Materialized: read the relation facts, then the join decodes + builds tries.
            let t0 = Instant::now();
            let mut mat_ans = 0usize;
            for _ in 0..runs {
                let mut fact_bufs: Vec<Vec<u8>> = Vec::new();
                let mut rz = space.btm.read_zipper_at_path(&e_prefix);
                while rz.to_next_val() {
                    fact_bufs.push(rz.origin_path().to_vec());
                }
                let fact_slices: Vec<&[u8]> = fact_bufs.iter().map(Vec::as_slice).collect();
                let ans = crate::unify_join::leapfrog_unify_join_encoded(&body, &fact_slices);
                mat_ans = ans.len();
                std::hint::black_box(&ans);
            }
            let mat = t0.elapsed() / runs;

            // Zipper-native: seek the PathMap directly, no materialization.
            let t1 = Instant::now();
            let mut zip_ans = 0usize;
            for _ in 0..runs {
                let rows =
                    crate::zipper_join::unify_join_zipper(&space.btm, &factors, &order, nvars);
                zip_ans = rows.len();
                std::hint::black_box(&rows);
            }
            let zip = t1.elapsed() / runs;

            // Counts differ by design: the materialized join keeps non-ground answers, the zipper
            // join drops them (the live route's projection). Printed to confirm comparable work.
            println!(
                "{s:5} {nfacts:6}   {:>10.3?}   {:>10.3?}   {:>6.2}x   (mat_ans={mat_ans} zip_ans={zip_ans})",
                mat,
                zip,
                mat.as_secs_f64() / zip.as_secs_f64()
            );
        }
    }

    /// The decisive comparison: the zipper-native join against MORK's own ground path, the
    /// ProductZipper (`query_multi`), on the same selective two-path as the space fills with junk.
    /// This settles whether the ProductZipper already seeks the query (so the zero-copy win is
    /// only over the materialized unification join) or scans (so the zipper beats it directly).
    #[test]
    #[ignore]
    fn bench_zipper_vs_product_zipper() {
        use std::time::Instant;
        let p_e = vec![
            item_byte(Tag::Arity(3)),
            item_byte(Tag::SymbolSize(1)),
            b'e',
        ];
        let a = vec![item_byte(Tag::SymbolSize(1)), b'a'];
        let factors = vec![
            crate::zipper_join::Factor {
                prefix: p_e.clone(),
                cols: vec![
                    crate::zipper_join::FactorColumn::Term(crate::zipper_join::EncodedTerm {
                        bytes: a,
                        intro: 0,
                    }),
                    crate::zipper_join::FactorColumn::Var(0),
                ],
            },
            crate::zipper_join::Factor::var_cols(p_e, vec![0, 1]),
        ];
        let order = vec![0usize, 1];
        let nvars = 2usize;
        let runs = 50u32;

        println!("\n  junk   facts   ProductZipper      zipper     speedup");
        for &s in &[0usize, 256, 1024, 4096, 16384, 65536] {
            let mut prog = String::new();
            for t in 0..5 {
                prog.push_str(&format!("(e a t{t})\n"));
                for u in 0..5 {
                    prog.push_str(&format!("(e t{t} u{u})\n"));
                }
            }
            for j in 0..s {
                prog.push_str(&format!("(e p{j} q{j})\n"));
            }

            let mut space = Space::new();
            space.add_all_sexpr(prog.as_bytes()).unwrap();
            let nfacts = space.btm.val_count();
            // Bracket notation: (, (e a $y) (e $y $z)) with $y coreferenced across the two
            // patterns (the join). The head symbol is dropped as args[0], so it is cosmetic.
            let pat = crate::expr!(space, "[3] conj [3] e a $ [3] e _1 $");

            // ProductZipper: MORK's existing multi-pattern join over the same live snapshot.
            let t0 = Instant::now();
            let mut pz_ans = 0u64;
            for _ in 0..runs {
                let mut c = 0u64;
                Space::query_multi(&space.btm, pat, |_, _| {
                    c += 1;
                    true
                });
                pz_ans = c;
                std::hint::black_box(c);
            }
            let pz = t0.elapsed() / runs;

            // Zipper-native unification join: seek the PathMap directly, no materialization.
            let t1 = Instant::now();
            let mut zip_ans = 0usize;
            for _ in 0..runs {
                let rows =
                    crate::zipper_join::unify_join_zipper(&space.btm, &factors, &order, nvars);
                zip_ans = rows.len();
                std::hint::black_box(&rows);
            }
            let zip = t1.elapsed() / runs;

            println!(
                "{s:5} {nfacts:6}   {:>10.3?}   {:>10.3?}   {:>6.2}x   (pz_ans={pz_ans} zip_ans={zip_ans})",
                pz,
                zip,
                pz.as_secs_f64() / zip.as_secs_f64()
            );
        }
    }

    /// Beat the cycle. On the triangle (e $x $y)(e $y $z)(e $z $x) over a hub-blowup space, the WCO
    /// join must prune the s^2 two-paths the ProductZipper materializes. The zipper join re-indexes
    /// only the inverted factor (e $z $x) and seeks the other two, so it recovers worst-case
    /// optimality at the cost of one partial materialization rather than the live route's decode of
    /// every factor into tries. All three return the same triangle count.
    #[test]
    #[ignore]
    fn bench_cyclic_triangle() {
        use std::time::Instant;
        let patterns = ["(e $x $y)", "(e $y $z)", "(e $z $x)"];
        let (factors, nvars) = parse_var_only_factors(&patterns);
        let order: Vec<usize> = (0..nvars).collect();
        let body = mork_uni_join::term::parse("(, (e $x $y) (e $y $z) (e $z $x))").encode();
        let e_prefix = parse_var_only_factors(&["(e $a $b)"]).0[0].prefix.clone();
        let runs = 10u32;

        println!("\n  s   facts   ProductZipper   materialized   reindex-zipper");
        for &s in &[64usize, 128, 256, 512, 1024, 2048] {
            let mut prog = String::new();
            for i in 0..s {
                prog.push_str(&format!("(e hub o{i})\n(e i{i} hub)\n"));
            }
            for a in 0..6 {
                for b in 0..6 {
                    if a != b {
                        prog.push_str(&format!("(e c{a} c{b})\n"));
                    }
                }
            }
            let mut space = Space::new();
            space.add_all_sexpr(prog.as_bytes()).unwrap();
            let nfacts = space.btm.val_count();
            let pat = crate::expr!(space, "[4] conj [3] e $ $ [3] e _2 $ [3] e _3 _1");

            let t0 = Instant::now();
            let mut pz_ans = 0u64;
            for _ in 0..runs {
                let mut c = 0u64;
                Space::query_multi(&space.btm, pat, |_, _| {
                    c += 1;
                    true
                });
                pz_ans = c;
                std::hint::black_box(c);
            }
            let pz = t0.elapsed() / runs;

            let t1 = Instant::now();
            let mut mat_ans = 0usize;
            for _ in 0..runs {
                let mut fact_bufs: Vec<Vec<u8>> = Vec::new();
                let mut rz = space.btm.read_zipper_at_path(&e_prefix);
                while rz.to_next_val() {
                    fact_bufs.push(rz.origin_path().to_vec());
                }
                let fact_slices: Vec<&[u8]> = fact_bufs.iter().map(Vec::as_slice).collect();
                let ans = crate::unify_join::leapfrog_unify_join_encoded(&body, &fact_slices);
                mat_ans = ans.len();
                std::hint::black_box(&ans);
            }
            let mat = t1.elapsed() / runs;

            let t2 = Instant::now();
            let mut zip_ans = 0usize;
            for _ in 0..runs {
                let rows =
                    crate::zipper_join::unify_join_zipper(&space.btm, &factors, &order, nvars);
                zip_ans = rows.len();
                std::hint::black_box(&rows);
            }
            let zip = t2.elapsed() / runs;

            println!(
                "{s:5} {nfacts:6}   {pz:>11.3?}   {mat:>11.3?}   {zip:>11.3?}   (pz={pz_ans} mat={mat_ans} zip={zip_ans})"
            );
        }
    }

    /// End-to-end: the cyclic schematic body through the live flip (`metta_calculus`), the zipper
    /// kernel vs the materialized one. The zipper also skips the relation fact-read (it seeks), so the
    /// live speedup is on top of the faster join. A schematic edge on a join key reaches the route.
    #[test]
    #[ignore]
    fn bench_live_route_zipper_vs_materialized() {
        use std::sync::atomic::Ordering;
        use std::time::Instant;
        let runs = 10u32;
        let exec = "(exec 0 (, (e $x $y) (e $y $z) (e $z $x)) (, (out $x $y $z)))\n";
        println!("\n  s   facts   materialized-route   zipper-route   speedup");
        for &s in &[64usize, 256, 1024, 2048] {
            let mut prog = String::new();
            for i in 0..s {
                prog.push_str(&format!("(e hub o{i})\n(e i{i} hub)\n"));
            }
            for a in 0..6 {
                for b in 0..6 {
                    if a != b {
                        prog.push_str(&format!("(e c{a} c{b})\n"));
                    }
                }
            }
            prog.push_str("(e c0 $w)\n"); // schematic edge on a join key: reaches the unify route

            let mut nfacts = 0usize;
            let mut run_kernel = |zipper: bool| -> std::time::Duration {
                SIDECAR_ZIPPER_JOIN_ENABLED.store(zipper, Ordering::Relaxed);
                let mut total = std::time::Duration::ZERO;
                for _ in 0..runs {
                    let mut space = Space::new();
                    space.add_all_sexpr(prog.as_bytes()).unwrap();
                    space.add_all_sexpr(exec.as_bytes()).unwrap();
                    nfacts = space.btm.val_count();
                    let t = Instant::now();
                    space.metta_calculus(1);
                    total += t.elapsed();
                }
                total / runs
            };

            let mat = run_kernel(false);
            let zip = run_kernel(true);
            SIDECAR_ZIPPER_JOIN_ENABLED.store(true, Ordering::Relaxed);
            println!(
                "{s:5} {nfacts:6}   {mat:>14.3?}   {zip:>12.3?}   {:>6.2}x",
                mat.as_secs_f64() / zip.as_secs_f64()
            );
        }
    }

    #[test]
    fn zipper_join_matches_product_zipper_random() {
        struct R(u64);
        impl R {
            fn nx(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.0 = x;
                x.wrapping_mul(0x2545F4914F6CDD1D)
            }
            fn below(&mut self, n: usize) -> usize {
                (self.nx() % n as u64) as usize
            }
        }
        // In-scope corpus: flat facts only (no non-ground compounds, the case the production gate
        // declines), so the unification join must equal the ProductZipper. Columns are a small
        // symbol set or a fact-local variable (slot 0/1), giving ground, single-variable,
        // coreferent, and two-variable schematic edges.
        let shapes: &[&[&str]] = &[
            &["(e $x $y)", "(e $y $z)"],
            &["(e $x $y)", "(e $x $z)"],
            &["(e $x $y)", "(e $y $z)", "(e $z $x)"],
            &["(e $x $y)", "(e $y $z)", "(e $x $z)"],
            &["(e $w $x)", "(e $x $y)", "(e $y $z)", "(e $z $w)"],
            &["(e $x $y)", "(e $y $x)"],
        ];
        let syms = ["a", "b", "c"];
        let mut rng = R(0x9E3779B97F4A7C15);
        for seed in 0..250u64 {
            rng.0 = seed.wrapping_mul(0xD1B54A32D192ED03).wrapping_add(1);
            let nfacts = 4 + rng.below(7);
            let mut facts: Vec<String> = Vec::new();
            for _ in 0..nfacts {
                let mut col = |r: &mut R| {
                    if r.below(3) == 0 {
                        format!("$w{}", r.below(2))
                    } else {
                        syms[r.below(syms.len())].to_string()
                    }
                };
                facts.push(format!("(e {} {})", col(&mut rng), col(&mut rng)));
            }
            let fact_refs: Vec<&str> = facts.iter().map(|s| s.as_str()).collect();
            for shape in shapes {
                let fork: BTreeSet<String> = fork_answer_paths(shape, &fact_refs)
                    .iter()
                    .map(|p| serialize(p))
                    .collect();
                let mine = zipper_answer_strings(shape, &fact_refs);
                assert_eq!(
                    fork,
                    mine,
                    "seed {seed} shape {shape:?} facts {facts:?}\n  fork-only={:?}\n  mine-only={:?}",
                    fork.difference(&mine).collect::<Vec<_>>(),
                    mine.difference(&fork).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn cross_validate_prototype_against_product_zipper() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            (
                "ground triangle",
                &["(e $x $y)", "(e $y $z)", "(e $x $z)"],
                &["(e a b)", "(e a c)", "(e b c)", "(e b d)"],
            ),
            (
                "func-type unification (ground join key)",
                &["(: ($f) A)", "(: $f (-> A))"],
                &["(: (f) A)", "(: f (-> A))"],
            ),
            (
                "schematic fact admitted (var not at a join position)",
                &["(rel $x b)"],
                &["(rel a $w)", "(rel c b)", "(rel d e)"],
            ),
            (
                "shared-key conjunction",
                &["(p $x)", "(q $x)"],
                &["(p a)", "(q a)", "(q b)", "(p c)"],
            ),
        ];
        let mut mismatches = 0;
        for (name, pats, facts) in cases {
            let (fork, proto, nonground, path) = cross_check(pats, facts);
            let agree = fork == proto;
            if !agree {
                mismatches += 1;
            }
            eprintln!(
                "[{}] {name}: fork={} proto_ground={} proto_nonground={} -> {}",
                path,
                fork.len(),
                proto.len(),
                nonground,
                if agree { "AGREE" } else { "MISMATCH" }
            );
            if !agree {
                eprintln!(
                    "    fork  only: {:?}",
                    fork.difference(&proto).collect::<Vec<_>>()
                );
                eprintln!(
                    "    proto only: {:?}",
                    proto.difference(&fork).collect::<Vec<_>>()
                );
            }
        }
        assert_eq!(
            mismatches, 0,
            "prototype must agree with the ProductZipper on ground answers"
        );
    }

    // The same cross-validation over a random corpus: random conjunctive bodies
    // (1-3 patterns over relation `r`) against random spaces (~40% schematic facts),
    // the prototype's routed `uni_join` against MORK's real ProductZipper. Ground
    // answers must agree on every case. Exercises both routing paths.
    #[test]
    fn cross_validate_random_against_product_zipper() {
        struct R(u64);
        impl R {
            fn nx(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.0 = x;
                x.wrapping_mul(0x2545F4914F6CDD1D)
            }
            fn below(&mut self, n: usize) -> usize {
                (self.nx() % n as u64) as usize
            }
            fn chance(&mut self, a: usize, b: usize) -> bool {
                self.below(b) < a
            }
        }
        const SYMS: &[&str] = &["a", "b", "c"];
        // A random term string: a symbol, a variable `$<prefix><k>`, or a small compound.
        fn rand_term(
            r: &mut R,
            depth: usize,
            allow_var: bool,
            pool: usize,
            prefix: &str,
        ) -> String {
            if depth == 0 || r.chance(3, 5) {
                if allow_var && r.chance(2, 5) {
                    format!("${prefix}{}", r.below(pool))
                } else {
                    SYMS[r.below(SYMS.len())].to_string()
                }
            } else {
                let arity = 1 + r.below(2);
                let parts: Vec<String> = (0..arity)
                    .map(|_| rand_term(r, depth - 1, allow_var, pool, prefix))
                    .collect();
                format!("({})", parts.join(" "))
            }
        }

        let mut r = R(0xDEAD_BEEF_1234_5678);
        // exact: identical ground answers. superset: the prototype's full unification finds ground
        // answers the native matcher did not emit. fork_extra: MORK finds a ground answer the
        // prototype misses, which would be a prototype completeness bug and must stay zero.
        let mut exact = 0;
        let mut superset = 0;
        let mut fork_extra = 0;
        let mut leapfrog = 0;
        let mut coupled = 0;
        let mut nonempty = 0;
        let mut nonground_dropped = 0;
        const N: usize = 500;
        for _ in 0..N {
            let npat = 1 + r.below(3);
            let var_pool = 1 + r.below(2);
            // Pattern vars share the `p` scope across patterns; each fact's `d` vars
            // are independent (parsed per fact on both sides).
            let patterns: Vec<String> = (0..npat)
                .map(|_| {
                    format!(
                        "(r {} {})",
                        rand_term(&mut r, 2, true, var_pool, "p"),
                        rand_term(&mut r, 1, true, var_pool, "p")
                    )
                })
                .collect();
            let nfacts = r.below(6);
            let facts: Vec<String> = (0..nfacts)
                .map(|_| {
                    let sch = r.chance(2, 5);
                    format!(
                        "(r {} {})",
                        rand_term(&mut r, 2, sch, 2, "d"),
                        rand_term(&mut r, 1, sch, 2, "d")
                    )
                })
                .collect();

            let pats: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
            let fcts: Vec<&str> = facts.iter().map(|s| s.as_str()).collect();
            let (fork, proto, nonground, path) = cross_check(&pats, &fcts);
            nonground_dropped += nonground;
            if path == "Leapfrog" {
                leapfrog += 1;
            } else {
                coupled += 1;
            }
            if !fork.is_empty() {
                nonempty += 1;
            }
            let fork_only: Vec<_> = fork.difference(&proto).collect();
            let proto_only: Vec<_> = proto.difference(&fork).collect();
            if !fork_only.is_empty() {
                fork_extra += 1;
                eprintln!(
                    "FORK-EXTRA (prototype missed a real-matcher answer) [{path}]\n  patterns={patterns:?}\n  facts={facts:?}\n  fork only: {fork_only:?}"
                );
            } else if proto_only.is_empty() {
                exact += 1;
            } else {
                superset += 1;
                if superset <= 5 {
                    eprintln!(
                        "SUPERSET (prototype-only ground answers) [{path}]\n  patterns={patterns:?}\n  facts={facts:?}\n  proto only: {proto_only:?}"
                    );
                }
            }
        }
        eprintln!(
            "random cross-validation: {N} cases | exact={exact} superset={superset} fork_extra={fork_extra} | leapfrog={leapfrog} coupled={coupled} nonempty={nonempty} nonground_dropped={nonground_dropped}"
        );
        // The soundness/completeness floor: the prototype must never miss a ground
        // answer MORK's real matcher produces.
        assert_eq!(
            fork_extra, 0,
            "prototype must never miss a ground answer the ProductZipper finds"
        );
        assert!(leapfrog > 100, "corpus must exercise the leapfrog path");
        assert!(coupled > 5, "corpus must exercise the coupled path");
        assert!(nonempty > 30, "corpus must produce real answers");
    }

    // Project the body's answers through a TEMPLATE that keeps only `proj` (a subset of the
    // body variables), exactly as the real integration does: `(exec 0 (, body) (, (out proj)))`.
    // A join variable omitted from `proj` can be bound by the match and still leave a ground
    // projected tuple. Returns the ground `out` answers the real ProductZipper emits.
    fn fork_proj(patterns: &[&str], facts: &[&str], proj: &[&str]) -> BTreeSet<String> {
        fork_projected_strings("out", patterns, facts, proj, true)
    }

    // The same projection through the prototype's FULL leapfrog-unify: decode each answer
    // tuple (positionally over the first-occurrence query-variable order), keep the `proj`
    // columns, and render the ground ones. Returns (ground answers, count of projected tuples
    // dropped for being non-ground).
    fn proto_proj(patterns: &[&str], facts: &[&str], proj: &[&str]) -> (BTreeSet<String>, usize) {
        use mork_uni_join::oracle::Conj;
        use mork_uni_join::term::{Term as PTerm, parse as pparse};
        use mork_uni_join::unijoin::leapfrog_unify_join;
        let order = first_occurrence_vars(patterns);
        let proj_pos: Vec<usize> = proj
            .iter()
            .map(|n| order.iter().position(|o| o == n).expect("proj var in body"))
            .collect();
        let q = Conj::parse(patterns);
        let sp: Vec<PTerm> = facts.iter().map(|f| pparse(f)).collect();
        let sols = leapfrog_unify_join(&q, &sp);
        let mut ground = BTreeSet::new();
        let mut nonground = 0usize;
        for key in &sols {
            let tuple = match PTerm::decode(key) {
                PTerm::App(a) => a,
                t => vec![t],
            };
            let mut out = vec![PTerm::sym("out")];
            for &p in &proj_pos {
                out.push(tuple[p].clone());
            }
            let wrapped = PTerm::App(out);
            if wrapped.is_ground() {
                ground.insert(serialize(&wrapped.encode()));
            } else {
                nonground += 1;
            }
        }
        (ground, nonground)
    }

    // The live integration's emit, validated as a function. Decode the pattern body and the
    // space's facts to the prototype's term model (MORK's own byte encoding, so coreference is
    // preserved), run the worst-case-optimal leapfrog-unification join, map each ground answer
    // back to the fork's (u8,u8) variable keys positionally (both number variables in
    // first-occurrence order across factors), and emit through the fork's OWN template
    // applicator. The output paths must equal the ProductZipper's on every routable body,
    // including the join-capture case the equality join has to decline.
    fn unify_glue_outputs(
        pat_expr: Expr,
        sources: &[ExprEnv],
        templates: &[Expr],
        facts: &[Vec<u8>],
    ) -> BTreeSet<Vec<u8>> {
        use mork_uni_join::oracle::Conj;
        use mork_uni_join::term::Term as PTerm;
        use mork_uni_join::unijoin::leapfrog_unify_join;

        // fork variable keys, densely numbered in first-occurrence order (== BindingVar order).
        let mut variable_for_key: BTreeMap<(u8, u8), crate::binding_space::BindingVar> =
            BTreeMap::new();
        for &source in sources {
            for key in Space::query_factor_variables(source) {
                let next = crate::binding_space::BindingVar(variable_for_key.len() as u8);
                variable_for_key.entry(key).or_insert(next);
            }
        }
        let mut dense_keys: Vec<(u8, u8)> = vec![(0, 0); variable_for_key.len()];
        for (&key, &bv) in &variable_for_key {
            dense_keys[bv.0 as usize] = key;
        }

        // prototype query from the pattern body bytes; drop the ',' head, keep the factors.
        let body = PTerm::decode(unsafe { &*pat_expr.span() });
        let factors: Vec<PTerm> = match body {
            PTerm::App(mut a) => {
                if !a.is_empty() {
                    a.remove(0);
                }
                a
            }
            _ => Vec::new(),
        };
        let mut query_vars = Vec::new();
        for p in &factors {
            for v in p.var_ids() {
                if !query_vars.contains(&v) {
                    query_vars.push(v);
                }
            }
        }
        assert_eq!(
            query_vars.len(),
            dense_keys.len(),
            "prototype variable count != fork key count (coreference mapping mismatch)"
        );
        let q = Conj {
            patterns: factors,
            query_vars,
        };

        let pfacts: Vec<PTerm> = facts.iter().map(|f| PTerm::decode(f)).collect();
        let answers = leapfrog_unify_join(&q, &pfacts);

        let mut out = BTreeSet::new();
        let mut buffer = template_output_buffer();
        let mut stack = Vec::new();
        let mut assignments = Vec::new();
        for key in &answers {
            let tuple = PTerm::decode(key);
            let comps: Vec<PTerm> = match tuple {
                PTerm::App(a) => a,
                t => vec![t],
            };
            if comps.len() != dense_keys.len() {
                continue;
            }
            // Bind only the GROUND components. A ground term has no variables, so its ExprEnv
            // carries no (n,v) identity that could collide; a non-ground value would alias the
            // pattern's own variables under ExprEnv's (n,v) scheme. A template that references an
            // unbound (non-ground) variable instantiates a fresh variable, yielding a non-ground
            // output the exec discards, while a template over only ground variables still emits.
            let value_bufs: Vec<Option<Vec<u8>>> = comps
                .iter()
                .map(|c| c.is_ground().then(|| c.encode()))
                .collect();
            let mut bindings: BTreeMap<(u8, u8), ExprEnv> = BTreeMap::new();
            for (i, &k) in dense_keys.iter().enumerate() {
                if let Some(bytes) = &value_bufs[i] {
                    bindings.insert(
                        k,
                        ExprEnv::new(
                            0,
                            Expr {
                                ptr: bytes.as_ptr() as *mut u8,
                            },
                        ),
                    );
                }
            }
            Space::apply_templates_from_bindings(
                &bindings,
                pat_expr,
                templates,
                &mut buffer,
                &mut stack,
                &mut assignments,
                |path| {
                    out.insert(path.to_vec());
                },
            );
        }
        out
    }

    #[test]
    fn unify_glue_emit_matches_product_zipper() {
        // (facts, pattern body, template body): routable schematic and ground bodies. Each must
        // emit byte-identically to the ProductZipper through the unification-join glue, the same
        // path the live integration takes. Includes the join-capture case (a schematic var on a
        // join key, grounded by another factor) the equality join declines.
        let cases: &[(&str, &'static str, &'static str)] = &[
            (
                "(edge a b)\n(edge b c)\n(edge c d)\n",
                "[3] , [3] edge $ $ [3] edge _2 $",
                "[2] , [3] path _1 _3",
            ),
            (
                "(edge a b)\n(edge b c)\n(edge a $w)\n",
                "[3] , [3] edge $ $ [3] edge _2 $",
                "[2] , [3] path _1 _3",
            ),
            (
                "(edge a b)\n(edge a d)\n(label b $w)\n(label d e)\n",
                "[3] , [3] edge $ $ [3] label _2 $",
                "[2] , [4] out _1 _2 _3",
            ),
            (
                "(edge a b)\n(edge b c)\n(edge c a)\n(label a la)\n(label b lb)\n(label c lc)\n",
                "[5] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1 [3] label _1 $",
                "[2] , [5] out _1 _2 _3 _4",
            ),
            (
                "(s a $w)\n(s b c)\n(s d c)\n",
                "[3] , [3] s $ c",
                "[2] , [2] got _1",
            ),
            // stored-to-stored on the OMITTED join key _2: template (path _1 _3) leaves _2 out,
            // so the non-ground _2 still yields a ground output. The glue must emit it.
            (
                "(edge a $w)\n(edge $u c)\n(edge a c)\n",
                "[3] , [3] edge $ $ [3] edge _2 $",
                "[2] , [3] path _1 _3",
            ),
        ];
        for (i, (facts, pat, tpl)) in cases.iter().enumerate() {
            let mut space = Space::new();
            space.add_all_sexpr(facts.as_bytes()).unwrap();
            let mut fact_bytes: Vec<Vec<u8>> = Vec::new();
            space
                .btm
                .try_for_each_value::<_, ()>(|p, _| {
                    fact_bytes.push(p.to_vec());
                    Ok(())
                })
                .unwrap();
            let (pat_expr, sources) = query_pattern_and_sources(&mut space, pat);
            let (tpl_expr, _) = query_pattern_and_sources(&mut space, tpl);
            let mut tpl_args = Vec::new();
            ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
            let templates: Vec<Expr> = tpl_args[1..].iter().map(|ee| ee.subsexpr()).collect();

            // The exec apply keeps only ground results, so compare the ground output sets: the
            // glue must reproduce exactly the ground outputs the ProductZipper produces.
            let ground = |s: &BTreeSet<Vec<u8>>| -> BTreeSet<Vec<u8>> {
                s.iter()
                    .filter(|p| serialized_is_ground(&serialize(p)))
                    .cloned()
                    .collect()
            };
            let product = ground(&Space::product_template_outputs(
                &space.btm, pat_expr, &templates,
            ));
            let glue = ground(&unify_glue_outputs(
                pat_expr,
                &sources,
                &templates,
                &fact_bytes,
            ));
            assert_eq!(
                glue,
                product,
                "case {i} glue != ProductZipper\n  glue-only={:?}\n  product-only={:?}",
                glue.difference(&product)
                    .map(|p| serialize(p))
                    .collect::<Vec<_>>(),
                product
                    .difference(&glue)
                    .map(|p| serialize(p))
                    .collect::<Vec<_>>()
            );
        }
    }

    // Exploratory: probe the projection boundary on the canonical hard shapes, especially a
    // join variable that is bound stored-variable <-> stored-variable and projected OUT. Prints
    // the real matcher's answers against the prototype's full unification, projected the same
    // way. No assertions: this characterizes where the two diverge so the gate can be built to
    // match the measured boundary, not a predicted one.
    #[test]
    fn probe_projection_boundary() {
        let cases: &[(&str, &[&str], &[&str], &[&str])] = &[
            // case 1: schematic on a join key, grounded by another ground factor. AGREE expected.
            (
                "join-capture ground",
                &["(e $x $y)", "(e $y $z)"],
                &["(e a b)", "(e b c)", "(e a $w)"],
                &["x", "z"],
            ),
            // case 2: two schematic facts meet at the omitted join key ($y) -> stored<->stored.
            // The discriminating answer is (out a c), which needs $w (from (e a $w)) to alias
            // $u (from (e $u c)). Does the ProductZipper emit it?
            (
                "stored<->stored omitted join",
                &["(e $x $y)", "(e $y $z)"],
                &["(e a $w)", "(e $u c)"],
                &["x", "z"],
            ),
            (
                "stored<->stored + ground",
                &["(e $x $y)", "(e $y $z)"],
                &["(e a $w)", "(e $u c)", "(e a c)"],
                &["x", "z"],
            ),
            // case 3: data-side capture of a query compound, projected. DIVERGE expected.
            (
                "data-side capture proj",
                &["(r (a $p) b)", "(r (b) $p)"],
                &["(r $d b)", "(r a b)"],
                &["p"],
            ),
            // case 4: schematic meets a constant / ground both sides. AGREE expected.
            (
                "stored meets ground both",
                &["(p $x)", "(q $x)"],
                &["(p $w)", "(q a)", "(p b)"],
                &["x"],
            ),
            (
                "stored at join, project join",
                &["(e $x $y)", "(e $y $z)"],
                &["(e a $w)", "(e $u c)"],
                &["x", "y", "z"],
            ),
            // a schematic fact whose var meets a query constant, projected away.
            (
                "stored meets query const",
                &["(s $x c)"],
                &["(s a $w)", "(s $u c)", "(s b c)"],
                &["x"],
            ),
            // capture via the JOIN: a shared query var carries a value across factors. Here $x
            // binds data var $d in factor 1 and ground compound (h a) in factor 2; full unify
            // forces $d = (h a). Does the ProductZipper propagate it?
            (
                "capture-via-join proj y",
                &["(e $x $y)", "(g $x)"],
                &["(e $d c)", "(g (h a))"],
                &["y"],
            ),
            (
                "capture-via-join proj x",
                &["(e $x $y)", "(g $x)"],
                &["(e $d c)", "(g (h a))"],
                &["x"],
            ),
            // does the ProductZipper bind a data var to a GROUND query compound? (line 6048
            // declines this regardless of groundness; if the PZ captures it, 6048 over-declines.)
            (
                "fact var vs ground query compound",
                &["(r (a b) $y)"],
                &["(r $d c)", "(r (a b) e)"],
                &["y"],
            ),
            // non-ground query compound vs fact var, but the compound's var is grounded by
            // another factor before the capture matters.
            (
                "nonground compound vs fact var, $p grounded",
                &["(r (a $p) $y)", "(z $p)"],
                &["(r $d c)", "(z b)"],
                &["y", "p"],
            ),
            // --- genuine-unification capture cases (triejoin-unification suite) ---
            // data-side coreference must force the two join vars equal: full-unify -> {(a a)} only;
            // a matcher that treats each data $u as an independent wildcard also emits (a b).
            (
                "data-coref forces join equal",
                &["(e $x $y)", "(p $x)", "(q $y)"],
                &["(e $u $u)", "(p a)", "(q a)", "(q b)"],
                &["x", "y"],
            ),
            // a data var must capture a NON-ground query compound (f $x), grounded by (s $x):
            // full-unify -> {(c b)}; native ProductZipper must emit the same answer.
            (
                "compound capture grounded by join",
                &["(r (f $x) $y)", "(s $x)"],
                &["(r $d b)", "(s c)"],
                &["x", "y"],
            ),
            // occurs: (f $x) must unify with the coreferent data var already bound to $x -> cycle -> no answer.
            (
                "occurs via data coref",
                &["(e $x (f $x))"],
                &["(e $w $w)"],
                &["x"],
            ),
        ];
        let mut divergences = 0;
        for (name, pats, facts, proj) in cases {
            let fork = fork_proj(pats, facts, proj);
            let (proto, nonground) = proto_proj(pats, facts, proj);
            let agree = fork == proto;
            if !agree {
                divergences += 1;
            }
            eprintln!(
                "[{}] {name}: fork={} proto={} proto_nonground_dropped={} -> {}",
                if agree { "AGREE" } else { "DIVERGE" },
                fork.len(),
                proto.len(),
                nonground,
                if agree { "ok" } else { "***" }
            );
            if !agree {
                eprintln!(
                    "    fork  only: {:?}",
                    fork.difference(&proto).collect::<Vec<_>>()
                );
                eprintln!(
                    "    proto only: {:?}",
                    proto.difference(&fork).collect::<Vec<_>>()
                );
            } else {
                eprintln!("    both: {:?}", fork.iter().collect::<Vec<_>>());
            }
        }
        eprintln!(
            "probe_projection_boundary: {} / {} cases diverge",
            divergences,
            cases.len()
        );
    }

    // === Triejoin-unification suite (Phase 1) ===
    // The oracle is full first-order unification with occurs-check: `proto_proj` runs the
    // prototype leapfrog-unify join, cross-validated byte-for-byte against the native
    // `leapfrog_unify_join_encoded` on 5000 schematic cases
    // (unify_join.rs::random_encoded_join_matches_prototype_byte_for_byte); its leaf unifier is
    // Robinson with occurs-check. A SWI-Prolog `unify_with_occurs_check` gold-check seals it
    // independently (unification_capture_matches_prolog). `fork_proj` is the live exec/matcher
    // route. Both are projected to ground outputs (exec keeps only ground answers).

    // Regression guard: the unification behaviour the live fast path ALREADY gets right, on data
    // that genuinely needs unification (data-side variables, flat coreference, occurs). These must
    // stay extensionally equal to full unification.
    #[test]
    fn unification_capture_handled_matches_full_unify() {
        let cases: &[(&str, &[&str], &[&str], &[&str])] = &[
            // flat data-side coreference forces the two join vars equal: only (a a), never (a b).
            (
                "flat data coref forces equal",
                &["(e $x $y)", "(p $x)", "(q $y)"],
                &["(e $u $u)", "(p a)", "(q a)", "(q b)"],
                &["x", "y"],
            ),
            // occurs through a coreferent data var -> cycle -> no answer.
            (
                "occurs via data coref yields nothing",
                &["(e $x (f $x))"],
                &["(e $w $w)"],
                &["x"],
            ),
            // a data var meeting a query constant, projected away.
            (
                "stored meets query const",
                &["(s $x c)"],
                &["(s a $w)", "(s $u c)", "(s b c)"],
                &["x"],
            ),
            // ground cyclic triangle (no data variables) — the WCO join's home turf.
            (
                "ground triangle",
                &["(e $x $y)", "(e $y $z)", "(e $z $x)"],
                &["(e a b)", "(e b c)", "(e c a)", "(e a c)"],
                &["x", "y", "z"],
            ),
        ];
        for (name, pats, facts, proj) in cases {
            let fork = fork_proj(pats, facts, proj);
            let (proto, _ng) = proto_proj(pats, facts, proj);
            assert_eq!(
                fork,
                proto,
                "case `{name}`: live route diverged from full unification\n  fork  only: {:?}\n  proto only: {:?}",
                fork.difference(&proto).collect::<Vec<_>>(),
                proto.difference(&fork).collect::<Vec<_>>()
            );
        }
    }

    // CLOSED (issue-29 / compound_capture): a data variable captures a NON-ground query compound
    // (e.g. data `(r $d b)` absorbing query `(r (f $x) $y)`, binding $d = (f $x)). Full unification
    // finds these; the ProductZipper the live fast path declines to MISSES them. With the capture
    // route ON, the live exec route equals full unification on exactly those cases. This assertion
    // IS the build's acceptance contract, now GREEN. The capture toggle is process-global, so this
    // test shares the serial-execution requirement of the live A/B tests (run --test-threads=1);
    // it saves and restores the flag, and restores BEFORE asserting so a failure never leaves it on.
    #[test]
    fn unification_capture_target_matches_full_unify() {
        #[cfg(feature = "semi_naive_ic")]
        let _sni_guard = SniDisarmGuard::new();
        let _route_guard = sidecar_route_toggle_lock().lock().unwrap();
        let prev = SIDECAR_CAPTURE_ENABLED.swap(true, std::sync::atomic::Ordering::Relaxed);
        let cases: &[(&str, &[&str], &[&str], &[&str])] = &[
            (
                "data var captures (a $p), $p fixed by join",
                &["(r (a $p) b)", "(r (b) $p)"],
                &["(r $d b)", "(r a b)"],
                &["p"],
            ),
            (
                "data var captures (a $p), $p grounded",
                &["(r (a $p) $y)", "(z $p)"],
                &["(r $d c)", "(z b)"],
                &["y", "p"],
            ),
            (
                "data var captures (f $x), x grounded by join",
                &["(r (f $x) $y)", "(s $x)"],
                &["(r $d b)", "(s c)"],
                &["x", "y"],
            ),
        ];
        let mut failures = Vec::new();
        for (name, pats, facts, proj) in cases {
            let fork = fork_proj(pats, facts, proj);
            let (proto, _ng) = proto_proj(pats, facts, proj);
            if fork != proto {
                failures.push(format!(
                    "case `{name}`: live capture route != full unification\n    proto only: {:?}\n    fork only: {:?}",
                    proto.difference(&fork).collect::<Vec<_>>(),
                    fork.difference(&proto).collect::<Vec<_>>()
                ));
            }
        }
        SIDECAR_CAPTURE_ENABLED.store(prev, std::sync::atomic::Ordering::Relaxed);
        assert!(
            failures.is_empty(),
            "the capture route MISSES a capture answer full unification finds:\n{}",
            failures.join("\n")
        );
    }

    // The capture route at scale, end to end through the LIVE flip. With the capture route ON, drive
    // random data-side-capture bodies (acyclic via `transform_via_capture`, cyclic via the sidecar
    // capture gate) through `metta_calculus` and assert the emitted ground answers equal full
    // first-order unification (`proto_proj`, the leapfrog sealed against SWI-Prolog occurs-check) —
    // and native ProductZipper. This hardens the whole production path
    // (route selection, the capture join over the live trie, the template emit, the live insert) on
    // arbitrary structure, the half the join-only differential does not reach. Process-global
    // toggle: run --test-threads=1 (it saves and restores the flag, restoring BEFORE asserting).
    // Bridge-route equality contract: meaningful only with the bridge compiled in
    // (pre-existing no-bridge failure, control-verified on the pre-delta branch).
    #[cfg(feature = "sidecar_bridge_emit")]
    #[test]
    fn capture_route_matches_full_unification_live_random() {
        #[cfg(feature = "semi_naive_ic")]
        let _sni_guard = SniDisarmGuard::new();
        use std::sync::atomic::Ordering;
        let _route_guard = sidecar_route_toggle_lock().lock().unwrap();
        let prev = SIDECAR_CAPTURE_ENABLED.swap(true, Ordering::Relaxed);
        let mut r = UjRng(0xCA97_5EED_1234_9F31);
        let mut captures = 0usize;
        let mut nonempty = 0usize;
        let mut routed = 0usize;
        let mut failures = Vec::new();
        const N: usize = 1200;
        for i in 0..N {
            let (patterns, facts) = if i % 2 == 0 {
                uj_random_acyclic_capture_case(&mut r)
            } else {
                uj_random_cyclic_case(&mut r)
            };
            let pats: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
            let fcts: Vec<&str> = facts.iter().map(|s| s.as_str()).collect();
            let order = first_occurrence_vars(&pats);
            if order.is_empty() {
                continue;
            }
            let mask = 1 + r.below((1usize << order.len()) - 1);
            let proj: Vec<String> = order
                .iter()
                .enumerate()
                .filter(|(k, _)| mask & (1 << k) != 0)
                .map(|(_, v)| v.clone())
                .collect();
            let proj_refs: Vec<&str> = proj.iter().map(|s| s.as_str()).collect();

            let before_rec = SIDECAR_CAPTURE_RECOVERS.load(Ordering::Relaxed);
            let fork = fork_proj(&pats, &fcts, &proj_refs);
            let (proto, _ng) = proto_proj(&pats, &fcts, &proj_refs);
            if SIDECAR_CAPTURE_RECOVERS.load(Ordering::Relaxed) > before_rec {
                routed += 1;
            }
            if fork != proto {
                failures.push(format!(
                    "case {i}\n  patterns={patterns:?}\n  facts={facts:?}\n  proj={proj:?}\n  proto-only={:?}\n  fork-only={:?}",
                    proto.difference(&fork).collect::<Vec<_>>(),
                    fork.difference(&proto).collect::<Vec<_>>()
                ));
            }
            if !proto.is_empty() {
                nonempty += 1;
            }
            if case_has_compound_capture(&pats, &fcts) {
                captures += 1;
            }
        }
        SIDECAR_CAPTURE_ENABLED.store(prev, Ordering::Relaxed);
        eprintln!(
            "live capture A/B: {N} cases | route fired {routed} | data-side-capture {captures} | nonempty {nonempty}"
        );
        assert!(
            failures.is_empty(),
            "the live capture route diverged from full unification on {} case(s):\n{}",
            failures.len(),
            failures
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
        assert!(
            captures > 50,
            "corpus must exercise real data-side capture: {captures}"
        );
        assert!(
            routed > 20,
            "the capture route must actually fire end to end: {routed}"
        );
        assert!(
            nonempty > 50,
            "corpus must produce real answers: {nonempty}"
        );
    }

    // The routing gate, validated at scale. A schematic body is byte-identical between full
    // leapfrog-unification and the real ProductZipper UNLESS a stored (data) variable must bind
    // a query subterm that is non-ground under one factor's own match (data-side capture of a
    // non-ground query compound). `compound_capture` is the sound, structural over-approximation
    // of that condition: a fact-variable position aligned with a NON-ground query compound. The
    // dense differential below asserts `!compound_capture => byte-identical to the ProductZipper`
    // across random bodies and random output projections (a join variable is often projected
    // out, the case the all-variable `cross_check` never reached), and that full unification
    // never misses a ProductZipper answer.

    // Sound over-approximation of "could a data variable capture a non-ground query compound
    // when this query factor matches this fact". Structural: recurse the aligned positions, flag
    // a query compound (non-ground) facing a fact variable. A query variable binds anything; a
    // fact variable facing a query symbol or a GROUND query compound is a ground capture the
    // ProductZipper also performs, so it is not flagged.
    fn proto_compound_capture(
        query: &mork_uni_join::term::Term,
        fact: &mork_uni_join::term::Term,
    ) -> bool {
        use mork_uni_join::term::Term as PTerm;
        match (query, fact) {
            (PTerm::App(_), PTerm::Var(_)) => !query.is_ground(),
            (PTerm::App(qs), PTerm::App(fs)) if qs.len() == fs.len() => qs
                .iter()
                .zip(fs.iter())
                .any(|(q, f)| proto_compound_capture(q, f)),
            _ => false,
        }
    }

    fn case_has_compound_capture(patterns: &[&str], facts: &[&str]) -> bool {
        use mork_uni_join::term::parse as pparse;
        let pats: Vec<_> = patterns.iter().map(|p| pparse(p)).collect();
        let fcts: Vec<_> = facts.iter().map(|f| pparse(f)).collect();
        pats.iter()
            .any(|p| fcts.iter().any(|f| proto_compound_capture(p, f)))
    }

    #[test]
    fn routing_gate_is_sound_under_projection() {
        struct R(u64);
        impl R {
            fn nx(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.0 = x;
                x.wrapping_mul(0x2545F4914F6CDD1D)
            }
            fn below(&mut self, n: usize) -> usize {
                (self.nx() % n as u64) as usize
            }
            fn chance(&mut self, a: usize, b: usize) -> bool {
                self.below(b) < a
            }
        }
        const SYMS: &[&str] = &["a", "b", "c"];
        // A term: a symbol, a variable `$<prefix><k>`, or a small compound. `allow_var` and the
        // pool control how often variables appear and how many distinct ones (so factors share).
        fn rt(
            r: &mut R,
            depth: usize,
            allow_var: bool,
            pool: usize,
            prefix: &str,
            compound: usize,
        ) -> String {
            if depth > 0 && r.chance(compound, 10) {
                let arity = 1 + r.below(2);
                let parts: Vec<String> = (0..arity)
                    .map(|_| rt(r, depth - 1, allow_var, pool, prefix, compound))
                    .collect();
                format!("({})", parts.join(" "))
            } else if allow_var && r.chance(2, 5) {
                format!("${prefix}{}", r.below(pool))
            } else {
                SYMS[r.below(SYMS.len())].to_string()
            }
        }

        let mut r = R(0x0123_4567_89AB_CDEF);
        let mut agree = 0usize;
        let mut diverge_capture = 0usize;
        let mut exercised_capture = 0usize;
        let mut fork_extra = 0usize; // ProductZipper found an answer full unification missed
        let mut gate_violation = 0usize; // !capture but DIVERGED -- must stay zero
        let mut coupled_nonempty = 0usize;
        const N: usize = 4000;
        for _ in 0..N {
            let npat = 1 + r.below(3);
            let pool = 1 + r.below(3);
            // Pattern args: relation `e`, two args, each a term over a shared `p` var scope with a
            // moderate chance of a compound (so data-side capture can arise).
            let patterns: Vec<String> = (0..npat)
                .map(|_| {
                    format!(
                        "(e {} {})",
                        rt(&mut r, 2, true, pool, "p", 4),
                        rt(&mut r, 2, true, pool, "p", 3)
                    )
                })
                .collect();
            // Facts: relation `e`, schematic-heavy (data vars on either side, sometimes a compound).
            let nfacts = 1 + r.below(5);
            let facts: Vec<String> = (0..nfacts)
                .map(|_| {
                    let sch = r.chance(3, 5);
                    format!(
                        "(e {} {})",
                        rt(&mut r, 2, sch, 2, "d", 3),
                        rt(&mut r, 2, sch, 2, "d", 3)
                    )
                })
                .collect();

            let pats: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
            let fcts: Vec<&str> = facts.iter().map(|s| s.as_str()).collect();

            // The query variables in first-occurrence order; pick a non-empty projection subset.
            let order = first_occurrence_vars(&pats);
            if order.is_empty() {
                continue;
            }
            let mask = 1 + r.below((1usize << order.len()) - 1);
            let proj: Vec<String> = order
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, v)| v.clone())
                .collect();
            let proj_refs: Vec<&str> = proj.iter().map(|s| s.as_str()).collect();

            let fork = fork_proj(&pats, &fcts, &proj_refs);
            let (proto, _ng) = proto_proj(&pats, &fcts, &proj_refs);
            let capture = case_has_compound_capture(&pats, &fcts);

            if !fork.is_subset(&proto) {
                fork_extra += 1;
                eprintln!(
                    "FORK-EXTRA (full unification missed a ProductZipper answer)\n  patterns={patterns:?}\n  facts={facts:?}\n  proj={proj:?}\n  fork-only={:?}",
                    fork.difference(&proto).collect::<Vec<_>>()
                );
            }
            let agreed = fork == proto;
            if agreed {
                agree += 1;
                if capture {
                    exercised_capture += 1;
                }
            } else {
                diverge_capture += 1;
                if !capture {
                    gate_violation += 1;
                    eprintln!(
                        "GATE VIOLATION (!capture but DIVERGED)\n  patterns={patterns:?}\n  facts={facts:?}\n  proj={proj:?}\n  fork-only={:?}\n  proto-only={:?}",
                        fork.difference(&proto).collect::<Vec<_>>(),
                        proto.difference(&fork).collect::<Vec<_>>()
                    );
                }
            }
            if !proto.is_empty() {
                coupled_nonempty += 1;
            }
        }
        eprintln!(
            "routing gate: {N} cases | agree={agree} diverge={diverge_capture} exercised_capture={exercised_capture} | gate_violations={gate_violation} fork_extra={fork_extra} nonempty={coupled_nonempty}"
        );
        assert_eq!(
            fork_extra, 0,
            "full unification must never miss a ProductZipper answer"
        );
        assert_eq!(
            gate_violation, 0,
            "the routing gate admitted a body that diverges from the ProductZipper"
        );
        assert_eq!(
            diverge_capture, 0,
            "native ProductZipper should match full unification on this corpus"
        );
        assert!(
            exercised_capture > 20,
            "corpus must exercise real data-side-capture cases"
        );
        assert!(coupled_nonempty > 100, "corpus must produce real answers");
    }

    // The integration's end-to-end correctness gate: drive random schematic bodies through the
    // LIVE flip (metta_calculus) twice, once with the worst-case-optimal unification route ON and
    // once OFF (forced onto the ProductZipper), and assert the emitted ground answers are
    // byte-identical. Run it isolated (`--test-threads=1`): the route toggle is process-global.
    // Bridge-route equality contract: meaningful only with the bridge compiled in
    // (no-bridge builds route differently; control-verified pre-delta).
    #[cfg(feature = "sidecar_bridge_emit")]
    #[test]
    fn unify_route_is_byte_identical_to_product_zipper_live() {
        let _route_guard = sidecar_route_toggle_lock().lock().unwrap();
        let mut r = UjRng(0x5EED_0F1C_2B3A_4D59);
        let mut recovered = 0usize;
        let mut nonempty = 0usize;
        const N: usize = 1500;
        for _ in 0..N {
            // A cyclic body is the case the worst-case-optimal join engages on (the acyclic
            // ProductZipper is already output-optimal otherwise, so the route never sees acyclic
            // bodies). Two families exercise different arities and the same join structure: an
            // arity-2 edge cycle (triangle, 4-cycle, or triangle-with-pendant), and an arity-3
            // rotation cycle. Every variable sits on a join key, so a schematic fact there is
            // never output-only-admissible to the equality join and the body reaches the route.
            // An occasional compound query argument exercises the decline (data-side capture)
            // branch, and nested schematic facts exercise the wiring at depth.
            let (patterns, facts) = uj_random_cyclic_case(&mut r);
            let pats: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
            let order = first_occurrence_vars(&pats);
            if order.is_empty() {
                continue;
            }
            let mask = 1 + r.below((1usize << order.len()) - 1);
            let proj: Vec<String> = order
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, v)| v.clone())
                .collect();

            let (routed, ne) = uj_assert_identical(&facts, &patterns, &proj);
            if routed {
                recovered += 1;
            }
            if ne {
                nonempty += 1;
            }
        }
        SIDECAR_UNIFY_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "live A/B: {N} cases | unification route taken on {recovered} | nonempty {nonempty}"
        );
        assert!(
            recovered > 30,
            "corpus must exercise the unification route: {recovered}"
        );
        assert!(
            nonempty > 50,
            "corpus must produce real answers: {nonempty}"
        );
    }

    fn compound_live_case(r: &mut UjRng) -> (&'static str, Vec<String>, Vec<String>, Vec<String>) {
        let v0 = format!("v{}", r.below(5));
        let v1 = format!("v{}", (r.below(5) + 1) % 6);
        match r.below(7) {
            0 => (
                "acyclic-free-compound",
                vec!["(r (a $p))".to_string()],
                vec!["(r $d)".to_string(), format!("(r (a {v0}))")],
                vec!["p".to_string()],
            ),
            1 => (
                "acyclic-occurs",
                vec!["(e $x (f $x))".to_string()],
                vec!["(e $w $w)".to_string(), format!("(e {v0} (f {v1}))")],
                vec!["x".to_string()],
            ),
            2 => (
                "cyclic-query-compound-capture",
                vec![
                    "(r (a $x) $y)".to_string(),
                    "(s $y $z)".to_string(),
                    "(t $z $x)".to_string(),
                ],
                vec![
                    "(r $d v0)".to_string(),
                    "(s v0 v1)".to_string(),
                    "(t v1 b)".to_string(),
                    format!("(r (a {v0}) junk)"),
                ],
                vec!["x".to_string(), "y".to_string(), "z".to_string()],
            ),
            3 => (
                "cyclic-nested-coref",
                vec![
                    "(r (a (f $x) $x) $y)".to_string(),
                    "(s $y $z)".to_string(),
                    "(t $z $x)".to_string(),
                ],
                vec![
                    "(r $d v0)".to_string(),
                    "(s v0 v1)".to_string(),
                    "(t v1 b)".to_string(),
                    "(r (a (f a) b) nope)".to_string(),
                ],
                vec!["x".to_string(), "y".to_string(), "z".to_string()],
            ),
            4 => (
                "join-propagated-compound-capture",
                vec![
                    "(e (k $x0) $x1)".to_string(),
                    "(e (k $x1) $x2)".to_string(),
                    "(h $x2 $x0)".to_string(),
                ],
                vec![
                    "(e (k $s2) v0)".to_string(),
                    "(e $s1 $s1)".to_string(),
                    "(h $s0 $s0)".to_string(),
                    "(h junk junk)".to_string(),
                ],
                vec!["x0".to_string(), "x1".to_string()],
            ),
            5 => (
                "fact-schematic-compound-under-ground-query",
                vec![
                    "(r (a b) $y)".to_string(),
                    "(s $y $z)".to_string(),
                    "(t $z b)".to_string(),
                ],
                vec![
                    "(r (a $q) v0)".to_string(),
                    "(s v0 v1)".to_string(),
                    "(t v1 b)".to_string(),
                    "(r (a c) decoy)".to_string(),
                ],
                vec!["y".to_string(), "z".to_string()],
            ),
            _ => (
                "witness-shape-with-extra-cycle",
                vec![
                    "(r (a $p) b)".to_string(),
                    "(r (b) $p)".to_string(),
                    "(u $p $q)".to_string(),
                    "(v $q $p)".to_string(),
                ],
                vec![
                    "(r $d b)".to_string(),
                    "(r a b)".to_string(),
                    "(u b c)".to_string(),
                    "(v c b)".to_string(),
                    "(u $du $du)".to_string(),
                ],
                vec!["p".to_string(), "q".to_string()],
            ),
        }
    }

    // Bridge-route equality contract: meaningful only with the bridge compiled in
    // (no-bridge builds route differently; control-verified pre-delta).
    #[cfg(feature = "sidecar_bridge_emit")]
    #[test]
    fn zipper_compound_live_differential_random() {
        let _route_guard = sidecar_route_toggle_lock().lock().unwrap();
        let mut r = UjRng(0xC0DE_CAFE_D15EA5E);
        let mut leapfrog = 0usize;
        let mut fallback = 0usize;
        let mut materialized_unify = 0usize;
        let mut nonempty = 0usize;
        let mut nonground_outputs = 0usize;
        let mut routed_shapes: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut fallback_shapes: BTreeMap<&'static str, usize> = BTreeMap::new();
        let mut materialized_shapes: BTreeMap<&'static str, usize> = BTreeMap::new();
        const N: usize = 4000;
        for i in 0..N {
            let (shape, patterns, facts, proj) = compound_live_case(&mut r);
            let (with_unify, rec, zrec, nonground) =
                uj_live_run_raw(&facts, &patterns, &proj, true);
            let (product, _, _, product_nonground) =
                uj_live_run_raw(&facts, &patterns, &proj, false);
            assert_eq!(
                with_unify,
                product,
                "compound live differential mismatch at trial {i} shape={shape}\n  patterns={patterns:?}\n  facts={facts:?}\n  proj={proj:?}\n  unify-only={:?}\n  product-only={:?}",
                with_unify.difference(&product).collect::<Vec<_>>(),
                product.difference(&with_unify).collect::<Vec<_>>()
            );
            if zrec > 0 {
                leapfrog += 1;
                *routed_shapes.entry(shape).or_default() += 1;
            } else if rec > 0 {
                materialized_unify += 1;
                *materialized_shapes.entry(shape).or_default() += 1;
            } else {
                fallback += 1;
                *fallback_shapes.entry(shape).or_default() += 1;
            }
            if !with_unify.is_empty() {
                nonempty += 1;
            }
            nonground_outputs += nonground.max(product_nonground);
        }
        eprintln!(
            "compound live differential: trials={N} leapfrog={leapfrog} materialized_unify={materialized_unify} fallback={fallback} nonempty={nonempty} nonground_outputs={nonground_outputs} routed_shapes={routed_shapes:?} materialized_shapes={materialized_shapes:?} fallback_shapes={fallback_shapes:?}"
        );
        assert_eq!(leapfrog + materialized_unify + fallback, N);
        assert!(
            leapfrog >= 1500,
            "compound corpus must exercise the zipper kernel: {leapfrog}"
        );
        assert_eq!(
            materialized_unify, 0,
            "zipper-native route should cover every routed compound shape"
        );
        assert!(
            fallback > 0,
            "compound corpus must exercise the remaining decline"
        );
        assert!(
            nonground_outputs > 0,
            "compound corpus must render non-ground outputs"
        );
        for shape in [
            "acyclic-free-compound",
            "acyclic-occurs",
            "cyclic-nested-coref",
            "cyclic-query-compound-capture",
            "fact-schematic-compound-under-ground-query",
            "witness-shape-with-extra-cycle",
        ] {
            assert!(
                routed_shapes.get(shape).copied().unwrap_or(0) > 0,
                "shape {shape} should route through the zipper kernel"
            );
        }
        assert_eq!(
            fallback_shapes
                .get("join-propagated-compound-capture")
                .copied(),
            Some(fallback),
            "only join-propagated-compound-capture should remain on ProductZipper: {fallback_shapes:?}"
        );
    }

    #[test]
    fn unify_route_declines_compound_capture_with_fact_varref() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                b"(e v0 v1)
                  (e v1 v2)
                  (e v2 v0)
                  (h v0 v1)
                  (h v1 v2)
                  (h v2 v0)
                  (e $s0 $u0)
                  (e (k v1) $s1)
                  (h $s2 $s2)
                  ",
            )
            .unwrap();
        let (pat_expr, _sources) =
            query_pattern_and_sources(&mut space, "[4] , [3] e $ $ [3] h [2] k _1 $ [3] e _2 _0");
        let body = unsafe { &*pat_expr.span() };
        assert!(
            !crate::zipper_join::unify_join_zipper_body_routable(&space.btm, body),
            "compound capture with fact VarRef must stay on ProductZipper"
        );
    }

    // Adversarial hardening of the same gate. Cyclic bodies over MIXED relations, an occasional
    // compound-wrapped endpoint (the data-side-capture trigger), and a schematic-fact generator
    // that emits coreferent facts (`(e $s $s)`), facts nested two deep (`(e (k (k $s)) v)`), and
    // compounds on either position, over a larger corpus. Every case still asserts the route is
    // byte-identical to the ProductZipper, and the corpus is checked to exercise BOTH the route
    // and the decline boundary. Run isolated (`--test-threads=1`): the route toggle is global.
    // Bridge-route equality contract: meaningful only with the bridge compiled in
    // (pre-existing no-bridge failure, control-verified on the pre-delta branch).
    #[cfg(feature = "sidecar_bridge_emit")]
    #[test]
    fn unify_route_adversarial_byte_identity() {
        #[cfg(feature = "semi_naive_ic")]
        let _sni_guard = SniDisarmGuard::new();
        let _route_guard = sidecar_route_toggle_lock().lock().unwrap();
        let mut r = UjRng(0x00C0_FFEE_1234_5678);
        let rels = ["e", "g", "h"];
        let mut routed = 0usize;
        let mut declined = 0usize;
        let mut nonempty = 0usize;
        const N: usize = 3000;
        for _ in 0..N {
            let len = 3 + r.below(2); // cycle length 3..=4
            let mut pats: Vec<String> = Vec::with_capacity(len);
            for i in 0..len {
                let rel = rels[r.below(rels.len())];
                let mut a = format!("$x{i}");
                let mut b = format!("$x{}", (i + 1) % len);
                if r.chance(1, 4) {
                    a = format!("(k {a})");
                }
                if r.chance(1, 4) {
                    b = format!("(w {b})");
                }
                pats.push(format!("({rel} {a} {b})"));
            }
            let nv = len + r.below(3);
            let mut facts: Vec<String> = Vec::new();
            for &rel in &rels {
                for i in 0..len {
                    facts.push(format!("({rel} v{i} v{})", (i + 1) % len));
                }
            }
            for _ in 0..(2 + r.below(5)) {
                let rel = rels[r.below(rels.len())];
                facts.push(format!("({rel} v{} v{})", r.below(nv), r.below(nv)));
            }
            let nsch = 1 + r.below(4);
            for k in 0..nsch {
                let rel = rels[r.below(rels.len())];
                let v = r.below(nv);
                facts.push(match r.below(8) {
                    0 => format!("({rel} v{v} $s{k})"),
                    1 => format!("({rel} $s{k} v{v})"),
                    2 => format!("({rel} $s{k} $s{k})"),
                    3 => format!("({rel} (k $s{k}) v{v})"),
                    4 => format!("({rel} (k (k $s{k})) v{v})"),
                    5 => format!("({rel} v{v} (w $s{k}))"),
                    6 => format!("({rel} (k v{v}) $s{k})"),
                    _ => format!("({rel} $s{k} $u{k})"),
                });
            }
            let pats_ref: Vec<&str> = pats.iter().map(|s| s.as_str()).collect();
            let order = first_occurrence_vars(&pats_ref);
            if order.is_empty() {
                continue;
            }
            let mask = 1 + r.below((1usize << order.len()) - 1);
            let proj: Vec<String> = order
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, v)| v.clone())
                .collect();

            let (did_route, ne) = uj_assert_identical(&facts, &pats, &proj);
            if did_route {
                routed += 1;
            } else {
                declined += 1;
            }
            if ne {
                nonempty += 1;
            }
        }
        SIDECAR_UNIFY_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "adversarial A/B: {N} cases | routed {routed} | declined {declined} | nonempty {nonempty}"
        );
        assert!(routed > 100, "corpus must exercise the route: {routed}");
        assert!(
            declined > 100,
            "corpus must exercise the decline boundary: {declined}"
        );
        assert!(
            nonempty > 100,
            "corpus must produce real answers: {nonempty}"
        );
    }

    // Data-side capture pinned against the full-unification prototype. The first
    // factor lets stored `$d0` capture `(a $p0)`, then the second factor grounds or
    // aliases `$p0`. Native ProductZipper must emit the same ground answers the
    // prototype finds.
    #[test]
    fn data_side_capture_matches_full_unification() {
        let (fork, proto, _nonground, path) = cross_check(
            &["(r (a $p0) b)", "(r (b) $p0)"],
            &["(r $d0 b)", "(r $d1 $d1)", "(r a b)"],
        );
        assert_eq!(path, "Coupled", "a data variable reaches the join position");
        assert_eq!(
            proto.len(),
            2,
            "full unification finds both spec-correct answers"
        );
        assert_eq!(
            fork, proto,
            "native ProductZipper must capture data-side variables here"
        );
    }

    // One-time tool: emit the golden fixture of REAL ProductZipper answers, so the
    // standalone prototype can validate its routed join against the actual matcher with
    // zero fork. Each line is `pat | pat ;; fact | fact ;; ans | ans`, with answers
    // rendered through the prototype's own decoder so it compares them directly. Run, then
    // paste the output into mork-uni-join/tests/mork_fixture.txt:
    //   cargo +nightly test -p mork --lib --release print_prototype_fixture -- --ignored --nocapture
    #[ignore = "regenerates the prototype's golden fixture; paste output into mork-uni-join"]
    #[test]
    fn print_prototype_fixture() {
        use mork_uni_join::term::Term as PTerm;
        let cases: &[(&[&str], &[&str])] = &[
            (
                &["(e $x $y)", "(e $y $z)", "(e $x $z)"],
                &["(e a b)", "(e a c)", "(e b c)", "(e b d)"],
            ),
            (
                &["(e $x $y)", "(e $y $z)"],
                &["(e a b)", "(e b c)", "(e c d)"],
            ),
            (&["(p $x)", "(q $x)"], &["(p a)", "(q a)", "(q b)", "(p c)"]),
            (&["(e $x $x)"], &["(e a a)", "(e a b)", "(e c c)"]),
            (&["(e $x $y)"], &["(e a b)", "(e b c)"]),
            (
                &["(: ($f) A)", "(: $f (-> A))"],
                &["(: (f) A)", "(: f (-> A))"],
            ),
            (&["(f (g $x))"], &["(f (g a))", "(f (g b))", "(f h)"]),
            (
                &["(pair $x $y)", "(pair $y $x)"],
                &["(pair a b)", "(pair b a)", "(pair c c)"],
            ),
            (&["(rel $x b)"], &["(rel a $w)", "(rel c b)", "(rel d e)"]),
            (&["(s $x c)"], &["(s a $w)", "(s $u c)", "(s b c)"]),
            (
                &["(kv k1 $v)", "(kv k2 $v)"],
                &["(kv k1 $a)", "(kv k2 x)", "(kv k1 x)"],
            ),
            (
                &["(r (a $p0) b)", "(r (b) $p0)"],
                &["(r $d0 b)", "(r $d1 $d1)", "(r a b)"],
            ),
            (
                &["(r $x $y)", "(r $y $x)"],
                &["(r a $w)", "(r b a)", "(r a b)"],
            ),
            (&["(t $x $y)"], &["(t a b)", "(t c d)"]),
            (&["(k $x)"], &["(m a)"]),
            (
                &["(w $x)", "(w (s $x))"],
                &["(w z)", "(w (s z))", "(w (s (s z)))"],
            ),
            (
                &["(c1 $x)", "(c2 $x)", "(c3 $x)"],
                &["(c1 a)", "(c2 a)", "(c3 a)", "(c1 b)", "(c2 b)"],
            ),
            (
                &["(edge $x $y)", "(edge $y $z)", "(edge $z $x)"],
                &["(edge a b)", "(edge b c)", "(edge c a)", "(edge a c)"],
            ),
            (
                &["(type $f $t)", "(val $f $v)"],
                &["(type foo int)", "(val foo num)", "(type bar str)"],
            ),
            (&["(g $x)"], &["(g a)", "(g $w)", "(g b)"]),
            (
                &["(h $x $y)", "(h $x $z)"],
                &["(h a b)", "(h a c)", "(h d e)"],
            ),
            (&["(rel a $x)"], &["(rel a (foo b))", "(rel a c)"]),
            (
                &["(m $x (n $x))"],
                &["(m a (n a))", "(m b (n c))", "(m d (n d))"],
            ),
            (
                &["(link $x $y)", "(link $y $z)", "(link $z $w)"],
                &["(link a b)", "(link b c)", "(link c d)"],
            ),
        ];
        for (pats, facts) in cases {
            let answers: BTreeSet<String> = fork_answer_paths(pats, facts)
                .iter()
                .map(|p| PTerm::decode(p).to_string())
                .collect();
            println!(
                "{} ;; {} ;; {}",
                pats.join(" | "),
                facts.join(" | "),
                answers.into_iter().collect::<Vec<_>>().join(" | ")
            );
        }
    }

    // Live real-MORK probe for data-side capture. Runs the exact witness through the actual matcher
    // (ProductZipper, capture route off) and prints what MORK emits. Upstream MORK, reference
    // Hyperon, and SWI-Prolog all return (ans b).
    //   cargo +nightly test -p mork --lib --release probe_capture_bug_live -- --ignored --nocapture
    #[ignore = "checks data-side capture under matcher optimization toggles"]
    #[test]
    fn probe_capture_bug_live() {
        use mork_uni_join::term::Term as PTerm;
        let run = |pats: &[&str], facts: &[&str]| -> Vec<String> {
            fork_answer_paths(pats, facts)
                .iter()
                .map(|p| PTerm::decode(p).to_string())
                .collect()
        };
        // sanity: the supported direction (query var binds a fact subterm) must always work.
        println!(
            "sanity (r $x b): {:?}",
            run(&["(r $x b)"], &["(r (a c) b)", "(r a b)"])
        );
        // Witness: upstream MORK and SWI-Prolog both return (ans b). Sweep matcher optimization
        // toggles so regressions show which path dropped capture.
        let wp = &["(r (a $p) b)", "(r (b) $p)"];
        let wf = &["(r $d b)", "(r a b)"];
        for &interp in &[false, true] {
            for &recheck in &[true, false] {
                for &compcap in &[true, false] {
                    super::FORCE_INTERPRETED_MATCHER.with(|c| c.set(interp));
                    super::VARREF_FAST_RECHECK.with(|c| c.set(recheck));
                    super::COMPILED_MATCHER_COMPOUND_CAPTURE.with(|c| c.set(compcap));
                    let ans = run(wp, wf);
                    let hit = if ans.iter().any(|a| a == "(ans b)") {
                        "   <-- CAPTURES (ans b)"
                    } else {
                        ""
                    };
                    println!("interp={interp} recheck={recheck} compcap={compcap} -> {ans:?}{hit}");
                }
            }
        }
        super::FORCE_INTERPRETED_MATCHER.with(|c| c.set(false));
        super::VARREF_FAST_RECHECK.with(|c| c.set(true));
        super::COMPILED_MATCHER_COMPOUND_CAPTURE.with(|c| c.set(true));
    }

    // The exec-level (ground-filtered) differential: the sidecar emit versus the
    // ProductZipper, keeping only the ground outputs the exec apply keeps. On a schematic
    // body this is the admissibility oracle. A schematic fact whose variable is output-only
    // yields only non-ground outputs (dropped on both sides), so the sidecar stays complete.
    // A fact captured to a ground answer (a variable meeting a constant or a join key) makes
    // the ProductZipper emit a ground tuple the equality join misses, so they diverge.
    fn sidecar_emit_ground_matches_product(
        facts: &str,
        pat: &'static str,
        tpl: &'static str,
    ) -> Option<bool> {
        let mut space = Space::new();
        space.add_all_sexpr(facts.as_bytes()).unwrap();
        let (pat_expr, _) = query_pattern_and_sources(&mut space, pat);
        let (tpl_expr, _) = query_pattern_and_sources(&mut space, tpl);
        let (sidecar_set, _) =
            Space::sidecar_emit_output_set(&space.btm, pat_expr, tpl_expr, false)?;
        let mut tpl_args = Vec::new();
        ExprEnv::new(0, tpl_expr).args(&mut tpl_args);
        let templates: Vec<Expr> = tpl_args.get(1..)?.iter().map(|ee| ee.subsexpr()).collect();
        let product_set = Space::product_template_outputs(&space.btm, pat_expr, &templates);
        let ground_only = |s: &BTreeSet<Vec<u8>>| -> BTreeSet<Vec<u8>> {
            s.iter()
                .filter(|p| serialized_is_ground(&serialize(p)))
                .cloned()
                .collect()
        };
        Some(ground_only(&sidecar_set) == ground_only(&product_set))
    }

    // The admissibility oracle for per-position schematic admission. Pins which schematic
    // bodies the sidecar handles soundly (ground output identical to the ProductZipper),
    // so the static gate can be checked against it. This is the safety net for the wiring.
    #[test]
    fn sidecar_admissibility_oracle() {
        // (facts, pattern body, template body, admissible)
        let cases: &[(&str, &str, &str, bool)] = &[
            // Ground transitive: trivially admissible.
            (
                "(edge a b)\n(edge b c)\n(edge c d)\n",
                "[3] , [3] edge $ $ [3] edge _2 $",
                "[2] , [3] path _1 _3",
                true,
            ),
            // Case A: a schematic variable on an output-only column ($t). Admissible.
            (
                "(edge a b)\n(edge a d)\n(label b $w)\n(label d e)\n",
                "[3] , [3] edge $ $ [3] label _2 $",
                "[2] , [4] out _1 _2 _3",
                true,
            ),
            // Case A-inert: a schematic label for an edgeless node contributes nothing.
            (
                "(edge a b)\n(edge b c)\n(edge c a)\n(label a la)\n(label b lb)\n(label c lc)\n(label zz $w)\n",
                "[5] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1 [3] label _1 $",
                "[2] , [5] out _1 _2 _3 _4",
                true,
            ),
            // Two output-only schematic facts: still admissible.
            (
                "(edge a b)\n(label a $u)\n(label b $w)\n",
                "[3] , [3] edge $ $ [3] label _1 $",
                "[2] , [4] out _1 _2 _3",
                true,
            ),
            // Case B: a schematic variable meeting a constant (b). Capture needed; NOT admissible.
            (
                "(edge a b)\n(rel b $w)\n(rel b q)\n",
                "[3] , [3] edge $ $ [3] rel _2 b",
                "[2] , [3] out _1 _2",
                false,
            ),
            // Join capture: a schematic variable at a join position ($y), grounded by the
            // other edge factor. The ProductZipper captures it; NOT admissible.
            (
                "(edge a b)\n(edge b c)\n(edge a $w)\n",
                "[3] , [3] edge $ $ [3] edge _2 $",
                "[2] , [3] path _1 _3",
                false,
            ),
        ];
        for (i, (facts, pat, tpl, admissible)) in cases.iter().enumerate() {
            let got = sidecar_emit_ground_matches_product(facts, pat, tpl)
                .unwrap_or_else(|| panic!("case {i} did not lower"));
            assert_eq!(
                got, *admissible,
                "case {i} admissibility (facts: {facts:?})"
            );
        }
    }

    // The static gate's decision for a body+space: does `schematic_facts_safe_to_admit`
    // admit it. Builds the subspace sidecar and the lowered sources the runtime gate sees.
    fn gate_admits(facts: &str, pat: &'static str) -> Option<bool> {
        let mut space = Space::new();
        space.add_all_sexpr(facts.as_bytes()).unwrap();
        let (pat_expr, _) = query_pattern_and_sources(&mut space, pat);
        let mut args = Vec::new();
        ExprEnv::new(0, pat_expr).args(&mut args);
        let sources = &args[1..];
        let sidecar = Space::build_subspace_sidecar(&space.btm, sources)?;
        Some(Space::schematic_facts_safe_to_admit(&sidecar, sources))
    }

    // The soundness gate: the static admission check must NEVER admit a body whose sidecar
    // ground output differs from the ProductZipper's. Random spaces (ground and schematic
    // facts at random positions) over body shapes that exercise admit (output-only var),
    // decline-by-constant, decline-by-join, and the triangle+pendant. For every case
    // `gate_admits ==> oracle-admissible`. The gate may be conservative (decline an
    // admissible body), but it may never admit an inadmissible one.
    #[test]
    fn gate_admissions_are_sound_random() {
        struct R(u64);
        impl R {
            fn nx(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                self.0 = x;
                x.wrapping_mul(0x2545F4914F6CDD1D)
            }
            fn below(&mut self, n: usize) -> usize {
                (self.nx() % n as u64) as usize
            }
        }
        // (pattern body, template body, relations the body reads as (name, arg count)).
        let shapes: &[(&'static str, &'static str, &[(&str, usize)])] = &[
            (
                "[5] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1 [3] label _1 $",
                "[2] , [5] out _1 _2 _3 _4",
                &[("edge", 2), ("label", 2)],
            ),
            (
                "[3] , [3] edge $ $ [3] edge _2 $",
                "[2] , [3] path _1 _3",
                &[("edge", 2)],
            ),
            (
                "[3] , [3] edge $ $ [3] rel _2 b",
                "[2] , [3] out _1 _2",
                &[("edge", 2), ("rel", 2)],
            ),
            (
                "[3] , [3] edge $ $ [3] label _1 $",
                "[2] , [4] out _1 _2 _3",
                &[("edge", 2), ("label", 2)],
            ),
            // A nested query factor: (typeof $y (arrow $a $b)) decomposes the second column.
            (
                "[3] , [3] edge $ $ [3] typeof _2 [3] arrow $ $",
                "[2] , [5] out _1 _2 _3 _4",
                &[("edge", 2), ("typeof", 2)],
            ),
        ];
        let syms = ["a", "b", "c"];
        let mut r = R(0x1234_5678_9ABC_DEF0);
        let mut admitted = 0;
        let mut checked = 0;
        for (pat, tpl, rels) in shapes {
            for _ in 0..120 {
                // Build a random space for this shape's relations: each fact's args are a
                // symbol or (one third of the time) a fresh variable, so facts land ground
                // or schematic at random positions.
                let mut facts = String::new();
                let mut vctr = 0;
                for (rel, nargs) in rels.iter() {
                    for _ in 0..(1 + r.below(4)) {
                        let mut line = format!("({rel}");
                        for _ in 0..*nargs {
                            match r.below(4) {
                                // A bare variable.
                                0 => {
                                    line.push_str(&format!(" $w{vctr}"));
                                    vctr += 1;
                                }
                                // A compound argument, ground or carrying a variable, so both
                                // nested facts and nested query factors are exercised. `(arrow
                                // X Y)` matches the nested typeof query shape; `(foo X)` does
                                // not, exercising the shape-mismatch path.
                                1 => {
                                    let (head, inner_n) = if r.below(2) == 0 {
                                        ("foo", 1)
                                    } else {
                                        ("arrow", 2)
                                    };
                                    let mut c = format!("({head}");
                                    for _ in 0..inner_n {
                                        if r.below(2) == 0 {
                                            c.push_str(&format!(" $w{vctr}"));
                                            vctr += 1;
                                        } else {
                                            c.push_str(&format!(" {}", syms[r.below(syms.len())]));
                                        }
                                    }
                                    c.push(')');
                                    line.push(' ');
                                    line.push_str(&c);
                                }
                                // A symbol.
                                _ => line.push_str(&format!(" {}", syms[r.below(syms.len())])),
                            }
                        }
                        line.push(')');
                        facts.push('\n');
                        facts.push_str(&line);
                    }
                }
                facts.push('\n');

                let (Some(gate), Some(oracle)) = (
                    gate_admits(&facts, pat),
                    sidecar_emit_ground_matches_product(&facts, pat, tpl),
                ) else {
                    continue;
                };
                checked += 1;
                if gate {
                    admitted += 1;
                    assert!(
                        oracle,
                        "UNSOUND: gate admitted a body whose sidecar output differs from the ProductZipper\n  pat={pat}\n  facts={facts}"
                    );
                }
            }
        }
        eprintln!("gate soundness: checked={checked} admitted={admitted}");
        assert!(
            checked > 200,
            "the corpus must lower a representative set, got {checked}"
        );
        assert!(
            admitted > 0,
            "the gate must admit some schematic bodies, else it is useless"
        );
    }

    #[test]
    fn gate_admits_nested_output_only_facts() {
        // A schematic fact with NESTED structure carrying the unknown at an output position.
        // The flat gate declined any nested fact; the generalized gate admits it because the
        // non-ground compound sits only on the output column $t, yielding non-ground rows the
        // exec drops. The oracle confirms the sidecar's ground output equals the ProductZipper.
        let facts = "(edge a b)\n(edge a d)\n(label b (foo $w))\n(label d (bar e))\n";
        let pat = "[3] , [3] edge $ $ [3] label _2 $";
        let tpl = "[2] , [4] out _1 _2 _3";
        assert_eq!(
            gate_admits(facts, pat),
            Some(true),
            "nested output-only schematic fact must be admitted"
        );
        assert_eq!(
            sidecar_emit_ground_matches_product(facts, pat, tpl),
            Some(true),
            "and it must be sound"
        );

        // The same nested structure at the JOIN position ($x, shared with edge) is declined:
        // a non-ground compound there is a join key the equality join cannot intersect.
        let facts2 = "(edge a b)\n(edge b c)\n(label (foo $w) lx)\n";
        let pat2 = "[3] , [3] edge $ $ [3] label _1 $";
        assert_eq!(
            gate_admits(facts2, pat2),
            Some(false),
            "a nested non-ground compound at a join position is declined"
        );
    }

    #[test]
    fn gate_declines_schematic_in_decomposed_compound() {
        // A nested query factor (typeof $y (arrow $a $b)) decomposes the second column. The
        // sidecar emits it correctly for GROUND facts, but a schematic fact with a variable
        // INSIDE the decomposed compound, (arrow int $u), breaks the equality join's
        // projection. So the gate must decline it, even though the variable is at an output
        // position; the decline is necessary, not merely conservative.
        let pat = "[3] , [3] edge $ $ [3] typeof _2 [3] arrow $ $";
        let tpl = "[2] , [5] out _1 _2 _3 _4";
        let ground = "(edge p q)\n(typeof q (arrow int str))\n";
        assert_eq!(
            sidecar_emit_ground_matches_product(ground, pat, tpl),
            Some(true),
            "a ground nested query factor lowers and is sound"
        );
        let schematic = "(edge p q)\n(typeof q (arrow int $u))\n(typeof q (arrow int str))\n";
        assert_eq!(
            sidecar_emit_ground_matches_product(schematic, pat, tpl),
            Some(false),
            "a schematic variable inside the decomposed compound is unsound"
        );
        assert_eq!(
            gate_admits(schematic, pat),
            Some(false),
            "so the gate declines it"
        );
    }

    #[ignore = "profiling harness; run under callgrind/perf, dense cyclic emit"]
    #[test]
    fn bench_triangle_dense_emit_profile() {
        use std::time::Instant;
        // A dense directed graph (near-clique) so the triangle rewrite emits a
        // large factorizable output: many (tri x y z) sharing the (tri x y *)
        // prefix. This is the workload the factorized emit targets.
        let nodes = 46usize;
        let mut space = Space::new();
        let mut program = String::new();
        for a in 0..nodes {
            for b in 0..nodes {
                if a != b {
                    program.push_str(&format!("(edge n{a} n{b})\n"));
                }
            }
        }
        space.add_all_sexpr(program.as_bytes()).unwrap();
        let start = Instant::now();
        space
            .add_all_sexpr(
                b"(exec 0 (, (edge $x $y) (edge $y $z) (edge $z $x)) (, (tri $x $y $z)))\n",
            )
            .unwrap();
        space.metta_calculus(1);
        let elapsed = start.elapsed();
        eprintln!(
            "nodes={nodes} btm_val_count={} elapsed={elapsed:?}",
            space.btm.val_count()
        );
    }

    #[cfg(feature = "semi_naive_fixpoint")]
    #[test]
    fn streaming_closure_rebuilds_after_edge_removal() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(path a b)
(path b c)
(exec 0 (, (edge $x $y) (path $y $z)) (, (path $x $z)))
"#,
            )
            .unwrap();
        assert_eq!(space.metta_calculus(1), 1);

        // Remove edge a->b and signal it (a RemoveSink transform bumps this).
        space.load_all_sexpr_impl(b"(edge a b)\n", false).unwrap();
        space.bridge_remove_gen += 1;

        // Stream edge c->d and re-fire. The closure rebuilds from the live edges
        // (b->c, c->d), so `a` cannot reach `d` any more.
        space
            .add_all_sexpr(
                br#"
(edge c d)
(exec 1 (, (edge $x $y) (path $y $z)) (, (path $x $z)))
"#,
            )
            .unwrap();
        assert_eq!(space.metta_calculus(1), 1);

        // (path a d) must NOT be derived: adding it is a genuine insert (+1). The
        // insertion-only closure would wrongly route a->b->c->d without the rebuild.
        let before = space.btm.val_count();
        space.add_all_sexpr(b"(path a d)\n").unwrap();
        assert_eq!(
            space.btm.val_count(),
            before + 1,
            "(path a d) is not derived once a->b is removed"
        );
    }

    #[test]
    fn validate_lowered_plan_against_product_confirms_match() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge a d)
(color a red)
(color b red)
(color c blue)
"#,
            )
            .unwrap();

        // Two arrangements (transitive edge) and an arrangement joined with a
        // pattern factor (edge plus the constant-bearing color) both agree with
        // the ProductZipper through the end-to-end bridge harness.
        let (transitive, _) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ $ [3] edge _2 $");
        let comparison = Space::validate_lowered_plan_against_product(&space.btm, transitive)
            .expect("transitive body lowers to a sidecar plan");
        assert!(comparison.matched);
        assert_eq!(comparison.missing_term_mappings, 0);

        let (edge_color, _) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ $ [3] color _1 red");
        let comparison = Space::validate_lowered_plan_against_product(&space.btm, edge_color)
            .expect("edge plus color body lowers to a sidecar plan");
        assert!(comparison.matched);
        assert_eq!(comparison.missing_term_mappings, 0);
    }

    #[test]
    fn validate_lowered_plan_covers_multiway_bodies() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge c a)
(edge c d)
(edge a c)
(color a red)
(color b red)
(color c blue)
(color d blue)
"#,
            )
            .unwrap();

        // Three-plus-factor bodies route to the multiway kernels (cyclic triangle
        // to GHD or trie, acyclic chain to Yannakakis, arrangement plus patterns).
        // Each lowers and matches the ProductZipper through whichever kernel the
        // planner selects.
        let bodies = [
            "[4] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1",
            "[4] , [3] edge $ $ [3] edge _2 $ [3] edge _3 $",
            "[4] , [3] edge $ $ [3] color _1 red [3] color _2 blue",
        ];
        for body in bodies {
            let (pat, _) = query_pattern_and_sources(&mut space, body);
            let comparison = Space::validate_lowered_plan_against_product(&space.btm, pat)
                .unwrap_or_else(|| panic!("multiway body should lower: {body}"));
            assert!(
                comparison.matched,
                "sidecar must match the ProductZipper for: {body}"
            );
            assert_eq!(comparison.missing_term_mappings, 0);
        }
    }

    #[test]
    fn subspace_interning_touches_only_relation_facts() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(color a red)
(color b blue)
(color c red)
(weight a 5)
"#,
            )
            .unwrap();

        let mut whole = crate::term_identity::TermIdentitySidecar::new();
        whole.extend_from_pathmap(&space.btm).unwrap();

        let edge_prefix = source_prefix(query_sources(&mut space, "[2] , [3] edge $ $")[0]);
        let mut subspace = crate::term_identity::TermIdentitySidecar::new();
        let interned = subspace
            .extend_from_pathmap_under_prefix(&space.btm, &edge_prefix)
            .unwrap();

        // The subspace intern reads only the two edge facts; the whole-space
        // intern reads all six. This is the bound the bridge relies on: a query
        // interns its own relations, not the entire space.
        assert_eq!(interned, 2);
        assert_eq!(subspace.stats().facts, 2);
        assert_eq!(whole.stats().facts, 6);
    }

    #[test]
    fn remove_fact_tombstones_revives_and_excludes_from_arrangement() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge a d)
"#,
            )
            .unwrap();

        let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&space.btm).unwrap();
        assert_eq!(sidecar.live_fact_count(), 3);

        let edge = sidecar
            .term_id_for_encoded(&encoded_expr_bytes(&mut space, "edge"))
            .unwrap();
        let descriptor = crate::arrangements::ArrangementDescriptor::new(edge, 2, [0, 1]).unwrap();
        let before =
            crate::arrangements::ArrangementIndex::build(&sidecar, descriptor.clone()).unwrap();
        assert_eq!(before.stats().rows, 3);

        // Tombstone (edge a b): gone from the live count and the arrangement;
        // removing it again is a no-op.
        let ab = encoded_expr_bytes(&mut space, "[3] edge a b");
        assert!(sidecar.remove_fact(&ab));
        assert!(!sidecar.remove_fact(&ab));
        assert_eq!(sidecar.live_fact_count(), 2);
        let after =
            crate::arrangements::ArrangementIndex::build(&sidecar, descriptor.clone()).unwrap();
        assert_eq!(after.stats().rows, 2);

        // Re-inserting revives the same fact and the row returns.
        sidecar.insert_fact(&ab).unwrap();
        assert_eq!(sidecar.live_fact_count(), 3);
        let revived = crate::arrangements::ArrangementIndex::build(&sidecar, descriptor).unwrap();
        assert_eq!(revived.stats().rows, 3);
    }

    #[test]
    fn apply_fact_delta_adds_and_removes() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
"#,
            )
            .unwrap();
        let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&space.btm).unwrap();
        assert_eq!(sidecar.live_fact_count(), 2);

        let ab = encoded_expr_bytes(&mut space, "[3] edge a b");
        let cd = encoded_expr_bytes(&mut space, "[3] edge c d");

        // Add (edge c d), remove (edge a b): a net-zero change in count, but the
        // membership shifts.
        sidecar
            .apply_fact_delta(&[cd.as_slice()], &[ab.as_slice()])
            .unwrap();
        assert_eq!(sidecar.live_fact_count(), 2);
        assert!(!sidecar.remove_fact(&ab));
        assert!(sidecar.remove_fact(&cd));
        assert_eq!(sidecar.live_fact_count(), 1);
    }

    // Stage 1 of the semi-naive delta lever: a per-step COW snapshot plus the
    // set difference both ways yields the added/removed delta the m-delta-rule
    // matches against. See kernel/resources/semi_naive_delta_design.md.
    #[test]
    fn delta_since_captures_added_and_removed_facts() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(petri a)
(petri b)
"#,
            )
            .unwrap();
        let before = space.delta_snapshot();
        assert_eq!(before.val_count(), 2);

        // Add (petri c), remove (petri a): the live space becomes {b, c}.
        let c = encoded_expr_bytes(&mut space, "[2] petri c");
        let a = encoded_expr_bytes(&mut space, "[2] petri a");
        space.btm.insert(&c[..], ());
        space.btm.remove(&a[..]);
        assert_eq!(space.btm.val_count(), 2);

        let (added, removed) = space.delta_since(&before);
        assert_eq!(added.val_count(), 1, "exactly (petri c) was added");
        assert_eq!(removed.val_count(), 1, "exactly (petri a) was removed");
        // The COW snapshot is independent of the live mutation.
        assert_eq!(before.val_count(), 2);
    }

    #[test]
    fn ground_triangle_join_is_output_optimal_like_product_zipper() {
        // A directed triangle n0 -> n1 -> n2 -> n0 plus a dangling two-path
        // n0 -> n3 -> n4 that never closes. This measures whether MORK's
        // ProductZipper wastes work on a ground conjunctive query relative to the
        // worst-case-optimal sidecar join.
        //
        // Finding: it does not. `coreferential_transition` walks the byte trie
        // enforcing coreference during the walk, so for a ground query the
        // candidates it unifies equal the complete matches (the dangling two-path
        // is pruned in the walk, never unified). The ProductZipper is already an
        // output-sensitive variable-at-a-time trie join, not the top-down
        // backtracking with a-posteriori equality checks that relational
        // e-matching (Zhang et al., POPL 2022) speeds up. So the sidecar matches
        // its efficiency here; the sidecar's speed lever is elsewhere
        // (incremental semi-naive matching across exec steps, and schematic
        // patterns where unification does real work), per
        // resources/relational_ematching_prior_art.md.
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge n0 n1)
(edge n1 n2)
(edge n2 n0)
(edge n0 n3)
(edge n3 n4)
"#,
            )
            .unwrap();

        let (product_pattern, sources) = query_pattern_and_sources(
            &mut space,
            "[4] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1",
        );
        let left = Space::query_factor_variables(sources[0]);
        let right = Space::query_factor_variables(sources[1]);

        let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&space.btm).unwrap();
        let plan = Space::lower_query_to_sidecar_plan(&sources, &mut sidecar)
            .expect("triangle body lowers to three arrangements");

        let selected = plan.execute_selected(&sidecar).unwrap();
        let product_trace = Space::trace_query_projection_product_candidates(
            &space.btm,
            product_pattern,
            [
                (BindingVar(0), left[0]),
                (BindingVar(1), left[1]),
                (BindingVar(2), right[1]),
            ],
            plan.variable_order().to_vec(),
        );

        eprintln!(
            "ground triangle: product raw_candidates={} successful={} | sidecar rows={}",
            product_trace.raw.raw_candidates,
            product_trace.successful_candidates,
            selected.relation.positive_rows().count(),
        );

        // The three rotations of the one directed triangle, and nothing wasted.
        assert_eq!(selected.relation.positive_rows().count(), 3);
        assert_eq!(product_trace.successful_candidates, 3);
        // Output-optimal: the ProductZipper unified exactly the matches, the
        // dangling two-path was pruned in the walk, never unified.
        assert_eq!(
            product_trace.raw.raw_candidates,
            product_trace.successful_candidates
        );
    }

    #[test]
    fn product_walk_steps_grow_faster_than_sidecar_trie_steps() {
        let _guard = ForceInterpretedMatcherGuard::new(true);
        // The single-shot worst-case-optimal advantage, measured. On the star
        // triangle the ProductZipper's fixed byte-order coreferential walk
        // enumerates the O(n^2) two-path intermediate (counted by the global
        // TRANSITIONS), while the sidecar's worst-case-optimal join is
        // output-sensitive (linear trie_steps). The unification count
        // (raw_candidates) is 0 for both and hides this; the walk-step count
        // reveals the gap, and the gap grows with the data, which is the
        // asymptotic speedup the bridge exists to capture.
        let sizes = [8usize, 16, 32];
        let mut product_walk = Vec::new();
        let mut trie_walk = Vec::new();
        let mut sidecar_total = Vec::new();
        for &n in &sizes {
            let mut space = Space::new();
            let mut facts = String::new();
            for i in 0..n {
                facts.push_str(&format!("(edge a s{i})\n(edge s{i} a)\n"));
            }
            space.add_all_sexpr(facts.as_bytes()).unwrap();

            let (product_pattern, sources) = query_pattern_and_sources(
                &mut space,
                "[4] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1",
            );
            let left = Space::query_factor_variables(sources[0]);
            let right = Space::query_factor_variables(sources[1]);
            let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
            sidecar.extend_from_pathmap(&space.btm).unwrap();
            let plan = Space::lower_query_to_sidecar_plan(&sources, &mut sidecar).unwrap();

            unsafe {
                TRANSITIONS = 0;
            }
            let product_trace = Space::trace_query_projection_product_candidates(
                &space.btm,
                product_pattern,
                [
                    (BindingVar(0), left[0]),
                    (BindingVar(1), left[1]),
                    (BindingVar(2), right[1]),
                ],
                plan.variable_order().to_vec(),
            );
            let product_transitions = unsafe { TRANSITIONS };
            let steps = plan
                .explain_selected_trie_trace(&sidecar)
                .unwrap()
                .trie_trace
                .as_ref()
                .map(|t| {
                    compare_query_projection_product_trace_to_trie_trace(&product_trace, t)
                        .unwrap()
                        .trie_steps
                })
                .expect("triangle selects the trie kernel");

            // Same (empty) answer, but the ProductZipper walked far more.
            assert_eq!(product_trace.successful_candidates, 0);
            assert!(
                product_transitions > steps,
                "n={n}: ProductZipper walk {product_transitions} should exceed sidecar trie steps {steps}"
            );
            product_walk.push(product_transitions);
            trie_walk.push(steps);
            // The sidecar's full work: interning the relation's facts plus the
            // worst-case-optimal join steps. The ProductZipper interns nothing but
            // walks the O(n^2) intermediate.
            sidecar_total.push(sidecar.stats().facts + steps);
        }

        // The advantage widens: as the star quadruples (8 -> 32), the
        // ProductZipper's walk grows strictly faster than the sidecar's.
        let product_growth = product_walk[2] as f64 / product_walk[0] as f64;
        let trie_growth = trie_walk[2] as f64 / trie_walk[0] as f64;
        eprintln!(
            "walk steps product={product_walk:?} sidecar={trie_walk:?} (growth product={product_growth:.1}x sidecar={trie_growth:.1}x)"
        );
        assert!(
            product_growth > trie_growth,
            "ProductZipper walk grew {product_growth:.1}x but sidecar only {trie_growth:.1}x; the gap should widen"
        );

        // End to end, the sidecar's full work (intern plus join) beats the
        // ProductZipper's walk at every size, and the advantage widens.
        eprintln!("sidecar total (intern+join)={sidecar_total:?} vs product walk={product_walk:?}");
        for i in 0..sizes.len() {
            assert!(
                sidecar_total[i] < product_walk[i],
                "n={}: sidecar work {} should beat ProductZipper walk {}",
                sizes[i],
                sidecar_total[i],
                product_walk[i]
            );
        }
        let sidecar_growth = sidecar_total[2] as f64 / sidecar_total[0] as f64;
        assert!(
            product_growth > sidecar_growth,
            "ProductZipper walk grew {product_growth:.1}x but sidecar total only {sidecar_growth:.1}x"
        );
    }

    #[test]
    fn query_projection_zipper_factors_match_selected_sidecar_contract_from_query_projection() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge Alice Bob)
(edge Bob Carol)
(edge Alice Dana)
(edge Dana Carol)
(edge Carol Erin)
(edge X Y)
"#,
            )
            .unwrap();

        let (product_pattern, sources) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ $ [3] edge _2 $");
        let left_query_variables = Space::query_factor_variables(sources[0]);
        let right_query_variables = Space::query_factor_variables(sources[1]);

        assert_eq!(left_query_variables.len(), 2);
        assert_eq!(right_query_variables.len(), 2);
        assert_eq!(
            left_query_variables[1], right_query_variables[0],
            "second edge source should share the middle variable with the first"
        );

        let mut sidecar = crate::term_identity::TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&space.btm).unwrap();
        let edge = sidecar
            .term_id_for_encoded(&encoded_expr_bytes(&mut space, "edge"))
            .unwrap();

        let descriptor = crate::arrangements::ArrangementDescriptor::new(edge, 2, [0, 1]).unwrap();
        let xy = crate::binding_plan::BindingAccessPlan::Arrangement {
            descriptor: descriptor.clone(),
            projection: crate::arrangements::ArrangementProjection::new(
                2,
                [BindingVar(0), BindingVar(1)],
                [0, 1],
            )
            .unwrap(),
        };
        let yz = crate::binding_plan::BindingAccessPlan::Arrangement {
            descriptor,
            projection: crate::arrangements::ArrangementProjection::new(
                2,
                [BindingVar(1), BindingVar(2)],
                [0, 1],
            )
            .unwrap(),
        };
        let plan = crate::binding_plan::BindingSidecarPlan::new(
            [xy, yz],
            [BindingVar(0), BindingVar(1), BindingVar(2)],
        );
        let report = plan
            .explain_selected_trie_cursor_contract(&sidecar)
            .unwrap();
        let contract = report
            .cursor_contract
            .as_ref()
            .expect("edge transitive plan should select trie cursor contract");
        let selected_variable_order = report.execution.choice.variable_order.clone();
        let factors = [
            QueryProjectionZipperRelationFactor::new(
                &space.btm,
                sources[0],
                [
                    (left_query_variables[0], BindingVar(0)),
                    (left_query_variables[1], BindingVar(1)),
                ],
                [BindingVar(1), BindingVar(0)],
            )
            .unwrap(),
            QueryProjectionZipperRelationFactor::new(
                &space.btm,
                sources[1],
                [
                    (right_query_variables[0], BindingVar(1)),
                    (right_query_variables[1], BindingVar(2)),
                ],
                [BindingVar(1), BindingVar(2)],
            )
            .unwrap(),
        ];
        let comparison = compare_query_projection_zipper_factors_to_trie_contract(
            &factors,
            &selected_variable_order,
            contract,
            |term| {
                sidecar
                    .get_term(term)
                    .map(|record| record.encoded().to_vec())
            },
        );

        assert_eq!(
            report.execution.choice.kernel,
            crate::binding_plan::BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        assert_eq!(
            selected_variable_order.as_ref(),
            [BindingVar(1), BindingVar(0), BindingVar(2)]
        );
        assert_eq!(comparison.relation_indexes, 2);
        assert_eq!(comparison.factor_requirements, 2);
        assert_eq!(comparison.contexts, 9);
        assert_eq!(comparison.matched_contexts, comparison.contexts);
        assert_eq!(comparison.mismatched_contexts, 0);
        assert_eq!(comparison.missing_factors, 0);
        assert_eq!(comparison.missing_term_mappings, 0);
        assert!(
            comparison
                .context_results
                .iter()
                .all(|context| context.matched)
        );

        let first_telemetry = sum_zipper_domain_telemetry(&factors);
        let first_cached_domains: usize = factors
            .iter()
            .map(QueryProjectionZipperRelationFactor::cached_domain_count)
            .sum();
        assert_eq!(first_telemetry.opens, comparison.contexts);
        assert!(first_telemetry.scans > 0);
        assert_eq!(first_telemetry.read_zipper_scans, first_telemetry.scans);
        assert_eq!(first_telemetry.product_zipper_scans, 0);
        assert_eq!(first_telemetry.candidates, first_telemetry.unifications);
        assert_eq!(first_cached_domains, first_telemetry.scans);

        for factor in &factors {
            factor.clear_telemetry();
        }

        let second_comparison = compare_query_projection_zipper_factors_to_trie_contract(
            &factors,
            &selected_variable_order,
            contract,
            |term| {
                sidecar
                    .get_term(term)
                    .map(|record| record.encoded().to_vec())
            },
        );
        let second_telemetry = sum_zipper_domain_telemetry(&factors);
        let returned_values: usize = second_comparison
            .context_results
            .iter()
            .map(|context| context.actual_domain.len())
            .sum();

        assert_eq!(second_comparison, comparison);
        assert_eq!(second_telemetry.opens, comparison.contexts);
        assert_eq!(second_telemetry.cache_hits, comparison.contexts);
        assert_eq!(second_telemetry.scans, 0);
        assert_eq!(second_telemetry.read_zipper_scans, 0);
        assert_eq!(second_telemetry.product_zipper_scans, 0);
        assert_eq!(second_telemetry.candidates, 0);
        assert_eq!(second_telemetry.unifications, 0);
        assert_eq!(second_telemetry.rows, 0);
        assert_eq!(second_telemetry.rows_matching_prefix, 0);
        assert_eq!(second_telemetry.domain_values, returned_values);

        let product_factors = [
            QueryProjectionZipperRelationFactor::new_product_zipper(
                &space.btm,
                sources[0],
                [
                    (left_query_variables[0], BindingVar(0)),
                    (left_query_variables[1], BindingVar(1)),
                ],
                [BindingVar(1), BindingVar(0)],
            )
            .unwrap(),
            QueryProjectionZipperRelationFactor::new_product_zipper(
                &space.btm,
                sources[1],
                [
                    (right_query_variables[0], BindingVar(1)),
                    (right_query_variables[1], BindingVar(2)),
                ],
                [BindingVar(1), BindingVar(2)],
            )
            .unwrap(),
        ];
        let product_comparison = compare_query_projection_zipper_factors_to_trie_contract(
            &product_factors,
            &selected_variable_order,
            contract,
            |term| {
                sidecar
                    .get_term(term)
                    .map(|record| record.encoded().to_vec())
            },
        );
        let product_telemetry = sum_zipper_domain_telemetry(&product_factors);

        assert_eq!(product_comparison, comparison);
        assert_eq!(product_telemetry.opens, comparison.contexts);
        assert_eq!(
            product_telemetry.scans + product_telemetry.cache_hits,
            product_telemetry.opens
        );
        assert_eq!(
            product_telemetry.product_zipper_scans,
            product_telemetry.scans
        );
        assert_eq!(product_telemetry.read_zipper_scans, 0);
        assert!(product_telemetry.scans > 0);
        assert_eq!(product_telemetry.candidates, product_telemetry.unifications);

        let selected = plan.execute_selected(&sidecar).unwrap();
        let product_trace = Space::trace_query_projection_product_candidates(
            &space.btm,
            product_pattern,
            [
                (BindingVar(0), left_query_variables[0]),
                (BindingVar(1), left_query_variables[1]),
                (BindingVar(2), right_query_variables[1]),
            ],
            selected_variable_order.clone(),
        );
        let trace_comparison = compare_query_projection_product_trace_to_binding_relation(
            &product_trace,
            &selected.relation,
            |term| {
                sidecar
                    .get_term(term)
                    .map(|record| record.encoded().to_vec())
            },
        );
        let selected_trie_trace = plan.explain_selected_trie_trace(&sidecar).unwrap();
        let trie_trace = selected_trie_trace
            .trie_trace
            .as_ref()
            .expect("selected edge plan should expose a trie trace");
        let product_vs_trie =
            compare_query_projection_product_trace_to_trie_trace(&product_trace, trie_trace)
                .unwrap();

        assert_eq!(product_trace.factor_count, 2);
        assert_eq!(
            product_trace.rows.len(),
            product_trace.successful_candidates
        );
        assert_eq!(
            product_trace.raw.successful_unifications,
            product_trace.successful_candidates
        );
        assert_eq!(
            product_trace.raw.raw_candidates,
            product_trace.raw.general_unifications
        );
        assert_eq!(
            product_trace.raw.raw_candidates,
            product_trace.raw.successful_unifications + product_trace.raw.rejected_unifications
        );
        assert_eq!(product_trace.missing_binding_rows, 0);
        assert_eq!(product_trace.non_binding_results, 0);
        assert_eq!(
            trace_comparison.product_raw_candidates,
            product_trace.raw.raw_candidates
        );
        assert_eq!(
            trace_comparison.product_rejected_candidates,
            product_trace.raw.rejected_unifications
        );
        assert_eq!(
            product_vs_trie.product_raw_candidates,
            product_trace.raw.raw_candidates
        );
        assert_eq!(
            product_vs_trie.product_successful_candidates,
            product_trace.successful_candidates
        );
        assert_eq!(
            product_vs_trie.product_rejected_candidates,
            product_trace.raw.rejected_unifications
        );
        assert_eq!(
            product_vs_trie.trie_candidate_bindings,
            selected.relation.positive_rows().count()
        );
        assert!(product_vs_trie.successful_candidate_counts_match);
        assert!(product_vs_trie.unique_row_counts_match);
        assert_eq!(product_vs_trie.raw_candidate_overhead, 0);
        assert!(product_vs_trie.trie_steps > 0);
        assert!(product_vs_trie.trie_domain_sources > 0);
        assert!(product_vs_trie.trie_cursor_seeks > 0);
        assert_eq!(trace_comparison.missing_term_mappings, 0);
        assert_eq!(trace_comparison.missing_binding_rows, 0);
        assert_eq!(trace_comparison.non_binding_results, 0);
        assert_eq!(
            trace_comparison.product_unique_rows,
            selected.relation.positive_rows().count()
        );
        assert!(trace_comparison.matched);
    }

    #[test]
    fn query_multi_raw_candidate_counters_report_unification_rejections() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge Alice Bob)
(edge Bob Carol)
(edge Alice Dana)
(edge Dana Carol)
(edge Carol Erin)
(edge X Y)
"#,
            )
            .unwrap();

        let (_, search_sources) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ $ [3] edge $ $");
        let (_, unify_sources) =
            query_pattern_and_sources(&mut space, "[3] , [3] edge $ $ [3] edge _2 $");
        let mut product = ProductZipper::new(
            space.btm.read_zipper(),
            (0..(search_sources.len() - 1)).map(|_| space.btm.read_zipper()),
        );
        reserve_query_product_buffers(&mut product);

        let mut counters = QueryProjectionProductRawCandidateCounters::default();
        let successful = Space::query_multi_raw_with_unification_sources(
            &mut product,
            &search_sources,
            &unify_sources,
            0,
            Some(&mut counters),
            |_, _| true,
        );

        assert_eq!(successful, counters.successful_unifications);
        assert_eq!(counters.raw_candidates, counters.general_unifications);
        assert_eq!(
            counters.raw_candidates,
            counters.successful_unifications + counters.rejected_unifications
        );
        assert!(counters.raw_candidates > counters.successful_unifications);
        assert!(counters.difference_rejections > 0);
        assert_eq!(counters.occurs_rejections, 0);
        assert_eq!(counters.max_iter_rejections, 0);
    }

    #[test]
    fn query_projection_side_index_reuses_pathmap_projection_maps() {
        let mut space = Space::new();
        let mut program = String::new();
        for i in 0..10 {
            program.push_str(&format!("(ProjectionReuse bucket{} item{i})\n", i % 5));
        }
        space.add_all_sexpr(program.as_bytes()).unwrap();

        let sources = query_sources(&mut space, "[2] , [3] ProjectionReuse $ $");
        let before = Space::query_projection_side_index_stats();
        let (_, first) = query_projection_maps_for_source(&space, sources[0]);
        let (_, second) = query_projection_maps_for_source(&space, sources[0]);
        let after = Space::query_projection_side_index_stats();

        assert_eq!(first.matches, 10);
        assert_eq!(second.matches, first.matches);
        assert_eq!(first.variable_maps.len(), 2);
        assert_eq!(first.variable_maps[0].val_count(), 5);
        assert_eq!(first.variable_maps[1].val_count(), 10);
        assert_eq!(
            second.variable_maps[1].val_count(),
            first.variable_maps[1].val_count()
        );
        assert!(after.inserts >= before.inserts + 1);
        assert!(after.hits >= before.hits + 1);
        assert!(after.domain_values >= before.domain_values + 1);
        assert!(after.projection_maps >= before.projection_maps + 2);
        assert!(after.avoided_projection_scans >= before.avoided_projection_scans + 1);
    }

    #[test]
    fn query_factor_plan_uses_projected_domain_for_cardinality_ties() {
        let mut space = Space::new();
        let mut program = String::new();
        for i in 0..32 {
            program.push_str(&format!("(WideAA key{i} item{i})\n"));
            program.push_str(&format!("(NaroAA bucket{} item{i})\n", i % 2));
        }
        space.add_all_sexpr(program.as_bytes()).unwrap();

        let sources = query_sources(&mut space, "[3] , [3] WideAA $ $ [3] NaroAA $ $");

        assert_eq!(Space::query_factor_plan(&space.btm, &sources), vec![1, 0]);
    }

    #[test]
    fn query_factor_plan_metrics_records_projected_domain_refinements() {
        let mut space = Space::new();
        let mut program = String::new();
        for i in 0..16 {
            program.push_str(&format!("(DomMet bucket{} item)\n", i % 4));
        }
        space.add_all_sexpr(program.as_bytes()).unwrap();

        let sources = query_sources(&mut space, "[2] , [3] DomMet $ item");
        let before = Space::query_factor_plan_metrics_snapshot();

        let _plan = Space::query_factor_plan(&space.btm, &sources);

        let after = Space::query_factor_plan_metrics_snapshot();
        assert!(after.variable_domain_refinements >= before.variable_domain_refinements + 1);
        assert!(
            after.min_variable_domain_cardinality_sum
                >= before.min_variable_domain_cardinality_sum + 4
        );
        assert!(after.max_variable_domain_cardinality >= 4);
    }

    #[test]
    fn query_factor_plan_metrics_records_shared_variable_domain_intersections() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(WideShared key0 item0)
(WideShared key1 item1)
(WideShared key2 item2)
(WideShared key3 item3)
(WideShared key4 item4)
(WideShared key5 item5)
(NarrowShared bucket0 item0)
(NarrowShared bucket1 item1)
(NarrowShared bucket2 item2)
(NarrowShared bucket3 item7)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[3] , [3] WideShared $ $ [3] NarrowShared $ _2");
        let before = Space::query_factor_plan_metrics_snapshot();

        let _plan = Space::query_factor_plan(&space.btm, &sources);

        let after = Space::query_factor_plan_metrics_snapshot();
        assert!(
            after.shared_variable_domain_intersections
                >= before.shared_variable_domain_intersections + 1
        );
        assert!(
            after.shared_variable_domain_cardinality_sum
                >= before.shared_variable_domain_cardinality_sum + 3
        );
        assert!(after.max_shared_variable_domain_cardinality >= 3);
        assert!(
            after.prunable_shared_variable_domains >= before.prunable_shared_variable_domains + 1
        );
        assert!(
            after.shared_variable_domain_product_upper_bound_sum
                >= before.shared_variable_domain_product_upper_bound_sum + 24
        );
        assert!(
            after.shared_variable_domain_pruning_upper_bound_sum
                >= before.shared_variable_domain_pruning_upper_bound_sum + 21
        );
        assert!(after.max_shared_variable_domain_product_upper_bound >= 24);
    }

    #[test]
    fn query_variable_order_steps_prefer_smallest_shared_domain() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(WideOrder key0 item0)
(WideOrder key1 item1)
(WideOrder key2 item2)
(WideOrder key3 item3)
(WideOrder key4 item4)
(WideOrder key5 item5)
(NarrowOrder bucket0 item0)
(NarrowOrder bucket1 item1)
(NarrowOrder bucket2 item2)
(NarrowOrder bucket3 item7)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[3] , [3] WideOrder $ $ [3] NarrowOrder $ _2");
        let mut prefix_cardinalities = BTreeMap::new();
        let mut shape_cardinalities = BTreeMap::new();
        let ranks = sources
            .iter()
            .copied()
            .map(|source| {
                Space::query_factor_rank(
                    &space.btm,
                    source,
                    &mut prefix_cardinalities,
                    &mut shape_cardinalities,
                )
            })
            .collect::<Vec<_>>();

        let order = QueryFactorPlanMetrics::variable_order_steps(&ranks);
        let first = &order[0];

        assert_eq!(order.len(), 3);
        assert_eq!(first.domain_cardinality, 3);
        assert_eq!(first.factor_domain_count, 2);
        assert_eq!(first.product_upper_bound, 24);
        assert_eq!(first.pruning_upper_bound, 21);
        assert!(
            order
                .windows(2)
                .all(|pair| pair[0].domain_cardinality <= pair[1].domain_cardinality)
        );
    }

    #[test]
    fn query_factor_plan_metrics_records_variable_order_bounds() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(WideVarOrder key0 item0)
(WideVarOrder key1 item1)
(WideVarOrder key2 item2)
(WideVarOrder key3 item3)
(WideVarOrder key4 item4)
(WideVarOrder key5 item5)
(NarrowVarOrder bucket0 item0)
(NarrowVarOrder bucket1 item1)
(NarrowVarOrder bucket2 item2)
(NarrowVarOrder bucket3 item7)
"#,
            )
            .unwrap();

        let sources = query_sources(
            &mut space,
            "[3] , [3] WideVarOrder $ $ [3] NarrowVarOrder $ _2",
        );
        let before = Space::query_factor_plan_metrics_snapshot();

        let _plan = Space::query_factor_plan(&space.btm, &sources);

        let after = Space::query_factor_plan_metrics_snapshot();
        assert!(after.variable_order_plans >= before.variable_order_plans + 1);
        assert!(after.variable_order_variables >= before.variable_order_variables + 3);
        assert!(
            after.variable_order_shared_variables >= before.variable_order_shared_variables + 1
        );
        assert!(
            after.variable_order_first_domain_cardinality_sum
                >= before.variable_order_first_domain_cardinality_sum + 3
        );
        assert!(
            after.variable_order_assignment_upper_bound_sum
                >= before.variable_order_assignment_upper_bound_sum + 72
        );
        assert!(after.max_variable_order_assignment_upper_bound >= 72);
        assert!(after.max_variable_order_domain_cardinality >= 6);
        assert!(
            after.variable_order_pruning_upper_bound_sum
                >= before.variable_order_pruning_upper_bound_sum + 21
        );
    }

    #[test]
    fn query_factor_plan_cache_reuses_exact_repeated_shape() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(A target left)
(A decoy left)
(A decoy right)
(Guard target)
(Rare target)
"#,
            )
            .unwrap();

        let sources = query_sources(
            &mut space,
            "[5] , [3] A $ left [2] Rare target [2] Guard $ $",
        );
        let before = Space::query_factor_plan_cache_stats();
        let first = Space::query_factor_plan(&space.btm, &sources);
        let second = Space::query_factor_plan(&space.btm, &sources);
        let after = Space::query_factor_plan_cache_stats();

        assert_eq!(first, vec![2, 1, 0, 3]);
        assert_eq!(second, first);
        assert!(after.entries >= before.entries);
        assert!(after.hits > before.hits);
        assert!(after.misses >= before.misses);
        assert!(after.inserts >= before.inserts);
    }

    #[test]
    fn query_factor_plan_cache_key_is_shape_only() {
        // The key freezes the plan per query shape: no data-dependent parts, so
        // computing it never walks the space (odd_even_sort paid O(relation) per
        // step through the old bucketed key). Growth anywhere, related or not,
        // must leave the key unchanged.
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(CacheDepEdge target left)
(CacheDepEdge decoy left)
(CacheDepEdge spare left)
(CacheDepEdge filler left)
(CacheDepGuard target)
"#,
            )
            .unwrap();

        let sources = query_sources(
            &mut space,
            "[4] , [3] CacheDepEdge $ left [2] CacheDepGuard $ $",
        );
        let before_count = space.btm.val_count();
        let before = Space::query_factor_plan_cache_key(&sources).unwrap();

        // Mutations that touch no queried prefix never change the key.
        let mut unrelated = String::new();
        for i in 0..256 {
            unrelated.push_str(&format!("(CacheDepNoise unrelated{i})\n"));
        }
        space.add_all_sexpr(unrelated.as_bytes()).unwrap();
        assert!(space.btm.val_count() >= before_count + 256);
        assert_eq!(
            Space::query_factor_plan_cache_key(&sources).unwrap(),
            before
        );

        // Even band-crossing growth of a queried relation (4 -> 8 facts) keeps
        // the key: replanning on data growth is no longer the key's job.
        space
            .add_all_sexpr(
                b"(CacheDepEdge fresh left)\n(CacheDepEdge g6 left)\n(CacheDepEdge g7 left)\n(CacheDepEdge g8 left)\n",
            )
            .unwrap();
        assert_eq!(
            Space::query_factor_plan_cache_key(&sources).unwrap(),
            before
        );
    }

    #[test]
    fn query_shape_side_index_reuses_summary_across_rank_calls() {
        let mut space = Space::new();
        let mut program = String::new();
        for i in 0..12 {
            program.push_str(&format!("(SideIdxReuse bucket{} item{i})\n", i % 3));
        }
        space.add_all_sexpr(program.as_bytes()).unwrap();

        let sources = query_sources(&mut space, "[2] , [3] SideIdxReuse $ $");
        let before = Space::query_shape_side_index_stats();

        let first = Space::query_factor_rank(
            &space.btm,
            sources[0],
            &mut BTreeMap::new(),
            &mut BTreeMap::new(),
        );
        let second = Space::query_factor_rank(
            &space.btm,
            sources[0],
            &mut BTreeMap::new(),
            &mut BTreeMap::new(),
        );

        let after = Space::query_shape_side_index_stats();
        assert_eq!(first.estimated_cardinality, 12);
        assert_eq!(first.min_variable_domain_cardinality, Some(3));
        assert!(first.shape_cardinality_scan);
        assert!(first.shape_side_index_insert);
        assert_eq!(second.estimated_cardinality, first.estimated_cardinality);
        assert_eq!(
            second.min_variable_domain_cardinality,
            first.min_variable_domain_cardinality
        );
        assert!(second.shape_side_index_hit);
        assert!(!second.shape_cardinality_scan);
        assert!(after.hits >= before.hits + 1);
        assert!(after.inserts >= before.inserts + 1);
        assert!(after.estimated_bytes >= before.estimated_bytes);
        assert!(after.key_bytes > 0);
        assert!(after.summary_bytes > 0);
        assert!(after.domain_values >= before.domain_values + 1);
        assert!(after.avoided_shape_scans >= before.avoided_shape_scans + 1);
    }

    #[test]
    fn query_factor_plan_metrics_bucket_prefix_cardinalities() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(MetricHot h1 value)
(MetricHot h2 value)
(MetricHot h3 value)
(MetricHot h4 value)
(MetricHot h5 value)
(MetricHot h6 value)
(MetricHot h7 value)
(MetricHot h8 value)
(MetricHot h9 value)
(MetricHot h10 value)
(MetricExact target)
"#,
            )
            .unwrap();

        let sources = query_sources(
            &mut space,
            "[5] , [3] MetricHot $ value [3] MetricHot $ value [2] MetricExact target [2] MetricCold $",
        );
        let before = Space::query_factor_plan_metrics_snapshot();

        let _plan = Space::query_factor_plan(&space.btm, &sources);

        let after = Space::query_factor_plan_metrics_snapshot();
        assert!(after.plans_ranked >= before.plans_ranked + 1);
        assert!(after.factors_ranked >= before.factors_ranked + 4);
        assert!(after.prefix_cardinality_lookups >= before.prefix_cardinality_lookups + 4);
        assert!(after.prefix_cardinality_cache_hits >= before.prefix_cardinality_cache_hits + 1);
        assert!(after.zero_cardinality_factors >= before.zero_cardinality_factors + 1);
        assert!(after.one_cardinality_factors >= before.one_cardinality_factors + 1);
        assert!(after.le64_cardinality_factors >= before.le64_cardinality_factors + 2);
        assert!(after.max_estimated_cardinality >= 10);
    }

    #[test]
    fn query_factor_plan_metrics_records_mode_signatures() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(ModeRel target hub)
(ModeNeedle target)
(ModePair target target)
"#,
            )
            .unwrap();

        let sources = query_sources(
            &mut space,
            "[5] , [3] ModeRel $ $ [2] ModeNeedle target [3] ModePair $ _1 $",
        );
        let mut prefix_cardinalities = BTreeMap::new();
        let mut shape_cardinalities = BTreeMap::new();

        let anchored = Space::query_factor_rank(
            &space.btm,
            sources[0],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert!(anchored.prefix_len > 0);
        assert_eq!(anchored.new_var_items, 2);
        assert_eq!(anchored.var_ref_items, 0);

        let ground = Space::query_factor_rank(
            &space.btm,
            sources[1],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert!(ground.prefix_len > 0);
        assert_eq!(ground.variable_items, 0);

        let repeated = Space::query_factor_rank(
            &space.btm,
            sources[2],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert!(repeated.prefix_len > 0);
        assert_eq!(repeated.new_var_items, 1);
        assert_eq!(repeated.var_ref_items, 1);

        let pure = Space::query_factor_rank(
            &space.btm,
            sources[3],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );
        assert_eq!(pure.prefix_len, 0);
        assert_eq!(pure.constant_items, 0);
        assert_eq!(pure.new_var_items, 1);

        let before = Space::query_factor_plan_metrics_snapshot();

        let _plan = Space::query_factor_plan(&space.btm, &sources);

        let after = Space::query_factor_plan_metrics_snapshot();
        assert!(after.plans_ranked >= before.plans_ranked + 1);
        assert!(after.ground_factors >= before.ground_factors + 1);
        assert!(after.anchored_variable_factors >= before.anchored_variable_factors + 2);
        assert!(after.unanchored_variable_factors >= before.unanchored_variable_factors + 1);
        assert!(after.repeated_variable_factors >= before.repeated_variable_factors + 1);
        assert!(after.pure_variable_factors >= before.pure_variable_factors + 1);
        assert!(after.new_var_items >= before.new_var_items + 4);
        assert!(after.var_ref_items >= before.var_ref_items + 1);
        assert!(after.variable_items_sum >= before.variable_items_sum + 5);
        assert!(after.max_variables_per_factor >= 2);
        assert!(after.max_prefix_len >= anchored.prefix_len);
    }

    #[test]
    fn query_factor_plan_metrics_records_ground_and_schematic_shape_roots() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(RootKind ground)
(RootKind $)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[2] , [2] RootKind $");
        let mut prefix_cardinalities = BTreeMap::new();
        let mut shape_cardinalities = BTreeMap::new();

        let rank = Space::query_factor_rank(
            &space.btm,
            sources[0],
            &mut prefix_cardinalities,
            &mut shape_cardinalities,
        );

        assert_eq!(rank.ground_root_matches, 1);
        assert_eq!(rank.schematic_root_matches, 1);

        let before = Space::query_factor_plan_metrics_snapshot();

        let _plan = Space::query_factor_plan(&space.btm, &sources);

        let after = Space::query_factor_plan_metrics_snapshot();
        assert!(after.shape_ground_root_matches >= before.shape_ground_root_matches + 1);
        assert!(after.shape_schematic_root_matches >= before.shape_schematic_root_matches + 1);
        assert!(after.schematic_shape_factors >= before.schematic_shape_factors + 1);
    }

    #[test]
    fn query_execution_storage_metrics_records_renormalized_factor_buffers() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(StorageRel target hub)
(StorageNeedle target)
"#,
            )
            .unwrap();

        let sources = query_sources(&mut space, "[3] , [3] StorageRel $ $ [2] StorageNeedle $");
        let before = Space::query_execution_storage_metrics_snapshot();

        let (buffers, _planned_sources) =
            Space::renormalize_query_factors(&sources, &[1, 0]).unwrap();

        let expected_len_sum = buffers.iter().map(Vec::len).sum::<usize>() as u128;
        let expected_capacity_sum = buffers.iter().map(Vec::capacity).sum::<usize>() as u128;
        let expected_max_len = buffers.iter().map(Vec::len).max().unwrap_or(0);
        let expected_max_capacity = buffers.iter().map(Vec::capacity).max().unwrap_or(0);
        let after = Space::query_execution_storage_metrics_snapshot();

        assert!(after.renormalized_plans >= before.renormalized_plans + 1);
        assert!(after.renormalized_factors >= before.renormalized_factors + 2);
        assert!(
            after.renormalized_factor_len_sum
                >= before.renormalized_factor_len_sum + expected_len_sum
        );
        assert!(
            after.renormalized_factor_capacity_sum
                >= before.renormalized_factor_capacity_sum + expected_capacity_sum
        );
        assert!(after.max_renormalized_factor_len >= expected_max_len);
        assert!(after.max_renormalized_factor_capacity >= expected_max_capacity);
    }

    #[test]
    fn query_execution_storage_metrics_records_candidate_pair_buffers() {
        let mut space = Space::new();
        let program = br#"
(StorageA target left)
(StorageA target side0)
(StorageDecoy other left)
(StorageGuard target)

(exec 0
  (, (StorageA $x left) (StorageA $x $side) (StorageGuard $x))
  (, (StorageResult $x $side)))
"#;

        space.add_all_sexpr(program).unwrap();
        let before = Space::query_execution_storage_metrics_snapshot();

        assert_eq!(space.metta_calculus(1), 1);

        let after = Space::query_execution_storage_metrics_snapshot();
        assert!(after.raw_searches >= before.raw_searches + 1);
        assert!(after.candidate_pair_vectors >= before.candidate_pair_vectors + 1);
        assert!(after.candidate_pair_entries_sum >= before.candidate_pair_entries_sum + 3);
        assert!(after.max_candidate_pair_entries >= 3);
        assert!(after.max_candidate_pair_capacity >= after.max_candidate_pair_entries);
        assert!(after.general_unifications >= before.general_unifications + 1);
        assert!(after.successful_unifications >= before.successful_unifications + 1);
        let general_delta = after.general_unifications - before.general_unifications;
        let success_delta = after.successful_unifications - before.successful_unifications;
        let failure_delta = after.unification_failures - before.unification_failures;
        assert_eq!(general_delta, success_delta + failure_delta);
    }

    #[test]
    fn take_first_exec_path_claims_execs_in_encoded_priority_order() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(exec 1 (, (Never high)) (, (Out high)))
(exec 0 (, (Never low)) (, (Out low)))
"#,
            )
            .unwrap();

        let mut claimed = Vec::new();

        assert!(space.take_first_exec_path(&mut claimed));
        let first = serialize(&claimed);
        assert!(first.contains("exec 0"), "{first}");
        assert!(space.btm.get(&claimed).is_none());

        assert!(space.take_first_exec_path(&mut claimed));
        let second = serialize(&claimed);
        assert!(second.contains("exec 1"), "{second}");
        assert!(space.btm.get(&claimed).is_none());

        assert!(!space.take_first_exec_path(&mut claimed));
        assert!(claimed.is_empty());
    }

    #[test]
    fn take_first_exec_path_does_not_claim_non_exec_values() {
        let mut space = Space::new();
        space.add_all_sexpr(b"(fact still-here)").unwrap();

        let mut claimed = b"stale".to_vec();

        assert!(!space.take_first_exec_path(&mut claimed));
        assert!(claimed.is_empty());
        assert_eq!(space.btm.val_count(), 1);
    }

    #[test]
    fn metta_calculus_consumes_exec_even_when_pattern_has_no_matches() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(exec 0 (, (Missing $x)) (, (Out $x)))
"#,
            )
            .unwrap();

        assert_eq!(space.metta_calculus(1), 1);
        assert_eq!(space.metta_calculus(1), 0);

        let mut output = Vec::new();
        space.dump_all_sexpr(&mut output).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(!output.contains("(exec "), "{output}");
        assert!(!output.contains("(Out "), "{output}");
    }

    #[test]
    fn query_factor_plan_cache_preserves_coreferential_query_results() {
        let mut space = Space::new();
        let program = br#"
(A target left)
(A target side0)
(A decoy left)
(Guard target)

(exec 0
  (, (A $x left) (A $x $side) (Guard $x))
  (, (R $x $side)))
"#;

        space.add_all_sexpr(program).unwrap();

        assert_eq!(space.metta_calculus(1), 1);

        let mut output = Vec::new();
        space.dump_sexpr(
            crate::expr!(space, "[3] R $ $"),
            crate::expr!(space, "[3] R _1 _2"),
            &mut output,
        );
        let output = String::from_utf8(output).unwrap();

        assert!(output.contains("(R target left)"), "{output}");
        assert!(output.contains("(R target side0)"), "{output}");
        assert_eq!(output.matches("(R ").count(), 2, "{output}");
        assert!(!output.contains("(R decoy left)"), "{output}");
    }

    #[test]
    fn prefix_subsumption_prefers_shortest_ancestor_prefix() {
        let prefixes: Vec<&[u8]> = vec![b"abcd", b"ab", b"abc", b"xy", b"abcd"];

        assert_eq!(Space::prefix_subsumption(&prefixes), vec![1, 1, 1, 3, 1]);
    }

    #[test]
    fn prefix_subsumption_keeps_earliest_duplicate_prefix() {
        let prefixes: Vec<&[u8]> = vec![b"foo", b"bar", b"foo"];

        assert_eq!(Space::prefix_subsumption(&prefixes), vec![0, 1, 0]);
    }

    #[test]
    fn prefix_subsumption_empty_prefix_subsumes_everything() {
        let prefixes: Vec<&[u8]> = vec![b"abc", b"", b"a", b""];

        assert_eq!(Space::prefix_subsumption(&prefixes), vec![1, 1, 1, 1]);
    }

    #[test]
    fn prefix_subsumption_resources_uses_byte_prefixes_for_btm_requests() {
        use crate::sinks::WriteResourceRequest::BTM;

        let requests = vec![
            BTM(b"abcd"),
            BTM(b"ab"),
            BTM(b"abc"),
            BTM(b"xy"),
            BTM(b"abcd"),
        ];

        assert_eq!(
            Space::prefix_subsumption_resources(&requests),
            vec![1, 1, 1, 3, 1]
        );
    }

    #[test]
    fn prefix_subsumption_resources_preserves_mixed_resource_fallback() {
        use crate::sinks::WriteResourceRequest::{ACT, BTM};

        let requests = vec![BTM(b"abc"), ACT("arena"), BTM(b"a"), ACT("arena")];

        assert_eq!(
            Space::prefix_subsumption_resources(&requests),
            vec![2, 1, 2, 1]
        );
    }

    #[test]
    fn prefix_subsumption_resources_indexes_mixed_resource_kinds_independently() {
        use crate::sinks::WriteResourceRequest::{ACT, BTM};

        let requests = vec![
            ACT("left"),
            BTM(b"out/item-0"),
            BTM(b"out"),
            ACT("right"),
            ACT("left"),
            BTM(b""),
            BTM(b"out/item-1"),
            ACT("right"),
        ];

        assert_eq!(
            Space::prefix_subsumption_resources(&requests),
            vec![0, 5, 5, 3, 0, 5, 5, 3]
        );
    }

    #[test]
    fn prefix_subsumption_resources_groups_nested_btm_requests_by_shortest_prefix() {
        use crate::sinks::WriteResourceRequest::BTM;

        let requests = vec![
            BTM(b"out/group-a/item-0"),
            BTM(b"out/group-a"),
            BTM(b"out/group-a/item-1"),
            BTM(b"out/group-b/item-0"),
            BTM(b"out/group-b"),
            BTM(b"out/group-b/item-1"),
            BTM(b"out/group-c"),
        ];

        assert_eq!(
            Space::prefix_subsumption_resources(&requests),
            vec![1, 1, 1, 4, 4, 4, 6]
        );
    }

    #[test]
    fn renormalize_query_factors_rejects_unencodable_variable_offsets() {
        let bytes = [item_byte(Tag::NewVar)];
        let source = ExprEnv {
            n: 0,
            v: 64,
            offset: 0,
            base: Expr {
                ptr: bytes.as_ptr().cast_mut(),
            },
        };

        assert!(Space::renormalize_query_factors(&[source], &[0]).is_none());
    }

    // ---- Phase 6a: the semi-naive IC soundness corpus + gate ----
    //
    // The semi-naive delta (feature `semi_naive_ic`) replaces the naive
    // full-rematch of every `,`->`,` rule in the IC loop with a per-rule delta
    // match. Datalog semi-naive evaluation is proven equal to naive only for
    // monotone (add-only) set-semantics programs; the known failure modes are
    // (1) non-monotonicity / retraction and (2) multiplicity. This corpus runs a
    // broad adversarial set of MM2 meta-rewrite programs both ways and asserts
    // the final dish is byte-identical, pinning down exactly where the equivalence
    // holds and where (if anywhere) it breaks. It is the correctness gate for ever
    // defaulting the feature.
    #[cfg(feature = "semi_naive_ic")]
    mod semi_naive_ic_corpus {
        use super::*;

        // Build a space by loading every top-level sexpr in `setup` as a literal
        // fact (identity transform), exactly like the existing oracle's `build`.
        fn build(setup: &str) -> Space {
            let mut s = Space::new();
            let pat = crate::expr!(s, "$");
            let tpl = crate::expr!(s, "_1");
            s.add_sexpr(setup.as_bytes(), pat, tpl).unwrap();
            s
        }

        // Naive reference: hand-drive the IC loop with the delta hook inert
        // (`sni_rule_seen` stays None, so every transform_multi_multi_ takes the
        // naive full-space branch). This is exactly the feature-off loop.
        fn run_naive(setup: &str) -> Space {
            let mut s = build(setup);
            let mut exec_path = Vec::new();
            while s.take_first_exec_path(&mut exec_path) {
                let xe = Expr { ptr: exec_path.as_mut_ptr() };
                let _ = s.interpret(xe);
                debug_assert!(s.sni_rule_seen.is_none());
            }
            s
        }

        // Semi-naive: the wired metta_calculus loop maintaining the per-rule delta.
        // A huge step budget drives it to the same fixpoint as the naive loop
        // (both stop when no exec remains).
        fn run_semi(setup: &str) -> Space {
            let mut s = build(setup);
            s.metta_calculus(1_000_000_000_000_000);
            s
        }

        // Semi-naive with a chosen retraction mode (Naive / Dred / RawSemiNoGate).
        fn run_semi_mode(setup: &str, mode: SniRetractMode) -> Space {
            let mut s = build(setup);
            s.sni_retract_mode = mode;
            s.metta_calculus(1_000_000_000_000_000);
            s
        }

        fn dump(s: &mut Space) -> String {
            let mut v = Vec::new();
            s.dump_all_sexpr(&mut v).unwrap();
            String::from_utf8_lossy(&v).into_owned()
        }

        // The set of dish facts present in `a` but not in `b` (line-wise on the
        // canonical dump), for divergence reporting.
        fn dish_minus(a: &str, b: &str) -> Vec<String> {
            let bl: std::collections::HashSet<&str> = b.lines().collect();
            a.lines().filter(|l| !bl.contains(l)).map(|l| l.to_string()).collect()
        }

        // Run `setup` both ways and return the two full-dish dumps. The dump is a
        // canonical sorted serialization of the whole space, so byte-equality of
        // the two strings is byte-equality of the two final spaces.
        fn both_ways(setup: &str) -> (String, String) {
            let mut naive = run_naive(setup);
            let mut semi = run_semi(setup);
            let mut nv = Vec::new();
            naive.dump_all_sexpr(&mut nv).unwrap();
            let mut sv = Vec::new();
            semi.dump_all_sexpr(&mut sv).unwrap();
            (
                String::from_utf8_lossy(&nv).into_owned(),
                String::from_utf8_lossy(&sv).into_owned(),
            )
        }

        // The first line where the two dumps differ, for divergence reporting.
        fn first_divergence(naive: &str, semi: &str) -> Option<String> {
            if naive == semi {
                return None;
            }
            let nlines: Vec<&str> = naive.lines().collect();
            let slines: Vec<&str> = semi.lines().collect();
            for i in 0..nlines.len().max(slines.len()) {
                let n = nlines.get(i).copied().unwrap_or("<none>");
                let s = slines.get(i).copied().unwrap_or("<none>");
                if n != s {
                    return Some(format!("line {i}: naive={n:?} semi={s:?}"));
                }
            }
            Some("(differ only in length)".to_string())
        }

        // ---- The adversarial corpus ----
        //
        // Each case is (name, setup). Every case is driven through the IC loop;
        // the `(exec N ...)` wrappers fire in priority order each round until
        // fixpoint. The default `(exec 0 ...)` / `(exec 1 ...)` numeric tags are
        // re-armed automatically by the loop the same way process_calculus's are.
        fn corpus() -> Vec<(&'static str, String)> {
            let mut cases: Vec<(&'static str, String)> = Vec::new();

            // -- 1. transitive closure: the canonical recursive `,`->`,` rule.
            // The recursive relation (path) appears once in the body alongside a
            // source relation (edge). This is the monotone case semi-naive targets.
            cases.push((
                "transitive_closure_chain",
                r#"
(exec 0 (, (edge $x $y) (path $y $z)) (, (path $x $z)))
(edge a b) (edge b c) (edge c d) (edge d e)
(path a b) (path b c) (path c d) (path d e)
"#
                .to_string(),
            ));

            // -- 2. recursive relation appears >2 times in one body (a 3-hop
            // closure step). Stresses the m-delta-rule's per-factor delta union.
            cases.push((
                "recursive_relation_thrice_in_body",
                r#"
(exec 0 (, (path $a $b) (path $b $c) (path $c $d)) (, (path $a $d)))
(path a b) (path b c) (path c d) (path d e) (path e f)
"#
                .to_string(),
            ));

            // -- 3. pure-source body: the template relation does NOT appear in the
            // body (recursive relation appears 0 times). A non-recursive `,`->`,`
            // rule; fires once then idles (no new matches feed it).
            cases.push((
                "pure_source_no_recursion",
                r#"
(exec 0 (, (a $x) (b $x)) (, (ab $x)))
(a 1) (a 2) (a 3) (b 2) (b 3) (b 4)
"#
                .to_string(),
            ));

            // -- 4. multi-relation body: factors over different relation heads,
            // recursive in one (reach), sourced from another (link).
            cases.push((
                "multi_relation_body",
                r#"
(exec 0 (, (link $x $y) (reach $y $z)) (, (reach $x $z)))
(link a b) (link b c) (link c d)
(reach a b) (reach b c) (reach c d)
"#
                .to_string(),
            ));

            // -- 5. coreference: repeated variable within one factor (r $x $x).
            // The self-loop edges are the only ones matching (r $x $x).
            cases.push((
                "coref_repeated_var_in_factor",
                r#"
(exec 0 (, (r $x $x) (mark $x)) (, (found $x)))
(r a a) (r b c) (r d d) (mark a) (mark d) (mark e)
"#
                .to_string(),
            ));

            // -- 6. coreference across factors plus a self-coref factor.
            cases.push((
                "coref_across_and_within",
                r#"
(exec 0 (, (p $x $x) (q $x $y) (p $y $y)) (, (pp $x $y)))
(p a a) (p b b) (p c c) (q a b) (q b c) (q a c)
"#
                .to_string(),
            ));

            // -- 7. issue-29 specificity ordering: a MORE-specific coreferential
            // factor (eq $x $x) appears BEFORE a less-specific coreferential use
            // (rel $x $y) that shares $x. The matcher must bind $x from the
            // specific factor first; the delta reorder must preserve that.
            cases.push((
                "issue29_specificity_ordering",
                r#"
(exec 0 (, (eq $x $x) (rel $x $y)) (, (out $x $y)))
(eq a a) (eq b b) (rel a p) (rel a q) (rel b r) (rel c s)
"#
                .to_string(),
            ));

            // -- 8. non-ground / schematic receiver: the body matches facts that
            // THEMSELVES carry variables (the petri reaction shape). The template
            // re-emits a variable-bearing fact.
            cases.push((
                "schematic_variable_bearing_facts",
                r#"
(exec 0 (, (rule (src $a) (dst $b)) (active $a)) (, (fire $a $b)))
(rule (src x) (dst (out $z))) (rule (src y) (dst (out $w)))
(active x) (active y)
"#
                .to_string(),
            ));

            // -- 9. single-factor body: the delta form's emit returns early on the
            // raw-matcher Ok branch; a single-factor `,`->`,` must still match.
            cases.push((
                "single_factor_body",
                r#"
(exec 0 (, (n $x)) (, (m $x)))
(n 1) (n 2) (n 3)
"#
                .to_string(),
            ));

            // -- 10. idempotent re-derivation: the rule re-derives facts that
            // already exist (the template output is already present). Must be a
            // no-op, fixpoint reached immediately.
            cases.push((
                "idempotent_rederive_existing",
                r#"
(exec 0 (, (e $x $y)) (, (e $x $y)))
(e a b) (e b c)
"#
                .to_string(),
            ));

            // -- 11. fixpoint reached early then idled: a short chain whose closure
            // completes in 2 rounds, but the exec keeps re-arming and finding
            // nothing new for many rounds (the redundancy the lever targets).
            cases.push((
                "fixpoint_early_then_idle",
                r#"
(exec 0 (, (edge $x $y) (tc $y $z)) (, (tc $x $z)))
(edge a b) (edge b c)
(tc a b) (tc b c)
"#
                .to_string(),
            ));

            // -- 12. cross-rule deltas: two recursive `,`->`,` rules feeding each
            // other. Rule 0 derives `even` from `odd`, rule 1 derives `odd` from
            // `even`; they fire out of lockstep across IC steps, so each rule's
            // per-rule frontier must be independent.
            cases.push((
                "cross_rule_two_recursive",
                r#"
(exec 0 (, (succ $x $y) (even $x)) (, (odd $y)))
(exec 1 (, (succ $x $y) (odd $x)) (, (even $y)))
(succ 0 1) (succ 1 2) (succ 2 3) (succ 3 4) (succ 4 5)
(even 0)
"#
                .to_string(),
            ));

            // -- 13. three mutually-feeding rules over a shared relation.
            cases.push((
                "cross_rule_three_feeding",
                r#"
(exec 0 (, (gen $x) (next $x $y)) (, (gen $y)))
(exec 1 (, (gen $x) (tag $x)) (, (seen $x)))
(exec 2 (, (seen $x) (next $x $y)) (, (tag $y)))
(gen a) (tag a) (next a b) (next b c) (next c d)
"#
                .to_string(),
            ));

            // ---- The suspected weak spot: RETRACTION (non-monotone) ----
            // The semi-naive path is hooked ONLY into transform_multi_multi_ (the
            // `,`->`,` add-only case). The `O`/`-` removing template runs through
            // transform_multi_multi_o, always naive. These cases mix a removing
            // rule with a recursive `,`->`,` rule, to probe whether a retraction
            // by the naive rule invalidates the semi-naive rule's snapshot.

            // -- 14. a removing rule alone (sanity: pure naive path, must match).
            cases.push((
                "remove_rule_alone",
                r#"
(exec 0 (, (junk $x)) (O (- (junk $x))))
(junk a) (junk b) (junk c) (keep d)
"#
                .to_string(),
            ));

            // -- 15. removal feeds a recursive add-rule: rule 0 (O/-) removes a
            // `gate`, rule 1 (`,`->`,`) closes `reach` over `link`. The removal
            // changes the dish under the recursive rule's feet between its firings.
            cases.push((
                "remove_then_recursive_add",
                r#"
(exec 0 (, (gate $x)) (O (- (gate $x)) (+ (open $x))))
(exec 1 (, (link $x $y) (reach $y $z)) (, (reach $x $z)))
(gate g1) (gate g2)
(link a b) (link b c) (link c d)
(reach a b) (reach b c) (reach c d)
"#
                .to_string(),
            ));

            // -- 16. the recursive relation ITSELF is retracted by an O/- rule
            // while the `,`->`,` rule is closing over it. This is the hard
            // non-monotone case: a `reach` fact the closure depends on is removed,
            // so the naive rule (matching the post-removal dish) and the semi-naive
            // rule (whose snapshot still records the removed fact) could diverge.
            cases.push((
                "retract_recursive_relation_midclosure",
                r#"
(exec 0 (, (kill $x $y)) (O (- (reach $x $y))))
(exec 1 (, (edge $x $y) (reach $y $z)) (, (reach $x $z)))
(kill b c)
(edge a b) (edge b c) (edge c d)
(reach a b) (reach b c) (reach c d)
"#
                .to_string(),
            ));

            // -- 17. retract a SOURCE fact the recursive rule reads. Rule 0 (O/-)
            // removes an `edge` (a source factor of the closure), rule 1 closes
            // `path` over `edge`. The edge is gone before the closure finishes, so
            // the post-removal naive match and the snapshot-bearing semi-naive
            // match could disagree on which path facts are derivable. (MM2
            // consumption is via the O/- removing template, not the I functor,
            // which is a builtin/comparison source.)
            cases.push((
                "retract_source_edge_midclosure",
                r#"
(exec 0 (, (cut $x $y)) (O (- (edge $x $y))))
(exec 1 (, (edge $x $y) (path $y $z)) (, (path $x $z)))
(cut b c)
(edge a b) (edge b c) (edge c d)
(path a b) (path b c) (path c d)
"#
                .to_string(),
            ));

            // ---- empty / degenerate shapes ----

            // -- 18. no-match step: the body references a relation with no facts,
            // so the rule fires but matches nothing every round.
            cases.push((
                "no_match_empty_relation",
                r#"
(exec 0 (, (ghost $x) (real $x)) (, (out $x)))
(real 1) (real 2)
"#
                .to_string(),
            ));

            // -- 19. a program with only facts, no exec: the loop does nothing,
            // both dumps are the loaded facts.
            cases.push((
                "facts_only_no_exec",
                r#"
(alpha 1) (beta 2) (gamma 3)
"#
                .to_string(),
            ));

            // -- 20. self-matching exec (the IC-driver shape): a `,`->`,` rule
            // whose body matches the exec wrapper itself, re-arming each round.
            cases.push((
                "self_rearming_counter",
                r#"
(exec (C 0)
      (, (exec (C $n) $p $t) (count $n $m))
      (, (exec (C $m) $p $t) (tick $n)))
(count 0 1) (count 1 2) (count 2 3) (count 3 stop)
"#
                .to_string(),
            ));

            // -- 21. the actual process_calculus petri reaction at small size
            // (add(3,3)), the canonical lever target, as a corpus member.
            cases.push((
                "process_calculus_add_3_3",
                process_calculus_setup_sexpr(50, 3, 3),
            ));

            // ---- re-arming (multi-firing) shapes ----
            //
            // A bare `(exec N ...)` fires once then is consumed. The cases above
            // therefore exercise each rule's FIRST firing only, where the delta is
            // trivially the whole space (no snapshot yet) so semi-naive == naive by
            // construction. To stress the per-rule snapshot ACROSS firings (the
            // only place the delta restriction can diverge), a rule must re-arm.
            // These use the process_calculus IC-driver shape: a driver exec that
            // each round re-emits itself with a decremented Peano counter and re-
            // arms a worker exec from a stored `(W..)` rule fact. The driver is the
            // multi-firing `,`->`,` rule that goes through the semi-naive path many
            // times; the worker closes a relation. See `rearming_program`.

            // -- 22. re-arming transitive closure: the worker fires every round and
            // computes the full path closure (3-hop and 4-hop facts appear, which
            // need >=3 firings). This is the multi-firing monotone baseline.
            cases.push((
                "rearming_transitive_closure",
                rearming_program(
                    6,
                    "(, (edge $x $y) (path $y $z)) (, (path $x $z))",
                    "(edge a b) (edge b c) (edge c d) (edge d e)\n\
                     (path a b) (path b c) (path c d) (path d e)",
                    "",
                ),
            ));

            // -- 23. re-arming closure with a SOURCE edge retracted mid-loop. A
            // separate O/- rule removes (edge b c) when a `(trigger)` fires, which
            // the driver schedules a few rounds in. The worker's snapshot (taken
            // before the removal) still records the edge; naive (post-removal) does
            // not. This is the decisive non-monotone adversary: the recursive rule
            // fires repeatedly AND a fact it reads is retracted between firings.
            cases.push((
                "rearming_retract_source_edge",
                rearming_program(
                    8,
                    "(, (edge $x $y) (path $y $z)) (, (path $x $z))",
                    "(edge a b) (edge b c) (edge c d) (edge d e)\n\
                     (path a b) (path b c) (path c d) (path d e)\n\
                     (armed yes)",
                    "(exec rm (, (armed yes)) (O (- (edge b c)) (- (armed yes))))",
                ),
            ));

            // -- 24. re-arming closure where the RECURSIVE relation itself (a path
            // fact) is retracted mid-loop. Even sharper: removing (path c d) pulls
            // a derived fact the closure chains through.
            cases.push((
                "rearming_retract_recursive_path",
                rearming_program(
                    8,
                    "(, (edge $x $y) (path $y $z)) (, (path $x $z))",
                    "(edge a b) (edge b c) (edge c d) (edge d e)\n\
                     (path a b) (path b c) (path c d) (path d e)\n\
                     (armed yes)",
                    "(exec rm (, (armed yes)) (O (- (path c d)) (- (armed yes))))",
                ),
            ));

            // -- 25. re-arming with an idempotent worker that re-derives only
            // existing facts every round (snapshot grows but no new facts). Probes
            // the empty-delta path across many firings.
            cases.push((
                "rearming_idempotent_worker",
                rearming_program(
                    5,
                    "(, (e $x $y)) (, (e $x $y))",
                    "(e a b) (e b c) (e c d)",
                    "",
                ),
            ));

            // -- 26. re-arming worker with the recursive relation appearing 3x in
            // the body, firing repeatedly (multi-factor delta union across firings).
            cases.push((
                "rearming_three_factor_recursion",
                rearming_program(
                    7,
                    "(, (tc $a $b) (tc $b $c) (tc $c $d)) (, (tc $a $d))",
                    "(tc a b) (tc b c) (tc c d) (tc d e) (tc e f)",
                    "",
                ),
            ));

            // -- 27. re-arming closure with a SOURCE edge ADDED mid-loop (monotone
            // but late): a new edge appears after the closure has partly formed, so
            // the worker must fold it in on a later firing (delta correctness for a
            // fact added after the first firing, the normal semi-naive case but
            // under retraction-free re-arming).
            cases.push((
                "rearming_add_source_edge_late",
                rearming_program(
                    8,
                    "(, (edge $x $y) (path $y $z)) (, (path $x $z))",
                    "(edge a b) (edge b c) (edge c d)\n\
                     (path a b) (path b c) (path c d)\n\
                     (armed yes)",
                    "(exec ad (, (armed yes)) (O (+ (edge c d2)) (+ (path c d2)) (- (armed yes))))",
                ),
            ));

            // ---- remove-then-re-add: the classic semi-naive non-monotone hazard ----
            //
            // The snapshot is `read_copy` at the worker's last firing; the delta is
            // `read_copy_now \ snapshot`. A fact present in the snapshot, REMOVED,
            // then RE-ADDED is in BOTH snapshot and read_copy_now, so `subtract`
            // excludes it from the delta -- semi-naive treats it as "old, already
            // processed". If a NEW derivation depends on that fact's re-presence
            // combined with another fact, semi-naive can MISS it where naive (which
            // re-scans the whole current dish) finds it. These cases drive exactly
            // that window using counter-gated O/- and O/+ rules that fire at
            // distinct rounds (the driver's Peano counter is the clock).

            // -- 28. remove-then-re-add an `edge` the closure reads. The gate worker
            // removes (edge q r)+(path q r) at counter (S(S(S(S(S(S(S Z))))))) and
            // re-adds them plus a fresh (edge r s)+(path r s) two rounds later. The
            // closure worker's snapshot recorded edge q r before removal, so on the
            // re-add round `read_copy \ snapshot` excludes the re-added edge: the
            // delta restriction could MISS the r->s extension that chains to p only
            // through the re-added q->r. (The gate fires off the live `(exec (D $c))`
            // driver exec, so it lands between closure firings.)
            cases.push((
                "rearming_remove_then_readd_edge",
                rearming_gated_program(
                    12,
                    "(, (edge $x $y) (path $y $z)) (, (path $x $z))",
                    // gate: counter==7 remove edge q r; counter==5 re-add it + r->s.
                    "(, (exec (D $c) $a $b) (gate $c rm)) \
                     (O (- (edge q r)) (- (path q r)) (- (gate $c rm)))",
                    "(edge p q) (edge q r)\n\
                     (path p q) (path q r)\n\
                     (gate (S (S (S (S (S (S (S Z)))))))  rm)",
                ),
            ));

            // -- 29. a second gated remove-then-re-add over a distinct relation,
            // with the re-add carrying a NEW joinable fact, so a divergence would
            // show as a missing `conn` derivation. Stresses the same snapshot-
            // staleness window on a fresh relation/order.
            cases.push((
                "rearming_toggle_fact_dependent_derive",
                rearming_gated_program(
                    12,
                    "(, (link $x $y) (conn $y $z)) (, (conn $x $z))",
                    "(, (exec (D $c) $a $b) (gate $c go)) \
                     (O (- (conn n o)) (+ (link x m)) (+ (conn m n)) (- (gate $c go)))",
                    "(link m n) (conn n o)\n\
                     (conn m n)\n\
                     (gate (S (S (S (S (S (S Z)))))) go)",
                ),
            ));

            cases
        }

        // Build a process_calculus-style re-arming program: a driver exec that
        // each round re-arms a stored worker rule `(W..)` and decrements a Peano
        // counter, plus `extra` standalone rules (e.g. an interleaved O/- remover).
        // `worker` is "<body> <template>" (the `,`->`,` rule the worker fires).
        // This is the only shape that fires the worker rule MORE THAN ONCE, so it
        // is the only one that stresses the per-rule snapshot across firings.
        fn rearming_program(steps: usize, worker: &str, facts: &str, extra: &str) -> String {
            format!(
                r#"
(exec (D {counter})
      (, (exec (D (S $c)) $sp $st) ((W) $p $t))
      (, (exec (D $c) $sp $st) (exec (R) $p $t)))
((W) {worker})
{extra}
{facts}
"#,
                counter = peano_sexpr(steps),
            )
        }

        // A re-arming driver that re-arms TWO stored workers each round: the
        // closure worker `(W)` (a `,`->`,` rule on the semi-naive path) and a gate
        // worker `(G)` (an O/-/+ rule, always naive) that self-schedules off the
        // driver's Peano counter. Each round the driver re-emits itself decremented
        // and re-arms BOTH `(exec (R) ...)` (closure) and `(exec (GR) ...)` (gate).
        // The gate worker's body reads the live `(exec (D $c) ...)` driver exec to
        // see the current counter, so its O/-/+ effects fire at a chosen round. This
        // is what makes remove-then-re-add land BETWEEN closure firings.
        fn rearming_gated_program(
            steps: usize,
            worker: &str,
            gate: &str,
            facts: &str,
        ) -> String {
            format!(
                r#"
(exec (D {counter})
      (, (exec (D (S $c)) $sp $st) ((W) $p $t) ((G) $gp $gt))
      (, (exec (D $c) $sp $st) (exec (R) $p $t) (exec (GR) $gp $gt)))
((W) {worker})
((G) {gate})
{facts}
"#,
                counter = peano_sexpr(steps),
            )
        }

        // A tiny deterministic PRNG (splitmix64) so the random corpus is seeded and
        // reproducible without pulling in `rand`.
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^ (z >> 31)
            }
            fn below(&mut self, n: usize) -> usize {
                (self.next() % n as u64) as usize
            }
        }

        // Generate a seeded random small program: a re-arming transitive-closure
        // worker over a random sparse `edge` graph (seeded `path` = the edges),
        // optionally with a gate worker that removes and/or adds a random edge
        // mid-loop. Re-arming (the driver) makes the worker fire many rounds, so the
        // per-rule snapshot is exercised across firings under the random mutation.
        fn random_program(seed: u64) -> String {
            let mut rng = Rng(seed);
            let nodes = 4 + rng.below(4); // 4..7 nodes
            let n_edges = 3 + rng.below(6); // 3..8 edges
            let mut facts = String::new();
            let mut edges: Vec<(usize, usize)> = Vec::new();
            for _ in 0..n_edges {
                let a = rng.below(nodes);
                let b = rng.below(nodes);
                if a == b {
                    continue; // skip self-loops (keep the closure finite-ish)
                }
                if edges.contains(&(a, b)) {
                    continue;
                }
                edges.push((a, b));
                facts.push_str(&format!("(edge n{a} n{b}) (path n{a} n{b})\n"));
            }
            // Optional gate: with prob ~1/2, remove a random existing edge; with
            // prob ~1/2, add a fresh random edge. Both fire once, mid-loop, off the
            // live driver counter at a random round.
            let mut gate_effects = String::new();
            if rng.below(2) == 0 && !edges.is_empty() {
                let (a, b) = edges[rng.below(edges.len())];
                gate_effects.push_str(&format!("(- (edge n{a} n{b})) (- (path n{a} n{b})) "));
            }
            if rng.below(2) == 0 {
                let a = rng.below(nodes);
                let b = rng.below(nodes);
                if a != b {
                    gate_effects.push_str(&format!("(+ (edge n{a} n{b})) (+ (path n{a} n{b})) "));
                }
            }
            let steps = 5 + rng.below(8); // 5..12 rounds
            if gate_effects.is_empty() {
                rearming_program(
                    steps,
                    "(, (edge $x $y) (path $y $z)) (, (path $x $z))",
                    &facts,
                    "",
                )
            } else {
                let gate = format!(
                    "(, (exec (D $c) $a $b) (gate $c go)) (O {gate_effects}(- (gate $c go)))"
                );
                let mut facts_with_gate = facts;
                // Fire the gate around the middle of the run.
                let fire_at = 1 + rng.below(steps.saturating_sub(1).max(1));
                facts_with_gate.push_str(&format!("(gate {} go)\n", peano_sexpr(fire_at)));
                rearming_gated_program(
                    steps,
                    "(, (edge $x $y) (path $y $z)) (, (path $x $z))",
                    &gate,
                    &facts_with_gate,
                )
            }
        }

        // Run a large batch of seeded random programs both ways; assert every one
        // is byte-identical. This sweeps interleavings of re-arming recursion with
        // random add/remove mutation that the hand-written cases cannot enumerate.
        #[test]
        fn random_corpus_byte_identical() {
            let mut diverged = Vec::new();
            for seed in 0..200u64 {
                let setup = random_program(seed.wrapping_mul(0x100000001b3).wrapping_add(1));
                let (naive, semi) = both_ways(&setup);
                if let Some(d) = first_divergence(&naive, &semi) {
                    eprintln!("[DIVERGES] seed {seed}: {d}\nSETUP:\n{setup}");
                    diverged.push(seed);
                }
            }
            assert!(
                diverged.is_empty(),
                "seeded random programs must be byte-identical naive vs semi-naive; diverged seeds: {diverged:?}"
            );
        }

        // Generate a seeded random program that ALWAYS retracts: like
        // `random_program`, but the gate effects are forced to include at least one
        // `O`/`-` removal of an existing edge, so every program exercises the
        // retraction path (and therefore DRed). Re-arming makes the worker fire many
        // rounds, so the per-rule snapshot is stressed across firings under the
        // mutation -- exactly where the semi-naive delta can diverge.
        fn random_retracting_program(seed: u64) -> String {
            let mut rng = Rng(seed);
            let nodes = 4 + rng.below(4); // 4..7 nodes
            let n_edges = 4 + rng.below(6); // 4..9 edges (>=4 so a removal leaves a graph)
            let mut edges: Vec<(usize, usize)> = Vec::new();
            let mut facts = String::new();
            for _ in 0..n_edges {
                let a = rng.below(nodes);
                let b = rng.below(nodes);
                if a == b || edges.contains(&(a, b)) {
                    continue;
                }
                edges.push((a, b));
                facts.push_str(&format!("(edge n{a} n{b}) (path n{a} n{b})\n"));
            }
            // Force a removal of an existing edge+path (the retraction under test).
            let mut gate_effects = String::new();
            if !edges.is_empty() {
                let (a, b) = edges[rng.below(edges.len())];
                gate_effects.push_str(&format!("(- (edge n{a} n{b})) (- (path n{a} n{b})) "));
            }
            // Optionally also add a fresh edge mid-loop (mixed add+remove batch).
            if rng.below(2) == 0 {
                let a = rng.below(nodes);
                let b = rng.below(nodes);
                if a != b {
                    gate_effects.push_str(&format!("(+ (edge n{a} n{b})) (+ (path n{a} n{b})) "));
                }
            }
            let steps = 6 + rng.below(8); // 6..13 rounds (enough re-arming firings)
            let gate =
                format!("(, (exec (D $c) $a $b) (gate $c go)) (O {gate_effects}(- (gate $c go)))");
            let mut facts_with_gate = facts;
            let fire_at = 1 + rng.below(steps.saturating_sub(1).max(1));
            facts_with_gate.push_str(&format!("(gate {} go)\n", peano_sexpr(fire_at)));
            rearming_gated_program(
                steps,
                "(, (edge $x $y) (path $y $z)) (, (path $x $z))",
                &gate,
                &facts_with_gate,
            )
        }

        // The hard DRed random assertion: >=200 seeded random RETRACTING programs,
        // each with an `O`/`-` removal mid-loop, must be byte-identical naive vs
        // DRed. Also asserts DRed actually ENGAGED (repairs > 0 on a non-trivial
        // fraction), so the test is not vacuously passing through the naive fallback.
        #[test]
        fn random_retracting_corpus_byte_identical_dred() {
            let mut diverged = Vec::new();
            let mut engaged = 0usize;
            let mut fellback = 0usize;
            let total = 240u64;
            for seed in 0..total {
                let setup =
                    random_retracting_program(seed.wrapping_mul(0x100000001b3).wrapping_add(7));
                let mut naive = run_naive(&setup);
                let mut dred = run_semi_mode(&setup, SniRetractMode::Dred);
                if dred.sni_dred_repairs > 0 {
                    engaged += 1;
                }
                if dred.sni_dred_fallbacks > 0 {
                    fellback += 1;
                }
                let nd = dump(&mut naive);
                let dd = dump(&mut dred);
                if nd != dd {
                    eprintln!(
                        "[DIVERGES] seed {seed}: naive HAS {:?} ; dred HAS {:?}\nSETUP:\n{setup}",
                        dish_minus(&nd, &dd),
                        dish_minus(&dd, &nd),
                    );
                    diverged.push(seed);
                }
            }
            eprintln!(
                "DRed random retracting corpus: {total} programs, {engaged} engaged DRed (repairs>0), {fellback} hit a fallback"
            );
            assert!(
                diverged.is_empty(),
                "seeded random RETRACTING programs must be byte-identical naive vs DRed; diverged seeds: {diverged:?}"
            );
            assert!(
                engaged >= total as usize / 4,
                "DRed must actually engage (repairs>0) on a non-trivial fraction; only {engaged}/{total} did"
            );
        }

        // Soft probe: run the whole corpus, report byte-identical vs divergent
        // (with the first divergent fact) WITHOUT asserting. This is the
        // measurement that maps where the equivalence holds, run before designing
        // the gate. Always "passes"; its value is the eprintln map.
        #[test]
        fn corpus_divergence_map() {
            let mut diverged = Vec::new();
            for (name, setup) in corpus() {
                let (naive, semi) = both_ways(&setup);
                match first_divergence(&naive, &semi) {
                    None => eprintln!("[byte-identical] {name}"),
                    Some(d) => {
                        eprintln!("[DIVERGES]       {name}: {d}");
                        diverged.push(name);
                    }
                }
            }
            eprintln!("\n=== divergent cases: {diverged:?} ===");
        }

        // The hard corpus assertion: every hand-written case must be byte-identical
        // naive vs semi-naive WITH the soundness gate. The retraction cases pass
        // because the gate routes them to naive once a removal happens.
        #[test]
        fn corpus_all_byte_identical() {
            for (name, setup) in corpus() {
                let (naive, semi) = both_ways(&setup);
                if let Some(d) = first_divergence(&naive, &semi) {
                    panic!("case {name} diverges WITH the gate: {d}");
                }
            }
        }

        // An adversarial mix: an `I`-source (builtin `==`) rule that RETRACTS via
        // `O`/`-` AND a recursive `,`->`,` closure rule, both in one program. The
        // `I`-source rule never enters the semi-naive delta path (the gate lives only
        // in `transform_multi_multi_`), so it is always naive and re-derives on its
        // own; the catch-up only repairs the `,`->`,` rule. This proves DRed stays
        // byte-identical when a retraction is driven by a NON-`,`->`,` (builtin/`I`)
        // rule -- the conservative boundary the research flags.
        #[test]
        fn dred_byte_identical_with_i_source_retraction() {
            let setup = r#"
(exec 0 (I (== (key $p) $o)) (O (- (edge $o $o)) (+ (removed $o))))
(exec 1 (, (edge $x $y) (path $y $z)) (, (path $x $z)))
(key k1) 
(edge a b) (edge b c) (edge c d) (edge a a)
(path a b) (path b c) (path c d) (path a a)
"#;
            let mut naive = run_naive(setup);
            let mut dred = run_semi_mode(setup, SniRetractMode::Dred);
            let nd = dump(&mut naive);
            let dd = dump(&mut dred);
            assert_eq!(
                nd, dd,
                "DRed must be byte-identical to naive with an I-source retraction; \
                 naive HAS {:?} ; dred HAS {:?}",
                dish_minus(&nd, &dd),
                dish_minus(&dd, &nd),
            );
        }

        // CONSERVATIVE SAFETY: the DRed `Fallback` arm (taken when a shape cannot be
        // proven byte-identical) must itself produce naive's dish. Force the fallback
        // on every retraction repro and assert the whole retraction corpus is STILL
        // byte-identical to naive -- proving the safety net is correct, so routing an
        // un-handled shape to it never risks a wrong dish.
        #[test]
        fn dred_fallback_is_byte_identical_to_naive() {
            for (name, setup) in corpus() {
                let mut naive = run_naive(&setup);
                let mut forced = build(&setup);
                forced.sni_retract_mode = SniRetractMode::Dred;
                forced.sni_dred_force_fallback = true;
                forced.metta_calculus(1_000_000_000_000_000);
                let nd = dump(&mut naive);
                let fd = dump(&mut forced);
                assert_eq!(
                    nd, fd,
                    "case {name}: the DRed conservative fallback must reach naive's dish; \
                     naive HAS {:?} ; fallback HAS {:?}",
                    dish_minus(&nd, &fd),
                    dish_minus(&fd, &nd),
                );
                // Where a retraction happened, the forced fallback must have recorded
                // at least one fallback (proving the arm is actually reachable).
                if forced.sni_removal_seen {
                    assert!(
                        forced.sni_dred_fallbacks >= 1 || forced.sni_dred_repairs == 0,
                        "case {name}: forced fallback should record a fallback on a retraction"
                    );
                }
            }
        }

        // The hard DRed assertion: every hand-written case must be byte-identical
        // naive vs DRed (`SniRetractMode::Dred`). The retraction cases now take the
        // re-derivation catch-up (instead of falling to naive), so this proves DRed
        // reproduces naive's dish on EVERY shape including all the retraction cases.
        #[test]
        fn corpus_all_byte_identical_dred() {
            for (name, setup) in corpus() {
                let mut naive = run_naive(&setup);
                let mut dred = run_semi_mode(&setup, SniRetractMode::Dred);
                let nd = dump(&mut naive);
                let dd = dump(&mut dred);
                if nd != dd {
                    panic!(
                        "case {name} diverges under Dred: naive HAS {:?} ; dred HAS {:?}",
                        dish_minus(&nd, &dd),
                        dish_minus(&dd, &nd),
                    );
                }
            }
        }

        // MUTATION TEST: the DRed re-derivation is LOAD-BEARING. Run the minimal
        // retraction repro under Dred with the re-derivation SKIPPED
        // (`sni_dred_skip_rederive`): the dish MUST diverge from naive (it loses the
        // re-derived (path n1 n3), the exact bug). Then WITHOUT the skip it MUST be
        // byte-identical. If the re-derivation were dead code, the skipped run would
        // still match naive and this test would (correctly) fail.
        #[test]
        fn dred_rederive_is_load_bearing_mutation() {
            let minimal = r#"
(exec (D (S (S (S (S (S (S Z)))))))
      (, (exec (D (S $c)) $sp $st) ((W) $p $t) ((G) $gp $gt))
      (, (exec (D $c) $sp $st) (exec (R) $p $t) (exec (GR) $gp $gt)))
((W) (, (edge $x $y) (path $y $z)) (, (path $x $z)))
((G) (, (exec (D $c) $a $b) (gate $c go)) (O (- (edge n1 n3)) (- (path n1 n3)) (- (gate $c go))))
(edge n1 n0) (path n1 n0)
(edge n0 n1) (path n0 n1)
(edge n1 n3) (path n1 n3)
(edge n3 n0) (path n3 n0)
(gate (S Z) go)
"#;
            let mut naive = run_naive(minimal);
            let nd = dump(&mut naive);

            // MUTANT: skip the re-derivation -> must DIVERGE (loses (path n1 n3)).
            let mut mutant = build(minimal);
            mutant.sni_retract_mode = SniRetractMode::Dred;
            mutant.sni_dred_skip_rederive = true;
            mutant.metta_calculus(1_000_000_000_000_000);
            let md = dump(&mut mutant);
            assert_ne!(
                nd, md,
                "mutation (skip re-derivation) MUST diverge from naive; if it matches, \
                 the re-derivation is dead code"
            );
            assert!(
                dish_minus(&nd, &md).iter().any(|l| l.contains("(path n1 n3)")),
                "the skipped-re-derivation mutant must specifically lose (path n1 n3)"
            );

            // RESTORED: the real DRed (re-derivation ON) must be byte-identical.
            let mut dred = run_semi_mode(minimal, SniRetractMode::Dred);
            let dd = dump(&mut dred);
            assert_eq!(
                nd, dd,
                "with the re-derivation restored, DRed must be byte-identical to naive"
            );
        }

        // DRed actually ENGAGES on the retraction cases (not silently falling back
        // to naive for all of them): at least the minimal-repro-style re-arming
        // retraction cases must record DRed repairs > 0. Documents which retraction
        // shapes DRed handles incrementally vs routes to naive (fallbacks).
        #[test]
        fn dred_engages_on_retraction_cases() {
            let mut any_repaired = false;
            for (name, setup) in corpus() {
                let mut dred = run_semi_mode(&setup, SniRetractMode::Dred);
                let repairs = dred.sni_dred_repairs;
                let fallbacks = dred.sni_dred_fallbacks;
                let removal = dred.sni_removal_seen;
                if removal {
                    eprintln!("[Dred] {name}: repairs={repairs} fallbacks={fallbacks}");
                    if repairs > 0 {
                        any_repaired = true;
                    }
                }
            }
            assert!(
                any_repaired,
                "DRed must run >=1 re-derivation repair across the retraction corpus"
            );
        }

        // Without the gate, a retraction read by a recursive worker diverges; this
        // documents the exact boundary the gate guards. The reduced seed-198
        // program (a 2-cycle n0<->n1 plus n1->n3->n0, gate removes edge/path n1->n3
        // at round 1) loses the re-derivation of (path n1 n3) under semi-naive: the
        // worker's snapshot recorded n1->n3 before removal, so `btm \ snapshot`
        // excludes it and the closure never re-derives it through n1->n0->...->n3,
        // which naive (full re-scan) does. WITH the gate the removal latches
        // `sni_removal_seen` and every later worker runs naive, so it is byte-
        // identical. This is the load-bearing proof that the gate closes the hole.
        #[test]
        fn gate_closes_the_retraction_divergence() {
            let setup = r#"
(exec (D (S (S (S (S (S (S Z)))))))
      (, (exec (D (S $c)) $sp $st) ((W) $p $t) ((G) $gp $gt))
      (, (exec (D $c) $sp $st) (exec (R) $p $t) (exec (GR) $gp $gt)))
((W) (, (edge $x $y) (path $y $z)) (, (path $x $z)))
((G) (, (exec (D $c) $a $b) (gate $c go)) (O (- (edge n1 n3)) (- (path n1 n3)) (- (gate $c go))))
(edge n1 n0) (path n1 n0)
(edge n0 n1) (path n0 n1)
(edge n1 n3) (path n1 n3)
(edge n3 n0) (path n3 n0)
(gate (S Z) go)
"#;
            // The gate makes it byte-identical.
            let (naive, semi) = both_ways(setup);
            assert_eq!(naive, semi, "the gate must close the retraction divergence");
            // And it really does close a HOLE: naive re-derives (path n1 n3).
            assert!(
                naive.contains("(path n1 n3)"),
                "naive must re-derive the removed (path n1 n3) through the cycle"
            );
            // Sanity: the gate actually tripped (a removal happened in the loop).
            let mut s = run_semi(setup);
            assert!(
                s.sni_removal_seen,
                "the gate flag must latch after the O/- rule retracts"
            );
            // dump to settle `s` use without warnings.
            let mut v = Vec::new();
            s.dump_all_sexpr(&mut v).unwrap();
        }

        // DIAGNOSTIC (always passes; its value is the eprintln ground truth): for
        // every corpus case that RETRACTS, print the exact facts where the un-gated
        // raw semi-naive (`RawSemiNoGate`) diverges from naive. This is the
        // "test, don't predict" map of what DRed must repair: the facts naive HAS
        // that raw-semi MISSES (DRed must re-derive these) and the facts raw-semi
        // HAS that naive lacks (DRed must over-delete these). Run with --nocapture.
        #[test]
        fn dred_divergence_ground_truth() {
            // The documented minimal repro (the gate_closes program) first.
            let minimal = r#"
(exec (D (S (S (S (S (S (S Z)))))))
      (, (exec (D (S $c)) $sp $st) ((W) $p $t) ((G) $gp $gt))
      (, (exec (D $c) $sp $st) (exec (R) $p $t) (exec (GR) $gp $gt)))
((W) (, (edge $x $y) (path $y $z)) (, (path $x $z)))
((G) (, (exec (D $c) $a $b) (gate $c go)) (O (- (edge n1 n3)) (- (path n1 n3)) (- (gate $c go))))
(edge n1 n0) (path n1 n0)
(edge n0 n1) (path n0 n1)
(edge n1 n3) (path n1 n3)
(edge n3 n0) (path n3 n0)
(gate (S Z) go)
"#;
            {
                let mut naive = run_naive(minimal);
                let mut raw = run_semi_mode(minimal, SniRetractMode::RawSemiNoGate);
                let raw_calls = raw.sni_delta_calls;
                let nd = dump(&mut naive);
                let rd = dump(&mut raw);
                if nd != rd {
                    eprintln!("[DIVERGES] MINIMAL_REPRO (raw, semi delta firings={raw_calls})");
                    eprintln!("  naive HAS, raw-semi MISSES (DRed must re-derive): {:?}", dish_minus(&nd, &rd));
                    eprintln!("  raw-semi HAS, naive LACKS (DRed must over-delete): {:?}", dish_minus(&rd, &nd));
                } else {
                    eprintln!("[identical] MINIMAL_REPRO under RawSemiNoGate (firings={raw_calls})");
                }
                // And under DRed: must be byte-identical, with repairs > 0.
                let mut dred = run_semi_mode(minimal, SniRetractMode::Dred);
                let dred_repairs = dred.sni_dred_repairs;
                let dred_fallbacks = dred.sni_dred_fallbacks;
                let dd = dump(&mut dred);
                if dd == nd {
                    eprintln!("[Dred OK] MINIMAL_REPRO byte-identical under Dred (repairs={dred_repairs} fallbacks={dred_fallbacks})");
                } else {
                    eprintln!("[Dred DIVERGES] MINIMAL_REPRO (repairs={dred_repairs} fallbacks={dred_fallbacks})");
                    eprintln!("  naive HAS, dred MISSES: {:?}", dish_minus(&nd, &dd));
                    eprintln!("  dred HAS, naive LACKS: {:?}", dish_minus(&dd, &nd));
                }
            }
            // STALENESS PROBE: derive (path a c) from (edge a b)+(path b c), then
            // remove ONLY (edge a b)+(path b c) (the support), keeping (path a c).
            // Does NAIVE keep the now-unsupported (path a c)? If yes, naive is a
            // MONOTONE TRACE (keeps stale derived facts), NOT mat(Pi, survivors),
            // and DRed must NOT over-delete derived facts whose support is removed.
            {
                let staleness = r#"
(exec (D (S (S (S Z))))
      (, (exec (D (S $c)) $sp $st) ((W) $p $t) ((G) $gp $gt))
      (, (exec (D $c) $sp $st) (exec (R) $p $t) (exec (GR) $gp $gt)))
((W) (, (edge $x $y) (path $y $z)) (, (path $x $z)))
((G) (, (exec (D $c) $a $b) (gate $c go)) (O (- (edge a b)) (- (path b c)) (- (gate $c go))))
(edge a b) (path a b)
(edge b c) (path b c)
(gate (S Z) go)
"#;
                let mut naive = run_naive(staleness);
                let nd = dump(&mut naive);
                eprintln!(
                    "[STALENESS] naive keeps unsupported (path a c)? {} ; keeps (path a b)? {}",
                    nd.contains("(path a c)"),
                    nd.contains("(path a b)"),
                );
            }
            for (name, setup) in corpus() {
                let mut naive = run_naive(&setup);
                let mut raw = run_semi_mode(&setup, SniRetractMode::RawSemiNoGate);
                let raw_calls = raw.sni_delta_calls;
                let raw_removal = raw.sni_removal_seen;
                let nd = dump(&mut naive);
                let rd = dump(&mut raw);
                if nd == rd {
                    if raw_removal {
                        eprintln!(
                            "[identical, has-removal] {name}: raw-semi byte-identical despite a removal (semi delta firings={raw_calls})"
                        );
                    }
                    continue;
                }
                let naive_has_raw_misses = dish_minus(&nd, &rd);
                let raw_has_naive_lacks = dish_minus(&rd, &nd);
                eprintln!("[DIVERGES] {name} (semi delta firings={raw_calls}, removal_seen={raw_removal})");
                eprintln!("  naive HAS, raw-semi MISSES (DRed must re-derive): {naive_has_raw_misses:?}");
                eprintln!("  raw-semi HAS, naive LACKS (DRed must over-delete): {raw_has_naive_lacks:?}");
            }
        }

        // The fast path is preserved: a monotone (no-removal) re-arming closure and
        // the full process_calculus reaction both STILL take the semi-naive delta
        // path (the gate does not trip), so the perf win is intact. Proven by the
        // SNI_DELTA_CALLS counter being positive after the run, and the gate flag
        // staying false.
        #[test]
        fn gate_preserves_fast_path_on_process_calculus() {
            // process_calculus add(8,8): no removals anywhere, so the gate must NOT
            // trip and the delta transform MUST run many times.
            let setup = process_calculus_setup_sexpr(100, 8, 8);
            let mut s = build(&setup);
            unsafe {
                SNI_DELTA_CALLS = 0;
            }
            s.metta_calculus(1_000_000_000_000_000);
            let calls = unsafe { SNI_DELTA_CALLS };
            assert!(
                !s.sni_removal_seen,
                "process_calculus has no removals, the gate must not trip"
            );
            assert!(
                calls > 0,
                "the semi-naive delta path must run on process_calculus (fast path preserved); SNI_DELTA_CALLS={calls}"
            );
            // Confirm the answer is still correct (peano(16) on result).
            let pat = crate::expr!(s, "[2] petri [3] ! result $");
            let tpl = crate::expr!(s, "_1");
            let mut v = Vec::new();
            s.dump_sexpr(pat, tpl, &mut v);
            assert_eq!(
                String::from_utf8_lossy(&v),
                format!("{}\n", peano_sexpr(16)),
                "process_calculus must still compute add(8,8)=peano(16) on the gated fast path"
            );
        }

        // A monotone re-arming closure also keeps the fast path (delta runs, gate
        // stays armed), to show the gate is not over-eager: it trips ONLY on a
        // removal, never on a plain add-only recursive program.
        #[test]
        fn gate_does_not_trip_on_monotone_recursion() {
            let setup = rearming_program(
                8,
                "(, (edge $x $y) (path $y $z)) (, (path $x $z))",
                "(edge a b) (edge b c) (edge c d) (edge d e)\n\
                 (path a b) (path b c) (path c d) (path d e)",
                "",
            );
            let mut s = build(&setup);
            unsafe {
                SNI_DELTA_CALLS = 0;
            }
            s.metta_calculus(1_000_000_000_000_000);
            assert!(!s.sni_removal_seen, "no removal => gate must stay armed");
            assert!(
                unsafe { SNI_DELTA_CALLS } > 0,
                "the delta path must run on a monotone re-arming closure"
            );
        }

        // Run `setup` with the cost gate live (`sni_force_naive = false`, the
        // default) and with the runtime revert switch forcing naive
        // (`sni_force_naive = true`), and return the two dumps plus the
        // SNI_DELTA_CALLS the gated run made. The forced run is exactly the
        // feature-off behaviour reached without a rebuild.
        // Run `setup` three ways and return (dump, semi-firings) for each:
        //  - GATED:  the default build (cost gate live, `sni_force_naive = false`);
        //  - FORCED: the runtime revert switch on (`sni_force_naive = true`), which
        //            must reach exactly the feature-off naive behaviour at runtime;
        //  - REF:    the inert-hook reference (`run_naive`, `sni_rule_seen` never
        //            armed -- the feature-OFF loop driven by hand).
        // The semi-firing count is read from the PER-SPACE `sni_delta_calls` (not the
        // process-global `SNI_DELTA_CALLS`), so it is correct even though `cargo test`
        // runs tests in parallel threads that share the global counter.
        fn run_three_ways(setup: &str) -> ((String, usize), (String, usize), String) {
            fn dump(s: &mut Space) -> String {
                let mut v = Vec::new();
                s.dump_all_sexpr(&mut v).unwrap();
                String::from_utf8_lossy(&v).into_owned()
            }

            let mut gated = build(setup);
            gated.sni_force_naive = false;
            gated.metta_calculus(1_000_000_000_000_000);
            let gated_calls = gated.sni_delta_calls;

            let mut forced = build(setup);
            forced.sni_force_naive = true;
            forced.metta_calculus(1_000_000_000_000_000);
            let forced_calls = forced.sni_delta_calls;

            let mut reference = run_naive(setup);

            (
                (dump(&mut gated), gated_calls),
                (dump(&mut forced), forced_calls),
                dump(&mut reference),
            )
        }

        // The cost gate is RESULT-NEUTRAL: the gated path (semi where the delta is
        // small, naive where it is large) writes a dish byte-identical to the
        // always-naive path, AND the runtime revert switch (`sni_force_naive`)
        // reaches that same naive dish. Checked on BOTH gate regimes:
        //  - a SMALL program where the gate routes EVERY firing to naive (its only
        //    firings have delta ~ dish), so the gated run takes zero delta passes;
        //  - a LARGE monotone recursive closure with many small-delta firings, so
        //    the gate picks semi (the gated run takes >0 delta passes).
        // If the cost gate ever changed an answer (it must not -- it only chooses
        // HOW to compute the same immediate consequences) the dumps would diverge.
        // All assertions use race-free dump comparisons and the per-Space firing
        // count, so the test is correct under parallel execution.
        #[test]
        fn cost_gate_is_result_neutral() {
            // SMALL: a pure-source join over a tiny dish; every firing has
            // delta ~ dish (m == 2), so the gate routes them all to naive.
            let small = r#"
(exec 0 (, (a $x) (b $x)) (, (ab $x)))
(a 1) (a 2) (a 3) (b 2) (b 3) (b 4)
"#;
            let ((g, g_calls), (f, _), r) = run_three_ways(small);
            assert_eq!(g, r, "cost gate changed the result on the small program");
            assert_eq!(f, r, "sni_force_naive must reach the feature-off naive dish (small)");
            assert_eq!(
                g_calls, 0,
                "the small program (delta ~ dish every firing) must be gated entirely to naive"
            );

            // LARGE: a re-arming transitive closure over a longer chain. Many
            // firings have delta << dish, so the gate picks semi (delta path runs),
            // and the result must still be byte-identical to always-naive.
            let large = rearming_program(
                12,
                "(, (edge $x $y) (path $y $z)) (, (path $x $z))",
                "(edge a b) (edge b c) (edge c d) (edge d e) (edge e f) (edge f g)\n\
                 (edge g h) (edge h i) (edge i j) (edge j k) (edge k l)\n\
                 (path a b) (path b c) (path c d) (path d e) (path e f) (path f g)\n\
                 (path g h) (path h i) (path i j) (path j k) (path k l)",
                "",
            );
            let ((g2, g2_calls), (f2, _), r2) = run_three_ways(&large);
            assert_eq!(g2, r2, "cost gate changed the result on the large program");
            assert_eq!(f2, r2, "sni_force_naive must reach the feature-off naive dish (large)");
            assert!(
                g2_calls > 0,
                "the large recursive closure must exercise the semi-naive delta path (gate picked semi)"
            );
        }
    }
}
