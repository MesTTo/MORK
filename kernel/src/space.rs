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
    dependencies: Vec<QueryFactorPlanDependency>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct QueryFactorPlanDependency {
    prefix: Vec<u8>,
    prefix_cardinality_bucket: usize,
}

/// Order-of-magnitude bucket of a cardinality: its bit length (0 for 0, else
/// floor(log2)+1). The query plan is only a factor ordering, never affecting the
/// join's output, so keying the plan cache on the bucket instead of the exact
/// count lets a cached plan survive the small per-step cardinality drift of a
/// mutating space. The key changes only when a cardinality crosses a
/// power-of-two band, where re-ranking is actually worthwhile. This is what
/// turns the per-step replanning on workloads like process_calculus (the space
/// changes every exec step, so an exact-count key misses every time) into cache
/// hits, without changing any result.
fn cardinality_bucket(count: usize) -> usize {
    (usize::BITS - count.leading_zeros()) as usize
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
        self.max_renormalized_factor_len =
            self.max_renormalized_factor_len.max(other.max_renormalized_factor_len);
        self.max_renormalized_factor_capacity =
            self.max_renormalized_factor_capacity.max(other.max_renormalized_factor_capacity);
        self.raw_searches += other.raw_searches;
        self.raw_stack_entries_sum += other.raw_stack_entries_sum;
        self.max_raw_stack_entries = self.max_raw_stack_entries.max(other.max_raw_stack_entries);
        self.candidate_pair_vectors += other.candidate_pair_vectors;
        self.candidate_pair_entries_sum += other.candidate_pair_entries_sum;
        self.candidate_pair_capacity_sum += other.candidate_pair_capacity_sum;
        self.max_candidate_pair_entries =
            self.max_candidate_pair_entries.max(other.max_candidate_pair_entries);
        self.max_candidate_pair_capacity =
            self.max_candidate_pair_capacity.max(other.max_candidate_pair_capacity);
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
fn query_execution_storage_metrics_registry(
) -> &'static Mutex<Vec<std::sync::Arc<Mutex<QueryExecutionStorageMetrics>>>> {
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
                        if e.n == 0 {
                            references.push(loc.path().len() as u32);
                        } else {
                            trace!(target: "coref trans", "not putting {} {}", e.n, e.show());
                            // trace!(target: "coref trans", "not putting against {:?}", loc.child_mask());
                        }
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

                        if e.n == 0 {
                            references.pop();
                        }
                    }
                    Tag::VarRef(i) => {
                        if e.n == 0 {
                            if i as usize >= references.len() {
                                trace!(target: "coref trans", "i {i} #references {}", references.len());
                                stack.push(e);
                                return;
                            }
                            // The data subterm bound to this variable at its first
                            // occurrence (`references[i]` is its start in the data path).
                            let bound = Expr {
                                ptr: loc
                                    .path()
                                    .as_ptr()
                                    .cast_mut()
                                    .offset(references[i as usize] as _),
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
                                    if loc.child_mask().and(&ByteMask(VARS)).iter().next().is_some()
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
    /// `Tag::NewVar`: match any term here; `introduces` records a coreference path.
    Any { introduces: bool },
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
    let mut introduced = 0usize;
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
    introduced: &mut usize,
) -> Option<()> {
    let n = e.n;
    let mut ptr = e.subsexpr().ptr;
    // Open compounds awaiting children: (op index of the Compound, children left).
    let mut open: Vec<(usize, u32)> = Vec::new();
    loop {
        match unsafe { byte_item(*ptr) } {
            Tag::NewVar => {
                let introduces = n == 0;
                if introduces {
                    *introduced += 1;
                }
                ops.push(MatchOp::Any { introduces });
                ptr = unsafe { ptr.byte_add(1) };
                close_compound_child(&mut open, ops);
            }
            Tag::VarRef(index) => {
                if n == 0 && (index as usize) < *introduced {
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
    introduced: &mut usize,
    args_scratch: &mut Vec<ExprEnv>,
) -> Option<()> {
    let ptr = e.subsexpr().ptr;
    match unsafe { byte_item(*ptr) } {
        Tag::NewVar => {
            let introduces = e.n == 0;
            if introduces {
                *introduced += 1;
            }
            ops.push(MatchOp::Any { introduces });
        }
        Tag::VarRef(index) => {
            if e.n == 0 && (index as usize) < *introduced {
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
    unsafe {
        let Some(op) = ops.get(pc) else {
            f(loc);
            return;
        };
        match op {
            MatchOp::Any { introduces } => {
                if *introduces {
                    references.push(loc.path().len() as u32);
                }
                match_any_term(loc, 1, ops, pc + 1, references, scratch, f);
                if *introduces {
                    references.pop();
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
}

fn rematch_bound_then_program<Z, F>(
    loc: &mut Z,
    bound: Expr,
    ops: &[MatchOp],
    pc_after: usize,
    references: &mut Vec<u32>,
    scratch: &mut MatchProgramScratch,
    f: &mut F,
) where
    Z: ZipperMoving + Zipper + ZipperAbsolutePath + ZipperIteration,
    F: FnMut(&mut Z) -> (),
{
    let addition = ExprEnv {
        n: 254,
        v: 0,
        offset: 0,
        base: bound,
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
    if index >= references.len() {
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
            rematch_bound_then_program(loc, bound, ops, pc_after, references, scratch, f);
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
        rematch_bound_then_program(loc, bound, ops, pc_after, references, scratch, f);
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
            0, 0, 0, pat_expr, bindings, buffer, stack, assignments
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
            0, oi, ni, template, bindings, buffer, stack, assignments
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
        let Some((oi, ni)) =
            Self::pattern_template_intros(bindings, pat_expr, &mut *buffer, &mut *stack, &mut *assignments)
        else {
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
                        .cmp(&ranks[b].min_variable_domain_cardinality.unwrap_or(usize::MAX))
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

        // The sidecar join is relational: it equates interned ground tuples.
        // Schematic stored facts need first-order unification, so ProductZipper
        // remains authoritative for any relation that contains variables.
        if sidecar.any_schematic_fact_under_prefixes(&prefixes) {
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

    fn query_factor_plan_cache_key(
        btm: &PathMap<()>,
        sources: &[ExprEnv],
    ) -> Option<QueryFactorPlanCacheKey> {
        let mut factors = Vec::with_capacity(sources.len());
        let mut dependency_counts = BTreeMap::new();
        for source in sources {
            let span = unsafe { source.subsexpr().span().as_ref()? };
            factors.push(span.to_vec());
            if let Some(prefix) = query_source_prefix(*source).filter(|prefix| !prefix.is_empty()) {
                dependency_counts
                    .entry(prefix)
                    .or_insert_with_key(|prefix| btm.read_zipper_at_path(prefix).val_count());
            }
        }
        let dependencies = dependency_counts
            .into_iter()
            .map(|(prefix, prefix_cardinality)| QueryFactorPlanDependency {
                prefix,
                prefix_cardinality_bucket: cardinality_bucket(prefix_cardinality),
            })
            .collect();
        Some(QueryFactorPlanCacheKey {
            factors,
            dependencies,
        })
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
                        .cmp(&ranks[b].min_variable_domain_cardinality.unwrap_or(usize::MAX))
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
        let Some(cache_key) = Self::query_factor_plan_cache_key(btm, sources) else {
            return Self::query_factor_plan_uncached(btm, sources, None);
        };

        {
            let mut cache = query_factor_plan_cache().lock().unwrap();
            if let Some(plan) = cache.get(&cache_key) {
                debug!(
                    target: "query_plan_cache",
                    "hit factors={} dependencies={}",
                    cache_key.factors.len(),
                    cache_key.dependencies.len()
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
            "miss factors={} dependencies={}",
            cache_key.factors.len(),
            cache_key.dependencies.len()
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

        for &source_idx in plan {
            let source = sources[source_idx];
            let capacity = unsafe { source.subsexpr().span().as_ref().unwrap().len() };
            let mut buffer = Vec::with_capacity(capacity);
            Self::append_renormalized_query_factor(
                source,
                &mut var_map,
                &mut next_var,
                &mut buffer,
            )?;
            buffers.push(buffer);
        }

        record_storage_metrics(|m| m.record_renormalized_plan(&buffers));

        let planned_sources = buffers
            .iter()
            .map(|buffer| {
                ExprEnv::new(
                    0,
                    Expr {
                        ptr: buffer.as_ptr().cast_mut(),
                    },
                )
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
        // Single-factor fast path. A one-source query has an unconditional plan of `[0]`
        // (the `n <= 1` early return in query_factor_plan_uncached), and a 0-normalized source
        // renormalizes to itself, so matching it directly is byte-identical to the planned
        // path. Taking it avoids both `query_factor_plan`'s global cache mutex and
        // `renormalize_query_factors`'s metrics mutex, each locked once per query; profiling a
        // single-pattern point-query loop showed those locks dominate per-query cost and
        // serialize parallel queries (throughput collapses past ~8 threads). Multi-factor
        // queries are unchanged.
        let (_planned_buffers, planned_sources, planned_unify_sources, primary_source_index, plan_reordered) =
            if sources.len() == 1 {
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
                    None => (Vec::new(), sources.to_vec(), sources.to_vec(), 0, plan_reordered),
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
        let plan = Self::query_factor_plan(btm, sources);
        let planned = Self::renormalize_query_factors(sources, &plan);
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

        // Drive the template writes from the sidecar's worst-case-optimal join
        // when the body lowers (a ground `,` conjunction). The emit is proven
        // equal to the ProductZipper's (validate_sidecar_emit_against_product),
        // and the walk-step measurement shows the join takes asymptotically fewer
        // steps. A cheap structural pre-check (the join graph from the pattern's
        // variables, no data) skips acyclic bodies before paying the O(relation)
        // interning, since the worst-case-optimal join only beats the ProductZipper
        // on cyclic bodies. Falls through to the ProductZipper otherwise.
        // The worst-case-optimal join only wins on a genuinely cyclic, growing
        // join. A body whose cardinality order is disconnected (a function table
        // whose `args` input sorts last, as in finite_domain) reorders to a cheap
        // connected ProductZipper plan, so route it straight there and skip the
        // O(relation) sidecar sync+probe entirely. The check is cheap btm stats
        // and short-circuits after `body_is_cyclic`, so non-cyclic bodies pay
        // nothing and a real cyclic pattern (clique, transitive's triangle) still
        // takes the WCO path.
        #[cfg(feature = "sidecar_bridge_emit")]
        if Self::body_is_cyclic(pat_expr)
            && Self::body_cardinality_order_connected(&self.btm, pat_expr)
        {
            let mut read_copy = self.btm.clone();
            read_copy.insert(unsafe { add.span().as_ref().unwrap() }, ());
            if let Some(result) = self.transform_via_sidecar(&read_copy, pat_expr, tpl_expr) {
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
                                0, 0, 0, pat_expr, bindings, void, trace, assignments
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
        });
        for wz in template_wzs {
            zh.cleanup_write_zipper(wz);
        }
        (touched, any_new)
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
                                0, 0, 0, pat_expr, bindings, void, trace, assignments
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
        #[cfg(feature = "sidecar_bridge_emit")]
        if sinks.iter().any(|s| s.is_remove()) {
            self.bridge_remove_gen = self.bridge_remove_gen.wrapping_add(1);
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
        #[cfg(feature = "sidecar_bridge_emit")]
        if sinks.iter().any(|s| s.is_remove()) {
            self.bridge_remove_gen = self.bridge_remove_gen.wrapping_add(1);
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
            dependencies: Vec::new(),
        }
    }

    #[test]
    fn cardinality_bucket_is_stable_within_a_power_of_two_band() {
        // Counts inside one [2^(k-1), 2^k) band share a bucket, so the small
        // per-step cardinality drift of a mutating space keeps the plan-cache
        // key stable and the plan is reused. The key only changes when a count
        // crosses a band, where re-ranking is worthwhile.
        assert_eq!(cardinality_bucket(0), 0);
        assert_eq!(cardinality_bucket(1), 1);
        assert_eq!(cardinality_bucket(2), cardinality_bucket(3));
        assert_eq!(cardinality_bucket(64), cardinality_bucket(127));
        assert_eq!(cardinality_bucket(128), cardinality_bucket(255));
        assert_ne!(cardinality_bucket(127), cardinality_bucket(128));
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
        space.add_all_sexpr(b"(implies (Frog $x) (Green $x))\n").unwrap();
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
        assert_eq!(count, 1, "callback fired more than once: data coreference double-counted");
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
        assert_eq!(touched, 1, "matcher under-produces: missed compound binding to a reused data var");
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
        space.add_all_sexpr(b"(= (lam $v $b) (got $v $b))\n").unwrap();
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
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge a b)
(edge b c)
(edge c a)
(if (S $n) $x $y $x)
"#,
            )
            .unwrap();

        let (pat_expr, sources) = query_pattern_and_sources(
            &mut space,
            "[5] , [3] edge $ $ [3] edge _2 $ [3] edge _3 _1 [5] if $ [2] S _1 fallback $",
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
            "the (if ...) relation contains a schematic stored fact"
        );

        let read_copy = space.btm.clone();
        assert!(
            space
                .transform_via_sidecar(&read_copy, pat_expr, tpl_expr)
                .is_none(),
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
                    "[2] q foo",                 // SymbolSize child
                    "[2] q [2] S Z",             // Arity-2 child
                    "[2] q [3] t a b",           // Arity-3 child
                    "[2] q x",                   // another SymbolSize
                    "[2] q [2] K [2] S Z",       // nested arity
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
            "[2] q a",                                       // flat ground source
            "[3] q $ $",                                     // flat var sources
            "[2] q [2] S [2] S [2] S [2] S [2] S Z",         // deep ground chain S^5(Z)
            "[3] q [2] S [2] S [2] S [2] S $ a",             // deep chain, bottom var (adversarial)
            "[4] state $ $ [2] S [2] S Z",                   // counter_machine shape: vars + deep ground
            "[3] f [3] g a b [2] h c",                       // nested branching ground
            "[3] p $ _1",                                    // bound varref re-check
            "[2] q [1] Z",                                   // arity-1 compound
            "[3] mix [2] S [2] S $ [2] T [2] T Z",           // mixed deep var + deep ground
        ];
        for case in cases {
            let pat = crate::expr!(space, case);
            let mut args = Vec::new();
            ExprEnv::new(0, pat).args(&mut args);
            let sources = &args[1..];

            let mut lin = Vec::new();
            let mut intro_l = 0usize;
            let mut accept_l = true;
            for &s in sources {
                if compile_match_factor(s, &mut lin, &mut intro_l).is_none() {
                    accept_l = false;
                    break;
                }
            }

            let mut rec = Vec::new();
            let mut intro_r = 0usize;
            let mut scratch = Vec::new();
            let mut accept_r = true;
            for &s in sources {
                if compile_match_factor_recursive(s, &mut rec, &mut intro_r, &mut scratch).is_none() {
                    accept_r = false;
                    break;
                }
            }

            assert_eq!(accept_l, accept_r, "accept mismatch: {case}");
            assert_eq!(intro_l, intro_r, "introduced count mismatch: {case}");
            assert_eq!(lin, rec, "compiled ops differ (linear vs recursive): {case}");
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
    fn query_factor_plan_cache_key_buckets_dependency_cardinalities() {
        // Start CacheDepEdge with 4 atoms (cardinality bucket 3, the band
        // [4, 8)), so we can probe both an in-band and a band-crossing change.
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
        let before = Space::query_factor_plan_cache_key(&space.btm, &sources).unwrap();

        // Mutations that touch no queried prefix never change the key.
        let mut unrelated = String::new();
        for i in 0..256 {
            unrelated.push_str(&format!("(CacheDepNoise unrelated{i})\n"));
        }
        space.add_all_sexpr(unrelated.as_bytes()).unwrap();
        assert!(space.btm.val_count() >= before_count + 256);
        assert_eq!(
            Space::query_factor_plan_cache_key(&space.btm, &sources).unwrap(),
            before
        );

        // A small related change that stays inside the cardinality band (4 -> 5,
        // still bucket 3) reuses the plan: the key is unchanged. This is the
        // optimization that lets a mutating space keep hitting the plan cache.
        space.add_all_sexpr(b"(CacheDepEdge fresh left)\n").unwrap();
        assert_eq!(
            Space::query_factor_plan_cache_key(&space.btm, &sources).unwrap(),
            before
        );

        // Growing CacheDepEdge across the band boundary (to 8, bucket 4) is where
        // re-ranking is worthwhile, so the key changes.
        space
            .add_all_sexpr(b"(CacheDepEdge g6 left)\n(CacheDepEdge g7 left)\n(CacheDepEdge g8 left)\n")
            .unwrap();
        assert_ne!(
            Space::query_factor_plan_cache_key(&space.btm, &sources).unwrap(),
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
}
