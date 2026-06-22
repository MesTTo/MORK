#![feature(gen_blocks)]
#![feature(coroutine_trait)]
#![feature(coroutines)]
#![feature(stmt_expr_attributes)]

pub mod arrangements;
pub mod binding_dag;
pub mod binding_env;
pub mod binding_plan;
pub mod binding_space;
pub mod critical_pairs;
pub mod egraph;
#[cfg(test)]
mod encoded_test_helpers;
pub mod expression_trie;
pub mod formal_lowering;
pub mod json_path_query;
#[cfg(feature = "experimental_dnf")]
pub mod path_dnf;
pub mod path_space_ops;
#[cfg(test)]
mod pathmap_zipper_laws;
pub mod pattern_relations;
mod pure;
mod sinks;
mod sources;
pub mod prefix {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct Prefix<'a> {
        pub slice: &'a [u8],
    }

    impl<'a> Prefix<'a> {
        pub fn path(&self) -> &'a [u8] {
            self.slice
        }
    }
}
#[cfg(feature = "einsum")]
pub mod graph_tensor;
pub mod semiring;
pub mod shard_zipper;
pub mod space;
#[cfg(feature = "einsum")]
mod tensor_ops;
pub mod term_identity;
pub mod weighted_paths;

/// Curated public API for derived term, BindingSpace, and sidecar-planner
/// execution surfaces.
///
/// The underlying modules remain public for detailed experiments. This module
/// provides a stable import point for callers that need the optimized
/// non-materializing count/existence paths, execution reports, and sidecar
/// relation primitives without depending on the current file layout.
pub mod api {
    pub use crate::arrangements::{
        ArrangementDescriptor, ArrangementError, ArrangementIndex, ArrangementProjection,
        ArrangementRow, ArrangementStats,
    };
    pub use crate::binding_env::{
        Binding, BindingEnv, BindingEnvError, BindingMark, BindingSnapshot, BindingSource,
        MAX_BINDING_SLOTS,
    };
    pub use crate::binding_plan::{
        BindingAccessPlan, BindingAdaptiveEpisodePlan, BindingAdaptiveOrderEstimate,
        BindingAdaptiveTaskBox, BindingAdaptiveTaskRange, BindingParallelSharePlan,
        BindingSidecarAggregateKernel, BindingSidecarAnalysis, BindingSidecarCountReport,
        BindingSidecarExecutionChoice, BindingSidecarExecutionKernel,
        BindingSidecarExecutionReason, BindingSidecarExecutionReport, BindingSidecarExistsReport,
        BindingSidecarPlan, BindingSidecarPlanError, BindingSidecarPrepared, BindingSidecarResult,
        BindingSidecarSelectedResult, BindingSidecarStats, BindingSidecarTrieCursorContractReport,
        BindingSidecarTrieJoinResult, BindingSidecarTrieTraceDiffReport,
        BindingSidecarTrieTraceReport, BindingVariableDomainStats, BindingVariableOrderCandidate,
        BindingVariableOrderSource, BindingVariableShare, PatternProjection,
    };
    pub use crate::binding_space::{
        BindingDomainCursor, BindingDomainIntersection, BindingRelation, BindingRelationError,
        BindingRow, BindingVar, FactorGroup, FactorizedJoin, GenericJoinCount,
        GenericJoinExistence, PreparedTrieJoin, SemiNaiveTransitiveClosure, SemijoinReduction,
        TrieJoinCount, TrieJoinCursorContract, TrieJoinExistence, TrieJoinFactorCursorContext,
        TrieJoinFactorCursorRequirement, TrieJoinResult, TrieJoinStats, TrieJoinTrace,
        TrieJoinTraceFactorDomain, TrieJoinTraceImpact, TrieJoinTraceReplayBranch,
        TrieJoinTraceReplayDiff, TrieJoinTraceReplayDiffEntry, TrieJoinTraceReplayDiffKind,
        TrieJoinTraceReplayNode, TrieJoinTraceReplayShape, TrieJoinTraceShapeError,
        TrieJoinTraceStep, TrieJoinTraceSummary, delta_join, generic_join, generic_join_count,
        generic_join_exists, intersect_binding_domain_cursors, natural_join,
        semi_naive_transitive_closure, semijoin_presence, semijoin_reduce_presence, trie_join,
        trie_join_count, trie_join_exists, trie_join_trace,
    };
    pub use crate::critical_pairs::{
        AdditiveSaturationComparisonReport, AdditiveSaturationReport, AdditiveSaturationRound,
        AdditiveSaturationRoundWorkSavings, AdditiveSaturationRuleRound,
        AdditiveSaturationWorkSavings, CriticalPairWitness, Rule, RuleExtractionError, StateRule,
        StateStep, Term, compare_additive_saturation_reports, first_non_joinable_witness,
        ground_facts_from_mm2_program, rules_from_mm2_program, saturate_additive_state,
        saturate_additive_state_report, saturate_additive_state_semi_naive,
        saturate_additive_state_semi_naive_report, state_rule_successors,
        state_rules_from_mm2_program,
    };
    pub use crate::expression_trie::{
        ExpressionFeature, ExpressionTrieCandidates, ExpressionTrieError, ExpressionTrieIndex,
        ExpressionTrieMatches, ExpressionTrieStats, ExpressionTrieToken,
    };
    pub use crate::formal_lowering::{
        FormalContextHole, FormalContextOwner, FormalEffectKind, FormalExecPlan, FormalExecSummary,
        FormalHoleKind, FormalListMode, FormalLoweringError, FormalMettaSpecialForm,
        FormalMettaSpecialResult, FormalMinimalInstruction, FormalSourceKind, FormalSourcePlan,
        FormalTemplateEffect, lower_exec, metta_special_form,
    };
    pub use crate::json_path_query::{
        JsonPathQueryError, JsonPathSegment, parse_singular_json_path,
    };
    #[cfg(feature = "experimental_dnf")]
    pub use crate::path_dnf::{
        DnfPathSetCountResult, DnfPathSetError, DnfPathSetExistenceResult, DnfPathSetResult,
        DnfPathSetStats, evaluate_pathmap_dnf, evaluate_pathmap_dnf_count,
        evaluate_pathmap_dnf_exists, evaluate_pathmap_dnf_zipper_merge,
    };
    pub use crate::path_space_ops::{
        prefix_maximal_values, prefix_minimal_values, shared_prefix_witnesses,
    };
    pub use crate::pattern_relations::{
        ApplicationAtom, PatternLoweringError, PatternRelationMatchError,
        PatternRelationMatchStats, PatternRelationMatches, PatternRelationPlan, PatternRelationRow,
        PlanValue, PlanVariable, PlanVariableKind, PlanVariableRecord, lower_pattern,
        match_fact_ids, match_facts,
    };
    pub use crate::sinks::{
        WASM_LINEAR_MEMORY_GUARD_BYTES, WASM_LINEAR_MEMORY_GUARD_ENV,
        WASM_LINEAR_MEMORY_RESERVATION_BYTES, WASM_LINEAR_MEMORY_RESERVATION_ENV,
        WasmLinearMemoryPolicy, WasmLinearMemoryPolicyError, wasm_linear_memory_policy,
        wasm_linear_memory_policy_from_env, wasm_linear_memory_policy_from_values,
    };
    pub use crate::space::{
        QueryProjectionByteDomainCursor, QueryProjectionDomainCursor,
        QueryProjectionDomainIntersection, QueryProjectionProductCandidateTrace,
        QueryProjectionProductRawCandidateCounters, QueryProjectionProductTraceComparison,
        QueryProjectionProductVsTrieTraceComparison, QueryProjectionRelationDomain,
        QueryProjectionRelationFactor, QueryProjectionTrieContractComparison,
        QueryProjectionTrieContractContextComparison, QueryProjectionZipperDomainTelemetry,
        QueryProjectionZipperRelationFactor, Space,
        compare_query_projection_product_trace_to_binding_relation,
        compare_query_projection_product_trace_to_trie_trace,
        compare_query_projection_relation_factors_to_trie_contract,
        compare_query_projection_zipper_factors_to_trie_contract,
        intersect_query_projection_byte_domain_cursors, intersect_query_projection_domain_values,
        intersect_query_projection_domains,
    };
    pub use crate::term_identity::{
        FactId, FactRecord, TermFlags, TermId, TermIdentitySidecar, TermIdentityStats, TermKind,
        TermParseError, TermParseErrorKind, TermRecord,
    };
    pub use crate::weighted_paths::{
        WeightedPathError, WeightedPathIndex, WeightedPathStats, WeightedSelectionTree,
        WeightedSelectionTreeStats,
    };
    pub use pathmap::PathMap;
}

#[doc(hidden)]
pub use mork_expr as __mork_expr;
#[doc(hidden)]
pub use mork_frontend as __mork_frontend;
