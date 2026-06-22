use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};

use crate::arrangements::{
    ArrangementDescriptor, ArrangementError, ArrangementIndex, ArrangementProjection,
};
use crate::binding_env::MAX_BINDING_SLOTS;
use crate::binding_space::{
    BindingRelation, BindingRelationError, BindingRow, BindingVar, HyperTreeDecomposition,
    PreparedTrieJoin, SemijoinReduction, TrieJoinCursorContract, TrieJoinStats, TrieJoinTrace,
    TrieJoinTraceReplayDiff, TrieJoinTraceShapeError, agm_size_bound, generic_join,
    generic_join_count, generic_join_exists, ghd_join, ghd_join_count, ghd_join_exists,
    ghd_size_cost, gyo_join_tree, hypertree_decomposition, selectivity_variable_order,
    semijoin_reduce_presence,
};
use crate::expression_trie::{ExpressionTrieError, ExpressionTrieIndex};
use crate::pattern_relations::PatternRelationRow;
use crate::term_identity::{TermId, TermIdentitySidecar};

/// Physical BindingSpace access selected by a compiled sidecar plan.
///
/// This describes derived access over the term snapshot. It does not change
/// the canonical PathMap/ACT pathspace semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BindingAccessPlan {
    /// Build or use an argument-order arrangement, then project it into a
    /// BindingSpace relation.
    Arrangement {
        descriptor: ArrangementDescriptor,
        projection: ArrangementProjection,
    },
    /// Use the typed expression trie to retrieve candidate facts for one
    /// pattern, exact-filter them, then project user-visible slots into a
    /// BindingSpace relation.
    Pattern {
        pattern: TermId,
        projection: PatternProjection,
    },
}

/// Projection from a matched pattern row into a BindingSpace relation schema.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatternProjection {
    /// Output BindingSpace schema.
    pub schema: Box<[BindingVar]>,
    /// User-visible pattern slots for each schema variable.
    pub user_slots: Box<[u8]>,
}

impl PatternProjection {
    /// Creates a projection after validating arity and six-bit slot bounds.
    pub fn new(
        schema: impl Into<Box<[BindingVar]>>,
        user_slots: impl Into<Box<[u8]>>,
    ) -> Result<Self, BindingSidecarPlanError> {
        let schema = schema.into();
        let user_slots = user_slots.into();
        if schema.len() != user_slots.len() {
            return Err(BindingSidecarPlanError::PatternProjectionArityMismatch {
                schema_len: schema.len(),
                slots_len: user_slots.len(),
            });
        }
        for &slot in user_slots.iter() {
            if usize::from(slot) >= MAX_BINDING_SLOTS {
                return Err(BindingSidecarPlanError::InvalidPatternSlot { slot });
            }
        }

        Ok(Self { schema, user_slots })
    }
}

/// Sidecar join plan over derived BindingSpace relations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarPlan {
    factors: Box<[BindingAccessPlan]>,
    variable_order: Box<[BindingVar]>,
}

/// Opened sidecar factors and selected physical execution for one immutable
/// term snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarPrepared {
    relations: Box<[BindingRelation]>,
    execution: BindingSidecarExecutionReport,
    trie_join: Option<PreparedTrieJoin>,
    /// GYO join-tree `(child, parent)` edges driving the Yannakakis full
    /// reducer. `Some` exactly when the selected kernel is `AcyclicYannakakis`.
    join_tree_edges: Option<Box<[(usize, usize)]>>,
    /// Hypertree decomposition driving Yannakakis-over-bags. `Some` exactly when
    /// the selected kernel is `GhdYannakakis`.
    decomposition: Option<HyperTreeDecomposition>,
}

/// Result of executing a sidecar join plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarResult {
    /// Flat relation returned by the current reference Generic Join oracle.
    pub relation: BindingRelation,
    /// Execution counters for checking that the sidecar plan shape is visible.
    pub stats: BindingSidecarStats,
}

/// Result of executing a sidecar plan with the trie-backed join kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarTrieJoinResult {
    /// Flat relation returned by the trie-backed variable-at-a-time join.
    pub relation: BindingRelation,
    /// Variable order used by the trie-backed join.
    pub variable_order: Box<[BindingVar]>,
    /// Execution counters for physical factor opening.
    pub stats: BindingSidecarStats,
    /// Execution counters for the trie-backed join itself.
    pub trie_stats: TrieJoinStats,
}

/// Kernel selected for one sidecar execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingSidecarExecutionKernel {
    /// Reference variable-at-a-time Generic Join over opened relations.
    GenericJoin,
    /// Trie-backed LFTJ-style join using the root-domain suggested order.
    TrieJoinSuggested,
    /// Yannakakis full reducer over the GYO join tree, then Generic Join on the
    /// dangling-tuple-free relations. Selected for multiway (>= 3 factor)
    /// alpha-acyclic bodies, where a semijoin full reducer removes tuples that
    /// cannot join before the final materialization. The output set is identical
    /// to the plain Generic Join (Yannakakis, VLDB 1981); only intermediate work
    /// shrinks. See `binding_space::gyo_join_tree`.
    AcyclicYannakakis,
    /// Hypertree-decomposition evaluation: materialize each bag (join its
    /// relations), then run Yannakakis over the bag relations. Selected for cyclic
    /// bodies that are mostly acyclic with small bounded-width cyclic cores, where
    /// a single global worst-case-optimal join would still pay for the acyclic
    /// part. A tight cyclic core (a bare triangle) stays on the trie kernel
    /// instead. See `binding_space::ghd_join`.
    GhdYannakakis,
}

/// Why the sidecar planner selected an execution kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingSidecarExecutionReason {
    /// A single factor does not need a multiway trie join.
    SingleFactor,
    /// No variable occurs in more than one factor, so there is no domain
    /// intersection to exploit before producing the Cartesian product.
    NoSharedVariables,
    /// The explicit opened sidecar input is too small to justify building
    /// trie indexes for this experimental kernel.
    SmallExplicitInput,
    /// Shared-variable domains are visible and the opened input is large
    /// enough that LFTJ-style pruning is the better physical experiment.
    SharedVariablePruning,
    /// The body is multiway (>= 3 factors) and alpha-acyclic, so its GYO join
    /// tree admits a Yannakakis full reducer that prunes dangling tuples before
    /// the final join.
    AcyclicJoinTree,
    /// The body is cyclic but has a low-width hypertree decomposition whose
    /// acyclic structure dominates, so Yannakakis-over-bags beats a single global
    /// worst-case-optimal join.
    BoundedWidthDecomposition,
}

/// Planner-visible sidecar execution choice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarExecutionChoice {
    /// Selected physical kernel.
    pub kernel: BindingSidecarExecutionKernel,
    /// Human-readable planner reason encoded as a stable enum.
    pub reason: BindingSidecarExecutionReason,
    /// Variable order consumed by the selected kernel.
    pub variable_order: Box<[BindingVar]>,
    /// Positive rows available across opened factor projections.
    pub projected_rows: usize,
    /// Variables appearing in more than one factor.
    pub shared_variables: usize,
    /// Smallest root-domain intersection across shared variables.
    pub min_shared_root_domain_len: Option<usize>,
}

/// Result of executing the sidecar plan through its selected kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarSelectedResult {
    /// Flat relation returned by the selected kernel.
    pub relation: BindingRelation,
    /// Planner choice used for this execution.
    pub choice: BindingSidecarExecutionChoice,
    /// Execution counters for physical factor opening.
    pub stats: BindingSidecarStats,
    /// Trie counters when the trie-backed kernel was selected.
    pub trie_stats: Option<TrieJoinStats>,
}

/// Aggregate kernel used for a selected sidecar aggregate query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingSidecarAggregateKernel {
    /// The selected physical plan stayed with Generic Join, so the aggregate
    /// used the non-materializing reference count traversal.
    GenericJoinCount,
    /// The selected physical plan stayed with Generic Join, so existence used
    /// the non-materializing reference early-stop traversal.
    GenericJoinExists,
    /// Count was served by the non-materializing trie aggregate traversal.
    TrieJoinCount,
    /// Existence was served by the early-stop trie aggregate traversal.
    TrieJoinExists,
}

/// Result of counting selected sidecar matches.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarCountReport {
    /// Planner analysis and selected execution kernel.
    pub execution: BindingSidecarExecutionReport,
    /// Aggregate kernel used to answer the count.
    pub aggregate_kernel: BindingSidecarAggregateKernel,
    /// Exact positive output-row count.
    pub rows: usize,
    /// Execution counters for physical factor opening and aggregate work.
    pub stats: BindingSidecarStats,
    /// Trie counters when a trie aggregate kernel was used.
    pub trie_stats: Option<TrieJoinStats>,
    /// Whether this aggregate had to materialize the flat output relation.
    pub materialized_output: bool,
}

/// Result of checking whether selected sidecar matches exist.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarExistsReport {
    /// Planner analysis and selected execution kernel.
    pub execution: BindingSidecarExecutionReport,
    /// Aggregate kernel used to answer existence.
    pub aggregate_kernel: BindingSidecarAggregateKernel,
    /// Whether at least one positive output row exists.
    pub exists: bool,
    /// Execution counters for physical factor opening and aggregate work.
    pub stats: BindingSidecarStats,
    /// Trie counters when a trie aggregate kernel was used.
    pub trie_stats: Option<TrieJoinStats>,
    /// Whether this aggregate had to materialize the flat output relation.
    pub materialized_output: bool,
}

/// Explain-only sidecar execution report.
///
/// This reports the physical kernel choice and the opened-factor statistics
/// used to choose it, but deliberately does not execute the final join.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarExecutionReport {
    /// Root-domain and factor statistics gathered while opening the plan.
    pub analysis: BindingSidecarAnalysis,
    /// Planner choice derived from the analysis.
    pub choice: BindingSidecarExecutionChoice,
}

/// Non-materializing trace for the selected trie-backed sidecar execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarTrieTraceReport {
    /// Planner analysis and selected execution kernel.
    pub execution: BindingSidecarExecutionReport,
    /// BindingRelation trie traversal trace when the selected kernel is
    /// trie-backed; `None` when the selector conservatively keeps Generic Join.
    pub trie_trace: Option<TrieJoinTrace>,
}

/// Explain-only diff between selected trie traces for two sidecar snapshots.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarTrieTraceDiffReport {
    /// Selected trace report for the previous snapshot.
    pub old: BindingSidecarTrieTraceReport,
    /// Selected trace report for the newer snapshot.
    pub new: BindingSidecarTrieTraceReport,
    /// Replay-shape diff when both snapshots select the trie-backed kernel.
    pub replay_diff: Option<TrieJoinTraceReplayDiff>,
}

/// Explain-only cursor contract for a selected trie-backed sidecar execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarTrieCursorContractReport {
    /// Planner analysis and selected execution kernel.
    pub execution: BindingSidecarExecutionReport,
    /// Per-factor cursor contract when the selected kernel is trie-backed;
    /// `None` when the selector conservatively keeps Generic Join.
    pub cursor_contract: Option<TrieJoinCursorContract>,
}

/// Explain-only cost comparison behind the GHD routing decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindingSidecarRoutingCost {
    /// AGM size bound of a single global worst-case-optimal join over the opened
    /// relations: the smallest size product over edge covers.
    pub global_agm_bound: u128,
    /// Width of the hypertree decomposition within `GHD_MAX_WIDTH`, when one
    /// exists.
    pub decomposition_width: Option<usize>,
    /// Cost of the decomposition (its largest bag's size product), when one
    /// exists. The selector routes to GHD when this is below `global_agm_bound`
    /// and the width is at least 2.
    pub decomposition_cost: Option<u128>,
}

/// Analysis of a sidecar join plan before choosing a production join kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingSidecarAnalysis {
    /// Counters collected while opening the planned sidecar factors.
    pub stats: BindingSidecarStats,
    /// Root-level variable domains visible before any binding is chosen.
    pub variables: Box<[BindingVariableDomainStats]>,
    /// Heuristic variable order suitable for the current Generic Join oracle
    /// and for trie-backed LFTJ-style execution.
    pub suggested_variable_order: Box<[BindingVar]>,
    /// Candidate orders exposed for adaptive WCOJ experiments.
    pub variable_order_candidates: Box<[BindingVariableOrderCandidate]>,
}

/// Root-level domain statistics for one BindingSpace variable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindingVariableDomainStats {
    /// Query variable.
    pub variable: BindingVar,
    /// Number of factors containing this variable.
    pub factor_count: usize,
    /// Size of the intersection of positive domains from all containing
    /// factors, before any earlier variable is bound.
    pub root_domain_len: usize,
    /// Smallest positive domain contributed by one containing factor.
    pub min_factor_domain_len: usize,
    /// Largest positive domain contributed by one containing factor.
    pub max_factor_domain_len: usize,
}

/// Source of a candidate variable order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingVariableOrderSource {
    /// Order requested by the caller or compiled sidecar plan.
    Planned,
    /// Order selected by the current root-domain heuristic.
    Suggested,
    /// Order sorted only by root-domain selectivity.
    RootDomainAscending,
}

/// Candidate attribute order for adaptive WCOJ-sidecar experiments.
///
/// This mirrors the ADOPT/HoneyComb lesson at the planning boundary: variable
/// order is a first-class choice, so the engine should preserve alternatives
/// and their cheap static scores before any runtime feedback loop exists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingVariableOrderCandidate {
    /// How this candidate was generated.
    pub source: BindingVariableOrderSource,
    /// Variables in the order consumed by Generic Join/LFTJ-style kernels.
    pub variable_order: Box<[BindingVar]>,
    /// Sum of root-domain fanout weighted by depth. Lower is a cheaper static
    /// guess; runtime feedback should supersede it once available.
    pub estimated_depth_weighted_domain_work: usize,
    /// Number of non-root positions that connect to an already-bound variable
    /// through some relation factor.
    pub connected_prefix_steps: usize,
}

/// HoneyComb-style shared-memory partition-share diagnostic.
///
/// This does not execute a parallel join. It only records how a future
/// partitioned WCOJ executor would split work across all variables instead of
/// only the top loop.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingParallelSharePlan {
    /// Requested upper bound on task-grid cells.
    pub target_tasks: usize,
    /// Product of the selected variable shares.
    pub realized_tasks: usize,
    /// Variable order used by the share planner.
    pub variable_order: Box<[BindingVar]>,
    /// Per-variable shares in `variable_order`.
    pub variables: Box<[BindingVariableShare]>,
}

/// One variable's partition share in a [`BindingParallelSharePlan`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindingVariableShare {
    /// Query variable.
    pub variable: BindingVar,
    /// Number of hash/range buckets assigned to this variable.
    pub share: usize,
    /// Root-domain cardinality used by the diagnostic share heuristic.
    pub root_domain_len: usize,
    /// Number of factors containing this variable.
    pub factor_count: usize,
}

/// Explain-only adaptive WCOJ episode plan.
///
/// ADOPT's useful execution contract is that different attribute orders can be
/// tried in bounded episodes while a task manager records non-overlapping
/// hypercubes of the attribute domain. This structure captures that contract
/// without changing live query execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingAdaptiveEpisodePlan {
    /// Requested upper bound on task-grid cells.
    pub target_tasks: usize,
    /// Fixed per-order episode budget used for the static reward estimate.
    pub episode_budget_steps: usize,
    /// Non-overlapping task boxes over the selected share grid.
    pub task_boxes: Box<[BindingAdaptiveTaskBox]>,
    /// Per-candidate order estimates for an ADOPT-style exploration loop.
    pub order_estimates: Box<[BindingAdaptiveOrderEstimate]>,
}

/// One non-overlapping task box in the adaptive episode grid.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingAdaptiveTaskBox {
    /// Stable task id in lexicographic share-grid order.
    pub task_id: usize,
    /// Half-open bucket ranges, one per partitioned variable.
    pub ranges: Box<[BindingAdaptiveTaskRange]>,
}

/// One variable's bucket interval inside a task box.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindingAdaptiveTaskRange {
    /// Query variable.
    pub variable: BindingVar,
    /// Inclusive bucket start.
    pub start_bucket: usize,
    /// Exclusive bucket end.
    pub end_bucket: usize,
    /// Total buckets for this variable.
    pub share: usize,
}

/// Static reward estimate for one candidate variable order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingAdaptiveOrderEstimate {
    /// Candidate source.
    pub source: BindingVariableOrderSource,
    /// Candidate variable order.
    pub variable_order: Box<[BindingVar]>,
    /// Static work denominator inherited from the candidate order score.
    pub estimated_depth_weighted_domain_work: usize,
    /// Estimated domain cells covered by one bounded episode.
    pub estimated_covered_domain_cells: usize,
    /// Reward numerator. This is domain coverage, not output rows, so an order
    /// can receive credit for quickly discarding empty search space.
    pub reward_numerator: usize,
    /// Reward denominator. This is the total root-domain volume, making the
    /// rational reward a normalized covered-volume estimate in `[0, 1]`.
    pub reward_denominator: usize,
}

impl BindingSidecarAnalysis {
    /// Computes a HoneyComb-style task-grid share plan for the current
    /// suggested variable order.
    ///
    /// The plan is bounded by `target_tasks`; the realized task-grid product
    /// may be smaller when the target cannot be reached by integer share
    /// increments. This is explain-only and does not affect live MM2 matching.
    pub fn parallel_share_plan(&self, target_tasks: usize) -> BindingParallelSharePlan {
        let target_tasks = target_tasks.max(1);
        let stats_by_variable = self
            .variables
            .iter()
            .map(|stats| (stats.variable, *stats))
            .collect::<BTreeMap<_, _>>();
        let mut variables = self
            .suggested_variable_order
            .iter()
            .filter_map(|variable| stats_by_variable.get(variable).copied())
            .map(|stats| BindingVariableShare {
                variable: stats.variable,
                share: 1,
                root_domain_len: stats.root_domain_len,
                factor_count: stats.factor_count,
            })
            .collect::<Vec<_>>();
        let mut realized_tasks = 1usize;

        while realized_tasks < target_tasks && !variables.is_empty() {
            let Some(index) = variables
                .iter()
                .enumerate()
                .filter(|(_, variable)| {
                    let next = realized_tasks / variable.share * (variable.share + 1);
                    next <= target_tasks
                })
                .max_by_key(|(_, variable)| {
                    let residual_domain =
                        variable.root_domain_len.saturating_add(variable.share - 1)
                            / variable.share;
                    (
                        residual_domain.saturating_mul(variable.factor_count.max(1)),
                        variable.root_domain_len,
                        variable.factor_count,
                    )
                })
                .map(|(index, _)| index)
            else {
                break;
            };

            realized_tasks = realized_tasks / variables[index].share * (variables[index].share + 1);
            variables[index].share += 1;
        }

        BindingParallelSharePlan {
            target_tasks,
            realized_tasks,
            variable_order: self.suggested_variable_order.clone(),
            variables: variables.into_boxed_slice(),
        }
    }

    /// Builds an ADOPT-style explain plan for adaptive WCOJ-style execution.
    ///
    /// The task boxes are non-overlapping cells of the share grid. The reward
    /// estimate uses domain cells covered per bounded episode rather than
    /// output rows, matching ADOPT's observation that failed/empty regions are
    /// still useful progress.
    pub fn adaptive_episode_plan(
        &self,
        target_tasks: usize,
        episode_budget_steps: usize,
    ) -> BindingAdaptiveEpisodePlan {
        let episode_budget_steps = episode_budget_steps.max(1);
        let share_plan = self.parallel_share_plan(target_tasks);
        let task_boxes = adaptive_task_boxes(&share_plan);
        let domain_cells = root_domain_volume(&self.variables).max(1);
        let order_estimates = self
            .variable_order_candidates
            .iter()
            .map(|candidate| {
                let work = candidate.estimated_depth_weighted_domain_work.max(1);
                let covered = domain_cells
                    .saturating_mul(episode_budget_steps)
                    .saturating_div(work)
                    .clamp(usize::from(domain_cells > 0), domain_cells);

                BindingAdaptiveOrderEstimate {
                    source: candidate.source,
                    variable_order: candidate.variable_order.clone(),
                    estimated_depth_weighted_domain_work: work,
                    estimated_covered_domain_cells: covered,
                    reward_numerator: covered,
                    reward_denominator: domain_cells,
                }
            })
            .collect();

        BindingAdaptiveEpisodePlan {
            target_tasks: share_plan.target_tasks,
            episode_budget_steps,
            task_boxes,
            order_estimates,
        }
    }
}

/// Coarse execution counters for sidecar plans.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BindingSidecarStats {
    /// Number of physical factor accesses opened by the plan.
    pub factors: usize,
    /// Complete fact roots inspected while constructing arrangements.
    pub facts_scanned: usize,
    /// Rows retained by constructed arrangements.
    pub arrangement_rows: usize,
    /// Relation/arity facts skipped by arrangements because they contain variables.
    pub arrangement_schematic_rows_skipped: usize,
    /// Expression-trie indexes built while opening this plan.
    pub expression_trie_builds: usize,
    /// Complete facts indexed into expression tries.
    pub expression_trie_facts_indexed: usize,
    /// Candidate facts returned by expression-trie prefix retrieval.
    pub expression_trie_candidates: usize,
    /// Application atoms checked by exact relationalized pattern filters.
    pub pattern_app_atoms_checked: usize,
    /// Exact pattern matches before BindingSpace projection.
    pub pattern_matches: usize,
    /// Positive BindingSpace rows produced by factor projections.
    pub projected_rows: usize,
    /// Positive rows emitted by the final join.
    pub output_rows: usize,
    /// Positive rows removed by the Yannakakis full reducer before the final
    /// join. Zero unless the acyclic-Yannakakis kernel was selected; nonzero
    /// only when the GYO join tree found dangling tuples to prune.
    pub semijoin_removed_rows: usize,
}

const MIN_TRIE_SIDE_INPUT_ROWS: usize = 8;

/// Largest hypertree-decomposition width the selector will consider. Width 2
/// covers triangles and 4-cliques; width 3 covers 5-cliques. Beyond this a global
/// worst-case-optimal join is the better default.
const GHD_MAX_WIDTH: usize = 3;

/// Error from sidecar plan execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingSidecarPlanError {
    /// A physical arrangement could not be built or projected.
    Arrangement(ArrangementError),
    /// Expression-trie candidate retrieval or exact filtering failed.
    ExpressionTrie(ExpressionTrieError),
    /// A BindingSpace relation operation failed.
    Binding(BindingRelationError),
    /// A generated trie trace failed replay-shape validation.
    TraceShape(TrieJoinTraceShapeError),
    /// Pattern projection schema and user-slot list have different lengths.
    PatternProjectionArityMismatch { schema_len: usize, slots_len: usize },
    /// Pattern projection requested a slot outside MORK's six-bit domain.
    InvalidPatternSlot { slot: u8 },
    /// Exact pattern matching did not bind a projected user slot.
    MissingPatternBinding { slot: u8 },
}

impl BindingSidecarPlan {
    /// Creates a sidecar plan from factor accesses and a variable order.
    pub fn new(
        factors: impl Into<Box<[BindingAccessPlan]>>,
        variable_order: impl Into<Box<[BindingVar]>>,
    ) -> Self {
        Self {
            factors: factors.into(),
            variable_order: variable_order.into(),
        }
    }

    /// Planned factor accesses.
    pub fn factors(&self) -> &[BindingAccessPlan] {
        &self.factors
    }

    /// Variable order consumed by Generic Join and trie-backed LFTJ kernels.
    pub fn variable_order(&self) -> &[BindingVar] {
        &self.variable_order
    }

    /// Whether the body's variable hypergraph is alpha-acyclic (GYO ear removal
    /// over the factor schemas, data-independent). The worst-case-optimal join
    /// beats the ProductZipper only on cyclic bodies, where a cyclic constraint
    /// cannot be checked until every variable is bound, so the ProductZipper's
    /// fixed-order walk enumerates the intermediate. On an acyclic body the
    /// ProductZipper's trie walk is already output-sensitive (it prunes dangling
    /// like a semijoin reducer), so the sidecar has no asymptotic advantage and
    /// only pays the interning cost. The live flip uses this to gate.
    pub fn body_is_acyclic(&self) -> bool {
        let relations = self
            .factors
            .iter()
            .map(|factor| {
                let schema = match factor {
                    BindingAccessPlan::Arrangement { projection, .. } => projection.schema.clone(),
                    BindingAccessPlan::Pattern { projection, .. } => projection.schema.clone(),
                };
                BindingRelation::new(schema)
            })
            .collect::<Vec<_>>();
        gyo_join_tree(&relations).acyclic
    }

    /// Executes the plan against one term snapshot.
    pub fn execute(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarResult, BindingSidecarPlanError> {
        let (relations, mut stats) = self.open_relations(sidecar)?;

        let relation = generic_join(&relations, &self.variable_order)
            .map_err(BindingSidecarPlanError::Binding)?;
        stats.output_rows = relation.positive_rows().count();

        Ok(BindingSidecarResult { relation, stats })
    }

    /// Executes the plan using the trie-backed variable-at-a-time join kernel.
    ///
    /// The opened factors are still derived sidecars over the term snapshot;
    /// this does not change the canonical PathMap/ACT pathspace semantics.
    pub fn execute_trie_join(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarTrieJoinResult, BindingSidecarPlanError> {
        let (relations, mut stats) = self.open_relations(sidecar)?;

        let joined = PreparedTrieJoin::prepare(&relations, &self.variable_order)
            .and_then(|prepared| prepared.execute())
            .map_err(BindingSidecarPlanError::Binding)?;
        stats.output_rows = joined.relation.positive_rows().count();

        Ok(BindingSidecarTrieJoinResult {
            relation: joined.relation,
            variable_order: self.variable_order.clone(),
            stats,
            trie_stats: joined.stats,
        })
    }

    /// Executes the plan with a root-domain heuristic variable order.
    ///
    /// This is still a sidecar experiment over opened `BindingRelation`s, not a
    /// replacement for the live ProductZipper matcher. It removes the need for
    /// tests and prototypes to hand-author the global LFTJ order when the plan
    /// already has exact root-domain evidence.
    pub fn execute_trie_join_suggested(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarTrieJoinResult, BindingSidecarPlanError> {
        let (relations, mut stats) = self.open_relations(sidecar)?;
        let variable_stats = variable_domain_stats(&relations);
        let variable_order = suggest_variable_order(&relations, &variable_stats);

        let joined = PreparedTrieJoin::prepare(&relations, &variable_order)
            .and_then(|prepared| prepared.execute())
            .map_err(BindingSidecarPlanError::Binding)?;
        stats.output_rows = joined.relation.positive_rows().count();

        Ok(BindingSidecarTrieJoinResult {
            relation: joined.relation,
            variable_order: variable_order.into_boxed_slice(),
            stats,
            trie_stats: joined.stats,
        })
    }

    /// Chooses the physical sidecar kernel from opened-factor statistics.
    ///
    /// This selection is deliberately conservative. It keeps the reference
    /// Generic Join for one-factor, disconnected, and tiny sidecar inputs, and
    /// only selects the trie-backed LFTJ-style kernel when shared-variable
    /// domain pruning can pay for query-specific index construction.
    pub fn choose_execution(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarExecutionChoice, BindingSidecarPlanError> {
        Ok(self.explain_execution(sidecar)?.choice)
    }

    /// Opens all sidecar factors once and chooses the selected physical kernel.
    ///
    /// Reuse the returned prepared state when a caller needs multiple reports
    /// for the same immutable term snapshot, such as explain + count + exists.
    pub fn prepare(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarPrepared, BindingSidecarPlanError> {
        let (relations, stats) = self.open_relations(sidecar)?;
        BindingSidecarPrepared::new(relations, &self.variable_order, stats)
    }

    /// Executes the sidecar plan through the selected physical kernel.
    ///
    /// This is still a derived sidecar path over the immutable term snapshot;
    /// it does not alter the canonical PathMap/ACT matching semantics.
    pub fn execute_selected(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarSelectedResult, BindingSidecarPlanError> {
        self.prepare(sidecar)?.execute_selected()
    }

    /// Counts selected sidecar matches through the selected physical aggregate
    /// path.
    ///
    /// Trie-backed selections use `trie_join_count` and do not allocate a flat
    /// output relation. Generic Join selections use the matching reference
    /// aggregate traversal instead of building a flat output relation.
    pub fn count_selected(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarCountReport, BindingSidecarPlanError> {
        self.prepare(sidecar)?.count_selected()
    }

    /// Checks whether selected sidecar matches exist through the selected
    /// physical aggregate path.
    ///
    /// Trie-backed selections use the early-stop `trie_join_exists` kernel.
    /// Generic Join selections use the matching early-stop reference traversal.
    pub fn exists_selected(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarExistsReport, BindingSidecarPlanError> {
        self.prepare(sidecar)?.exists_selected()
    }

    /// Explains the selected physical sidecar kernel without running the join.
    ///
    /// This is the BindingSidecar analogue of an optimizer explain plan: it
    /// opens the immutable snapshot factors to gather estimated/cardinality
    /// evidence, but leaves `output_rows` at zero because no result is emitted.
    pub fn explain_execution(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarExecutionReport, BindingSidecarPlanError> {
        Ok(self.prepare(sidecar)?.execution().clone())
    }

    /// Explains the selected execution and, when applicable, traces the
    /// trie-backed variable-domain traversal without materializing join rows.
    pub fn explain_selected_trie_trace(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarTrieTraceReport, BindingSidecarPlanError> {
        self.prepare(sidecar)?.explain_selected_trie_trace()
    }

    /// Explains and diffs selected trie traces across two immutable snapshots.
    ///
    /// The final join is still not executed. If either snapshot keeps the
    /// Generic Join kernel, `replay_diff` is `None` because there is no trie
    /// replay shape to compare.
    pub fn explain_selected_trie_trace_diff(
        &self,
        old_sidecar: &TermIdentitySidecar,
        new_sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarTrieTraceDiffReport, BindingSidecarPlanError> {
        let old = self.explain_selected_trie_trace(old_sidecar)?;
        let new = self.explain_selected_trie_trace(new_sidecar)?;

        let replay_diff = match (&old.trie_trace, &new.trie_trace) {
            (Some(old_trace), Some(new_trace)) => {
                let old_shape = old_trace
                    .replay_shape()
                    .map_err(BindingSidecarPlanError::TraceShape)?;
                let new_shape = new_trace
                    .replay_shape()
                    .map_err(BindingSidecarPlanError::TraceShape)?;
                Some(old_shape.diff(&new_shape))
            }
            _ => None,
        };

        Ok(BindingSidecarTrieTraceDiffReport {
            old,
            new,
            replay_diff,
        })
    }

    /// Explains the selected execution and, when trie-backed, returns the
    /// per-factor cursor contract that a PathMap/ReadZipper-backed kernel must
    /// reproduce.
    ///
    /// The final join is not executed. This is a bridge artifact between the
    /// sidecar LFTJ trace and a physical zipper-factor implementation.
    pub fn explain_selected_trie_cursor_contract(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarTrieCursorContractReport, BindingSidecarPlanError> {
        self.prepare(sidecar)?
            .explain_selected_trie_cursor_contract()
    }

    /// Opens all sidecar factors and reports root-domain statistics. This is a
    /// planning aid; it does not mutate the canonical PathMap/ACT storage.
    pub fn analyze(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<BindingSidecarAnalysis, BindingSidecarPlanError> {
        let (relations, stats) = self.open_relations(sidecar)?;
        Ok(analyze_relations(&relations, &self.variable_order, stats))
    }

    /// Explains how the current analysis would split work across variables for
    /// a parallel WCOJ executor.
    ///
    /// This opens sidecar factors for cardinality evidence but does not execute
    /// the final join. Use [`BindingSidecarPrepared::parallel_share_plan`] when
    /// the caller already prepared this immutable snapshot.
    pub fn explain_parallel_share_plan(
        &self,
        sidecar: &TermIdentitySidecar,
        target_tasks: usize,
    ) -> Result<BindingParallelSharePlan, BindingSidecarPlanError> {
        Ok(self.analyze(sidecar)?.parallel_share_plan(target_tasks))
    }

    /// Explains bounded adaptive WCOJ episodes for the current sidecar
    /// analysis.
    ///
    /// The returned task boxes and order estimates are diagnostic only: they do
    /// not switch the live execution kernel or mutate the canonical PathMap
    /// storage.
    pub fn explain_adaptive_episode_plan(
        &self,
        sidecar: &TermIdentitySidecar,
        target_tasks: usize,
        episode_budget_steps: usize,
    ) -> Result<BindingAdaptiveEpisodePlan, BindingSidecarPlanError> {
        Ok(self
            .analyze(sidecar)?
            .adaptive_episode_plan(target_tasks, episode_budget_steps))
    }

    fn open_relations(
        &self,
        sidecar: &TermIdentitySidecar,
    ) -> Result<(Vec<BindingRelation>, BindingSidecarStats), BindingSidecarPlanError> {
        let mut stats = BindingSidecarStats::default();
        let mut context = BindingOpenContext::default();
        let mut relations = Vec::with_capacity(self.factors.len());

        for factor in self.factors.iter() {
            stats.factors += 1;
            let relation = factor.open(sidecar, &mut stats, &mut context)?;
            stats.projected_rows += relation.positive_rows().count();
            relations.push(relation);
        }

        Ok((relations, stats))
    }
}

impl BindingSidecarPrepared {
    fn new(
        relations: Vec<BindingRelation>,
        planned_order: &[BindingVar],
        stats: BindingSidecarStats,
    ) -> Result<Self, BindingSidecarPlanError> {
        let execution = explain_execution_choice(&relations, planned_order, stats);
        let trie_join = match execution.choice.kernel {
            BindingSidecarExecutionKernel::GenericJoin
            | BindingSidecarExecutionKernel::AcyclicYannakakis
            | BindingSidecarExecutionKernel::GhdYannakakis => None,
            BindingSidecarExecutionKernel::TrieJoinSuggested => Some(
                PreparedTrieJoin::prepare(&relations, &execution.choice.variable_order)
                    .map_err(BindingSidecarPlanError::Binding)?,
            ),
        };
        let join_tree_edges = match execution.choice.kernel {
            BindingSidecarExecutionKernel::AcyclicYannakakis => {
                // Recompute the GYO tree (cheap for the small conjunctions MM2
                // admits) so the reducer has its edges; the selector used the
                // same construction to decide alpha-acyclicity.
                Some(gyo_join_tree(&relations).edges)
            }
            BindingSidecarExecutionKernel::GenericJoin
            | BindingSidecarExecutionKernel::TrieJoinSuggested
            | BindingSidecarExecutionKernel::GhdYannakakis => None,
        };
        let decomposition = match execution.choice.kernel {
            BindingSidecarExecutionKernel::GhdYannakakis => {
                // Recompute the decomposition (cheap for tiny conjunctions) so the
                // bag plan is available; the selector used it to route here.
                hypertree_decomposition(&relations, GHD_MAX_WIDTH)
            }
            BindingSidecarExecutionKernel::GenericJoin
            | BindingSidecarExecutionKernel::TrieJoinSuggested
            | BindingSidecarExecutionKernel::AcyclicYannakakis => None,
        };

        Ok(Self {
            relations: relations.into_boxed_slice(),
            execution,
            trie_join,
            join_tree_edges,
            decomposition,
        })
    }

    /// Opened factor relations for this prepared snapshot.
    pub fn relations(&self) -> &[BindingRelation] {
        &self.relations
    }

    /// Selected execution report derived from the opened factor relations.
    pub fn execution(&self) -> &BindingSidecarExecutionReport {
        &self.execution
    }

    /// Prepared trie indexes when the selected kernel is trie-backed.
    pub fn prepared_trie_join(&self) -> Option<&PreparedTrieJoin> {
        self.trie_join.as_ref()
    }

    fn require_prepared_trie_join(&self) -> &PreparedTrieJoin {
        self.trie_join
            .as_ref()
            .expect("trie-backed prepared sidecar should have prepared trie indexes")
    }

    /// Applies the Yannakakis full reducer over the prepared GYO join tree,
    /// returning dangling-tuple-free relations whose join output set equals the
    /// un-reduced join (Yannakakis, VLDB 1981). Only valid when the selected
    /// kernel is `AcyclicYannakakis`.
    fn reduced_relations(&self) -> Result<SemijoinReduction, BindingSidecarPlanError> {
        let edges = self
            .join_tree_edges
            .as_deref()
            .expect("acyclic-Yannakakis prepared sidecar should carry GYO join-tree edges");
        semijoin_reduce_presence(&self.relations, edges).map_err(BindingSidecarPlanError::Binding)
    }

    fn require_decomposition(&self) -> &HyperTreeDecomposition {
        self.decomposition
            .as_ref()
            .expect("ghd-Yannakakis prepared sidecar should carry a hypertree decomposition")
    }

    /// Computes a HoneyComb-style share plan from the already-opened prepared
    /// factor analysis.
    ///
    /// This is useful when a caller needs several explain/count/trace reports
    /// for the same snapshot without rebuilding arrangements or expression
    /// tries.
    pub fn parallel_share_plan(&self, target_tasks: usize) -> BindingParallelSharePlan {
        self.execution.analysis.parallel_share_plan(target_tasks)
    }

    /// Computes an ADOPT-style bounded episode plan from already-opened
    /// prepared factor analysis.
    pub fn adaptive_episode_plan(
        &self,
        target_tasks: usize,
        episode_budget_steps: usize,
    ) -> BindingAdaptiveEpisodePlan {
        self.execution
            .analysis
            .adaptive_episode_plan(target_tasks, episode_budget_steps)
    }

    /// Executes the prepared sidecar through its selected physical kernel.
    pub fn execute_selected(
        &self,
    ) -> Result<BindingSidecarSelectedResult, BindingSidecarPlanError> {
        let choice = self.execution.choice.clone();
        let mut stats = self.execution.analysis.stats;

        match choice.kernel {
            BindingSidecarExecutionKernel::GenericJoin => {
                let relation = generic_join(&self.relations, &choice.variable_order)
                    .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = relation.positive_rows().count();

                Ok(BindingSidecarSelectedResult {
                    relation,
                    choice,
                    stats,
                    trie_stats: None,
                })
            }
            BindingSidecarExecutionKernel::TrieJoinSuggested => {
                let joined = self
                    .require_prepared_trie_join()
                    .execute()
                    .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = joined.relation.positive_rows().count();

                Ok(BindingSidecarSelectedResult {
                    relation: joined.relation,
                    choice,
                    stats,
                    trie_stats: Some(joined.stats),
                })
            }
            BindingSidecarExecutionKernel::AcyclicYannakakis => {
                let reduced = self.reduced_relations()?;
                let relation = generic_join(&reduced.relations, &choice.variable_order)
                    .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = relation.positive_rows().count();
                stats.semijoin_removed_rows = reduced.removed_rows;

                Ok(BindingSidecarSelectedResult {
                    relation,
                    choice,
                    stats,
                    trie_stats: None,
                })
            }
            BindingSidecarExecutionKernel::GhdYannakakis => {
                let relation = ghd_join(
                    &self.relations,
                    self.require_decomposition(),
                    &choice.variable_order,
                )
                .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = relation.positive_rows().count();

                Ok(BindingSidecarSelectedResult {
                    relation,
                    choice,
                    stats,
                    trie_stats: None,
                })
            }
        }
    }

    /// Streams each output row of the selected join to `emit` as
    /// `(binding, weight)` without materializing the relation, when the selected
    /// kernel has a streaming variant (the trie join). Returns `None` for the
    /// materialising kernels (generic/Yannakakis/GHD) so the caller keeps the
    /// relation path. The plan-level entry for the v3 `Factorise` fused emit:
    /// the caller writes each template output directly from the binding.
    pub(crate) fn for_each_selected<E>(
        &self,
        emit: E,
    ) -> Option<Result<crate::binding_space::TrieJoinStats, BindingSidecarPlanError>>
    where
        E: FnMut(&crate::binding_space::BindingAssignment, i64),
    {
        match self.execution.choice.kernel {
            BindingSidecarExecutionKernel::TrieJoinSuggested => Some(
                self.require_prepared_trie_join()
                    .for_each(emit)
                    .map_err(BindingSidecarPlanError::Binding),
            ),
            _ => None,
        }
    }

    /// Counts prepared sidecar matches through the selected aggregate path.
    pub fn count_selected(&self) -> Result<BindingSidecarCountReport, BindingSidecarPlanError> {
        let execution = self.execution.clone();
        let mut stats = execution.analysis.stats;

        match execution.choice.kernel {
            BindingSidecarExecutionKernel::GenericJoin => {
                let count = generic_join_count(&self.relations, &execution.choice.variable_order)
                    .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = count.rows;

                Ok(BindingSidecarCountReport {
                    execution,
                    aggregate_kernel: BindingSidecarAggregateKernel::GenericJoinCount,
                    rows: count.rows,
                    stats,
                    trie_stats: None,
                    materialized_output: false,
                })
            }
            BindingSidecarExecutionKernel::TrieJoinSuggested => {
                let count = self
                    .require_prepared_trie_join()
                    .count()
                    .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = count.rows;

                Ok(BindingSidecarCountReport {
                    execution,
                    aggregate_kernel: BindingSidecarAggregateKernel::TrieJoinCount,
                    rows: count.rows,
                    stats,
                    trie_stats: Some(count.stats),
                    materialized_output: false,
                })
            }
            BindingSidecarExecutionKernel::AcyclicYannakakis => {
                let reduced = self.reduced_relations()?;
                let count =
                    generic_join_count(&reduced.relations, &execution.choice.variable_order)
                        .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = count.rows;
                stats.semijoin_removed_rows = reduced.removed_rows;

                Ok(BindingSidecarCountReport {
                    execution,
                    aggregate_kernel: BindingSidecarAggregateKernel::GenericJoinCount,
                    rows: count.rows,
                    stats,
                    trie_stats: None,
                    materialized_output: false,
                })
            }
            BindingSidecarExecutionKernel::GhdYannakakis => {
                let count = ghd_join_count(
                    &self.relations,
                    self.require_decomposition(),
                    &execution.choice.variable_order,
                )
                .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = count.rows;

                Ok(BindingSidecarCountReport {
                    execution,
                    aggregate_kernel: BindingSidecarAggregateKernel::GenericJoinCount,
                    rows: count.rows,
                    stats,
                    trie_stats: None,
                    materialized_output: false,
                })
            }
        }
    }

    /// Checks whether prepared sidecar matches exist through the selected
    /// aggregate path.
    pub fn exists_selected(&self) -> Result<BindingSidecarExistsReport, BindingSidecarPlanError> {
        let execution = self.execution.clone();
        let mut stats = execution.analysis.stats;

        match execution.choice.kernel {
            BindingSidecarExecutionKernel::GenericJoin => {
                let existence =
                    generic_join_exists(&self.relations, &execution.choice.variable_order)
                        .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = usize::from(existence.exists);

                Ok(BindingSidecarExistsReport {
                    execution,
                    aggregate_kernel: BindingSidecarAggregateKernel::GenericJoinExists,
                    exists: existence.exists,
                    stats,
                    trie_stats: None,
                    materialized_output: false,
                })
            }
            BindingSidecarExecutionKernel::TrieJoinSuggested => {
                let existence = self
                    .require_prepared_trie_join()
                    .exists()
                    .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = existence.stats.output_rows;

                Ok(BindingSidecarExistsReport {
                    execution,
                    aggregate_kernel: BindingSidecarAggregateKernel::TrieJoinExists,
                    exists: existence.exists,
                    stats,
                    trie_stats: Some(existence.stats),
                    materialized_output: false,
                })
            }
            BindingSidecarExecutionKernel::AcyclicYannakakis => {
                let reduced = self.reduced_relations()?;
                let existence =
                    generic_join_exists(&reduced.relations, &execution.choice.variable_order)
                        .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = usize::from(existence.exists);
                stats.semijoin_removed_rows = reduced.removed_rows;

                Ok(BindingSidecarExistsReport {
                    execution,
                    aggregate_kernel: BindingSidecarAggregateKernel::GenericJoinExists,
                    exists: existence.exists,
                    stats,
                    trie_stats: None,
                    materialized_output: false,
                })
            }
            BindingSidecarExecutionKernel::GhdYannakakis => {
                let existence = ghd_join_exists(
                    &self.relations,
                    self.require_decomposition(),
                    &execution.choice.variable_order,
                )
                .map_err(BindingSidecarPlanError::Binding)?;
                stats.output_rows = usize::from(existence.exists);

                Ok(BindingSidecarExistsReport {
                    execution,
                    aggregate_kernel: BindingSidecarAggregateKernel::GenericJoinExists,
                    exists: existence.exists,
                    stats,
                    trie_stats: None,
                    materialized_output: false,
                })
            }
        }
    }

    /// Explains and traces the prepared trie-backed traversal without
    /// materializing join rows.
    pub fn explain_selected_trie_trace(
        &self,
    ) -> Result<BindingSidecarTrieTraceReport, BindingSidecarPlanError> {
        let execution = self.execution.clone();
        let trie_trace = match execution.choice.kernel {
            BindingSidecarExecutionKernel::GenericJoin
            | BindingSidecarExecutionKernel::AcyclicYannakakis
            | BindingSidecarExecutionKernel::GhdYannakakis => None,
            BindingSidecarExecutionKernel::TrieJoinSuggested => Some(
                self.require_prepared_trie_join()
                    .trace()
                    .map_err(BindingSidecarPlanError::Binding)?,
            ),
        };

        Ok(BindingSidecarTrieTraceReport {
            execution,
            trie_trace,
        })
    }

    /// Explains the cursor contract required by a prepared trie-backed
    /// traversal.
    pub fn explain_selected_trie_cursor_contract(
        &self,
    ) -> Result<BindingSidecarTrieCursorContractReport, BindingSidecarPlanError> {
        let report = self.explain_selected_trie_trace()?;
        let cursor_contract = report
            .trie_trace
            .as_ref()
            .map(TrieJoinTrace::cursor_contract)
            .transpose()
            .map_err(BindingSidecarPlanError::TraceShape)?;

        Ok(BindingSidecarTrieCursorContractReport {
            execution: report.execution,
            cursor_contract,
        })
    }

    /// Reports the size-based cost comparison the selector weighs when choosing
    /// between a global join and the hypertree decomposition.
    pub fn explain_routing_cost(&self) -> BindingSidecarRoutingCost {
        let decomposition = hypertree_decomposition(&self.relations, GHD_MAX_WIDTH);
        BindingSidecarRoutingCost {
            global_agm_bound: agm_size_bound(&self.relations),
            decomposition_width: decomposition.as_ref().map(|tree| tree.width),
            decomposition_cost: decomposition
                .as_ref()
                .map(|tree| ghd_size_cost(&self.relations, tree)),
        }
    }
}

fn explain_execution_choice(
    relations: &[BindingRelation],
    planned_order: &[BindingVar],
    stats: BindingSidecarStats,
) -> BindingSidecarExecutionReport {
    let analysis = analyze_relations(relations, planned_order, stats);
    let choice = select_execution_choice(relations, planned_order, &analysis);

    BindingSidecarExecutionReport { analysis, choice }
}

fn select_execution_choice(
    relations: &[BindingRelation],
    planned_order: &[BindingVar],
    analysis: &BindingSidecarAnalysis,
) -> BindingSidecarExecutionChoice {
    let relation_count = relations.len();
    let shared_variables = analysis
        .variables
        .iter()
        .filter(|stats| stats.factor_count > 1)
        .count();
    let min_shared_root_domain_len = analysis
        .variables
        .iter()
        .filter(|stats| stats.factor_count > 1)
        .map(|stats| stats.root_domain_len)
        .min();

    let (kernel, reason, variable_order) = if relation_count <= 1 {
        (
            BindingSidecarExecutionKernel::GenericJoin,
            BindingSidecarExecutionReason::SingleFactor,
            planned_order.to_vec(),
        )
    } else if shared_variables == 0 {
        (
            BindingSidecarExecutionKernel::GenericJoin,
            BindingSidecarExecutionReason::NoSharedVariables,
            planned_order.to_vec(),
        )
    } else if analysis.stats.projected_rows <= MIN_TRIE_SIDE_INPUT_ROWS {
        (
            BindingSidecarExecutionKernel::GenericJoin,
            BindingSidecarExecutionReason::SmallExplicitInput,
            planned_order.to_vec(),
        )
    } else if relation_count >= 3 && gyo_join_tree(relations).acyclic {
        // Multiway alpha-acyclic body: a GYO join tree admits a Yannakakis full
        // reducer (Yannakakis, VLDB 1981) that prunes dangling tuples in linear
        // semijoin passes before the final join. The `&&` short-circuits, so GYO
        // only runs once the cheaper guards have been ruled out and the body is
        // wide enough (>= 3 factors) for a multiway reducer to matter; for two
        // relations there is no intermediate result to reduce. Cyclic multiway
        // cores fall through to the worst-case-optimal trie kernel below.
        (
            BindingSidecarExecutionKernel::AcyclicYannakakis,
            BindingSidecarExecutionReason::AcyclicJoinTree,
            selectivity_variable_order(relations),
        )
    } else if ghd_decomposition_helps(relations) {
        // Cyclic but mostly acyclic with a small bounded-width core: a hypertree
        // decomposition runs Yannakakis over bags, which beats a single global
        // worst-case-optimal join that would still pay for the acyclic part. A
        // tight cyclic core fails this check and stays on the trie kernel, which
        // is worst-case-optimal (N^rho* <= N^width). The routing heuristic is
        // conservative and will be replaced by the cardinality cost model.
        (
            BindingSidecarExecutionKernel::GhdYannakakis,
            BindingSidecarExecutionReason::BoundedWidthDecomposition,
            selectivity_variable_order(relations),
        )
    } else {
        (
            BindingSidecarExecutionKernel::TrieJoinSuggested,
            BindingSidecarExecutionReason::SharedVariablePruning,
            analysis.suggested_variable_order.to_vec(),
        )
    };

    BindingSidecarExecutionChoice {
        kernel,
        reason,
        variable_order: variable_order.into_boxed_slice(),
        projected_rows: analysis.stats.projected_rows,
        shared_variables,
        min_shared_root_domain_len,
    }
}

/// Decides whether a cyclic body should run as a hypertree decomposition instead
/// of a single global worst-case-optimal join.
///
/// It routes to the decomposition when the widest bag's materialization cost is
/// strictly below the AGM size bound of a single global worst-case-optimal join.
/// Both costs use measured relation sizes: `ghd_size_cost` is the largest bag's
/// size product, `agm_size_bound` is the smallest size product over edge covers.
/// A bare triangle's widest bag equals its AGM bound, so it stays on the trie
/// kernel; a body that is mostly a long acyclic structure has a high cover product
/// but a small core bag, so it routes here. Returns false when no decomposition
/// within `GHD_MAX_WIDTH` exists.
///
/// Both costs are integral (size) products, an upper bound on the fractional AGM
/// cost, so this is a proxy that improves on the structural heuristic by using
/// data sizes.
fn ghd_decomposition_helps(relations: &[BindingRelation]) -> bool {
    let Some(decomposition) = hypertree_decomposition(relations, GHD_MAX_WIDTH) else {
        return false;
    };
    // Only a genuinely cyclic body benefits: an acyclic body has width 1 and stays
    // on its existing kernel (the trie join, or the acyclic Yannakakis reducer for
    // wider bodies).
    decomposition.width >= 2 && ghd_size_cost(relations, &decomposition) < agm_size_bound(relations)
}

impl BindingAccessPlan {
    /// Creates an arrangement-backed factor with a validated binding projection.
    pub fn arrangement(
        descriptor: ArrangementDescriptor,
        schema: impl Into<Box<[BindingVar]>>,
        argument_positions: impl Into<Box<[u8]>>,
    ) -> Result<Self, BindingSidecarPlanError> {
        let projection =
            ArrangementProjection::new(descriptor.argument_count, schema, argument_positions)
                .map_err(BindingSidecarPlanError::Arrangement)?;
        Ok(Self::Arrangement {
            descriptor,
            projection,
        })
    }

    fn open(
        &self,
        sidecar: &TermIdentitySidecar,
        stats: &mut BindingSidecarStats,
        context: &mut BindingOpenContext,
    ) -> Result<BindingRelation, BindingSidecarPlanError> {
        match self {
            BindingAccessPlan::Arrangement {
                descriptor,
                projection,
            } => {
                let arrangement = context.arrangement(sidecar, descriptor, stats)?;
                arrangement
                    .project_bindings(sidecar, projection)
                    .map_err(BindingSidecarPlanError::Arrangement)
            }
            BindingAccessPlan::Pattern {
                pattern,
                projection,
            } => {
                let index = context.expression_trie(sidecar, stats)?;
                let matches = index
                    .match_pattern(sidecar, *pattern)
                    .map_err(BindingSidecarPlanError::ExpressionTrie)?;
                stats.expression_trie_candidates += matches.candidates.facts.len();
                stats.pattern_app_atoms_checked += matches.exact.stats.app_atoms_checked;
                stats.pattern_matches += matches.exact.stats.matches;

                let mut relation = BindingRelation::new(projection.schema.clone());
                for row in matches.exact.rows {
                    let binding_row = pattern_binding_row(&row, &projection.user_slots)?;
                    relation
                        .add(binding_row, 1)
                        .map_err(BindingSidecarPlanError::Binding)?;
                }
                Ok(relation)
            }
        }
    }
}

fn pattern_binding_row(
    row: &PatternRelationRow,
    user_slots: &[u8],
) -> Result<BindingRow, BindingSidecarPlanError> {
    match user_slots {
        [] => Ok(Box::from([])),
        [first] => Ok(Box::from([required_pattern_binding(row, *first)?])),
        [first, second] => Ok(Box::from([
            required_pattern_binding(row, *first)?,
            required_pattern_binding(row, *second)?,
        ])),
        [first, second, third] => Ok(Box::from([
            required_pattern_binding(row, *first)?,
            required_pattern_binding(row, *second)?,
            required_pattern_binding(row, *third)?,
        ])),
        [first, second, third, fourth] => Ok(Box::from([
            required_pattern_binding(row, *first)?,
            required_pattern_binding(row, *second)?,
            required_pattern_binding(row, *third)?,
            required_pattern_binding(row, *fourth)?,
        ])),
        _ => user_slots
            .iter()
            .map(|&slot| required_pattern_binding(row, slot))
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice),
    }
}

fn required_pattern_binding(
    row: &PatternRelationRow,
    slot: u8,
) -> Result<TermId, BindingSidecarPlanError> {
    row.user_binding(slot)
        .ok_or(BindingSidecarPlanError::MissingPatternBinding { slot })
}

#[derive(Default)]
struct BindingOpenContext {
    expression_trie: Option<ExpressionTrieIndex>,
    arrangements: BTreeMap<ArrangementDescriptor, ArrangementIndex>,
}

impl BindingOpenContext {
    fn arrangement<'a>(
        &'a mut self,
        sidecar: &TermIdentitySidecar,
        descriptor: &ArrangementDescriptor,
        stats: &mut BindingSidecarStats,
    ) -> Result<&'a ArrangementIndex, BindingSidecarPlanError> {
        match self.arrangements.entry(descriptor.clone()) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let arrangement = ArrangementIndex::build(sidecar, descriptor.clone())
                    .map_err(BindingSidecarPlanError::Arrangement)?;
                let arrangement_stats = arrangement.stats();
                stats.facts_scanned += arrangement_stats.facts_scanned;
                stats.arrangement_rows += arrangement_stats.rows;
                stats.arrangement_schematic_rows_skipped +=
                    arrangement_stats.schematic_rows_skipped;
                Ok(entry.insert(arrangement))
            }
        }
    }

    fn expression_trie<'a>(
        &'a mut self,
        sidecar: &TermIdentitySidecar,
        stats: &mut BindingSidecarStats,
    ) -> Result<&'a ExpressionTrieIndex, BindingSidecarPlanError> {
        if self.expression_trie.is_none() {
            let index = ExpressionTrieIndex::build(sidecar)
                .map_err(BindingSidecarPlanError::ExpressionTrie)?;
            stats.expression_trie_builds += 1;
            stats.expression_trie_facts_indexed += index.stats().facts_indexed;
            self.expression_trie = Some(index);
        }

        Ok(self
            .expression_trie
            .as_ref()
            .expect("expression trie was initialized above"))
    }
}

fn analyze_relations(
    relations: &[BindingRelation],
    planned_order: &[BindingVar],
    stats: BindingSidecarStats,
) -> BindingSidecarAnalysis {
    let variable_stats = variable_domain_stats(relations);
    let factor_schemas = factor_schemas(relations);
    let suggested_variable_order = suggest_variable_order(relations, &variable_stats);
    let variable_order_candidates = variable_order_candidates(
        planned_order,
        &suggested_variable_order,
        &variable_stats,
        &factor_schemas,
    );

    BindingSidecarAnalysis {
        stats,
        variables: variable_stats.into_boxed_slice(),
        suggested_variable_order: suggested_variable_order.into_boxed_slice(),
        variable_order_candidates: variable_order_candidates.into_boxed_slice(),
    }
}

fn variable_domain_stats(relations: &[BindingRelation]) -> Vec<BindingVariableDomainStats> {
    let mut domains_by_variable = BTreeMap::<BindingVar, Vec<Vec<TermId>>>::new();

    for relation in relations {
        for (index, &variable) in relation.schema().iter().enumerate() {
            let mut domain = relation
                .positive_rows()
                .map(|row| row[index])
                .collect::<Vec<_>>();
            domain.sort_unstable();
            domain.dedup();
            domains_by_variable
                .entry(variable)
                .or_default()
                .push(domain);
        }
    }

    domains_by_variable
        .into_iter()
        .map(|(variable, domains)| {
            let root_domain_len = intersect_domain_len(&domains);
            let min_factor_domain_len = domains.iter().map(Vec::len).min().unwrap_or(0);
            let max_factor_domain_len = domains.iter().map(Vec::len).max().unwrap_or(0);

            BindingVariableDomainStats {
                variable,
                factor_count: domains.len(),
                root_domain_len,
                min_factor_domain_len,
                max_factor_domain_len,
            }
        })
        .collect()
}

fn intersect_domain_len(domains: &[Vec<TermId>]) -> usize {
    let Some((smallest_index, smallest)) = domains
        .iter()
        .enumerate()
        .min_by_key(|(_, domain)| domain.len())
    else {
        return 0;
    };

    smallest
        .iter()
        .filter(|&&value| {
            domains.iter().enumerate().all(|(index, domain)| {
                index == smallest_index || domain.binary_search(&value).is_ok()
            })
        })
        .count()
}

fn suggest_variable_order(
    relations: &[BindingRelation],
    stats: &[BindingVariableDomainStats],
) -> Vec<BindingVar> {
    let stats_by_variable = stats
        .iter()
        .map(|stats| (stats.variable, *stats))
        .collect::<BTreeMap<_, _>>();
    let factor_schemas = relations
        .iter()
        .map(|relation| relation.schema().iter().copied().collect::<BTreeSet<_>>())
        .collect::<Vec<_>>();
    let mut remaining = stats.iter().map(|stats| stats.variable).collect::<Vec<_>>();
    let mut order = Vec::with_capacity(remaining.len());

    while !remaining.is_empty() {
        let has_connected = !order.is_empty()
            && remaining
                .iter()
                .any(|&variable| variable_connects_to_bound(variable, &order, &factor_schemas));
        let has_non_lonely = remaining
            .iter()
            .any(|variable| stats_by_variable[variable].factor_count > 1);

        let next_index = (0..remaining.len())
            .min_by(|&left_index, &right_index| {
                compare_suggested_variable(
                    remaining[left_index],
                    remaining[right_index],
                    &order,
                    &factor_schemas,
                    &stats_by_variable,
                    has_connected,
                    has_non_lonely,
                )
            })
            .expect("non-empty remaining variables have a next candidate");

        order.push(remaining.swap_remove(next_index));
    }

    order
}

fn compare_suggested_variable(
    left: BindingVar,
    right: BindingVar,
    bound_order: &[BindingVar],
    factor_schemas: &[BTreeSet<BindingVar>],
    stats_by_variable: &BTreeMap<BindingVar, BindingVariableDomainStats>,
    has_connected: bool,
    has_non_lonely: bool,
) -> Ordering {
    let left_connected = variable_connects_to_bound(left, bound_order, factor_schemas);
    let right_connected = variable_connects_to_bound(right, bound_order, factor_schemas);
    if has_connected && left_connected != right_connected {
        return right_connected.cmp(&left_connected);
    }

    let left_stats = stats_by_variable[&left];
    let right_stats = stats_by_variable[&right];
    let left_lonely = left_stats.factor_count == 1;
    let right_lonely = right_stats.factor_count == 1;
    if has_non_lonely && left_lonely != right_lonely {
        return left_lonely.cmp(&right_lonely);
    }

    left_stats
        .root_domain_len
        .cmp(&right_stats.root_domain_len)
        .then_with(|| right_stats.factor_count.cmp(&left_stats.factor_count))
        .then_with(|| left.cmp(&right))
}

fn variable_order_candidates(
    planned_order: &[BindingVar],
    suggested_order: &[BindingVar],
    stats: &[BindingVariableDomainStats],
    factor_schemas: &[BTreeSet<BindingVar>],
) -> Vec<BindingVariableOrderCandidate> {
    let mut candidates = Vec::new();
    push_variable_order_candidate(
        &mut candidates,
        BindingVariableOrderSource::Planned,
        planned_order.to_vec(),
        stats,
        factor_schemas,
    );
    push_variable_order_candidate(
        &mut candidates,
        BindingVariableOrderSource::Suggested,
        suggested_order.to_vec(),
        stats,
        factor_schemas,
    );

    let stats_by_variable = stats
        .iter()
        .map(|stats| (stats.variable, *stats))
        .collect::<BTreeMap<_, _>>();
    let mut root_domain_order = stats.iter().map(|stats| stats.variable).collect::<Vec<_>>();
    root_domain_order.sort_by(|left, right| {
        stats_by_variable[left]
            .root_domain_len
            .cmp(&stats_by_variable[right].root_domain_len)
            .then_with(|| right.cmp(left))
    });
    push_variable_order_candidate(
        &mut candidates,
        BindingVariableOrderSource::RootDomainAscending,
        root_domain_order,
        stats,
        factor_schemas,
    );

    candidates
}

fn push_variable_order_candidate(
    candidates: &mut Vec<BindingVariableOrderCandidate>,
    source: BindingVariableOrderSource,
    variable_order: Vec<BindingVar>,
    stats: &[BindingVariableDomainStats],
    factor_schemas: &[BTreeSet<BindingVar>],
) {
    if variable_order.is_empty()
        || candidates
            .iter()
            .any(|candidate| candidate.variable_order.as_ref() == variable_order.as_slice())
    {
        return;
    }

    candidates.push(BindingVariableOrderCandidate {
        source,
        estimated_depth_weighted_domain_work: depth_weighted_domain_work(&variable_order, stats),
        connected_prefix_steps: connected_prefix_steps(&variable_order, factor_schemas),
        variable_order: variable_order.into_boxed_slice(),
    });
}

fn depth_weighted_domain_work(
    variable_order: &[BindingVar],
    stats: &[BindingVariableDomainStats],
) -> usize {
    let stats_by_variable = stats
        .iter()
        .map(|stats| (stats.variable, *stats))
        .collect::<BTreeMap<_, _>>();
    variable_order
        .iter()
        .enumerate()
        .filter_map(|(index, variable)| {
            stats_by_variable.get(variable).map(|stats| {
                (index + 1)
                    .saturating_mul(stats.root_domain_len.max(1))
                    .saturating_mul(stats.factor_count.max(1))
            })
        })
        .sum()
}

fn connected_prefix_steps(
    variable_order: &[BindingVar],
    factor_schemas: &[BTreeSet<BindingVar>],
) -> usize {
    variable_order
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(index, variable)| {
            variable_connects_to_bound(**variable, &variable_order[..*index], factor_schemas)
        })
        .count()
}

fn factor_schemas(relations: &[BindingRelation]) -> Vec<BTreeSet<BindingVar>> {
    relations
        .iter()
        .map(|relation| relation.schema().iter().copied().collect())
        .collect()
}

fn variable_connects_to_bound(
    variable: BindingVar,
    bound: &[BindingVar],
    factor_schemas: &[BTreeSet<BindingVar>],
) -> bool {
    factor_schemas.iter().any(|schema| {
        schema.contains(&variable)
            && bound
                .iter()
                .any(|bound_variable| schema.contains(bound_variable))
    })
}

fn root_domain_volume(stats: &[BindingVariableDomainStats]) -> usize {
    stats
        .iter()
        .map(|stats| stats.root_domain_len.max(1))
        .fold(1usize, usize::saturating_mul)
}

fn adaptive_task_boxes(share_plan: &BindingParallelSharePlan) -> Box<[BindingAdaptiveTaskBox]> {
    let mut boxes = Vec::with_capacity(share_plan.realized_tasks);
    let mut current = Vec::with_capacity(share_plan.variables.len());
    push_adaptive_task_boxes(&share_plan.variables, 0, &mut current, &mut boxes);
    boxes.into_boxed_slice()
}

fn push_adaptive_task_boxes(
    variables: &[BindingVariableShare],
    index: usize,
    current: &mut Vec<BindingAdaptiveTaskRange>,
    boxes: &mut Vec<BindingAdaptiveTaskBox>,
) {
    if index == variables.len() {
        boxes.push(BindingAdaptiveTaskBox {
            task_id: boxes.len(),
            ranges: current.clone().into_boxed_slice(),
        });
        return;
    }

    let variable = variables[index];
    for bucket in 0..variable.share {
        current.push(BindingAdaptiveTaskRange {
            variable: variable.variable,
            start_bucket: bucket,
            end_bucket: bucket + 1,
            share: variable.share,
        });
        push_adaptive_task_boxes(variables, index + 1, current, boxes);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::Space;
    use crate::term_identity::TermIdentitySidecar;

    const TRANSITIVE_EDGE_FACTS: &[u8] = br#"
(edge Alice Bob)
(edge Bob Carol)
(edge Alice Dana)
(edge Dana Carol)
(edge Carol Erin)
(edge X Y)
"#;
    const TRIANGLE_EDGE_FACTS: &[u8] = br#"
(edge A B)
(edge B C)
(edge C A)
(edge C D)
(edge D E)
(edge E F)
"#;
    const SCHEMATIC_EDGE_FACTS: &[u8] = br#"
(edge Alice Bob)
(edge Carol Bob)
(edge $x Bob)
(edge Alice $y)
(node Bob)
"#;
    const THREE_EDGE_FACTS: &[u8] = br#"
(edge Alice Bob)
(edge Bob Carol)
(edge Alice Dana)
"#;
    const TWO_EDGE_FACTS: &[u8] = br#"
(edge Alice Bob)
(edge Bob Carol)
"#;
    const REPEATED_EDGE_FACTS: &[u8] = br#"
(edge Alice (f Alice))
(edge Alice (f Bob))
(edge Bob (f Bob))
(edge Carol (g Carol))
(edge Dave (f Eve))
(node Alice)
(tag Bob)
"#;
    const EDGE_COLOR_FACTS: &[u8] = br#"
(edge Alice (f Alice))
(edge Bob (f Bob))
(edge Carol (f Carol))
(edge Dave (f Eve))
(color Alice red)
(color Bob blue)
(color Carol red)
(color Eve red)
"#;
    const TRANSITIVE_PRODUCT_PATTERN: &str = "[3] , [3] edge $ $ [3] edge _2 $";
    const REPEATED_EDGE_PATTERN: &str = "[3] edge $ [2] f _1";
    const REPEATED_EDGE_PRODUCT_PATTERN: &str = "[2] , [3] edge $ [2] f _1";
    const EDGE_COLOR_PRODUCT_PATTERN: &str = "[3] , [3] edge $ [2] f _1 [3] color _1 $";

    fn encoded_expr(space: &mut Space, expr: &'static str) -> Vec<u8> {
        let expr = crate::expr!(space, expr);
        unsafe { expr.span().as_ref().unwrap() }.to_vec()
    }

    fn sidecar_fixture(facts: &[u8]) -> (Space, TermIdentitySidecar) {
        let mut space = Space::new();
        space.add_all_sexpr(facts).unwrap();

        let mut sidecar = TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&space.btm).unwrap();

        (space, sidecar)
    }

    fn sidecar_fixture_with_patterns(
        facts: &[u8],
        patterns: &[&'static str],
    ) -> (Space, TermIdentitySidecar, Vec<TermId>) {
        let mut space = Space::new();
        space.add_all_sexpr(facts).unwrap();

        let mut sidecar = TermIdentitySidecar::new();
        let pattern_ids = patterns
            .iter()
            .map(|pattern| {
                sidecar
                    .insert_term(&encoded_expr(&mut space, pattern))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        sidecar.extend_from_pathmap(&space.btm).unwrap();

        (space, sidecar, pattern_ids)
    }

    fn domain(values: &[u64]) -> Vec<TermId> {
        let mut domain = values.iter().copied().map(TermId).collect::<Vec<_>>();
        domain.sort_unstable();
        domain.dedup();
        domain
    }

    fn sample_pattern_row() -> PatternRelationRow {
        PatternRelationRow {
            root: TermId(99),
            user_bindings: vec![
                (0, TermId(10)),
                (2, TermId(20)),
                (4, TermId(40)),
                (6, TermId(60)),
                (8, TermId(80)),
            ]
            .into_boxed_slice(),
        }
    }

    #[test]
    fn pattern_binding_row_preserves_small_and_wide_orders() {
        let row = sample_pattern_row();

        assert_eq!(pattern_binding_row(&row, &[]).unwrap().as_ref(), &[]);
        assert_eq!(
            pattern_binding_row(&row, &[4]).unwrap().as_ref(),
            &[TermId(40)]
        );
        assert_eq!(
            pattern_binding_row(&row, &[4, 0]).unwrap().as_ref(),
            &[TermId(40), TermId(10)]
        );
        assert_eq!(
            pattern_binding_row(&row, &[4, 0, 8, 2]).unwrap().as_ref(),
            &[TermId(40), TermId(10), TermId(80), TermId(20)]
        );
        assert_eq!(
            pattern_binding_row(&row, &[8, 6, 4, 2, 0])
                .unwrap()
                .as_ref(),
            &[TermId(80), TermId(60), TermId(40), TermId(20), TermId(10)]
        );
    }

    #[test]
    fn pattern_binding_row_reports_missing_slot() {
        let row = sample_pattern_row();

        assert_eq!(
            pattern_binding_row(&row, &[4, 5]).unwrap_err(),
            BindingSidecarPlanError::MissingPatternBinding { slot: 5 }
        );
    }

    #[test]
    fn intersect_domain_len_counts_members_from_smallest_domain() {
        let domains = vec![
            domain(&[1, 2, 3, 4]),
            domain(&[4, 2, 4]),
            domain(&[0, 2, 4, 8]),
        ];

        assert_eq!(intersect_domain_len(&domains), 2);
        assert_eq!(intersect_domain_len(&[domain(&[1, 2]), domain(&[3, 4])]), 0);
        assert_eq!(intersect_domain_len(&[]), 0);
    }

    fn relation_descriptor(
        space: &mut Space,
        sidecar: &TermIdentitySidecar,
        relation: &'static str,
    ) -> ArrangementDescriptor {
        let relation = sidecar
            .term_id_for_encoded(&encoded_expr(space, relation))
            .unwrap();
        ArrangementDescriptor::new(relation, 2, [0, 1]).unwrap()
    }

    fn relation_fixture(
        facts: &[u8],
        relation: &'static str,
    ) -> (Space, TermIdentitySidecar, ArrangementDescriptor) {
        let (mut space, sidecar) = sidecar_fixture(facts);
        let descriptor = relation_descriptor(&mut space, &sidecar, relation);

        (space, sidecar, descriptor)
    }

    fn edge_fixture(facts: &[u8]) -> (Space, TermIdentitySidecar, ArrangementDescriptor) {
        relation_fixture(facts, "edge")
    }

    fn query_product_count(space: &Space, pattern: &str) -> usize {
        let product_pattern = crate::expr!(space, pattern);
        Space::query_multi(&space.btm, product_pattern, |_, _| true)
    }

    fn transitive_product_count(space: &Space) -> usize {
        query_product_count(space, TRANSITIVE_PRODUCT_PATTERN)
    }

    fn pattern_factor(
        pattern: TermId,
        schema: impl Into<Box<[BindingVar]>>,
        variable_positions: impl Into<Box<[u8]>>,
    ) -> BindingAccessPlan {
        BindingAccessPlan::Pattern {
            pattern,
            projection: PatternProjection::new(schema, variable_positions).unwrap(),
        }
    }

    fn single_edge_arrangement_plan(descriptor: ArrangementDescriptor) -> BindingSidecarPlan {
        BindingSidecarPlan::new(
            [
                BindingAccessPlan::arrangement(descriptor, [BindingVar(0), BindingVar(1)], [0, 1])
                    .unwrap(),
            ],
            [BindingVar(0), BindingVar(1)],
        )
    }

    fn rows_in_order(relation: &BindingRelation, order: &[BindingVar]) -> BTreeSet<Vec<TermId>> {
        let positions = order
            .iter()
            .map(|variable| {
                relation
                    .schema()
                    .iter()
                    .position(|candidate| candidate == variable)
                    .expect("requested variable should be present in relation schema")
            })
            .collect::<Vec<_>>();

        relation
            .positive_rows()
            .map(|row| positions.iter().map(|&position| row[position]).collect())
            .collect()
    }

    fn transitive_edge_fixture() -> (Space, TermIdentitySidecar, ArrangementDescriptor) {
        edge_fixture(TRANSITIVE_EDGE_FACTS)
    }

    fn triangle_edge_fixture() -> (Space, TermIdentitySidecar, ArrangementDescriptor) {
        edge_fixture(TRIANGLE_EDGE_FACTS)
    }

    /// Four-factor body: a triangle edge(a,b), edge(b,c), edge(c,a) plus an
    /// acyclic tail edge(c,d). Mostly acyclic with a small cyclic core, so the
    /// selector routes it to the hypertree-decomposition kernel.
    fn triangle_tail_edge_plan(descriptor: &ArrangementDescriptor) -> BindingSidecarPlan {
        BindingSidecarPlan::new(
            [
                edge_factor(descriptor, 0, 1),
                edge_factor(descriptor, 1, 2),
                edge_factor(descriptor, 2, 0),
                edge_factor(descriptor, 2, 3),
            ],
            [BindingVar(0), BindingVar(1), BindingVar(2), BindingVar(3)],
        )
    }

    /// Six-factor body: a triangle edge(a,b), edge(b,c), edge(c,a) plus a 3-edge
    /// acyclic chain edge(c,d), edge(d,e), edge(e,f). The width is 2 (the triangle
    /// core) but the edge cover is 3, so Yannakakis-over-bags beats a global join
    /// and the selector routes here.
    fn triangle_chain_edge_plan(descriptor: &ArrangementDescriptor) -> BindingSidecarPlan {
        BindingSidecarPlan::new(
            [
                edge_factor(descriptor, 0, 1),
                edge_factor(descriptor, 1, 2),
                edge_factor(descriptor, 2, 0),
                edge_factor(descriptor, 2, 3),
                edge_factor(descriptor, 3, 4),
                edge_factor(descriptor, 4, 5),
            ],
            [
                BindingVar(0),
                BindingVar(1),
                BindingVar(2),
                BindingVar(3),
                BindingVar(4),
                BindingVar(5),
            ],
        )
    }

    fn transitive_edge_plan(
        descriptor: &ArrangementDescriptor,
        variable_order: [BindingVar; 3],
    ) -> BindingSidecarPlan {
        let xy = BindingAccessPlan::arrangement(
            descriptor.clone(),
            [BindingVar(0), BindingVar(1)],
            [0, 1],
        )
        .unwrap();
        let yz = BindingAccessPlan::arrangement(
            descriptor.clone(),
            [BindingVar(1), BindingVar(2)],
            [0, 1],
        )
        .unwrap();
        BindingSidecarPlan::new([xy, yz], variable_order)
    }

    /// One edge arrangement factor projected onto binding variables `(a, b)`.
    fn edge_factor(descriptor: &ArrangementDescriptor, a: u8, b: u8) -> BindingAccessPlan {
        BindingAccessPlan::arrangement(descriptor.clone(), [BindingVar(a), BindingVar(b)], [0, 1])
            .unwrap()
    }

    /// Three-factor chain `edge(a,b), edge(b,c), edge(c,d)`: a multiway
    /// alpha-acyclic (path) body that selects the Yannakakis full reducer.
    fn chain_edge_plan(descriptor: &ArrangementDescriptor) -> BindingSidecarPlan {
        BindingSidecarPlan::new(
            [
                edge_factor(descriptor, 0, 1),
                edge_factor(descriptor, 1, 2),
                edge_factor(descriptor, 2, 3),
            ],
            [BindingVar(0), BindingVar(1), BindingVar(2), BindingVar(3)],
        )
    }

    /// Three-factor triangle `edge(a,b), edge(b,c), edge(c,a)`: a multiway
    /// alpha-cyclic body with no full reducer, so the selector keeps the trie
    /// (worst-case-optimal) kernel.
    fn triangle_edge_plan(descriptor: &ArrangementDescriptor) -> BindingSidecarPlan {
        BindingSidecarPlan::new(
            [
                edge_factor(descriptor, 0, 1),
                edge_factor(descriptor, 1, 2),
                edge_factor(descriptor, 2, 0),
            ],
            [BindingVar(0), BindingVar(1), BindingVar(2)],
        )
    }

    #[test]
    fn arrangement_sidecar_plan_matches_transitive_product_query() {
        let (space, sidecar, descriptor) = transitive_edge_fixture();
        let plan = transitive_edge_plan(&descriptor, [BindingVar(1), BindingVar(0), BindingVar(2)]);

        let result = plan.execute(&sidecar).unwrap();
        let product_count = transitive_product_count(&space);

        assert_eq!(product_count, 4);
        assert_eq!(result.relation.positive_rows().count(), product_count);
        assert_eq!(
            result.stats,
            BindingSidecarStats {
                factors: 2,
                facts_scanned: 6,
                arrangement_rows: 6,
                projected_rows: 12,
                output_rows: 4,
                ..BindingSidecarStats::default()
            }
        );
    }

    #[test]
    fn arrangement_sidecar_plan_skips_schematic_relation_rows() {
        let (_, sidecar, descriptor) = edge_fixture(SCHEMATIC_EDGE_FACTS);
        let plan = single_edge_arrangement_plan(descriptor);

        let result = plan.execute(&sidecar).unwrap();

        assert_eq!(result.relation.positive_rows().count(), 2);
        assert_eq!(result.stats.factors, 1);
        // Only the four edge facts are scanned, not (node Bob): build reads the
        // relation's bucket, not the whole sidecar.
        assert_eq!(result.stats.facts_scanned, 4);
        assert_eq!(result.stats.arrangement_rows, 2);
        assert_eq!(result.stats.arrangement_schematic_rows_skipped, 2);
        assert_eq!(result.stats.projected_rows, 2);
        assert_eq!(result.stats.output_rows, 2);
    }

    #[test]
    fn trie_sidecar_plan_matches_generic_and_product_query() {
        let (space, sidecar, descriptor) = transitive_edge_fixture();
        let plan = transitive_edge_plan(&descriptor, [BindingVar(1), BindingVar(0), BindingVar(2)]);

        let generic = plan.execute(&sidecar).unwrap();
        let trie = plan.execute_trie_join(&sidecar).unwrap();
        let product_count = transitive_product_count(&space);

        assert_eq!(trie.relation, generic.relation);
        assert_eq!(trie.relation.positive_rows().count(), product_count);
        assert_eq!(trie.trie_stats.output_rows, product_count);
        assert_eq!(trie.trie_stats.relation_indexes, 2);
        assert_eq!(trie.trie_stats.indexed_rows, 12);
        assert_eq!(trie.trie_stats.domain_intersections, 8);
        assert_eq!(trie.trie_stats.domain_cursor_opens, 2);
        assert!(trie.trie_stats.domain_cursor_opens < trie.trie_stats.domain_sources);
        assert!(trie.trie_stats.domain_cursor_seeks >= trie.trie_stats.domain_cursor_opens);
    }

    #[test]
    fn suggested_trie_join_order_preserves_rows_and_uses_small_shared_domain() {
        let (space, sidecar, descriptor) = transitive_edge_fixture();
        let plan = transitive_edge_plan(&descriptor, [BindingVar(0), BindingVar(1), BindingVar(2)]);

        let manual = plan.execute_trie_join(&sidecar).unwrap();
        let suggested = plan.execute_trie_join_suggested(&sidecar).unwrap();
        let product_count = transitive_product_count(&space);

        assert_eq!(
            suggested.variable_order.as_ref(),
            [BindingVar(1), BindingVar(0), BindingVar(2)]
        );
        assert_eq!(manual.variable_order.as_ref(), plan.variable_order());
        assert_eq!(suggested.relation.positive_rows().count(), product_count);
        assert_eq!(suggested.trie_stats.output_rows, product_count);
        assert_eq!(
            rows_in_order(&suggested.relation, plan.variable_order()),
            rows_in_order(&manual.relation, plan.variable_order())
        );
    }

    #[test]
    fn selected_execution_uses_suggested_trie_join_for_shared_domains() {
        let (space, sidecar, descriptor) = transitive_edge_fixture();
        let plan = transitive_edge_plan(&descriptor, [BindingVar(0), BindingVar(1), BindingVar(2)]);

        let selected = plan.execute_selected(&sidecar).unwrap();
        let product_count = transitive_product_count(&space);

        assert_eq!(
            selected.choice.kernel,
            BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        assert_eq!(
            selected.choice.reason,
            BindingSidecarExecutionReason::SharedVariablePruning
        );
        assert_eq!(
            selected.choice.variable_order.as_ref(),
            [BindingVar(1), BindingVar(0), BindingVar(2)]
        );
        assert_eq!(selected.choice.projected_rows, 12);
        assert_eq!(selected.choice.shared_variables, 1);
        assert_eq!(selected.choice.min_shared_root_domain_len, Some(3));
        assert_eq!(selected.relation.positive_rows().count(), product_count);
        assert_eq!(selected.trie_stats.unwrap().output_rows, product_count);
    }

    #[test]
    fn selected_execution_uses_acyclic_yannakakis_for_multiway_chain() {
        let (_, sidecar, descriptor) = transitive_edge_fixture();
        let plan = chain_edge_plan(&descriptor);
        let order = [BindingVar(0), BindingVar(1), BindingVar(2), BindingVar(3)];

        let prepared = plan.prepare(&sidecar).unwrap();
        assert_eq!(prepared.relations().len(), 3);
        assert_eq!(
            prepared.execution().choice.kernel,
            BindingSidecarExecutionKernel::AcyclicYannakakis
        );
        assert_eq!(
            prepared.execution().choice.reason,
            BindingSidecarExecutionReason::AcyclicJoinTree
        );
        // The Yannakakis kernel reduces then Generic-Joins; it builds no trie.
        assert!(prepared.prepared_trie_join().is_none());

        let selected = prepared.execute_selected().unwrap();
        let generic = plan.execute(&sidecar).unwrap();

        // Output set equals the plain Generic Join oracle (Yannakakis 1981: the
        // full reducer deletes only dangling tuples, preserving the join set).
        assert_eq!(
            rows_in_order(&selected.relation, &order),
            rows_in_order(&generic.relation, &order)
        );
        // The two 3-hop paths Alice->{Bob,Dana}->Carol->Erin.
        assert_eq!(selected.relation.positive_rows().count(), 2);
        assert_eq!(generic.relation.positive_rows().count(), 2);
        // Dangling edges that cannot extend to a full 3-hop path were pruned
        // (e.g. X->Y, and any edge into Erin which has no outgoing edge).
        assert!(selected.stats.semijoin_removed_rows > 0);
        assert!(selected.trie_stats.is_none());
    }

    #[test]
    fn acyclic_yannakakis_count_and_exists_match_generic_oracle() {
        let (_, sidecar, descriptor) = transitive_edge_fixture();
        let plan = chain_edge_plan(&descriptor);

        let count = plan.count_selected(&sidecar).unwrap();
        let exists = plan.exists_selected(&sidecar).unwrap();
        let expected = plan
            .execute(&sidecar)
            .unwrap()
            .relation
            .positive_rows()
            .count();

        assert_eq!(
            count.execution.choice.kernel,
            BindingSidecarExecutionKernel::AcyclicYannakakis
        );
        assert_eq!(
            count.aggregate_kernel,
            BindingSidecarAggregateKernel::GenericJoinCount
        );
        assert_eq!(count.rows, expected);
        assert!(!count.materialized_output);
        assert!(count.trie_stats.is_none());
        assert!(count.stats.semijoin_removed_rows > 0);

        assert_eq!(
            exists.aggregate_kernel,
            BindingSidecarAggregateKernel::GenericJoinExists
        );
        assert_eq!(exists.exists, expected > 0);
        assert!(exists.stats.semijoin_removed_rows > 0);
    }

    #[test]
    fn selected_execution_keeps_trie_for_cyclic_triangle() {
        let (_, sidecar, descriptor) = transitive_edge_fixture();
        let plan = triangle_edge_plan(&descriptor);
        let order = [BindingVar(0), BindingVar(1), BindingVar(2)];

        let prepared = plan.prepare(&sidecar).unwrap();
        assert_eq!(prepared.relations().len(), 3);
        // A 3-cycle is alpha-cyclic: no full reducer exists, so the selector
        // keeps the worst-case-optimal trie kernel rather than reducing.
        assert_eq!(
            prepared.execution().choice.kernel,
            BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        assert_eq!(
            prepared.execution().choice.reason,
            BindingSidecarExecutionReason::SharedVariablePruning
        );
        assert!(prepared.prepared_trie_join().is_some());

        let selected = prepared.execute_selected().unwrap();
        let generic = plan.execute(&sidecar).unwrap();
        assert_eq!(
            rows_in_order(&selected.relation, &order),
            rows_in_order(&generic.relation, &order)
        );
        // No 3-cycle exists in the fixture, and the reducer never ran.
        assert_eq!(selected.stats.semijoin_removed_rows, 0);
    }

    #[test]
    fn selected_execution_uses_ghd_for_long_mostly_acyclic_body() {
        let (_, sidecar, descriptor) = triangle_edge_fixture();
        let plan = triangle_chain_edge_plan(&descriptor);
        let order = [
            BindingVar(0),
            BindingVar(1),
            BindingVar(2),
            BindingVar(3),
            BindingVar(4),
            BindingVar(5),
        ];

        let prepared = plan.prepare(&sidecar).unwrap();
        assert_eq!(prepared.relations().len(), 6);
        // Width 2 (triangle core) is below the edge cover 3, so GHD wins.
        assert_eq!(
            prepared.execution().choice.kernel,
            BindingSidecarExecutionKernel::GhdYannakakis
        );
        assert_eq!(
            prepared.execution().choice.reason,
            BindingSidecarExecutionReason::BoundedWidthDecomposition
        );
        assert!(prepared.prepared_trie_join().is_none());

        let selected = prepared.execute_selected().unwrap();
        let generic = plan.execute(&sidecar).unwrap();
        assert_eq!(
            rows_in_order(&selected.relation, &order),
            rows_in_order(&generic.relation, &order)
        );
        // The triangle closes and the chain c->d->e->f extends it, so the
        // decomposed join is non-empty.
        assert!(selected.relation.positive_rows().count() > 0);

        let count = plan.count_selected(&sidecar).unwrap();
        assert_eq!(
            count.execution.choice.kernel,
            BindingSidecarExecutionKernel::GhdYannakakis
        );
        assert_eq!(count.rows, generic.relation.positive_rows().count());
        assert!(plan.exists_selected(&sidecar).unwrap().exists);
    }

    #[test]
    fn adaptive_planner_routes_each_topology_to_the_right_kernel() {
        // One demonstration that the planner picks the right kernel per topology
        // and every kernel matches the Generic Join oracle: a 2-factor join and a
        // tight triangle stay on the trie kernel, an acyclic chain uses the
        // Yannakakis reducer, and a triangle with a long acyclic chain uses the
        // hypertree decomposition.
        let (_, sidecar, descriptor) = triangle_edge_fixture();
        let two_factor =
            transitive_edge_plan(&descriptor, [BindingVar(0), BindingVar(1), BindingVar(2)]);
        let acyclic_chain = chain_edge_plan(&descriptor);
        let triangle = triangle_edge_plan(&descriptor);
        let triangle_chain = triangle_chain_edge_plan(&descriptor);

        let cases: [(&BindingSidecarPlan, BindingSidecarExecutionKernel); 4] = [
            (
                &two_factor,
                BindingSidecarExecutionKernel::TrieJoinSuggested,
            ),
            (
                &acyclic_chain,
                BindingSidecarExecutionKernel::AcyclicYannakakis,
            ),
            (&triangle, BindingSidecarExecutionKernel::TrieJoinSuggested),
            (
                &triangle_chain,
                BindingSidecarExecutionKernel::GhdYannakakis,
            ),
        ];

        for (plan, expected_kernel) in cases {
            let prepared = plan.prepare(&sidecar).unwrap();
            assert_eq!(
                prepared.execution().choice.kernel,
                expected_kernel,
                "unexpected kernel for a {expected_kernel:?} topology"
            );
            let selected = prepared.execute_selected().unwrap();
            let generic = plan.execute(&sidecar).unwrap();
            let order = plan.variable_order().to_vec();
            assert_eq!(
                rows_in_order(&selected.relation, &order),
                rows_in_order(&generic.relation, &order),
                "kernel {expected_kernel:?} output must match the oracle"
            );
        }
    }

    #[test]
    fn explain_routing_cost_exposes_the_ghd_decision() {
        let (_, sidecar, descriptor) = triangle_edge_fixture();
        let plan = triangle_chain_edge_plan(&descriptor);
        let prepared = plan.prepare(&sidecar).unwrap();

        let cost = prepared.explain_routing_cost();
        assert_eq!(cost.decomposition_width, Some(2));
        let ghd_cost = cost.decomposition_cost.unwrap();
        // The decomposition is cheaper than a global join, which is why GHD won.
        assert!(ghd_cost < cost.global_agm_bound);
        assert_eq!(
            prepared.execution().choice.kernel,
            BindingSidecarExecutionKernel::GhdYannakakis
        );
    }

    #[test]
    fn selected_execution_keeps_trie_for_short_cyclic_body() {
        // Triangle plus a one-edge tail: width 2 equals the edge cover 2, so the
        // decomposition has no cost advantage and the trie kernel stays.
        let (_, sidecar, descriptor) = triangle_edge_fixture();
        let plan = triangle_tail_edge_plan(&descriptor);
        let order = [BindingVar(0), BindingVar(1), BindingVar(2), BindingVar(3)];

        let prepared = plan.prepare(&sidecar).unwrap();
        assert_eq!(
            prepared.execution().choice.kernel,
            BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        let selected = prepared.execute_selected().unwrap();
        let generic = plan.execute(&sidecar).unwrap();
        assert_eq!(
            rows_in_order(&selected.relation, &order),
            rows_in_order(&generic.relation, &order)
        );
    }

    #[test]
    fn acyclic_yannakakis_explain_reports_kernel_without_trie_trace() {
        let (_, sidecar, descriptor) = transitive_edge_fixture();
        let plan = chain_edge_plan(&descriptor);
        let prepared = plan.prepare(&sidecar).unwrap();

        let report = prepared.explain_selected_trie_trace().unwrap();
        assert_eq!(
            report.execution.choice.kernel,
            BindingSidecarExecutionKernel::AcyclicYannakakis
        );
        // The Yannakakis kernel has no trie cursor, so there is no trie trace.
        assert!(report.trie_trace.is_none());
    }

    #[test]
    fn selected_aggregate_reports_use_trie_without_materializing_rows() {
        let (space, sidecar, descriptor) = transitive_edge_fixture();
        let plan = transitive_edge_plan(&descriptor, [BindingVar(0), BindingVar(1), BindingVar(2)]);

        let count = plan.count_selected(&sidecar).unwrap();
        let exists = plan.exists_selected(&sidecar).unwrap();
        let product_count = transitive_product_count(&space);

        assert_eq!(
            count.execution.choice.kernel,
            BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        assert_eq!(
            count.aggregate_kernel,
            BindingSidecarAggregateKernel::TrieJoinCount
        );
        assert_eq!(count.rows, product_count);
        assert!(!count.materialized_output);
        assert_eq!(count.stats.output_rows, product_count);

        assert_eq!(
            exists.aggregate_kernel,
            BindingSidecarAggregateKernel::TrieJoinExists
        );
        assert!(exists.exists);
        assert!(!exists.materialized_output);
        assert_eq!(exists.stats.output_rows, 1);
        assert!(
            exists.trie_stats.unwrap().weight_lookups < count.trie_stats.unwrap().weight_lookups
        );
    }

    #[test]
    fn prepared_sidecar_reuses_opened_relations_for_multiple_reports() {
        let (_, sidecar, descriptor) = transitive_edge_fixture();
        let plan = transitive_edge_plan(&descriptor, [BindingVar(0), BindingVar(1), BindingVar(2)]);

        let prepared = plan.prepare(&sidecar).unwrap();
        let count = prepared.count_selected().unwrap();
        let exists = prepared.exists_selected().unwrap();
        let selected = prepared.execute_selected().unwrap();

        assert_eq!(prepared.relations().len(), 2);
        assert_eq!(
            prepared.execution().choice.kernel,
            BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        assert!(prepared.prepared_trie_join().is_some());
        assert_eq!(
            prepared.prepared_trie_join().unwrap().variable_order(),
            prepared.execution().choice.variable_order.as_ref()
        );
        assert_eq!(prepared.execution().analysis.stats.factors, 2);
        assert_eq!(count.rows, selected.relation.positive_rows().count());
        assert_eq!(
            count.stats.factors,
            prepared.execution().analysis.stats.factors
        );
        assert_eq!(
            exists.stats.factors,
            prepared.execution().analysis.stats.factors
        );
        assert_eq!(
            selected.stats.factors,
            prepared.execution().analysis.stats.factors
        );
        assert!(!count.materialized_output);
        assert!(!exists.materialized_output);
    }

    #[test]
    fn selected_aggregate_reports_use_generic_without_materializing_rows() {
        let (_, sidecar, descriptor) = edge_fixture(THREE_EDGE_FACTS);
        let plan = single_edge_arrangement_plan(descriptor);
        let prepared = plan.prepare(&sidecar).unwrap();

        let count = plan.count_selected(&sidecar).unwrap();
        let exists = plan.exists_selected(&sidecar).unwrap();

        assert_eq!(
            prepared.execution().choice.reason,
            BindingSidecarExecutionReason::SingleFactor
        );
        assert!(prepared.prepared_trie_join().is_none());
        assert_eq!(
            count.execution.choice.reason,
            BindingSidecarExecutionReason::SingleFactor
        );
        assert_eq!(
            count.aggregate_kernel,
            BindingSidecarAggregateKernel::GenericJoinCount
        );
        assert_eq!(count.rows, 3);
        assert_eq!(count.stats.output_rows, 3);
        assert!(!count.materialized_output);
        assert_eq!(count.trie_stats, None);

        assert_eq!(
            exists.aggregate_kernel,
            BindingSidecarAggregateKernel::GenericJoinExists
        );
        assert!(exists.exists);
        assert_eq!(exists.stats.output_rows, 1);
        assert!(!exists.materialized_output);
        assert_eq!(exists.trie_stats, None);
    }

    #[test]
    fn execution_report_explains_selected_kernel_without_joining() {
        let (_, sidecar, descriptor) = transitive_edge_fixture();
        let plan = transitive_edge_plan(&descriptor, [BindingVar(0), BindingVar(1), BindingVar(2)]);

        let report = plan.explain_execution(&sidecar).unwrap();

        assert_eq!(
            report.choice.kernel,
            BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        assert_eq!(
            report.choice.variable_order.as_ref(),
            [BindingVar(1), BindingVar(0), BindingVar(2)]
        );
        assert_eq!(report.analysis.stats.projected_rows, 12);
        assert_eq!(report.analysis.stats.output_rows, 0);
        assert_eq!(
            report.analysis.suggested_variable_order.as_ref(),
            report.choice.variable_order.as_ref()
        );
    }

    #[test]
    fn selected_trie_trace_matches_product_query_count_without_materializing_relation() {
        let (space, sidecar, descriptor) = transitive_edge_fixture();
        let plan = transitive_edge_plan(&descriptor, [BindingVar(0), BindingVar(1), BindingVar(2)]);

        let report = plan.explain_selected_trie_trace(&sidecar).unwrap();
        let product_count = transitive_product_count(&space);
        let trace = report.trie_trace.unwrap();
        let summary = trace.summarize().unwrap();

        assert_eq!(
            report.execution.choice.kernel,
            BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        assert_eq!(report.execution.analysis.stats.output_rows, 0);
        assert_eq!(
            trace.variable_order.as_ref(),
            [BindingVar(1), BindingVar(0), BindingVar(2)]
        );
        assert_eq!(trace.candidate_bindings, product_count);
        assert_eq!(trace.relation_indexes, 2);
        assert_eq!(trace.steps[0].variable, BindingVar(1));
        assert_eq!(trace.steps[0].participating_relations.as_ref(), [0, 1]);
        assert_eq!(trace.steps[0].relation_domain_lens.len(), 2);
        assert_eq!(trace.steps[0].domain_sources, 2);
        assert_eq!(trace.steps[0].intersection.len(), 3);
        assert_eq!(summary.candidate_bindings, product_count);
        assert_eq!(summary.max_participating_relations, 2);
        assert_eq!(summary.empty_intersections, 0);
    }

    #[test]
    fn selected_trie_cursor_contract_names_factor_replay_contexts_without_joining() {
        let (space, sidecar, descriptor) = transitive_edge_fixture();
        let plan = transitive_edge_plan(&descriptor, [BindingVar(0), BindingVar(1), BindingVar(2)]);

        let report = plan
            .explain_selected_trie_cursor_contract(&sidecar)
            .unwrap();
        let product_count = transitive_product_count(&space);
        let contract = report.cursor_contract.unwrap();

        assert_eq!(
            report.execution.choice.kernel,
            BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        assert_eq!(report.execution.analysis.stats.output_rows, 0);
        assert_eq!(contract.summary.candidate_bindings, product_count);
        assert_eq!(contract.relation_indexes, 2);
        assert_eq!(contract.factor_requirements.len(), 2);
        assert_eq!(contract.factor_requirements[0].relation_index, 0);
        assert_eq!(contract.factor_requirements[0].domain_contexts, 4);
        assert_eq!(contract.factor_requirements[1].relation_index, 1);
        assert_eq!(contract.factor_requirements[1].domain_contexts, 5);
        assert_eq!(
            contract.factor_requirements[0].contexts[0]
                .bound_prefix
                .as_ref(),
            []
        );
        assert_eq!(
            contract.factor_requirements[0].contexts[0].variable,
            BindingVar(1)
        );
    }

    #[test]
    fn selected_trie_trace_diff_compares_sidecar_snapshots_without_joining() {
        let (mut space, old_sidecar, descriptor) = transitive_edge_fixture();
        let alice = old_sidecar
            .term_id_for_encoded(&encoded_expr(&mut space, "Alice"))
            .unwrap();
        let bob = old_sidecar
            .term_id_for_encoded(&encoded_expr(&mut space, "Bob"))
            .unwrap();

        let mut new_sidecar = old_sidecar.clone();
        new_sidecar
            .insert_fact(&encoded_expr(&mut space, "[3] edge Bob Erin"))
            .unwrap();

        let plan = transitive_edge_plan(&descriptor, [BindingVar(0), BindingVar(1), BindingVar(2)]);

        let report = plan
            .explain_selected_trie_trace_diff(&old_sidecar, &new_sidecar)
            .unwrap();
        let diff = report.replay_diff.as_ref().unwrap();

        assert_eq!(
            report.old.execution.choice.kernel,
            BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        assert_eq!(
            report.new.execution.choice.kernel,
            BindingSidecarExecutionKernel::TrieJoinSuggested
        );
        assert_eq!(report.old.execution.analysis.stats.output_rows, 0);
        assert_eq!(report.new.execution.analysis.stats.output_rows, 0);
        assert!(!diff.is_empty());
        assert_eq!(diff.changed_contexts, 1);
        assert_eq!(diff.added_contexts, 0);
        assert_eq!(diff.removed_contexts, 0);
        assert_eq!(diff.frontier_contexts, 1);
        assert_eq!(diff.old_replay_steps_touched, 1);
        assert_eq!(diff.new_replay_steps_touched, 1);
        assert_eq!(diff.old_candidate_bindings_touched, 1);
        assert_eq!(diff.new_candidate_bindings_touched, 2);
        assert_eq!(diff.entries.len(), 1);
        assert_eq!(diff.entries[0].bound_prefix.as_ref(), [bob, alice]);
    }

    #[test]
    fn execution_report_keeps_generic_join_for_tiny_shared_input() {
        let (_, sidecar, descriptor) = edge_fixture(TWO_EDGE_FACTS);
        let plan = transitive_edge_plan(&descriptor, [BindingVar(0), BindingVar(1), BindingVar(2)]);

        let report = plan.explain_execution(&sidecar).unwrap();

        assert_eq!(
            report.choice.kernel,
            BindingSidecarExecutionKernel::GenericJoin
        );
        assert_eq!(
            report.choice.reason,
            BindingSidecarExecutionReason::SmallExplicitInput
        );
        assert_eq!(report.choice.projected_rows, 4);
        assert_eq!(report.choice.shared_variables, 1);
        assert_eq!(report.choice.min_shared_root_domain_len, Some(1));
    }

    #[test]
    fn selected_trie_trace_is_absent_when_selector_keeps_generic_join() {
        let (_, sidecar, descriptor) = edge_fixture(TWO_EDGE_FACTS);
        let plan = transitive_edge_plan(&descriptor, [BindingVar(0), BindingVar(1), BindingVar(2)]);

        let report = plan.explain_selected_trie_trace(&sidecar).unwrap();

        assert_eq!(
            report.execution.choice.kernel,
            BindingSidecarExecutionKernel::GenericJoin
        );
        assert_eq!(
            report.execution.choice.reason,
            BindingSidecarExecutionReason::SmallExplicitInput
        );
        assert_eq!(report.trie_trace, None);

        let cursor_report = plan
            .explain_selected_trie_cursor_contract(&sidecar)
            .unwrap();
        assert_eq!(
            cursor_report.execution.choice.kernel,
            BindingSidecarExecutionKernel::GenericJoin
        );
        assert_eq!(cursor_report.cursor_contract, None);
    }

    #[test]
    fn selected_execution_keeps_generic_join_for_single_factor() {
        let (_, sidecar, descriptor) = edge_fixture(TWO_EDGE_FACTS);
        let plan = single_edge_arrangement_plan(descriptor);

        let choice = plan.choose_execution(&sidecar).unwrap();
        let selected = plan.execute_selected(&sidecar).unwrap();

        assert_eq!(choice.kernel, BindingSidecarExecutionKernel::GenericJoin);
        assert_eq!(choice.reason, BindingSidecarExecutionReason::SingleFactor);
        assert_eq!(choice.variable_order.as_ref(), plan.variable_order());
        assert_eq!(selected.choice, choice);
        assert_eq!(selected.relation.positive_rows().count(), 2);
        assert_eq!(selected.trie_stats, None);
    }

    #[test]
    fn suggested_variable_order_preserves_deterministic_tie_breaks() {
        let stats = [
            BindingVariableDomainStats {
                variable: BindingVar(2),
                factor_count: 1,
                root_domain_len: 5,
                min_factor_domain_len: 5,
                max_factor_domain_len: 5,
            },
            BindingVariableDomainStats {
                variable: BindingVar(0),
                factor_count: 1,
                root_domain_len: 5,
                min_factor_domain_len: 5,
                max_factor_domain_len: 5,
            },
            BindingVariableDomainStats {
                variable: BindingVar(1),
                factor_count: 1,
                root_domain_len: 5,
                min_factor_domain_len: 5,
                max_factor_domain_len: 5,
            },
        ];

        let order = suggest_variable_order(&[], &stats);

        assert_eq!(order, [BindingVar(0), BindingVar(1), BindingVar(2)]);
    }

    #[test]
    fn analysis_suggests_selective_connected_variable_order() {
        let (_, sidecar, descriptor) = transitive_edge_fixture();
        let plan = BindingSidecarPlan::new(
            [
                BindingAccessPlan::arrangement(
                    descriptor.clone(),
                    [BindingVar(0), BindingVar(1)],
                    [0, 1],
                )
                .unwrap(),
                BindingAccessPlan::arrangement(descriptor, [BindingVar(1), BindingVar(2)], [0, 1])
                    .unwrap(),
            ],
            [BindingVar(0), BindingVar(1), BindingVar(2)],
        );

        let analysis = plan.analyze(&sidecar).unwrap();

        assert_eq!(
            analysis.suggested_variable_order.as_ref(),
            [BindingVar(1), BindingVar(0), BindingVar(2)]
        );
        assert_eq!(
            analysis
                .variable_order_candidates
                .iter()
                .map(|candidate| (candidate.source, candidate.variable_order.as_ref()))
                .collect::<Vec<_>>(),
            vec![
                (
                    BindingVariableOrderSource::Planned,
                    &[BindingVar(0), BindingVar(1), BindingVar(2)][..],
                ),
                (
                    BindingVariableOrderSource::Suggested,
                    &[BindingVar(1), BindingVar(0), BindingVar(2)][..],
                ),
                (
                    BindingVariableOrderSource::RootDomainAscending,
                    &[BindingVar(1), BindingVar(2), BindingVar(0)][..],
                ),
            ]
        );
        assert_eq!(
            analysis
                .variable_order_candidates
                .iter()
                .map(|candidate| candidate.connected_prefix_steps)
                .collect::<Vec<_>>(),
            vec![2, 2, 2]
        );
        assert!(
            analysis.variable_order_candidates[1].estimated_depth_weighted_domain_work
                < analysis.variable_order_candidates[0].estimated_depth_weighted_domain_work
        );
        assert_eq!(
            analysis.parallel_share_plan(8),
            BindingParallelSharePlan {
                target_tasks: 8,
                realized_tasks: 8,
                variable_order: [BindingVar(1), BindingVar(0), BindingVar(2)].into(),
                variables: [
                    BindingVariableShare {
                        variable: BindingVar(1),
                        share: 2,
                        root_domain_len: 3,
                        factor_count: 2,
                    },
                    BindingVariableShare {
                        variable: BindingVar(0),
                        share: 2,
                        root_domain_len: 5,
                        factor_count: 1,
                    },
                    BindingVariableShare {
                        variable: BindingVar(2),
                        share: 2,
                        root_domain_len: 5,
                        factor_count: 1,
                    },
                ]
                .into(),
            }
        );
        assert_eq!(
            analysis.variables.as_ref(),
            [
                BindingVariableDomainStats {
                    variable: BindingVar(0),
                    factor_count: 1,
                    root_domain_len: 5,
                    min_factor_domain_len: 5,
                    max_factor_domain_len: 5,
                },
                BindingVariableDomainStats {
                    variable: BindingVar(1),
                    factor_count: 2,
                    root_domain_len: 3,
                    min_factor_domain_len: 5,
                    max_factor_domain_len: 5,
                },
                BindingVariableDomainStats {
                    variable: BindingVar(2),
                    factor_count: 1,
                    root_domain_len: 5,
                    min_factor_domain_len: 5,
                    max_factor_domain_len: 5,
                },
            ]
        );

        let adaptive = analysis.adaptive_episode_plan(8, 16);
        let task_coordinates = adaptive
            .task_boxes
            .iter()
            .map(|task| {
                assert_eq!(task.ranges.len(), 3);
                task.ranges
                    .iter()
                    .map(|range| {
                        assert_eq!(range.end_bucket, range.start_bucket + 1);
                        assert!(range.end_bucket <= range.share);
                        (
                            range.variable,
                            range.start_bucket,
                            range.end_bucket,
                            range.share,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(adaptive.target_tasks, 8);
        assert_eq!(adaptive.episode_budget_steps, 16);
        assert_eq!(adaptive.task_boxes.len(), 8);
        assert_eq!(task_coordinates.len(), adaptive.task_boxes.len());
        assert_eq!(
            adaptive
                .order_estimates
                .iter()
                .map(|estimate| (
                    estimate.source,
                    estimate.estimated_depth_weighted_domain_work,
                    estimate.estimated_covered_domain_cells,
                    estimate.reward_denominator,
                ))
                .collect::<Vec<_>>(),
            vec![
                (BindingVariableOrderSource::Planned, 32, 37, 75),
                (BindingVariableOrderSource::Suggested, 31, 38, 75),
                (BindingVariableOrderSource::RootDomainAscending, 31, 38, 75),
            ]
        );
        assert!(
            adaptive
                .order_estimates
                .iter()
                .all(|estimate| estimate.reward_numerator <= estimate.reward_denominator)
        );
    }

    #[test]
    fn expression_trie_pattern_factor_projects_repeated_variable_bindings() {
        let (mut space, sidecar, patterns) =
            sidecar_fixture_with_patterns(REPEATED_EDGE_FACTS, &[REPEATED_EDGE_PATTERN]);
        let plan = BindingSidecarPlan::new(
            [pattern_factor(patterns[0], [BindingVar(0)], [0])],
            [BindingVar(0)],
        );

        let result = plan.execute_trie_join(&sidecar).unwrap();
        let product_count = query_product_count(&space, REPEATED_EDGE_PRODUCT_PATTERN);
        let alice = sidecar
            .term_id_for_encoded(&encoded_expr(&mut space, "Alice"))
            .unwrap();
        let bob = sidecar
            .term_id_for_encoded(&encoded_expr(&mut space, "Bob"))
            .unwrap();
        let projected = result
            .relation
            .positive_rows()
            .map(|row| row[0])
            .collect::<BTreeSet<_>>();

        assert_eq!(product_count, 2);
        assert_eq!(result.relation.positive_rows().count(), product_count);
        assert_eq!(projected, BTreeSet::from([alice, bob]));
        assert_eq!(result.stats.factors, 1);
        assert_eq!(result.stats.expression_trie_builds, 1);
        assert_eq!(result.stats.expression_trie_facts_indexed, 7);
        assert_eq!(result.stats.expression_trie_candidates, 4);
        assert_eq!(result.stats.pattern_matches, 2);
        assert_eq!(result.stats.projected_rows, 2);
        assert_eq!(result.trie_stats.output_rows, 2);
    }

    #[test]
    fn expression_trie_pattern_factor_joins_with_arrangement_factor() {
        let (mut space, sidecar, patterns) =
            sidecar_fixture_with_patterns(EDGE_COLOR_FACTS, &[REPEATED_EDGE_PATTERN]);
        let descriptor = relation_descriptor(&mut space, &sidecar, "color");
        let plan = BindingSidecarPlan::new(
            [
                pattern_factor(patterns[0], [BindingVar(0)], [0]),
                BindingAccessPlan::arrangement(descriptor, [BindingVar(0), BindingVar(1)], [0, 1])
                    .unwrap(),
            ],
            [BindingVar(0), BindingVar(1)],
        );

        let generic = plan.execute(&sidecar).unwrap();
        let trie = plan.execute_trie_join(&sidecar).unwrap();
        let product_count = query_product_count(&space, EDGE_COLOR_PRODUCT_PATTERN);

        assert_eq!(product_count, 3);
        assert_eq!(trie.relation, generic.relation);
        assert_eq!(trie.relation.positive_rows().count(), product_count);
        assert_eq!(trie.stats.expression_trie_builds, 1);
        assert_eq!(trie.stats.expression_trie_candidates, 4);
        assert_eq!(trie.stats.pattern_matches, 3);
        assert_eq!(trie.stats.arrangement_rows, 4);
        assert_eq!(trie.trie_stats.output_rows, product_count);
    }

    #[test]
    fn expression_trie_pattern_factors_share_one_index_build() {
        let (space, sidecar, patterns) = sidecar_fixture_with_patterns(
            EDGE_COLOR_FACTS,
            &[REPEATED_EDGE_PATTERN, "[3] color $ $"],
        );
        let plan = BindingSidecarPlan::new(
            [
                pattern_factor(patterns[0], [BindingVar(0)], [0]),
                pattern_factor(patterns[1], [BindingVar(0), BindingVar(1)], [0, 1]),
            ],
            [BindingVar(0), BindingVar(1)],
        );

        let result = plan.execute_trie_join(&sidecar).unwrap();
        let product_count = query_product_count(&space, EDGE_COLOR_PRODUCT_PATTERN);

        assert_eq!(product_count, 3);
        assert_eq!(result.relation.positive_rows().count(), product_count);
        assert_eq!(result.stats.factors, 2);
        assert_eq!(result.stats.expression_trie_builds, 1);
        assert_eq!(result.stats.expression_trie_facts_indexed, 8);
        assert_eq!(result.stats.expression_trie_candidates, 8);
        assert_eq!(result.stats.pattern_matches, 7);
        assert_eq!(result.stats.projected_rows, 7);
        assert_eq!(result.trie_stats.output_rows, product_count);
    }
}
