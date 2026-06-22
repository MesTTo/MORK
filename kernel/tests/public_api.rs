use mork::api::{
    AdditiveSaturationComparisonReport, AdditiveSaturationReport, AdditiveSaturationRound,
    AdditiveSaturationRoundWorkSavings, AdditiveSaturationRuleRound, AdditiveSaturationWorkSavings,
    ArrangementDescriptor, ArrangementIndex, ArrangementProjection, ArrangementRow,
    ArrangementStats, Binding, BindingAccessPlan, BindingAdaptiveEpisodePlan, BindingDomainCursor,
    BindingDomainIntersection, BindingEnv, BindingEnvError, BindingParallelSharePlan,
    BindingRelation, BindingRelationError, BindingSidecarAggregateKernel,
    BindingSidecarExecutionKernel, BindingSidecarExecutionReason, BindingSidecarPlan,
    BindingSnapshot, BindingSource, BindingVar, CriticalPairWitness, ExpressionFeature,
    ExpressionTrieCandidates, ExpressionTrieIndex, ExpressionTrieMatches, ExpressionTrieStats,
    ExpressionTrieToken, FactId, FactorizedJoin, FormalExecSummary, FormalListMode,
    FormalMettaSpecialForm, FormalMettaSpecialResult, FormalMinimalInstruction, FormalSourceKind,
    JsonPathQueryError, JsonPathSegment, MAX_BINDING_SLOTS, PathMap, PlanValue, PlanVariableKind,
    PreparedTrieJoin, QueryProjectionByteDomainCursor, QueryProjectionDomainCursor,
    QueryProjectionRelationFactor, QueryProjectionTrieContractComparison, Rule,
    SemiNaiveTransitiveClosure, SemijoinReduction, Space, Term, TermId, TermIdentitySidecar,
    TermKind, WASM_LINEAR_MEMORY_GUARD_BYTES, WASM_LINEAR_MEMORY_GUARD_ENV,
    WASM_LINEAR_MEMORY_RESERVATION_BYTES, WASM_LINEAR_MEMORY_RESERVATION_ENV,
    WasmLinearMemoryPolicy, WasmLinearMemoryPolicyError, WeightedPathError, WeightedPathIndex,
    WeightedPathStats, WeightedSelectionTree, WeightedSelectionTreeStats,
    compare_additive_saturation_reports,
    compare_query_projection_relation_factors_to_trie_contract, delta_join,
    first_non_joinable_witness, generic_join_count, generic_join_exists,
    ground_facts_from_mm2_program, intersect_binding_domain_cursors,
    intersect_query_projection_byte_domain_cursors, intersect_query_projection_domain_values,
    intersect_query_projection_domains, lower_exec, lower_pattern, match_facts, metta_special_form,
    natural_join, parse_singular_json_path, prefix_maximal_values, prefix_minimal_values,
    saturate_additive_state_report, saturate_additive_state_semi_naive,
    saturate_additive_state_semi_naive_report, semi_naive_transitive_closure, semijoin_presence,
    semijoin_reduce_presence, shared_prefix_witnesses, state_rules_from_mm2_program,
    trie_join_count, trie_join_exists, trie_join_trace, wasm_linear_memory_policy,
    wasm_linear_memory_policy_from_env, wasm_linear_memory_policy_from_values,
};
#[cfg(feature = "experimental_dnf")]
use mork::api::{
    DnfPathSetCountResult, DnfPathSetError, DnfPathSetExistenceResult, evaluate_pathmap_dnf,
    evaluate_pathmap_dnf_count, evaluate_pathmap_dnf_exists, evaluate_pathmap_dnf_zipper_merge,
};
#[cfg(feature = "experimental_dnf")]
use pathmap::experimental::zipper_algebra::zipper_merge_dnf;

fn repeated_edge_pattern_sidecar() -> (TermIdentitySidecar, TermId) {
    let mut space = Space::new();
    space
        .add_all_sexpr(
            br#"
(edge Alice (f Alice))
(edge Alice (f Bob))
(edge Bob (f Bob))
(edge Carol (g Carol))
(edge Dave (f Eve))
(edge $x (f $x))
"#,
        )
        .unwrap();

    let mut sidecar = TermIdentitySidecar::new();
    assert_eq!(sidecar.extend_from_pathmap(&space.btm).unwrap(), 6);
    let pattern_root = sidecar
        .facts()
        .iter()
        .find(|fact| fact.flags.contains_vars)
        .expect("fixture should include one schematic pattern fact")
        .root;

    (sidecar, pattern_root)
}

fn term(id: u64) -> TermId {
    TermId(id)
}

fn var(id: u8) -> BindingVar {
    BindingVar(id)
}

fn advance_position_until<T>(
    domain: &[T],
    position: &mut usize,
    is_before_target: impl Fn(&T) -> bool,
) -> usize {
    let start = *position;
    while *position < domain.len() && is_before_target(&domain[*position]) {
        *position += 1;
    }
    *position - start
}

struct PublicCursor<T> {
    domain: Vec<T>,
    position: usize,
}

impl<T> PublicCursor<T> {
    fn new(domain: Vec<T>) -> Self {
        Self {
            domain,
            position: 0,
        }
    }

    fn key_ref(&self) -> Option<&T> {
        self.domain.get(self.position)
    }

    fn at_end(&self) -> bool {
        self.position >= self.domain.len()
    }

    fn next(&mut self) {
        if !self.at_end() {
            self.position += 1;
        }
    }

    fn domain_len(&self) -> usize {
        self.domain.len()
    }
}

impl<T: Ord> PublicCursor<T> {
    fn new_sorted_deduped(mut domain: Vec<T>) -> Self {
        domain.sort();
        domain.dedup();
        Self::new(domain)
    }
}

impl BindingDomainCursor for PublicCursor<TermId> {
    fn key(&self) -> Option<TermId> {
        self.key_ref().copied()
    }

    fn at_end(&self) -> bool {
        PublicCursor::at_end(self)
    }

    fn next(&mut self) {
        PublicCursor::next(self);
    }

    fn seek(&mut self, target: TermId) -> usize {
        advance_position_until(&self.domain, &mut self.position, |value| *value < target)
    }
}

impl QueryProjectionByteDomainCursor for PublicCursor<Vec<u8>> {
    fn key(&self) -> Option<&[u8]> {
        self.key_ref().map(Vec::as_slice)
    }

    fn at_end(&self) -> bool {
        PublicCursor::at_end(self)
    }

    fn next(&mut self) {
        PublicCursor::next(self);
    }

    fn seek(&mut self, target: &[u8]) -> usize {
        advance_position_until(&self.domain, &mut self.position, |value| {
            value.as_slice() < target
        })
    }

    fn domain_len(&self) -> usize {
        PublicCursor::domain_len(self)
    }
}

fn path_map(paths: &[&[u8]]) -> PathMap<()> {
    PathMap::from_iter(paths.iter().copied())
}

fn sorted_paths(map: &PathMap<()>) -> Vec<Vec<u8>> {
    let mut paths = Vec::new();
    map.for_each_value(|path, _| paths.push(path.to_vec()));
    paths.sort();
    paths
}

fn sorted_atoms(space: &Space) -> Vec<String> {
    let mut output = Vec::new();
    space.dump_all_sexpr(&mut output).unwrap();
    let mut atoms = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    atoms.sort();
    atoms
}

fn first_relation_functor(sidecar: &TermIdentitySidecar) -> TermId {
    for fact in sidecar.facts() {
        let Some(root) = sidecar.get_term(fact.root) else {
            continue;
        };
        if matches!(root.kind, TermKind::Application { arity: 3 }) {
            return root.children()[0];
        }
    }

    panic!("test fixture should contain a binary relation fact");
}

fn critical_fact(name: &str, args: &[&str]) -> Term {
    Term::app(name, args.iter().map(|arg| Term::sym(*arg)).collect())
}

const TRANSITIVE_EDGE_EXEC: &str = r#"
(exec transitive
  (, (edge $x $y) (edge $y $z))
  (O (+ (edge $x $z))))
"#;

#[test]
fn binding_space_api_exposes_relation_and_aggregate_kernels() {
    let mut left = BindingRelation::new([var(0), var(1)]);
    left.add(vec![term(1), term(10)], 1).unwrap();
    left.add(vec![term(2), term(10)], 1).unwrap();

    let mut right = BindingRelation::new([var(1), var(2)]);
    right.add(vec![term(10), term(100)], 1).unwrap();
    right.add(vec![term(10), term(101)], 1).unwrap();

    let joined = natural_join(&left, &right).unwrap();
    let factorized = FactorizedJoin::from_relations(&left, &right).unwrap();
    let relations = [left, right];
    let variable_order = [var(0), var(1), var(2)];
    let count = generic_join_count(&relations, &variable_order).unwrap();
    let trie_count = trie_join_count(&relations, &variable_order).unwrap();
    let trie_existence = trie_join_exists(&relations, &variable_order).unwrap();
    let existence = generic_join_exists(&[joined], &[var(0), var(1), var(2)]).unwrap();
    let visible_error = BindingRelationError::CardinalityOverflow;
    let distinct = {
        let mut relation = BindingRelation::new([var(0)]);
        relation.add(vec![term(1)], 5).unwrap();
        relation.add(vec![term(2)], -3).unwrap();
        relation.distinct()
    };

    assert_eq!(factorized.count(), 4);
    assert_eq!(factorized.checked_count().unwrap(), 4);
    assert_eq!(factorized.binding_rows().count(), 4);
    assert!(factorized.checked_factorized_node_count().unwrap() < count.rows + 6);
    assert_eq!(count.rows, 4);
    assert_eq!(trie_count.rows, count.rows);
    assert_eq!(trie_count.stats.output_rows, count.rows);
    assert!(existence.exists);
    assert!(trie_existence.exists);
    assert_eq!(visible_error, BindingRelationError::CardinalityOverflow);
    assert_eq!(distinct.weight(&[term(1)]), 1);
    assert_eq!(distinct.weight(&[term(2)]), 1);
}

#[test]
fn binding_space_api_exposes_differential_and_semijoin_diagnostics() {
    let x = var(0);
    let y = var(1);
    let z = var(2);

    let mut left = BindingRelation::new([x, y]);
    left.add(vec![term(1), term(10)], 1).unwrap();
    left.add(vec![term(2), term(20)], 1).unwrap();
    left.add(vec![term(9), term(90)], 1).unwrap();

    let mut middle = BindingRelation::new([y, z]);
    middle.add(vec![term(10), term(100)], 1).unwrap();
    middle.add(vec![term(20), term(200)], 1).unwrap();
    middle.add(vec![term(90), term(900)], 1).unwrap();
    middle.add(vec![term(30), term(100)], 1).unwrap();

    let mut right = BindingRelation::new([z]);
    right.add(vec![term(100)], 1).unwrap();
    right.add(vec![term(200)], 1).unwrap();

    let relations = [left.clone(), middle.clone(), right];
    let reduction: SemijoinReduction = semijoin_reduce_presence(&relations, &[(0, 1), (2, 1)])
        .expect("acyclic semijoin reduction should succeed");
    let filtered_left = semijoin_presence(&left, &middle).unwrap();

    assert_eq!(reduction.removed_rows, 3);
    assert_eq!(reduction.bottom_up_passes, 2);
    assert_eq!(reduction.top_down_passes, 2);
    assert_eq!(reduction.relations[0].positive_rows().count(), 2);
    assert_eq!(filtered_left.positive_rows().count(), 3);

    let old_left = left;
    let delta_left = {
        let mut relation = BindingRelation::new([x, y]);
        relation.add(vec![term(3), term(10)], 1).unwrap();
        relation
    };
    let old_right = {
        let mut relation = BindingRelation::new([y, z]);
        relation.add(vec![term(10), term(100)], 1).unwrap();
        relation.add(vec![term(10), term(101)], 1).unwrap();
        relation
    };
    let delta_right = {
        let mut relation = BindingRelation::new([y, z]);
        relation.add(vec![term(10), term(100)], -1).unwrap();
        relation.add(vec![term(10), term(102)], 1).unwrap();
        relation
    };
    let mut new_right = old_right.clone();
    new_right.union_assign(&delta_right).unwrap();

    let delta = delta_join(&old_left, &new_right, &delta_left, &delta_right).unwrap();
    assert_eq!(delta.weight(&[term(1), term(10), term(100)]), -1);
    assert_eq!(delta.weight(&[term(1), term(10), term(102)]), 1);
    assert_eq!(delta.weight(&[term(3), term(10), term(101)]), 1);
    assert_eq!(delta.weight(&[term(3), term(10), term(102)]), 1);

    let closure: SemiNaiveTransitiveClosure =
        semi_naive_transitive_closure(&old_right, y, z).unwrap();
    assert_eq!(closure.relation.weight(&[term(10), term(100)]), 1);
    assert_eq!(closure.relation.weight(&[term(10), term(101)]), 1);
    assert_eq!(closure.rounds, 1);
}

#[test]
fn binding_domain_cursor_api_can_be_implemented_by_callers() {
    let mut cursor = PublicCursor::new(vec![term(1), term(3), term(5)]);

    assert_eq!(cursor.key(), Some(term(1)));
    assert_eq!(cursor.seek(term(4)), 2);
    assert_eq!(cursor.key(), Some(term(5)));
    cursor.next();
    assert!(cursor.at_end());
    assert_eq!(cursor.seek(term(9)), 0);
}

#[test]
fn binding_domain_cursor_intersection_api_accepts_caller_cursors() {
    let mut cursors = [
        PublicCursor::new_sorted_deduped(vec![term(1), term(3), term(5), term(8)]),
        PublicCursor::new_sorted_deduped(vec![term(2), term(3), term(5), term(9)]),
        PublicCursor::new_sorted_deduped(vec![term(3), term(4), term(5), term(10)]),
    ];

    let intersection: BindingDomainIntersection = intersect_binding_domain_cursors(&mut cursors);

    assert_eq!(intersection.values, vec![term(3), term(5)]);
    assert_eq!(intersection.domain_sources, 3);
    assert_eq!(intersection.cursor_opens, 3);
    assert!(intersection.cursor_seeks >= intersection.cursor_opens);
    assert_eq!(intersection.cursor_skips, 8);
    assert_eq!(intersection.cursor_nexts, 2);
}

#[test]
fn query_projection_byte_domain_cursor_api_can_be_implemented_by_callers() {
    let mut cursors = [
        PublicCursor::new_sorted_deduped(vec![vec![1], vec![3], vec![5]]),
        PublicCursor::new_sorted_deduped(vec![vec![0], vec![3], vec![4], vec![5]]),
    ];

    let intersection = intersect_query_projection_byte_domain_cursors(&mut cursors);

    assert_eq!(intersection.values, vec![vec![3], vec![5]]);
    assert_eq!(intersection.domain_sources, 2);
    assert_eq!(intersection.domain_values, 7);
    assert!(intersection.cursor_seeks > 0);
    assert_eq!(intersection.cursor_nexts, 2);
}

#[test]
fn query_projection_api_exposes_ordered_byte_domain_contracts() {
    let mut left = BindingRelation::new([var(0), var(1)]);
    left.add(vec![term(1), term(10)], 1).unwrap();
    left.add(vec![term(2), term(10)], 1).unwrap();
    left.add(vec![term(3), term(11)], 1).unwrap();

    let mut right = BindingRelation::new([var(1), var(2)]);
    right.add(vec![term(10), term(20)], 1).unwrap();
    right.add(vec![term(10), term(21)], 1).unwrap();
    right.add(vec![term(11), term(22)], 1).unwrap();
    right.add(vec![term(12), term(23)], 1).unwrap();

    let trace = trie_join_trace(&[left, right], &[var(1), var(0), var(2)]).unwrap();
    let contract = trace.cursor_contract().unwrap();
    let factors = [
        QueryProjectionRelationFactor::from_rows_with_variables(
            [var(1), var(0)],
            [
                vec![vec![10], vec![1]],
                vec![vec![10], vec![2]],
                vec![vec![11], vec![3]],
            ],
        ),
        QueryProjectionRelationFactor::from_rows_with_variables(
            [var(1), var(2)],
            [
                vec![vec![10], vec![20]],
                vec![vec![10], vec![21]],
                vec![vec![11], vec![22]],
                vec![vec![12], vec![23]],
            ],
        ),
    ];
    let comparison: QueryProjectionTrieContractComparison =
        compare_query_projection_relation_factors_to_trie_contract(
            &factors,
            &[var(1), var(0), var(2)],
            &contract,
            |term| u8::try_from(term.0).ok().map(|value| vec![value]),
        );
    let left_y_domain = factors[0].open_domain(0, &[]);
    let right_y_domain = factors[1].open_domain(0, &[]);
    let mut generic_cursors = [
        QueryProjectionDomainCursor::from_values(left_y_domain.values.clone()),
        QueryProjectionDomainCursor::from_values(right_y_domain.values.clone()),
    ];
    let generic_intersection = intersect_query_projection_byte_domain_cursors(&mut generic_cursors);
    let domain_intersection = intersect_query_projection_domain_values(&[
        left_y_domain.values.as_slice(),
        right_y_domain.values.as_slice(),
    ]);
    let left_y_map = path_map(&[&[10], &[11]]);
    let right_y_map = path_map(&[&[10], &[11], &[12]]);
    let pathmap_intersection = intersect_query_projection_domains(&[&left_y_map, &right_y_map]);

    assert_eq!(comparison.contexts, comparison.matched_contexts);
    assert_eq!(comparison.mismatched_contexts, 0);
    assert_eq!(comparison.missing_factors, 0);
    assert_eq!(comparison.missing_term_mappings, 0);
    assert_eq!(generic_intersection.values, vec![vec![10], vec![11]]);
    assert_eq!(generic_intersection, domain_intersection);
    assert_eq!(domain_intersection.values, vec![vec![10], vec![11]]);
    assert_eq!(pathmap_intersection.values, domain_intersection.values);
    assert!(domain_intersection.cursor_seeks > 0);
}

#[test]
fn pathspace_api_exposes_prefix_oracles() {
    let source = path_map(&[b"foo", b"foo/bar", b"foo/bar/baz", b"other"]);

    assert_eq!(
        sorted_paths(&prefix_minimal_values(&source)),
        vec![b"foo".to_vec(), b"other".to_vec()]
    );
    assert_eq!(
        sorted_paths(&prefix_maximal_values(&source)),
        vec![b"foo/bar/baz".to_vec(), b"other".to_vec()]
    );

    let left = path_map(&[b"alpha/red", b"beta"]);
    let right = path_map(&[b"alpha/blue", b"betamax"]);

    assert_eq!(
        sorted_paths(&shared_prefix_witnesses(&left, &right)),
        vec![b"alpha/".to_vec(), b"beta".to_vec()]
    );
}

#[test]
fn wasm_memory_policy_api_exposes_runtime_tunables() {
    let policy: WasmLinearMemoryPolicy = wasm_linear_memory_policy();

    assert_eq!(
        policy.reservation_bytes,
        WASM_LINEAR_MEMORY_RESERVATION_BYTES
    );
    assert_eq!(policy.guard_bytes, WASM_LINEAR_MEMORY_GUARD_BYTES);
    assert_eq!(policy.reservation_bytes, 1 << 32);
    assert_eq!(policy.guard_bytes, 32 * 1024 * 1024);
    assert!(policy.guard_bytes < policy.reservation_bytes);
    assert!(policy.multi_memory_enabled);
    assert!(policy.signals_based_traps_enabled);
    assert_eq!(WasmLinearMemoryPolicy::default(), policy);
    let _from_env: fn() -> Result<WasmLinearMemoryPolicy, WasmLinearMemoryPolicyError> =
        wasm_linear_memory_policy_from_env;

    let tuned = wasm_linear_memory_policy_from_values(Some("67108864"), Some("1048576")).unwrap();
    assert_eq!(tuned.reservation_bytes, 64 * 1024 * 1024);
    assert_eq!(tuned.guard_bytes, 1024 * 1024);

    let error: WasmLinearMemoryPolicyError =
        wasm_linear_memory_policy_from_values(Some("not-bytes"), None).unwrap_err();
    assert_eq!(error.variable, WASM_LINEAR_MEMORY_RESERVATION_ENV);
    assert_eq!(error.value, "not-bytes");
    assert_eq!(
        format!("{error}"),
        format!(
            "{} must be an unsigned byte count, got {:?}",
            WASM_LINEAR_MEMORY_RESERVATION_ENV, "not-bytes"
        )
    );
    assert_eq!(
        WASM_LINEAR_MEMORY_GUARD_ENV,
        "MORK_WASM_LINEAR_MEMORY_GUARD_BYTES"
    );
}

#[test]
fn space_api_exposes_whole_atom_algebra() {
    let root = Space::new();
    let mut left = root.fork_empty();
    let mut right = root.fork_empty();

    left.add_all_sexpr(b"(A one)\n(A shared)\n(B keep)")
        .unwrap();
    right
        .add_all_sexpr(b"(A shared)\n(B drop)\n(C two)")
        .unwrap();

    assert_eq!(
        sorted_atoms(&left.atom_union(&right).unwrap()),
        vec![
            "(A one)".to_string(),
            "(A shared)".to_string(),
            "(B drop)".to_string(),
            "(B keep)".to_string(),
            "(C two)".to_string(),
        ]
    );
    assert_eq!(
        sorted_atoms(&left.atom_intersection(&right).unwrap()),
        vec!["(A shared)".to_string()]
    );
    assert_eq!(
        sorted_atoms(&left.atom_subtract(&right).unwrap()),
        vec!["(A one)".to_string(), "(B keep)".to_string()]
    );
}

#[test]
fn sidecar_plan_api_exposes_selected_execution_reports() {
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

    let mut sidecar = TermIdentitySidecar::new();
    assert_eq!(sidecar.extend_from_pathmap(&space.btm).unwrap(), 6);

    let edge = first_relation_functor(&sidecar);
    let descriptor = ArrangementDescriptor::new(edge, 2, [0, 1]).unwrap();
    let mut arrangement = ArrangementIndex::build(&sidecar, descriptor.clone()).unwrap();
    let stats_before_lookup: ArrangementStats = arrangement.stats();
    let mut prefix_rows: Vec<ArrangementRow> = Vec::new();
    let prefix_row_count = arrangement.for_each_prefix_row(&[], |row| prefix_rows.push(row));

    assert_eq!(stats_before_lookup.facts_scanned, 6);
    assert_eq!(stats_before_lookup.rows, 6);
    assert_eq!(prefix_row_count, 6);
    assert_eq!(prefix_rows.len(), 6);
    assert_eq!(arrangement.stats().prefix_lookups, 1);
    assert_eq!(arrangement.stats().prefix_rows_returned, 6);

    let prefix_projection = ArrangementProjection::new(2, [var(0), var(1)], [0, 1]).unwrap();
    let prefix_relation = arrangement
        .project_prefix_bindings(&sidecar, &[], &prefix_projection)
        .unwrap();
    assert_eq!(prefix_relation.positive_rows().count(), 6);
    assert_eq!(arrangement.stats().prefix_lookups, 2);
    assert_eq!(arrangement.stats().prefix_rows_returned, 12);

    let xy = BindingAccessPlan::arrangement(descriptor.clone(), [var(0), var(1)], [0, 1]).unwrap();
    let yz = BindingAccessPlan::arrangement(descriptor, [var(1), var(2)], [0, 1]).unwrap();
    let plan = BindingSidecarPlan::new([xy, yz], [var(0), var(1), var(2)]);
    let prepared = plan.prepare(&sidecar).unwrap();
    let prepared_trie: &PreparedTrieJoin = prepared
        .prepared_trie_join()
        .expect("transitive sidecar should select prepared trie join");
    let count = prepared.count_selected().unwrap();
    let exists = prepared.exists_selected().unwrap();
    let selected = prepared.execute_selected().unwrap();
    let cursor_contract = plan
        .explain_selected_trie_cursor_contract(&sidecar)
        .unwrap();
    let direct_share_plan: BindingParallelSharePlan =
        plan.explain_parallel_share_plan(&sidecar, 8).unwrap();
    let prepared_share_plan: BindingParallelSharePlan = prepared.parallel_share_plan(8);
    let direct_episode_plan: BindingAdaptiveEpisodePlan =
        plan.explain_adaptive_episode_plan(&sidecar, 8, 16).unwrap();
    let prepared_episode_plan: BindingAdaptiveEpisodePlan = prepared.adaptive_episode_plan(8, 16);

    assert_eq!(prepared.relations().len(), 2);
    assert_eq!(
        prepared.execution().choice.kernel,
        BindingSidecarExecutionKernel::TrieJoinSuggested
    );
    assert_eq!(
        prepared_trie.variable_order(),
        prepared.execution().choice.variable_order.as_ref()
    );
    assert_eq!(
        prepared.execution().choice.reason,
        BindingSidecarExecutionReason::SharedVariablePruning
    );
    assert_eq!(
        count.aggregate_kernel,
        BindingSidecarAggregateKernel::TrieJoinCount
    );
    assert!(!count.materialized_output);
    assert_eq!(count.rows, 4);
    assert_eq!(
        exists.aggregate_kernel,
        BindingSidecarAggregateKernel::TrieJoinExists
    );
    assert!(exists.exists);
    assert!(!exists.materialized_output);
    assert_eq!(selected.relation.positive_rows().count(), count.rows);
    assert!(selected.trie_stats.is_some());
    assert!(cursor_contract.cursor_contract.is_some());
    assert_eq!(direct_share_plan, prepared_share_plan);
    assert_eq!(prepared_share_plan.realized_tasks, 8);
    assert_eq!(
        prepared_share_plan.variable_order.as_ref(),
        [var(1), var(0), var(2)]
    );
    assert_eq!(
        prepared_share_plan
            .variables
            .iter()
            .map(|share| share.share)
            .collect::<Vec<_>>(),
        vec![2, 2, 2]
    );
    assert_eq!(direct_episode_plan, prepared_episode_plan);
    assert_eq!(
        prepared_episode_plan.task_boxes.len(),
        prepared_share_plan.realized_tasks
    );
    assert_eq!(prepared_episode_plan.order_estimates.len(), 3);
    assert!(
        prepared_episode_plan
            .order_estimates
            .iter()
            .all(|estimate| estimate.reward_denominator > 0)
    );
}

#[test]
fn pattern_relation_api_exposes_repeated_variable_matching() {
    let (sidecar, pattern_root) = repeated_edge_pattern_sidecar();
    let plan = lower_pattern(&sidecar, pattern_root).unwrap();
    let matches = match_facts(&sidecar, &plan).unwrap();

    assert!(matches!(plan.root(), PlanValue::Variable(_)));
    assert_eq!(plan.user_slot_count(), 1);
    let user_slot = plan.user_slot(0).expect("slot 0 should be lowered");
    assert_eq!(
        plan.variables()[usize::from(user_slot.0)].kind,
        PlanVariableKind::UserSlot(0)
    );
    assert_eq!(plan.atoms().len(), 2);
    assert_eq!(matches.stats.facts_scanned, sidecar.stats().facts);
    assert_eq!(matches.stats.matches, 2);
    assert_eq!(matches.rows.len(), 2);

    let mut bound_terms = matches
        .rows
        .iter()
        .map(|row| {
            assert!(
                sidecar
                    .get_term(row.root)
                    .expect("matched row root should be interned")
                    .flags
                    .ground
            );
            assert_eq!(row.user_bindings.len(), 1);
            assert_eq!(row.user_bindings[0].0, 0);
            row.user_bindings[0].1
        })
        .collect::<Vec<_>>();
    bound_terms.sort();
    bound_terms.dedup();
    assert_eq!(bound_terms.len(), 2);
}

#[test]
fn expression_trie_api_exposes_candidate_filtering_before_exact_matching() {
    let (sidecar, pattern_root) = repeated_edge_pattern_sidecar();
    let index = ExpressionTrieIndex::build(&sidecar).unwrap();
    let stats: ExpressionTrieStats = index.stats();
    let trie_matches: ExpressionTrieMatches = index.match_pattern(&sidecar, pattern_root).unwrap();
    let candidates: &ExpressionTrieCandidates = &trie_matches.candidates;

    assert_eq!(stats.source_generation, sidecar.stats().generation);
    assert_eq!(stats.facts_indexed, sidecar.stats().facts);
    assert!(stats.trie_nodes > 1);
    assert!(stats.tokens_indexed >= stats.facts_indexed);
    assert!(stats.features_indexed >= stats.facts_indexed);

    assert_eq!(candidates.prefix[0], ExpressionTrieToken::App(3));
    assert_eq!(candidates.prefix.len(), 2);
    assert_eq!(candidates.features.len(), 4);
    assert!(candidates.facts.len() < sidecar.stats().facts);
    assert!(
        candidates
            .features
            .iter()
            .any(|feature: &ExpressionFeature| feature.position.as_ref() == [2, 0])
    );

    assert_eq!(
        trie_matches.exact.stats.facts_scanned,
        candidates.facts.len()
    );
    assert_eq!(trie_matches.exact.stats.matches, 2);
    assert_eq!(trie_matches.exact.rows.len(), 2);
}

#[test]
fn json_path_api_exposes_singular_parser_and_value_pattern() {
    let segments = parse_singular_json_path("$.profile['scores'][1]").unwrap();
    assert_eq!(
        segments,
        vec![
            JsonPathSegment::Member("profile".to_string()),
            JsonPathSegment::Member("scores".to_string()),
            JsonPathSegment::Index(1),
        ]
    );
    assert_eq!(
        parse_singular_json_path("$.profile.scores[*]"),
        Err(JsonPathQueryError::UnsupportedSelector { offset: 17 })
    );

    let space = Space::new();
    let pattern = space
        .singular_json_path_value_pattern("$.profile.scores[1]")
        .unwrap();
    let mut sidecar = TermIdentitySidecar::new();
    let pattern_root = sidecar.insert_term(&pattern).unwrap();
    let plan = lower_pattern(&sidecar, pattern_root).unwrap();

    assert!(matches!(plan.root(), PlanValue::Variable(_)));
    assert_eq!(plan.atoms().len(), 3);
    assert_eq!(plan.user_slot_count(), 1);
    let capture = plan.user_slot(0).expect("JSONPath value capture slot");
    assert_eq!(
        plan.variables()[usize::from(capture.0)].kind,
        PlanVariableKind::UserSlot(0)
    );
}

#[test]
fn formal_lowering_api_exposes_exec_summary_and_special_forms() {
    let mut space = Space::new();
    space
        .add_all_sexpr(
            br#"
Empty
NotReducible
()
(chain (eval foo) $x $x)
(exec formal-query
  (I (== (Knowledge $value) $whole))
  (O (+ (query-result $whole))))
"#,
        )
        .unwrap();

    let mut sidecar = TermIdentitySidecar::new();
    sidecar.extend_from_pathmap(&space.btm).unwrap();

    let plan = sidecar
        .facts()
        .iter()
        .find_map(|fact| lower_exec(&sidecar, fact.root).ok())
        .expect("fixture should include one exec fact");
    assert_eq!(plan.source_mode, FormalListMode::SourceList);
    assert_eq!(plan.template_mode, FormalListMode::OutputList);
    assert_eq!(plan.sources.len(), 1);
    assert!(matches!(
        plan.sources[0].kind,
        FormalSourceKind::Equation { .. }
    ));
    assert_eq!(
        plan.summary,
        FormalExecSummary {
            sources: 1,
            equation_sources: 1,
            path_sources: 0,
            add_effects: 1,
            remove_effects: 0,
            binding_sensitive_effects: 1,
            binding_insensitive_effects: 0,
            added_execs: 0,
            later_step_outputs: 1,
            context_holes: 3,
        }
    );

    let forms = sidecar
        .facts()
        .iter()
        .filter_map(|fact| metta_special_form(&sidecar, fact.root).unwrap())
        .collect::<Vec<_>>();
    assert!(forms.contains(&FormalMettaSpecialForm::SpecialResult(
        FormalMettaSpecialResult::Empty
    )));
    assert!(forms.contains(&FormalMettaSpecialForm::SpecialResult(
        FormalMettaSpecialResult::NotReducible
    )));
    assert!(forms.contains(&FormalMettaSpecialForm::SpecialResult(
        FormalMettaSpecialResult::Unit
    )));
    assert!(forms.contains(&FormalMettaSpecialForm::MinimalInstruction(
        FormalMinimalInstruction::Chain
    )));
}

#[test]
fn critical_pair_api_exposes_bounded_witness_rendering() {
    let rules = vec![
        Rule::new(
            "outer-rule",
            Term::app(
                "f",
                vec![Term::app("g", vec![Term::var("x")]), Term::var("z")],
            ),
            Term::app("left", vec![Term::var("x"), Term::var("z")]),
        ),
        Rule::new(
            "inner-rule",
            Term::app("g", vec![Term::var("y")]),
            Term::app("right", vec![Term::var("y")]),
        ),
    ];

    let witness: CriticalPairWitness = first_non_joinable_witness(&rules, 8).unwrap();

    assert_eq!(witness.position_name(), "p0");
    assert_eq!(
        witness.to_metta_atom(),
        "(critical-pair outer-rule inner-rule p0 (f (g a) b) (left a b) (f (right a) b))"
    );
}

#[test]
fn critical_pair_api_exposes_additive_saturation_reports() {
    let program = format!(
        r#"
(edge a b)
(edge b c)
{TRANSITIVE_EDGE_EXEC}"#
    );
    let rules = state_rules_from_mm2_program(&program).unwrap();
    let initial = ground_facts_from_mm2_program(&program).unwrap();

    let report: AdditiveSaturationReport =
        saturate_additive_state_report(initial.clone(), &rules, 4);
    let truncated = saturate_additive_state_report(initial, &rules, 0);

    assert_eq!(report.initial_facts, 2);
    assert_eq!(report.final_facts, 3);
    assert_eq!(report.derived_facts, 1);
    assert_eq!(report.rounds_executed, 2);
    assert!(report.converged);
    assert!(report.state.contains(&critical_fact("edge", &["a", "c"])));
    assert_eq!(
        report.rounds,
        vec![
            AdditiveSaturationRound {
                round: 1,
                state_facts_before: 2,
                generated_steps: 1,
                generated_facts: 1,
                unique_generated_facts: 1,
                candidate_steps: 1,
                candidate_facts: 1,
                new_facts: 1,
                state_facts_after: 3,
                rules: vec![AdditiveSaturationRuleRound {
                    rule: "transitive".to_string(),
                    generated_steps: 1,
                    generated_facts: 1,
                    unique_generated_facts: 1,
                    candidate_steps: 1,
                    candidate_facts: 1,
                    new_facts: 1,
                }],
            },
            AdditiveSaturationRound {
                round: 2,
                state_facts_before: 3,
                generated_steps: 1,
                generated_facts: 1,
                unique_generated_facts: 1,
                candidate_steps: 0,
                candidate_facts: 0,
                new_facts: 0,
                state_facts_after: 3,
                rules: vec![AdditiveSaturationRuleRound {
                    rule: "transitive".to_string(),
                    generated_steps: 1,
                    generated_facts: 1,
                    unique_generated_facts: 1,
                    candidate_steps: 0,
                    candidate_facts: 0,
                    new_facts: 0,
                }],
            },
        ]
    );

    assert_eq!(truncated.final_facts, 2);
    assert_eq!(truncated.rounds_executed, 0);
    assert!(!truncated.converged);
    assert!(truncated.rounds.is_empty());
}

#[test]
fn critical_pair_api_exposes_additive_saturation_redundancy_gaps() {
    let program = r#"
(edge a b)
(edge b c)

(exec duplicate-output
  (, (edge $x $y) (edge $y $z))
  (O (+ (edge $x $z)) (+ (edge $x $z))))

"#;
    let rules = state_rules_from_mm2_program(program).unwrap();
    let initial = ground_facts_from_mm2_program(program).unwrap();

    let report = saturate_additive_state_report(initial, &rules, 4);
    assert_eq!(report.rounds_executed, 2);

    let first_round = &report.rounds[0];
    assert_eq!(first_round.generated_facts, 2);
    assert_eq!(first_round.unique_generated_facts, 1);
    assert_eq!(first_round.new_facts, 1);
    assert_eq!(first_round.duplicate_generated_facts(), 1);
    assert_eq!(first_round.already_known_generated_facts(), 0);
    assert_eq!(first_round.redundant_generated_facts(), 1);

    let first_rule = &first_round.rules[0];
    assert_eq!(first_rule.duplicate_generated_facts(), 1);
    assert_eq!(first_rule.already_known_generated_facts(), 0);
    assert_eq!(first_rule.redundant_generated_facts(), 1);

    let second_round = &report.rounds[1];
    assert_eq!(second_round.generated_facts, 2);
    assert_eq!(second_round.unique_generated_facts, 1);
    assert_eq!(second_round.new_facts, 0);
    assert_eq!(second_round.duplicate_generated_facts(), 1);
    assert_eq!(second_round.already_known_generated_facts(), 1);
    assert_eq!(second_round.redundant_generated_facts(), 2);

    let second_rule = &second_round.rules[0];
    assert_eq!(second_rule.duplicate_generated_facts(), 1);
    assert_eq!(second_rule.already_known_generated_facts(), 1);
    assert_eq!(second_rule.redundant_generated_facts(), 2);
}

#[test]
fn critical_pair_api_exposes_semi_naive_additive_saturation_reports() {
    let program = format!(
        r#"
(edge a b)
(edge b c)
(edge c d)
{TRANSITIVE_EDGE_EXEC}"#
    );
    let rules = state_rules_from_mm2_program(&program).unwrap();
    let initial = ground_facts_from_mm2_program(&program).unwrap();

    let naive = saturate_additive_state_report(initial.clone(), &rules, 8);
    let semi_naive = saturate_additive_state_semi_naive_report(initial.clone(), &rules, 8);
    let comparison: AdditiveSaturationComparisonReport =
        compare_additive_saturation_reports(initial.clone(), &rules, 8);

    assert_eq!(&semi_naive.state, &naive.state);
    assert_eq!(
        saturate_additive_state_semi_naive(initial, &rules, 8),
        naive.state
    );
    assert_eq!(semi_naive.initial_facts, 3);
    assert_eq!(semi_naive.final_facts, 6);
    assert_eq!(semi_naive.derived_facts, 3);
    assert_eq!(semi_naive.rounds_executed, 3);
    assert!(semi_naive.converged);
    assert_eq!(
        semi_naive.rounds,
        vec![
            AdditiveSaturationRound {
                round: 1,
                state_facts_before: 3,
                generated_steps: 2,
                generated_facts: 2,
                unique_generated_facts: 2,
                candidate_steps: 2,
                candidate_facts: 2,
                new_facts: 2,
                state_facts_after: 5,
                rules: vec![AdditiveSaturationRuleRound {
                    rule: "transitive".to_string(),
                    generated_steps: 2,
                    generated_facts: 2,
                    unique_generated_facts: 2,
                    candidate_steps: 2,
                    candidate_facts: 2,
                    new_facts: 2,
                }],
            },
            AdditiveSaturationRound {
                round: 2,
                state_facts_before: 5,
                generated_steps: 2,
                generated_facts: 2,
                unique_generated_facts: 1,
                candidate_steps: 1,
                candidate_facts: 1,
                new_facts: 1,
                state_facts_after: 6,
                rules: vec![AdditiveSaturationRuleRound {
                    rule: "transitive".to_string(),
                    generated_steps: 2,
                    generated_facts: 2,
                    unique_generated_facts: 1,
                    candidate_steps: 1,
                    candidate_facts: 1,
                    new_facts: 1,
                }],
            },
            AdditiveSaturationRound {
                round: 3,
                state_facts_before: 6,
                generated_steps: 0,
                generated_facts: 0,
                unique_generated_facts: 0,
                candidate_steps: 0,
                candidate_facts: 0,
                new_facts: 0,
                state_facts_after: 6,
                rules: Vec::new(),
            },
        ]
    );
    assert!(semi_naive.rounds[1].generated_facts < naive.rounds[1].generated_facts);
    assert!(comparison.same_final_state());
    assert!(comparison.same_convergence_status());
    assert_eq!(comparison.naive, naive);
    assert_eq!(comparison.semi_naive, semi_naive);
    assert_eq!(
        comparison.semi_naive_savings,
        AdditiveSaturationWorkSavings {
            generated_steps: 6,
            generated_facts: 6,
            unique_generated_facts: 5,
            candidate_steps: 0,
            candidate_facts: 0,
            new_facts: 0,
        }
    );
    assert!(comparison.semi_naive_savings.any());
    assert_eq!(
        comparison.round_savings,
        vec![
            AdditiveSaturationRoundWorkSavings {
                round: 1,
                savings: AdditiveSaturationWorkSavings {
                    generated_steps: 0,
                    generated_facts: 0,
                    unique_generated_facts: 0,
                    candidate_steps: 0,
                    candidate_facts: 0,
                    new_facts: 0,
                },
            },
            AdditiveSaturationRoundWorkSavings {
                round: 2,
                savings: AdditiveSaturationWorkSavings {
                    generated_steps: 2,
                    generated_facts: 2,
                    unique_generated_facts: 2,
                    candidate_steps: 0,
                    candidate_facts: 0,
                    new_facts: 0,
                },
            },
            AdditiveSaturationRoundWorkSavings {
                round: 3,
                savings: AdditiveSaturationWorkSavings {
                    generated_steps: 4,
                    generated_facts: 4,
                    unique_generated_facts: 3,
                    candidate_steps: 0,
                    candidate_facts: 0,
                    new_facts: 0,
                },
            },
        ]
    );
    assert!(!comparison.round_savings[0].any());
    assert!(comparison.round_savings[1].any());
}

#[test]
fn binding_env_api_exposes_fixed_slot_scoped_rollback() {
    let mut env = BindingEnv::new();

    assert_eq!(MAX_BINDING_SLOTS, 64);
    assert!(env.is_empty());
    assert_eq!(env.bind_term(1, term(10)), Ok(true));
    assert_eq!(env.bind_term(1, term(10)), Ok(true));
    assert_eq!(env.trail_len(), 1);

    let mark = env.mark();
    assert_eq!(env.bind_term(2, term(20)), Ok(true));
    assert_eq!(
        env.bind(
            3,
            Binding {
                term: term(30),
                fact: Some(FactId(7)),
                source: BindingSource(42),
            },
        ),
        Ok(true)
    );
    assert_eq!(env.bind_term(2, term(21)), Ok(false));
    assert_eq!(env.get(2), Ok(Some(Binding::from_term(term(20)))));

    let snapshot: BindingSnapshot = env.capture();
    let scoped_len = env
        .with_rollback(|env| {
            assert_eq!(env.bind_term(4, term(40)), Ok(true));
            env.len()
        })
        .unwrap();

    assert_eq!(scoped_len, 4);
    assert_eq!(env.get(4), Ok(None));
    assert_eq!(env.len(), 3);
    assert_eq!(
        env.iter_bound()
            .map(|(slot, binding)| (slot, binding.term))
            .collect::<Vec<_>>(),
        vec![(1, term(10)), (2, term(20)), (3, term(30))]
    );

    assert_eq!(env.rollback_to(mark), Ok(()));
    assert_eq!(env.get(1), Ok(Some(Binding::from_term(term(10)))));
    assert_eq!(env.get(2), Ok(None));
    assert_eq!(env.get(3), Ok(None));

    env.restore(&snapshot);
    assert_eq!(env.trail_len(), 0);
    assert_eq!(env.len(), 3);
    assert_eq!(
        env.bind_term(64, term(64)),
        Err(BindingEnvError::SlotOutOfRange { slot: 64 })
    );
}

#[test]
fn weighted_path_api_exposes_sidecar_selection_tree() -> Result<(), WeightedPathError> {
    let mut index = WeightedPathIndex::new();

    index.apply_delta(b"edge/Alice/Bob", 2)?;
    index.apply_delta(b"edge/Alice/Carol", 3)?;
    index.apply_delta(b"edge/Dana/Erin", -4)?;

    assert_eq!(index.total_positive_weight(), 5);
    assert_eq!(index.weight(b"edge/Dana/Erin"), -4);
    assert_eq!(
        index.stats(),
        WeightedPathStats {
            entries: 3,
            positive_entries: 2,
            non_positive_entries: 1,
            total_positive_weight: 5,
            updates: 3,
        }
    );

    let tree: WeightedSelectionTree = index.selection_tree()?;
    let tree_stats: WeightedSelectionTreeStats = tree.stats();

    assert_eq!(tree.total_positive_weight(), index.total_positive_weight());
    assert_eq!(tree_stats.total_positive_weight, 5);
    assert_eq!(tree_stats.positive_value_nodes, 2);
    assert!(tree_stats.nodes >= index.stats().entries);
    assert!(tree_stats.child_edges > 0);

    for offset in 0..index.total_positive_weight() {
        assert_eq!(
            tree.select_by_offset(offset),
            index.select_by_offset(offset),
            "tree selector should match linear selector at offset {offset}"
        );
    }
    assert_eq!(tree.select_by_offset(5), None);
    assert_eq!(
        index.apply_delta(b"edge/Alice/Bob", i64::MAX),
        Err(WeightedPathError::WeightOverflow {
            current: 2,
            delta: i64::MAX,
        })
    );
    Ok(())
}

#[cfg(feature = "experimental_dnf")]
#[test]
fn dnf_api_exposes_feature_gated_pathset_evaluator() {
    let left = path_map(&[b"alpha/red", b"alpha/blue"]);
    let middle = path_map(&[b"alpha/red", b"beta"]);
    let right = path_map(&[b"alpha/blue", b"gamma"]);

    let result = evaluate_pathmap_dnf(&[&[&left, &middle], &[&left, &right]]).unwrap();
    let fast_result =
        evaluate_pathmap_dnf_zipper_merge(&[&[&left, &middle], &[&left, &right]]).unwrap();

    assert_eq!(
        sorted_paths(&result.map),
        vec![b"alpha/blue".to_vec(), b"alpha/red".to_vec()]
    );
    assert_eq!(sorted_paths(&fast_result), sorted_paths(&result.map));

    let mut z_left_for_middle = left.read_zipper();
    let mut z_middle = middle.read_zipper();
    let mut z_left_for_right = left.read_zipper();
    let mut z_right = right.read_zipper();
    let mut zipper_result = PathMap::new();
    zipper_merge_dnf(
        &mut [
            &mut [&mut z_left_for_middle, &mut z_middle],
            &mut [&mut z_left_for_right, &mut z_right],
        ],
        &mut zipper_result.write_zipper(),
    );
    assert_eq!(sorted_paths(&zipper_result), sorted_paths(&result.map));

    assert_eq!(result.stats.clauses, 2);
    assert_eq!(result.stats.factors, 4);
    assert_eq!(result.stats.factors_evaluated, 4);
    assert_eq!(result.stats.short_circuit_skipped_factors, 0);
    assert_eq!(result.stats.meet_ops, 2);
    assert_eq!(result.stats.join_ops, 1);
    assert_eq!(result.stats.distinct_factor_refs, 3);
    assert_eq!(result.stats.repeated_factor_refs, 1);
    assert_eq!(result.stats.factor_input_values, 8);
    assert_eq!(result.stats.clause_output_values, 2);
    assert_eq!(result.stats.duplicate_clause_values, 0);
    assert_eq!(result.stats.empty_clause_results, 0);
    assert_eq!(result.stats.non_empty_clause_results, 2);
    assert_eq!(result.stats.count_disjoint_clause_additions, 0);
    assert_eq!(result.stats.count_overlap_check_ops, 0);
    assert_eq!(result.stats.peak_clause_values, 1);
    assert_eq!(result.stats.peak_result_values, 2);
    assert_eq!(
        evaluate_pathmap_dnf(&[&[&left], &[]]).unwrap_err(),
        DnfPathSetError::EmptyClause { clause_index: 1 }
    );

    let existence: DnfPathSetExistenceResult =
        evaluate_pathmap_dnf_exists(&[&[&left, &middle], &[&right]]).unwrap();
    assert!(existence.exists);
    assert!(!existence.final_output_materialized);
    assert_eq!(existence.stats.clauses, 2);
    assert_eq!(existence.stats.clauses_evaluated, 1);
    assert_eq!(existence.stats.short_circuit_skipped_clauses, 1);
    assert_eq!(existence.stats.factors, 2);
    assert_eq!(existence.stats.factors_evaluated, 2);
    assert_eq!(existence.stats.short_circuit_skipped_factors, 0);
    assert_eq!(existence.stats.meet_ops, 1);
    assert_eq!(existence.stats.join_ops, 0);

    let count: DnfPathSetCountResult =
        evaluate_pathmap_dnf_count(&[&[&left], &[&left, &middle]]).unwrap();
    assert_eq!(count.count, left.val_count());
    assert!(count.final_output_materialized);
    assert_eq!(count.stats.clauses, 2);
    assert_eq!(count.stats.clauses_evaluated, 2);
    assert_eq!(count.stats.short_circuit_skipped_clauses, 0);
    assert_eq!(count.stats.factors, 3);
    assert_eq!(count.stats.factors_evaluated, 3);
    assert_eq!(count.stats.short_circuit_skipped_factors, 0);
    assert_eq!(count.stats.meet_ops, 1);
    assert_eq!(count.stats.join_ops, 1);
    assert_eq!(count.stats.clause_output_values, 3);
    assert_eq!(count.stats.duplicate_clause_values, 1);
    assert_eq!(count.stats.empty_clause_results, 0);
    assert_eq!(count.stats.non_empty_clause_results, 2);
    assert_eq!(count.stats.count_disjoint_clause_additions, 0);
    assert_eq!(count.stats.count_overlap_check_ops, 1);
    let overflow: DnfPathSetError = DnfPathSetError::CardinalityOverflow {
        left: usize::MAX,
        right: 1,
    };
    assert!(matches!(
        overflow,
        DnfPathSetError::CardinalityOverflow {
            left: usize::MAX,
            right: 1,
        }
    ));
}
