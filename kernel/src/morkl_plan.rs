//! Pure symbolic analysis for separable MORKL exec bodies.
//!
//! The partition is over query variables from the encoded body pattern. Variables stored in data
//! do not connect body factors. Output templates form the plan's union branches; execution remains
//! in the existing unifying join and transform paths.
//!
//! Admission only proves that a body and its templates have a supported dependency shape. It does
//! not prove that the whole body has a solution. The planned transform must existence-check every
//! component before emitting any template, including a ground template.
//!
//! Any dispatch change must pass `ai-bench-guard` in every feature configuration.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use mork_expr::{Tag, maybe_byte_item};
use pathmap::PathMap;

use crate::union_find::{UnionFind, UnionFindId};
use crate::zipper_join::{
    DISPATCH_MIN_FACTS, Factor, bounded_fact_count, collect_factor_vars,
    factor_count_lower_bounds_single_factor_solutions,
};

/// Number of planned-route firings that emitted at least one template application.
pub static PLANNED_FIRINGS: AtomicUsize = AtomicUsize::new(0);

/// Read the process-wide planned-route emission count.
pub fn planned_firings() -> usize {
    PLANNED_FIRINGS.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn record_planned_firing() {
    PLANNED_FIRINGS.fetch_add(1, Ordering::Relaxed);
}

/// Planned-transform dispatch mode: off, the default structurally gated policy, or every
/// structurally admissible body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DispatchMode {
    Off,
    Gated,
    All,
}

impl DispatchMode {
    fn from_env(value: Option<&str>) -> Self {
        match value {
            Some("0") => Self::Off,
            Some("all") => Self::All,
            _ => Self::Gated,
        }
    }
}

thread_local! {
    /// Per-thread planned-transform dispatch mode. `MORK_MORKL_PLAN=0` turns dispatch off at
    /// thread start, `MORK_MORKL_PLAN=all` selects every structurally admissible body, and every
    /// other value uses the default gated policy. The environment is read once when each thread
    /// first accesses the knob. Thread-local state lets differentials pin one run to stock without
    /// racing parallel tests.
    static MORKL_PLAN_DISPATCH: std::cell::Cell<DispatchMode> = std::cell::Cell::new(
        DispatchMode::from_env(std::env::var("MORK_MORKL_PLAN").as_deref().ok()),
    );
}

/// Whether structurally admitted transforms may use the planned route on this thread.
pub fn morkl_plan_dispatch_enabled() -> bool {
    MORKL_PLAN_DISPATCH.with(|mode| mode.get()) != DispatchMode::Off
}

/// Whether this thread uses the default profitability gate rather than the explicit `all` mode.
pub(crate) fn morkl_plan_profitability_required() -> bool {
    MORKL_PLAN_DISPATCH.with(|mode| mode.get()) == DispatchMode::Gated
}

/// Turn the default structurally gated planned route on or off for this thread.
/// Differentials use this to keep the reference run on the stock route.
pub fn set_morkl_plan_dispatch(on: bool) {
    MORKL_PLAN_DISPATCH.with(|mode| {
        mode.set(if on {
            DispatchMode::Gated
        } else {
            DispatchMode::Off
        })
    })
}

/// Force structural admission for the current thread, bypassing only the profitability gate.
/// Differential and scaling tests use this to exercise the planned executor on small fixtures.
pub fn set_morkl_plan_dispatch_all() {
    MORKL_PLAN_DISPATCH.with(|mode| mode.set(DispatchMode::All));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FactorId(u32);

impl UnionFindId for FactorId {
    fn from_index(index: u32) -> Self {
        Self(index)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct BodyPlan {
    component_offsets: Box<[usize]>,
    factor_indices: Box<[usize]>,
    var_components: Box<[Option<usize>]>,
}

/// Independent factor components and the component that owns each query variable.
///
/// Components use factor indices from the parsed body. The flat representation keeps construction
/// allocation-light.

impl BodyPlan {
    /// Number of independent body components.
    pub fn component_count(&self) -> usize {
        self.component_offsets.len() - 1
    }

    /// Factors in one component, in body order.
    pub fn component(&self, component: usize) -> Option<&[usize]> {
        let start = *self.component_offsets.get(component)?;
        let end = *self.component_offsets.get(component + 1)?;
        Some(&self.factor_indices[start..end])
    }

    /// Every component in first-factor order.
    pub fn components(&self) -> impl ExactSizeIterator<Item = &[usize]> + '_ {
        self.component_offsets
            .windows(2)
            .map(|range| &self.factor_indices[range[0]..range[1]])
    }

    /// Component containing `var`, or `None` when the parsed body does not use it.
    pub fn var_component(&self, var: usize) -> Option<usize> {
        self.var_components.get(var).copied().flatten()
    }

    /// Component ownership for all query-variable ids below the parser's `nvars`.
    pub fn var_components(&self) -> &[Option<usize>] {
        &self.var_components
    }
}

/// Partition parsed body factors by shared query variables.
///
/// A fully ground factor has no query-variable edge and forms its own existential component.
pub fn partition_components(factors: &[Factor], nvars: usize) -> BodyPlan {
    debug_assert!(factors.len() <= u32::MAX as usize);

    let mut classes = UnionFind::<FactorId>::new();
    for _ in factors {
        classes.make_set();
    }

    let mut first_factor_by_var = vec![None; nvars];
    let mut factor_vars = BTreeSet::new();
    for (factor_index, factor) in factors.iter().enumerate() {
        factor_vars.clear();
        collect_factor_vars(factor, &mut factor_vars);
        let factor_id = FactorId(factor_index as u32);
        for &var in &factor_vars {
            debug_assert!(var < nvars);
            match first_factor_by_var[var] {
                Some(first) => {
                    classes.union(first, factor_id);
                }
                None => first_factor_by_var[var] = Some(factor_id),
            }
        }
    }

    let mut root_to_component = vec![usize::MAX; factors.len()];
    let mut factor_components = Vec::with_capacity(factors.len());
    let mut component_count = 0;
    for factor_index in 0..factors.len() {
        let root = classes.find(FactorId(factor_index as u32)).index();
        let component = &mut root_to_component[root];
        if *component == usize::MAX {
            *component = component_count;
            component_count += 1;
        }
        factor_components.push(*component);
    }

    let mut component_sizes = root_to_component;
    component_sizes.truncate(component_count);
    component_sizes.fill(0);
    for &component in &factor_components {
        component_sizes[component] += 1;
    }

    let mut component_offsets = Vec::with_capacity(component_count + 1);
    component_offsets.push(0);
    let mut next_offset = 0;
    for &size in &component_sizes {
        next_offset += size;
        component_offsets.push(next_offset);
    }

    let mut factor_indices = vec![0; factors.len()];
    component_sizes.fill(0);
    for (factor_index, &component) in factor_components.iter().enumerate() {
        let destination = component_offsets[component] + component_sizes[component];
        factor_indices[destination] = factor_index;
        component_sizes[component] += 1;
    }

    let var_components = first_factor_by_var
        .into_iter()
        .map(|factor| factor.map(|factor| factor_components[factor.index()]))
        .collect::<Vec<_>>();

    BodyPlan {
        component_offsets: component_offsets.into_boxed_slice(),
        factor_indices: factor_indices.into_boxed_slice(),
        var_components: var_components.into_boxed_slice(),
    }
}

/// Dependency class of one output template.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemplateClass {
    /// No variable occurrence. Emit once only after every body component is known nonempty.
    Ground,
    /// Every referenced query variable belongs to one body component.
    SingleComponent(usize),
    /// Referenced query variables belong to more than one component.
    Mixed,
    /// Contains a fresh `NewVar`, an unowned `VarRef`, or invalid encoded bytes.
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TemplateVars {
    Unsupported,
    Refs(u64),
}

/// Scan one exact encoded template without allocating. Symbol payload bytes must be skipped because
/// their high bits overlap tag encodings.
fn scan_template_vars(span: &[u8]) -> TemplateVars {
    if span.is_empty() {
        return TemplateVars::Unsupported;
    }

    let mut refs = 0u64;
    let mut offset = 0;
    while offset < span.len() {
        let Ok(tag) = maybe_byte_item(span[offset]) else {
            return TemplateVars::Unsupported;
        };
        match tag {
            Tag::NewVar => return TemplateVars::Unsupported,
            Tag::VarRef(var) => {
                refs |= 1u64 << var;
                offset += 1;
            }
            Tag::SymbolSize(size) => {
                let next = offset + 1 + size as usize;
                if next > span.len() {
                    return TemplateVars::Unsupported;
                }
                offset = next;
            }
            Tag::Arity(_) => offset += 1,
        }
    }
    TemplateVars::Refs(refs)
}

/// Classify a template against query-variable ownership in `body`.
pub fn classify_template(span: &[u8], body: &BodyPlan) -> TemplateClass {
    let TemplateVars::Refs(mut refs) = scan_template_vars(span) else {
        return TemplateClass::Unsupported;
    };
    if refs == 0 {
        return TemplateClass::Ground;
    }

    let mut owner = None;
    while refs != 0 {
        let var = refs.trailing_zeros() as usize;
        refs &= refs - 1;
        let Some(component) = body.var_component(var) else {
            return TemplateClass::Unsupported;
        };
        match owner {
            None => owner = Some(component),
            Some(first) if first != component => return TemplateClass::Mixed,
            Some(_) => {}
        }
    }
    let Some(owner) = owner else {
        return TemplateClass::Unsupported;
    };
    TemplateClass::SingleComponent(owner)
}

#[derive(Debug, Eq, PartialEq)]
pub struct MorklJoinPlan {
    body: BodyPlan,
    template_classes: Box<[TemplateClass]>,
    factor_counts: Option<Box<[usize]>>,
}

impl MorklJoinPlan {
    /// Partition of the body factors.
    pub fn body(&self) -> &BodyPlan {
        &self.body
    }

    /// Classification corresponding positionally to the analyzed template spans.
    pub fn template_classes(&self) -> &[TemplateClass] {
        &self.template_classes
    }

    /// Capped relation counts retained by gated analysis for deterministic probe ordering.
    pub(crate) fn factor_counts(&self) -> Option<&[usize]> {
        self.factor_counts.as_deref()
    }
}

/// Admit a pre-partitioned body when every template is ground or component-local.
///
/// Declined templates allocate nothing. Successful classification is repeated once to build the
/// owned result after the allocation-free admission pass.
pub fn admit(body: BodyPlan, template_spans: &[&[u8]]) -> Option<MorklJoinPlan> {
    if body.component_count() < 2 {
        return None;
    }
    if template_spans.iter().any(|span| {
        !matches!(
            classify_template(span, &body),
            TemplateClass::Ground | TemplateClass::SingleComponent(_)
        )
    }) {
        return None;
    }

    let template_classes = template_spans
        .iter()
        .map(|span| classify_template(span, &body))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Some(MorklJoinPlan {
        body,
        template_classes,
        factor_counts: None,
    })
}

fn analysis_preflight(factors: &[Factor], template_spans: &[&[u8]]) -> bool {
    factors.len() >= 2
        && !template_spans
            .iter()
            .any(|span| scan_template_vars(span) == TemplateVars::Unsupported)
}

fn analyze_preflighted(
    factors: &[Factor],
    nvars: usize,
    template_spans: &[&[u8]],
) -> Option<MorklJoinPlan> {
    let body = partition_components(factors, nvars);
    admit(body, template_spans)
}

/// Whether a factorised projection has enough certified work to avoid before it is admitted.
///
/// For a Cartesian product, projection can be pushed through an independent component with an
/// emptiness guard: `pi_A(A x B) = A` when `B` is nonempty, and `{}` otherwise. This is the
/// factorised result bound from Olteanu and Zavodny, ACM TODS 2015, doi:10.1145/2656335. A flat
/// template writes across the product; the plan writes each component-local template only across
/// its component and each ground template once.
///
/// A relation count contributes only for a single-factor component whose scan is a certified lower
/// bound on solutions. Every other shape returns zero and therefore declines. Capping a positive
/// lower bound preserves the direction of the estimate because avoided applications are monotone
/// in every independent component size. This intentionally misses possible wins rather than admit
/// a relation whose joins or filters could collapse its apparent size.
fn profitable(factor_counts: &[usize], factors: &[Factor], plan: &MorklJoinPlan) -> bool {
    let mut component_lower_bounds = Vec::with_capacity(plan.body().component_count());
    for component in plan.body().components() {
        let [factor_index] = component else {
            return false;
        };
        let lower_bound = factor_counts[*factor_index];
        if lower_bound == 0 {
            return false;
        }
        component_lower_bounds.push(lower_bound);
    }

    let flat_applications = component_lower_bounds
        .iter()
        .copied()
        .fold(plan.template_classes().len(), usize::saturating_mul);
    let factorised_applications = plan
        .template_classes()
        .iter()
        .map(|class| match class {
            TemplateClass::Ground => 1,
            TemplateClass::SingleComponent(component) => component_lower_bounds[*component],
            TemplateClass::Mixed | TemplateClass::Unsupported => unreachable!(),
        })
        .fold(0usize, usize::saturating_add);

    flat_applications.saturating_sub(factorised_applications) >= DISPATCH_MIN_FACTS
}

/// Partition and admit a parsed body as a pure function of the map, rule shape, and templates.
///
/// The constant-time factor-count check and allocation-free `NewVar` preflight precede trie reads.
/// The bounded counts run before component construction. Every relation below the existing
/// [`DISPATCH_MIN_FACTS`] boundary declines, matching the measured tile-puzzle boundary where
/// 56-fact inequality tables lost on extra join machinery.
pub fn analyze(
    map: &PathMap<()>,
    factors: &[Factor],
    nvars: usize,
    template_spans: &[&[u8]],
) -> Option<MorklJoinPlan> {
    if !analysis_preflight(factors, template_spans) {
        return None;
    }
    if !factors
        .iter()
        .all(factor_count_lower_bounds_single_factor_solutions)
    {
        return None;
    }

    let factor_counts = factors
        .iter()
        .map(|factor| bounded_fact_count(map, factor, DISPATCH_MIN_FACTS))
        .collect::<Vec<_>>();
    if factor_counts
        .iter()
        .all(|count| *count < DISPATCH_MIN_FACTS)
    {
        return None;
    }

    let mut plan = analyze_preflighted(factors, nvars, template_spans)?;
    if !profitable(&factor_counts, factors, &plan) {
        return None;
    }
    plan.factor_counts = Some(factor_counts.into_boxed_slice());
    Some(plan)
}

/// Structural admission for the explicit `MORK_MORKL_PLAN=all` diagnostic mode.
pub(crate) fn analyze_ungated(
    factors: &[Factor],
    nvars: usize,
    template_spans: &[&[u8]],
) -> Option<MorklJoinPlan> {
    analysis_preflight(factors, template_spans)
        .then(|| analyze_preflighted(factors, nvars, template_spans))
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::{
        BodyPlan, DispatchMode, TemplateClass, admit, analyze, classify_template,
        morkl_plan_dispatch_enabled, partition_components, set_morkl_plan_dispatch,
    };
    use crate::space::Space;
    use crate::zipper_join::{Factor, parse_body_factors};

    type PartitionCase = (
        &'static str,
        &'static [&'static [usize]],
        &'static [Option<usize>],
    );

    fn parsed_body(source: &str) -> (Vec<Factor>, usize) {
        let space = Space::new();
        let body = crate::expr!(space, source);
        let span = unsafe { body.span().as_ref().unwrap() };
        parse_body_factors(span).expect("body must parse")
    }

    #[test]
    fn dispatch_mode_defaults_to_gated_and_honors_environment_values() {
        assert_eq!(DispatchMode::from_env(None), DispatchMode::Gated);
        assert_eq!(DispatchMode::from_env(Some("0")), DispatchMode::Off);
        assert_eq!(DispatchMode::from_env(Some("all")), DispatchMode::All);
        assert_eq!(DispatchMode::from_env(Some("unknown")), DispatchMode::Gated);
    }

    #[test]
    fn dispatch_override_toggles_current_thread() {
        set_morkl_plan_dispatch(false);
        assert!(!morkl_plan_dispatch_enabled());

        set_morkl_plan_dispatch(true);
        assert!(morkl_plan_dispatch_enabled());
    }

    fn plan(source: &str) -> BodyPlan {
        let (factors, nvars) = parsed_body(source);
        partition_components(&factors, nvars)
    }

    fn encoded(source: &str) -> Vec<u8> {
        let space = Space::new();
        let expression = crate::expr!(space, source);
        unsafe { expression.span().as_ref().unwrap() }.to_vec()
    }

    #[test]
    fn partition_components_follows_shared_query_variables() {
        let cases: &[PartitionCase] = &[
            ("[3] , [2] a $ [2] b $", &[&[0], &[1]], &[Some(0), Some(1)]),
            ("[3] , [2] a $ [2] b _1", &[&[0, 1]], &[Some(0)]),
            (
                "[4] , [2] a $ [2] b $ [3] lt _1 _2",
                &[&[0, 1, 2]],
                &[Some(0), Some(0)],
            ),
            ("[3] , [2] a 1 [2] b $", &[&[0], &[1]], &[Some(1)]),
        ];

        for (source, expected_components, expected_var_components) in cases {
            let plan = plan(source);
            assert_eq!(
                plan.components().collect::<Vec<_>>(),
                *expected_components,
                "body: {}",
                source
            );
            assert_eq!(
                plan.var_components(),
                *expected_var_components,
                "body: {}",
                source
            );
        }
    }

    #[test]
    fn classify_template_covers_f1_f2_and_f8_shapes() {
        let body = plan("[3] , [2] a $ [2] b $");
        let cases = [
            ("[2] seen-a _1", TemplateClass::SingleComponent(0)),
            ("[1] done", TemplateClass::Ground),
            ("[3] pair _1 _2", TemplateClass::Mixed),
            ("[2] fresh $", TemplateClass::Unsupported),
        ];

        for (source, expected) in cases {
            assert_eq!(
                classify_template(&encoded(source), &body),
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn admit_accepts_f1_supported_templates() {
        let body = plan("[3] , [2] a $ [2] b $");
        let templates = [encoded("[2] seen-a _1"), encoded("[1] done")];
        let spans = templates.iter().map(Vec::as_slice).collect::<Vec<_>>();

        let admitted = admit(body, &spans).expect("F1 must be admitted");

        assert_eq!(
            admitted.template_classes(),
            &[TemplateClass::SingleComponent(0), TemplateClass::Ground]
        );
    }

    #[test]
    fn admit_rejects_single_component_body() {
        let body = plan("[3] , [2] a $ [2] b _1");
        let templates = [encoded("[2] seen _1")];
        let spans = templates.iter().map(Vec::as_slice).collect::<Vec<_>>();

        assert!(admit(body, &spans).is_none());
    }

    fn analysis_is_rejected(body: &str, template: &str) -> bool {
        let (factors, nvars) = parsed_body(body);
        let template = encoded(template);
        let map = Space::new();
        analyze(&map.btm, &factors, nvars, &[template.as_slice()]).is_none()
    }

    #[test]
    fn analyze_rejects_one_factor_before_partitioning() {
        assert!(analysis_is_rejected("[2] , [2] a $", "[2] seen _1"));
    }

    #[test]
    fn analyze_rejects_new_var_before_partitioning() {
        assert!(analysis_is_rejected("[3] , [2] a $ [2] b $", "[2] fresh $"));
    }

    fn product_is_admitted(a_count: usize, b_count: usize) -> bool {
        let mut space = Space::new();
        let mut facts = String::new();
        for value in 0..a_count {
            facts.push_str(&format!("(a a{value})\n"));
        }
        for value in 0..b_count {
            facts.push_str(&format!("(b b{value})\n"));
        }
        space.add_all_sexpr(facts.as_bytes()).unwrap();
        let body = crate::expr!(space, "[3] , [2] a $ [2] b $");
        let body_span = unsafe { body.span().as_ref().unwrap() };
        let (factors, nvars) = parse_body_factors(body_span).unwrap();
        let templates = [encoded("[2] seen-a _1"), encoded("[1] done")];
        let spans = templates.iter().map(Vec::as_slice).collect::<Vec<_>>();
        analyze(&space.btm, &factors, nvars, &spans).is_some()
    }

    fn encoded_body_passes_cardinality_preflight(body: &str) -> bool {
        let mut space = Space::new();
        let body = crate::expr!(space, body);
        let span = unsafe { body.span().as_ref().unwrap() };
        crate::zipper_join::body_has_independent_full_relation_scans(span)
    }

    #[test]
    fn encoded_cardinality_preflight_accepts_only_independent_full_relation_scans() {
        assert!(encoded_body_passes_cardinality_preflight(
            "[3] , [2] a $ [2] b $"
        ));
        assert!(!encoded_body_passes_cardinality_preflight(
            "[3] , [2] a $ [2] b _1"
        ));
        assert!(!encoded_body_passes_cardinality_preflight(
            "[3] , [2] a [2] f $ [2] b $"
        ));
        assert!(!encoded_body_passes_cardinality_preflight(
            "[3] , [3] a $ _1 [2] b $"
        ));
    }

    #[test]
    fn analyze_declines_when_every_relation_is_below_the_measured_boundary() {
        assert!(!product_is_admitted(
            crate::zipper_join::DISPATCH_MIN_FACTS - 1,
            crate::zipper_join::DISPATCH_MIN_FACTS - 1,
        ));
    }

    #[test]
    fn analyze_declines_when_avoided_applications_stay_below_the_boundary() {
        assert!(!product_is_admitted(
            crate::zipper_join::DISPATCH_MIN_FACTS,
            1,
        ));
    }

    #[test]
    fn analyze_admits_when_bounded_product_savings_cross_the_boundary() {
        assert!(product_is_admitted(
            crate::zipper_join::DISPATCH_MIN_FACTS,
            2,
        ));
    }
}
