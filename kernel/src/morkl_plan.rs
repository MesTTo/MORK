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
//! The PR3 dispatch must memoize analysis by the stable rule key formed from pattern bytes followed
//! by template bytes, following `sni_rule_seen`. Any dispatch change must pass `ai-bench-guard` in
//! every feature configuration. Declining the plan on a small firing must not add repeated analysis
//! cost.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::union_find::{UnionFind, UnionFindId};
use crate::zipper_join::{Factor, collect_factor_vars};

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
struct BodyPlanData {
    component_offsets: Box<[usize]>,
    factor_indices: Box<[usize]>,
    var_components: Box<[Option<usize>]>,
}

/// Independent factor components and the component that owns each query variable.
///
/// Components use factor indices from the parsed body. The flat, shared representation keeps
/// construction allocation-light and makes a cached plan constant-time to clone.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BodyPlan(Arc<BodyPlanData>);

impl BodyPlan {
    /// Number of independent body components.
    pub fn component_count(&self) -> usize {
        self.0.component_offsets.len() - 1
    }

    /// Factors in one component, in body order.
    pub fn component(&self, component: usize) -> Option<&[usize]> {
        let start = *self.0.component_offsets.get(component)?;
        let end = *self.0.component_offsets.get(component + 1)?;
        Some(&self.0.factor_indices[start..end])
    }

    /// Every component in first-factor order.
    pub fn components(&self) -> impl ExactSizeIterator<Item = &[usize]> + '_ {
        self.0
            .component_offsets
            .windows(2)
            .map(|range| &self.0.factor_indices[range[0]..range[1]])
    }

    /// Component containing `var`, or `None` when the parsed body does not use it.
    pub fn var_component(&self, var: usize) -> Option<usize> {
        self.0.var_components.get(var).copied().flatten()
    }

    /// Component ownership for all query-variable ids below the parser's `nvars`.
    pub fn var_components(&self) -> &[Option<usize>] {
        &self.0.var_components
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

    BodyPlan(Arc::new(BodyPlanData {
        component_offsets: component_offsets.into_boxed_slice(),
        factor_indices: factor_indices.into_boxed_slice(),
        var_components: var_components.into_boxed_slice(),
    }))
}

#[cfg(test)]
mod tests {
    use super::{BodyPlan, partition_components};
    use crate::space::Space;
    use crate::zipper_join::parse_body_factors;

    fn plan(source: &str) -> BodyPlan {
        let space = Space::new();
        let body = crate::expr!(space, source);
        let span = unsafe { body.span().as_ref().unwrap() };
        let (factors, nvars) = parse_body_factors(span).expect("body must parse");
        partition_components(&factors, nvars)
    }

    #[test]
    fn partition_components_follows_shared_query_variables() {
        let cases: &[(&str, &[&[usize]], &[Option<usize>])] = &[
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
}
