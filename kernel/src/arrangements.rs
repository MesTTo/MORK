use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::convert::Infallible;

use crate::binding_space::{BindingRelation, BindingRelationError, BindingRow, BindingVar};
use crate::term_identity::{FactId, TermId, TermIdentitySidecar, TermKind};

/// Physical argument-order sidecar for relation-like facts.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ArrangementDescriptor {
    /// Relation/functor term at child position 0.
    pub relation: TermId,
    /// Number of relation arguments, excluding the relation/functor child.
    pub argument_count: u8,
    /// Argument positions used as the key, zero-based after the relation child.
    pub key_order: Box<[u8]>,
}

impl ArrangementDescriptor {
    /// Creates a descriptor after validating the key order.
    pub fn new(
        relation: TermId,
        argument_count: u8,
        key_order: impl Into<Box<[u8]>>,
    ) -> Result<Self, ArrangementError> {
        let key_order = key_order.into();
        validate_key_order(argument_count, &key_order)?;

        Ok(Self {
            relation,
            argument_count,
            key_order,
        })
    }
}

/// One arranged fact row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArrangementRow {
    /// Complete fact identity.
    pub fact: FactId,
    /// Root term for the fact.
    pub root: TermId,
}

/// Projection from arranged facts into a BindingSpace relation schema.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArrangementProjection {
    /// Output BindingSpace schema.
    pub schema: Box<[BindingVar]>,
    /// Argument positions for each schema variable, zero-based after the
    /// relation/functor child.
    pub argument_positions: Box<[u8]>,
}

impl ArrangementProjection {
    /// Creates a projection after validating arity and argument positions.
    pub fn new(
        argument_count: u8,
        schema: impl Into<Box<[BindingVar]>>,
        argument_positions: impl Into<Box<[u8]>>,
    ) -> Result<Self, ArrangementError> {
        let schema = schema.into();
        let argument_positions = argument_positions.into();
        if schema.len() != argument_positions.len() {
            return Err(ArrangementError::ProjectionArityMismatch {
                schema_len: schema.len(),
                positions_len: argument_positions.len(),
            });
        }
        for &position in argument_positions.iter() {
            if position >= argument_count {
                return Err(ArrangementError::InvalidKeyPosition {
                    position,
                    argument_count,
                });
            }
        }

        Ok(Self {
            schema,
            argument_positions,
        })
    }
}

/// Snapshot-local arrangement index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArrangementIndex {
    descriptor: ArrangementDescriptor,
    rows_by_key: BTreeMap<ArrangementKey, Vec<ArrangementRow>>,
    stats: ArrangementStats,
}

/// Counters for arrangement construction and lookup.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArrangementStats {
    /// Complete fact roots inspected during build.
    pub facts_scanned: usize,
    /// Facts whose root matched the relation and arity.
    pub rows: usize,
    /// Relation/arity facts skipped because they contain encoded variables.
    pub schematic_rows_skipped: usize,
    /// Distinct arranged keys.
    pub distinct_keys: usize,
    /// Rows returned by prefix lookups.
    pub prefix_rows_returned: usize,
    /// Prefix lookup calls.
    pub prefix_lookups: usize,
}

/// Errors from arrangement construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArrangementError {
    /// The relation term is absent from the term sidecar.
    UnknownRelation { relation: TermId },
    /// A key position is outside the declared argument count.
    InvalidKeyPosition { position: u8, argument_count: u8 },
    /// Duplicate key positions would not define a useful arrangement order.
    DuplicateKeyPosition { position: u8 },
    /// A fact referenced a missing term record.
    UnknownTerm { term: TermId },
    /// Encoded arity would overflow when adding the relation/functor child.
    ArityOverflow { argument_count: u8 },
    /// Projection schema and argument-position list have different lengths.
    ProjectionArityMismatch {
        schema_len: usize,
        positions_len: usize,
    },
    /// Projecting duplicate rows overflowed the signed binding weight.
    BindingWeightOverflow,
}

impl ArrangementIndex {
    /// Builds a derived arrangement from a term snapshot.
    pub fn build(
        sidecar: &TermIdentitySidecar,
        descriptor: ArrangementDescriptor,
    ) -> Result<Self, ArrangementError> {
        if sidecar.get_term(descriptor.relation).is_none() {
            return Err(ArrangementError::UnknownRelation {
                relation: descriptor.relation,
            });
        }

        let encoded_arity =
            descriptor
                .argument_count
                .checked_add(1)
                .ok_or(ArrangementError::ArityOverflow {
                    argument_count: descriptor.argument_count,
                })?;
        let mut stats = ArrangementStats::default();
        let mut rows_by_key: BTreeMap<ArrangementKey, Vec<ArrangementRow>> = BTreeMap::new();

        for &fact_id in sidecar.facts_for_relation(descriptor.relation) {
            if !sidecar.is_fact_live(fact_id) {
                continue;
            }
            let Some(fact) = sidecar.get_fact(fact_id) else {
                continue;
            };
            stats.facts_scanned += 1;
            let Some(root) = sidecar.get_term(fact.root) else {
                return Err(ArrangementError::UnknownTerm { term: fact.root });
            };
            if root.kind
                != (TermKind::Application {
                    arity: encoded_arity,
                })
            {
                continue;
            }
            let children = root.children();
            if children.first().copied() != Some(descriptor.relation) {
                continue;
            }
            if fact.flags.contains_vars {
                stats.schematic_rows_skipped += 1;
                continue;
            }

            let key = ArrangementKey::from_arguments(&children[1..], &descriptor.key_order);
            rows_by_key.entry(key).or_default().push(ArrangementRow {
                fact: fact.id,
                root: fact.root,
            });
            stats.rows += 1;
        }

        stats.distinct_keys = rows_by_key.len();
        Ok(Self {
            descriptor,
            rows_by_key,
            stats,
        })
    }

    /// Folds a batch of newly interned facts into the arrangement, adding exactly
    /// the rows a rebuild would, in time proportional to the batch rather than the
    /// whole relation. The Stage 4 incremental-index step: a streaming relation
    /// patches its arrangement on each delta instead of rebuilding it. The caller
    /// passes the fact ids interned since the last build or patch; non-matching,
    /// schematic, or dead facts are skipped, exactly as in `build`.
    pub fn patch(
        &mut self,
        sidecar: &TermIdentitySidecar,
        new_facts: impl IntoIterator<Item = crate::term_identity::FactId>,
    ) -> Result<(), ArrangementError> {
        let encoded_arity = self.descriptor.argument_count.checked_add(1).ok_or(
            ArrangementError::ArityOverflow {
                argument_count: self.descriptor.argument_count,
            },
        )?;
        for fact_id in new_facts {
            if !sidecar.is_fact_live(fact_id) {
                continue;
            }
            let Some(fact) = sidecar.get_fact(fact_id) else {
                continue;
            };
            self.stats.facts_scanned += 1;
            let Some(root) = sidecar.get_term(fact.root) else {
                return Err(ArrangementError::UnknownTerm { term: fact.root });
            };
            if root.kind
                != (TermKind::Application {
                    arity: encoded_arity,
                })
            {
                continue;
            }
            let children = root.children();
            if children.first().copied() != Some(self.descriptor.relation) {
                continue;
            }
            if fact.flags.contains_vars {
                self.stats.schematic_rows_skipped += 1;
                continue;
            }
            let key = ArrangementKey::from_arguments(&children[1..], &self.descriptor.key_order);
            self.rows_by_key
                .entry(key)
                .or_default()
                .push(ArrangementRow {
                    fact: fact.id,
                    root: fact.root,
                });
            self.stats.rows += 1;
        }
        self.stats.distinct_keys = self.rows_by_key.len();
        Ok(())
    }

    /// Arrangement descriptor.
    pub fn descriptor(&self) -> &ArrangementDescriptor {
        &self.descriptor
    }

    /// Returns counters accumulated by build and lookup calls.
    pub fn stats(&self) -> ArrangementStats {
        self.stats
    }

    /// Exact number of rows whose leading arranged columns equal `prefix`: a
    /// correlated multi-constant count. Unlike a product of per-column distinct
    /// counts, this sees the correlation between columns, so it never over- or
    /// under-estimates a partially-bound relation (the planner's
    /// independent-column assumption is what it replaces). `prefix` is given in
    /// the arrangement's key order; an over-long prefix returns 0. Scans the
    /// sorted keys, so it is exact, trading the trie's constant-time probe for no
    /// extra stored cardinalities.
    pub fn prefix_count(&self, prefix: &[TermId]) -> usize {
        if prefix.len() > self.descriptor.key_order.len() {
            return 0;
        }
        self.rows_by_key
            .iter()
            .filter(|(key, _)| key.starts_with(prefix))
            .map(|(_, rows)| rows.len())
            .sum()
    }

    /// Exact number of distinct values in the next arranged column below
    /// `prefix`: the conditional domain size a worst-case-optimal join uses to
    /// bound the next variable given the constants already chosen. Returns 0 when
    /// `prefix` already covers the whole key order.
    pub fn conditional_distinct_count(&self, prefix: &[TermId]) -> usize {
        if prefix.len() >= self.descriptor.key_order.len() {
            return 0;
        }
        let mut seen = std::collections::HashSet::new();
        for key in self.rows_by_key.keys() {
            if key.starts_with(prefix) {
                if let Some(next) = key.get(prefix.len()) {
                    seen.insert(next);
                }
            }
        }
        seen.len()
    }

    /// Returns all rows whose arranged key starts with `prefix`.
    pub fn seek_prefix(&mut self, prefix: &[TermId]) -> Vec<ArrangementRow> {
        let mut rows = Vec::new();
        self.for_each_prefix_row(prefix, |row| rows.push(row));
        rows
    }

    /// Fallibly visits every row whose arranged key starts with `prefix`.
    ///
    /// The returned count is the number of rows delivered to `visit`. If
    /// `visit` returns an error, traversal stops immediately and lookup
    /// counters include only the rows delivered before that error.
    pub fn try_for_each_prefix_row<E>(
        &mut self,
        prefix: &[TermId],
        mut visit: impl FnMut(ArrangementRow) -> Result<(), E>,
    ) -> Result<usize, E> {
        self.stats.prefix_lookups += 1;

        let mut returned = 0;
        for (key, key_rows) in self.rows_by_key.range(ArrangementKey::from_slice(prefix)..) {
            if !key.starts_with(prefix) {
                break;
            }
            for &row in key_rows {
                returned += 1;
                if let Err(error) = visit(row) {
                    self.stats.prefix_rows_returned += returned;
                    return Err(error);
                }
            }
        }

        self.stats.prefix_rows_returned += returned;
        Ok(returned)
    }

    /// Visits every row whose arranged key starts with `prefix`.
    ///
    /// This is the allocation-free form of [`Self::seek_prefix`] for callers
    /// that can stream rows directly into a join, aggregate, or existence
    /// check.
    pub fn for_each_prefix_row(
        &mut self,
        prefix: &[TermId],
        mut visit: impl FnMut(ArrangementRow),
    ) -> usize {
        match self.try_for_each_prefix_row(prefix, |row| {
            visit(row);
            Ok::<(), Infallible>(())
        }) {
            Ok(count) => count,
            Err(error) => match error {},
        }
    }

    /// Exact arranged-key lookup.
    pub fn get_exact(&self, key: &[TermId]) -> &[ArrangementRow] {
        self.rows_by_key
            .get(&ArrangementKey::from_slice(key))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Projects arranged facts into a BindingSpace relation.
    pub fn project_bindings(
        &self,
        sidecar: &TermIdentitySidecar,
        projection: &ArrangementProjection,
    ) -> Result<BindingRelation, ArrangementError> {
        let mut relation = BindingRelation::new(projection.schema.clone());

        for row in self.rows_by_key.values().flatten() {
            add_projected_row(&self.descriptor, sidecar, projection, &mut relation, *row)?;
        }

        Ok(relation)
    }

    /// Projects only arranged facts whose key starts with `prefix`.
    ///
    /// This is the BindingSpace projection counterpart to
    /// [`Self::for_each_prefix_row`]. It lets bound-variable callers seek an
    /// arrangement prefix and build only the compatible relation factor.
    pub fn project_prefix_bindings(
        &mut self,
        sidecar: &TermIdentitySidecar,
        prefix: &[TermId],
        projection: &ArrangementProjection,
    ) -> Result<BindingRelation, ArrangementError> {
        let descriptor = self.descriptor.clone();
        let mut relation = BindingRelation::new(projection.schema.clone());

        self.try_for_each_prefix_row(prefix, |row| {
            add_projected_row(&descriptor, sidecar, projection, &mut relation, row)
        })?;

        Ok(relation)
    }
}

fn add_projected_row(
    descriptor: &ArrangementDescriptor,
    sidecar: &TermIdentitySidecar,
    projection: &ArrangementProjection,
    relation: &mut BindingRelation,
    row: ArrangementRow,
) -> Result<(), ArrangementError> {
    let arguments = root_arguments_for(descriptor, sidecar, row.root)?;
    let binding_row = projected_binding_row(arguments, &projection.argument_positions);
    relation
        .add(binding_row, 1)
        .map_err(|error| map_binding_relation_error(error, projection))
}

fn projected_binding_row(arguments: &[TermId], positions: &[u8]) -> BindingRow {
    match positions {
        [] => Box::from([]),
        [first] => Box::from([arguments[usize::from(*first)]]),
        [first, second] => Box::from([
            arguments[usize::from(*first)],
            arguments[usize::from(*second)],
        ]),
        [first, second, third] => Box::from([
            arguments[usize::from(*first)],
            arguments[usize::from(*second)],
            arguments[usize::from(*third)],
        ]),
        [first, second, third, fourth] => Box::from([
            arguments[usize::from(*first)],
            arguments[usize::from(*second)],
            arguments[usize::from(*third)],
            arguments[usize::from(*fourth)],
        ]),
        _ => positions
            .iter()
            .map(|&position| arguments[usize::from(position)])
            .collect(),
    }
}

fn map_binding_relation_error(
    error: BindingRelationError,
    projection: &ArrangementProjection,
) -> ArrangementError {
    match error {
        BindingRelationError::ArityMismatch { expected, actual } => {
            ArrangementError::ProjectionArityMismatch {
                schema_len: expected,
                positions_len: actual,
            }
        }
        BindingRelationError::SchemaMismatch
        | BindingRelationError::UnknownVariable { .. }
        | BindingRelationError::InvalidVariableOrder => ArrangementError::ProjectionArityMismatch {
            schema_len: projection.schema.len(),
            positions_len: projection.argument_positions.len(),
        },
        BindingRelationError::WeightOverflow | BindingRelationError::CardinalityOverflow => {
            ArrangementError::BindingWeightOverflow
        }
    }
}

fn root_arguments_for<'a>(
    descriptor: &ArrangementDescriptor,
    sidecar: &'a TermIdentitySidecar,
    root: TermId,
) -> Result<&'a [TermId], ArrangementError> {
    let Some(record) = sidecar.get_term(root) else {
        return Err(ArrangementError::UnknownTerm { term: root });
    };
    let encoded_arity =
        descriptor
            .argument_count
            .checked_add(1)
            .ok_or(ArrangementError::ArityOverflow {
                argument_count: descriptor.argument_count,
            })?;
    if record.kind
        != (TermKind::Application {
            arity: encoded_arity,
        })
    {
        return Err(ArrangementError::UnknownTerm { term: root });
    }
    let children = record.children();
    if children.first().copied() != Some(descriptor.relation) {
        return Err(ArrangementError::UnknownTerm { term: root });
    }
    Ok(&children[1..])
}

/// Arrangement key representation that avoids heap allocation for common
/// relation-like unary and binary key orders while preserving slice
/// lexicographic ordering.
#[derive(Clone, Debug, Eq, PartialEq)]
enum ArrangementKey {
    Empty,
    One(TermId),
    Two(TermId, TermId),
    Many(Box<[TermId]>),
}

impl ArrangementKey {
    fn from_slice(key: &[TermId]) -> Self {
        match key {
            [] => Self::Empty,
            [first] => Self::One(*first),
            [first, second] => Self::Two(*first, *second),
            _ => Self::Many(key.to_vec().into_boxed_slice()),
        }
    }

    fn from_arguments(arguments: &[TermId], positions: &[u8]) -> Self {
        match positions {
            [] => Self::Empty,
            [first] => Self::One(arguments[usize::from(*first)]),
            [first, second] => Self::Two(
                arguments[usize::from(*first)],
                arguments[usize::from(*second)],
            ),
            _ => Self::Many(
                positions
                    .iter()
                    .map(|&position| arguments[usize::from(position)])
                    .collect(),
            ),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::One(_) => 1,
            Self::Two(_, _) => 2,
            Self::Many(key) => key.len(),
        }
    }

    fn get(&self, index: usize) -> Option<TermId> {
        match self {
            Self::Empty => None,
            Self::One(first) => (index == 0).then_some(*first),
            Self::Two(first, second) => match index {
                0 => Some(*first),
                1 => Some(*second),
                _ => None,
            },
            Self::Many(key) => key.get(index).copied(),
        }
    }

    fn starts_with(&self, prefix: &[TermId]) -> bool {
        if prefix.len() > self.len() {
            return false;
        }

        prefix
            .iter()
            .enumerate()
            .all(|(index, term)| self.get(index) == Some(*term))
    }
}

impl Ord for ArrangementKey {
    fn cmp(&self, other: &Self) -> Ordering {
        for index in 0..self.len().min(other.len()) {
            let order = self
                .get(index)
                .expect("index is within key length")
                .cmp(&other.get(index).expect("index is within key length"));
            if order != Ordering::Equal {
                return order;
            }
        }
        self.len().cmp(&other.len())
    }
}

impl PartialOrd for ArrangementKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn validate_key_order(argument_count: u8, key_order: &[u8]) -> Result<(), ArrangementError> {
    let mut seen = 0u64;
    for &position in key_order {
        if position >= argument_count {
            return Err(ArrangementError::InvalidKeyPosition {
                position,
                argument_count,
            });
        }
        let bit = 1u64 << position;
        if seen & bit != 0 {
            return Err(ArrangementError::DuplicateKeyPosition { position });
        }
        seen |= bit;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding_space::{BindingVar, generic_join};
    use crate::space::Space;
    use crate::term_identity::TermIdentitySidecar;
    use std::collections::BTreeSet;

    fn encoded_roots(
        sidecar: &TermIdentitySidecar,
        rows: impl IntoIterator<Item = ArrangementRow>,
    ) -> BTreeSet<Vec<u8>> {
        rows.into_iter()
            .map(|row| sidecar.get_term(row.root).unwrap().encoded().to_vec())
            .collect()
    }

    fn encoded_expr(space: &mut Space, expr: &'static str) -> Vec<u8> {
        let expr = crate::expr!(space, expr);
        unsafe { expr.span().as_ref().unwrap() }.to_vec()
    }

    fn sidecar_from(space: &Space) -> TermIdentitySidecar {
        let mut sidecar = TermIdentitySidecar::new();
        sidecar.extend_from_pathmap(&space.btm).unwrap();
        sidecar
    }

    fn suffix_edge_space() -> Space {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge Alice Bob)
(edge Carol Bob)
(edge Alice Dave)
(edge Eve Frank)
(note Bob Alice)
"#,
            )
            .unwrap();
        space
    }

    fn term_id(space: &mut Space, sidecar: &TermIdentitySidecar, expr: &'static str) -> TermId {
        sidecar
            .term_id_for_encoded(&encoded_expr(space, expr))
            .unwrap()
    }

    fn build_arrangement(
        sidecar: &TermIdentitySidecar,
        relation: TermId,
        key_order: [u8; 2],
    ) -> ArrangementIndex {
        let descriptor = ArrangementDescriptor::new(relation, 2, key_order).unwrap();
        ArrangementIndex::build(sidecar, descriptor).unwrap()
    }

    fn term(id: u64) -> TermId {
        TermId(id)
    }

    #[test]
    fn arrangement_patch_matches_rebuild() {
        let mut space = Space::new();
        space
            .add_all_sexpr(b"(edge Alice Bob)\n(edge Carol Bob)\n(edge Alice Dave)\n")
            .unwrap();
        let mut facts: Vec<Vec<u8>> = Vec::new();
        space
            .btm
            .try_for_each_value::<_, ()>(|path, _| {
                facts.push(path.to_vec());
                Ok(())
            })
            .unwrap();

        let mut sidecar = TermIdentitySidecar::new();
        let first = sidecar.insert_fact(&facts[0]).unwrap();
        sidecar.insert_fact(&facts[1]).unwrap();
        // The edge relation head is the first child of an edge fact's root term.
        let root = sidecar.get_fact(first).unwrap().root;
        let edge = sidecar.get_term(root).unwrap().children()[0];
        let descriptor = ArrangementDescriptor::new(edge, 2, [0, 1]).unwrap();
        let mut patched = ArrangementIndex::build(&sidecar, descriptor.clone()).unwrap();

        // Fold the third fact in incrementally instead of rebuilding.
        let third = sidecar.insert_fact(&facts[2]).unwrap();
        patched.patch(&sidecar, [third]).unwrap();

        let mut rebuilt = ArrangementIndex::build(&sidecar, descriptor).unwrap();

        let dump = |arrangement: &mut ArrangementIndex| {
            let mut rows: Vec<_> = arrangement
                .seek_prefix(&[])
                .iter()
                .map(|row| (row.fact, row.root))
                .collect();
            rows.sort_unstable();
            rows
        };
        assert_eq!(dump(&mut patched), dump(&mut rebuilt));
        assert_eq!(patched.stats().rows, 3);
    }

    #[test]
    fn arrangement_key_preserves_slice_lexicographic_order() {
        let mut keys = [
            ArrangementKey::from_slice(&[term(2)]),
            ArrangementKey::from_slice(&[term(1), term(2), term(0)]),
            ArrangementKey::from_slice(&[]),
            ArrangementKey::from_slice(&[term(1)]),
            ArrangementKey::from_slice(&[term(1), term(1)]),
            ArrangementKey::from_slice(&[term(1), term(2)]),
        ];

        keys.sort();

        assert_eq!(
            keys,
            [
                ArrangementKey::Empty,
                ArrangementKey::One(term(1)),
                ArrangementKey::Two(term(1), term(1)),
                ArrangementKey::Two(term(1), term(2)),
                ArrangementKey::Many(vec![term(1), term(2), term(0)].into_boxed_slice()),
                ArrangementKey::One(term(2)),
            ]
        );
    }

    #[test]
    fn arrangement_key_builds_compact_keys_and_checks_prefixes() {
        let arguments = [term(10), term(20), term(30), term(40)];

        assert_eq!(
            ArrangementKey::from_arguments(&arguments, &[]),
            ArrangementKey::Empty
        );
        assert_eq!(
            ArrangementKey::from_arguments(&arguments, &[2]),
            ArrangementKey::One(term(30))
        );
        assert_eq!(
            ArrangementKey::from_arguments(&arguments, &[2, 0]),
            ArrangementKey::Two(term(30), term(10))
        );
        assert_eq!(
            ArrangementKey::from_arguments(&arguments, &[2, 0, 1]),
            ArrangementKey::Many(vec![term(30), term(10), term(20)].into_boxed_slice())
        );

        let key = ArrangementKey::from_slice(&[term(30), term(10), term(20)]);
        assert!(key.starts_with(&[]));
        assert!(key.starts_with(&[term(30)]));
        assert!(key.starts_with(&[term(30), term(10)]));
        assert!(key.starts_with(&[term(30), term(10), term(20)]));
        assert!(!key.starts_with(&[term(30), term(20)]));
        assert!(!key.starts_with(&[term(30), term(10), term(20), term(40)]));
    }

    #[test]
    fn projected_binding_row_preserves_small_and_wide_orders() {
        let arguments = [term(10), term(20), term(30), term(40), term(50), term(60)];

        assert_eq!(projected_binding_row(&arguments, &[]).as_ref(), &[]);
        assert_eq!(
            projected_binding_row(&arguments, &[2]).as_ref(),
            &[term(30)]
        );
        assert_eq!(
            projected_binding_row(&arguments, &[2, 0]).as_ref(),
            &[term(30), term(10)]
        );
        assert_eq!(
            projected_binding_row(&arguments, &[2, 0, 4, 1]).as_ref(),
            &[term(30), term(10), term(50), term(20)]
        );
        assert_eq!(
            projected_binding_row(&arguments, &[5, 4, 3, 2, 1]).as_ref(),
            &[term(60), term(50), term(40), term(30), term(20)]
        );
    }

    #[test]
    fn suffix_bound_arrangement_matches_product_query_roots() {
        let mut space = suffix_edge_space();

        let sidecar = sidecar_from(&space);
        let edge = term_id(&mut space, &sidecar, "edge");
        let bob = term_id(&mut space, &sidecar, "Bob");
        let alice = term_id(&mut space, &sidecar, "Alice");

        let mut arrangement = build_arrangement(&sidecar, edge, [1, 0]);
        let mut bob_rows = Vec::new();
        let returned = arrangement.for_each_prefix_row(&[bob], |row| bob_rows.push(row));
        let exact_rows = arrangement.get_exact(&[bob, alice]);

        let product_pattern = crate::expr!(space, "[2] , [3] edge $ Bob");
        let mut product_roots = BTreeSet::new();
        let product_count = Space::query_multi(&space.btm, product_pattern, |_, loc| {
            let span = unsafe { loc.span().as_ref().unwrap() };
            product_roots.insert(span.to_vec());
            true
        });

        assert_eq!(product_count, 2);
        assert_eq!(returned, 2);
        assert_eq!(encoded_roots(&sidecar, bob_rows), product_roots);
        assert_eq!(exact_rows.len(), 1);

        let stats = arrangement.stats();
        assert_eq!(stats.rows, 4);
        assert_eq!(stats.schematic_rows_skipped, 0);
        assert_eq!(stats.prefix_lookups, 1);
        assert_eq!(stats.prefix_rows_returned, 2);
    }

    #[test]
    fn prefix_projection_projects_only_matching_rows() {
        let mut space = suffix_edge_space();
        let sidecar = sidecar_from(&space);
        let edge = term_id(&mut space, &sidecar, "edge");
        let alice = term_id(&mut space, &sidecar, "Alice");
        let bob = term_id(&mut space, &sidecar, "Bob");
        let carol = term_id(&mut space, &sidecar, "Carol");
        let eve = term_id(&mut space, &sidecar, "Eve");
        let frank = term_id(&mut space, &sidecar, "Frank");

        let mut arrangement = build_arrangement(&sidecar, edge, [1, 0]);
        let projection =
            ArrangementProjection::new(2, [BindingVar(0), BindingVar(1)], [0, 1]).unwrap();
        let relation = arrangement
            .project_prefix_bindings(&sidecar, &[bob], &projection)
            .unwrap();
        let rows = relation
            .positive_rows()
            .map(<[TermId]>::to_vec)
            .collect::<BTreeSet<_>>();

        assert_eq!(rows, BTreeSet::from([vec![alice, bob], vec![carol, bob]]));
        assert_eq!(relation.weight(&[eve, frank]), 0);
        assert_eq!(arrangement.stats().prefix_lookups, 1);
        assert_eq!(arrangement.stats().prefix_rows_returned, 2);
    }

    #[test]
    fn try_prefix_row_stops_after_visitor_error() {
        let mut space = suffix_edge_space();
        let sidecar = sidecar_from(&space);
        let edge = term_id(&mut space, &sidecar, "edge");
        let bob = term_id(&mut space, &sidecar, "Bob");

        let mut arrangement = build_arrangement(&sidecar, edge, [1, 0]);
        let result = arrangement.try_for_each_prefix_row(&[bob], |_| Err("stop"));

        assert_eq!(result, Err("stop"));
        assert_eq!(arrangement.stats().prefix_lookups, 1);
        assert_eq!(arrangement.stats().prefix_rows_returned, 1);
    }

    #[test]
    fn arrangement_index_keeps_schematic_facts_out_of_ground_rows() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge Alice Bob)
(edge Carol Bob)
(edge $x Bob)
(edge Alice $y)
(node Bob)
"#,
            )
            .unwrap();

        let sidecar = sidecar_from(&space);
        let edge = term_id(&mut space, &sidecar, "edge");
        let bob = term_id(&mut space, &sidecar, "Bob");

        let mut arrangement = build_arrangement(&sidecar, edge, [1, 0]);
        let bob_rows = arrangement.seek_prefix(&[bob]);
        let stats = arrangement.stats();

        // Only the four edge facts are scanned, not the unrelated (node Bob):
        // arrangement build reads the relation's bucket, not the whole sidecar.
        assert_eq!(stats.facts_scanned, 4);
        assert_eq!(stats.rows, 2);
        assert_eq!(stats.schematic_rows_skipped, 2);
        assert_eq!(bob_rows.len(), 2);
        assert!(bob_rows.iter().all(|row| {
            !sidecar
                .get_term(row.root)
                .expect("arrangement row roots should be interned")
                .flags
                .contains_vars
        }));
    }

    #[test]
    fn descriptor_rejects_invalid_key_orders() {
        assert_eq!(
            ArrangementDescriptor::new(TermId(0), 2, [2]),
            Err(ArrangementError::InvalidKeyPosition {
                position: 2,
                argument_count: 2,
            })
        );
        assert_eq!(
            ArrangementDescriptor::new(TermId(0), 2, [1, 1]),
            Err(ArrangementError::DuplicateKeyPosition { position: 1 })
        );
    }

    #[test]
    fn arrangement_projection_feeds_generic_join_for_transitive_edges() {
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

        let sidecar = sidecar_from(&space);
        let edge = term_id(&mut space, &sidecar, "edge");

        let arrangement = build_arrangement(&sidecar, edge, [0, 1]);
        let xy = arrangement
            .project_bindings(
                &sidecar,
                &ArrangementProjection::new(2, [BindingVar(0), BindingVar(1)], [0, 1]).unwrap(),
            )
            .unwrap();
        let yz = arrangement
            .project_bindings(
                &sidecar,
                &ArrangementProjection::new(2, [BindingVar(1), BindingVar(2)], [0, 1]).unwrap(),
            )
            .unwrap();
        let joined =
            generic_join(&[xy, yz], &[BindingVar(1), BindingVar(0), BindingVar(2)]).unwrap();

        let product_pattern = crate::expr!(space, "[3] , [3] edge $ $ [3] edge _2 $");
        let product_count = Space::query_multi(&space.btm, product_pattern, |_, _| true);

        assert_eq!(product_count, 4);
        assert_eq!(joined.positive_rows().count(), product_count);
    }

    #[test]
    fn correlated_prefix_counts_beat_the_independent_estimate() {
        use crate::encoded_test_helpers::{app, sym};

        // Correlated relation: a only pairs with t1, b only with t2. Two distinct
        // sources and two distinct targets, but only two of the four source-target
        // combinations actually occur.
        let mut sidecar = TermIdentitySidecar::new();
        sidecar
            .insert_fact(&app(&[sym(b"edge"), sym(b"a"), sym(b"t1")]))
            .unwrap();
        sidecar
            .insert_fact(&app(&[sym(b"edge"), sym(b"b"), sym(b"t2")]))
            .unwrap();
        let id = |s: &[u8]| sidecar.term_id_for_encoded(&sym(s)).unwrap();
        let (edge, a, b, t1, t2) = (id(b"edge"), id(b"a"), id(b"b"), id(b"t1"), id(b"t2"));

        let descriptor = ArrangementDescriptor::new(edge, 2, [0, 1]).unwrap();
        let index = ArrangementIndex::build(&sidecar, descriptor).unwrap();

        // Exact correlated row counts.
        assert_eq!(index.prefix_count(&[]), 2);
        assert_eq!(index.prefix_count(&[a]), 1);
        assert_eq!(index.prefix_count(&[b]), 1);
        assert_eq!(index.prefix_count(&[a, t1]), 1);
        // The correlation: a never pairs with t2. An independent product of
        // distinct(source) = 2 and distinct(target) = 2 over 2 rows would predict
        // a positive count for (a, t2); the exact correlated count is 0.
        assert_eq!(index.prefix_count(&[a, t2]), 0);

        // Conditional domain of the target given a chosen source is 1, not the
        // unconditional 2 distinct targets the planner would otherwise assume.
        assert_eq!(index.conditional_distinct_count(&[]), 2); // distinct sources
        assert_eq!(index.conditional_distinct_count(&[a]), 1); // only t1 follows a
        assert_eq!(index.conditional_distinct_count(&[b]), 1); // only t2 follows b
        // A full-length prefix has no next column.
        assert_eq!(index.conditional_distinct_count(&[a, t1]), 0);
    }
}
