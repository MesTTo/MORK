use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashSet, btree_map::Entry},
};

use crate::term_identity::TermId;

/// Query variable identifier used by BindingSpace sidecars.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BindingVar(pub u8);

/// Compact binding row.
pub type BindingRow = Box<[TermId]>;

/// Signed relation over canonical term bindings.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BindingRelation {
    schema: Box<[BindingVar]>,
    weights: BTreeMap<BindingRow, i64>,
}

/// Errors from relation operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingRelationError {
    /// Row width did not match relation schema.
    ArityMismatch { expected: usize, actual: usize },
    /// Operation requires equal schemas.
    SchemaMismatch,
    /// Variable was not present in the schema.
    UnknownVariable { variable: BindingVar },
    /// Variable order does not contain exactly the variables in the input.
    InvalidVariableOrder,
    /// Signed relation weight arithmetic overflowed.
    WeightOverflow,
    /// Factorized cardinality arithmetic overflowed.
    CardinalityOverflow,
}

fn checked_add_weight(lhs: i64, rhs: i64) -> Result<i64, BindingRelationError> {
    lhs.checked_add(rhs)
        .ok_or(BindingRelationError::WeightOverflow)
}

fn checked_mul_weight(lhs: i64, rhs: i64) -> Result<i64, BindingRelationError> {
    lhs.checked_mul(rhs)
        .ok_or(BindingRelationError::WeightOverflow)
}

fn checked_neg_weight(weight: i64) -> Result<i64, BindingRelationError> {
    weight
        .checked_neg()
        .ok_or(BindingRelationError::WeightOverflow)
}

fn checked_add_cardinality(lhs: usize, rhs: usize) -> Result<usize, BindingRelationError> {
    lhs.checked_add(rhs)
        .ok_or(BindingRelationError::CardinalityOverflow)
}

fn checked_mul_cardinality(lhs: usize, rhs: usize) -> Result<usize, BindingRelationError> {
    lhs.checked_mul(rhs)
        .ok_or(BindingRelationError::CardinalityOverflow)
}

fn checked_removed_rows(before: usize, after: usize) -> Result<usize, BindingRelationError> {
    before
        .checked_sub(after)
        .ok_or(BindingRelationError::CardinalityOverflow)
}

fn checked_add_removed_rows(
    total: usize,
    before: usize,
    after: usize,
) -> Result<usize, BindingRelationError> {
    checked_add_cardinality(total, checked_removed_rows(before, after)?)
}

const BINDING_SLOT_COUNT: usize = (u8::MAX as usize) + 1;
const MISSING_TRIE_POSITION: usize = usize::MAX;

#[derive(Clone)]
pub(crate) struct BindingAssignment {
    values: [TermId; BINDING_SLOT_COUNT],
    bound_words: [u64; 4],
}

impl Default for BindingAssignment {
    fn default() -> Self {
        Self {
            values: [TermId(0); BINDING_SLOT_COUNT],
            bound_words: [0; 4],
        }
    }
}

impl BindingAssignment {
    #[inline]
    fn slot(variable: BindingVar) -> usize {
        usize::from(variable.0)
    }

    #[inline]
    fn word_and_bit(variable: BindingVar) -> (usize, u64) {
        let slot = Self::slot(variable);
        (
            slot / u64::BITS as usize,
            1_u64 << (slot % u64::BITS as usize),
        )
    }

    #[inline]
    fn insert(&mut self, variable: BindingVar, value: TermId) {
        let (word, bit) = Self::word_and_bit(variable);
        self.values[Self::slot(variable)] = value;
        self.bound_words[word] |= bit;
    }

    #[inline]
    fn remove(&mut self, variable: BindingVar) {
        let (word, bit) = Self::word_and_bit(variable);
        self.bound_words[word] &= !bit;
    }

    #[inline]
    pub(crate) fn get(&self, variable: BindingVar) -> Option<TermId> {
        let (word, bit) = Self::word_and_bit(variable);
        (self.bound_words[word] & bit != 0).then_some(self.values[Self::slot(variable)])
    }

    #[inline]
    fn contains_all(&self, variables: &[BindingVar]) -> bool {
        variables
            .iter()
            .all(|&variable| self.get(variable).is_some())
    }

    #[inline]
    fn bound(&self, variable: BindingVar) -> TermId {
        self.get(variable)
            .expect("join variable should be bound at this recursion depth")
    }
}

impl BindingRelation {
    /// Creates an empty relation with the given schema.
    pub fn new(schema: impl Into<Box<[BindingVar]>>) -> Self {
        Self {
            schema: schema.into(),
            weights: BTreeMap::new(),
        }
    }

    /// Relation schema.
    pub fn schema(&self) -> &[BindingVar] {
        &self.schema
    }

    /// Number of retained non-zero rows.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Returns true when no non-zero rows remain.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Adds a signed row weight, removing the row when the total reaches zero.
    pub fn add(
        &mut self,
        row: impl Into<BindingRow>,
        weight: i64,
    ) -> Result<(), BindingRelationError> {
        let row = row.into();
        if row.len() != self.schema.len() {
            return Err(BindingRelationError::ArityMismatch {
                expected: self.schema.len(),
                actual: row.len(),
            });
        }

        match self.weights.entry(row) {
            Entry::Vacant(entry) => {
                if weight != 0 {
                    entry.insert(weight);
                }
            }
            Entry::Occupied(mut entry) => {
                let updated = checked_add_weight(*entry.get(), weight)?;
                if updated == 0 {
                    entry.remove();
                } else {
                    *entry.get_mut() = updated;
                }
            }
        }
        Ok(())
    }

    /// Inserts `row` with unit positive weight only when it is not already
    /// retained.
    ///
    /// This is the set-style fast path for differential frontiers and other
    /// deduplicated kernels. It uses one ordered-map entry search instead of a
    /// lookup followed by a separate insert.
    pub fn insert_unit_if_absent(
        &mut self,
        row: impl Into<BindingRow>,
    ) -> Result<bool, BindingRelationError> {
        let row = row.into();
        if row.len() != self.schema.len() {
            return Err(BindingRelationError::ArityMismatch {
                expected: self.schema.len(),
                actual: row.len(),
            });
        }

        match self.weights.entry(row) {
            Entry::Vacant(entry) => {
                entry.insert(1);
                Ok(true)
            }
            Entry::Occupied(_) => Ok(false),
        }
    }

    /// Returns the signed weight for `row`, or zero if absent.
    pub fn weight(&self, row: &[TermId]) -> i64 {
        self.weights.get(row).copied().unwrap_or(0)
    }

    fn weight_from_binding(&self, binding: &BindingAssignment) -> i64 {
        match self.schema.as_ref() {
            [] => self.weight(&[]),
            [a] => {
                let row = [binding.bound(*a)];
                self.weight(&row)
            }
            [a, b] => {
                let row = [binding.bound(*a), binding.bound(*b)];
                self.weight(&row)
            }
            [a, b, c] => {
                let row = [binding.bound(*a), binding.bound(*b), binding.bound(*c)];
                self.weight(&row)
            }
            [a, b, c, d] => {
                let row = [
                    binding.bound(*a),
                    binding.bound(*b),
                    binding.bound(*c),
                    binding.bound(*d),
                ];
                self.weight(&row)
            }
            variables => {
                let row = variables
                    .iter()
                    .map(|&variable| binding.bound(variable))
                    .collect::<Vec<_>>();
                self.weight(&row)
            }
        }
    }

    /// Iterates all retained rows and weights.
    pub fn rows(&self) -> impl Iterator<Item = (&[TermId], i64)> + '_ {
        self.weights
            .iter()
            .map(|(row, &weight)| (row.as_ref(), weight))
    }

    /// Iterates rows with positive visible weight.
    pub fn positive_rows(&self) -> impl Iterator<Item = &[TermId]> + '_ {
        self.rows()
            .filter_map(|(row, weight)| (weight > 0).then_some(row))
    }

    /// Adds every row from `other`.
    pub fn union_assign(&mut self, other: &Self) -> Result<(), BindingRelationError> {
        if self.schema != other.schema {
            return Err(BindingRelationError::SchemaMismatch);
        }
        for (row, weight) in &other.weights {
            self.add(row.clone(), *weight)?;
        }
        Ok(())
    }

    /// Set-style normalization: retain every non-zero signed row with unit
    /// positive weight.
    pub fn distinct(&self) -> Self {
        let mut weights = self.weights.clone();
        for weight in weights.values_mut() {
            *weight = 1;
        }
        Self {
            schema: self.schema.clone(),
            weights,
        }
    }

    /// Signed difference: subtract every row weight from `other`.
    pub fn signed_difference(&self, other: &Self) -> Result<Self, BindingRelationError> {
        if self.schema != other.schema {
            return Err(BindingRelationError::SchemaMismatch);
        }
        let mut out = self.clone();
        for (row, weight) in &other.weights {
            out.add(row.clone(), checked_neg_weight(*weight)?)?;
        }
        Ok(out)
    }

    /// Presence difference: positive rows from `self` that are not visible in
    /// `other`.
    pub fn difference_presence(&self, other: &Self) -> Result<Self, BindingRelationError> {
        if self.schema != other.schema {
            return Err(BindingRelationError::SchemaMismatch);
        }

        let visible_other = other
            .rows()
            .filter_map(|(row, weight)| {
                (weight > 0).then_some(NaturalJoinKey::from_complete_row(row))
            })
            .collect::<BTreeSet<_>>();
        let mut out = Self::new(self.schema.clone());
        for (row, weight) in self.rows() {
            if weight > 0 {
                let key = NaturalJoinKey::from_complete_row(row);
                if !visible_other.contains(&key) {
                    out.add(key.into_row(), weight)?;
                }
            }
        }
        Ok(out)
    }

    /// Set projection onto `variables`, a subset of the schema in the given
    /// order. Each distinct projection of a positive row is emitted once with
    /// weight 1, so the result follows MM2 set semantics. Errors if a variable
    /// is repeated or absent from the schema.
    ///
    /// This materializes a hypertree-decomposition bag relation: the bag holds
    /// the join of its covering relations projected onto the bag variables. See
    /// `kernel/resources/hypertree_decomposition_plan.md`.
    pub fn project(&self, variables: &[BindingVar]) -> Result<Self, BindingRelationError> {
        if has_duplicates(variables) {
            return Err(BindingRelationError::InvalidVariableOrder);
        }
        let positions = indexes(self, variables)?;
        let mut out = Self::new(variables.to_vec());
        for row in self.positive_rows() {
            let projected = positions
                .iter()
                .map(|&position| row[position])
                .collect::<Vec<_>>();
            out.insert_unit_if_absent(projected)?;
        }
        Ok(out)
    }

    /// Number of distinct value tuples over `variables` among positive rows: the
    /// exact projected cardinality. A cost model uses this to size joins and order
    /// variables. Errors on an absent or repeated variable.
    pub fn distinct_prefix_count(
        &self,
        variables: &[BindingVar],
    ) -> Result<usize, BindingRelationError> {
        Ok(self.project(variables)?.positive_rows().count())
    }

    /// Number of distinct values of a single variable among positive rows: its
    /// exact domain size.
    pub fn distinct_value_count(
        &self,
        variable: BindingVar,
    ) -> Result<usize, BindingRelationError> {
        self.distinct_prefix_count(std::slice::from_ref(&variable))
    }

    /// Relabels the schema variables, preserving rows and signed weights
    /// positionally. Plugs a relation into a rule body that uses different
    /// variable names (for example a recursive relation joined back into its own
    /// body). `new_schema` must match the current arity and have no duplicates.
    pub fn relabel(&self, new_schema: &[BindingVar]) -> Result<Self, BindingRelationError> {
        if new_schema.len() != self.schema.len() {
            return Err(BindingRelationError::ArityMismatch {
                expected: self.schema.len(),
                actual: new_schema.len(),
            });
        }
        if has_duplicates(new_schema) {
            return Err(BindingRelationError::InvalidVariableOrder);
        }
        let mut out = Self::new(new_schema.to_vec());
        for (row, weight) in self.rows() {
            out.add(row.to_vec(), weight)?;
        }
        Ok(out)
    }

    fn schema_index(&self, variable: BindingVar) -> Option<usize> {
        self.schema.iter().position(|&value| value == variable)
    }
}

/// Right-side natural-join bucket optimized for the common unique-key case.
enum NaturalJoinBucket<'a> {
    One((&'a [TermId], i64)),
    Many(Vec<(&'a [TermId], i64)>),
}

impl<'a> NaturalJoinBucket<'a> {
    fn push(&mut self, row: (&'a [TermId], i64)) {
        match self {
            Self::One(first) => {
                let first = *first;
                *self = Self::Many(vec![first, row]);
            }
            Self::Many(rows) => rows.push(row),
        }
    }

    fn try_for_each<E>(
        &self,
        mut visit: impl FnMut(&'a [TermId], i64) -> Result<(), E>,
    ) -> Result<(), E> {
        match self {
            Self::One((row, weight)) => visit(row, *weight),
            Self::Many(rows) => {
                for &(row, weight) in rows {
                    visit(row, weight)?;
                }
                Ok(())
            }
        }
    }
}

fn insert_natural_join_bucket<'a>(
    index: &mut BTreeMap<NaturalJoinKey, NaturalJoinBucket<'a>>,
    key: NaturalJoinKey,
    row: &'a [TermId],
    weight: i64,
) {
    index
        .entry(key)
        .and_modify(|bucket| bucket.push((row, weight)))
        .or_insert(NaturalJoinBucket::One((row, weight)));
}

/// Natural-join key representation that avoids allocating for common unary and
/// binary join keys.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum NaturalJoinKey {
    Empty,
    One(TermId),
    Two(TermId, TermId),
    Many(Box<[TermId]>),
}

impl NaturalJoinKey {
    fn from_complete_row(row: &[TermId]) -> Self {
        match row {
            [] => Self::Empty,
            [first] => Self::One(*first),
            [first, second] => Self::Two(*first, *second),
            _ => Self::Many(row.to_vec().into_boxed_slice()),
        }
    }

    fn from_row(row: &[TermId], indexes: &[usize]) -> Self {
        match indexes {
            [] => Self::Empty,
            [first] => Self::One(row[*first]),
            [first, second] => Self::Two(row[*first], row[*second]),
            _ => Self::Many(indexes.iter().map(|&index| row[index]).collect()),
        }
    }

    fn from_bound_prefix(variables: &[BindingVar], binding: &BindingAssignment) -> Option<Self> {
        match variables {
            [] => Some(Self::Empty),
            [first] => Some(Self::One(binding.get(*first)?)),
            [first, second] => Some(Self::Two(binding.get(*first)?, binding.get(*second)?)),
            _ => variables
                .iter()
                .map(|&variable| binding.get(variable))
                .collect::<Option<Vec<_>>>()
                .map(Vec::into_boxed_slice)
                .map(Self::Many),
        }
    }

    fn cmp_bound_prefix(&self, variables: &[BindingVar], binding: &BindingAssignment) -> Ordering {
        match (self, variables.len()) {
            (Self::Empty, 0) => Ordering::Equal,
            (Self::Empty, _) => Ordering::Less,
            (Self::One(_), 0) => Ordering::Greater,
            (Self::One(value), 1) => value.cmp(&binding.bound(variables[0])),
            (Self::One(_), _) => Ordering::Less,
            (Self::Two(_, _), 0 | 1) => Ordering::Greater,
            (Self::Two(first, second), 2) => first
                .cmp(&binding.bound(variables[0]))
                .then_with(|| second.cmp(&binding.bound(variables[1]))),
            (Self::Two(_, _), _) => Ordering::Less,
            (Self::Many(_), 0..=2) => Ordering::Greater,
            (Self::Many(row), _) => cmp_row_with_bound_prefix(row, variables, binding),
        }
    }

    /// Same ordering as [`cmp_bound_prefix`], but against an already-extracted
    /// bound prefix (`prefix[i] == binding.bound(variables[i])`). Used by the
    /// join seek to compare against a prefix it read once, instead of re-reading
    /// each `binding.bound(var)` (a word/bit computation) inside every
    /// binary-search comparison.
    fn cmp_prefix_values(&self, prefix: &[TermId]) -> Ordering {
        match (self, prefix.len()) {
            (Self::Empty, 0) => Ordering::Equal,
            (Self::Empty, _) => Ordering::Less,
            (Self::One(_), 0) => Ordering::Greater,
            (Self::One(value), 1) => value.cmp(&prefix[0]),
            (Self::One(_), _) => Ordering::Less,
            (Self::Two(_, _), 0 | 1) => Ordering::Greater,
            (Self::Two(first, second), 2) => {
                first.cmp(&prefix[0]).then_with(|| second.cmp(&prefix[1]))
            }
            (Self::Two(_, _), _) => Ordering::Less,
            (Self::Many(_), 0..=2) => Ordering::Greater,
            (Self::Many(row), _) => cmp_row_with_values(row, prefix),
        }
    }

    fn into_row(self) -> BindingRow {
        match self {
            Self::Empty => Box::new([]),
            Self::One(value) => Box::new([value]),
            Self::Two(first, second) => Box::new([first, second]),
            Self::Many(row) => row,
        }
    }
}

fn cmp_row_with_bound_prefix(
    row: &[TermId],
    variables: &[BindingVar],
    binding: &BindingAssignment,
) -> Ordering {
    for (&value, &variable) in row.iter().zip(variables.iter()) {
        match value.cmp(&binding.bound(variable)) {
            Ordering::Equal => {}
            ordering => return ordering,
        }
    }

    row.len().cmp(&variables.len())
}

fn cmp_row_with_values(row: &[TermId], prefix: &[TermId]) -> Ordering {
    for (&value, &bound) in row.iter().zip(prefix.iter()) {
        match value.cmp(&bound) {
            Ordering::Equal => {}
            ordering => return ordering,
        }
    }

    row.len().cmp(&prefix.len())
}

fn binary_search_bound_prefix(
    keys: &[NaturalJoinKey],
    variables: &[BindingVar],
    binding: &BindingAssignment,
) -> Option<Result<usize, usize>> {
    binary_search_bound_prefix_from(keys, variables, binding, 0)
}

/// Exponential ("galloping") search, returning the same `Ok(index)`/`Err(insert)`
/// as `keys.binary_search_by(cmp)` for sorted unique keys. The leapfrog seek
/// advances monotonically from a hint, so the target is usually a few positions
/// past it; galloping finds it in `O(log distance)` instead of `O(log n)`, and
/// is never worse than about twice the comparisons of a plain binary search.
/// This is the seek the Leapfrog Triejoin paper specifies (Veldhuizen, ICDT'14).
fn exponential_search_by(
    keys: &[NaturalJoinKey],
    mut cmp: impl FnMut(&NaturalJoinKey) -> Ordering,
) -> Result<usize, usize> {
    let n = keys.len();
    if n == 0 {
        return Err(0);
    }
    // Grow the bracket until its top key is not below the target. `lo` then
    // holds a key strictly below the target (or 0), bracketing the answer.
    let mut hi = 1;
    while hi < n && cmp(&keys[hi - 1]) == Ordering::Less {
        hi *= 2;
    }
    let lo = hi / 2;
    let hi = hi.min(n);
    keys[lo..hi]
        .binary_search_by(&mut cmp)
        .map(|index| lo + index)
        .map_err(|index| lo + index)
}

fn binary_search_bound_prefix_from(
    keys: &[NaturalJoinKey],
    variables: &[BindingVar],
    binding: &BindingAssignment,
    start: usize,
) -> Option<Result<usize, usize>> {
    if !binding.contains_all(variables) {
        return None;
    }

    // The bound prefix is invariant across the whole search, so read it once
    // instead of re-deriving each `binding.bound(var)` (a word/bit computation)
    // inside every one of the ~log2(n) binary-search comparisons. This is the
    // named path-comparison cost: profiling the worst-case-optimal join put ~67%
    // of a join-heavy workload in `cmp_bound_prefix` plus this search. Real join
    // keys have few variables, so a small stack buffer holds the prefix without
    // allocating; the rare wider key falls back to the per-comparison path.
    const PREFIX_STACK: usize = 32;
    let mut prefix_buf = [TermId(0); PREFIX_STACK];
    if let Some(prefix) = prefix_buf.get_mut(..variables.len()) {
        for (slot, &variable) in prefix.iter_mut().zip(variables.iter()) {
            *slot = binding.bound(variable);
        }
        let prefix: &[TermId] = &prefix_buf[..variables.len()];
        let search_from = start.min(keys.len());
        let can_reuse_hint = search_from == 0
            || keys[search_from - 1].cmp_prefix_values(prefix) == Ordering::Less;
        if !can_reuse_hint {
            return Some(keys.binary_search_by(|key| key.cmp_prefix_values(prefix)));
        }
        return Some(
            exponential_search_by(&keys[search_from..], |key| key.cmp_prefix_values(prefix))
                .map(|index| search_from + index)
                .map_err(|index| search_from + index),
        );
    }

    let search_from = start.min(keys.len());
    let can_reuse_hint = search_from == 0
        || keys[search_from - 1].cmp_bound_prefix(variables, binding) == Ordering::Less;
    if !can_reuse_hint {
        return Some(keys.binary_search_by(|key| key.cmp_bound_prefix(variables, binding)));
    }

    Some(
        keys[search_from..]
            .binary_search_by(|key| key.cmp_bound_prefix(variables, binding))
            .map(|index| search_from + index)
            .map_err(|index| search_from + index),
    )
}

/// Reference natural join over signed relations.
pub fn natural_join(
    left: &BindingRelation,
    right: &BindingRelation,
) -> Result<BindingRelation, BindingRelationError> {
    let common = left
        .schema()
        .iter()
        .copied()
        .filter(|variable| right.schema().contains(variable))
        .collect::<Vec<_>>();
    let right_new = right
        .schema()
        .iter()
        .copied()
        .filter(|variable| !left.schema().contains(variable))
        .collect::<Vec<_>>();

    let mut out_schema = left.schema().to_vec();
    out_schema.extend_from_slice(&right_new);
    let mut out = BindingRelation::new(out_schema);

    let left_common = indexes(left, &common)?;
    let right_common = indexes(right, &common)?;
    let right_new_indexes = indexes(right, &right_new)?;

    if common.is_empty() {
        for (left_row, left_weight) in left.rows() {
            for (right_row, right_weight) in right.rows() {
                let output = joined_output_row(left_row, right_row, &right_new_indexes);
                out.add(output, checked_mul_weight(left_weight, right_weight)?)?;
            }
        }
        return Ok(out);
    }

    if left.len() <= right.len() {
        let mut left_index: BTreeMap<NaturalJoinKey, NaturalJoinBucket<'_>> = BTreeMap::new();
        for (row, weight) in left.rows() {
            insert_natural_join_bucket(
                &mut left_index,
                NaturalJoinKey::from_row(row, &left_common),
                row,
                weight,
            );
        }

        for (right_row, right_weight) in right.rows() {
            let key = NaturalJoinKey::from_row(right_row, &right_common);
            if let Some(bucket) = left_index.get(&key) {
                bucket.try_for_each(|left_row, left_weight| {
                    let output = joined_output_row(left_row, right_row, &right_new_indexes);
                    out.add(output, checked_mul_weight(left_weight, right_weight)?)
                })?;
            }
        }
    } else {
        let mut right_index: BTreeMap<NaturalJoinKey, NaturalJoinBucket<'_>> = BTreeMap::new();
        for (row, weight) in right.rows() {
            insert_natural_join_bucket(
                &mut right_index,
                NaturalJoinKey::from_row(row, &right_common),
                row,
                weight,
            );
        }

        for (left_row, left_weight) in left.rows() {
            let key = NaturalJoinKey::from_row(left_row, &left_common);
            if let Some(bucket) = right_index.get(&key) {
                bucket.try_for_each(|right_row, right_weight| {
                    let output = joined_output_row(left_row, right_row, &right_new_indexes);
                    out.add(output, checked_mul_weight(left_weight, right_weight)?)
                })?;
            }
        }
    }

    Ok(out)
}

fn joined_output_row(
    left_row: &[TermId],
    right_row: &[TermId],
    right_new_indexes: &[usize],
) -> BindingRow {
    let output_len = left_row.len() + right_new_indexes.len();
    match output_len {
        0 => Box::from([]),
        1 => Box::from([joined_output_value(
            left_row,
            right_row,
            right_new_indexes,
            0,
        )]),
        2 => Box::from([
            joined_output_value(left_row, right_row, right_new_indexes, 0),
            joined_output_value(left_row, right_row, right_new_indexes, 1),
        ]),
        3 => Box::from([
            joined_output_value(left_row, right_row, right_new_indexes, 0),
            joined_output_value(left_row, right_row, right_new_indexes, 1),
            joined_output_value(left_row, right_row, right_new_indexes, 2),
        ]),
        4 => Box::from([
            joined_output_value(left_row, right_row, right_new_indexes, 0),
            joined_output_value(left_row, right_row, right_new_indexes, 1),
            joined_output_value(left_row, right_row, right_new_indexes, 2),
            joined_output_value(left_row, right_row, right_new_indexes, 3),
        ]),
        _ => {
            let mut output = Vec::with_capacity(output_len);
            output.extend_from_slice(left_row);
            output.extend(right_new_indexes.iter().map(|&index| right_row[index]));
            output.into_boxed_slice()
        }
    }
}

fn joined_output_value(
    left_row: &[TermId],
    right_row: &[TermId],
    right_new_indexes: &[usize],
    output_index: usize,
) -> TermId {
    if output_index < left_row.len() {
        left_row[output_index]
    } else {
        right_row[right_new_indexes[output_index - left_row.len()]]
    }
}

/// Retains visible positive rows from `left` that have at least one visible
/// join partner in `right`.
///
/// This is a sidecar planning primitive for Yannakakis-style acyclic reduction.
/// It deliberately follows the current BindingSpace query convention of
/// pruning over positive visible rows; signed differential algebra remains
/// represented by `natural_join` and `delta_join`.
pub fn semijoin_presence(
    left: &BindingRelation,
    right: &BindingRelation,
) -> Result<BindingRelation, BindingRelationError> {
    let common = left
        .schema()
        .iter()
        .copied()
        .filter(|variable| right.schema().contains(variable))
        .collect::<Vec<_>>();
    if common.is_empty() {
        let mut out = BindingRelation::new(left.schema.clone());
        if right.positive_rows().next().is_some() {
            for (row, weight) in left.rows() {
                if weight > 0 {
                    out.add(NaturalJoinKey::from_complete_row(row).into_row(), weight)?;
                }
            }
        }
        return Ok(out);
    }

    let left_common = indexes(left, &common)?;
    let right_common = indexes(right, &common)?;
    let right_keys = right
        .positive_rows()
        .map(|row| NaturalJoinKey::from_row(row, &right_common))
        .collect::<BTreeSet<_>>();

    let mut out = BindingRelation::new(left.schema.clone());
    for (row, weight) in left.rows() {
        if weight > 0 && right_keys.contains(&NaturalJoinKey::from_row(row, &left_common)) {
            out.add(NaturalJoinKey::from_complete_row(row).into_row(), weight)?;
        }
    }
    Ok(out)
}

/// Result of applying bottom-up then top-down semijoin reduction over an
/// acyclic relation forest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemijoinReduction {
    /// Reduced relations in the original input order.
    pub relations: Box<[BindingRelation]>,
    /// Number of child-to-parent reductions applied.
    pub bottom_up_passes: usize,
    /// Number of parent-to-child reductions applied.
    pub top_down_passes: usize,
    /// Visible positive rows removed across both sweeps.
    pub removed_rows: usize,
}

/// Applies a Yannakakis-style full reducer over visible BindingSpace rows.
///
/// `edges` are `(child, parent)` relation indexes in GYO ear-removal order.
/// The forward pass reduces parents from children; the reverse pass reduces
/// children from their already-reduced parents. The function does not execute
/// the final join and does not affect canonical PathMap/ACT semantics.
pub fn semijoin_reduce_presence(
    relations: &[BindingRelation],
    edges: &[(usize, usize)],
) -> Result<SemijoinReduction, BindingRelationError> {
    let mut reduced = relations.to_vec();
    let mut removed_rows = 0usize;

    for &(child, parent) in edges {
        validate_semijoin_edge(reduced.len(), child, parent)?;
        let before = reduced[parent].positive_rows().count();
        let next = semijoin_presence(&reduced[parent], &reduced[child])?;
        reduced[parent] = next;
        removed_rows = checked_add_removed_rows(
            removed_rows,
            before,
            reduced[parent].positive_rows().count(),
        )?;
    }

    for &(child, parent) in edges.iter().rev() {
        validate_semijoin_edge(reduced.len(), child, parent)?;
        let before = reduced[child].positive_rows().count();
        let next = semijoin_presence(&reduced[child], &reduced[parent])?;
        reduced[child] = next;
        removed_rows =
            checked_add_removed_rows(removed_rows, before, reduced[child].positive_rows().count())?;
    }

    Ok(SemijoinReduction {
        relations: reduced.into_boxed_slice(),
        bottom_up_passes: edges.len(),
        top_down_passes: edges.len(),
        removed_rows,
    })
}

/// α-acyclicity decision and join tree produced by Graham–Yu–Özsoyoğlu (GYO)
/// ear removal over the binding-relation join hypergraph.
///
/// The join hypergraph has one vertex per [`BindingVar`] and one hyperedge per
/// input relation (its schema). A relation is an *ear* when the variables it
/// shares with any other still-present relation are all contained in one other
/// relation (its *witness*); a relation sharing nothing with the rest is an ear
/// with no witness. GYO repeatedly removes ears: the hypergraph is α-acyclic
/// iff every relation is removed this way, and the witnesses form a join tree.
///
/// References:
/// - M. Yannakakis, "Algorithms for acyclic database schemes," VLDB 1981,
///   pp. 82–94, full reducer + join-tree evaluation of acyclic joins.
/// - C. Beeri, R. Fagin, D. Maier, M. Yannakakis, "On the desirability of
///   acyclic database schemes," J. ACM 30(3):479–513, 1983, α-acyclic ⇔ has a
///   join tree ⇔ admits a full reducer; the GYO test.
/// - M. H. Graham, "On the universal relation," Univ. Toronto TR, 1979; and
///   C. T. Yu, M. Z. Özsoyoğlu, "An algorithm for tree-query membership of a
///   distributed query," COMPSAC 1979, the GYO ear-removal procedure
///   (discovered independently).
/// - P. A. Bernstein, N. Goodman, "Power of natural semijoins," SIAM J. Comput.
///   10(4):751–771, 1981, the two-pass leaf↔root natural-semijoin full reducer
///   over a join tree that [`semijoin_reduce_presence`] runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GyoJoinTree {
    /// True iff the join hypergraph is α-acyclic (GYO reduces it to empty).
    pub acyclic: bool,
    /// Relation indexes in ear-removal order. A child is always removed before
    /// its witness parent, so this is a valid leaves→root reduction order.
    pub ear_order: Box<[usize]>,
    /// Join-tree edges `(child, parent)` in ear-removal order, exactly the
    /// edges [`semijoin_reduce_presence`] consumes to run a full reducer. A
    /// relation sharing no variables with the rest (a separate join-tree
    /// component / Cartesian factor) contributes no edge.
    pub edges: Box<[(usize, usize)]>,
    /// Relations still present when ear removal got stuck: the α-cyclic core.
    /// Empty iff `acyclic` is true.
    pub residual: Box<[usize]>,
}

/// Computes the GYO ear-removal join tree of the relations' join hypergraph.
///
/// This is a pure planning primitive: it inspects only relation *schemas*
/// (variable sets), never row data, and never touches PathMap/ACT or `exec`
/// selection. When `acyclic`, [`GyoJoinTree::edges`] drives a Yannakakis full
/// reducer ([`semijoin_reduce_presence`]) that removes exactly the dangling
/// tuples, preserving the MM2 result *set* of the join while shrinking
/// intermediate work. When cyclic (e.g. the triangle query `R(a,b), S(b,c),
/// T(c,a)`), `residual` names the core a worst-case-optimal kernel (LFTJ /
/// Generic Join) must still handle.
///
/// Ear and witness choice is deterministic (lowest index first) so the tree is
/// reproducible; GYO's acyclicity verdict is confluent regardless of the order
/// ears are removed (BFMY 1983). Cost is `O(n³·a)` for `n` relations of arity
/// `a`, which is negligible for the small pattern conjunctions MM2 admits (at
/// most 64 variable mentions per computation step).
pub fn gyo_join_tree(relations: &[BindingRelation]) -> GyoJoinTree {
    let relation_count = relations.len();
    let var_sets = relations
        .iter()
        .map(|relation| {
            let mut vars = relation.schema().to_vec();
            vars.sort_unstable();
            vars.dedup();
            vars.into_boxed_slice()
        })
        .collect::<Vec<Box<[BindingVar]>>>();

    let mut remaining = vec![true; relation_count];
    let mut remaining_count = relation_count;
    let mut ear_order = Vec::with_capacity(relation_count);
    let mut edges = Vec::new();

    while remaining_count > 0 {
        let mut ear = None;
        'scan: for candidate in 0..relation_count {
            if !remaining[candidate] {
                continue;
            }
            let shared = shared_remaining_vertices(candidate, &var_sets, &remaining);
            if shared.is_empty() {
                // Isolated relation: an ear with no witness, a join-tree root
                // (and a Cartesian factor relative to the other components).
                ear = Some((candidate, None));
                break 'scan;
            }
            for witness in 0..relation_count {
                if witness == candidate || !remaining[witness] {
                    continue;
                }
                if is_sorted_subset(&shared, &var_sets[witness]) {
                    ear = Some((candidate, Some(witness)));
                    break 'scan;
                }
            }
        }

        match ear {
            Some((removed, witness)) => {
                remaining[removed] = false;
                remaining_count -= 1;
                ear_order.push(removed);
                if let Some(parent) = witness {
                    edges.push((removed, parent));
                }
            }
            // No remaining relation is an ear: the rest is an α-cyclic core.
            None => break,
        }
    }

    let residual = (0..relation_count)
        .filter(|&index| remaining[index])
        .collect::<Vec<_>>();
    GyoJoinTree {
        acyclic: remaining_count == 0,
        ear_order: ear_order.into_boxed_slice(),
        edges: edges.into_boxed_slice(),
        residual: residual.into_boxed_slice(),
    }
}

/// Variables of relation `index` that also occur in some other still-remaining
/// relation. The schema is already sorted-unique, so the result is sorted and
/// deduplicated. These "attachment" vertices are what an ear's witness must
/// fully contain.
fn shared_remaining_vertices(
    index: usize,
    var_sets: &[Box<[BindingVar]>],
    remaining: &[bool],
) -> Vec<BindingVar> {
    var_sets[index]
        .iter()
        .copied()
        .filter(|variable| {
            var_sets
                .iter()
                .enumerate()
                .any(|(other, vars)| other != index && remaining[other] && vars.contains(variable))
        })
        .collect()
}

/// Tests sorted-unique `needle` ⊆ sorted-unique `haystack` with a single
/// forward merge walk.
fn is_sorted_subset(needle: &[BindingVar], haystack: &[BindingVar]) -> bool {
    let mut haystack = haystack.iter();
    'next: for variable in needle {
        for candidate in haystack.by_ref() {
            match candidate.cmp(variable) {
                Ordering::Less => {}
                Ordering::Equal => continue 'next,
                Ordering::Greater => return false,
            }
        }
        return false;
    }
    true
}

/// Largest conjunction the hypertree-decomposition search will try. The search
/// enumerates set partitions, so it is bounded by the Bell number; MM2
/// conjunctions are small, and beyond this the planner falls back to a global
/// worst-case-optimal join.
const HYPERTREE_MAX_RELATIONS: usize = 8;

/// One bag of a hypertree decomposition: the relations joined at this node and
/// the union of their variables.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HyperTreeBag {
    /// Relation indexes joined at this bag (the cover lambda; `len <= width`).
    pub relations: Box<[usize]>,
    /// Bag variables: the sorted-unique union of the joined relations' schemas.
    pub variables: Box<[BindingVar]>,
}

/// A width-bounded hypertree decomposition of the join hypergraph.
///
/// The bags partition the relations (each relation in exactly one bag), and the
/// bag variable-sets form an alpha-acyclic hypergraph. Joining the bag relations
/// reproduces the original join (join grouping), and the acyclic bag hypergraph
/// admits Yannakakis-over-bags. See `kernel/resources/hypertree_decomposition_plan.md`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HyperTreeDecomposition {
    /// Bags in partition order (each block's first relation index is increasing).
    pub bags: Box<[HyperTreeBag]>,
    /// Decomposition width: the largest bag's relation count. Width 1 means the
    /// body is already alpha-acyclic.
    pub width: usize,
}

/// Searches for a hypertree decomposition of `relations` of width `<= max_width`.
///
/// It partitions the relations into bags of at most `max_width` relations and
/// returns the smallest-width partition whose bag variable-sets are alpha-acyclic
/// (checked by reusing [`gyo_join_tree`] on the bag schemas). Returns `None` when
/// no such partition exists within `max_width`, or when the conjunction is larger
/// than [`HYPERTREE_MAX_RELATIONS`]; the caller then keeps a global join kernel.
///
/// This is a pure, schema-only planning primitive. A partition (each relation in
/// one bag) is a sound generalized hypertree decomposition: the join of the bag
/// relations equals the original join because natural join is associative and
/// commutative, so a wrong or missing decomposition only changes performance,
/// never answers. A partition width is an upper bound on the generalized
/// hypertree width; a cover that shares relations across bags can be narrower
/// (the det-k-decomp refinement in the design plan).
pub fn hypertree_decomposition(
    relations: &[BindingRelation],
    max_width: usize,
) -> Option<HyperTreeDecomposition> {
    let relation_count = relations.len();
    if relation_count == 0 || max_width == 0 || relation_count > HYPERTREE_MAX_RELATIONS {
        return None;
    }

    let var_sets = relations
        .iter()
        .map(|relation| {
            let mut vars = relation.schema().to_vec();
            vars.sort_unstable();
            vars.dedup();
            vars.into_boxed_slice()
        })
        .collect::<Vec<Box<[BindingVar]>>>();

    // Among all partitions with blocks of at most `max_width`, pick the smallest
    // width, and among those the most bags (the fewest merges, so only the cyclic
    // cores get grouped and acyclic relations stay in their own bag). Tiny `n`
    // makes the full enumeration cheap, and a deterministic choice keeps the
    // planner's routing heuristic stable.
    let mut best: Option<(usize, usize, Vec<Vec<usize>>)> = None;
    let mut blocks = Vec::new();
    enumerate_bounded_partitions(0, relation_count, max_width, &mut blocks, &mut |blocks| {
        let width = blocks.iter().map(Vec::len).max().unwrap_or(0);
        let bag_count = blocks.len();
        if let Some((best_width, best_bags, _)) = &best {
            if width > *best_width || (width == *best_width && bag_count <= *best_bags) {
                return false;
            }
        }
        // Reject bags whose relations do not share variables: joining them would
        // be a Cartesian product, which defeats the point of the bag.
        if !blocks
            .iter()
            .all(|block| block_is_connected(block, &var_sets))
        {
            return false;
        }
        let bag_relations = blocks
            .iter()
            .map(|block| BindingRelation::new(bag_variables(block, &var_sets)))
            .collect::<Vec<_>>();
        if gyo_join_tree(&bag_relations).acyclic {
            best = Some((width, bag_count, blocks.iter().cloned().collect()));
        }
        false
    });

    best.map(|(width, _, blocks)| {
        let bags = blocks
            .into_iter()
            .map(|block| HyperTreeBag {
                variables: bag_variables(&block, &var_sets).into_boxed_slice(),
                relations: block.into_boxed_slice(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        HyperTreeDecomposition { bags, width }
    })
}

/// Sorted-unique union of the variables of the relations in `block`.
fn bag_variables(block: &[usize], var_sets: &[Box<[BindingVar]>]) -> Vec<BindingVar> {
    let mut vars = block
        .iter()
        .flat_map(|&index| var_sets[index].iter().copied())
        .collect::<Vec<_>>();
    vars.sort_unstable();
    vars.dedup();
    vars
}

/// Whether the relations in `block` are connected through shared variables, so
/// joining them is not a Cartesian product. A singleton block is connected.
fn block_is_connected(block: &[usize], var_sets: &[Box<[BindingVar]>]) -> bool {
    if block.len() <= 1 {
        return true;
    }
    let mut reached = vec![false; block.len()];
    reached[0] = true;
    let mut frontier = vec![0usize];
    let mut reached_count = 1usize;
    while let Some(current) = frontier.pop() {
        for other in 0..block.len() {
            if !reached[other]
                && share_sorted_variable(&var_sets[block[current]], &var_sets[block[other]])
            {
                reached[other] = true;
                reached_count += 1;
                frontier.push(other);
            }
        }
    }
    reached_count == block.len()
}

/// Whether two sorted-unique variable lists share any variable.
fn share_sorted_variable(left: &[BindingVar], right: &[BindingVar]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => return true,
        }
    }
    false
}

/// Minimum number of relations whose variables cover every variable in the body:
/// the integral edge cover number of the join hypergraph. It upper-bounds the
/// fractional edge cover, so `N^min_edge_cover` upper-bounds the AGM output bound
/// of a single global worst-case-optimal join. A hypertree decomposition of width
/// below this is worth running instead of a global join.
///
/// Exact for up to [`HYPERTREE_MAX_RELATIONS`] relations (a bitmask search over
/// subsets); beyond that it returns the relation count, a safe upper bound.
pub fn min_edge_cover(relations: &[BindingRelation]) -> usize {
    let relation_count = relations.len();
    if relation_count == 0 {
        return 0;
    }
    let var_sets = relations
        .iter()
        .map(|relation| {
            let mut vars = relation.schema().to_vec();
            vars.sort_unstable();
            vars.dedup();
            vars.into_boxed_slice()
        })
        .collect::<Vec<Box<[BindingVar]>>>();
    let mut all_vars = var_sets
        .iter()
        .flat_map(|vars| vars.iter().copied())
        .collect::<Vec<_>>();
    all_vars.sort_unstable();
    all_vars.dedup();
    if all_vars.is_empty() {
        return 0;
    }
    if relation_count > HYPERTREE_MAX_RELATIONS {
        return relation_count;
    }

    let mut best = relation_count;
    for mask in 1u32..(1u32 << relation_count) {
        let size = mask.count_ones() as usize;
        if size >= best {
            continue;
        }
        if mask_covers(mask, &var_sets, &all_vars) {
            best = size;
        }
    }
    best
}

/// Whether the relations selected by `mask` together cover every variable in
/// `all_vars`.
fn mask_covers(mask: u32, var_sets: &[Box<[BindingVar]>], all_vars: &[BindingVar]) -> bool {
    all_vars.iter().all(|variable| {
        (0..var_sets.len()).any(|index| {
            (mask & (1u32 << index)) != 0 && var_sets[index].binary_search(variable).is_ok()
        })
    })
}

/// A variable-at-a-time join order that puts more selective variables first.
///
/// Each variable is keyed by the smallest distinct-value domain among the
/// relations that mention it: binding such a variable early prunes the search.
/// Exact, since it reads the materialized sidecar relations. The order covers
/// exactly the variables appearing in `relations`, with the variable id as a
/// deterministic tie-break. A cost-based planner uses this in place of the
/// root-domain heuristic order.
pub fn selectivity_variable_order(relations: &[BindingRelation]) -> Vec<BindingVar> {
    let mut variables = relations
        .iter()
        .flat_map(|relation| relation.schema().iter().copied())
        .collect::<Vec<_>>();
    variables.sort_unstable();
    variables.dedup();

    let smallest_domain = |variable: BindingVar| -> usize {
        relations
            .iter()
            .filter(|relation| relation.schema().contains(&variable))
            .map(|relation| {
                relation
                    .distinct_value_count(variable)
                    .unwrap_or(usize::MAX)
            })
            .min()
            .unwrap_or(usize::MAX)
    };

    variables.sort_by_key(|&variable| (smallest_domain(variable), variable.0));
    variables
}

/// Integral AGM bound for the sub-query induced by a subset of variables (a bit
/// mask over `all_vars`): the minimum over relation covers of `mask` of the
/// product of the cover relations' sizes, using only the schemas and sizes (no
/// data). Empty mask is 1. Helper for [`dp_variable_order`].
fn agm_for_mask(relations: &[BindingRelation], all_vars: &[BindingVar], mask: u32) -> u128 {
    let needed: Vec<BindingVar> = all_vars
        .iter()
        .enumerate()
        .filter(|(index, _)| mask & (1u32 << index) != 0)
        .map(|(_, &variable)| variable)
        .collect();
    if needed.is_empty() {
        return 1;
    }
    let relation_count = relations.len();
    let mut best = u128::MAX;
    for cover in 1u32..(1u32 << relation_count) {
        let mut covered = 0u32;
        let mut product = 1u128;
        for (relation_index, relation) in relations.iter().enumerate() {
            if cover & (1u32 << relation_index) == 0 {
                continue;
            }
            product = product.saturating_mul(relation.positive_rows().count() as u128);
            for (need_index, variable) in needed.iter().enumerate() {
                if relation.schema().contains(variable) {
                    covered |= 1u32 << need_index;
                }
            }
        }
        if covered == (1u32 << needed.len()) - 1 {
            best = best.min(product);
        }
    }
    best
}

/// Scale turning a base-2 logarithm into an integer weight. 1024 keeps about
/// three decimal digits of each log2, ample for an ordering heuristic.
const LOG_WEIGHT_SCALE: f64 = 1024.0;
/// Saturating sentinel for a variable subset no relation cover reaches.
const COVER_INF: u64 = u64::MAX / 4;

/// Minimum log-domain AGM cost for every variable subset, built in one pass.
///
/// `cost[S]` is `round(SCALE * log2(agm_for_mask(S)))`: the cheapest relation
/// cover of `S` measured by the product of the cover relations' sizes, taken in
/// the log domain where that product is a sum. So the integral min-product cover
/// becomes a min-weight set cover with log-cardinality weights. We enumerate the
/// `2^m` relation subsets once (each reuses its predecessor with the low bit
/// cleared, so no inner relation loop), record the cheapest summed weight that
/// reaches each covered-variable union, then run a min-over-supersets transform
/// so every subset inherits the best cover of any superset. This replaces the
/// per-subset `O(2^m)` rescan in [`agm_for_mask`] with `O(2^m + n*2^n)` total,
/// and the log domain avoids the u128 product saturation that otherwise collapses
/// every large cover to `u128::MAX` and erases the ordering. Lengths are bounded
/// by [`dp_variable_order`]'s guard, so the temporary `2^m` arrays stay small.
fn cover_cost_table(relations: &[BindingRelation], all_vars: &[BindingVar]) -> Vec<u64> {
    let n = all_vars.len();
    let m = relations.len();
    let var_state_count = 1usize << n;

    // Per-relation variable mask over `all_vars`, and its integer log2 weight.
    // An empty relation drives the product to zero; clamping its row count to 1
    // gives it weight 0, so a cover using it is correctly the cheapest.
    let mut rel_var_mask = vec![0u64; m];
    let mut rel_weight = vec![0u64; m];
    for (j, relation) in relations.iter().enumerate() {
        let mut var_mask = 0u64;
        for (index, &var) in all_vars.iter().enumerate() {
            if relation.schema().contains(&var) {
                var_mask |= 1u64 << index;
            }
        }
        rel_var_mask[j] = var_mask;
        let rows = relation.positive_rows().count().max(1) as f64;
        rel_weight[j] = (rows.log2() * LOG_WEIGHT_SCALE).round() as u64;
    }

    let mut cost = vec![COVER_INF; var_state_count];
    cost[0] = 0;

    // Subset recurrence over the 2^m relation subsets: each state reuses the
    // union and summed weight of itself with the least-significant bit removed.
    let rel_state_count = 1usize << m;
    let mut unions = vec![0u64; rel_state_count];
    let mut sums = vec![0u64; rel_state_count];
    for subset in 1..rel_state_count {
        let previous = subset & (subset - 1);
        let bit = subset.trailing_zeros() as usize;
        let union = unions[previous] | rel_var_mask[bit];
        let sum = sums[previous].saturating_add(rel_weight[bit]).min(COVER_INF);
        unions[subset] = union;
        sums[subset] = sum;
        let reached = union as usize;
        if sum < cost[reached] {
            cost[reached] = sum;
        }
    }

    // Min-over-supersets: cost[S] becomes the minimum cost at any superset of S.
    for bit in 0..n {
        let bit_mask = 1usize << bit;
        for mask in 0..var_state_count {
            if mask & bit_mask == 0 {
                let superset_cost = cost[mask | bit_mask];
                if superset_cost < cost[mask] {
                    cost[mask] = superset_cost;
                }
            }
        }
    }
    cost
}

/// Optimal variable order by subset dynamic programming over the AGM cost model:
/// `dp[S] = agm(S) + min over v in S of dp[S \ {v}]`, so the order minimizes the
/// sum of the prefix AGM bounds, the worst-case-optimal join's intermediate work.
/// This is the Selinger subset DP (Selinger et al., SIGMOD 1979) applied to the
/// variable-elimination order; unlike greedy selectivity it weighs the whole
/// order, so it wins on asymmetric queries where a globally small domain is the
/// wrong first choice. The `2^n * n` states each need the AGM cost of their
/// variable subset; [`cover_cost_table`] precomputes all of them once in
/// `O(2^m + n*2^n)` instead of rescanning the `2^m` covers per state, so the
/// exact DP stays affordable up to the guard below before deferring to the
/// greedy selectivity order.
pub fn dp_variable_order(relations: &[BindingRelation]) -> Vec<BindingVar> {
    let mut all_vars = relations
        .iter()
        .flat_map(|relation| relation.schema().iter().copied())
        .collect::<Vec<_>>();
    all_vars.sort_unstable();
    all_vars.dedup();

    let n = all_vars.len();
    if n == 0 {
        return all_vars;
    }
    // The cover table and DP arrays are sized 2^n and 2^m; 20 keeps the largest
    // transient near a million entries, computed once per prepared query.
    if n > 20 || relations.len() > 20 {
        return selectivity_variable_order(relations);
    }

    let cover_cost = cover_cost_table(relations, &all_vars);
    let full = 1usize << n;
    let mut dp = vec![COVER_INF; full];
    let mut last_added = vec![usize::MAX; full];
    dp[0] = 0;
    for mask in 1..full {
        let agm = cover_cost[mask];
        for v in 0..n {
            if mask & (1 << v) == 0 {
                continue;
            }
            let prev = mask & !(1 << v);
            if dp[prev] == COVER_INF {
                continue;
            }
            let cost = dp[prev].saturating_add(agm).min(COVER_INF);
            if cost < dp[mask] {
                dp[mask] = cost;
                last_added[mask] = v;
            }
        }
    }

    let mut order = Vec::with_capacity(n);
    let mut mask = full - 1;
    while mask != 0 {
        let v = last_added[mask];
        order.push(all_vars[v]);
        mask &= !(1 << v);
    }
    order.reverse();
    order
}

/// Saturating product of the positive-row counts of the relations at `indices`.
fn size_product(relations: &[BindingRelation], indices: impl IntoIterator<Item = usize>) -> u128 {
    indices.into_iter().fold(1u128, |product, index| {
        product.saturating_mul(relations[index].positive_rows().count() as u128)
    })
}

/// Upper bound on the join output size: the minimum over edge covers of the
/// product of the cover relations' sizes (the integral AGM bound). It bounds the
/// cost of a single global worst-case-optimal join using measured relation sizes,
/// not just the cover count. Returns 0 for no relations or no variables; beyond
/// [`HYPERTREE_MAX_RELATIONS`] it returns the full product, a safe upper bound.
pub fn agm_size_bound(relations: &[BindingRelation]) -> u128 {
    let relation_count = relations.len();
    if relation_count == 0 {
        return 0;
    }
    let var_sets = relations
        .iter()
        .map(|relation| {
            let mut vars = relation.schema().to_vec();
            vars.sort_unstable();
            vars.dedup();
            vars.into_boxed_slice()
        })
        .collect::<Vec<Box<[BindingVar]>>>();
    let mut all_vars = var_sets
        .iter()
        .flat_map(|vars| vars.iter().copied())
        .collect::<Vec<_>>();
    all_vars.sort_unstable();
    all_vars.dedup();
    if all_vars.is_empty() {
        return 0;
    }
    if relation_count > HYPERTREE_MAX_RELATIONS {
        return size_product(relations, 0..relation_count);
    }

    let mut best = u128::MAX;
    for mask in 1u32..(1u32 << relation_count) {
        if mask_covers(mask, &var_sets, &all_vars) {
            let indices = (0..relation_count).filter(|&index| (mask & (1u32 << index)) != 0);
            best = best.min(size_product(relations, indices));
        }
    }
    best
}

const LP_EPSILON: f64 = 1e-9;

/// The fractional AGM bound on a join's output size: `exp` of the optimum of the
/// fractional edge-cover LP weighted by log-cardinalities. Unlike the integral
/// [`agm_size_bound`] (a minimum over integer covers), this is the true AGM
/// bound: a triangle of N-row edges is N^1.5, not N^2. Returns 0 when a relation
/// is empty (the join is empty) and `INFINITY` when a variable is uncovered.
/// The dual cover LP is solved by a small primal simplex. References Atserias,
/// Grohe, Marx, "Size Bounds and Query Plans for Relational Joins", FOCS 2008.
pub fn fractional_agm_bound(relations: &[BindingRelation]) -> f64 {
    let mut variables: Vec<BindingVar> = relations
        .iter()
        .flat_map(|relation| relation.schema().iter().copied())
        .collect();
    variables.sort_unstable();
    variables.dedup();
    if variables.is_empty() {
        return 1.0;
    }
    let bit = |variable: BindingVar| variables.iter().position(|&v| v == variable).unwrap();

    let mut matrix: Vec<Vec<f64>> = Vec::new();
    let mut bounds: Vec<f64> = Vec::new();
    for relation in relations {
        let size = relation.positive_rows().count();
        if size == 0 {
            return 0.0;
        }
        let mut row = vec![0.0; variables.len()];
        for &variable in relation.schema() {
            row[bit(variable)] = 1.0;
        }
        matrix.push(row);
        bounds.push((size as f64).ln());
    }
    for index in 0..variables.len() {
        if !matrix.iter().any(|row| row[index] != 0.0) {
            return f64::INFINITY;
        }
    }
    let objective = vec![1.0; variables.len()];
    LpSimplex::new(matrix, bounds, objective)
        .solve()
        .map(f64::exp)
        .unwrap_or(f64::INFINITY)
}

/// Small deterministic primal simplex for `max c*x` subject to `A*x <= b`,
/// `x >= 0`. The fractional-cover dual starts feasible at `x = 0`, so no phase-I
/// artificial-variable pass is needed.
struct LpSimplex {
    rows: usize,
    columns: usize,
    basis: Vec<usize>,
    nonbasis: Vec<usize>,
    tableau: Vec<Vec<f64>>,
}

impl LpSimplex {
    fn new(matrix: Vec<Vec<f64>>, bounds: Vec<f64>, objective: Vec<f64>) -> Self {
        let rows = bounds.len();
        let columns = objective.len();
        let mut tableau = vec![vec![0.0; columns + 1]; rows + 1];
        for row in 0..rows {
            tableau[row][..columns].copy_from_slice(&matrix[row][..columns]);
            tableau[row][columns] = bounds[row];
        }
        for column in 0..columns {
            tableau[rows][column] = -objective[column];
        }
        Self {
            rows,
            columns,
            basis: (0..rows).map(|row| columns + row).collect(),
            nonbasis: (0..columns).collect(),
            tableau,
        }
    }

    fn pivot(&mut self, leaving: usize, entering: usize) {
        let inverse = 1.0 / self.tableau[leaving][entering];
        for row in 0..=self.rows {
            if row == leaving {
                continue;
            }
            for column in 0..=self.columns {
                if column == entering {
                    continue;
                }
                self.tableau[row][column] -=
                    self.tableau[leaving][column] * self.tableau[row][entering] * inverse;
            }
        }
        for column in 0..=self.columns {
            if column != entering {
                self.tableau[leaving][column] *= inverse;
            }
        }
        for row in 0..=self.rows {
            if row != leaving {
                self.tableau[row][entering] *= -inverse;
            }
        }
        self.tableau[leaving][entering] = inverse;
        std::mem::swap(&mut self.basis[leaving], &mut self.nonbasis[entering]);
    }

    fn solve(mut self) -> Option<f64> {
        loop {
            let entering = (0..self.columns)
                .filter(|column| self.tableau[self.rows][*column] < -LP_EPSILON)
                .min_by(|left, right| {
                    self.tableau[self.rows][*left]
                        .total_cmp(&self.tableau[self.rows][*right])
                        .then_with(|| self.nonbasis[*left].cmp(&self.nonbasis[*right]))
                });
            let Some(entering) = entering else {
                return Some(self.tableau[self.rows][self.columns].max(0.0));
            };
            let leaving = (0..self.rows)
                .filter(|row| self.tableau[*row][entering] > LP_EPSILON)
                .min_by(|left, right| {
                    let left_ratio =
                        self.tableau[*left][self.columns] / self.tableau[*left][entering];
                    let right_ratio =
                        self.tableau[*right][self.columns] / self.tableau[*right][entering];
                    left_ratio
                        .total_cmp(&right_ratio)
                        .then_with(|| self.basis[*left].cmp(&self.basis[*right]))
                });
            let leaving = leaving?;
            self.pivot(leaving, entering);
        }
    }
}

/// Cost of evaluating `relations` through `decomposition`: the largest bag's
/// materialization upper bound (the product of its relations' sizes). Yannakakis
/// over the bags is dominated by this widest bag.
pub fn ghd_size_cost(
    relations: &[BindingRelation],
    decomposition: &HyperTreeDecomposition,
) -> u128 {
    decomposition
        .bags
        .iter()
        .map(|bag| size_product(relations, bag.relations.iter().copied()))
        .max()
        .unwrap_or(0)
}

/// Enumerates every set partition of `{0..relation_count}` whose blocks have at
/// most `max_block` elements, calling `visit` on each. Blocks are kept in
/// canonical order (each block's first element increases), so every partition is
/// produced once. `visit` returns `true` to stop the search early.
fn enumerate_bounded_partitions(
    item: usize,
    relation_count: usize,
    max_block: usize,
    blocks: &mut Vec<Vec<usize>>,
    visit: &mut dyn FnMut(&[Vec<usize>]) -> bool,
) -> bool {
    if item == relation_count {
        return visit(blocks);
    }

    for index in 0..blocks.len() {
        if blocks[index].len() < max_block {
            blocks[index].push(item);
            if enumerate_bounded_partitions(item + 1, relation_count, max_block, blocks, visit) {
                blocks[index].pop();
                return true;
            }
            blocks[index].pop();
        }
    }

    blocks.push(vec![item]);
    let stop = enumerate_bounded_partitions(item + 1, relation_count, max_block, blocks, visit);
    blocks.pop();
    stop
}

/// Materializes each bag of a decomposition by joining its relations into one
/// relation over the bag variables.
fn materialize_bags(
    relations: &[BindingRelation],
    decomposition: &HyperTreeDecomposition,
) -> Result<Vec<BindingRelation>, BindingRelationError> {
    decomposition
        .bags
        .iter()
        .map(|bag| {
            let inputs = bag
                .relations
                .iter()
                .map(|&index| relations[index].clone())
                .collect::<Vec<_>>();
            generic_join(&inputs, &bag.variables)
        })
        .collect()
}

/// Evaluates the join of `relations` through a hypertree decomposition:
/// materialize each bag, then run the Yannakakis full reducer and final join over
/// the bag relations. The output set equals [`generic_join`] over all relations,
/// because joining the bags reproduces the original join and the reducer removes
/// only dangling tuples. `output_order` must hold exactly the variables of all
/// relations.
///
/// When the bags are not alpha-acyclic (a malformed decomposition), it falls back
/// to a direct [`generic_join`], so a wrong decomposition can only cost time, not
/// correctness.
pub fn ghd_join(
    relations: &[BindingRelation],
    decomposition: &HyperTreeDecomposition,
    output_order: &[BindingVar],
) -> Result<BindingRelation, BindingRelationError> {
    let bag_relations = materialize_bags(relations, decomposition)?;
    let tree = gyo_join_tree(&bag_relations);
    if !tree.acyclic {
        return generic_join(relations, output_order);
    }
    let reduced = semijoin_reduce_presence(&bag_relations, &tree.edges)?;
    generic_join(&reduced.relations, output_order)
}

/// Counts the join output through a hypertree decomposition without
/// materializing the flat result. Matches [`generic_join_count`] over all
/// relations.
pub fn ghd_join_count(
    relations: &[BindingRelation],
    decomposition: &HyperTreeDecomposition,
    output_order: &[BindingVar],
) -> Result<GenericJoinCount, BindingRelationError> {
    let bag_relations = materialize_bags(relations, decomposition)?;
    let tree = gyo_join_tree(&bag_relations);
    if !tree.acyclic {
        return generic_join_count(relations, output_order);
    }
    let reduced = semijoin_reduce_presence(&bag_relations, &tree.edges)?;
    generic_join_count(&reduced.relations, output_order)
}

/// Checks whether the decomposed join has any output row, matching
/// [`generic_join_exists`] over all relations.
pub fn ghd_join_exists(
    relations: &[BindingRelation],
    decomposition: &HyperTreeDecomposition,
    output_order: &[BindingVar],
) -> Result<GenericJoinExistence, BindingRelationError> {
    let bag_relations = materialize_bags(relations, decomposition)?;
    let tree = gyo_join_tree(&bag_relations);
    if !tree.acyclic {
        return generic_join_exists(relations, output_order);
    }
    let reduced = semijoin_reduce_presence(&bag_relations, &tree.edges)?;
    generic_join_exists(&reduced.relations, output_order)
}

/// Differential natural join: Δ(A ⋈ B) = ΔA ⋈ B' + A ⋈ ΔB.
pub fn delta_join(
    old_left: &BindingRelation,
    new_right: &BindingRelation,
    delta_left: &BindingRelation,
    delta_right: &BindingRelation,
) -> Result<BindingRelation, BindingRelationError> {
    let mut out = natural_join(delta_left, new_right)?;
    out.union_assign(&natural_join(old_left, delta_right)?)?;
    Ok(out)
}

/// Folds a left-deep natural join over `relations` in index order. The fixed
/// order makes the output schema deterministic, so several folds can be summed.
/// Returns `None` for an empty input.
fn natural_join_all(
    relations: &[BindingRelation],
) -> Result<Option<BindingRelation>, BindingRelationError> {
    let mut accumulated: Option<BindingRelation> = None;
    for relation in relations {
        accumulated = Some(match accumulated {
            None => relation.clone(),
            Some(joined) => natural_join(&joined, relation)?,
        });
    }
    Ok(accumulated)
}

/// Multiway differential natural join: the signed change to
/// `R_0 join ... join R_{n-1}` when each factor moves from `olds[i]` to
/// `news[i] = olds[i] + deltas[i]`.
///
/// It sums one delta term per factor: in term `i`, factors before `i` are old,
/// factor `i` is its delta, and factors after `i` are new. Telescoping shows the
/// sum equals `join(news) - join(olds)`, and it expands the product change once
/// per factor rather than over the `2^n - 1` naive subsets. The binary case
/// (`n == 2`) is [`delta_join`]. This lifts incremental maintenance from a single
/// join to a whole multi-pattern conjunction (recurring MM2 rules).
///
/// All three slices must have the same length. Returns an empty-schema relation
/// for `n == 0`.
pub fn delta_multi_join(
    olds: &[BindingRelation],
    news: &[BindingRelation],
    deltas: &[BindingRelation],
) -> Result<BindingRelation, BindingRelationError> {
    let factor_count = olds.len();
    if news.len() != factor_count || deltas.len() != factor_count {
        return Err(BindingRelationError::SchemaMismatch);
    }
    if factor_count == 0 {
        return Ok(BindingRelation::new(Vec::new()));
    }

    let mut result: Option<BindingRelation> = None;
    for changed in 0..factor_count {
        // term = R_0 join ... join R_{n-1} folded in index order, with factor
        // `changed` as its delta, earlier factors old, later factors new.
        let mut term: Option<BindingRelation> = None;
        for index in 0..factor_count {
            let factor = if index < changed {
                &olds[index]
            } else if index == changed {
                &deltas[changed]
            } else {
                &news[index]
            };
            term = Some(match term {
                None => factor.clone(),
                Some(joined) => natural_join(&joined, factor)?,
            });
        }
        let term = term.expect("factor_count >= 1 so the term fold produced a relation");
        result = Some(match result {
            None => term,
            Some(mut accumulated) => {
                accumulated.union_assign(&term)?;
                accumulated
            }
        });
    }
    Ok(result.expect("factor_count >= 1 so at least one term was summed"))
}

/// An incrementally maintained natural join over a set of factor relations.
///
/// The join is kept as a signed relation. `apply_deltas` updates it in place via
/// [`delta_multi_join`] rather than recomputing from scratch, so a recurring rule
/// pays only for the change. The join and the deltas share the left-deep
/// index-order schema, so the maintained join stays equal to a full
/// recomputation. This is the incremental-view-maintenance primitive for hot or
/// recursive MM2 plans (DBSP/F-IVM style), kept internal and signed; the global
/// MM2 space stays set-valued.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaintainedJoin {
    factors: Box<[BindingRelation]>,
    join: BindingRelation,
}

impl MaintainedJoin {
    /// Builds the initial join over `factors`, folded left-deep in index order.
    /// Requires at least one factor.
    pub fn new(factors: Vec<BindingRelation>) -> Result<Self, BindingRelationError> {
        let join = natural_join_all(&factors)?.ok_or(BindingRelationError::InvalidVariableOrder)?;
        Ok(Self {
            factors: factors.into_boxed_slice(),
            join,
        })
    }

    /// The current maintained join (signed).
    pub fn join(&self) -> &BindingRelation {
        &self.join
    }

    /// The current factor relations.
    pub fn factors(&self) -> &[BindingRelation] {
        &self.factors
    }

    /// Applies one signed delta per factor and folds the join's delta in, leaving
    /// the maintained join equal to a recomputation over the updated factors.
    pub fn apply_deltas(&mut self, deltas: &[BindingRelation]) -> Result<(), BindingRelationError> {
        if deltas.len() != self.factors.len() {
            return Err(BindingRelationError::SchemaMismatch);
        }
        let news = self
            .factors
            .iter()
            .zip(deltas)
            .map(|(old, delta)| {
                let mut updated = old.clone();
                updated.union_assign(delta)?;
                Ok(updated)
            })
            .collect::<Result<Vec<_>, BindingRelationError>>()?;
        let join_delta = delta_multi_join(&self.factors, &news, deltas)?;
        self.join.union_assign(&join_delta)?;
        self.factors = news.into_boxed_slice();
        Ok(())
    }
}

/// Least fixpoint of a linear-recursive rule by semi-naive evaluation:
/// `Q = base union project_output( fixed_factors join relabel(Q, recursive_schema) )`.
///
/// `base` holds Q's non-recursive facts and its schema is `output`. Each round
/// plugs the newly derived facts into the body (relabeled to `recursive_schema`
/// so they join the fixed factors), joins, projects back to `output`, and keeps
/// only rows not seen before. It stops when a round derives nothing new.
/// Terminates because joins create no new terms, so the closure is finite.
///
/// `output` must equal `base`'s schema and must appear in the body;
/// `recursive_schema` must match Q's arity. Generalizes
/// [`semi_naive_transitive_closure`] from one fixed edge relation to any linear
/// body (a recursive relation joined once with fixed factors).
pub fn semi_naive_linear_fixpoint(
    base: &BindingRelation,
    fixed_factors: &[BindingRelation],
    recursive_schema: &[BindingVar],
    output: &[BindingVar],
) -> Result<BindingRelation, BindingRelationError> {
    if base.schema() != output || recursive_schema.len() != output.len() {
        return Err(BindingRelationError::SchemaMismatch);
    }
    let mut total = base.distinct();
    let mut delta = total.clone();
    while delta.positive_rows().next().is_some() {
        let mut body = delta.relabel(recursive_schema)?;
        for factor in fixed_factors {
            body = natural_join(&body, factor)?;
        }
        let derived = body.project(output)?;
        let fresh = derived.difference_presence(&total)?;
        if fresh.positive_rows().next().is_none() {
            break;
        }
        total.union_assign(&fresh)?;
        delta = fresh;
    }
    Ok(total)
}

/// Least fixpoint of a linear-recursive rule, like [`semi_naive_linear_fixpoint`]
/// but joining the body with the worst-case-optimal `generic_join` over all body
/// relations at once instead of a left-deep chain of binary `natural_join`s. For
/// a single fixed factor the two agree; for a multi-factor cyclic body (a
/// recursive multi-way join) this avoids the intermediate blow-up and meets the
/// AGM bound, the recursive-semi-naive-over-WCO combination (Worst-Case Optimal
/// GPU Datalog, arXiv 2604.20073). The variable order is the recursive schema
/// followed by the fixed factors' remaining variables.
pub fn semi_naive_linear_fixpoint_wco(
    base: &BindingRelation,
    fixed_factors: &[BindingRelation],
    recursive_schema: &[BindingVar],
    output: &[BindingVar],
) -> Result<BindingRelation, BindingRelationError> {
    if base.schema() != output || recursive_schema.len() != output.len() {
        return Err(BindingRelationError::SchemaMismatch);
    }
    let mut body_order: Vec<BindingVar> = recursive_schema.to_vec();
    for factor in fixed_factors {
        for &variable in factor.schema() {
            if !body_order.contains(&variable) {
                body_order.push(variable);
            }
        }
    }
    let mut total = base.distinct();
    let mut delta = total.clone();
    while delta.positive_rows().next().is_some() {
        let mut inputs = Vec::with_capacity(fixed_factors.len() + 1);
        inputs.push(delta.relabel(recursive_schema)?);
        inputs.extend(fixed_factors.iter().cloned());
        let derived = generic_join(&inputs, &body_order)?.project(output)?;
        let fresh = derived.difference_presence(&total)?;
        if fresh.positive_rows().next().is_none() {
            break;
        }
        total.union_assign(&fresh)?;
        delta = fresh;
    }
    Ok(total)
}

/// Diagnostics from semi-naive binary transitive closure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemiNaiveTransitiveClosure {
    /// Visible closure rows in `[source, target]` schema order.
    pub relation: BindingRelation,
    /// Non-empty frontier rounds evaluated.
    pub rounds: usize,
    /// Candidate extension rows examined before duplicate filtering.
    pub candidate_extensions: usize,
    /// New closure rows inserted after the initial edge set.
    pub derived_rows: usize,
}

/// Semi-naive transitive closure over a binary edge relation.
///
/// This is a deterministic reference/diagnostic kernel for recurring-rule
/// planning. It keeps signed relation storage internal, but treats visible
/// positive edge rows as set facts and only joins the current frontier against
/// the base edge relation on each round.
pub fn semi_naive_transitive_closure(
    edges: &BindingRelation,
    source: BindingVar,
    target: BindingVar,
) -> Result<SemiNaiveTransitiveClosure, BindingRelationError> {
    if source == target {
        return Err(BindingRelationError::InvalidVariableOrder);
    }
    let source_index = edges
        .schema_index(source)
        .ok_or(BindingRelationError::UnknownVariable { variable: source })?;
    let target_index = edges
        .schema_index(target)
        .ok_or(BindingRelationError::UnknownVariable { variable: target })?;

    let schema = vec![source, target];
    let mut outgoing = BTreeMap::<TermId, Vec<TermId>>::new();
    let seen_capacity = edges.len().saturating_mul(4).min(1 << 20);
    let mut seen = HashSet::<(TermId, TermId)>::with_capacity(seen_capacity);
    let mut frontier = Vec::<(TermId, TermId)>::with_capacity(edges.len());
    for row in edges.positive_rows() {
        let from = row[source_index];
        let to = row[target_index];
        outgoing.entry(from).or_default().push(to);
        if seen.insert((from, to)) {
            frontier.push((from, to));
        }
    }
    for targets in outgoing.values_mut() {
        targets.sort_unstable();
        targets.dedup();
    }

    let mut next = Vec::<(TermId, TermId)>::new();
    let mut rounds = 0;
    let mut candidate_extensions = 0usize;
    let mut derived_rows = 0usize;
    while !frontier.is_empty() {
        rounds = checked_add_cardinality(rounds, 1)?;
        next.clear();
        if next.capacity() < frontier.len() {
            next.reserve(frontier.len() - next.capacity());
        }

        for &(from, bridge) in frontier.iter() {
            let Some(targets) = outgoing.get(&bridge) else {
                continue;
            };
            for &to in targets {
                candidate_extensions = checked_add_cardinality(candidate_extensions, 1)?;
                if seen.insert((from, to)) {
                    next.push((from, to));
                }
            }
        }
        if next.is_empty() {
            break;
        }
        derived_rows = checked_add_cardinality(derived_rows, next.len())?;
        std::mem::swap(&mut frontier, &mut next);
    }

    let mut closure = BindingRelation::new(schema.clone());
    for (from, to) in seen {
        closure.insert_unit_if_absent([from, to])?;
    }

    Ok(SemiNaiveTransitiveClosure {
        relation: closure,
        rounds,
        candidate_extensions,
        derived_rows,
    })
}

/// A transitive closure maintained incrementally under edge inserts. It keeps,
/// per node, the set it reaches (`succs`) and the set that reaches it (`preds`),
/// so inserting an edge `(a,b)` adds exactly the pairs
/// `(ancestors(a) + a) x (descendants(b) + b)` the edge creates, in time
/// proportional to those new pairs rather than the whole closure (Italiano,
/// "Amortized efficiency of a path retrieval data structure", TCS 48, 1986). This
/// is the incremental-recursion case of differential/DBSP view maintenance
/// (Budiu, McSherry, Ryzhyk, Tannen, 2022): a recursive view maintained under
/// base inserts. Persisting one of these across exec steps makes a streaming
/// closure cost O(new pairs) per edge instead of O(closure) per recompute.
#[derive(Clone, Debug, Default)]
pub struct MaintainedTransitiveClosure {
    succs: BTreeMap<TermId, BTreeSet<TermId>>,
    preds: BTreeMap<TermId, BTreeSet<TermId>>,
    pairs: usize,
}

impl MaintainedTransitiveClosure {
    /// An empty closure.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `from` reaches `to`.
    pub fn reaches(&self, from: TermId, to: TermId) -> bool {
        self.succs.get(&from).is_some_and(|set| set.contains(&to))
    }

    /// Number of reachable ordered pairs.
    pub fn len(&self) -> usize {
        self.pairs
    }

    /// Whether no pairs are stored.
    pub fn is_empty(&self) -> bool {
        self.pairs == 0
    }

    /// Loads an already-closed relation directly, without re-deriving it.
    /// `source` and `target` index the from and to columns.
    pub fn load_closed(&mut self, closure: &BindingRelation, source: usize, target: usize) {
        for row in closure.positive_rows() {
            let (from, to) = (row[source], row[target]);
            if self.succs.entry(from).or_default().insert(to) {
                self.preds.entry(to).or_default().insert(from);
                self.pairs += 1;
            }
        }
    }

    /// Inserts edge `(from, to)` and adds the pairs it creates, returning their
    /// count. Assumes the structure is currently closed.
    pub fn insert_edge(&mut self, from: TermId, to: TermId) -> usize {
        let mut created = Vec::new();
        self.insert_edge_into(from, to, &mut created);
        created.len()
    }

    /// Inserts edge `(from, to)`, pushing each newly created pair onto `created`.
    /// Lets a streaming caller write exactly the delta. Assumes the structure is
    /// currently closed.
    pub fn insert_edge_into(
        &mut self,
        from: TermId,
        to: TermId,
        created: &mut Vec<(TermId, TermId)>,
    ) {
        if self.reaches(from, to) {
            return;
        }
        // Ancestors of `from` (including itself) now reach `to`'s descendants
        // (including itself). Snapshot both sides before mutating.
        let ancestors: Vec<TermId> = self
            .preds
            .get(&from)
            .into_iter()
            .flatten()
            .copied()
            .chain(std::iter::once(from))
            .collect();
        let descendants: Vec<TermId> = self
            .succs
            .get(&to)
            .into_iter()
            .flatten()
            .copied()
            .chain(std::iter::once(to))
            .collect();
        for &x in &ancestors {
            for &y in &descendants {
                if self.succs.entry(x).or_default().insert(y) {
                    self.preds.entry(y).or_default().insert(x);
                    self.pairs += 1;
                    created.push((x, y));
                }
            }
        }
    }

    /// Materializes the closure as a relation with schema `[source, target]`.
    pub fn to_relation(
        &self,
        source: BindingVar,
        target: BindingVar,
    ) -> Result<BindingRelation, BindingRelationError> {
        let mut relation = BindingRelation::new(vec![source, target]);
        for (&from, tos) in &self.succs {
            for &to in tos {
                relation.add(vec![from, to], 1)?;
            }
        }
        Ok(relation)
    }
}

/// Extends a closed relation with a batch of new edges, returning the new
/// closure, via [`MaintainedTransitiveClosure`]. Standalone this rebuilds the
/// reachability index, so it is O(|closure| + new pairs), not a win over a full
/// recompute. The O(new pairs) win comes from persisting the maintained closure
/// across calls (the streaming case), not from this one-shot wrapper.
pub fn semi_naive_incremental_transitive_closure(
    closure: &BindingRelation,
    new_edges: &BindingRelation,
    source: BindingVar,
    target: BindingVar,
) -> Result<BindingRelation, BindingRelationError> {
    if source == target {
        return Err(BindingRelationError::InvalidVariableOrder);
    }
    let s = closure
        .schema_index(source)
        .ok_or(BindingRelationError::UnknownVariable { variable: source })?;
    let t = closure
        .schema_index(target)
        .ok_or(BindingRelationError::UnknownVariable { variable: target })?;
    let ns = new_edges
        .schema_index(source)
        .ok_or(BindingRelationError::UnknownVariable { variable: source })?;
    let nt = new_edges
        .schema_index(target)
        .ok_or(BindingRelationError::UnknownVariable { variable: target })?;
    let mut maintained = MaintainedTransitiveClosure::new();
    maintained.load_closed(closure, s, t);
    for row in new_edges.positive_rows() {
        maintained.insert_edge(row[ns], row[nt]);
    }
    maintained.to_relation(source, target)
}

/// A linear-recursive view maintained incrementally under base inserts. The rule
/// is `Q = base union project_output( fixed_factors join relabel(Q, recursive
/// schema) )`, the shape `semi_naive_linear_fixpoint` evaluates. `insert` folds a
/// batch of new base facts and new fixed-factor facts into the view by semi-naive
/// delta propagation, so a streaming workload pays for the newly derived facts,
/// not a full recompute. This generalizes the maintained transitive closure past
/// transitive closure to any linear-recursive rule. It is F-IVM / DBSP
/// incremental recursion (Kara, Nikolic, Olteanu, Zhang, F-IVM, VLDB Journal 32,
/// 2023; Budiu, McSherry, Ryzhyk, Tannen, DBSP, 2022).
#[derive(Clone, Debug)]
pub struct MaintainedLinearFixpoint {
    view: BindingRelation,
    fixed_factors: Vec<BindingRelation>,
    recursive_schema: Vec<BindingVar>,
    output: Vec<BindingVar>,
}

impl MaintainedLinearFixpoint {
    /// Builds the initial least fixpoint of the rule.
    pub fn new(
        base: &BindingRelation,
        fixed_factors: &[BindingRelation],
        recursive_schema: &[BindingVar],
        output: &[BindingVar],
    ) -> Result<Self, BindingRelationError> {
        let view = semi_naive_linear_fixpoint(base, fixed_factors, recursive_schema, output)?;
        Ok(Self {
            view,
            fixed_factors: fixed_factors.to_vec(),
            recursive_schema: recursive_schema.to_vec(),
            output: output.to_vec(),
        })
    }

    /// The maintained view (the current least fixpoint).
    pub fn view(&self) -> &BindingRelation {
        &self.view
    }

    /// Folds new base facts and new fixed-factor facts (each `(factor index,
    /// delta)`) into the view, returning the number of new view rows. The result
    /// equals rebuilding the fixpoint over the updated base and factors. Deltas
    /// are applied one at a time, each fully propagated, so derivations that need
    /// more than one new fact are not missed.
    pub fn insert(
        &mut self,
        base_delta: &BindingRelation,
        fixed_deltas: &[(usize, BindingRelation)],
    ) -> Result<usize, BindingRelationError> {
        let mut added = self.propagate(base_delta.distinct())?;
        for (index, delta) in fixed_deltas {
            // Derivations that route through this new fixed fact: the current view
            // joined with the delta and the other factors at full.
            let mut body = self.view.relabel(&self.recursive_schema)?;
            for (factor_index, factor) in self.fixed_factors.iter().enumerate() {
                let operand = if factor_index == *index {
                    delta
                } else {
                    factor
                };
                body = natural_join(&body, operand)?;
            }
            let immediate = body.project(&self.output)?;
            if let Some(factor) = self.fixed_factors.get_mut(*index) {
                factor.union_assign(delta)?;
            }
            added += self.propagate(immediate)?;
        }
        Ok(added)
    }

    /// Semi-naive propagation: each fresh view row is a new recursive fact that
    /// derives more, until nothing new appears.
    fn propagate(&mut self, seed: BindingRelation) -> Result<usize, BindingRelationError> {
        let mut frontier = seed.difference_presence(&self.view)?;
        let mut added = 0;
        while frontier.positive_rows().next().is_some() {
            self.view.union_assign(&frontier)?;
            added += frontier.positive_rows().count();
            let mut body = frontier.relabel(&self.recursive_schema)?;
            for factor in &self.fixed_factors {
                body = natural_join(&body, factor)?;
            }
            frontier = body
                .project(&self.output)?
                .difference_presence(&self.view)?;
        }
        Ok(added)
    }
}

/// Reference variable-at-a-time Generic Join using leapfrog-style domain
/// intersection. This is a semantic oracle, not a production kernel.
pub fn generic_join(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
) -> Result<BindingRelation, BindingRelationError> {
    validate_variable_order(relations, variable_order)?;

    let mut out = BindingRelation::new(variable_order.to_vec());
    let mut binding = BindingAssignment::default();
    let mut scratch = generic_join_scratch(variable_order.len());
    generic_join_recurse(
        relations,
        variable_order,
        variable_order,
        &mut binding,
        &mut out,
        &mut scratch,
    )?;
    Ok(out)
}

/// Result of counting a reference Generic Join without materializing rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericJoinCount {
    /// Number of positive output rows represented by the join.
    pub rows: usize,
}

/// Result of checking whether a reference Generic Join has any positive row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericJoinExistence {
    /// Whether at least one positive output row exists.
    pub exists: bool,
}

/// Counts positive output rows from the reference Generic Join without
/// materializing the output relation.
pub fn generic_join_count(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
) -> Result<GenericJoinCount, BindingRelationError> {
    let rows = generic_join_aggregate(relations, variable_order, false)?;
    Ok(GenericJoinCount { rows })
}

/// Checks whether the reference Generic Join has any positive output row,
/// stopping after the first witness.
pub fn generic_join_exists(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
) -> Result<GenericJoinExistence, BindingRelationError> {
    let rows = generic_join_aggregate(relations, variable_order, true)?;
    Ok(GenericJoinExistence { exists: rows > 0 })
}

/// Result of executing a trie-backed variable-at-a-time join.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinResult {
    /// Flat relation in the requested variable order.
    pub relation: BindingRelation,
    /// Physical work counters for the trie-backed executor.
    pub stats: TrieJoinStats,
}

/// Result of counting a trie-backed variable-at-a-time join without
/// materializing the output relation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinCount {
    /// Number of positive output rows represented by the join.
    pub rows: usize,
    /// Physical work counters for the aggregate traversal.
    pub stats: TrieJoinStats,
}

/// Result of checking whether a trie-backed variable-at-a-time join has any
/// positive output row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinExistence {
    /// Whether at least one positive output row exists.
    pub exists: bool,
    /// Physical work counters for the early-stop traversal.
    pub stats: TrieJoinStats,
}

/// Reusable trie indexes for one variable-at-a-time join order.
///
/// Build this once when several reports are needed for the same immutable
/// relation snapshot and variable order, then call [`execute`](Self::execute),
/// [`count`](Self::count), [`exists`](Self::exists), or [`trace`](Self::trace)
/// without rebuilding relation trie indexes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedTrieJoin {
    variable_order: Box<[BindingVar]>,
    indexes: Box<[RelationTrieIndex]>,
    indexed_rows: usize,
    trie_nodes: usize,
}

/// Result of intersecting caller-provided TermId domain cursors.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BindingDomainIntersection {
    /// Ordered TermIds that survived every input domain.
    pub values: Vec<TermId>,
    /// Number of input domains.
    pub domain_sources: usize,
    /// Domain cursors opened.
    pub cursor_opens: usize,
    /// Monotone seek calls issued.
    pub cursor_seeks: usize,
    /// Domain values skipped by seeks.
    pub cursor_skips: usize,
    /// Next calls issued after aligned values.
    pub cursor_nexts: usize,
}

/// Counters from the trie-backed variable-at-a-time join.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TrieJoinStats {
    /// Input relation indexes constructed for the join.
    pub relation_indexes: usize,
    /// Positive rows retained across all relation indexes.
    pub indexed_rows: usize,
    /// Distinct trie prefixes with outgoing children across all indexes.
    pub trie_nodes: usize,
    /// Variable-domain intersections performed during traversal.
    pub domain_intersections: usize,
    /// Relation domains participating in those intersections.
    pub domain_sources: usize,
    /// Domain values presented to leapfrog intersection.
    pub domain_values: usize,
    /// LFTJ-style domain cursors opened for variable intersections.
    pub domain_cursor_opens: usize,
    /// Monotone seek calls issued against domain cursors.
    pub domain_cursor_seeks: usize,
    /// Domain values skipped by monotone seek calls.
    pub domain_cursor_skips: usize,
    /// Next calls issued after a cursor-aligned result is emitted.
    pub domain_cursor_nexts: usize,
    /// Final relation row-weight probes.
    pub weight_lookups: usize,
    /// Positive rows emitted by the join.
    pub output_rows: usize,
}

/// Non-materializing trace of a trie-backed variable-at-a-time join.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinTrace {
    /// Variable order used by the traced traversal.
    pub variable_order: Box<[BindingVar]>,
    /// Input relation indexes constructed for the trace.
    pub relation_indexes: usize,
    /// Positive rows retained across all relation indexes.
    pub indexed_rows: usize,
    /// Distinct trie prefixes with outgoing children across all indexes.
    pub trie_nodes: usize,
    /// Domain-intersection contexts visited in traversal order.
    pub steps: Box<[TrieJoinTraceStep]>,
    /// Complete satisfying bindings reached by domain traversal before relation
    /// materialization or row-weight probing.
    pub candidate_bindings: usize,
}

/// Aggregate counters and shape facts derived from a trie-join trace.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TrieJoinTraceSummary {
    /// Input relation indexes constructed for the trace.
    pub relation_indexes: usize,
    /// Positive rows retained across all relation indexes.
    pub indexed_rows: usize,
    /// Distinct trie prefixes with outgoing children across all indexes.
    pub trie_nodes: usize,
    /// Domain-intersection contexts visited in traversal order.
    pub steps: usize,
    /// Complete satisfying bindings reached by the traced traversal.
    pub candidate_bindings: usize,
    /// Variable-domain intersections represented by the trace.
    pub domain_intersections: usize,
    /// Relation domains participating in those intersections.
    pub domain_sources: usize,
    /// Domain values presented to leapfrog intersection.
    pub domain_values: usize,
    /// Values that survived all recorded intersections.
    pub intersection_values: usize,
    /// Intersections whose survivor set was empty.
    pub empty_intersections: usize,
    /// Largest bound-prefix depth represented by one step.
    pub max_bound_prefix_len: usize,
    /// Largest relation-domain fan-in represented by one step.
    pub max_participating_relations: usize,
    /// Largest survivor set represented by one step.
    pub max_intersection_len: usize,
    /// Cursor opens recorded across all trace steps.
    pub cursor_opens: usize,
    /// Cursor seeks recorded across all trace steps.
    pub cursor_seeks: usize,
    /// Cursor-skipped values recorded across all trace steps.
    pub cursor_skips: usize,
    /// Cursor next calls recorded across all trace steps.
    pub cursor_nexts: usize,
}

/// Cursor contract extracted from a validated trie-join trace.
///
/// This is the checklist a future PathMap/ReadZipper-backed relation factor
/// must satisfy for the same variable order: for each relation index, open the
/// listed bound-prefix contexts and expose ordered domains with the recorded
/// values and cardinalities. The contract is diagnostic only; it does not
/// execute or maintain join answers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinCursorContract {
    /// Aggregate trace summary validated before the contract was built.
    pub summary: TrieJoinTraceSummary,
    /// Number of relation factors represented by the trace.
    pub relation_indexes: usize,
    /// Per-relation cursor requirements.
    pub factor_requirements: Box<[TrieJoinFactorCursorRequirement]>,
}

/// Cursor contexts required from one relation factor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinFactorCursorRequirement {
    /// Relation/factor index in the opened sidecar plan.
    pub relation_index: usize,
    /// Domain contexts this factor must expose during replay.
    pub contexts: Box<[TrieJoinFactorCursorContext]>,
    /// Number of domain contexts for this factor.
    pub domain_contexts: usize,
    /// Total domain values observed across those contexts.
    pub domain_values: usize,
    /// Largest domain required from this factor.
    pub max_domain_len: usize,
}

/// One variable-depth domain opening required from a relation factor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinFactorCursorContext {
    /// Original trace step index.
    pub step_index: usize,
    /// Depth in the global variable order.
    pub level: usize,
    /// Variable whose ordered domain is opened at this context.
    pub variable: BindingVar,
    /// Values already bound before opening this domain.
    pub bound_prefix: BindingRow,
    /// Ordered values this factor exposed before intersection.
    pub domain: BindingRow,
    /// Number of values the factor exposed before intersection.
    pub domain_len: usize,
    /// Number of values that survived the multi-factor intersection.
    pub intersection_len: usize,
}

/// Replayable prefix tree reconstructed from a trie-join trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinTraceReplayShape {
    /// Locally validated aggregate trace summary.
    pub summary: TrieJoinTraceSummary,
    /// Root variable-depth context, absent only for zero-variable traces.
    pub root: Option<TrieJoinTraceReplayNode>,
}

/// Trace-local work affected by replaying from one bound-prefix context.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TrieJoinTraceImpact {
    /// Bound-prefix context whose replay subtree was found.
    pub bound_prefix: BindingRow,
    /// Original trace step indexes covered by this replay subtree.
    pub step_indexes: Box<[usize]>,
    /// Domain-intersection steps covered by this replay subtree.
    pub steps: usize,
    /// Complete satisfying bindings below this replay subtree.
    pub candidate_bindings: usize,
    /// Relation domains participating below this replay subtree.
    pub domain_sources: usize,
    /// Domain values exposed before leapfrog intersection below this subtree.
    pub domain_values: usize,
    /// Values that survived intersection below this replay subtree.
    pub intersection_values: usize,
    /// Deepest variable-order level reached by this replay subtree.
    pub max_level: usize,
}

/// One bound-prefix context changed between two replay shapes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinTraceReplayDiffEntry {
    /// Bound-prefix context compared between replay shapes.
    pub bound_prefix: BindingRow,
    /// Kind of replay-shape change at this exact context.
    pub kind: TrieJoinTraceReplayDiffKind,
    /// Previous trace-local impact, absent for newly added contexts.
    pub old_impact: Option<TrieJoinTraceImpact>,
    /// New trace-local impact, absent for removed contexts.
    pub new_impact: Option<TrieJoinTraceImpact>,
}

/// Replay-shape change kind for one bound-prefix context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrieJoinTraceReplayDiffKind {
    /// Context exists only in the new replay shape.
    Added,
    /// Context exists only in the old replay shape.
    Removed,
    /// Context exists in both shapes but its local intersection metadata changed.
    Changed,
}

/// Trace-local diff between two replay shapes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TrieJoinTraceReplayDiff {
    /// Bound-prefix contexts present in both shapes with identical local metadata.
    pub unchanged_contexts: usize,
    /// Bound-prefix contexts present in both shapes with changed local metadata.
    pub changed_contexts: usize,
    /// Bound-prefix contexts present only in the new shape.
    pub added_contexts: usize,
    /// Bound-prefix contexts present only in the old shape.
    pub removed_contexts: usize,
    /// Non-overlapping changed-context roots used for touched-work estimates.
    pub frontier_contexts: usize,
    /// Old replay steps under the non-overlapping change frontier.
    pub old_replay_steps_touched: usize,
    /// New replay steps under the non-overlapping change frontier.
    pub new_replay_steps_touched: usize,
    /// Old candidate bindings under the non-overlapping change frontier.
    pub old_candidate_bindings_touched: usize,
    /// New candidate bindings under the non-overlapping change frontier.
    pub new_candidate_bindings_touched: usize,
    /// Exact changed, added, and removed contexts.
    pub entries: Box<[TrieJoinTraceReplayDiffEntry]>,
}

impl TrieJoinTraceReplayDiff {
    /// Returns true when the two replay shapes expose the same local contexts.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One replay node for a variable-depth domain intersection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinTraceReplayNode {
    /// Original trace step index.
    pub step_index: usize,
    /// Depth in the global variable order.
    pub level: usize,
    /// Variable intersected at this depth.
    pub variable: BindingVar,
    /// Values already bound before this depth.
    pub bound_prefix: BindingRow,
    /// Relation index numbers whose trie domains constrained this variable.
    pub participating_relations: Box<[usize]>,
    /// Domain length contributed by each participating relation.
    pub relation_domain_lens: Box<[usize]>,
    /// Exact relation domains contributed at this context.
    pub relation_domains: Box<[TrieJoinTraceFactorDomain]>,
    /// Values that survived the intersection at this context.
    pub intersection: Box<[TermId]>,
    /// Per-survivor continuation in traversal order.
    pub branches: Box<[TrieJoinTraceReplayBranch]>,
}

/// One survivor from a replay node's domain intersection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinTraceReplayBranch {
    /// Bound value for the node variable.
    pub value: TermId,
    /// Child context after binding `value`, or `None` at leaf depth.
    pub child: Option<Box<TrieJoinTraceReplayNode>>,
}

impl TrieJoinTraceReplayShape {
    /// Returns the replay subtree affected at `bound_prefix`, if this trace
    /// visited that exact variable-prefix context.
    ///
    /// This is a trace-local sensitivity summary. It identifies the existing
    /// replay work under a bound prefix, but it does not maintain or recompute
    /// join answers for a changed relation.
    pub fn impact_for_bound_prefix(&self, bound_prefix: &[TermId]) -> Option<TrieJoinTraceImpact> {
        if bound_prefix.is_empty() && self.root.is_none() {
            return Some(TrieJoinTraceImpact {
                bound_prefix: Vec::new().into_boxed_slice(),
                candidate_bindings: self.summary.candidate_bindings,
                ..TrieJoinTraceImpact::default()
            });
        }

        self.root
            .as_ref()?
            .find_bound_prefix(bound_prefix)
            .map(TrieJoinTraceReplayNode::impact)
    }

    /// Compares two replay shapes at exact bound-prefix contexts.
    ///
    /// This is an edit-distance-style diagnostic for traces. It does not apply
    /// deltas or decide semantic equivalence of the underlying relations.
    pub fn diff(&self, newer: &Self) -> TrieJoinTraceReplayDiff {
        let mut old_nodes = BTreeMap::new();
        let mut new_nodes = BTreeMap::new();
        self.collect_nodes(&mut old_nodes);
        newer.collect_nodes(&mut new_nodes);

        let prefixes = old_nodes
            .keys()
            .chain(new_nodes.keys())
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut entries = Vec::new();
        let mut diff = TrieJoinTraceReplayDiff::default();

        for prefix in prefixes {
            match (old_nodes.get(&prefix), new_nodes.get(&prefix)) {
                (Some(old), Some(new)) if old.local_context_eq(new) => {
                    diff.unchanged_contexts += 1;
                }
                (Some(old), Some(new)) => {
                    diff.changed_contexts += 1;
                    entries.push(TrieJoinTraceReplayDiffEntry {
                        bound_prefix: prefix,
                        kind: TrieJoinTraceReplayDiffKind::Changed,
                        old_impact: Some(old.impact()),
                        new_impact: Some(new.impact()),
                    });
                }
                (Some(old), None) => {
                    diff.removed_contexts += 1;
                    entries.push(TrieJoinTraceReplayDiffEntry {
                        bound_prefix: prefix,
                        kind: TrieJoinTraceReplayDiffKind::Removed,
                        old_impact: Some(old.impact()),
                        new_impact: None,
                    });
                }
                (None, Some(new)) => {
                    diff.added_contexts += 1;
                    entries.push(TrieJoinTraceReplayDiffEntry {
                        bound_prefix: prefix,
                        kind: TrieJoinTraceReplayDiffKind::Added,
                        old_impact: None,
                        new_impact: Some(new.impact()),
                    });
                }
                (None, None) => {}
            }
        }

        entries.sort_by(|left, right| {
            left.bound_prefix
                .len()
                .cmp(&right.bound_prefix.len())
                .then_with(|| left.bound_prefix.as_ref().cmp(right.bound_prefix.as_ref()))
        });
        diff.add_frontier_estimate(&entries);
        diff.entries = entries.into_boxed_slice();
        diff
    }

    fn collect_nodes<'a>(&'a self, nodes: &mut BTreeMap<BindingRow, &'a TrieJoinTraceReplayNode>) {
        if let Some(root) = &self.root {
            root.collect_nodes(nodes);
        }
    }
}

impl TrieJoinTraceReplayNode {
    fn find_bound_prefix(&self, bound_prefix: &[TermId]) -> Option<&TrieJoinTraceReplayNode> {
        if self.bound_prefix.as_ref() == bound_prefix {
            return Some(self);
        }
        if bound_prefix.len() <= self.bound_prefix.len()
            || !bound_prefix.starts_with(self.bound_prefix.as_ref())
        {
            return None;
        }

        let next_value = bound_prefix[self.bound_prefix.len()];
        self.branches
            .iter()
            .find(|branch| branch.value == next_value)?
            .child
            .as_deref()?
            .find_bound_prefix(bound_prefix)
    }

    fn impact(&self) -> TrieJoinTraceImpact {
        let mut impact = TrieJoinTraceImpact {
            bound_prefix: self.bound_prefix.clone(),
            max_level: self.level,
            ..TrieJoinTraceImpact::default()
        };
        let mut step_indexes = Vec::new();
        self.accumulate_impact(&mut impact, &mut step_indexes);
        impact.steps = step_indexes.len();
        impact.step_indexes = step_indexes.into_boxed_slice();
        impact
    }

    fn accumulate_impact(&self, impact: &mut TrieJoinTraceImpact, step_indexes: &mut Vec<usize>) {
        step_indexes.push(self.step_index);
        impact.domain_sources += self.participating_relations.len();
        impact.domain_values += self.relation_domain_lens.iter().sum::<usize>();
        impact.intersection_values += self.intersection.len();
        impact.max_level = impact.max_level.max(self.level);

        for branch in self.branches.iter() {
            if let Some(child) = branch.child.as_deref() {
                child.accumulate_impact(impact, step_indexes);
            } else {
                impact.candidate_bindings += 1;
            }
        }
    }

    fn collect_nodes<'a>(&'a self, nodes: &mut BTreeMap<BindingRow, &'a TrieJoinTraceReplayNode>) {
        nodes.insert(self.bound_prefix.clone(), self);
        for branch in self.branches.iter() {
            if let Some(child) = branch.child.as_deref() {
                child.collect_nodes(nodes);
            }
        }
    }

    fn local_context_eq(&self, other: &Self) -> bool {
        self.level == other.level
            && self.variable == other.variable
            && self.participating_relations == other.participating_relations
            && self.relation_domain_lens == other.relation_domain_lens
            && self.relation_domains == other.relation_domains
            && self.intersection == other.intersection
    }
}

impl TrieJoinTraceReplayDiff {
    fn add_frontier_estimate(&mut self, entries: &[TrieJoinTraceReplayDiffEntry]) {
        let mut frontier_prefixes = Vec::<BindingRow>::new();
        for entry in entries {
            if frontier_prefixes
                .iter()
                .any(|prefix| entry.bound_prefix.starts_with(prefix.as_ref()))
            {
                continue;
            }

            frontier_prefixes.push(entry.bound_prefix.clone());
            self.frontier_contexts += 1;
            if let Some(impact) = &entry.old_impact {
                self.old_replay_steps_touched += impact.steps;
                self.old_candidate_bindings_touched += impact.candidate_bindings;
            }
            if let Some(impact) = &entry.new_impact {
                self.new_replay_steps_touched += impact.steps;
                self.new_candidate_bindings_touched += impact.candidate_bindings;
            }
        }
    }
}

/// Shape error found while summarizing a trie-join trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrieJoinTraceShapeError {
    /// A step refers past the configured variable order.
    InvalidStepLevel {
        step: usize,
        level: usize,
        variable_count: usize,
    },
    /// A step variable does not match the trace's variable order at its level.
    VariableMismatch {
        step: usize,
        expected: BindingVar,
        actual: BindingVar,
    },
    /// Bound-prefix length must equal the variable depth.
    BoundPrefixLengthMismatch {
        step: usize,
        expected: usize,
        actual: usize,
    },
    /// A participating relation index is outside the trace relation set.
    UnknownParticipatingRelation {
        step: usize,
        relation_index: usize,
        relation_indexes: usize,
    },
    /// Participating relation IDs and domain lengths must align one-for-one.
    ParticipationLengthMismatch {
        step: usize,
        participating_relations: usize,
        relation_domain_lens: usize,
    },
    /// Participating relation IDs and exact relation domains must align one-for-one.
    RelationDomainCountMismatch {
        step: usize,
        participating_relations: usize,
        relation_domains: usize,
    },
    /// Exact relation-domain metadata must preserve participating-relation order.
    RelationDomainIndexMismatch {
        step: usize,
        position: usize,
        expected: usize,
        actual: usize,
    },
    /// Exact relation-domain values must agree with the recorded domain length.
    RelationDomainLengthMismatch {
        step: usize,
        relation_index: usize,
        expected: usize,
        actual: usize,
    },
    /// Exact relation-domain values must be unique and strictly ordered.
    RelationDomainOrderMismatch {
        step: usize,
        relation_index: usize,
        previous: TermId,
        actual: TermId,
    },
    /// Aggregate domain-source count must match participating relation IDs.
    DomainSourceMismatch {
        step: usize,
        expected: usize,
        actual: usize,
    },
    /// Aggregate domain-value count must match summed relation-domain lengths.
    DomainValueMismatch {
        step: usize,
        expected: usize,
        actual: usize,
    },
    /// Non-empty domains open one cursor per participating relation.
    CursorOpenMismatch {
        step: usize,
        expected: usize,
        actual: usize,
    },
    /// A replay step appears at a different variable depth than traversal order expects.
    ReplayLevelMismatch {
        step: usize,
        expected: usize,
        actual: usize,
    },
    /// A replay step is missing for a reachable bound prefix.
    MissingReplayStep {
        level: usize,
        bound_prefix_len: usize,
    },
    /// A locally valid step belongs to a different bound-prefix branch.
    BoundPrefixValueMismatch {
        step: usize,
        position: usize,
        expected: TermId,
        actual: TermId,
    },
    /// Steps remain after replaying every reachable branch.
    TrailingReplayStep { step: usize },
    /// Leaf replay count disagrees with the trace candidate total.
    CandidateBindingMismatch { expected: usize, actual: usize },
}

/// One variable-depth domain intersection observed while tracing LFTJ traversal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinTraceStep {
    /// Depth in the global variable order.
    pub level: usize,
    /// Variable intersected at this depth.
    pub variable: BindingVar,
    /// Values already bound for variables before `level`, in variable-order
    /// prefix order.
    pub bound_prefix: BindingRow,
    /// Relation index numbers whose trie domains constrained this variable at
    /// the current bound prefix.
    pub participating_relations: Box<[usize]>,
    /// Domain length contributed by each participating relation, in
    /// `participating_relations` order.
    pub relation_domain_lens: Box<[usize]>,
    /// Exact domains contributed by each participating relation, in
    /// `participating_relations` order.
    pub relation_domains: Box<[TrieJoinTraceFactorDomain]>,
    /// Relation domains that constrained this variable in the current context.
    pub domain_sources: usize,
    /// Total values exposed by those relation domains before intersection.
    pub domain_values: usize,
    /// Values that survived domain synchronization at this context.
    pub intersection: Box<[TermId]>,
    /// Cursor opens issued for this intersection.
    pub cursor_opens: usize,
    /// Cursor seeks issued for this intersection.
    pub cursor_seeks: usize,
    /// Domain values skipped by cursor seeks for this intersection.
    pub cursor_skips: usize,
    /// Cursor next calls issued after aligned values.
    pub cursor_nexts: usize,
}

/// Exact ordered domain that one relation factor exposed at a trace step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrieJoinTraceFactorDomain {
    /// Relation/factor index in the opened sidecar plan.
    pub relation_index: usize,
    /// Ordered unique domain values exposed before intersection.
    pub values: BindingRow,
}

impl TrieJoinTrace {
    /// Validates step-local trace shape and returns aggregate replay counters.
    pub fn summarize(&self) -> Result<TrieJoinTraceSummary, TrieJoinTraceShapeError> {
        let mut summary = TrieJoinTraceSummary {
            relation_indexes: self.relation_indexes,
            indexed_rows: self.indexed_rows,
            trie_nodes: self.trie_nodes,
            steps: self.steps.len(),
            candidate_bindings: self.candidate_bindings,
            ..TrieJoinTraceSummary::default()
        };

        for (step_index, step) in self.steps.iter().enumerate() {
            let Some(&expected_variable) = self.variable_order.get(step.level) else {
                return Err(TrieJoinTraceShapeError::InvalidStepLevel {
                    step: step_index,
                    level: step.level,
                    variable_count: self.variable_order.len(),
                });
            };

            if step.variable != expected_variable {
                return Err(TrieJoinTraceShapeError::VariableMismatch {
                    step: step_index,
                    expected: expected_variable,
                    actual: step.variable,
                });
            }

            if step.bound_prefix.len() != step.level {
                return Err(TrieJoinTraceShapeError::BoundPrefixLengthMismatch {
                    step: step_index,
                    expected: step.level,
                    actual: step.bound_prefix.len(),
                });
            }

            for &relation_index in step.participating_relations.iter() {
                if relation_index >= self.relation_indexes {
                    return Err(TrieJoinTraceShapeError::UnknownParticipatingRelation {
                        step: step_index,
                        relation_index,
                        relation_indexes: self.relation_indexes,
                    });
                }
            }

            if step.participating_relations.len() != step.relation_domain_lens.len() {
                return Err(TrieJoinTraceShapeError::ParticipationLengthMismatch {
                    step: step_index,
                    participating_relations: step.participating_relations.len(),
                    relation_domain_lens: step.relation_domain_lens.len(),
                });
            }

            if step.participating_relations.len() != step.relation_domains.len() {
                return Err(TrieJoinTraceShapeError::RelationDomainCountMismatch {
                    step: step_index,
                    participating_relations: step.participating_relations.len(),
                    relation_domains: step.relation_domains.len(),
                });
            }

            for (position, ((&relation_index, &domain_len), relation_domain)) in step
                .participating_relations
                .iter()
                .zip(step.relation_domain_lens.iter())
                .zip(step.relation_domains.iter())
                .enumerate()
            {
                if relation_domain.relation_index != relation_index {
                    return Err(TrieJoinTraceShapeError::RelationDomainIndexMismatch {
                        step: step_index,
                        position,
                        expected: relation_index,
                        actual: relation_domain.relation_index,
                    });
                }
                if relation_domain.values.len() != domain_len {
                    return Err(TrieJoinTraceShapeError::RelationDomainLengthMismatch {
                        step: step_index,
                        relation_index,
                        expected: domain_len,
                        actual: relation_domain.values.len(),
                    });
                }
                if let Some(window) = relation_domain
                    .values
                    .windows(2)
                    .find(|window| window[0] >= window[1])
                {
                    return Err(TrieJoinTraceShapeError::RelationDomainOrderMismatch {
                        step: step_index,
                        relation_index,
                        previous: window[0],
                        actual: window[1],
                    });
                }
            }

            if step.domain_sources != step.participating_relations.len() {
                return Err(TrieJoinTraceShapeError::DomainSourceMismatch {
                    step: step_index,
                    expected: step.participating_relations.len(),
                    actual: step.domain_sources,
                });
            }

            let domain_values = step.relation_domain_lens.iter().sum::<usize>();
            if step.domain_values != domain_values {
                return Err(TrieJoinTraceShapeError::DomainValueMismatch {
                    step: step_index,
                    expected: domain_values,
                    actual: step.domain_values,
                });
            }

            let expected_cursor_opens = if step.relation_domain_lens.contains(&0)
                || step.participating_relations.len() <= 1
            {
                0
            } else {
                step.participating_relations.len()
            };
            if step.cursor_opens != expected_cursor_opens {
                return Err(TrieJoinTraceShapeError::CursorOpenMismatch {
                    step: step_index,
                    expected: expected_cursor_opens,
                    actual: step.cursor_opens,
                });
            }

            summary.domain_intersections += 1;
            summary.domain_sources += step.domain_sources;
            summary.domain_values += step.domain_values;
            summary.intersection_values += step.intersection.len();
            summary.empty_intersections += usize::from(step.intersection.is_empty());
            summary.max_bound_prefix_len =
                summary.max_bound_prefix_len.max(step.bound_prefix.len());
            summary.max_participating_relations = summary
                .max_participating_relations
                .max(step.participating_relations.len());
            summary.max_intersection_len =
                summary.max_intersection_len.max(step.intersection.len());
            summary.cursor_opens += step.cursor_opens;
            summary.cursor_seeks += step.cursor_seeks;
            summary.cursor_skips += step.cursor_skips;
            summary.cursor_nexts += step.cursor_nexts;
        }

        Ok(summary)
    }

    /// Reconstructs the variable-prefix replay tree represented by this trace.
    ///
    /// `summarize` checks each step independently. This method additionally
    /// checks that the steps form the preorder traversal that an LFTJ-style
    /// cursor would replay: each survivor at a non-leaf depth must be followed
    /// by exactly one child context whose bound prefix extends the parent.
    pub fn replay_shape(&self) -> Result<TrieJoinTraceReplayShape, TrieJoinTraceShapeError> {
        let summary = self.summarize()?;

        if self.variable_order.is_empty() {
            if !self.steps.is_empty() {
                return Err(TrieJoinTraceShapeError::TrailingReplayStep { step: 0 });
            }
            if self.candidate_bindings != 1 {
                return Err(TrieJoinTraceShapeError::CandidateBindingMismatch {
                    expected: self.candidate_bindings,
                    actual: 1,
                });
            }
            return Ok(TrieJoinTraceReplayShape {
                summary,
                root: None,
            });
        }

        let mut step_index = 0;
        let mut candidate_bindings = 0;
        let root = self.replay_node(0, &[], &mut step_index, &mut candidate_bindings)?;

        if step_index != self.steps.len() {
            return Err(TrieJoinTraceShapeError::TrailingReplayStep { step: step_index });
        }
        if candidate_bindings != self.candidate_bindings {
            return Err(TrieJoinTraceShapeError::CandidateBindingMismatch {
                expected: self.candidate_bindings,
                actual: candidate_bindings,
            });
        }

        Ok(TrieJoinTraceReplayShape {
            summary,
            root: Some(root),
        })
    }

    /// Builds the per-factor cursor contract implied by this trace.
    ///
    /// This first reconstructs the replay shape, so callers get the stronger
    /// preorder and candidate-count validation rather than only step-local
    /// metadata checks.
    pub fn cursor_contract(&self) -> Result<TrieJoinCursorContract, TrieJoinTraceShapeError> {
        let replay = self.replay_shape()?;
        let mut contexts_by_relation = BTreeMap::<usize, Vec<TrieJoinFactorCursorContext>>::new();

        for (step_index, step) in self.steps.iter().enumerate() {
            for relation_domain in step.relation_domains.iter() {
                contexts_by_relation
                    .entry(relation_domain.relation_index)
                    .or_default()
                    .push(TrieJoinFactorCursorContext {
                        step_index,
                        level: step.level,
                        variable: step.variable,
                        bound_prefix: step.bound_prefix.clone(),
                        domain: relation_domain.values.clone(),
                        domain_len: relation_domain.values.len(),
                        intersection_len: step.intersection.len(),
                    });
            }
        }

        let factor_requirements = contexts_by_relation
            .into_iter()
            .map(|(relation_index, contexts)| {
                let domain_values = contexts.iter().map(|context| context.domain_len).sum();
                let max_domain_len = contexts
                    .iter()
                    .map(|context| context.domain_len)
                    .max()
                    .unwrap_or(0);
                TrieJoinFactorCursorRequirement {
                    relation_index,
                    domain_contexts: contexts.len(),
                    contexts: contexts.into_boxed_slice(),
                    domain_values,
                    max_domain_len,
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(TrieJoinCursorContract {
            summary: replay.summary,
            relation_indexes: self.relation_indexes,
            factor_requirements,
        })
    }

    fn replay_node(
        &self,
        level: usize,
        expected_prefix: &[TermId],
        step_index: &mut usize,
        candidate_bindings: &mut usize,
    ) -> Result<TrieJoinTraceReplayNode, TrieJoinTraceShapeError> {
        let current_step = *step_index;
        let Some(step) = self.steps.get(current_step) else {
            return Err(TrieJoinTraceShapeError::MissingReplayStep {
                level,
                bound_prefix_len: expected_prefix.len(),
            });
        };

        if step.level != level {
            return Err(TrieJoinTraceShapeError::ReplayLevelMismatch {
                step: current_step,
                expected: level,
                actual: step.level,
            });
        }

        for (position, (&expected, &actual)) in expected_prefix
            .iter()
            .zip(step.bound_prefix.iter())
            .enumerate()
        {
            if expected != actual {
                return Err(TrieJoinTraceShapeError::BoundPrefixValueMismatch {
                    step: current_step,
                    position,
                    expected,
                    actual,
                });
            }
        }

        *step_index += 1;
        let leaf = level + 1 == self.variable_order.len();
        let mut branches = Vec::with_capacity(step.intersection.len());
        for &value in step.intersection.iter() {
            let child = if leaf {
                *candidate_bindings += 1;
                None
            } else {
                let mut child_prefix = Vec::with_capacity(expected_prefix.len() + 1);
                child_prefix.extend_from_slice(expected_prefix);
                child_prefix.push(value);
                Some(Box::new(self.replay_node(
                    level + 1,
                    &child_prefix,
                    step_index,
                    candidate_bindings,
                )?))
            };
            branches.push(TrieJoinTraceReplayBranch { value, child });
        }

        Ok(TrieJoinTraceReplayNode {
            step_index: current_step,
            level: step.level,
            variable: step.variable,
            bound_prefix: step.bound_prefix.clone(),
            participating_relations: step.participating_relations.clone(),
            relation_domain_lens: step.relation_domain_lens.clone(),
            relation_domains: step.relation_domains.clone(),
            intersection: step.intersection.clone(),
            branches: branches.into_boxed_slice(),
        })
    }
}

/// Minimal LFTJ-style cursor over one ordered variable domain.
///
/// This is the physical contract a future PathMap/ReadZipper-backed factor
/// must expose: ordered current key, monotone `seek`, linear `next`, and an
/// end marker. The current implementation below adapts in-memory relation
/// trie domains to the same contract.
pub trait BindingDomainCursor {
    /// Current key at this trie depth, or `None` at end.
    fn key(&self) -> Option<TermId>;
    /// Whether the cursor has exhausted its current domain.
    fn at_end(&self) -> bool;
    /// Advance to the next key in the current domain.
    fn next(&mut self);
    /// Advance to the least key greater than or equal to `target`.
    ///
    /// Returns the number of domain values skipped while advancing.
    fn seek(&mut self, target: TermId) -> usize;
}

/// Intersects already-opened TermId domain cursors with LFTJ-style seek/next.
///
/// This is the public cursor-level bridge used by external relation factors:
/// each cursor supplies one ordered domain for the current variable under a
/// bound prefix, and the helper returns the ordered intersection without
/// materializing pairwise intermediate sets.
pub fn intersect_binding_domain_cursors<C>(cursors: &mut [C]) -> BindingDomainIntersection
where
    C: BindingDomainCursor,
{
    let mut stats = TrieJoinStats::default();
    let values = leapfrog_intersection_cursors(cursors, &mut stats);
    BindingDomainIntersection {
        values,
        domain_sources: cursors.len(),
        cursor_opens: stats.domain_cursor_opens,
        cursor_seeks: stats.domain_cursor_seeks,
        cursor_skips: stats.domain_cursor_skips,
        cursor_nexts: stats.domain_cursor_nexts,
    }
}

impl PreparedTrieJoin {
    /// Builds reusable relation trie indexes for `relations` under
    /// `variable_order`.
    pub fn prepare(
        relations: &[BindingRelation],
        variable_order: &[BindingVar],
    ) -> Result<Self, BindingRelationError> {
        validate_variable_order(relations, variable_order)?;

        let indexes = relations
            .iter()
            .map(|relation| RelationTrieIndex::build(relation, variable_order))
            .collect::<Result<Vec<_>, _>>()?;
        let indexed_rows = indexes.iter().map(|index| index.weight_values.len()).sum();
        let trie_nodes = indexes
            .iter()
            .map(|index| index.child_range_starts.len())
            .sum();

        Ok(Self {
            variable_order: variable_order.to_vec().into_boxed_slice(),
            indexes: indexes.into_boxed_slice(),
            indexed_rows,
            trie_nodes,
        })
    }

    /// Variable order used by this prepared trie join.
    pub fn variable_order(&self) -> &[BindingVar] {
        &self.variable_order
    }

    fn base_stats(&self) -> TrieJoinStats {
        TrieJoinStats {
            relation_indexes: self.indexes.len(),
            indexed_rows: self.indexed_rows,
            trie_nodes: self.trie_nodes,
            ..TrieJoinStats::default()
        }
    }

    /// Executes the prepared trie-backed join and materializes output rows.
    pub fn execute(&self) -> Result<TrieJoinResult, BindingRelationError> {
        let mut stats = self.base_stats();
        let mut relation = BindingRelation::new(self.variable_order.to_vec());
        let mut binding = BindingAssignment::default();
        let mut scratch = trie_join_scratch(self.variable_order.len());
        let mut weight_hints = vec![0; self.indexes.len()];
        {
            let mut sink = RelationSink(&mut relation);
            trie_join_recurse(
                &self.indexes,
                &self.variable_order,
                0,
                &mut binding,
                &mut sink,
                &mut stats,
                &mut scratch,
                &mut weight_hints,
            )?;
        }
        stats.output_rows = relation.positive_rows().count();

        Ok(TrieJoinResult { relation, stats })
    }

    /// Streams each positive output row of the join to `emit` as
    /// `(binding, weight)` without materializing a `BindingRelation`. The v3
    /// `Factorise` substrate: the caller reads each template variable's bound
    /// `TermId` from the binding and writes the output directly, so the join
    /// result is never built as a relation. Returns the join stats.
    pub(crate) fn for_each<E: FnMut(&BindingAssignment, i64)>(
        &self,
        emit: E,
    ) -> Result<TrieJoinStats, BindingRelationError> {
        let mut stats = self.base_stats();
        let mut binding = BindingAssignment::default();
        let mut scratch = trie_join_scratch(self.variable_order.len());
        let mut weight_hints = vec![0; self.indexes.len()];
        let mut sink = CallbackSink { emit, rows: 0 };
        trie_join_recurse(
            &self.indexes,
            &self.variable_order,
            0,
            &mut binding,
            &mut sink,
            &mut stats,
            &mut scratch,
            &mut weight_hints,
        )?;
        stats.output_rows = sink.rows;
        Ok(stats)
    }

    /// Counts prepared trie-backed join rows without materializing output rows.
    pub fn count(&self) -> Result<TrieJoinCount, BindingRelationError> {
        let (rows, stats) = self.aggregate(false)?;
        Ok(TrieJoinCount { rows, stats })
    }

    /// Checks whether a prepared trie-backed join has a positive row.
    pub fn exists(&self) -> Result<TrieJoinExistence, BindingRelationError> {
        let (rows, stats) = self.aggregate(true)?;
        Ok(TrieJoinExistence {
            exists: rows > 0,
            stats,
        })
    }

    fn aggregate(
        &self,
        stop_after_first: bool,
    ) -> Result<(usize, TrieJoinStats), BindingRelationError> {
        let mut stats = self.base_stats();
        let mut binding = BindingAssignment::default();
        let mut rows = 0usize;
        let mut scratch = trie_join_scratch(self.variable_order.len());
        let mut weight_hints = vec![0; self.indexes.len()];
        trie_join_aggregate_recurse(
            &self.indexes,
            &self.variable_order,
            0,
            &mut binding,
            &mut stats,
            &mut rows,
            stop_after_first,
            &mut scratch,
            &mut weight_hints,
        )?;
        stats.output_rows = rows;

        Ok((rows, stats))
    }

    /// Traces the prepared trie-backed traversal without materializing rows.
    pub fn trace(&self) -> Result<TrieJoinTrace, BindingRelationError> {
        let mut binding = BindingAssignment::default();
        let mut steps = Vec::new();
        let mut candidate_bindings = 0;
        trie_join_trace_recurse(
            &self.indexes,
            &self.variable_order,
            0,
            &mut binding,
            &mut steps,
            &mut candidate_bindings,
        )?;

        Ok(TrieJoinTrace {
            variable_order: self.variable_order.clone(),
            relation_indexes: self.indexes.len(),
            indexed_rows: self.indexed_rows,
            trie_nodes: self.trie_nodes,
            steps: steps.into_boxed_slice(),
            candidate_bindings,
        })
    }
}

/// Trie-backed variable-at-a-time join over positive BindingSpace rows.
///
/// This is the first physical LFTJ-style sidecar kernel: each input relation is
/// re-keyed as a trie compatible with the global `variable_order`, then the
/// executor synchronizes the current variable's sorted domains and recurses.
pub fn trie_join(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
) -> Result<TrieJoinResult, BindingRelationError> {
    PreparedTrieJoin::prepare(relations, variable_order)?.execute()
}

/// Streams each positive output row of the trie-backed join to `emit` as
/// `(binding, weight)` without materializing the output relation: the v3
/// `Factorise` operator substrate for the fused emit. The caller reads each
/// output variable's bound `TermId` from the binding and writes the template
/// output directly.
pub(crate) fn trie_join_for_each<E: FnMut(&BindingAssignment, i64)>(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
    emit: E,
) -> Result<TrieJoinStats, BindingRelationError> {
    PreparedTrieJoin::prepare(relations, variable_order)?.for_each(emit)
}

/// Counts output rows from the trie-backed variable-at-a-time join without
/// materializing the output relation.
pub fn trie_join_count(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
) -> Result<TrieJoinCount, BindingRelationError> {
    PreparedTrieJoin::prepare(relations, variable_order)?.count()
}

/// Checks whether the trie-backed variable-at-a-time join has any positive
/// output row, stopping after the first witness.
pub fn trie_join_exists(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
) -> Result<TrieJoinExistence, BindingRelationError> {
    PreparedTrieJoin::prepare(relations, variable_order)?.exists()
}

/// Traces the trie-backed variable-at-a-time traversal without materializing
/// the final joined relation.
///
/// The trace records every variable-depth domain intersection that a future
/// PathMap/ReadZipper-backed factor must reproduce: bound prefix, constraining
/// domain count, presented domain values, surviving intersection values, and
/// cursor operation counts.
pub fn trie_join_trace(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
) -> Result<TrieJoinTrace, BindingRelationError> {
    PreparedTrieJoin::prepare(relations, variable_order)?.trace()
}

fn generic_join_recurse(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
    remaining_order: &[BindingVar],
    binding: &mut BindingAssignment,
    out: &mut BindingRelation,
    scratch: &mut [GenericJoinLevelScratch],
) -> Result<(), BindingRelationError> {
    let Some((&variable, child_order)) = remaining_order.split_first() else {
        let weight = generic_binding_weight(relations, binding)?;
        if weight != 0 {
            out.add(bound_output_row(variable_order, binding), weight)?;
        }
        return Ok(());
    };

    let Some((level_scratch, child_scratch)) = scratch.split_first_mut() else {
        return Err(BindingRelationError::InvalidVariableOrder);
    };
    let domain_count = generic_domains_into(
        relations,
        variable,
        binding,
        &mut level_scratch.domains,
        &mut level_scratch.bound_filters,
    );

    try_for_each_leapfrog_intersection(
        &level_scratch.domains[..domain_count],
        &mut level_scratch.positions,
        |value| {
            binding.insert(variable, value);
            let result = generic_join_recurse(
                relations,
                variable_order,
                child_order,
                binding,
                out,
                child_scratch,
            );
            binding.remove(variable);
            result?;
            Ok(false)
        },
    )?;
    Ok(())
}

fn generic_join_aggregate(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
    stop_after_first: bool,
) -> Result<usize, BindingRelationError> {
    validate_variable_order(relations, variable_order)?;

    let mut binding = BindingAssignment::default();
    let mut rows = 0usize;
    let mut scratch = generic_join_scratch(variable_order.len());
    generic_join_aggregate_recurse(
        relations,
        variable_order,
        variable_order,
        &mut binding,
        &mut rows,
        stop_after_first,
        &mut scratch,
    )?;

    Ok(rows)
}

fn generic_join_aggregate_recurse(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
    remaining_order: &[BindingVar],
    binding: &mut BindingAssignment,
    rows: &mut usize,
    stop_after_first: bool,
    scratch: &mut [GenericJoinLevelScratch],
) -> Result<bool, BindingRelationError> {
    let Some((&variable, child_order)) = remaining_order.split_first() else {
        if generic_binding_weight(relations, binding)? > 0 {
            *rows = checked_add_cardinality(*rows, 1)?;
            return Ok(stop_after_first);
        }
        return Ok(false);
    };

    let Some((level_scratch, child_scratch)) = scratch.split_first_mut() else {
        return Err(BindingRelationError::InvalidVariableOrder);
    };
    let domain_count = generic_domains_into(
        relations,
        variable,
        binding,
        &mut level_scratch.domains,
        &mut level_scratch.bound_filters,
    );

    let stopped = try_for_each_leapfrog_intersection(
        &level_scratch.domains[..domain_count],
        &mut level_scratch.positions,
        |value| {
            binding.insert(variable, value);
            let result = generic_join_aggregate_recurse(
                relations,
                variable_order,
                child_order,
                binding,
                rows,
                stop_after_first,
                child_scratch,
            );
            binding.remove(variable);
            result
        },
    )?;
    Ok(stopped)
}

fn generic_binding_weight(
    relations: &[BindingRelation],
    binding: &BindingAssignment,
) -> Result<i64, BindingRelationError> {
    relations.iter().try_fold(1i64, |weight, relation| {
        checked_mul_weight(weight, relation.weight_from_binding(binding))
    })
}

#[derive(Default)]
struct GenericJoinLevelScratch {
    domains: Vec<Vec<TermId>>,
    positions: Vec<usize>,
    bound_filters: Vec<(usize, TermId)>,
}

fn generic_join_scratch(variable_count: usize) -> Vec<GenericJoinLevelScratch> {
    (0..variable_count)
        .map(|_| GenericJoinLevelScratch::default())
        .collect()
}

fn generic_domains_into(
    relations: &[BindingRelation],
    variable: BindingVar,
    binding: &BindingAssignment,
    domains: &mut Vec<Vec<TermId>>,
    bound_filters: &mut Vec<(usize, TermId)>,
) -> usize {
    let mut domain_count = 0usize;

    for relation in relations {
        let Some(variable_index) = relation.schema_index(variable) else {
            continue;
        };

        bound_filters.clear();
        for (index, &relation_var) in relation.schema().iter().enumerate() {
            if relation_var == variable {
                continue;
            }
            if let Some(bound) = binding.get(relation_var) {
                bound_filters.push((index, bound));
            }
        }

        if domain_count == domains.len() {
            domains.push(Vec::new());
        }
        let domain = &mut domains[domain_count];
        domain.clear();

        for (row, weight) in relation.rows() {
            if weight <= 0 {
                continue;
            }
            if bound_filters
                .iter()
                .all(|&(index, bound)| row[index] == bound)
            {
                domain.push(row[variable_index]);
            }
        }

        domain.sort_unstable();
        domain.dedup();
        domain_count += 1;
    }

    for domain in domains.iter_mut().skip(domain_count) {
        domain.clear();
    }
    bound_filters.clear();

    domain_count
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RelationTrieIndex {
    schema: Box<[BindingVar]>,
    positions: [usize; BINDING_SLOT_COUNT],
    child_prefix_keys: Box<[NaturalJoinKey]>,
    child_range_starts: Box<[usize]>,
    child_range_ends: Box<[usize]>,
    child_values: Box<[TermId]>,
    weight_keys: Box<[NaturalJoinKey]>,
    weight_values: Box<[i64]>,
}

impl RelationTrieIndex {
    fn build(
        relation: &BindingRelation,
        variable_order: &[BindingVar],
    ) -> Result<Self, BindingRelationError> {
        if has_duplicates(relation.schema()) {
            return Err(BindingRelationError::InvalidVariableOrder);
        }

        let schema = variable_order
            .iter()
            .copied()
            .filter(|variable| relation.schema().contains(variable))
            .collect::<Vec<_>>();
        let source_indexes = indexes(relation, &schema)?;

        let mut child_lists = BTreeMap::<NaturalJoinKey, Vec<TermId>>::new();
        let mut weights = BTreeMap::<NaturalJoinKey, i64>::new();
        for (row, weight) in relation.rows() {
            if weight <= 0 {
                continue;
            }

            let entry = weights
                .entry(NaturalJoinKey::from_row(row, &source_indexes))
                .or_default();
            *entry = checked_add_weight(*entry, weight)?;

            for depth in 0..source_indexes.len() {
                child_lists
                    .entry(NaturalJoinKey::from_row(row, &source_indexes[..depth]))
                    .or_default()
                    .push(row[source_indexes[depth]]);
            }
        }

        let mut positions = [MISSING_TRIE_POSITION; BINDING_SLOT_COUNT];
        for (index, variable) in schema.iter().copied().enumerate() {
            positions[usize::from(variable.0)] = index;
        }
        let child_prefix_count = child_lists.len();
        let mut child_prefix_keys = Vec::with_capacity(child_prefix_count);
        let mut child_range_starts = Vec::with_capacity(child_prefix_count);
        let mut child_range_ends = Vec::with_capacity(child_prefix_count);
        let mut child_values = Vec::new();
        for (prefix, mut children) in child_lists {
            children.sort_unstable();
            children.dedup();
            let start = child_values.len();
            child_values.extend_from_slice(&children);
            let end = child_values.len();
            child_prefix_keys.push(prefix);
            child_range_starts.push(start);
            child_range_ends.push(end);
        }
        let weight_count = weights.len();
        let mut weight_keys = Vec::with_capacity(weight_count);
        let mut weight_values = Vec::with_capacity(weight_count);
        for (key, weight) in weights {
            weight_keys.push(key);
            weight_values.push(weight);
        }

        Ok(Self {
            schema: schema.into_boxed_slice(),
            positions,
            child_prefix_keys: child_prefix_keys.into_boxed_slice(),
            child_range_starts: child_range_starts.into_boxed_slice(),
            child_range_ends: child_range_ends.into_boxed_slice(),
            child_values: child_values.into_boxed_slice(),
            weight_keys: weight_keys.into_boxed_slice(),
            weight_values: weight_values.into_boxed_slice(),
        })
    }

    fn domain(&self, variable: BindingVar, binding: &BindingAssignment) -> Option<&[TermId]> {
        let mut hint = 0;
        self.domain_with_hint(variable, binding, &mut hint)
    }

    fn domain_with_hint(
        &self,
        variable: BindingVar,
        binding: &BindingAssignment,
        hint: &mut usize,
    ) -> Option<&[TermId]> {
        let position = self.positions[usize::from(variable.0)];
        if position == MISSING_TRIE_POSITION {
            return None;
        }

        let prefix_variables = &self.schema[..position];
        let search = binary_search_bound_prefix_from(
            &self.child_prefix_keys,
            prefix_variables,
            binding,
            *hint,
        )?;

        *hint = match search {
            Ok(index) | Err(index) => index,
        };

        Some(search.map_or(&[], |index| {
            let start = self.child_range_starts[index];
            let end = self.child_range_ends[index];
            &self.child_values[start..end]
        }))
    }

    fn weight(&self, binding: &BindingAssignment) -> i64 {
        let mut hint = 0;
        self.weight_with_hint(binding, &mut hint)
    }

    fn weight_with_hint(&self, binding: &BindingAssignment, hint: &mut usize) -> i64 {
        let Some(search) =
            binary_search_bound_prefix_from(&self.weight_keys, &self.schema, binding, *hint)
        else {
            return 0;
        };

        *hint = match search {
            Ok(index) | Err(index) => index,
        };

        search.map_or(0, |index| self.weight_values[index])
    }
}

#[derive(Clone, Copy, Debug)]
struct SliceDomainCursor<'a> {
    domain: &'a [TermId],
    position: usize,
}

impl<'a> SliceDomainCursor<'a> {
    fn new(domain: &'a [TermId]) -> Self {
        Self {
            domain,
            position: 0,
        }
    }
}

fn lower_bound_from(domain: &[TermId], start: usize, target: TermId) -> usize {
    let n = domain.len();
    if start >= n {
        return n;
    }

    // Galloping lower bound: the leapfrog seek advances forward monotonically,
    // usually by a small amount, so expand the bracket exponentially from `start`
    // and binary-search only the bracket. O(log distance) instead of O(log
    // remaining), the seek the Leapfrog Triejoin paper specifies. The lower bound
    // is unique, so the result is identical to a full partition_point.
    let mut bound = 1;
    while start + bound < n && domain[start + bound - 1] < target {
        bound *= 2;
    }
    let lo = start + bound / 2;
    let hi = (start + bound).min(n);
    lo + domain[lo..hi].partition_point(|&value| value < target)
}

impl BindingDomainCursor for SliceDomainCursor<'_> {
    fn key(&self) -> Option<TermId> {
        self.domain.get(self.position).copied()
    }

    fn at_end(&self) -> bool {
        self.position >= self.domain.len()
    }

    fn next(&mut self) {
        if !self.at_end() {
            self.position += 1;
        }
    }

    fn seek(&mut self, target: TermId) -> usize {
        if self.at_end() {
            return 0;
        }

        let next_position = lower_bound_from(self.domain, self.position, target);
        let skipped = next_position - self.position;
        self.position = next_position;
        skipped
    }
}

struct LeapfrogIntersection<'a, C> {
    cursors: &'a mut [C],
    target: TermId,
    exhausted: bool,
}

impl<'a, C: BindingDomainCursor> LeapfrogIntersection<'a, C> {
    fn new(cursors: &'a mut [C], stats: &mut TrieJoinStats) -> Option<Self> {
        stats.domain_cursor_opens += cursors.len();

        if cursors.is_empty() || cursors.iter().any(BindingDomainCursor::at_end) {
            return None;
        }

        let target = cursors
            .iter()
            .filter_map(BindingDomainCursor::key)
            .max()
            .expect("non-empty cursors have keys");

        Some(Self {
            cursors,
            target,
            exhausted: false,
        })
    }

    fn next(&mut self, stats: &mut TrieJoinStats) -> Option<TermId> {
        if self.exhausted {
            return None;
        }

        loop {
            let mut changed = false;
            for cursor in self.cursors.iter_mut() {
                stats.domain_cursor_seeks += 1;
                stats.domain_cursor_skips += cursor.seek(self.target);
                if cursor.at_end() {
                    return None;
                }
                let Some(key) = cursor.key() else {
                    return None;
                };
                if key > self.target {
                    self.target = key;
                    changed = true;
                }
            }
            if !changed {
                let value = self.target;
                stats.domain_cursor_nexts += 1;
                self.cursors[0].next();
                if self.cursors[0].at_end() {
                    self.exhausted = true;
                } else {
                    self.target = self.cursors[0]
                        .key()
                        .expect("cursor just checked as not at end");
                }
                return Some(value);
            }
        }
    }
}

/// Leaf action of the trie-backed join, so one recursion serves both the
/// materializing `execute` and the streaming `for_each` (the v3 `Factorise`
/// substrate: emit each binding without building a `BindingRelation`). Zero-cost,
/// monomorphized and inlined.
trait TrieJoinSink {
    fn emit(
        &mut self,
        variable_order: &[BindingVar],
        binding: &BindingAssignment,
        weight: i64,
    ) -> Result<(), BindingRelationError>;
}

/// Materializing sink: each join row becomes a row of the output relation.
struct RelationSink<'r>(&'r mut BindingRelation);
impl TrieJoinSink for RelationSink<'_> {
    #[inline]
    fn emit(
        &mut self,
        variable_order: &[BindingVar],
        binding: &BindingAssignment,
        weight: i64,
    ) -> Result<(), BindingRelationError> {
        self.0.add(bound_output_row(variable_order, binding), weight)
    }
}

/// Streaming sink: hands each binding (and weight) to a callback and counts rows,
/// never materializing the relation.
struct CallbackSink<E> {
    emit: E,
    rows: usize,
}
impl<E: FnMut(&BindingAssignment, i64)> TrieJoinSink for CallbackSink<E> {
    #[inline]
    fn emit(
        &mut self,
        _variable_order: &[BindingVar],
        binding: &BindingAssignment,
        weight: i64,
    ) -> Result<(), BindingRelationError> {
        (self.emit)(binding, weight);
        self.rows += 1;
        Ok(())
    }
}

fn trie_join_recurse<'a, O: TrieJoinSink>(
    indexes: &'a [RelationTrieIndex],
    variable_order: &[BindingVar],
    level: usize,
    binding: &mut BindingAssignment,
    out: &mut O,
    stats: &mut TrieJoinStats,
    scratch: &mut [TrieJoinLevelScratch<'a>],
    weight_hints: &mut [usize],
) -> Result<(), BindingRelationError> {
    if level == variable_order.len() {
        let mut weight = 1i64;
        for (index_id, index) in indexes.iter().enumerate() {
            stats.weight_lookups += 1;
            weight = checked_mul_weight(
                weight,
                index.weight_with_hint(binding, &mut weight_hints[index_id]),
            )?;
        }
        if weight != 0 {
            out.emit(variable_order, binding, weight)?;
        }
        return Ok(());
    }

    let variable = variable_order[level];
    let Some((level_scratch, child_scratch)) = scratch.split_first_mut() else {
        return Err(BindingRelationError::InvalidVariableOrder);
    };
    let domain_count = trie_domains_into(
        indexes,
        variable,
        binding,
        &mut level_scratch.domains,
        &mut level_scratch.domain_hints,
    );
    if domain_count == 0 {
        return Err(BindingRelationError::InvalidVariableOrder);
    }
    let domains = &level_scratch.domains[..domain_count];

    let Some(()) = record_domain_intersection(domains, stats) else {
        return Ok(());
    };
    if let [domain] = domains {
        for &value in *domain {
            binding.insert(variable, value);
            trie_join_recurse(
                indexes,
                variable_order,
                level + 1,
                binding,
                out,
                stats,
                child_scratch,
                weight_hints,
            )?;
        }
        binding.remove(variable);
        return Ok(());
    }

    slice_domain_cursors_from_domains(domains, &mut level_scratch.cursors);
    let Some(mut intersection) = LeapfrogIntersection::new(&mut level_scratch.cursors, stats)
    else {
        return Ok(());
    };

    while let Some(value) = intersection.next(stats) {
        binding.insert(variable, value);
        trie_join_recurse(
            indexes,
            variable_order,
            level + 1,
            binding,
            out,
            stats,
            child_scratch,
            weight_hints,
        )?;
    }
    binding.remove(variable);
    Ok(())
}

fn trie_join_aggregate_recurse<'a>(
    indexes: &'a [RelationTrieIndex],
    variable_order: &[BindingVar],
    level: usize,
    binding: &mut BindingAssignment,
    stats: &mut TrieJoinStats,
    rows: &mut usize,
    stop_after_first: bool,
    scratch: &mut [TrieJoinLevelScratch<'a>],
    weight_hints: &mut [usize],
) -> Result<bool, BindingRelationError> {
    if level == variable_order.len() {
        let mut weight = 1i64;
        for (index_id, index) in indexes.iter().enumerate() {
            stats.weight_lookups += 1;
            weight = checked_mul_weight(
                weight,
                index.weight_with_hint(binding, &mut weight_hints[index_id]),
            )?;
        }
        if weight > 0 {
            *rows = checked_add_cardinality(*rows, 1)?;
            return Ok(stop_after_first);
        }
        return Ok(false);
    }

    let variable = variable_order[level];
    let Some((level_scratch, child_scratch)) = scratch.split_first_mut() else {
        return Err(BindingRelationError::InvalidVariableOrder);
    };
    let domain_count = trie_domains_into(
        indexes,
        variable,
        binding,
        &mut level_scratch.domains,
        &mut level_scratch.domain_hints,
    );
    if domain_count == 0 {
        return Err(BindingRelationError::InvalidVariableOrder);
    }
    let domains = &level_scratch.domains[..domain_count];

    let Some(()) = record_domain_intersection(domains, stats) else {
        return Ok(false);
    };
    if let [domain] = domains {
        for &value in *domain {
            binding.insert(variable, value);
            let stop = trie_join_aggregate_recurse(
                indexes,
                variable_order,
                level + 1,
                binding,
                stats,
                rows,
                stop_after_first,
                child_scratch,
                weight_hints,
            )?;
            binding.remove(variable);
            if stop {
                return Ok(true);
            }
        }
        return Ok(false);
    }

    slice_domain_cursors_from_domains(domains, &mut level_scratch.cursors);
    let Some(mut intersection) = LeapfrogIntersection::new(&mut level_scratch.cursors, stats)
    else {
        return Ok(false);
    };

    while let Some(value) = intersection.next(stats) {
        binding.insert(variable, value);
        let stop = trie_join_aggregate_recurse(
            indexes,
            variable_order,
            level + 1,
            binding,
            stats,
            rows,
            stop_after_first,
            child_scratch,
            weight_hints,
        )?;
        binding.remove(variable);
        if stop {
            return Ok(true);
        }
    }
    Ok(false)
}

fn trie_join_trace_recurse(
    indexes: &[RelationTrieIndex],
    variable_order: &[BindingVar],
    level: usize,
    binding: &mut BindingAssignment,
    steps: &mut Vec<TrieJoinTraceStep>,
    candidate_bindings: &mut usize,
) -> Result<(), BindingRelationError> {
    if level == variable_order.len() {
        *candidate_bindings = checked_add_cardinality(*candidate_bindings, 1)?;
        return Ok(());
    }

    let variable = variable_order[level];
    let domain_entries = indexes
        .iter()
        .enumerate()
        .filter_map(|(relation_index, index)| {
            index
                .domain(variable, binding)
                .map(|domain| (relation_index, domain))
        })
        .collect::<Vec<_>>();
    let domains = domain_entries
        .iter()
        .map(|(_, domain)| *domain)
        .collect::<Vec<_>>();
    if domains.is_empty() {
        return Err(BindingRelationError::InvalidVariableOrder);
    }

    let mut stats = TrieJoinStats::default();
    let intersection = match domains.as_slice() {
        [domain] => domain.to_vec(),
        _ => leapfrog_intersection_slices(&domains, &mut stats),
    };
    steps.push(TrieJoinTraceStep {
        level,
        variable,
        bound_prefix: bound_output_row(&variable_order[..level], binding),
        participating_relations: domain_entries
            .iter()
            .map(|(relation_index, _)| *relation_index)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        relation_domain_lens: domain_entries
            .iter()
            .map(|(_, domain)| domain.len())
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        relation_domains: domain_entries
            .iter()
            .map(|(relation_index, domain)| TrieJoinTraceFactorDomain {
                relation_index: *relation_index,
                values: domain.to_vec().into_boxed_slice(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        domain_sources: domains.len(),
        domain_values: domains.iter().map(|domain| domain.len()).sum(),
        intersection: intersection.clone().into_boxed_slice(),
        cursor_opens: stats.domain_cursor_opens,
        cursor_seeks: stats.domain_cursor_seeks,
        cursor_skips: stats.domain_cursor_skips,
        cursor_nexts: stats.domain_cursor_nexts,
    });

    for value in intersection {
        binding.insert(variable, value);
        trie_join_trace_recurse(
            indexes,
            variable_order,
            level + 1,
            binding,
            steps,
            candidate_bindings,
        )?;
    }
    binding.remove(variable);
    Ok(())
}

/// Factorized binary equijoin grouped by shared variables.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorizedJoin {
    /// Shared variables.
    pub shared_schema: Box<[BindingVar]>,
    /// Variables only on the left input.
    pub left_only_schema: Box<[BindingVar]>,
    /// Variables only on the right input.
    pub right_only_schema: Box<[BindingVar]>,
    /// Groups keyed by shared assignment.
    pub groups: Box<[FactorGroup]>,
}

/// One factorized group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorGroup {
    /// Shared assignment.
    pub key: BindingRow,
    /// Distinct left residual assignments.
    pub left_residuals: Box<[BindingRow]>,
    /// Distinct right residual assignments.
    pub right_residuals: Box<[BindingRow]>,
}

impl FactorGroup {
    /// Checked number of flat rows represented by this group.
    pub fn checked_row_count(&self) -> Result<usize, BindingRelationError> {
        checked_mul_cardinality(self.left_residuals.len(), self.right_residuals.len())
    }

    /// Number of flat rows represented by this group.
    ///
    /// Panics if the represented row count exceeds `usize::MAX`. Use
    /// [`FactorGroup::checked_row_count`] when a caller needs to handle that
    /// boundary explicitly.
    pub fn row_count(&self) -> usize {
        self.checked_row_count()
            .expect("factorized group row count overflow")
    }
}

impl FactorizedJoin {
    /// Builds a factorized set join from positive rows in both relations.
    pub fn from_relations(
        left: &BindingRelation,
        right: &BindingRelation,
    ) -> Result<Self, BindingRelationError> {
        let shared_schema = left
            .schema()
            .iter()
            .copied()
            .filter(|variable| right.schema().contains(variable))
            .collect::<Vec<_>>();
        let left_only_schema = left
            .schema()
            .iter()
            .copied()
            .filter(|variable| !shared_schema.contains(variable))
            .collect::<Vec<_>>();
        let right_only_schema = right
            .schema()
            .iter()
            .copied()
            .filter(|variable| !shared_schema.contains(variable))
            .collect::<Vec<_>>();

        let left_shared = indexes(left, &shared_schema)?;
        let right_shared = indexes(right, &shared_schema)?;
        let left_only = indexes(left, &left_only_schema)?;
        let right_only = indexes(right, &right_only_schema)?;

        let mut left_groups: BTreeMap<NaturalJoinKey, BTreeSet<NaturalJoinKey>> = BTreeMap::new();
        let mut right_groups: BTreeMap<NaturalJoinKey, BTreeSet<NaturalJoinKey>> = BTreeMap::new();

        for row in left.positive_rows() {
            left_groups
                .entry(NaturalJoinKey::from_row(row, &left_shared))
                .or_default()
                .insert(NaturalJoinKey::from_row(row, &left_only));
        }
        for row in right.positive_rows() {
            right_groups
                .entry(NaturalJoinKey::from_row(row, &right_shared))
                .or_default()
                .insert(NaturalJoinKey::from_row(row, &right_only));
        }

        let mut groups = Vec::new();
        for (key, left_residuals) in left_groups {
            if let Some(right_residuals) = right_groups.remove(&key) {
                groups.push(FactorGroup {
                    key: key.into_row(),
                    left_residuals: rows_from_key_set(left_residuals),
                    right_residuals: rows_from_key_set(right_residuals),
                });
            }
        }

        Ok(Self {
            shared_schema: shared_schema.into_boxed_slice(),
            left_only_schema: left_only_schema.into_boxed_slice(),
            right_only_schema: right_only_schema.into_boxed_slice(),
            groups: groups.into_boxed_slice(),
        })
    }

    /// Checked number of flat rows represented by the factorized join.
    pub fn checked_count(&self) -> Result<usize, BindingRelationError> {
        self.groups.iter().try_fold(0usize, |total, group| {
            checked_add_cardinality(total, group.checked_row_count()?)
        })
    }

    /// Number of flat rows represented by the factorized join.
    ///
    /// Panics if the represented row count exceeds `usize::MAX`. Use
    /// [`FactorizedJoin::checked_count`] when a caller needs a recoverable
    /// overflow boundary.
    pub fn count(&self) -> usize {
        self.checked_count()
            .expect("factorized join row count overflow")
    }

    /// Checked structural node count for comparing factorization to flat rows.
    pub fn checked_factorized_node_count(&self) -> Result<usize, BindingRelationError> {
        let groups = self.groups.iter().try_fold(0usize, |total, group| {
            let group_nodes = checked_add_cardinality(
                1,
                checked_add_cardinality(group.left_residuals.len(), group.right_residuals.len())?,
            )?;
            checked_add_cardinality(total, group_nodes)
        })?;
        checked_add_cardinality(1, groups)
    }

    /// Rough structural node count for comparing factorization to flat rows.
    pub fn factorized_node_count(&self) -> usize {
        self.checked_factorized_node_count()
            .expect("factorized join node count overflow")
    }

    /// Enumerates flat rows as compact binding rows in schema order:
    /// shared + left-only + right-only.
    pub fn binding_rows(&self) -> impl Iterator<Item = BindingRow> + '_ {
        self.groups.iter().flat_map(|group| {
            group.left_residuals.iter().flat_map(move |left| {
                group
                    .right_residuals
                    .iter()
                    .map(move |right| factorized_output_row(&group.key, left, right))
            })
        })
    }

    /// Enumerates flat rows in schema order: shared + left-only + right-only.
    pub fn rows(&self) -> impl Iterator<Item = Vec<TermId>> + '_ {
        self.binding_rows().map(Vec::from)
    }
}

fn indexes(
    relation: &BindingRelation,
    variables: &[BindingVar],
) -> Result<Vec<usize>, BindingRelationError> {
    variables
        .iter()
        .map(|&variable| {
            relation
                .schema_index(variable)
                .ok_or(BindingRelationError::UnknownVariable { variable })
        })
        .collect()
}

fn rows_from_key_set(rows: BTreeSet<NaturalJoinKey>) -> Box<[BindingRow]> {
    rows.into_iter()
        .map(NaturalJoinKey::into_row)
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn factorized_output_row(shared: &[TermId], left: &[TermId], right: &[TermId]) -> BindingRow {
    let output_len = shared.len() + left.len() + right.len();
    match output_len {
        0 => Box::from([]),
        1 => Box::from([factorized_output_value(shared, left, right, 0)]),
        2 => Box::from([
            factorized_output_value(shared, left, right, 0),
            factorized_output_value(shared, left, right, 1),
        ]),
        3 => Box::from([
            factorized_output_value(shared, left, right, 0),
            factorized_output_value(shared, left, right, 1),
            factorized_output_value(shared, left, right, 2),
        ]),
        4 => Box::from([
            factorized_output_value(shared, left, right, 0),
            factorized_output_value(shared, left, right, 1),
            factorized_output_value(shared, left, right, 2),
            factorized_output_value(shared, left, right, 3),
        ]),
        _ => {
            let mut output = Vec::with_capacity(output_len);
            output.extend_from_slice(shared);
            output.extend_from_slice(left);
            output.extend_from_slice(right);
            output.into_boxed_slice()
        }
    }
}

fn factorized_output_value(
    shared: &[TermId],
    left: &[TermId],
    right: &[TermId],
    output_index: usize,
) -> TermId {
    if output_index < shared.len() {
        shared[output_index]
    } else {
        let residual_index = output_index - shared.len();
        if residual_index < left.len() {
            left[residual_index]
        } else {
            right[residual_index - left.len()]
        }
    }
}

fn bound_output_row(variables: &[BindingVar], binding: &BindingAssignment) -> BindingRow {
    match variables {
        [] => Box::from([]),
        [a] => Box::from([binding.bound(*a)]),
        [a, b] => Box::from([binding.bound(*a), binding.bound(*b)]),
        [a, b, c] => Box::from([binding.bound(*a), binding.bound(*b), binding.bound(*c)]),
        [a, b, c, d] => Box::from([
            binding.bound(*a),
            binding.bound(*b),
            binding.bound(*c),
            binding.bound(*d),
        ]),
        variables => variables
            .iter()
            .map(|&variable| binding.bound(variable))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    }
}

fn validate_semijoin_edge(
    relation_count: usize,
    child: usize,
    parent: usize,
) -> Result<(), BindingRelationError> {
    if child == parent || child >= relation_count || parent >= relation_count {
        return Err(BindingRelationError::InvalidVariableOrder);
    }
    Ok(())
}

fn validate_variable_order(
    relations: &[BindingRelation],
    variable_order: &[BindingVar],
) -> Result<(), BindingRelationError> {
    if has_duplicates(variable_order)
        || relations
            .iter()
            .any(|relation| has_duplicates(relation.schema()))
    {
        return Err(BindingRelationError::InvalidVariableOrder);
    }

    let all_variables = relations
        .iter()
        .flat_map(|relation| relation.schema().iter().copied())
        .collect::<BTreeSet<_>>();
    let ordered = variable_order.iter().copied().collect::<BTreeSet<_>>();
    if all_variables != ordered {
        return Err(BindingRelationError::InvalidVariableOrder);
    }

    Ok(())
}

fn has_duplicates(variables: &[BindingVar]) -> bool {
    let mut seen = BTreeSet::new();
    variables
        .iter()
        .copied()
        .any(|variable| !seen.insert(variable))
}

fn try_for_each_leapfrog_intersection<E>(
    domains: &[Vec<TermId>],
    positions: &mut Vec<usize>,
    mut visit: impl FnMut(TermId) -> Result<bool, E>,
) -> Result<bool, E> {
    if domains.is_empty() || domains.iter().any(Vec::is_empty) {
        return Ok(false);
    }

    positions.clear();
    positions.resize(domains.len(), 0);
    let mut target = domains
        .iter()
        .map(|domain| domain[0])
        .max()
        .expect("domains are non-empty");

    loop {
        let mut changed = false;
        for (index, domain) in domains.iter().enumerate() {
            while positions[index] < domain.len() && domain[positions[index]] < target {
                positions[index] += 1;
            }
            if positions[index] >= domain.len() {
                return Ok(false);
            }
            if domain[positions[index]] > target {
                target = domain[positions[index]];
                changed = true;
            }
        }
        if !changed {
            if visit(target)? {
                return Ok(true);
            }
            positions[0] += 1;
            if positions[0] >= domains[0].len() {
                return Ok(false);
            }
            target = domains[0][positions[0]];
        }
    }
}

fn leapfrog_intersection_slices(domains: &[&[TermId]], stats: &mut TrieJoinStats) -> Vec<TermId> {
    let mut cursors = Vec::new();
    let Some(()) = slice_domain_cursors_into(domains, &mut cursors, stats) else {
        return Vec::new();
    };

    leapfrog_intersection_cursors(&mut cursors, stats)
}

#[derive(Default)]
struct TrieJoinLevelScratch<'a> {
    domains: Vec<&'a [TermId]>,
    cursors: Vec<SliceDomainCursor<'a>>,
    domain_hints: Vec<usize>,
}

fn trie_join_scratch<'a>(variable_count: usize) -> Vec<TrieJoinLevelScratch<'a>> {
    (0..variable_count)
        .map(|_| TrieJoinLevelScratch::default())
        .collect()
}

fn trie_domains_into<'a>(
    indexes: &'a [RelationTrieIndex],
    variable: BindingVar,
    binding: &BindingAssignment,
    out: &mut Vec<&'a [TermId]>,
    hints: &mut Vec<usize>,
) -> usize {
    out.clear();
    if hints.len() < indexes.len() {
        hints.resize(indexes.len(), 0);
    }

    for (index_id, index) in indexes.iter().enumerate() {
        if let Some(domain) = index.domain_with_hint(variable, binding, &mut hints[index_id]) {
            out.push(domain);
        }
    }
    out.len()
}

fn slice_domain_cursors_into<'a>(
    domains: &[&'a [TermId]],
    cursors: &mut Vec<SliceDomainCursor<'a>>,
    stats: &mut TrieJoinStats,
) -> Option<()> {
    cursors.clear();
    record_domain_intersection(domains, stats)?;
    slice_domain_cursors_from_domains(domains, cursors);
    Some(())
}

fn record_domain_intersection(domains: &[&[TermId]], stats: &mut TrieJoinStats) -> Option<()> {
    stats.domain_intersections += 1;
    stats.domain_sources += domains.len();
    stats.domain_values += domains.iter().map(|domain| domain.len()).sum::<usize>();

    if domains.is_empty() || domains.iter().any(|domain| domain.is_empty()) {
        return None;
    }

    Some(())
}

fn slice_domain_cursors_from_domains<'a>(
    domains: &[&'a [TermId]],
    cursors: &mut Vec<SliceDomainCursor<'a>>,
) {
    cursors.clear();
    cursors.extend(domains.iter().map(|&domain| SliceDomainCursor::new(domain)));
}

fn leapfrog_intersection_cursors<C: BindingDomainCursor>(
    cursors: &mut [C],
    stats: &mut TrieJoinStats,
) -> Vec<TermId> {
    let Some(mut intersection) = LeapfrogIntersection::new(cursors, stats) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    while let Some(value) = intersection.next(stats) {
        out.push(value);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(id: u64) -> TermId {
        TermId(id)
    }

    fn v(id: u8) -> BindingVar {
        BindingVar(id)
    }

    fn relation(schema: &[u8], rows: &[&[u64]]) -> BindingRelation {
        let mut relation =
            BindingRelation::new(schema.iter().copied().map(BindingVar).collect::<Vec<_>>());
        for row in rows {
            relation
                .add(row.iter().copied().map(TermId).collect::<Vec<_>>(), 1)
                .unwrap();
        }
        relation
    }

    #[test]
    fn trie_join_for_each_streams_the_same_rows_as_execute() {
        // The Factorise substrate (for_each) must stream exactly the rows that
        // execute materializes, so the fused emit agrees with the relation path.
        let r = relation(&[0, 1], &[&[1, 2], &[1, 3], &[2, 3]]);
        let s = relation(&[1, 2], &[&[2, 9], &[3, 9], &[3, 8]]);
        let relations = [r, s];
        let order = [v(0), v(1), v(2)];

        let executed = trie_join(&relations, &order).unwrap();
        let mut materialized: Vec<Vec<TermId>> = executed
            .relation
            .positive_rows()
            .map(|row| row.to_vec())
            .collect();

        let mut streamed: Vec<Vec<TermId>> = Vec::new();
        trie_join_for_each(&relations, &order, |binding, weight| {
            assert_eq!(weight, 1);
            streamed.push(order.iter().map(|&var| binding.get(var).unwrap()).collect());
        })
        .unwrap();

        materialized.sort();
        streamed.sort();
        assert_eq!(streamed, materialized);
        assert_eq!(streamed.len(), 5);
    }

    fn signed_relation(schema: &[u8], rows: &[(&[u64], i64)]) -> BindingRelation {
        let mut relation =
            BindingRelation::new(schema.iter().copied().map(BindingVar).collect::<Vec<_>>());
        for (row, weight) in rows {
            relation
                .add(row.iter().copied().map(TermId).collect::<Vec<_>>(), *weight)
                .unwrap();
        }
        relation
    }

    fn unary_relation(variable: u8, values: impl IntoIterator<Item = u64>) -> BindingRelation {
        let mut relation = BindingRelation::new(vec![BindingVar(variable)]);
        for value in values {
            relation.add(vec![TermId(value)], 1).unwrap();
        }
        relation
    }

    fn brute_force_domain(
        relation: &BindingRelation,
        schema: &[BindingVar],
        variable: BindingVar,
        binding: &BindingAssignment,
    ) -> Option<Vec<TermId>> {
        let variable_position = schema.iter().position(|&candidate| candidate == variable)?;
        let source_indexes = indexes(relation, schema).unwrap();
        let mut domain = Vec::new();

        'rows: for (row, weight) in relation.rows() {
            if weight <= 0 {
                continue;
            }

            for &prefix_variable in &schema[..variable_position] {
                if let Some(bound) = binding.get(prefix_variable) {
                    let source_index = relation
                        .schema()
                        .iter()
                        .position(|&candidate| candidate == prefix_variable)
                        .unwrap();
                    if row[source_index] != bound {
                        continue 'rows;
                    }
                } else {
                    return None;
                }
            }

            domain.push(row[source_indexes[variable_position]]);
        }

        domain.sort_unstable();
        domain.dedup();
        Some(domain)
    }

    #[test]
    fn relation_weight_from_binding_handles_small_and_large_arities() {
        let mut binding = BindingAssignment::default();
        for variable in 0..=4 {
            binding.insert(v(variable), t(100 + u64::from(variable)));
        }

        let empty_row: &[u64] = &[];
        let empty = signed_relation(&[], &[(empty_row, 7)]);
        let unary = signed_relation(&[0], &[(&[100], 2)]);
        let binary = signed_relation(&[0, 1], &[(&[100, 101], 3)]);
        let ternary = signed_relation(&[0, 1, 2], &[(&[100, 101, 102], 5)]);
        let quaternary = signed_relation(&[0, 1, 2, 3], &[(&[100, 101, 102, 103], 11)]);
        let larger = signed_relation(&[0, 1, 2, 3, 4], &[(&[100, 101, 102, 103, 104], 13)]);

        assert_eq!(empty.weight_from_binding(&binding), 7);
        assert_eq!(unary.weight_from_binding(&binding), 2);
        assert_eq!(binary.weight_from_binding(&binding), 3);
        assert_eq!(ternary.weight_from_binding(&binding), 5);
        assert_eq!(quaternary.weight_from_binding(&binding), 11);
        assert_eq!(larger.weight_from_binding(&binding), 13);
    }

    #[test]
    fn natural_join_bucket_keeps_unique_key_inline() {
        let first = vec![t(1), t(10)];
        let second = vec![t(1), t(20)];
        let mut bucket = NaturalJoinBucket::One((first.as_slice(), 3));

        match bucket {
            NaturalJoinBucket::One((row, weight)) => {
                assert_eq!(row, first.as_slice());
                assert_eq!(weight, 3);
            }
            NaturalJoinBucket::Many(_) => panic!("unique key should remain inline"),
        }

        bucket.push((second.as_slice(), 5));
        let mut visited = Vec::new();
        bucket
            .try_for_each::<()>(|row, weight| {
                visited.push((row.to_vec(), weight));
                Ok(())
            })
            .unwrap();

        assert_eq!(visited, vec![(first, 3), (second, 5)]);
    }

    #[test]
    fn natural_join_key_avoids_allocating_common_small_keys() {
        let row = [t(1), t(2), t(3), t(4)];

        assert_eq!(NaturalJoinKey::from_row(&row, &[]), NaturalJoinKey::Empty);
        assert_eq!(
            NaturalJoinKey::from_row(&row, &[2]),
            NaturalJoinKey::One(t(3))
        );
        assert_eq!(
            NaturalJoinKey::from_row(&row, &[2, 0]),
            NaturalJoinKey::Two(t(3), t(1))
        );
        assert_eq!(
            NaturalJoinKey::from_row(&row, &[3, 1, 0]),
            NaturalJoinKey::Many(vec![t(4), t(2), t(1)].into_boxed_slice())
        );
        assert_eq!(
            NaturalJoinKey::from_row(&row, &[2, 0]).into_row(),
            vec![t(3), t(1)].into_boxed_slice()
        );
    }

    #[test]
    fn natural_join_key_builds_compact_complete_rows() {
        let row = [t(1), t(2), t(3), t(4)];

        assert_eq!(
            NaturalJoinKey::from_complete_row(&[]),
            NaturalJoinKey::Empty
        );
        assert_eq!(
            NaturalJoinKey::from_complete_row(&row[..1]),
            NaturalJoinKey::One(t(1))
        );
        assert_eq!(
            NaturalJoinKey::from_complete_row(&row[..2]),
            NaturalJoinKey::Two(t(1), t(2))
        );
        assert_eq!(
            NaturalJoinKey::from_complete_row(&row),
            NaturalJoinKey::Many(row.into())
        );
    }

    #[test]
    fn natural_join_key_into_row_preserves_complete_row_order() {
        let row = [t(1), t(2), t(3), t(4)];

        assert_eq!(
            NaturalJoinKey::from_complete_row(&[]).into_row().as_ref(),
            &[]
        );
        assert_eq!(
            NaturalJoinKey::from_complete_row(&row[..1])
                .into_row()
                .as_ref(),
            &[t(1)]
        );
        assert_eq!(
            NaturalJoinKey::from_complete_row(&row[..2])
                .into_row()
                .as_ref(),
            &[t(1), t(2)]
        );
        assert_eq!(
            NaturalJoinKey::from_complete_row(&row).into_row().as_ref(),
            &row
        );
    }

    #[test]
    fn natural_join_key_builds_compact_prefixes_from_rows_and_bindings() {
        let row = [t(1), t(2), t(3), t(4)];

        assert_eq!(NaturalJoinKey::from_row(&row, &[]), NaturalJoinKey::Empty);
        assert_eq!(
            NaturalJoinKey::from_row(&row, &[0]),
            NaturalJoinKey::One(t(1))
        );
        assert_eq!(
            NaturalJoinKey::from_row(&row, &[0, 1]),
            NaturalJoinKey::Two(t(1), t(2))
        );
        assert_eq!(
            NaturalJoinKey::from_row(&row, &[0, 1, 2, 3]),
            NaturalJoinKey::Many(row.into())
        );

        let mut binding = BindingAssignment::default();
        binding.insert(v(0), t(10));
        binding.insert(v(1), t(20));
        binding.insert(v(2), t(30));
        assert_eq!(
            NaturalJoinKey::from_bound_prefix(&[v(0), v(1)], &binding),
            Some(NaturalJoinKey::Two(t(10), t(20)))
        );
        assert_eq!(
            NaturalJoinKey::from_bound_prefix(&[v(0), v(9)], &binding),
            None
        );
    }

    #[test]
    fn natural_join_key_searches_bound_prefixes_without_materializing_wide_keys() {
        let keys = [
            NaturalJoinKey::Empty,
            NaturalJoinKey::One(t(1)),
            NaturalJoinKey::Two(t(1), t(2)),
            NaturalJoinKey::Many(vec![t(1), t(2), t(3)].into_boxed_slice()),
            NaturalJoinKey::Many(vec![t(1), t(2), t(4)].into_boxed_slice()),
        ];
        let mut binding = BindingAssignment::default();
        binding.insert(v(0), t(1));
        binding.insert(v(1), t(2));
        binding.insert(v(2), t(3));

        assert_eq!(
            binary_search_bound_prefix(&keys, &[v(0), v(1), v(2)], &binding),
            Some(Ok(3))
        );

        binding.insert(v(2), t(5));
        assert_eq!(
            binary_search_bound_prefix(&keys, &[v(0), v(1), v(2)], &binding),
            Some(Err(5))
        );

        assert_eq!(
            binary_search_bound_prefix(&keys, &[v(0), v(9)], &binding),
            None
        );
    }

    #[test]
    fn natural_join_key_reuses_monotone_bound_prefix_hints() {
        let keys = [
            NaturalJoinKey::Empty,
            NaturalJoinKey::One(t(1)),
            NaturalJoinKey::Two(t(1), t(2)),
            NaturalJoinKey::Many(vec![t(1), t(2), t(3)].into_boxed_slice()),
            NaturalJoinKey::Many(vec![t(1), t(2), t(4)].into_boxed_slice()),
            NaturalJoinKey::Many(vec![t(2), t(0), t(0)].into_boxed_slice()),
        ];
        let mut binding = BindingAssignment::default();
        binding.insert(v(0), t(1));
        binding.insert(v(1), t(2));
        binding.insert(v(2), t(3));

        assert_eq!(
            binary_search_bound_prefix_from(&keys, &[v(0), v(1), v(2)], &binding, 3),
            Some(Ok(3))
        );

        binding.insert(v(2), t(4));
        assert_eq!(
            binary_search_bound_prefix_from(&keys, &[v(0), v(1), v(2)], &binding, 4),
            Some(Ok(4))
        );

        binding.insert(v(2), t(2));
        assert_eq!(
            binary_search_bound_prefix_from(&keys, &[v(0), v(1), v(2)], &binding, 4),
            Some(Err(3))
        );

        assert_eq!(
            binary_search_bound_prefix_from(&keys, &[v(0), v(9)], &binding, 4),
            None
        );
    }

    #[test]
    fn exponential_search_matches_binary_search() {
        // Galloping must return exactly what binary_search_by returns on sorted
        // unique keys, at targets before, on, between, and past every key.
        let keys: Vec<NaturalJoinKey> =
            (0..40u64).map(|i| NaturalJoinKey::One(t(i * 2))).collect();
        for target in 0..=82u64 {
            let probe = t(target);
            let expected = keys.binary_search_by(|key| match key {
                NaturalJoinKey::One(value) => value.cmp(&probe),
                _ => unreachable!(),
            });
            let got = exponential_search_by(&keys, |key| match key {
                NaturalJoinKey::One(value) => value.cmp(&probe),
                _ => unreachable!(),
            });
            assert_eq!(got, expected, "target={target}");
        }
        assert_eq!(exponential_search_by(&[], |_| Ordering::Equal), Err(0));
    }

    #[test]
    fn joined_output_row_preserves_small_and_wide_join_order() {
        let left = [t(1), t(2), t(3)];
        let right = [t(10), t(20), t(30), t(40)];

        assert_eq!(joined_output_row(&[], &right, &[]).as_ref(), &[]);
        assert_eq!(joined_output_row(&[], &right, &[2]).as_ref(), &[t(30)]);
        assert_eq!(
            joined_output_row(&left[..1], &right, &[1]).as_ref(),
            &[t(1), t(20)]
        );
        assert_eq!(
            joined_output_row(&left[..1], &right, &[0, 2]).as_ref(),
            &[t(1), t(10), t(30)]
        );
        assert_eq!(
            joined_output_row(&left[..2], &right, &[3, 0]).as_ref(),
            &[t(1), t(2), t(40), t(10)]
        );
        assert_eq!(
            joined_output_row(&left, &right, &[0, 1]).as_ref(),
            &[t(1), t(2), t(3), t(10), t(20)]
        );
    }

    #[test]
    fn factorized_output_row_preserves_small_and_wide_join_order() {
        let shared = [t(1), t(2)];
        let left = [t(10), t(20)];
        let right = [t(100), t(200), t(300)];

        assert_eq!(factorized_output_row(&[], &[], &[]).as_ref(), &[]);
        assert_eq!(
            factorized_output_row(&shared[..1], &[], &[]).as_ref(),
            &[t(1)]
        );
        assert_eq!(
            factorized_output_row(&shared[..1], &left[..1], &[]).as_ref(),
            &[t(1), t(10)]
        );
        assert_eq!(
            factorized_output_row(&shared[..1], &left[..1], &right[..1]).as_ref(),
            &[t(1), t(10), t(100)]
        );
        assert_eq!(
            factorized_output_row(&shared, &left[..1], &right[..1]).as_ref(),
            &[t(1), t(2), t(10), t(100)]
        );
        assert_eq!(
            factorized_output_row(&shared, &left, &right[..1]).as_ref(),
            &[t(1), t(2), t(10), t(20), t(100)]
        );
    }

    #[test]
    fn bound_output_row_preserves_small_and_wide_variable_order() {
        let mut binding = BindingAssignment::default();
        for index in 0..5 {
            binding.insert(v(index), t(u64::from(index) + 10));
        }

        assert_eq!(bound_output_row(&[], &binding).as_ref(), &[]);
        assert_eq!(bound_output_row(&[v(3)], &binding).as_ref(), &[t(13)]);
        assert_eq!(
            bound_output_row(&[v(3), v(1)], &binding).as_ref(),
            &[t(13), t(11)]
        );
        assert_eq!(
            bound_output_row(&[v(3), v(1), v(4)], &binding).as_ref(),
            &[t(13), t(11), t(14)]
        );
        assert_eq!(
            bound_output_row(&[v(3), v(1), v(4), v(0)], &binding).as_ref(),
            &[t(13), t(11), t(14), t(10)]
        );
        assert_eq!(
            bound_output_row(&[v(3), v(1), v(4), v(0), v(2)], &binding).as_ref(),
            &[t(13), t(11), t(14), t(10), t(12)]
        );
    }

    #[test]
    fn binding_assignment_remove_hides_stale_slot() {
        let mut binding = BindingAssignment::default();

        assert_eq!(binding.get(v(7)), None);
        binding.insert(v(7), t(10));
        assert_eq!(binding.get(v(7)), Some(t(10)));
        binding.remove(v(7));
        assert_eq!(binding.get(v(7)), None);
        binding.insert(v(7), t(11));
        assert_eq!(binding.bound(v(7)), t(11));
    }

    #[test]
    fn natural_join_without_common_variables_is_cartesian_product() {
        let left = relation(&[0], &[&[1], &[2]]);
        let right = relation(&[1], &[&[10], &[20], &[30]]);

        let joined = natural_join(&left, &right).unwrap();
        let rows = joined
            .positive_rows()
            .map(Vec::from)
            .collect::<BTreeSet<_>>();

        assert_eq!(joined.schema(), &[v(0), v(1)]);
        assert_eq!(
            rows,
            BTreeSet::from([
                vec![t(1), t(10)],
                vec![t(1), t(20)],
                vec![t(1), t(30)],
                vec![t(2), t(10)],
                vec![t(2), t(20)],
                vec![t(2), t(30)],
            ])
        );
    }

    #[test]
    fn natural_join_preserves_output_when_either_side_is_smaller_build_side() {
        let small = signed_relation(&[0, 1], &[(&[1, 10], 2), (&[2, 20], 3)]);
        let large = signed_relation(
            &[1, 2],
            &[
                (&[10, 100], 5),
                (&[10, 101], -7),
                (&[20, 200], 11),
                (&[30, 300], 13),
            ],
        );

        let left_smaller = natural_join(&small, &large).unwrap();
        assert_eq!(left_smaller.schema(), &[v(0), v(1), v(2)]);
        assert_eq!(left_smaller.weight(&[t(1), t(10), t(100)]), 10);
        assert_eq!(left_smaller.weight(&[t(1), t(10), t(101)]), -14);
        assert_eq!(left_smaller.weight(&[t(2), t(20), t(200)]), 33);
        assert_eq!(left_smaller.weight(&[t(3), t(30), t(300)]), 0);

        let right_smaller = natural_join(&large, &small).unwrap();
        assert_eq!(right_smaller.schema(), &[v(1), v(2), v(0)]);
        assert_eq!(right_smaller.weight(&[t(10), t(100), t(1)]), 10);
        assert_eq!(right_smaller.weight(&[t(10), t(101), t(1)]), -14);
        assert_eq!(right_smaller.weight(&[t(20), t(200), t(2)]), 33);
        assert_eq!(right_smaller.weight(&[t(30), t(300), t(3)]), 0);
    }

    #[test]
    fn natural_and_generic_join_agree_on_variable_order() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10], &[3, 20]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101], &[30, 300]]);

        let natural = natural_join(&left, &right).unwrap();
        let generic = generic_join(&[left, right], &[v(1), v(0), v(2)]).unwrap();
        let normalized_natural = natural
            .positive_rows()
            .map(|row| vec![row[1], row[0], row[2]])
            .collect::<BTreeSet<_>>();
        let generic_rows = generic
            .positive_rows()
            .map(Vec::from)
            .collect::<BTreeSet<_>>();

        assert_eq!(generic_rows, normalized_natural);
        assert_eq!(generic_rows.len(), 4);
    }

    #[test]
    fn semijoin_presence_reduces_non_participants_over_visible_rows() {
        let left = signed_relation(
            &[0, 1],
            &[(&[1, 10], 1), (&[2, 20], 1), (&[3, 30], -1), (&[4, 40], 2)],
        );
        let right = signed_relation(&[1], &[(&[10], 1), (&[20], -1), (&[40], 1)]);

        let reduced = semijoin_presence(&left, &right).unwrap();
        let rows = reduced
            .positive_rows()
            .map(Vec::from)
            .collect::<BTreeSet<_>>();

        assert_eq!(rows, BTreeSet::from([vec![t(1), t(10)], vec![t(4), t(40)]]));
        assert_eq!(reduced.weight(&[t(4), t(40)]), 2);
        assert_eq!(reduced.weight(&[t(2), t(20)]), 0);
        assert_eq!(reduced.weight(&[t(3), t(30)]), 0);
    }

    #[test]
    fn semijoin_presence_without_common_variables_uses_right_visibility() {
        let left = signed_relation(&[0], &[(&[1], 2), (&[2], -3), (&[3], 5)]);
        let right_visible = signed_relation(&[1], &[(&[10], -1), (&[20], 1)]);

        let reduced = semijoin_presence(&left, &right_visible).unwrap();

        assert_eq!(reduced.schema(), left.schema());
        assert_eq!(reduced.weight(&[t(1)]), 2);
        assert_eq!(reduced.weight(&[t(2)]), 0);
        assert_eq!(reduced.weight(&[t(3)]), 5);

        let right_hidden = signed_relation(&[1], &[(&[10], -1)]);
        let empty = semijoin_presence(&left, &right_hidden).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn semijoin_presence_matches_complete_shared_key() {
        let left = relation(&[0, 1, 2], &[&[1, 10, 100], &[2, 10, 200], &[3, 20, 100]]);
        let right = relation(&[1, 2], &[&[10, 100], &[20, 200]]);

        let reduced = semijoin_presence(&left, &right).unwrap();

        assert_eq!(reduced.weight(&[t(1), t(10), t(100)]), 1);
        assert_eq!(reduced.weight(&[t(2), t(10), t(200)]), 0);
        assert_eq!(reduced.weight(&[t(3), t(20), t(100)]), 0);
    }

    #[test]
    fn presence_filters_preserve_compact_and_wide_output_rows() {
        let binary_left = signed_relation(&[0, 1], &[(&[1, 10], 2), (&[2, 20], 3)]);
        let binary_right = relation(&[0, 1], &[&[1, 10]]);

        let binary_difference = binary_left.difference_presence(&binary_right).unwrap();

        assert_eq!(binary_difference.weight(&[t(1), t(10)]), 0);
        assert_eq!(binary_difference.weight(&[t(2), t(20)]), 3);

        let wide_left = relation(&[0, 1, 2, 3], &[&[1, 10, 100, 1000], &[2, 20, 200, 2000]]);
        let wide_right = relation(&[1, 3], &[&[10, 1000]]);

        let wide_semijoin = semijoin_presence(&wide_left, &wide_right).unwrap();

        assert_eq!(wide_semijoin.weight(&[t(1), t(10), t(100), t(1000)]), 1);
        assert_eq!(wide_semijoin.weight(&[t(2), t(20), t(200), t(2000)]), 0);
    }

    #[test]
    fn semijoin_reduce_presence_preserves_acyclic_join_output() {
        let x = v(0);
        let y = v(1);
        let z = v(2);
        let left = relation(&[0, 1], &[&[1, 10], &[2, 20], &[9, 90]]);
        let middle = relation(&[1, 2], &[&[10, 100], &[20, 200], &[90, 900], &[30, 100]]);
        let right = relation(&[2], &[&[100], &[200]]);
        let relations = [left, middle, right];
        let variable_order = [x, y, z];

        let original = generic_join(&relations, &variable_order).unwrap();
        let reduction = semijoin_reduce_presence(&relations, &[(0, 1), (2, 1)]).unwrap();
        let reduced = generic_join(&reduction.relations, &variable_order).unwrap();

        assert_eq!(reduction.bottom_up_passes, 2);
        assert_eq!(reduction.top_down_passes, 2);
        assert_eq!(reduction.removed_rows, 3);
        assert_eq!(reduction.relations[0].positive_rows().count(), 2);
        assert_eq!(reduction.relations[1].positive_rows().count(), 2);
        assert_eq!(reduction.relations[2].positive_rows().count(), 2);
        assert_eq!(reduced, original);
        assert_eq!(reduced.positive_rows().count(), 2);
    }

    #[test]
    fn semijoin_reduce_presence_rejects_invalid_edges() {
        let relation = relation(&[0], &[&[1]]);

        assert!(matches!(
            semijoin_reduce_presence(std::slice::from_ref(&relation), &[(0, 0)]),
            Err(BindingRelationError::InvalidVariableOrder)
        ));
        assert!(matches!(
            semijoin_reduce_presence(&[relation], &[(0, 1)]),
            Err(BindingRelationError::InvalidVariableOrder)
        ));
    }

    #[test]
    fn project_dedups_subset_columns_with_unit_weight() {
        // schema (a,b,c); project onto (c,a). Two rows that agree on (c,a)
        // collapse to one positive row of weight 1.
        let relation = relation(&[0, 1, 2], &[&[1, 2, 9], &[1, 5, 9], &[7, 8, 9]]);
        let projected = relation.project(&[v(2), v(0)]).unwrap();
        assert_eq!(projected.schema(), &[v(2), v(0)]);
        assert_eq!(
            positive_row_set(&projected),
            BTreeSet::from([vec![t(9), t(1)], vec![t(9), t(7)]])
        );
        for row in projected.positive_rows() {
            assert_eq!(projected.weight(row), 1);
        }
    }

    #[test]
    fn project_full_permutation_preserves_distinct_rows() {
        let relation = relation(&[0, 1], &[&[1, 2], &[3, 4]]);
        let projected = relation.project(&[v(1), v(0)]).unwrap();
        assert_eq!(
            positive_row_set(&projected),
            BTreeSet::from([vec![t(2), t(1)], vec![t(4), t(3)]])
        );
    }

    #[test]
    fn project_onto_empty_yields_nullary_existence() {
        // Projecting onto no variables gives the nullary "is there any row"
        // relation: one empty row when non-empty, no rows when empty.
        let non_empty = relation(&[0, 1], &[&[1, 2], &[3, 4]]);
        let nullary = non_empty.project(&[]).unwrap();
        assert_eq!(nullary.positive_rows().count(), 1);
        assert!(nullary.positive_rows().any(<[TermId]>::is_empty));

        let empty = relation(&[0, 1], &[]);
        assert_eq!(empty.project(&[]).unwrap().positive_rows().count(), 0);
    }

    #[test]
    fn project_ignores_non_positive_rows() {
        let relation = signed_relation(&[0, 1], &[(&[1, 2], 1), (&[3, 4], -1), (&[5, 6], 0)]);
        let projected = relation.project(&[v(0)]).unwrap();
        assert_eq!(positive_row_set(&projected), BTreeSet::from([vec![t(1)]]));
    }

    #[test]
    fn project_rejects_unknown_and_repeated_variables() {
        let relation = relation(&[0, 1], &[&[1, 2]]);
        assert!(matches!(
            relation.project(&[v(9)]),
            Err(BindingRelationError::UnknownVariable { variable }) if variable == v(9)
        ));
        assert!(matches!(
            relation.project(&[v(0), v(0)]),
            Err(BindingRelationError::InvalidVariableOrder)
        ));
    }

    #[test]
    fn distinct_cardinalities_count_projected_values() {
        let relation = relation(&[0, 1, 2], &[&[1, 2, 9], &[1, 5, 9], &[3, 2, 9]]);
        // var 0 has values {1, 3}.
        assert_eq!(relation.distinct_value_count(v(0)).unwrap(), 2);
        // var 2 has value {9}.
        assert_eq!(relation.distinct_value_count(v(2)).unwrap(), 1);
        // prefix (0,1): (1,2), (1,5), (3,2) are all distinct.
        assert_eq!(relation.distinct_prefix_count(&[v(0), v(1)]).unwrap(), 3);
        // prefix (0,2): (1,9) appears twice, (3,9) once.
        assert_eq!(relation.distinct_prefix_count(&[v(0), v(2)]).unwrap(), 2);
        assert!(relation.distinct_value_count(v(9)).is_err());
    }

    #[test]
    fn relabel_preserves_rows_and_enables_realignment() {
        let r = relation(&[0, 1], &[&[1, 2], &[3, 4]]);
        let relabeled = r.relabel(&[v(5), v(2)]).unwrap();
        assert_eq!(relabeled.schema(), &[v(5), v(2)]);
        assert_eq!(positive_row_set(&relabeled), positive_row_set(&r));

        // Realigned onto variable 2, it now joins another relation on it.
        let s = relation(&[2, 3], &[&[2, 9], &[4, 8]]);
        let joined = natural_join(&relabeled, &s).unwrap();
        assert_eq!(joined.positive_rows().count(), 2);

        assert!(r.relabel(&[v(5)]).is_err());
        assert!(r.relabel(&[v(5), v(5)]).is_err());
    }

    /// Positions of relation indexes in the ear-removal order, for asserting
    /// children precede parents.
    fn ear_positions(tree: &GyoJoinTree) -> Vec<usize> {
        let mut position = vec![usize::MAX; tree.ear_order.len()];
        for (slot, &index) in tree.ear_order.iter().enumerate() {
            position[index] = slot;
        }
        position
    }

    fn positive_row_set(relation: &BindingRelation) -> BTreeSet<Vec<TermId>> {
        relation.positive_rows().map(<[TermId]>::to_vec).collect()
    }

    #[test]
    fn gyo_chain_query_is_acyclic_with_valid_reduction_order() {
        // R(a,b), S(b,c), T(c,d): a path query, α-acyclic.
        let relations = [
            relation(&[0, 1], &[&[1, 2]]),
            relation(&[1, 2], &[&[2, 3]]),
            relation(&[2, 3], &[&[3, 4]]),
        ];
        let tree = gyo_join_tree(&relations);
        assert!(tree.acyclic);
        assert!(tree.residual.is_empty());
        assert_eq!(tree.ear_order.len(), 3);
        // Each (child, parent) edge removes the child before its witness parent,
        // i.e. a valid leaves→root full-reduction order.
        let position = ear_positions(&tree);
        for &(child, parent) in tree.edges.iter() {
            assert!(position[child] < position[parent]);
        }
    }

    #[test]
    fn gyo_triangle_query_is_cyclic() {
        // R(a,b), S(b,c), T(c,a): the canonical triangle, α-cyclic, no ear, no
        // full reducer exists (BFMY 1983).
        let relations = [
            relation(&[0, 1], &[&[1, 2]]),
            relation(&[1, 2], &[&[2, 3]]),
            relation(&[2, 0], &[&[3, 1]]),
        ];
        let tree = gyo_join_tree(&relations);
        assert!(!tree.acyclic);
        assert_eq!(tree.residual.len(), 3);
        assert!(tree.edges.len() < 3);
    }

    #[test]
    fn gyo_star_query_is_acyclic_with_a_spanning_join_tree() {
        // Center C(a,b,c) joined to a leaf on each variable, α-acyclic. Each
        // leaf shares one variable with the center, so GYO yields a spanning
        // join tree on all four relations (3 edges). The exact rooting is not
        // fixed: once two leaves are peeled the center itself can become an ear
        // of the last leaf, and every valid GYO tree is still a full reducer.
        let relations = [
            relation(&[0, 1, 2], &[&[1, 2, 3]]),
            relation(&[0, 3], &[&[1, 4]]),
            relation(&[1, 4], &[&[2, 5]]),
            relation(&[2, 5], &[&[3, 6]]),
        ];
        let tree = gyo_join_tree(&relations);
        assert!(tree.acyclic);
        assert!(tree.residual.is_empty());
        assert_eq!(
            tree.edges.len(),
            3,
            "spanning tree on 4 relations has 3 edges"
        );
        // Every non-root relation is a child exactly once, and each child is
        // removed before its witness parent.
        let position = ear_positions(&tree);
        let mut children = tree
            .edges
            .iter()
            .map(|&(child, _)| child)
            .collect::<Vec<_>>();
        children.sort_unstable();
        children.dedup();
        assert_eq!(children.len(), 3, "three of four relations are children");
        for &(child, parent) in tree.edges.iter() {
            assert!(position[child] < position[parent]);
        }
    }

    #[test]
    fn gyo_disconnected_relations_form_a_forest_without_edges() {
        // R(a,b) and S(c,d) share nothing: two roots, no join-tree edges, but
        // still α-acyclic (a Cartesian product of two trivial trees).
        let relations = [relation(&[0, 1], &[&[1, 2]]), relation(&[2, 3], &[&[3, 4]])];
        let tree = gyo_join_tree(&relations);
        assert!(tree.acyclic);
        assert!(tree.edges.is_empty());
        assert_eq!(tree.ear_order.len(), 2);
    }

    #[test]
    fn gyo_full_reducer_preserves_acyclic_join_output() {
        // Yannakakis' correctness property: applying the GYO-tree full reducer
        // and then joining the reduced relations yields the SAME binding set as
        // joining the originals. The reducer only deletes dangling tuples, which
        // contribute nothing under MM2 set semantics. `b=9` and `b=8` dangle.
        let relations = [
            relation(&[0, 1], &[&[1, 2], &[1, 9]]),
            relation(&[1, 2], &[&[2, 3], &[8, 3]]),
            relation(&[2, 3], &[&[3, 4]]),
        ];
        let order = [v(0), v(1), v(2), v(3)];
        let tree = gyo_join_tree(&relations);
        assert!(tree.acyclic);

        let reduced = semijoin_reduce_presence(&relations, &tree.edges).unwrap();
        let direct = generic_join(&relations, &order).unwrap();
        let via_reducer = generic_join(&reduced.relations, &order).unwrap();

        assert_eq!(positive_row_set(&direct), positive_row_set(&via_reducer));
        assert_eq!(direct, via_reducer);
        // The dangling tuples were actually pruned (otherwise this test would
        // not exercise the reducer at all).
        assert!(reduced.removed_rows > 0);
    }

    #[test]
    fn gyo_full_reducer_matches_generic_join_across_factor_orders() {
        // The full-reduced join must equal the direct join for every input
        // ordering of an acyclic body (the ear-removal order is internal).
        let base = [
            relation(&[0, 1], &[&[1, 2], &[1, 5], &[7, 2]]),
            relation(&[1, 2], &[&[2, 3], &[5, 9]]),
            relation(&[2, 3], &[&[3, 4], &[9, 9]]),
        ];
        let order = [v(0), v(1), v(2), v(3)];
        let expected = positive_row_set(&generic_join(&base, &order).unwrap());
        for permutation in [[0, 1, 2], [2, 1, 0], [1, 0, 2], [1, 2, 0]] {
            let relations = permutation
                .iter()
                .map(|&i| base[i].clone())
                .collect::<Vec<_>>();
            let tree = gyo_join_tree(&relations);
            assert!(
                tree.acyclic,
                "permutation {permutation:?} should be acyclic"
            );
            let reduced = semijoin_reduce_presence(&relations, &tree.edges).unwrap();
            let via_reducer = positive_row_set(&generic_join(&reduced.relations, &order).unwrap());
            assert_eq!(expected, via_reducer, "permutation {permutation:?}");
        }
    }

    #[test]
    fn gyo_single_relation_is_trivially_acyclic() {
        let relations = [relation(&[0, 1], &[&[1, 2]])];
        let tree = gyo_join_tree(&relations);
        assert!(tree.acyclic);
        assert!(tree.edges.is_empty());
        assert_eq!(&*tree.ear_order, &[0]);
    }

    #[test]
    fn gyo_contained_relation_is_an_ear_of_its_superset() {
        // R(a,b) ⊆ S(a,b,c): R is an ear with witness S (a contained edge).
        let relations = [
            relation(&[0, 1], &[&[1, 2]]),
            relation(&[0, 1, 2], &[&[1, 2, 3]]),
        ];
        let tree = gyo_join_tree(&relations);
        assert!(tree.acyclic);
        assert_eq!(&*tree.edges, &[(0, 1)]);
    }

    fn assert_decomposition_partitions(
        decomposition: &HyperTreeDecomposition,
        relation_count: usize,
    ) {
        let mut covered = decomposition
            .bags
            .iter()
            .flat_map(|bag| bag.relations.iter().copied())
            .collect::<Vec<_>>();
        covered.sort_unstable();
        assert_eq!(covered, (0..relation_count).collect::<Vec<_>>());
    }

    fn assert_bag_var_sets_acyclic(decomposition: &HyperTreeDecomposition) {
        let bag_relations = decomposition
            .bags
            .iter()
            .map(|bag| BindingRelation::new(bag.variables.to_vec()))
            .collect::<Vec<_>>();
        assert!(gyo_join_tree(&bag_relations).acyclic);
    }

    #[test]
    fn hypertree_decomposition_finds_width_two_for_triangle() {
        // R(a,b), S(b,c), T(c,a): ghw 2. A width-2 partition groups two of the
        // three edges into one bag, leaving the third as the second bag.
        let relations = [
            relation(&[0, 1], &[&[1, 2]]),
            relation(&[1, 2], &[&[2, 3]]),
            relation(&[2, 0], &[&[3, 1]]),
        ];
        let decomposition = hypertree_decomposition(&relations, 3).unwrap();
        assert_eq!(decomposition.width, 2);
        assert_decomposition_partitions(&decomposition, relations.len());
        assert_bag_var_sets_acyclic(&decomposition);
    }

    #[test]
    fn hypertree_decomposition_finds_width_two_for_four_cycle() {
        // R(a,b), S(b,c), T(c,d), U(d,a): ghw 2.
        let relations = [
            relation(&[0, 1], &[&[1, 2]]),
            relation(&[1, 2], &[&[2, 3]]),
            relation(&[2, 3], &[&[3, 4]]),
            relation(&[3, 0], &[&[4, 1]]),
        ];
        let decomposition = hypertree_decomposition(&relations, 3).unwrap();
        assert_eq!(decomposition.width, 2);
        assert_decomposition_partitions(&decomposition, relations.len());
        assert_bag_var_sets_acyclic(&decomposition);
    }

    #[test]
    fn hypertree_decomposition_is_width_one_for_acyclic_body() {
        // An acyclic chain decomposes at width 1 (all singletons = the join
        // tree). The selector handles this via AcyclicYannakakis, but the search
        // should still report width 1.
        let relations = [
            relation(&[0, 1], &[&[1, 2]]),
            relation(&[1, 2], &[&[2, 3]]),
            relation(&[2, 3], &[&[3, 4]]),
        ];
        let decomposition = hypertree_decomposition(&relations, 3).unwrap();
        assert_eq!(decomposition.width, 1);
        assert_eq!(decomposition.bags.len(), 3);
        assert_decomposition_partitions(&decomposition, relations.len());
    }

    #[test]
    fn hypertree_decomposition_rejects_cyclic_body_below_its_width() {
        let relations = [
            relation(&[0, 1], &[&[1, 2]]),
            relation(&[1, 2], &[&[2, 3]]),
            relation(&[2, 0], &[&[3, 1]]),
        ];
        // The triangle has no width-1 (alpha-acyclic) decomposition.
        assert!(hypertree_decomposition(&relations, 1).is_none());
    }

    #[test]
    fn ghd_join_matches_generic_join_on_triangle() {
        // R(a,b), S(b,c), T(c,a) over a directed 3-cycle 1->2->3->1. The closed
        // triangles are the three rotations.
        let edge: &[&[u64]] = &[&[1, 2], &[2, 3], &[3, 1]];
        let relations = [
            relation(&[0, 1], edge),
            relation(&[1, 2], edge),
            relation(&[2, 0], edge),
        ];
        let order = [v(0), v(1), v(2)];
        let decomposition = hypertree_decomposition(&relations, 3).unwrap();
        assert_eq!(decomposition.width, 2);

        let via_ghd = ghd_join(&relations, &decomposition, &order).unwrap();
        let direct = generic_join(&relations, &order).unwrap();
        assert_eq!(positive_row_set(&via_ghd), positive_row_set(&direct));
        assert_eq!(via_ghd.positive_rows().count(), 3);
        assert_eq!(
            ghd_join_count(&relations, &decomposition, &order)
                .unwrap()
                .rows,
            direct.positive_rows().count()
        );
        assert!(
            ghd_join_exists(&relations, &decomposition, &order)
                .unwrap()
                .exists
        );
    }

    #[test]
    fn ghd_join_matches_generic_join_on_four_cycle() {
        // R(a,b), S(b,c), T(c,d), U(d,a) over a directed 4-cycle 1->2->3->4->1.
        let edge: &[&[u64]] = &[&[1, 2], &[2, 3], &[3, 4], &[4, 1]];
        let relations = [
            relation(&[0, 1], edge),
            relation(&[1, 2], edge),
            relation(&[2, 3], edge),
            relation(&[3, 0], edge),
        ];
        let order = [v(0), v(1), v(2), v(3)];
        let decomposition = hypertree_decomposition(&relations, 3).unwrap();
        assert_eq!(decomposition.width, 2);

        let via_ghd = ghd_join(&relations, &decomposition, &order).unwrap();
        let direct = generic_join(&relations, &order).unwrap();
        assert_eq!(positive_row_set(&via_ghd), positive_row_set(&direct));
        assert_eq!(via_ghd.positive_rows().count(), 4);
    }

    #[test]
    fn ghd_join_matches_generic_join_with_dangling_tuples() {
        // Edges that include rows which cannot close a triangle, so the bag
        // reducer must prune them and still match the oracle.
        let edge: &[&[u64]] = &[&[1, 2], &[2, 3], &[3, 1], &[5, 6], &[2, 9]];
        let relations = [
            relation(&[0, 1], edge),
            relation(&[1, 2], edge),
            relation(&[2, 0], edge),
        ];
        let order = [v(0), v(1), v(2)];
        let decomposition = hypertree_decomposition(&relations, 3).unwrap();
        let via_ghd = ghd_join(&relations, &decomposition, &order).unwrap();
        let direct = generic_join(&relations, &order).unwrap();
        assert_eq!(positive_row_set(&via_ghd), positive_row_set(&direct));
    }

    #[test]
    fn min_edge_cover_counts_relations_to_cover_all_variables() {
        // Triangle: 3 edges over 3 variables, 2 edges cover all.
        let triangle = [
            relation(&[0, 1], &[&[1, 2]]),
            relation(&[1, 2], &[&[2, 3]]),
            relation(&[2, 0], &[&[3, 1]]),
        ];
        assert_eq!(min_edge_cover(&triangle), 2);

        // 4-cycle: 4 edges over 4 variables, 2 opposite edges cover all.
        let four_cycle = [
            relation(&[0, 1], &[&[1, 2]]),
            relation(&[1, 2], &[&[2, 3]]),
            relation(&[2, 3], &[&[3, 4]]),
            relation(&[3, 0], &[&[4, 1]]),
        ];
        assert_eq!(min_edge_cover(&four_cycle), 2);

        // A single relation covers itself.
        assert_eq!(min_edge_cover(&[relation(&[0, 1], &[&[1, 2]])]), 1);

        // Two relations with disjoint variables both required.
        let disjoint = [relation(&[0, 1], &[&[1, 2]]), relation(&[2, 3], &[&[3, 4]])];
        assert_eq!(min_edge_cover(&disjoint), 2);

        assert_eq!(min_edge_cover(&[]), 0);
    }

    #[test]
    fn selectivity_variable_order_puts_small_domains_first() {
        // var 0 has one distinct value, var 2 has two, var 1 has three.
        let r = relation(&[0, 1], &[&[9, 1], &[9, 2], &[9, 3]]);
        let s = relation(&[2, 0], &[&[5, 9], &[6, 9]]);
        assert_eq!(selectivity_variable_order(&[r, s]), vec![v(0), v(2), v(1)]);
    }

    #[test]
    fn dp_variable_order_is_a_valid_permutation_and_optimal() {
        // R(0,1) is wide on variable 0; S(1,2) and T(0,2) are small. A triangle.
        let r = relation(
            &[0, 1],
            &[&[0, 0], &[1, 0], &[2, 0], &[3, 1], &[4, 1], &[5, 1]],
        );
        let s = relation(&[1, 2], &[&[0, 0], &[1, 1]]);
        let t = relation(&[0, 2], &[&[0, 0], &[1, 1]]);
        let relations = [r, s, t];

        let order = dp_variable_order(&relations);
        let mut distinct = order.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct,
            vec![v(0), v(1), v(2)],
            "a valid permutation of the variables"
        );

        // The DP minimizes the sum of prefix AGM bounds, so it is no worse than
        // the greedy selectivity order by that objective.
        let all_vars = vec![v(0), v(1), v(2)];
        let order_cost = |order: &[BindingVar]| -> u128 {
            let mut mask = 0u32;
            let mut total = 0u128;
            for &var in order {
                let index = all_vars.iter().position(|&x| x == var).unwrap();
                mask |= 1u32 << index;
                total = total.saturating_add(agm_for_mask(&relations, &all_vars, mask));
            }
            total
        };
        let greedy = selectivity_variable_order(&relations);
        assert!(
            order_cost(&order) <= order_cost(&greedy),
            "the DP order is no worse than greedy"
        );
    }

    #[test]
    fn cover_cost_table_ranks_subsets_like_exact_agm() {
        // The log-domain table must induce the same ordering over variable
        // subsets as the exact u128-product `agm_for_mask`, since the DP only
        // uses the cost to compare orders. Relations small enough that the exact
        // product never saturates, so the comparison is meaningful.
        let r = relation(&[0, 1], &[&[0, 0], &[1, 0], &[2, 1], &[3, 1], &[4, 2]]);
        let s = relation(&[1, 2], &[&[0, 0], &[1, 1]]);
        let t = relation(&[0, 2], &[&[0, 0], &[1, 1], &[2, 2]]);
        let relations = [r, s, t];
        let all_vars = vec![v(0), v(1), v(2)];

        let table = cover_cost_table(&relations, &all_vars);
        let full = 1u32 << all_vars.len();
        for a in 0..full {
            for b in 0..full {
                let exact_a = agm_for_mask(&relations, &all_vars, a);
                let exact_b = agm_for_mask(&relations, &all_vars, b);
                // Both subsets are coverable here, so neither cost is the sentinel.
                assert!(table[a as usize] < COVER_INF && table[b as usize] < COVER_INF);
                assert_eq!(
                    table[a as usize].cmp(&table[b as usize]),
                    exact_a.cmp(&exact_b),
                    "log-domain rank disagrees with exact AGM for masks {a:#b} vs {b:#b}",
                );
            }
        }
        // The empty subset costs nothing and the full subset is the most costly.
        assert_eq!(table[0], 0);
        assert_eq!(
            table.iter().copied().max(),
            Some(table[(full - 1) as usize])
        );
    }

    #[test]
    fn cover_cost_table_marks_uncoverable_subsets() {
        // A variable touched by no relation can never be covered, so its subsets
        // keep the sentinel cost while the reachable ones do not.
        let only = relation(&[0], &[&[0], &[1]]);
        let all_vars = vec![v(0), v(1)];
        let table = cover_cost_table(&[only], &all_vars);
        assert_eq!(table[0b00], 0); // empty subset: no cover needed.
        assert!(table[0b01] < COVER_INF); // {v0}: covered by the single relation.
        assert_eq!(table[0b10], COVER_INF); // {v1}: no relation mentions it.
        assert_eq!(table[0b11], COVER_INF); // {v0,v1}: still uncoverable.
    }

    #[test]
    fn agm_size_bound_and_ghd_cost_use_relation_sizes() {
        // Triangle, each edge 4 rows: a 2-edge cover bounds the output by 4*4=16.
        let edge: &[&[u64]] = &[&[1, 2], &[2, 3], &[3, 4], &[4, 1]];
        let triangle = [
            relation(&[0, 1], edge),
            relation(&[1, 2], edge),
            relation(&[2, 0], edge),
        ];
        assert_eq!(agm_size_bound(&triangle), 16);
        let decomposition = hypertree_decomposition(&triangle, 3).unwrap();
        // The triangle bag merges two 4-row edges: 4*4 = 16, equal to the bound,
        // so a bare triangle has no decomposition advantage.
        assert_eq!(ghd_size_cost(&triangle, &decomposition), 16);

        // Two disjoint relations of sizes 3 and 2: both needed, product 6.
        let disjoint = [
            relation(&[0, 1], &[&[1, 1], &[2, 2], &[3, 3]]),
            relation(&[2, 3], &[&[1, 1], &[2, 2]]),
        ];
        assert_eq!(agm_size_bound(&disjoint), 6);
    }

    #[test]
    fn fractional_agm_bound_is_tighter_than_integral() {
        let edge: &[&[u64]] = &[&[1, 2], &[2, 3], &[3, 4], &[4, 1]];
        let triangle = [
            relation(&[0, 1], edge),
            relation(&[1, 2], edge),
            relation(&[2, 0], edge),
        ];
        // Integral bound is a 2-edge cover, 4*4 = 16; the fractional bound is the
        // half-half-half cover 4^(3/2) = 8, the true AGM bound.
        assert_eq!(agm_size_bound(&triangle), 16);
        assert!(
            (fractional_agm_bound(&triangle) - 8.0).abs() < 1e-6,
            "triangle fractional AGM is 8, got {}",
            fractional_agm_bound(&triangle)
        );

        // An acyclic two-edge path needs both edges: fractional equals integral 16.
        let path = [relation(&[0, 1], edge), relation(&[1, 2], edge)];
        assert!((fractional_agm_bound(&path) - 16.0).abs() < 1e-6);

        // An empty relation makes the join empty.
        let no_rows: &[&[u64]] = &[];
        let empty = [relation(&[0, 1], edge), relation(&[1, 2], no_rows)];
        assert_eq!(fractional_agm_bound(&empty), 0.0);
    }

    #[test]
    fn generic_join_materializes_rows_in_variable_order() {
        let relation = relation(&[0, 1, 2, 3, 4], &[&[10, 11, 12, 13, 14]]);
        let variable_order = [v(4), v(0), v(3), v(1), v(2)];

        let joined = generic_join(&[relation], &variable_order).unwrap();

        assert_eq!(joined.weight(&[t(14), t(10), t(13), t(11), t(12)]), 1);
    }

    #[test]
    fn generic_join_aggregates_match_materialized_join_without_rows() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10], &[3, 20]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101], &[30, 300]]);
        let variable_order = [v(1), v(0), v(2)];

        let materialized = generic_join(&[left.clone(), right.clone()], &variable_order).unwrap();
        let count = generic_join_count(&[left.clone(), right.clone()], &variable_order).unwrap();
        let exists = generic_join_exists(&[left.clone(), right.clone()], &variable_order).unwrap();

        assert_eq!(count.rows, materialized.positive_rows().count());
        assert!(exists.exists);

        let disjoint_right = relation(&[1, 2], &[&[30, 300]]);
        let empty = generic_join_exists(&[left, disjoint_right], &variable_order).unwrap();
        assert!(!empty.exists);
    }

    #[test]
    fn generic_join_aggregates_ignore_non_visible_signed_rows() {
        let left = signed_relation(&[0, 1], &[(&[1, 10], -1)]);
        let right = relation(&[1, 2], &[&[10, 100]]);
        let variable_order = [v(1), v(0), v(2)];

        let count = generic_join_count(&[left.clone(), right.clone()], &variable_order).unwrap();
        let exists = generic_join_exists(&[left.clone(), right.clone()], &variable_order).unwrap();
        let materialized = generic_join(&[left, right], &variable_order).unwrap();

        assert_eq!(count.rows, 0);
        assert!(!exists.exists);
        assert_eq!(materialized.positive_rows().count(), 0);
    }

    #[test]
    fn generic_domains_into_reuses_scratch_without_stale_domains() {
        let left = relation(&[0, 1], &[&[2, 10], &[1, 10], &[1, 20]]);
        let right = relation(&[1, 2], &[&[10, 100], &[30, 300]]);
        let mut binding = BindingAssignment::default();
        let mut domains = Vec::new();
        let mut bound_filters = Vec::new();

        let count = generic_domains_into(
            &[left.clone(), right.clone()],
            v(1),
            &binding,
            &mut domains,
            &mut bound_filters,
        );
        assert_eq!(count, 2);
        assert_eq!(domains[0], vec![t(10), t(20)]);
        assert_eq!(domains[1], vec![t(10), t(30)]);

        binding.insert(v(1), t(10));
        let count = generic_domains_into(&[left], v(0), &binding, &mut domains, &mut bound_filters);
        assert_eq!(count, 1);
        assert_eq!(domains[0], vec![t(1), t(2)]);
        assert!(domains[1].is_empty());
        assert!(bound_filters.is_empty());
    }

    #[test]
    fn generic_domains_into_reuses_bound_filter_scratch_without_stale_filters() {
        let relation = relation(&[0, 1, 2], &[&[1, 10, 100], &[2, 20, 100], &[3, 20, 200]]);
        let mut binding = BindingAssignment::default();
        let mut domains = Vec::new();
        let mut bound_filters = Vec::new();

        binding.insert(v(2), t(100));
        let count = generic_domains_into(
            std::slice::from_ref(&relation),
            v(1),
            &binding,
            &mut domains,
            &mut bound_filters,
        );
        assert_eq!(count, 1);
        assert_eq!(domains[0], vec![t(10), t(20)]);
        assert!(bound_filters.is_empty());

        binding.remove(v(2));
        let count = generic_domains_into(
            &[relation],
            v(0),
            &binding,
            &mut domains,
            &mut bound_filters,
        );
        assert_eq!(count, 1);
        assert_eq!(domains[0], vec![t(1), t(2), t(3)]);
        assert!(bound_filters.is_empty());
    }

    #[test]
    fn generic_join_count_reports_cardinality_overflow() {
        let mut rows = usize::MAX;
        let mut binding = BindingAssignment::default();
        let mut scratch = Vec::new();
        let result = generic_join_aggregate_recurse(
            &[],
            &[],
            &[],
            &mut binding,
            &mut rows,
            false,
            &mut scratch,
        );

        assert_eq!(result, Err(BindingRelationError::CardinalityOverflow));
        assert_eq!(rows, usize::MAX);
    }

    #[test]
    fn trie_join_matches_generic_join_with_index_counters() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10], &[3, 20]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101], &[30, 300]]);

        let generic = generic_join(&[left.clone(), right.clone()], &[v(1), v(0), v(2)]).unwrap();
        let trie = trie_join(&[left, right], &[v(1), v(0), v(2)]).unwrap();

        assert_eq!(trie.relation, generic);
        assert_eq!(
            trie.stats,
            TrieJoinStats {
                relation_indexes: 2,
                indexed_rows: 6,
                trie_nodes: 6,
                domain_intersections: 4,
                domain_sources: 5,
                domain_values: 10,
                domain_cursor_opens: 2,
                domain_cursor_seeks: 5,
                domain_cursor_skips: 2,
                domain_cursor_nexts: 1,
                weight_lookups: 8,
                output_rows: 4,
            }
        );
    }

    #[test]
    fn trie_join_trace_matches_generic_model_on_seeded_small_queries() {
        fn seeded_relation(seed: u64, schema: &[u8]) -> BindingRelation {
            let mut relation =
                BindingRelation::new(schema.iter().copied().map(BindingVar).collect::<Vec<_>>());
            let mut seen = BTreeSet::new();

            for row_seed in 0_u64..10 {
                let row = schema
                    .iter()
                    .map(|&variable| {
                        let value = 1
                            + ((seed
                                .wrapping_mul(13)
                                .wrapping_add(row_seed.wrapping_mul(7))
                                .wrapping_add(u64::from(variable).wrapping_mul(11)))
                                % 5);
                        t(u64::from(variable) * 100 + value)
                    })
                    .collect::<Vec<_>>();

                if seen.insert(row.clone()) {
                    relation.add(row, 1).unwrap();
                }
            }

            relation
        }

        let orders = [[v(0), v(1), v(2)], [v(1), v(0), v(2)], [v(2), v(1), v(0)]];

        for seed in 0_u64..16 {
            let relations = [
                seeded_relation(seed, &[0, 1]),
                seeded_relation(seed + 1, &[1, 2]),
                seeded_relation(seed + 2, &[0, 2]),
            ];

            for variable_order in orders {
                let generic = generic_join(&relations, &variable_order).unwrap();
                let trie = trie_join(&relations, &variable_order).unwrap();
                let count = trie_join_count(&relations, &variable_order).unwrap();
                let exists = trie_join_exists(&relations, &variable_order).unwrap();
                let trace = trie_join_trace(&relations, &variable_order).unwrap();
                let summary = trace.summarize().unwrap();
                let replay = trace.replay_shape().unwrap();
                let expected_rows = generic.positive_rows().count();

                assert_eq!(trie.relation, generic);
                assert_eq!(count.rows, expected_rows);
                assert_eq!(exists.exists, expected_rows > 0);
                assert_eq!(trace.candidate_bindings, expected_rows);
                assert_eq!(summary.candidate_bindings, expected_rows);
                assert_eq!(replay.summary, summary);

                for step in trace.steps.iter() {
                    let expected_cursor_opens = if step.relation_domain_lens.contains(&0)
                        || step.participating_relations.len() <= 1
                    {
                        0
                    } else {
                        step.participating_relations.len()
                    };
                    assert_eq!(step.cursor_opens, expected_cursor_opens);

                    if step.participating_relations.len() == 1 {
                        assert_eq!(
                            step.intersection.as_ref(),
                            step.relation_domains[0].values.as_ref()
                        );
                        assert_eq!(step.cursor_seeks, 0);
                        assert_eq!(step.cursor_skips, 0);
                        assert_eq!(step.cursor_nexts, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn relation_trie_index_uses_compact_prefix_keys() {
        let relation = relation(&[0, 1, 2], &[&[1, 10, 100], &[1, 20, 200], &[2, 10, 300]]);
        let index = RelationTrieIndex::build(&relation, &[v(0), v(1), v(2)]).unwrap();

        assert!(
            index
                .child_prefix_keys
                .binary_search(&NaturalJoinKey::Empty)
                .is_ok()
        );
        assert!(
            index
                .child_prefix_keys
                .binary_search(&NaturalJoinKey::One(t(1)))
                .is_ok()
        );
        assert!(
            index
                .child_prefix_keys
                .binary_search(&NaturalJoinKey::Two(t(1), t(10)))
                .is_ok()
        );
        assert_eq!(
            index.child_prefix_keys.len(),
            index.child_range_starts.len()
        );
        assert_eq!(index.child_prefix_keys.len(), index.child_range_ends.len());
        assert!(
            index
                .child_prefix_keys
                .windows(2)
                .all(|window| window[0] < window[1])
        );
        assert_eq!(
            index.child_values.as_ref(),
            &[t(1), t(2), t(10), t(20), t(10), t(100), t(200), t(300)]
        );
        assert_eq!(index.weight_keys.len(), index.weight_values.len());
        assert!(
            index
                .weight_keys
                .windows(2)
                .all(|window| window[0] < window[1])
        );
        assert!(
            index
                .weight_keys
                .iter()
                .any(|key| key == &NaturalJoinKey::Many(vec![t(1), t(10), t(100)].into()))
        );
        assert_eq!(index.weight_values.as_ref(), &[1, 1, 1],);

        let mut binding = BindingAssignment::default();
        assert_eq!(index.domain(v(0), &binding), Some(&[t(1), t(2)][..]));
        binding.insert(v(0), t(1));
        assert_eq!(index.domain(v(1), &binding), Some(&[t(10), t(20)][..]));
        binding.insert(v(1), t(10));
        assert_eq!(index.domain(v(2), &binding), Some(&[t(100)][..]));
        binding.insert(v(2), t(100));
        assert_eq!(index.weight(&binding), 1);

        let reordered = RelationTrieIndex::build(&relation, &[v(2), v(0), v(1)]).unwrap();
        assert!(
            reordered
                .child_prefix_keys
                .binary_search(&NaturalJoinKey::One(t(100)))
                .is_ok()
        );
        assert!(
            reordered
                .weight_keys
                .iter()
                .any(|key| key == &NaturalJoinKey::Many(vec![t(100), t(1), t(10)].into()))
        );

        let mut reordered_binding = BindingAssignment::default();
        assert_eq!(
            reordered.domain(v(2), &reordered_binding),
            Some(&[t(100), t(200), t(300)][..])
        );
        reordered_binding.insert(v(2), t(100));
        assert_eq!(
            reordered.domain(v(0), &reordered_binding),
            Some(&[t(1)][..])
        );
        reordered_binding.insert(v(0), t(1));
        assert_eq!(
            reordered.domain(v(1), &reordered_binding),
            Some(&[t(10)][..])
        );
        reordered_binding.insert(v(1), t(10));
        assert_eq!(reordered.weight(&reordered_binding), 1);
    }

    #[test]
    fn relation_trie_index_bulk_builds_sorted_unique_domains() {
        let relation = relation(
            &[0, 1, 2],
            &[&[2, 20, 100], &[1, 10, 100], &[1, 10, 200], &[1, 20, 300]],
        );
        let index = RelationTrieIndex::build(&relation, &[v(0), v(1), v(2)]).unwrap();
        let mut binding = BindingAssignment::default();

        assert_eq!(index.domain(v(0), &binding), Some(&[t(1), t(2)][..]));
        binding.insert(v(0), t(1));
        assert_eq!(index.domain(v(1), &binding), Some(&[t(10), t(20)][..]));
        binding.insert(v(1), t(10));
        assert_eq!(index.domain(v(2), &binding), Some(&[t(100), t(200)][..]));
    }

    #[test]
    fn relation_trie_index_uses_dense_variable_positions() {
        let relation = relation(&[2, 0], &[&[20, 1], &[20, 2], &[30, 1]]);
        let index = RelationTrieIndex::build(&relation, &[v(0), v(2)]).unwrap();

        assert_eq!(index.positions[usize::from(v(0).0)], 0);
        assert_eq!(index.positions[usize::from(v(2).0)], 1);
        assert_eq!(index.positions[usize::from(v(9).0)], MISSING_TRIE_POSITION);

        let mut binding = BindingAssignment::default();
        assert_eq!(index.domain(v(0), &binding), Some(&[t(1), t(2)][..]));
        assert_eq!(index.domain(v(9), &binding), None);
        binding.insert(v(0), t(1));
        assert_eq!(index.domain(v(2), &binding), Some(&[t(20), t(30)][..]));
    }

    #[test]
    fn relation_trie_index_domains_match_bruteforce_model() {
        for relation_seed in 0_u64..18 {
            let mut relation = BindingRelation::new(vec![v(7), v(3), v(11)]);
            for row_seed in 0_u64..9 {
                let row = [
                    t(1 + ((relation_seed + row_seed * 2) % 5)),
                    t(10 + ((relation_seed * 3 + row_seed) % 7)),
                    t(20 + ((relation_seed + row_seed * row_seed) % 6)),
                ];
                let weight = if (relation_seed + row_seed) % 5 == 0 {
                    -1
                } else {
                    1
                };
                relation.add(row.to_vec(), weight).unwrap();
            }

            let orders = [
                [v(7), v(3), v(11)],
                [v(3), v(11), v(7)],
                [v(11), v(7), v(3)],
            ];
            for order in orders {
                let index = RelationTrieIndex::build(&relation, &order).unwrap();
                for depth in 0..order.len() {
                    let mut binding = BindingAssignment::default();
                    for &prefix_variable in &order[..depth] {
                        let source_index = relation
                            .schema()
                            .iter()
                            .position(|&candidate| candidate == prefix_variable)
                            .unwrap();
                        let seed_row = relation
                            .positive_rows()
                            .next()
                            .expect("generated relation keeps a visible positive row");
                        binding.insert(prefix_variable, seed_row[source_index]);
                    }

                    let expected =
                        brute_force_domain(&relation, &order, order[depth], &binding).unwrap();
                    assert_eq!(
                        index.domain(order[depth], &binding),
                        Some(expected.as_slice())
                    );
                }
            }
        }
    }

    #[test]
    fn relation_trie_index_packs_weight_values_in_sorted_key_order() {
        let relation = signed_relation(&[0], &[(&[2], 5), (&[1], 7), (&[3], 11)]);
        let index = RelationTrieIndex::build(&relation, &[v(0)]).unwrap();

        assert_eq!(
            index.weight_keys.as_ref(),
            &[
                NaturalJoinKey::One(t(1)),
                NaturalJoinKey::One(t(2)),
                NaturalJoinKey::One(t(3))
            ]
        );
        assert_eq!(index.weight_values.as_ref(), &[7, 5, 11]);

        let mut binding = BindingAssignment::default();
        binding.insert(v(0), t(2));
        assert_eq!(index.weight(&binding), 5);
        binding.insert(v(0), t(9));
        assert_eq!(index.weight(&binding), 0);
    }

    #[test]
    fn relation_trie_index_packs_prefix_ranges_in_sorted_key_order() {
        let relation = relation(&[0, 1], &[&[2, 20], &[1, 10], &[1, 30]]);
        let index = RelationTrieIndex::build(&relation, &[v(0), v(1)]).unwrap();

        assert_eq!(
            index.child_prefix_keys.as_ref(),
            &[
                NaturalJoinKey::Empty,
                NaturalJoinKey::One(t(1)),
                NaturalJoinKey::One(t(2))
            ]
        );
        assert_eq!(
            index.child_values.as_ref(),
            &[t(1), t(2), t(10), t(30), t(20)]
        );
        assert_eq!(index.child_range_starts.as_ref(), &[0, 2, 4]);
        assert_eq!(index.child_range_ends.as_ref(), &[2, 4, 5]);

        let mut binding = BindingAssignment::default();
        assert_eq!(index.domain(v(0), &binding), Some(&[t(1), t(2)][..]));
        binding.insert(v(0), t(1));
        assert_eq!(index.domain(v(1), &binding), Some(&[t(10), t(30)][..]));
        binding.insert(v(0), t(9));
        assert_eq!(index.domain(v(1), &binding), Some(&[][..]));
    }

    #[test]
    fn relation_trie_index_searches_wide_bound_prefixes_without_materialized_keys() {
        let relation = relation(
            &[0, 1, 2, 3],
            &[
                &[1, 10, 100, 1000],
                &[1, 10, 100, 1001],
                &[1, 10, 200, 2000],
                &[2, 20, 300, 3000],
            ],
        );
        let index = RelationTrieIndex::build(&relation, &[v(0), v(1), v(2), v(3)]).unwrap();
        let mut binding = BindingAssignment::default();

        binding.insert(v(0), t(1));
        binding.insert(v(1), t(10));
        binding.insert(v(2), t(100));
        assert_eq!(index.domain(v(3), &binding), Some(&[t(1000), t(1001)][..]));

        binding.insert(v(3), t(1001));
        assert_eq!(index.weight(&binding), 1);

        binding.insert(v(2), t(999));
        assert_eq!(index.domain(v(3), &binding), Some(&[][..]));
        binding.insert(v(3), t(1001));
        assert_eq!(index.weight(&binding), 0);
    }

    #[test]
    fn relation_trie_index_reuses_monotone_domain_and_weight_hints() {
        let relation = relation(
            &[0, 1, 2, 3],
            &[
                &[1, 10, 100, 1000],
                &[1, 10, 100, 1001],
                &[1, 10, 200, 2000],
                &[2, 20, 300, 3000],
            ],
        );
        let index = RelationTrieIndex::build(&relation, &[v(0), v(1), v(2), v(3)]).unwrap();
        let mut binding = BindingAssignment::default();
        let mut domain_hint = 0;

        binding.insert(v(0), t(1));
        binding.insert(v(1), t(10));
        binding.insert(v(2), t(100));
        assert_eq!(
            index.domain_with_hint(v(3), &binding, &mut domain_hint),
            Some(&[t(1000), t(1001)][..])
        );
        let first_domain_hint = domain_hint;

        binding.insert(v(2), t(200));
        assert_eq!(
            index.domain_with_hint(v(3), &binding, &mut domain_hint),
            Some(&[t(2000)][..])
        );
        assert!(domain_hint >= first_domain_hint);

        let mut weight_hint = 0;
        binding.insert(v(2), t(100));
        binding.insert(v(3), t(1000));
        assert_eq!(index.weight_with_hint(&binding, &mut weight_hint), 1);
        let first_weight_hint = weight_hint;

        binding.insert(v(3), t(1001));
        assert_eq!(index.weight_with_hint(&binding, &mut weight_hint), 1);
        assert!(weight_hint >= first_weight_hint);

        binding.insert(v(3), t(999));
        assert_eq!(index.weight_with_hint(&binding, &mut weight_hint), 0);
    }

    #[test]
    fn trie_join_count_matches_materialized_join_without_rows() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10], &[3, 20]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101], &[30, 300]]);
        let variable_order = [v(1), v(0), v(2)];

        let trie = trie_join(&[left.clone(), right.clone()], &variable_order).unwrap();
        let count = trie_join_count(&[left, right], &variable_order).unwrap();

        assert_eq!(count.rows, trie.relation.positive_rows().count());
        assert_eq!(count.stats.output_rows, count.rows);
        assert_eq!(count.stats.relation_indexes, trie.stats.relation_indexes);
        assert_eq!(count.stats.indexed_rows, trie.stats.indexed_rows);
        assert_eq!(
            count.stats.domain_intersections,
            trie.stats.domain_intersections
        );
        assert_eq!(count.stats.weight_lookups, trie.stats.weight_lookups);
    }

    #[test]
    fn prepared_trie_join_reuses_indexes_for_multiple_reports() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10], &[3, 20]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101], &[30, 300]]);
        let relations = [left, right];
        let variable_order = [v(1), v(0), v(2)];

        let prepared = PreparedTrieJoin::prepare(&relations, &variable_order).unwrap();
        let executed = prepared.execute().unwrap();
        let counted = prepared.count().unwrap();
        let existed = prepared.exists().unwrap();
        let traced = prepared.trace().unwrap();

        assert_eq!(prepared.variable_order(), variable_order);
        assert_eq!(counted.rows, executed.relation.positive_rows().count());
        assert!(existed.exists);
        assert_eq!(traced.candidate_bindings, counted.rows);
        assert_eq!(
            executed.stats.relation_indexes,
            counted.stats.relation_indexes
        );
        assert_eq!(executed.stats.indexed_rows, counted.stats.indexed_rows);
        assert_eq!(executed.stats.trie_nodes, counted.stats.trie_nodes);
        assert_eq!(traced.indexed_rows, counted.stats.indexed_rows);
        assert_eq!(traced.trie_nodes, counted.stats.trie_nodes);
    }

    #[test]
    fn trie_join_count_reports_cardinality_overflow() {
        let mut stats = TrieJoinStats::default();
        let mut rows = usize::MAX;
        let mut scratch = trie_join_scratch(0);
        let mut weight_hints = Vec::new();
        let result = trie_join_aggregate_recurse(
            &[],
            &[],
            0,
            &mut BindingAssignment::default(),
            &mut stats,
            &mut rows,
            false,
            &mut scratch,
            &mut weight_hints,
        );

        assert_eq!(result, Err(BindingRelationError::CardinalityOverflow));
        assert_eq!(rows, usize::MAX);
        assert_eq!(stats.weight_lookups, 0);
    }

    #[test]
    fn trie_join_domain_scratch_clears_between_prefixes() {
        let relation = relation(&[0, 1], &[&[1, 10], &[1, 20], &[2, 30], &[3, 30]]);
        let index = RelationTrieIndex::build(&relation, &[v(0), v(1)]).unwrap();
        let mut binding = BindingAssignment::default();
        let mut domains = Vec::new();
        let mut hints = Vec::new();

        let root_count = trie_domains_into(
            std::slice::from_ref(&index),
            v(0),
            &binding,
            &mut domains,
            &mut hints,
        );
        assert_eq!(root_count, 1);
        assert_eq!(domains[0], &[t(1), t(2), t(3)]);

        binding.insert(v(0), t(1));
        let child_count = trie_domains_into(
            std::slice::from_ref(&index),
            v(1),
            &binding,
            &mut domains,
            &mut hints,
        );
        assert_eq!(child_count, 1);
        assert_eq!(domains[0], &[t(10), t(20)]);

        binding.insert(v(0), t(99));
        let missing_count = trie_domains_into(
            std::slice::from_ref(&index),
            v(1),
            &binding,
            &mut domains,
            &mut hints,
        );
        assert_eq!(missing_count, 1);
        assert!(domains[0].is_empty());
    }

    #[test]
    fn trie_join_streams_singleton_domains_without_leapfrog_cursors() {
        let relation = relation(&[0, 1], &[&[1, 10], &[2, 20]]);
        let result = trie_join(&[relation], &[v(0), v(1)]).unwrap();

        assert_eq!(result.relation.positive_rows().count(), 2);
        assert_eq!(result.stats.domain_intersections, 3);
        assert_eq!(result.stats.domain_sources, 3);
        assert_eq!(result.stats.domain_values, 4);
        assert_eq!(result.stats.domain_cursor_opens, 0);
        assert_eq!(result.stats.domain_cursor_seeks, 0);
        assert_eq!(result.stats.domain_cursor_nexts, 0);
    }

    #[test]
    fn trie_join_trace_streams_singleton_domains_without_leapfrog_cursors() {
        let relation = relation(&[0, 1], &[&[1, 10], &[2, 20]]);
        let trace = trie_join_trace(&[relation], &[v(0), v(1)]).unwrap();

        assert_eq!(trace.candidate_bindings, 2);
        assert_eq!(trace.steps.len(), 3);
        assert!(trace.steps.iter().all(|step| step.domain_sources == 1));
        assert!(trace.steps.iter().all(|step| step.cursor_opens == 0));
        assert!(trace.steps.iter().all(|step| step.cursor_seeks == 0));
        assert!(trace.steps.iter().all(|step| step.cursor_skips == 0));
        assert!(trace.steps.iter().all(|step| step.cursor_nexts == 0));
        assert_eq!(trace.steps[0].intersection.as_ref(), [t(1), t(2)]);
        assert_eq!(
            trace.summarize().unwrap(),
            TrieJoinTraceSummary {
                relation_indexes: 1,
                indexed_rows: 2,
                trie_nodes: 3,
                steps: 3,
                candidate_bindings: 2,
                domain_intersections: 3,
                domain_sources: 3,
                domain_values: 4,
                intersection_values: 4,
                empty_intersections: 0,
                max_bound_prefix_len: 1,
                max_participating_relations: 1,
                max_intersection_len: 2,
                cursor_opens: 0,
                cursor_seeks: 0,
                cursor_skips: 0,
                cursor_nexts: 0,
            }
        );
    }

    #[test]
    fn trie_join_cursor_scratch_clears_between_intersections() {
        let first = [t(1), t(2)];
        let second = [t(2), t(3)];
        let empty: [TermId; 0] = [];
        let mut stats = TrieJoinStats::default();
        let mut cursors = Vec::new();

        assert_eq!(
            slice_domain_cursors_into(&[&first, &second], &mut cursors, &mut stats),
            Some(())
        );
        assert_eq!(cursors.len(), 2);

        assert_eq!(
            slice_domain_cursors_into(&[&empty], &mut cursors, &mut stats),
            None
        );
        assert!(cursors.is_empty());
        assert_eq!(stats.domain_intersections, 2);
    }

    #[test]
    fn trie_join_exists_stops_after_first_positive_witness() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10], &[3, 20]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101], &[30, 300]]);
        let variable_order = [v(1), v(0), v(2)];

        let count = trie_join_count(&[left.clone(), right.clone()], &variable_order).unwrap();
        let exists = trie_join_exists(&[left.clone(), right.clone()], &variable_order).unwrap();

        assert!(exists.exists);
        assert_eq!(exists.stats.output_rows, 1);
        assert!(exists.stats.weight_lookups < count.stats.weight_lookups);
        assert!(exists.stats.domain_intersections <= count.stats.domain_intersections);

        let disjoint_right = relation(&[1, 2], &[&[30, 300]]);
        let empty = trie_join_exists(&[left, disjoint_right], &variable_order).unwrap();
        assert!(!empty.exists);
        assert_eq!(empty.stats.output_rows, 0);
    }

    #[test]
    fn trie_join_exists_streams_domain_intersection_until_first_witness() {
        let left = unary_relation(0, 1..=100);
        let right = unary_relation(0, 1..=100);
        let variable_order = [v(0)];

        let count = trie_join_count(&[left.clone(), right.clone()], &variable_order).unwrap();
        let exists = trie_join_exists(&[left, right], &variable_order).unwrap();

        assert_eq!(count.rows, 100);
        assert!(exists.exists);
        assert_eq!(exists.stats.output_rows, 1);
        assert_eq!(exists.stats.weight_lookups, 2);
        assert_eq!(exists.stats.domain_cursor_nexts, 1);
        assert!(exists.stats.domain_cursor_nexts < count.stats.domain_cursor_nexts);
        assert!(exists.stats.domain_cursor_seeks < count.stats.domain_cursor_seeks);
    }

    #[test]
    fn trie_join_aggregates_ignore_non_visible_signed_rows() {
        let left = signed_relation(&[0, 1], &[(&[1, 10], -1)]);
        let right = relation(&[1, 2], &[&[10, 100]]);
        let variable_order = [v(1), v(0), v(2)];

        let count = trie_join_count(&[left.clone(), right.clone()], &variable_order).unwrap();
        let exists = trie_join_exists(&[left.clone(), right.clone()], &variable_order).unwrap();
        let materialized = trie_join(&[left, right], &variable_order).unwrap();

        assert_eq!(count.rows, 0);
        assert!(!exists.exists);
        assert_eq!(materialized.relation.positive_rows().count(), 0);
    }

    #[test]
    fn trie_join_trace_records_variable_depth_domain_contexts_without_materializing() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10], &[3, 20]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101], &[30, 300]]);
        let variable_order = [v(1), v(0), v(2)];

        let trace = trie_join_trace(&[left.clone(), right.clone()], &variable_order).unwrap();
        let trie = trie_join(&[left, right], &variable_order).unwrap();

        assert_eq!(trace.variable_order.as_ref(), variable_order);
        assert_eq!(
            trace.candidate_bindings,
            trie.relation.positive_rows().count()
        );
        assert_eq!(trace.steps.len(), 4);
        assert_eq!(trace.relation_indexes, 2);
        assert_eq!(trace.indexed_rows, 6);
        assert_eq!(trace.trie_nodes, 6);
        assert_eq!(
            trace.summarize().unwrap(),
            TrieJoinTraceSummary {
                relation_indexes: 2,
                indexed_rows: 6,
                trie_nodes: 6,
                steps: 4,
                candidate_bindings: 4,
                domain_intersections: 4,
                domain_sources: 5,
                domain_values: 10,
                intersection_values: 7,
                empty_intersections: 0,
                max_bound_prefix_len: 2,
                max_participating_relations: 2,
                max_intersection_len: 2,
                cursor_opens: 2,
                cursor_seeks: 5,
                cursor_skips: 2,
                cursor_nexts: 1,
            }
        );

        let root = &trace.steps[0];
        assert_eq!(root.level, 0);
        assert_eq!(root.variable, v(1));
        assert!(root.bound_prefix.is_empty());
        assert_eq!(root.participating_relations.as_ref(), [0, 1]);
        assert_eq!(root.relation_domain_lens.as_ref(), [2, 2]);
        assert_eq!(root.relation_domains.len(), 2);
        assert_eq!(root.relation_domains[0].relation_index, 0);
        assert_eq!(root.relation_domains[0].values.as_ref(), [t(10), t(20)]);
        assert_eq!(root.relation_domains[1].relation_index, 1);
        assert_eq!(root.relation_domains[1].values.as_ref(), [t(10), t(30)]);
        assert_eq!(root.domain_sources, 2);
        assert_eq!(root.domain_values, 4);
        assert_eq!(root.intersection.as_ref(), [t(10)]);
        assert_eq!(root.cursor_opens, 2);
        assert!(root.cursor_seeks >= root.cursor_opens);
        assert_eq!(root.cursor_skips, 2);
        assert_eq!(root.cursor_nexts, 1);

        assert_eq!(
            trace
                .steps
                .iter()
                .filter(|step| step.variable == v(2))
                .count(),
            2
        );
        assert!(
            trace
                .steps
                .iter()
                .filter(|step| step.variable == v(2))
                .all(|step| step.bound_prefix.len() == 2
                    && step.participating_relations.as_ref() == [1]
                    && step.relation_domain_lens.as_ref() == [2]
                    && step.intersection.as_ref() == [t(100), t(101)])
        );

        let replay = trace.replay_shape().unwrap();
        assert_eq!(replay.summary, trace.summarize().unwrap());
        let root = replay.root.as_ref().unwrap();
        assert_eq!(root.step_index, 0);
        assert_eq!(root.variable, v(1));
        assert_eq!(root.intersection.as_ref(), [t(10)]);
        assert_eq!(root.branches.len(), 1);
        assert_eq!(root.branches[0].value, t(10));

        let x_node = root.branches[0].child.as_ref().unwrap();
        assert_eq!(x_node.step_index, 1);
        assert_eq!(x_node.variable, v(0));
        assert_eq!(x_node.bound_prefix.as_ref(), [t(10)]);
        assert_eq!(x_node.intersection.as_ref(), [t(1), t(2)]);
        assert_eq!(x_node.branches.len(), 2);

        for branch in x_node.branches.iter() {
            let z_node = branch.child.as_ref().unwrap();
            assert_eq!(z_node.variable, v(2));
            assert_eq!(z_node.bound_prefix.as_ref(), [t(10), branch.value]);
            assert_eq!(z_node.intersection.as_ref(), [t(100), t(101)]);
            assert!(z_node.branches.iter().all(|leaf| leaf.child.is_none()));
        }

        let root_impact = replay.impact_for_bound_prefix(&[]).unwrap();
        assert_eq!(root_impact.bound_prefix.as_ref(), []);
        assert_eq!(root_impact.step_indexes.as_ref(), [0, 1, 2, 3]);
        assert_eq!(root_impact.steps, 4);
        assert_eq!(root_impact.candidate_bindings, 4);
        assert_eq!(root_impact.domain_sources, 5);
        assert_eq!(root_impact.domain_values, 10);
        assert_eq!(root_impact.intersection_values, 7);
        assert_eq!(root_impact.max_level, 2);

        let y10_impact = replay.impact_for_bound_prefix(&[t(10)]).unwrap();
        assert_eq!(y10_impact.bound_prefix.as_ref(), [t(10)]);
        assert_eq!(y10_impact.step_indexes.as_ref(), [1, 2, 3]);
        assert_eq!(y10_impact.steps, 3);
        assert_eq!(y10_impact.candidate_bindings, 4);
        assert_eq!(y10_impact.domain_sources, 3);
        assert_eq!(y10_impact.domain_values, 6);
        assert_eq!(y10_impact.intersection_values, 6);
        assert_eq!(y10_impact.max_level, 2);

        let x1_impact = replay.impact_for_bound_prefix(&[t(10), t(1)]).unwrap();
        assert_eq!(x1_impact.bound_prefix.as_ref(), [t(10), t(1)]);
        assert_eq!(x1_impact.step_indexes.as_ref(), [2]);
        assert_eq!(x1_impact.steps, 1);
        assert_eq!(x1_impact.candidate_bindings, 2);
        assert_eq!(x1_impact.domain_sources, 1);
        assert_eq!(x1_impact.domain_values, 2);
        assert_eq!(x1_impact.intersection_values, 2);
        assert_eq!(x1_impact.max_level, 2);

        assert_eq!(replay.impact_for_bound_prefix(&[t(20)]), None);
        assert_eq!(replay.impact_for_bound_prefix(&[t(10), t(3)]), None);
    }

    #[test]
    fn trie_join_trace_cursor_contract_lists_per_factor_replay_contexts() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10], &[3, 20]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101], &[30, 300]]);
        let variable_order = [v(1), v(0), v(2)];

        let trace = trie_join_trace(&[left, right], &variable_order).unwrap();
        let contract = trace.cursor_contract().unwrap();

        assert_eq!(contract.summary, trace.summarize().unwrap());
        assert_eq!(contract.relation_indexes, 2);
        assert_eq!(contract.factor_requirements.len(), 2);

        let left_factor = &contract.factor_requirements[0];
        assert_eq!(left_factor.relation_index, 0);
        assert_eq!(left_factor.domain_contexts, 2);
        assert_eq!(left_factor.domain_values, 4);
        assert_eq!(left_factor.max_domain_len, 2);
        assert_eq!(left_factor.contexts[0].step_index, 0);
        assert_eq!(left_factor.contexts[0].variable, v(1));
        assert_eq!(left_factor.contexts[0].bound_prefix.as_ref(), []);
        assert_eq!(left_factor.contexts[0].domain.as_ref(), [t(10), t(20)]);
        assert_eq!(left_factor.contexts[0].domain_len, 2);
        assert_eq!(left_factor.contexts[0].intersection_len, 1);
        assert_eq!(left_factor.contexts[1].step_index, 1);
        assert_eq!(left_factor.contexts[1].variable, v(0));
        assert_eq!(left_factor.contexts[1].bound_prefix.as_ref(), [t(10)]);
        assert_eq!(left_factor.contexts[1].domain.as_ref(), [t(1), t(2)]);

        let right_factor = &contract.factor_requirements[1];
        assert_eq!(right_factor.relation_index, 1);
        assert_eq!(right_factor.domain_contexts, 3);
        assert_eq!(right_factor.domain_values, 6);
        assert_eq!(right_factor.max_domain_len, 2);
        assert_eq!(right_factor.contexts[0].step_index, 0);
        assert_eq!(right_factor.contexts[0].variable, v(1));
        assert_eq!(right_factor.contexts[0].bound_prefix.as_ref(), []);
        assert_eq!(right_factor.contexts[0].domain.as_ref(), [t(10), t(30)]);
        assert_eq!(right_factor.contexts[1].step_index, 2);
        assert_eq!(right_factor.contexts[1].variable, v(2));
        assert_eq!(
            right_factor.contexts[1].bound_prefix.as_ref(),
            [t(10), t(1)]
        );
        assert_eq!(right_factor.contexts[1].domain.as_ref(), [t(100), t(101)]);
        assert_eq!(right_factor.contexts[2].step_index, 3);
        assert_eq!(right_factor.contexts[2].variable, v(2));
        assert_eq!(
            right_factor.contexts[2].bound_prefix.as_ref(),
            [t(10), t(2)]
        );
        assert_eq!(right_factor.contexts[2].domain.as_ref(), [t(100), t(101)]);
    }

    #[test]
    fn trie_join_trace_replay_diff_reports_context_edits_without_nested_double_counting() {
        let old_left = relation(&[0, 1], &[&[1, 10], &[2, 10]]);
        let new_left = relation(&[0, 1], &[&[1, 10], &[2, 10], &[3, 10]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101]]);
        let variable_order = [v(1), v(0), v(2)];

        let old_replay = trie_join_trace(&[old_left, right.clone()], &variable_order)
            .unwrap()
            .replay_shape()
            .unwrap();
        let new_replay = trie_join_trace(&[new_left, right], &variable_order)
            .unwrap()
            .replay_shape()
            .unwrap();

        let diff = old_replay.diff(&new_replay);

        assert!(!diff.is_empty());
        assert_eq!(diff.unchanged_contexts, 3);
        assert_eq!(diff.changed_contexts, 1);
        assert_eq!(diff.added_contexts, 1);
        assert_eq!(diff.removed_contexts, 0);
        assert_eq!(diff.frontier_contexts, 1);
        assert_eq!(diff.old_replay_steps_touched, 3);
        assert_eq!(diff.new_replay_steps_touched, 4);
        assert_eq!(diff.old_candidate_bindings_touched, 4);
        assert_eq!(diff.new_candidate_bindings_touched, 6);
        assert_eq!(diff.entries.len(), 2);

        let changed_x = &diff.entries[0];
        assert_eq!(changed_x.kind, TrieJoinTraceReplayDiffKind::Changed);
        assert_eq!(changed_x.bound_prefix.as_ref(), [t(10)]);
        assert_eq!(
            changed_x.old_impact.as_ref().unwrap().step_indexes.as_ref(),
            [1, 2, 3]
        );
        assert_eq!(
            changed_x.new_impact.as_ref().unwrap().step_indexes.as_ref(),
            [1, 2, 3, 4]
        );

        let added_x3 = &diff.entries[1];
        assert_eq!(added_x3.kind, TrieJoinTraceReplayDiffKind::Added);
        assert_eq!(added_x3.bound_prefix.as_ref(), [t(10), t(3)]);
        assert_eq!(added_x3.old_impact, None);
        assert_eq!(added_x3.new_impact.as_ref().unwrap().candidate_bindings, 2);

        let reverse = new_replay.diff(&old_replay);
        assert_eq!(reverse.added_contexts, 0);
        assert_eq!(reverse.removed_contexts, 1);
        assert_eq!(reverse.changed_contexts, 1);
        assert_eq!(reverse.frontier_contexts, 1);
        assert_eq!(reverse.old_candidate_bindings_touched, 6);
        assert_eq!(reverse.new_candidate_bindings_touched, 4);
    }

    #[test]
    fn trie_join_trace_replay_diff_compares_exact_factor_domains() {
        let old_left = relation(&[0, 1], &[&[1, 10], &[3, 20]]);
        let new_left = relation(&[0, 1], &[&[1, 10], &[3, 25]]);
        let right = relation(&[1, 2], &[&[10, 100]]);
        let variable_order = [v(1), v(0), v(2)];

        let old_replay = trie_join_trace(&[old_left, right.clone()], &variable_order)
            .unwrap()
            .replay_shape()
            .unwrap();
        let new_replay = trie_join_trace(&[new_left, right], &variable_order)
            .unwrap()
            .replay_shape()
            .unwrap();

        let diff = old_replay.diff(&new_replay);

        assert!(!diff.is_empty());
        assert_eq!(diff.unchanged_contexts, 2);
        assert_eq!(diff.changed_contexts, 1);
        assert_eq!(diff.added_contexts, 0);
        assert_eq!(diff.removed_contexts, 0);
        assert_eq!(diff.frontier_contexts, 1);
        assert_eq!(diff.old_candidate_bindings_touched, 1);
        assert_eq!(diff.new_candidate_bindings_touched, 1);
        assert_eq!(diff.entries.len(), 1);
        assert_eq!(diff.entries[0].bound_prefix.as_ref(), []);
        assert_eq!(diff.entries[0].kind, TrieJoinTraceReplayDiffKind::Changed);
        assert_eq!(
            old_replay.summary.candidate_bindings,
            new_replay.summary.candidate_bindings
        );
        assert_eq!(
            old_replay.root.as_ref().unwrap().intersection,
            new_replay.root.as_ref().unwrap().intersection
        );
    }

    #[test]
    fn trie_join_trace_summary_rejects_inconsistent_domain_metadata() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101]]);
        let variable_order = [v(1), v(0), v(2)];
        let mut trace = trie_join_trace(&[left, right], &variable_order).unwrap();

        trace.steps[0].domain_values += 1;

        assert_eq!(
            trace.summarize().unwrap_err(),
            TrieJoinTraceShapeError::DomainValueMismatch {
                step: 0,
                expected: 2,
                actual: 3,
            }
        );
    }

    #[test]
    fn trie_join_trace_summary_rejects_unordered_exact_domain_values() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101]]);
        let variable_order = [v(1), v(0), v(2)];
        let mut trace = trie_join_trace(&[left, right], &variable_order).unwrap();

        trace.steps[1].relation_domains[0].values = vec![t(2), t(1)].into_boxed_slice();

        assert_eq!(
            trace.summarize().unwrap_err(),
            TrieJoinTraceShapeError::RelationDomainOrderMismatch {
                step: 1,
                relation_index: 0,
                previous: t(2),
                actual: t(1),
            }
        );
    }

    #[test]
    fn trie_join_trace_replay_rejects_missing_child_context() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101]]);
        let variable_order = [v(1), v(0), v(2)];
        let mut trace = trie_join_trace(&[left, right], &variable_order).unwrap();

        trace.steps = trace.steps[..trace.steps.len() - 1]
            .to_vec()
            .into_boxed_slice();

        assert_eq!(
            trace.replay_shape().unwrap_err(),
            TrieJoinTraceShapeError::MissingReplayStep {
                level: 2,
                bound_prefix_len: 2,
            }
        );
    }

    #[test]
    fn trie_join_trace_replay_rejects_wrong_bound_prefix_branch() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101]]);
        let variable_order = [v(1), v(0), v(2)];
        let mut trace = trie_join_trace(&[left, right], &variable_order).unwrap();
        trace.steps[2].bound_prefix[1] = t(2);

        assert_eq!(
            trace.replay_shape().unwrap_err(),
            TrieJoinTraceShapeError::BoundPrefixValueMismatch {
                step: 2,
                position: 1,
                expected: t(1),
                actual: t(2),
            }
        );
    }

    #[test]
    fn generic_leapfrog_intersection_streams_values_and_can_stop_early() {
        let domains = vec![
            vec![t(1), t(3), t(5), t(8)],
            vec![t(2), t(3), t(5), t(9)],
            vec![t(3), t(4), t(5), t(10)],
        ];
        let mut positions = Vec::new();
        let mut streamed = Vec::new();
        let stopped = try_for_each_leapfrog_intersection(&domains, &mut positions, |value| {
            streamed.push(value);
            Ok::<bool, BindingRelationError>(false)
        })
        .unwrap();

        assert!(!stopped);
        assert_eq!(streamed, vec![t(3), t(5)]);

        let mut stopped_at = Vec::new();
        let stopped = try_for_each_leapfrog_intersection(&domains, &mut positions, |value| {
            stopped_at.push(value);
            Ok::<bool, BindingRelationError>(true)
        })
        .unwrap();

        assert!(stopped);
        assert_eq!(stopped_at, vec![t(3)]);
    }

    #[test]
    fn cursor_leapfrog_intersection_uses_monotone_seek_contract() {
        let left = [t(1), t(3), t(5), t(8)];
        let right = [t(2), t(3), t(5), t(9)];
        let guard = [t(3), t(4), t(5), t(10)];
        let mut cursors = [
            SliceDomainCursor::new(&left),
            SliceDomainCursor::new(&right),
            SliceDomainCursor::new(&guard),
        ];
        let mut stats = TrieJoinStats::default();

        let intersection = leapfrog_intersection_cursors(&mut cursors, &mut stats);

        assert_eq!(intersection, vec![t(3), t(5)]);
        assert_eq!(stats.domain_cursor_opens, 3);
        assert_eq!(stats.domain_cursor_nexts, 2);
        assert!(stats.domain_cursor_seeks >= stats.domain_cursor_opens);
        assert_eq!(stats.domain_cursor_skips, 8);
    }

    #[test]
    fn lower_bound_from_reports_first_value_at_or_after_target() {
        let domain = [t(1), t(3), t(5), t(8)];

        assert_eq!(lower_bound_from(&domain, 0, t(0)), 0);
        assert_eq!(lower_bound_from(&domain, 0, t(1)), 0);
        assert_eq!(lower_bound_from(&domain, 0, t(2)), 1);
        assert_eq!(lower_bound_from(&domain, 1, t(5)), 2);
        assert_eq!(lower_bound_from(&domain, 2, t(6)), 3);
        assert_eq!(lower_bound_from(&domain, 3, t(9)), domain.len());
        assert_eq!(lower_bound_from(&domain, domain.len(), t(9)), domain.len());
    }

    #[test]
    fn trie_join_handles_cyclic_triangle_without_binary_intermediate() {
        let xy = relation(&[0, 1], &[&[1, 2], &[1, 3], &[2, 3], &[3, 1]]);
        let yz = relation(&[1, 2], &[&[2, 3], &[3, 1], &[3, 4], &[1, 2]]);
        let zx = relation(&[2, 0], &[&[3, 1], &[1, 2], &[2, 3], &[4, 1]]);

        let generic =
            generic_join(&[xy.clone(), yz.clone(), zx.clone()], &[v(0), v(1), v(2)]).unwrap();
        let trie = trie_join(&[xy, yz, zx], &[v(0), v(1), v(2)]).unwrap();

        assert_eq!(trie.relation, generic);
        assert_eq!(
            trie.relation
                .positive_rows()
                .map(Vec::from)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                vec![t(1), t(2), t(3)],
                vec![t(1), t(3), t(4)],
                vec![t(2), t(3), t(1)],
                vec![t(3), t(1), t(2)],
            ])
        );
        assert_eq!(trie.stats.output_rows, 4);
        assert!(trie.stats.domain_intersections < 10);
    }

    #[test]
    fn factorized_join_counts_without_flattening() {
        let left = relation(&[0, 1], &[&[1, 10], &[2, 10]]);
        let right = relation(&[1, 2], &[&[10, 100], &[10, 101], &[10, 102]]);

        let factorized = FactorizedJoin::from_relations(&left, &right).unwrap();

        assert_eq!(factorized.checked_count().unwrap(), 6);
        assert_eq!(factorized.count(), 6);
        assert_eq!(factorized.checked_factorized_node_count().unwrap(), 7);
        assert_eq!(factorized.rows().count(), 6);
        assert_eq!(
            factorized
                .binding_rows()
                .map(Vec::from)
                .collect::<BTreeSet<_>>(),
            factorized.rows().collect::<BTreeSet<_>>()
        );
        assert_eq!(
            factorized.rows().collect::<BTreeSet<_>>(),
            BTreeSet::from([
                vec![t(10), t(1), t(100)],
                vec![t(10), t(1), t(101)],
                vec![t(10), t(1), t(102)],
                vec![t(10), t(2), t(100)],
                vec![t(10), t(2), t(101)],
                vec![t(10), t(2), t(102)],
            ])
        );
        assert_eq!(factorized.groups.len(), 1);
        assert_eq!(factorized.groups[0].key.as_ref(), &[t(10)]);
        assert_eq!(factorized.groups[0].left_residuals.len(), 2);
        assert_eq!(factorized.groups[0].right_residuals.len(), 3);
        assert!(factorized.factorized_node_count() < factorized.count() + 6);
    }

    #[test]
    fn factorized_count_helpers_report_cardinality_overflow() {
        assert_eq!(
            checked_mul_cardinality(usize::MAX, 2),
            Err(BindingRelationError::CardinalityOverflow)
        );
        assert_eq!(
            checked_add_cardinality(usize::MAX, 1),
            Err(BindingRelationError::CardinalityOverflow)
        );
    }

    #[test]
    fn semijoin_removed_row_helpers_report_cardinality_invariants() {
        assert_eq!(checked_removed_rows(7, 3), Ok(4));
        assert_eq!(
            checked_removed_rows(3, 7),
            Err(BindingRelationError::CardinalityOverflow)
        );
        assert_eq!(
            checked_add_removed_rows(usize::MAX, 1, 0),
            Err(BindingRelationError::CardinalityOverflow)
        );
    }

    #[test]
    fn signed_rows_cancel_and_participate_in_joins() {
        let mut left = BindingRelation::new([v(0)]);
        left.add(vec![t(1)], 2).unwrap();
        left.add(vec![t(1)], -2).unwrap();
        assert!(left.is_empty());

        left.add(vec![t(1)], -1).unwrap();
        let right = relation(&[0], &[&[1]]);
        let joined = natural_join(&left, &right).unwrap();

        assert_eq!(joined.weight(&[t(1)]), -1);
    }

    #[test]
    fn signed_row_zero_weight_adds_and_cancellations_do_not_retain_rows() {
        let mut relation = BindingRelation::new([v(0)]);

        relation.add(vec![t(1)], 0).unwrap();
        assert!(relation.is_empty());

        relation.add(vec![t(1)], 5).unwrap();
        relation.add(vec![t(2)], 7).unwrap();
        relation.add(vec![t(1)], -5).unwrap();

        assert_eq!(relation.len(), 1);
        assert_eq!(relation.weight(&[t(1)]), 0);
        assert_eq!(relation.weight(&[t(2)]), 7);
    }

    #[test]
    fn insert_unit_if_absent_reports_new_rows_without_overwriting_signed_rows() {
        let mut relation = BindingRelation::new([v(0)]);

        assert_eq!(relation.insert_unit_if_absent([t(1)]), Ok(true));
        assert_eq!(relation.insert_unit_if_absent([t(1)]), Ok(false));
        assert_eq!(relation.weight(&[t(1)]), 1);

        relation.add([t(2)], -3).unwrap();
        assert_eq!(relation.insert_unit_if_absent([t(2)]), Ok(false));
        assert_eq!(relation.weight(&[t(2)]), -3);

        assert_eq!(
            relation.insert_unit_if_absent([t(1), t(2)]),
            Err(BindingRelationError::ArityMismatch {
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn distinct_normalizes_retained_signed_rows_to_unit_weight() {
        let relation = signed_relation(&[0], &[(&[1], 5), (&[2], -3)]);

        let distinct = relation.distinct();

        assert_eq!(distinct.schema(), relation.schema());
        assert_eq!(distinct.len(), 2);
        assert_eq!(distinct.weight(&[t(1)]), 1);
        assert_eq!(distinct.weight(&[t(2)]), 1);
    }

    #[test]
    fn signed_row_addition_reports_weight_overflow() {
        let mut relation = BindingRelation::new([v(0)]);
        relation.add(vec![t(1)], i64::MAX).unwrap();

        let result = relation.add(vec![t(1)], 1);

        assert_eq!(result, Err(BindingRelationError::WeightOverflow));
        assert_eq!(relation.weight(&[t(1)]), i64::MAX);
    }

    #[test]
    fn signed_difference_reports_min_weight_negation_overflow() {
        let left = BindingRelation::new([v(0)]);
        let right = signed_relation(&[0], &[(&[1], i64::MIN)]);

        let result = left.signed_difference(&right);

        assert_eq!(result, Err(BindingRelationError::WeightOverflow));
    }

    #[test]
    fn natural_join_reports_weight_product_overflow() {
        let left = signed_relation(&[0], &[(&[1], i64::MAX)]);
        let right = signed_relation(&[0], &[(&[1], 2)]);

        let result = natural_join(&left, &right);

        assert_eq!(result, Err(BindingRelationError::WeightOverflow));
    }

    #[test]
    fn trie_join_reports_weight_product_overflow() {
        let left = signed_relation(&[0], &[(&[1], i64::MAX)]);
        let right = signed_relation(&[0], &[(&[1], 2)]);

        let result = trie_join(&[left, right], &[v(0)]);

        assert!(matches!(result, Err(BindingRelationError::WeightOverflow)));
    }

    #[test]
    fn signed_difference_subtracts_weights_without_presence_semantics() {
        let left = signed_relation(&[0], &[(&[1], 3), (&[2], -2)]);
        let right = signed_relation(&[0], &[(&[1], 1), (&[2], -2), (&[3], 5)]);

        let diff = left.signed_difference(&right).unwrap();

        assert_eq!(diff.weight(&[t(1)]), 2);
        assert_eq!(diff.weight(&[t(2)]), 0);
        assert_eq!(diff.weight(&[t(3)]), -5);
    }

    #[test]
    fn delta_multi_join_matches_recomputation() {
        // Chain R(a,b), S(b,c), T(c,d). Adding one edge to each opens a new chain
        // 1->5->6->7 alongside the old 1->2->3->4.
        let olds = [
            relation(&[0, 1], &[&[1, 2]]),
            relation(&[1, 2], &[&[2, 3]]),
            relation(&[2, 3], &[&[3, 4]]),
        ];
        let news = [
            relation(&[0, 1], &[&[1, 2], &[1, 5]]),
            relation(&[1, 2], &[&[2, 3], &[5, 6]]),
            relation(&[2, 3], &[&[3, 4], &[6, 7]]),
        ];
        let deltas = [
            relation(&[0, 1], &[&[1, 5]]),
            relation(&[1, 2], &[&[5, 6]]),
            relation(&[2, 3], &[&[6, 7]]),
        ];
        let delta = delta_multi_join(&olds, &news, &deltas).unwrap();
        // The only new full chain is (1,5,6,7).
        assert_eq!(
            positive_row_set(&delta),
            BTreeSet::from([vec![t(1), t(5), t(6), t(7)]])
        );

        // Cross-check against a full recomputation: the delta's positive rows are
        // exactly the new join rows that were not in the old join.
        let new_join = natural_join_all(&news).unwrap().unwrap();
        let old_join = natural_join_all(&olds).unwrap().unwrap();
        let expected = new_join.difference_presence(&old_join).unwrap();
        assert_eq!(positive_row_set(&delta), positive_row_set(&expected));

        // Length mismatch is an error; empty input yields an empty relation.
        assert!(delta_multi_join(&olds, &news, &deltas[..2]).is_err());
        assert_eq!(
            delta_multi_join(&[], &[], &[])
                .unwrap()
                .positive_rows()
                .count(),
            0
        );
    }

    #[test]
    fn maintained_join_tracks_recomputation_under_deltas() {
        let olds = vec![
            relation(&[0, 1], &[&[1, 2]]),
            relation(&[1, 2], &[&[2, 3]]),
            relation(&[2, 3], &[&[3, 4]]),
        ];
        let deltas = vec![
            relation(&[0, 1], &[&[1, 5]]),
            relation(&[1, 2], &[&[5, 6]]),
            relation(&[2, 3], &[&[6, 7]]),
        ];

        let mut view = MaintainedJoin::new(olds.clone()).unwrap();
        assert_eq!(
            positive_row_set(view.join()),
            BTreeSet::from([vec![t(1), t(2), t(3), t(4)]])
        );

        view.apply_deltas(&deltas).unwrap();
        let news = olds
            .iter()
            .zip(&deltas)
            .map(|(old, delta)| {
                let mut updated = old.clone();
                updated.union_assign(delta).unwrap();
                updated
            })
            .collect::<Vec<_>>();
        let recomputed = natural_join_all(&news).unwrap().unwrap();
        assert_eq!(positive_row_set(view.join()), positive_row_set(&recomputed));
        assert_eq!(
            positive_row_set(view.join()),
            BTreeSet::from([vec![t(1), t(2), t(3), t(4)], vec![t(1), t(5), t(6), t(7)]])
        );

        assert!(view.apply_deltas(&deltas[..2]).is_err());
    }

    #[test]
    fn semi_naive_linear_fixpoint_computes_transitive_closure() {
        // reach(s,t) = edge(s,t) union project_{s,t}( edge(s,m) join reach(m,t) ).
        let edges = relation(&[0, 1], &[&[1, 2], &[2, 3], &[3, 4]]);
        // base = edge(s,t); fixed factor = edge(s,m) = edges as (0,2); the
        // recursive reach(m,t) is Q relabeled (0,1) -> (2,1); output is (s,t).
        let edge_sm = edges.relabel(&[v(0), v(2)]).unwrap();
        let reach =
            semi_naive_linear_fixpoint(&edges, &[edge_sm], &[v(2), v(1)], &[v(0), v(1)]).unwrap();

        assert_eq!(
            positive_row_set(&reach),
            BTreeSet::from([
                vec![t(1), t(2)],
                vec![t(2), t(3)],
                vec![t(3), t(4)],
                vec![t(1), t(3)],
                vec![t(2), t(4)],
                vec![t(1), t(4)],
            ])
        );

        // Agrees with the specialized transitive closure.
        let specialized = semi_naive_transitive_closure(&edges, v(0), v(1)).unwrap();
        assert_eq!(
            positive_row_set(&reach),
            positive_row_set(&specialized.relation)
        );

        // Schema guards.
        assert!(semi_naive_linear_fixpoint(&edges, &[], &[v(2)], &[v(0), v(1)]).is_err());
    }

    #[test]
    fn semi_naive_linear_fixpoint_wco_matches_the_binary_body() {
        // reach(s,t) = edge(s,t) union project_{s,t}( edge(s,m) join reach(m,t) ).
        let edges = relation(&[0, 1], &[&[1, 2], &[2, 3], &[3, 4]]);
        let edge_sm = edges.relabel(&[v(0), v(2)]).unwrap();
        let binary =
            semi_naive_linear_fixpoint(&edges, &[edge_sm.clone()], &[v(2), v(1)], &[v(0), v(1)])
                .unwrap();
        // The worst-case-optimal body join gives the same closure.
        let wco = semi_naive_linear_fixpoint_wco(&edges, &[edge_sm], &[v(2), v(1)], &[v(0), v(1)])
            .unwrap();
        assert_eq!(positive_row_set(&binary), positive_row_set(&wco));
        assert_eq!(
            positive_row_set(&wco),
            BTreeSet::from([
                vec![t(1), t(2)],
                vec![t(2), t(3)],
                vec![t(3), t(4)],
                vec![t(1), t(3)],
                vec![t(2), t(4)],
                vec![t(1), t(4)],
            ])
        );
    }

    #[test]
    fn delta_join_matches_full_recomputation_with_signed_deletions() {
        let old_left = relation(&[0, 1], &[&[1, 10], &[2, 10]]);
        let delta_left = relation(&[0, 1], &[&[3, 10]]);
        let old_right = relation(&[1, 2], &[&[10, 100], &[10, 101]]);
        let delta_right = signed_relation(&[1, 2], &[(&[10, 100], -1), (&[10, 102], 1)]);
        let mut new_left = old_left.clone();
        new_left.union_assign(&delta_left).unwrap();
        let mut new_right = old_right.clone();
        new_right.union_assign(&delta_right).unwrap();

        let delta = delta_join(&old_left, &new_right, &delta_left, &delta_right).unwrap();
        let recomputed = natural_join(&new_left, &new_right)
            .unwrap()
            .signed_difference(&natural_join(&old_left, &old_right).unwrap())
            .unwrap();

        assert_eq!(delta, recomputed);
        assert_eq!(delta.weight(&[t(1), t(10), t(100)]), -1);
        assert_eq!(delta.weight(&[t(2), t(10), t(100)]), -1);
        assert_eq!(delta.weight(&[t(1), t(10), t(102)]), 1);
        assert_eq!(delta.weight(&[t(2), t(10), t(102)]), 1);
        assert_eq!(delta.weight(&[t(3), t(10), t(101)]), 1);
        assert_eq!(delta.weight(&[t(3), t(10), t(102)]), 1);
    }

    #[test]
    fn incremental_transitive_closure_matches_full_recompute() {
        let source = v(0);
        let target = v(1);
        // Close a 1->2->3 chain, then add edges that extend both ends (3->4) and
        // prepend (0->1). Incremental maintenance must equal a full recompute.
        let old_edges = relation(&[0, 1], &[&[1, 2], &[2, 3]]);
        let closure = semi_naive_transitive_closure(&old_edges, source, target)
            .unwrap()
            .relation;
        let new_edges = relation(&[0, 1], &[&[3, 4], &[0, 1]]);
        let incremental =
            semi_naive_incremental_transitive_closure(&closure, &new_edges, source, target)
                .unwrap();

        let all_edges = relation(&[0, 1], &[&[1, 2], &[2, 3], &[3, 4], &[0, 1]]);
        let full = semi_naive_transitive_closure(&all_edges, source, target)
            .unwrap()
            .relation;
        assert_eq!(positive_row_set(&incremental), positive_row_set(&full));

        // Adding an edge that closes a cycle (4->0) reaches every pair.
        let cycle_edge = relation(&[0, 1], &[&[4, 0]]);
        let closed =
            semi_naive_incremental_transitive_closure(&incremental, &cycle_edge, source, target)
                .unwrap();
        // Five nodes in one cycle: all 25 ordered pairs including self-loops.
        assert_eq!(positive_row_set(&closed).len(), 25);
    }

    #[test]
    fn maintained_linear_fixpoint_matches_full_recompute_after_insert() {
        // reach(s,t) = edge(s,t) union project_{s,t}( edge(s,m) join reach(m,t) ).
        let edges = relation(&[0, 1], &[&[1, 2], &[2, 3]]);
        let edge_sm = edges.relabel(&[v(0), v(2)]).unwrap();
        let mut maintained =
            MaintainedLinearFixpoint::new(&edges, &[edge_sm], &[v(2), v(1)], &[v(0), v(1)])
                .unwrap();

        // Insert edge 3->4: a new base fact (reach holds the edge) and a new fixed
        // fact (the recursion's edge(s,m)).
        let new_edge = relation(&[0, 1], &[&[3, 4]]);
        let new_edge_sm = new_edge.relabel(&[v(0), v(2)]).unwrap();
        maintained.insert(&new_edge, &[(0, new_edge_sm)]).unwrap();
        let all = relation(&[0, 1], &[&[1, 2], &[2, 3], &[3, 4]]);
        let full = semi_naive_transitive_closure(&all, v(0), v(1))
            .unwrap()
            .relation;
        assert_eq!(positive_row_set(maintained.view()), positive_row_set(&full));

        // A second insert closes the cycle 4->1: every node reaches every node.
        let cycle = relation(&[0, 1], &[&[4, 1]]);
        let cycle_sm = cycle.relabel(&[v(0), v(2)]).unwrap();
        maintained.insert(&cycle, &[(0, cycle_sm)]).unwrap();
        let all2 = relation(&[0, 1], &[&[1, 2], &[2, 3], &[3, 4], &[4, 1]]);
        let full2 = semi_naive_transitive_closure(&all2, v(0), v(1))
            .unwrap()
            .relation;
        assert_eq!(
            positive_row_set(maintained.view()),
            positive_row_set(&full2)
        );
        assert_eq!(positive_row_set(maintained.view()).len(), 16);
    }

    #[ignore]
    #[test]
    fn bench_maintained_vs_recompute_transitive_closure() {
        // Stream the edges of a chain one at a time. A persisted maintained
        // closure costs O(new pairs) per edge (O(closure) total); recomputing the
        // closure from scratch after each edge costs O(closure) each time, so
        // O(edges * closure) total. Run with `--ignored --nocapture`.
        let source = v(0);
        let target = v(1);
        for n in [100usize, 200, 400] {
            let start = std::time::Instant::now();
            let mut maintained = MaintainedTransitiveClosure::new();
            for i in 0..n {
                maintained.insert_edge(t(i as u64), t(i as u64 + 1));
            }
            let maintained_us = start.elapsed().as_micros();

            let start = std::time::Instant::now();
            let mut edges: Vec<Vec<u64>> = Vec::new();
            for i in 0..n {
                edges.push(vec![i as u64, i as u64 + 1]);
                let refs: Vec<&[u64]> = edges.iter().map(|r| r.as_slice()).collect();
                let relation = relation(&[0, 1], &refs);
                let _ = semi_naive_transitive_closure(&relation, source, target).unwrap();
            }
            let recompute_us = start.elapsed().as_micros();

            eprintln!(
                "n={n}: maintained {maintained_us}us, recompute-each {recompute_us}us, speedup {:.1}x",
                recompute_us as f64 / maintained_us.max(1) as f64
            );
        }
    }

    #[test]
    fn semi_naive_transitive_closure_matches_graph_reference_on_cycles() {
        let source = v(0);
        let target = v(1);
        let mut edges = BindingRelation::new([source, target]);
        let mut adjacency = BTreeMap::<TermId, BTreeSet<TermId>>::new();
        for node in 0..18u64 {
            let from = t(node);
            let next = t((node + 1) % 18);
            edges.add(vec![from, next], 1).unwrap();
            adjacency.entry(from).or_default().insert(next);
            if node % 3 == 0 {
                let skip = t((node * 5 + 7) % 18);
                edges.add(vec![from, skip], 1).unwrap();
                adjacency.entry(from).or_default().insert(skip);
            }
        }

        let actual = semi_naive_transitive_closure(&edges, source, target).unwrap();
        let mut expected = BindingRelation::new([source, target]);
        for &start in adjacency.keys() {
            let mut seen = BTreeSet::new();
            let mut stack = adjacency
                .get(&start)
                .into_iter()
                .flat_map(|targets| targets.iter().copied())
                .collect::<Vec<_>>();
            while let Some(node) = stack.pop() {
                if !seen.insert(node) {
                    continue;
                }
                expected.add(vec![start, node], 1).unwrap();
                if let Some(targets) = adjacency.get(&node) {
                    stack.extend(targets.iter().copied());
                }
            }
        }

        assert_eq!(actual.relation, expected);
        assert!(actual.rounds > 0);
        assert!(actual.candidate_extensions >= actual.derived_rows);
    }

    #[test]
    fn semi_naive_transitive_closure_handles_empty_and_self_loop_relations() {
        let source = v(0);
        let target = v(1);
        let empty = BindingRelation::new([source, target]);
        let empty_closure = semi_naive_transitive_closure(&empty, source, target).unwrap();
        assert!(empty_closure.relation.is_empty());
        assert_eq!(empty_closure.rounds, 0);

        let loops = relation(&[0, 1], &[&[1, 1], &[1, 2]]);
        let closure = semi_naive_transitive_closure(&loops, source, target).unwrap();
        assert_eq!(closure.relation.len(), 2);
        assert_eq!(closure.relation.weight(&[t(1), t(1)]), 1);
        assert_eq!(closure.relation.weight(&[t(1), t(2)]), 1);
    }

    #[test]
    fn presence_difference_uses_positive_visibility() {
        let left = relation(&[0], &[&[1], &[2]]);
        let mut right = BindingRelation::new([v(0)]);
        right.add(vec![t(1)], -1).unwrap();
        right.add(vec![t(2)], 1).unwrap();

        let diff = left.difference_presence(&right).unwrap();

        assert_eq!(diff.weight(&[t(1)]), 1);
        assert_eq!(diff.weight(&[t(2)]), 0);
    }
}
