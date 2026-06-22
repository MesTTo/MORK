use vstd::prelude::*;

verus! {

/// A relation row split into the parts a semijoin cares about. `key` is the
/// projection onto the shared join variables (what `semijoin_presence` matches
/// on); `payload` is the rest of the row, carried through unchanged.
/// `semijoin_presence` keeps a whole left row when its shared-key projection
/// occurs among the right relation's keys, so two left rows with the same key
/// are kept or dropped together. Signed weights and the `BTreeMap` dedup of
/// identical rows do not affect that keep/drop decision and are not modeled here.
#[derive(PartialEq, Eq, Structural, Clone, Copy)]
pub struct Row {
    pub key: u64,
    pub payload: u64,
}

/// Executable membership test over a sequence of join keys, mirroring the
/// `Seq::contains` spec used in the postconditions.
pub fn contains_key(keys: &Vec<u64>, value: u64) -> (found: bool)
    ensures
        found == keys@.contains(value),
{
    let mut index: usize = 0;
    while index < keys.len()
        invariant
            index <= keys.len(),
            forall|k: int| 0 <= k < index ==> keys@[k] != value,
        decreases keys.len() - index,
    {
        if keys[index] == value {
            assert(keys@.contains(value)) by {
                assert(keys@[index as int] == value);
            }
            return true;
        }
        index = index + 1;
    }
    assert(!keys@.contains(value));
    false
}

/// Verified model of the per-edge semijoin step that the Yannakakis full reducer
/// (`BindingSpace::semijoin_reduce_presence`) iterates: retain each `left` row
/// whose shared-key projection has a partner in `right_keys`.
///
/// The postconditions pin the result down to exactly `{ r in left : r.key in
/// right_keys }`, the full characterization of a semijoin:
/// 1. No fabrication: every surviving row came from `left`.
/// 2. Sound keep: every surviving row has a `right` partner.
/// 3. No over-reduction (completeness): every left row whose key has a `right`
///    partner is retained.
///
/// Property 3 is what guarantees Yannakakis correctness: the reducer never drops
/// a participating tuple, so the reduced join keeps the same answer set.
/// Properties 1 and 2 alone (the earlier seed) hold for the empty result, so
/// they did not rule out over-reduction. Property 3 rejects it.
pub fn semijoin_filter(left: &Vec<Row>, right_keys: &Vec<u64>) -> (result: Vec<Row>)
    ensures
        forall|i: int| 0 <= i < result.len() ==> left@.contains(#[trigger] result@[i]),
        forall|i: int| 0 <= i < result.len() ==> right_keys@.contains(#[trigger] result@[i].key),
        forall|j: int|
            0 <= j < left.len() && right_keys@.contains(left@[j].key)
                ==> #[trigger] result@.contains(left@[j]),
{
    let mut result: Vec<Row> = Vec::new();
    let mut index: usize = 0;
    while index < left.len()
        invariant
            index <= left.len(),
            forall|k: int| 0 <= k < result.len() ==> left@.contains(#[trigger] result@[k]),
            forall|k: int| 0 <= k < result.len() ==> right_keys@.contains(#[trigger] result@[k].key),
            forall|j: int|
                0 <= j < index && right_keys@.contains(left@[j].key)
                    ==> #[trigger] result@.contains(left@[j]),
        decreases left.len() - index,
    {
        let ghost before = result@;
        // Carry the loop invariant onto the immutable snapshot `before`.
        assert(forall|j: int|
            0 <= j < index && right_keys@.contains(left@[j].key)
                ==> #[trigger] before.contains(left@[j]));
        let row = left[index];
        let keep = contains_key(right_keys, row.key);
        if keep {
            assert(left@.contains(row)) by {
                assert(left@[index as int] == row);
            }
            result.push(row);
        }
        // `result@` is either `before` (no push) or `before.push(row)`; either
        // way `before` is a prefix and the length only grows.
        assert(before.len() <= result.len());
        assert(forall|k: int| #![trigger result@[k]] 0 <= k < before.len() ==> result@[k] == before[k]);
        // Re-establish completeness for j <= index.
        assert forall|j: int|
            0 <= j < index + 1 && right_keys@.contains(left@[j].key) implies #[trigger] result@.contains(
            left@[j],
        ) by {
            if j < index {
                // Prior invariant: `before` already contained `left@[j]`; the
                // preserved prefix carries the same witness index into `result@`.
                assert(before.contains(left@[j]));
                let w = choose|w: int| 0 <= w < before.len() && before[w] == left@[j];
                assert(0 <= w < before.len() && before[w] == left@[j]);
                assert(result@[w] == before[w]);
                assert(0 <= w < result.len() && result@[w] == left@[j]);
            } else {
                // j == index: the antecedent forces `keep` (since `row ==
                // left@[index]`), so `row` was just pushed as the last element.
                assert(row == left@[index as int]);
                assert(keep);
                assert(result@[result.len() - 1] == row);
                assert(0 <= result.len() - 1 < result.len() && result@[result.len() - 1] == left@[j]);
            }
        }
        index = index + 1;
    }
    result
}

fn main() {}

}
