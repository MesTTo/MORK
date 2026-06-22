use vstd::prelude::*;

verus! {

/// Executable membership test over a sequence of projected row keys.
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

/// Verified model of `BindingRelation::project` at the projected-key level. Each
/// input element is the projection of one positive row onto the bag variables;
/// the result keeps each distinct projection once, which is the set semantics the
/// hypertree-decomposition bag materialization relies on.
///
/// The postconditions are the safety contract: projection
/// 1. never fabricates a row (every output key is the projection of some input
///    row);
/// 2. is a set (the output has no duplicates); and
/// 3. is monotone (the output is no larger than the input).
///
/// So a bag relation only ever holds projections of real joined rows, and the
/// join-of-bags cannot invent answers.
pub fn project_distinct(input: &Vec<u64>) -> (output: Vec<u64>)
    ensures
        forall|i: int| 0 <= i < output.len() ==> input@.contains(#[trigger] output@[i]),
        forall|i: int, j: int| 0 <= i < j < output.len() ==> output@[i] != output@[j],
        output.len() <= input.len(),
{
    let mut output: Vec<u64> = Vec::new();
    let mut index: usize = 0;
    while index < input.len()
        invariant
            index <= input.len(),
            output.len() <= index,
            forall|i: int| 0 <= i < output.len() ==> input@.contains(#[trigger] output@[i]),
            forall|i: int, j: int| 0 <= i < j < output.len() ==> output@[i] != output@[j],
        decreases input.len() - index,
    {
        let value = input[index];
        if !contains_key(&output, value) {
            assert(input@.contains(value)) by {
                assert(input@[index as int] == value);
            }
            assert(forall|i: int| 0 <= i < output.len() ==> output@[i] != value);
            output.push(value);
        }
        index = index + 1;
    }
    output
}

fn main() {}

}
