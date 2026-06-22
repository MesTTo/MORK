use vstd::prelude::*;

verus! {

pub open spec fn sorted(values: Seq<u64>) -> bool {
    forall|i: int, j: int|
        0 <= i < j < values.len() ==> values[i] <= values[j]
}

/// Verified seek primitive for leapfrog-style ordered intersection.
pub fn lower_bound_from(values: &Vec<u64>, start: usize, target: u64) -> (result: usize)
    requires
        start <= values.len(),
        sorted(values@),
    ensures
        start <= result <= values.len(),
        forall|i: int| start <= i < result ==> values[i] < target,
        result < values.len() ==> values[result as int] >= target,
{
    let mut index = start;
    while index < values.len() && values[index] < target
        invariant
            start <= index <= values.len(),
            sorted(values@),
            forall|i: int| start <= i < index ==> values[i] < target,
        decreases values.len() - index,
    {
        index = index + 1;
    }
    index
}

fn main() {}

}
