use vstd::prelude::*;

verus! {

/// Scalar executable model for the common-prefix contract used by BTM
/// write-resource grouping. Optimized implementations may scan differently,
/// but must return the same boundary.
pub fn common_prefix_len_model(a: &Vec<u8>, b: &Vec<u8>) -> (n: usize)
    ensures
        n <= a.len(),
        n <= b.len(),
        forall|i: int| 0 <= i < n ==> a[i] == b[i],
        n < a.len() && n < b.len() ==> a[n as int] != b[n as int],
{
    let limit = if a.len() < b.len() { a.len() } else { b.len() };
    let mut n: usize = 0;
    while n < limit && a[n] == b[n]
        invariant
            limit <= a.len(),
            limit <= b.len(),
            n <= limit,
            forall|i: int| 0 <= i < n ==> a[i] == b[i],
        decreases limit - n,
    {
        n = n + 1;
    }
    n
}

fn main() {}

}
