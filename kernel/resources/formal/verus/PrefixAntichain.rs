use vstd::prelude::*;

verus! {

pub open spec fn is_prefix(prefix: Seq<u8>, path: Seq<u8>) -> bool {
    prefix.len() <= path.len()
        && forall|i: int| 0 <= i < prefix.len() ==> #[trigger] prefix[i] == path[i]
}

pub open spec fn prefix_antichain(prefixes: Seq<Seq<u8>>) -> bool {
    forall|i: int, j: int|
        0 <= i < j < prefixes.len() ==>
            !is_prefix(prefixes[i], prefixes[j]) && !is_prefix(prefixes[j], prefixes[i])
}

pub open spec fn unrelated_to_all(prefixes: Seq<Seq<u8>>, prefix: Seq<u8>) -> bool {
    forall|i: int|
        #![trigger prefixes[i]]
        0 <= i < prefixes.len() ==>
            !is_prefix(prefixes[i], prefix) && !is_prefix(prefix, prefixes[i])
}

pub open spec fn mutually_unrelated(left: Seq<Seq<u8>>, right: Seq<Seq<u8>>) -> bool {
    forall|i: int, j: int|
        0 <= i < left.len() && 0 <= j < right.len() ==>
            !is_prefix(left[i], right[j]) && !is_prefix(right[j], left[i])
}

pub proof fn lemma_prefix_reflexive(path: Seq<u8>)
    ensures
        is_prefix(path, path),
{
}

pub proof fn lemma_prefix_transitive(left: Seq<u8>, middle: Seq<u8>, right: Seq<u8>)
    requires
        is_prefix(left, middle),
        is_prefix(middle, right),
    ensures
        is_prefix(left, right),
{
    assert(left.len() <= right.len());
    assert forall|i: int| 0 <= i < left.len() implies #[trigger] left[i] == right[i] by {
        assert(left[i] == middle[i]);
        assert(middle[i] == right[i]);
    }
}

pub proof fn lemma_push_unrelated_preserves_antichain(prefixes: Seq<Seq<u8>>, prefix: Seq<u8>)
    requires
        prefix_antichain(prefixes),
        unrelated_to_all(prefixes, prefix),
    ensures
        prefix_antichain(prefixes.push(prefix)),
{
    assert forall|i: int, j: int|
        0 <= i < j < prefixes.push(prefix).len() implies
            !is_prefix(prefixes.push(prefix)[i], prefixes.push(prefix)[j])
                && !is_prefix(prefixes.push(prefix)[j], prefixes.push(prefix)[i])
    by {
        if j < prefixes.len() {
            assert(prefixes.push(prefix)[i] == prefixes[i]);
            assert(prefixes.push(prefix)[j] == prefixes[j]);
        } else {
            assert(j == prefixes.len());
            assert(i < prefixes.len());
            assert(prefixes.push(prefix)[i] == prefixes[i]);
            assert(prefixes.push(prefix)[j] == prefix);
        }
    }
}

pub proof fn lemma_append_antichains_preserves_antichain(
    left: Seq<Seq<u8>>,
    right: Seq<Seq<u8>>,
)
    requires
        prefix_antichain(left),
        prefix_antichain(right),
        mutually_unrelated(left, right),
    ensures
        prefix_antichain(left + right),
{
    let merged = left + right;
    assert forall|i: int, j: int|
        0 <= i < j < merged.len() implies
            !is_prefix(merged[i], merged[j])
                && !is_prefix(merged[j], merged[i])
    by {
        if j < left.len() {
            assert(merged[i] == left[i]);
            assert(merged[j] == left[j]);
        } else if i >= left.len() {
            let ri = i - left.len();
            let rj = j - left.len();
            assert(0 <= ri < rj < right.len());
            assert(merged[i] == right[ri]);
            assert(merged[j] == right[rj]);
        } else {
            let rj = j - left.len();
            assert(0 <= i < left.len());
            assert(0 <= rj < right.len());
            assert(merged[i] == left[i]);
            assert(merged[j] == right[rj]);
        }
    }
}

fn main() {}

}
