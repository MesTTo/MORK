use vstd::prelude::*;

verus! {

pub open spec fn bit_rank(mask: Seq<bool>, end: nat) -> nat
    recommends
        end <= mask.len(),
    decreases end,
{
    if end == 0 {
        0
    } else {
        bit_rank(mask, (end - 1) as nat)
            + if mask[(end - 1) as int] {
                1nat
            } else {
                0nat
            }
    }
}

pub fn rank_prefix(mask: &Vec<bool>, end: usize) -> (count: usize)
    requires
        end <= mask.len(),
    ensures
        count == bit_rank(mask@, end as nat),
        count <= end,
{
    let mut index: usize = 0;
    let mut count: usize = 0;
    while index < end
        invariant
            index <= end,
            end <= mask.len(),
            count == bit_rank(mask@, index as nat),
            count <= index,
        decreases end - index,
    {
        if mask[index] {
            count = count + 1;
        }
        index = index + 1;
    }
    count
}

pub proof fn lemma_rank_prefix_bounds(mask: Seq<bool>, end: nat)
    requires
        end <= mask.len(),
    ensures
        bit_rank(mask, end) <= end,
    decreases end,
{
    if end == 0 {
    } else {
        lemma_rank_prefix_bounds(mask, (end - 1) as nat);
    }
}

pub proof fn lemma_rank_monotonic_one(mask: Seq<bool>, end: nat)
    requires
        end < mask.len(),
    ensures
        bit_rank(mask, end) <= bit_rank(mask, end + 1),
        bit_rank(mask, end + 1) <= bit_rank(mask, end) + 1,
{
}

pub proof fn lemma_rank_set_bit_adds_one(mask: Seq<bool>, end: nat)
    requires
        end < mask.len(),
        mask[end as int],
    ensures
        bit_rank(mask, end + 1) == bit_rank(mask, end) + 1,
{
}

pub proof fn lemma_rank_unset_bit_preserves(mask: Seq<bool>, end: nat)
    requires
        end < mask.len(),
        !mask[end as int],
    ensures
        bit_rank(mask, end + 1) == bit_rank(mask, end),
{
}

pub proof fn lemma_rank_is_bounded_by_256(mask: Seq<bool>, end: nat)
    requires
        mask.len() == 256,
        end <= 256,
    ensures
        bit_rank(mask, end) <= 256,
{
    lemma_rank_prefix_bounds(mask, end);
}

fn main() {}

}
