use vstd::prelude::*;

verus! {

pub open spec fn seek_from(domain: Seq<int>, position: nat, target: int) -> nat
    recommends
        position <= domain.len(),
    decreases domain.len() - position,
{
    if position >= domain.len() {
        position
    } else if domain[position as int] >= target {
        position
    } else {
        seek_from(domain, position + 1, target)
    }
}

pub proof fn lemma_seek_from_bounds(domain: Seq<int>, position: nat, target: int)
    requires
        position <= domain.len(),
    ensures
        position <= seek_from(domain, position, target) <= domain.len(),
    decreases domain.len() - position,
{
    if position >= domain.len() {
    } else if domain[position as int] >= target {
    } else {
        lemma_seek_from_bounds(domain, position + 1, target);
    }
}

pub proof fn lemma_seek_from_stops_at_end_or_target(domain: Seq<int>, position: nat, target: int)
    requires
        position <= domain.len(),
    ensures
        seek_from(domain, position, target) == domain.len()
            || domain[seek_from(domain, position, target) as int] >= target,
    decreases domain.len() - position,
{
    if position >= domain.len() {
    } else if domain[position as int] >= target {
    } else {
        lemma_seek_from_stops_at_end_or_target(domain, position + 1, target);
    }
}

pub proof fn lemma_seek_from_skips_only_less_than_target(
    domain: Seq<int>,
    position: nat,
    target: int,
)
    requires
        position <= domain.len(),
    ensures
        forall|i: int|
            position <= i < seek_from(domain, position, target) ==> #[trigger] domain[i] < target,
    decreases domain.len() - position,
{
    if position >= domain.len() {
    } else if domain[position as int] >= target {
    } else {
        lemma_seek_from_skips_only_less_than_target(domain, position + 1, target);
        assert forall|i: int|
            position <= i < seek_from(domain, position, target) implies #[trigger] domain[i] < target
        by {
            if i == position {
            } else {
                assert(position + 1 <= i);
            }
        }
    }
}

pub proof fn lemma_seek_from_cursor_contract(domain: Seq<int>, position: nat, target: int)
    requires
        position <= domain.len(),
    ensures
        position <= seek_from(domain, position, target) <= domain.len(),
        seek_from(domain, position, target) == domain.len()
            || domain[seek_from(domain, position, target) as int] >= target,
        forall|i: int|
            position <= i < seek_from(domain, position, target) ==> #[trigger] domain[i] < target,
{
    lemma_seek_from_bounds(domain, position, target);
    lemma_seek_from_stops_at_end_or_target(domain, position, target);
    lemma_seek_from_skips_only_less_than_target(domain, position, target);
}

fn main() {}

}
