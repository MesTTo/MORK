use vstd::prelude::*;

verus! {

pub open spec fn valid_slot(slot: nat) -> bool {
    slot < 64
}

pub open spec fn term_matches(bindings: Map<nat, int>, slot: nat, term: int) -> bool {
    bindings.dom().contains(slot) && bindings[slot] == term
}

pub open spec fn bind_ok(bindings: Map<nat, int>, slot: nat, term: int) -> bool {
    !bindings.dom().contains(slot) || bindings[slot] == term
}

pub open spec fn bind_state(
    bindings: Map<nat, int>,
    trail: Seq<nat>,
    slot: nat,
    term: int,
) -> (Map<nat, int>, Seq<nat>) {
    if !valid_slot(slot) {
        (bindings, trail)
    } else if !bindings.dom().contains(slot) {
        (bindings.insert(slot, term), trail.push(slot))
    } else {
        (bindings, trail)
    }
}

pub open spec fn rollback_state(
    bindings: Map<nat, int>,
    trail: Seq<nat>,
    mark: nat,
) -> (Map<nat, int>, Seq<nat>)
    recommends
        mark <= trail.len(),
    decreases trail.len() - mark,
{
    if trail.len() <= mark {
        (bindings, trail)
    } else {
        let slot = trail[trail.len() as int - 1];
        let next_bindings = bindings.remove(slot);
        let next_trail = trail.drop_last();
        rollback_state(next_bindings, next_trail, mark)
    }
}

pub proof fn lemma_bind_new_slot_records_term_and_extends_trail(
    bindings: Map<nat, int>,
    trail: Seq<nat>,
    slot: nat,
    term: int,
)
    requires
        valid_slot(slot),
        !bindings.dom().contains(slot),
    ensures
        bind_state(bindings, trail, slot, term).0.dom().contains(slot),
        bind_state(bindings, trail, slot, term).0[slot] == term,
        bind_state(bindings, trail, slot, term).1 == trail.push(slot),
{
    assert(bind_state(bindings, trail, slot, term).0 =~= bindings.insert(slot, term));
}

pub proof fn lemma_bind_same_term_does_not_extend_trail(
    bindings: Map<nat, int>,
    trail: Seq<nat>,
    slot: nat,
    term: int,
)
    requires
        bindings.dom().contains(slot),
        bindings[slot] == term,
    ensures
        bind_ok(bindings, slot, term),
        bind_state(bindings, trail, slot, term).0 == bindings,
        bind_state(bindings, trail, slot, term).1 == trail,
        term_matches(bind_state(bindings, trail, slot, term).0, slot, term),
{
}

pub proof fn lemma_bind_conflict_does_not_mutate(
    bindings: Map<nat, int>,
    trail: Seq<nat>,
    slot: nat,
    existing: int,
    incoming: int,
)
    requires
        bindings.dom().contains(slot),
        bindings[slot] == existing,
        existing != incoming,
    ensures
        !bind_ok(bindings, slot, incoming),
        bind_state(bindings, trail, slot, incoming).0 == bindings,
        bind_state(bindings, trail, slot, incoming).1 == trail,
        bind_state(bindings, trail, slot, incoming).0[slot] == existing,
{
}

pub proof fn lemma_rollback_to_current_mark_is_identity(
    bindings: Map<nat, int>,
    trail: Seq<nat>,
)
    ensures
        rollback_state(bindings, trail, trail.len()).0 == bindings,
        rollback_state(bindings, trail, trail.len()).1 == trail,
{
}

pub proof fn lemma_rollback_removes_latest_binding(
    bindings: Map<nat, int>,
    trail: Seq<nat>,
    slot: nat,
    term: int,
)
    requires
        valid_slot(slot),
        !bindings.dom().contains(slot),
    ensures
        rollback_state(
            bind_state(bindings, trail, slot, term).0,
            bind_state(bindings, trail, slot, term).1,
            trail.len(),
        ).0 =~= bindings,
        rollback_state(
            bind_state(bindings, trail, slot, term).0,
            bind_state(bindings, trail, slot, term).1,
            trail.len(),
        ).1 =~= trail,
{
    lemma_bind_new_slot_records_term_and_extends_trail(bindings, trail, slot, term);
    let next = bind_state(bindings, trail, slot, term);
    assert(next.1.drop_last() == trail);
    assert(next.0.remove(slot) =~= bindings);
    assert(next.1.len() == trail.len() + 1);
    assert(next.1[next.1.len() as int - 1] == slot);
    assert(rollback_state(next.0, next.1, trail.len()) =~= rollback_state(
        next.0.remove(slot),
        next.1.drop_last(),
        trail.len(),
    ));
    lemma_rollback_to_current_mark_is_identity(next.0.remove(slot), next.1.drop_last());
    assert(rollback_state(next.0, next.1, trail.len()).0 =~= bindings);
    assert(rollback_state(next.0, next.1, trail.len()).1 =~= trail);
}

pub proof fn lemma_slot_bit_is_inside_bound_mask_domain(slot: u64)
    requires
        slot < 64,
    ensures
        (1u64 << slot) != 0,
        ((1u64 << slot) & !(1u64 << slot)) == 0,
{
    assert((1u64 << slot) != 0) by(bit_vector)
        requires
            slot < 64,
    ;
    assert(((1u64 << slot) & !(1u64 << slot)) == 0) by(bit_vector)
        requires
            slot < 64,
    ;
}

fn main() {}

}
