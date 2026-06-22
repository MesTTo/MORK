use vstd::prelude::*;

verus! {

// The pattern-factor renumbering (`Space::lower_pattern_factor`) rewrites a query
// factor into a self-contained schematic term: the first occurrence of each
// variable becomes a `NewVar` assigned the next slot, and every later occurrence
// becomes a `VarRef` to that slot. A cross-factor reference, whose variable was
// introduced in another factor, is itself a first occurrence here and so also
// becomes a `NewVar`.
//
// This proves the safety property the interning relies on: every emitted `VarRef`
// slot is a valid back-reference, strictly less than the number of variables
// introduced so far, so the term handed to `insert_term` is well-formed with no
// dangling reference.
//
// State: `slots` maps each introduced variable key to its slot; `count` is the
// number of variables introduced (the number of `NewVar`s emitted). Slots are
// assigned 0, 1, 2, ... in first-occurrence order.

/// The renumber invariant: every assigned slot is a valid index, strictly less
/// than the number of variables introduced so far.
pub open spec fn renumber_inv(slots: Map<u64, nat>, count: nat) -> bool {
    forall|k: u64| #![auto] slots.contains_key(k) ==> slots[k] < count
}

/// The empty state (no variable introduced yet) satisfies the invariant.
pub proof fn renumber_start()
    ensures
        renumber_inv(Map::<u64, nat>::empty(), 0),
{
}

/// Introducing a variable on its first occurrence assigns it the current slot
/// `count` and bumps the count to `count + 1`. The invariant is preserved: the
/// new slot equals `count` (less than `count + 1`), and every previously assigned
/// slot was less than `count` (so less than `count + 1`).
pub proof fn renumber_introduce(slots: Map<u64, nat>, count: nat, key: u64)
    requires
        renumber_inv(slots, count),
        !slots.contains_key(key),
    ensures
        renumber_inv(slots.insert(key, count), (count + 1) as nat),
{
    let next = slots.insert(key, count);
    assert forall|k: u64| #![auto] next.contains_key(k) implies next[k] < (count + 1) as nat by {
        if k == key {
            assert(next[k] == count);
        } else {
            assert(slots.contains_key(k));
            assert(slots[k] < count);
        }
    }
}

/// A repeated occurrence of an already-introduced variable emits
/// `VarRef(slots[key])`. That slot is a valid back-reference, strictly less than
/// the number of variables introduced so far, so it points at a `NewVar` already
/// emitted.
pub proof fn renumber_reference(slots: Map<u64, nat>, count: nat, key: u64)
    requires
        renumber_inv(slots, count),
        slots.contains_key(key),
    ensures
        slots[key] < count,
{
}

/// Final `(slots, count)` after processing a sequence of variable occurrences in
/// order: each occurrence of a key not yet seen introduces it (bumping the
/// count), each repeat leaves the state unchanged.
pub open spec fn renumber_fold(keys: Seq<u64>) -> (Map<u64, nat>, nat)
    decreases keys.len(),
{
    if keys.len() == 0 {
        (Map::<u64, nat>::empty(), 0nat)
    } else {
        let prev = renumber_fold(keys.drop_last());
        let key = keys.last();
        if prev.0.contains_key(key) {
            prev
        } else {
            (prev.0.insert(key, prev.1), (prev.1 + 1) as nat)
        }
    }
}

/// Processing any sequence of variable occurrences from the empty state keeps the
/// invariant. By induction over the sequence, reusing the per-step lemmas: every
/// reference emitted along the way is therefore a valid back-reference
/// (`renumber_reference`), so the whole renumbered term is well-formed.
pub proof fn renumber_fold_inv(keys: Seq<u64>)
    ensures
        renumber_inv(renumber_fold(keys).0, renumber_fold(keys).1),
    decreases keys.len(),
{
    if keys.len() == 0 {
        renumber_start();
    } else {
        renumber_fold_inv(keys.drop_last());
        let prev = renumber_fold(keys.drop_last());
        let key = keys.last();
        if !prev.0.contains_key(key) {
            renumber_introduce(prev.0, prev.1, key);
        }
    }
}

fn main() {}

}
