use vstd::prelude::*;

verus! {

// Companion to VarRefRecheck.rs. That proof covers the GROUND fast-path of the
// compiled VarRef re-check (`match_varref_program`). This one covers the NON-GROUND
// branch: the query variable was bound at its first occurrence to a data subterm that
// itself contains variables (e.g. matching `(implies ($P $x) (Green $x))` against the
// stored `(implies (Frog $x) (Green $x))` binds the query `$x` to the data variable
// `$x`; re-checking the second `$x` lands here).
//
// At the re-check position the data exposes a list of child branches. Each is either a
// data VARIABLE (a wildcard) or a CONCRETE byte. Two compiled strategies exist:
//
//   * `vs_match_program`        — descends each VARIABLE child and continues the program
//                                 (the data-side-capture shortcut): one per variable child.
//   * `rematch_bound_then_program` — re-matches the bound value via
//                                 `coreferential_transition`; the bound value is a single
//                                 variable, which descends EVERY child once: one per child.
//
// The interpreted oracle `coreferential_transition` descends every child exactly once and
// is the reference (it emits once for the a3 case; verified by trace). The defect: the old
// non-ground branch ran BOTH strategies, so every data-variable child was counted twice
// (once captured by `vs`, once re-descended by the rematch). The fix runs the rematch
// alone. These theorems pin the over-count to exactly the data-variable children and prove
// the fix equals the oracle.

pub enum Child {
    Var,
    Concrete,
}

pub open spec fn is_var(c: Child) -> bool {
    match c {
        Child::Var => true,
        Child::Concrete => false,
    }
}

// Reference: the interpreted oracle descends each child once.
pub open spec fn oracle_count(cs: Seq<Child>) -> nat {
    cs.len() as nat
}

// Number of data-variable children at this position.
pub open spec fn var_children(cs: Seq<Child>) -> nat
    decreases cs.len(),
{
    if cs.len() == 0 {
        0
    } else {
        (if is_var(cs[0]) { 1nat } else { 0nat }) + var_children(cs.subrange(1, cs.len() as int))
    }
}

// The fix: rematch alone, one solution per child (variable or concrete).
pub open spec fn fixed_count(cs: Seq<Child>) -> nat
    decreases cs.len(),
{
    if cs.len() == 0 {
        0
    } else {
        1nat + fixed_count(cs.subrange(1, cs.len() as int))
    }
}

// The defect: `vs` (one per variable child) PLUS rematch (one per child). A variable
// child is therefore counted twice, a concrete child once.
pub open spec fn buggy_count(cs: Seq<Child>) -> nat
    decreases cs.len(),
{
    if cs.len() == 0 {
        0
    } else {
        (if is_var(cs[0]) { 2nat } else { 1nat }) + buggy_count(cs.subrange(1, cs.len() as int))
    }
}

// THEOREM 1 (the fix equals the oracle): rematch alone descends each child once, so the
// fixed compiled path returns exactly the oracle's solution count.
pub proof fn fixed_equals_oracle(cs: Seq<Child>)
    ensures
        fixed_count(cs) == oracle_count(cs),
    decreases cs.len(),
{
    if cs.len() == 0 {
    } else {
        fixed_equals_oracle(cs.subrange(1, cs.len() as int));
    }
}

// THEOREM 2 (the defect over-counts by exactly the data-variable children): the old
// non-ground path's count is the oracle's plus one for every data-variable child.
pub proof fn buggy_overcounts_by_var_children(cs: Seq<Child>)
    ensures
        buggy_count(cs) == oracle_count(cs) + var_children(cs),
    decreases cs.len(),
{
    if cs.len() == 0 {
    } else {
        buggy_overcounts_by_var_children(cs.subrange(1, cs.len() as int));
    }
}

// COROLLARY (the fix removes exactly the spurious solutions): the defect exceeds the fix
// by the data-variable count, and they coincide iff the matched data has no variable
// child at the re-check position. The a3 case has one such child ($x), hence the 2-vs-1.
pub proof fn fix_removes_exactly_the_double_count(cs: Seq<Child>)
    ensures
        buggy_count(cs) == fixed_count(cs) + var_children(cs),
        buggy_count(cs) == fixed_count(cs) <==> var_children(cs) == 0,
{
    fixed_equals_oracle(cs);
    buggy_overcounts_by_var_children(cs);
}

} // verus!

fn main() {}
