use vstd::prelude::*;

verus! {

// MORK runs a `,`-conjunction transform two ways. For a cyclic body it uses the
// worst-case-optimal sidecar join (`transform_via_sidecar`), which interns each
// joined relation as ground relational tuples and matches a query factor by
// equality on the shared columns. Otherwise, and as the fallback, it uses the
// ProductZipper walk plus first-order `unify`. The ProductZipper path is the
// complete matcher: a data-side variable in a stored fact captures the query
// subterm at that position, and a data-side `VarRef` enforces coreference.
//
// The relational join is correct only on ground stored facts. A schematic stored
// fact, one carrying a `NewVar` or `VarRef` such as `(if (S $n) $x $y $x)`, is
// not a ground tuple. Matching it needs unification, which the join does not
// perform, so the join silently drops those matches. The counter_machine
// original `(if ...)` meta form stalled exactly this way. The runtime fix makes
// the sidecar decline a body whose joined relations hold any schematic fact via
// `TermIdentitySidecar::any_schematic_fact_under_prefixes`, falling back to the
// complete ProductZipper plus `unify` path.
//
// This file models the matcher abstractly and proves the condition the fix
// relies on:
//   - the relational join is sound. It never produces a wrong match.
//   - it is complete on ground facts. It agrees with `unify` there.
//   - it is incomplete on schematic facts. A witnessed match is dropped.
//   - the routed matcher equals the complete `unify` matcher on every input.
//
// A term is a symbol, a data variable, or a compound. `Var` models a variable
// occurring in the stored data-side fact. Under complete semantics it captures
// any query subterm at its position. The query is ground because the join's
// shared columns are instantiated to concrete values before a factor is probed,
// and the transform's apply also requires a ground result.

pub enum Term {
    Sym(nat),
    Var,
    Cmp(Seq<Term>),
}

pub open spec fn ground(t: Term) -> bool
    decreases t,
{
    match t {
        Term::Sym(_) => true,
        Term::Var => false,
        Term::Cmp(ts) => forall|i: int| 0 <= i < ts.len() ==> ground(#[trigger] ts[i]),
    }
}

// Complete ProductZipper plus `unify` semantics. A data-side `Var` in the
// stored fact matches any query subterm. Otherwise shapes must agree and
// children match pairwise.
pub open spec fn complete_match(q: Term, f: Term) -> bool
    decreases f,
{
    match f {
        Term::Var => true,
        Term::Sym(b) => match q {
            Term::Sym(a) => a == b,
            _ => false,
        },
        Term::Cmp(fs) => match q {
            Term::Cmp(qs) => qs.len() == fs.len() && forall|i: int|
                0 <= i < fs.len() ==> complete_match(#[trigger] qs[i], fs[i]),
            _ => false,
        },
    }
}

// Relational sidecar matching. The sidecar can equate only ground tuples, so a
// schematic fact does not participate in this match relation.
pub open spec fn relational_match(q: Term, f: Term) -> bool {
    ground(f) && q == f
}

pub proof fn relational_is_sound(q: Term, f: Term)
    requires
        relational_match(q, f),
    ensures
        complete_match(q, f),
    decreases f,
{
    assert(ground(f));
    assert(q == f);
    ground_complete_matches_itself(f);
}

pub proof fn ground_complete_matches_itself(f: Term)
    requires
        ground(f),
    ensures
        complete_match(f, f),
    decreases f,
{
    match f {
        Term::Sym(_) => {},
        Term::Var => {
            assert(false);
        },
        Term::Cmp(fs) => {
            assert forall|i: int| 0 <= i < fs.len() implies complete_match(#[trigger] fs[i], fs[i]) by {
                assert(ground(fs[i]));
                ground_complete_matches_itself(fs[i]);
            }
        },
    }
}

pub proof fn relational_complete_on_ground(q: Term, f: Term)
    requires
        ground(f),
    ensures
        relational_match(q, f) <==> complete_match(q, f),
    decreases f,
{
    if complete_match(q, f) {
        complete_match_on_ground_is_equality(q, f);
        assert(q == f);
        assert(relational_match(q, f));
    }
    if relational_match(q, f) {
        relational_is_sound(q, f);
    }
}

pub proof fn complete_match_on_ground_is_equality(q: Term, f: Term)
    requires
        ground(f),
        complete_match(q, f),
    ensures
        q == f,
    decreases f,
{
    match f {
        Term::Sym(_b) => {},
        Term::Var => {
            assert(false);
        },
        Term::Cmp(fs) => {
            match q {
                Term::Sym(_) => {},
                Term::Var => {},
                Term::Cmp(qs) => {
                    assert forall|i: int| 0 <= i < fs.len() implies qs[i] == fs[i] by {
                        assert(ground(fs[i]));
                        assert(complete_match(qs[i], fs[i]));
                        complete_match_on_ground_is_equality(qs[i], fs[i]);
                    }
                    assert(qs =~= fs);
                },
            }
        },
    }
}

pub proof fn schematic_breaks_relational()
    ensures
        exists|q: Term, f: Term|
            !ground(f) && #[trigger] complete_match(q, f) && !relational_match(q, f),
{
    let f = Term::Var;
    let q = Term::Cmp(seq![Term::Sym(0), Term::Sym(1)]);
    assert(!ground(f));
    assert(complete_match(q, f));
    assert(!ground(f));
    assert(!relational_match(q, f));
    assert(!ground(f) && complete_match(q, f) && !relational_match(q, f));
}

pub open spec fn routed_match(q: Term, f: Term) -> bool {
    if ground(f) {
        relational_match(q, f)
    } else {
        complete_match(q, f)
    }
}

pub proof fn routed_equals_complete(q: Term, f: Term)
    ensures
        routed_match(q, f) <==> complete_match(q, f),
{
    if ground(f) {
        relational_complete_on_ground(q, f);
        assert(routed_match(q, f) == relational_match(q, f));
    } else {
        assert(routed_match(q, f) == complete_match(q, f));
    }
}

pub proof fn routed_sound_and_complete(q: Term, f: Term)
    ensures
        routed_match(q, f) ==> complete_match(q, f),
        complete_match(q, f) ==> routed_match(q, f),
{
    routed_equals_complete(q, f);
}

fn main() {}

} // verus!
