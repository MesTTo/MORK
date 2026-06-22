use vstd::prelude::*;

verus! {

/// The bilinear delta identity underlying `BindingSpace::delta_join`.
///
/// `delta_join` computes `delta_left join new_right` plus `old_left join
/// delta_right`, where `new_right = old_right + delta_right`. At the per-tuple
/// signed-weight level (the ring the join's weight multiplication and the delta
/// sum obey), this equals the true delta of the join:
///
/// `(a + da) * (b + db) - a * b == da * (b + db) + a * db`.
///
/// `a` and `b` are the old left and right weights for a matching tuple, `da` and
/// `db` the deltas. The left side is the change in the joined weight; the right
/// side is exactly what `delta_join` accumulates. So the maintained join stays
/// equal to a recomputation, which is the correctness `delta_join` relies on.
pub proof fn delta_join_bilinear(a: int, da: int, b: int, db: int)
    ensures
        (a + da) * (b + db) - a * b == da * (b + db) + a * db,
{
    assert((a + da) * (b + db) - a * b == da * (b + db) + a * db) by (nonlinear_arith);
}

/// Symmetric form using both new sides, matching the more common textbook
/// statement `D(A join B) = dA join B + A join B + dA join dB` consolidated.
/// Here it is the same ring identity split as `da*b + a*db + da*db`.
pub proof fn delta_join_expanded(a: int, da: int, b: int, db: int)
    ensures
        (a + da) * (b + db) - a * b == da * b + a * db + da * db,
{
    assert((a + da) * (b + db) - a * b == da * b + a * db + da * db) by (nonlinear_arith);
}

/// Right distribution of a sum over a product, `(x + y) * z == x*z + y*z`. A
/// degree-two fact (x, y, z are atoms here), so `nonlinear_arith` proves it
/// reliably regardless of the surrounding context. The three-factor lemma below
/// instantiates this to discharge its one cubic step instead of asking Z3 to
/// solve a three-factor nonlinear goal in a single shot.
proof fn mul_distributes_right(x: int, y: int, z: int)
    ensures
        (x + y) * z == x * z + y * z,
{
    assert((x + y) * z == x * z + y * z) by (nonlinear_arith);
}

/// The 3-factor telescoping identity that `delta_multi_join` obeys for a tuple
/// matching all three factors: the change in the product equals the sum of the
/// per-factor delta terms, with earlier factors old and later factors new. This
/// is the `n = 3` case of the multiway delta join, certifying that summing the
/// terms equals a full recomputation at the per-tuple weight level.
pub proof fn delta_multi_join_three(a: int, da: int, b: int, db: int, c: int, dc: int)
    ensures
        (a + da) * (b + db) * (c + dc) - a * b * c == da * (b + db) * (c + dc) + a * db * (c + dc)
            + a * b * dc,
{
    // Derive the cubic from already-proven degree-two lemmas, so Z3 never solves
    // a three-factor nonlinear goal directly (that single shot is context
    // sensitive and was flaking). p_old and p_new are the old and new products
    // of the first two factors; note a*b*c parses as (a*b)*c == p_old*c and the
    // triple product as ((a+da)*(b+db))*(c+dc) == p_new*(c+dc).
    let p_old = a * b;
    let p_new = (a + da) * (b + db);
    // Two-factor delta on the first two factors:
    //   p_new - p_old == da*(b+db) + a*db.
    delta_join_bilinear(a, da, b, db);
    // Two-factor delta of that combined factor against the third:
    //   p_new*(c+dc) - p_old*c == (p_new - p_old)*(c+dc) + p_old*dc.
    delta_join_bilinear(p_old, p_new - p_old, c, dc);
    // Distribute (p_new - p_old) == da*(b+db) + a*db across (c+dc).
    mul_distributes_right(da * (b + db), a * db, c + dc);
}

/// Set membership read off a signed support count flips only at the zero
/// boundary. A tuple is visible exactly when its count is positive, so it can
/// only appear when the count rises from zero and only vanish when it returns to
/// zero. This is what lets the maintained views and the semi-naive fixpoint
/// publish a tuple the first time its count becomes positive and retract it only
/// when the count falls back to zero, rather than re-deriving the whole relation.
pub proof fn positive_visibility_transition(old_count: int, delta: int)
    requires
        old_count >= 0,
        old_count + delta >= 0,
    ensures
        (old_count == 0 && old_count + delta > 0) ==> (!(old_count > 0) && old_count + delta > 0),
        (old_count > 0 && old_count + delta == 0) ==> (old_count > 0 && !(old_count + delta > 0)),
{
}

/// Under insert-only maintenance every delta is nonnegative, so a tuple already
/// visible stays visible. The maintained transitive closure depends on this:
/// Italiano's insert step only adds ancestor-descendant pairs, never removes
/// one, so a reachable pair stays reachable as later edges arrive.
pub proof fn insert_only_keeps_visible(old_count: int, delta: int)
    requires
        old_count >= 0,
        delta >= 0,
    ensures
        (old_count > 0) ==> (old_count + delta > 0),
{
}

/// Running total after folding the first `n` deltas of an insert stream.
pub open spec fn prefix_sum(deltas: Seq<int>, n: int) -> int
    decreases n,
{
    if n <= 0 {
        0
    } else {
        prefix_sum(deltas, n - 1) + deltas[n - 1]
    }
}

/// Folding a stream of nonnegative deltas grows the running total
/// monotonically. Streaming edges into the maintained closure adds a nonnegative
/// count of new pairs per edge, so the closure size never shrinks while edges
/// are inserted. That monotonicity is what makes advancing the watermark over
/// already-folded facts sound: past contributions are never undone.
pub proof fn prefix_sum_monotone(deltas: Seq<int>, i: int, j: int)
    requires
        0 <= i <= j <= deltas.len(),
        forall|k: int| 0 <= k < deltas.len() ==> deltas[k] >= 0,
    ensures
        prefix_sum(deltas, i) <= prefix_sum(deltas, j),
    decreases j - i,
{
    if i < j {
        prefix_sum_monotone(deltas, i, j - 1);
        assert(deltas[j - 1] >= 0);
    }
}

fn main() {}

}
