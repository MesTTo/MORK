use vstd::prelude::*;

verus! {

// A hand-encoded state machine (state type + init predicate + step relation +
// inductive safety invariant) for the streaming maintained-closure protocol in
// binding_space.rs / space.rs. The macro-based state_machine! framework emits
// trusted, unverified token constructors that the --no-cheating harness rejects
// (vstd verifies those macros under its own trusted build, not no-cheating), so
// the protocol is modeled directly here. This is the same inductive-invariant
// method without the trusted token machinery.
//
// The streaming closure reads new facts from facts_for_relation(head)[watermark..]
// and folds each into the maintained relation, then advances the watermark. The
// safety property is that the watermark never runs past the fact stream and the
// maintained aggregate is exactly the fold over the folded prefix, so advancing
// the watermark never reprocesses a fact and never skips one.

/// Protocol state. Each fact carries an integer weight; for the maintained
/// transitive closure that weight is the count of new pairs the streamed edge
/// produces. `total` abstracts the maintained aggregate and `watermark` counts
/// the facts already folded into it.
pub struct StreamState {
    pub facts: Seq<int>,
    pub watermark: nat,
    pub total: int,
}

/// Fold of the first `n` fact weights.
pub open spec fn prefix_sum(deltas: Seq<int>, n: int) -> int
    decreases n,
{
    if n <= 0 {
        0
    } else {
        prefix_sum(deltas, n - 1) + deltas[n - 1]
    }
}

/// Safety invariant: the watermark stays within the stream and the maintained
/// total equals the fold over exactly the folded prefix.
pub open spec fn inv(s: StreamState) -> bool {
    &&& s.watermark <= s.facts.len()
    &&& s.total == prefix_sum(s.facts, s.watermark as int)
}

/// Initial state: empty stream, nothing folded, zero total.
pub open spec fn init(s: StreamState) -> bool {
    &&& s.facts == Seq::<int>::empty()
    &&& s.watermark == 0
    &&& s.total == 0
}

/// Append a fact (a write into the space). The watermark and the maintained
/// total are untouched: a write does not retroactively fold itself in.
pub open spec fn step_append(s: StreamState, t: StreamState, f: int) -> bool {
    &&& t.facts == s.facts.push(f)
    &&& t.watermark == s.watermark
    &&& t.total == s.total
}

/// Fold the next unfolded fact and advance the watermark by one (one streamed
/// edge handed to insert_edge).
pub open spec fn step_fold_next(s: StreamState, t: StreamState) -> bool {
    &&& s.watermark < s.facts.len()
    &&& t.facts == s.facts
    &&& t.watermark == s.watermark + 1
    &&& t.total == s.total + s.facts[s.watermark as int]
}

/// One protocol step is either an append or a fold.
pub open spec fn step(s: StreamState, t: StreamState) -> bool {
    ||| (exists|f: int| step_append(s, t, f))
    ||| step_fold_next(s, t)
}

/// Appending past index `n` leaves the fold of the first `n` weights unchanged,
/// which is why a write never disturbs the already-maintained total.
proof fn push_preserves_prefix_sum(deltas: Seq<int>, f: int, n: int)
    requires
        0 <= n <= deltas.len(),
    ensures
        prefix_sum(deltas.push(f), n) == prefix_sum(deltas, n),
    decreases n,
{
    if n > 0 {
        push_preserves_prefix_sum(deltas, f, n - 1);
        assert(deltas.push(f)[n - 1] == deltas[n - 1]);
    }
}

/// The initial state satisfies the invariant.
proof fn init_establishes_inv(s: StreamState)
    requires
        init(s),
    ensures
        inv(s),
{
}

/// Every protocol step preserves the invariant: the inductive safety proof.
proof fn step_preserves_inv(s: StreamState, t: StreamState)
    requires
        inv(s),
        step(s, t),
    ensures
        inv(t),
{
    if exists|f: int| step_append(s, t, f) {
        let f = choose|f: int| step_append(s, t, f);
        push_preserves_prefix_sum(s.facts, f, s.watermark as int);
    } else {
        // The remaining disjunct is the fold step; prefix_sum unfolds by one at
        // index watermark, matching the added fact weight.
    }
}

/// Corollary for an insert-only stream: when every fact weight is nonnegative
/// (Italiano inserts only add pairs), folding never shrinks the maintained
/// total. This is the monotonicity the watermark advance relies on.
proof fn fold_next_monotone(s: StreamState, t: StreamState)
    requires
        inv(s),
        step_fold_next(s, t),
        forall|k: int| 0 <= k < s.facts.len() ==> s.facts[k] >= 0,
    ensures
        t.total >= s.total,
{
    assert(s.facts[s.watermark as int] >= 0);
}

fn main() {
}

}
