#![cfg(feature = "witness_select")]

use mork::{reset_witness_select_stats, witness_select_stats};
use mork::space::Space;
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn run(program: &[u8]) -> String {
    let mut s = Space::new();
    s.add_all_sexpr(program).unwrap();
    s.metta_calculus(100);
    let mut out = Vec::new();
    s.dump_all_sexpr(&mut out).unwrap();
    String::from_utf8_lossy(&out).into_owned()
}

/// The point of the sink: `guard` reads the pre-step snapshot, so same-key
/// siblings produced INSIDE one firing all survive it. `choose` keeps one.
#[test]
fn one_witness_per_key_within_a_firing() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_witness_select_stats();
    let dump = run(b"(src a p1)
(src a p2)
(src a p3)
(src b p1)
(src b p2)
(exec 0 (, (src $k $p)) (O (choose (idx $k) (out $k $p))
                           (+ (idx $k))))
");
    let outs = dump.lines().filter(|l| l.starts_with("(out ")).count();
    assert_eq!(outs, 2, "one witness per key, not one per payload:\n{dump}");
    assert!(dump.contains("(out a p1)"), "first in emit order wins:\n{dump}");
    assert!(dump.contains("(out b p1)"), "every key keeps a witness:\n{dump}");
    let (kept, dropped) = witness_select_stats();
    assert_eq!(kept, 2);
    assert_eq!(dropped, 3);
}

/// A `guard` under the same table would behave identically here: the states
/// are already in the snapshot.
#[test]
fn a_known_state_emits_no_witness() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_witness_select_stats();
    let dump = run(b"(idx a)
(src a p1)
(src a p2)
(src b p9)
(exec 0 (, (src $k $p)) (O (choose (idx $k) (out $k $p))
                           (+ (idx $k))))
");
    assert!(!dump.contains("(out a "), "known state re-emitted:\n{dump}");
    assert!(dump.contains("(out b p9)"), "new state lost:\n{dump}");
}

/// `choose` matches keys EXACTLY -- a De Bruijn key is already alpha-
/// canonical, so exact membership is state identity, and the probe is O(key)
/// no matter how large the state table grows. It deliberately does NOT
/// subsume instances under stored schemas: on the backward proof space that
/// was measured at 1.2x fewer states, against 47x for the payload collapse
/// this sink exists for, and it costs a branching walk over a table that
/// grows with the search. Subsumption remains `guard`'s job (see
/// guarded_emit.rs), and the two compose.
#[test]
fn keys_are_matched_exactly_not_by_subsumption() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_witness_select_stats();
    let dump = run(b"(idx (f a))
(src (f a) p1)
(src (f b) p2)
(src (g a) p3)
(src (g a) p4)
(exec 0 (, (src $k $p)) (O (choose (idx $k) (out $k $p))
                           (+ (idx $k))))
");
    assert!(!dump.contains("(out (f a)"), "known state re-emitted:\n{dump}");
    assert!(dump.contains("(out (f b) p2)"), "unknown state lost:\n{dump}");
    assert_eq!(
        dump.lines().filter(|l| l.starts_with("(out (g a)")).count(),
        1,
        "one witness per state:\n{dump}"
    );
    assert!(dump.contains("(out (g a) p3)"), "{dump}");
}

/// Distinct keys are never conflated, including keys that differ only inside
/// a compound (the De Bruijn trap the guard walk was fixed for).
#[test]
fn distinct_keys_each_keep_a_witness() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_witness_select_stats();
    let dump = run(b"(src (k a a) p1)
(src (k a b) p2)
(src (k b a) p3)
(exec 0 (, (src $k $p)) (O (choose (idx $k) (out $k $p))
                           (+ (idx $k))))
");
    assert_eq!(
        dump.lines().filter(|l| l.starts_with("(out ")).count(),
        3,
        "distinct keys were conflated:\n{dump}"
    );
}

/// Rounds compose: a witness kept in one round puts its state in the index,
/// so later rounds neither re-derive it nor multiply its payload.
#[test]
fn witnesses_do_not_accumulate_across_rounds() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_witness_select_stats();
    let dump = run(b"(step 0)
(nxt 0 1)
(nxt 1 2)
(src p1)
(src p2)
(src p3)
((grow rule)
  (, ((grow rule) $sp $st)
     (step $n)
     (nxt $n $m)
     (src $p))
  (O (choose (idx $m) (out $m $p))
     (+ (idx $m))))
((tick rule)
  (, ((tick rule) $ap $at)
     ((grow rule) $gp $gt)
     (step $n)
     (nxt $n $m))
  (O (- (step $n))
     (+ (step $m))
     (+ (exec (30 grow) $gp $gt))
     (+ (exec (quiesce 31 tick) $ap $at))))
(exec (10 init)
  (, ((grow rule) $gp $gt)
     ((tick rule) $ap $at))
  (, (armed init)
     (exec (30 grow) $gp $gt)
     (exec (quiesce 31 tick) $ap $at)))
");
    let outs: Vec<&str> = dump.lines().filter(|l| l.starts_with("(out ")).collect();
    assert_eq!(
        outs.len(),
        2,
        "one witness per reached state, not one per (state, payload):\n{dump}"
    );
}
