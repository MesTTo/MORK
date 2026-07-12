#![cfg(feature = "guarded_emit")]

use mork::space::Space;
use mork::{guarded_emit_stats, reset_guarded_emit_stats};
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());
const STRUCTURAL_GUARD_PROGRAM: &[u8] = b"(blocked (pair $x $x))
(seed (pair a a))
(seed (pair a b))
(exec 0 (, (seed $k)) (O (guard (blocked $k) (out $k))))
";

fn run(program: &[u8]) -> String {
    let mut s = Space::new();
    s.add_all_sexpr(program).unwrap();
    s.metta_calculus(100);
    let mut out = Vec::new();
    s.dump_all_sexpr(&mut out).unwrap();
    String::from_utf8_lossy(&out).into_owned()
}

#[test]
fn guard_drops_candidates_covered_by_a_repeated_variable_schema() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_guarded_emit_stats();
    let dump = run(STRUCTURAL_GUARD_PROGRAM);

    assert!(
        !dump.contains("(out (pair a a))"),
        "covered candidate was emitted:\n{dump}"
    );
    assert!(
        dump.contains("(out (pair a b))"),
        "uncovered candidate was not emitted:\n{dump}"
    );
    let stats = guarded_emit_stats();
    assert_eq!(stats.consulted, 2);
    assert_eq!(stats.dropped, 1);
}

#[test]
fn guard_decimal_bound_drops_only_when_stored_bound_is_greater_or_equal() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_guarded_emit_stats();
    let dump = run(b"(nogood (premise $x) 3)
(seed (premise a) 1)
(seed (premise b) 3)
(seed (premise c) 4)
(exec 0 (, (seed $k $b)) (O (guard (nogood $k $b) (sol $k $b))))
");

    assert!(
        !dump.contains("(sol (premise a) 1)"),
        "lower candidate bound was not pruned:\n{dump}"
    );
    assert!(
        !dump.contains("(sol (premise b) 3)"),
        "equal candidate bound was not pruned:\n{dump}"
    );
    assert!(
        dump.contains("(sol (premise c) 4)"),
        "higher candidate bound was pruned:\n{dump}"
    );
    let stats = guarded_emit_stats();
    assert_eq!(stats.consulted, 3);
    assert_eq!(stats.dropped, 2);
}

#[test]
fn guard_matches_reference_join_and_remove_program() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_guarded_emit_stats();
    let guarded = run(STRUCTURAL_GUARD_PROGRAM);
    let reference = run(b"(blocked (pair $x $x))
(seed (pair a a))
(seed (pair a b))
(exec 0 (, (seed $k)) (O (+ (out $k))))
(exec 1 (, (out $k) (blocked $k)) (O (- (out $k))))
");

    assert_eq!(guarded, reference);
}

#[test]
fn guard_rebases_output_variables_introduced_in_the_key() {
    let _guard = TEST_LOCK.lock().unwrap();
    reset_guarded_emit_stats();
    let dump = run(
        b"(seed b c)
(exec 0 (, (seed $b $c))
        (O (guard (nogood (pair $a $b) 0)
                  (out (pair $a $b) (pair $a $c)))))
",
    );

    assert!(
        dump.contains("(out (pair $a b) (pair $a c))"),
        "output kept dangling guard-introduced variables:\n{dump}"
    );
    let stats = guarded_emit_stats();
    assert_eq!(stats.consulted, 1);
    assert_eq!(stats.dropped, 0);
}
