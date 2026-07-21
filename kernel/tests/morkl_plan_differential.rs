//! Differential oracle for the MORKL symbolic plan route.
//!
//! Every fixture runs the same MM2 program on two Spaces - stock dispatch pinned off via
//! set_morkl_plan_dispatch(false), planned dispatch on - and asserts byte-identical
//! dump_all_sexpr output plus identical metta_calculus step counts. In PR1 the planned
//! route does not exist yet, so both sides run stock and the tests pin TODAY's behavior
//! (dumps recorded as insta-style string constants would rot; we compare live-vs-live).

use mork::space::Space;

/// Run `prog` to quiescence and return (steps, sorted dump).
fn run_and_dump(prog: &str, max_steps: usize) -> (usize, String) {
    let mut s = Space::new();
    s.add_all_sexpr(prog.as_bytes()).unwrap();
    let steps = s.metta_calculus(max_steps);
    let mut v = Vec::new();
    s.dump_all_sexpr(&mut v).unwrap();
    let mut lines: Vec<&str> = std::str::from_utf8(&v).unwrap().lines().collect();
    lines.sort_unstable();
    (steps, lines.join("\n"))
}

/// F1: two independent components, one SingleComponent template + one Ground template.
/// Planned route (PR3) must cut template applications from |A|*|B|*2 to |A|+1 while the
/// final space stays byte-identical.
const F1_HOIST: &str = r#"
(a 1) (a 2) (a 3)
(b x) (b y)
(exec 0 (, (a $i) (b $j)) (, (seen-a $i) (done)))
"#;

/// F3: empty component. (b ...) has no facts, so the body has NO solutions and NOTHING may
/// be emitted - not even the ground template. The existential trap fixture.
const F3_EMPTY_COMPONENT: &str = r#"
(a 1) (a 2)
(exec 0 (, (a $i) (b $j)) (, (seen-a $i) (done)))
"#;

#[test]
fn f1_stock_shape() {
    let (steps, dump) = run_and_dump(F1_HOIST, 100);
    assert_eq!(steps, 1);
    // The exec is consumed; outputs are the three seen-a plus done plus the base facts.
    for expected in [
        "(seen-a 1)",
        "(seen-a 2)",
        "(seen-a 3)",
        "(done)",
        "(a 1)",
        "(a 2)",
        "(a 3)",
        "(b x)",
        "(b y)",
    ] {
        assert!(dump.contains(expected), "missing {expected} in:\n{dump}");
    }
    assert!(!dump.contains("exec"), "exec must be consumed:\n{dump}");
}

#[test]
fn f3_empty_component_emits_nothing() {
    let (steps, dump) = run_and_dump(F3_EMPTY_COMPONENT, 100);
    assert_eq!(steps, 1); // the exec fires (and is consumed) even with zero matches
    assert!(
        !dump.contains("seen-a"),
        "no solutions => no emission:\n{dump}"
    );
    assert!(
        !dump.contains("(done)"),
        "ground template needs >=1 body solution:\n{dump}"
    );
}
