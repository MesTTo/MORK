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

/// F2: Mixed template (uses both components) - planned route must DECLINE (v1) and the
/// differential must still hold trivially.
const F2_MIXED: &str = r#"
(a 1) (a 2)
(b x)
(exec 0 (, (a $i) (b $j)) (, (pair $i $j)))
"#;

/// F4: schematic DATA (a stored fact contains a variable). Component routability must
/// decline or handle identically - the leapfrog silent-drop hazard class.
const F4_SCHEMATIC: &str = r#"
(a $q)
(a 1)
(b x)
(exec 0 (, (a $i) (b $j)) (, (seen-a $i) (done)))
"#;

/// F5: shared variable => ONE component - planned route must decline (no gain).
const F5_SHARED: &str = r#"
(a 1) (a 2)
(b 1)
(exec 0 (, (a $i) (b $i)) (, (seen $i)))
"#;

/// F6: compound column + repeated var inside a factor.
const F6_COMPOUND: &str = r#"
(a (f 1)) (a (f 2)) (a (g 1 1))
(b x)
(exec 0 (, (a (f $i)) (b $j)) (, (seen-f $i) (done)))
"#;

/// F7: duplicate-producing component (two factors in one component derive the same
/// binding twice) - set semantics must dedup identically.
const F7_DUPS: &str = r#"
(a 1) (a2 1)
(b x) (b y)
(exec 0 (, (a $i) (a2 $i) (b $j)) (, (seen $i)))
"#;

/// F8 hazard pin: NewVar in a template, THREADED across the template list. v1 declines
/// this shape (invariant I6); this test documents what stock does so any later widening
/// has an exact target.
const F8_NEWVAR_TEMPLATE: &str = r#"
(a 1)
(b x)
(exec 0 (, (a $i) (b $j)) (, (fresh $n) (also $n $i)))
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

#[test]
fn f2_mixed_stock_shape() {
    let (_, dump) = run_and_dump(F2_MIXED, 100);
    assert!(
        dump.contains("(pair 1 x)") && dump.contains("(pair 2 x)"),
        "{dump}"
    );
}

#[test]
fn f4_schematic_stock_shape() {
    let (_, dump) = run_and_dump(F4_SCHEMATIC, 100);
    assert_eq!(
        dump,
        "(a $a)\n(a 1)\n(b x)\n(done)\n(seen-a $a)\n(seen-a 1)"
    );
}

#[test]
fn f5_shared_stock_shape() {
    let (_, dump) = run_and_dump(F5_SHARED, 100);
    assert!(
        dump.contains("(seen 1)") && !dump.contains("(seen 2)"),
        "{dump}"
    );
}

#[test]
fn f6_compound_stock_shape() {
    let (_, dump) = run_and_dump(F6_COMPOUND, 100);
    assert!(
        dump.contains("(seen-f 1)") && dump.contains("(seen-f 2)"),
        "{dump}"
    );
    assert!(
        !dump.contains("(seen-f (g"),
        "g-facts must not match (f $i): {dump}"
    );
}

#[test]
fn f7_dups_stock_shape() {
    let (_, dump) = run_and_dump(F7_DUPS, 100);
    let count = dump.matches("(seen 1)").count();
    assert_eq!(count, 1, "set semantics: exactly one (seen 1): {dump}");
}

#[test]
fn f8_newvar_template_stock_shape() {
    let (_, dump) = run_and_dump(F8_NEWVAR_TEMPLATE, 100);
    assert_eq!(dump, "(a 1)\n(also $a 1)\n(b x)\n(fresh $a)");
}
