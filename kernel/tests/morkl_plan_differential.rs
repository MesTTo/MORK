//! Differential oracle for the MORKL symbolic plan route.
//!
//! Every fixture runs the same MM2 program on two Spaces. Stock dispatch is pinned off and planned
//! dispatch is pinned on before each corresponding step. The runner asserts byte-identical
//! `dump_all_sexpr` output and identical `metta_calculus` step counts.

use std::sync::Mutex;
use std::sync::atomic::Ordering;

use mork::morkl_plan::{PLANNED_FIRINGS, planned_firings, set_morkl_plan_dispatch};
use mork::space::Space;

static ROUTING_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Run `prog` to quiescence and return (steps, sorted dump).
fn run_and_dump(prog: &str, max_steps: usize) -> (usize, String) {
    set_morkl_plan_dispatch(false);
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

/// F6B: repeated query variable inside one factor. The factor itself constrains both
/// columns to the same value, independently of the second body component.
const F6B_REPEATED_VAR: &str = r#"
(p 1 2) (p 2 2)
(b x)
(exec 0 (, (p $i $i) (b $j)) (, (seen $i) (done)))
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
fn f6b_repeated_var_stock_shape() {
    let (_, dump) = run_and_dump(F6B_REPEATED_VAR, 100);
    assert_eq!(dump, "(b x)\n(done)\n(p 1 2)\n(p 2 2)\n(seen 2)");
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

/// Run one program on stock and planned routes and return the planned emission count.
fn assert_lockstep(prog: &str, max_steps: usize) -> usize {
    PLANNED_FIRINGS.store(0, Ordering::Relaxed);
    let mut stock = Space::new();
    let mut plan = Space::new();
    stock.add_all_sexpr(prog.as_bytes()).unwrap();
    plan.add_all_sexpr(prog.as_bytes()).unwrap();
    for step in 0..max_steps {
        set_morkl_plan_dispatch(false);
        let a = stock.metta_calculus(1);
        set_morkl_plan_dispatch(true);
        let b = plan.metta_calculus(1);
        assert_eq!(a, b, "step-count divergence at step {step}");
        let (mut va, mut vb) = (Vec::new(), Vec::new());
        stock.dump_all_sexpr(&mut va).unwrap();
        plan.dump_all_sexpr(&mut vb).unwrap();
        assert_eq!(va, vb, "space divergence after step {step}");
        if a == 0 {
            break;
        }
    }
    set_morkl_plan_dispatch(false);
    planned_firings()
}

#[test]
fn lockstep_all_fixtures() {
    let _guard = ROUTING_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (name, prog, should_emit) in [
        ("F1", F1_HOIST, Some(true)),
        ("F2", F2_MIXED, Some(false)),
        ("F3", F3_EMPTY_COMPONENT, None),
        ("F4", F4_SCHEMATIC, None),
        ("F5", F5_SHARED, Some(false)),
        ("F6", F6_COMPOUND, Some(true)),
        ("F6B", F6B_REPEATED_VAR, Some(true)),
        ("F7", F7_DUPS, Some(true)),
        ("F8", F8_NEWVAR_TEMPLATE, Some(false)),
    ] {
        let firings = assert_lockstep(prog, 32);
        println!("{name} PLANNED_FIRINGS={firings}");
        match should_emit {
            Some(true) => assert!(firings > 0, "{name} must emit through the planned route"),
            Some(false) => assert_eq!(firings, 0, "{name} must decline the planned route"),
            None => {}
        }
    }
}

/// Deterministic pseudo-random program generator for property differentials: k relations,
/// random facts, one exec whose body samples 2-4 relations with fresh or shared vars.
/// Uses a hand-rolled LCG so the corpus is reproducible without new dependencies.
fn gen_program(seed: u64, inject_common_fact: bool) -> (String, bool) {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = move |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % m
    };
    let nrel = 2 + next(3) as usize; // 2..=4 relations r0..r3
    let mut prog = String::new();
    for r in 0..nrel {
        // Keep every shared-variable component satisfiable so a structurally qualifying case must
        // emit and can prove planned-route engagement.
        if inject_common_fact {
            prog.push_str(&format!("(r{r} v0)\n"));
        }
        for _ in 0..(1 + next(5)) {
            prog.push_str(&format!("(r{r} v{})\n", next(4)));
        }
    }
    let mut body = String::new();
    let mut vars_used = 0usize;
    for r in 0..nrel {
        let v = if vars_used > 0 && next(3) == 0 {
            next(vars_used as u64) as usize
        } else {
            vars_used += 1;
            vars_used - 1
        };
        body.push_str(&format!(" (r{r} $x{v})"));
    }
    // template uses the first variable only => SingleComponent on split bodies
    prog.push_str(&format!("(exec 0 (,{body}) (, (out $x0) (done)))\n"));
    (prog, vars_used >= 2)
}

#[test]
fn lockstep_generated_corpus() {
    let _guard = ROUTING_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for seed in 0..200u64 {
        let (program, qualifies) = gen_program(seed, true);
        let firings = assert_lockstep(&program, 8);
        if qualifies {
            assert!(
                firings > 0,
                "qualifying generated case seed={seed} must emit through the planned route"
            );
        } else {
            assert_eq!(firings, 0, "single-component generated seed={seed}");
        }
    }
    for seed in 0..200u64 {
        let (program, _) = gen_program(seed, false);
        assert_lockstep(&program, 8);
    }
}
