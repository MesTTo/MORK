//! Differential oracle for interpreted equality-source leapfrog routing.

use mork::space::{ACT_PATH, Space};

#[cfg(feature = "leapfrog")]
use mork::zipper_join::{
    io_eq_leapfrog_queries, reset_io_eq_leapfrog_queries, set_leapfrog_dispatch,
    set_leapfrog_dispatch_all,
};

fn run_program(program: &str, max_steps: usize) -> (usize, Vec<u8>) {
    #[cfg(feature = "leapfrog")]
    set_leapfrog_dispatch(false);
    let mut space = Space::new();
    space.add_all_sexpr(program.as_bytes()).unwrap();
    let steps = space.metta_calculus(max_steps);
    let mut dump = Vec::new();
    space.dump_all_sexpr(&mut dump).unwrap();
    #[cfg(feature = "leapfrog")]
    set_leapfrog_dispatch(true);
    (steps, dump)
}

#[cfg(feature = "leapfrog")]
#[derive(Clone, Copy)]
enum RouteMode {
    Gated,
    All,
}

#[cfg(feature = "leapfrog")]
fn set_route_mode(mode: RouteMode) {
    match mode {
        RouteMode::Gated => set_leapfrog_dispatch(true),
        RouteMode::All => set_leapfrog_dispatch_all(),
    }
}

#[cfg(feature = "leapfrog")]
fn dump(space: &Space) -> Vec<u8> {
    let mut out = Vec::new();
    space.dump_all_sexpr(&mut out).unwrap();
    out
}

/// Run stock and routed spaces one firing at a time and compare the exact dish after every firing.
#[cfg(feature = "leapfrog")]
fn assert_lockstep(program: &str, max_steps: usize, mode: RouteMode) -> usize {
    let mut stock = Space::new();
    stock.add_all_sexpr(program.as_bytes()).unwrap();
    let mut routed = Space::new();
    routed.add_all_sexpr(program.as_bytes()).unwrap();

    reset_io_eq_leapfrog_queries();
    for step in 0..max_steps {
        set_leapfrog_dispatch(false);
        let stock_steps = stock.metta_calculus(1);
        set_route_mode(mode);
        let routed_steps = routed.metta_calculus(1);
        assert_eq!(routed_steps, stock_steps, "step count diverged at firing {step}");
        assert_eq!(dump(&routed), dump(&stock), "dish diverged at firing {step}");
        if stock_steps == 0 {
            break;
        }
    }
    let queries = io_eq_leapfrog_queries();
    set_leapfrog_dispatch(true);
    queries
}

const WHOLE_FACT_CAPTURE: &str = r#"
(LHS (foo $y))
(RHS ($x bar))
(exec 0 (I (BTM (LHS $p)) (== (RHS $p) $out)) (, (REM $out)))
"#;

const REMOVAL_CONTENTION: &str = r#"
(token a)
(candidate a 1)
(candidate a 2)
(exec 0 (I (== (token $x) $token)
           (== (candidate $x $value) $candidate))
        (O (- $token) (- $candidate) (+ (seen $value))))
"#;

const MIXED_EQ_AND_BTM: &str = r#"
(r a m)
(s m z)
(t z)
(exec 0 (I (== (r $x $mid) $r_fact)
           (== (s $mid $late) $s_fact)
           (BTM (t $late)))
        (, (joined $x $mid $late $r_fact $s_fact)))
"#;

const GROUND_CAPTURE: &str = r#"
(r a)
(r b)
(exec 0 (I (== (r $x) (r a))) (, (ground $x)))
"#;

const REPEATED_CAPTURE: &str = r#"
(r a)
(keep (r a))
(exec 0 (I (== (r $x) $fact) (BTM (keep $fact)))
        (, (reused $x $fact)))
"#;

const NOT_EQUAL_SOURCE: &str = r#"
(VAL X)
(VAL Y)
(VAL Z)
(exec 0 (I (!= (VAL $x) (VAL $y))) (, (OUT ($x != $y))))
"#;

const SCHEMATIC_EQ_CAPTURES: &str = r#"
(schematic a ($x tail))
(link a ok)
(exec 0 (I (== (schematic $key $term) $schematic_fact)
           (== (link $key $value) $link_fact))
        (, (captured $term $value $schematic_fact $link_fact)))
"#;

fn pc_reaction_program(receives: usize) -> String {
    let mut program = String::new();
    for index in 0..receives {
        program.push_str(&format!("(petri (? c{index} p{index} (done r{index})))\n"));
    }
    let matching = receives / 2;
    program.push_str(&format!("(petri (! c{matching} p{matching}))\n"));
    program.push_str(
        r#"(exec 0
      (I (== (petri (? $channel $payload $body)) $recv)
         (== (petri (! $channel $payload)) $send))
      (O (+ (petri $body)) (- $recv) (- $send)))
"#,
    );
    program
}

fn create_act_fixture(label: &str) -> (String, String) {
    let name = format!("io_eq_leapfrog_{label}_{}", std::process::id());
    let path = format!("{ACT_PATH}{name}.act");
    let mut external = Space::new();
    external.add_all_sexpr(b"(external a)\n").unwrap();
    external.backup_tree(&path).unwrap();
    let program = format!("(exec 0 (I (ACT {name} (external $x))) (, (from-act $x)))\n");
    (path, program)
}

#[cfg(feature = "leapfrog")]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

#[cfg(feature = "leapfrog")]
fn generated_eq_program(seed: u64) -> String {
    let mut state = seed;
    let left_count = 1 + (lcg_next(&mut state) % 5) as usize;
    let right_count = 1 + (lcg_next(&mut state) % 5) as usize;
    let guard_count = 1 + (lcg_next(&mut state) % 4) as usize;
    let three_factors = lcg_next(&mut state) & 1 == 0;
    let mut program = String::new();
    for index in 0..left_count {
        let key = lcg_next(&mut state) % 4;
        program.push_str(&format!("(left k{key} l{index})\n"));
    }
    for index in 0..right_count {
        let key = lcg_next(&mut state) % 4;
        program.push_str(&format!("(right k{key} r{index})\n"));
    }
    for index in 0..guard_count {
        let key = lcg_next(&mut state) % 4;
        program.push_str(&format!("(guard k{key} g{index})\n"));
    }
    if three_factors {
        program.push_str(
            "(exec 0 (I (== (left $key $left) $left_fact)\n\
                         (== (right $key $right) $right_fact)\n\
                         (== (guard $key $guard) $guard_fact))\n\
                      (, (row $key $left $right $guard $left_fact $right_fact $guard_fact)))\n",
        );
    } else {
        program.push_str(
            "(exec 0 (I (== (left $key $left) $left_fact)\n\
                         (== (right $key $right) $right_fact))\n\
                      (, (row $key $left $right $left_fact $right_fact)))\n",
        );
    }
    program
}

#[test]
fn stock_equality_capture_binds_the_whole_schematic_fact() {
    let (steps, dump) = run_program(WHOLE_FACT_CAPTURE, 8);
    assert_eq!(steps, 1);
    assert_eq!(
        dump,
        b"(LHS (foo $a))\n(REM (RHS ($a bar)))\n(RHS ($a bar))\n"
    );
}

#[test]
fn stock_removal_contention_enumerates_before_finalizing() {
    let (steps, dump) = run_program(REMOVAL_CONTENTION, 8);
    assert_eq!(steps, 1);
    assert_eq!(dump, b"(seen 1)\n(seen 2)\n");
}

#[test]
fn stock_mixed_sources_preserve_variables_after_capture_gaps() {
    let (steps, dump) = run_program(MIXED_EQ_AND_BTM, 8);
    assert_eq!(steps, 1);
    assert_eq!(
        dump,
        b"(t z)\n(r a m)\n(s m z)\n(joined a m z (r a m) (s m z))\n"
    );
}

#[test]
fn stock_decline_shapes_have_pinned_results() {
    let (ground_steps, ground_dump) = run_program(GROUND_CAPTURE, 8);
    assert_eq!(ground_steps, 1);
    assert_eq!(ground_dump, b"(r a)\n(r b)\n(ground a)\n");

    let (repeated_steps, repeated_dump) = run_program(REPEATED_CAPTURE, 8);
    assert_eq!(repeated_steps, 1);
    assert_eq!(repeated_dump, b"(r a)\n(keep (r a))\n(reused a (r a))\n");

    let (ne_steps, ne_dump) = run_program(NOT_EQUAL_SOURCE, 8);
    assert_eq!(ne_steps, 1);
    assert_eq!(
        ne_dump,
        b"(OUT (X != Y))\n(OUT (X != Z))\n(OUT (Y != X))\n\
          (OUT (Y != Z))\n(OUT (Z != X))\n(OUT (Z != Y))\n\
          (VAL X)\n(VAL Y)\n(VAL Z)\n"
    );
}

#[test]
fn stock_process_calculus_reaction_shape_is_pinned_at_several_sizes() {
    for receives in [2usize, 7, 16] {
        let matching = receives / 2;
        let (steps, dump) = run_program(&pc_reaction_program(receives), 8);
        assert_eq!(steps, 1, "receives={receives}");
        let dump = std::str::from_utf8(&dump).unwrap();
        assert!(
            dump.contains(&format!("(petri (done r{matching}))")),
            "receives={receives}: {dump}"
        );
        assert!(
            !dump.contains(&format!("(petri (! c{matching} p{matching}))")),
            "receives={receives}: {dump}"
        );
        assert!(!dump.contains("(exec "), "receives={receives}: {dump}");
    }
}

#[test]
fn stock_act_source_fixture_is_pinned() {
    let (path, program) = create_act_fixture("stock");
    let result = run_program(&program, 8);
    let cleanup = std::fs::remove_file(&path);
    cleanup.unwrap();

    assert_eq!(result.0, 1);
    assert_eq!(result.1, b"(from-act a)\n");
}

#[cfg(feature = "leapfrog")]
#[test]
fn routed_equality_sources_match_stock_after_every_firing() {
    for program in [
        WHOLE_FACT_CAPTURE,
        REMOVAL_CONTENTION,
        MIXED_EQ_AND_BTM,
        SCHEMATIC_EQ_CAPTURES,
    ] {
        let queries = assert_lockstep(program, 8, RouteMode::All);
        assert!(queries > 0, "eligible fixture did not exercise the route");
    }
    for receives in [2usize, 7, 16] {
        let program = pc_reaction_program(receives);
        let queries = assert_lockstep(&program, 8, RouteMode::All);
        assert!(queries > 0, "receives={receives} did not exercise the route");
    }
}

#[cfg(feature = "leapfrog")]
#[test]
fn routed_equality_sources_decline_ineligible_interpreted_bodies() {
    for program in [GROUND_CAPTURE, REPEATED_CAPTURE, NOT_EQUAL_SOURCE] {
        let queries = assert_lockstep(program, 8, RouteMode::All);
        assert_eq!(queries, 0, "ineligible fixture entered the route");
    }

    let (path, program) = create_act_fixture("decline");
    let queries = assert_lockstep(&program, 8, RouteMode::All);
    std::fs::remove_file(&path).unwrap();
    assert_eq!(queries, 0, "ACT source entered the BTM-only route");
}

#[cfg(feature = "leapfrog")]
#[test]
fn deterministic_gate_routes_the_large_process_calculus_shape() {
    let program = pc_reaction_program(160);
    let queries = assert_lockstep(&program, 8, RouteMode::Gated);
    assert!(queries > 0, "large selective equality join did not pass the gate");
}

#[cfg(feature = "leapfrog")]
#[test]
fn generated_equality_body_corpus_is_byte_identical() {
    for seed in 0..128u64 {
        let program = generated_eq_program(seed ^ 0x6a09_e667_f3bc_c909);
        let queries = assert_lockstep(&program, 8, RouteMode::All);
        assert!(queries > 0, "seed {seed} did not exercise the route");
    }
}
