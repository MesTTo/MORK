//! Differential oracle for interpreted equality-source leapfrog routing.

use mork::space::{ACT_PATH, Space};

fn run_program(program: &str, max_steps: usize) -> (usize, Vec<u8>) {
    let mut space = Space::new();
    space.add_all_sexpr(program.as_bytes()).unwrap();
    let steps = space.metta_calculus(max_steps);
    let mut dump = Vec::new();
    space.dump_all_sexpr(&mut dump).unwrap();
    (steps, dump)
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
    let name = format!("io_eq_leapfrog_differential_{}", std::process::id());
    let path = format!("{ACT_PATH}{name}.act");
    let mut external = Space::new();
    external.add_all_sexpr(b"(external a)\n").unwrap();
    external.backup_tree(&path).unwrap();

    let program = format!("(exec 0 (I (ACT {name} (external $x))) (, (from-act $x)))\n");
    let result = run_program(&program, 8);
    let cleanup = std::fs::remove_file(&path);
    cleanup.unwrap();

    assert_eq!(result.0, 1);
    assert_eq!(result.1, b"(from-act a)\n");
}
