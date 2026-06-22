use std::collections::BTreeSet;

use mork::critical_pairs::{
    Rule, Term, first_non_joinable_witness, ground_facts_from_mm2_program, rules_from_mm2_program,
    saturate_additive_state, state_rule_successors, state_rules_from_mm2_program,
};
use mork::expr;
use mork::space::Space;
use mork_expr::Expr;

fn dump(space: &Space, pattern: Expr, template: Expr) -> String {
    let mut output = Vec::new();
    space.dump_sexpr(pattern, template, &mut output);
    String::from_utf8(output).unwrap()
}

fn overlapping_rules() -> Vec<Rule> {
    vec![
        Rule::new(
            "outer-rule",
            Term::app(
                "f",
                vec![Term::app("g", vec![Term::var("x")]), Term::var("z")],
            ),
            Term::app("left", vec![Term::var("x"), Term::var("z")]),
        ),
        Rule::new(
            "inner-rule",
            Term::app("g", vec![Term::var("y")]),
            Term::app("right", vec![Term::var("y")]),
        ),
    ]
}

fn fact(name: &str, args: &[&str]) -> Term {
    Term::app(name, args.iter().map(|arg| Term::sym(*arg)).collect())
}

fn render_terms(terms: &[Term]) -> String {
    terms
        .iter()
        .map(Term::to_metta)
        .collect::<Vec<_>>()
        .join(", ")
}

#[test]
fn critical_pair_witness_reports_non_joinable_overlap_as_mork_atom() {
    let witness = first_non_joinable_witness(&overlapping_rules(), 8).unwrap();

    assert_eq!(
        witness.to_metta_atom(),
        "(critical-pair outer-rule inner-rule p0 (f (g a) b) (left a b) (f (right a) b))"
    );

    let mut space = Space::new();
    space
        .add_all_sexpr(witness.to_metta_atom().as_bytes())
        .unwrap();

    assert_eq!(
        dump(
            &space,
            expr!(space, "[7] critical-pair $ $ $ $ $ $"),
            expr!(space, "[7] critical-pair _1 _2 _3 _4 _5 _6"),
        ),
        "(critical-pair outer-rule inner-rule p0 (f (g a) b) (left a b) (f (right a) b))\n"
    );
}

#[test]
fn critical_pair_witness_treats_bounded_joinable_overlap_as_closed() {
    let mut rules = overlapping_rules();
    rules.push(Rule::new(
        "close-rule",
        Term::app(
            "f",
            vec![Term::app("right", vec![Term::var("u")]), Term::var("v")],
        ),
        Term::app("left", vec![Term::var("u"), Term::var("v")]),
    ));

    assert!(first_non_joinable_witness(&rules, 8).is_none());
}

#[test]
fn critical_pair_witness_extracts_rules_from_real_mm2_exec_program_text() {
    let program = r#"
; Only finite first-order, single-pattern execs are extracted.
(exec outer-rule
  (, (f (g $x) $z))
  (O (- (f (g $x) $z))
     (+ (left $x $z))))

(exec inner-rule
  (, (g $y))
  (O (+ (right $y))))

; Multi-pattern execs are ordinary MM2, but not finite first-order rewrite rules.
(exec skipped-rule
  (, (guard $x) (f $x))
  (O (+ (guarded $x))))
"#;

    let rules = rules_from_mm2_program(program).unwrap();
    assert_eq!(rules.len(), 2);

    let witness = first_non_joinable_witness(&rules, 8).unwrap();
    assert_eq!(
        witness.to_metta_atom(),
        "(critical-pair outer-rule inner-rule p0 (f (g a) b) (left a b) (f (right a) b))"
    );
}

#[test]
fn critical_pair_witness_projects_multi_output_single_pattern_execs() {
    let rules = rules_from_mm2_program(include_str!("../resources/transitive.mm2")).unwrap();

    let rendered = rules
        .iter()
        .map(|rule| {
            format!(
                "{}: {} -> {}",
                rule.name,
                rule.lhs.to_metta(),
                rule.rhs.to_metta()
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(
        rendered,
        vec![
            "0-out-0: (triangle $x $y $z) -> (edge $z $x)",
            "0-out-1: (triangle $x $y $z) -> (edge $y $z)",
            "0-out-2: (triangle $x $y $z) -> (edge $x $y)",
        ]
    );
}

#[test]
fn mm2_state_rules_extract_multi_pattern_execs_from_transitive_program() {
    let state_rules = state_rules_from_mm2_program(include_str!("../resources/transitive.mm2"))
        .expect("transitive fixture should parse");

    let rendered = state_rules
        .iter()
        .map(|rule| {
            format!(
                "{}: patterns=[{}] remove=[{}] add=[{}]",
                rule.name,
                render_terms(&rule.patterns),
                render_terms(&rule.remove),
                render_terms(&rule.add)
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(
        rendered,
        vec![
            "0: patterns=[(triangle $x $y $z)] remove=[] add=[(edge $z $x), (edge $y $z), (edge $x $y)]",
            "1: patterns=[(edge $x $y), (edge $y $z)] remove=[] add=[(edge $x $z)]",
            "2: patterns=[(edge $z $x), (edge $y $z), (edge $x $y)] remove=[] add=[(triangle $x $y $z)]",
        ]
    );
}

#[test]
fn mm2_state_rule_successors_share_bindings_across_pattern_conjunctions() {
    let state_rules = state_rules_from_mm2_program(include_str!("../resources/transitive.mm2"))
        .expect("transitive fixture should parse");
    let state = BTreeSet::from([
        fact("edge", &["brussels", "paris"]),
        fact("edge", &["paris", "london"]),
    ]);

    let steps = state_rule_successors(&state, &state_rules[1]);

    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].rule, "1");
    assert_eq!(steps[0].add, vec![fact("edge", &["brussels", "london"])]);
    assert!(
        steps[0]
            .after
            .contains(&fact("edge", &["brussels", "london"]))
    );
}

#[test]
fn mm2_state_rule_successors_apply_remove_and_add_effects() {
    let program = r#"
(state todo task-1)

(exec move-task
  (, (state todo $task))
  (O (- (state todo $task))
     (+ (state done $task))))
"#;
    let state_rules = state_rules_from_mm2_program(program).expect("program should parse");
    let initial_facts = ground_facts_from_mm2_program(program).expect("initial facts should parse");

    let steps = state_rule_successors(&initial_facts, &state_rules[0]);

    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].remove, vec![fact("state", &["todo", "task-1"])]);
    assert_eq!(steps[0].add, vec![fact("state", &["done", "task-1"])]);
    assert!(!steps[0].after.contains(&fact("state", &["todo", "task-1"])));
    assert!(steps[0].after.contains(&fact("state", &["done", "task-1"])));
}

#[test]
fn mm2_state_rule_analyzer_saturates_transitive_program_facts() {
    let program = include_str!("../resources/transitive.mm2");
    let state_rules =
        state_rules_from_mm2_program(program).expect("transitive fixture should parse");
    let initial_facts = ground_facts_from_mm2_program(program).expect("initial facts should parse");

    assert_eq!(initial_facts.len(), 2);

    let saturated = saturate_additive_state(initial_facts, &state_rules, 8);

    assert_eq!(saturated.len(), 150);
    assert!(saturated.contains(&fact("edge", &["london", "istanbul"])));
    assert!(saturated.contains(&fact("edge", &["st-petersburg", "paris"])));
    assert!(saturated.contains(&fact("triangle", &["paris", "st-petersburg", "london"],)));
}
