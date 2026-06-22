use mork::api::{
    metta_special_form, FormalMettaSpecialForm, FormalMettaSpecialResult, FormalMinimalInstruction,
    Space, TermId, TermIdentitySidecar,
};
use mork::expr;
use mork_expr::Expr;

fn term_for(
    space: &mut Space,
    sidecar: &mut TermIdentitySidecar,
    expression: &'static str,
) -> TermId {
    let expression = expr!(space, expression);
    let encoded = unsafe { expression.span().as_ref().unwrap() };
    sidecar.insert_term(encoded).unwrap()
}

fn special_form_for(
    space: &mut Space,
    sidecar: &mut TermIdentitySidecar,
    expression: &'static str,
) -> Option<FormalMettaSpecialForm> {
    let term = term_for(space, sidecar, expression);
    metta_special_form(sidecar, term).unwrap()
}

fn dump(space: &Space, pattern: Expr, template: Expr) -> String {
    let mut output = Vec::new();
    space.dump_sexpr(pattern, template, &mut output);
    String::from_utf8(output).unwrap()
}

#[test]
fn repeated_variable_query_filters_non_unifying_facts() {
    let mut space = Space::new();
    let program = br#"
(Pair same same)
(Pair left right)

(exec repeated-variable-query
  (, (Pair $value $value))
  (O (+ (same-pair $value))))
"#;

    space.add_all_sexpr(program).unwrap();

    assert_eq!(space.metta_calculus(1), 1);
    let same_pairs = dump(
        &space,
        expr!(space, "[2] same-pair $"),
        expr!(space, "[2] same-pair _1"),
    );

    assert!(same_pairs.contains("(same-pair same)"), "{same_pairs}");
    assert!(!same_pairs.contains("(same-pair left)"), "{same_pairs}");
    assert!(!same_pairs.contains("(same-pair right)"), "{same_pairs}");
}

#[test]
fn equality_source_query_materializes_the_matched_term() {
    let mut space = Space::new();
    let program = br#"
(Knowledge (Some input))

(exec formal-query
  (I (== (Knowledge $value) $whole))
  (O (+ (query-result $whole))))
"#;

    space.add_all_sexpr(program).unwrap();

    assert_eq!(space.metta_calculus(1), 1);
    assert_eq!(
        dump(
            &space,
            expr!(space, "[2] query-result $"),
            expr!(space, "[2] query-result _1"),
        ),
        "(query-result (Knowledge (Some input)))\n"
    );
}

#[test]
fn special_result_parity_keeps_empty_not_reducible_unit_and_error_distinct() {
    let mut space = Space::new();
    let mut sidecar = TermIdentitySidecar::new();

    assert_eq!(
        special_form_for(&mut space, &mut sidecar, "Empty"),
        Some(FormalMettaSpecialForm::SpecialResult(
            FormalMettaSpecialResult::Empty
        ))
    );
    assert_eq!(
        special_form_for(&mut space, &mut sidecar, "NotReducible"),
        Some(FormalMettaSpecialForm::SpecialResult(
            FormalMettaSpecialResult::NotReducible
        ))
    );
    assert_eq!(
        special_form_for(&mut space, &mut sidecar, "[0]"),
        Some(FormalMettaSpecialForm::SpecialResult(
            FormalMettaSpecialResult::Unit
        ))
    );

    let bad = term_for(&mut space, &mut sidecar, "bad");
    let message = term_for(&mut space, &mut sidecar, "message");
    assert_eq!(
        special_form_for(&mut space, &mut sidecar, "[3] Error bad message"),
        Some(FormalMettaSpecialForm::SpecialResult(
            FormalMettaSpecialResult::Error { atom: bad, message }
        ))
    );

    for expression in [
        "Error",
        "[1] Error",
        "[2] Error bad",
        "[4] Error bad message extra",
    ] {
        assert_eq!(
            special_form_for(&mut space, &mut sidecar, expression),
            None,
            "{expression}"
        );
    }
}

#[test]
fn minimal_metta_instruction_recognition_requires_exact_arity() {
    let mut space = Space::new();
    let mut sidecar = TermIdentitySidecar::new();

    for (expression, instruction) in [
        ("[2] eval target", FormalMinimalInstruction::Eval),
        ("[3] evalc target ctx", FormalMinimalInstruction::Evalc),
        (
            "[4] chain [2] eval target $ _1",
            FormalMinimalInstruction::Chain,
        ),
        (
            "[5] unify $ Empty then else",
            FormalMinimalInstruction::Unify,
        ),
        (
            "[2] decons-atom [2] A B",
            FormalMinimalInstruction::DeconsAtom,
        ),
        ("[3] cons-atom A [1] B", FormalMinimalInstruction::ConsAtom),
        (
            "[2] function [2] return Done",
            FormalMinimalInstruction::Function,
        ),
        ("[2] return Done", FormalMinimalInstruction::Return),
        (
            "[2] collapse-bind [2] eval target",
            FormalMinimalInstruction::CollapseBind,
        ),
        (
            "[2] superpose-bind [1] [2] target bindings",
            FormalMinimalInstruction::SuperposeBind,
        ),
        (
            "[4] metta target Atom space",
            FormalMinimalInstruction::Metta,
        ),
        ("[1] context-space", FormalMinimalInstruction::ContextSpace),
        (
            "[4] call-native name pointer args",
            FormalMinimalInstruction::CallNative,
        ),
    ] {
        assert_eq!(
            special_form_for(&mut space, &mut sidecar, expression),
            Some(FormalMettaSpecialForm::MinimalInstruction(instruction)),
            "{expression}"
        );
    }

    for expression in [
        "eval",
        "[1] eval",
        "[3] eval target extra",
        "[2] evalc target",
        "[4] evalc target ctx extra",
        "[3] chain atom var",
        "[5] chain atom var body extra",
        "[4] unify atom pattern then",
        "[6] unify atom pattern then else extra",
        "[1] decons-atom",
        "[3] decons-atom target extra",
        "[2] cons-atom head",
        "[4] cons-atom head tail extra",
        "[1] function",
        "[3] function body extra",
        "[1] return",
        "[3] return atom extra",
        "[1] collapse-bind",
        "[3] collapse-bind atom extra",
        "[1] superpose-bind",
        "[3] superpose-bind atom extra",
        "[3] metta target type",
        "[5] metta target type space extra",
        "context-space",
        "[2] context-space extra",
        "[3] call-native name pointer",
        "[5] call-native name pointer args extra",
    ] {
        assert_eq!(
            special_form_for(&mut space, &mut sidecar, expression),
            None,
            "{expression}"
        );
    }
}

#[test]
fn transform_over_facts_preserves_conjunctive_bindings() {
    let mut space = Space::new();
    let program = br#"
(parent Pam Bob)
(parent Tom Bob)
(female Pam)

(exec transform-parent
  (, (parent $candidate Bob) (female $candidate))
  (O (+ (mother Bob $candidate))))
"#;

    space.add_all_sexpr(program).unwrap();

    assert_eq!(space.metta_calculus(1), 1);
    let mothers = dump(
        &space,
        expr!(space, "[3] mother Bob $"),
        expr!(space, "[3] mother Bob _1"),
    );

    assert!(mothers.contains("(mother Bob Pam)"), "{mothers}");
    assert!(!mothers.contains("Tom"), "{mothers}");
}

#[test]
fn context_rewrite_keeps_the_surrounding_expression_shape() {
    let mut space = Space::new();
    let program = br#"
(Wrapped (Some input))

(exec context-rewrite
  (, (Wrapped (Some $value)))
  (O (+ (Wrapped (Result $value)))))
"#;

    space.add_all_sexpr(program).unwrap();

    assert_eq!(space.metta_calculus(1), 1);
    assert_eq!(
        dump(
            &space,
            expr!(space, "[2] Wrapped [2] Result $"),
            expr!(space, "[2] Wrapped [2] Result _1"),
        ),
        "(Wrapped (Result input))\n"
    );
}

#[test]
fn query_to_chain_output_requires_a_later_machine_step() {
    let mut space = Space::new();
    let program = br#"
(seed start)

(exec a-stage
  (, (seed start))
  (O (+ (working-message (Wrapped input)))))

(exec z-stage
  (, (working-message (Wrapped $value)))
  (O (+ (formal-output (Wrapped $value)))))
"#;

    space.add_all_sexpr(program).unwrap();

    assert_eq!(space.metta_calculus(1), 1);
    assert_eq!(
        dump(
            &space,
            expr!(space, "[2] formal-output $"),
            expr!(space, "[2] formal-output _1"),
        ),
        ""
    );
    assert_eq!(
        dump(
            &space,
            expr!(space, "[2] working-message $"),
            expr!(space, "[2] working-message _1"),
        ),
        "(working-message (Wrapped input))\n"
    );

    assert_eq!(space.metta_calculus(1), 1);
    assert_eq!(
        dump(
            &space,
            expr!(space, "[2] formal-output $"),
            expr!(space, "[2] formal-output _1"),
        ),
        "(formal-output (Wrapped input))\n"
    );
}
