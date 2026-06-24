use mork::expr;
use mork::formal_lowering::{
    FormalMettaSpecialForm, FormalMettaSpecialResult, FormalMinimalInstruction, metta_special_form,
};
use mork::space::Space;
use mork::term_identity::{TermId, TermIdentitySidecar};

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
