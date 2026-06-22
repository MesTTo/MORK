#![cfg(feature = "grounding")]

use mork::expr;
use mork::space::Space;

#[test]
fn pure_sink_accepts_matching_fixed_symbol_guard() {
    let mut space = Space::new();
    let program = br#"
(x a)

(exec 0
  (, (x $v))
  (O (pure (fixed racecar) racecar (reverse_symbol racecar))))
"#;

    space.add_all_sexpr(program).unwrap();
    assert_eq!(space.metta_calculus(1), 1);

    let mut output = Vec::new();
    space.dump_sexpr(
        expr!(space, "[2] fixed $"),
        expr!(space, "[2] fixed _1"),
        &mut output,
    );
    let output = String::from_utf8(output).unwrap();

    assert_eq!(output, "(fixed racecar)\n");
}

#[test]
fn pure_sink_accepts_matching_fixed_expression_guard() {
    let mut space = Space::new();
    let program = br#"
(x a)

(exec 0
  (, (x $v))
  (O (pure
       (fixed (wrap racecar))
       (wrap racecar)
       (tuple wrap (reverse_symbol racecar)))))
"#;

    space.add_all_sexpr(program).unwrap();
    assert_eq!(space.metta_calculus(1), 1);

    let mut output = Vec::new();
    space.dump_sexpr(
        expr!(space, "[2] fixed $"),
        expr!(space, "[2] fixed _1"),
        &mut output,
    );
    let output = String::from_utf8(output).unwrap();

    assert_eq!(output, "(fixed (wrap racecar))\n");
}
