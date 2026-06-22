use mork::expr;
use mork::space::Space;
use std::collections::BTreeSet;

fn dump_lines(output: String) -> BTreeSet<String> {
    output.lines().map(str::to_owned).collect()
}

#[test]
fn planned_query_preserves_coreferential_variables() {
    let mut space = Space::new();
    let program = br#"
(A a)
(A b)
(B c)
(C a)
(C c)

(exec 0
  (, (A $x) (B $y) (C $x))
  (, (R $x $y)))
"#;

    space.add_all_sexpr(program).unwrap();
    assert_eq!(space.metta_calculus(1), 1);

    let mut output = Vec::new();
    space.dump_sexpr(
        expr!(space, "[3] R $ $"),
        expr!(space, "[3] R _1 _2"),
        &mut output,
    );
    let output = String::from_utf8(output).unwrap();

    assert!(output.contains("(R a c)"), "{output}");
    assert!(!output.contains("(R c c)"), "{output}");
    assert!(!output.contains("(R b c)"), "{output}");
}

#[test]
fn planned_query_preserves_refs_when_selective_factor_runs_first() {
    let mut space = Space::new();
    let program = br#"
(A a ok)
(A a bad)
(A b ok)
(A c ok)
(B target a)
(C ok)

(exec 0
  (, (A $x $z) (B target $x) (C $z))
  (, (R $x $z)))
"#;

    space.add_all_sexpr(program).unwrap();
    assert_eq!(space.metta_calculus(1), 1);

    let mut output = Vec::new();
    space.dump_sexpr(
        expr!(space, "[3] R $ $"),
        expr!(space, "[3] R _1 _2"),
        &mut output,
    );
    let output = String::from_utf8(output).unwrap();

    assert!(output.contains("(R a ok)"), "{output}");
    assert_eq!(output.matches("(R ").count(), 1, "{output}");
    assert!(!output.contains("(R a bad)"), "{output}");
    assert!(!output.contains("(R b ok)"), "{output}");
    assert!(!output.contains("(R c ok)"), "{output}");
}

#[test]
fn planned_query_handles_repeated_prefix_factors_after_cardinality_cache() {
    let mut space = Space::new();
    let program = br#"
(A a left)
(A a right)
(A b left)
(Guard a)

(exec 0
  (, (A $x left) (A $x $side) (Guard $x))
  (, (R $x $side)))
"#;

    space.add_all_sexpr(program).unwrap();
    assert_eq!(space.metta_calculus(1), 1);

    let mut output = Vec::new();
    space.dump_sexpr(
        expr!(space, "[3] R $ $"),
        expr!(space, "[3] R _1 _2"),
        &mut output,
    );
    let output = String::from_utf8(output).unwrap();

    assert!(output.contains("(R a left)"), "{output}");
    assert!(output.contains("(R a right)"), "{output}");
    assert_eq!(output.matches("(R ").count(), 2, "{output}");
    assert!(!output.contains("(R b left)"), "{output}");
}

#[test]
fn planned_query_falls_back_for_variable_bearing_rule_template() {
    let mut space = Space::new();
    let program = br#"
(inputfile 0 (arg 1390) (arg 0.9257))
(inputfile 1 (arg 3490) (arg 1.2329))

(normalize $e (normalized-value $e))

(exec 0
  (, (inputfile $i (arg $x) (arg $y))
     (normalize
       (product_f64 (i64_as_f64 (i64_from_string $x)) (f64_from_string $y))
       $normalized))
  (, (normalized $i $normalized)))
"#;

    space.add_all_sexpr(program).unwrap();
    assert_eq!(space.metta_calculus(1), 1);

    let mut output = Vec::new();
    space.dump_sexpr(
        expr!(space, "[3] normalized $ $"),
        expr!(space, "[3] normalized _1 _2"),
        &mut output,
    );
    let output = String::from_utf8(output).unwrap();

    assert!(
        output.contains(
            "(normalized 0 (normalized-value (product_f64 (i64_as_f64 (i64_from_string 1390)) (f64_from_string 0.9257))))"
        ),
        "{output}"
    );
    assert!(
        output.contains(
            "(normalized 1 (normalized-value (product_f64 (i64_as_f64 (i64_from_string 3490)) (f64_from_string 1.2329))))"
        ),
        "{output}"
    );
}

#[test]
fn planned_source_query_preserves_coreferential_variables() {
    let mut space = Space::new();
    let program = br#"
(A a ok)
(A a bad)
(A b ok)
(A c ok)
(B target a)
(C ok)

(exec 0
  (I (BTM (A $x $z)) (BTM (B target $x)) (BTM (C $z)))
  (, (IR $x $z)))
"#;

    space.add_all_sexpr(program).unwrap();
    assert_eq!(space.metta_calculus(1), 1);

    let mut output = Vec::new();
    space.dump_sexpr(
        expr!(space, "[3] IR $ $"),
        expr!(space, "[3] IR _1 _2"),
        &mut output,
    );
    let output = String::from_utf8(output).unwrap();

    assert!(output.contains("(IR a ok)"), "{output}");
    assert_eq!(output.matches("(IR ").count(), 1, "{output}");
    assert!(!output.contains("(IR a bad)"), "{output}");
    assert!(!output.contains("(IR b ok)"), "{output}");
    assert!(!output.contains("(IR c ok)"), "{output}");
}

#[test]
fn planned_query_preserves_bindings_when_projected_domain_tie_breaks() {
    let mut space = Space::new();
    let mut program = String::new();
    for i in 0..16 {
        program.push_str(&format!("(WideAA key{i} item{i})\n"));
        program.push_str(&format!("(NaroAA bucket{} item{i})\n", i % 2));
    }
    program.push_str(
        r#"
(exec 0
  (, (WideAA $wide $payload) (NaroAA $bucket $payload))
  (, (DomainR $bucket $wide $payload)))
"#,
    );

    space.add_all_sexpr(program.as_bytes()).unwrap();
    assert_eq!(space.metta_calculus(1), 1);

    let mut output = Vec::new();
    space.dump_sexpr(
        expr!(space, "[4] DomainR $ $ $"),
        expr!(space, "[4] DomainR _1 _2 _3"),
        &mut output,
    );
    let output = String::from_utf8(output).unwrap();

    assert!(output.contains("(DomainR bucket0 key0 item0)"), "{output}");
    assert!(output.contains("(DomainR bucket1 key1 item1)"), "{output}");
    assert_eq!(output.matches("(DomainR ").count(), 16, "{output}");
}

#[test]
fn planned_query_preserves_results_for_all_factor_orders() {
    let mut space = Space::new();
    let mut program = String::from(
        r#"
(A alice food)
(A bob tool)
(A carol food)
(B food edible)
(B tool usable)
(C alice ok)
(C bob blocked)
(C carol ok)
"#,
    );

    let factors = ["(A $person $kind)", "(B $kind $class)", "(C $person ok)"];
    let permutations = [
        ("R012", [0, 1, 2]),
        ("R021", [0, 2, 1]),
        ("R102", [1, 0, 2]),
        ("R120", [1, 2, 0]),
        ("R201", [2, 0, 1]),
        ("R210", [2, 1, 0]),
    ];

    for (relation, order) in permutations {
        program.push_str(&format!(
            r#"
(exec {relation}
  (, {} {} {})
  (, ({relation} $person $kind $class)))
"#,
            factors[order[0]], factors[order[1]], factors[order[2]]
        ));
    }

    space.add_all_sexpr(program.as_bytes()).unwrap();
    assert_eq!(space.metta_calculus(6), 6);

    let expected = BTreeSet::from([
        "(R012 alice food edible)".to_owned(),
        "(R012 carol food edible)".to_owned(),
    ]);
    let r012 = dump_lines({
        let mut output = Vec::new();
        space.dump_sexpr(
            expr!(space, "[4] R012 $ $ $"),
            expr!(space, "[4] R012 _1 _2 _3"),
            &mut output,
        );
        String::from_utf8(output).unwrap()
    });
    assert_eq!(r012, expected);

    let expected_suffixes = BTreeSet::from([
        "alice food edible".to_owned(),
        "carol food edible".to_owned(),
    ]);
    let relation_outputs = [
        (
            "R021",
            expr!(space, "[4] R021 $ $ $"),
            expr!(space, "[4] R021 _1 _2 _3"),
        ),
        (
            "R102",
            expr!(space, "[4] R102 $ $ $"),
            expr!(space, "[4] R102 _1 _2 _3"),
        ),
        (
            "R120",
            expr!(space, "[4] R120 $ $ $"),
            expr!(space, "[4] R120 _1 _2 _3"),
        ),
        (
            "R201",
            expr!(space, "[4] R201 $ $ $"),
            expr!(space, "[4] R201 _1 _2 _3"),
        ),
        (
            "R210",
            expr!(space, "[4] R210 $ $ $"),
            expr!(space, "[4] R210 _1 _2 _3"),
        ),
    ];

    for (relation, pattern, template) in relation_outputs {
        let mut output = Vec::new();
        space.dump_sexpr(pattern, template, &mut output);
        let output = String::from_utf8(output).unwrap();
        let suffixes = dump_lines(output)
            .into_iter()
            .map(|line| {
                line.strip_prefix(&format!("({relation} "))
                    .and_then(|line| line.strip_suffix(')'))
                    .unwrap()
                    .to_owned()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(suffixes, expected_suffixes, "{relation}");
    }
}
