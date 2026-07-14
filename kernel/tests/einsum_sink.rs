use mork::expr;
use mork::space::Space;
use std::collections::BTreeMap;

fn run_one_step_and_dump(program: &[u8], query: &str, template: &str) -> String {
    let mut space = Space::new();
    space.add_all_sexpr(program).unwrap();
    assert_eq!(space.metta_calculus(1), 1);
    dump_selection(&space, query, template)
}

fn dump_selection(space: &Space, query: &str, template: &str) -> String {
    let mut output = Vec::new();
    space.dump_sexpr(expr!(space, query), expr!(space, template), &mut output);
    String::from_utf8(output).unwrap()
}

fn assert_contains_all(output: &str, expected: &[&str]) {
    for fact in expected {
        assert!(output.contains(fact), "{output}");
    }
}

fn assert_excludes_all(output: &str, unexpected: &[&str]) {
    for fact in unexpected {
        assert!(!output.contains(fact), "{output}");
    }
}

fn parse_dumped_tensor_cells(output: &str, name: &str) -> BTreeMap<Vec<usize>, f32> {
    let mut cells = BTreeMap::new();
    for line in output.lines() {
        let cell = line
            .strip_prefix('(')
            .and_then(|line| line.strip_suffix(')'))
            .unwrap_or_else(|| panic!("dumped tensor cell is not parenthesized: {line:?}"));
        let parts: Vec<&str> = cell.split_whitespace().collect();
        assert!(
            parts.len() >= 2,
            "dumped tensor cell must contain a name and value: {line:?}"
        );
        assert_eq!(parts[0], name, "dumped tensor cell name changed");
        let value = parts[parts.len() - 1]
            .parse::<f32>()
            .unwrap_or_else(|err| panic!("could not parse dumped tensor value {line:?}: {err}"));
        let indices = parts[1..parts.len() - 1]
            .iter()
            .map(|index| {
                index.parse::<usize>().unwrap_or_else(|err| {
                    panic!("could not parse dumped tensor index {line:?}: {err}")
                })
            })
            .collect();
        cells.insert(indices, value);
    }
    cells
}

fn assert_tensor_cells_close<const R: usize>(
    output: &str,
    name: &str,
    expected: &[([usize; R], f32)],
) {
    let cells = parse_dumped_tensor_cells(output, name);
    assert_eq!(cells.len(), expected.len(), "{output}");
    for (indices, expected_value) in expected {
        let actual = cells
            .get(&indices[..])
            .unwrap_or_else(|| panic!("missing {name}{indices:?} in:\n{output}"));
        let diff = (*actual - *expected_value).abs();
        assert!(
            diff <= 1.0e-6,
            "{name}{indices:?}: expected {expected_value}, got {actual}, diff {diff}"
        );
    }
}

const ATTENTION_APPLY_FIXTURE: &[u8] = br#"
(Q 0 0 0 0 0)
(Q 0 0 0 1 0)
(Q 0 0 1 0 0)
(Q 0 0 1 1 0)

(K 0 0 0 0 1)
(K 0 0 0 1 2)
(K 0 0 1 0 3)
(K 0 0 1 1 4)

(V 0 0 0 0 10)
(V 0 0 0 1 20)
(V 0 0 1 0 14)
(V 0 0 1 1 28)
"#;

fn format_tensor_value(value: f32) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn matrix_cells(name: &str, rows: usize, cols: usize, values: &[f32]) -> String {
    assert_eq!(values.len(), rows * cols);
    let mut out = String::new();
    for row in 0..rows {
        for col in 0..cols {
            let value = format_tensor_value(values[row * cols + col]);
            out.push_str(&format!("({name} {row} {col} {value})\n"));
        }
    }
    out
}

fn dense_matmul_2x3_3x2_cells() -> String {
    [
        matrix_cells("A", 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "\n".to_string(),
        matrix_cells("B", 3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    ]
    .concat()
}

fn dense_matmul_2x3_3x2_program(exec: &str) -> Vec<u8> {
    format!("{}\n{exec}\n", dense_matmul_2x3_3x2_cells()).into_bytes()
}

fn tensor_op_einsum_exec(pattern: &str, inputs: &str, output: &str, extra: &str) -> String {
    format!(
        r#"
(exec 0
  {pattern}
  (O (tensor-op-f32
        (op einsum ab,bc->ac)
        (inputs {inputs})
        (output {output})
        (from (A $i $k $av)
              (B $k $j $bv)){extra}
        (backend auto))))
"#
    )
}

fn sparse_dense_matmul_program(output_decl: &str) -> Vec<u8> {
    format!(
        r#"
(A 0 1 2)
(A 0 2 3)
(A 1 0 1)

{}

(exec 0
  (, (A $i $k $av)
     (X $k $j $xv))
  (O (einsum-f32 ab,bc->ac
        (csr A 3 3)
        (X 3 2)
        {output_decl}
        (A $i $k $av)
        (X $k $j $xv))))
"#,
        matrix_cells("X", 3, 2, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    )
    .into_bytes()
}

#[test]
fn einsum_f32_sink_runs_dense_matmul_from_mork_program() {
    let program = dense_matmul_2x3_3x2_program(
        r#"
(exec 0
  (, (A $i $k $av)
     (B $k $j $bv))
  (O (einsum-f32 ab,bc->ac
        (A 2 3)
        (B 3 2)
        (C 2 2)
        (A $i $k $av)
        (B $k $j $bv))))
"#,
    );

    let output = run_one_step_and_dump(&program, "[4] C $ $ $", "[4] C _1 _2 _3");
    assert_contains_all(
        &output,
        &["(C 0 0 58)", "(C 0 1 64)", "(C 1 0 139)", "(C 1 1 154)"],
    );
}

#[test]
fn einsum_f32_sink_scans_inputs_from_one_shot_trigger() {
    let program = format!(
        "{}\n{}\n",
        [
            matrix_cells("A", 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "\n".to_string(),
            matrix_cells(
                "B",
                3,
                4,
                &[
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0
                ],
            ),
        ]
        .concat(),
        r#"
(exec 0
  (, (A 0 0 $a00)
     (B 0 0 $b00))
  (O (einsum-f32 ab,bc->ac
        (A 2 3)
        (B 3 4)
        (C 2 4)
        (A $i $k $av)
        (B $k $j $bv))))
"#,
    )
    .into_bytes();

    let output = run_one_step_and_dump(&program, "[4] C $ $ $", "[4] C _1 _2 _3");
    assert_tensor_cells_close(
        &output,
        "C",
        &[
            ([0, 0], 74.0),
            ([0, 1], 80.0),
            ([0, 2], 86.0),
            ([0, 3], 92.0),
            ([1, 0], 173.0),
            ([1, 1], 188.0),
            ([1, 2], 203.0),
            ([1, 3], 218.0),
        ],
    );
}

#[test]
fn tensor_op_f32_runs_dense_matmul_from_operator_syntax() {
    let program = dense_matmul_2x3_3x2_program(&tensor_op_einsum_exec(
        "(, (A $i $k $av)\n     (B $k $j $bv))",
        "(A dense 2 3) (B dense 3 2)",
        "(C dense)",
        "",
    ));

    let output = run_one_step_and_dump(&program, "[4] C $ $ $", "[4] C _1 _2 _3");
    assert_contains_all(
        &output,
        &["(C 0 0 58)", "(C 0 1 64)", "(C 1 0 139)", "(C 1 1 154)"],
    );
}

#[test]
fn tensor_op_f32_einsum_scans_inputs_from_one_shot_trigger() {
    let program = br#"
(A 0 0 2)
(A 0 1 3)

(B 0 0 5)
(B 1 0 7)

(exec 0
  (, (A 0 0 $a00)
     (B 0 0 $b00))
  (O (tensor-op-f32
        (op einsum ab,bc->ac)
        (inputs (A dense 1 2) (B dense 2 1))
        (output (C dense))
        (from (A $i $k $av)
              (B $k $j $bv))
        (backend auto))))
"#;

    let output = run_one_step_and_dump(program, "[4] C $ $ $", "[4] C _1 _2 _3");
    assert_tensor_cells_close(&output, "C", &[([0, 0], 31.0)]);
}

#[test]
fn tensor_op_f32_runs_add_from_operator_syntax() {
    let program = br#"
(A 0 0 1)
(A 0 1 2)
(A 1 0 3)
(A 1 1 4)

(B 0 0 10)
(B 0 1 20)
(B 1 0 30)
(B 1 1 40)

(exec 0
  (, (A $i $j $av)
     (B $i $j $bv))
  (O (tensor-op-f32
        (op add)
        (inputs (A dense 2 2) (B dense 2 2))
        (output (C dense))
        (from (A $i $j $av)
              (B $i $j $bv))
        (backend auto))))
"#;

    let output = run_one_step_and_dump(program, "[4] C $ $ $", "[4] C _1 _2 _3");
    assert_tensor_cells_close(
        &output,
        "C",
        &[
            ([0, 0], 11.0),
            ([0, 1], 22.0),
            ([1, 0], 33.0),
            ([1, 1], 44.0),
        ],
    );
}

#[test]
fn tensor_op_f32_runs_layernorm_from_operator_syntax() {
    let program = br#"
(X 0 0 1)
(X 0 1 2)
(X 0 2 3)
(X 0 3 4)

(G 0 1)
(G 1 1)
(G 2 1)
(G 3 1)

(B 0 0)
(B 1 0)
(B 2 0)
(B 3 0)

(exec 0
  (, (X $row $d $xv)
     (G $d $gv)
     (B $d $bv))
  (O (tensor-op-f32
        (op layernorm 1e-5)
        (inputs (X dense 1 4) (G dense 4) (B dense 4))
        (output (Y dense))
        (from (X $row $d $xv)
              (G $d $gv)
              (B $d $bv))
        (backend auto))))
"#;

    let output = run_one_step_and_dump(program, "[4] Y $ $ $", "[4] Y _1 _2 _3");
    assert_tensor_cells_close(
        &output,
        "Y",
        &[
            ([0, 0], -1.341_635_5),
            ([0, 1], -0.447_211_83),
            ([0, 2], 0.447_211_83),
            ([0, 3], 1.341_635_5),
        ],
    );
}

#[test]
fn tensor_op_f32_runs_gelu_from_operator_syntax() {
    let program = br#"
(X 0 -1)
(X 1 0)
(X 2 1)
(X 3 2)

(exec 0
  (, (X $i $xv))
  (O (tensor-op-f32
        (op gelu)
        (inputs (X dense 4))
        (output (Y dense))
        (from (X $i $xv))
        (backend auto))))
"#;

    let output = run_one_step_and_dump(program, "[3] Y $ $", "[3] Y _1 _2");
    assert_tensor_cells_close(
        &output,
        "Y",
        &[
            ([0], -0.158_808_01),
            ([1], 0.0),
            ([2], 0.841_192),
            ([3], 1.954_597_7),
        ],
    );
}

#[test]
fn tensor_op_f32_runs_softmax_from_operator_syntax() {
    let program = br#"
(X 0 0 1)
(X 0 1 2)
(X 0 2 3)

(exec 0
  (, (X $row $d $xv))
  (O (tensor-op-f32
        (op softmax)
        (inputs (X dense 1 3))
        (output (Y dense))
        (from (X $row $d $xv))
        (backend auto))))
"#;

    let output = run_one_step_and_dump(program, "[4] Y $ $ $", "[4] Y _1 _2 _3");
    assert_tensor_cells_close(
        &output,
        "Y",
        &[
            ([0, 0], 0.090_030_57),
            ([0, 1], 0.244_728_48),
            ([0, 2], 0.665_240_94),
        ],
    );
}

#[test]
fn tensor_op_f32_reshapes_2x4_to_2x2x2_row_major() {
    let program = br#"
(X 0 0 0)
(X 0 1 1)
(X 0 2 2)
(X 0 3 3)
(X 1 0 4)
(X 1 1 5)
(X 1 2 6)
(X 1 3 7)

(exec 0
  (, (X $i $j $xv))
  (O (tensor-op-f32
        (op reshape)
        (inputs (X dense 2 4))
        (output (OUT dense 2 2 2))
        (from (X $i $j $xv))
        (backend auto))))
"#;

    let output = run_one_step_and_dump(program, "[5] OUT $ $ $ $", "[5] OUT _1 _2 _3 _4");
    assert_tensor_cells_close(
        &output,
        "OUT",
        &[
            ([0, 0, 0], 0.0),
            ([0, 0, 1], 1.0),
            ([0, 1, 0], 2.0),
            ([0, 1, 1], 3.0),
            ([1, 0, 0], 4.0),
            ([1, 0, 1], 5.0),
            ([1, 1, 0], 6.0),
            ([1, 1, 1], 7.0),
        ],
    );
}

#[test]
fn tensor_op_f32_reshapes_2x2x2_to_2x4_row_major() {
    let program = br#"
(X 0 0 0 0)
(X 0 0 1 1)
(X 0 1 0 2)
(X 0 1 1 3)
(X 1 0 0 4)
(X 1 0 1 5)
(X 1 1 0 6)
(X 1 1 1 7)

(exec 0
  (, (X $i $j $k $xv))
  (O (tensor-op-f32
        (op reshape)
        (inputs (X dense 2 2 2))
        (output (OUT dense 2 4))
        (from (X $i $j $k $xv))
        (backend auto))))
"#;

    let output = run_one_step_and_dump(program, "[4] OUT $ $ $", "[4] OUT _1 _2 _3");
    assert_tensor_cells_close(
        &output,
        "OUT",
        &[
            ([0, 0], 0.0),
            ([0, 1], 1.0),
            ([0, 2], 2.0),
            ([0, 3], 3.0),
            ([1, 0], 4.0),
            ([1, 1], 5.0),
            ([1, 2], 6.0),
            ([1, 3], 7.0),
        ],
    );
}

#[test]
#[should_panic(expected = "explicit output shape does not match inferred operator shape")]
fn tensor_op_f32_rejects_wrong_explicit_output_shape() {
    let mut space = Space::new();
    let program = format!(
        r#"
(A 0 0 1)
(B 0 0 2)
"#
    ) + &tensor_op_einsum_exec(
        "(, (A $i $k $av)\n     (B $k $j $bv))",
        "(A dense 1 1) (B dense 1 1)",
        "(C dense 2 1)",
        "",
    );

    space.add_all_sexpr(program.as_bytes()).unwrap();
    let _ = space.metta_calculus(1);
}

#[test]
fn tensor_op_f32_emit_threshold_materializes_selected_cells() {
    let exec = format!(
        r#"
(CT 0 0 999)
"#
    ) + &tensor_op_einsum_exec(
        "(, (A $i $k $av)\n     (B $k $j $bv))",
        "(A dense 2 3) (B dense 3 2)",
        "(CT dense 2 2)",
        "\n        (emit threshold 100)",
    );
    let program = dense_matmul_2x3_3x2_program(&exec);

    let output = run_one_step_and_dump(&program, "[4] CT $ $ $", "[4] CT _1 _2 _3");
    assert_excludes_all(&output, &["(CT 0 0 999)", "(CT 0 0 58)", "(CT 0 1 64)"]);
    assert_contains_all(&output, &["(CT 1 0 139)", "(CT 1 1 154)"]);
}

#[cfg(feature = "stratified_quiescence")]
#[test]
fn tensor_op_f32_stratified_shrink_rewrite_counts_removed_output_cells() {
    let mut program = dense_matmul_2x3_3x2_cells();
    program.push_str(
        r#"
(CT 0 0 999)
(CT 0 1 888)
(CT 1 0 777)
(CT 1 1 666)

((tensor rewrite)
  (, ((tensor rewrite) $p $t)
     (A $i $k $av)
     (B $k $j $bv))
  (O (tensor-op-f32
        (op einsum ab,bc->ac)
        (inputs (A dense 2 3) (B dense 3 2))
        (output (CT dense 2 2))
        (from (A $i $k $av)
              (B $k $j $bv))
        (emit threshold 100)
        (backend auto))
     (+ (exec (stage tensor rewrite) $p $t))))

(exec (stage tensor rewrite)
      (, ((tensor rewrite) $p $t)
         (A $i $k $av)
         (B $k $j $bv))
      (O (tensor-op-f32
            (op einsum ab,bc->ac)
            (inputs (A dense 2 3) (B dense 3 2))
            (output (CT dense 2 2))
            (from (A $i $k $av)
                  (B $k $j $bv))
            (emit threshold 100)
            (backend auto))
         (+ (exec (stage tensor rewrite) $p $t))))

(exec (quiesce tensor ready)
      (, (CT 1 0 $v))
      (O (+ (ready tensor))))
"#,
    );

    let mut space = Space::new();
    space.add_all_sexpr(program.as_bytes()).unwrap();

    assert_eq!(space.metta_calculus(2), 2);
    let output = dump_selection(&space, "[4] CT $ $ $", "[4] CT _1 _2 _3");
    assert_excludes_all(&output, &["(CT 0 0 999)", "(CT 0 1 888)", "(CT 1 0 777)", "(CT 1 1 666)"]);
    assert_excludes_all(&output, &["(CT 0 0 58)", "(CT 0 1 64)"]);
    assert_contains_all(&output, &["(CT 1 0 139)", "(CT 1 1 154)"]);
    assert!(
        dump_selection(&space, "[2] ready $", "[2] ready _1").is_empty(),
        "barrier advanced before the shrinking rewrite reached quiescence"
    );

    assert_eq!(space.metta_calculus(1), 1);
    assert_eq!(
        dump_selection(&space, "[2] ready $", "[2] ready _1"),
        "(ready tensor)\n"
    );
}

#[test]
fn tensor_op_f32_exec_is_consumed_when_patterns_do_not_match() {
    let mut space = Space::new();
    let program = format!(
        r#"
(A 0 0 1)
"#
    ) + &tensor_op_einsum_exec(
        "(, (A $i $k $av)\n     (B $k $j $bv))",
        "(A dense 1 1) (B dense 1 1)",
        "(C dense 1 1)",
        "",
    );

    space.add_all_sexpr(program.as_bytes()).unwrap();
    assert_eq!(space.metta_calculus(1), 1);
    assert_eq!(space.metta_calculus(1), 0);

    let output = dump_selection(&space, "[4] C $ $ $", "[4] C _1 _2 _3");

    assert!(output.is_empty(), "{output}");
}

#[test]
fn einsum_f32_sink_runs_sparse_dense_matmul_from_mork_program() {
    let program = sparse_dense_matmul_program("(Y 3 2)");

    let output = run_one_step_and_dump(&program, "[4] Y $ $ $", "[4] Y _1 _2 _3");
    assert_contains_all(
        &output,
        &[
            "(Y 0 0 210)",
            "(Y 0 1 260)",
            "(Y 1 0 10)",
            "(Y 1 1 20)",
            "(Y 2 0 0)",
            "(Y 2 1 0)",
        ],
    );
}

#[test]
fn einsum_f32_sink_can_materialize_only_nonzero_output_cells() {
    let program = sparse_dense_matmul_program("(nonzero Y 3 2)");

    let output = run_one_step_and_dump(&program, "[4] Y $ $ $", "[4] Y _1 _2 _3");
    assert_contains_all(
        &output,
        &["(Y 0 0 210)", "(Y 0 1 260)", "(Y 1 0 10)", "(Y 1 1 20)"],
    );
    assert_excludes_all(&output, &["(Y 2 0 0)", "(Y 2 1 0)"]);
}

#[test]
fn einsum_f32_sink_runs_attention_scores_from_mork_program() {
    let program = br#"
(Q 0 0 0 0 1)
(Q 0 0 0 1 2)
(Q 0 0 0 2 3)
(Q 0 0 1 0 4)
(Q 0 0 1 1 5)
(Q 0 0 1 2 6)

(K 0 0 0 0 7)
(K 0 0 0 1 8)
(K 0 0 0 2 9)
(K 0 0 1 0 10)
(K 0 0 1 1 11)
(K 0 0 1 2 12)

(exec 0
  (, (Q $b $h $q $d $qv)
     (K $b $h $k $d $kv))
  (O (einsum-f32 bhqd,bhkd->bhqk
        (Q 1 1 2 3)
        (K 1 1 2 3)
        (Score 1 1 2 2)
        (Q $b $h $q $d $qv)
        (K $b $h $k $d $kv))))
"#;

    let output = run_one_step_and_dump(program, "[6] Score $ $ $ $ $", "[6] Score _1 _2 _3 _4 _5");
    assert_contains_all(
        &output,
        &[
            "(Score 0 0 0 0 50)",
            "(Score 0 0 0 1 68)",
            "(Score 0 0 1 0 122)",
            "(Score 0 0 1 1 167)",
        ],
    );
}

#[test]
fn tensor_op_f32_runs_scaled_dot_product_attention_from_operator_syntax() {
    let program = [
        ATTENTION_APPLY_FIXTURE,
        br#"
(exec 0
  (, (Q $b $h $q $d $qv)
     (K $b $h $k $d $kv)
     (V $b $h $k $vd $vv))
  (O (tensor-op-f32
        (op attention scaled-dot)
        (inputs (Q dense 1 1 2 2)
                (K dense 1 1 2 2)
                (V dense 1 1 2 2))
        (output (Ctx dense))
        (from (Q $b $h $q $d $qv)
              (K $b $h $k $d $kv)
              (V $b $h $k $vd $vv))
        (backend auto))))
"#,
    ]
    .concat();

    let output = run_one_step_and_dump(&program, "[6] Ctx $ $ $ $ $", "[6] Ctx _1 _2 _3 _4 _5");
    assert_contains_all(
        &output,
        &[
            "(Ctx 0 0 0 0 12)",
            "(Ctx 0 0 0 1 24)",
            "(Ctx 0 0 1 0 12)",
            "(Ctx 0 0 1 1 24)",
        ],
    );
}

#[test]
fn attention_f32_sink_runs_scaled_dot_product_attention_from_mork_program() {
    let program = [
        ATTENTION_APPLY_FIXTURE,
        br#"
(exec 0
  (, (Q $b $h $q $d $qv)
     (K $b $h $k $d $kv)
     (V $b $h $k $vd $vv))
  (O (attention-f32
        (Q 1 1 2 2)
        (K 1 1 2 2)
        (V 1 1 2 2)
        (Ctx 1 1 2 2)
        (Q $b $h $q $d $qv)
        (K $b $h $k $d $kv)
        (V $b $h $k $vd $vv))))
"#,
    ]
    .concat();

    let output = run_one_step_and_dump(&program, "[6] Ctx $ $ $ $ $", "[6] Ctx _1 _2 _3 _4 _5");
    assert_contains_all(
        &output,
        &[
            "(Ctx 0 0 0 0 12)",
            "(Ctx 0 0 0 1 24)",
            "(Ctx 0 0 1 0 12)",
            "(Ctx 0 0 1 1 24)",
        ],
    );
}
