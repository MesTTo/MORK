use mork::expr;
use mork::space::Space;

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

#[test]
fn einsum_f32_sink_runs_dense_matmul_from_mork_program() {
    let program = br#"
(A 0 0 1)
(A 0 1 2)
(A 0 2 3)
(A 1 0 4)
(A 1 1 5)
(A 1 2 6)

(B 0 0 7)
(B 0 1 8)
(B 1 0 9)
(B 1 1 10)
(B 2 0 11)
(B 2 1 12)

(exec 0
  (, (A $i $k $av)
     (B $k $j $bv))
  (O (einsum-f32 ab,bc->ac
        (A 2 3)
        (B 3 2)
        (C 2 2)
        (A $i $k $av)
        (B $k $j $bv))))
"#;

    let output = run_one_step_and_dump(program, "[4] C $ $ $", "[4] C _1 _2 _3");
    assert_contains_all(
        &output,
        &["(C 0 0 58)", "(C 0 1 64)", "(C 1 0 139)", "(C 1 1 154)"],
    );
}

#[test]
fn tensor_op_f32_runs_dense_matmul_from_operator_syntax() {
    let program = br#"
(A 0 0 1)
(A 0 1 2)
(A 0 2 3)
(A 1 0 4)
(A 1 1 5)
(A 1 2 6)

(B 0 0 7)
(B 0 1 8)
(B 1 0 9)
(B 1 1 10)
(B 2 0 11)
(B 2 1 12)

(exec 0
  (, (A $i $k $av)
     (B $k $j $bv))
  (O (tensor-op-f32
        (op einsum ab,bc->ac)
        (inputs (A dense 2 3) (B dense 3 2))
        (output (C dense))
        (from (A $i $k $av)
              (B $k $j $bv))
        (backend auto))))
"#;

    let output = run_one_step_and_dump(program, "[4] C $ $ $", "[4] C _1 _2 _3");
    assert_contains_all(
        &output,
        &["(C 0 0 58)", "(C 0 1 64)", "(C 1 0 139)", "(C 1 1 154)"],
    );
}

#[test]
#[should_panic(expected = "explicit output shape does not match inferred operator shape")]
fn tensor_op_f32_rejects_wrong_explicit_output_shape() {
    let mut space = Space::new();
    let program = br#"
(A 0 0 1)
(B 0 0 2)

(exec 0
  (, (A $i $k $av)
     (B $k $j $bv))
  (O (tensor-op-f32
        (op einsum ab,bc->ac)
        (inputs (A dense 1 1) (B dense 1 1))
        (output (C dense 2 1))
        (from (A $i $k $av)
              (B $k $j $bv))
        (backend auto))))
"#;

    space.add_all_sexpr(program).unwrap();
    let _ = space.metta_calculus(1);
}

#[test]
fn tensor_op_f32_emit_threshold_materializes_selected_cells() {
    let program = br#"
(A 0 0 1)
(A 0 1 2)
(A 0 2 3)
(A 1 0 4)
(A 1 1 5)
(A 1 2 6)

(B 0 0 7)
(B 0 1 8)
(B 1 0 9)
(B 1 1 10)
(B 2 0 11)
(B 2 1 12)
(CT 0 0 999)

(exec 0
  (, (A $i $k $av)
     (B $k $j $bv))
  (O (tensor-op-f32
        (op einsum ab,bc->ac)
        (inputs (A dense 2 3) (B dense 3 2))
        (output (CT dense 2 2))
        (from (A $i $k $av)
              (B $k $j $bv))
        (emit threshold 100)
        (backend auto))))
"#;

    let output = run_one_step_and_dump(program, "[4] CT $ $ $", "[4] CT _1 _2 _3");
    assert_excludes_all(&output, &["(CT 0 0 999)", "(CT 0 0 58)", "(CT 0 1 64)"]);
    assert_contains_all(&output, &["(CT 1 0 139)", "(CT 1 1 154)"]);
}

#[test]
fn tensor_op_f32_exec_is_consumed_when_patterns_do_not_match() {
    let mut space = Space::new();
    let program = br#"
(A 0 0 1)

(exec 0
  (, (A $i $k $av)
     (B $k $j $bv))
  (O (tensor-op-f32
        (op einsum ab,bc->ac)
        (inputs (A dense 1 1) (B dense 1 1))
        (output (C dense 1 1))
        (from (A $i $k $av)
              (B $k $j $bv))
        (backend auto))))
"#;

    space.add_all_sexpr(program).unwrap();
    assert_eq!(space.metta_calculus(1), 1);
    assert_eq!(space.metta_calculus(1), 0);

    let output = dump_selection(&space, "[4] C $ $ $", "[4] C _1 _2 _3");

    assert!(output.is_empty(), "{output}");
}

#[test]
fn einsum_f32_sink_runs_sparse_dense_matmul_from_mork_program() {
    let program = br#"
(A 0 1 2)
(A 0 2 3)
(A 1 0 1)

(X 0 0 10)
(X 0 1 20)
(X 1 0 30)
(X 1 1 40)
(X 2 0 50)
(X 2 1 60)

(exec 0
  (, (A $i $k $av)
     (X $k $j $xv))
  (O (einsum-f32 ab,bc->ac
        (csr A 3 3)
        (X 3 2)
        (Y 3 2)
        (A $i $k $av)
        (X $k $j $xv))))
"#;

    let output = run_one_step_and_dump(program, "[4] Y $ $ $", "[4] Y _1 _2 _3");
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
    let program = br#"
(A 0 1 2)
(A 0 2 3)
(A 1 0 1)

(X 0 0 10)
(X 0 1 20)
(X 1 0 30)
(X 1 1 40)
(X 2 0 50)
(X 2 1 60)

(exec 0
  (, (A $i $k $av)
     (X $k $j $xv))
  (O (einsum-f32 ab,bc->ac
        (csr A 3 3)
        (X 3 2)
        (nonzero Y 3 2)
        (A $i $k $av)
        (X $k $j $xv))))
"#;

    let output = run_one_step_and_dump(program, "[4] Y $ $ $", "[4] Y _1 _2 _3");
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
