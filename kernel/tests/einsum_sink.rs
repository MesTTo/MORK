use mork::expr;
use mork::space::Space;
use std::collections::BTreeMap;

// dense16 coverage intentionally excludes attention-f32 and tensor-op attention.
// Those paths consume scalar cells from each rule match instead of sharing the
// direct EinsumInput gather path that can scan all strip widths.

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

fn linear_to_indices(mut linear: usize, shape: &[usize], out: &mut [usize]) {
    for axis in (0..shape.len()).rev() {
        out[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
}

fn dense_map_from_values(shape: &[usize], values: &[f32]) -> BTreeMap<Vec<usize>, f32> {
    let expected_len: usize = shape.iter().product();
    assert_eq!(values.len(), expected_len);
    let mut out = BTreeMap::new();
    let mut indices = vec![0usize; shape.len()];
    for (linear, &value) in values.iter().enumerate() {
        linear_to_indices(linear, shape, &mut indices);
        out.insert(indices.clone(), value);
    }
    out
}

fn assert_dense_maps_close(
    actual: &BTreeMap<Vec<usize>, f32>,
    expected: &BTreeMap<Vec<usize>, f32>,
) {
    assert_eq!(actual.len(), expected.len(), "actual {actual:?}");
    for (indices, expected_value) in expected {
        let actual_value = actual
            .get(indices)
            .unwrap_or_else(|| panic!("missing tensor cell {indices:?} in {actual:?}"));
        let diff = (*actual_value - *expected_value).abs();
        assert!(
            diff <= 1.0e-5,
            "{indices:?}: expected {expected_value}, got {actual_value}, diff {diff}"
        );
    }
}

fn dense16_atom_count(shape: &[usize]) -> usize {
    assert!(!shape.is_empty());
    let last = shape[shape.len() - 1];
    let rows = shape[..shape.len() - 1].iter().product::<usize>();
    rows * last.div_ceil(16)
}

fn dense16_cells(name: &str, shape: &[usize], values: &[f32]) -> String {
    assert!(!shape.is_empty());
    let expected_len: usize = shape.iter().product();
    assert_eq!(values.len(), expected_len);
    let rank = shape.len();
    let last = shape[rank - 1];
    assert!(last > 0);
    let row_count = values.len() / last;
    let mut base_indices = vec![0usize; rank - 1];
    let mut out = String::new();

    for row in 0..row_count {
        linear_to_indices(row, &shape[..rank - 1], &mut base_indices);
        let row_start = row * last;
        for strip_start in (0..last).step_by(16) {
            let value_count = 16.min(last - strip_start);
            out.push('(');
            out.push_str(name);
            for index in &base_indices {
                out.push(' ');
                out.push_str(&index.to_string());
            }
            out.push(' ');
            out.push_str(&(strip_start / 16).to_string());
            for value in &values[row_start + strip_start..row_start + strip_start + value_count] {
                out.push(' ');
                out.push_str(&format_tensor_value(*value));
            }
            out.push_str(")\n");
        }
    }

    out
}

fn repeated_token(token: &str, count: usize) -> String {
    vec![token; count].join(" ")
}

fn numbered_vars(count: usize) -> String {
    (1..=count)
        .map(|index| format!("_{index}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn dump_dense16_selection(space: &Space, name: &str, rank: usize) -> String {
    let mut output = String::new();
    for value_count in 1..=16 {
        let arity = rank + value_count + 1;
        let query = format!("[{arity}] {name} {}", repeated_token("$", arity - 1));
        let template = format!("[{arity}] {name} {}", numbered_vars(arity - 1));
        output.push_str(&dump_selection(space, &query, &template));
    }
    output
}

fn parse_dumped_dense16_cells(output: &str, name: &str, rank: usize) -> BTreeMap<Vec<usize>, f32> {
    let mut cells = BTreeMap::new();
    for line in output.lines() {
        let cell = line
            .strip_prefix('(')
            .and_then(|line| line.strip_suffix(')'))
            .unwrap_or_else(|| panic!("dumped tensor cell is not parenthesized: {line:?}"));
        let parts: Vec<&str> = cell.split_whitespace().collect();
        assert_eq!(parts[0], name, "dumped tensor cell name changed");
        let value_count = parts.len() - (rank + 1);
        assert!(
            (1..=16).contains(&value_count),
            "dense16 dump has invalid value count {value_count}: {line:?}"
        );
        let base_indices: Vec<usize> = parts[1..rank]
            .iter()
            .map(|index| {
                index.parse::<usize>().unwrap_or_else(|err| {
                    panic!("could not parse dumped tensor index {line:?}: {err}")
                })
            })
            .collect();
        let strip = parts[rank]
            .parse::<usize>()
            .unwrap_or_else(|err| panic!("could not parse dense16 strip {line:?}: {err}"));
        for (offset, value) in parts[rank + 1..].iter().enumerate() {
            let mut indices = base_indices.clone();
            indices.push(strip * 16 + offset);
            let value = value.parse::<f32>().unwrap_or_else(|err| {
                panic!("could not parse dumped tensor value {line:?}: {err}")
            });
            cells.insert(indices, value);
        }
    }
    cells
}

fn run_one_step_and_parse_dense16(
    program: &[u8],
    name: &str,
    rank: usize,
) -> BTreeMap<Vec<usize>, f32> {
    let mut space = Space::new();
    space.add_all_sexpr(program).unwrap();
    assert_eq!(space.metta_calculus(1), 1);
    let output = dump_dense16_selection(&space, name, rank);
    parse_dumped_dense16_cells(&output, name, rank)
}

fn strip_template(name: &str, rank: usize, value_count: usize) -> String {
    let mut parts = vec![name.to_string()];
    parts.extend((0..rank - 1).map(|index| format!("$i{index}")));
    parts.push("$sb".to_string());
    parts.extend((0..value_count).map(|index| format!("$v{index}")));
    format!("({})", parts.join(" "))
}

fn ground_scalar_template(name: &str, rank: usize) -> String {
    let mut parts = vec![name.to_string()];
    parts.extend(std::iter::repeat_n("0".to_string(), rank + 1));
    format!("({})", parts.join(" "))
}

fn ground_strip_template(name: &str, rank: usize, value_count: usize) -> String {
    let mut parts = vec![name.to_string()];
    parts.extend(std::iter::repeat_n("0".to_string(), rank - 1));
    parts.push("0".to_string());
    parts.extend(std::iter::repeat_n("0".to_string(), value_count));
    format!("({})", parts.join(" "))
}

fn identity_matrix_values(size: usize) -> Vec<f32> {
    let mut values = vec![0.0; size * size];
    for index in 0..size {
        values[index * size + index] = 1.0;
    }
    values
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

fn einsum_copy_program(
    a_cells: &str,
    b_cells: &str,
    a_decl: &str,
    b_decl: &str,
    output_decl: &str,
    a_template: &str,
    b_template: &str,
) -> Vec<u8> {
    format!(
        r#"
{a_cells}
{b_cells}
(go)

(exec 0
  (, (go))
  (O (einsum-f32 ab,bc->ac
        {a_decl}
        {b_decl}
        {output_decl}
        {a_template}
        {b_template})))
"#
    )
    .into_bytes()
}

#[test]
fn einsum_f32_dense16_matches_scalar_with_ragged_last_dim() {
    let a_shape = [5, 17];
    let b_shape = [17, 17];
    let a_values: Vec<f32> = (0..85).map(|index| index as f32 * 0.25 - 7.0).collect();
    let b_values = identity_matrix_values(17);

    let scalar_program = einsum_copy_program(
        &matrix_cells("A", 5, 17, &a_values),
        &matrix_cells("B", 17, 17, &b_values),
        "(A 5 17)",
        "(B 17 17)",
        "(C 5 17)",
        &ground_scalar_template("A", 2),
        &ground_scalar_template("B", 2),
    );
    let mut scalar_space = Space::new();
    let scalar_loaded = scalar_space.add_all_sexpr(&scalar_program).unwrap();
    assert_eq!(scalar_loaded, a_values.len() + b_values.len() + 2);
    assert_eq!(scalar_space.metta_calculus(1), 1);
    let scalar_output = dump_selection(&scalar_space, "[4] C $ $ $", "[4] C _1 _2 _3");
    let scalar_cells = parse_dumped_tensor_cells(&scalar_output, "C");

    let strip_program = einsum_copy_program(
        &dense16_cells("A", &a_shape, &a_values),
        &dense16_cells("B", &b_shape, &b_values),
        "(A dense16 5 17)",
        "(B dense16 17 17)",
        "(C dense16 5 17)",
        &ground_strip_template("A", 2, 16),
        &ground_strip_template("B", 2, 16),
    );
    let mut strip_space = Space::new();
    let strip_loaded = strip_space.add_all_sexpr(&strip_program).unwrap();
    assert_eq!(
        strip_loaded,
        dense16_atom_count(&a_shape) + dense16_atom_count(&b_shape) + 2
    );
    assert_eq!(strip_space.metta_calculus(1), 1);
    let strip_output = dump_dense16_selection(&strip_space, "C", 2);
    let strip_cells = parse_dumped_dense16_cells(&strip_output, "C", 2);

    assert_dense_maps_close(&strip_cells, &scalar_cells);
}

#[test]
fn einsum_f32_mixes_dense16_and_scalar_storage() {
    let a_shape = [2, 17];
    let b_shape = [17, 17];
    let a_values: Vec<f32> = (0..34).map(|index| index as f32 - 3.0).collect();
    let b_values = identity_matrix_values(17);
    let expected = dense_map_from_values(&a_shape, &a_values);

    let strip_input_scalar_output = einsum_copy_program(
        &dense16_cells("A", &a_shape, &a_values),
        &dense16_cells("B", &b_shape, &b_values),
        "(A dense16 2 17)",
        "(B dense16 17 17)",
        "(C 2 17)",
        &ground_strip_template("A", 2, 16),
        &ground_strip_template("B", 2, 16),
    );
    let scalar_output =
        run_one_step_and_dump(&strip_input_scalar_output, "[4] C $ $ $", "[4] C _1 _2 _3");
    let scalar_output = parse_dumped_tensor_cells(&scalar_output, "C");
    assert_dense_maps_close(&scalar_output, &expected);

    let scalar_input_strip_output = einsum_copy_program(
        &matrix_cells("A", 2, 17, &a_values),
        &matrix_cells("B", 17, 17, &b_values),
        "(A 2 17)",
        "(B 17 17)",
        "(D dense16 2 17)",
        &ground_scalar_template("A", 2),
        &ground_scalar_template("B", 2),
    );
    let mut space = Space::new();
    space.add_all_sexpr(&scalar_input_strip_output).unwrap();
    assert_eq!(space.metta_calculus(1), 1);
    let strip_output = dump_dense16_selection(&space, "D", 2);
    let strip_output = parse_dumped_dense16_cells(&strip_output, "D", 2);
    assert_dense_maps_close(&strip_output, &expected);
}

#[test]
fn tensor_op_f32_einsum_uses_dense16_direct_input_and_output() {
    let a_shape = [2, 17];
    let b_shape = [17, 17];
    let a_values: Vec<f32> = (0..34).map(|index| index as f32 * 0.5 + 1.0).collect();
    let b_values = identity_matrix_values(17);
    let program = format!(
        r#"
{}
{}
(go)

(exec 0
  (, (go))
  (O (tensor-op-f32
        (op einsum ab,bc->ac)
        (inputs (A dense16 2 17) (B dense16 17 17))
        (output (T dense16))
        (from {}
              {})
        (backend auto))))
"#,
        dense16_cells("A", &a_shape, &a_values),
        dense16_cells("B", &b_shape, &b_values),
        ground_strip_template("A", 2, 16),
        ground_strip_template("B", 2, 16),
    )
    .into_bytes();

    let mut space = Space::new();
    let loaded = space.add_all_sexpr(&program).unwrap();
    assert_eq!(
        loaded,
        dense16_atom_count(&a_shape) + dense16_atom_count(&b_shape) + 2
    );
    assert_eq!(space.metta_calculus(1), 1);
    let output = dump_dense16_selection(&space, "T", 2);
    let output = parse_dumped_dense16_cells(&output, "T", 2);
    let expected = dense_map_from_values(&a_shape, &a_values);
    assert_dense_maps_close(&output, &expected);
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
fn tensor_op_f32_runs_add_with_dense16_cells() {
    let a_shape = [2, 4];
    let a_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let program = format!(
        r#"
{}
{}

(exec 0
  (, (A $i $s $a0 $a1 $a2 $a3)
     (B $i $s $b0 $b1 $b2 $b3))
  (O (tensor-op-f32
        (op add)
        (inputs (A dense16 2 4) (B dense16 2 4))
        (output (C dense16))
        (from (A $i $s $a0 $a1 $a2 $a3)
              (B $i $s $b0 $b1 $b2 $b3))
        (backend auto))))
"#,
        dense16_cells("A", &a_shape, &a_values),
        dense16_cells("B", &a_shape, &b_values),
    )
    .into_bytes();

    let output = run_one_step_and_parse_dense16(&program, "C", 2);
    let expected_values: Vec<f32> = a_values
        .iter()
        .zip(b_values)
        .map(|(&lhs, rhs)| lhs + rhs)
        .collect();
    let expected = dense_map_from_values(&a_shape, &expected_values);
    assert_dense_maps_close(&output, &expected);
}

#[test]
fn tensor_op_f32_runs_layernorm_with_dense16_cells() {
    let x_shape = [1, 4];
    let scale_shape = [4];
    let program = format!(
        r#"
{}
{}
{}

(exec 0
  (, (X $row $s $x0 $x1 $x2 $x3)
     (G $s $g0 $g1 $g2 $g3)
     (B $s $b0 $b1 $b2 $b3))
  (O (tensor-op-f32
        (op layernorm 1e-5)
        (inputs (X dense16 1 4) (G dense16 4) (B dense16 4))
        (output (Y dense16))
        (from (X $row $s $x0 $x1 $x2 $x3)
              (G $s $g0 $g1 $g2 $g3)
              (B $s $b0 $b1 $b2 $b3))
        (backend auto))))
"#,
        dense16_cells("X", &x_shape, &[1.0, 2.0, 3.0, 4.0]),
        dense16_cells("G", &scale_shape, &[1.0, 1.0, 1.0, 1.0]),
        dense16_cells("B", &scale_shape, &[0.0, 0.0, 0.0, 0.0]),
    )
    .into_bytes();

    let output = run_one_step_and_parse_dense16(&program, "Y", 2);
    let expected = dense_map_from_values(
        &x_shape,
        &[-1.341_635_5, -0.447_211_83, 0.447_211_83, 1.341_635_5],
    );
    assert_dense_maps_close(&output, &expected);
}

#[test]
fn tensor_op_f32_runs_gelu_with_dense16_cells() {
    let shape = [4];
    let program = format!(
        r#"
{}

(exec 0
  (, (X $s $x0 $x1 $x2 $x3))
  (O (tensor-op-f32
        (op gelu)
        (inputs (X dense16 4))
        (output (Y dense16))
        (from (X $s $x0 $x1 $x2 $x3))
        (backend auto))))
"#,
        dense16_cells("X", &shape, &[-1.0, 0.0, 1.0, 2.0]),
    )
    .into_bytes();

    let output = run_one_step_and_parse_dense16(&program, "Y", 1);
    let expected = dense_map_from_values(&shape, &[-0.158_808_01, 0.0, 0.841_192, 1.954_597_7]);
    assert_dense_maps_close(&output, &expected);
}

#[test]
fn tensor_op_f32_runs_softmax_with_dense16_cells() {
    let shape = [1, 4];
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = input.iter().map(|value| (*value - max).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    let expected_values: Vec<f32> = exp_values.iter().map(|value| value / sum).collect();
    let program = format!(
        r#"
{}

(exec 0
  (, (X $row $s $x0 $x1 $x2 $x3))
  (O (tensor-op-f32
        (op softmax)
        (inputs (X dense16 1 4))
        (output (Y dense16))
        (from (X $row $s $x0 $x1 $x2 $x3))
        (backend auto))))
"#,
        dense16_cells("X", &shape, &input),
    )
    .into_bytes();

    let output = run_one_step_and_parse_dense16(&program, "Y", 2);
    let expected = dense_map_from_values(&shape, &expected_values);
    assert_dense_maps_close(&output, &expected);
}

#[test]
fn tensor_op_f32_runs_reshape_with_dense16_cells() {
    let input_shape = [2, 4];
    let output_shape = [2, 2, 2];
    let values: Vec<f32> = (0..8).map(|value| value as f32).collect();
    let program = format!(
        r#"
{}

(exec 0
  (, (X $row $s $x0 $x1 $x2 $x3))
  (O (tensor-op-f32
        (op reshape)
        (inputs (X dense16 2 4))
        (output (OUT dense16 2 2 2))
        (from (X $row $s $x0 $x1 $x2 $x3))
        (backend auto))))
"#,
        dense16_cells("X", &input_shape, &values),
    )
    .into_bytes();

    let output = run_one_step_and_parse_dense16(&program, "OUT", 3);
    let expected = dense_map_from_values(&output_shape, &values);
    assert_dense_maps_close(&output, &expected);
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
    assert_excludes_all(
        &output,
        &[
            "(CT 0 0 999)",
            "(CT 0 1 888)",
            "(CT 1 0 777)",
            "(CT 1 1 666)",
        ],
    );
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

#[cfg(feature = "stratified_quiescence")]
#[test]
fn einsum_f32_stratified_dense16_shrink_rewrite_counts_removed_output_strips() {
    let a_shape = [1, 17];
    let b_shape = [17, 17];
    let a_values: Vec<f32> = (0..17).map(|index| index as f32 + 1.0).collect();
    let b_values = identity_matrix_values(17);
    let sink = format!(
        r#"(einsum-f32 ab,bc->ac
        (A dense16 1 17)
        (B dense16 17 17)
        (CT dense16 1 17)
        {}
        {})"#,
        ground_strip_template("A", 2, 16),
        ground_strip_template("B", 2, 16),
    );
    let program = format!(
        r#"
{}
{}
(go)
(CT 0 0 999 998 997 996 995 994 993 992 991 990 989 988 987 986 985 984)
(CT 0 1 888)
(CT 0 2 777)

((tensor rewrite)
  (, ((tensor rewrite) $p $t)
     (go))
  (O {sink}
     (+ (exec (stage tensor rewrite) $p $t))))

(exec (stage tensor rewrite)
      (, ((tensor rewrite) $p $t)
         (go))
      (O {sink}
         (+ (exec (stage tensor rewrite) $p $t))))

(exec (quiesce tensor ready)
      (, (CT 0 1 $v))
      (O (+ (ready tensor))))
"#,
        dense16_cells("A", &a_shape, &a_values),
        dense16_cells("B", &b_shape, &b_values),
    );

    let mut space = Space::new();
    space.add_all_sexpr(program.as_bytes()).unwrap();

    assert_eq!(space.metta_calculus(2), 2);
    let output = dump_dense16_selection(&space, "CT", 2);
    let cells = parse_dumped_dense16_cells(&output, "CT", 2);
    let expected = dense_map_from_values(&a_shape, &a_values);
    assert_dense_maps_close(&cells, &expected);
    assert!(
        !cells.contains_key(&vec![0, 32]),
        "stale strip survived: {cells:?}"
    );
    assert!(
        dump_selection(&space, "[2] ready $", "[2] ready _1").is_empty(),
        "barrier advanced before the dense16 shrinking rewrite reached quiescence"
    );

    assert_eq!(space.metta_calculus(1), 1);
    assert_eq!(
        dump_selection(&space, "[2] ready $", "[2] ready _1"),
        "(ready tensor)\n"
    );
}

#[test]
fn einsum_f32_dense16_ragged_tail_round_trips_through_output_then_input() {
    let a_shape = [5, 17];
    let b_shape = [17, 17];
    let a_values: Vec<f32> = (0..85).map(|index| index as f32 / 3.0 - 4.0).collect();
    let b_values = identity_matrix_values(17);
    let first_sink = format!(
        r#"(einsum-f32 ab,bc->ac
        (A dense16 5 17)
        (B dense16 17 17)
        (C dense16 5 17)
        {}
        {})"#,
        ground_strip_template("A", 2, 16),
        ground_strip_template("B", 2, 16),
    );
    let second_sink = format!(
        r#"(einsum-f32 ab,bc->ac
        (C dense16 5 17)
        (B dense16 17 17)
        (D dense16 5 17)
        {}
        {})"#,
        ground_strip_template("C", 2, 16),
        ground_strip_template("B", 2, 16),
    );
    let program = format!(
        r#"
{}
{}
(roundtrip start)

(exec 0
  (, (roundtrip start))
  (O {first_sink}
     (+ (roundtrip written))))

(exec 0
  (, (roundtrip written))
  (O {second_sink}))
"#,
        dense16_cells("A", &a_shape, &a_values),
        dense16_cells("B", &b_shape, &b_values),
    )
    .into_bytes();

    let mut space = Space::new();
    space.add_all_sexpr(&program).unwrap();
    assert_eq!(space.metta_calculus(1), 1);
    assert_eq!(space.metta_calculus(1), 1);

    let output = dump_dense16_selection(&space, "D", 2);
    let output = parse_dumped_dense16_cells(&output, "D", 2);
    let expected = dense_map_from_values(&a_shape, &a_values);
    assert_dense_maps_close(&output, &expected);
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
