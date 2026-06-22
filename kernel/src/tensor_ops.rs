use linalg::csr::Csr;
use linalg::dense::Dense;
use linalg::einsum::infer_output_shapes;
use linalg::jit::{EinsumF32Plan, JitInput};
use linalg::tensor::NDIndex;
use log::trace;
use mork_expr::{Expr, ExprEnv, Tag, item_byte};
use pathmap::PathMap;
use pathmap::utils::BitMask;
use pathmap::zipper::*;
use std::collections::BTreeMap;

pub(crate) fn expr_args(e: Expr) -> Vec<Expr> {
    let mut envs = Vec::new();
    ExprEnv::new(0, e).args(&mut envs);
    envs.into_iter().map(|env| env.subsexpr()).collect()
}

pub(crate) fn symbol_string(e: Expr) -> String {
    let bytes = unsafe {
        e.symbol()
            .and_then(|s| s.as_ref())
            .unwrap_or_else(|| panic!("expected symbol, got {:?}", e))
    };
    std::str::from_utf8(bytes)
        .unwrap_or_else(|_| panic!("symbol is not utf-8: {bytes:?}"))
        .to_string()
}

pub(crate) fn symbol_usize(e: Expr) -> usize {
    let s = symbol_string(e);
    s.parse::<usize>()
        .unwrap_or_else(|_| panic!("expected usize symbol, got {s:?}"))
}

pub(crate) fn symbol_f32(e: Expr) -> f32 {
    let s = symbol_string(e);
    s.parse::<f32>()
        .unwrap_or_else(|_| panic!("expected f32 symbol, got {s:?}"))
}

fn parse_usize_shape(args: &[Expr]) -> Vec<usize> {
    args.iter().map(|&arg| symbol_usize(arg)).collect()
}

pub(crate) fn parse_tensor_decl(e: Expr) -> (String, Vec<usize>) {
    let args = expr_args(e);
    assert!(
        args.len() >= 2,
        "tensor declaration must be shaped like (Name dim...)"
    );
    let name = symbol_string(args[0]);
    let shape = parse_usize_shape(&args[1..]);
    (name, shape)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorInputKind {
    Dense,
    Csr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum TensorOutputKind {
    Dense,
    NonZero,
    Threshold(f32),
}

impl TensorOutputKind {
    fn should_emit(self, value: f32) -> bool {
        match self {
            Self::Dense => true,
            Self::NonZero => value != 0.0,
            Self::Threshold(min_abs) => value.abs() >= min_abs,
        }
    }
}

fn parse_qualified_tensor_decl<T: Copy>(
    args: &[Expr],
    kind_for: impl Fn(&str) -> Option<T>,
) -> Option<(T, String, Vec<usize>)> {
    if args.len() < 3 {
        return None;
    }

    let first = symbol_string(args[0]);
    if let Some(kind) = kind_for(&first) {
        return Some((kind, symbol_string(args[1]), parse_usize_shape(&args[2..])));
    }

    let second = symbol_string(args[1]);
    kind_for(&second).map(|kind| (kind, first, parse_usize_shape(&args[2..])))
}

fn tensor_input_kind(token: &str) -> Option<TensorInputKind> {
    match token {
        "dense" => Some(TensorInputKind::Dense),
        "csr" => Some(TensorInputKind::Csr),
        _ => None,
    }
}

fn tensor_output_kind(token: &str) -> Option<TensorOutputKind> {
    match token {
        "dense" => Some(TensorOutputKind::Dense),
        "nonzero" | "nz" | "sparse" => Some(TensorOutputKind::NonZero),
        _ => None,
    }
}

pub(crate) fn parse_input_tensor_decl(e: Expr) -> (TensorInputKind, String, Vec<usize>) {
    let args = expr_args(e);
    assert!(
        args.len() >= 2,
        "tensor declaration must be shaped like (Name dim...) or (csr Name dim...)"
    );

    if let Some(parsed) = parse_qualified_tensor_decl(&args, tensor_input_kind) {
        return parsed;
    }

    let first = symbol_string(args[0]);
    let shape = parse_usize_shape(&args[1..]);
    (TensorInputKind::Dense, first, shape)
}

pub(crate) fn parse_output_tensor_decl(e: Expr) -> (TensorOutputKind, String, Vec<usize>) {
    let args = expr_args(e);
    assert!(
        args.len() >= 2,
        "tensor output declaration must be shaped like (Name dim...) or (nonzero Name dim...)"
    );

    if let Some(parsed) = parse_qualified_tensor_decl(&args, tensor_output_kind) {
        return parsed;
    }

    let first = symbol_string(args[0]);
    let shape = parse_usize_shape(&args[1..]);
    (TensorOutputKind::Dense, first, shape)
}

pub(crate) fn parse_cell(e: Expr) -> (String, Vec<usize>, f32) {
    let args = expr_args(e);
    assert!(
        args.len() >= 2,
        "tensor cell must be shaped like (Name index... value)"
    );
    let name = symbol_string(args[0]);
    let last = args.len() - 1;
    let indices = args[1..last].iter().map(|&arg| symbol_usize(arg)).collect();
    let value = symbol_f32(args[last]);
    (name, indices, value)
}

fn write_symbol(buf: &mut Vec<u8>, symbol: &str) {
    assert!(
        !symbol.is_empty() && symbol.len() < 64,
        "MORK symbols must have length 1..63, got {symbol:?}"
    );
    buf.push(item_byte(Tag::SymbolSize(symbol.len() as u8)));
    buf.extend_from_slice(symbol.as_bytes());
}

fn write_tensor_cell(buf: &mut Vec<u8>, name: &str, indices: &[usize], value: f32) {
    let arity = indices.len() + 2;
    assert!(arity < 64, "tensor cell arity is too large: {arity}");
    buf.push(item_byte(Tag::Arity(arity as u8)));
    write_symbol(buf, name);
    for index in indices {
        write_symbol(buf, &index.to_string());
    }
    let value = if value == 0.0 {
        "0".to_string()
    } else {
        value.to_string()
    };
    write_symbol(buf, &value);
}

pub(crate) fn tensor_cell_prefix(name: &str, rank: usize) -> &'static [u8] {
    let mut prefix = Vec::new();
    let arity = rank + 2;
    assert!(arity < 64, "tensor cell arity is too large: {arity}");
    prefix.push(item_byte(Tag::Arity(arity as u8)));
    write_symbol(&mut prefix, name);
    Box::leak(prefix.into_boxed_slice())
}

fn linear_to_index(mut linear: usize, shape: &[usize], out: &mut [usize]) {
    for axis in (0..shape.len()).rev() {
        out[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
}

pub(crate) fn validate_tensor_cell_template(template: Expr, name: &str, rank: usize) {
    let cell_args = expr_args(template);
    assert_eq!(
        symbol_string(cell_args[0]),
        name,
        "input cell template name does not match input declaration"
    );
    assert_eq!(
        cell_args.len(),
        rank + 2,
        "input cell template rank does not match input declaration"
    );
}

fn flatten_csr_row(indices: &[usize], shape: &[usize]) -> usize {
    let mut row = 0usize;
    for axis in 0..shape.len() - 1 {
        row = row * shape[axis] + indices[axis];
    }
    row
}

fn csr_from_entries(shape: &[usize], entries: &BTreeMap<Vec<usize>, f32>) -> Csr<u32, f32> {
    assert!(
        shape.len() >= 2,
        "CSR input shape needs rank >= 2, got {shape:?}"
    );
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let cols = shape[shape.len() - 1];
    let mut row_ptr = vec![0usize; rows + 1];
    let mut col_idx = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    let mut cur_row = 0usize;

    for (indices, &value) in entries {
        assert_eq!(
            indices.len(),
            shape.len(),
            "CSR cell rank does not match declaration"
        );
        if value == 0.0 {
            continue;
        }
        for (axis, (&index, &dim)) in indices.iter().zip(shape).enumerate() {
            assert!(
                index < dim,
                "CSR index {index} on axis {axis} is outside dimension {dim}"
            );
        }
        let row = flatten_csr_row(indices, shape);
        while cur_row <= row {
            row_ptr[cur_row] = col_idx.len();
            cur_row += 1;
        }
        let col = indices[shape.len() - 1];
        assert!(col < cols, "CSR column {col} outside dimension {cols}");
        let col = u32::try_from(col).unwrap_or_else(|_| panic!("CSR column too large: {col}"));
        col_idx.push(col);
        values.push(value);
    }
    while cur_row <= rows {
        row_ptr[cur_row] = col_idx.len();
        cur_row += 1;
    }

    Csr::from_parts(shape.to_vec(), row_ptr, col_idx, values)
}

pub(crate) enum EinsumInput {
    Dense {
        name: String,
        tensor: Dense<f32>,
    },
    Csr {
        name: String,
        shape: Vec<usize>,
        entries: BTreeMap<Vec<usize>, f32>,
        tensor: Option<Csr<u32, f32>>,
    },
}

impl EinsumInput {
    pub(crate) fn new(kind: TensorInputKind, name: String, shape: Vec<usize>) -> Self {
        match kind {
            TensorInputKind::Dense => Self::Dense {
                name,
                tensor: Dense::<f32>::zeros(shape),
            },
            TensorInputKind::Csr => {
                assert!(
                    shape.len() >= 2,
                    "CSR input shape needs rank >= 2, got {shape:?}"
                );
                Self::Csr {
                    name,
                    shape,
                    entries: BTreeMap::new(),
                    tensor: None,
                }
            }
        }
    }

    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Dense { name, .. } | Self::Csr { name, .. } => name,
        }
    }

    pub(crate) fn shape(&self) -> &[usize] {
        match self {
            Self::Dense { tensor, .. } => &tensor.shape,
            Self::Csr { shape, .. } => shape,
        }
    }

    pub(crate) fn set(&mut self, indices: Vec<usize>, value: f32) {
        match self {
            Self::Dense { tensor, .. } => tensor.set(&indices, value),
            Self::Csr {
                shape,
                entries,
                tensor,
                ..
            } => {
                assert_eq!(indices.len(), shape.len(), "CSR cell rank changed");
                entries.insert(indices, value);
                *tensor = None;
            }
        }
    }

    pub(crate) fn prepare(&mut self) {
        if let Self::Csr {
            shape,
            entries,
            tensor,
            ..
        } = self
        {
            if tensor.is_none() {
                *tensor = Some(csr_from_entries(shape, entries));
            }
        }
    }

    pub(crate) fn jit_input(&self) -> JitInput<'_> {
        match self {
            Self::Dense { tensor, .. } => JitInput::Dense(tensor),
            Self::Csr { tensor, .. } => JitInput::Csr(
                tensor
                    .as_ref()
                    .expect("CSR input must be prepared before JIT execution"),
            ),
        }
    }
}

pub(crate) fn validate_attention_shapes(
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
    output_shape: &[usize],
) {
    assert_eq!(
        q_shape.len(),
        4,
        "Q must have shape [batch, heads, query, dim]"
    );
    assert_eq!(
        k_shape.len(),
        4,
        "K must have shape [batch, heads, key, dim]"
    );
    assert_eq!(
        v_shape.len(),
        4,
        "V must have shape [batch, heads, key, value_dim]"
    );
    assert_eq!(
        output_shape.len(),
        4,
        "attention output must have shape [batch, heads, query, value_dim]"
    );
    assert!(
        q_shape
            .iter()
            .chain(k_shape)
            .chain(v_shape)
            .all(|&dim| dim > 0),
        "attention tensors must have non-zero dimensions"
    );

    let expected_k = [q_shape[0], q_shape[1], k_shape[2], q_shape[3]];
    assert_eq!(k_shape, expected_k, "K shape must match Q batch/head/dim");
    let expected_v = [q_shape[0], q_shape[1], k_shape[2], v_shape[3]];
    assert_eq!(v_shape, expected_v, "V shape must match K batch/head/key");
    let expected_output = [q_shape[0], q_shape[1], q_shape[2], v_shape[3]];
    assert_eq!(
        output_shape, expected_output,
        "attention output shape must be [batch, heads, query, value_dim]"
    );
}

pub(crate) fn softmax_attention_rows(scores: &mut Dense<f32>, scale: f32) {
    let key = scores.shape[3];
    for row in scores.data.chunks_exact_mut(key) {
        let mut max = f32::NEG_INFINITY;
        for score in row.iter_mut() {
            *score *= scale;
            max = max.max(*score);
        }

        let mut sum = 0.0;
        for score in row.iter_mut() {
            *score = (*score - max).exp();
            sum += *score;
        }

        let inv_sum = 1.0 / sum;
        for score in row {
            *score *= inv_sum;
        }
    }
}

pub(crate) fn write_dense_output_cells(
    wz: &mut WriteZipperTracked<'_, '_, ()>,
    output: &Dense<f32>,
    output_name: &str,
    output_kind: TensorOutputKind,
) -> bool {
    wz.reset();
    let root_len = wz.root_prefix_path().len();
    let mut changed = wz.val().is_some() || wz.child_mask().count_bits() != 0;
    wz.graft_map(PathMap::new());

    let mut indices = vec![0usize; output.shape.len()];
    let mut encoded = Vec::new();

    for linear in 0..output.data.len() {
        let value = output.data[linear];
        if !output_kind.should_emit(value) {
            continue;
        }
        linear_to_index(linear, &output.shape, &mut indices);
        encoded.clear();
        write_tensor_cell(&mut encoded, output_name, &indices, value);
        wz.move_to_path(&encoded[root_len..]);
        changed |= wz.set_val(()).is_none();
    }

    changed
}

pub(crate) fn parse_expr_group(e: Expr, names: &[&str], label: &str) -> Vec<Expr> {
    let args = expr_args(e);
    assert!(
        args.len() >= 2,
        "{label} group must be shaped like ({} ...)",
        names[0]
    );
    let name = symbol_string(args[0]);
    assert!(
        names.iter().any(|candidate| *candidate == name),
        "{label} group must start with one of {names:?}, got {name:?}"
    );
    args[1..].to_vec()
}

#[derive(Debug, Clone)]
pub(crate) struct TensorOpOutputDecl {
    kind: TensorOutputKind,
    name: String,
    shape: Option<Vec<usize>>,
}

pub(crate) fn parse_single_output_group(e: Expr) -> TensorOpOutputDecl {
    let outputs = parse_expr_group(e, &["output"], "tensor output");
    assert_eq!(
        outputs.len(),
        1,
        "tensor-op-f32 currently supports exactly one output"
    );
    parse_tensor_op_output_decl(outputs[0])
}

fn parse_emit_option(args: &[Expr]) -> TensorOutputKind {
    assert!(
        args.len() >= 2,
        "tensor-op-f32 emit option must be (emit dense), (emit nonzero), or (emit threshold eps)"
    );
    let mode = symbol_string(args[1]);
    match mode.as_str() {
        "dense" => {
            assert_eq!(
                args.len(),
                2,
                "tensor-op-f32 dense emit option does not take an argument"
            );
            TensorOutputKind::Dense
        }
        "nonzero" => {
            assert_eq!(
                args.len(),
                2,
                "tensor-op-f32 nonzero emit option does not take an argument"
            );
            TensorOutputKind::NonZero
        }
        "threshold" => {
            assert_eq!(
                args.len(),
                3,
                "tensor-op-f32 threshold emit option must be (emit threshold eps)"
            );
            let eps = symbol_f32(args[2]);
            assert!(
                eps.is_finite() && eps >= 0.0,
                "tensor-op-f32 threshold must be a finite non-negative f32"
            );
            TensorOutputKind::Threshold(eps)
        }
        _ => panic!("unsupported tensor-op-f32 emit mode {mode:?}"),
    }
}

fn parse_tensor_op_input_decl(e: Expr) -> (TensorInputKind, String, Vec<usize>) {
    let args = expr_args(e);
    assert!(
        args.len() >= 3,
        "tensor-op-f32 input declaration must be shaped like (Name dense dim...) or (Name csr dim...)"
    );
    let name = symbol_string(args[0]);
    let kind = match symbol_string(args[1]).as_str() {
        "dense" => TensorInputKind::Dense,
        "csr" => TensorInputKind::Csr,
        kind => panic!("unsupported tensor-op-f32 input kind {kind:?}"),
    };
    let shape = args[2..].iter().map(|&arg| symbol_usize(arg)).collect();
    (kind, name, shape)
}

fn parse_tensor_op_output_decl(e: Expr) -> TensorOpOutputDecl {
    let args = expr_args(e);
    assert!(
        args.len() >= 2,
        "tensor-op-f32 output declaration must be shaped like (Name dense [dim...]) or (Name nonzero [dim...])"
    );
    let name = symbol_string(args[0]);
    let kind = match symbol_string(args[1]).as_str() {
        "dense" => TensorOutputKind::Dense,
        "nonzero" | "sparse" => TensorOutputKind::NonZero,
        kind => panic!("unsupported tensor-op-f32 output kind {kind:?}"),
    };
    let shape = if args.len() == 2 {
        None
    } else {
        Some(args[2..].iter().map(|&arg| symbol_usize(arg)).collect())
    };
    TensorOpOutputDecl { kind, name, shape }
}

fn validate_backend_clause(args: &[Expr]) {
    assert_eq!(
        args.len(),
        2,
        "tensor-op-f32 backend clause must be (backend auto)"
    );
    let backend = symbol_string(args[1]);
    assert!(
        backend == "auto",
        "tensor-op-f32 backend clause currently only supports (backend auto)"
    );
}

fn parse_operator_clause(e: Expr) -> (String, Vec<Expr>) {
    let args = expr_args(e);
    assert!(
        args.len() >= 2,
        "tensor-op-f32 operator clause must be (op <operator> ...)"
    );
    let head = symbol_string(args[0]);
    assert!(
        head == "op",
        "tensor-op-f32 operator clause must start with op"
    );
    (symbol_string(args[1]), args[2..].to_vec())
}

fn parse_source_cells(e: Expr) -> Vec<Expr> {
    parse_expr_group(e, &["from"], "tensor source cells")
}

fn infer_attention_output_shape(input_decls: &[Expr]) -> Vec<usize> {
    assert_eq!(
        input_decls.len(),
        3,
        "attention tensor-op-f32 needs Q, K, and V input declarations"
    );
    let (_, q_shape) = parse_dense_input_decl(input_decls[0]);
    let (_, k_shape) = parse_dense_input_decl(input_decls[1]);
    let (_, v_shape) = parse_dense_input_decl(input_decls[2]);
    assert_eq!(
        q_shape.len(),
        4,
        "Q must have shape [batch, heads, query, dim]"
    );
    assert_eq!(
        k_shape.len(),
        4,
        "K must have shape [batch, heads, key, dim]"
    );
    assert_eq!(
        v_shape.len(),
        4,
        "V must have shape [batch, heads, key, value_dim]"
    );
    let output_shape = vec![q_shape[0], q_shape[1], q_shape[2], v_shape[3]];
    validate_attention_shapes(&q_shape, &k_shape, &v_shape, &output_shape);
    output_shape
}

fn infer_tensor_op_output_shape(
    op_name: &str,
    op_args: &[Expr],
    input_decls: &[Expr],
    output: &TensorOpOutputDecl,
) -> Vec<usize> {
    let inferred = match op_name {
        "einsum" => {
            assert_eq!(
                op_args.len(),
                1,
                "einsum tensor-op-f32 op clause must be (op einsum spec)"
            );
            let spec = symbol_string(op_args[0]);
            let input_shapes: Vec<Vec<usize>> = input_decls
                .iter()
                .map(|&decl| {
                    let (_, _, shape) = parse_tensor_op_input_decl(decl);
                    shape
                })
                .collect();
            let input_shape_refs: Vec<&[usize]> = input_shapes.iter().map(Vec::as_slice).collect();
            let mut output_shapes = infer_output_shapes(&spec, &input_shape_refs)
                .unwrap_or_else(|err| panic!("tensor-op-f32 einsum shape inference failed: {err}"));
            assert_eq!(
                output_shapes.len(),
                1,
                "tensor-op-f32 currently supports exactly one inferred output"
            );
            output_shapes.remove(0)
        }
        "attention" => {
            assert!(
                op_args.is_empty()
                    || (op_args.len() == 1
                        && matches!(
                            symbol_string(op_args[0]).as_str(),
                            "scaled-dot" | "scaled-dot-product"
                        )),
                "attention tensor-op-f32 op clause must be (op attention) or (op attention scaled-dot)"
            );
            infer_attention_output_shape(input_decls)
        }
        _ => panic!("unsupported tensor-op-f32 operator {op_name:?}"),
    };

    if let Some(explicit) = &output.shape {
        assert_eq!(
            explicit, &inferred,
            "tensor-op-f32 explicit output shape does not match inferred operator shape"
        );
    }
    output.shape.clone().unwrap_or(inferred)
}

/// Parsed surface form of a `tensor-op-f32` expression.
///
/// This keeps the MORK syntax boundary explicit: sink code does not need to
/// know the grammar once an expression has been accepted here.
pub(crate) struct TensorOpF32Syntax {
    op_name: String,
    op_args: Vec<Expr>,
    input_decls: Vec<Expr>,
    pub(crate) output_kind: TensorOutputKind,
    pub(crate) output_name: String,
    pub(crate) output_shape: Vec<usize>,
    cell_templates: Vec<Expr>,
}

impl TensorOpF32Syntax {
    pub(crate) fn parse(e: Expr) -> Self {
        let args = expr_args(e);
        assert!(
            args.len() >= 5,
            "tensor-op-f32 shape is (tensor-op-f32 (op ...) (inputs ...) (output ...) (from ...) [(emit ...)] [(backend auto)])"
        );

        let mut op = None;
        let mut input_decls = None;
        let mut output = None;
        let mut cell_templates = None;
        let mut emit = None;

        for &clause in &args[1..] {
            let clause_args = expr_args(clause);
            assert!(
                !clause_args.is_empty(),
                "tensor-op-f32 clause cannot be empty"
            );
            let clause_name = symbol_string(clause_args[0]);
            match clause_name.as_str() {
                "op" => {
                    assert!(op.is_none(), "tensor-op-f32 cannot repeat op clause");
                    op = Some(parse_operator_clause(clause));
                }
                "inputs" => {
                    assert!(
                        input_decls.is_none(),
                        "tensor-op-f32 cannot repeat inputs clause"
                    );
                    input_decls = Some(clause_args[1..].to_vec());
                }
                "output" => {
                    assert!(
                        output.is_none(),
                        "tensor-op-f32 cannot repeat output clause"
                    );
                    output = Some(parse_single_output_group(clause));
                }
                "from" => {
                    assert!(
                        cell_templates.is_none(),
                        "tensor-op-f32 cannot repeat from clause"
                    );
                    cell_templates = Some(parse_source_cells(clause));
                }
                "emit" => {
                    assert!(emit.is_none(), "tensor-op-f32 cannot repeat emit clause");
                    emit = Some(parse_emit_option(&clause_args));
                }
                "backend" => validate_backend_clause(&clause_args),
                _ => panic!("unsupported tensor-op-f32 clause {clause_name:?}"),
            }
        }

        let (op_name, op_args) = op.expect("tensor-op-f32 requires an (op ...) clause");
        let input_decls = input_decls.expect("tensor-op-f32 requires an (inputs ...) clause");
        let output = output.expect("tensor-op-f32 requires an (output ...) clause");
        let cell_templates = cell_templates.expect("tensor-op-f32 requires a (from ...) clause");
        let output_shape = infer_tensor_op_output_shape(&op_name, &op_args, &input_decls, &output);
        let output_kind = emit.unwrap_or(output.kind);

        Self {
            op_name,
            op_args,
            input_decls,
            output_kind,
            output_name: output.name,
            output_shape,
            cell_templates,
        }
    }

    pub(crate) fn output_prefix(&self) -> &'static [u8] {
        tensor_cell_prefix(&self.output_name, self.output_shape.len())
    }

    pub(crate) fn matched_cells(e: Expr) -> Vec<Expr> {
        let args = expr_args(e);
        for &clause in &args[1..] {
            let clause_args = expr_args(clause);
            if clause_args.is_empty() {
                continue;
            }
            let clause_name = symbol_string(clause_args[0]);
            if clause_name == "from" {
                return parse_source_cells(clause);
            }
        }
        panic!("tensor-op-f32 requires a (from ...) clause")
    }
}

fn parse_dense_input_decl(e: Expr) -> (String, Vec<usize>) {
    let (kind, name, shape) = parse_tensor_op_input_decl(e);
    assert_eq!(
        kind,
        TensorInputKind::Dense,
        "this tensor-op-f32 operator only accepts dense inputs"
    );
    (name, shape)
}

enum TensorOpF32Kernel {
    Einsum {
        spec: String,
        plan: Option<EinsumF32Plan>,
        inputs: Vec<EinsumInput>,
    },
    AttentionScaledDot(Box<AttentionScaledDotKernel>),
}

struct AttentionScaledDotKernel {
    score_plan: Option<EinsumF32Plan>,
    value_plan: Option<EinsumF32Plan>,
    score_shape: Vec<usize>,
    q_name: String,
    k_name: String,
    v_name: String,
    q: Dense<f32>,
    k: Dense<f32>,
    v: Dense<f32>,
    scale: f32,
}

/// Parsed and verified executable form of a `tensor-op-f32` declaration.
///
/// MORK syntax stays fact-like and pathmap-friendly, while this enum carries
/// the typed storage and compiled backend plans needed for repeated execution.
pub(crate) struct TensorOpF32Plan {
    kernel: TensorOpF32Kernel,
}

impl TensorOpF32Plan {
    pub(crate) fn from_syntax(syntax: &TensorOpF32Syntax) -> Self {
        let kernel = match syntax.op_name.as_str() {
            "einsum" => {
                assert_eq!(
                    syntax.op_args.len(),
                    1,
                    "einsum tensor-op-f32 op clause must be (op einsum spec)"
                );
                assert_eq!(
                    syntax.input_decls.len(),
                    syntax.cell_templates.len(),
                    "einsum tensor-op-f32 needs one cell template per input"
                );
                let spec = symbol_string(syntax.op_args[0]);
                let mut inputs = Vec::with_capacity(syntax.input_decls.len());
                for (decl, template) in syntax.input_decls.iter().zip(&syntax.cell_templates) {
                    let (kind, name, shape) = parse_tensor_op_input_decl(*decl);
                    let input = EinsumInput::new(kind, name, shape);
                    validate_tensor_cell_template(*template, input.name(), input.shape().len());
                    inputs.push(input);
                }
                TensorOpF32Kernel::Einsum {
                    spec,
                    plan: None,
                    inputs,
                }
            }
            "attention" => {
                assert!(
                    syntax.op_args.is_empty()
                        || (syntax.op_args.len() == 1
                            && matches!(
                                symbol_string(syntax.op_args[0]).as_str(),
                                "scaled-dot" | "scaled-dot-product"
                            )),
                    "attention tensor-op-f32 op clause must be (op attention) or (op attention scaled-dot)"
                );
                assert_eq!(
                    syntax.input_decls.len(),
                    3,
                    "attention tensor-op-f32 needs Q, K, and V input declarations"
                );
                assert_eq!(
                    syntax.cell_templates.len(),
                    3,
                    "attention tensor-op-f32 needs Q, K, and V cell templates"
                );

                let (q_name, q_shape) = parse_dense_input_decl(syntax.input_decls[0]);
                let (k_name, k_shape) = parse_dense_input_decl(syntax.input_decls[1]);
                let (v_name, v_shape) = parse_dense_input_decl(syntax.input_decls[2]);
                validate_attention_shapes(&q_shape, &k_shape, &v_shape, &syntax.output_shape);
                validate_tensor_cell_template(syntax.cell_templates[0], &q_name, q_shape.len());
                validate_tensor_cell_template(syntax.cell_templates[1], &k_name, k_shape.len());
                validate_tensor_cell_template(syntax.cell_templates[2], &v_name, v_shape.len());

                let q = Dense::<f32>::zeros(q_shape);
                let k = Dense::<f32>::zeros(k_shape);
                let v = Dense::<f32>::zeros(v_shape);
                let score_shape = vec![q.shape[0], q.shape[1], q.shape[2], k.shape[2]];
                let scale = 1.0 / (q.shape[3] as f32).sqrt();
                TensorOpF32Kernel::AttentionScaledDot(Box::new(AttentionScaledDotKernel {
                    score_plan: None,
                    value_plan: None,
                    score_shape,
                    q_name,
                    k_name,
                    v_name,
                    q,
                    k,
                    v,
                    scale,
                }))
            }
            _ => panic!("unsupported tensor-op-f32 operator {:?}", syntax.op_name),
        };

        Self { kernel }
    }

    pub(crate) fn sink_cells(&mut self, cells: &[Expr]) {
        match &mut self.kernel {
            TensorOpF32Kernel::Einsum { inputs, .. } => {
                assert_eq!(
                    cells.len(),
                    inputs.len(),
                    "einsum tensor-op-f32 cell count changed"
                );
                for (input, &cell) in inputs.iter_mut().zip(cells) {
                    let (name, indices, value) = parse_cell(cell);
                    assert_eq!(name, input.name(), "input cell name changed");
                    input.set(indices, value);
                }
            }
            TensorOpF32Kernel::AttentionScaledDot(kernel) => {
                let kernel = kernel.as_mut();
                assert_eq!(cells.len(), 3, "attention tensor-op-f32 cell count changed");
                let (parsed_q_name, q_indices, q_value) = parse_cell(cells[0]);
                let (parsed_k_name, k_indices, k_value) = parse_cell(cells[1]);
                let (parsed_v_name, v_indices, v_value) = parse_cell(cells[2]);
                assert_eq!(parsed_q_name, kernel.q_name, "Q cell name changed");
                assert_eq!(parsed_k_name, kernel.k_name, "K cell name changed");
                assert_eq!(parsed_v_name, kernel.v_name, "V cell name changed");
                kernel.q.set(&q_indices, q_value);
                kernel.k.set(&k_indices, k_value);
                kernel.v.set(&v_indices, v_value);
            }
        }
    }

    pub(crate) fn run(&mut self, output_shape: &Vec<usize>) -> Dense<f32> {
        match &mut self.kernel {
            TensorOpF32Kernel::Einsum { spec, plan, inputs } => {
                let mut output = Dense::<f32>::zeros(output_shape.clone());
                for input in inputs.iter_mut() {
                    input.prepare();
                }
                let jit_inputs: Vec<JitInput<'_>> =
                    inputs.iter().map(EinsumInput::jit_input).collect();
                if plan.is_none() {
                    *plan = Some(
                        EinsumF32Plan::compile(
                            spec,
                            &jit_inputs,
                            std::slice::from_ref(output_shape),
                        )
                        .unwrap_or_else(|err| panic!("tensor-op-f32 einsum compile failed: {err}")),
                    );
                }
                {
                    let mut outputs = [&mut output];
                    let plan = plan.as_ref().unwrap();
                    plan.try_run(&jit_inputs, &mut outputs)
                        .unwrap_or_else(|err| panic!("tensor-op-f32 einsum failed: {err}"));
                    trace!(target: "sink", "tensor-op-f32 einsum backend {:?}", plan.backend());
                }
                output
            }
            TensorOpF32Kernel::AttentionScaledDot(kernel) => {
                let kernel = kernel.as_mut();
                let mut scores = Dense::<f32>::zeros(kernel.score_shape.clone());
                if kernel.score_plan.is_none() {
                    kernel.score_plan = Some(
                        EinsumF32Plan::compile(
                            "bhqd,bhkd->bhqk",
                            &[JitInput::Dense(&kernel.q), JitInput::Dense(&kernel.k)],
                            std::slice::from_ref(&kernel.score_shape),
                        )
                        .unwrap_or_else(|err| {
                            panic!("tensor-op-f32 attention score compile failed: {err}")
                        }),
                    );
                }
                {
                    let mut outputs = [&mut scores];
                    let score_plan = kernel.score_plan.as_ref().unwrap();
                    score_plan
                        .try_run(
                            &[JitInput::Dense(&kernel.q), JitInput::Dense(&kernel.k)],
                            &mut outputs,
                        )
                        .unwrap_or_else(|err| {
                            panic!("tensor-op-f32 attention score pass failed: {err}")
                        });
                    trace!(
                        target: "sink",
                        "tensor-op-f32 attention score backend {:?}",
                        score_plan.backend()
                    );
                }

                softmax_attention_rows(&mut scores, kernel.scale);

                let mut output = Dense::<f32>::zeros(output_shape.clone());
                if kernel.value_plan.is_none() {
                    kernel.value_plan = Some(
                        EinsumF32Plan::compile(
                            "bhqk,bhkd->bhqd",
                            &[JitInput::Dense(&scores), JitInput::Dense(&kernel.v)],
                            std::slice::from_ref(output_shape),
                        )
                        .unwrap_or_else(|err| {
                            panic!("tensor-op-f32 attention value compile failed: {err}")
                        }),
                    );
                }
                {
                    let mut outputs = [&mut output];
                    let value_plan = kernel.value_plan.as_ref().unwrap();
                    value_plan
                        .try_run(
                            &[JitInput::Dense(&scores), JitInput::Dense(&kernel.v)],
                            &mut outputs,
                        )
                        .unwrap_or_else(|err| {
                            panic!("tensor-op-f32 attention value pass failed: {err}")
                        });
                    trace!(
                        target: "sink",
                        "tensor-op-f32 attention value backend {:?}",
                        value_plan.backend()
                    );
                }
                output
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TensorOpF32Kernel;

    #[test]
    fn tensor_op_kernel_keeps_attention_payload_boxed() {
        assert!(
            std::mem::size_of::<TensorOpF32Kernel>() <= 128,
            "large attention payload should stay behind indirection"
        );
    }
}
