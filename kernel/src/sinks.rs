use crate::pure;
use crate::space::ACT_PATH;
#[cfg(feature = "einsum")]
use crate::tensor_ops::{
    EinsumInput, TensorOpF32Plan, TensorOpF32Syntax, TensorOutputKind, expr_args, parse_cell,
    parse_input_tensor_decl, parse_output_tensor_decl, parse_tensor_decl, softmax_attention_rows,
    symbol_string, tensor_cell_prefix, validate_attention_shapes, validate_tensor_cell_template,
    write_dense_output_cells,
};
use core::f64;
use eval::EvalScope;
use eval_ffi::ExprSource;
use log::*;
use mork_expr::{Expr, ExprEnv, ExprZipper, Tag, byte_item, destruct, item_byte, serialize};
use pathmap::PathMap;
use pathmap::morphisms::Catamorphism;
use pathmap::ring::AlgebraicStatus;
use pathmap::utils::{BitMask, ByteMask};
use pathmap::zipper::*;
#[cfg(test)]
use std::cmp::Ordering;
#[cfg(feature = "z3")]
use std::io::Write;
use std::marker::PhantomData;
use std::ops::{AddAssign, MulAssign};
use std::ptr::slice_from_raw_parts;
#[cfg(feature = "wasm")]
use std::sync::LazyLock;

#[cfg(feature = "einsum")]
use linalg::dense::Dense;
#[cfg(feature = "einsum")]
use linalg::jit::{EinsumF32Plan, JitInput};
#[cfg(feature = "einsum")]
use linalg::tensor::NDIndex;

/// Default Wasmtime linear-memory reservation for MORK WASM sinks.
pub const WASM_LINEAR_MEMORY_RESERVATION_BYTES: u64 = 1 << 32;
/// Default end-guard size for MORK WASM sink linear memories.
pub const WASM_LINEAR_MEMORY_GUARD_BYTES: u64 = 32 * 1024 * 1024;
/// Environment override for the WASM sink linear-memory reservation, in bytes.
pub const WASM_LINEAR_MEMORY_RESERVATION_ENV: &str = "MORK_WASM_LINEAR_MEMORY_RESERVATION_BYTES";
/// Environment override for the WASM sink linear-memory end guard, in bytes.
pub const WASM_LINEAR_MEMORY_GUARD_ENV: &str = "MORK_WASM_LINEAR_MEMORY_GUARD_BYTES";

/// Runtime memory policy used when MORK is built with the `wasm` sink feature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WasmLinearMemoryPolicy {
    /// Bytes reserved for each linear memory.
    pub reservation_bytes: u64,
    /// Bytes reserved as the end guard after each linear memory.
    pub guard_bytes: u64,
    /// Whether Wasmtime's multi-memory proposal support is enabled.
    pub multi_memory_enabled: bool,
    /// Whether Wasmtime can use signal traps for memory faults.
    pub signals_based_traps_enabled: bool,
}

/// Returns the default linear-memory policy used by MORK WASM sinks.
pub const fn wasm_linear_memory_policy() -> WasmLinearMemoryPolicy {
    WasmLinearMemoryPolicy {
        reservation_bytes: WASM_LINEAR_MEMORY_RESERVATION_BYTES,
        guard_bytes: WASM_LINEAR_MEMORY_GUARD_BYTES,
        multi_memory_enabled: true,
        signals_based_traps_enabled: true,
    }
}

impl Default for WasmLinearMemoryPolicy {
    fn default() -> Self {
        wasm_linear_memory_policy()
    }
}

/// Error returned when a WASM sink memory-policy override is not a valid byte count.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WasmLinearMemoryPolicyError {
    /// Name of the environment variable or configuration field that failed.
    pub variable: &'static str,
    /// Rejected value.
    pub value: String,
}

impl std::fmt::Display for WasmLinearMemoryPolicyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} must be an unsigned byte count, got {:?}",
            self.variable, self.value
        )
    }
}

impl std::error::Error for WasmLinearMemoryPolicyError {}

fn parse_wasm_linear_memory_bytes(
    variable: &'static str,
    value: Option<&str>,
    default: u64,
) -> Result<u64, WasmLinearMemoryPolicyError> {
    match value {
        Some(raw) => raw.parse::<u64>().map_err(|_| WasmLinearMemoryPolicyError {
            variable,
            value: raw.to_owned(),
        }),
        None => Ok(default),
    }
}

fn wasm_linear_memory_env_value(
    variable: &'static str,
) -> Result<Option<String>, WasmLinearMemoryPolicyError> {
    match std::env::var(variable) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(value)) => Err(WasmLinearMemoryPolicyError {
            variable,
            value: value.to_string_lossy().into_owned(),
        }),
    }
}

/// Builds a WASM sink linear-memory policy from explicit optional byte strings.
pub fn wasm_linear_memory_policy_from_values(
    reservation_bytes: Option<&str>,
    guard_bytes: Option<&str>,
) -> Result<WasmLinearMemoryPolicy, WasmLinearMemoryPolicyError> {
    let default = wasm_linear_memory_policy();
    Ok(WasmLinearMemoryPolicy {
        reservation_bytes: parse_wasm_linear_memory_bytes(
            WASM_LINEAR_MEMORY_RESERVATION_ENV,
            reservation_bytes,
            default.reservation_bytes,
        )?,
        guard_bytes: parse_wasm_linear_memory_bytes(
            WASM_LINEAR_MEMORY_GUARD_ENV,
            guard_bytes,
            default.guard_bytes,
        )?,
        ..default
    })
}

/// Builds a WASM sink linear-memory policy from MORK environment overrides.
pub fn wasm_linear_memory_policy_from_env()
-> Result<WasmLinearMemoryPolicy, WasmLinearMemoryPolicyError> {
    let reservation_bytes = wasm_linear_memory_env_value(WASM_LINEAR_MEMORY_RESERVATION_ENV)?;
    let guard_bytes = wasm_linear_memory_env_value(WASM_LINEAR_MEMORY_GUARD_ENV)?;
    wasm_linear_memory_policy_from_values(reservation_bytes.as_deref(), guard_bytes.as_deref())
}

#[derive(Eq, PartialEq, Debug)]
pub(crate) enum WriteResourceRequest {
    BTM(&'static [u8]),
    ACT(&'static str),
    #[cfg(feature = "z3")]
    Z3(&'static str),
}

#[cfg(test)]
impl WriteResourceRequest {
    pub(crate) fn pjoin(&self, other: &Self) -> Option<Self> {
        match self {
            WriteResourceRequest::BTM(s) => match other {
                WriteResourceRequest::BTM(o) => {
                    Some(WriteResourceRequest::BTM(btm_common_prefix(s, o)))
                }
                _ => None,
            },
            WriteResourceRequest::ACT(s) => match other {
                WriteResourceRequest::ACT(o) if s == o => Some(WriteResourceRequest::ACT(s)),
                _ => None,
            },
            #[cfg(feature = "z3")]
            WriteResourceRequest::Z3(s) => match other {
                WriteResourceRequest::Z3(o) if s == o => Some(WriteResourceRequest::Z3(s)),
                _ => None,
            },
        }
    }
}

#[cfg(test)]
fn btm_common_prefix<'a>(left: &'a [u8], right: &[u8]) -> &'a [u8] {
    &left[..btm_common_prefix_len(left, right)]
}

#[cfg(test)]
fn btm_common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    pathmap::utils::find_prefix_overlap(left, right)
}

const SINK_EXPR_SCRATCH_INITIAL_CAPACITY: usize = 4096;

fn sink_expr_scratch_buffer() -> Vec<u8> {
    Vec::with_capacity(SINK_EXPR_SCRATCH_INITIAL_CAPACITY)
}

fn substitute_one_de_bruijn_into_buffer(
    expr: &[u8],
    idx: u8,
    substitution: &mut [u8],
    output: &mut Vec<u8>,
) -> usize {
    let expr = Expr {
        ptr: expr.as_ptr().cast_mut(),
    };
    let substitution = Expr {
        ptr: substitution.as_mut_ptr(),
    };
    let len = expr.substitute_one_de_bruijn_len(idx, substitution);

    output.clear();
    output.reserve(len);
    let mut oz = ExprZipper::new(Expr {
        ptr: output.as_mut_ptr(),
    });
    expr.substitute_one_de_bruijn(idx, substitution, &mut oz);
    debug_assert_eq!(oz.loc, len);
    // SAFETY: `reserve(len)` ensured capacity for `len` bytes, and
    // `substitute_one_de_bruijn` initialized exactly `oz.loc` bytes.
    unsafe { output.set_len(oz.loc) };
    oz.loc
}

#[cfg(test)]
impl PartialOrd for WriteResourceRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self {
            WriteResourceRequest::BTM(s) => {
                if let WriteResourceRequest::BTM(o) = other {
                    s.partial_cmp(o)
                } else {
                    None
                }
            }
            WriteResourceRequest::ACT(s) => {
                if let WriteResourceRequest::ACT(o) = other {
                    if s == o { Some(Ordering::Equal) } else { None }
                } else {
                    None
                }
            }
            #[cfg(feature = "z3")]
            WriteResourceRequest::Z3(s) => {
                if let WriteResourceRequest::Z3(o) = other {
                    if s == o { Some(Ordering::Equal) } else { None }
                } else {
                    None
                }
            }
        }
    }
}

pub(crate) enum WriteResource<'w, 'a, 'k> {
    BTM(&'w mut WriteZipperTracked<'a, 'k, ()>),
    ACT(()),
    #[cfg(feature = "z3")]
    Z3(&'w mut subprocess::Popen),
}

// trait JoinLattice  {
//     fn join(x: Self, y: Self) -> Self;
// }
//
// impl JoinLattice for WriteResourceRequest {
//     fn join(x: Self, y: Self) -> Self {
//         match (x, y) {
//             (WriteResourceRequest::BTM(x), WriteResourceRequest::BTM(y)) => {
//                 let i = pathmap::utils::find_prefix_overlap(x, y);
//                 &x[..i] // equiv &y[..i]
//             }
//         }
//     }
// }
//
// impl std::cmp::PartialEq for JoinLattice {
//     fn eq(&self, other: &Self) -> bool {
//         Self::is_bottom(self.meet(other))
//     }
//
// }
//
// impl std::cmp::PartialOrd for JoinLattice {
//     fn lteq(x: Self, y: Self) -> bool {
//         x.join(y).eq(y)
//     }
// }

#[cfg(test)]
mod tests {
    use super::WriteResourceRequest::{ACT, BTM};
    use super::{
        ASink, SINK_EXPR_SCRATCH_INITIAL_CAPACITY, WASM_LINEAR_MEMORY_GUARD_ENV,
        WASM_LINEAR_MEMORY_RESERVATION_ENV, btm_common_prefix, btm_common_prefix_len,
        sink_expr_scratch_buffer, substitute_one_de_bruijn_into_buffer, wasm_linear_memory_policy,
        wasm_linear_memory_policy_from_values,
    };
    use mork_expr::{Expr, parse};

    #[test]
    fn btm_common_prefix_len_stops_before_first_mismatch() {
        assert_eq!(btm_common_prefix_len(b"abcdef", b"abcxyz"), 3);
        assert_eq!(btm_common_prefix_len(b"abc", b"abcxyz"), 3);
        assert_eq!(btm_common_prefix_len(b"abc", b"xyz"), 0);
    }

    #[test]
    fn btm_common_prefix_returns_left_borrowed_prefix() {
        assert_eq!(
            btm_common_prefix(b"out/group-a/item", b"out/group-b/item"),
            b"out/group-"
        );
        assert_eq!(
            btm_common_prefix(b"out/group", b"out/group/item"),
            b"out/group"
        );
    }

    #[test]
    fn aggregate_sink_keeps_large_attention_sink_boxed() {
        assert!(
            std::mem::size_of::<ASink>() <= 256,
            "large optional sink payloads should stay behind indirection"
        );
    }

    #[test]
    fn btm_pjoin_returns_longest_common_prefix_request() {
        assert_eq!(BTM(b"abc").pjoin(&BTM(b"abd")), Some(BTM(b"ab")));
        assert_eq!(BTM(b"abc").pjoin(&BTM(b"abc/child")), Some(BTM(b"abc")));
        assert_eq!(BTM(b"abc").pjoin(&BTM(b"xyz")), Some(BTM(b"")));
    }

    #[test]
    fn pjoin_preserves_non_btm_resource_boundaries() {
        assert_eq!(ACT("arena").pjoin(&ACT("arena")), Some(ACT("arena")));
        assert_eq!(ACT("arena").pjoin(&ACT("other")), None);
        assert_eq!(ACT("arena").pjoin(&BTM(b"arena")), None);
        assert_eq!(BTM(b"arena").pjoin(&ACT("arena")), None);
    }

    #[test]
    fn sink_expr_scratch_buffer_starts_bounded_and_empty() {
        let buffer = sink_expr_scratch_buffer();

        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), SINK_EXPR_SCRATCH_INITIAL_CAPACITY);
    }

    #[test]
    fn wasm_linear_memory_policy_values_default_when_unset() {
        assert_eq!(
            wasm_linear_memory_policy_from_values(None, None).unwrap(),
            wasm_linear_memory_policy()
        );
    }

    #[test]
    fn wasm_linear_memory_policy_values_override_byte_counts() {
        let policy =
            wasm_linear_memory_policy_from_values(Some("67108864"), Some("1048576")).unwrap();

        assert_eq!(policy.reservation_bytes, 64 * 1024 * 1024);
        assert_eq!(policy.guard_bytes, 1024 * 1024);
        assert!(policy.multi_memory_enabled);
        assert!(policy.signals_based_traps_enabled);
    }

    #[test]
    fn wasm_linear_memory_policy_values_reject_invalid_byte_counts() {
        let error = wasm_linear_memory_policy_from_values(Some("64MiB"), None).unwrap_err();

        assert_eq!(error.variable, WASM_LINEAR_MEMORY_RESERVATION_ENV);
        assert_eq!(error.value, "64MiB");

        let error = wasm_linear_memory_policy_from_values(None, Some("-1")).unwrap_err();

        assert_eq!(error.variable, WASM_LINEAR_MEMORY_GUARD_ENV);
        assert_eq!(error.value, "-1");
    }

    #[test]
    fn substitute_one_de_bruijn_into_buffer_writes_exact_len() {
        let expr = parse!("[3] result $ _1");
        let mut replacement = parse!("[2] count 123");
        let mut output = Vec::with_capacity(1);

        let len = substitute_one_de_bruijn_into_buffer(&expr, 0, &mut replacement, &mut output);

        assert_eq!(len, output.len());
        assert!(output.capacity() >= len);
        assert_eq!(
            format!(
                "{:?}",
                Expr {
                    ptr: output.as_mut_ptr()
                }
            ),
            "(result (count 123) (count 123))"
        );
    }
}

pub(crate) trait Sink {
    fn new(e: Expr) -> Self;
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest>;
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w;
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w;
}

pub struct CompatSink {
    e: Expr,
    changed: bool,
}

impl Sink for CompatSink {
    fn new(e: Expr) -> Self {
        CompatSink { e, changed: false }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| self.e.span())
                .as_ref()
                .unwrap()
        }[..];
        trace!(target: "sink", "+ (compat) requesting {}", serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        let mpath = &path[wz.root_prefix_path().len()..];
        trace!(target: "sink", "+ (compat) at '{}' sinking raw '{}'", serialize(wz.root_prefix_path()), serialize(path));
        trace!(target: "sink", "+ (compat) sinking '{}'", serialize(mpath));
        wz.move_to_path(mpath);
        self.changed |= wz.set_val(()).is_none();
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        _it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        trace!(target: "sink", "+ (compat) finalizing");
        self.changed
    }
}

pub struct AddSink {
    e: Expr,
    pending: PathMap<()>,
}
impl Sink for AddSink {
    fn new(e: Expr) -> Self {
        AddSink {
            e,
            pending: PathMap::new(),
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| self.e.span())
                .as_ref()
                .unwrap()
        }[3..];
        trace!(target: "sink", "+ requesting {}", serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        let mpath = &path[3 + wz.root_prefix_path().len()..];
        trace!(target: "sink", "+ at '{}' sinking raw '{}'", serialize(wz.root_prefix_path()), serialize(path));
        trace!(target: "sink", "+ sinking '{}'", serialize(mpath));
        self.pending.insert(mpath, ());
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        wz.reset();
        trace!(target: "sink", "+ finalizing by joining {} at '{}'", self.pending.val_count(), serialize(wz.origin_path()));

        let mut changed = self
            .pending
            .remove(&[])
            .is_some_and(|()| wz.set_val(()).is_none());

        changed |= match wz.join_into(&self.pending.read_zipper()) {
            AlgebraicStatus::Element => true,
            AlgebraicStatus::Identity => false,
            AlgebraicStatus::None => true,
        };
        changed
    }
}

// (U <expr>)
pub struct USink {
    e: Expr,
    buf: Option<Vec<u8>>,
    tmp: Option<Vec<u8>>,
    conflict: bool,
    tmp_expr_env: Vec<(ExprEnv, ExprEnv)>,
    tmp_stack: Vec<(u8, u8)>,
    tmp_assignments: Vec<(u8, u8)>,
    last_len: usize,
}
impl Sink for USink {
    fn new(e: Expr) -> Self {
        USink {
            e,
            buf: None,
            tmp: None,
            conflict: false,
            tmp_expr_env: Vec::new(),
            tmp_stack: Vec::new(),
            tmp_assignments: Vec::new(),
            last_len: usize::MAX,
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| self.e.span())
                .as_ref()
                .unwrap()
        }[3..];
        trace!(target: "sink", "U requesting {}", serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        _it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        // we could be way more parsimonious not unifying the prefix over and over again
        // let mpath = &path[3+wz.root_prefix_path().len()..];
        trace!(target: "sink", "U new expr '{}'", serialize(&path[3..]));
        if self.conflict {
            return;
        }
        if let Some(mut buf) = self.buf.take() {
            let tmp = self.tmp.get_or_insert_with(sink_expr_scratch_buffer);
            tmp.clear();
            let eau = Expr {
                ptr: buf.as_mut_ptr(),
            };

            if !mork_expr::unifies_reuse_state(
                eau,
                Expr {
                    ptr: path[3..].as_ptr().cast_mut(),
                },
                &mut *tmp,
                &mut self.tmp_expr_env,
                &mut self.tmp_stack,
                &mut self.tmp_assignments,
            ) {
                self.buf = Some(buf);
                self.conflict = true;
                return;
            }

            self.last_len = tmp.len();
            self.buf = self.tmp.take();
            self.tmp = Some(buf);
        } else {
            self.buf = Some(path[3..].to_vec());
            self.tmp = Some(sink_expr_scratch_buffer());
            self.last_len = path[3..].len();
        }
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        trace!(target: "sink", "U finalizing");
        if self.conflict {
            trace!(target: "sink", "U conflict");
            return false;
        }
        match self.buf.take() {
            None => {
                trace!(target: "sink", "U empty");
                false
            }
            Some(buf) => {
                let buf_slice = &buf[..self.last_len];
                trace!(target: "sink", "U unified expression '{}'", serialize(buf_slice));
                let WriteResource::BTM(wz) = it.next().unwrap() else {
                    unreachable!()
                };
                wz.move_to_path(&buf_slice[wz.root_prefix_path().len()..]);
                wz.set_val(());
                true
            }
        }
    }
}

// (AU <expr>)
pub struct AUSink {
    e: Expr,
    buf: Option<Box<[u8]>>,
    tmp: Option<Box<[u8]>>,
    last: usize,
}
impl Sink for AUSink {
    fn new(e: Expr) -> Self {
        AUSink {
            e,
            buf: None,
            tmp: None,
            last: usize::MAX,
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| self.e.span())
                .as_ref()
                .unwrap()
        }[4..];
        trace!(target: "sink", "AU requesting {}", serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        _it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        // we could be way more parsimonious not anti-unifying the prefix over and over again
        // let mpath = &path[4+wz.root_prefix_path().len()..];
        trace!(target: "sink", "AU new expr '{}'", serialize(&path[4..]));
        if let Some(e) = self.buf.as_mut() {
            let tmp = self.tmp.as_mut().unwrap();
            let eau = Expr {
                ptr: (*e).as_mut_ptr(),
            };
            let mut wz = ExprZipper::new(Expr {
                ptr: (*tmp).as_mut_ptr(),
            });
            eau.anti_unify(
                Expr {
                    ptr: path[4..].as_ptr().cast_mut(),
                },
                &mut wz,
            )
            .unwrap();
            std::mem::swap(&mut self.buf, &mut self.tmp);
            self.last = wz.loc;
        } else {
            self.buf = Some(path[4..].to_vec().into_boxed_slice());
            self.tmp = self.buf.clone();
        }
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        trace!(target: "sink", "AU finalizing");
        match self.buf.take() {
            None => {
                trace!(target: "sink", "AU empty");
                false
            }
            Some(buf) => {
                trace!(target: "sink", "AU anti-unified expression '{}'", serialize(&buf[..self.last]));
                let WriteResource::BTM(wz) = it.next().unwrap() else {
                    unreachable!()
                };
                wz.move_to_path(&buf[wz.root_prefix_path().len()..self.last]);
                wz.set_val(());
                true
            }
        }
    }
}

pub struct ACTSink {
    file: &'static str,
    tmp: PathMap<()>,
}
impl Sink for ACTSink {
    fn new(e: Expr) -> Self {
        destruct!(e, ("ACT" {act: &str} se), {
            return ACTSink { file: act, tmp: PathMap::new() }
        }, _err => { panic!("act not the right shape") });
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        trace!(target: "sink", "ACT requesting {}", self.file);
        std::iter::once(WriteResourceRequest::ACT(self.file))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        _it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        trace!(target: "sink", "ACT sinking '{}'", serialize(&path[1+1+3+1+self.file.len()..]));
        self.tmp
            .insert(&path[1 + 1 + 3 + 1 + self.file.len()..], ());
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        trace!(target: "sink", "ACT finalizing");
        let _resource = it.next().unwrap();
        if let Err(err) = pathmap::arena_compact::ArenaCompactTree::dump_from_zipper(
            self.tmp.read_zipper(),
            |_v| 0,
            format!("{}{}.act", ACT_PATH, self.file),
        )
        .map(|_tree| ())
        {
            error!(target: "sink", "ACT failed to dump '{}': {err}", self.file);
            return false;
        }
        true
    }
}

pub struct RemoveSink {
    e: Expr,
    remove: PathMap<()>,
}
// perhaps more performant to graft, remove*, and graft back?
impl Sink for RemoveSink {
    fn new(e: Expr) -> Self {
        RemoveSink {
            e,
            remove: PathMap::new(),
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        // !! we're never grabbing the full expression path, because then we don't have the ability to remove the root value
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| {
                    let s = self.e.span();
                    slice_from_raw_parts(self.e.ptr, s.len() - 1)
                })
                .as_ref()
                .unwrap()
        }[3..];
        trace!(target: "sink", "- requesting {}", serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        let mpath = &path[3 + wz.root_prefix_path().len()..];
        trace!(target: "sink", "- at '{}' sinking raw '{}'", serialize(wz.root_prefix_path()), serialize(path));
        trace!(target: "sink", "- sinking '{}'", serialize(mpath));
        self.remove.insert(mpath, ());
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        wz.reset();
        trace!(target: "sink", "- finalizing by subtracting {} at '{}'", self.remove.val_count(), serialize(wz.origin_path()));
        // match self.remove.remove(&[]) {
        //     None => {}
        //     Some(s) => {
        //         println!("has root");
        //         wz.remove_val(true);
        //         println!("val not removed");
        //     }
        // }
        match wz.subtract_into(&self.remove.read_zipper(), true) {
            AlgebraicStatus::Element => true,
            AlgebraicStatus::Identity => false,
            AlgebraicStatus::None => true, // GOAT maybe not?
        }
    }
}

pub struct HeadTailSink<const HEAD: bool> {
    e: Expr,
    extrema: PathMap<()>,
    skip: usize,
    count: usize,
    max: usize,
    extremum: Vec<u8>,
}
impl<const HEAD: bool> Sink for HeadTailSink<HEAD> {
    fn new(e: Expr) -> Self {
        let mut ez = ExprZipper::new(e);
        ez.next();
        ez.next();
        let max_s = ez
            .item()
            .err()
            .expect("cnt can not be an expression or variable");
        let max: usize = str::from_utf8(max_s)
            .expect("string encoded numbers for now")
            .parse()
            .expect("a number");
        assert_ne!(max, 0);
        Self {
            e,
            extrema: PathMap::new(),
            skip: 1 + 1 + 4 + 1 + max_s.len(),
            count: 0,
            max,
            extremum: vec![],
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| {
                    let s = self.e.span();
                    slice_from_raw_parts(self.e.ptr, s.len() - 1)
                })
                .as_ref()
                .unwrap()
        }[self.skip..];
        trace!(target: "sink", "head/tail requesting {}", serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        let mpath = &path[self.skip + wz.root_prefix_path().len()..];
        trace!(target: "sink", "head/tail at '{}' sinking raw '{}'", serialize(wz.root_prefix_path()), serialize(path));
        if self.count == self.max {
            if if HEAD {
                &self.extremum[..] <= mpath
            } else {
                &self.extremum[..] >= mpath
            } {
                trace!(target: "sink", "head/tail at max capacity ignoring '{}'", serialize(mpath));
                // doesn't displace any path
            } else {
                trace!(target: "sink", "head/tail at max capacity replacing '{}' with '{}'", serialize(&self.extremum[..]), serialize(mpath));
                assert!(self.extrema.insert(mpath, ()).is_none());
                self.extrema.remove(&self.extremum[..]);
                let mut rz = self.extrema.read_zipper();
                if HEAD {
                    rz.descend_last_path();
                } else {
                    rz.to_next_val();
                }
                self.extremum.clear();
                self.extremum.extend_from_slice(rz.path()); // yikes, throwing away our needless allocation
            }
        } else if self.extrema.insert(mpath, ()).is_none() {
            let update_extremum = self.count == 0
                || if HEAD {
                    &self.extremum[..] <= mpath
                } else {
                    &self.extremum[..] >= mpath
                };
            if update_extremum {
                trace!(target: "sink", "head/tail adding new extremum at '{}'", serialize(mpath));
                self.extremum.clear();
                self.extremum.extend_from_slice(mpath);
            } else {
                trace!(target: "sink", "head/tail adding '{}'", serialize(mpath));
            }
            self.count += 1;
        }
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        wz.reset();
        trace!(target: "sink", "head/tail finalizing by joining {} at '{}'", self.count, serialize(wz.origin_path()));

        match wz.join_into(&self.extrema.read_zipper()) {
            AlgebraicStatus::Element => true,
            AlgebraicStatus::Identity => false,
            AlgebraicStatus::None => true, // GOAT maybe not?
        }
    }
}

#[cfg(feature = "wasm")]
pub struct WASMSink {
    skip: usize,
    changed: bool,
    _module: wasmtime::Module,
    store: wasmtime::Store<()>,
    instance: wasmtime::Instance,
}

#[cfg(feature = "wasm")]
static ENGINE_LINKER: LazyLock<(wasmtime::Engine, wasmtime::Linker<()>)> = LazyLock::new(|| {
    let memory_policy = wasm_linear_memory_policy_from_env()
        .unwrap_or_else(|error| panic!("invalid WASM sink memory policy: {error}"));
    let mut config = wasmtime::Config::new();
    config.wasm_multi_memory(memory_policy.multi_memory_enabled);
    config.strategy(wasmtime::Strategy::Cranelift);
    config.signals_based_traps(memory_policy.signals_based_traps_enabled);
    config.memory_reservation(memory_policy.reservation_bytes);
    config.memory_guard_size(memory_policy.guard_bytes);
    #[cfg(all(target_feature = "avx2"))]
    unsafe {
        config.cranelift_flag_enable("has_sse3");
        config.cranelift_flag_enable("has_ssse3");
        config.cranelift_flag_enable("has_sse41");
        config.cranelift_flag_enable("has_sse42");
        config.cranelift_flag_enable("has_avx");
        config.cranelift_flag_enable("has_avx2");
        config.cranelift_flag_enable("has_bmi1");
        config.cranelift_flag_enable("has_bmi2");
        config.cranelift_flag_enable("has_lzcnt");
        config.cranelift_flag_enable("has_popcnt");
        config.cranelift_flag_enable("has_fma");
    }
    #[cfg(all(target_feature = "avx512f"))]
    unsafe {
        config.cranelift_flag_enable("has_avx512bitalg");
        config.cranelift_flag_enable("has_avx512dq");
        config.cranelift_flag_enable("has_avx512vl");
        config.cranelift_flag_enable("has_avx512vbmi");
        config.cranelift_flag_enable("has_avx512f");
    }

    let engine = wasmtime::Engine::new(&config).unwrap();

    let mut linker = wasmtime::Linker::new(&engine);

    linker
        .func_wrap("", "i32.bswap", |param: i32| param.to_be())
        .unwrap();
    linker
        .func_wrap("", "i64.bswap", |param: i64| param.to_be())
        .unwrap();

    (engine, linker)
});

#[cfg(feature = "wasm")]
macro_rules! wasm_ctx {
    () => {
        r#"
(module
  (import "" "i32.bswap" (func $i32.bswap (param i32) (result i32)))
  (import "" "i64.bswap" (func $i64.bswap (param i64) (result i64)))

  (memory $in 1)
  (export "in" (memory $in))
  (memory $out 1)
  (export "out" (memory $out))
  (memory $local 1)

  (func (export "_otf_grounding")
    {:?}
  )
)
"#
    };
}

#[cfg(feature = "wasm")]
impl Sink for WASMSink {
    fn new(e: Expr) -> Self {
        let mut ez = ExprZipper::new(e);
        ez.next();
        ez.next();
        let program_e = ez.subexpr();
        let wat = format!(wasm_ctx!(), program_e);
        let module = wasmtime::Module::new(&ENGINE_LINKER.0, wat).unwrap();
        let mut store = wasmtime::Store::new(&ENGINE_LINKER.0, ());
        let instance = (&ENGINE_LINKER.1).instantiate(&mut store, &module).unwrap();

        WASMSink {
            skip: 1 + 1 + 4 + program_e.span().len(),
            changed: false,
            _module: module,
            store,
            instance,
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        // let p = &unsafe { self.e.prefix().unwrap_or_else(|_| { let s = self.e.span(); slice_from_raw_parts(self.e.ptr, s.len() - 1) }).as_ref().unwrap() }[self.skip..];
        // trace!(target: "sink", "wasm requesting {}", serialize(p));
        // std::iter::once(p)
        std::iter::once(WriteResourceRequest::BTM(&[]))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        let mpath = &path[self.skip + wz.root_prefix_path().len()..];
        trace!(target: "sink", "wasm at '{}' sinking raw '{}'", serialize(wz.root_prefix_path()), serialize(path));
        trace!(target: "sink", "wasm input '{}'", serialize(mpath));
        let imem = self.instance.get_memory(&mut self.store, "in").unwrap();
        imem.write(&mut self.store, 0, mpath).unwrap();
        let run = self
            .instance
            .get_typed_func::<(), ()>(&mut self.store, "_otf_grounding")
            .unwrap();
        match run.call(&mut self.store, ()) {
            Ok(()) => {
                let omem = self
                    .instance
                    .get_memory(&mut self.store, "out")
                    .unwrap()
                    .data(&mut self.store);
                let ospan = unsafe {
                    Expr {
                        ptr: omem.as_ptr().cast_mut(),
                    }
                    .span()
                    .as_ref()
                    .unwrap()
                };
                trace!(target: "sink", "wasm output '{}'", serialize(ospan));
                wz.move_to_path(ospan);
                self.changed |= wz.set_val(()).is_none();
            }
            Err(e) => {
                trace!(target: "sink", "wasm error {:?}", e);
            }
        }
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        _it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        trace!(target: "sink", "wasm finalizing");
        self.changed
    }
}

// ($k $x) (f $x $y)
// (count (count of $k is $i) $i ($x $y))   unify
// (count (count of r2 is $i) $i (P Q))
// (count (count of r2 is 3) 3 ($x $y))
pub struct CountSink {
    e: Expr,
    unique: PathMap<()>,
}
impl Sink for CountSink {
    fn new(e: Expr) -> Self {
        CountSink {
            e,
            unique: PathMap::new(),
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| {
                    let s = self.e.span();
                    slice_from_raw_parts(self.e.ptr, s.len() - 1)
                })
                .as_ref()
                .unwrap()
        }[7..];
        trace!(target: "sink", "count requesting {}", serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        let mpath = &path[7 + wz.root_prefix_path().len()..];
        trace!(target: "sink", "count at '{}' sinking raw '{}'", serialize(wz.root_prefix_path()), serialize(path));
        trace!(target: "sink", "count registering in ctx {:?}", serialize(mpath));
        self.unique.insert(mpath, ());
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        wz.reset();
        trace!(target: "sink", "count finalizing by reducing {} at '{}'", self.unique.val_count(), serialize(wz.origin_path()));

        let mut _to_swap = PathMap::new();
        std::mem::swap(&mut self.unique, &mut _to_swap);
        let mut rooted_input = PathMap::new();
        rooted_input
            .write_zipper_at_path(wz.root_prefix_path())
            .graft_map(_to_swap);

        static QUERY_VAR: &'static [u8] = &[item_byte(Tag::NewVar)];
        let mut prz = OneFactor::new(rooted_input.into_read_zipper(&[]));
        let prz_ptr = (&prz) as *const OneFactor<_>;
        let mut changed = false;
        let mut buffer = sink_expr_scratch_buffer();
        crate::space::Space::query_multi_raw(
            unsafe { prz_ptr.cast_mut().as_mut().unwrap() },
            &[ExprEnv::new(
                0,
                Expr {
                    ptr: QUERY_VAR.as_ptr().cast_mut(),
                },
            )],
            |_refs_bindings, _loc| {
                let cnt = prz.val_count();
                trace!(target: "sink", "'{}' and under {}", serialize(prz.path()), cnt);
                let _clen = prz.path().len();
                let cnt_str = cnt.to_string();
                if prz.descend_to_existing_byte(item_byte(Tag::SymbolSize(cnt_str.len() as _))) {
                    let descended = prz.descend_to_existing(cnt_str.as_bytes());
                    if descended == cnt_str.len() {
                        let fixed = &prz.path()[..prz.path().len() - (1 + cnt_str.len())];
                        trace!(target: "sink", "fixed guard {}", serialize(fixed));
                        wz.move_to_path(fixed);
                        wz.set_val(());
                        changed |= true;
                    }
                    prz.ascend(descended + 1);
                }
                if prz.descend_to_existing_byte(item_byte(Tag::NewVar)) {
                    let ignored = &prz.path()[..prz.path().len() - 1];
                    trace!(target: "sink", "ignored guard {}", serialize(ignored));
                    wz.move_to_path(ignored);
                    wz.set_val(());
                    changed |= true;
                    prz.ascend_byte();
                }
                if prz.descend_first_byte() {
                    if let Tag::VarRef(k) = byte_item(prz.path()[prz.path().len() - 1]) {
                        let mut cntv = vec![item_byte(Tag::SymbolSize(cnt_str.len() as _))];
                        cntv.extend_from_slice(cnt_str.as_bytes());
                        let varref = &prz.path()[..prz.path().len() - 1];
                        trace!(target: "sink", "ref guard '{}' var {:?} with '{}'", serialize(varref), k, serialize(&cntv[..]));
                        let output_len =
                            substitute_one_de_bruijn_into_buffer(varref, k, &mut cntv, &mut buffer);
                        trace!(target: "sink", "ref guard subs '{:?}'", serialize(&buffer[..output_len]));
                        wz.move_to_path(&buffer[wz.root_prefix_path().len()..output_len]);
                        wz.set_val(());
                        changed |= true
                    }
                    prz.ascend_byte();
                }
                true
            },
        );
        changed
    }
}

pub struct HashSink {
    e: Expr,
    unique: PathMap<()>,
}
impl Sink for HashSink {
    fn new(e: Expr) -> Self {
        Self {
            e,
            unique: PathMap::new(),
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| {
                    let s = self.e.span();
                    slice_from_raw_parts(self.e.ptr, s.len() - 1)
                })
                .as_ref()
                .unwrap()
        }[6..];
        trace!(target: "sink", "hash requesting {}", serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        let mpath = &path[6 + wz.root_prefix_path().len()..];
        trace!(target: "sink", "hash at '{}' sinking raw '{}'", serialize(wz.root_prefix_path()), serialize(path));
        trace!(target: "sink", "hash registering in ctx {:?}", serialize(mpath));
        self.unique.insert(mpath, ());
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        wz.reset();
        trace!(target: "sink", "hash finalizing by reducing {} at '{}'", self.unique.val_count(), serialize(wz.origin_path()));

        let mut _to_swap = PathMap::new();
        std::mem::swap(&mut self.unique, &mut _to_swap);
        let mut rooted_input = PathMap::new();
        rooted_input
            .write_zipper_at_path(wz.root_prefix_path())
            .graft_map(_to_swap);

        static QUERY_VAR: &'static [u8] = &[item_byte(Tag::NewVar)];
        let mut prz = OneFactor::new(rooted_input.into_read_zipper(&[]));
        let prz_ptr = (&prz) as *const OneFactor<_>;
        let mut changed = false;
        let mut buffer = sink_expr_scratch_buffer();
        crate::space::Space::query_multi_raw(
            unsafe { prz_ptr.cast_mut().as_mut().unwrap() },
            &[ExprEnv::new(
                0,
                Expr {
                    ptr: QUERY_VAR.as_ptr().cast_mut(),
                },
            )],
            |_refs_bindings, _loc| {
                for b in prz.child_mask().and(&ByteMask(crate::space::SIZES)).iter() {
                    let Tag::SymbolSize(size) = byte_item(b) else {
                        unreachable!()
                    };
                    // if size != 16 { trace!(target: "sink", "hash guard not 16 bytes {size}"); continue }
                    prz.descend_to_byte(b);
                    debug_assert!(prz.path_exists());
                    if !prz.descend_first_k_path(size as _) {
                        unreachable!()
                    }
                    loop {
                        let clen = prz.origin_path().len();

                        let hash = prz.fork_read_zipper().hash();

                        let cnt_str = hash.to_be_bytes();
                        trace!(target: "sink", "'{}' and under {}", serialize(prz.origin_path()), hash);
                        assert_eq!(prz.origin_path().len(), clen);

                        let fixed_number =
                            &prz.origin_path()[prz.origin_path().len() - (size as usize)..];
                        if fixed_number == &cnt_str[..] {
                            let fixed =
                                &prz.origin_path()[..prz.origin_path().len() - (1 + size as usize)];
                            trace!(target: "sink", "fixed payload {}", serialize(fixed));
                            wz.move_to_path(fixed);
                            wz.set_val(());
                            changed |= true;
                        }

                        if !prz.to_next_k_path(size as _) {
                            break;
                        }
                    }
                    if !prz.ascend_byte() {
                        unreachable!()
                    }
                }

                if prz.descend_to_existing_byte(item_byte(Tag::NewVar)) {
                    let ignored = &prz.path()[..prz.path().len() - 1];
                    trace!(target: "sink", "ignored guard {}", serialize(ignored));
                    wz.move_to_path(ignored);
                    wz.set_val(());
                    changed |= true;
                    prz.ascend_byte();
                }
                if prz.descend_first_byte() {
                    if let Tag::VarRef(k) = byte_item(prz.path()[prz.path().len() - 1]) {
                        let hash = prz.fork_read_zipper().hash();
                        let cnt_str = hash.to_be_bytes();

                        let mut cntv = vec![item_byte(Tag::SymbolSize(cnt_str.len() as _))];
                        cntv.extend_from_slice(&cnt_str[..]);
                        let varref = &prz.path()[..prz.path().len() - 1];
                        trace!(target: "sink", "hash ref guard '{}' var {:?} with '{}'", serialize(varref), k, serialize(&cntv[..]));
                        let output_len =
                            substitute_one_de_bruijn_into_buffer(varref, k, &mut cntv, &mut buffer);
                        trace!(target: "sink", "hash ref guard subs '{:?}'", serialize(&buffer[..output_len]));
                        wz.move_to_path(&buffer[wz.root_prefix_path().len()..output_len]);
                        wz.set_val(());
                        changed |= true
                    }
                    prz.ascend_byte();
                }
                true
            },
        );
        changed
    }
}

pub struct AndSink {
    e: Expr,
    unique: PathMap<()>,
}
impl Sink for AndSink {
    fn new(e: Expr) -> Self {
        Self {
            e,
            unique: PathMap::new(),
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| {
                    let s = self.e.span();
                    slice_from_raw_parts(self.e.ptr, s.len() - 1)
                })
                .as_ref()
                .unwrap()
        }[5..];
        trace!(target: "sink", "and requesting {}", serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        let mpath = &path[5 + wz.root_prefix_path().len()..];
        trace!(target: "sink", "and at '{}' sinking raw '{}'", serialize(wz.root_prefix_path()), serialize(path));
        trace!(target: "sink", "and registering in ctx {:?}", serialize(mpath));
        self.unique.insert(mpath, ());
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        wz.reset();
        trace!(target: "sink", "and finalizing by reducing {} at '{}'", self.unique.val_count(), serialize(wz.origin_path()));

        let mut _to_swap = PathMap::new();
        std::mem::swap(&mut self.unique, &mut _to_swap);
        let mut rooted_input = PathMap::new();
        rooted_input
            .write_zipper_at_path(wz.root_prefix_path())
            .graft_map(_to_swap);

        static QUERY_VAR: &'static [u8] = &[item_byte(Tag::NewVar)];
        let mut prz = OneFactor::new(rooted_input.into_read_zipper(&[]));
        let prz_ptr = (&prz) as *const OneFactor<_>;
        let mut changed = false;
        let mut buffer = sink_expr_scratch_buffer();
        crate::space::Space::query_multi_raw(
            unsafe { prz_ptr.cast_mut().as_mut().unwrap() },
            &[ExprEnv::new(
                0,
                Expr {
                    ptr: QUERY_VAR.as_ptr().cast_mut(),
                },
            )],
            |_refs_bindings, _loc| {
                for b in prz.child_mask().and(&ByteMask(crate::space::SIZES)).iter() {
                    let Tag::SymbolSize(size) = byte_item(b) else {
                        unreachable!()
                    };
                    println!("and size {size}");
                    prz.descend_to_byte(b);
                    debug_assert!(prz.path_exists());
                    if !prz.descend_first_k_path(size as _) {
                        unreachable!()
                    }
                    loop {
                        let mut total = !0u8;
                        let clen = prz.origin_path().len();

                        let mut rz = prz.fork_read_zipper();
                        while rz.to_next_val() {
                            let p = rz.origin_path();
                            trace!(target: "sink", "path number {:?}", serialize(&p[clen..]));
                            total &= p[clen + 1];
                        }
                        let cnt_str = [total];
                        trace!(target: "sink", "'{}' and under {}", serialize(prz.origin_path()), total);
                        assert_eq!(prz.origin_path().len(), clen);

                        let fixed_number =
                            &prz.origin_path()[prz.origin_path().len() - (size as usize)..];
                        if fixed_number == &cnt_str[..] {
                            let fixed =
                                &prz.origin_path()[..prz.origin_path().len() - (1 + size as usize)];
                            trace!(target: "sink", "fixed payload {}", serialize(fixed));
                            wz.move_to_path(fixed);
                            wz.set_val(());
                            changed |= true;
                        }

                        if !prz.to_next_k_path(size as _) {
                            break;
                        }
                    }
                    if !prz.ascend_byte() {
                        unreachable!()
                    }
                }

                if prz.descend_to_existing_byte(item_byte(Tag::NewVar)) {
                    let ignored = &prz.path()[..prz.path().len() - 1];
                    trace!(target: "sink", "ignored guard {}", serialize(ignored));
                    wz.move_to_path(ignored);
                    wz.set_val(());
                    changed |= true;
                    prz.ascend_byte();
                }
                if prz.descend_first_byte() {
                    if let Tag::VarRef(k) = byte_item(prz.path()[prz.path().len() - 1]) {
                        let mut total = !0u8;
                        let clen = prz.path().len();
                        let mut rz = prz.fork_read_zipper();
                        while rz.to_next_val() {
                            let p = rz.origin_path();
                            trace!(target: "sink", "and path {:?}", serialize(p));
                            trace!(target: "sink", "and path {:?}", serialize(&p[clen+1..]));
                            total &= p[clen + 1];
                        }
                        let cnt_str = [total];

                        let mut cntv = vec![item_byte(Tag::SymbolSize(cnt_str.len() as _))];
                        cntv.extend_from_slice(&cnt_str[..]);
                        let varref = &prz.path()[..prz.path().len() - 1];
                        trace!(target: "sink", "and ref guard '{}' var {:?} with '{}'", serialize(varref), k, serialize(&cntv[..]));
                        let output_len =
                            substitute_one_de_bruijn_into_buffer(varref, k, &mut cntv, &mut buffer);
                        trace!(target: "sink", "and ref guard subs '{:?}'", serialize(&buffer[..output_len]));
                        wz.move_to_path(&buffer[wz.root_prefix_path().len()..output_len]);
                        wz.set_val(());
                        changed |= true
                    }
                    prz.ascend_byte();
                }
                true
            },
        );
        changed
    }
}

pub struct SumSink {
    e: Expr,
    unique: PathMap<()>,
}
impl Sink for SumSink {
    fn new(e: Expr) -> Self {
        SumSink {
            e,
            unique: PathMap::new(),
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| {
                    let s = self.e.span();
                    slice_from_raw_parts(self.e.ptr, s.len() - 1)
                })
                .as_ref()
                .unwrap()
        }[5..];
        trace!(target: "sink", "sum requesting {}", serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        let mpath = &path[5 + wz.root_prefix_path().len()..];
        trace!(target: "sink", "sum at '{}' sinking raw '{}'", serialize(wz.root_prefix_path()), serialize(path));
        trace!(target: "sink", "sum registering in ctx {:?}", serialize(mpath));
        self.unique.insert(mpath, ());
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        wz.reset();
        trace!(target: "sink", "sum finalizing by reducing {} at '{}'", self.unique.val_count(), serialize(wz.origin_path()));

        let mut _to_swap = PathMap::new();
        std::mem::swap(&mut self.unique, &mut _to_swap);
        let mut rooted_input = PathMap::new();
        rooted_input
            .write_zipper_at_path(wz.root_prefix_path())
            .graft_map(_to_swap);

        static QUERY_VAR: &'static [u8] = &[item_byte(Tag::NewVar)];
        let mut prz = OneFactor::new(rooted_input.into_read_zipper(&[]));
        let prz_ptr = (&prz) as *const OneFactor<_>;
        let mut changed = false;
        let mut buffer = sink_expr_scratch_buffer();
        crate::space::Space::query_multi_raw(
            unsafe { prz_ptr.cast_mut().as_mut().unwrap() },
            &[ExprEnv::new(
                0,
                Expr {
                    ptr: QUERY_VAR.as_ptr().cast_mut(),
                },
            )],
            |_refs_bindings, _loc| {
                for b in prz.child_mask().and(&ByteMask(crate::space::SIZES)).iter() {
                    let Tag::SymbolSize(size) = byte_item(b) else {
                        unreachable!()
                    };
                    prz.descend_to_byte(b);
                    debug_assert!(prz.path_exists());
                    if !prz.descend_first_k_path(size as _) {
                        unreachable!()
                    }
                    loop {
                        let mut total = 0u32;
                        let clen = prz.origin_path().len();

                        let mut rz = prz.fork_read_zipper();
                        while rz.to_next_val() {
                            let p = rz.origin_path();
                            trace!(target: "sink", "path number {:?}", serialize(&p[clen..]));
                            total +=
                                u32::from_str_radix(str::from_utf8(&p[clen + 1..]).unwrap(), 10)
                                    .unwrap();
                        }
                        let cnt_str = total.to_string();
                        trace!(target: "sink", "'{}' and under {}", serialize(prz.origin_path()), total);
                        assert_eq!(prz.origin_path().len(), clen);

                        let fixed_number =
                            &prz.origin_path()[prz.origin_path().len() - (size as usize)..];
                        if fixed_number == cnt_str.as_bytes() {
                            let fixed =
                                &prz.origin_path()[..prz.origin_path().len() - (1 + size as usize)];
                            trace!(target: "sink", "fixed payload {}", serialize(fixed));
                            wz.move_to_path(fixed);
                            wz.set_val(());
                            changed |= true;
                        }

                        if !prz.to_next_k_path(size as _) {
                            break;
                        }
                    }
                    if !prz.ascend_byte() {
                        unreachable!()
                    }
                }

                if prz.descend_to_existing_byte(item_byte(Tag::NewVar)) {
                    let ignored = &prz.path()[..prz.path().len() - 1];
                    trace!(target: "sink", "ignored guard {}", serialize(ignored));
                    wz.move_to_path(ignored);
                    wz.set_val(());
                    changed |= true;
                    prz.ascend_byte();
                }
                if prz.descend_first_byte() {
                    if let Tag::VarRef(k) = byte_item(prz.path()[prz.path().len() - 1]) {
                        let mut total = 0u32;
                        let clen = prz.path().len();
                        let mut rz = prz.fork_read_zipper();
                        while rz.to_next_val() {
                            let p = rz.origin_path();
                            trace!(target: "sink", "path {:?}", serialize(p));
                            trace!(target: "sink", "path {:?}", serialize(&p[clen+1..]));
                            total +=
                                u32::from_str_radix(str::from_utf8(&p[clen + 1..]).unwrap(), 10)
                                    .unwrap();
                        }
                        let cnt_str = total.to_string();

                        let mut cntv = vec![item_byte(Tag::SymbolSize(cnt_str.len() as _))];
                        cntv.extend_from_slice(cnt_str.as_bytes());
                        let varref = &prz.path()[..prz.path().len() - 1];
                        trace!(target: "sink", "ref guard '{}' var {:?} with '{}'", serialize(varref), k, serialize(&cntv[..]));
                        let output_len =
                            substitute_one_de_bruijn_into_buffer(varref, k, &mut cntv, &mut buffer);
                        trace!(target: "sink", "ref guard subs '{:?}'", serialize(&buffer[..output_len]));
                        wz.move_to_path(&buffer[wz.root_prefix_path().len()..output_len]);
                        wz.set_val(());
                        changed |= true
                    }
                    prz.ascend_byte();
                }
                true
            },
        );
        changed
    }
}

pub(crate) struct Sum;
pub(crate) struct Min;
pub(crate) struct Max;
pub(crate) struct Prod;

trait FloatReduction {
    const NAME: &'static str;
    const ACC: f64;
    fn op(acc: &mut f64, new: f64);
}
impl FloatReduction for Sum {
    const NAME: &'static str = "fsum";
    const ACC: f64 = 0.0;
    fn op(acc: &mut f64, new: f64) {
        acc.add_assign(new);
    }
}
impl FloatReduction for Min {
    const NAME: &'static str = "fmin";
    const ACC: f64 = f64::MAX;
    fn op(acc: &mut f64, new: f64) {
        *acc = (*acc).min(new)
    }
}
impl FloatReduction for Max {
    const NAME: &'static str = "fmax";
    const ACC: f64 = f64::MIN;
    fn op(acc: &mut f64, new: f64) {
        *acc = (*acc).max(new)
    }
}
impl FloatReduction for Prod {
    const NAME: &'static str = "fprod";
    const ACC: f64 = 1.0;
    fn op(acc: &mut f64, new: f64) {
        acc.mul_assign(new)
    }
}

pub struct FloatReductionSink<Reduction> {
    e: Expr,
    unique: PathMap<()>,
    boo: PhantomData<Reduction>,
}
impl<Reduction: FloatReduction> Sink for FloatReductionSink<Reduction> {
    fn new(e: Expr) -> Self {
        Self {
            e,
            unique: PathMap::new(),
            boo: PhantomData,
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| {
                    let s = self.e.span();
                    slice_from_raw_parts(self.e.ptr, s.len() - 1)
                })
                .as_ref()
                .unwrap()
        }[2 + Reduction::NAME.len()..];
        trace!(target: "sink", "{} requesting {}", Reduction::NAME, serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        let mpath = &path[2 + Reduction::NAME.len() + wz.root_prefix_path().len()..];
        trace!(target: "sink", "{} at '{}' sinking raw '{}'", Reduction::NAME, serialize(wz.root_prefix_path()), serialize(path));
        trace!(target: "sink", "{} registering in ctx {:?}", Reduction::NAME, serialize(mpath));
        self.unique.insert(mpath, ());
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        wz.reset();
        trace!(target: "sink", "{} finalizing by reducing {} at '{}'", Reduction::NAME, self.unique.val_count(), serialize(wz.origin_path()));

        let mut _to_swap = PathMap::new();
        std::mem::swap(&mut self.unique, &mut _to_swap);
        let mut rooted_input = PathMap::new();
        rooted_input
            .write_zipper_at_path(wz.root_prefix_path())
            .graft_map(_to_swap);

        static QUERY_VAR: &'static [u8] = &[item_byte(Tag::NewVar)];
        let mut prz = OneFactor::new(rooted_input.into_read_zipper(&[]));
        let prz_ptr = (&prz) as *const OneFactor<_>;
        let mut changed = false;
        let mut buffer = sink_expr_scratch_buffer();
        crate::space::Space::query_multi_raw(
            unsafe { prz_ptr.cast_mut().as_mut().unwrap() },
            &[ExprEnv::new(
                0,
                Expr {
                    ptr: QUERY_VAR.as_ptr().cast_mut(),
                },
            )],
            |_refs_bindings, _loc| {
                for b in prz.child_mask().and(&ByteMask(crate::space::SIZES)).iter() {
                    let Tag::SymbolSize(size) = byte_item(b) else {
                        unreachable!()
                    };
                    prz.descend_to_byte(b);
                    debug_assert!(prz.path_exists());
                    if !prz.descend_first_k_path(size as _) {
                        unreachable!()
                    }
                    loop {
                        let mut total = Reduction::ACC;
                        let clen = prz.origin_path().len();

                        let mut rz = prz.fork_read_zipper();
                        while rz.to_next_val() {
                            let p = rz.origin_path();
                            trace!(target: "sink", "path number {:?}", serialize(&p[clen..]));
                            Reduction::op(
                                &mut total,
                                str::parse::<f64>(str::from_utf8(&p[clen + 1..]).unwrap()).unwrap(),
                            );
                        }
                        let min_str = total.to_string();
                        trace!(target: "sink", "'{}' and under {}", serialize(prz.origin_path()), total);
                        assert_eq!(prz.origin_path().len(), clen);

                        let fixed_number =
                            &prz.origin_path()[prz.origin_path().len() - (size as usize)..];
                        if fixed_number == min_str.as_bytes() {
                            let fixed =
                                &prz.origin_path()[..prz.origin_path().len() - (1 + size as usize)];
                            trace!(target: "sink", "fixed payload {}", serialize(fixed));
                            wz.move_to_path(fixed);
                            wz.set_val(());
                            changed |= true;
                        }

                        if !prz.to_next_k_path(size as _) {
                            break;
                        }
                    }
                    if !prz.ascend_byte() {
                        unreachable!()
                    }
                }

                if prz.descend_to_existing_byte(item_byte(Tag::NewVar)) {
                    let ignored = &prz.path()[..prz.path().len() - 1];
                    trace!(target: "sink", "ignored guard {}", serialize(ignored));
                    wz.move_to_path(ignored);
                    wz.set_val(());
                    changed |= true;
                    prz.ascend_byte();
                }
                if prz.descend_first_byte() {
                    if let Tag::VarRef(k) = byte_item(prz.path()[prz.path().len() - 1]) {
                        let mut total = Reduction::ACC;
                        let clen = prz.path().len();
                        let mut rz = prz.fork_read_zipper();
                        while rz.to_next_val() {
                            let p = rz.origin_path();
                            trace!(target: "sink", "path {:?}", serialize(p));
                            trace!(target: "sink", "path {:?}", serialize(&p[clen+1..]));
                            Reduction::op(
                                &mut total,
                                str::parse::<f64>(str::from_utf8(&p[clen + 1..]).unwrap()).unwrap(),
                            );
                        }
                        let min_str = total.to_string();

                        let mut cntv = vec![item_byte(Tag::SymbolSize(min_str.len() as _))];
                        cntv.extend_from_slice(min_str.as_bytes());
                        let varref = &prz.path()[..prz.path().len() - 1];
                        trace!(target: "sink", "ref guard '{}' var {:?} with '{}'", serialize(varref), k, serialize(&cntv[..]));
                        let output_len =
                            substitute_one_de_bruijn_into_buffer(varref, k, &mut cntv, &mut buffer);
                        trace!(target: "sink", "ref guard subs '{:?}'", serialize(&buffer[..output_len]));
                        wz.move_to_path(&buffer[wz.root_prefix_path().len()..output_len]);
                        wz.set_val(());
                        changed |= true
                    }
                    prz.ascend_byte();
                }
                true
            },
        );
        changed
    }
}

// (pure (result $x) $x (f32_from_string 0.2))
#[cfg(feature = "grounding")]
pub struct PureSink {
    e: Expr,
    unique: PathMap<()>,
    scope: EvalScope,
}
#[cfg(feature = "grounding")]
impl Sink for PureSink {
    fn new(e: Expr) -> Self {
        let mut scope = EvalScope::new();
        pure::register(&mut scope);
        PureSink {
            e,
            unique: PathMap::new(),
            scope,
        }
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        let p = &unsafe {
            self.e
                .prefix()
                .unwrap_or_else(|_| {
                    let s = self.e.span();
                    slice_from_raw_parts(self.e.ptr, s.len() - 1)
                })
                .as_ref()
                .unwrap()
        }[6..];
        trace!(target: "sink", "count requesting {}", serialize(p));
        std::iter::once(WriteResourceRequest::BTM(p))
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        let mpath = &path[6 + wz.root_prefix_path().len()..];
        trace!(target: "sink", "pure at '{}' sinking raw '{}'", serialize(wz.root_prefix_path()), serialize(path));
        trace!(target: "sink", "pure registering in ctx {:?}", serialize(mpath));
        self.unique.insert(mpath, ());
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        wz.reset();
        trace!(target: "sink", "pure finalizing by reducing {} at '{}'", self.unique.val_count(), serialize(wz.origin_path()));

        let mut _to_swap = PathMap::new();
        std::mem::swap(&mut self.unique, &mut _to_swap);
        let mut rooted_input = PathMap::new();
        rooted_input
            .write_zipper_at_path(wz.root_prefix_path())
            .graft_map(_to_swap);

        static QUERY_VAR: &'static [u8] = &[item_byte(Tag::NewVar)];
        let mut prz = OneFactor::new(rooted_input.into_read_zipper(&[]));
        let prz_ptr = (&prz) as *const OneFactor<_>;
        let mut changed = false;
        let mut buffer = sink_expr_scratch_buffer();
        crate::space::Space::query_multi_raw(
            unsafe { prz_ptr.cast_mut().as_mut().unwrap() },
            &[ExprEnv::new(
                0,
                Expr {
                    ptr: QUERY_VAR.as_ptr().cast_mut(),
                },
            )],
            |_refs_bindings, _loc| {
                for b in prz.child_mask().and(&ByteMask(crate::space::SIZES)).iter() {
                    let Tag::SymbolSize(size) = byte_item(b) else {
                        unreachable!()
                    };
                    prz.descend_to_byte(b);
                    debug_assert!(prz.path_exists());
                    if !prz.descend_first_k_path(size as _) {
                        unreachable!()
                    }
                    loop {
                        let clen = prz.origin_path().len();
                        let fixed_value_len = 1 + size as usize;
                        let fixed_value_start = clen - fixed_value_len;
                        let fixed_value = &prz.origin_path()[fixed_value_start..clen];
                        let fixed = &prz.origin_path()[..fixed_value_start];

                        let mut rz = prz.fork_read_zipper();
                        'vals: while rz.to_next_val() {
                            let p = rz.origin_path();
                            trace!(target: "sink", "path number {:?}", serialize(&p[clen..]));
                            if p.len() == clen {
                                continue 'vals;
                            }

                            let res = match self.scope.eval(ExprSource::new(&p[clen])) {
                                Ok(res) => res,
                                Err(er) => {
                                    trace!(target: "pure", "err {}", er);
                                    continue 'vals;
                                }
                            };

                            trace!(target: "sink", "fixed symbol guard '{}' with result '{}'", serialize(fixed_value), serialize(&res[..]));
                            if res.as_slice() == fixed_value {
                                wz.move_to_path(fixed);
                                wz.set_val(());
                                changed |= true;
                            }
                            self.scope.return_alloc(res);
                        }

                        if !prz.to_next_k_path(size as _) {
                            break;
                        }
                    }
                    if !prz.ascend_byte() {
                        unreachable!()
                    }
                }

                for b in prz
                    .child_mask()
                    .and(&ByteMask(crate::space::ARITIES))
                    .iter()
                {
                    prz.descend_to_byte(b);
                    let fixed_expr_start = prz.path().len() - 1;
                    let mut rz = prz.fork_read_zipper();
                    'vals: while rz.to_next_val() {
                        let p = rz.origin_path();
                        let fixed_expr = Expr {
                            ptr: p[fixed_expr_start..].as_ptr().cast_mut(),
                        };
                        let fixed_expr_end = fixed_expr_start + fixed_expr.span().len();
                        if fixed_expr_end >= p.len() {
                            continue 'vals;
                        }

                        let res = match self.scope.eval(ExprSource::new(&p[fixed_expr_end])) {
                            Ok(res) => res,
                            Err(er) => {
                                trace!(target: "pure", "err {}", er);
                                continue 'vals;
                            }
                        };

                        let fixed_expr_bytes = &p[fixed_expr_start..fixed_expr_end];
                        trace!(target: "sink", "fixed expr guard '{}' with result '{}'", serialize(fixed_expr_bytes), serialize(&res[..]));
                        if res.as_slice() == fixed_expr_bytes {
                            let fixed = &p[..fixed_expr_start];
                            wz.move_to_path(fixed);
                            wz.set_val(());
                            changed |= true;
                        }
                        self.scope.return_alloc(res);
                    }
                    prz.ascend_byte();
                }

                if prz.descend_to_existing_byte(item_byte(Tag::NewVar)) {
                    let ignored = &prz.path()[..prz.path().len() - 1];
                    trace!(target: "sink", "ignored guard {}", serialize(ignored));
                    wz.move_to_path(ignored);
                    wz.set_val(());
                    changed |= true;
                    prz.ascend_byte();
                }
                if prz.descend_first_byte() {
                    if let Tag::VarRef(k) = byte_item(prz.path()[prz.path().len() - 1]) {
                        let clen = prz.path().len();
                        let mut rz = prz.fork_read_zipper();
                        'vals: while rz.to_next_val() {
                            let p = rz.origin_path();
                            trace!(target: "sink", "path {:?}", serialize(p));
                            trace!(target: "sink", "path {:?}", serialize(&p[clen..]));

                            let mut res = match self.scope.eval(ExprSource::new(&p[clen])) {
                                Ok(res) => res,
                                Err(er) => {
                                    trace!(target: "pure", "err {}", er);
                                    continue 'vals;
                                }
                            };

                            trace!(target: "sink", "result {:?}", serialize(&res[..]));

                            let varref = &prz.path()[..prz.path().len() - 1];
                            trace!(target: "sink", "ref guard '{}' var {:?} with '{}'", serialize(varref), k, serialize(&res[..]));
                            let output_len = substitute_one_de_bruijn_into_buffer(
                                varref,
                                k,
                                &mut res,
                                &mut buffer,
                            );
                            trace!(target: "sink", "ref guard subs '{:?}'", serialize(&buffer[..output_len]));
                            wz.move_to_path(&buffer[wz.root_prefix_path().len()..output_len]);
                            wz.set_val(());
                            changed |= true;
                            self.scope.return_alloc(res);
                        }
                    }
                    prz.ascend_byte();
                }
                true
            },
        );
        changed
    }
}

// (z3 <instance> <declaration or assertion>)
#[cfg(feature = "z3")]
pub struct Z3Sink {
    buffer: Vec<u8>,
    ins: &'static str,
}
#[cfg(feature = "z3")]
impl Sink for Z3Sink {
    fn new(e: Expr) -> Self {
        destruct!(e, ("z3" {instance: &str} {decl: Expr}), {
            trace!(target: "sink", "z3 requesting instance {instance}");
            Z3Sink { buffer: vec![], ins: instance }
        }, _err => { unreachable!() })
    }
    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        return std::iter::once(WriteResourceRequest::Z3(self.ins));
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        _it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        let spath = &path[1 + 1 + 2 + 1 + self.ins.bytes().len()..];
        trace!(target: "sink", "z3 sinking '{}'", serialize(spath));
        let e = Expr {
            ptr: spath.as_ptr().cast_mut(),
        };
        e.serialize(&mut self.buffer, |e| std::str::from_utf8(e).unwrap());
        self.buffer.push(b'\n');
    }
    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        trace!(target: "sink", "z3 writing buffer {:?}", std::str::from_utf8(&self.buffer[..]).unwrap());
        let WriteResource::Z3(ref mut p) = it.next().unwrap() else {
            unreachable!()
        };
        let stdin = p.stdin.as_mut().unwrap();
        stdin.write_all(&self.buffer[..]).unwrap();
        stdin.flush().unwrap();
        true
    }
}

fn expr_functor_is(e: Expr, name: &[u8]) -> bool {
    unsafe {
        matches!(byte_item(*e.ptr), Tag::Arity(_))
            && *e.ptr.add(1) == item_byte(Tag::SymbolSize(name.len() as u8))
            && slice_from_raw_parts(e.ptr.add(2), name.len())
                .as_ref()
                .is_some_and(|s| s == name)
    }
}

/// Generic f32 tensor operator sink.
///
/// Syntax:
///
/// ```text
/// (tensor-op-f32
///   (op <operator> ...)
///   (inputs <input-decl>...)
///   (output <output-decl>)
///   (from <input-cell>...)
///   [(emit dense|nonzero|threshold <eps>)]
///   [(backend auto)])
/// ```
///
/// Supported operators are currently `(op einsum <spec>)` and
/// `(op attention scaled-dot)`.
#[cfg(feature = "einsum")]
pub struct TensorOpF32Sink {
    op: TensorOpF32Plan,
    output_name: String,
    output_kind: TensorOutputKind,
    output_shape: Vec<usize>,
    output_prefix: &'static [u8],
    seen: bool,
}

#[cfg(feature = "einsum")]
impl Sink for TensorOpF32Sink {
    fn new(e: Expr) -> Self {
        let syntax = TensorOpF32Syntax::parse(e);
        let op = TensorOpF32Plan::from_syntax(&syntax);
        let output_prefix = syntax.output_prefix();

        Self {
            op,
            output_name: syntax.output_name,
            output_kind: syntax.output_kind,
            output_shape: syntax.output_shape,
            output_prefix,
            seen: false,
        }
    }

    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        std::iter::once(WriteResourceRequest::BTM(self.output_prefix))
    }

    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        _it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        self.seen = true;
        let cells = TensorOpF32Syntax::matched_cells(Expr {
            ptr: path.as_ptr().cast_mut(),
        });
        self.op.sink_cells(&cells);
    }

    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        if !self.seen {
            return false;
        }

        let output = self.op.run(&self.output_shape);
        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        write_dense_output_cells(wz, &output, &self.output_name, self.output_kind)
    }
}

/// Dense f32 einsum grounded sink.
///
/// Syntax:
///
/// ```text
/// (einsum-f32 <spec> <input-decl>... <output-decl> <input-cell>...)
/// ```
///
/// with declarations shaped like `(A 2 3)` and cells like `(A i j value)`.
#[cfg(feature = "einsum")]
pub struct EinsumF32Sink {
    spec: String,
    plan: Option<EinsumF32Plan>,
    output_name: String,
    output_kind: TensorOutputKind,
    output_shape: Vec<usize>,
    output_prefix: &'static [u8],
    inputs: Vec<EinsumInput>,
    seen: bool,
}

#[cfg(feature = "einsum")]
impl Sink for EinsumF32Sink {
    fn new(e: Expr) -> Self {
        let args = expr_args(e);
        assert!(
            args.len() >= 5 && (args.len() - 3) % 2 == 0,
            "einsum-f32 shape is (einsum-f32 spec input-decl... output-decl input-cell...)"
        );

        let spec = symbol_string(args[1]);
        let input_count = (args.len() - 3) / 2;
        let mut inputs = Vec::with_capacity(input_count);
        for i in 0..input_count {
            let (kind, name, shape) = parse_input_tensor_decl(args[2 + i]);
            inputs.push(EinsumInput::new(kind, name, shape));
        }

        let (output_kind, output_name, output_shape) =
            parse_output_tensor_decl(args[2 + input_count]);
        let output_prefix = tensor_cell_prefix(&output_name, output_shape.len());

        for i in 0..input_count {
            validate_tensor_cell_template(
                args[3 + input_count + i],
                inputs[i].name(),
                inputs[i].shape().len(),
            );
        }
        Self {
            spec,
            plan: None,
            output_name,
            output_kind,
            output_shape,
            output_prefix,
            inputs,
            seen: false,
        }
    }

    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        std::iter::once(WriteResourceRequest::BTM(self.output_prefix))
    }

    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        _it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        self.seen = true;
        let args = expr_args(Expr {
            ptr: path.as_ptr().cast_mut(),
        });
        let input_count = self.inputs.len();
        for i in 0..input_count {
            let (name, indices, value) = parse_cell(args[3 + input_count + i]);
            assert_eq!(name, self.inputs[i].name(), "input cell name changed");
            self.inputs[i].set(indices, value);
        }
    }

    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        if !self.seen {
            return false;
        }

        let mut output = Dense::<f32>::zeros(self.output_shape.clone());
        for input in &mut self.inputs {
            input.prepare();
        }
        let jit_inputs: Vec<JitInput<'_>> =
            self.inputs.iter().map(EinsumInput::jit_input).collect();
        if self.plan.is_none() {
            self.plan = Some(
                EinsumF32Plan::compile(
                    &self.spec,
                    &jit_inputs,
                    std::slice::from_ref(&self.output_shape),
                )
                .unwrap_or_else(|err| panic!("einsum-f32 compile failed: {err}")),
            );
        }
        {
            let mut outputs = [&mut output];
            let plan = self.plan.as_ref().unwrap();
            plan.try_run(&jit_inputs, &mut outputs)
                .unwrap_or_else(|err| panic!("einsum-f32 failed: {err}"));
            trace!(target: "sink", "einsum-f32 selected backend {:?}", plan.backend());
        }

        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        write_dense_output_cells(wz, &output, &self.output_name, self.output_kind)
    }
}

/// Dense f32 scaled dot-product attention grounded sink.
///
/// Syntax:
///
/// ```text
/// (attention-f32 <q-decl> <k-decl> <v-decl> <output-decl>
///                <q-cell> <k-cell> <v-cell>)
/// ```
///
/// The sink computes `softmax((QK^T) / sqrt(dim)) V` for tensors shaped as
/// Q `[batch, heads, query, dim]`, K `[batch, heads, key, dim]`, V
/// `[batch, heads, key, value_dim]`, and output
/// `[batch, heads, query, value_dim]`.
#[cfg(feature = "einsum")]
pub struct AttentionF32Sink {
    score_plan: Option<EinsumF32Plan>,
    value_plan: Option<EinsumF32Plan>,
    score_shape: Vec<usize>,
    q_name: String,
    k_name: String,
    v_name: String,
    output_name: String,
    output_kind: TensorOutputKind,
    output_shape: Vec<usize>,
    output_prefix: &'static [u8],
    q: Dense<f32>,
    k: Dense<f32>,
    v: Dense<f32>,
    scale: f32,
    seen: bool,
}

#[cfg(feature = "einsum")]
impl Sink for AttentionF32Sink {
    fn new(e: Expr) -> Self {
        let args = expr_args(e);
        assert_eq!(
            args.len(),
            8,
            "attention-f32 shape is (attention-f32 q-decl k-decl v-decl output-decl q-cell k-cell v-cell)"
        );

        let (q_name, q_shape) = parse_tensor_decl(args[1]);
        let (k_name, k_shape) = parse_tensor_decl(args[2]);
        let (v_name, v_shape) = parse_tensor_decl(args[3]);
        let (output_kind, output_name, output_shape) = parse_output_tensor_decl(args[4]);
        validate_attention_shapes(&q_shape, &k_shape, &v_shape, &output_shape);

        validate_tensor_cell_template(args[5], &q_name, q_shape.len());
        validate_tensor_cell_template(args[6], &k_name, k_shape.len());
        validate_tensor_cell_template(args[7], &v_name, v_shape.len());

        let output_prefix = tensor_cell_prefix(&output_name, output_shape.len());
        let q = Dense::<f32>::zeros(q_shape);
        let k = Dense::<f32>::zeros(k_shape);
        let v = Dense::<f32>::zeros(v_shape);
        let score_shape = vec![q.shape[0], q.shape[1], q.shape[2], k.shape[2]];
        let scale = 1.0 / (q.shape[3] as f32).sqrt();

        Self {
            score_plan: None,
            value_plan: None,
            score_shape,
            q_name,
            k_name,
            v_name,
            output_name,
            output_kind,
            output_shape,
            output_prefix,
            q,
            k,
            v,
            scale,
            seen: false,
        }
    }

    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        std::iter::once(WriteResourceRequest::BTM(self.output_prefix))
    }

    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        _it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        self.seen = true;
        let args = expr_args(Expr {
            ptr: path.as_ptr().cast_mut(),
        });

        let (q_name, q_indices, q_value) = parse_cell(args[5]);
        let (k_name, k_indices, k_value) = parse_cell(args[6]);
        let (v_name, v_indices, v_value) = parse_cell(args[7]);
        assert_eq!(q_name, self.q_name, "Q cell name changed");
        assert_eq!(k_name, self.k_name, "K cell name changed");
        assert_eq!(v_name, self.v_name, "V cell name changed");
        self.q.set(&q_indices, q_value);
        self.k.set(&k_indices, k_value);
        self.v.set(&v_indices, v_value);
    }

    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        mut it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        if !self.seen {
            return false;
        }

        let mut scores = Dense::<f32>::zeros(self.score_shape.clone());
        if self.score_plan.is_none() {
            self.score_plan = Some(
                EinsumF32Plan::compile(
                    "bhqd,bhkd->bhqk",
                    &[JitInput::Dense(&self.q), JitInput::Dense(&self.k)],
                    std::slice::from_ref(&self.score_shape),
                )
                .unwrap_or_else(|err| panic!("attention-f32 score compile failed: {err}")),
            );
        }
        {
            let mut outputs = [&mut scores];
            let score_plan = self.score_plan.as_ref().unwrap();
            score_plan
                .try_run(
                    &[JitInput::Dense(&self.q), JitInput::Dense(&self.k)],
                    &mut outputs,
                )
                .unwrap_or_else(|err| panic!("attention-f32 score pass failed: {err}"));
            trace!(target: "sink", "attention-f32 score backend {:?}", score_plan.backend());
        }

        softmax_attention_rows(&mut scores, self.scale);

        let mut output = Dense::<f32>::zeros(self.output_shape.clone());
        if self.value_plan.is_none() {
            self.value_plan = Some(
                EinsumF32Plan::compile(
                    "bhqk,bhkd->bhqd",
                    &[JitInput::Dense(&scores), JitInput::Dense(&self.v)],
                    std::slice::from_ref(&self.output_shape),
                )
                .unwrap_or_else(|err| panic!("attention-f32 value compile failed: {err}")),
            );
        }
        {
            let mut outputs = [&mut output];
            let value_plan = self.value_plan.as_ref().unwrap();
            value_plan
                .try_run(
                    &[JitInput::Dense(&scores), JitInput::Dense(&self.v)],
                    &mut outputs,
                )
                .unwrap_or_else(|err| panic!("attention-f32 value pass failed: {err}"));
            trace!(target: "sink", "attention-f32 value backend {:?}", value_plan.backend());
        }

        let WriteResource::BTM(wz) = it.next().unwrap() else {
            unreachable!()
        };
        write_dense_output_cells(wz, &output, &self.output_name, self.output_kind)
    }
}

pub(crate) enum ASink {
    AddSink(AddSink),
    RemoveSink(RemoveSink),
    HeadSink(HeadTailSink<true>),
    TailSink(HeadTailSink<false>),
    CountSink(CountSink),
    HashSink(HashSink),
    SumSink(SumSink),
    AndSink(AndSink),
    ACTSink(ACTSink),
    #[cfg(feature = "einsum")]
    TensorOpF32Sink(TensorOpF32Sink),
    #[cfg(feature = "einsum")]
    EinsumF32Sink(EinsumF32Sink),
    #[cfg(feature = "einsum")]
    AttentionF32Sink(Box<AttentionF32Sink>),
    #[cfg(feature = "wasm")]
    WASMSink(WASMSink),
    #[cfg(feature = "grounding")]
    PureSink(PureSink),
    #[cfg(feature = "z3")]
    Z3Sink(Z3Sink),
    AUSink(AUSink),
    USink(USink),
    CompatSink(CompatSink),
    FSumSink(FloatReductionSink<Sum>),
    FMinSink(FloatReductionSink<Min>),
    FMaxSink(FloatReductionSink<Max>),
    FProdSink(FloatReductionSink<Prod>),
}

impl ASink {
    pub fn compat(e: Expr) -> Self {
        ASink::CompatSink(CompatSink::new(e))
    }

    /// Whether this sink deletes facts (a `(- ...)` template). Used to signal the
    /// persistent join sidecar that a removal happened, so it re-syncs.
    pub fn is_remove(&self) -> bool {
        matches!(self, ASink::RemoveSink(_))
    }
}

impl Sink for ASink {
    fn new(e: Expr) -> Self {
        if unsafe {
            *e.ptr == item_byte(Tag::Arity(2))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(1))
                && *e.ptr.offset(2) == b'-'
        } {
            ASink::RemoveSink(RemoveSink::new(e))
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(2))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(1))
                && *e.ptr.offset(2) == b'+'
        } {
            ASink::AddSink(AddSink::new(e))
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(2))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(1))
                && *e.ptr.offset(2) == b'U'
        } {
            ASink::USink(USink::new(e))
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(2))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(2))
                && *e.ptr.offset(2) == b'A'
                && *e.ptr.offset(3) == b'U'
        } {
            ASink::AUSink(AUSink::new(e))
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(3))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(4))
                && *e.ptr.offset(2) == b'h'
                && *e.ptr.offset(3) == b'e'
                && *e.ptr.offset(4) == b'a'
                && *e.ptr.offset(5) == b'd'
        } {
            ASink::HeadSink(HeadTailSink::new(e))
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(3))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(4))
                && *e.ptr.offset(2) == b't'
                && *e.ptr.offset(3) == b'a'
                && *e.ptr.offset(4) == b'i'
                && *e.ptr.offset(5) == b'l'
        } {
            ASink::TailSink(HeadTailSink::new(e))
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(4))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(5))
                && *e.ptr.offset(2) == b'c'
                && *e.ptr.offset(3) == b'o'
                && *e.ptr.offset(4) == b'u'
                && *e.ptr.offset(5) == b'n'
                && *e.ptr.offset(6) == b't'
        } {
            ASink::CountSink(CountSink::new(e))
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(4))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(4))
                && *e.ptr.offset(2) == b'h'
                && *e.ptr.offset(3) == b'a'
                && *e.ptr.offset(4) == b's'
                && *e.ptr.offset(5) == b'h'
        } {
            ASink::HashSink(HashSink::new(e))
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(4))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(3))
                && *e.ptr.offset(2) == b's'
                && *e.ptr.offset(3) == b'u'
                && *e.ptr.offset(4) == b'm'
        } {
            return ASink::SumSink(SumSink::new(e));
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(4))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(4))
                && *e.ptr.offset(2) == b'f'
                && *e.ptr.offset(3) == b's'
                && *e.ptr.offset(4) == b'u'
                && *e.ptr.offset(5) == b'm'
        } {
            return ASink::FSumSink(FloatReductionSink::new(e));
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(4))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(4))
                && *e.ptr.offset(2) == b'f'
                && *e.ptr.offset(3) == b'm'
                && *e.ptr.offset(4) == b'i'
                && *e.ptr.offset(5) == b'n'
        } {
            return ASink::FMinSink(FloatReductionSink::new(e));
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(4))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(4))
                && *e.ptr.offset(2) == b'f'
                && *e.ptr.offset(3) == b'm'
                && *e.ptr.offset(4) == b'a'
                && *e.ptr.offset(5) == b'x'
        } {
            return ASink::FMaxSink(FloatReductionSink::new(e));
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(4))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(5))
                && *e.ptr.offset(2) == b'f'
                && *e.ptr.offset(3) == b'p'
                && *e.ptr.offset(4) == b'r'
                && *e.ptr.offset(5) == b'o'
                && *e.ptr.offset(6) == b'd'
        } {
            return ASink::FProdSink(FloatReductionSink::new(e));
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(4))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(3))
                && *e.ptr.offset(2) == b'a'
                && *e.ptr.offset(3) == b'n'
                && *e.ptr.offset(4) == b'd'
        } {
            return ASink::AndSink(AndSink::new(e));
        } else if expr_functor_is(e, b"tensor-op-f32") {
            #[cfg(feature = "einsum")]
            return ASink::TensorOpF32Sink(TensorOpF32Sink::new(e));
            #[cfg(not(feature = "einsum"))]
            panic!(
                "MORK was not built with the einsum feature, yet trying to call {:?}",
                e
            );
        } else if expr_functor_is(e, b"einsum-f32") {
            #[cfg(feature = "einsum")]
            return ASink::EinsumF32Sink(EinsumF32Sink::new(e));
            #[cfg(not(feature = "einsum"))]
            panic!(
                "MORK was not built with the einsum feature, yet trying to call {:?}",
                e
            );
        } else if expr_functor_is(e, b"attention-f32") {
            #[cfg(feature = "einsum")]
            return ASink::AttentionF32Sink(Box::new(AttentionF32Sink::new(e)));
            #[cfg(not(feature = "einsum"))]
            panic!(
                "MORK was not built with the einsum feature, yet trying to call {:?}",
                e
            );
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(3))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(3))
                && *e.ptr.offset(2) == b'A'
                && *e.ptr.offset(3) == b'C'
                && *e.ptr.offset(4) == b'T'
        } {
            return ASink::ACTSink(ACTSink::new(e));
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(3))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(4))
                && *e.ptr.offset(2) == b'w'
                && *e.ptr.offset(3) == b'a'
                && *e.ptr.offset(4) == b's'
                && *e.ptr.offset(5) == b'm'
        } {
            #[cfg(feature = "wasm")]
            return ASink::WASMSink(WASMSink::new(e));
            #[cfg(not(feature = "wasm"))]
            panic!(
                "MORK was not built with the wasm feature, yet trying to call {:?}",
                e
            );
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(4))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(4))
                && *e.ptr.offset(2) == b'p'
                && *e.ptr.offset(3) == b'u'
                && *e.ptr.offset(4) == b'r'
                && *e.ptr.offset(5) == b'e'
        } {
            #[cfg(feature = "grounding")]
            return ASink::PureSink(PureSink::new(e));
            #[cfg(not(feature = "grounding"))]
            panic!(
                "MORK was not built with the grounding feature, yet trying to call {:?}",
                e
            );
        } else if unsafe {
            *e.ptr == item_byte(Tag::Arity(3))
                && *e.ptr.offset(1) == item_byte(Tag::SymbolSize(2))
                && *e.ptr.offset(2) == b'z'
                && *e.ptr.offset(3) == b'3'
        } {
            #[cfg(feature = "z3")]
            return ASink::Z3Sink(Z3Sink::new(e));
            #[cfg(not(feature = "z3"))]
            panic!(
                "MORK was not built with the z3 feature, yet trying to call {:?}",
                e
            );
        } else {
            panic!("unrecognized sink")
        }
    }

    fn request(&self) -> impl Iterator<Item = WriteResourceRequest> {
        gen move {
            match self {
                ASink::AddSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::USink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::AUSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::RemoveSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::HeadSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::TailSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::CountSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::HashSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::SumSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::AndSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::ACTSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                #[cfg(feature = "einsum")]
                ASink::TensorOpF32Sink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                #[cfg(feature = "einsum")]
                ASink::EinsumF32Sink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                #[cfg(feature = "einsum")]
                ASink::AttentionF32Sink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                #[cfg(feature = "wasm")]
                ASink::WASMSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                #[cfg(feature = "grounding")]
                ASink::PureSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                #[cfg(feature = "z3")]
                ASink::Z3Sink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::CompatSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::FSumSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::FMinSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::FMaxSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
                ASink::FProdSink(s) => {
                    for i in s.request().into_iter() {
                        yield i
                    }
                }
            }
        }
    }
    fn sink<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        it: It,
        path: &[u8],
    ) where
        'a: 'w,
        'k: 'w,
    {
        match self {
            ASink::AddSink(s) => s.sink(it, path),
            ASink::USink(s) => s.sink(it, path),
            ASink::AUSink(s) => s.sink(it, path),
            ASink::RemoveSink(s) => s.sink(it, path),
            ASink::HeadSink(s) => s.sink(it, path),
            ASink::TailSink(s) => s.sink(it, path),
            ASink::CountSink(s) => s.sink(it, path),
            ASink::HashSink(s) => s.sink(it, path),
            ASink::SumSink(s) => s.sink(it, path),
            ASink::AndSink(s) => s.sink(it, path),
            ASink::ACTSink(s) => s.sink(it, path),
            #[cfg(feature = "einsum")]
            ASink::TensorOpF32Sink(s) => s.sink(it, path),
            #[cfg(feature = "einsum")]
            ASink::EinsumF32Sink(s) => s.sink(it, path),
            #[cfg(feature = "einsum")]
            ASink::AttentionF32Sink(s) => s.sink(it, path),
            #[cfg(feature = "wasm")]
            ASink::WASMSink(s) => s.sink(it, path),
            #[cfg(feature = "grounding")]
            ASink::PureSink(s) => s.sink(it, path),
            #[cfg(feature = "z3")]
            ASink::Z3Sink(s) => s.sink(it, path),
            ASink::CompatSink(s) => s.sink(it, path),
            ASink::FSumSink(s) => s.sink(it, path),
            ASink::FMinSink(s) => s.sink(it, path),
            ASink::FMaxSink(s) => s.sink(it, path),
            ASink::FProdSink(s) => s.sink(it, path),
        }
    }

    fn finalize<'w, 'a, 'k, It: Iterator<Item = WriteResource<'w, 'a, 'k>>>(
        &mut self,
        it: It,
    ) -> bool
    where
        'a: 'w,
        'k: 'w,
    {
        match self {
            ASink::AddSink(s) => s.finalize(it),
            ASink::USink(s) => s.finalize(it),
            ASink::AUSink(s) => s.finalize(it),
            ASink::RemoveSink(s) => s.finalize(it),
            ASink::HeadSink(s) => s.finalize(it),
            ASink::TailSink(s) => s.finalize(it),
            ASink::CountSink(s) => s.finalize(it),
            ASink::HashSink(s) => s.finalize(it),
            ASink::SumSink(s) => s.finalize(it),
            ASink::AndSink(s) => s.finalize(it),
            ASink::ACTSink(s) => s.finalize(it),
            #[cfg(feature = "einsum")]
            ASink::TensorOpF32Sink(s) => s.finalize(it),
            #[cfg(feature = "einsum")]
            ASink::EinsumF32Sink(s) => s.finalize(it),
            #[cfg(feature = "einsum")]
            ASink::AttentionF32Sink(s) => s.finalize(it),
            #[cfg(feature = "wasm")]
            ASink::WASMSink(s) => s.finalize(it),
            #[cfg(feature = "grounding")]
            ASink::PureSink(s) => s.finalize(it),
            #[cfg(feature = "z3")]
            ASink::Z3Sink(s) => s.finalize(it),
            ASink::CompatSink(s) => s.finalize(it),
            ASink::FSumSink(s) => s.finalize(it),
            ASink::FMinSink(s) => s.finalize(it),
            ASink::FMaxSink(s) => s.finalize(it),
            ASink::FProdSink(s) => s.finalize(it),
        }
    }
}
