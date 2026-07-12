#![feature(gen_blocks)]
#![feature(coroutine_trait)]
#![feature(coroutines)]
#![feature(stmt_expr_attributes)]
#![feature(more_float_constants)]

#[cfg(feature = "experimental_dnf")]
pub mod path_dnf;
#[cfg(feature = "pathspace_oracles")]
pub mod path_space_ops;
pub mod ghd;
pub mod space;
pub mod json_path_query;
pub mod zipper_join;
mod sources;
mod sinks;
#[cfg(feature = "retrieval_join")]
pub mod retrieval;
mod pure;

pub use sinks::WriteResourceRequest;
pub use sources::ResourceRequest;
pub mod egraph;
#[cfg(feature = "einsum")]
pub mod graph_tensor;
pub mod term_identity;

#[doc(hidden)]
pub use mork_expr as __mork_expr;
#[doc(hidden)]
pub use mork_frontend as __mork_frontend;
pub mod weighted_paths;
#[cfg(feature = "guarded_emit")]
pub use sinks::{guarded_emit_stats, reset_guarded_emit_stats, GuardedEmitStats};
#[cfg(feature = "witness_select")]
pub use sinks::{reset_witness_select_stats, witness_select_stats};
pub use sinks::{
    wasm_linear_memory_policy, WasmLinearMemoryPolicy, WASM_LINEAR_MEMORY_GUARD_BYTES,
    WASM_LINEAR_MEMORY_RESERVATION_BYTES,
};

pub mod prefix {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct Prefix<'a> {
        pub slice: &'a [u8],
    }

    impl<'a> Prefix<'a> {
        pub fn path(&self) -> &'a [u8] {
            self.slice
        }
    }
}
