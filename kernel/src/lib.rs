#![feature(gen_blocks)]
#![feature(coroutine_trait)]
#![feature(coroutines)]
#![feature(stmt_expr_attributes)]
#![feature(more_float_constants)]

#[cfg(feature = "experimental_dnf")]
pub mod path_dnf;
#[cfg(feature = "pathspace_oracles")]
pub mod path_space_ops;
pub mod space;
pub mod json_path_query;
mod sources;
mod sinks;
mod pure;

pub use sinks::WriteResourceRequest;
pub use sources::ResourceRequest;
pub mod term_identity;

#[doc(hidden)]
pub use mork_expr as __mork_expr;
