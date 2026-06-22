#![no_std]

#[cfg(feature = "std")]
extern crate alloc;

pub mod sink;
pub mod source;

pub use mork_expr::{SourceItem, Tag};
pub use sink::ExprSink;
pub use source::ExprSource;

pub type FuncPtr = fn(*mut ExprSource, *mut ExprSink) -> Result<(), EvalError>;
pub type ExternFuncPtr = extern "C" fn(*mut ExprSource, *mut ExprSink) -> EvalStatus;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvalError {
    NotEnoughSpace,
    Msg { ptr: *const u8, len: usize },
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvalStatus {
    pub code: u32,
    pub msg_ptr: *const u8,
    pub msg_len: usize,
}

impl EvalStatus {
    pub const OK: u32 = 0;
    pub const NOT_ENOUGH_SPACE: u32 = 1;
    pub const MSG: u32 = 2;

    pub const fn ok() -> Self {
        Self {
            code: Self::OK,
            msg_ptr: core::ptr::null(),
            msg_len: 0,
        }
    }

    pub fn from_result(result: Result<(), EvalError>) -> Self {
        match result {
            Ok(()) => Self::ok(),
            Err(EvalError::NotEnoughSpace) => Self {
                code: Self::NOT_ENOUGH_SPACE,
                msg_ptr: core::ptr::null(),
                msg_len: 0,
            },
            Err(EvalError::Msg { ptr, len }) => Self {
                code: Self::MSG,
                msg_ptr: ptr,
                msg_len: len,
            },
        }
    }

    pub fn into_result(self) -> Result<(), EvalError> {
        match self.code {
            Self::OK => Ok(()),
            Self::NOT_ENOUGH_SPACE => Err(EvalError::NotEnoughSpace),
            Self::MSG => Err(EvalError::Msg {
                ptr: self.msg_ptr,
                len: self.msg_len,
            }),
            _ => Err(EvalError::from("unknown eval status code")),
        }
    }
}

impl core::fmt::Display for EvalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EvalError::NotEnoughSpace => write!(f, "EvalError: not enough space"),
            EvalError::Msg { ptr, len } => {
                let msg = unsafe { core::slice::from_raw_parts(*ptr, *len) };
                write!(f, "EvalError: {:?}", core::str::from_utf8(msg))
            }
        }
    }
}

impl core::convert::From<&'static str> for EvalError {
    fn from(s: &'static str) -> Self {
        EvalError::Msg {
            ptr: s.as_ptr(),
            len: s.len(),
        }
    }
}

impl core::error::Error for EvalError {}
