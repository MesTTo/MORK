use mork_expr::{Tag, item_byte};

pub(crate) fn sym(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![item_byte(Tag::SymbolSize(bytes.len() as u8))];
    out.extend_from_slice(bytes);
    out
}

pub(crate) fn app(children: &[Vec<u8>]) -> Vec<u8> {
    let mut out = vec![item_byte(Tag::Arity(children.len() as u8))];
    for child in children {
        out.extend_from_slice(child);
    }
    out
}

pub(crate) fn var() -> Vec<u8> {
    vec![item_byte(Tag::NewVar)]
}

pub(crate) fn var_ref(slot: u8) -> Vec<u8> {
    vec![item_byte(Tag::VarRef(slot))]
}

/// A bare arity tag, for building truncated or hand-laid item sequences that
/// [`app`] (which requires all children) cannot express.
pub(crate) fn arity(n: u8) -> Vec<u8> {
    vec![item_byte(Tag::Arity(n))]
}

/// Raw concatenation of encoded fragments.
pub(crate) fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.iter().flatten().copied().collect()
}
