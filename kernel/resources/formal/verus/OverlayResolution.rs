use vstd::prelude::*;

verus! {

#[derive(PartialEq, Eq)]
pub enum OverlayEdit {
    Inherit,
    Put(u64),
    Delete,
}

pub open spec fn resolve_spec(
    base: Option<u64>,
    edit: OverlayEdit,
    hidden_by_cut: bool,
) -> Option<u64> {
    match edit {
        OverlayEdit::Put(value) => Some(value),
        OverlayEdit::Delete => None,
        OverlayEdit::Inherit => if hidden_by_cut { None } else { base },
    }
}

pub fn resolve(
    base: Option<u64>,
    edit: OverlayEdit,
    hidden_by_cut: bool,
) -> (result: Option<u64>)
    ensures
        result == resolve_spec(base, edit, hidden_by_cut),
{
    match edit {
        OverlayEdit::Put(value) => Some(value),
        OverlayEdit::Delete => None,
        OverlayEdit::Inherit => if hidden_by_cut { None } else { base },
    }
}

pub proof fn lemma_descendant_put_revives_below_cut(base: Option<u64>, value: u64)
    ensures
        resolve_spec(base, OverlayEdit::Put(value), true) == Some(value),
{
}

pub proof fn lemma_cut_hides_inherited_base(base: Option<u64>)
    ensures
        resolve_spec(base, OverlayEdit::Inherit, true) == None,
{
}

pub proof fn lemma_without_cut_inherit_preserves_base(base: Option<u64>)
    ensures
        resolve_spec(base, OverlayEdit::Inherit, false) == base,
{
}

pub proof fn lemma_delete_wins_over_base(base: Option<u64>, hidden_by_cut: bool)
    ensures
        resolve_spec(base, OverlayEdit::Delete, hidden_by_cut) == None,
{
}

pub proof fn lemma_put_wins_over_base_and_cut(base: Option<u64>, value: u64, hidden_by_cut: bool)
    ensures
        resolve_spec(base, OverlayEdit::Put(value), hidden_by_cut) == Some(value),
{
}

fn main() {}

}
