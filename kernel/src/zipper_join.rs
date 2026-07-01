//! Zipper-native worst-case-optimal unification leapfrog over variable-width MORK terms.
//!
//! MORK answers a conjunctive query with the ProductZipper, a relation-at-a-time join that
//! materializes the intermediate product before pruning it. This module seeks directly instead,
//! variable-at-a-time, on the PathMap byte-trie: a join variable's value is a variable-width
//! subterm, found by descending the trie with `child_mask` + `descend_to_byte`, its boundary
//! tracked by a parse stack, and a stored variable in the data is a wildcard that unifies. No
//! domain is materialized.
//!
//! Built bottom-up, each layer validated before the next: the byte-scan and the subterm parser
//! here, then the zipper subterm cursor, then the unification leapfrog, gated against the
//! ProductZipper.

use pathmap::PathMap;
use pathmap::utils::ByteMask;
use pathmap::zipper::{Zipper, ZipperAbsolutePath, ZipperIteration, ZipperMoving, ZipperValues};
use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

// MORK tag bytes (top two bits select the tag).
const TOP2: u8 = 0b1100_0000;
const TAG_ARITY: u8 = 0b0000_0000;
const TAG_VARREF: u8 = 0b1000_0000;
#[cfg(test)]
const TAG_SYMSIZE: u8 = 0b1100_0000;
const NEWVAR_BYTE: u8 = 0b1100_0000;
const LOW6: u8 = 0b0011_1111;

/// The least byte present in `mask` that is `>= k`, or `None` if every set bit is below `k`.
/// `ByteMask::next_bit` returns the least bit strictly above its argument, so test `k` itself
/// first. This is the per-byte leapfrog seek on a trie node's children.
#[inline]
pub fn least_ge(mask: &ByteMask, k: u8) -> Option<u8> {
    if (mask.0[(k >> 6) as usize] >> (k & 63)) & 1 == 1 {
        Some(k)
    } else {
        mask.next_bit(k)
    }
}

/// Parse the first complete subterm at `bytes[0..]`, returning its byte length and whether it is
/// ground. The encoding is prefix-free: an `Arity(k)` consumes the next `k` subterms, a
/// `SymbolSize(s)` consumes `s` payload bytes, a `VarRef`/`NewVar` is one byte. Walking a "need one
/// more complete term" counter to zero gives the span. Panics on a truncated term.
#[inline]
fn parse_first_subterm(bytes: &[u8]) -> (usize, bool) {
    try_parse_first_subterm(bytes).expect("truncated encoded subterm")
}

fn try_parse_first_subterm(bytes: &[u8]) -> Option<(usize, bool)> {
    let mut i = 0usize;
    let mut remaining = 1usize;
    let mut ground = true;
    while remaining > 0 {
        let b = *bytes.get(i)?;
        i += 1;
        remaining -= 1;
        match b & TOP2 {
            TAG_ARITY => remaining += (b & LOW6) as usize,
            TAG_VARREF => ground = false,
            _ => {
                // 0b11xxxxxx: NewVar (exactly 0xC0) is a one-byte variable; SymbolSize carries `s`
                // payload bytes.
                if b == NEWVAR_BYTE {
                    ground = false;
                } else {
                    i = i.checked_add((b & LOW6) as usize)?;
                    if i > bytes.len() {
                        return None;
                    }
                }
            }
        }
    }
    Some((i, ground))
}

/// Byte length of the first complete subterm at `bytes[0..]`.
pub fn first_subterm_len(bytes: &[u8]) -> usize {
    parse_first_subterm(bytes).0
}

/// Whether the first complete subterm at `bytes[0..]` is ground (contains no variable).
pub fn first_subterm_is_ground(bytes: &[u8]) -> bool {
    parse_first_subterm(bytes).1
}

/// One step of the incremental parse: consume byte `b`, updating how many complete subterms are
/// still owed (`subterms`) and how many raw symbol-payload bytes are still owed (`payload`). A
/// payload byte completes nothing; a tag byte completes one slot, then an `Arity(k)` owes `k` more
/// subterms and a `SymbolSize(s)` owes `s` payload bytes.
#[inline]
fn step_parse(b: u8, subterms: &mut usize, payload: &mut usize) {
    if *payload > 0 {
        *payload -= 1;
    } else {
        *subterms -= 1;
        match b & TOP2 {
            TAG_ARITY => *subterms += (b & LOW6) as usize,
            TAG_VARREF => {}
            _ => {
                if b != NEWVAR_BYTE {
                    *payload += (b & LOW6) as usize;
                }
            }
        }
    }
}

/// Whether `bytes` (from the column-start focus) spell exactly one complete subterm. Recomputed
/// per descent step; subterms are short, so the O(len) replay is cheap and keeps the navigation
/// free of incremental-state bugs.
#[inline]
fn is_complete(bytes: &[u8]) -> bool {
    let (mut subterms, mut payload) = (1usize, 0usize);
    for &b in bytes {
        step_parse(b, &mut subterms, &mut payload);
    }
    subterms == 0 && payload == 0
}

#[inline]
fn has_bit(mask: &ByteMask, b: u8) -> bool {
    (mask.0[(b >> 6) as usize] >> (b & 63)) & 1 == 1
}

/// A cursor over the complete variable-width subterms branching from a PathMap zipper's focus, in
/// ascending lexicographic order, with a leapfrog `seek`. This is the zipper-native replacement for
/// a materialized per-variable domain: it seeks on the live byte-trie instead of scanning a `Vec`.
///
/// `key` holds the bytes of the current subterm relative to the focus the cursor was built at
/// (its "floor"). The cursor descends with `descend_to_byte` and ascends with `ascend_byte`, never
/// above the floor (it stops when `key` is empty), so the zipper is left at the floor between
/// re-seeks and at the subterm boundary while positioned.
pub struct SubtermCursor<Z> {
    z: Z,
    key: Vec<u8>,
    at_end: bool,
}

impl<Z: Zipper + ZipperMoving> SubtermCursor<Z> {
    /// Build a cursor at the zipper's current focus. Not positioned until `first`/`seek` is called.
    pub fn new(z: Z) -> Self {
        SubtermCursor {
            z,
            key: Vec::new(),
            at_end: true,
        }
    }

    /// Ascend back to the floor (column start), clearing the key.
    fn reset_to_floor(&mut self) {
        while self.key.pop().is_some() {
            self.z.ascend_byte();
        }
        self.at_end = false;
    }

    /// Descend the least child at each step until the key forms a complete subterm. Returns false
    /// if a node runs out of children before completion (malformed/empty branch).
    fn complete_leftmost(&mut self) -> bool {
        while !is_complete(&self.key) {
            let mask = self.z.child_mask();
            match least_ge(&mask, 0) {
                Some(b) => {
                    self.z.descend_to_byte(b);
                    self.key.push(b);
                }
                None => return false,
            }
        }
        true
    }

    /// From the current complete subterm, move to the least subterm strictly greater: ascend until a
    /// level offers a larger sibling, take the least such, then complete leftmost. False = exhausted.
    fn backtrack_then_leftmost(&mut self) -> bool {
        loop {
            let Some(last) = self.key.pop() else {
                return false;
            };
            self.z.ascend_byte();
            let mask = self.z.child_mask();
            if let Some(b) = mask.next_bit(last) {
                self.z.descend_to_byte(b);
                self.key.push(b);
                return self.complete_leftmost();
            }
        }
    }

    /// Position at the least subterm.
    pub fn first(&mut self) {
        self.reset_to_floor();
        if !self.complete_leftmost() {
            self.at_end = true;
        }
    }

    /// Advance to the next subterm.
    pub fn next(&mut self) {
        if self.at_end {
            return;
        }
        if !self.backtrack_then_leftmost() {
            self.at_end = true;
        }
    }

    /// The current subterm bytes, or `None` when exhausted.
    pub fn key(&self) -> Option<&[u8]> {
        if self.at_end { None } else { Some(&self.key) }
    }

    pub fn at_end(&self) -> bool {
        self.at_end
    }

    /// Position at the least subterm `>= target`. `target` must itself be a complete subterm (the
    /// leapfrog only ever seeks to another factor's bound subterm value). Because the encoding is
    /// prefix-free and `target` is complete, a completed descent matches `target` exactly; any
    /// divergence is handled by taking the least larger child (then completing leftmost) or, when no
    /// larger child exists at that level, backtracking to an ancestor that offers one.
    pub fn seek(&mut self, target: &[u8]) {
        self.reset_to_floor();
        let mut ti = 0usize;
        loop {
            if is_complete(&self.key) {
                self.at_end = false;
                return;
            }
            let mask = self.z.child_mask();
            if ti < target.len() {
                let t = target[ti];
                if has_bit(&mask, t) {
                    self.z.descend_to_byte(t);
                    self.key.push(t);
                    ti += 1;
                    continue;
                }
                match mask.next_bit(t) {
                    Some(b) => {
                        self.z.descend_to_byte(b);
                        self.key.push(b);
                        if !self.complete_leftmost() {
                            self.at_end = true;
                        }
                        return;
                    }
                    None => {
                        if !self.backtrack_then_leftmost() {
                            self.at_end = true;
                        }
                        return;
                    }
                }
            } else {
                if !self.complete_leftmost() {
                    self.at_end = true;
                }
                return;
            }
        }
    }
}

/// Leapfrog intersection of several subterm cursors: the subterm values present in ALL of them, in
/// ascending order. The textbook leapfrog step seeks every cursor to the current maximum key; when
/// they all agree, that key is in the intersection, then one cursor steps past it. Each step either
/// emits a match and advances, or jumps a cursor forward, so it terminates and is worst-case-optimal
/// on the cursors' sizes.
fn intersect<Z: Zipper + ZipperMoving>(cursors: &mut [SubtermCursor<Z>]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if cursors.is_empty() {
        return out;
    }
    for c in cursors.iter_mut() {
        c.first();
        if c.at_end() {
            return out;
        }
    }
    loop {
        let max = cursors
            .iter()
            .map(|c| c.key().unwrap())
            .max()
            .unwrap()
            .to_vec();
        let mut all_match = true;
        for c in cursors.iter_mut() {
            if c.key().unwrap() != max.as_slice() {
                c.seek(&max);
                if c.at_end() {
                    return out;
                }
                if c.key().unwrap() != max.as_slice() {
                    all_match = false;
                }
            }
        }
        if all_match {
            out.push(max);
            cursors[0].next();
            if cursors[0].at_end() {
                return out;
            }
        }
    }
}

/// A non-ground compound query argument. `arity` is the leading arity byte, and `children`
/// are the compound's encoded children in order, including the head.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompoundPat {
    pub arity: u8,
    pub children: Vec<Col>,
}

/// One query argument column in a factor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Col {
    Var(usize),
    Ground(Vec<u8>),
    Compound(Box<CompoundPat>),
}

/// A query factor: its relation prefix in the PathMap, and every argument column in syntactic
/// column order. The prefix is only the arity byte plus relation head. Ground arguments stay as
/// columns so they can unify with stored data variables at that trie position.
#[derive(Clone, Debug)]
pub struct Factor {
    pub prefix: Vec<u8>,
    pub cols: Vec<Col>,
}

impl Factor {
    pub fn var_cols(prefix: Vec<u8>, cols: Vec<usize>) -> Self {
        Factor {
            prefix,
            cols: cols.into_iter().map(Col::Var).collect(),
        }
    }
}

/// Ground worst-case-optimal join over PathMap factors, seeking variable-width subterms directly on
/// the byte-trie with no materialized domain. `var_order` lists the global variables in binding
/// order; it must be compatible with every factor's column order (each factor's variables, in
/// `var_order`, occur in column order), which holds for any acyclic query under a suitable order.
/// Cyclic queries that admit no compatible order are handled by re-indexing in a later layer.
///
/// Returns one row per answer: `row[v]` is the bound subterm bytes for global variable `v`.
pub fn ground_join(
    map: &PathMap<()>,
    factors: &[Factor],
    var_order: &[usize],
    nvars: usize,
) -> Vec<Vec<Vec<u8>>> {
    let mut state = GroundJoin {
        map,
        factors,
        var_order,
        bound: vec![Vec::new(); factors.len()],
        next_col: vec![0; factors.len()],
        binding: vec![Vec::new(); nvars],
        out: Vec::new(),
    };
    state.recurse(0);
    state.out
}

struct GroundJoin<'a> {
    map: &'a PathMap<()>,
    factors: &'a [Factor],
    var_order: &'a [usize],
    bound: Vec<Vec<u8>>,
    next_col: Vec<usize>,
    binding: Vec<Vec<u8>>,
    out: Vec<Vec<Vec<u8>>>,
}

impl GroundJoin<'_> {
    fn recurse(&mut self, i: usize) {
        self.catch_up(i, 0);
    }

    fn recurse_after_catch_up(&mut self, i: usize) {
        if i == self.var_order.len() {
            if (0..self.factors.len()).all(|f| self.factor_has_value(f)) {
                self.out.push(self.binding.clone());
            }
            return;
        }
        let v = self.var_order[i];
        let parts: Vec<usize> = (0..self.factors.len())
            .filter(|&f| {
                let nc = self.next_col[f];
                matches!(self.factors[f].cols.get(nc), Some(Col::Var(cv)) if *cv == v)
            })
            .collect();

        // Open one cursor per participating factor at its current position (relation prefix plus the
        // bytes of its already-bound columns), then leapfrog-intersect their next-column subterms.
        let mut cursors: Vec<_> = parts
            .iter()
            .map(|&f| {
                let mut path = self.factors[f].prefix.clone();
                path.extend_from_slice(&self.bound[f]);
                SubtermCursor::new(self.map.read_zipper_at_path(&path))
            })
            .collect();
        let vals = intersect(&mut cursors);
        drop(cursors);

        for val in vals {
            for &f in &parts {
                self.bound[f].extend_from_slice(&val);
                self.next_col[f] += 1;
            }
            self.binding[v] = val.clone();
            self.recurse(i + 1);
            self.binding[v].clear();
            for &f in &parts {
                let len = self.bound[f].len() - val.len();
                self.bound[f].truncate(len);
                self.next_col[f] -= 1;
            }
        }
    }

    fn factor_path(&self, f: usize) -> Vec<u8> {
        let mut path = self.factors[f].prefix.clone();
        path.extend_from_slice(&self.bound[f]);
        path
    }

    fn factor_has_value(&self, f: usize) -> bool {
        if self.next_col[f] != self.factors[f].cols.len() {
            return false;
        }
        let path = self.factor_path(f);
        self.map.read_zipper_at_path(&path).val().is_some()
    }

    fn consume_exact_column(&mut self, f: usize, target: &[u8]) -> bool {
        let path = self.factor_path(f);
        let mut cur = SubtermCursor::new(self.map.read_zipper_at_path(&path));
        cur.seek(target);
        if cur.key() == Some(target) {
            self.bound[f].extend_from_slice(target);
            self.next_col[f] += 1;
            true
        } else {
            false
        }
    }

    fn catch_up(&mut self, i: usize, f: usize) {
        if f == self.factors.len() {
            self.recurse_after_catch_up(i);
            return;
        }
        let Some(col) = self.factors[f].cols.get(self.next_col[f]).cloned() else {
            self.catch_up(i, f + 1);
            return;
        };
        let target = match col {
            Col::Ground(g) => Some(g),
            Col::Var(v) if !self.binding[v].is_empty() => Some(self.binding[v].clone()),
            Col::Var(_) => None,
            Col::Compound(_) => None,
        };
        let Some(target) = target else {
            self.catch_up(i, f + 1);
            return;
        };
        if self.consume_exact_column(f, &target) {
            self.catch_up(i, f);
            let len = self.bound[f].len() - target.len();
            self.bound[f].truncate(len);
            self.next_col[f] -= 1;
        }
    }
}

// ---- unification layer: schematic data (stored variables in facts act as wildcards) ----

/// Trail-backed first-order unifier over MORK byte terms. Query variables hold ids `0..nvars`;
/// stored data variables get fresh ids past that, allocated per fact descent so they are renamed
/// apart across facts.
struct Env {
    slots: Vec<Option<Col>>,
    trail: Vec<usize>,
}

impl Env {
    fn new(nvars: usize) -> Env {
        Env {
            slots: vec![None; nvars],
            trail: Vec::new(),
        }
    }

    /// Allocate a fresh (unbound) variable id, for a stored data variable.
    fn fresh(&mut self) -> usize {
        self.slots.push(None);
        self.slots.len() - 1
    }

    fn mark(&self) -> usize {
        self.trail.len()
    }

    fn rollback(&mut self, m: usize) {
        while self.trail.len() > m {
            let id = self.trail.pop().unwrap();
            self.slots[id] = None;
        }
    }

    fn bind_term(&mut self, id: usize, term: Col) {
        self.slots[id] = Some(term);
        self.trail.push(id);
    }

    fn resolve_term(&self, term: &Col) -> Col {
        match term {
            Col::Var(id) => match &self.slots[*id] {
                Some(bound) => self.resolve_term(bound),
                None => Col::Var(*id),
            },
            Col::Ground(g) => Col::Ground(g.clone()),
            Col::Compound(compound) => Col::Compound(Box::new(CompoundPat {
                arity: compound.arity,
                children: compound
                    .children
                    .iter()
                    .map(|child| self.resolve_term(child))
                    .collect(),
            })),
        }
    }

    fn unify_terms(&mut self, a: &Col, b: &Col) -> bool {
        let mark = self.mark();
        if self.unify_terms_inner(a, b) {
            true
        } else {
            self.rollback(mark);
            false
        }
    }

    fn unify_terms_inner(&mut self, a: &Col, b: &Col) -> bool {
        let a = self.resolve_term(a).structural_ground();
        let b = self.resolve_term(b).structural_ground();
        match (&a, &b) {
            (Col::Var(x), Col::Var(y)) if x == y => true,
            (Col::Var(x), _) => {
                if self.occurs(*x, &b) {
                    false
                } else {
                    self.bind_term(*x, b);
                    true
                }
            }
            (_, Col::Var(y)) => {
                if self.occurs(*y, &a) {
                    false
                } else {
                    self.bind_term(*y, a);
                    true
                }
            }
            (Col::Ground(ga), Col::Ground(gb)) => ga == gb,
            (Col::Compound(ca), Col::Compound(cb)) => {
                if ca.arity != cb.arity || ca.children.len() != cb.children.len() {
                    return false;
                }
                ca.children
                    .iter()
                    .zip(&cb.children)
                    .all(|(x, y)| self.unify_terms_inner(x, y))
            }
            _ => false,
        }
    }

    fn occurs(&self, id: usize, term: &Col) -> bool {
        match self.resolve_term(term).structural_ground() {
            Col::Var(v) => v == id,
            Col::Ground(_) => false,
            Col::Compound(compound) => compound.children.iter().any(|child| self.occurs(id, child)),
        }
    }

    /// Unify variable `id` with a ground value `g`. False on a clash.
    fn unify_var_ground(&mut self, id: usize, g: &[u8]) -> bool {
        self.unify_terms(&Col::Var(id), &Col::Ground(g.to_vec()))
    }

    /// Unify variable `id` with `term`. False on a clash or occurs-check failure.
    fn unify_var_term(&mut self, id: usize, term: &Col) -> bool {
        self.unify_terms(&Col::Var(id), term)
    }

    /// Unify two variables. False on a clash or occurs-check failure.
    fn unify_var_var(&mut self, a: usize, b: usize) -> bool {
        self.unify_terms(&Col::Var(a), &Col::Var(b))
    }

    /// The ground value a query variable resolved to, or `None` if it is still free or schematic.
    #[cfg(test)]
    fn ground_of(&self, id: usize) -> Option<Vec<u8>> {
        self.term_bytes_of(id)
            .filter(|bytes| first_subterm_is_ground(bytes))
    }

    /// The resolved bytes for a query variable. A free variable returns `None`; a variable bound to
    /// a compound containing free variables returns canonical schematic bytes.
    fn term_bytes_of(&self, id: usize) -> Option<Vec<u8>> {
        match self.resolve_term(&Col::Var(id)) {
            Col::Var(_) => None,
            term => Some(self.encode_schematic(&term)),
        }
    }

    fn encode_schematic(&self, term: &Col) -> Vec<u8> {
        let mut out = Vec::new();
        let mut intro = std::collections::BTreeMap::new();
        self.encode_schematic_into(term, &mut out, &mut intro);
        out
    }

    fn encode_schematic_into(
        &self,
        term: &Col,
        out: &mut Vec<u8>,
        intro: &mut std::collections::BTreeMap<usize, u8>,
    ) {
        match self.resolve_term(term) {
            Col::Var(id) => match intro.get(&id) {
                Some(&level) => out.push(TAG_VARREF | level),
                None => {
                    let level = intro.len() as u8;
                    intro.insert(id, level);
                    out.push(NEWVAR_BYTE);
                }
            },
            Col::Ground(g) => out.extend_from_slice(&g),
            Col::Compound(compound) => {
                out.push(compound.arity);
                for child in &compound.children {
                    self.encode_schematic_into(child, out, intro);
                }
            }
        }
    }

    /// Encode the whole answer tuple (query variables `0..nvars`, in order) through ONE shared
    /// intro map, so a free variable that appears in more than one answer position (data-induced
    /// coreference) emits as coordinated NewVar/VarRef, the way MORK's exec emit numbers the answer.
    /// Ground and schematic components encode exactly as `encode_schematic` does.
    fn encode_tuple_coordinated(&self, nvars: usize) -> Vec<u8> {
        let mut out = Vec::new();
        let mut intro = std::collections::BTreeMap::new();
        for v in 0..nvars {
            self.encode_schematic_into(&Col::Var(v), &mut out, &mut intro);
        }
        out
    }
}

impl Col {
    fn structural_ground(self) -> Col {
        match self {
            Col::Ground(g) if !g.is_empty() && g[0] & TOP2 == TAG_ARITY => {
                Col::Compound(Box::new(compound_from_ground_bytes(&g)))
            }
            other => other,
        }
    }

    fn min_var_pos(&self, var_pos: &[usize]) -> Option<usize> {
        match self {
            Col::Var(v) => Some(var_pos[*v]),
            Col::Ground(_) => None,
            Col::Compound(compound) => compound
                .children
                .iter()
                .filter_map(|child| child.min_var_pos(var_pos))
                .min(),
        }
    }
}

fn compound_from_ground_bytes(bytes: &[u8]) -> CompoundPat {
    debug_assert!(!bytes.is_empty() && bytes[0] & TOP2 == TAG_ARITY);
    let arity = bytes[0];
    let mut children = Vec::with_capacity((arity & LOW6) as usize);
    let mut pos = 1usize;
    for _ in 0..(arity & LOW6) as usize {
        let len = first_subterm_len(&bytes[pos..]);
        children.push(Col::Ground(bytes[pos..pos + len].to_vec()));
        pos += len;
    }
    debug_assert_eq!(pos, bytes.len());
    CompoundPat { arity, children }
}

/// A candidate child at a factor's column: a ground subterm value, or a stored-variable wildcard
/// (its one tag byte: NewVar `0xC0` or VarRef `0x80|i`).
enum Cand {
    Ground(Vec<u8>),
    Wild(u8),
}

#[inline]
fn is_wildcard_byte(k: &[u8]) -> bool {
    k.len() == 1 && (0x80..=0xC0).contains(&k[0])
}

/// The children of a factor's current column when the join variable is still free. This is the
/// lead, which enumerates the whole column (structured children and wildcards alike).
fn free_candidates<Z: Zipper + ZipperMoving>(cur: &mut SubtermCursor<Z>) -> Vec<Cand> {
    let mut out = Vec::new();
    cur.first();
    while let Some(k) = cur.key() {
        if is_wildcard_byte(k) {
            out.push(Cand::Wild(k[0]));
        } else {
            out.push(Cand::Ground(k.to_vec()));
        }
        cur.next();
    }
    out
}

fn add_ground_matches<Z: Zipper + ZipperMoving>(
    cur: &mut SubtermCursor<Z>,
    g: &[u8],
    out: &mut Vec<Cand>,
) {
    cur.seek(g);
    if cur.key() == Some(g) {
        out.push(Cand::Ground(g.to_vec()));
    }
    cur.seek(&[0x80]);
    while let Some(k) = cur.key() {
        if is_wildcard_byte(k) {
            out.push(Cand::Wild(k[0]));
            cur.next();
        } else {
            break;
        }
    }
}

fn ground_candidates<Z: Zipper + ZipperMoving>(cur: &mut SubtermCursor<Z>, g: &[u8]) -> Vec<Cand> {
    let mut out = Vec::new();
    add_ground_matches(cur, g, &mut out);
    out
}

/// A factor is inverted when its columns are not in `var_order` order, so the join cannot seek it
/// forward (a later column's variable is bound before an earlier one). The triangle's third factor
/// `(e $z $x)` under order `$x,$y,$z` is the case: its `$x` column comes second but binds first.
fn is_inverted(factor: &Factor, var_pos: &[usize]) -> bool {
    let mut prev = None;
    for col in &factor.cols {
        if let Some(pos) = col.min_var_pos(var_pos) {
            if prev.is_some_and(|p| p > pos) {
                return true;
            }
            prev = Some(pos);
        }
    }
    false
}

/// One position in a re-emitted subterm: a literal byte, or a variable identified by its original
/// id (so the re-index can renumber it canonically in the new column order).
enum Item {
    Byte(u8),
    Var(usize),
}

/// Split a fact's column bytes (everything after the relation prefix) into its `ncols` subterms.
fn split_columns(bytes: &[u8], ncols: usize) -> Vec<&[u8]> {
    let mut cols = Vec::with_capacity(ncols);
    let mut i = 0;
    for _ in 0..ncols {
        let len = first_subterm_len(&bytes[i..]);
        cols.push(&bytes[i..i + len]);
        i += len;
    }
    cols
}

/// Decode each column into items, tagging every variable with its original id. NewVar takes the next
/// id in encounter order across the whole fact; VarRef(i) refers to id `i`. This is what lets the
/// re-index renumber a coreferent schematic fact, say `(e $u $u)`, correctly after its columns move.
fn columns_to_items(cols: &[&[u8]]) -> Vec<Vec<Item>> {
    let mut next_orig = 0usize;
    let mut out = Vec::with_capacity(cols.len());
    for col in cols {
        let mut items = Vec::new();
        let mut i = 0;
        while i < col.len() {
            let b = col[i];
            i += 1;
            match b & TOP2 {
                TAG_ARITY => items.push(Item::Byte(b)),
                TAG_VARREF => items.push(Item::Var((b & LOW6) as usize)),
                _ => {
                    if b == NEWVAR_BYTE {
                        items.push(Item::Var(next_orig));
                        next_orig += 1;
                    } else {
                        items.push(Item::Byte(b));
                        for _ in 0..(b & LOW6) as usize {
                            items.push(Item::Byte(col[i]));
                            i += 1;
                        }
                    }
                }
            }
        }
        out.push(items);
    }
    out
}

/// Re-emit the columns in `new_order`, renumbering variables so the first reference to each original
/// id (in the new order) is a NewVar and later references are a VarRef of its new index. Produces a
/// canonical, self-consistent encoding for the re-indexed key.
fn emit_reordered(items_by_col: &[Vec<Item>], new_order: &[usize]) -> Vec<u8> {
    use std::collections::HashMap;
    let mut out = Vec::new();
    let mut renum: HashMap<usize, usize> = HashMap::new();
    for &c in new_order {
        for item in &items_by_col[c] {
            match item {
                Item::Byte(b) => out.push(*b),
                Item::Var(orig) => match renum.get(orig) {
                    Some(&new_id) => out.push(TAG_VARREF | new_id as u8),
                    None => {
                        renum.insert(*orig, renum.len());
                        out.push(NEWVAR_BYTE);
                    }
                },
            }
        }
    }
    out
}

/// Re-index an inverted factor: copy its facts into a fresh PathMap with the columns permuted into
/// `var_order` position order (variables renumbered to stay canonical). Returns that map and the new
/// column-variable list, now non-decreasing, so the join seeks it like any compatible factor. This
/// is the one partial materialization the cyclic case needs, and only the inverted factor pays it;
/// re-keying into another attribute order is the standard worst-case-optimal answer to a cycle.
fn build_reindex(map: &PathMap<()>, factor: &Factor, var_pos: &[usize]) -> (PathMap<()>, Vec<Col>) {
    let ncols = factor.cols.len();
    let mut new_order: Vec<usize> = (0..ncols).collect();
    new_order.sort_by_key(|&c| match &factor.cols[c] {
        Col::Ground(_) => (0usize, 0usize, c),
        col => (
            col.min_var_pos(var_pos).map_or(usize::MAX, |pos| pos + 1),
            1usize,
            c,
        ),
    });
    let new_cols: Vec<Col> = new_order.iter().map(|&c| factor.cols[c].clone()).collect();

    let mut reindex = PathMap::<()>::new();
    let plen = factor.prefix.len();
    let mut rz = map.read_zipper_at_path(&factor.prefix);
    while rz.to_next_val() {
        let full = rz.origin_path();
        let cols = split_columns(&full[plen..], ncols);
        let items = columns_to_items(&cols);
        reindex.insert(&emit_reordered(&items, &new_order), ());
    }
    (reindex, new_cols)
}

/// Worst-case-optimal leapfrog-UNIFICATION join directly on the PathMap byte-trie, returning the
/// fully-ground answer rows (`row[v]` = global variable `v`'s value). A row with any still-free query
/// variable is dropped here; the live route uses [`unify_join_zipper_partial`] instead, to keep it
/// and bind only its ground components, exactly as the materialized route does.
pub fn unify_join_zipper(
    map: &PathMap<()>,
    factors: &[Factor],
    var_order: &[usize],
    nvars: usize,
) -> BTreeSet<Vec<Vec<u8>>> {
    unify_join_zipper_partial(map, factors, var_order, nvars)
        .into_iter()
        .filter_map(|row| {
            row.into_iter()
                .map(|component| component.filter(|bytes| first_subterm_is_ground(bytes)))
                .collect::<Option<Vec<Vec<u8>>>>()
        })
        .collect()
}

/// As [`unify_join_zipper`], but each answer component is `Some(bytes)` when the query variable
/// resolved to a concrete term (ground or schematic) and `None` when it stayed free. Generalizes
/// [`ground_join`]: a stored variable in the data is a wildcard that unifies with the join variable
/// through the trail. Inverted factors (a cyclic query has one) are re-indexed up front so the join
/// can seek them; every other factor stays zero-copy on the live map.
pub fn unify_join_zipper_partial(
    map: &PathMap<()>,
    factors: &[Factor],
    var_order: &[usize],
    nvars: usize,
) -> BTreeSet<Vec<Option<Vec<u8>>>> {
    run_unify_join(map, factors, var_order, nvars, false).out
}

/// As [`unify_join_zipper_partial`], but returns each answer as one variable-coordinated tuple
/// encoding (query variables `0..nvars` in order, sharing one intro map), so a free variable that
/// spans answer positions renders with coordinated NewVar/VarRef the way MORK's emit does.
fn unify_join_zipper_coordinated(
    map: &PathMap<()>,
    factors: &[Factor],
    var_order: &[usize],
    nvars: usize,
) -> BTreeSet<Vec<u8>> {
    run_unify_join(map, factors, var_order, nvars, true).coordinated
}

/// Build the join state and run it. When `want_coordinated`, also collect each answer as one
/// variable-coordinated tuple encoding (see [`unify_join_zipper_coordinated`]).
fn run_unify_join<'a>(
    map: &'a PathMap<()>,
    factors: &[Factor],
    var_order: &'a [usize],
    nvars: usize,
    want_coordinated: bool,
) -> UnifyJoin<'a> {
    let nf = factors.len();
    let mut var_pos = vec![0usize; nvars];
    for (pos, &v) in var_order.iter().enumerate() {
        var_pos[v] = pos;
    }

    // Re-index inverted factors so the join can seek them in var_order; a compatible factor keeps its
    // live-map prefix and pays nothing. `factor_src[f]` selects which map factor `f` reads from.
    let mut owned: Vec<Factor> = Vec::with_capacity(nf);
    let mut reindexes: Vec<PathMap<()>> = Vec::new();
    let mut factor_src: Vec<Option<usize>> = Vec::with_capacity(nf);
    for factor in factors {
        if is_inverted(factor, &var_pos) {
            let (ri, new_cols) = build_reindex(map, factor, &var_pos);
            factor_src.push(Some(reindexes.len()));
            reindexes.push(ri);
            owned.push(Factor {
                prefix: Vec::new(),
                cols: new_cols,
            });
        } else {
            factor_src.push(None);
            owned.push(factor.clone());
        }
    }

    let mut state = UnifyJoin {
        map,
        reindexes,
        factor_src,
        factors: owned,
        var_order,
        var_pos,
        nvars,
        bound: vec![Vec::new(); nf],
        next_col: vec![0; nf],
        stored_slots: vec![Vec::new(); nf],
        env: Env::new(nvars),
        out: BTreeSet::new(),
        want_coordinated,
        coordinated: BTreeSet::new(),
    };
    state.recurse(0);
    state
}

/// Parse an encoded conjunction body `(, p1 .. pk)` into factors, threading the body's variable
/// numbering (a NewVar takes the next id in first-occurrence order, a VarRef back-references one).
/// Returns the factors and the variable count.
pub fn parse_body_factors(body: &[u8]) -> Option<(Vec<Factor>, usize)> {
    let (body_len, _) = try_parse_first_subterm(body)?;
    if body_len != body.len() {
        return None;
    }
    if body.is_empty() || body[0] & TOP2 != TAG_ARITY {
        return None;
    }
    let nconj = (body[0] & LOW6) as usize;
    if nconj == 0 {
        return None;
    }
    let mut i = 1;
    i += try_parse_first_subterm(body.get(i..)?)?.0; // skip the `,` conjunction head
    let mut factors = Vec::with_capacity(nconj - 1);
    let mut nvars = 0usize;
    for _ in 0..nconj - 1 {
        let plen = try_parse_first_subterm(body.get(i..)?)?.0;
        factors.push(parse_pattern_factor(&body[i..i + plen], &mut nvars)?);
        i += plen;
    }
    if i != body.len() {
        return None;
    }
    Some((factors, nvars))
}

/// One conjunct `(rel arg..)` to a factor. The prefix is the arity byte and relation symbol only.
/// Each argument becomes a query variable, a ground subterm, or a recursive compound pattern.
fn parse_pattern_factor(pat: &[u8], nvars: &mut usize) -> Option<Factor> {
    if pat.is_empty() || pat[0] & TOP2 != TAG_ARITY {
        return None;
    }
    let mut args_left = (pat[0] & LOW6) as usize;
    if args_left == 0 {
        return None;
    }
    args_left -= 1;
    let mut j = 1 + try_parse_first_subterm(pat.get(1..)?)?.0; // past the arity byte and the head symbol
    let prefix = pat[0..j].to_vec();
    let mut cols = Vec::with_capacity(args_left);
    while args_left > 0 {
        let len = try_parse_first_subterm(pat.get(j..)?)?.0;
        cols.push(parse_pattern_col(&pat[j..j + len], nvars)?);
        j += len;
        args_left -= 1;
    }
    if j != pat.len() {
        return None;
    }
    Some(Factor { prefix, cols })
}

fn parse_pattern_col(bytes: &[u8], nvars: &mut usize) -> Option<Col> {
    if bytes.is_empty() {
        return None;
    }
    let b = bytes[0];
    if bytes.len() == 1 && (0x80..=0xC0).contains(&b) {
        return Some(Col::Var(if b == NEWVAR_BYTE {
            let id = *nvars;
            *nvars += 1;
            id
        } else {
            let id = (b & LOW6) as usize;
            if id >= *nvars {
                return None;
            }
            id
        }));
    }
    if first_subterm_is_ground(bytes) {
        return Some(Col::Ground(bytes.to_vec()));
    }
    if b & TOP2 != TAG_ARITY {
        return None;
    }
    let arity = (b & LOW6) as usize;
    let mut children = Vec::with_capacity(arity);
    let mut pos = 1usize;
    for _ in 0..arity {
        let len = try_parse_first_subterm(bytes.get(pos..)?)?.0;
        children.push(parse_pattern_col(&bytes[pos..pos + len], nvars)?);
        pos += len;
    }
    if pos != bytes.len() {
        return None;
    }
    Some(Col::Compound(Box::new(CompoundPat { arity: b, children })))
}

/// Live-route entry: parse the conjunction body into factors and run the join on the live map.
/// Variables bind in first-occurrence order, the order the emit numbers the answer components in.
/// None if the body is not a relation-prefix conjunction.
pub fn unify_join_zipper_body(map: &PathMap<()>, body: &[u8]) -> Option<BTreeSet<Vec<Vec<u8>>>> {
    let (factors, nvars) = parse_body_factors(body)?;
    let var_order: Vec<usize> = (0..nvars).collect();
    Some(unify_join_zipper(map, &factors, &var_order, nvars))
}

/// Returns true when `body` is inside the ProductZipper-identical zipper-join fragment.
///
/// The predicate is self-contained: it uses only the encoded body, the live `PathMap`, and the
/// same factor parser as the join. It declines any shape whose byte output is owned by the
/// ProductZipper compatibility path.
pub fn unify_join_zipper_body_routable(map: &PathMap<()>, body: &[u8]) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        let Some((factors, _)) = parse_body_factors(body) else {
            return false;
        };
        body_factors_routable_to_zipper_join(map, &factors)
    }))
    .unwrap_or(false)
}

/// Parse, route-check, and run the zipper join for bodies whose all-variable answer rows can be
/// represented exactly as ground bytes. Bodies that route but produce any free or schematic
/// answer component return `None`; callers that can render partial rows should use
/// [`unify_join_zipper_body_partial_safe`].
pub fn unify_join_zipper_body_safe(
    map: &PathMap<()>,
    body: &[u8],
) -> Option<BTreeSet<Vec<Vec<u8>>>> {
    let (_, rows) = unify_join_zipper_body_partial_safe(map, body)?;
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|component| component.filter(|bytes| first_subterm_is_ground(bytes)))
                .collect::<Option<Vec<Vec<u8>>>>()
        })
        .collect()
}

/// Parse, route-check, and run the zipper join, preserving free or schematic answer components for
/// the live template renderer. `Some` means the routing decision is self-contained and sound;
/// `None` keeps the caller on the ProductZipper path.
pub fn unify_join_zipper_body_partial_safe(
    map: &PathMap<()>,
    body: &[u8],
) -> Option<(usize, BTreeSet<Vec<Option<Vec<u8>>>>)> {
    catch_unwind(AssertUnwindSafe(|| {
        let (factors, nvars) = parse_body_factors(body)?;
        if !body_factors_routable_to_zipper_join(map, &factors) {
            return None;
        }
        let var_order: Vec<usize> = (0..nvars).collect();
        Some((
            nvars,
            unify_join_zipper_partial(map, &factors, &var_order, nvars),
        ))
    }))
    .ok()
    .flatten()
}

/// Parse, route-check, and run the zipper join, returning each answer as one variable-coordinated
/// tuple encoding. Unlike [`unify_join_zipper_body_safe`], a free-variable answer is kept: a
/// variable shared across answer positions emits as one coordinated variable, so the caller can
/// render and compare free-variable answers up to consistent renaming. `None` keeps the caller on
/// the ProductZipper path.
pub fn unify_join_zipper_body_rows_rendered(
    map: &PathMap<()>,
    body: &[u8],
) -> Option<BTreeSet<Vec<u8>>> {
    catch_unwind(AssertUnwindSafe(|| {
        let (factors, nvars) = parse_body_factors(body)?;
        if !body_factors_routable_to_zipper_join(map, &factors) {
            return None;
        }
        let var_order: Vec<usize> = (0..nvars).collect();
        Some(unify_join_zipper_coordinated(map, &factors, &var_order, nvars))
    }))
    .ok()
    .flatten()
}

fn body_factors_routable_to_zipper_join(map: &PathMap<()>, factors: &[Factor]) -> bool {
    for factor in factors {
        let compound_count = factor
            .cols
            .iter()
            .filter(|col| matches!(col, Col::Compound(_)))
            .count();
        if compound_count > 1 {
            return false;
        }
        if factor.prefix.is_empty() {
            return false;
        }
    }

    for factor in factors {
        let mut rz = map.read_zipper_at_path(&factor.prefix);
        while rz.to_next_val() {
            let fact = rz.origin_path();
            if !fact_routable_for_factor(factor, fact, factors.len()) {
                return false;
            }
        }
    }
    true
}

fn fact_routable_for_factor(factor: &Factor, fact: &[u8], factor_count: usize) -> bool {
    let Some(rest) = fact.get(factor.prefix.len()..) else {
        return false;
    };
    let Some(cols) = try_split_columns(rest, factor.cols.len()) else {
        return false;
    };
    for (query_col, fact_col) in factor.cols.iter().zip(cols) {
        if subterm_is_nonground_compound(fact_col) && !ground_query_column_can_absorb(query_col) {
            return false;
        }
    }
    if factor_count > 1
        && factor
            .cols
            .iter()
            .any(|col| matches!(col, Col::Compound(_)))
    {
        let Some(has_varref) = expr_has_varref(fact) else {
            return false;
        };
        if has_varref {
            return false;
        }
    }
    true
}

fn try_split_columns(bytes: &[u8], ncols: usize) -> Option<Vec<&[u8]>> {
    let mut cols = Vec::with_capacity(ncols);
    let mut pos = 0usize;
    for _ in 0..ncols {
        let len = try_parse_first_subterm(bytes.get(pos..)?)?.0;
        cols.push(bytes.get(pos..pos + len)?);
        pos += len;
    }
    (pos == bytes.len()).then_some(cols)
}

fn subterm_is_nonground_compound(bytes: &[u8]) -> bool {
    bytes.first().is_some_and(|b| {
        b & TOP2 == TAG_ARITY && !try_parse_first_subterm(bytes).is_some_and(|(_, ground)| ground)
    })
}

fn ground_query_column_can_absorb(col: &Col) -> bool {
    match col {
        Col::Ground(_) | Col::Compound(_) => true,
        Col::Var(_) => false,
    }
}

fn expr_has_varref(bytes: &[u8]) -> Option<bool> {
    let (len, _) = try_parse_first_subterm(bytes)?;
    if len != bytes.len() {
        return None;
    }
    let mut i = 0usize;
    let mut remaining = 1usize;
    while remaining > 0 {
        let b = *bytes.get(i)?;
        i += 1;
        remaining -= 1;
        match b & TOP2 {
            TAG_ARITY => remaining += (b & LOW6) as usize,
            TAG_VARREF => return Some(true),
            _ => {
                if b != NEWVAR_BYTE {
                    let payload = (b & LOW6) as usize;
                    i = i.checked_add(payload)?;
                    if i > bytes.len() {
                        return None;
                    }
                }
            }
        }
    }
    (i == bytes.len()).then_some(false)
}

struct UnifyJoin<'a> {
    map: &'a PathMap<()>,
    /// Re-indexed copies of inverted factors; `factor_src[f] = Some(i)` reads `reindexes[i]`.
    reindexes: Vec<PathMap<()>>,
    factor_src: Vec<Option<usize>>,
    /// Owned because a re-indexed factor's prefix and columns differ from the input factor's.
    factors: Vec<Factor>,
    var_order: &'a [usize],
    /// `var_pos[v]` = position of global variable `v` in `var_order`, for the catch-up test.
    var_pos: Vec<usize>,
    nvars: usize,
    bound: Vec<Vec<u8>>,
    next_col: Vec<usize>,
    stored_slots: Vec<Vec<usize>>,
    env: Env,
    /// Answer rows, one `Option` per query variable: `Some(bytes)` for a resolved term, `None` for
    /// a still-free variable. The all-ground entry filters to fully-ground rows.
    out: BTreeSet<Vec<Option<Vec<u8>>>>,
    /// When set, also collect each answer as one variable-coordinated tuple encoding in `coordinated`.
    want_coordinated: bool,
    /// Answer tuples encoded through one shared intro map, so free-variable coreference across answer
    /// positions survives for the live renderer. Empty unless `want_coordinated`.
    coordinated: BTreeSet<Vec<u8>>,
}

impl UnifyJoin<'_> {
    fn recurse(&mut self, i: usize) {
        self.catch_up(i, 0);
    }

    fn recurse_after_catch_up(&mut self, i: usize) {
        if i == self.var_order.len() {
            if !(0..self.factors.len()).all(|f| self.factor_has_value(f)) {
                return;
            }
            // Keep every component that resolved to a term, ground or schematic. A variable that is
            // still free is None; the live emit can then render schematic compounds and leave free
            // variables fresh, matching the ProductZipper's byte output.
            let row: Vec<Option<Vec<u8>>> =
                (0..self.nvars).map(|v| self.env.term_bytes_of(v)).collect();
            self.out.insert(row);
            if self.want_coordinated {
                self.coordinated
                    .insert(self.env.encode_tuple_coordinated(self.nvars));
            }
            return;
        }
        let v = self.var_order[i];
        let mut parts: Vec<usize> = (0..self.factors.len())
            .filter(|&f| {
                let nc = self.next_col[f];
                matches!(self.factors[f].cols.get(nc), Some(Col::Var(cv)) if *cv == v)
            })
            .collect();
        if parts.is_empty() {
            self.recurse(i + 1);
            return;
        }
        if !matches!(self.env.resolve_term(&Col::Var(v)), Col::Var(_)) {
            self.consume_bound_var_parts(&parts, 0, v, i);
            return;
        }
        // The leapfrog principle: lead with the smallest domain so the leading factor enumerates
        // few candidates and the rest seek. A bounded subterm count under each factor's current
        // position is the estimate. This is what makes a selective factor, say (e a $y) with a few
        // edges, drive the join instead of the whole relation.
        parts.sort_by_key(|&f| self.domain_estimate(f));
        self.intersect_unify(&parts, 0, v, i);
    }

    /// Domain-size estimate for lead selection, bounded so it is independent of the space size.
    /// Count the distinct subterm values under the factor's current position, but stop at a small
    /// cap. The leapfrog only needs to know which factor has the fewest candidates, not the exact
    /// count, so a bounded count suffices and stays O(cap). A full `val_count` is O(subtree), which
    /// would make a selective join's cost climb with the whole relation rather than the answer.
    /// The map factor `f` reads from: its re-indexed copy if it was inverted, else the live map.
    fn src_map(&self, f: usize) -> &PathMap<()> {
        match self.factor_src[f] {
            Some(ri) => &self.reindexes[ri],
            None => self.map,
        }
    }

    fn domain_estimate(&self, f: usize) -> usize {
        const CAP: usize = 32;
        let mut path = self.factors[f].prefix.clone();
        path.extend_from_slice(&self.bound[f]);
        let mut cur = SubtermCursor::new(self.src_map(f).read_zipper_at_path(&path));
        cur.first();
        let mut n = 0;
        while !cur.at_end() && n < CAP {
            n += 1;
            cur.next();
        }
        n
    }

    fn factor_path(&self, f: usize) -> Vec<u8> {
        let mut path = self.factors[f].prefix.clone();
        path.extend_from_slice(&self.bound[f]);
        path
    }

    fn factor_has_value(&self, f: usize) -> bool {
        if self.next_col[f] != self.factors[f].cols.len() {
            return false;
        }
        let path = self.factor_path(f);
        self.src_map(f).read_zipper_at_path(&path).val().is_some()
    }

    /// The candidates at factor `f`'s current column when a query variable is still free.
    fn open_free_candidates(&self, f: usize) -> Vec<Cand> {
        let path = self.factor_path(f);
        let mut cur = SubtermCursor::new(self.src_map(f).read_zipper_at_path(&path));
        free_candidates(&mut cur)
    }

    /// The candidates at factor `f`'s current ground column that can unify with the fixed query
    /// value. A data-side wildcard captures that ground value and keeps its stored slot for later
    /// VarRef columns in the same fact.
    fn open_ground_candidates(&self, f: usize, ground: &[u8]) -> Vec<Cand> {
        let path = self.factor_path(f);
        let mut cur = SubtermCursor::new(self.src_map(f).read_zipper_at_path(&path));
        ground_candidates(&mut cur, ground)
    }

    fn child_bytes_at_current(&self, f: usize) -> Vec<u8> {
        let path = self.factor_path(f);
        self.src_map(f)
            .read_zipper_at_path(&path)
            .child_mask()
            .iter()
            .collect()
    }

    fn wildcard_children_at_current(&self, f: usize) -> Vec<u8> {
        self.child_bytes_at_current(f)
            .into_iter()
            .filter(|&b| (0x80..=0xC0).contains(&b))
            .collect()
    }

    fn child_exists_at_current(&self, f: usize, b: u8) -> bool {
        let path = self.factor_path(f);
        has_bit(&self.src_map(f).read_zipper_at_path(&path).child_mask(), b)
    }

    /// Unify `qvar` with a candidate child of factor `f`, returning whether it held and the trie
    /// bytes to descend. A NewVar wildcard allocates a fresh stored variable (pushed to `f`'s slots
    /// so a later VarRef in the same fact corefers); a VarRef reads the slot it introduced.
    fn apply_cand(&mut self, qvar: usize, cand: &Cand, f: usize) -> (bool, Vec<u8>) {
        match cand {
            Cand::Ground(g) => {
                let term = self.data_term_from_bytes(f, g);
                (self.env.unify_var_term(qvar, &term), g.clone())
            }
            Cand::Wild(w) => {
                let id = self.stored_var_for_wildcard(f, *w);
                (self.env.unify_var_var(qvar, id), vec![*w])
            }
        }
    }

    fn stored_var_for_wildcard(&mut self, f: usize, w: u8) -> usize {
        if w == 0xC0 {
            let id = self.env.fresh();
            self.stored_slots[f].push(id);
            id
        } else {
            self.stored_slots[f][(w & 0x3F) as usize]
        }
    }

    fn apply_ground_cand(&mut self, ground: &[u8], cand: &Cand, f: usize) -> (bool, Vec<u8>) {
        match cand {
            Cand::Ground(g) => (g == ground, g.clone()),
            Cand::Wild(w) => {
                let id = self.stored_var_for_wildcard(f, *w);
                (self.env.unify_var_ground(id, ground), vec![*w])
            }
        }
    }

    fn data_term_from_bytes(&mut self, f: usize, bytes: &[u8]) -> Col {
        let mut pos = 0usize;
        let term = self.parse_data_term_at(f, bytes, &mut pos);
        debug_assert_eq!(pos, bytes.len());
        term
    }

    fn parse_data_term_at(&mut self, f: usize, bytes: &[u8], pos: &mut usize) -> Col {
        let start = *pos;
        let b = bytes[*pos];
        *pos += 1;
        match b & TOP2 {
            TAG_ARITY => {
                let arity = (b & LOW6) as usize;
                let mut children = Vec::with_capacity(arity);
                for _ in 0..arity {
                    children.push(self.parse_data_term_at(f, bytes, pos));
                }
                if children.iter().all(|child| matches!(child, Col::Ground(_))) {
                    Col::Ground(bytes[start..*pos].to_vec())
                } else {
                    Col::Compound(Box::new(CompoundPat { arity: b, children }))
                }
            }
            TAG_VARREF => Col::Var(self.stored_var_for_wildcard(f, b)),
            _ if b == NEWVAR_BYTE => Col::Var(self.stored_var_for_wildcard(f, b)),
            _ => {
                let len = (b & LOW6) as usize;
                *pos += len;
                Col::Ground(bytes[start..*pos].to_vec())
            }
        }
    }

    fn intersect_unify(&mut self, parts: &[usize], pi: usize, v: usize, i: usize) {
        if pi == parts.len() {
            self.recurse(i + 1);
            return;
        }
        // The lead factor (pi == 0, `v` still free) enumerates its small domain and binds `v`; every
        // later factor SEEKS the now-bound `v` instead of enumerating its own relation. `consume_col`
        // resolves `v` and does exactly that: it enumerates while `v` is free and seeks once it is
        // bound (a data-side wildcard still captures the value). The seek is what keeps a k-factor
        // join O(answer) rather than O(relation^k); enumerating every factor made the triangle O(s^2).
        let f = parts[pi];
        self.consume_col(f, Col::Var(v), &mut |this| {
            this.intersect_unify(parts, pi + 1, v, i);
        });
    }

    fn consume_bound_var_parts(&mut self, parts: &[usize], pi: usize, v: usize, i: usize) {
        if pi == parts.len() {
            self.recurse(i + 1);
            return;
        }
        let f = parts[pi];
        self.consume_col(f, Col::Var(v), &mut |this| {
            this.consume_bound_var_parts(parts, pi + 1, v, i);
        });
    }

    fn consume_col(&mut self, f: usize, col: Col, cont: &mut dyn FnMut(&mut Self)) {
        self.match_col_at_current(f, col, &mut |this| {
            this.next_col[f] += 1;
            cont(this);
            this.next_col[f] -= 1;
        });
    }

    fn with_bound_bytes(&mut self, f: usize, bytes: &[u8], cont: &mut dyn FnMut(&mut Self)) {
        let len = self.bound[f].len();
        self.bound[f].extend_from_slice(bytes);
        cont(self);
        self.bound[f].truncate(len);
    }

    fn match_col_at_current(&mut self, f: usize, col: Col, cont: &mut dyn FnMut(&mut Self)) {
        match self.env.resolve_term(&col).structural_ground() {
            Col::Var(qvar) => {
                let cands = self.open_free_candidates(f);
                for cand in cands {
                    let mark = self.env.mark();
                    let slots_len = self.stored_slots[f].len();
                    let (ok, bytes) = self.apply_cand(qvar, &cand, f);
                    if ok {
                        self.with_bound_bytes(f, &bytes, cont);
                    }
                    self.stored_slots[f].truncate(slots_len);
                    self.env.rollback(mark);
                }
            }
            Col::Ground(g) => self.match_ground_at_current(f, &g, cont),
            Col::Compound(compound) => self.match_compound_at_current(f, *compound, cont),
        }
    }

    fn match_ground_at_current(
        &mut self,
        f: usize,
        ground: &[u8],
        cont: &mut dyn FnMut(&mut Self),
    ) {
        if !ground.is_empty() && ground[0] & TOP2 == TAG_ARITY {
            self.match_compound_at_current(f, compound_from_ground_bytes(ground), cont);
            return;
        }
        let cands = self.open_ground_candidates(f, ground);
        for cand in cands {
            let mark = self.env.mark();
            let slots_len = self.stored_slots[f].len();
            let (ok, bytes) = self.apply_ground_cand(ground, &cand, f);
            if ok {
                self.with_bound_bytes(f, &bytes, cont);
            }
            self.stored_slots[f].truncate(slots_len);
            self.env.rollback(mark);
        }
    }

    fn match_compound_at_current(
        &mut self,
        f: usize,
        compound: CompoundPat,
        cont: &mut dyn FnMut(&mut Self),
    ) {
        let capture_term = Col::Compound(Box::new(compound.clone()));
        for w in self.wildcard_children_at_current(f) {
            let mark = self.env.mark();
            let slots_len = self.stored_slots[f].len();
            let id = self.stored_var_for_wildcard(f, w);
            if self.env.unify_var_term(id, &capture_term) {
                self.with_bound_bytes(f, &[w], cont);
            }
            self.stored_slots[f].truncate(slots_len);
            self.env.rollback(mark);
        }
        if self.child_exists_at_current(f, compound.arity) {
            self.with_bound_bytes(f, &[compound.arity], &mut |this| {
                this.match_compound_children(f, &compound.children, 0, cont);
            });
        }
    }

    fn match_compound_children(
        &mut self,
        f: usize,
        children: &[Col],
        idx: usize,
        cont: &mut dyn FnMut(&mut Self),
    ) {
        if idx == children.len() {
            cont(self);
            return;
        }
        let child = children[idx].clone();
        self.match_col_at_current(f, child, &mut |this| {
            this.match_compound_children(f, children, idx + 1, cont);
        });
    }

    /// Before each scheduled variable, consume every column whose value is already known: ground
    /// query arguments, compound arguments, and repeated or inverted variables already bound by
    /// earlier levels. Columns can branch because a stored data variable may capture the fixed query
    /// value or compound.
    fn catch_up(&mut self, i: usize, f: usize) {
        if f == self.factors.len() {
            self.recurse_after_catch_up(i);
            return;
        }
        let Some(col) = self.factors[f].cols.get(self.next_col[f]).cloned() else {
            self.catch_up(i, f + 1);
            return;
        };
        match col {
            Col::Var(vp) if self.var_pos[vp] < i => {
                self.consume_col(f, Col::Var(vp), &mut |this| this.catch_up(i, f));
            }
            Col::Var(_) => {
                self.catch_up(i, f + 1);
            }
            other => {
                self.consume_col(f, other, &mut |this| this.catch_up(i, f));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_of(bytes: &[u8]) -> ByteMask {
        let mut m = [0u64; 4];
        for &b in bytes {
            m[(b >> 6) as usize] |= 1u64 << (b & 63);
        }
        ByteMask(m)
    }

    fn sym(s: &str) -> Vec<u8> {
        let mut v = vec![TAG_SYMSIZE | s.len() as u8];
        v.extend_from_slice(s.as_bytes());
        v
    }

    /// `(rel a0 a1 ...)` encoded: Arity(1+n), Sym(rel), then each arg's bytes.
    fn nest(rel: &str, args: &[Vec<u8>]) -> Vec<u8> {
        let mut v = vec![TAG_ARITY | (1 + args.len()) as u8];
        v.extend(sym(rel));
        for a in args {
            v.extend_from_slice(a);
        }
        v
    }

    fn conj(factors: &[Vec<u8>]) -> Vec<u8> {
        nest(",", factors)
    }

    fn new_var() -> Vec<u8> {
        vec![NEWVAR_BYTE]
    }

    fn var_ref(idx: u8) -> Vec<u8> {
        vec![TAG_VARREF | idx]
    }

    /// The stored-path prefix for a relation of the given total arity (head + args).
    fn relation_prefix(rel: &str, total_arity: usize) -> Vec<u8> {
        let mut v = vec![TAG_ARITY | total_arity as u8];
        v.extend(sym(rel));
        v
    }

    #[test]
    fn safe_body_routes_flat_ground_answers() {
        let mut map = PathMap::<()>::new();
        map.insert(&nest("e", &[sym("a"), sym("b")]), ());
        map.insert(&nest("e", &[sym("b"), sym("c")]), ());
        let body = conj(&[
            nest("e", &[new_var(), new_var()]),
            nest("e", &[var_ref(1), new_var()]),
        ]);

        let rows = unify_join_zipper_body_safe(&map, &body).expect("flat body routes");
        let expected = BTreeSet::from([vec![sym("a"), sym("b"), sym("c")]]);
        assert_eq!(rows, expected);
    }

    #[test]
    fn safe_body_partial_routes_when_all_ground_entry_cannot_represent_rows() {
        let mut map = PathMap::<()>::new();
        map.insert(&nest("r", &[new_var()]), ());
        map.insert(&nest("r", &[nest("a", &[sym("v0")])]), ());
        let body = conj(&[nest("r", &[nest("a", &[new_var()])])]);

        assert!(unify_join_zipper_body_routable(&map, &body));
        let (_nvars, rows) =
            unify_join_zipper_body_partial_safe(&map, &body).expect("partial route is safe");
        assert!(
            rows.iter()
                .any(|row| row.iter().any(|component| component.is_none())),
            "the live renderer must preserve the free non-ground row"
        );
        assert!(
            unify_join_zipper_body_safe(&map, &body).is_none(),
            "the all-ground entry must not silently drop non-ground rows"
        );
    }

    #[test]
    fn coordinated_rows_preserve_free_var_coreference() {
        // A schematic fact (e $u $u) couples the two query variables: matching (e $x $y) binds both
        // to one free variable. The coordinated tuple must share it (NewVar then VarRef(0)), the way
        // MORK's emit numbers a coreferent answer.
        let mut coref = PathMap::<()>::new();
        coref.insert(&nest("e", &[new_var(), var_ref(0)]), ()); // (e $u $u)
        let body = conj(&[nest("e", &[new_var(), new_var()])]); // (e $x $y)
        let rows = unify_join_zipper_body_rows_rendered(&coref, &body).expect("flat body routes");
        assert_eq!(rows.len(), 1, "one answer: $x and $y are the same free variable");
        assert_eq!(
            rows.iter().next().unwrap(),
            &vec![NEWVAR_BYTE, TAG_VARREF | 0],
            "coreferent free variables must coordinate to NewVar, VarRef(0)"
        );

        // Two independent data variables stay distinct: two NewVars, no back-reference.
        let mut indep = PathMap::<()>::new();
        indep.insert(&nest("e", &[new_var(), new_var()]), ()); // (e $u $w)
        let indep_rows =
            unify_join_zipper_body_rows_rendered(&indep, &body).expect("flat body routes");
        assert_eq!(
            indep_rows.iter().next().unwrap(),
            &vec![NEWVAR_BYTE, NEWVAR_BYTE],
            "independent free variables must stay distinct NewVars"
        );
    }

    #[test]
    fn safe_body_routes_goal2_boundary_and_declines_propagated_capture() {
        let mut occurs = PathMap::<()>::new();
        occurs.insert(&nest("e", &[new_var(), var_ref(0)]), ());
        occurs.insert(&nest("e", &[sym("v0"), nest("f", &[sym("v1")])]), ());
        let occurs_body = conj(&[nest("e", &[new_var(), nest("f", &[var_ref(0)])])]);

        let mut ground_query = PathMap::<()>::new();
        ground_query.insert(&nest("r", &[nest("a", &[new_var()]), sym("v0")]), ());
        ground_query.insert(&nest("s", &[sym("v0"), sym("v1")]), ());
        ground_query.insert(&nest("t", &[sym("v1"), sym("b")]), ());
        let ground_query_body = conj(&[
            nest("r", &[nest("a", &[sym("b")]), new_var()]),
            nest("s", &[var_ref(0), new_var()]),
            nest("t", &[var_ref(1), sym("b")]),
        ]);

        let mut propagated = PathMap::<()>::new();
        propagated.insert(&nest("e", &[nest("k", &[new_var()]), sym("v0")]), ());
        propagated.insert(&nest("e", &[new_var(), var_ref(0)]), ());
        propagated.insert(&nest("h", &[new_var(), var_ref(0)]), ());
        let propagated_body = conj(&[
            nest("e", &[nest("k", &[new_var()]), new_var()]),
            nest("e", &[nest("k", &[var_ref(1)]), new_var()]),
            nest("h", &[var_ref(2), var_ref(0)]),
        ]);

        for (name, map, body) in [
            ("acyclic-occurs", &occurs, &occurs_body),
            (
                "fact-schematic-compound-under-ground-query",
                &ground_query,
                &ground_query_body,
            ),
        ] {
            assert!(
                unify_join_zipper_body_routable(map, body),
                "{name} must be inside the zipper-owned safe route"
            );
            assert!(
                unify_join_zipper_body_partial_safe(map, body).is_some(),
                "{name} must route safely"
            );
        }

        for (name, map, body) in [(
            "join-propagated-compound-capture",
            &propagated,
            &propagated_body,
        )] {
            assert!(
                !unify_join_zipper_body_routable(map, body),
                "{name} must stay on the ProductZipper boundary"
            );
            assert!(
                unify_join_zipper_body_partial_safe(map, body).is_none(),
                "{name} must decline safely"
            );
        }
    }

    #[test]
    fn subterm_cursor_enumerates_and_seeks_arg1() {
        // First arguments of various shapes: a compound (sorts first, tag 0x02 < symbol tag 0xC1),
        // several one-byte-length symbols, and a two-byte-length one (sorts last, 0xC2 > 0xC1).
        let a_terms: Vec<Vec<u8>> = vec![
            sym("a"),
            sym("b"),
            sym("c"),
            sym("z"),
            sym("bb"),
            nest("k", &[sym("v")]),
        ];
        // Each arg1 appears in two facts (distinct arg2) to exercise trie merging / distinctness.
        let mut facts = Vec::new();
        for (i, a) in a_terms.iter().enumerate() {
            facts.push(nest("e", &[a.clone(), sym(&format!("p{i}"))]));
            facts.push(nest("e", &[a.clone(), sym(&format!("q{i}"))]));
        }
        // A different relation under the same map, to confirm the prefix scopes the cursor.
        facts.push(nest("h", &[sym("a"), sym("a")]));

        let mut map = PathMap::<()>::new();
        for f in &facts {
            map.insert(f, ());
        }
        let pfx = relation_prefix("e", 3);

        // Oracle: distinct arg1 subterms in byte-lex order.
        let mut want: Vec<Vec<u8>> = a_terms.clone();
        want.sort();
        want.dedup();

        let mut cur = SubtermCursor::new(map.read_zipper_at_path(&pfx));
        cur.first();
        let mut got = Vec::new();
        while let Some(k) = cur.key() {
            got.push(k.to_vec());
            cur.next();
        }
        assert_eq!(
            got, want,
            "enumeration must be the distinct arg1 subterms in lex order"
        );

        // seek to each oracle value and to a few off-key targets; compare to least >= target.
        let mut targets = want.clone();
        targets.push(nest("k", &[sym("a")])); // a compound just below (k v)
        targets.push(sym("ba")); // between b and bb in byte order? [0xC2,'b','a'] vs [0xC2,'b','b']
        for target in &targets {
            cur.seek(target);
            let expect = want
                .iter()
                .find(|w| w.as_slice() >= target.as_slice())
                .cloned();
            assert_eq!(
                cur.key().map(<[u8]>::to_vec),
                expect,
                "seek({target:?}) must land on the least subterm >= target"
            );
        }

        // seek past every subterm -> exhausted.
        cur.seek(&sym("zz"));
        assert!(
            cur.at_end(),
            "seek past the maximum must exhaust the cursor"
        );
    }

    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493))
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// A random variable-width term over a two-byte symbol alphabet, so symbols share prefixes and
    /// force multi-level backtracking in seek; with nested compounds when depth allows.
    fn rand_term(rng: &mut Lcg, depth: usize) -> Vec<u8> {
        const ALPHA: &[u8] = b"ab";
        if depth == 0 || rng.below(3) != 0 {
            let len = 1 + rng.below(3);
            let mut v = vec![TAG_SYMSIZE | len as u8];
            for _ in 0..len {
                v.push(ALPHA[rng.below(ALPHA.len())]);
            }
            v
        } else {
            let n = 1 + rng.below(2);
            let mut v = vec![TAG_ARITY | (1 + n) as u8];
            v.extend(sym("f"));
            for _ in 0..n {
                v.extend(rand_term(rng, depth - 1));
            }
            v
        }
    }

    #[test]
    fn subterm_cursor_property_vs_brute_force() {
        for seed in 0..300u64 {
            let mut rng = Lcg::new(seed.wrapping_add(1));
            let n = 1 + rng.below(12);
            let a_terms: Vec<Vec<u8>> = (0..n).map(|_| rand_term(&mut rng, 2)).collect();

            let mut map = PathMap::<()>::new();
            for (i, a) in a_terms.iter().enumerate() {
                map.insert(&nest("e", &[a.clone(), sym(&format!("z{}", i % 3))]), ());
            }
            let pfx = relation_prefix("e", 3);

            let mut want: Vec<Vec<u8>> = a_terms.clone();
            want.sort();
            want.dedup();

            let mut cur = SubtermCursor::new(map.read_zipper_at_path(&pfx));
            cur.first();
            let mut got = Vec::new();
            while let Some(k) = cur.key() {
                got.push(k.to_vec());
                cur.next();
            }
            assert_eq!(got, want, "seed {seed}: enumeration");

            let mut targets = want.clone();
            for _ in 0..12 {
                targets.push(rand_term(&mut rng, 2));
            }
            for target in &targets {
                cur.seek(target);
                let expect = want
                    .iter()
                    .find(|w| w.as_slice() >= target.as_slice())
                    .cloned();
                assert_eq!(
                    cur.key().map(<[u8]>::to_vec),
                    expect,
                    "seed {seed}: seek({target:?})"
                );
            }
        }
    }

    /// Reference join: nested loop over one matching fact per factor, binding shared variables and
    /// rejecting on conflict. `factor_rows[f]` is the column-subterm list of factor f's facts.
    fn brute_rec(
        f: usize,
        factors: &[Factor],
        factor_rows: &[Vec<Vec<Vec<u8>>>],
        binding: &mut Vec<Option<Vec<u8>>>,
        out: &mut Vec<Vec<Vec<u8>>>,
    ) {
        if f == factors.len() {
            out.push(binding.iter().map(|b| b.clone().unwrap()).collect());
            return;
        }
        for row in &factor_rows[f] {
            let mut undo: Vec<usize> = Vec::new();
            let mut ok = true;
            for (ci, col) in factors[f].cols.iter().enumerate() {
                match col {
                    Col::Ground(g) => {
                        if g != &row[ci] {
                            ok = false;
                            break;
                        }
                    }
                    Col::Var(v) => {
                        if let Some(existing) = &binding[*v] {
                            if existing != &row[ci] {
                                ok = false;
                                break;
                            }
                        } else {
                            binding[*v] = Some(row[ci].clone());
                            undo.push(*v);
                        }
                    }
                    Col::Compound(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                brute_rec(f + 1, factors, factor_rows, binding, out);
            }
            for v in undo.into_iter().rev() {
                binding[v] = None;
            }
        }
    }

    #[test]
    fn ground_join_matches_brute_force() {
        for seed in 0..150u64 {
            let mut rng = Lcg::new(seed.wrapping_add(7));
            let nnodes = 3 + rng.below(4);
            let nodes: Vec<Vec<u8>> = (0..nnodes)
                .map(|i| sym(&((b'a' + i as u8) as char).to_string()))
                .collect();

            let mut map = PathMap::<()>::new();
            let mut e_facts: Vec<Vec<Vec<u8>>> = Vec::new();
            let mut f_facts: Vec<Vec<Vec<u8>>> = Vec::new();
            let nedges = 4 + rng.below(8);
            for _ in 0..nedges {
                let a = nodes[rng.below(nnodes)].clone();
                let b = nodes[rng.below(nnodes)].clone();
                if map
                    .insert(&nest("e", &[a.clone(), b.clone()]), ())
                    .is_none()
                {
                    e_facts.push(vec![a, b]);
                }
                let c = nodes[rng.below(nnodes)].clone();
                let d = nodes[rng.below(nnodes)].clone();
                if map
                    .insert(&nest("f", &[c.clone(), d.clone()]), ())
                    .is_none()
                {
                    f_facts.push(vec![c, d]);
                }
            }
            let pe = relation_prefix("e", 3);
            let pf = relation_prefix("f", 3);

            let queries: Vec<(Vec<Factor>, Vec<usize>, usize)> = vec![
                // path  (e $0 $1)(e $1 $2)
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![1, 2]),
                    ],
                    vec![0, 1, 2],
                    3,
                ),
                // star  (e $0 $1)(e $0 $2)
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![0, 2]),
                    ],
                    vec![0, 1, 2],
                    3,
                ),
                // two-relation path  (e $0 $1)(f $1 $2)
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pf.clone(), vec![1, 2]),
                    ],
                    vec![0, 1, 2],
                    3,
                ),
                // three-path  (e $0 $1)(e $1 $2)(e $2 $3)
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![1, 2]),
                        Factor::var_cols(pe.clone(), vec![2, 3]),
                    ],
                    vec![0, 1, 2, 3],
                    4,
                ),
                // triangle  (e $0 $1)(e $1 $2)(e $2 $0) -- cyclic: factor 2's columns invert the
                // variable order, so it must validate $0 after binding $2 (the catch-up path).
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![1, 2]),
                        Factor::var_cols(pe.clone(), vec![2, 0]),
                    ],
                    vec![0, 1, 2],
                    3,
                ),
                // same triangle under a rotated variable order (different participation pattern).
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![1, 2]),
                        Factor::var_cols(pe.clone(), vec![2, 0]),
                    ],
                    vec![1, 2, 0],
                    3,
                ),
                // four-cycle  (e $0 $1)(e $1 $2)(e $2 $3)(e $3 $0)
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![1, 2]),
                        Factor::var_cols(pe.clone(), vec![2, 3]),
                        Factor::var_cols(pe.clone(), vec![3, 0]),
                    ],
                    vec![0, 1, 2, 3],
                    4,
                ),
                // intra-factor coreference  (e $0 $0): only the self-loops, via catch-up on col 1.
                (vec![Factor::var_cols(pe.clone(), vec![0, 0])], vec![0], 1),
                // triangle with a pendant  (e $0 $1)(e $1 $2)(e $2 $0)(f $0 $3)
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![1, 2]),
                        Factor::var_cols(pe.clone(), vec![2, 0]),
                        Factor::var_cols(pf.clone(), vec![0, 3]),
                    ],
                    vec![0, 1, 2, 3],
                    4,
                ),
            ];

            for (factors, order, nvars) in &queries {
                let factor_rows: Vec<Vec<Vec<Vec<u8>>>> = factors
                    .iter()
                    .map(|fac| {
                        if fac.prefix == pe {
                            e_facts.clone()
                        } else {
                            f_facts.clone()
                        }
                    })
                    .collect();

                let mut got = ground_join(&map, factors, order, *nvars);
                let mut want = {
                    let mut binding = vec![None; *nvars];
                    let mut out = Vec::new();
                    brute_rec(0, factors, &factor_rows, &mut binding, &mut out);
                    out
                };
                got.sort();
                got.dedup();
                want.sort();
                want.dedup();
                assert_eq!(
                    got, want,
                    "seed {seed}: join answers must match the nested loop"
                );
            }
        }
    }

    /// A structured fact column for the schematic differential: a ground value, or a stored variable
    /// identified by a fact-local slot (so the same slot twice in one fact is coreferent).
    #[derive(Clone)]
    enum FCol {
        G(Vec<u8>),
        V(usize),
    }

    /// Encode a fact, assigning MORK's NewVar to a slot's first occurrence and VarRef to repeats,
    /// exactly as the stored encoding represents schematic facts and their coreference.
    fn encode_fact(rel: &str, cols: &[FCol]) -> Vec<u8> {
        let mut v = vec![TAG_ARITY | (1 + cols.len()) as u8];
        v.extend(sym(rel));
        let mut introduced: Vec<usize> = Vec::new();
        for col in cols {
            match col {
                FCol::G(g) => v.extend_from_slice(g),
                FCol::V(slot) => {
                    if let Some(idx) = introduced.iter().position(|s| s == slot) {
                        v.push(TAG_VARREF | idx as u8);
                    } else {
                        introduced.push(*slot);
                        v.push(NEWVAR_BYTE);
                    }
                }
            }
        }
        v
    }

    /// A random binary fact: each column ground (small symbol set) or a stored variable (slot 0/1),
    /// so the corpus mixes ground, single-variable, coreferent, and two-variable schematic facts.
    fn gen_fact(rng: &mut Lcg, syms: &[Vec<u8>]) -> Vec<FCol> {
        (0..2)
            .map(|_| {
                if rng.below(3) == 0 {
                    FCol::V(rng.below(2))
                } else {
                    FCol::G(syms[rng.below(syms.len())].clone())
                }
            })
            .collect()
    }

    /// Reference unification join: nested loop over one fact per factor, each fact renamed apart,
    /// unifying its columns with the query variables through the same trail. Collects fully-ground
    /// answer rows, the same projection the zipper join emits.
    fn naive_rec(
        fi: usize,
        factors: &[Factor],
        factor_facts: &[Vec<Vec<FCol>>],
        env: &mut Env,
        nvars: usize,
        out: &mut BTreeSet<Vec<Vec<u8>>>,
    ) {
        if fi == factors.len() {
            let mut row = Vec::with_capacity(nvars);
            for v in 0..nvars {
                match env.ground_of(v) {
                    Some(g) => row.push(g),
                    None => return,
                }
            }
            out.insert(row);
            return;
        }
        for fact in &factor_facts[fi] {
            let mark = env.mark();
            let mut slot_ids: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            let mut ok = true;
            for (ci, fact_col) in fact.iter().enumerate() {
                let mut slot_id = |slot: usize, env: &mut Env| match slot_ids.get(&slot) {
                    Some(&id) => id,
                    None => {
                        let id = env.fresh();
                        slot_ids.insert(slot, id);
                        id
                    }
                };
                let cok = match (&factors[fi].cols[ci], fact_col) {
                    (Col::Ground(q), FCol::G(g)) => q == g,
                    (Col::Ground(q), FCol::V(slot)) => {
                        let id = slot_id(*slot, env);
                        env.unify_var_ground(id, q)
                    }
                    (Col::Var(v), FCol::G(g)) => env.unify_var_ground(*v, g),
                    (Col::Var(v), FCol::V(slot)) => {
                        let id = slot_id(*slot, env);
                        env.unify_var_var(*v, id)
                    }
                    (Col::Compound(q), FCol::G(g)) => {
                        env.unify_terms(&Col::Compound(q.clone()), &Col::Ground(g.clone()))
                    }
                    (Col::Compound(q), FCol::V(slot)) => {
                        let id = slot_id(*slot, env);
                        env.unify_var_term(id, &Col::Compound(q.clone()))
                    }
                };
                if !cok {
                    ok = false;
                    break;
                }
            }
            if ok {
                naive_rec(fi + 1, factors, factor_facts, env, nvars, out);
            }
            env.rollback(mark);
        }
    }

    #[test]
    fn unify_join_matches_naive() {
        for seed in 0..400u64 {
            let mut rng = Lcg::new(seed.wrapping_add(11));
            let nsyms = 2 + rng.below(2);
            let syms: Vec<Vec<u8>> = (0..nsyms)
                .map(|i| sym(&((b'a' + i as u8) as char).to_string()))
                .collect();

            let mut map = PathMap::<()>::new();
            let mut e_facts: Vec<Vec<FCol>> = Vec::new();
            let mut f_facts: Vec<Vec<FCol>> = Vec::new();
            let nfacts = 3 + rng.below(6);
            for _ in 0..nfacts {
                let fe = gen_fact(&mut rng, &syms);
                if map.insert(&encode_fact("e", &fe), ()).is_none() {
                    e_facts.push(fe);
                }
                let ff = gen_fact(&mut rng, &syms);
                if map.insert(&encode_fact("f", &ff), ()).is_none() {
                    f_facts.push(ff);
                }
            }
            let pe = relation_prefix("e", 3);
            let pf = relation_prefix("f", 3);

            let queries: Vec<(Vec<Factor>, Vec<usize>, usize)> = vec![
                (
                    vec![Factor::var_cols(pe.clone(), vec![0, 1])],
                    vec![0, 1],
                    2,
                ),
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![1, 2]),
                    ],
                    vec![0, 1, 2],
                    3,
                ),
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![0, 2]),
                    ],
                    vec![0, 1, 2],
                    3,
                ),
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pf.clone(), vec![1, 2]),
                    ],
                    vec![0, 1, 2],
                    3,
                ),
                // cyclic: triangle over schematic edges (the catch-up-with-unification path).
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![1, 2]),
                        Factor::var_cols(pe.clone(), vec![2, 0]),
                    ],
                    vec![0, 1, 2],
                    3,
                ),
                // cyclic four-cycle over schematic edges.
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![1, 2]),
                        Factor::var_cols(pe.clone(), vec![2, 3]),
                        Factor::var_cols(pe.clone(), vec![3, 0]),
                    ],
                    vec![0, 1, 2, 3],
                    4,
                ),
                // intra-query coreference exercised against schematic data: (e $0 $1)(e $1 $0).
                (
                    vec![
                        Factor::var_cols(pe.clone(), vec![0, 1]),
                        Factor::var_cols(pe.clone(), vec![1, 0]),
                    ],
                    vec![0, 1],
                    2,
                ),
            ];

            for (factors, order, nvars) in &queries {
                let factor_facts: Vec<Vec<Vec<FCol>>> = factors
                    .iter()
                    .map(|fac| {
                        if fac.prefix == pe {
                            e_facts.clone()
                        } else {
                            f_facts.clone()
                        }
                    })
                    .collect();

                let got = unify_join_zipper(&map, factors, order, *nvars);
                let mut env = Env::new(*nvars);
                let mut want = BTreeSet::new();
                naive_rec(0, factors, &factor_facts, &mut env, *nvars, &mut want);

                assert_eq!(
                    got, want,
                    "seed {seed}: unify join must match the naive unifier"
                );
            }
        }
    }

    #[test]
    fn least_ge_matches_brute_force() {
        let sets: &[&[u8]] = &[
            &[],
            &[0],
            &[255],
            &[0, 1, 2, 63, 64, 65, 127, 128, 191, 192, 255],
            &[10, 50, 90, 130, 170, 210, 250],
            &[63, 64],
        ];
        for set in sets {
            let mask = mask_of(set);
            for k in 0u8..=255 {
                let want = set.iter().copied().filter(|&b| b >= k).min();
                assert_eq!(least_ge(&mask, k), want, "set={set:?} k={k}");
            }
        }
    }

    #[test]
    fn first_subterm_len_parses_each_shape() {
        // symbol "ab": SymbolSize(2), 'a', 'b'  -> 3 bytes
        let sym = [TAG_SYMSIZE | 2, b'a', b'b'];
        assert_eq!(first_subterm_len(&sym), 3);
        assert!(first_subterm_is_ground(&sym));

        // NewVar -> 1 byte, non-ground
        let nv = [NEWVAR_BYTE];
        assert_eq!(first_subterm_len(&nv), 1);
        assert!(!first_subterm_is_ground(&nv));

        // VarRef(0) -> 1 byte, non-ground
        let vr = [TAG_VARREF | 0];
        assert_eq!(first_subterm_len(&vr), 1);
        assert!(!first_subterm_is_ground(&vr));

        // (k v0):  Arity(2), Sym("k"), Sym("v0")
        let k = TAG_SYMSIZE | 1;
        let v0 = TAG_SYMSIZE | 2;
        let compound = [TAG_ARITY | 2, k, b'k', v0, b'v', b'0'];
        assert_eq!(first_subterm_len(&compound), 6);
        assert!(first_subterm_is_ground(&compound));

        // (k $x): Arity(2), Sym("k"), NewVar  -> 4 bytes, non-ground
        let compound_var = [TAG_ARITY | 2, k, b'k', NEWVAR_BYTE];
        assert_eq!(first_subterm_len(&compound_var), 4);
        assert!(!first_subterm_is_ground(&compound_var));

        // trailing bytes after the first subterm are ignored: (e A B) prefix then junk
        let mut buf = compound.to_vec();
        buf.extend_from_slice(&[0xFF, 0xFF]);
        assert_eq!(first_subterm_len(&buf), 6);
    }
}
