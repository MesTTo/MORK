//! Unifiability retrieval into a stored region: enumerate every stored fact
//! under a trie region that UNIFIES with a (possibly schematic) key, by a
//! discrimination walk instead of scan-and-unify-per-row.
//!
//! This is the join-side generalization of the guarded-emit sink's one-way
//! coverage walk: where that walk asks "does some stored schema generalize the
//! key" and answers a boolean, this one binds variables on BOTH sides and
//! yields every unifying stored fact. Variable identity discipline follows the
//! guard walk's hard-won rule (see the fc71e65 commit): key-side occurrences
//! are compared through a resolved shadow in which every occurrence carries its
//! canonical variable index, because raw De Bruijn bytes make every first
//! occurrence the same NewVar tag and conflate distinct variables.
//!
//! Cost: O(matching trie paths) per call, exact-byte descent wherever the key
//! is ground, stored-variable branches bound to key subterms, and full branch
//! enumeration only below key-variable positions. Never a region scan.

#![cfg(feature = "retrieval_join")]

use mork_expr::{byte_item, item_byte, Tag};
use pathmap::zipper::{Zipper, ZipperIteration, ZipperMoving, ZipperValues};

/// Byte length of the complete subterm starting at `bytes[pos..]`.
fn subexpr_len_at(bytes: &[u8], pos: usize) -> Option<usize> {
    let (mut owed, mut payload, mut i) = (1usize, 0usize, pos);
    while owed > 0 || payload > 0 {
        let b = *bytes.get(i)?;
        i += 1;
        if payload > 0 {
            payload -= 1;
            continue;
        }
        owed -= 1;
        match byte_item(b) {
            Tag::Arity(k) => owed += k as usize,
            Tag::SymbolSize(s) => payload += s as usize,
            Tag::NewVar | Tag::VarRef(_) => {}
        }
    }
    Some(i - pos)
}

/// Rewrites every key variable occurrence to `VarRef(canonical index)` so
/// binding comparisons see variable identity (the guard walk's discipline).
fn resolve_key_vars(key: &[u8]) -> Vec<u8> {
    let mut resolved = key.to_vec();
    let (mut next, mut i) = (0u8, 0usize);
    while i < resolved.len() {
        match byte_item(resolved[i]) {
            Tag::NewVar => {
                resolved[i] = item_byte(Tag::VarRef(next));
                next = next.saturating_add(1);
                i += 1;
            }
            Tag::VarRef(_) => i += 1,
            Tag::SymbolSize(s) => i += 1 + s as usize,
            Tag::Arity(_) => i += 1,
        }
    }
    resolved
}

/// One unifier found by [`retrieve_unifiable`]: the stored fact's full path
/// bytes (relative to the region root), what each STORED variable bound to
/// (key-byte spans, by stored variable index), and what each KEY variable
/// bound to (stored-byte spans, by canonical key variable index). A key
/// variable bound to a stored variable records the stored span of that
/// variable occurrence.
pub struct RetrievedUnifier<'w> {
    pub stored_path: &'w [u8],
    pub stored_bindings: &'w [(u8, Vec<u8>)],
    pub key_bindings: &'w [(u8, Vec<u8>)],
}

struct Walk<'k, Z, F> {
    rz: Z,
    key: &'k [u8],
    resolved: &'k [u8],
    /// Stored paths already emitted this call: a fact reachable through both a
    /// stored-variable arm and a key-variable arm (variable-variable meetings)
    /// is one CANDIDATE; consumers re-derive exact unifiers per fact.
    emitted: std::collections::HashSet<Vec<u8>>,
    /// stored variable index -> bound key span (from `resolved`).
    stored_bindings: Vec<(u8, Vec<u8>)>,
    /// canonical key variable index -> bound stored span (trie bytes).
    key_bindings: Vec<(u8, Vec<u8>)>,
    emit: F,
    stopped: bool,
}

impl<'k, Z, F> Walk<'k, Z, F>
where
    Z: Zipper + ZipperMoving + ZipperIteration + ZipperValues<()>,
    F: FnMut(RetrievedUnifier) -> bool,
{
    fn emit_here(&mut self) {
        let path = self.rz.path().to_vec();
        if !self.emitted.insert(path.clone()) {
            return;
        }
        let u = RetrievedUnifier {
            stored_path: &path,
            stored_bindings: &self.stored_bindings,
            key_bindings: &self.key_bindings,
        };
        if !(self.emit)(u) {
            self.stopped = true;
        }
    }

    /// Descend every stored branch that completes exactly one stored subterm,
    /// calling `cont` at each completion with the traversed span recorded.
    /// Bounded by the branches below the current node.
    fn each_stored_subterm(&mut self, cont: &mut dyn FnMut(&mut Self, &[u8])) {
        fn go<'k, Z, F>(
            w: &mut Walk<'k, Z, F>,
            owed: usize,
            payload: usize,
            span: &mut Vec<u8>,
            cont: &mut dyn FnMut(&mut Walk<'k, Z, F>, &[u8]),
        ) where
            Z: Zipper + ZipperMoving + ZipperIteration + ZipperValues<()>,
            F: FnMut(RetrievedUnifier) -> bool,
        {
            if w.stopped {
                return;
            }
            if owed == 0 && payload == 0 {
                cont(w, span);
                return;
            }
            let mask = w.rz.child_mask();
            let mut it = mask.iter();
            while let Some(b) = it.next() {
                if w.stopped {
                    return;
                }
                if !w.rz.descend_to_existing_byte(b) {
                    continue;
                }
                span.push(b);
                if payload > 0 {
                    go(w, owed, payload - 1, span, cont);
                } else {
                    let (o, p) = match byte_item(b) {
                        Tag::Arity(k) => (owed - 1 + k as usize, 0),
                        Tag::SymbolSize(s) => (owed - 1, s as usize),
                        Tag::NewVar | Tag::VarRef(_) => (owed - 1, 0),
                    };
                    go(w, o, p, span, cont);
                }
                span.pop();
                w.rz.ascend_byte();
            }
        }
        let mut span = Vec::new();
        go(self, 1, 0, &mut span, cont);
    }

    fn step(&mut self, pos: usize) {
        if self.stopped {
            return;
        }
        if pos == self.key.len() {
            if self.rz.value().is_some() {
                self.emit_here();
            }
            return;
        }

        // Stored NewVar: binds the whole key subterm at `pos`.
        if self.rz.descend_to_existing_byte(item_byte(Tag::NewVar)) {
            if let Some(len) = subexpr_len_at(self.key, pos) {
                let idx = self.stored_bindings.len() as u8;
                self.stored_bindings
                    .push((idx, self.resolved[pos..pos + len].to_vec()));
                self.step(pos + len);
                self.stored_bindings.pop();
            }
            self.rz.ascend_byte();
        }

        // Stored VarRef(i): key subterm must equal binding i (resolved spans).
        {
            let mask = self.rz.child_mask();
            let mut it = mask.iter();
            while let Some(b) = it.next() {
                if self.stopped {
                    return;
                }
                let Tag::VarRef(i) = byte_item(b) else { continue };
                let Some((_, bound)) =
                    self.stored_bindings.iter().find(|(k, _)| *k == i)
                else {
                    continue;
                };
                let bound = bound.clone();
                if self.resolved[pos..].starts_with(&bound)
                    && self.rz.descend_to_existing_byte(b)
                {
                    self.step(pos + bound.len());
                    self.rz.ascend_byte();
                }
            }
        }

        match byte_item(self.key[pos]) {
            // Key variable: first occurrence binds ANY stored subterm below;
            // re-occurrence must find exactly the bound stored span.
            Tag::NewVar | Tag::VarRef(_) => {
                let Tag::VarRef(kidx) = byte_item(self.resolved[pos]) else {
                    return;
                };
                let existing = self
                    .key_bindings
                    .iter()
                    .find(|(k, _)| *k == kidx)
                    .map(|(_, s)| s.clone());
                match existing {
                    Some(span) => {
                        if self.rz.descend_to_check(&span) {
                            self.step(pos + 1);
                        }
                        self.rz.ascend(span.len());
                    }
                    None => {
                        self.each_stored_subterm(&mut |w, span| {
                            w.key_bindings.push((kidx, span.to_vec()));
                            w.step(pos + 1);
                            w.key_bindings.pop();
                        });
                    }
                }
            }
            Tag::SymbolSize(size) => {
                let next = pos + 1 + size as usize;
                if next > self.key.len() {
                    return;
                }
                if self.rz.descend_to_existing_byte(self.key[pos]) {
                    // descend_to_check moves the full payload even when the
                    // path does not exist, so the payload ascent is
                    // unconditional (the sinks walk's balance rule).
                    if self.rz.descend_to_check(&self.key[pos + 1..next]) {
                        self.step(next);
                    }
                    self.rz.ascend(size as usize);
                    self.rz.ascend_byte();
                }
            }
            Tag::Arity(_) => {
                if self.rz.descend_to_existing_byte(self.key[pos]) {
                    self.step(pos + 1);
                    self.rz.ascend_byte();
                }
            }
        }
    }
}

/// Enumerate every stored fact under `region` unifiable with `key`, calling
/// `emit` per unifier; `emit` returning false stops the walk. Returns the
/// number of unifiers emitted.
pub fn retrieve_unifiable<Z, F>(region: Z, key: &[u8], mut emit: F) -> usize
where
    Z: Zipper + ZipperMoving + ZipperIteration + ZipperValues<()>,
    F: FnMut(RetrievedUnifier) -> bool,
{
    let resolved = resolve_key_vars(key);
    let mut count = 0usize;
    let mut walk = Walk {
        rz: region,
        key,
        resolved: &resolved,
        emitted: std::collections::HashSet::new(),
        stored_bindings: Vec::new(),
        key_bindings: Vec::new(),
        emit: |u: RetrievedUnifier| {
            count += 1;
            emit(u)
        },
        stopped: false,
    };
    walk.step(0);
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use pathmap::PathMap;

    fn enc(s: &str) -> Vec<u8> {
        // Tiny encoder for tests: tokens are single ASCII symbols, $x-style
        // variables in first-occurrence order, parenthesized arity groups.
        fn go(toks: &[String], i: &mut usize, vars: &mut Vec<String>, out: &mut Vec<u8>) {
            let t = toks[*i].clone();
            *i += 1;
            if t == "(" {
                let start = out.len();
                out.push(item_byte(Tag::Arity(0)));
                let mut n = 0u8;
                while toks[*i] != ")" {
                    go(toks, i, vars, out);
                    n += 1;
                }
                *i += 1;
                out[start] = item_byte(Tag::Arity(n));
            } else if let Some(name) = t.strip_prefix('$') {
                match vars.iter().position(|v| v == name) {
                    Some(k) => out.push(item_byte(Tag::VarRef(k as u8))),
                    None => {
                        vars.push(name.to_string());
                        out.push(item_byte(Tag::NewVar));
                    }
                }
            } else {
                out.push(item_byte(Tag::SymbolSize(t.len() as u8)));
                out.extend_from_slice(t.as_bytes());
            }
        }
        let spaced = s.replace('(', " ( ").replace(')', " ) ");
        let toks: Vec<String> = spaced.split_whitespace().map(str::to_string).collect();
        let (mut i, mut vars, mut out) = (0usize, Vec::new(), Vec::new());
        go(&toks, &mut i, &mut vars, &mut out);
        out
    }

    fn region(facts: &[&str]) -> PathMap<()> {
        let mut m = PathMap::new();
        for f in facts {
            m.insert(&enc(f), ());
        }
        m
    }

    fn hits(m: &PathMap<()>, key: &str) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        retrieve_unifiable(m.read_zipper(), &enc(key), |u| {
            out.push(u.stored_path.to_vec());
            true
        });
        out.sort();
        out
    }

    #[test]
    fn ground_key_exact_and_stored_var_matches() {
        let m = region(&["(f a b)", "(f $x b)", "(f a c)", "(g a b)"]);
        let got = hits(&m, "(f a b)");
        assert_eq!(got.len(), 2, "exact fact and (f $x b) both unify");
    }

    #[test]
    fn key_variable_enumerates_stored_branches() {
        let m = region(&["(f a b)", "(f (h a) b)", "(f a c)"]);
        let got = hits(&m, "(f $k b)");
        assert_eq!(got.len(), 2, "$k binds a and (h a); (f a c) does not unify");
    }

    #[test]
    fn repeated_key_variable_requires_equal_stored_spans() {
        let m = region(&["(f a a)", "(f a b)", "(f (h c) (h c))"]);
        let got = hits(&m, "(f $k $k)");
        assert_eq!(got.len(), 2, "(f a a) and (f (h c) (h c)) only");
    }

    #[test]
    fn repeated_stored_variable_requires_equal_key_spans() {
        let m = region(&["(f $x $x)", "(f $x $y)"]);
        assert_eq!(hits(&m, "(f a b)").len(), 1, "only (f $x $y) covers (f a b)");
        assert_eq!(hits(&m, "(f a a)").len(), 2, "both cover (f a a)");
    }

    #[test]
    fn distinct_key_variables_do_not_conflate() {
        // The fc71e65 trap, unification-side: stored (f $x $x) must NOT unify
        // key (f $a $b) into a ground answer set claim... it DOES unify (by
        // equating $a with $b), and must be reported: unification is two-way.
        let m = region(&["(f $x $x)"]);
        assert_eq!(hits(&m, "(f $a $b)").len(), 1, "unifies by equating key vars");
        // But a repeated KEY var against distinct stored structure must fail.
        let m2 = region(&["(f a b)"]);
        assert_eq!(hits(&m2, "(f $k $k)").len(), 0);
    }

    #[test]
    fn schematic_meet_shape() {
        // The MITM meet's failing class: key contains a variable where stored
        // facts hold structure, and structure where stored facts hold schemas.
        let m = region(&[
            "(: p1 (i (n $a) (i $a $b)))",
            "(: p2 (i $a (i $b $a)))",
            "(: p3 (i c d))",
        ]);
        // With symbol y, p2 would need stored $a to equal both (n x) and y --
        // no unifier, and the walk correctly rejects it. With key variable $y,
        // p2 unifies by binding $y to (n x).
        let got = hits(&m, "(: $p (i (n x) (i x y)))");
        assert_eq!(got.len(), 1, "only p1 unifies against the all-symbol key");
        let got = hits(&m, "(: $p (i (n x) (i x $y)))");
        assert_eq!(got.len(), 2, "p1 and p2 unify once $y is a variable; p3 does not");
    }
}
