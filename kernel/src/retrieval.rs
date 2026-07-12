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

/// Tag-aware scan: does this encoded span contain any variable occurrence?
/// Payload bytes are skipped (raw symbol bytes alias the tag ranges).
pub(crate) fn span_contains_vars(bytes: &[u8]) -> bool {
    let mut i = 0usize;
    while i < bytes.len() {
        match byte_item(bytes[i]) {
            Tag::NewVar | Tag::VarRef(_) => return true,
            Tag::SymbolSize(s) => i += 1 + s as usize,
            Tag::Arity(_) => i += 1,
        }
    }
    false
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
    pub stored_bindings: &'w [Option<Vec<u8>>],
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
    /// Stored NewVars in OCCURRENCE order (VarRef indices align with this):
    /// `Some(span)` when the stored variable bound a key subterm exactly,
    /// `None` (wildcard) when it was consumed inside a key-variable's swallowed
    /// span -- its coreference constraints are enforced by the consumer's full
    /// unification, so wildcard occurrences accept any key subterm here (the
    /// walk is a COMPLETE overapproximation; it never misses a unifiable fact).
    stored_bindings: Vec<Option<Vec<u8>>>,
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
                    let mut pushed = false;
                    let (o, p) = match byte_item(b) {
                        Tag::Arity(k) => (owed - 1 + k as usize, 0),
                        Tag::SymbolSize(s) => (owed - 1, s as usize),
                        Tag::NewVar => {
                            // A stored variable consumed inside a swallowed
                            // span: keep the occurrence index aligned and mark
                            // it wildcard for later VarRef re-occurrences.
                            w.stored_bindings.push(None);
                            pushed = true;
                            (owed - 1, 0)
                        }
                        Tag::VarRef(_) => (owed - 1, 0),
                    };
                    go(w, o, p, span, cont);
                    if pushed {
                        w.stored_bindings.pop();
                    }
                }
                span.pop();
                w.rz.ascend_byte();
            }
        }
        let mut span = Vec::new();
        go(self, 1, 0, &mut span, cont);
    }

    /// Stored VarRef(i) branches at the cursor: a re-occurrence may match
    /// the subterm `sub` when the spans are byte-equal or either side still
    /// holds variables (byte-equality is only complete for ground spans --
    /// the WAM unify_value constraint); a wildcarded stored variable accepts
    /// anything. The consumer's full unification is the final judge either
    /// way, so acceptance here only needs to never reject a unifiable pair.
    fn each_stored_varref(&mut self, sub: &[u8], cont: &mut dyn FnMut(&mut Self)) {
        let mask = self.rz.child_mask();
        let mut it = mask.iter();
        while let Some(b) = it.next() {
            if self.stopped {
                return;
            }
            let Tag::VarRef(i) = byte_item(b) else { continue };
            let ok = match self.stored_bindings.get(i as usize) {
                Some(Some(bound)) => {
                    sub == &bound[..]
                        || span_contains_vars(bound)
                        || span_contains_vars(sub)
                }
                Some(None) => true,
                None => false,
            };
            if ok && self.rz.descend_to_existing_byte(b) {
                cont(self);
                self.rz.ascend_byte();
            }
        }
    }

    /// Walk the stored trie in lockstep with a GROUND resolved span (a
    /// repeated key variable's earlier binding), honoring stored-variable
    /// branches exactly as the main walk does: a stored NewVar binds the span
    /// subterm at the cursor, a stored VarRef re-checks per the binding
    /// rules. Exact byte descent alone is incomplete here -- the stored side
    /// may hold variables below this position that unify with the ground
    /// span. Resumes the main key walk at `resume_pos` once the span is
    /// exhausted.
    fn walk_span(&mut self, span: &[u8], spos: usize, resume_pos: usize) {
        if self.stopped {
            return;
        }
        if spos == span.len() {
            self.step(resume_pos);
            return;
        }

        // Stored NewVar: binds the whole span subterm at `spos`. The span is
        // ground, so the binding it records is exact (no wildcard needed).
        if self.rz.descend_to_existing_byte(item_byte(Tag::NewVar)) {
            if let Some(len) = subexpr_len_at(span, spos) {
                self.stored_bindings
                    .push(Some(span[spos..spos + len].to_vec()));
                self.walk_span(span, spos + len, resume_pos);
                self.stored_bindings.pop();
            }
            self.rz.ascend_byte();
        }

        // Stored VarRef(i): shared binding rules; the span is ground, so the
        // var-containing-sub disjunct is vacuously false here.
        if let Some(len) = subexpr_len_at(span, spos) {
            self.each_stored_varref(&span[spos..spos + len], &mut |w| {
                w.walk_span(span, spos + len, resume_pos)
            });
        }

        match byte_item(span[spos]) {
            Tag::SymbolSize(size) => {
                let next = spos + 1 + size as usize;
                if next > span.len() {
                    return;
                }
                if self.rz.descend_to_existing_byte(span[spos]) {
                    // descend_to_check moves the full payload even on failure,
                    // so the payload ascent is unconditional.
                    if self.rz.descend_to_check(&span[spos + 1..next]) {
                        self.walk_span(span, next, resume_pos);
                    }
                    self.rz.ascend(size as usize);
                    self.rz.ascend_byte();
                }
            }
            Tag::Arity(_) => {
                if self.rz.descend_to_existing_byte(span[spos]) {
                    self.walk_span(span, spos + 1, resume_pos);
                    self.rz.ascend_byte();
                }
            }
            Tag::NewVar | Tag::VarRef(_) => {
                debug_assert!(false, "walk_span requires a ground span");
            }
        }
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
                self.stored_bindings
                    .push(Some(self.resolved[pos..pos + len].to_vec()));
                self.step(pos + len);
                self.stored_bindings.pop();
            }
            self.rz.ascend_byte();
        }

        // Stored VarRef(i): key re-occurrence under the shared binding rules
        // (resolved shadow carries key variable identity).
        let resolved = self.resolved;
        if let Some(len) = subexpr_len_at(resolved, pos) {
            self.each_stored_varref(&resolved[pos..pos + len], &mut |w| {
                w.step(pos + len)
            });
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
                    Some(span) if !span_contains_vars(&span) => {
                        // Exact descent alone would miss stored-variable
                        // branches that unify with the ground span; the
                        // sub-walk honors them (the completeness oracle
                        // caught exactly this on the barrier corpus).
                        self.walk_span(&span, 0, pos + 1);
                    }
                    Some(_) => {
                        // The bound stored span contains stored variables whose
                        // bytes are position-dependent; enumerate and let the
                        // consumer's unification enforce equality.
                        self.each_stored_subterm(&mut |w, _span| {
                            w.step(pos + 1);
                        });
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
    fn repeated_key_variable_takes_stored_variable_branches() {
        // The barrier-corpus miss: the re-occurrence's ground binding must
        // still admit stored facts holding variables at or below that
        // position -- (n $s) unifies with the bound (n a) by $s := a, and a
        // bare stored $w swallows the whole span. Exact byte descent alone
        // finds only the ground twin.
        let m = region(&[
            "(f (n a) (n a))",
            "(f (n a) (n $s))",
            "(f (n a) $w)",
            "(f (n a) (n b))",
            "(f (n a) (m $s))",
        ]);
        let got = hits(&m, "(f $k $k)");
        assert_eq!(got.len(), 3, "ground twin, (n $s), and bare $w unify");
        // Wildcard arm inside the span sub-walk: stored $y is swallowed by a
        // DIFFERENT key variable ($m), so its VarRef under the re-occurrence
        // span carries no binding and must accept the ground subterm.
        let m2 = region(&[
            "(g (n a) x (n a))",
            "(g (n a) $y (n $y))",
            "(g (n a) $y (n b))",
        ]);
        assert_eq!(
            hits(&m2, "(g $k $m $k)").len(),
            2,
            "ground twin and the $y-linked fact; (n b) breaks the twin"
        );
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

    /// Randomized differential: over random (region, key) instances, every
    /// stored fact the whole-pair unification accepts must appear in the
    /// walk's candidate set (the walk may overapproximate, never miss).
    /// Deterministic LCG seeds keep failures reproducible.
    #[test]
    fn randomized_walk_never_misses_a_unifiable_fact() {
        use mork_expr::{unify as pair_unify, Expr, ExprEnv};

        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> u64 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.0 >> 33
            }
            fn pick(&mut self, n: usize) -> usize {
                (self.next() % n as u64) as usize
            }
        }

        fn term(r: &mut Lcg, depth: usize, vars: &[&str]) -> String {
            let syms = ["a", "b", "c", "f", "g", "n"];
            let roll = r.pick(10);
            if depth == 0 || roll < 4 {
                if roll < 2 && !vars.is_empty() {
                    format!("${}", vars[r.pick(vars.len())])
                } else {
                    syms[r.pick(syms.len())].to_string()
                }
            } else {
                let n = 2 + r.pick(2);
                let items: Vec<String> =
                    (0..n).map(|_| term(r, depth - 1, vars)).collect();
                format!("({})", items.join(" "))
            }
        }

        for seed in 0..400u64 {
            let mut r = Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
            let nfacts = 3 + r.pick(6);
            let facts: Vec<String> = (0..nfacts)
                .map(|_| term(&mut r, 3, &["x", "y", "z"]))
                .collect();
            let facts: Vec<String> =
                facts.into_iter().filter(|f| f.starts_with('(')).collect();
            if facts.is_empty() {
                continue;
            }
            let key = loop {
                let k = term(&mut r, 3, &["k", "m"]);
                if k.starts_with('(') {
                    break k;
                }
            };
            let m = region(&facts.iter().map(String::as_str).collect::<Vec<_>>());
            let got: std::collections::HashSet<Vec<u8>> =
                hits(&m, &key).into_iter().collect();
            let kb = enc(&key);
            for f in &facts {
                let fb = enc(f);
                let mut ps = vec![(
                    ExprEnv::new(
                        0,
                        Expr {
                            ptr: kb.as_ptr().cast_mut(),
                        },
                    ),
                    ExprEnv::new(
                        1,
                        Expr {
                            ptr: fb.as_ptr().cast_mut(),
                        },
                    ),
                )];
                if pair_unify(&mut ps).is_ok() && !got.contains(&fb) {
                    panic!(
                        "seed {seed}: walk missed unifiable fact {f} for key {key}"
                    );
                }
            }
        }
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
