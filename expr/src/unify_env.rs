//! Reversible, backtrackable unification environment for the unify-capable join.
//!
//! The leapfrog/variable-at-a-time join over schematic (variable-bearing) facts
//! needs a substitution environment that can be refined as the cursor descends
//! and rolled back when it backtracks. MORK's existing `unify` (lib.rs:3173)
//! builds a fresh `BTreeMap` substitution per call and is not reversible; this is
//! the incremental version: union-find equivalence classes over variables plus a
//! trail for O(1) rollback, occurs-check before binding, and a finite-tree vs
//! rational-tree mode. The design follows the friend's v8 §133 `UnifyEnv` and the
//! reversible-union-find ("union-find with undo") used in CP/SAT solvers, which
//! avoids path compression so every change is a single trail entry.
//!
//! This module is deliberately over an abstract `Term` (symbol / variable /
//! compound), the same model proved correct in
//! `kernel/resources/formal/verus/VarRefRecheck.rs`, so the core is testable and
//! Verus-friendly. The `Expr` (byte-encoded) bridge lives separately.

/// A dense variable id. Query-side and data-side variables are alpha-normalized
/// into one disjoint id space before unification (v8 §133.1: namespaces must be
/// disjoint, identity carries scope, not incidental byte offsets).
pub type VarId = u32;

/// Whether cycles (`x = f(x)`) are rejected (finite trees, with occurs-check) or
/// admitted (rational/infinite trees). This is a term-theory choice, not an
/// optimization: it changes which problems are solvable, so it belongs in the
/// plan-cache key (v8 §119.1), never silently mixed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TreeMode {
    FiniteWithOccursCheck,
    Rational,
}

/// Abstract term: a symbol, a variable, or a compound of sub-terms. Mirrors the
/// `Term` enum the Verus proof reasons about.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Term {
    Sym(u64),
    Var(VarId),
    Cmp(Vec<Term>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnifyFail {
    /// Symbol clash or arity/shape mismatch.
    Clash,
    /// Occurs-check failure in finite-tree mode (`x` occurs in the term bound to
    /// `x`), i.e. an attempt to build a cyclic term.
    OccursCycle(VarId),
}

/// One reversible change, recorded so `rollback` can undo to a `mark`.
enum TrailEntry {
    /// `parent[child]` was set from `child` (a root) to a new root. Undo: make
    /// `child` a root again, and if the union bumped the surviving root's rank,
    /// drop it back.
    Union { child: VarId, rank_bumped_root: Option<VarId> },
    /// `binding[class]` was set from `None` to `Some(_)`. Undo: clear it.
    Bind { class: VarId },
}

/// Backtrackable union-find over variables, with an optional bound term per class
/// representative. `find` follows parent links without path compression (so each
/// mutation is one trail entry and rollback is exact); union-by-rank keeps `find`
/// at O(log n).
pub struct UnifyEnv {
    parent: Vec<VarId>,
    rank: Vec<u8>,
    binding: Vec<Option<Term>>,
    trail: Vec<TrailEntry>,
    mode: TreeMode,
}

impl UnifyEnv {
    pub fn new(mode: TreeMode) -> Self {
        UnifyEnv { parent: Vec::new(), rank: Vec::new(), binding: Vec::new(), trail: Vec::new(), mode }
    }

    /// Allocate `n` fresh, unbound, singleton variables and return the first id.
    pub fn with_vars(n: usize, mode: TreeMode) -> Self {
        let mut e = Self::new(mode);
        for _ in 0..n { e.fresh_var(); }
        e
    }

    /// Allocate a fresh unbound variable as its own class.
    pub fn fresh_var(&mut self) -> VarId {
        let id = self.parent.len() as VarId;
        self.parent.push(id);
        self.rank.push(0);
        self.binding.push(None);
        id
    }

    /// Class representative of `v` (parent chain to the root). No path
    /// compression, so this performs no recorded mutation and is rollback-safe.
    pub fn find(&self, v: VarId) -> VarId {
        let mut r = v;
        while self.parent[r as usize] != r {
            r = self.parent[r as usize];
        }
        r
    }

    /// A rollback point: pass the returned mark to `rollback`.
    #[inline]
    pub fn mark(&self) -> usize {
        self.trail.len()
    }

    /// Undo every change recorded after `mark`, restoring the environment exactly.
    pub fn rollback(&mut self, mark: usize) {
        while self.trail.len() > mark {
            match self.trail.pop().unwrap() {
                TrailEntry::Union { child, rank_bumped_root } => {
                    self.parent[child as usize] = child;
                    if let Some(root) = rank_bumped_root {
                        self.rank[root as usize] -= 1;
                    }
                }
                TrailEntry::Bind { class } => {
                    self.binding[class as usize] = None;
                }
            }
        }
    }

    /// The term bound to `v`'s class, if any (looked up on the representative).
    pub fn binding_of(&self, v: VarId) -> Option<&Term> {
        self.binding[self.find(v) as usize].as_ref()
    }

    /// Fully resolve `t` under the current substitution: replace every variable by
    /// its class representative, and by its bound term where one exists. In
    /// `Rational` mode a bound variable that (transitively) contains itself yields
    /// a finite unfolding bounded by `fuel` to stay total.
    pub fn resolve(&self, t: &Term) -> Term {
        self.resolve_fueled(t, u32::MAX)
    }

    fn resolve_fueled(&self, t: &Term, fuel: u32) -> Term {
        match t {
            Term::Sym(s) => Term::Sym(*s),
            Term::Cmp(ts) => Term::Cmp(ts.iter().map(|x| self.resolve_fueled(x, fuel)).collect()),
            Term::Var(v) => {
                let r = self.find(*v);
                match (&self.binding[r as usize], fuel) {
                    (Some(b), f) if f > 0 => self.resolve_fueled(b, f - 1),
                    _ => Term::Var(r),
                }
            }
        }
    }

    /// Does variable `v` occur in the resolved form of `t`? Used as the
    /// occurs-check before binding in finite-tree mode. Follows bound variables so
    /// indirect cycles (created only after earlier bindings) are caught, exactly
    /// as v8 §63 requires.
    fn occurs(&self, v: VarId, t: &Term) -> bool {
        match t {
            Term::Sym(_) => false,
            Term::Cmp(ts) => ts.iter().any(|x| self.occurs(v, x)),
            Term::Var(w) => {
                let r = self.find(*w);
                if r == v { return true; }
                match &self.binding[r as usize] {
                    Some(b) => self.occurs(v, b),
                    None => false,
                }
            }
        }
    }

    /// Union the class of `child_root` under `root`, recording the trail entry.
    /// Caller guarantees both are distinct roots.
    fn link(&mut self, a: VarId, b: VarId) -> VarId {
        debug_assert!(self.parent[a as usize] == a && self.parent[b as usize] == b && a != b);
        let (root, child) = if self.rank[a as usize] >= self.rank[b as usize] { (a, b) } else { (b, a) };
        self.parent[child as usize] = root;
        let mut bumped = None;
        if self.rank[root as usize] == self.rank[child as usize] {
            self.rank[root as usize] += 1;
            bumped = Some(root);
        }
        self.trail.push(TrailEntry::Union { child, rank_bumped_root: bumped });
        root
    }

    /// Bind class `class` (a root) to `t`, recording the trail entry. Caller
    /// guarantees `class` is an unbound root.
    fn set_binding(&mut self, class: VarId, t: Term) {
        debug_assert!(self.parent[class as usize] == class && self.binding[class as usize].is_none());
        self.binding[class as usize] = Some(t);
        self.trail.push(TrailEntry::Bind { class });
    }

    /// Unify two terms, refining the environment. On failure the environment is
    /// left partially refined; the caller rolls back to a `mark` taken before the
    /// call (the leapfrog cursor always does). Returns `Ok` when a unifier exists.
    pub fn unify(&mut self, a: &Term, b: &Term) -> Result<(), UnifyFail> {
        let mut stack: Vec<(Term, Term)> = Vec::new();
        stack.push((a.clone(), b.clone()));
        while let Some((x, y)) = stack.pop() {
            match (x, y) {
                (Term::Sym(p), Term::Sym(q)) => {
                    if p != q { return Err(UnifyFail::Clash); }
                }
                (Term::Cmp(ps), Term::Cmp(qs)) => {
                    if ps.len() != qs.len() { return Err(UnifyFail::Clash); }
                    for (pi, qi) in ps.into_iter().zip(qs.into_iter()) {
                        stack.push((pi, qi));
                    }
                }
                (Term::Var(v), other) | (other, Term::Var(v)) => {
                    self.bind_var(v, other, &mut stack)?;
                }
                (Term::Sym(_), Term::Cmp(_)) | (Term::Cmp(_), Term::Sym(_)) => {
                    return Err(UnifyFail::Clash);
                }
            }
        }
        Ok(())
    }

    /// Unify variable `v` with `other`, pushing decomposed sub-problems onto
    /// `stack`. Resolves `v` to its class first; if already bound, unify the bound
    /// term with `other`; otherwise bind (var-var = union, var-term = occurs-check
    /// then bind).
    fn bind_var(&mut self, v: VarId, other: Term, stack: &mut Vec<(Term, Term)>) -> Result<(), UnifyFail> {
        let rv = self.find(v);
        // If v's class is already bound, unify its binding with `other`.
        if let Some(bound) = self.binding[rv as usize].clone() {
            stack.push((bound, other));
            return Ok(());
        }
        match other {
            Term::Var(w) => {
                let rw = self.find(w);
                if rv == rw { return Ok(()); }
                match self.binding[rw as usize].clone() {
                    // rw bound, rv not: union, the bound term survives on the new root.
                    Some(wb) => {
                        let root = self.link(rv, rw);
                        if self.binding[root as usize].is_none() {
                            // root is rv (took rv's rank); move the binding onto it.
                            self.set_binding(root, wb);
                        }
                    }
                    None => { self.link(rv, rw); }
                }
            }
            t => {
                if self.mode == TreeMode::FiniteWithOccursCheck && self.occurs(rv, &t) {
                    return Err(UnifyFail::OccursCycle(rv));
                }
                self.set_binding(rv, t);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: VarId) -> Term { Term::Var(i) }
    fn s(i: u64) -> Term { Term::Sym(i) }
    fn c(ts: Vec<Term>) -> Term { Term::Cmp(ts) }

    #[test]
    fn sym_clash_and_match() {
        let mut e = UnifyEnv::new(TreeMode::FiniteWithOccursCheck);
        assert!(e.unify(&s(1), &s(1)).is_ok());
        assert_eq!(e.unify(&s(1), &s(2)), Err(UnifyFail::Clash));
    }

    #[test]
    fn var_binds_and_resolves() {
        let mut e = UnifyEnv::with_vars(2, TreeMode::FiniteWithOccursCheck);
        // $0 = f($1), $1 = a  =>  $0 resolves to f(a)
        assert!(e.unify(&v(0), &c(vec![s(9), v(1)])).is_ok());
        assert!(e.unify(&v(1), &s(0)).is_ok());
        assert_eq!(e.resolve(&v(0)), c(vec![s(9), s(0)]));
    }

    #[test]
    fn occurs_check_rejects_cycle_in_finite_mode() {
        let mut e = UnifyEnv::with_vars(1, TreeMode::FiniteWithOccursCheck);
        // $0 = f($0) must fail with occurs-check.
        assert_eq!(e.unify(&v(0), &c(vec![s(9), v(0)])), Err(UnifyFail::OccursCycle(0)));
    }

    #[test]
    fn rational_mode_accepts_cycle() {
        let mut e = UnifyEnv::with_vars(1, TreeMode::Rational);
        // $0 = f($0) succeeds in rational mode (no occurs-check).
        assert!(e.unify(&v(0), &c(vec![s(9), v(0)])).is_ok());
    }

    #[test]
    fn indirect_occurs_cycle_caught() {
        let mut e = UnifyEnv::with_vars(2, TreeMode::FiniteWithOccursCheck);
        // $0 = f($1) ok; then $1 = g($0) would create an indirect cycle -> fail.
        assert!(e.unify(&v(0), &c(vec![s(1), v(1)])).is_ok());
        assert_eq!(e.unify(&v(1), &c(vec![s(2), v(0)])), Err(UnifyFail::OccursCycle(1)));
    }

    #[test]
    fn rollback_restores_exactly() {
        let mut e = UnifyEnv::with_vars(3, TreeMode::FiniteWithOccursCheck);
        let m = e.mark();
        assert!(e.unify(&v(0), &v(1)).is_ok());
        assert!(e.unify(&v(1), &s(7)).is_ok());
        assert_eq!(e.resolve(&v(0)), s(7));
        e.rollback(m);
        // After rollback, $0 and $1 are independent unbound roots again.
        assert_eq!(e.find(0), 0);
        assert_eq!(e.find(1), 1);
        assert_eq!(e.resolve(&v(0)), v(0));
        assert_eq!(e.resolve(&v(1)), v(1));
        assert_eq!(e.mark(), m);
    }

    #[test]
    fn var_var_then_term_propagates_to_class() {
        let mut e = UnifyEnv::with_vars(3, TreeMode::FiniteWithOccursCheck);
        // $0=$1, $1=$2, $2=a  =>  all resolve to a.
        assert!(e.unify(&v(0), &v(1)).is_ok());
        assert!(e.unify(&v(1), &v(2)).is_ok());
        assert!(e.unify(&v(2), &s(5)).is_ok());
        assert_eq!(e.resolve(&v(0)), s(5));
        assert_eq!(e.resolve(&v(1)), s(5));
        assert_eq!(e.resolve(&v(2)), s(5));
    }

    // The issue-29 minimal witness (v8 §119.3), order-independence of the match
    // set: two equations sharing a substitution must unify to the SAME resolved
    // binding regardless of the order they are processed. Encoded abstractly:
    //   eq1:  (: ($f) $t)  =  (: $a A)
    //   eq2:  (: $f (-> $t)) = (: f (-> A))
    // with the shared solution {$f -> f, $t -> A, $a -> (f)}.
    // Symbols: ':'=10, '->'=11, 'A'=20, 'f'=21.  Vars: $f=0 $t=1 $a=2.
    fn issue29_eqs() -> ((Term, Term), (Term, Term)) {
        let colon = 10; let arrow = 11; let a_sym = 20; let f_sym = 21;
        let eq1 = (
            c(vec![s(colon), c(vec![v(0)]), v(1)]),        // (: ($f) $t)
            c(vec![s(colon), v(2), s(a_sym)]),             // (: $a A)
        );
        let eq2 = (
            c(vec![s(colon), v(0), c(vec![s(arrow), v(1)])]),       // (: $f (-> $t))
            c(vec![s(colon), s(f_sym), c(vec![s(arrow), s(a_sym)])]), // (: f (-> A))
        );
        (eq1, eq2)
    }

    #[test]
    fn issue29_order_independent() {
        let f_sym = 21; let a_sym = 20;
        let (eq1, eq2) = issue29_eqs();
        // Order A: eq1 then eq2.
        let mut e1 = UnifyEnv::with_vars(3, TreeMode::FiniteWithOccursCheck);
        assert!(e1.unify(&eq1.0, &eq1.1).is_ok());
        assert!(e1.unify(&eq2.0, &eq2.1).is_ok());
        // Order B: eq2 then eq1 (the order the one-way matcher dropped).
        let mut e2 = UnifyEnv::with_vars(3, TreeMode::FiniteWithOccursCheck);
        assert!(e2.unify(&eq2.0, &eq2.1).is_ok());
        assert!(e2.unify(&eq1.0, &eq1.1).is_ok());
        // Both orders yield the same resolved bindings: $f->f, $t->A, $a->(f).
        for e in [&e1, &e2] {
            assert_eq!(e.resolve(&v(0)), s(f_sym));        // $f -> f
            assert_eq!(e.resolve(&v(1)), s(a_sym));        // $t -> A
            assert_eq!(e.resolve(&v(2)), c(vec![s(f_sym)])); // $a -> (f)
        }
    }
}
