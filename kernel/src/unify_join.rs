//! Encoded MORK terms joined with leapfrog unification.
//!
//! The public entry point accepts MORK's tag-byte encoding directly. Internally
//! the join uses the same small term model, trail substitution, and trie walk as
//! the validated prototype, with symbols stored as raw bytes.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Term {
    Sym(Vec<u8>),
    Var(u32),
    App(Vec<Term>),
}

const TOP2: u8 = 0b1100_0000;
const TAG_ARITY: u8 = 0b0000_0000;
const TAG_VARREF: u8 = 0b1000_0000;
const TAG_SYMSIZE: u8 = 0b1100_0000;
const NEWVAR_BYTE: u8 = 0b1100_0000;
const LOW6: u8 = 0b0011_1111;
const MAX6: usize = 63;
const DATA_VAR_BASE: u32 = 1_000_000;
const DATA_VAR_STRIDE: u32 = 64;

impl Term {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut intro = HashMap::new();
        self.encode_into(&mut out, &mut intro);
        out
    }

    fn encode_into(&self, out: &mut Vec<u8>, intro: &mut HashMap<u32, u8>) {
        match self {
            Term::Sym(bytes) => {
                assert!(
                    (1..=MAX6).contains(&bytes.len()),
                    "symbol length {} out of 1..=63",
                    bytes.len()
                );
                out.push(TAG_SYMSIZE | bytes.len() as u8);
                out.extend_from_slice(bytes);
            }
            Term::Var(id) => {
                if let Some(&level) = intro.get(id) {
                    out.push(TAG_VARREF | level);
                } else {
                    let level = intro.len();
                    assert!(level < 64, "more than 64 distinct variables");
                    intro.insert(*id, level as u8);
                    out.push(NEWVAR_BYTE);
                }
            }
            Term::App(args) => {
                assert!(args.len() <= MAX6, "arity {} out of 0..=63", args.len());
                out.push(TAG_ARITY | args.len() as u8);
                for arg in args {
                    arg.encode_into(out, intro);
                }
            }
        }
    }

    fn decode(bytes: &[u8]) -> Term {
        let mut pos = 0usize;
        let mut next_level = 0u32;
        let term = Term::decode_at(bytes, &mut pos, &mut next_level);
        assert_eq!(pos, bytes.len(), "trailing bytes after a complete term");
        term
    }

    fn decode_at(bytes: &[u8], pos: &mut usize, next_level: &mut u32) -> Term {
        let b = bytes[*pos];
        *pos += 1;
        match b & TOP2 {
            TAG_ARITY => {
                let arity = (b & LOW6) as usize;
                let mut args = Vec::with_capacity(arity);
                for _ in 0..arity {
                    args.push(Term::decode_at(bytes, pos, next_level));
                }
                Term::App(args)
            }
            TAG_VARREF => Term::Var((b & LOW6) as u32),
            _ => {
                if b == NEWVAR_BYTE {
                    let level = *next_level;
                    *next_level += 1;
                    Term::Var(level)
                } else {
                    let len = (b & LOW6) as usize;
                    let sym = bytes[*pos..*pos + len].to_vec();
                    *pos += len;
                    Term::Sym(sym)
                }
            }
        }
    }

    fn is_ground(&self) -> bool {
        match self {
            Term::Sym(_) => true,
            Term::Var(_) => false,
            Term::App(args) => args.iter().all(Term::is_ground),
        }
    }

    fn var_ids(&self) -> Vec<u32> {
        let mut seen = Vec::new();
        self.collect_vars(&mut seen);
        seen
    }

    fn collect_vars(&self, seen: &mut Vec<u32>) {
        match self {
            Term::Sym(_) => {}
            Term::Var(id) => {
                if !seen.contains(id) {
                    seen.push(*id);
                }
            }
            Term::App(args) => {
                for arg in args {
                    arg.collect_vars(seen);
                }
            }
        }
    }

    fn rename_apart(&self, offset: u32) -> Term {
        match self {
            Term::Sym(bytes) => Term::Sym(bytes.clone()),
            Term::Var(id) => Term::Var(id + offset),
            Term::App(args) => Term::App(args.iter().map(|arg| arg.rename_apart(offset)).collect()),
        }
    }
}

#[derive(Default)]
struct Env {
    bindings: HashMap<u32, Term>,
    trail: Vec<u32>,
}

impl Env {
    fn new() -> Env {
        Env::default()
    }

    fn mark(&self) -> usize {
        self.trail.len()
    }

    fn rollback(&mut self, mark: usize) {
        while self.trail.len() > mark {
            let v = self.trail.pop().unwrap();
            self.bindings.remove(&v);
        }
    }

    fn bind(&mut self, var: u32, term: Term) {
        self.bindings.insert(var, term);
        self.trail.push(var);
    }

    fn walk<'a>(&'a self, term: &'a Term) -> &'a Term {
        let mut cur = term;
        while let Term::Var(var) = cur {
            match self.bindings.get(var) {
                Some(next) => cur = next,
                None => break,
            }
        }
        cur
    }

    fn resolve(&self, term: &Term) -> Term {
        match self.walk(term) {
            Term::Sym(bytes) => Term::Sym(bytes.clone()),
            Term::Var(var) => Term::Var(*var),
            Term::App(args) => Term::App(args.iter().map(|arg| self.resolve(arg)).collect()),
        }
    }

    fn unify(&mut self, a: &Term, b: &Term) -> bool {
        let mark = self.mark();
        if self.unify_rec(a, b) {
            true
        } else {
            self.rollback(mark);
            false
        }
    }

    fn unify_rec(&mut self, a: &Term, b: &Term) -> bool {
        let a = self.walk(a).clone();
        let b = self.walk(b).clone();
        match (&a, &b) {
            (Term::Var(x), Term::Var(y)) if x == y => true,
            (Term::Var(x), _) => {
                if self.occurs(*x, &b) {
                    false
                } else {
                    self.bind(*x, b);
                    true
                }
            }
            (_, Term::Var(y)) => {
                if self.occurs(*y, &a) {
                    false
                } else {
                    self.bind(*y, a);
                    true
                }
            }
            (Term::Sym(p), Term::Sym(q)) => p == q,
            (Term::App(xs), Term::App(ys)) => {
                if xs.len() != ys.len() {
                    return false;
                }
                for (x, y) in xs.iter().zip(ys.iter()) {
                    if !self.unify_rec(x, y) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn occurs(&self, needle: u32, term: &Term) -> bool {
        match self.walk(term) {
            Term::Var(var) => needle == *var,
            Term::Sym(_) => false,
            Term::App(args) => args.iter().any(|arg| self.occurs(needle, arg)),
        }
    }
}

struct Conj {
    patterns: Vec<Term>,
    query_vars: Vec<u32>,
}

fn answer_key(env: &Env, query_vars: &[u32]) -> Vec<u8> {
    Term::App(
        query_vars
            .iter()
            .map(|var| env.resolve(&Term::Var(*var)))
            .collect(),
    )
    .encode()
}

struct Child {
    key: Vec<u8>,
    val: Term,
    sub: Node,
}

struct Node {
    children: Vec<Child>,
    n_ground: usize,
}

impl Node {
    fn leaf() -> Node {
        Node {
            children: Vec::new(),
            n_ground: 0,
        }
    }
}

struct Trie {
    rel_vars: Vec<u32>,
    root: Node,
}

fn insert(node: &mut Node, vals: &[Term]) {
    if vals.is_empty() {
        return;
    }
    let head = &vals[0];
    let idx = match node.children.iter().position(|child| &child.val == head) {
        Some(idx) => idx,
        None => {
            node.children.push(Child {
                key: head.encode(),
                val: head.clone(),
                sub: Node::leaf(),
            });
            node.children.len() - 1
        }
    };
    insert(&mut node.children[idx].sub, &vals[1..]);
}

fn finalize(node: &mut Node) {
    for child in &mut node.children {
        finalize(&mut child.sub);
    }
    node.children
        .sort_by(|a, b| match (a.val.is_ground(), b.val.is_ground()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => a.key.cmp(&b.key),
        });
    node.n_ground = node
        .children
        .iter()
        .filter(|child| child.val.is_ground())
        .count();
}

fn build(vars: &[u32], tuples: &[BTreeMap<u32, Term>], order: &[u32]) -> Trie {
    let rel_vars: Vec<u32> = order
        .iter()
        .copied()
        .filter(|var| vars.contains(var))
        .collect();
    let mut root = Node::leaf();
    for tuple in tuples {
        let vals: Vec<Term> = rel_vars.iter().map(|var| tuple[var].clone()).collect();
        insert(&mut root, &vals);
    }
    finalize(&mut root);
    Trie { rel_vars, root }
}

fn materialize(query: &Conj, space: &[Term]) -> Option<Vec<(Vec<u32>, Vec<BTreeMap<u32, Term>>)>> {
    let mut fresh = DATA_VAR_BASE;
    let mut rels = Vec::with_capacity(query.patterns.len());
    for pattern in &query.patterns {
        let vars = pattern.var_ids();
        let mut tuples = Vec::new();
        let mut seen = BTreeSet::new();
        for fact in space {
            let renamed_fact = fact.rename_apart(fresh);
            fresh += DATA_VAR_STRIDE;
            let mut env = Env::new();
            if env.unify(pattern, &renamed_fact) {
                let mut tuple = BTreeMap::new();
                for &var in &vars {
                    tuple.insert(var, env.resolve(&Term::Var(var)));
                }
                let key = Term::App(vars.iter().map(|var| tuple[var].clone()).collect()).encode();
                if seen.insert(key) {
                    tuples.push(tuple);
                }
            }
        }
        if tuples.is_empty() {
            return None;
        }
        rels.push((vars, tuples));
    }
    Some(rels)
}

fn candidates<'a>(node: &'a Node, vb: &Term) -> Vec<&'a Child> {
    match vb {
        Term::Var(_) => node.children.iter().collect(),
        _ if vb.is_ground() => {
            let key = vb.encode();
            let mut res = Vec::new();
            if let Ok(idx) =
                node.children[..node.n_ground].binary_search_by(|child| child.key.cmp(&key))
            {
                res.push(&node.children[idx]);
            }
            res.extend(node.children[node.n_ground..].iter());
            res
        }
        _ => node.children.iter().collect(),
    }
}

fn leapfrog_unify_join(query: &Conj, space: &[Term]) -> BTreeSet<Vec<u8>> {
    let rels = match materialize(query, space) {
        Some(rels) => rels,
        None => return BTreeSet::new(),
    };
    let order = &query.query_vars;
    let tries: Vec<Trie> = rels
        .iter()
        .map(|(vars, tuples)| build(vars, tuples, order))
        .collect();

    let mut out = BTreeSet::new();
    let mut cursors: Vec<&Node> = tries.iter().map(|trie| &trie.root).collect();
    let mut depths = vec![0usize; tries.len()];
    let mut env = Env::new();
    descend(
        order,
        0,
        &query.query_vars,
        &tries,
        &mut cursors,
        &mut depths,
        &mut env,
        &mut out,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn descend<'a>(
    order: &[u32],
    k: usize,
    query_vars: &[u32],
    tries: &'a [Trie],
    cursors: &mut Vec<&'a Node>,
    depths: &mut [usize],
    env: &mut Env,
    out: &mut BTreeSet<Vec<u8>>,
) {
    if k == order.len() {
        out.insert(answer_key(env, query_vars));
        return;
    }
    let v = order[k];
    let mut parts: Vec<usize> = (0..tries.len())
        .filter(|&i| depths[i] < tries[i].rel_vars.len() && tries[i].rel_vars[depths[i]] == v)
        .collect();
    if parts.is_empty() {
        descend(order, k + 1, query_vars, tries, cursors, depths, env, out);
        return;
    }
    parts.sort_by_key(|&i| cursors[i].children.len());
    intersect(
        order, k, &parts, 0, v, query_vars, tries, cursors, depths, env, out,
    );
}

#[allow(clippy::too_many_arguments)]
fn intersect<'a>(
    order: &[u32],
    k: usize,
    parts: &[usize],
    pi: usize,
    v: u32,
    query_vars: &[u32],
    tries: &'a [Trie],
    cursors: &mut Vec<&'a Node>,
    depths: &mut [usize],
    env: &mut Env,
    out: &mut BTreeSet<Vec<u8>>,
) {
    if pi == parts.len() {
        descend(order, k + 1, query_vars, tries, cursors, depths, env, out);
        return;
    }
    let i = parts[pi];
    let saved_cursor = cursors[i];
    let saved_depth = depths[i];
    let vb = env.resolve(&Term::Var(v));
    for child in candidates(saved_cursor, &vb) {
        let mark = env.mark();
        if env.unify(&Term::Var(v), &child.val) {
            cursors[i] = &child.sub;
            depths[i] = saved_depth + 1;
            intersect(
                order,
                k,
                parts,
                pi + 1,
                v,
                query_vars,
                tries,
                cursors,
                depths,
                env,
                out,
            );
        }
        cursors[i] = saved_cursor;
        depths[i] = saved_depth;
        env.rollback(mark);
    }
}

fn body_patterns_and_query_vars(body: &[u8]) -> (Vec<Term>, Vec<u32>) {
    let body = Term::decode(body);
    let patterns = match body {
        Term::App(mut args) => {
            if !args.is_empty() {
                args.remove(0);
            }
            args
        }
        _ => Vec::new(),
    };
    let query_vars = query_vars(&patterns);
    (patterns, query_vars)
}

fn query_vars(patterns: &[Term]) -> Vec<u32> {
    let mut query_vars = Vec::new();
    for pattern in patterns {
        for var in pattern.var_ids() {
            if !query_vars.contains(&var) {
                query_vars.push(var);
            }
        }
    }
    query_vars
}

/// Run the leapfrog-unification join over an encoded conjunction body and encoded facts.
pub fn leapfrog_unify_join_encoded(
    body: &[u8],
    facts: &[&[u8]],
) -> std::collections::BTreeSet<Vec<u8>> {
    let (patterns, query_vars) = body_patterns_and_query_vars(body);
    let query = Conj {
        patterns,
        query_vars,
    };
    let space: Vec<Term> = facts.iter().map(|fact| Term::decode(fact)).collect();
    leapfrog_unify_join(&query, &space)
}

/// Return the number of distinct query variables in an encoded conjunction body.
pub fn body_var_count(body: &[u8]) -> usize {
    let (_, query_vars) = body_patterns_and_query_vars(body);
    query_vars.len()
}

/// Split an encoded arity tuple into canonically re-encoded child terms.
pub fn split_tuple(tuple: &[u8], n: usize) -> Vec<Vec<u8>> {
    let tuple = Term::decode(tuple);
    let Term::App(args) = tuple else {
        panic!("split_tuple expected an Arity({n}) expression");
    };
    assert_eq!(args.len(), n, "tuple arity mismatch");
    args.into_iter().map(|arg| arg.encode()).collect()
}

/// Return true when the encoded term contains no variables.
pub fn term_is_ground(bytes: &[u8]) -> bool {
    Term::decode(bytes).is_ground()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mork_uni_join::oracle::Conj as ProtoConj;
    use mork_uni_join::term::Term as ProtoTerm;
    use mork_uni_join::unijoin::leapfrog_unify_join as proto_join;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn chance(&mut self, num: usize, den: usize) -> bool {
            self.below(den) < num
        }
    }

    const SYMS: &[&str] = &["a", "b", "c", "d"];

    fn proto_query_vars(patterns: &[ProtoTerm]) -> Vec<u32> {
        let mut vars = Vec::new();
        for pattern in patterns {
            for var in pattern.var_ids() {
                if !vars.contains(&var) {
                    vars.push(var);
                }
            }
        }
        vars
    }

    fn gen_proto_term(rng: &mut Rng, depth: usize, allow_var: bool, var_pool: usize) -> ProtoTerm {
        if depth == 0 || rng.chance(3, 5) {
            if allow_var && rng.chance(2, 5) {
                ProtoTerm::Var(rng.below(var_pool) as u32)
            } else {
                ProtoTerm::sym(SYMS[rng.below(SYMS.len())])
            }
        } else {
            let arity = 1 + rng.below(2);
            ProtoTerm::App(
                (0..arity)
                    .map(|_| gen_proto_term(rng, depth - 1, allow_var, var_pool))
                    .collect(),
            )
        }
    }

    fn gen_proto_factor(rng: &mut Rng, allow_var: bool, var_pool: usize) -> ProtoTerm {
        ProtoTerm::App(vec![
            ProtoTerm::sym("r"),
            gen_proto_term(rng, 2, allow_var, var_pool),
            gen_proto_term(rng, 1, allow_var, var_pool),
        ])
    }

    #[test]
    fn split_tuple_reencodes_ground_children() {
        let tuple = Term::App(vec![
            Term::Sym(b"a".to_vec()),
            Term::App(vec![Term::Sym(b"b".to_vec()), Term::Sym(b"c".to_vec())]),
        ])
        .encode();
        let children = split_tuple(&tuple, 2);
        assert_eq!(
            children,
            vec![
                Term::Sym(b"a".to_vec()).encode(),
                Term::App(vec![Term::Sym(b"b".to_vec()), Term::Sym(b"c".to_vec())]).encode(),
            ]
        );
    }

    #[test]
    fn split_tuple_reencodes_variable_children_as_standalone_terms() {
        let tuple = Term::App(vec![
            Term::Var(7),
            Term::App(vec![Term::Sym(b"f".to_vec()), Term::Var(7)]),
        ])
        .encode();
        let children = split_tuple(&tuple, 2);
        assert_eq!(
            children,
            vec![
                Term::Var(0).encode(),
                Term::App(vec![Term::Sym(b"f".to_vec()), Term::Var(0)]).encode(),
            ]
        );
    }

    #[test]
    fn term_is_ground_detects_variables_and_accepts_raw_symbol_bytes() {
        let ground = Term::App(vec![Term::Sym(vec![0xff]), Term::Sym(b"a".to_vec())]).encode();
        let nonground = Term::App(vec![Term::Sym(b"a".to_vec()), Term::Var(0)]).encode();
        assert!(term_is_ground(&ground));
        assert!(!term_is_ground(&nonground));
    }

    #[test]
    fn random_encoded_join_matches_prototype_byte_for_byte() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut nonempty = 0usize;
        let mut schematic_answers = 0usize;

        for case in 0..5000 {
            let npat = 1 + rng.below(3);
            let var_pool = 1 + rng.below(3);
            let patterns: Vec<ProtoTerm> = (0..npat)
                .map(|_| gen_proto_factor(&mut rng, true, var_pool))
                .collect();
            let query_vars = proto_query_vars(&patterns);
            let query = ProtoConj {
                patterns: patterns.clone(),
                query_vars: query_vars.clone(),
            };

            let nfacts = rng.below(6);
            let facts: Vec<ProtoTerm> = (0..nfacts)
                .map(|_| {
                    let schematic = rng.chance(1, 2);
                    gen_proto_factor(&mut rng, schematic, 2)
                })
                .collect();

            let body = {
                let mut args = Vec::with_capacity(patterns.len() + 1);
                args.push(ProtoTerm::sym(","));
                args.extend(patterns.iter().cloned());
                ProtoTerm::App(args).encode()
            };
            let fact_bytes: Vec<Vec<u8>> = facts.iter().map(ProtoTerm::encode).collect();
            let fact_slices: Vec<&[u8]> = fact_bytes.iter().map(Vec::as_slice).collect();

            let expected = proto_join(&query, &facts);
            let got = leapfrog_unify_join_encoded(&body, &fact_slices);
            assert_eq!(
                got, expected,
                "case {case}: native encoded join differed from prototype"
            );
            assert_eq!(body_var_count(&body), query_vars.len());

            if !got.is_empty() {
                nonempty += 1;
            }
            if got
                .iter()
                .any(|key| key.iter().any(|&byte| byte == NEWVAR_BYTE))
            {
                schematic_answers += 1;
            }
        }

        assert!(
            nonempty > 100,
            "random corpus produced too few answers: {nonempty}"
        );
        assert!(
            schematic_answers > 10,
            "random corpus produced too few schematic answers: {schematic_answers}"
        );
    }
}
