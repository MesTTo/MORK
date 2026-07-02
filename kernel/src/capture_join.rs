//! The capture-aware unification join over the LIVE PathMap.
//!
//! MORK's flat leapfrog (`zipper_join`) declines a body once a non-ground compound is in
//! play and falls back to the materialized ProductZipper. This routes that case to the
//! prototype's capture join (`mork_uni_join::trie_join`), run directly over the live
//! byte-trie by implementing its four-method `SubtermZip` contract for a PathMap
//! `ReadZipper`. No data is copied and there is no second join implementation: the same
//! descent that is sealed against SWI-Prolog occurs-check (the prototype's `prolog_seal`)
//! runs here over the real space.

use mork_uni_join::oracle::Conj;
use mork_uni_join::term::Term as PTerm;
use mork_uni_join::trie::SubtermZip;
use mork_uni_join::trie_join::unify_join_z;
use pathmap::zipper::{Zipper, ZipperMoving};
use pathmap::PathMap;
use std::collections::BTreeSet;

/// Stored (data) variables get ids from here up, disjoint from the small query-variable ids.
const DATA_VAR_BASE: u32 = 1_000_000;

/// Adapts a PathMap `ReadZipper` to the prototype's `SubtermZip` contract. The four methods
/// are a direct rename of the PathMap zipper API, so the capture join descends the live trie
/// exactly as it descends the prototype's `ByteTrie`.
struct PathMapZip<Z>(Z);

impl<Z: Zipper + ZipperMoving> SubtermZip for PathMapZip<Z> {
    fn child_mask_words(&self) -> [u64; 4] {
        self.0.child_mask().0
    }
    fn descend_byte(&mut self, b: u8) -> bool {
        self.0.descend_to_existing_byte(b)
    }
    fn ascend(&mut self) {
        self.0.ascend_byte();
    }
    fn save_path(&self) -> Vec<u8> {
        self.0.path().to_vec()
    }
    fn restore_path(&mut self, path: &[u8]) {
        self.0.reset();
        self.0.descend_to(path);
    }
}

/// Build the prototype conjunctive query from an encoded `(, p1 .. pk)` body: drop the `,`
/// head, keep the factors, number the head variables in first-occurrence order (the order
/// the answer tuple is keyed by).
pub fn conj_from_body(body: &[u8]) -> Conj {
    let factors: Vec<PTerm> = match PTerm::decode(body) {
        PTerm::App(mut a) => {
            if !a.is_empty() {
                a.remove(0);
            }
            a
        }
        _ => Vec::new(),
    };
    let mut query_vars = Vec::new();
    for p in &factors {
        for v in p.var_ids() {
            if !query_vars.contains(&v) {
                query_vars.push(v);
            }
        }
    }
    Conj { patterns: factors, query_vars }
}

/// All answers to the conjunctive query `q` against the live PathMap `map`, under full
/// unification with data-side capture, as canonical answer-tuple keys (the prototype's
/// encoding over `q.query_vars`). Byte-identical to `trie_unify_join` over the same facts,
/// hence to `leapfrog_unify_join` and to SWI-Prolog occurs-check.
pub fn capture_join_live(map: &PathMap<()>, q: &Conj) -> BTreeSet<Vec<u8>> {
    unify_join_z(PathMapZip(map.read_zipper()), q, DATA_VAR_BASE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::Space;
    use mork_uni_join::corpus;
    use mork_uni_join::randgen::{gen_facts, gen_query, Rng};
    use mork_uni_join::term::parse as pparse;
    use mork_uni_join::term::Term as PTermT;
    use mork_uni_join::trie_join::trie_unify_join;
    use mork_uni_join::unijoin::leapfrog_unify_join;

    /// Add the facts (as their s-expression text) to a fresh space, returning it.
    fn space_of(facts: &[PTermT]) -> Space {
        let mut space = Space::new();
        let mut program = String::new();
        for f in facts {
            program.push_str(&f.to_string());
            program.push('\n');
        }
        space.add_all_sexpr(program.as_bytes()).unwrap();
        space
    }

    /// The live-PathMap capture join must equal the prototype's in-process trie join (and the
    /// materialized leapfrog) on every genuine-unification corpus case. This exercises the
    /// PathMap `SubtermZip` impl: same engine, real space, no copy.
    #[test]
    fn live_capture_join_matches_prototype_on_corpus() {
        for case in corpus::cases() {
            // Build a live space holding exactly the facts.
            let mut space = Space::new();
            let mut program = String::new();
            for f in case.facts {
                program.push_str(f);
                program.push('\n');
            }
            space.add_all_sexpr(program.as_bytes()).unwrap();

            let q = Conj::parse(case.patterns);
            let facts: Vec<PTerm> = case.facts.iter().map(|f| pparse(f)).collect();

            let live = capture_join_live(&space.btm, &q);
            let proto = trie_unify_join(&q, &facts);
            let leap = leapfrog_unify_join(&q, &facts);

            assert_eq!(live, proto, "case {:?}: live PathMap join != prototype trie join", case.name);
            assert_eq!(live, leap, "case {:?}: live PathMap join != materialized leapfrog", case.name);
        }
    }

    /// The same agreement over a large random schematic distribution: build a real space from
    /// random facts, run the join over the live PathMap, and compare to the prototype's
    /// in-process trie join. This exercises the PathMap zipper adapter (path/reset/child_mask)
    /// on arbitrary structure, not just the hand-written corpus shapes.
    #[test]
    fn live_capture_join_matches_prototype_random() {
        let mut rng = Rng(0xD1B54A32D192ED03);
        let trials = 4000;
        let mut nonempty = 0usize;
        let mut nonground = 0usize;
        for i in 0..trials {
            let q = gen_query(&mut rng);
            let facts = gen_facts(&mut rng);
            let space = space_of(&facts);
            let live = capture_join_live(&space.btm, &q);
            let proto = trie_unify_join(&q, &facts);
            assert_eq!(
                live, proto,
                "trial {i}: live PathMap join != prototype\n  query={:?}\n  facts={:?}",
                q.patterns.iter().map(|t| t.to_string()).collect::<Vec<_>>(),
                facts.iter().map(|t| t.to_string()).collect::<Vec<_>>(),
            );
            if !live.is_empty() {
                nonempty += 1;
            }
            if live.iter().any(|k| !PTermT::decode(k).is_ground()) {
                nonground += 1;
            }
        }
        eprintln!("live random differential: {trials} trials, {nonempty} non-empty, {nonground} capture");
        assert!(nonempty > trials / 10, "too few non-empty results ({nonempty}/{trials})");
        assert!(nonground > 25, "too few capture (non-ground) results ({nonground})");
    }
}
