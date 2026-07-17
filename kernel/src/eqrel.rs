//! Equivalence-relation state for the `eqrel` sink.
//!
//! The sink is insert-only. It consumes union events and writes quotient facts;
//! it does not retract older quotient facts if later unions choose a smaller
//! representative. Use it at a monotone boundary where the union relation only
//! grows, the same boundary required by the semi-naive immediate-consequence
//! path.
//!
//! Example:
//!
//! ```text
//! (edge a b)
//! (edge b c)
//! (exec 0 (, (edge $x $y))
//!         (O (eqrel $x $y (quot $e $rep))))
//! (exec 1 (, (quot $x $r) (quot $y $r))
//!         (, (same $x $y)))
//! ```
//!
//! The first rule emits `(quot a a)`, `(quot b a)`, and `(quot c a)`. The
//! second rule can answer same-class queries by joining on the quotient
//! representative, without materializing every closure pair up front.

use std::collections::HashMap;

use crate::union_find::{UnionFind, UnionFindId};

/// Dense element id in one eqrel sink instance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct EqRelId(u32);

impl UnionFindId for EqRelId {
    fn from_index(index: u32) -> Self {
        Self(index)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// Union-find-backed equivalence relation over encoded MORK expressions.
#[derive(Clone, Debug, Default)]
pub struct EqRel {
    ids: HashMap<Box<[u8]>, EqRelId>,
    elements: Vec<Box<[u8]>>,
    classes: UnionFind<EqRelId>,
}

impl EqRel {
    pub fn new() -> Self {
        Self {
            ids: HashMap::new(),
            elements: Vec::new(),
            classes: UnionFind::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Interns an encoded expression to a dense id.
    fn intern(&mut self, bytes: &[u8]) -> EqRelId {
        if let Some(id) = self.ids.get(bytes).copied() {
            return id;
        }
        let id = self.classes.make_set();
        let owned: Box<[u8]> = bytes.into();
        self.elements.push(owned.clone());
        self.ids.insert(owned, id);
        id
    }

    /// Adds a union event for two encoded expressions.
    pub fn union(&mut self, left: &[u8], right: &[u8]) -> bool {
        let left = self.intern(left);
        let right = self.intern(right);
        self.classes.union(left, right).1
    }

    /// Emits the deterministic quotient: element -> byte-smallest class member.
    pub fn for_each_quotient<F>(&self, mut f: F)
    where
        F: FnMut(&[u8], &[u8]),
    {
        let mut mins = vec![None::<usize>; self.elements.len()];
        for id in self.classes.ids() {
            let root = self.classes.find(id);
            let slot = &mut mins[root.index()];
            if slot
                .map(|current| self.elements[id.index()] < self.elements[current])
                .unwrap_or(true)
            {
                *slot = Some(id.index());
            }
        }

        let mut ids: Vec<usize> = (0..self.elements.len()).collect();
        ids.sort_unstable_by(|left, right| self.elements[*left].cmp(&self.elements[*right]));
        for id in ids {
            let root = self.classes.find(EqRelId(id as u32));
            let rep = mins[root.index()].expect("class root must have a minimum element");
            f(&self.elements[id], &self.elements[rep]);
        }
    }

    /// Owned quotient pairs for tests and small callers.
    pub fn quotient_pairs(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut pairs = Vec::with_capacity(self.len());
        self.for_each_quotient(|element, rep| pairs.push((element.to_vec(), rep.to_vec())));
        pairs
    }
}
