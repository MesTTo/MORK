//! Factorized binding relations: a d-representation of a query answer set.
//!
//! A factorized result represents unions and Cartesian products without
//! repeating complete rows, so it can be exponentially smaller than the flat row
//! list (Olteanu and Zavodny, factorized databases). Consumers like COUNT and
//! EXISTS read the DAG without enumerating every binding: `row_count` is linear
//! in the DAG, not the answer set. Nodes are hash-consed, so shared subtrees are
//! stored once.
//!
//! Factorized answer DAGs over MORK's `BindingRelation` / `BindingVar` /
//! `TermId`. The shapes are:
//!
//! - `Empty` (no rows), `Unit { weight }` (one row carrying a signed weight),
//! - `Trie { variable, branches }` (a value-indexed branch on one variable, the
//!   prefix-sharing that `from_relation` builds),
//! - `Union` (sum of disjoint child answer sets) and `Product` (Cartesian product
//!   of children over disjoint variables).

use std::collections::{BTreeMap, HashMap};

use crate::binding_space::{BindingRelation, BindingVar};
use crate::term_identity::TermId;

/// Identity of a node in a [`BindingDag`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingNodeId(pub u32);

impl BindingNodeId {
    #[inline]
    fn index(self) -> usize {
        self.0 as usize
    }
}

/// One factorized node.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum BindingNode {
    Empty,
    Unit {
        weight: i64,
    },
    Trie {
        variable: BindingVar,
        branches: Box<[(TermId, BindingNodeId)]>,
    },
    Union(Box<[BindingNodeId]>),
    Product(Box<[BindingNodeId]>),
}

/// A hash-consed DAG of factorized binding nodes.
#[derive(Clone, Debug, Default)]
pub struct BindingDag {
    nodes: Vec<BindingNode>,
    exact: HashMap<BindingNode, BindingNodeId>,
}

impl BindingDag {
    /// A fresh DAG holding only the `Empty` node (id 0).
    pub fn new() -> Self {
        let mut dag = Self::default();
        dag.intern(BindingNode::Empty);
        dag
    }

    pub fn node(&self, id: BindingNodeId) -> &BindingNode {
        &self.nodes[id.index()]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }

    /// Intern a node, returning the shared id of an identical existing node.
    pub fn intern(&mut self, node: BindingNode) -> BindingNodeId {
        if let Some(id) = self.exact.get(&node) {
            return *id;
        }
        let id = BindingNodeId(self.nodes.len() as u32);
        self.nodes.push(node.clone());
        self.exact.insert(node, id);
        id
    }

    pub fn empty(&self) -> BindingNodeId {
        BindingNodeId(0)
    }

    /// A unit node carrying a signed weight; weight 0 collapses to `Empty`.
    pub fn unit(&mut self, weight: i64) -> BindingNodeId {
        if weight == 0 {
            self.empty()
        } else {
            self.intern(BindingNode::Unit { weight })
        }
    }

    /// Union of child answer sets, flattening nested unions, dropping `Empty`,
    /// and deduplicating shared children. A single survivor returns directly.
    pub fn union(&mut self, children: impl IntoIterator<Item = BindingNodeId>) -> BindingNodeId {
        let mut flattened = Vec::new();
        for child in children {
            if child == self.empty() {
                continue;
            }
            match self.node(child) {
                BindingNode::Union(nested) => flattened.extend(nested.iter().copied()),
                _ => flattened.push(child),
            }
        }
        flattened.sort();
        flattened.dedup();
        match flattened.len() {
            0 => self.empty(),
            1 => flattened[0],
            _ => self.intern(BindingNode::Union(flattened.into_boxed_slice())),
        }
    }

    /// Cartesian product of children, flattening nested products, dropping the
    /// `Unit { weight: 1 }` identity, and short-circuiting to `Empty` on any
    /// empty factor.
    pub fn product(&mut self, children: impl IntoIterator<Item = BindingNodeId>) -> BindingNodeId {
        let mut flattened = Vec::new();
        for child in children {
            if child == self.empty() {
                return self.empty();
            }
            match self.node(child) {
                BindingNode::Product(nested) => flattened.extend(nested.iter().copied()),
                BindingNode::Unit { weight: 1 } => {}
                _ => flattened.push(child),
            }
        }
        match flattened.len() {
            0 => self.unit(1),
            1 => flattened[0],
            _ => self.intern(BindingNode::Product(flattened.into_boxed_slice())),
        }
    }

    /// Factorize a relation into a value-indexed trie over `order`, then any
    /// remaining schema variables. `order` variables must be in the schema.
    /// Equal subtrees are hash-consed, so a shared suffix is stored once.
    pub fn from_relation(
        relation: &BindingRelation,
        order: &[BindingVar],
    ) -> (Self, BindingNodeId) {
        let mut dag = Self::new();
        let mut full_order = order.to_vec();
        for variable in relation.schema() {
            if !full_order.contains(variable) {
                full_order.push(*variable);
            }
        }
        let positions = full_order
            .iter()
            .map(|variable| {
                relation
                    .schema()
                    .iter()
                    .position(|candidate| candidate == variable)
                    .expect("order variable must be in the relation schema")
            })
            .collect::<Vec<_>>();
        let rows = relation
            .rows()
            .map(|(row, weight)| {
                (
                    positions
                        .iter()
                        .map(|&position| row[position])
                        .collect::<Vec<_>>(),
                    weight,
                )
            })
            .collect::<Vec<_>>();

        fn build(
            dag: &mut BindingDag,
            rows: &[(Vec<TermId>, i64)],
            order: &[BindingVar],
            depth: usize,
        ) -> BindingNodeId {
            if rows.is_empty() {
                return dag.empty();
            }
            if depth == order.len() {
                let weight = rows.iter().map(|(_, weight)| *weight).sum();
                return dag.unit(weight);
            }
            let mut groups: BTreeMap<TermId, Vec<(Vec<TermId>, i64)>> = BTreeMap::new();
            for (row, weight) in rows {
                groups
                    .entry(row[depth])
                    .or_default()
                    .push((row.clone(), *weight));
            }
            let mut branches = Vec::new();
            for (value, group) in groups {
                let child = build(dag, &group, order, depth + 1);
                if child != dag.empty() {
                    branches.push((value, child));
                }
            }
            if branches.is_empty() {
                dag.empty()
            } else {
                dag.intern(BindingNode::Trie {
                    variable: order[depth],
                    branches: branches.into_boxed_slice(),
                })
            }
        }

        let root = build(&mut dag, &rows, &full_order, 0);
        (dag, root)
    }

    /// Number of rows the DAG represents, computed without enumerating them:
    /// trie and union sum their children, product multiplies, all memoized, so
    /// the cost is linear in the DAG, not the answer set.
    pub fn row_count(&self, root: BindingNodeId) -> u128 {
        fn recurse(
            dag: &BindingDag,
            id: BindingNodeId,
            memo: &mut HashMap<BindingNodeId, u128>,
        ) -> u128 {
            if let Some(value) = memo.get(&id) {
                return *value;
            }
            let value = match dag.node(id) {
                BindingNode::Empty => 0,
                BindingNode::Unit { weight } => u128::from(*weight != 0),
                BindingNode::Trie { branches, .. } => branches
                    .iter()
                    .map(|(_, child)| recurse(dag, *child, memo))
                    .sum(),
                BindingNode::Union(children) => children
                    .iter()
                    .map(|child| recurse(dag, *child, memo))
                    .sum(),
                BindingNode::Product(children) => children
                    .iter()
                    .map(|child| recurse(dag, *child, memo))
                    .product(),
            };
            memo.insert(id, value);
            value
        }
        recurse(self, root, &mut HashMap::new())
    }

    /// Materialize every row as a variable-to-value map. Product children are
    /// cross-joined, skipping any pair that would bind a variable twice.
    pub fn enumerate(&self, root: BindingNodeId) -> Vec<BTreeMap<BindingVar, TermId>> {
        fn recurse(dag: &BindingDag, id: BindingNodeId) -> Vec<BTreeMap<BindingVar, TermId>> {
            match dag.node(id) {
                BindingNode::Empty => Vec::new(),
                BindingNode::Unit { weight } => {
                    if *weight == 0 {
                        Vec::new()
                    } else {
                        vec![BTreeMap::new()]
                    }
                }
                BindingNode::Trie { variable, branches } => {
                    let mut output = Vec::new();
                    for (value, child) in branches.iter().copied() {
                        for mut row in recurse(dag, child) {
                            row.insert(*variable, value);
                            output.push(row);
                        }
                    }
                    output
                }
                BindingNode::Union(children) => children
                    .iter()
                    .flat_map(|child| recurse(dag, *child))
                    .collect(),
                BindingNode::Product(children) => {
                    let mut rows = vec![BTreeMap::new()];
                    for child in children.iter().copied() {
                        let right = recurse(dag, child);
                        let mut next = Vec::new();
                        for left_row in &rows {
                            for right_row in &right {
                                if left_row.keys().any(|key| right_row.contains_key(key)) {
                                    continue;
                                }
                                let mut row = left_row.clone();
                                row.extend(right_row);
                                next.push(row);
                            }
                        }
                        rows = next;
                    }
                    rows
                }
            }
        }
        recurse(self, root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relation(rows: &[[u64; 3]]) -> BindingRelation {
        let mut relation = BindingRelation::new(vec![BindingVar(0), BindingVar(1), BindingVar(2)]);
        for row in rows {
            relation
                .add(row.iter().map(|&v| TermId(v)).collect::<Vec<_>>(), 1)
                .unwrap();
        }
        relation
    }

    #[test]
    fn factorization_preserves_rows_and_counts() {
        let relation = relation(&[[1, 10, 20], [2, 10, 20], [1, 11, 21]]);
        let (dag, root) =
            BindingDag::from_relation(&relation, &[BindingVar(1), BindingVar(0), BindingVar(2)]);
        assert_eq!(dag.row_count(root), 3);
        assert_eq!(dag.enumerate(root).len(), 3);
        assert!(dag.len() < 20);
    }

    #[test]
    fn shared_suffix_is_hash_consed() {
        // Three rows share the suffix (10, 20); the trie stores that suffix once,
        // so the node count is well under three full paths.
        let relation = relation(&[[1, 10, 20], [2, 10, 20], [3, 10, 20]]);
        let (dag, root) =
            BindingDag::from_relation(&relation, &[BindingVar(0), BindingVar(1), BindingVar(2)]);
        assert_eq!(dag.row_count(root), 3);
        assert_eq!(dag.enumerate(root).len(), 3);
        // Empty, the shared unit, the shared (11->...) chain: the three rows do
        // not cost three independent depth-3 paths.
        assert!(
            dag.len() <= 6,
            "shared suffix should collapse, got {}",
            dag.len()
        );
    }

    #[test]
    fn product_counts_without_enumerating() {
        // Two independent columns: a 5-value variable and a 4-value variable.
        // Their product is 20 rows, but the factorized DAG stays tiny.
        let mut dag = BindingDag::new();
        let unit = dag.unit(1);
        let col0 = dag.intern(BindingNode::Trie {
            variable: BindingVar(0),
            branches: (1..=5).map(|v| (TermId(v), unit)).collect(),
        });
        let col1 = dag.intern(BindingNode::Trie {
            variable: BindingVar(1),
            branches: (10..=13).map(|v| (TermId(v), unit)).collect(),
        });
        let root = dag.product([col0, col1]);

        assert_eq!(dag.row_count(root), 20);
        assert_eq!(dag.enumerate(root).len(), 20);
        // The whole DAG is a handful of nodes, not 20 rows.
        assert!(
            dag.len() < 10,
            "factorized DAG should be small, got {}",
            dag.len()
        );
    }
}
