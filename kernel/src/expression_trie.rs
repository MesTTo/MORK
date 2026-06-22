use std::collections::BTreeMap;

use crate::pattern_relations::{
    PatternLoweringError, PatternRelationMatchError, PatternRelationMatches, lower_pattern,
    match_fact_ids,
};
use crate::term_identity::{FactId, TermId, TermIdentitySidecar, TermKind};

/// Typed preorder token used by the derived expression trie.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ExpressionTrieToken {
    /// Application node with encoded arity.
    App(u8),
    /// Complete interned symbol item.
    Symbol(TermId),
    /// Stored schematic new-variable token.
    NewVar,
    /// Stored schematic variable reference token.
    VarRef(u8),
}

/// Snapshot-local discrimination-trie style index over canonical term roots.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpressionTrieIndex {
    nodes: Vec<ExpressionTrieNode>,
    feature_postings: BTreeMap<ExpressionFeature, Vec<FactId>>,
    stats: ExpressionTrieStats,
}

/// Counters for expression-trie build and candidate lookup.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpressionTrieStats {
    /// Source term-sidecar generation used to build this snapshot-local index.
    pub source_generation: u32,
    /// Cheap order-independent fingerprint of indexed fact roots.
    ///
    /// This is diagnostic only: exact `TermId` matching remains the semantic
    /// gate, and this fingerprint must not be used as a correctness proof.
    pub source_fingerprint: u128,
    /// Complete facts indexed.
    pub facts_indexed: usize,
    /// Trie nodes allocated, including the root.
    pub trie_nodes: usize,
    /// Typed preorder tokens inserted across all facts.
    pub tokens_indexed: usize,
    /// Structural ground features inserted across all facts.
    pub features_indexed: usize,
    /// Distinct structural feature posting lists.
    pub feature_postings: usize,
}

/// Structural feature used to narrow candidate facts.
///
/// Positions are child-index paths from the root. This keeps sibling constants
/// useful even when an earlier sibling in preorder is a variable-sized pattern
/// variable.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ExpressionFeature {
    /// Child-index path from the root term to this feature.
    pub position: Box<[u8]>,
    /// Exact typed token required at `position`.
    pub token: ExpressionTrieToken,
}

/// Candidate lookup result before exact filtering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpressionTrieCandidates {
    /// Conservative typed prefix used for trie descent.
    pub prefix: Box<[ExpressionTrieToken]>,
    /// Ground structural features used for posting-list intersection.
    pub features: Box<[ExpressionFeature]>,
    /// Complete facts below the prefix.
    pub facts: Box<[FactId]>,
}

/// Match result from expression-trie candidate retrieval plus exact filtering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpressionTrieMatches {
    /// Prefix-filter candidates.
    pub candidates: ExpressionTrieCandidates,
    /// Exact relationalized pattern matches over the candidate facts.
    pub exact: PatternRelationMatches,
}

/// Errors from expression-trie construction or matching.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpressionTrieError {
    /// A term referenced by a fact or pattern is absent from the sidecar.
    UnknownTerm { term: TermId },
    /// Pattern lowering failed.
    Lowering(PatternLoweringError),
    /// Exact candidate filtering failed.
    Match(PatternRelationMatchError),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ExpressionTrieNode {
    children: BTreeMap<ExpressionTrieToken, usize>,
    facts: Vec<FactId>,
}

impl ExpressionTrieIndex {
    /// Builds a derived typed expression trie from complete fact roots.
    pub fn build(sidecar: &TermIdentitySidecar) -> Result<Self, ExpressionTrieError> {
        let sidecar_stats = sidecar.stats();
        let mut index = Self {
            nodes: vec![ExpressionTrieNode::default()],
            feature_postings: BTreeMap::new(),
            stats: ExpressionTrieStats {
                source_generation: sidecar_stats.generation,
                trie_nodes: 1,
                ..ExpressionTrieStats::default()
            },
        };

        for fact in sidecar.facts() {
            if !sidecar.is_fact_live(fact.id) {
                continue;
            }
            index.insert_fact(sidecar, fact.id, fact.root)?;
        }
        for postings in index.feature_postings.values_mut() {
            postings.sort_unstable();
            postings.dedup();
        }

        Ok(index)
    }

    /// Build and lookup counters.
    pub fn stats(&self) -> ExpressionTrieStats {
        self.stats
    }

    /// Returns candidate fact IDs for the grounded typed prefix of `pattern`.
    ///
    /// Pattern variables stop prefix extraction because they match complete
    /// subterms of unknown length. Later constants are checked by the exact
    /// relationalized matcher.
    pub fn candidates_for_pattern(
        &self,
        sidecar: &TermIdentitySidecar,
        pattern: TermId,
    ) -> Result<ExpressionTrieCandidates, ExpressionTrieError> {
        let mut prefix = Vec::new();
        append_ground_prefix(sidecar, pattern, &mut prefix)?;
        let mut features = Vec::new();
        append_ground_features(sidecar, pattern, &mut Vec::new(), &mut features)?;
        features.sort();
        features.dedup();
        let facts = self.facts_below_prefix_and_features(&prefix, &features);

        Ok(ExpressionTrieCandidates {
            prefix: prefix.into_boxed_slice(),
            features: features.into_boxed_slice(),
            facts: facts.into_boxed_slice(),
        })
    }

    /// Prefix-filtered exact matching for one pattern term.
    pub fn match_pattern(
        &self,
        sidecar: &TermIdentitySidecar,
        pattern: TermId,
    ) -> Result<ExpressionTrieMatches, ExpressionTrieError> {
        let candidates = self.candidates_for_pattern(sidecar, pattern)?;
        let plan = lower_pattern(sidecar, pattern).map_err(ExpressionTrieError::Lowering)?;
        let exact = match_fact_ids(sidecar, &plan, candidates.facts.iter().copied())
            .map_err(ExpressionTrieError::Match)?;

        Ok(ExpressionTrieMatches { candidates, exact })
    }

    fn insert_fact(
        &mut self,
        sidecar: &TermIdentitySidecar,
        fact: FactId,
        root: TermId,
    ) -> Result<(), ExpressionTrieError> {
        let Some(root_record) = sidecar.get_term(root) else {
            return Err(ExpressionTrieError::UnknownTerm { term: root });
        };
        let mut features = Vec::new();
        append_exact_features(sidecar, root, &mut Vec::new(), &mut features)?;
        let path = features
            .iter()
            .map(|feature| feature.token)
            .collect::<Vec<_>>();

        let mut node = 0usize;
        for token in path {
            self.stats.tokens_indexed += 1;
            if let Some(&child) = self.nodes[node].children.get(&token) {
                node = child;
                continue;
            }

            let child = self.nodes.len();
            self.nodes.push(ExpressionTrieNode::default());
            self.nodes[node].children.insert(token, child);
            self.stats.trie_nodes += 1;
            node = child;
        }

        for feature in features {
            let postings = self.feature_postings.entry(feature).or_insert_with(|| {
                self.stats.feature_postings += 1;
                Vec::new()
            });
            postings.push(fact);
            self.stats.features_indexed += 1;
        }

        self.stats.source_fingerprint = self.stats.source_fingerprint.wrapping_add(
            root_record
                .structural_hash
                .rotate_left(u32::from(fact.0 % 127) + 1),
        );
        self.nodes[node].facts.push(fact);
        self.stats.facts_indexed += 1;
        Ok(())
    }

    fn facts_below_prefix_and_features(
        &self,
        prefix: &[ExpressionTrieToken],
        features: &[ExpressionFeature],
    ) -> Vec<FactId> {
        let mut facts = self.facts_below_prefix(prefix);
        if facts.is_empty() || features.is_empty() {
            return facts;
        }

        let mut features_by_selectivity = features.iter().collect::<Vec<_>>();
        features_by_selectivity.sort_by_key(|feature| {
            self.feature_postings
                .get(*feature)
                .map(|postings| postings.len())
                .unwrap_or(0)
        });

        for feature in features_by_selectivity {
            let Some(postings) = self.feature_postings.get(feature) else {
                return Vec::new();
            };
            facts = intersect_sorted_fact_ids(&facts, postings);
            if facts.is_empty() {
                break;
            }
        }

        facts
    }

    fn facts_below_prefix(&self, prefix: &[ExpressionTrieToken]) -> Vec<FactId> {
        let mut node = 0usize;
        for token in prefix {
            let Some(&child) = self.nodes[node].children.get(token) else {
                return Vec::new();
            };
            node = child;
        }

        let mut facts = Vec::new();
        self.collect_facts(node, &mut facts);
        facts.sort_unstable();
        facts
    }

    fn collect_facts(&self, node: usize, facts: &mut Vec<FactId>) {
        facts.extend_from_slice(&self.nodes[node].facts);
        for &child in self.nodes[node].children.values() {
            self.collect_facts(child, facts);
        }
    }
}

fn intersect_sorted_fact_ids(left: &[FactId], right: &[FactId]) -> Vec<FactId> {
    let mut result = Vec::new();
    let mut left_index = 0usize;
    let mut right_index = 0usize;

    while let (Some(&left_id), Some(&right_id)) = (left.get(left_index), right.get(right_index)) {
        match left_id.cmp(&right_id) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Equal => {
                result.push(left_id);
                left_index += 1;
                right_index += 1;
            }
            std::cmp::Ordering::Greater => right_index += 1,
        }
    }

    result
}

fn append_exact_features(
    sidecar: &TermIdentitySidecar,
    term: TermId,
    position: &mut Vec<u8>,
    out: &mut Vec<ExpressionFeature>,
) -> Result<(), ExpressionTrieError> {
    append_features(sidecar, term, position, out, true)
}

fn append_ground_features(
    sidecar: &TermIdentitySidecar,
    term: TermId,
    position: &mut Vec<u8>,
    out: &mut Vec<ExpressionFeature>,
) -> Result<(), ExpressionTrieError> {
    append_features(sidecar, term, position, out, false)
}

fn append_features(
    sidecar: &TermIdentitySidecar,
    term: TermId,
    position: &mut Vec<u8>,
    out: &mut Vec<ExpressionFeature>,
    include_variables: bool,
) -> Result<(), ExpressionTrieError> {
    let Some(record) = sidecar.get_term(term) else {
        return Err(ExpressionTrieError::UnknownTerm { term });
    };

    match record.kind {
        TermKind::Symbol => out.push(ExpressionFeature {
            position: position.clone().into_boxed_slice(),
            token: token_for_term(record.kind, term),
        }),
        TermKind::Application { .. } => {
            out.push(ExpressionFeature {
                position: position.clone().into_boxed_slice(),
                token: token_for_term(record.kind, term),
            });
            for (index, &child) in record.children().iter().enumerate() {
                let Ok(index) = u8::try_from(index) else {
                    continue;
                };
                position.push(index);
                append_features(sidecar, child, position, out, include_variables)?;
                position.pop();
            }
        }
        TermKind::NewVar if include_variables => out.push(ExpressionFeature {
            position: position.clone().into_boxed_slice(),
            token: token_for_term(record.kind, term),
        }),
        TermKind::VarRef(_) if include_variables => out.push(ExpressionFeature {
            position: position.clone().into_boxed_slice(),
            token: token_for_term(record.kind, term),
        }),
        TermKind::NewVar | TermKind::VarRef(_) => {}
    }

    Ok(())
}

fn token_for_term(kind: TermKind, term: TermId) -> ExpressionTrieToken {
    match kind {
        TermKind::Symbol => ExpressionTrieToken::Symbol(term),
        TermKind::Application { arity } => ExpressionTrieToken::App(arity),
        TermKind::NewVar => ExpressionTrieToken::NewVar,
        TermKind::VarRef(level) => ExpressionTrieToken::VarRef(level),
    }
}

#[cfg(test)]
fn feature(position: &[u8], token: ExpressionTrieToken) -> ExpressionFeature {
    ExpressionFeature {
        position: position.into(),
        token,
    }
}

fn append_ground_prefix(
    sidecar: &TermIdentitySidecar,
    term: TermId,
    out: &mut Vec<ExpressionTrieToken>,
) -> Result<bool, ExpressionTrieError> {
    let Some(record) = sidecar.get_term(term) else {
        return Err(ExpressionTrieError::UnknownTerm { term });
    };

    match record.kind {
        TermKind::Symbol => {
            out.push(ExpressionTrieToken::Symbol(term));
            Ok(true)
        }
        TermKind::Application { arity } => {
            out.push(ExpressionTrieToken::App(arity));
            for &child in record.children() {
                if !append_ground_prefix(sidecar, child, out)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        TermKind::NewVar | TermKind::VarRef(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::encoded_test_helpers::{app, sym, var, var_ref};
    use crate::space::Space;

    fn index_matches(
        sexpr: &[u8],
        pattern: Vec<u8>,
    ) -> (Space, TermIdentitySidecar, ExpressionTrieMatches) {
        let mut space = Space::new();
        space.add_all_sexpr(sexpr).unwrap();

        let mut sidecar = TermIdentitySidecar::new();
        let pattern_root = sidecar.insert_term(&pattern).unwrap();
        sidecar.extend_from_pathmap(&space.btm).unwrap();
        let index = ExpressionTrieIndex::build(&sidecar).unwrap();
        let matches = index.match_pattern(&sidecar, pattern_root).unwrap();

        (space, sidecar, matches)
    }

    fn build_index_for_pattern(
        space: &Space,
        pattern: &[u8],
    ) -> (TermIdentitySidecar, ExpressionTrieIndex, TermId) {
        let mut sidecar = TermIdentitySidecar::new();
        let pattern_root = sidecar.insert_term(pattern).unwrap();
        sidecar.extend_from_pathmap(&space.btm).unwrap();
        let index = ExpressionTrieIndex::build(&sidecar).unwrap();

        (sidecar, index, pattern_root)
    }

    #[test]
    fn typed_expression_trie_filters_repeated_variable_pattern_before_exact_match() {
        let pattern = app(&[sym(b"edge"), var(), app(&[sym(b"f"), var_ref(0)])]);
        let (space, sidecar, matches) = index_matches(
            br#"
(edge Alice (f Alice))
(edge Alice (f Bob))
(edge Bob (f Bob))
(edge Carol (g Carol))
(edge Dave (f Eve))
(node Alice)
(tag Bob)
"#,
            pattern,
        );
        let product_pattern = crate::expr!(space, "[2] , [3] edge $ [2] f _1");
        let mut product_roots = BTreeSet::new();
        let product_count = Space::query_multi(&space.btm, product_pattern, |_, loc| {
            let span = unsafe { loc.span().as_ref().unwrap() };
            product_roots.insert(span.to_vec());
            true
        });
        let trie_roots = matches
            .exact
            .rows
            .iter()
            .map(|row| sidecar.get_term(row.root).unwrap().encoded().to_vec())
            .collect::<BTreeSet<_>>();

        assert_eq!(product_count, 2);
        assert_eq!(matches.exact.stats.matches, product_count);
        assert_eq!(trie_roots, product_roots);
        assert_eq!(matches.candidates.prefix.len(), 2);
        assert_eq!(matches.candidates.features.len(), 4);
        assert_eq!(matches.candidates.facts.len(), 4);
        assert!(matches.candidates.facts.len() < sidecar.stats().facts);
        assert_eq!(
            matches.exact.stats.facts_scanned,
            matches.candidates.facts.len()
        );
    }

    #[test]
    fn typed_expression_trie_exact_ground_prefix_returns_one_candidate() {
        let pattern = app(&[sym(b"edge"), sym(b"Alice"), sym(b"Bob")]);
        let (_, _, matches) = index_matches(
            br#"
(edge Alice Bob)
(edge Bob Carol)
(node Alice)
"#,
            pattern,
        );

        assert_eq!(matches.candidates.facts.len(), 1);
        assert_eq!(matches.candidates.prefix.len(), 4);
        assert_eq!(matches.candidates.features.len(), 4);
        assert_eq!(matches.exact.stats.matches, 1);
        assert_eq!(matches.exact.stats.facts_scanned, 1);
    }

    #[test]
    fn expression_feature_index_prunes_suffix_bound_patterns() {
        let pattern = app(&[sym(b"edge"), var(), sym(b"Bob")]);
        let (space, sidecar, matches) = index_matches(
            br#"
(edge Alice Bob)
(edge Carol Bob)
(edge Alice Dana)
(edge (f Alice) Bob)
(edge Eve (f Bob))
(node Bob)
"#,
            pattern,
        );
        let product_pattern = crate::expr!(space, "[2] , [3] edge $ Bob");
        let product_count = Space::query_multi(&space.btm, product_pattern, |_, _| true);

        assert_eq!(product_count, 3);
        assert_eq!(matches.exact.stats.matches, product_count);
        assert_eq!(matches.candidates.prefix.len(), 2);
        assert_eq!(matches.candidates.features.len(), 3);
        let bob = sidecar.term_id_for_encoded(&sym(b"Bob")).unwrap();
        assert!(
            matches
                .candidates
                .features
                .contains(&feature(&[2], ExpressionTrieToken::Symbol(bob)))
        );
        assert_eq!(matches.candidates.facts.len(), 3);
        assert!(matches.candidates.facts.len() < sidecar.stats().facts);
        assert_eq!(
            matches.exact.stats.facts_scanned,
            matches.candidates.facts.len()
        );
    }

    #[test]
    fn expression_index_rebuild_tracks_mutated_snapshot() {
        let mut space = Space::new();
        space
            .add_all_sexpr(
                br#"
(edge Alice Bob)
(edge Carol Bob)
(edge Alice Dana)
(node Bob)
"#,
            )
            .unwrap();

        let pattern = app(&[sym(b"edge"), var(), sym(b"Bob")]);
        let removed = app(&[sym(b"edge"), sym(b"Alice"), sym(b"Bob")]);
        let added = app(&[sym(b"edge"), sym(b"Eve"), sym(b"Bob")]);
        let (before_sidecar, before_index, before_pattern) =
            build_index_for_pattern(&space, &pattern);
        let before_matches = before_index
            .match_pattern(&before_sidecar, before_pattern)
            .unwrap();
        let before_stats = before_index.stats();

        assert_eq!(before_stats.source_generation, 4);
        assert_eq!(before_matches.candidates.facts.len(), 2);
        assert!(before_matches.exact.rows.iter().any(|row| {
            before_sidecar
                .get_term(row.root)
                .is_some_and(|record| record.encoded() == removed)
        }));

        space.remove_all_sexpr(b"(edge Alice Bob)").unwrap();
        space.add_all_sexpr(b"(edge Eve Bob)").unwrap();

        let (after_sidecar, after_index, after_pattern) = build_index_for_pattern(&space, &pattern);
        let after_matches = after_index
            .match_pattern(&after_sidecar, after_pattern)
            .unwrap();
        let after_stats = after_index.stats();

        assert_eq!(
            after_stats.source_generation,
            before_stats.source_generation
        );
        assert_ne!(
            after_stats.source_fingerprint,
            before_stats.source_fingerprint
        );
        assert_eq!(after_matches.candidates.facts.len(), 2);
        assert!(after_sidecar.term_id_for_encoded(&removed).is_none());
        assert!(after_matches.exact.rows.iter().any(|row| {
            after_sidecar
                .get_term(row.root)
                .is_some_and(|record| record.encoded() == added)
        }));
    }
}
