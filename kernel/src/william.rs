//! WILLIAM's validation loop (whitepaper 5.12, S3): factor a heavy term-boundary
//! pattern into one definition plus per-atom references, measure the realized byte
//! gain by independently re-walking the store, and iterate greedily to a fixpoint.
//!
//! The predicted gain of a pattern of `len` bytes shared by `count` atoms is the
//! index weight `(count - 1) * len - count * REF_COST`. The factoring defined here
//! also pays one `REF_COST` id header on the definition atom itself, so the exact
//! realized gain is `predicted - REF_COST`; the tests pin `bytes_before - bytes_after`
//! to that identity on constructed and adversarial corpora, and `expand` inverts the
//! whole factoring so losslessness is checked by round-trip equality, not argued.

use std::collections::BTreeMap;

use crate::weighted_paths::{WeightedPathIndex, decode_pattern};
use mork_expr::{Tag, item_byte};
use pathmap::PathMap;
use pathmap::zipper::{ZipperIteration, ZipperMoving, ZipperValues, ZipperWriting};

/// Byte length of a definition/reference id header: one `SymbolSize` tag plus the
/// 8-byte payload of [`ref_header`]/[`def_header`]. This is the `ref_cost` the gain
/// index must be built with for its weights to be this factoring's predicted gains.
pub const REF_COST: u64 = 9;

/// Reference id header: the symbol `~r` followed by six hex digits, as one encoded
/// item. A factored atom `pattern ++ suffix` is rewritten to `ref_header(id) ++ suffix`.
/// The `~r`/`~d` symbols are reserved by this transform; a corpus that already uses
/// them would collide, which [`factor_pattern`] rejects rather than corrupts.
pub fn ref_header(id: u32) -> Vec<u8> {
    id_header(b'r', id)
}

/// Definition id header: the symbol `~d` followed by six hex digits. The definition
/// atom is `def_header(id) ++ pattern`, storing the factored spine once.
pub fn def_header(id: u32) -> Vec<u8> {
    id_header(b'd', id)
}

fn id_header(kind: u8, id: u32) -> Vec<u8> {
    let payload = format!("~{}{id:06x}", kind as char);
    debug_assert_eq!(payload.len() as u64 + 1, REF_COST);
    let mut bytes = vec![item_byte(Tag::SymbolSize(payload.len() as u8))];
    bytes.extend_from_slice(payload.as_bytes());
    bytes
}

/// Total stored bytes: the sum of every atom's path length. The independent measure
/// the validation compares against; it never derives from the gain arithmetic.
pub fn store_bytes(atoms: &PathMap<()>) -> u64 {
    let mut total = 0u64;
    atoms.for_each_value(|path, _| total += path.len() as u64);
    total
}

/// One validated factoring: what was predicted, what a full re-measure realized.
#[derive(Clone, Debug)]
pub struct FactoringOutcome {
    /// The factored term-boundary pattern, raw encoded bytes.
    pub pattern: Vec<u8>,
    /// The pattern rendered as MeTTa (truncated argument slots as `…`).
    pub rendered: String,
    /// Definition id used for this round's headers.
    pub id: u32,
    /// Atoms that carried the pattern as a prefix (all rewritten).
    pub count: u64,
    /// `(count - 1) * len - count * REF_COST`: the index weight for this pattern.
    pub predicted_gain: i64,
    /// `bytes_before - bytes_after`, measured by re-walking the store.
    pub realized_gain: i64,
    /// Store bytes before this factoring.
    pub bytes_before: u64,
    /// Store bytes after this factoring.
    pub bytes_after: u64,
}

/// Errors a factoring refuses to proceed past (the store is left unchanged).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FactoringError {
    /// Fewer than two atoms extend the pattern; factoring cannot pay for itself.
    PatternNotShared { count: u64 },
    /// A rewrite target or the definition atom already exists in the store (the
    /// corpus uses the reserved `~r`/`~d` id symbols).
    ReservedIdCollision,
}

/// Factors `pattern` out of every atom extending it: each `pattern ++ suffix` is
/// replaced by `ref_header(id) ++ suffix` and one definition atom
/// `def_header(id) ++ pattern` is added. Byte accounting, exactly:
///
/// ```text
/// before = count * len + Σ|suffix|
/// after  = count * REF_COST + Σ|suffix| + (REF_COST + len)
/// realized = before - after = (count - 1) * len - (count + 1) * REF_COST
///          = predicted - REF_COST
/// ```
///
/// The realized figure in the outcome is measured by re-walking the store, not
/// computed from this identity; the identity is what the tests (and the Isabelle
/// theory `WilliamGain`) hold it to.
pub fn factor_pattern(
    atoms: &mut PathMap<()>,
    pattern: &[u8],
    id: u32,
) -> Result<FactoringOutcome, FactoringError> {
    let bytes_before = store_bytes(atoms);

    // Collect every atom extending the pattern (the pattern itself included) before
    // mutating; the zipper borrow ends before the writes start.
    let mut matched: Vec<Vec<u8>> = Vec::new();
    {
        let mut rz = atoms.read_zipper_at_path(pattern);
        if rz.val().is_some() {
            matched.push(pattern.to_vec());
        }
        while rz.to_next_val() {
            let mut full = pattern.to_vec();
            full.extend_from_slice(rz.path());
            matched.push(full);
        }
    }
    let count = matched.len() as u64;
    if count < 2 {
        return Err(FactoringError::PatternNotShared { count });
    }

    let refh = ref_header(id);
    let defh = def_header(id);

    // Refuse a corpus that already holds the reserved id headers instead of
    // silently merging with it: every rewrite target and the definition must be new.
    let mut def_atom = defh.clone();
    def_atom.extend_from_slice(pattern);
    let mut rewrites: Vec<Vec<u8>> = Vec::with_capacity(matched.len());
    for full in &matched {
        let mut rewritten = refh.clone();
        rewritten.extend_from_slice(&full[pattern.len()..]);
        rewrites.push(rewritten);
    }
    if atoms.get_val_at(&def_atom[..]).is_some()
        || rewrites.iter().any(|r| atoms.get_val_at(&r[..]).is_some())
    {
        return Err(FactoringError::ReservedIdCollision);
    }

    for (full, rewritten) in matched.iter().zip(&rewrites) {
        atoms.write_zipper_at_path(full).remove_val(true);
        let replaced = atoms.insert(&rewritten[..], ());
        debug_assert!(replaced.is_none(), "rewrite target existed despite the guard");
    }
    let replaced = atoms.insert(&def_atom[..], ());
    debug_assert!(replaced.is_none(), "definition existed despite the guard");

    let bytes_after = store_bytes(atoms);
    let len = pattern.len() as i64;
    Ok(FactoringOutcome {
        pattern: pattern.to_vec(),
        rendered: decode_pattern(pattern),
        id,
        count,
        predicted_gain: (count as i64 - 1) * len - count as i64 * REF_COST as i64,
        realized_gain: bytes_before as i64 - bytes_after as i64,
        bytes_before,
        bytes_after,
    })
}

/// Inverts every factoring in the store: definitions are read back, each
/// `ref_header(id) ++ suffix` atom becomes `pattern(id) ++ suffix`, and the
/// definition atoms are dropped. Applied repeatedly, so factorings stacked by
/// [`compression_loop`] (later patterns over earlier reference atoms) unwind in
/// reverse. Round-tripping `expand(factored) == original` is the losslessness check.
pub fn expand(atoms: &PathMap<()>) -> PathMap<()> {
    let mut current: BTreeMap<Vec<u8>, ()> = BTreeMap::new();
    atoms.for_each_value(|path, _| {
        current.insert(path.to_vec(), ());
    });

    // Definitions must survive across passes: an atom may only become
    // `ref_header(i)`-prefixed after a later round's reference is expanded (round j > i
    // can factor patterns that contain round i's header). A round's pattern holds only
    // bytes that existed when it was cut, so the definition-dependency graph is a DAG
    // over round order and the passes terminate. A pattern can also bury a definition
    // (round j factoring a `def_header(i)`-prefixed spine), so defs are re-collected
    // every pass as expansions resurface them.
    loop {
        let mut defs: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for path in current.keys() {
            if path.len() as u64 > REF_COST && is_id_header(path, b'd') {
                defs.insert(
                    ref_header_from_def(path),
                    path[REF_COST as usize..].to_vec(),
                );
            }
        }
        if defs.is_empty() {
            break;
        }

        let mut next: BTreeMap<Vec<u8>, ()> = BTreeMap::new();
        let mut changed = false;
        for path in current.keys() {
            if path.len() as u64 > REF_COST && is_id_header(path, b'd') {
                next.insert(path.clone(), ()); // retained until nothing references it
                continue;
            }
            match defs.iter().find(|(refh, _)| path.starts_with(refh.as_slice())) {
                Some((refh, pattern)) => {
                    let mut e = pattern.clone();
                    e.extend_from_slice(&path[refh.len()..]);
                    next.insert(e, ());
                    changed = true;
                }
                None => {
                    next.insert(path.clone(), ());
                }
            }
        }
        if !changed {
            // Every reference is expanded; the definition records have no referents
            // left and are dropped rather than expanded (they are bookkeeping, not data).
            for refh in defs.keys() {
                let mut defh = refh.clone();
                defh[2] = b'd';
                next.retain(|path, _| !path.starts_with(&defh[..]));
            }
            current = next;
            break;
        }
        current = next;
    }

    let mut out: PathMap<()> = PathMap::new();
    for path in current.keys() {
        out.insert(&path[..], ());
    }
    out
}

fn is_id_header(path: &[u8], kind: u8) -> bool {
    path.len() as u64 >= REF_COST
        && path[0] == item_byte(Tag::SymbolSize((REF_COST - 1) as u8))
        && path[1] == b'~'
        && path[2] == kind
}

fn ref_header_from_def(def_path: &[u8]) -> Vec<u8> {
    let mut refh = def_path[..REF_COST as usize].to_vec();
    refh[2] = b'r';
    refh
}

/// Runs the WILLIAM compression loop: per round, rebuild the boundary gain index over
/// the current store, take the heaviest maximal (prefix-free) candidates, factor the
/// first whose realized gain `weight - REF_COST` is positive, and stop when no
/// candidate pays or `max_rounds` is reached. Each outcome's realized gain is
/// re-measured; the caller can assert monotone byte decrease from the outcomes.
pub fn compression_loop(
    atoms: &mut PathMap<()>,
    k_probe: usize,
    max_rounds: usize,
) -> Vec<FactoringOutcome> {
    let mut outcomes = Vec::new();
    for round in 0..max_rounds {
        let index = WeightedPathIndex::from_compression_gain_on_boundaries(atoms, REF_COST);
        let candidates = match index.iter_any_topk_maximal(k_probe) {
            Ok(c) => c,
            Err(_) => break,
        };
        let Some((pattern, _)) = candidates
            .into_iter()
            .find(|(_, weight)| *weight > REF_COST)
        else {
            break;
        };
        match factor_pattern(atoms, &pattern, round as u32) {
            Ok(outcome) => outcomes.push(outcome),
            // A reserved-id collision or a race-shrunk pattern ends the loop; the
            // store is unchanged by the refused round.
            Err(_) => break,
        }
    }
    outcomes
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::encoded_test_helpers::{arity, cat};

    /// Encode a symbol item from text (the shared helper takes raw bytes).
    fn sym(s: &str) -> Vec<u8> {
        crate::encoded_test_helpers::sym(s.as_bytes())
    }

    fn atom_set(atoms: &PathMap<()>) -> std::collections::BTreeSet<Vec<u8>> {
        let mut s = std::collections::BTreeSet::new();
        atoms.for_each_value(|p, _| {
            s.insert(p.to_vec());
        });
        s
    }

    /// Rule corpus: `count` atoms sharing the `(rule (when a) _)` spine, plus bystanders.
    fn rule_corpus(count: usize) -> (PathMap<()>, Vec<u8>) {
        let mut atoms: PathMap<()> = PathMap::new();
        let spine = cat(&[arity(3), sym("rule"), cat(&[arity(2), sym("when"), sym("a")])]);
        for i in 0..count {
            let mut a = spine.clone();
            a.extend_from_slice(&sym(&format!("X{i:03}")));
            atoms.insert(&a[..], ());
        }
        atoms.insert(&cat(&[arity(2), sym("fact"), sym("z")])[..], ());
        (atoms, spine)
    }

    #[test]
    fn factoring_realizes_exactly_predicted_minus_the_definition_header() {
        for count in [2usize, 3, 7, 20] {
            let (mut atoms, spine) = rule_corpus(count);
            let original = atom_set(&atoms);
            let before = store_bytes(&atoms);

            let outcome = factor_pattern(&mut atoms, &spine, 0).unwrap();
            assert_eq!(outcome.count, count as u64);
            assert_eq!(outcome.bytes_before, before);
            assert_eq!(
                outcome.predicted_gain,
                (count as i64 - 1) * spine.len() as i64 - count as i64 * REF_COST as i64
            );
            // The measured identity: realized == predicted - REF_COST, exactly.
            assert_eq!(outcome.realized_gain, outcome.predicted_gain - REF_COST as i64);
            assert_eq!(outcome.bytes_after, before - outcome.realized_gain as u64);
            assert_eq!(outcome.rendered, "(rule (when a) …)");

            // Losslessness by round-trip, not by argument.
            assert_eq!(atom_set(&expand(&atoms)), original);
        }
    }

    #[test]
    fn factoring_handles_the_pattern_itself_and_pattern_bearing_suffixes() {
        // Adversarial: one atom IS the pattern (empty suffix) and one suffix contains
        // the pattern bytes again; bystanders survive untouched.
        let spine = cat(&[arity(2), sym("dup"), sym("q")]);
        let mut atoms: PathMap<()> = PathMap::new();
        atoms.insert(&spine[..], ());
        let mut nested = spine.clone();
        nested.extend_from_slice(&spine);
        atoms.insert(&nested[..], ());
        let bystander = cat(&[arity(2), sym("other"), sym("w")]);
        atoms.insert(&bystander[..], ());
        let original = atom_set(&atoms);

        let outcome = factor_pattern(&mut atoms, &spine, 5).unwrap();
        assert_eq!(outcome.count, 2);
        assert_eq!(outcome.realized_gain, outcome.predicted_gain - REF_COST as i64);

        let after = atom_set(&atoms);
        assert!(after.contains(&bystander), "bystander was disturbed");
        let mut ref_alone = ref_header(5);
        assert!(after.contains(&ref_alone), "empty-suffix rewrite missing");
        ref_alone.extend_from_slice(&spine);
        assert!(after.contains(&ref_alone), "pattern-bearing suffix rewrite missing");

        assert_eq!(atom_set(&expand(&atoms)), original);
    }

    #[test]
    fn factoring_refuses_unshared_patterns_and_reserved_collisions() {
        let (mut atoms, spine) = rule_corpus(3);
        let only_once = cat(&[arity(2), sym("fact"), sym("z")]);
        assert!(matches!(
            factor_pattern(&mut atoms, &only_once, 0),
            Err(FactoringError::PatternNotShared { count: 1 })
        ));

        // A corpus already holding the id-0 definition header collides.
        let mut poisoned = def_header(0);
        poisoned.extend_from_slice(&spine);
        atoms.insert(&poisoned[..], ());
        assert!(matches!(
            factor_pattern(&mut atoms, &spine, 0),
            Err(FactoringError::ReservedIdCollision)
        ));
    }

    #[test]
    fn compression_loop_decreases_bytes_monotonically_to_a_fixpoint() {
        // Skewed corpus: two heavy templates, a light tail (the william bench shape).
        let mut atoms: PathMap<()> = PathMap::new();
        let mut x = 9u64;
        for _ in 0..240 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = match x % 10 {
                0..=3 => cat(&[
                    arity(3),
                    sym("rule"),
                    cat(&[arity(2), sym("when"), sym(&format!("C{:01}", x % 4))]),
                    sym(&format!("A{:04x}", (x >> 8) % 65536)),
                ]),
                4..=6 => cat(&[
                    arity(3),
                    sym("edge"),
                    sym(&format!("N{:02x}", (x >> 4) % 64)),
                    sym(&format!("N{:02x}", (x >> 12) % 64)),
                ]),
                _ => cat(&[arity(2), sym("fact"), sym(&format!("F{:06x}", (x >> 8) % 0xffffff))]),
            };
            atoms.insert(&a[..], ());
        }
        let original = atom_set(&atoms);
        let start_bytes = store_bytes(&atoms);

        let outcomes = compression_loop(&mut atoms, 8, 64);
        assert!(outcomes.len() >= 2, "expected several factorings, got {}", outcomes.len());

        let mut expected_bytes = start_bytes;
        for o in &outcomes {
            assert!(o.realized_gain > 0, "unprofitable round shipped: {o:?}");
            assert_eq!(o.realized_gain, o.predicted_gain - REF_COST as i64);
            assert_eq!(o.bytes_before, expected_bytes);
            expected_bytes = o.bytes_after;
            assert!(o.bytes_after < o.bytes_before);
        }
        assert_eq!(store_bytes(&atoms), expected_bytes);

        // Fixpoint: another loop finds nothing that pays.
        assert!(compression_loop(&mut atoms, 8, 64).is_empty());

        // The whole stack of factorings unwinds to the original corpus.
        let expanded = atom_set(&expand(&atoms));
        let missing: Vec<String> =
            original.difference(&expanded).map(|a| decode_pattern(a)).take(4).collect();
        let extra: Vec<String> =
            expanded.difference(&original).map(|a| decode_pattern(a)).take(4).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "round-trip broke: missing {missing:?} extra {extra:?} (factored {:?})",
            outcomes.iter().map(|o| o.rendered.clone()).collect::<Vec<_>>()
        );
    }
}
