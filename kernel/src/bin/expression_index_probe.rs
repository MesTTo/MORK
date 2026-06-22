use std::env;
use std::fmt::Write;
use std::time::Instant;

use mork::expression_trie::{ExpressionTrieIndex, ExpressionTrieStats};
use mork::space::Space;
use mork::term_identity::{TermIdentitySidecar, TermIdentityStats};
use mork_expr::{Tag, item_byte};

const DEFAULT_KEYS: usize = 128;
const DEFAULT_ROUNDS: usize = 16;

#[derive(Clone, Copy, Debug)]
struct Config {
    keys: usize,
    rounds: usize,
}

#[derive(Clone, Copy, Debug)]
struct ProbeSample {
    term_stats: TermIdentityStats,
    trie_stats: ExpressionTrieStats,
    candidates: usize,
    exact_matches: usize,
    facts_scanned: usize,
    prefix_tokens: usize,
    features: usize,
    build_us: u128,
    match_us: u128,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("expression_index_probe: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let config = Config::parse(env::args().skip(1))?;
    let mut space = Space::new();
    load_fixture(&mut space, config.keys)?;
    let pattern = edge_to_bob_pattern();

    print_header();
    emit_sample(0, 0, 0, rebuild_and_match(&space, &pattern)?);

    for round in 1..=config.rounds {
        let old_key = round - 1;
        let new_key = config.keys + round - 1;
        let removed = format!("(edge n{old_key} Bob)");
        let added = format!("(edge fresh{new_key} Bob)");

        let removed_count = space.remove_all_sexpr(removed.as_bytes())?;
        if removed_count != 1 {
            return Err(format!(
                "expected to remove one fact for {removed:?}, removed {removed_count}"
            ));
        }
        let added_count = space.add_all_sexpr(added.as_bytes())?;
        if added_count != 1 {
            return Err(format!(
                "expected to add one fact for {added:?}, added {added_count}"
            ));
        }

        let sample = rebuild_and_match(&space, &pattern)?;
        if sample.exact_matches != config.keys {
            return Err(format!(
                "round {round} expected {} exact matches, got {}",
                config.keys, sample.exact_matches
            ));
        }
        if sample.candidates != sample.exact_matches {
            return Err(format!(
                "round {round} expected candidate pruning to be exact, got {} candidates for {} matches",
                sample.candidates, sample.exact_matches
            ));
        }
        emit_sample(round, removed_count, added_count, sample);
    }

    Ok(())
}

impl Config {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut config = Self {
            keys: DEFAULT_KEYS,
            rounds: DEFAULT_ROUNDS,
        };

        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--keys" => {
                    config.keys = parse_usize_arg("--keys", args.next())?;
                }
                "--rounds" => {
                    config.rounds = parse_usize_arg("--rounds", args.next())?;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                unknown => return Err(format!("unknown argument {unknown:?}")),
            }
        }

        if config.keys == 0 {
            return Err("--keys must be positive".to_owned());
        }
        if config.rounds > config.keys {
            return Err(format!(
                "--rounds must be <= --keys so each removed fact exists; got rounds={} keys={}",
                config.rounds, config.keys
            ));
        }

        Ok(config)
    }
}

fn parse_usize_arg(name: &str, value: Option<String>) -> Result<usize, String> {
    let Some(value) = value else {
        return Err(format!("{name} requires a value"));
    };
    value
        .parse::<usize>()
        .map_err(|error| format!("{name} expects a positive integer, got {value:?}: {error}"))
}

fn print_usage() {
    eprintln!(
        "usage: expression_index_probe [--keys N] [--rounds N]\n\
         Reports TSV rows for rebuilding the derived expression index after deterministic mutations."
    );
}

fn load_fixture(space: &mut Space, keys: usize) -> Result<(), String> {
    let capacity = keys
        .checked_mul(64)
        .ok_or_else(|| "fixture capacity overflow".to_owned())?;
    let expected = keys
        .checked_mul(3)
        .ok_or_else(|| "expected fact count overflow".to_owned())?;
    let mut input = String::with_capacity(capacity);
    for key in 0..keys {
        writeln!(input, "(edge n{key} Bob)").expect("writing to String cannot fail");
        writeln!(input, "(edge n{key} Dana)").expect("writing to String cannot fail");
        writeln!(input, "(node n{key})").expect("writing to String cannot fail");
    }
    let loaded = space.add_all_sexpr(input.as_bytes())?;
    if loaded != expected {
        return Err(format!(
            "expected to load {expected} facts, loaded {loaded}"
        ));
    }
    Ok(())
}

fn rebuild_and_match(space: &Space, pattern: &[u8]) -> Result<ProbeSample, String> {
    let build_start = Instant::now();
    let mut sidecar = TermIdentitySidecar::new();
    let pattern_root = sidecar
        .insert_term(pattern)
        .map_err(|error| format!("pattern parse failed: {error:?}"))?;
    sidecar
        .extend_from_pathmap(&space.btm)
        .map_err(|error| format!("pathmap sidecar build failed: {error:?}"))?;
    let index = ExpressionTrieIndex::build(&sidecar)
        .map_err(|error| format!("index build failed: {error:?}"))?;
    let build_us = build_start.elapsed().as_micros();

    let match_start = Instant::now();
    let matches = index
        .match_pattern(&sidecar, pattern_root)
        .map_err(|error| format!("pattern match failed: {error:?}"))?;
    let match_us = match_start.elapsed().as_micros();

    Ok(ProbeSample {
        term_stats: sidecar.stats(),
        trie_stats: index.stats(),
        candidates: matches.candidates.facts.len(),
        exact_matches: matches.exact.stats.matches,
        facts_scanned: matches.exact.stats.facts_scanned,
        prefix_tokens: matches.candidates.prefix.len(),
        features: matches.candidates.features.len(),
        build_us,
        match_us,
    })
}

fn print_header() {
    println!(
        "round\tremoved\tadded\tfacts\tterms\tgeneration\tfingerprint\ttrie_nodes\ttokens_indexed\tfeatures_indexed\tfeature_postings\tprefix_tokens\tfeatures\tcandidates\texact_matches\tfacts_scanned\tbuild_us\tmatch_us"
    );
}

fn emit_sample(round: usize, removed: usize, added: usize, sample: ProbeSample) {
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{:032x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        round,
        removed,
        added,
        sample.term_stats.facts,
        sample.term_stats.terms,
        sample.trie_stats.source_generation,
        sample.trie_stats.source_fingerprint,
        sample.trie_stats.trie_nodes,
        sample.trie_stats.tokens_indexed,
        sample.trie_stats.features_indexed,
        sample.trie_stats.feature_postings,
        sample.prefix_tokens,
        sample.features,
        sample.candidates,
        sample.exact_matches,
        sample.facts_scanned,
        sample.build_us,
        sample.match_us
    );
}

fn edge_to_bob_pattern() -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 1 + b"edge".len() + 1 + 1 + b"Bob".len());
    out.push(item_byte(Tag::Arity(3)));
    out.push(item_byte(Tag::SymbolSize(b"edge".len() as u8)));
    out.extend_from_slice(b"edge");
    out.push(item_byte(Tag::NewVar));
    out.push(item_byte(Tag::SymbolSize(b"Bob".len() as u8)));
    out.extend_from_slice(b"Bob");
    out
}
