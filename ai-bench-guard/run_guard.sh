#!/usr/bin/env bash
# MORK bench regression guard.
#
#   ./ai-bench-guard/run_guard.sh baseline [label]   capture a new baseline set
#   ./ai-bench-guard/run_guard.sh check    [label]   run and compare against baselines/current
#
# Guards the full `mork bench all` surface across the shipping feature configs.
# Correctness: normalized bench output must match the baseline byte-for-byte
# (timing lines stripped; counters, result counts, and every other line kept).
# Performance: instructions:u per bench is recorded and diffed (load-immune to
# ~1e-6 relative, validated 2026-07-15); wall/cycles are informational only.
#
# Build rules this script deliberately follows:
# - RUSTFLAGS is set EXPLICITLY to the full flag set. Relying on the project
#   .cargo/config.toml does not work on this machine: ~/.cargo/config.toml
#   defines target.x86_64-unknown-linux-gnu.rustflags (mold link-arg), and
#   cargo's precedence is env > target.<triple>.rustflags > build.rustflags
#   with NO merging across kinds, so the project's build.rustflags (including
#   --cfg gxhash) is silently ignored whenever env RUSTFLAGS is unset
#   (discovered 2026-07-17 when gxhash's aes check failed a cold build).
#   The explicit set below reproduces project flags + the user's mold linking.
# - Each feature config builds into its own CARGO_TARGET_DIR under this
#   directory, so the repo's main target/ (and any binary another session is
#   running from it) is never touched.
# - Everything runs under nice -n 15 so live experiments keep priority.
set -uo pipefail
export RUSTFLAGS="-C target-cpu=native -Awarnings --cfg gxhash -C link-arg=-fuse-ld=mold"
cd "$(dirname "$0")/.."
GUARD=ai-bench-guard
mkdir -p "$GUARD"

MODE="${1:?usage: run_guard.sh baseline|check [label]}"
DIRTY=""; git diff --quiet && git diff --cached --quiet || DIRTY="-dirty"
LABEL="${2:-$(git rev-parse --short HEAD)$DIRTY}"
NICE="nice -n 15"
# 1800 not 900: canonical-config odd_even_sort legitimately takes ~535 s (the
# leapfrog profit-gate pathology, found 2026-07-17); a timeout mid-cell would
# rot the baseline.
PER_BENCH_TIMEOUT=1800

# name:features (empty features = upstream default behavior)
CONFIGS=(
  "default:"
  "canonical:einsum,leapfrog,bulk_emit,factorized_aggregate"
  "canonical-strat:einsum,leapfrog,bulk_emit,factorized_aggregate,stratified_quiescence"
  "canonical-sni:einsum,leapfrog,bulk_emit,factorized_aggregate,semi_naive_ic"
  "morkl-plan:morkl_plan"
  "canonical-morkl:einsum,leapfrog,bulk_emit,factorized_aggregate,morkl_plan"
)
# `bench all` resolves the set itself (12 benches, +aggregate when
# factorized_aggregate is compiled in), but we run benches ONE AT A TIME so
# perf counters are per-bench.
BASE_BENCHES=(taxi_lts counter_machine transitive clique finite_domain process_calculus exponential exponential_fringe odd_even_sort logic_query tile_puzzle_states bfc)

# Timing-line filter applied to BOTH sides at compare time. Keep this
# conservative: it must never eat a counter line. Refine here if a bench adds
# a new timing format; recapturing baselines is not required for filter edits.
normalize() {
  # Mask timing VALUES on counter-bearing lines (e.g. "elapsed 6772 steps 4001
  # size 2005003" keeps its counters), then drop pure timing/rate lines.
  sed -E 's/(elapsed|took|wall|time)[[:space:]]+[0-9]+(\.[0-9]+)?/\1 <T>/Ig' "$1" \
    | grep -vE '(^|[^A-Za-z])[0-9]+(\.[0-9]+)?[[:space:]]*(ms|s|sec|secs|seconds|us|µs|ns|m)([[:space:]]|[.,)]|$)' \
    | grep -viE 'per second|/s([[:space:]]|$)|throughput|writes/sec|speed'
}

run_all() {
  local outroot="$1"
  mkdir -p "$outroot"
  { git rev-parse HEAD; git describe --always --dirty 2>/dev/null; rustc +nightly -V; date -u +%FT%TZ; uname -r; } > "$outroot/meta.txt"
  for entry in "${CONFIGS[@]}"; do
    local name="${entry%%:*}" feats="${entry#*:}"
    local tdir="$GUARD/target-$name" outdir="$outroot/$name"
    mkdir -p "$outdir"
    echo "--- building $name (${feats:-no features}) ---"
    if ! CARGO_TARGET_DIR="$tdir" $NICE cargo +nightly build --release -p mork --bin mork ${feats:+--features "$feats"} > "$outdir/build.log" 2>&1; then
      echo "BUILD FAILED for $name (see $outdir/build.log)"; return 2
    fi
    local benches=("${BASE_BENCHES[@]}")
    [[ "$feats" == *factorized_aggregate* ]] && benches+=(aggregate)
    printf '%s\n' "${benches[@]}" > "$outdir/benches.txt"
    for b in "${benches[@]}"; do
      echo "    $name / $b"
      $NICE perf stat -x, -e instructions:u,cycles:u,task-clock -o "$outdir/$b.perf" -- \
        timeout "$PER_BENCH_TIMEOUT" "$tdir/release/mork" bench "$b" > "$outdir/$b.out" 2> "$outdir/$b.err"
      echo "$?" > "$outdir/$b.exit"
    done
  done
  return 0
}

instr_of() { awk -F, '$3=="instructions:u"{print $1}' "$1" 2>/dev/null; }

case "$MODE" in
  baseline)
    DEST="$GUARD/baselines/$LABEL"
    if [ -e "$DEST" ]; then echo "baseline $DEST already exists; remove it or pick a label"; exit 1; fi
    run_all "$DEST" || exit 2
    ln -sfn "$LABEL" "$GUARD/baselines/current"
    echo "BASELINE CAPTURED: $DEST (baselines/current -> $LABEL)"
    ;;
  check)
    BASE="$GUARD/baselines/current"
    [ -e "$BASE" ] || { echo "no baseline at $BASE; run baseline first"; exit 1; }
    DEST="$GUARD/runs/$(date -u +%Y%m%dT%H%M%SZ)-$LABEL"
    run_all "$DEST" || exit 2
    FAIL=0
    echo; echo "=== comparison vs $(readlink -f "$BASE") ==="
    for entry in "${CONFIGS[@]}"; do
      name="${entry%%:*}"
      while read -r b; do
        be="$BASE/$name/$b" ne="$DEST/$name/$b"
        if [ ! -f "$be.out" ]; then echo "NEW BENCH (no baseline): $name/$b"; continue; fi
        if [ "$(cat "$ne.exit")" != "$(cat "$be.exit")" ]; then
          echo "FAIL exit-code $name/$b: $(cat "$be.exit") -> $(cat "$ne.exit")"; FAIL=1
        fi
        if ! diff -q <(normalize "$be.out") <(normalize "$ne.out") >/dev/null; then
          echo "FAIL output $name/$b (diff of normalized output):"
          diff <(normalize "$be.out") <(normalize "$ne.out") | head -20
          FAIL=1
        fi
        bi="$(instr_of "$be.perf")"; ni="$(instr_of "$ne.perf")"
        if [ -n "$bi" ] && [ -n "$ni" ] && [ "$bi" != 0 ]; then
          awk -v b="$bi" -v n="$ni" -v tag="$name/$b" 'BEGIN{
            d=100.0*(n-b)/b; flag=(d>1.0)?"  <-- WARN instructions regression":"";
            printf "  instr %-38s %+8.3f%% (%s -> %s)%s\n", tag, d, b, n, flag }'
        fi
      done < "$BASE/$name/benches.txt"
    done
    if [ "$FAIL" -ne 0 ]; then echo; echo "REGRESSION CHECK: FAILED"; exit 1; fi
    echo; echo "REGRESSION CHECK: output-identical on every config x bench"
    ;;
  *) echo "usage: run_guard.sh baseline|check [label]"; exit 1;;
esac
