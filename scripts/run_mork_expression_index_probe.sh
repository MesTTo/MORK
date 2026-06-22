#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-quick}"
LOG_DIR="${LOG_DIR:-/tmp}"
LOAD_MAX="${LOAD_MAX:-2.0}"
ALLOW_BUSY="${ALLOW_BUSY:-0}"
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
FEATURES="${FEATURES:-grounding}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
. "$ROOT_DIR/scripts/sandbox_report_shell.sh"

usage() {
  printf 'usage: %s [quick|full]\n' "$0" >&2
}

if (( $# > 1 )); then
  usage
  exit 2
fi

case "$MODE" in
  quick)
    : "${EXPRESSION_INDEX_KEYS:=128}"
    : "${EXPRESSION_INDEX_ROUNDS:=16}"
    ;;
  full)
    : "${EXPRESSION_INDEX_KEYS:=2048}"
    : "${EXPRESSION_INDEX_ROUNDS:=128}"
    ;;
  *)
    usage
    exit 2
    ;;
esac

if ! [[ "$EXPRESSION_INDEX_KEYS" =~ ^[0-9]+$ ]] || (( EXPRESSION_INDEX_KEYS < 1 )); then
  printf 'EXPRESSION_INDEX_KEYS must be a positive integer, got %s\n' "$EXPRESSION_INDEX_KEYS" >&2
  exit 2
fi

if ! [[ "$EXPRESSION_INDEX_ROUNDS" =~ ^[0-9]+$ ]] || (( EXPRESSION_INDEX_ROUNDS < 0 )); then
  printf 'EXPRESSION_INDEX_ROUNDS must be a non-negative integer, got %s\n' "$EXPRESSION_INDEX_ROUNDS" >&2
  exit 2
fi

if (( EXPRESSION_INDEX_ROUNDS > EXPRESSION_INDEX_KEYS )); then
  printf 'EXPRESSION_INDEX_ROUNDS must be <= EXPRESSION_INDEX_KEYS so every removal exists\n' >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="$(create_sandbox_report_dir "$LOG_DIR" "mork-expression-index-probe" "$stamp")"
commands_tsv="$out_dir/commands.tsv"
runs_tsv="$out_dir/runs.tsv"
printf 'name\tstatus\telapsed_ms\tlog\n' > "$commands_tsv"
printf 'round\tremoved\tadded\tfacts\tterms\tgeneration\tfingerprint\ttrie_nodes\ttokens_indexed\tfeatures_indexed\tfeature_postings\tprefix_tokens\tfeatures\tcandidates\texact_matches\tfacts_scanned\tbuild_us\tmatch_us\n' > "$runs_tsv"
load1="$(awk '{ print $1 }' /proc/loadavg)"

record_command() {
  local name="$1"
  local status="$2"
  local elapsed_ms="$3"
  local log_name="$4"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$elapsed_ms" "$log_name" >> "$commands_tsv"
}

write_summary() {
  local final_status="$1"
  local summary="$out_dir/summary.md"
  {
    printf '# MORK Expression Index Probe Summary\n\n'
    printf 'Timestamp: `%s`\n\n' "$stamp"
    printf -- '- Mode: `%s`.\n' "$MODE"
    printf -- '- Final status: `%s`.\n' "$final_status"
    printf -- '- Keys: `%s` matching `(edge n Bob)` facts plus equal nonmatching edge and node facts.\n' "$EXPRESSION_INDEX_KEYS"
    printf -- '- Mutation rounds: `%s` deterministic remove/add pairs.\n' "$EXPRESSION_INDEX_ROUNDS"
    printf -- '- Load: `%s` with LOAD_MAX=`%s`, ALLOW_BUSY=`%s`.\n' "$load1" "$LOAD_MAX" "$ALLOW_BUSY"
    printf -- '- RUSTFLAGS: `%s`.\n' "$RUSTFLAGS"
    printf -- '- Features: `%s`.\n' "$FEATURES"
    printf '\n## Purpose\n\n'
    printf 'This probe measures the current rebuild-from-snapshot baseline for the derived expression trie after small dynamic updates. It does not implement maintained postings yet; it provides a repeatable cost floor for deciding when generation-keyed maintained postings are worth adding.\n\n'
    printf 'Each row rebuilds the `TermIdentitySidecar` and `ExpressionTrieIndex`, matches `(edge $x Bob)`, and verifies candidate pruning remains exact after the mutation.\n\n'
    printf '## Runs\n\n'
    printf '| round | facts | terms | candidates | exact | scanned | build | match | fingerprint |\n'
    printf '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n'
    if [ -s "$runs_tsv" ]; then
      tail -n +2 "$runs_tsv" | while IFS=$'\t' read -r round _removed _added facts terms _generation fingerprint _trie_nodes _tokens_indexed _features_indexed _feature_postings _prefix_tokens _features candidates exact_matches facts_scanned build_us match_us; do
        printf '| `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s us` | `%s us` | `%s` |\n' \
          "$round" "$facts" "$terms" "$candidates" "$exact_matches" "$facts_scanned" "$build_us" "$match_us" "$fingerprint"
      done
    fi
    printf '\n## Aggregate\n\n'
    awk '
      NR > 1 {
        rows += 1
        build_sum += $17
        match_sum += $18
        if (rows == 1 || $17 < min_build) min_build = $17
        if (rows == 1 || $17 > max_build) max_build = $17
        if (rows == 1 || $18 < min_match) min_match = $18
        if (rows == 1 || $18 > max_match) max_match = $18
      }
      END {
        if (rows > 0) {
          printf "- Rows: `%d`.\n", rows
          printf "- Build: avg `%.1f us`, min `%d us`, max `%d us`.\n", build_sum / rows, min_build, max_build
          printf "- Match: avg `%.1f us`, min `%d us`, max `%d us`.\n", match_sum / rows, min_match, max_match
        }
      }
    ' "$runs_tsv"
    printf '\n## Machine Reports\n\n'
    printf -- '- `manifest.txt`\n'
    printf -- '- `commands.tsv`\n'
    printf -- '- `runs.tsv`\n'
    printf -- '- `report.json`\n'
    printf -- '- `junit.xml`\n'
    printf -- '- `report_verification.log`\n'
  } > "$summary"

  write_and_verify_sandbox_reports "mork-expression-index-probe-$MODE" "$out_dir" "$final_status"
}

trap 'finish_sandbox_exit "$?" write_summary' EXIT

{
  printf 'stamp_utc=%s\n' "$stamp"
  printf 'mode=%s\n' "$MODE"
  printf 'root_dir=%s\n' "$ROOT_DIR"
  printf 'load1=%s\n' "$load1"
  printf 'load_max=%s\n' "$LOAD_MAX"
  printf 'allow_busy=%s\n' "$ALLOW_BUSY"
  printf 'expression_index_keys=%s\n' "$EXPRESSION_INDEX_KEYS"
  printf 'expression_index_rounds=%s\n' "$EXPRESSION_INDEX_ROUNDS"
  printf 'rustflags=%s\n' "$RUSTFLAGS"
  printf 'features=%s\n' "$FEATURES"
  printf '\n'
  uname -a
  printf '\n'
  uptime
} > "$out_dir/manifest.txt"

if [ "$ALLOW_BUSY" != "1" ]; then
  if awk -v load="$load1" -v max="$LOAD_MAX" 'BEGIN { exit !(load > max) }'; then
    {
      printf 'Refusing to run clean expression index probe: load1=%s exceeds LOAD_MAX=%s\n' "$load1" "$LOAD_MAX"
      printf 'Set ALLOW_BUSY=1 to run anyway and label the result as noisy.\n'
    } | tee "$out_dir/ABORTED_BUSY.txt" >&2
    ps -eo pid,ppid,pcpu,pmem,comm,args --sort=-pcpu | sed -n '1,25p' > "$out_dir/top-processes.txt"
    record_command load_gate 75 0 ABORTED_BUSY.txt
    exit 75
  fi
fi
record_command load_gate 0 0 manifest.txt

cd "$ROOT_DIR"
build_log="$out_dir/build_probe.log"
build_start_ns="$(date +%s%N)"
set +e
{
  printf '$ env RUSTFLAGS=%q cargo +nightly build --release -q -p mork --bin expression_index_probe --features %q\n\n' \
    "$RUSTFLAGS" "$FEATURES"
  env RUSTFLAGS="$RUSTFLAGS" cargo +nightly build --release -q -p mork --bin expression_index_probe --features "$FEATURES"
} > "$build_log" 2>&1
build_status="$?"
set -e
build_end_ns="$(date +%s%N)"
build_elapsed_ms="$(((build_end_ns - build_start_ns) / 1000000))"
record_command build_probe "$build_status" "$build_elapsed_ms" build_probe.log
if [ "$build_status" != "0" ]; then
  exit "$build_status"
fi

run_log="$out_dir/run_probe.log"
run_output="$out_dir/probe_output.tsv"
run_start_ns="$(date +%s%N)"
set +e
{
  printf '$ target/release/expression_index_probe --keys %q --rounds %q\n\n' \
    "$EXPRESSION_INDEX_KEYS" "$EXPRESSION_INDEX_ROUNDS"
  target/release/expression_index_probe \
    --keys "$EXPRESSION_INDEX_KEYS" \
    --rounds "$EXPRESSION_INDEX_ROUNDS"
} > "$run_output" 2> "$run_log"
run_status="$?"
set -e
run_end_ns="$(date +%s%N)"
run_elapsed_ms="$(((run_end_ns - run_start_ns) / 1000000))"
record_command run_probe "$run_status" "$run_elapsed_ms" run_probe.log
if [ "$run_status" != "0" ]; then
  exit "$run_status"
fi

tail -n +4 "$run_output" >> "$runs_tsv"
row_count="$(tail -n +2 "$runs_tsv" | wc -l | tr -d ' ')"
expected_rows="$((EXPRESSION_INDEX_ROUNDS + 1))"
if [ "$row_count" != "$expected_rows" ]; then
  printf 'expected %s probe rows, got %s\n' "$expected_rows" "$row_count" >&2
  exit 1
fi

printf 'MORK expression index probe logs written to %s\n' "$out_dir"
