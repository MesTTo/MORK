#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
case "$MODE" in
  local|quick|full) ;;
  *)
    printf 'usage: %s [local|quick|full]\n' "$0" >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp}"
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
RUSTFLAGS_TEST="${RUSTFLAGS_TEST:-$RUSTFLAGS -Awarnings}"
MORK_FEATURES="${MORK_FEATURES:-grounding}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BENCH_MODE="${BENCH_MODE:-quick}"
QUERY_PLANNER_STRESS_MODE="${QUERY_PLANNER_STRESS_MODE:-$BENCH_MODE}"
WRITE_RESOURCE_STRESS_MODE="${WRITE_RESOURCE_STRESS_MODE:-quick}"
WILLIAM_PREDICTIVE_MODE="${WILLIAM_PREDICTIVE_MODE:-quick}"
RECURSIVE_FRONTIER_MODE="${RECURSIVE_FRONTIER_MODE:-quick}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
MORK_RUNS="${MORK_RUNS:-20}"
. "$ROOT_DIR/scripts/sandbox_report_shell.sh"

case "$MODE" in
  local)
    RUN_HYPERON="${RUN_HYPERON:-0}"
    RUN_BENCHMARKS="${RUN_BENCHMARKS:-0}"
    ;;
  quick)
    RUN_HYPERON="${RUN_HYPERON:-1}"
    RUN_BENCHMARKS="${RUN_BENCHMARKS:-0}"
    ;;
  full)
    RUN_HYPERON="${RUN_HYPERON:-1}"
    RUN_BENCHMARKS="${RUN_BENCHMARKS:-1}"
    ;;
esac

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="$(create_sandbox_report_dir "$LOG_DIR" "mork-integration-sandbox" "$stamp")"
commands_tsv="$out_dir/commands.tsv"
printf 'name\tstatus\telapsed_ms\tlog\n' > "$commands_tsv"

{
  printf 'stamp_utc=%s\n' "$stamp"
  printf 'mode=%s\n' "$MODE"
  printf 'root_dir=%s\n' "$ROOT_DIR"
  printf 'run_hyperon=%s\n' "$RUN_HYPERON"
  printf 'run_benchmarks=%s\n' "$RUN_BENCHMARKS"
  printf 'run_query_planner_stress=%s\n' "${RUN_QUERY_PLANNER_STRESS:-$RUN_BENCHMARKS}"
  printf 'run_write_resource_stress=%s\n' "${RUN_WRITE_RESOURCE_STRESS:-$RUN_BENCHMARKS}"
  printf 'run_william_predictive_probe=%s\n' "${RUN_WILLIAM_PREDICTIVE_PROBE:-$RUN_BENCHMARKS}"
  printf 'run_recursive_frontier_probe=%s\n' "${RUN_RECURSIVE_FRONTIER_PROBE:-$RUN_BENCHMARKS}"
  printf 'run_verus_formal=%s\n' "${RUN_VERUS_FORMAL:-0}"
  printf 'run_cli_error_boundary=%s\n' "${RUN_CLI_ERROR_BOUNDARY:-1}"
  printf 'bench_mode=%s\n' "$BENCH_MODE"
  printf 'query_planner_stress_mode=%s\n' "$QUERY_PLANNER_STRESS_MODE"
  printf 'write_resource_stress_mode=%s\n' "$WRITE_RESOURCE_STRESS_MODE"
  printf 'william_predictive_mode=%s\n' "$WILLIAM_PREDICTIVE_MODE"
  printf 'recursive_frontier_mode=%s\n' "$RECURSIVE_FRONTIER_MODE"
  printf 'load_max=%s\n' "${LOAD_MAX:-child-default}"
  printf 'allow_busy=%s\n' "${ALLOW_BUSY:-child-default}"
  printf 'rustflags=%s\n' "$RUSTFLAGS"
  printf 'rustflags_test=%s\n' "$RUSTFLAGS_TEST"
  printf 'mork_features=%s\n' "$MORK_FEATURES"
  printf 'openblas_num_threads=%s\n' "$OPENBLAS_NUM_THREADS"
  printf 'mork_runs=%s\n' "$MORK_RUNS"
  printf 'hyperon_dir=%s\n' "${HYPERON_DIR:-/home/user/Dev/hyperon-build-src}"
  printf '\n'
  uname -a
  printf '\n'
  uptime
} > "$out_dir/manifest.txt"

write_summary() {
  local final_status="$1"
  local summary="$out_dir/summary.md"
  local load_line
  load_line="$(grep -m1 'load average:' "$out_dir/manifest.txt" || true)"

  {
    printf '# MORK Integration Sandbox Summary\n\n'
    printf "Timestamp: \`%s\`\n\n" "$stamp"
    printf -- "- Mode: \`%s\`.\n" "$MODE"
    printf -- "- Final status: \`%s\`.\n" "$final_status"
    printf -- "- Root: \`%s\`.\n" "$ROOT_DIR"
    printf -- "- RUSTFLAGS: \`%s\`.\n" "$RUSTFLAGS"
    printf -- "- RUSTFLAGS_TEST: \`%s\`.\n" "$RUSTFLAGS_TEST"
    printf -- "- MORK features: \`%s\`.\n" "$MORK_FEATURES"
    if [ -n "$load_line" ]; then
      printf -- "- Recorded load: \`%s\`.\n" "$load_line"
    fi
    printf '\n## Gates\n\n'
    printf '| gate | status | elapsed | log |\n'
    printf '| --- | ---: | ---: | --- |\n'
    if [ -s "$commands_tsv" ]; then
      tail -n +2 "$commands_tsv" | while IFS=$'\t' read -r name status elapsed log_name; do
        printf "| \`%s\` | \`%s\` | \`%s ms\` | \`%s\` |\n" "$name" "$status" "$elapsed" "$log_name"
      done
    fi
    printf '\n## Nested Summaries\n\n'
    local nested_count=0
    for nested_summary in "$out_dir"/*/summary.md; do
      if [ -f "$nested_summary" ]; then
        nested_count=$((nested_count + 1))
        printf -- "- \`%s\`\n" "${nested_summary#"$out_dir"/}"
      fi
    done
    if [ "$nested_count" = "0" ]; then
      printf 'None.\n'
    fi
    printf '\n## Semantic Coverage\n\n'
    printf -- "- Local MORK gates cover expression unification, parser error boundaries, query prefix ranking, critical-pair witnesses, rewrite semantics, FormalMeTTa parity, grounding, crate checks, and the internal MM2 runner.\n"
    if [ "$RUN_HYPERON" = "1" ]; then
      printf -- "- Hyperon MORK sandbox was enabled for this run; its nested summary covers Rust \`mork-space\` backend behavior and Python MeTTa sandbox semantics.\n"
    else
      printf -- "- Hyperon MORK sandbox was disabled for this run; use \`RUN_HYPERON=1\` or mode \`quick\`/\`full\` when external Hyperon/MeTTa parity evidence is required.\n"
    fi
    printf '\n## Machine Reports\n\n'
    printf -- "- \`report.json\`\n"
    printf -- "- \`junit.xml\`\n"
    printf -- "- \`report_verification.log\`\n"
  } > "$summary"

  write_and_verify_sandbox_reports "mork-integration-$MODE" "$out_dir" "$final_status"
}

trap 'finish_sandbox_exit "$?" write_summary' EXIT

run_log() {
  local name="$1"
  shift
  local log="$out_dir/$name.log"
  local start_ns end_ns elapsed_ms status
  start_ns="$(date +%s%N)"
  set +e
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n\n'
    "$@"
  } > "$log" 2>&1
  status="$?"
  set -e
  end_ns="$(date +%s%N)"
  elapsed_ms="$(((end_ns - start_ns) / 1000000))"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$elapsed_ms" "${name}.log" >> "$commands_tsv"
  if [ "$status" != "0" ]; then
    return "$status"
  fi
}

cd "$ROOT_DIR"

run_log local_fmt \
  cargo fmt --check -p mork -p mork-expr -p mork-frontend

RUN_VERUS_FORMAL="${RUN_VERUS_FORMAL:-0}"
if [ "$RUN_VERUS_FORMAL" = "1" ]; then
  run_log local_verus_formal \
    "${ROOT_DIR}/scripts/run_verus_formal_checks.sh"
fi

run_log local_mork_expr_unification \
  env RUSTFLAGS="$RUSTFLAGS_TEST" cargo +nightly test -q -p mork-expr \
    --test unification_semantics

run_log local_mork_parser_errors \
  env RUSTFLAGS="$RUSTFLAGS_TEST" cargo +nightly test -q -p mork \
    --features "$MORK_FEATURES" --test parser_errors

run_log local_mork_query_prefix_rank \
  env RUSTFLAGS="$RUSTFLAGS_TEST" cargo +nightly test -q -p mork \
    --features "$MORK_FEATURES" query_factor_

run_log local_mork_critical_pair_witness \
  env RUSTFLAGS="$RUSTFLAGS_TEST" cargo +nightly test -q -p mork \
    --features "$MORK_FEATURES" --test critical_pair_witness

run_log local_mork_rewrite_semantics \
  env RUSTFLAGS="$RUSTFLAGS_TEST" cargo +nightly test -q -p mork \
    --features "$MORK_FEATURES" --test rewrite_semantics

run_log local_mork_formal_metta_parity \
  env RUSTFLAGS="$RUSTFLAGS_TEST" cargo +nightly test -q -p mork \
    --features "$MORK_FEATURES" --test formal_metta_parity

run_log local_mork_grounding_suite \
  env RUSTFLAGS="$RUSTFLAGS_TEST" cargo +nightly test -q -p mork --features "$MORK_FEATURES"

run_log local_mork_check \
  env RUSTFLAGS="$RUSTFLAGS" cargo +nightly check -p mork --features "$MORK_FEATURES" --all-targets

run_log local_mork_internal_test \
  env RUSTFLAGS="$RUSTFLAGS_TEST" cargo +nightly run -q -p mork --bin mork --features "$MORK_FEATURES" -- test

RUN_QUERY_PLANNER_STRESS="${RUN_QUERY_PLANNER_STRESS:-$RUN_BENCHMARKS}"
if [ "$RUN_QUERY_PLANNER_STRESS" = "1" ]; then
  run_log mork_query_planner_stress \
    env LOG_DIR="$out_dir" RUSTFLAGS="$RUSTFLAGS" FEATURES="$MORK_FEATURES" PYTHON_BIN="$PYTHON_BIN" \
      "${ROOT_DIR}/scripts/run_mork_query_planner_stress.sh" \
      "$QUERY_PLANNER_STRESS_MODE"
fi

RUN_WRITE_RESOURCE_STRESS="${RUN_WRITE_RESOURCE_STRESS:-$RUN_BENCHMARKS}"
if [ "$RUN_WRITE_RESOURCE_STRESS" = "1" ]; then
  run_log mork_write_resource_stress \
    env LOG_DIR="$out_dir" RUSTFLAGS="$RUSTFLAGS" FEATURES="$MORK_FEATURES" PYTHON_BIN="$PYTHON_BIN" \
      "${ROOT_DIR}/scripts/run_mork_write_resource_stress.sh" \
      "$WRITE_RESOURCE_STRESS_MODE"
fi

RUN_WILLIAM_PREDICTIVE_PROBE="${RUN_WILLIAM_PREDICTIVE_PROBE:-$RUN_BENCHMARKS}"
if [ "$RUN_WILLIAM_PREDICTIVE_PROBE" = "1" ]; then
  run_log mork_william_predictive_probe \
    env LOG_DIR="$out_dir" RUSTFLAGS="$RUSTFLAGS" FEATURES="$MORK_FEATURES" PYTHON_BIN="$PYTHON_BIN" \
      "${ROOT_DIR}/scripts/run_mork_william_predictive_probe.sh" \
      "$WILLIAM_PREDICTIVE_MODE"
fi

RUN_RECURSIVE_FRONTIER_PROBE="${RUN_RECURSIVE_FRONTIER_PROBE:-$RUN_BENCHMARKS}"
if [ "$RUN_RECURSIVE_FRONTIER_PROBE" = "1" ]; then
  run_log mork_recursive_frontier_probe \
    env LOG_DIR="$out_dir" RUSTFLAGS="$RUSTFLAGS" FEATURES="$MORK_FEATURES" PYTHON_BIN="$PYTHON_BIN" \
      "${ROOT_DIR}/scripts/run_mork_recursive_frontier_probe.sh" \
      "$RECURSIVE_FRONTIER_MODE"
fi

RUN_CLI_ERROR_BOUNDARY="${RUN_CLI_ERROR_BOUNDARY:-1}"
if [ "$RUN_CLI_ERROR_BOUNDARY" = "1" ]; then
  run_log mork_cli_error_boundary \
    env LOG_DIR="$out_dir" RUSTFLAGS="$RUSTFLAGS" FEATURES="$MORK_FEATURES" \
      "${ROOT_DIR}/scripts/run_mork_cli_error_boundary.sh"
fi

if [ "$RUN_HYPERON" = "1" ]; then
  run_log hyperon_mork_sandbox \
    env LOG_DIR="$out_dir" RUSTFLAGS="$RUSTFLAGS" PYTHON_BIN="$PYTHON_BIN" \
      "${ROOT_DIR}/scripts/run_hyperon_mork_sandbox.sh"
fi

if [ "$RUN_BENCHMARKS" = "1" ]; then
  run_log linalg_cpu_benchmarks \
    env RUSTFLAGS="$RUSTFLAGS" OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
      MORK_RUNS="$MORK_RUNS" PYTHON_BIN="$PYTHON_BIN" \
      "${ROOT_DIR}/linalg/bench_scripts/run_cpu_benchmarks.sh" \
      "$BENCH_MODE"
fi

printf 'MORK integration sandbox logs written to %s\n' "$out_dir"
