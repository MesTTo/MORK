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
    : "${CHAIN_LEN:=8}"
    : "${STEP_BUDGETS:=1 2 4 8}"
    ;;
  full)
    : "${CHAIN_LEN:=32}"
    : "${STEP_BUDGETS:=1 2 4 8 16 32}"
    ;;
  *)
    usage
    exit 2
    ;;
esac

if (( CHAIN_LEN < 1 || CHAIN_LEN > 63 )); then
  printf 'CHAIN_LEN must be between 1 and 63, got %s\n' "$CHAIN_LEN" >&2
  exit 2
fi

for budget in $STEP_BUDGETS; do
  if ! [[ "$budget" =~ ^[0-9]+$ ]] || (( budget < 1 || budget > CHAIN_LEN )); then
    printf 'STEP_BUDGETS entries must be positive integers <= CHAIN_LEN; got %s\n' "$budget" >&2
    exit 2
  fi
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="$(create_sandbox_report_dir "$LOG_DIR" "mork-recursive-frontier-probe" "$stamp")"
fixture="$out_dir/recursive_frontier.mm2"
commands_tsv="$out_dir/commands.tsv"
runs_tsv="$out_dir/runs.tsv"
printf 'name\tstatus\telapsed_ms\tlog\n' > "$commands_tsv"
printf 'steps\telapsed_ms\treach_count\texpected_reach\tnew_reach\tunifications\twrites\ttransitions\texecution_line\n' > "$runs_tsv"
load1="$(awk '{ print $1 }' /proc/loadavg)"

write_summary() {
  local final_status="$1"
  local summary="$out_dir/summary.md"
  {
    printf '# MORK Recursive Frontier Probe Summary\n\n'
    printf "Timestamp: \`%s\`\n\n" "$stamp"
    printf -- "- Mode: \`%s\`.\n" "$MODE"
    printf -- "- Final status: \`%s\`.\n" "$final_status"
    printf -- "- Chain length: \`%s\`.\n" "$CHAIN_LEN"
    printf -- "- Step budgets: \`%s\`.\n" "$STEP_BUDGETS"
    printf -- "- Load: \`%s\` with LOAD_MAX=\`%s\`, ALLOW_BUSY=\`%s\`.\n" "$load1" "$LOAD_MAX" "$ALLOW_BUSY"
    printf -- "- RUSTFLAGS: \`%s\`.\n" "$RUSTFLAGS"
    printf -- "- Features: \`%s\`.\n" "$FEATURES"
    printf '\n## Purpose\n\n'
    printf 'This is a MeTTa-facing recursive baseline, not a tabling implementation. It queues repeated transitive-reachability `exec` rules over a chain graph and records how many `Reach` facts are visible after bounded scheduler steps.\n\n'
    printf 'The expected frontier is `min(steps, chain_len) + 1`, including the seed `(Reach n0 n0)`. Later semi-naive or tabled execution work should preserve these counts while reducing repeated joins and unifications.\n\n'
    printf '## Runs\n\n'
    printf '| steps | elapsed | reach | expected | new reach | unifications | writes | transitions |\n'
    printf '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    if [ -s "$runs_tsv" ]; then
      tail -n +2 "$runs_tsv" | while IFS=$'\t' read -r steps elapsed reach expected new_reach unifications writes transitions _execution_line; do
        printf '| `%s` | `%s ms` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` |\n' \
          "$steps" "$elapsed" "$reach" "$expected" "$new_reach" "$unifications" "$writes" "$transitions"
      done
    fi
    printf '\n## Machine Reports\n\n'
    printf -- "- \`manifest.txt\`\n"
    printf -- "- \`commands.tsv\`\n"
    printf -- "- \`runs.tsv\`\n"
    printf -- "- \`report.json\`\n"
    printf -- "- \`junit.xml\`\n"
    printf -- "- \`report_verification.log\`\n"
  } > "$summary"

  write_and_verify_sandbox_reports "mork-recursive-frontier-probe-$MODE" "$out_dir" "$final_status"
}

trap 'finish_sandbox_exit "$?" write_summary' EXIT

record_command() {
  local name="$1"
  local status="$2"
  local elapsed_ms="$3"
  local log_name="$4"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$elapsed_ms" "$log_name" >> "$commands_tsv"
}

{
  printf 'stamp_utc=%s\n' "$stamp"
  printf 'mode=%s\n' "$MODE"
  printf 'root_dir=%s\n' "$ROOT_DIR"
  printf 'fixture=%s\n' "$fixture"
  printf 'load1=%s\n' "$load1"
  printf 'load_max=%s\n' "$LOAD_MAX"
  printf 'allow_busy=%s\n' "$ALLOW_BUSY"
  printf 'chain_len=%s\n' "$CHAIN_LEN"
  printf 'step_budgets=%s\n' "$STEP_BUDGETS"
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
      printf 'Refusing to run clean recursive frontier probe: load1=%s exceeds LOAD_MAX=%s\n' "$load1" "$LOAD_MAX"
      printf 'Set ALLOW_BUSY=1 to run anyway and label the result as noisy.\n'
    } | tee "$out_dir/ABORTED_BUSY.txt" >&2
    ps -eo pid,ppid,pcpu,pmem,comm,args --sort=-pcpu | sed -n '1,25p' > "$out_dir/top-processes.txt"
    record_command load_gate 75 0 ABORTED_BUSY.txt
    exit 75
  fi
fi
record_command load_gate 0 0 manifest.txt

CHAIN_LEN="$CHAIN_LEN" FIXTURE="$fixture" "$PYTHON_BIN" - <<'PY'
import os

chain_len = int(os.environ["CHAIN_LEN"])
fixture = os.environ["FIXTURE"]

with open(fixture, "w", encoding="utf-8") as out:
    for i in range(chain_len):
        out.write(f"(Edge n{i} n{i + 1})\n")
    out.write("(Reach n0 n0)\n\n")
    for step in range(chain_len):
        out.write(f"(exec {step}\n")
        out.write("  (, (Reach n0 $y) (Edge $y $z))\n")
        out.write("  (, (Reach n0 $z)))\n")
PY

cd "$ROOT_DIR"
build_log="$out_dir/build_release.log"
build_start_ns="$(date +%s%N)"
set +e
{
  printf '$ env RUSTFLAGS=%q cargo +nightly build --release -q -p mork --features %q\n\n' \
    "$RUSTFLAGS" "$FEATURES"
  env RUSTFLAGS="$RUSTFLAGS" cargo +nightly build --release -q -p mork --features "$FEATURES"
} > "$build_log" 2>&1
build_status="$?"
set -e
build_end_ns="$(date +%s%N)"
build_elapsed_ms="$(((build_end_ns - build_start_ns) / 1000000))"
record_command build_release "$build_status" "$build_elapsed_ms" build_release.log
if [ "$build_status" != "0" ]; then
  exit "$build_status"
fi

for steps in $STEP_BUDGETS; do
  log="$out_dir/run_steps_${steps}.log"
  output="$out_dir/output_steps_${steps}.metta"
  start_ns="$(date +%s%N)"
  set +e
  "target/release/mork" run "$fixture" --steps "$steps" --instrumentation 0 \
    --query-execution-stats "$output" > "$log" 2>&1
  command_status="$?"
  set -e
  end_ns="$(date +%s%N)"
  elapsed_ms="$(((end_ns - start_ns) / 1000000))"
  execution_line="$(grep -m1 '^executing ' "$log" || true)"
  unifications="$(sed -n 's/.*unifications \([0-9][0-9]*\).*/\1/p' <<< "$execution_line")"
  writes="$(sed -n 's/.*writes \([0-9][0-9]*\).*/\1/p' <<< "$execution_line")"
  transitions="$(sed -n 's/.*transitions \([0-9][0-9]*\).*/\1/p' <<< "$execution_line")"
  : "${unifications:=0}"
  : "${writes:=0}"
  : "${transitions:=0}"
  reach_count="$(grep -c '^(Reach ' "$output" || true)"
  expected_reach=$((steps + 1))
  new_reach="$steps"
  if (( expected_reach > CHAIN_LEN + 1 )); then
    expected_reach=$((CHAIN_LEN + 1))
    new_reach="$CHAIN_LEN"
  fi
  gate_status="$command_status"
  if [ "$command_status" = "0" ] && [ "$reach_count" != "$expected_reach" ]; then
    gate_status=1
  fi
  record_command "frontier_steps_$steps" "$gate_status" "$elapsed_ms" "run_steps_${steps}.log"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$steps" "$elapsed_ms" "$reach_count" "$expected_reach" "$new_reach" \
    "$unifications" "$writes" "$transitions" "$execution_line" >> "$runs_tsv"
  if [ "$command_status" != "0" ]; then
    printf 'mork run steps=%s exited with status %s; see %s\n' "$steps" "$command_status" "$log" >&2
    exit "$command_status"
  fi
  if [ "$reach_count" != "$expected_reach" ]; then
    printf 'expected %s Reach atoms, got %s in %s\n' "$expected_reach" "$reach_count" "$output" >&2
    exit 1
  fi
done

printf 'MORK recursive frontier probe logs written to %s\n' "$out_dir"
