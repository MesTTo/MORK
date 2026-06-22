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
STEPS="${STEPS:-1}"
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
    : "${WRITE_GROUPS:=3}"
    : "${TEMPLATES_PER_GROUP:=4}"
    : "${MORK_WRITE_RESOURCE_RUNS:=3}"
    ;;
  full)
    : "${WRITE_GROUPS:=12}"
    : "${TEMPLATES_PER_GROUP:=5}"
    : "${MORK_WRITE_RESOURCE_RUNS:=10}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
RUNS="$MORK_WRITE_RESOURCE_RUNS"

if (( WRITE_GROUPS < 1 )); then
  printf 'WRITE_GROUPS must be positive, got %s\n' "$WRITE_GROUPS" >&2
  exit 2
fi
if (( TEMPLATES_PER_GROUP < 1 )); then
  printf 'TEMPLATES_PER_GROUP must be positive, got %s\n' "$TEMPLATES_PER_GROUP" >&2
  exit 2
fi
if (( RUNS < 1 )); then
  printf 'MORK_WRITE_RESOURCE_RUNS must be positive, got %s\n' "$RUNS" >&2
  exit 2
fi
template_count=$((WRITE_GROUPS * TEMPLATES_PER_GROUP))
if (( template_count > 60 )); then
  printf 'WRITE_GROUPS*TEMPLATES_PER_GROUP must be at most 60, got %s\n' "$template_count" >&2
  printf 'MM2 arity tags are limited to 63 children; keep the generated O-template below that.\n' >&2
  exit 2
fi
expected_outputs="$template_count"
expected_exclusive_writers="$WRITE_GROUPS"
expected_reused_writers="$((template_count - WRITE_GROUPS))"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="$(create_sandbox_report_dir "$LOG_DIR" "mork-write-resource-stress" "$stamp")"
fixture="$out_dir/write_resource_stress.mm2"
commands_tsv="$out_dir/commands.tsv"
printf 'name\tstatus\telapsed_ms\tlog\n' > "$commands_tsv"
load1="$(awk '{ print $1 }' /proc/loadavg)"

write_reports() {
  local final_status="$1"
  write_and_verify_sandbox_reports "mork-write-resource-stress-$MODE" "$out_dir" "$final_status"
}

trap 'finish_sandbox_exit "$?" write_reports' EXIT

record_command() {
  local name="$1"
  local status="$2"
  local elapsed_ms="$3"
  local log_name="$4"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$elapsed_ms" "$log_name" >> "$commands_tsv"
}

run_logged_command() {
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
  record_command "$name" "$status" "$elapsed_ms" "${name}.log"
  if [ "$status" != "0" ]; then
    return "$status"
  fi
}

{
  printf 'stamp_utc=%s\n' "$stamp"
  printf 'mode=%s\n' "$MODE"
  printf 'root_dir=%s\n' "$ROOT_DIR"
  printf 'fixture=%s\n' "$fixture"
  printf 'load1=%s\n' "$load1"
  printf 'load_max=%s\n' "$LOAD_MAX"
  printf 'allow_busy=%s\n' "$ALLOW_BUSY"
  printf 'write_groups=%s\n' "$WRITE_GROUPS"
  printf 'templates_per_group=%s\n' "$TEMPLATES_PER_GROUP"
  printf 'template_count=%s\n' "$template_count"
  printf 'expected_outputs=%s\n' "$expected_outputs"
  printf 'expected_exclusive_writers=%s\n' "$expected_exclusive_writers"
  printf 'expected_reused_writers=%s\n' "$expected_reused_writers"
  printf 'runs=%s\n' "$RUNS"
  printf 'steps=%s\n' "$STEPS"
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
      printf 'Refusing to run clean write-resource stress: load1=%s exceeds LOAD_MAX=%s\n' "$load1" "$LOAD_MAX"
      printf 'Set ALLOW_BUSY=1 to run anyway and label the result as noisy.\n'
    } | tee "$out_dir/ABORTED_BUSY.txt" >&2
    ps -eo pid,ppid,pcpu,pmem,comm,args --sort=-pcpu | sed -n '1,25p' > "$out_dir/top-processes.txt"
    record_command load_gate 75 0 ABORTED_BUSY.txt
    exit 75
  fi
fi
record_command load_gate 0 0 manifest.txt

{
  printf 'stamp_utc=%s\n' "$stamp"
  printf 'mode=%s\n' "$MODE"
  printf 'load1=%s\n' "$load1"
  printf 'load_max=%s\n' "$LOAD_MAX"
  printf 'allow_busy=%s\n' "$ALLOW_BUSY"
  printf 'rustflags=%s\n' "$RUSTFLAGS"
  printf 'features=%s\n' "$FEATURES"
  printf '\n'
  uname -a
  printf '\n'
  uptime
  printf '\n'
  ps -eo pid,ppid,pcpu,pmem,comm,args --sort=-pcpu | sed -n '1,25p'
} > "$out_dir/system.txt"

WRITE_GROUPS="$WRITE_GROUPS" TEMPLATES_PER_GROUP="$TEMPLATES_PER_GROUP" FIXTURE="$fixture" \
  "$PYTHON_BIN" - <<'PY'
import os

groups = int(os.environ["WRITE_GROUPS"])
templates_per_group = int(os.environ["TEMPLATES_PER_GROUP"])
fixture = os.environ["FIXTURE"]

with open(fixture, "w", encoding="utf-8") as out:
    out.write("(Seed item0)\n\n")
    out.write("(exec 0\n")
    out.write("  (, (Seed $x))\n")
    out.write("  (O")
    for group in range(groups):
        for leaf in range(templates_per_group):
            out.write(f" (+ (WriteBench group{group} $x leaf{leaf}))")
    out.write("))\n")
PY

cd "$ROOT_DIR"
run_logged_command build_release \
  env RUSTFLAGS="$RUSTFLAGS" cargo +nightly build --release -q -p mork --features "$FEATURES"

printf 'run\telapsed_ms\toutputs\texclusive_writers\treused_writers\tplacement_line\texecution_line\n' > "$out_dir/runs.tsv"
for run in $(seq 1 "$RUNS"); do
  log="$out_dir/run_$run.log"
  output="$out_dir/output_$run.metta"
  start_ns="$(date +%s%N)"
  set +e
  RUST_LOG=transform=debug "target/release/mork" run "$fixture" --steps "$STEPS" --instrumentation 0 "$output" \
    > "$log" 2>&1
  command_status="$?"
  set -e
  end_ns="$(date +%s%N)"
  elapsed_ms="$(((end_ns - start_ns) / 1000000))"
  outputs="$(grep -c '^(WriteBench ' "$output" || true)"
  placement_line="$(grep -m1 'write_resource_placement' "$log" || true)"
  exclusive_writers="$(printf '%s\n' "$placement_line" | sed -n 's/.*exclusive_writers=\([0-9][0-9]*\).*/\1/p')"
  reused_writers="$(printf '%s\n' "$placement_line" | sed -n 's/.*reused_writers=\([0-9][0-9]*\).*/\1/p')"
  execution_line="$(grep -m1 '^executing ' "$log" || true)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run" "$elapsed_ms" "$outputs" "${exclusive_writers:-missing}" "${reused_writers:-missing}" \
    "$placement_line" "$execution_line" >> "$out_dir/runs.tsv"

  gate_status="$command_status"
  if [ "$command_status" = "0" ] && [ "$outputs" != "$expected_outputs" ]; then
    gate_status=1
  fi
  if [ "$command_status" = "0" ] && [ "${exclusive_writers:-}" != "$expected_exclusive_writers" ]; then
    gate_status=1
  fi
  if [ "$command_status" = "0" ] && [ "${reused_writers:-}" != "$expected_reused_writers" ]; then
    gate_status=1
  fi
  record_command "write_resource_run_$run" "$gate_status" "$elapsed_ms" "run_$run.log"
  if [ "$command_status" != "0" ]; then
    printf 'mork run %s exited with status %s; see %s\n' "$run" "$command_status" "$log" >&2
    exit "$command_status"
  fi
  if [ "$outputs" != "$expected_outputs" ]; then
    printf 'expected %s WriteBench atoms, got %s in %s\n' "$expected_outputs" "$outputs" "$output" >&2
    exit 1
  fi
  if [ "${exclusive_writers:-}" != "$expected_exclusive_writers" ]; then
    printf 'expected %s exclusive writers, got %s in %s\n' "$expected_exclusive_writers" "${exclusive_writers:-missing}" "$log" >&2
    exit 1
  fi
  if [ "${reused_writers:-}" != "$expected_reused_writers" ]; then
    printf 'expected %s reused writers, got %s in %s\n' "$expected_reused_writers" "${reused_writers:-missing}" "$log" >&2
    exit 1
  fi
done

RUNS_TSV="$out_dir/runs.tsv" MANIFEST="$out_dir/manifest.txt" SUMMARY="$out_dir/summary.md" \
  "$PYTHON_BIN" - <<'PY'
import csv
import os
import statistics
from pathlib import Path

runs_tsv = Path(os.environ["RUNS_TSV"])
manifest = Path(os.environ["MANIFEST"])
summary = Path(os.environ["SUMMARY"])

rows = []
with runs_tsv.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        row["elapsed_ms"] = int(row["elapsed_ms"])
        row["outputs"] = int(row["outputs"])
        row["exclusive_writers"] = int(row["exclusive_writers"])
        row["reused_writers"] = int(row["reused_writers"])
        rows.append(row)

elapsed = [row["elapsed_ms"] for row in rows]
manifest_text = manifest.read_text(encoding="utf-8", errors="replace")
manifest_kv = {}
for line in manifest_text.splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        manifest_kv[key] = value
load_line = next((line.strip() for line in manifest_text.splitlines() if "load average:" in line), "unknown")

def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

load1 = as_float(manifest_kv.get("load1"))
load_max = as_float(manifest_kv.get("load_max"))
allow_busy = manifest_kv.get("allow_busy", "unknown")
if allow_busy == "1":
    if load1 is not None and load_max is not None and load1 > load_max:
        load_gate_note = "bypassed by ALLOW_BUSY=1; noisy loaded-workstation result"
    else:
        load_gate_note = "bypassed by ALLOW_BUSY=1"
else:
    load_gate_note = "passed before build and run"

lines = [
    "# MORK Write Resource Stress Summary",
    "",
    f"Timestamp: `{manifest_kv.get('stamp_utc', 'unknown')}`",
    f"Mode: `{manifest_kv.get('mode', 'unknown')}`",
    f"Load gate: `{load_gate_note}`",
    "",
    "This is a command-level stress gate for grouped BTM sink output prefixes.",
    "It verifies both emitted output atoms and the debug placement telemetry from `RUST_LOG=transform=debug`.",
    "",
    "## Fixture",
    "",
    f"- Groups: `{manifest_kv.get('write_groups', 'unknown')}`.",
    f"- Templates per group: `{manifest_kv.get('templates_per_group', 'unknown')}`.",
    f"- Output template requests: `{manifest_kv.get('template_count', 'unknown')}`.",
    f"- Expected exclusive write zippers: `{manifest_kv.get('expected_exclusive_writers', 'unknown')}`.",
    f"- Expected reused writers: `{manifest_kv.get('expected_reused_writers', 'unknown')}`.",
    f"- Steps: `{manifest_kv.get('steps', 'unknown')}`.",
    "",
    "## Timing",
    "",
    "| runs | min | median | mean | max |",
    "| ---: | ---: | ---: | ---: | ---: |",
    f"| {len(rows)} | {min(elapsed)} ms | {statistics.median(elapsed):.2f} ms | {statistics.mean(elapsed):.2f} ms | {max(elapsed)} ms |",
    "",
    "## Raw Runs",
    "",
    "| run | elapsed | outputs | exclusive writers | reused writers | execution line |",
    "| ---: | ---: | ---: | ---: | ---: | --- |",
]
for row in rows:
    lines.append(
        f"| {row['run']} | {row['elapsed_ms']} ms | {row['outputs']} | "
        f"{row['exclusive_writers']} | {row['reused_writers']} | `{row['execution_line']}` |"
    )
lines.extend(
    [
        "",
        "## Placement Lines",
        "",
    ]
)
for row in rows:
    lines.append(f"- run {row['run']}: `{row['placement_line']}`")
lines.extend(
    [
        "",
        "## Environment",
        "",
        f"- LOAD_MAX: `{manifest_kv.get('load_max', 'unknown')}`.",
        f"- ALLOW_BUSY: `{manifest_kv.get('allow_busy', 'unknown')}`.",
        f"- RUSTFLAGS: `{manifest_kv.get('rustflags', 'unknown')}`.",
        f"- Features: `{manifest_kv.get('features', 'unknown')}`.",
        f"- Recorded load1: `{manifest_kv.get('load1', 'unknown')}`.",
        f"- Uptime line: `{load_line}`.",
        "",
        "## Machine Reports",
        "",
        "- `commands.tsv`",
        "- `system.txt`",
        "- `report.json`",
        "- `junit.xml`",
        "- `report_verification.log`",
        "",
    ]
)
summary.write_text("\n".join(lines), encoding="utf-8")
PY

printf '%s\n' "$out_dir/summary.md"
printf 'MORK write resource stress logs written to %s\n' "$out_dir"
