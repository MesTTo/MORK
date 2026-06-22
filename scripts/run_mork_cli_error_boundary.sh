#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp}"
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
FEATURES="${FEATURES:-grounding}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
. "$ROOT_DIR/scripts/sandbox_report_shell.sh"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="$(create_sandbox_report_dir "$LOG_DIR" "mork-cli-error-boundary" "$stamp")"
commands_tsv="$out_dir/commands.tsv"
printf 'name\tstatus\telapsed_ms\tlog\n' > "$commands_tsv"

too_many_vars="$out_dir/too_many_vars.mm2"
too_many_arity="$out_dir/too_many_arity.mm2"

write_reports() {
  local final_status="$1"
  write_and_verify_sandbox_reports "mork-cli-error-boundary" "$out_dir" "$final_status"
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
  printf "(TooMany \$x0 "
  for i in $(seq 1 64); do
    printf "(Tail \$x%s " "$i"
  done
  printf 'end'
  for _ in $(seq 1 64); do
    printf ')'
  done
  printf ')\n'
} > "$too_many_vars"

{
  printf '(TooWide'
  for i in $(seq 0 63); do
    printf ' arg%s' "$i"
  done
  printf ')\n'
} > "$too_many_arity"

{
  printf 'stamp_utc=%s\n' "$stamp"
  printf 'root_dir=%s\n' "$ROOT_DIR"
  printf 'features=%s\n' "$FEATURES"
  printf 'rustflags=%s\n' "$RUSTFLAGS"
  printf 'too_many_vars=%s\n' "$too_many_vars"
  printf 'too_many_arity=%s\n' "$too_many_arity"
  printf '\n'
  uname -a
  printf '\n'
  uptime
} > "$out_dir/manifest.txt"

cd "$ROOT_DIR"
run_logged_command build_release \
  env RUSTFLAGS="$RUSTFLAGS" cargo +nightly build --release -q -p mork --features "$FEATURES"

run_expected_error() {
  local name="$1"
  local fixture="$2"
  local needle="$3"
  local log="$out_dir/$name.log"
  local output="$out_dir/$name.out.metta"
  local start_ns end_ns elapsed_ms status gate_status

  start_ns="$(date +%s%N)"
  set +e
  "target/release/mork" run "$fixture" --steps 0 --instrumentation 0 "$output" \
    > "$log" 2>&1
  status="$?"
  set -e
  end_ns="$(date +%s%N)"
  elapsed_ms="$(((end_ns - start_ns) / 1000000))"

  gate_status=0
  if [ "$status" = "0" ]; then
    printf '%s unexpectedly succeeded\n' "$name" >&2
    gate_status=1
  elif ! grep -q "$needle" "$log"; then
    printf '%s did not report %s\n' "$name" "$needle" >&2
    gate_status=1
  elif grep -qiE 'panicked at|stack backtrace|thread .* panicked' "$log"; then
    printf '%s produced panic output\n' "$name" >&2
    gate_status=1
  fi

  record_command "$name" "$gate_status" "$elapsed_ms" "${name}.log"
  if [ "$gate_status" != "0" ]; then
    exit 1
  fi
  printf '%s\t%s\t%s\n' "$name" "$status" "$needle" >> "$out_dir/results.tsv"
}

printf 'case\texit_status\texpected_error\n' > "$out_dir/results.tsv"
run_expected_error too_many_vars "$too_many_vars" TooManyVars
run_expected_error too_many_arity "$too_many_arity" TooManyArity

{
  printf '# MORK CLI Error Boundary Summary\n\n'
  printf "Timestamp: \`%s\`\n\n" "$stamp"
  printf "The command-level \`mork run\` path rejected malformed MM2 input without panic output.\n\n"
  printf '| case | exit status | expected error |\n'
  printf '| --- | ---: | --- |\n'
  tail -n +2 "$out_dir/results.tsv" | while IFS=$'\t' read -r name status needle; do
    printf "| \`%s\` | \`%s\` | \`%s\` |\n" "$name" "$status" "$needle"
  done
  printf "\nLogs and fixtures are under \`%s\`.\n" "$out_dir"
  printf '\n## Machine Reports\n\n'
  printf -- "- \`commands.tsv\`\n"
  printf -- "- \`report.json\`\n"
  printf -- "- \`junit.xml\`\n"
  printf -- "- \`report_verification.log\`\n"
} > "$out_dir/summary.md"

printf 'MORK CLI error-boundary logs written to %s\n' "$out_dir"
