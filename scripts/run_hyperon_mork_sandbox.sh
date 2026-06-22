#!/usr/bin/env bash
set -euo pipefail

HYPERON_DIR="${HYPERON_DIR:-/home/user/Dev/hyperon-build-src}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
LOG_DIR="${LOG_DIR:-/tmp}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT_DIR/scripts/sandbox_report_shell.sh"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="$(create_sandbox_report_dir "$LOG_DIR" "hyperon-mork-sandbox" "$stamp")"
commands_tsv="$out_dir/commands.tsv"
printf 'name\tstatus\telapsed_ms\tlog\n' > "$commands_tsv"

{
  printf 'stamp_utc=%s\n' "$stamp"
  printf 'hyperon_dir=%s\n' "$HYPERON_DIR"
  printf 'python_bin=%s\n' "$PYTHON_BIN"
  printf 'rustflags=%s\n' "$RUSTFLAGS"
  printf 'log_dir=%s\n' "$LOG_DIR"
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
    printf '# Hyperon MORK Sandbox Summary\n\n'
    printf "Timestamp: \`%s\`\n\n" "$stamp"
    printf -- "- Final status: \`%s\`.\n" "$final_status"
    printf -- "- Hyperon checkout: \`%s\`.\n" "$HYPERON_DIR"
    printf -- "- Python: \`%s\`.\n" "$PYTHON_BIN"
    printf -- "- RUSTFLAGS: \`%s\`.\n" "$RUSTFLAGS"
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
    printf '\n## Semantic Coverage\n\n'
    printf -- "- Rust \`mork-space\` gates cover constructor, snapshot index selection, same-head visit counts, and same-head query stability.\n"
    printf -- '- Python sandbox gates cover multiset atoms, fork-space snapshot isolation, symbolic snapshot matching, joins, batch matching, and expected error boundaries.\n'
    printf '\n## Machine Reports\n\n'
    printf -- "- \`report.json\`\n"
    printf -- "- \`junit.xml\`\n"
    printf -- "- \`report_verification.log\`\n"
  } > "$summary"

  write_and_verify_sandbox_reports "hyperon-mork-sandbox" "$out_dir" "$final_status"
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

run_log hyperon_checkout_exists \
  test -d "$HYPERON_DIR"

run_log hyperon_python_package_exists \
  test -d "$HYPERON_DIR/python/hyperon"

cd "$HYPERON_DIR"
export RUSTFLAGS

run_log rust_mork_grounding_space \
  cargo test -q -p hyperon mork_grounding_space --lib

run_log rust_new_mork_space \
  cargo test -q -p hyperon new_mork_space --lib

run_log rust_mork_space_feature_backend \
  cargo test -q -p hyperon --features mork-space \
    grounding_space_uses_mork_snapshot_index_when_mork_space_feature_enabled --lib

run_log rust_mork_space_same_head_1500 \
  cargo test -q -p hyperon --features mork-space visit_counts_many_same_head_atoms --lib

run_log rust_mork_space_same_head_query \
  cargo test -q -p hyperon --features mork-space query_and_visit_survive_large_same_head_load --lib

run_log python_test_scripts \
  env PYTHONPATH=python:python/tests "$PYTHON_BIN" -m pytest -q \
    python/tests/test_run_metta.py::MeTTaTest::test_scripts

run_log python_mork_sandbox_self_check \
  env PYTHONPATH=python:python/tests "$PYTHON_BIN" - <<'PY'
from pathlib import Path

from hyperon import Environment, MeTTa

root = Path("python/sandbox/mork").resolve()
paths = [
    root / "atom_multiset_semantics.metta",
    root / "fork_snapshot_stress.metta",
    root / "portable_snapshot_parallel.metta",
]

for path in paths:
    metta = MeTTa(env_builder=Environment.test_env())
    result = metta.load_module_at_path(str(path))
    print(f"{path.name}: {result!r}")
PY

run_log python_mork_sandbox_error_boundary \
  env PYTHONPATH=python:python/tests "$PYTHON_BIN" - <<'PY'
from pathlib import Path

from hyperon import Environment, MeTTa

root = Path("python/sandbox/mork").resolve()
cases = [
    ("arithmetic_error_hardening.metta", "DivisionByZero"),
    ("interpreter_error_boundary.metta", "malformed-collapse-row"),
]

for name, needle in cases:
    metta = MeTTa(env_builder=Environment.test_env())
    try:
        metta.load_module_at_path(str(root / name))
    except RuntimeError as err:
        text = str(err)
        if needle not in text:
            raise AssertionError(f"{name}: expected {needle!r} in {text!r}") from err
        print(f"{name}: expected RuntimeError containing {needle!r}: {text}")
    else:
        raise AssertionError(f"{name}: expected RuntimeError containing {needle!r}")
PY

printf 'Hyperon MORK sandbox logs written to %s\n' "$out_dir"
