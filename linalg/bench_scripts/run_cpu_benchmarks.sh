#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
LOAD_MAX="${LOAD_MAX:-2.0}"
ALLOW_BUSY="${ALLOW_BUSY:-0}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
MORK_RUNS="${MORK_RUNS:-100}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "$MODE" in
  quick|full) ;;
  *)
    printf 'usage: %s [quick|full]\n' "$0" >&2
    exit 2
    ;;
esac
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
. "$ROOT_DIR/scripts/sandbox_report_shell.sh"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="$(create_sandbox_report_dir "linalg/bench_results" "" "$stamp")"
commands_tsv="$out_dir/commands.tsv"
printf 'name\tstatus\telapsed_ms\tlog\n' > "$commands_tsv"

load1="$(awk '{ print $1 }' /proc/loadavg)"

{
  printf 'stamp_utc=%s\n' "$stamp"
  printf 'mode=%s\n' "$MODE"
  printf 'root_dir=%s\n' "$ROOT_DIR"
  printf 'load1=%s\n' "$load1"
  printf 'load_max=%s\n' "$LOAD_MAX"
  printf 'allow_busy=%s\n' "$ALLOW_BUSY"
  printf 'openblas_num_threads=%s\n' "$OPENBLAS_NUM_THREADS"
  printf 'rustflags=%s\n' "$RUSTFLAGS"
  printf 'mork_runs=%s\n' "$MORK_RUNS"
  printf '\n'
  uname -a
  printf '\n'
  uptime
} > "$out_dir/manifest.txt"

write_reports() {
  local final_status="$1"
  write_and_verify_sandbox_reports "mork-cpu-benchmarks-$MODE" "$out_dir" "$final_status"
}

trap 'finish_sandbox_exit "$?" write_reports' EXIT

record_command() {
  local name="$1"
  local status="$2"
  local elapsed_ms="$3"
  local log_name="$4"
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$elapsed_ms" "$log_name" >> "$commands_tsv"
}

if [ "$ALLOW_BUSY" != "1" ]; then
  if awk -v load="$load1" -v max="$LOAD_MAX" 'BEGIN { exit !(load > max) }'; then
    {
      printf 'Refusing to run clean benchmarks: load1=%s exceeds LOAD_MAX=%s\n' "$load1" "$LOAD_MAX"
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
  printf 'allow_busy=%s\n' "$ALLOW_BUSY"
  printf 'openblas_num_threads=%s\n' "$OPENBLAS_NUM_THREADS"
  printf 'rustflags=%s\n' "$RUSTFLAGS"
  printf 'mork_runs=%s\n' "$MORK_RUNS"
  printf '\n'
  uname -a
  printf '\n'
  uptime
  printf '\n'
  lscpu
  printf '\n'
  free -h
  printf '\n'
  ps -eo pid,ppid,pcpu,pmem,comm,args --sort=-pcpu | sed -n '1,25p'
} > "$out_dir/system.txt"

export OPENBLAS_NUM_THREADS RUSTFLAGS

run_log() {
  local name="$1"
  shift
  local start_ns end_ns elapsed_ms status
  start_ns="$(date +%s%N)"
  set +e
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n\n'
    /usr/bin/time -f '\nreal_sec=%e user_sec=%U sys_sec=%S max_rss_kb=%M' "$@"
  } 2>&1 | tee "$out_dir/$name.log"
  status="${PIPESTATUS[0]}"
  set -e
  end_ns="$(date +%s%N)"
  elapsed_ms="$(((end_ns - start_ns) / 1000000))"
  record_command "$name" "$status" "$elapsed_ms" "$name.log"
  if [ "$status" != "0" ]; then
    return "$status"
  fi
}

run_log jit_bench \
  cargo +nightly run -p linalg --release --features jit,blas --example jit_bench

run_log mork_tensor_resource \
  cargo +nightly run -p mork --bin mork --release --features einsum_blas -- \
    run kernel/resources/einsum_f32_attention.mm2 --steps 7 --instrumentation 0

run_log "mork_tensor_resource_binary_${MORK_RUNS}x" \
  bash -lc "for ((i = 0; i < ${MORK_RUNS}; i++)); do target/release/mork run kernel/resources/einsum_f32_attention.mm2 --steps 7 --instrumentation 0 >/dev/null; done"

if [ "$MODE" = "full" ]; then
  run_log perf_bench \
    cargo +nightly bench -p linalg --features jit,blas --bench perf
  run_log crossover_bench \
    cargo +nightly bench -p linalg --features jit,blas --bench crossover
  run_log graph_bench \
    cargo +nightly bench -p linalg --features jit,blas --bench graph
fi

run_log summarize_benchmarks "$PYTHON_BIN" linalg/bench_scripts/summarize_benchmarks.py "$out_dir"
printf 'bench logs written to %s\n' "$out_dir"
