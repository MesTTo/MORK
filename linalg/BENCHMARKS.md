# Linalg and MORK Tensor Benchmarks

This directory has two benchmark layers:

- `linalg` microbenchmarks for dense, sparse, blocked, BLAS, and JIT kernels.
- MORK surface benchmarks that load MM2 programs and run tensor sinks through
  the actual `mork` binary.

The benchmark runner is:

```bash
linalg/bench_scripts/run_cpu_benchmarks.sh quick
```

The integration sandbox runner composes this benchmark layer with the local MORK
semantic checks and the Hyperon/Python sandbox:

```bash
RUN_QUERY_PLANNER_STRESS=1 scripts/run_mork_integration_sandbox.sh local
ALLOW_BUSY=1 MORK_RUNS=20 scripts/run_mork_integration_sandbox.sh full
```

Local mode also runs the parser error-boundary gate, which keeps the six-bit
variable and arity limits explicit before benchmark fixtures reach the byte trie,
plus `local_mork_query_prefix_rank`, which directly checks encoded byte-prefix
cardinality caching and query-factor ordering. It also runs
`scripts/run_mork_cli_error_boundary.sh` by default, proving the release
`mork run` command reports those malformed MM2 inputs without panic output.
Each sandbox run writes a top-level `summary.md`, `commands.tsv`,
`report.json`, and `junit.xml` beside the per-gate logs. Nested command-level
stress gates now write the same machine-readable trio in their own output
directories. `report.json` is the lossless local command ledger; `junit.xml` is
suitable for CI systems that ingest JUnit-style test reports.

When `RUN_HYPERON=1`, the sandbox delegates to
`scripts/run_hyperon_mork_sandbox.sh`. That runner now writes its own
`manifest.txt`, `commands.tsv`, `summary.md`, `report.json`, `junit.xml`, and
per-gate logs under `$LOG_DIR/hyperon-mork-sandbox-<timestamp>.<suffix>/`. The
top-level integration summary lists that nested Hyperon summary when the gate
runs.

`RUN_QUERY_PLANNER_STRESS=1` adds the lightweight
`scripts/run_mork_query_planner_stress.sh` gate. It generates an MM2 fixture with
many same-head facts and repeated same-prefix query factors, builds the release
`mork` binary once, and records per-run command timings plus the MORK execution
line in `runs.tsv`, `commands.tsv`, `summary.md`, `report.json`, and
`junit.xml`. The stress runner accepts `quick|full`, records `LOAD_MAX` and
`ALLOW_BUSY`, and preserves clean-run load refusals as exit-75 reports. `full`
enables this stress gate by default because it follows the benchmark path.

`RUN_WRITE_RESOURCE_STRESS=1` adds
`scripts/run_mork_write_resource_stress.sh`. It generates grouped BTM sink output
templates, runs the release `mork` binary with `RUST_LOG=transform=debug`, and
gates on emitted `WriteBench` atoms plus `exclusive_writers`/`reused_writers`
placement telemetry. It also accepts `quick|full` and uses the same load-gate and
report format as the query-planner stress runner.

`quick` runs `jit_bench`, the MORK tensor resource through `cargo run`, and a
direct release-binary MORK loop. The loop defaults to 100 invocations and can be
changed with `MORK_RUNS`. `full` also runs the broader `perf`, `crossover`, and
`graph` bench targets:

```bash
linalg/bench_scripts/run_cpu_benchmarks.sh full
```

The runner writes timestamped logs and a generated `summary.md` under
`linalg/bench_results/<timestamp>.<suffix>/`.  The random suffix prevents
same-second benchmark runs from sharing a report directory. That directory is
intentionally ignored by git because benchmark logs are environment-specific
artifacts. Each benchmark directory also writes `manifest.txt`, `commands.tsv`,
`report.json`, and `junit.xml`; the runner finalizer also writes
`report_verification.log`. A refused clean run records `load_gate` as the failing
report case and exits `75`.

Validate a benchmark directory, or a whole sandbox tree with nested reports, with:

```bash
python3 scripts/verify_sandbox_reports.py --require-success \
  linalg/bench_results/<timestamp>.<suffix>/
```

The report writer and verifier reject malformed command ledgers and command log
paths that are absolute, missing, or escape the report directory. This keeps
copied CI artifacts self-contained. Sandbox and benchmark runners call the
verifier from their exit finalizers and write the result to
`report_verification.log`; an otherwise successful run exits nonzero if report
generation or verification fails, while a failing benchmark keeps its original
gate status for diagnosis.

## Loaded Baseline

On this workstation, the normal baseline includes active Ollama, PeTTa/SWI-Prolog,
Docker/qemu, RustDesk, rust-analyzer, and other background developer tooling.
The numbers collected in this session were taken under that concurrent load, so
they should be read as loaded-workstation comparisons rather than idle-machine
absolute limits.  To measure that loaded baseline explicitly:

```bash
ALLOW_BUSY=1 OPENBLAS_NUM_THREADS=1 RUSTFLAGS='-C target-cpu=native' \
  linalg/bench_scripts/run_cpu_benchmarks.sh quick
```

For an idle-style run, leave `ALLOW_BUSY` unset.  The runner records `/proc/loadavg`
and refuses to run when `load1 > LOAD_MAX`:

```bash
LOAD_MAX=2.0 linalg/bench_scripts/run_cpu_benchmarks.sh quick
```

Use the loaded baseline when comparing against the numbers collected in this
session.  The per-row VM/JIT/BLAS comparisons are still meaningful because they
come from the same process run under the same background pressure.  Use the
load-gated baseline when tracking small regressions.

## What To Read

- `jit_bench.log`: direct VM vs Cranelift JIT vs BLAS/native planner numbers.
- `perf_bench.log`: broader dense/sparse/blocked kernel timings.
- `crossover_bench.log`: density ranges where sparse/blocked kernels beat dense
  BLAS.
- `graph_bench.log`: CSR graph powers and connectivity workloads.
- `mork_tensor_resource_binary_<MORK_RUNS>x.log`: process-level MORK overhead
  for the tensor MM2 resource.
- `commands.tsv`: load gate, benchmark command, and summary command ledger used
  to emit `report.json` and `junit.xml`.
- `manifest.txt`: minimal benchmark metadata, available even when the load gate
  refuses the run before `system.txt` is written.
- `mork-query-planner-stress-<timestamp>.<suffix>/runs.tsv`: process-level timing for
  repeated-prefix query planning over the generated MM2 stress fixture.
- `mork-query-planner-stress-<timestamp>.<suffix>/commands.tsv`: build and per-run gate
  ledger used to emit nested `report.json` and `junit.xml`.
- `mork-query-planner-stress-<timestamp>.<suffix>/summary.md`: compact min/median/mean/max
  view of the query-planner stress run, including fixture and load metadata.
- `mork-write-resource-stress-<timestamp>.<suffix>/runs.tsv`: process-level
  timing and placement telemetry for grouped BTM sink output templates.
- `mork-write-resource-stress-<timestamp>.<suffix>/summary.md`: compact
  write-resource placement summary, including output count and writer reuse
  gates.
- `summary.md`: generated report across all logs present in the result directory.

## Interpretation

The intended MORK architecture is not to make the generic interpreted VM compete
with BLAS directly.  MORK should use pathmap matching to stage symbolic work,
then dispatch recognized tensor workloads into specialized native plans:

- dense matmul and dense attention use BLAS-backed plans;
- CSR x dense uses native sparse row iteration;
- blocked attention is useful only at very low query density;
- interpreted einsum remains the correctness/general fallback.

This follows the usual Rust benchmarking split: microbenchmarks for kernel
changes, command-level runs for integration overhead, and saved baselines for
regression comparison.  Useful references:

- Rust Performance Book benchmarking guidance:
  <https://nnethercote.github.io/perf-book/benchmarking.html>
- Criterion baseline options:
  <https://bheisler.github.io/criterion.rs/book/user_guide/command_line_options.html>
- Hyperfine command-level timing model:
  <https://github.com/sharkdp/hyperfine>
