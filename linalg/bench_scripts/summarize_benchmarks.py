#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def kv(system: str, key: str, default: str = "unknown") -> str:
    m = re.search(rf"^{re.escape(key)}=(.*)$", system, re.MULTILINE)
    return m.group(1).strip() if m else default


def positive_int(value: str, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def first(pattern: str, text: str):
    return re.search(pattern, text, re.MULTILINE)


def all_matches(pattern: str, text: str):
    return list(re.finditer(pattern, text, re.MULTILINE))


def table(headers, rows) -> list[str]:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <bench-result-dir>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    system = read(root / "system.txt")
    jit = read(root / "jit_bench.log")
    perf = read(root / "perf_bench.log")
    crossover = read(root / "crossover_bench.log")
    graph = read(root / "graph_bench.log")
    mork = read(root / "mork_tensor_resource.log")
    mork_runs = positive_int(kv(system, "mork_runs", "100"), 100)
    mork_binary_path = root / f"mork_tensor_resource_binary_{mork_runs}x.log"
    if not mork_binary_path.exists():
        mork_binary_path = root / "mork_tensor_resource_binary_100x.log"
    mork_binary = read(mork_binary_path)

    lines: list[str] = [
        "# Loaded-System Benchmark Summary",
        "",
        f"Timestamp: `{kv(system, 'stamp_utc', root.name)}`",
        "",
        "These numbers are for the recorded workstation state, not an ideal idle CPU.",
        f"The run used `OPENBLAS_NUM_THREADS={kv(system, 'openblas_num_threads')}` and `RUSTFLAGS='{kv(system, 'rustflags')}'`.",
        f"Recorded load was `load1={kv(system, 'load1')}` with `allow_busy={kv(system, 'allow_busy')}`.",
    ]

    cpu = first(r"Model name:\s+(.*)", system)
    if cpu:
        lines.append(f"CPU: `{cpu.group(1).strip()}`.")
    top = first(r"^\s*\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+\S+\s+(.+)$", system)
    if top:
        lines.append(f"Top recorded process: `{top.group(1).strip()}`.")

    dense_rows = []
    for m in all_matches(
        r"matmul\s+(\d+)x\1\s+VM\s+([\d.]+) µs\s+JIT\s+([\d.]+) µs\s+([\d.]+)× faster",
        jit,
    ):
        shape, vm, jit_us, ratio = m.groups()
        dense_rows.append((shape, vm, jit_us, "", ratio))

    blas = {
        m.group(1): (m.group(2), m.group(3))
        for m in all_matches(
            r"matmul\s+(\d+)x\1\s+JIT\s+[\d.]+ µs\s+plan\(BlasMatmul\)\s+([\d.]+) µs\s+\(([\d.]+)× JIT\)",
            jit,
        )
    }
    dense_table = []
    for shape, vm, jit_us, _blank, ratio in dense_rows:
        plan = blas.get(shape)
        if plan:
            dense_table.append([f"{shape}x{shape}", f"{vm} us", f"{jit_us} us", f"{plan[0]} us", f"{plan[1]}x vs JIT"])
    if dense_table:
        lines += ["", "## Dense Matmul", ""]
        lines += table(["shape", "VM", "JIT", "BLAS plan", "planner speedup"], dense_table)

    score_rows = []
    for m in all_matches(
        r"b=(\d+)\s+h=(\d+)\s+q=k=(\d+)\s+d=(\d+)\s+JIT\s+([\d.]+) µs\s+plan\(BlasAttentionScores\)\s+([\d.]+) µs\s+\(([\d.]+)× JIT\)",
        jit,
    ):
        b, h, qk, d, jit_us, plan, speed = m.groups()
        score_rows.append([f"b={b} h={h} q=k={qk} d={d}", f"{jit_us} us", f"{plan} us", f"{speed}x"])
    if score_rows:
        lines += ["", "## Attention Score Backend", ""]
        lines += table(["shape", "JIT", "BLAS score plan", "speedup"], score_rows)

    apply_rows = []
    for m in all_matches(
        r"b=(\d+)\s+h=(\d+)\s+q=k=(\d+)\s+d=(\d+)\s+JIT\s+([\d.]+) µs\s+plan\(BlasAttentionApply\)\s+([\d.]+) µs\s+\(([\d.]+)× JIT\)",
        jit,
    ):
        b, h, qk, d, jit_us, plan, speed = m.groups()
        apply_rows.append([f"b={b} h={h} q=k={qk} d={d}", f"{jit_us} us", f"{plan} us", f"{speed}x"])
    if apply_rows:
        lines += ["", "## Attention Apply Backend", ""]
        lines += table(["shape", "JIT", "BLAS apply plan", "speedup"], apply_rows)

    sparse_rows = []
    for m in all_matches(
        r"CSR\((\d+)x\1, (\d+)/row\) x Dense\(\1x(\d+)\)\s+VM\s+([\d.]+) µs\s+JIT\s+([\d.]+) µs\s+plan\(SparseDenseMatmul\)\s+([\d.]+) µs\s+\(plan ([\d.]+)× VM, ([\d.]+)× JIT\)",
        jit,
    ):
        n, per_row, cols, vm, jit_us, plan, vs_vm, vs_jit = m.groups()
        sparse_rows.append([f"CSR({n}, {per_row}/row) x Dense({cols})", f"{vm} us", f"{jit_us} us", f"{plan} us", f"{vs_vm}x VM / {vs_jit}x JIT"])
    if sparse_rows:
        lines += ["", "## Sparse-Dense", ""]
        lines += table(["shape", "VM", "JIT", "plan", "speedup"], sparse_rows)

    crossover_lines = []
    for m in all_matches(r"→ n=(\d+) csr crossover ≈ ([\d.]+)% density", crossover):
        crossover_lines.append(f"- CSR sequential crossover at n={m.group(1)}: about `{m.group(2)}%` density.")
    for m in all_matches(r"→ n=(\d+) csr_par crossover ≈ ([\d.]+)% density", crossover):
        crossover_lines.append(f"- CSR parallel crossover at n={m.group(1)}: about `{m.group(2)}%` density.")
    for m in all_matches(r"→ ([^\n]+) Blocked(8|16) crossover ≈ ([\d.]+)% density", crossover):
        crossover_lines.append(f"- {m.group(1)} Blocked{m.group(2)} attention crossover: about `{m.group(3)}%` density.")
    if crossover_lines:
        lines += ["", "## Crossover Points", ""]
        lines += crossover_lines

    graph_rows = []
    for m in all_matches(r"^(\d+),(\d+),(\d+),(\d+),([\d.]+)$", graph):
        step, nnz, seq, par, speed = m.groups()
        graph_rows.append([step, nnz, f"{seq} us", f"{par} us", f"{speed}x"])
    if graph_rows:
        lines += ["", "## Graph Powers", ""]
        lines += table(["step", "nnz", "seq", "par", "speedup"], graph_rows)

    mork_step = first(r"executing 7 steps took ([^\n]+)", mork)
    mork_100 = first(r"real_sec=([\d.]+)", mork_binary)
    if mork_step or mork_100:
        lines += ["", "## MORK Surface Timing", ""]
        if mork_step:
            lines.append(f"- Release MORK resource run reported `{mork_step.group(0)}`.")
        if mork_100:
            per_run_ms = float(mork_100.group(1)) * 1000.0 / float(mork_runs)
            lines.append(f"- {mork_runs} direct release-binary invocations took `{mork_100.group(1)} s`, about `{per_run_ms:.2f} ms/run` including process startup, parse, 7 MM2 steps, and discarded output.")

    total = first(r"real_sec=([\d.]+)", jit)
    if total:
        lines += ["", "## Raw Logs", "", f"- `jit_bench.log` completed in `{total.group(1)} s` wall time."]
    for name in ["perf_bench.log", "crossover_bench.log", "graph_bench.log"]:
        text = read(root / name)
        m = first(r"real_sec=([\d.]+)", text)
        if m:
            lines.append(f"- `{name}` completed in `{m.group(1)} s` wall time.")

    lines += [
        "",
        "## Machine Reports",
        "",
        "- `commands.tsv`: command ledger used for report generation.",
        "- `report.json`: JSON summary of command status, elapsed time, and environment metadata.",
        "- `junit.xml`: JUnit-style test report for CI systems.",
        "- `report_verification.log`: verifier output for the generated machine reports.",
    ]

    lines += [
        "",
        "## Interpretation",
        "",
        "MORK should use pathmap matching to stage symbolic work, then dispatch recognized tensor workloads into specialized native plans.  The interpreted einsum VM is a correctness/general fallback; BLAS, sparse row walking, and blocked sparse kernels are the performance paths.",
        "",
    ]

    (root / "summary.md").write_text("\n".join(lines))
    print(root / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
