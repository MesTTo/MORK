# Bench regression guard

Standing gate: no kernel change ships while `mork bench all` regresses, on any
shipping feature config. Run `./ai-bench-guard/run_guard.sh check` before
calling kernel work done; capture a fresh baseline (`... baseline`) immediately
after a change LANDS with an accepted, explained delta.

## What it covers

Six builds x the full bench set, each bench run separately so counters are
per-bench:

- `default` (no features) = upstream behavior, the surface Adam's process
  cares about (`mork bench default` + `mork test` reports).
- `canonical` = einsum,leapfrog,bulk_emit,factorized_aggregate (the R11
  canonical binary; adds the `aggregate` bench).
- `canonical-strat` = + stratified_quiescence (the GPT-2/pcgraph combo).
- `canonical-sni` = + semi_naive_ic (the saturation/TC combo; its
  process_calculus counters differ from default BY DESIGN, which is why
  baselines are per-config).
- `morkl-plan` = morkl_plan with its runtime dispatch knob enabled by default.
- `canonical-morkl` = the canonical feature set + morkl_plan, also with its
  runtime dispatch knob enabled by default.

Benches: taxi_lts, counter_machine, transitive, clique, finite_domain,
process_calculus, exponential, exponential_fringe, odd_even_sort, logic_query,
tile_puzzle_states, bfc (+ aggregate where compiled). `flybase` and
`logic_query_act` are excluded from `bench all` upstream and stay excluded.

## What "regress" means here

1. HARD FAIL: any change in a bench's exit code, or any diff in its normalized
   output (timing lines stripped, every counter and result-count line kept).
   These are the deterministic oracles: unifications, engine instructions,
   transition counts, clique/path/result counts.
2. REPORTED, WARN over +1%: instructions:u per bench. Load-immune
   (+-0.00008% under full contention, validated 2026-07-15), so it is
   meaningful even on a busy box. Wall and cycles are recorded but never
   judged under load; judge wall only on an idle box.

## Build rules (why the script looks the way it does)

- The script exports the full required RUSTFLAGS value:
  `-C target-cpu=native -Awarnings --cfg gxhash -C link-arg=-fuse-ld=mold`.
  On this machine, `~/.cargo/config.toml` defines target-specific rustflags.
  Cargo selects environment flags over target-specific flags and
  target-specific flags over the project's build flags; it does not merge
  those sources. A bare build therefore ignores the project's `--cfg gxhash`
  and fails gxhash's AES check. Manual cargo commands must use the same full
  RUSTFLAGS value.
- Each config builds into `ai-bench-guard/target-<name>`, so the repo's main
  `target/` and anything running from it are never clobbered.
- Everything runs under `nice -n 15`.

## Cross-checks for a fresh baseline

A correct default-config baseline reproduces the known-good deterministic
counters (measured 2026-07-17 at HEAD 31cf8b9): process_calculus unifications
201401 / engine instructions 427949817; transitive trans 19917429 / detected
8716; cliques 7824 / 2320 / 102. canonical-sni flips process_calculus
unifications to 1602 by design. If any of these differ at capture time,
distrust the baseline, not the numbers.

## Expected runtime

A full six-config baseline or check takes several hours on this box:
logic_query is a ~10-minute cell on EVERY config (600 s-class by nature, not a
pathology), and odd_even_sort is ~9 minutes on the leapfrog configs (that one
IS a pathology, see the 2026-07-17 measurements addendum). Neither is a hang;
the per-bench timeout (1800 s) only exists for real wedges.

## Layout

- `baselines/<label>/<config>/<bench>.{out,err,exit,perf}` + `meta.txt`;
  `baselines/current` symlinks the active label.
- `runs/<timestamp>-<label>/...` = check-mode captures, kept for the audit
  trail.
- The normalize() filter in run_guard.sh may be refined at any time without
  recapturing baselines (it is applied to both sides at compare time).
