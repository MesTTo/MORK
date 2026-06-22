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
    : "${WILLIAM_SEQUENCE_REPEATS:=4}"
    : "${WILLIAM_DECOYS:=25}"
    ;;
  full)
    : "${WILLIAM_SEQUENCE_REPEATS:=20}"
    : "${WILLIAM_DECOYS:=500}"
    ;;
  *)
    usage
    exit 2
    ;;
esac

if (( WILLIAM_SEQUENCE_REPEATS < 1 )); then
  printf 'WILLIAM_SEQUENCE_REPEATS must be positive, got %s\n' "$WILLIAM_SEQUENCE_REPEATS" >&2
  exit 2
fi
if (( WILLIAM_DECOYS < 0 )); then
  printf 'WILLIAM_DECOYS must be non-negative, got %s\n' "$WILLIAM_DECOYS" >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="$(create_sandbox_report_dir "$LOG_DIR" "mork-william-predictive-probe" "$stamp")"
commands_tsv="$out_dir/commands.tsv"
fixtures_tsv="$out_dir/fixtures.tsv"
prediction_tsv="$out_dir/predictions.tsv"
printf 'name\tstatus\telapsed_ms\tlog\n' > "$commands_tsv"
load1="$(awk '{ print $1 }' /proc/loadavg)"

write_reports() {
  local final_status="$1"
  write_and_verify_sandbox_reports "mork-william-predictive-probe-$MODE" "$out_dir" "$final_status"
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

cache_stat_field() {
  local line="$1"
  local key="$2"
  awk -v key="$key" '
    {
      for (i = 1; i <= NF; i++) {
        split($i, kv, "=")
        if (kv[1] == key) {
          print kv[2]
          exit
        }
      }
    }
  ' <<< "$line"
}

{
  printf 'stamp_utc=%s\n' "$stamp"
  printf 'mode=%s\n' "$MODE"
  printf 'root_dir=%s\n' "$ROOT_DIR"
  printf 'load1=%s\n' "$load1"
  printf 'load_max=%s\n' "$LOAD_MAX"
  printf 'allow_busy=%s\n' "$ALLOW_BUSY"
  printf 'william_sequence_repeats=%s\n' "$WILLIAM_SEQUENCE_REPEATS"
  printf 'william_decoys=%s\n' "$WILLIAM_DECOYS"
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
      printf 'Refusing to run clean William predictive probe: load1=%s exceeds LOAD_MAX=%s\n' "$load1" "$LOAD_MAX"
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

generate_start_ns="$(date +%s%N)"
set +e
{
  printf 'Generating William predictive workload in %s\n' "$out_dir"
  OUT_DIR="$out_dir" FIXTURES_TSV="$fixtures_tsv" PREDICTION_TSV="$prediction_tsv" \
    WILLIAM_SEQUENCE_REPEATS="$WILLIAM_SEQUENCE_REPEATS" WILLIAM_DECOYS="$WILLIAM_DECOYS" \
    "$PYTHON_BIN" - <<'PY'
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
fixtures_tsv = Path(os.environ["FIXTURES_TSV"])
prediction_tsv = Path(os.environ["PREDICTION_TSV"])
repeats = int(os.environ["WILLIAM_SEQUENCE_REPEATS"])
decoys = int(os.environ["WILLIAM_DECOYS"])

base_cycle = ["seed", "profile", "neighbors", "score"]
sequence: list[str] = []
for i in range(repeats):
    sequence.extend(base_cycle)
    if i % 3 == 2:
        sequence.append("audit")

expected = {
    "seed": 1,
    "profile": 1,
    "neighbors": 2,
    "score": 2,
    "audit": 1,
}

def write_base(out) -> None:
    out.write("(User u1)\n")
    out.write("(Profile u1 hot)\n")
    out.write("(Edge u1 u2)\n")
    out.write("(Edge u1 u3)\n")
    out.write("(Score u2 10)\n")
    out.write("(Score u3 20)\n")
    out.write("(Audit u1 flag)\n")
    for i in range(1, decoys + 1):
        out.write(f"(User d{i})\n")
        out.write(f"(Profile d{i} cold)\n")
        out.write(f"(Edge d{i} shadow{i})\n")
        out.write(f"(Score shadow{i} 0)\n")
    out.write("\n")

def write_exec(out, shape: str) -> None:
    if shape == "seed":
        out.write("(exec 0 (, (User $u) (Profile $u hot)) (, (WilliamResult seed $u)))\n")
    elif shape == "profile":
        out.write("(exec 0 (, (Profile $u hot)) (, (WilliamResult profile $u)))\n")
    elif shape == "neighbors":
        out.write("(exec 0 (, (User u1) (Edge u1 $v)) (, (WilliamResult neighbors $v)))\n")
    elif shape == "score":
        out.write("(exec 0 (, (Edge u1 $v) (Score $v $s)) (, (WilliamResult score $v $s)))\n")
    elif shape == "audit":
        out.write("(exec 0 (, (User $u) (Audit $u flag)) (, (WilliamResult audit $u)))\n")
    else:
        raise ValueError(shape)

transitions: defaultdict[str, Counter[str]] = defaultdict(Counter)
previous: str | None = None
fixture_rows = []
prediction_rows = []
for step, shape in enumerate(sequence, start=1):
    prediction = ""
    outcome = "no_prediction"
    if previous is not None and transitions[previous]:
        prediction, _ = sorted(
            transitions[previous].items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        outcome = "hit" if prediction == shape else "miss"

    fixture = out_dir / f"william_step_{step:03d}_{shape}.mm2"
    with fixture.open("w", encoding="utf-8") as out:
        write_base(out)
        write_exec(out, shape)

    fixture_rows.append(
        {
            "step": step,
            "shape": shape,
            "expected_results": expected[shape],
            "fixture": str(fixture),
        }
    )
    prediction_rows.append(
        {
            "step": step,
            "previous_shape": previous or "",
            "actual_shape": shape,
            "predicted_shape": prediction,
            "outcome": outcome,
        }
    )
    if previous is not None:
        transitions[previous][shape] += 1
    previous = shape

with fixtures_tsv.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=["step", "shape", "expected_results", "fixture"],
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(fixture_rows)

with prediction_tsv.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=["step", "previous_shape", "actual_shape", "predicted_shape", "outcome"],
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(prediction_rows)

print(f"fixtures={len(fixture_rows)}")
print(f"predictions={len(prediction_rows)}")
PY
} > "$out_dir/generate_workload.log" 2>&1
generate_status="$?"
set -e
generate_end_ns="$(date +%s%N)"
generate_elapsed_ms="$(((generate_end_ns - generate_start_ns) / 1000000))"
record_command generate_workload "$generate_status" "$generate_elapsed_ms" generate_workload.log
if [ "$generate_status" != "0" ]; then
  exit "$generate_status"
fi

cd "$ROOT_DIR"
run_logged_command build_release \
  env RUSTFLAGS="$RUSTFLAGS" cargo +nightly build --release -q -p mork --features "$FEATURES"

printf 'step\tshape\telapsed_ms\twilliam_results\texpected_results\tcache_entries\tcache_hits\tcache_misses\tcache_inserts\tcache_lookups\tcache_hit_rate\tshape_index_entries\tshape_index_clears\tshape_index_generation\tshape_index_estimated_bytes\tshape_index_max_estimated_bytes\tshape_index_key_bytes\tshape_index_summary_bytes\tshape_index_domain_values\tshape_index_avoided_scans\tplanner_plans\tplanner_factors\tprefix_lookups\tprefix_cache_hits\tshape_side_index_lookups\tshape_side_index_hits\tshape_side_index_inserts\tvariable_domain_refinements\tmin_variable_domain_sum\tmax_variable_domain\tshared_variable_domain_intersections\tshared_variable_domain_sum\tmax_shared_variable_domain\tprunable_shared_variable_domains\tshared_variable_domain_product_sum\tshared_variable_domain_pruning_sum\tmax_shared_variable_domain_product\tvariable_order_plans\tvariable_order_variables\tvariable_order_shared_variables\tvariable_order_first_domain_sum\tvariable_order_assignment_sum\tmax_variable_order_assignment\tmax_variable_order_domain\tvariable_order_pruning_sum\tcard_unknown\tcard_zero\tcard_one\tcard_le8\tcard_le64\tcard_le512\tcard_le4096\tcard_gt4096\testimated_sum\tmax_estimated\tmax_factors_per_plan\tshape_ground_roots\tshape_schematic_roots\tall_ground_shape_factors\tschematic_shape_factors\tground_factors\tanchored_variable_factors\tunanchored_variable_factors\trepeated_variable_factors\tpure_variable_factors\tnew_var_items\tvar_ref_items\tvariable_items_sum\tmax_variables_per_factor\tmax_prefix_len\tstorage_line\texecution_line\n' > "$out_dir/runs.tsv"
tail -n +2 "$fixtures_tsv" | while IFS=$'\t' read -r step shape expected_results fixture; do
  fixture="${fixture%$'\r'}"
  expected_results="${expected_results%$'\r'}"
  shape="${shape%$'\r'}"
  step="${step%$'\r'}"
  log="$out_dir/run_${step}_${shape}.log"
  output="$out_dir/output_${step}_${shape}.metta"
  start_ns="$(date +%s%N)"
  set +e
  "target/release/mork" run "$fixture" --steps "$STEPS" --instrumentation 0 \
    --query-plan-cache-stats --query-planner-stats --query-execution-stats "$output" \
    > "$log" 2>&1
  command_status="$?"
  set -e
  end_ns="$(date +%s%N)"
  elapsed_ms="$(((end_ns - start_ns) / 1000000))"
  william_results="$(grep -c '^(WilliamResult ' "$output" || true)"
  cache_stats_line="$(grep -m1 '^query plan cache:' "$log" || true)"
  cache_entries="$(cache_stat_field "$cache_stats_line" entries)"
  cache_hits="$(cache_stat_field "$cache_stats_line" hits)"
  cache_misses="$(cache_stat_field "$cache_stats_line" misses)"
  cache_inserts="$(cache_stat_field "$cache_stats_line" inserts)"
  cache_lookups="$(cache_stat_field "$cache_stats_line" lookups)"
  cache_hit_rate="$(cache_stat_field "$cache_stats_line" hit_rate)"
  shape_index_stats_line="$(grep -m1 '^query shape side index:' "$log" || true)"
  shape_index_entries="$(cache_stat_field "$shape_index_stats_line" entries)"
  shape_index_clears="$(cache_stat_field "$shape_index_stats_line" clears)"
  shape_index_generation="$(cache_stat_field "$shape_index_stats_line" generation)"
  shape_index_estimated_bytes="$(cache_stat_field "$shape_index_stats_line" estimated_bytes)"
  shape_index_max_estimated_bytes="$(cache_stat_field "$shape_index_stats_line" max_estimated_bytes)"
  shape_index_key_bytes="$(cache_stat_field "$shape_index_stats_line" key_bytes)"
  shape_index_summary_bytes="$(cache_stat_field "$shape_index_stats_line" summary_bytes)"
  shape_index_domain_values="$(cache_stat_field "$shape_index_stats_line" domain_values)"
  shape_index_avoided_scans="$(cache_stat_field "$shape_index_stats_line" avoided_shape_scans)"
  planner_stats_line="$(grep -m1 '^query planner cardinality:' "$log" || true)"
  planner_plans="$(cache_stat_field "$planner_stats_line" plans)"
  planner_factors="$(cache_stat_field "$planner_stats_line" factors)"
  prefix_lookups="$(cache_stat_field "$planner_stats_line" prefix_lookups)"
  prefix_cache_hits="$(cache_stat_field "$planner_stats_line" prefix_cache_hits)"
  shape_side_index_lookups="$(cache_stat_field "$planner_stats_line" shape_side_index_lookups)"
  shape_side_index_hits="$(cache_stat_field "$planner_stats_line" shape_side_index_hits)"
  shape_side_index_inserts="$(cache_stat_field "$planner_stats_line" shape_side_index_inserts)"
  variable_domain_refinements="$(cache_stat_field "$planner_stats_line" variable_domain_refinements)"
  min_variable_domain_sum="$(cache_stat_field "$planner_stats_line" min_variable_domain_sum)"
  max_variable_domain="$(cache_stat_field "$planner_stats_line" max_variable_domain)"
  shared_variable_domain_intersections="$(cache_stat_field "$planner_stats_line" shared_variable_domain_intersections)"
  shared_variable_domain_sum="$(cache_stat_field "$planner_stats_line" shared_variable_domain_sum)"
  max_shared_variable_domain="$(cache_stat_field "$planner_stats_line" max_shared_variable_domain)"
  prunable_shared_variable_domains="$(cache_stat_field "$planner_stats_line" prunable_shared_variable_domains)"
  shared_variable_domain_product_sum="$(cache_stat_field "$planner_stats_line" shared_variable_domain_product_sum)"
  shared_variable_domain_pruning_sum="$(cache_stat_field "$planner_stats_line" shared_variable_domain_pruning_sum)"
  max_shared_variable_domain_product="$(cache_stat_field "$planner_stats_line" max_shared_variable_domain_product)"
  variable_order_plans="$(cache_stat_field "$planner_stats_line" variable_order_plans)"
  variable_order_variables="$(cache_stat_field "$planner_stats_line" variable_order_variables)"
  variable_order_shared_variables="$(cache_stat_field "$planner_stats_line" variable_order_shared_variables)"
  variable_order_first_domain_sum="$(cache_stat_field "$planner_stats_line" variable_order_first_domain_sum)"
  variable_order_assignment_sum="$(cache_stat_field "$planner_stats_line" variable_order_assignment_sum)"
  max_variable_order_assignment="$(cache_stat_field "$planner_stats_line" max_variable_order_assignment)"
  max_variable_order_domain="$(cache_stat_field "$planner_stats_line" max_variable_order_domain)"
  variable_order_pruning_sum="$(cache_stat_field "$planner_stats_line" variable_order_pruning_sum)"
  card_unknown="$(cache_stat_field "$planner_stats_line" unknown)"
  card_zero="$(cache_stat_field "$planner_stats_line" zero)"
  card_one="$(cache_stat_field "$planner_stats_line" one)"
  card_le8="$(cache_stat_field "$planner_stats_line" le8)"
  card_le64="$(cache_stat_field "$planner_stats_line" le64)"
  card_le512="$(cache_stat_field "$planner_stats_line" le512)"
  card_le4096="$(cache_stat_field "$planner_stats_line" le4096)"
  card_gt4096="$(cache_stat_field "$planner_stats_line" gt4096)"
  estimated_sum="$(cache_stat_field "$planner_stats_line" estimated_sum)"
  max_estimated="$(cache_stat_field "$planner_stats_line" max_estimated)"
  max_factors_per_plan="$(cache_stat_field "$planner_stats_line" max_factors_per_plan)"
  shape_ground_roots="$(cache_stat_field "$planner_stats_line" shape_ground_roots)"
  shape_schematic_roots="$(cache_stat_field "$planner_stats_line" shape_schematic_roots)"
  all_ground_shape_factors="$(cache_stat_field "$planner_stats_line" all_ground_shape_factors)"
  schematic_shape_factors="$(cache_stat_field "$planner_stats_line" schematic_shape_factors)"
  ground_factors="$(cache_stat_field "$planner_stats_line" ground)"
  anchored_variable_factors="$(cache_stat_field "$planner_stats_line" anchored_variable)"
  unanchored_variable_factors="$(cache_stat_field "$planner_stats_line" unanchored_variable)"
  repeated_variable_factors="$(cache_stat_field "$planner_stats_line" repeated_variable)"
  pure_variable_factors="$(cache_stat_field "$planner_stats_line" pure_variable)"
  new_var_items="$(cache_stat_field "$planner_stats_line" new_var_items)"
  var_ref_items="$(cache_stat_field "$planner_stats_line" var_ref_items)"
  variable_items_sum="$(cache_stat_field "$planner_stats_line" variable_items_sum)"
  max_variables_per_factor="$(cache_stat_field "$planner_stats_line" max_variables_per_factor)"
  max_prefix_len="$(cache_stat_field "$planner_stats_line" max_prefix_len)"
  : "${cache_entries:=0}"
  : "${cache_hits:=0}"
  : "${cache_misses:=0}"
  : "${cache_inserts:=0}"
  : "${cache_lookups:=0}"
  : "${cache_hit_rate:=0.00%}"
  : "${shape_index_entries:=0}"
  : "${shape_index_clears:=0}"
  : "${shape_index_generation:=0}"
  : "${shape_index_estimated_bytes:=0}"
  : "${shape_index_max_estimated_bytes:=0}"
  : "${shape_index_key_bytes:=0}"
  : "${shape_index_summary_bytes:=0}"
  : "${shape_index_domain_values:=0}"
  : "${shape_index_avoided_scans:=0}"
  : "${planner_plans:=0}"
  : "${planner_factors:=0}"
  : "${prefix_lookups:=0}"
  : "${prefix_cache_hits:=0}"
  : "${shape_side_index_lookups:=0}"
  : "${shape_side_index_hits:=0}"
  : "${shape_side_index_inserts:=0}"
  : "${variable_domain_refinements:=0}"
  : "${min_variable_domain_sum:=0}"
  : "${max_variable_domain:=0}"
  : "${shared_variable_domain_intersections:=0}"
  : "${shared_variable_domain_sum:=0}"
  : "${max_shared_variable_domain:=0}"
  : "${prunable_shared_variable_domains:=0}"
  : "${shared_variable_domain_product_sum:=0}"
  : "${shared_variable_domain_pruning_sum:=0}"
  : "${max_shared_variable_domain_product:=0}"
  : "${variable_order_plans:=0}"
  : "${variable_order_variables:=0}"
  : "${variable_order_shared_variables:=0}"
  : "${variable_order_first_domain_sum:=0}"
  : "${variable_order_assignment_sum:=0}"
  : "${max_variable_order_assignment:=0}"
  : "${max_variable_order_domain:=0}"
  : "${variable_order_pruning_sum:=0}"
  : "${card_unknown:=0}"
  : "${card_zero:=0}"
  : "${card_one:=0}"
  : "${card_le8:=0}"
  : "${card_le64:=0}"
  : "${card_le512:=0}"
  : "${card_le4096:=0}"
  : "${card_gt4096:=0}"
  : "${estimated_sum:=0}"
  : "${max_estimated:=0}"
  : "${max_factors_per_plan:=0}"
  : "${shape_ground_roots:=0}"
  : "${shape_schematic_roots:=0}"
  : "${all_ground_shape_factors:=0}"
  : "${schematic_shape_factors:=0}"
  : "${ground_factors:=0}"
  : "${anchored_variable_factors:=0}"
  : "${unanchored_variable_factors:=0}"
  : "${repeated_variable_factors:=0}"
  : "${pure_variable_factors:=0}"
  : "${new_var_items:=0}"
  : "${var_ref_items:=0}"
  : "${variable_items_sum:=0}"
  : "${max_variables_per_factor:=0}"
  : "${max_prefix_len:=0}"
  storage_stats_line="$(grep -m1 '^query execution storage:' "$log" || true)"
  execution_line="$(grep -m1 '^executing ' "$log" || true)"
  run_row=(
    "$step" "$shape" "$elapsed_ms" "$william_results" "$expected_results"
    "$cache_entries" "$cache_hits" "$cache_misses" "$cache_inserts" "$cache_lookups" "$cache_hit_rate"
    "$shape_index_entries" "$shape_index_clears" "$shape_index_generation"
    "$shape_index_estimated_bytes" "$shape_index_max_estimated_bytes" "$shape_index_key_bytes"
    "$shape_index_summary_bytes" "$shape_index_domain_values" "$shape_index_avoided_scans"
    "$planner_plans" "$planner_factors" "$prefix_lookups" "$prefix_cache_hits"
    "$shape_side_index_lookups" "$shape_side_index_hits" "$shape_side_index_inserts"
    "$variable_domain_refinements" "$min_variable_domain_sum" "$max_variable_domain"
    "$shared_variable_domain_intersections" "$shared_variable_domain_sum" "$max_shared_variable_domain"
    "$prunable_shared_variable_domains" "$shared_variable_domain_product_sum" "$shared_variable_domain_pruning_sum" "$max_shared_variable_domain_product"
    "$variable_order_plans" "$variable_order_variables" "$variable_order_shared_variables"
    "$variable_order_first_domain_sum" "$variable_order_assignment_sum" "$max_variable_order_assignment"
    "$max_variable_order_domain" "$variable_order_pruning_sum"
    "$card_unknown" "$card_zero" "$card_one" "$card_le8" "$card_le64" "$card_le512" "$card_le4096" "$card_gt4096"
    "$estimated_sum" "$max_estimated" "$max_factors_per_plan"
    "$shape_ground_roots" "$shape_schematic_roots" "$all_ground_shape_factors" "$schematic_shape_factors"
    "$ground_factors" "$anchored_variable_factors" "$unanchored_variable_factors" "$repeated_variable_factors" "$pure_variable_factors"
    "$new_var_items" "$var_ref_items" "$variable_items_sum" "$max_variables_per_factor" "$max_prefix_len"
    "$storage_stats_line" "$execution_line"
  )
  {
    printf '%s' "${run_row[0]}"
    for value in "${run_row[@]:1}"; do
      printf '\t%s' "$value"
    done
    printf '\n'
  } >> "$out_dir/runs.tsv"
  gate_status="$command_status"
  if [ "$command_status" = "0" ] && [ "$william_results" != "$expected_results" ]; then
    gate_status=1
  fi
  record_command "william_step_${step}_${shape}" "$gate_status" "$elapsed_ms" "run_${step}_${shape}.log"
  if [ "$command_status" != "0" ]; then
    printf 'mork run %s (%s) exited with status %s; see %s\n' "$step" "$shape" "$command_status" "$log" >&2
    exit "$command_status"
  fi
  if [ "$william_results" != "$expected_results" ]; then
    printf 'expected %s WilliamResult atoms, got %s in %s\n' "$expected_results" "$william_results" "$output" >&2
    exit 1
  fi
done

RUNS_TSV="$out_dir/runs.tsv" PREDICTION_TSV="$prediction_tsv" MANIFEST="$out_dir/manifest.txt" SUMMARY="$out_dir/summary.md" \
  "$PYTHON_BIN" - <<'PY'
import csv
import os
import re
import statistics
from pathlib import Path

runs_tsv = Path(os.environ["RUNS_TSV"])
prediction_tsv = Path(os.environ["PREDICTION_TSV"])
manifest = Path(os.environ["MANIFEST"])
summary = Path(os.environ["SUMMARY"])

manifest_kv: dict[str, str] = {}
manifest_text = manifest.read_text(encoding="utf-8", errors="replace")
for line in manifest_text.splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        manifest_kv[key] = value
load_line = next((line.strip() for line in manifest_text.splitlines() if "load average:" in line), "unknown")

runs = []

def stat_field(line, key):
    match = re.search(rf"{re.escape(key)}=([^ ]+)", line or "")
    return match.group(1) if match else "0"

storage_keys = [
    "renorm_plans",
    "renorm_factors",
    "renorm_len_sum",
    "renorm_capacity_sum",
    "max_renorm_len",
    "max_renorm_capacity",
    "renorm_len_le8",
    "renorm_len_le32",
    "renorm_len_le128",
    "renorm_len_le512",
    "renorm_len_le2048",
    "renorm_len_gt2048",
    "renorm_capacity_le8",
    "renorm_capacity_le32",
    "renorm_capacity_le128",
    "renorm_capacity_le512",
    "renorm_capacity_le2048",
    "renorm_capacity_gt2048",
    "raw_searches",
    "raw_stack_entries_sum",
    "max_raw_stack_entries",
    "candidate_pair_vectors",
    "candidate_pair_entries_sum",
    "candidate_pair_capacity_sum",
    "max_candidate_pair_entries",
    "max_candidate_pair_capacity",
    "candidate_pair_capacity_le8",
    "candidate_pair_capacity_le32",
    "candidate_pair_capacity_le128",
    "candidate_pair_capacity_le512",
    "candidate_pair_capacity_le2048",
    "candidate_pair_capacity_gt2048",
    "general_unifications",
    "successful_unifications",
    "unification_failures",
]
mode_keys = [
    "shape_ground_roots",
    "shape_schematic_roots",
    "all_ground_shape_factors",
    "schematic_shape_factors",
    "ground_factors",
    "anchored_variable_factors",
    "unanchored_variable_factors",
    "repeated_variable_factors",
    "pure_variable_factors",
    "new_var_items",
    "var_ref_items",
    "variable_items_sum",
    "max_variables_per_factor",
    "max_prefix_len",
]

with runs_tsv.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        row["elapsed_ms"] = int(row["elapsed_ms"])
        row["william_results"] = int(row["william_results"])
        row["expected_results"] = int(row["expected_results"])
        row["cache_entries"] = int(row.get("cache_entries") or 0)
        row["cache_hits"] = int(row.get("cache_hits") or 0)
        row["cache_misses"] = int(row.get("cache_misses") or 0)
        row["cache_inserts"] = int(row.get("cache_inserts") or 0)
        row["cache_lookups"] = int(row.get("cache_lookups") or 0)
        row["shape_index_entries"] = int(row.get("shape_index_entries") or 0)
        row["shape_index_clears"] = int(row.get("shape_index_clears") or 0)
        row["shape_index_generation"] = int(row.get("shape_index_generation") or 0)
        row["shape_index_estimated_bytes"] = int(row.get("shape_index_estimated_bytes") or 0)
        row["shape_index_max_estimated_bytes"] = int(row.get("shape_index_max_estimated_bytes") or 0)
        row["shape_index_key_bytes"] = int(row.get("shape_index_key_bytes") or 0)
        row["shape_index_summary_bytes"] = int(row.get("shape_index_summary_bytes") or 0)
        row["shape_index_domain_values"] = int(row.get("shape_index_domain_values") or 0)
        row["shape_index_avoided_scans"] = int(row.get("shape_index_avoided_scans") or 0)
        row["planner_plans"] = int(row.get("planner_plans") or 0)
        row["planner_factors"] = int(row.get("planner_factors") or 0)
        row["prefix_lookups"] = int(row.get("prefix_lookups") or 0)
        row["prefix_cache_hits"] = int(row.get("prefix_cache_hits") or 0)
        row["shape_side_index_lookups"] = int(row.get("shape_side_index_lookups") or 0)
        row["shape_side_index_hits"] = int(row.get("shape_side_index_hits") or 0)
        row["shape_side_index_inserts"] = int(row.get("shape_side_index_inserts") or 0)
        row["variable_domain_refinements"] = int(row.get("variable_domain_refinements") or 0)
        row["min_variable_domain_sum"] = int(row.get("min_variable_domain_sum") or 0)
        row["max_variable_domain"] = int(row.get("max_variable_domain") or 0)
        row["shared_variable_domain_intersections"] = int(row.get("shared_variable_domain_intersections") or 0)
        row["shared_variable_domain_sum"] = int(row.get("shared_variable_domain_sum") or 0)
        row["max_shared_variable_domain"] = int(row.get("max_shared_variable_domain") or 0)
        row["prunable_shared_variable_domains"] = int(row.get("prunable_shared_variable_domains") or 0)
        row["shared_variable_domain_product_sum"] = int(row.get("shared_variable_domain_product_sum") or 0)
        row["shared_variable_domain_pruning_sum"] = int(row.get("shared_variable_domain_pruning_sum") or 0)
        row["max_shared_variable_domain_product"] = int(row.get("max_shared_variable_domain_product") or 0)
        row["variable_order_plans"] = int(row.get("variable_order_plans") or 0)
        row["variable_order_variables"] = int(row.get("variable_order_variables") or 0)
        row["variable_order_shared_variables"] = int(row.get("variable_order_shared_variables") or 0)
        row["variable_order_first_domain_sum"] = int(row.get("variable_order_first_domain_sum") or 0)
        row["variable_order_assignment_sum"] = int(row.get("variable_order_assignment_sum") or 0)
        row["max_variable_order_assignment"] = int(row.get("max_variable_order_assignment") or 0)
        row["max_variable_order_domain"] = int(row.get("max_variable_order_domain") or 0)
        row["variable_order_pruning_sum"] = int(row.get("variable_order_pruning_sum") or 0)
        row["card_unknown"] = int(row.get("card_unknown") or 0)
        row["card_zero"] = int(row.get("card_zero") or 0)
        row["card_one"] = int(row.get("card_one") or 0)
        row["card_le8"] = int(row.get("card_le8") or 0)
        row["card_le64"] = int(row.get("card_le64") or 0)
        row["card_le512"] = int(row.get("card_le512") or 0)
        row["card_le4096"] = int(row.get("card_le4096") or 0)
        row["card_gt4096"] = int(row.get("card_gt4096") or 0)
        row["estimated_sum"] = int(row.get("estimated_sum") or 0)
        row["max_estimated"] = int(row.get("max_estimated") or 0)
        row["max_factors_per_plan"] = int(row.get("max_factors_per_plan") or 0)
        for key in mode_keys:
            row[key] = int(row.get(key) or 0)
        storage_line = row.get("storage_line", "")
        for key in storage_keys:
            row[key] = int(stat_field(storage_line, key))
        runs.append(row)

predictions = []
with prediction_tsv.open(newline="", encoding="utf-8") as fh:
    predictions = list(csv.DictReader(fh, delimiter="\t"))

outcomes = {name: sum(1 for row in predictions if row["outcome"] == name) for name in ("hit", "miss", "no_prediction")}
predicted = outcomes["hit"] + outcomes["miss"]
hit_rate = (outcomes["hit"] / predicted) if predicted else 0.0
elapsed = [row["elapsed_ms"] for row in runs]
total_results = sum(row["william_results"] for row in runs)
cache_hits = sum(row["cache_hits"] for row in runs)
cache_misses = sum(row["cache_misses"] for row in runs)
cache_lookups = sum(row["cache_lookups"] for row in runs)
cache_hit_rate = (cache_hits / cache_lookups) if cache_lookups else 0.0
max_shape_index_entries = max((row["shape_index_entries"] for row in runs), default=0)
shape_index_clears = sum(row["shape_index_clears"] for row in runs)
max_shape_index_generation = max((row["shape_index_generation"] for row in runs), default=0)
max_shape_index_estimated_bytes = max((row["shape_index_estimated_bytes"] for row in runs), default=0)
max_shape_index_high_water_bytes = max((row["shape_index_max_estimated_bytes"] for row in runs), default=0)
max_shape_index_key_bytes = max((row["shape_index_key_bytes"] for row in runs), default=0)
max_shape_index_summary_bytes = max((row["shape_index_summary_bytes"] for row in runs), default=0)
max_shape_index_domain_values = max((row["shape_index_domain_values"] for row in runs), default=0)
shape_index_avoided_scans = sum(row["shape_index_avoided_scans"] for row in runs)
planner_plans = sum(row["planner_plans"] for row in runs)
planner_factors = sum(row["planner_factors"] for row in runs)
prefix_lookups = sum(row["prefix_lookups"] for row in runs)
prefix_cache_hits = sum(row["prefix_cache_hits"] for row in runs)
shape_side_index_lookups = sum(row["shape_side_index_lookups"] for row in runs)
shape_side_index_hits = sum(row["shape_side_index_hits"] for row in runs)
shape_side_index_inserts = sum(row["shape_side_index_inserts"] for row in runs)
variable_domain_refinements = sum(row["variable_domain_refinements"] for row in runs)
min_variable_domain_sum = sum(row["min_variable_domain_sum"] for row in runs)
max_variable_domain = max((row["max_variable_domain"] for row in runs), default=0)
shared_variable_domain_intersections = sum(row["shared_variable_domain_intersections"] for row in runs)
shared_variable_domain_sum = sum(row["shared_variable_domain_sum"] for row in runs)
max_shared_variable_domain = max((row["max_shared_variable_domain"] for row in runs), default=0)
prunable_shared_variable_domains = sum(row["prunable_shared_variable_domains"] for row in runs)
shared_variable_domain_product_sum = sum(row["shared_variable_domain_product_sum"] for row in runs)
shared_variable_domain_pruning_sum = sum(row["shared_variable_domain_pruning_sum"] for row in runs)
max_shared_variable_domain_product = max((row["max_shared_variable_domain_product"] for row in runs), default=0)
variable_order_plans = sum(row["variable_order_plans"] for row in runs)
variable_order_variables = sum(row["variable_order_variables"] for row in runs)
variable_order_shared_variables = sum(row["variable_order_shared_variables"] for row in runs)
variable_order_first_domain_sum = sum(row["variable_order_first_domain_sum"] for row in runs)
variable_order_assignment_sum = sum(row["variable_order_assignment_sum"] for row in runs)
max_variable_order_assignment = max((row["max_variable_order_assignment"] for row in runs), default=0)
max_variable_order_domain = max((row["max_variable_order_domain"] for row in runs), default=0)
variable_order_pruning_sum = sum(row["variable_order_pruning_sum"] for row in runs)
card_buckets = {
    "unknown": sum(row["card_unknown"] for row in runs),
    "zero": sum(row["card_zero"] for row in runs),
    "one": sum(row["card_one"] for row in runs),
    "2-8": sum(row["card_le8"] for row in runs),
    "9-64": sum(row["card_le64"] for row in runs),
    "65-512": sum(row["card_le512"] for row in runs),
    "513-4096": sum(row["card_le4096"] for row in runs),
    ">4096": sum(row["card_gt4096"] for row in runs),
}
estimated_sum = sum(row["estimated_sum"] for row in runs)
max_estimated = max((row["max_estimated"] for row in runs), default=0)
max_factors_per_plan = max((row["max_factors_per_plan"] for row in runs), default=0)
known_factors = planner_factors - card_buckets["unknown"]
avg_estimated = (estimated_sum / known_factors) if known_factors else 0.0
shape_ground_roots = sum(row["shape_ground_roots"] for row in runs)
shape_schematic_roots = sum(row["shape_schematic_roots"] for row in runs)
all_ground_shape_factors = sum(row["all_ground_shape_factors"] for row in runs)
schematic_shape_factors = sum(row["schematic_shape_factors"] for row in runs)
ground_factors = sum(row["ground_factors"] for row in runs)
anchored_variable_factors = sum(row["anchored_variable_factors"] for row in runs)
unanchored_variable_factors = sum(row["unanchored_variable_factors"] for row in runs)
repeated_variable_factors = sum(row["repeated_variable_factors"] for row in runs)
pure_variable_factors = sum(row["pure_variable_factors"] for row in runs)
new_var_items = sum(row["new_var_items"] for row in runs)
var_ref_items = sum(row["var_ref_items"] for row in runs)
variable_items_sum = sum(row["variable_items_sum"] for row in runs)
max_variables_per_factor = max((row["max_variables_per_factor"] for row in runs), default=0)
max_prefix_len = max((row["max_prefix_len"] for row in runs), default=0)
renorm_plans = sum(row["renorm_plans"] for row in runs)
renorm_factors = sum(row["renorm_factors"] for row in runs)
renorm_len_sum = sum(row["renorm_len_sum"] for row in runs)
renorm_capacity_sum = sum(row["renorm_capacity_sum"] for row in runs)
max_renorm_len = max((row["max_renorm_len"] for row in runs), default=0)
max_renorm_capacity = max((row["max_renorm_capacity"] for row in runs), default=0)
renorm_len_buckets = {
    "<=8": sum(row["renorm_len_le8"] for row in runs),
    "9-32": sum(row["renorm_len_le32"] for row in runs),
    "33-128": sum(row["renorm_len_le128"] for row in runs),
    "129-512": sum(row["renorm_len_le512"] for row in runs),
    "513-2048": sum(row["renorm_len_le2048"] for row in runs),
    ">2048": sum(row["renorm_len_gt2048"] for row in runs),
}
renorm_capacity_buckets = {
    "<=8": sum(row["renorm_capacity_le8"] for row in runs),
    "9-32": sum(row["renorm_capacity_le32"] for row in runs),
    "33-128": sum(row["renorm_capacity_le128"] for row in runs),
    "129-512": sum(row["renorm_capacity_le512"] for row in runs),
    "513-2048": sum(row["renorm_capacity_le2048"] for row in runs),
    ">2048": sum(row["renorm_capacity_gt2048"] for row in runs),
}
raw_searches = sum(row["raw_searches"] for row in runs)
raw_stack_entries_sum = sum(row["raw_stack_entries_sum"] for row in runs)
max_raw_stack_entries = max((row["max_raw_stack_entries"] for row in runs), default=0)
candidate_pair_vectors = sum(row["candidate_pair_vectors"] for row in runs)
candidate_pair_entries_sum = sum(row["candidate_pair_entries_sum"] for row in runs)
candidate_pair_capacity_sum = sum(row["candidate_pair_capacity_sum"] for row in runs)
max_candidate_pair_entries = max((row["max_candidate_pair_entries"] for row in runs), default=0)
max_candidate_pair_capacity = max((row["max_candidate_pair_capacity"] for row in runs), default=0)
general_unifications = sum(row["general_unifications"] for row in runs)
successful_unifications = sum(row["successful_unifications"] for row in runs)
unification_failures = sum(row["unification_failures"] for row in runs)
candidate_pair_capacity_buckets = {
    "<=8": sum(row["candidate_pair_capacity_le8"] for row in runs),
    "9-32": sum(row["candidate_pair_capacity_le32"] for row in runs),
    "33-128": sum(row["candidate_pair_capacity_le128"] for row in runs),
    "129-512": sum(row["candidate_pair_capacity_le512"] for row in runs),
    "513-2048": sum(row["candidate_pair_capacity_le2048"] for row in runs),
    ">2048": sum(row["candidate_pair_capacity_gt2048"] for row in runs),
}
avg_renorm_len = (renorm_len_sum / renorm_factors) if renorm_factors else 0.0
avg_renorm_capacity = (renorm_capacity_sum / renorm_factors) if renorm_factors else 0.0
avg_raw_stack_entries = (raw_stack_entries_sum / raw_searches) if raw_searches else 0.0
avg_candidate_pair_entries = (
    candidate_pair_entries_sum / candidate_pair_vectors
    if candidate_pair_vectors
    else 0.0
)
avg_candidate_pair_capacity = (
    candidate_pair_capacity_sum / candidate_pair_vectors
    if candidate_pair_vectors
    else 0.0
)
unification_failure_rate = (
    unification_failures / general_unifications if general_unifications else 0.0
)
unifications_per_success = (
    general_unifications / successful_unifications if successful_unifications else 0.0
)

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
    "# MORK William Predictive Probe Summary",
    "",
    f"Timestamp: `{manifest_kv.get('stamp_utc', 'unknown')}`",
    f"Mode: `{manifest_kv.get('mode', 'unknown')}`",
    f"Load gate: `{load_gate_note}`",
    "",
    "This is an advisory probe for the William idea: learn normalized query-shape",
    "transitions and estimate whether pre-warming the next plan/prefix sidecar would",
    "be useful. It does not cache answers and does not change MORK execution.",
    "",
    "## Prediction",
    "",
    f"- Workload steps: `{len(predictions)}`.",
    f"- Predicted steps: `{predicted}`.",
    f"- Hits: `{outcomes['hit']}`.",
    f"- Misses / wasted pre-warms: `{outcomes['miss']}`.",
    f"- No prior prediction: `{outcomes['no_prediction']}`.",
    f"- Hit rate among predicted steps: `{hit_rate:.2%}`.",
    "",
    "## Query Plan Cache",
    "",
    "Counters are process-local and are printed by `mork run --query-plan-cache-stats`.",
    "William steps currently execute as separate MORK processes, so these counters measure",
    "per-step planner reuse rather than cross-step predictive caching.",
    f"- Cache hits: `{cache_hits}`.",
    f"- Cache misses: `{cache_misses}`.",
    f"- Cache lookups: `{cache_lookups}`.",
    f"- Cache hit rate: `{cache_hit_rate:.2%}`.",
    "",
    "Reusable shape side-index budget telemetry is approximate retained data, not allocator accounting.",
    f"- Max side-index entries: `{max_shape_index_entries}`.",
    f"- Side-index clears / max generation: `{shape_index_clears}` / `{max_shape_index_generation}`.",
    f"- Max estimated retained bytes / high-water bytes: `{max_shape_index_estimated_bytes}` / `{max_shape_index_high_water_bytes}`.",
    f"- Max key bytes / summary bytes: `{max_shape_index_key_bytes}` / `{max_shape_index_summary_bytes}`.",
    f"- Max retained projected domain values: `{max_shape_index_domain_values}`.",
    f"- Side-index avoided shape scans: `{shape_index_avoided_scans}`.",
    "",
    "## Planner Cardinality",
    "",
    "Counters are process-local and count uncached query-factor ranking work.",
    f"- Plans ranked: `{planner_plans}`.",
    f"- Factors ranked: `{planner_factors}`.",
    f"- Prefix cardinality lookups: `{prefix_lookups}`.",
    f"- Planner-local prefix cache hits: `{prefix_cache_hits}`.",
    f"- Reusable shape side-index lookups: `{shape_side_index_lookups}`.",
    f"- Reusable shape side-index hits: `{shape_side_index_hits}`.",
    f"- Reusable shape side-index inserts: `{shape_side_index_inserts}`.",
    f"- Projected variable-domain refinements: `{variable_domain_refinements}`.",
    f"- Sum of most selective projected domains: `{min_variable_domain_sum}`.",
    f"- Max projected variable domain: `{max_variable_domain}`.",
    f"- Shared-variable domain intersections: `{shared_variable_domain_intersections}`.",
    f"- Sum of exact shared-variable intersections: `{shared_variable_domain_sum}`.",
    f"- Max shared-variable intersection: `{max_shared_variable_domain}`.",
    f"- Prunable shared-variable domains: `{prunable_shared_variable_domains}`.",
    f"- Sum of shared-variable domain product upper bounds: `{shared_variable_domain_product_sum}`.",
    f"- Sum of shared-variable pruning upper bounds: `{shared_variable_domain_pruning_sum}`.",
    f"- Max shared-variable domain product upper bound: `{max_shared_variable_domain_product}`.",
    f"- Variable-order plans: `{variable_order_plans}`.",
    f"- Variable-order variables: `{variable_order_variables}`.",
    f"- Variable-order shared variables: `{variable_order_shared_variables}`.",
    f"- Variable-order first-domain sum: `{variable_order_first_domain_sum}`.",
    f"- Variable-order assignment upper-bound sum: `{variable_order_assignment_sum}`.",
    f"- Max variable-order assignment upper bound: `{max_variable_order_assignment}`.",
    f"- Max variable-order domain: `{max_variable_order_domain}`.",
    f"- Variable-order pruning upper-bound sum: `{variable_order_pruning_sum}`.",
    f"- Known-cardinality average estimate: `{avg_estimated:.2f}`.",
    f"- Max estimated cardinality: `{max_estimated}`.",
    f"- Max factors per plan: `{max_factors_per_plan}`.",
    f"- Shape-scan matched ground roots: `{shape_ground_roots}`.",
    f"- Shape-scan matched schematic roots: `{shape_schematic_roots}`.",
    f"- All-ground shape-refined factors: `{all_ground_shape_factors}`.",
    f"- Schematic shape-refined factors: `{schematic_shape_factors}`.",
    "",
    "| bucket | factors |",
    "| --- | ---: |",
]
for bucket, count in card_buckets.items():
    lines.append(f"| `{bucket}` | {count} |")
lines += [
    "",
    "## Planner Mode Signatures",
    "",
    "These counters show whether William's transition predictions are attached to",
    "query modes that exact symbolic indexes can exploit. Hamming/LSH-style",
    "similarity filters remain advisory only; exact MORK unification still decides results.",
    f"- Ground factors: `{ground_factors}`.",
    f"- Anchored variable factors: `{anchored_variable_factors}`.",
    f"- Unanchored variable factors: `{unanchored_variable_factors}`.",
    f"- Repeated-variable factors: `{repeated_variable_factors}`.",
    f"- Pure-variable factors: `{pure_variable_factors}`.",
    f"- New-variable items: `{new_var_items}`.",
    f"- Variable-reference items: `{var_ref_items}`.",
    f"- Total variable items: `{variable_items_sum}`.",
    f"- Max variables per factor: `{max_variables_per_factor}`.",
    f"- Max prefix length: `{max_prefix_len}` bytes.",
    "",
    "## Execution Storage",
    "",
    "Counters are process-local and describe temporary `Vec` length/capacity shape,",
    "not allocator events. `Vec::capacity()` is used as the retained-storage proxy.",
    f"- Renormalized plans: `{renorm_plans}`.",
    f"- Renormalized factors: `{renorm_factors}`.",
    f"- Average renormalized factor length / capacity: `{avg_renorm_len:.2f}` / `{avg_renorm_capacity:.2f}` bytes.",
    f"- Max renormalized factor length / capacity: `{max_renorm_len}` / `{max_renorm_capacity}` bytes.",
    f"- Raw searches: `{raw_searches}`.",
    f"- Average raw stack entries: `{avg_raw_stack_entries:.2f}`.",
    f"- Max raw stack entries: `{max_raw_stack_entries}`.",
    f"- Candidate pair vectors: `{candidate_pair_vectors}`.",
    f"- Average candidate pair entries / capacity: `{avg_candidate_pair_entries:.2f}` / `{avg_candidate_pair_capacity:.2f}`.",
    f"- Max candidate pair entries / capacity: `{max_candidate_pair_entries}` / `{max_candidate_pair_capacity}`.",
    f"- General unifications: `{general_unifications}`.",
    f"- Successful unifications: `{successful_unifications}`.",
    f"- Unification failures: `{unification_failures}` (`{unification_failure_rate:.2%}`).",
    f"- General unifications per successful binding: `{unifications_per_success:.2f}`.",
    "",
    "| renormalized factor length bucket | buffers |",
    "| --- | ---: |",
]
for bucket, count in renorm_len_buckets.items():
    lines.append(f"| `{bucket}` | {count} |")
lines += [
    "",
    "| renormalized factor capacity bucket | buffers |",
    "| --- | ---: |",
]
for bucket, count in renorm_capacity_buckets.items():
    lines.append(f"| `{bucket}` | {count} |")
lines += [
    "",
    "| candidate pair capacity bucket | vectors |",
    "| --- | ---: |",
]
for bucket, count in candidate_pair_capacity_buckets.items():
    lines.append(f"| `{bucket}` | {count} |")
lines += [
    "",
    "## Execution",
    "",
    f"- Total `WilliamResult` atoms: `{total_results}`.",
    f"- Steps per MORK run: `{manifest_kv.get('steps', 'unknown')}`.",
    "",
    "| runs | min | median | mean | max |",
    "| ---: | ---: | ---: | ---: | ---: |",
    (
        f"| {len(elapsed)} | {min(elapsed)} ms | "
        f"{statistics.median(elapsed):.2f} ms | "
        f"{statistics.mean(elapsed):.2f} ms | {max(elapsed)} ms |"
    ),
    "",
    "## Raw Runs",
    "",
    "| step | shape | elapsed | results | cache hits | cache misses | cache hit rate | execution line |",
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
]
for row in runs:
    lines.append(
        f"| {row['step']} | `{row['shape']}` | {row['elapsed_ms']} ms | "
        f"{row['william_results']}/{row['expected_results']} | "
        f"{row['cache_hits']} | {row['cache_misses']} | `{row.get('cache_hit_rate', '0.00%')}` | "
        f"`{row['execution_line']}` |"
    )

lines += [
    "",
    "## Environment",
    "",
    f"- LOAD_MAX: `{manifest_kv.get('load_max', 'unknown')}`.",
    f"- ALLOW_BUSY: `{manifest_kv.get('allow_busy', 'unknown')}`.",
    f"- RUSTFLAGS: `{manifest_kv.get('rustflags', 'unknown')}`.",
    f"- Features: `{manifest_kv.get('features', 'unknown')}`.",
    f"- Repeats: `{manifest_kv.get('william_sequence_repeats', 'unknown')}`.",
    f"- Decoys: `{manifest_kv.get('william_decoys', 'unknown')}`.",
    f"- Recorded load1: `{manifest_kv.get('load1', 'unknown')}`.",
    f"- Uptime line: `{load_line}`.",
    "",
    "## Machine Reports",
    "",
    "- `commands.tsv`",
    "- `fixtures.tsv`",
    "- `predictions.tsv`",
    "- `runs.tsv`",
    "- `system.txt`",
    "- `report.json`",
    "- `junit.xml`",
    "- `report_verification.log`",
    "",
]

summary.write_text("\n".join(lines), encoding="utf-8")
print(summary)
PY

printf 'MORK William predictive probe logs written to %s\n' "$out_dir"
