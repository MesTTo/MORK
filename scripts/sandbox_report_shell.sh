#!/usr/bin/env bash

create_sandbox_report_dir() {
  local base_dir="$1"
  local name_prefix="$2"
  local stamp="$3"
  local template

  mkdir -p "$base_dir"
  if [ -n "$name_prefix" ]; then
    template="$base_dir/${name_prefix}-${stamp}.XXXXXX"
  else
    template="$base_dir/${stamp}.XXXXXX"
  fi
  mktemp -d "$template"
}

write_and_verify_sandbox_reports() {
  local suite="$1"
  local report_dir="$2"
  local final_status="$3"
  local verification_log="$report_dir/report_verification.log"
  local verifier_args=()

  if "$PYTHON_BIN" "$ROOT_DIR/scripts/write_sandbox_reports.py" \
    --suite "$suite" \
    --out-dir "$report_dir" \
    --final-status "$final_status"; then
    if [ "$final_status" = "0" ]; then
      verifier_args=(--require-success)
    fi
    "$PYTHON_BIN" "$ROOT_DIR/scripts/verify_sandbox_reports.py" \
      "${verifier_args[@]}" "$report_dir" > "$verification_log" 2>&1
  else
    local writer_status="$?"
    {
      printf 'write_sandbox_reports.py failed with status %s\n' "$writer_status"
      printf 'suite=%s\n' "$suite"
      printf 'final_status=%s\n' "$final_status"
    } > "$verification_log"
    return "$writer_status"
  fi
}

finish_sandbox_exit() {
  local original_status="$1"
  shift
  local finalizer_status=0

  set +e
  "$@" "$original_status"
  finalizer_status="$?"
  set -e

  if [ "$original_status" = "0" ]; then
    exit "$finalizer_status"
  fi
  exit "$original_status"
}
