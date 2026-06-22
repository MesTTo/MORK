#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v opam >/dev/null 2>&1; then
  eval "$(opam env)"
fi

if ! command -v abella >/dev/null 2>&1; then
  echo "abella executable not found on PATH; install it with opam install abella" >&2
  exit 127
fi

for target in kernel/resources/formal/abella/*.thm; do
  printf 'checking %s\n' "$target"
  abella "$target"
done
