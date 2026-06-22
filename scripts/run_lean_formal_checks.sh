#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v lean >/dev/null 2>&1; then
  echo "lean executable not found on PATH" >&2
  exit 127
fi

for target in kernel/resources/formal/lean/*.lean; do
  printf 'checking %s\n' "$target"
  lean "$target"
done
