#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMAL_DIR="${VERUS_FORMAL_DIR:-$REPO_ROOT/kernel/resources/formal/verus}"
VERUS_BIN="${VERUS:-}"

if [[ -z "$VERUS_BIN" ]]; then
    if command -v verus >/dev/null 2>&1; then
        VERUS_BIN="$(command -v verus)"
    elif [[ -x "$HOME/.local/bin/verus" ]]; then
        VERUS_BIN="$HOME/.local/bin/verus"
    else
        echo "verus not found. Set VERUS=/path/to/verus or install it under ~/.local/bin/verus." >&2
        exit 1
    fi
fi

"$REPO_ROOT/scripts/check_verus_no_cheats.sh" "$FORMAL_DIR"

mapfile -d '' FILES < <(find "$FORMAL_DIR" -type f -name '*.rs' -print0 | sort -z)
if [[ "${#FILES[@]}" -eq 0 ]]; then
    echo "No Verus proof files found under $FORMAL_DIR" >&2
    exit 1
fi

for file in "${FILES[@]}"; do
    echo "verus ${file#$REPO_ROOT/}"
    "$VERUS_BIN" "$file" --no-cheating --expand-errors
done
