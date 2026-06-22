#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-kernel/resources/formal/verus}"

if [[ ! -d "$ROOT" ]]; then
    echo "Verus proof directory not found: $ROOT" >&2
    exit 1
fi

PATTERN='assume[[:space:]]*\(|admit[[:space:]]*\(|external_body|axiom|unimplemented![[:space:]]*\('

if rg -n --pcre2 "$PATTERN" "$ROOT"; then
    echo "Verus cheat check failed: forbidden proof shortcut found under $ROOT" >&2
    exit 1
fi

echo "Verus cheat check passed: $ROOT"
