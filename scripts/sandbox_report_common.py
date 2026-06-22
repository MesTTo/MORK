#!/usr/bin/env python3
"""Shared helpers for sandbox report generation and verification."""

from __future__ import annotations

import csv
from pathlib import Path


EXPECTED_COMMAND_FIELDS = ["name", "status", "elapsed_ms", "log"]


def read_manifest(path: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    if not path.exists():
        return manifest
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            manifest[key] = value
    return manifest


def resolve_log_path(report_dir: Path, log: str) -> Path:
    log_path = Path(log)
    if not log or log_path == Path("."):
        raise ValueError("log path must be non-empty")
    if log_path.is_absolute():
        raise ValueError(f"log path {log!r} must be relative to the report directory")
    if ".." in log_path.parts:
        raise ValueError(f"log path {log!r} must not escape the report directory")

    report_root = report_dir.resolve()
    resolved = (report_root / log_path).resolve()
    try:
        resolved.relative_to(report_root)
    except ValueError as err:
        raise ValueError(f"log path {log!r} must stay inside {report_dir}") from err
    return resolved


def read_commands(path: Path, report_dir: Path, require_log_exists: bool) -> list[dict[str, object]]:
    if not path.exists():
        return []

    commands: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != EXPECTED_COMMAND_FIELDS:
            raise ValueError(
                f"{path}: expected TSV header {EXPECTED_COMMAND_FIELDS}, got {reader.fieldnames}"
            )
        for line_no, row in enumerate(reader, start=2):
            if not row["name"]:
                raise ValueError(f"{path}:{line_no}: command name must be non-empty")
            try:
                status = int(row["status"])
                elapsed_ms = int(row["elapsed_ms"])
            except ValueError as err:
                raise ValueError(f"{path}:{line_no}: status and elapsed_ms must be integers") from err
            if elapsed_ms < 0:
                raise ValueError(f"{path}:{line_no}: elapsed_ms must be non-negative")

            log = row["log"]
            log_path = resolve_log_path(report_dir, log)
            if require_log_exists and not log_path.exists():
                raise ValueError(f"{path}:{line_no}: log {log!r} does not exist")

            commands.append(
                {
                    "name": row["name"],
                    "status": status,
                    "elapsed_ms": elapsed_ms,
                    "log": log,
                }
            )
    return commands
