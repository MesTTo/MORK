#!/usr/bin/env python3
"""Write machine-readable reports for sandbox command ledgers."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from sandbox_report_common import read_commands, read_manifest


def write_json(
    path: Path,
    suite: str,
    final_status: int,
    manifest: dict[str, str],
    commands: list[dict[str, object]],
) -> None:
    failures = sum(1 for command in commands if command["status"] != 0)
    report = {
        "suite": suite,
        "final_status": final_status,
        "tests": len(commands),
        "failures": failures,
        "elapsed_ms": sum(int(command["elapsed_ms"]) for command in commands),
        "manifest": manifest,
        "commands": commands,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_junit(
    path: Path,
    suite: str,
    final_status: int,
    manifest: dict[str, str],
    commands: list[dict[str, object]],
) -> None:
    synthetic_failure = final_status != 0 and not any(command["status"] != 0 for command in commands)
    tests = len(commands) + (1 if synthetic_failure else 0)
    failures = sum(1 for command in commands if command["status"] != 0) + (
        1 if synthetic_failure else 0
    )
    elapsed_seconds = sum(int(command["elapsed_ms"]) for command in commands) / 1000.0

    root = ET.Element(
        "testsuites",
        {
            "tests": str(tests),
            "failures": str(failures),
            "errors": "0",
            "skipped": "0",
            "time": f"{elapsed_seconds:.3f}",
        },
    )
    suite_node = ET.SubElement(
        root,
        "testsuite",
        {
            "name": suite,
            "tests": str(tests),
            "failures": str(failures),
            "errors": "0",
            "skipped": "0",
            "time": f"{elapsed_seconds:.3f}",
            "timestamp": manifest.get("stamp_utc", ""),
        },
    )
    properties = ET.SubElement(suite_node, "properties")
    for key, value in sorted(manifest.items()):
        ET.SubElement(properties, "property", {"name": key, "value": value})

    for command in commands:
        status = int(command["status"])
        elapsed_ms = int(command["elapsed_ms"])
        log = str(command["log"])
        case = ET.SubElement(
            suite_node,
            "testcase",
            {
                "classname": suite,
                "name": str(command["name"]),
                "time": f"{elapsed_ms / 1000.0:.3f}",
                "file": log,
            },
        )
        ET.SubElement(case, "system-out").text = f"log: {log}"
        if status != 0:
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "message": f"gate exited with status {status}",
                    "type": "SandboxGateFailure",
                },
            )
            failure.text = f"See {log} for command output."

    if synthetic_failure:
        case = ET.SubElement(
            suite_node,
            "testcase",
            {
                "classname": suite,
                "name": "sandbox_exit",
                "time": "0.000",
                "file": "summary.md",
            },
        )
        failure = ET.SubElement(
            case,
            "failure",
            {
                "message": f"sandbox exited with status {final_status}",
                "type": "SandboxExitFailure",
            },
        )
        failure.text = "Sandbox exited without a recorded failing gate."

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--final-status", required=True, type=int)
    args = parser.parse_args()

    manifest = read_manifest(args.out_dir / "manifest.txt")
    commands = read_commands(args.out_dir / "commands.tsv", args.out_dir, require_log_exists=True)
    write_json(args.out_dir / "report.json", args.suite, args.final_status, manifest, commands)
    write_junit(args.out_dir / "junit.xml", args.suite, args.final_status, manifest, commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
