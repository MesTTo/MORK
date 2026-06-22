#!/usr/bin/env python3
"""Verify sandbox report.json, junit.xml, and commands.tsv consistency."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from sandbox_report_common import read_commands, read_manifest, resolve_log_path


def int_attr(node: ET.Element, attr: str, context: str) -> int:
    value = node.get(attr)
    if value is None:
        raise ValueError(f"{context}: missing {attr!r} attribute")
    try:
        return int(value)
    except ValueError as err:
        raise ValueError(f"{context}: {attr!r} must be an integer, got {value!r}") from err


def first_testsuite(root: ET.Element, path: Path) -> ET.Element:
    if root.tag == "testsuite":
        return root
    if root.tag != "testsuites":
        raise ValueError(f"{path}: root element must be testsuites or testsuite, got {root.tag!r}")
    suites = root.findall("testsuite")
    if len(suites) != 1:
        raise ValueError(f"{path}: expected exactly one testsuite, got {len(suites)}")
    return suites[0]


def validate_junit(
    path: Path,
    suite: str,
    final_status: int,
    commands: list[dict[str, object]],
) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    suite_node = first_testsuite(root, path)
    if suite_node.get("name") != suite:
        raise ValueError(f"{path}: testsuite name {suite_node.get('name')!r} != {suite!r}")

    command_failures = sum(1 for command in commands if int(command["status"]) != 0)
    synthetic_failure = final_status != 0 and command_failures == 0
    expected_tests = len(commands) + (1 if synthetic_failure else 0)
    expected_failures = command_failures + (1 if synthetic_failure else 0)

    suite_tests = int_attr(suite_node, "tests", f"{path}: testsuite")
    suite_failures = int_attr(suite_node, "failures", f"{path}: testsuite")
    if suite_tests != expected_tests:
        raise ValueError(f"{path}: testsuite tests {suite_tests} != expected {expected_tests}")
    if suite_failures != expected_failures:
        raise ValueError(
            f"{path}: testsuite failures {suite_failures} != expected {expected_failures}"
        )

    if root.tag == "testsuites":
        root_tests = int_attr(root, "tests", f"{path}: testsuites")
        root_failures = int_attr(root, "failures", f"{path}: testsuites")
        if root_tests != expected_tests:
            raise ValueError(f"{path}: testsuites tests {root_tests} != expected {expected_tests}")
        if root_failures != expected_failures:
            raise ValueError(
                f"{path}: testsuites failures {root_failures} != expected {expected_failures}"
            )

    cases = suite_node.findall("testcase")
    if len(cases) != expected_tests:
        raise ValueError(f"{path}: testcase count {len(cases)} != expected {expected_tests}")
    cases_by_name: dict[str, ET.Element] = {}
    for case in cases:
        name = case.get("name")
        if not name:
            raise ValueError(f"{path}: testcase without name")
        if name in cases_by_name:
            raise ValueError(f"{path}: duplicate testcase name {name!r}")
        cases_by_name[name] = case

    for command in commands:
        name = str(command["name"])
        status = int(command["status"])
        log = str(command["log"])
        case = cases_by_name.get(name)
        if case is None:
            raise ValueError(f"{path}: missing testcase for command {name!r}")
        if case.get("file") != log:
            raise ValueError(f"{path}: testcase {name!r} file {case.get('file')!r} != {log!r}")
        failures = case.findall("failure")
        if status == 0 and failures:
            raise ValueError(f"{path}: passing testcase {name!r} has failure nodes")
        if status != 0 and len(failures) != 1:
            raise ValueError(f"{path}: failing testcase {name!r} should have one failure node")
        system_out = case.find("system-out")
        if system_out is None or log not in (system_out.text or ""):
            raise ValueError(f"{path}: testcase {name!r} system-out does not reference {log!r}")

    if synthetic_failure:
        case = cases_by_name.get("sandbox_exit")
        if case is None or len(case.findall("failure")) != 1:
            raise ValueError(f"{path}: missing synthetic sandbox_exit failure")

    return expected_tests, expected_failures


def validate_report_dir(path: Path, require_success: bool) -> tuple[str, int, int, int]:
    commands_path = path / "commands.tsv"
    report_path = path / "report.json"
    junit_path = path / "junit.xml"
    missing = [p.name for p in (commands_path, report_path, junit_path) if not p.exists()]
    if missing:
        raise ValueError(f"{path}: missing required report files: {', '.join(missing)}")

    commands = read_commands(commands_path, path, require_log_exists=True)
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    suite = str(report.get("suite", ""))
    if not suite:
        raise ValueError(f"{report_path}: missing suite")
    try:
        final_status = int(report["final_status"])
        tests = int(report["tests"])
        failures = int(report["failures"])
        elapsed_ms = int(report["elapsed_ms"])
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(
            f"{report_path}: final_status, tests, failures, elapsed_ms must exist as integers"
        ) from err

    expected_failures = sum(1 for command in commands if int(command["status"]) != 0)
    expected_elapsed = sum(int(command["elapsed_ms"]) for command in commands)
    if tests != len(commands):
        raise ValueError(f"{report_path}: tests {tests} != commands {len(commands)}")
    if failures != expected_failures:
        raise ValueError(f"{report_path}: failures {failures} != expected {expected_failures}")
    if elapsed_ms != expected_elapsed:
        raise ValueError(f"{report_path}: elapsed_ms {elapsed_ms} != expected {expected_elapsed}")
    if report.get("commands") != commands:
        raise ValueError(f"{report_path}: commands do not match commands.tsv")
    manifest = read_manifest(path / "manifest.txt")
    if report.get("manifest") != manifest:
        raise ValueError(f"{report_path}: manifest does not match manifest.txt")

    for command in commands:
        log = str(command["log"])
        log_path = resolve_log_path(path, log)
        if not log_path.exists():
            raise ValueError(f"{path}: command {command['name']!r} log {log!r} does not exist")

    junit_tests, junit_failures = validate_junit(junit_path, suite, final_status, commands)
    if require_success and (final_status != 0 or failures != 0 or junit_failures != 0):
        raise ValueError(
            f"{path}: report is internally valid but not successful "
            f"(final_status={final_status}, failures={failures})"
        )
    return suite, final_status, junit_tests, junit_failures


def discover_report_dirs(paths: list[Path], recursive: bool) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        candidates = [path]
        if recursive and path.is_dir():
            candidates.extend(report_path for report_path in path.rglob("report.json"))
        for candidate in candidates:
            directory = candidate.parent if candidate.name == "report.json" else candidate
            if not (directory / "report.json").exists():
                continue
            resolved = directory.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(directory)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="only verify the exact directories provided",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="fail if an internally valid report contains failed gates",
    )
    args = parser.parse_args()

    report_dirs = discover_report_dirs(args.paths, recursive=not args.no_recursive)
    if not report_dirs:
        print("no report.json files found", file=sys.stderr)
        return 1

    ok = True
    for report_dir in report_dirs:
        try:
            suite, final_status, tests, failures = validate_report_dir(
                report_dir, require_success=args.require_success
            )
        except Exception as err:
            ok = False
            print(f"FAIL {report_dir}: {err}", file=sys.stderr)
        else:
            print(
                f"OK {report_dir}: suite={suite} final_status={final_status} "
                f"tests={tests} failures={failures}"
            )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
