#!/usr/bin/env python3
"""Focused tests for sandbox report verification."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRITER = ROOT / "scripts" / "write_sandbox_reports.py"
VERIFIER = ROOT / "scripts" / "verify_sandbox_reports.py"
SHELL_HELPER = ROOT / "scripts" / "sandbox_report_shell.sh"


def run_python(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *(str(arg) for arg in args)],
        check=False,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class VerifySandboxReportsTest(unittest.TestCase):
    def run_writer(self, report_dir: Path) -> subprocess.CompletedProcess[str]:
        return run_python(
            WRITER,
            "--suite",
            "verifier-test",
            "--out-dir",
            report_dir,
            "--final-status",
            "0",
        )

    def write_report(self, report_dir: Path, log: str) -> subprocess.CompletedProcess[str]:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "manifest.txt").write_text("stamp_utc=2026-06-17T00:00:00Z\n")
        (report_dir / "commands.tsv").write_text(
            f"name\tstatus\telapsed_ms\tlog\nsample\t0\t7\t{log}\n",
            encoding="utf-8",
        )
        return self.run_writer(report_dir)

    def test_relative_nested_log_path_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report"
            (report_dir / "logs").mkdir(parents=True)
            (report_dir / "logs" / "sample.log").write_text("ok\n", encoding="utf-8")
            write_result = self.write_report(report_dir, "logs/sample.log")
            self.assertEqual(write_result.returncode, 0, write_result.stderr)

            result = run_python(VERIFIER, "--require-success", report_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("OK", result.stdout)

    def test_parent_relative_log_path_fails_during_report_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "report"
            (root / "outside.log").write_text("escaped\n", encoding="utf-8")

            result = self.write_report(report_dir, "../outside.log")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not escape", result.stderr)
            self.assertFalse((report_dir / "report.json").exists())

    def test_absolute_log_path_fails_during_report_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "report"
            outside = root / "absolute.log"
            outside.write_text("escaped\n", encoding="utf-8")

            result = self.write_report(report_dir, str(outside))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be relative", result.stderr)
            self.assertFalse((report_dir / "report.json").exists())

    def test_missing_log_fails_during_report_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report"

            result = self.write_report(report_dir, "missing.log")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not exist", result.stderr)
            self.assertFalse((report_dir / "report.json").exists())

    def test_invalid_command_header_fails_during_report_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report"
            report_dir.mkdir(parents=True)
            (report_dir / "manifest.txt").write_text("stamp_utc=2026-06-17T00:00:00Z\n")
            (report_dir / "sample.log").write_text("ok\n", encoding="utf-8")
            (report_dir / "commands.tsv").write_text(
                "name\tstatus\tlog\nsample\t0\tsample.log\n",
                encoding="utf-8",
            )

            result = self.run_writer(report_dir)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected TSV header", result.stderr)
            self.assertFalse((report_dir / "report.json").exists())


class SandboxReportShellTest(unittest.TestCase):
    def run_report_dir_helper(self, base_dir: Path, prefix: str, stamp: str) -> subprocess.CompletedProcess[str]:
        script = f"""
set -euo pipefail
. {str(SHELL_HELPER)!r}
first="$(create_sandbox_report_dir {str(base_dir)!r} {prefix!r} {stamp!r})"
second="$(create_sandbox_report_dir {str(base_dir)!r} {prefix!r} {stamp!r})"
test "$first" != "$second"
test -d "$first"
test -d "$second"
case "$(basename "$first")" in
  {prefix}-{stamp}.*) ;;
  *) printf 'unexpected first dir: %s\n' "$first" >&2; exit 1 ;;
esac
case "$(basename "$second")" in
  {prefix}-{stamp}.*) ;;
  *) printf 'unexpected second dir: %s\n' "$second" >&2; exit 1 ;;
esac
printf '%s\n%s\n' "$first" "$second"
"""
        return subprocess.run(
            ["bash", "-c", script],
            check=False,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def run_finish_helper(self, report_dir: Path, original_status: int) -> subprocess.CompletedProcess[str]:
        script = f"""
set -euo pipefail
ROOT_DIR={str(ROOT)!r}
PYTHON_BIN={sys.executable!r}
REPORT_DIR={str(report_dir)!r}
. {str(SHELL_HELPER)!r}
finalize() {{
  write_and_verify_sandbox_reports shell-helper-test "$REPORT_DIR" "$1"
}}
finish_sandbox_exit {original_status} finalize
"""
        return subprocess.run(
            ["bash", "-c", script],
            check=False,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write_command_ledger(self, report_dir: Path, log: str, create_log: bool) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "manifest.txt").write_text("stamp_utc=2026-06-17T00:00:00Z\n")
        (report_dir / "commands.tsv").write_text(
            f"name\tstatus\telapsed_ms\tlog\nsample\t0\t7\t{log}\n",
            encoding="utf-8",
        )
        if create_log:
            (report_dir / log).write_text("ok\n", encoding="utf-8")

    def test_create_sandbox_report_dir_uses_unique_suffixed_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "reports"
            stamp = "20260617T000000Z"

            result = self.run_report_dir_helper(base_dir, "collision-check", stamp)

            self.assertEqual(result.returncode, 0, result.stderr)
            paths = [Path(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(paths), 2, result.stdout)
            self.assertNotEqual(paths[0], paths[1])
            for path in paths:
                self.assertTrue(path.is_dir())
                self.assertEqual(path.parent, base_dir)
                self.assertRegex(path.name, rf"^collision-check-{stamp}\.[A-Za-z0-9]+$")

    def test_clean_run_with_valid_reports_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report"
            self.write_command_ledger(report_dir, "sample.log", create_log=True)

            result = self.run_finish_helper(report_dir, original_status=0)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("OK", (report_dir / "report_verification.log").read_text())

    def test_clean_run_with_invalid_reports_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report"
            self.write_command_ledger(report_dir, "missing.log", create_log=False)

            result = self.run_finish_helper(report_dir, original_status=0)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("write_sandbox_reports.py failed", (report_dir / "report_verification.log").read_text())

    def test_failed_run_preserves_original_status_when_report_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report"
            self.write_command_ledger(report_dir, "missing.log", create_log=False)

            result = self.run_finish_helper(report_dir, original_status=75)

            self.assertEqual(result.returncode, 75)
            self.assertIn("write_sandbox_reports.py failed", (report_dir / "report_verification.log").read_text())


if __name__ == "__main__":
    unittest.main()
