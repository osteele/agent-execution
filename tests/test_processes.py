"""Process-group containment, spawn admission, and survivor cleanup."""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Mapping
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from agent_execution import processes
from agent_execution.processes import (
    ProcessIdentity,
    census,
    guard_live_groups,
    install_thread_failure_guard,
    run_in_process_group,
    signal_group,
    thread_failures,
)

SPAWN_GATE = str(Path(processes.__file__).with_name("spawn_gate.py"))


class ContainmentTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        processes.take_telemetry()

    def run_command(
        self,
        args: list[str],
        *,
        timeout: float | None = 5.0,
        grace_seconds: float = processes.GRACE_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        return run_in_process_group(
            [sys.executable, "-c", *args],
            self.root,
            "",
            timeout,
            grace_seconds=grace_seconds,
        )

    def assert_no_group(self, pgid: int) -> None:
        self.assertEqual(census(pgid=pgid), ())

    def test_child_leads_its_session_and_process_group(self) -> None:
        completed = self.run_command(
            ["import os; print(os.getpid(), os.getpgrp(), os.getsid(0), flush=True)"]
        )
        pid, pgid, sid = (int(value) for value in completed.stdout.split())
        self.assertEqual((pgid, sid), (pid, pid))
        telemetry = processes.take_telemetry()
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        self.assertEqual(telemetry.pgid, pid)
        self.assert_no_group(pgid)

    def test_spawn_gate_requires_release_and_executes_after_admission(self) -> None:
        marker = self.root / "marker"
        release_read, release_write = os.pipe()
        os.close(release_write)
        try:
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                subprocess.run(
                    [
                        sys.executable,
                        SPAWN_GATE,
                        str(release_read),
                        sys.executable,
                        "-c",
                        "raise SystemExit(0)",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    pass_fds=(release_read,),
                    timeout=5.0,
                )
            self.assertIn(
                "spawn gate closed before execution was admitted", raised.exception.stderr
            )
        finally:
            os.close(release_read)
        completed = run_in_process_group(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('executed')",
            ],
            self.root,
            "",
            5.0,
            hold_before_exec=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(marker.read_text(), "executed")

    def test_prompt_delivery_deadline_kills_the_group(self) -> None:
        program = "import time; time.sleep(2)"
        with self.assertRaises(subprocess.TimeoutExpired):
            run_in_process_group(
                [sys.executable, "-c", program],
                self.root,
                "x" * (16 * 1024 * 1024),
                0.2,
            )
        telemetry = processes.take_telemetry()
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        self.assertIn(telemetry.kill_kind, {"group-term", "group-kill"})
        self.assert_no_group(telemetry.pgid or -1)

    def test_timeout_kills_the_descendant_group_and_preserves_partial_output(self) -> None:
        program = """
import subprocess, sys, time
print('partial-output', flush=True)
subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
time.sleep(30)
"""
        with self.assertRaises(subprocess.TimeoutExpired) as raised:
            self.run_command([program], timeout=0.2)
        self.assertIn("partial-output", raised.exception.output or "")
        telemetry = processes.take_telemetry()
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        self.assertIn(telemetry.kill_kind, {"group-term", "group-kill"})
        self.assert_no_group(telemetry.pgid or -1)

    def test_reaping_defaults_to_enabled(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("AGENT_EXECUTION_HARNESS_REAP", None)
            self.assertTrue(processes.reaping_enabled())

    def test_clean_exit_records_and_reaps_a_leftover_descendant(self) -> None:
        program = """
import subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
print(child.pid, flush=True)
"""
        with mock.patch.dict(os.environ, {"AGENT_EXECUTION_HARNESS_REAP": "1"}):
            completed = self.run_command([program], timeout=5.0)
        telemetry = processes.take_telemetry()
        self.assertEqual(completed.returncode, 0)
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        self.assertEqual(telemetry.kill_kind, "group-sweep")
        self.assertEqual(len(telemetry.survivors), 1)
        self.assertTrue(telemetry.survivors_reaped)
        self.assert_no_group(telemetry.pgid or -1)

    def test_reaping_can_be_disabled_without_claiming_cleanup(self) -> None:
        with mock.patch.dict(os.environ, {"AGENT_EXECUTION_HARNESS_REAP": "0"}):
            completed = self.run_command(
                [
                    (
                        "import subprocess, sys; "
                        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])"
                    )
                ],
                timeout=5.0,
            )
        telemetry = processes.take_telemetry()
        self.assertEqual(completed.returncode, 0)
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        self.assertEqual(telemetry.kill_kind, "group-sweep")
        self.assertEqual(len(telemetry.survivors), 1)
        self.assertFalse(telemetry.survivors_reaped)
        try:
            signal_group(telemetry.pgid or -1, signal.SIGKILL)
            time.sleep(processes.SETTLE_SECONDS)
        finally:
            self.assert_no_group(telemetry.pgid or -1)

    def test_spawn_observer_failure_never_executes_the_command(self) -> None:
        marker = self.root / "must-not-exist"

        def fail(identity: ProcessIdentity) -> None:
            raise ValueError("record lost")

        with self.assertRaisesRegex(ValueError, "record lost"):
            run_in_process_group(
                [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('x')",
                ],
                self.root,
                "",
                5.0,
                on_spawn=fail,
                hold_before_exec=True,
            )
        self.assertFalse(marker.exists())

    def test_guarded_interrupt_drains_inflight_groups_once(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        try:
            with self.assertRaises(RuntimeError), guard_live_groups():
                process = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    start_new_session=True,
                )
                processes._live.add(process.pid)
                raise RuntimeError("interrupted")
            assert process is not None
            process.wait(timeout=5.0)
            self.assertEqual(process.returncode, -signal.SIGKILL)
            self.assertEqual(processes.kill_live_groups(), ())
        finally:
            if process is not None:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5.0)
                processes._live.discard(process.pid)

    def test_telemetry_is_consumed_once_and_scoped_to_thread(self) -> None:
        self.run_command(["print('ok')"], timeout=5.0)
        first = processes.take_telemetry()
        self.assertIsNotNone(first)
        self.assertIsNone(processes.take_telemetry())
        seen: list[processes.GroupTelemetry | None] = []

        def worker() -> None:
            seen.append(processes.take_telemetry())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertEqual(seen, [None])

    def test_resource_samples_match_final_telemetry(self) -> None:
        samples: list[dict[str, object]] = []

        def record(sample: Mapping[str, object]) -> None:
            samples.append(dict(sample))

        completed = run_in_process_group(
            [sys.executable, "-c", "import time; time.sleep(0.2)"],
            self.root,
            "",
            5.0,
            on_resource_sample=record,
        )
        telemetry = processes.take_telemetry()
        self.assertEqual(completed.returncode, 0)
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        self.assertTrue(samples)
        final_sample = samples[-1]
        self.assertEqual(final_sample["sample_count"], telemetry.resource_sample_count)
        self.assertEqual(len(samples), telemetry.resource_sample_count)
        self.assertEqual(final_sample["peak_process_count"], telemetry.peak_process_count)
        self.assertEqual(final_sample["peak_rss_kib"], telemetry.peak_rss_kib)
        self.assertEqual(final_sample["peak_cpu_percent"], round(telemetry.peak_cpu_percent, 3))
        self.assertGreaterEqual(telemetry.peak_process_count, 1)
        self.assertGreater(telemetry.peak_rss_kib, 0)

    def test_resource_observer_failure_is_diagnostic(self) -> None:
        def fail(sample: Mapping[str, object]) -> None:
            raise ValueError("sample sink failed")

        completed = run_in_process_group(
            [sys.executable, "-c", "import time; time.sleep(0.1)"],
            self.root,
            "",
            5.0,
            on_resource_sample=fail,
        )
        telemetry = processes.take_telemetry()
        self.assertEqual(completed.returncode, 0)
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        self.assertEqual(telemetry.resource_observation_error, "ValueError: sample sink failed")

    def test_thread_failure_is_recorded(self) -> None:
        previous_hook = threading.excepthook
        self.addCleanup(setattr, threading, "excepthook", previous_hook)
        install_thread_failure_guard()

        def fail() -> None:
            raise RuntimeError("background failure")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            failing = threading.Thread(target=fail)
            failing.start()
            failing.join()
        self.assertIn("RuntimeError: background failure", stderr.getvalue())
        self.assertTrue(any("background failure" in failure for failure in thread_failures()))

    def test_census_and_signal_refuse_ambiguous_groups(self) -> None:
        for pgid in (0, 1, -1):
            with self.subTest(pgid=pgid), self.assertRaises(ValueError):
                census(pgid=pgid)
            with self.assertRaises(ValueError):
                signal_group(pgid, signal.SIGTERM)

    def test_ps_failure_is_unknown_not_empty(self) -> None:
        ps = self.root / "ps"
        ps.write_text("#!/bin/sh\nprintf 'ps failed\\n' >&2\nexit 1\n")
        ps.chmod(0o755)
        path = f"{self.root}{os.pathsep}{os.environ.get('PATH', '')}"
        with (
            mock.patch.dict(os.environ, {"PATH": path}),
            self.assertRaises(subprocess.CalledProcessError) as raised,
        ):
            processes._ps_lines()
        self.assertEqual(raised.exception.stderr.strip(), "ps failed")

    def test_group_termination_escalates_after_grace(self) -> None:
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_command(
                [
                    (
                        "import subprocess, sys, time; "
                        "subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']); "
                        "time.sleep(30)"
                    )
                ],
                timeout=0.2,
                grace_seconds=0.05,
            )
        telemetry = processes.take_telemetry()
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        self.assertEqual(telemetry.kill_kind, "group-kill")
        self.assert_no_group(telemetry.pgid or -1)
