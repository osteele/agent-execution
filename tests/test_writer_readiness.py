"""Writer-readiness protocol checks at the worker and SDK-subprocess boundaries.

Registry and credential policy is exercised against SDK interfaces in
``tests/omp_sdk.test.ts``; these tests cover request validation before launch,
the bounded subprocess boundary, and validation of the evidence it returns.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from typing import cast
from unittest import mock

from agent_execution.omp_execution import probe_omp_writers
from agent_execution.worker_cli import probe_writers

SELECTOR = "openai-codex/gpt-6.1-sol"
OTHER = "kimi-code/k3"
#: The external process boundary, patched where it is defined: the probe
#: imports it from this module at call time.
RUNNER = "agent_execution.processes.run_in_process_group"


def evidence(rows: list[dict[str, object]]) -> str:
    return json.dumps(
        {"schema_version": "agent-execution.omp-writer-readiness/v1", "writers": rows}
    )


def row(
    *, model: bool = True, credential: bool = True, selector: str = SELECTOR
) -> dict[str, object]:
    return {
        "selector": selector,
        "model_available": model,
        "credential_available": credential,
        "available": model and credential,
        "detail": "observed by pinned SDK fixture",
    }


def completed(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["bun"], returncode, stdout, stderr)


class WriterReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        for target in (
            "agent_execution.worker_cli.require_omp_sdk",
            "agent_execution.omp_execution.require_omp_sdk",
        ):
            patcher = mock.patch(target, return_value=("/sdk", "/bun"))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_requested_selectors_reach_bounded_sdk_launch_and_rows_are_validated(self) -> None:
        stdout = evidence([row(selector=OTHER, credential=False), row(selector=SELECTOR)])
        with mock.patch(RUNNER, return_value=completed(stdout)) as runner:
            response = probe_writers(selectors=[SELECTOR, OTHER], timeout=7)
        command, _cwd, prompt, timeout = runner.call_args.args
        self.assertIn("--probe-writers", command)
        self.assertEqual(
            json.loads(command[command.index("--probe-writers") - 1]), [SELECTOR, OTHER]
        )
        self.assertEqual(prompt, "")
        self.assertEqual(timeout, 7)
        self.assertEqual(response["schema_version"], "agent-execution.writer-readiness/v1")
        rows = response["writers"]
        assert isinstance(rows, list)
        self.assertEqual([item["selector"] for item in rows], [SELECTOR, OTHER])
        self.assertEqual([item["available"] for item in rows], [True, False])

    def test_probe_rejects_malformed_evidence(self) -> None:
        payloads = {
            "invalid json": "{not json",
            "wrong schema": json.dumps({"schema_version": "other/v1", "writers": [row()]}),
            "missing writers": json.dumps(
                {"schema_version": "agent-execution.omp-writer-readiness/v1"}
            ),
            "missing row": evidence([]),
            "duplicate row": evidence([row(), row()]),
            "mismatched selector": evidence([row(selector="openai-codex/gpt-6-sol")]),
            "non-bool flag": evidence([{**row(), "model_available": 1}]),
            "inconsistent available": evidence([{**row(), "available": False}]),
            "empty detail": evidence([{**row(), "detail": "  "}]),
            "nontext detail": evidence([{**row(), "detail": None}]),
        }
        for label, payload in payloads.items():
            with (
                self.subTest(label),
                mock.patch(RUNNER, return_value=completed(payload)),
                self.assertRaises(ValueError),
            ):
                probe_omp_writers([SELECTOR])

    def test_failed_probe_reports_helper_error(self) -> None:
        failure = completed("", returncode=1, stderr="SDK registry failed to load\n")
        with (
            mock.patch(RUNNER, return_value=failure),
            self.assertRaises(ValueError) as raised,
        ):
            probe_omp_writers([SELECTOR])
        self.assertIn("SDK registry failed to load", str(raised.exception))

    def test_probe_timeout_is_forwarded_and_propagates(self) -> None:
        timeout = subprocess.TimeoutExpired(["bun"], 0.01)
        with (
            mock.patch(RUNNER, side_effect=timeout) as runner,
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            probe_omp_writers([SELECTOR], timeout=0.01)
        self.assertEqual(runner.call_args.args[3], 0.01)

    def test_malformed_requests_are_rejected_before_sdk_launch(self) -> None:
        malformed_selectors: list[object] = [
            ["unknown/model"],
            ["not-exact"],
            [SELECTOR, SELECTOR],
            [1],
            [],
            SELECTOR,
            42,
        ]
        malformed_timeouts: list[object] = [0, -1, float("nan"), float("inf"), "30", None, True]
        with mock.patch(RUNNER) as runner:
            for selectors in malformed_selectors:
                with self.subTest(selectors=selectors), self.assertRaises(ValueError):
                    probe_writers(selectors=cast("list[str]", selectors), timeout=30)
            for timeout in malformed_timeouts:
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    probe_writers(selectors=[SELECTOR], timeout=cast(float, timeout))
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
