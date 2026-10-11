"""Renewable budget clocks, OMP evidence qualification, and result telemetry.

Deterministic boundary tests drive ``BudgetClock`` on an injected monotonic
clock. The subprocess scenarios run the actual process owner and the actual
``run_omp_command`` observer against a stub OMP SDK launcher whose children
emit the genuine event schema — ``tool_execution_start`` with ``args`` and
``tool_execution_end`` with ``result`` and an explicit ``isError``, and no
arguments on the end event.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest import mock
from unittest.mock import ANY

from agent_execution import processes
from agent_execution.budget import (
    MAX_DECISION_RECORDS,
    BudgetClock,
    RenewableBudget,
    validate_budget_stats,
)
from agent_execution.identity import worker_evidence_path
from agent_execution.omp_execution import OMP_SDK_VERSION, run_omp_command
from agent_execution.processes import (
    BudgetedCompletedProcess,
    BudgetTimeoutExpired,
    census,
    run_in_process_group,
)
from agent_execution.worker import (
    WORKER_RESULT_SCHEMA,
    WorkerResult,
    execute_worker,
    run_command_with_prompt,
)

WRITER_COMMAND = [
    "omp",
    "-p",
    "write the file",
    "--mode",
    "json",
    "--cwd",
    ".",
    "--model",
    "anthropic/claude-opus-5-5",
    "--execution-tool-policy",
    "workspace-write-no-shell",
]

STDIN_COMMAND = [
    "omp",
    "-p",
    "-",
    "--mode",
    "json",
    "--cwd",
    ".",
    "--model",
    "anthropic/claude-opus-5-5",
    "--execution-tool-policy",
    "workspace-write-no-shell",
]

EXHAUSTED_TELEMETRY: dict[str, object] = {
    "schema_version": 1,
    "policy": {
        "initial_seconds": 0.5,
        "extension_seconds": 0.5,
        "progress_window_seconds": 30.0,
        "max_seconds": 30.0,
    },
    "duration_seconds": 12.5,
    "termination_reason": "no_recent_progress",
    "decisions": [
        {"at_seconds": 0.5, "evidence_age_seconds": 0.4, "decision": "extended"},
        {"at_seconds": 12.5, "evidence_age_seconds": None, "decision": "no_recent_progress"},
    ],
    "qualifying_progress_count": 3,
    "changed_file_count": 2,
}

COMPLETED_TELEMETRY: dict[str, object] = {
    "schema_version": 1,
    "policy": {
        "initial_seconds": 3600.0,
        "extension_seconds": 1800.0,
        "progress_window_seconds": 900.0,
        "max_seconds": 14400.0,
    },
    "duration_seconds": 6211.0,
    "termination_reason": "completed",
    "decisions": [
        {"at_seconds": 3600.0, "evidence_age_seconds": 12.0, "decision": "extended"},
        {"at_seconds": 5400.0, "evidence_age_seconds": 3.5, "decision": "extended"},
    ],
    "qualifying_progress_count": 9,
    "changed_file_count": 4,
}


class FakeClock:
    """Deterministic monotonic clock advanced explicitly by tests."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def budget(**overrides: object) -> RenewableBudget:
    """A valid policy with named overrides; invalid values are the point."""
    values: dict[str, float] = {
        "initial_seconds": 1.0,
        "extension_seconds": 1.0,
        "progress_window_seconds": 1.0,
        "max_seconds": 10.0,
    }
    for name, value in overrides.items():
        values[name] = cast(float, value)
    return RenewableBudget(**values)


def telemetry(**overrides: object) -> dict[str, object]:
    """Valid budget telemetry with named overrides; invalid values are the point."""
    record = dict(EXHAUSTED_TELEMETRY)
    for name, value in overrides.items():
        record[name] = value
    return record


def tool_start(call_id: str, tool_name: str, arguments: dict[str, object]) -> str:
    """The authentic start event: toolCallId, toolName, args."""
    return json.dumps(
        {
            "type": "tool_execution_start",
            "toolCallId": call_id,
            "toolName": tool_name,
            "args": arguments,
        }
    )


def tool_end(
    call_id: str,
    tool_name: str,
    result: dict[str, object],
    *,
    is_error: bool = False,
) -> str:
    """The authentic end event: no args, explicit isError, full result."""
    return json.dumps(
        {
            "type": "tool_execution_end",
            "toolCallId": call_id,
            "toolName": tool_name,
            "isError": is_error,
            "result": result,
        }
    )


def write_result(*paths: str) -> dict[str, object]:
    return {
        "content": [{"type": "text", "text": "wrote " + ", ".join(paths)}],
        "details": {"paths": list(paths)},
    }


def read_result(path: str) -> dict[str, object]:
    return {
        "content": [{"type": "text", "text": "body of " + path}],
        "details": {"paths": [path]},
    }


def assistant_text(text: str) -> str:
    """A token-bearing assistant event, which is never qualifying evidence."""
    return json.dumps(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "provider": "anthropic",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": text}],
            },
        }
    )


def harness_script(events: list[str], tail: str) -> str:
    """A stub-SDK stdout script: print each event line, then run the tail."""
    printed = [f"printf '%s\\n' {shlex.quote(event)}" for event in events]
    return "\n".join([*printed, tail, ""])


# Real subprocess decisions allow scheduling jitter, not process teardown time.
SCHEDULER_TOLERANCE_SECONDS = 0.5

IDLE_SCRIPT = """\
printf '%s\\n' 'warming up'
sleep 30 &
sleep 6
"""

CONTINUOUS_PROGRESS_SCRIPT = """\
i=0
while [ "$i" -lt 40 ]; do
  printf '{"type":"tool_execution_start","toolCallId":"w%s","toolName":"execution_edit","args":{"path":"/w/f%s.txt","oldText":"a%s","newText":"b%s"}}\\n' "$i" "$i" "$i" "$i"
  printf '{"type":"tool_execution_end","toolCallId":"w%s","toolName":"execution_edit","isError":false,"result":{"content":[{"type":"text","text":"edited"}],"details":{"paths":["/w/f%s.txt"]}}}\\n' "$i" "$i"
  i=$((i+1))
  sleep 0.15
done
"""


class RenewableBudgetValidationTests(unittest.TestCase):
    def test_a_valid_policy_constructs(self) -> None:
        policy = RenewableBudget(
            initial_seconds=3600.0,
            extension_seconds=1800.0,
            progress_window_seconds=900.0,
            max_seconds=14400.0,
        )
        self.assertEqual(policy.max_seconds, 14400.0)

    def test_invalid_values_are_refused(self) -> None:
        for field, value in (
            ("initial_seconds", 0.0),
            ("initial_seconds", -1.0),
            ("initial_seconds", float("inf")),
            ("initial_seconds", float("nan")),
            ("initial_seconds", True),
            ("initial_seconds", "60"),
            ("extension_seconds", 0),
            ("extension_seconds", float("nan")),
            ("progress_window_seconds", -5.0),
            ("progress_window_seconds", None),
            ("max_seconds", 0.0),
            ("max_seconds", float("inf")),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                budget(**{field: value})

    def test_values_above_the_absolute_maximum_are_refused(self) -> None:
        for field in ("initial_seconds", "extension_seconds", "progress_window_seconds"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                budget(**{field: 20.0})


class BudgetClockBoundaryTests(unittest.TestCase):
    def test_fresh_evidence_extends_once_and_is_then_consumed(self) -> None:
        clock = FakeClock()
        clock_budget = BudgetClock(RenewableBudget(60.0, 30.0, 900.0, 14400.0), clock=clock)
        self.assertEqual(clock_budget.remaining(), 60.0)
        clock.advance(50.0)
        clock_budget.progress(changed_files=2)
        clock.advance(10.0)
        self.assertEqual(clock_budget.remaining(), 30.0)
        self.assertEqual(
            clock_budget.stats()["decisions"],
            [{"at_seconds": 60.0, "evidence_age_seconds": 10.0, "decision": "extended"}],
        )
        clock.advance(30.0)
        self.assertEqual(clock_budget.remaining(), 0.0)
        stats = clock_budget.stats()
        self.assertEqual(stats["termination_reason"], "no_recent_progress")
        self.assertEqual(stats["qualifying_progress_count"], 1)
        self.assertEqual(stats["changed_file_count"], 2)
        self.assertEqual(validate_budget_stats(stats), stats)

    def test_no_progress_ever_still_records_its_final_refusal(self) -> None:
        clock = FakeClock()
        clock_budget = BudgetClock(RenewableBudget(10.0, 30.0, 900.0, 14400.0), clock=clock)
        clock.advance(10.0)
        self.assertEqual(clock_budget.remaining(), 0.0)
        self.assertEqual(
            clock_budget.stats()["decisions"],
            [
                {
                    "at_seconds": 10.0,
                    "evidence_age_seconds": None,
                    "decision": "no_recent_progress",
                }
            ],
        )
        self.assertEqual(clock_budget.stats()["qualifying_progress_count"], 0)
        self.assertEqual(clock_budget.termination_reason, "no_recent_progress")

    def test_the_absolute_cap_wins_over_fresh_evidence(self) -> None:
        clock = FakeClock()
        clock_budget = BudgetClock(RenewableBudget(10.0, 30.0, 40.0, 40.0), clock=clock)
        clock.advance(9.0)
        clock_budget.progress()
        clock.advance(31.0)
        self.assertEqual(clock_budget.remaining(), 0.0)
        self.assertEqual(
            clock_budget.stats()["decisions"],
            [
                {
                    "at_seconds": 40.0,
                    "evidence_age_seconds": 31.0,
                    "decision": "absolute_cap",
                }
            ],
        )
        self.assertEqual(clock_budget.termination_reason, "absolute_cap")

    def test_extensions_never_pass_the_absolute_cap(self) -> None:
        clock = FakeClock()
        clock_budget = BudgetClock(RenewableBudget(30.0, 30.0, 20.0, 50.0), clock=clock)
        clock.advance(29.0)
        clock_budget.progress()
        clock.advance(1.0)
        self.assertEqual(clock_budget.remaining(), 20.0)
        clock.advance(20.0)
        self.assertEqual(clock_budget.remaining(), 0.0)
        self.assertEqual(clock_budget.termination_reason, "absolute_cap")
        self.assertLessEqual(cast(float, clock_budget.stats()["duration_seconds"]), 50.0 + 1e-9)

    def test_late_progress_preserves_eligible_evidence_for_one_renewal(self) -> None:
        clock = FakeClock()
        clock.now = 0.0
        owner = BudgetClock(RenewableBudget(10, 10, 10, 30), clock=clock)
        clock.now = 9.0
        owner.progress()
        clock.now = 10.1
        owner.progress()
        self.assertAlmostEqual(owner.remaining(), 9.9)
        self.assertAlmostEqual(owner.remaining(), 9.9)
        stats = owner.stats()
        self.assertEqual(stats["qualifying_progress_count"], 2)
        decisions = cast(list[dict[str, object]], stats["decisions"])
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["decision"], "extended")
        self.assertEqual(decisions[0]["at_seconds"], 10.1)
        self.assertAlmostEqual(cast(float, decisions[0]["evidence_age_seconds"]), 1.1)

        clock.now = 20.0
        self.assertEqual(owner.remaining(), 0)
        self.assertEqual(owner.termination_reason, "no_recent_progress")
        self.assertEqual(
            owner.stats()["decisions"],
            [
                decisions[0],
                {
                    "at_seconds": 20.0,
                    "evidence_age_seconds": None,
                    "decision": "no_recent_progress",
                },
            ],
        )
        owner.progress()
        self.assertEqual(owner.remaining(), 0)
        self.assertEqual(owner.stats()["qualifying_progress_count"], 3)
        self.assertEqual(owner.termination_reason, "no_recent_progress")

    def test_evidence_after_the_deadline_cannot_resurrect_an_attempt(self) -> None:
        clock = FakeClock()
        owner = BudgetClock(RenewableBudget(10, 10, 10, 30), clock=clock)
        clock.advance(11)
        owner.progress()
        self.assertEqual(owner.remaining(), 0)
        self.assertEqual(owner.termination_reason, "no_recent_progress")
        owner.progress()
        self.assertEqual(owner.remaining(), 0)
        self.assertEqual(owner.termination_reason, "no_recent_progress")

    def test_evidence_older_than_the_window_does_not_renew(self) -> None:
        clock = FakeClock()
        clock_budget = BudgetClock(RenewableBudget(10.0, 30.0, 5.0, 14400.0), clock=clock)
        clock.advance(1.0)
        clock_budget.progress()
        clock.advance(9.0)
        self.assertEqual(clock_budget.remaining(), 0.0)
        self.assertEqual(clock_budget.termination_reason, "no_recent_progress")
        decisions = cast(list[dict[str, object]], clock_budget.stats()["decisions"])
        self.assertEqual(decisions[0]["evidence_age_seconds"], 9.0)

    def test_evidence_exactly_at_the_window_boundary_renews(self) -> None:
        clock = FakeClock()
        clock_budget = BudgetClock(RenewableBudget(10.0, 30.0, 5.0, 14400.0), clock=clock)
        clock.advance(5.0)
        clock_budget.progress()
        clock.advance(5.0)
        self.assertEqual(clock_budget.remaining(), 30.0)
        decisions = cast(list[dict[str, object]], clock_budget.stats()["decisions"])
        self.assertEqual(decisions[0]["decision"], "extended")
        self.assertEqual(decisions[0]["evidence_age_seconds"], 5.0)

    def test_decision_records_are_bounded_while_counters_keep_counting(self) -> None:
        clock = FakeClock()
        clock_budget = BudgetClock(RenewableBudget(1.0, 1.0, 1e9, 1e9), clock=clock)
        for _ in range(MAX_DECISION_RECORDS + 50):
            clock_budget.progress()
            clock.advance(1.0)
            self.assertGreater(clock_budget.remaining(), 0.0)
        self.assertEqual(
            len(cast(list[object], clock_budget.stats()["decisions"])),
            MAX_DECISION_RECORDS,
        )
        self.assertEqual(
            clock_budget.stats()["qualifying_progress_count"],
            MAX_DECISION_RECORDS + 50,
        )


class BudgetTelemetryValidationTests(unittest.TestCase):
    def test_valid_records_pass_and_the_known_fixtures_are_valid(self) -> None:
        for record in (EXHAUSTED_TELEMETRY, COMPLETED_TELEMETRY):
            self.assertEqual(validate_budget_stats(record), record)
        self.assertEqual(
            validate_budget_stats(
                telemetry(decisions=[], qualifying_progress_count=0, changed_file_count=0)
            )["termination_reason"],
            "no_recent_progress",
        )

    def test_structural_refusals(self) -> None:
        policy = cast(dict[str, object], EXHAUSTED_TELEMETRY["policy"])
        missing_schema = {
            key: value for key, value in EXHAUSTED_TELEMETRY.items() if key != "schema_version"
        }
        missing_decisions = {
            key: value for key, value in EXHAUSTED_TELEMETRY.items() if key != "decisions"
        }
        for label, record in (
            ("not-an-object", "fast"),
            ("none", None),
            ("list", []),
            ("wrong-schema", telemetry(schema_version=2)),
            ("float-schema", telemetry(schema_version=1.0)),
            ("missing-schema", missing_schema),
            ("policy-missing", telemetry(policy="small")),
            ("policy-not-positive", telemetry(policy={**policy, "initial_seconds": 0})),
            ("policy-not-ordered", telemetry(policy={**policy, "initial_seconds": 90.0})),
            ("policy-missing-field", telemetry(policy={"initial_seconds": 1.0})),
            ("duration-negative", telemetry(duration_seconds=-0.1)),
            ("duration-not-finite", telemetry(duration_seconds=float("nan"))),
            ("duration-typed", telemetry(duration_seconds="12.5")),
            ("unknown-reason", telemetry(termination_reason="vibes")),
            ("list-reason", telemetry(termination_reason=["completed"])),
            (
                "list-decision",
                telemetry(
                    decisions=[{"at_seconds": 1, "evidence_age_seconds": None, "decision": []}]
                ),
            ),
            (
                "beyond-duration",
                telemetry(
                    decisions=[
                        {"at_seconds": 13, "evidence_age_seconds": None, "decision": "extended"}
                    ]
                ),
            ),
            (
                "reverse-order",
                telemetry(
                    decisions=[
                        {"at_seconds": 2, "evidence_age_seconds": None, "decision": "extended"},
                        {"at_seconds": 1, "evidence_age_seconds": None, "decision": "extended"},
                    ]
                ),
            ),
            ("decisions-missing", missing_decisions),
            ("decisions-not-list", telemetry(decisions="extended")),
            ("decision-not-object", telemetry(decisions=["extended"])),
            (
                "decision-unknown-kind",
                telemetry(
                    decisions=[
                        {"at_seconds": 1.0, "evidence_age_seconds": None, "decision": "maybe"}
                    ]
                ),
            ),
            (
                "decision-missing-at",
                telemetry(decisions=[{"evidence_age_seconds": None, "decision": "extended"}]),
            ),
            (
                "decision-bad-age",
                telemetry(
                    decisions=[
                        {
                            "at_seconds": 1.0,
                            "evidence_age_seconds": "old",
                            "decision": "extended",
                        }
                    ]
                ),
            ),
            ("progress-float", telemetry(qualifying_progress_count=1.5)),
            ("progress-bool", telemetry(qualifying_progress_count=True)),
            ("progress-negative", telemetry(qualifying_progress_count=-1)),
            ("changed-negative", telemetry(changed_file_count=-2)),
            ("changed-float", telemetry(changed_file_count=2.5)),
            (
                "too-many-decisions",
                telemetry(
                    decisions=[
                        {"at_seconds": index, "evidence_age_seconds": None, "decision": "extended"}
                        for index in range(MAX_DECISION_RECORDS + 1)
                    ]
                ),
            ),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_budget_stats(record)


def harness_failed_result_json(**overrides: object) -> str:
    record: dict[str, object] = {
        "schema_version": WORKER_RESULT_SCHEMA,
        "model_call_id": "budget-roundtrip",
        "provider": "agy",
        "status": "harness_failed",
        "worker_version": "0.1.0",
        "worker_protocol_version": 2,
        "worker_source_sha256": "",
        "prompt_sha256": "",
        "ctx_version": "",
        "worker_cwd": "/tmp/snapshot",
        "started_at": 1.0,
        "completed_at": 13.5,
        "model_call_started": True,
        "session_id": "",
        "harness": {"exit_status": 124, "stdout": "partial output\n", "stderr": ""},
        "evidence": None,
        "failure": "harness-timeout",
        "harness_timeout_seconds": None,
        "harness_started_at": 1.0,
    }
    record.update(overrides)
    return json.dumps(record)


class WorkerBudgetTelemetryTests(unittest.TestCase):
    def test_results_without_telemetry_stay_valid_and_uninvented(self) -> None:
        result = WorkerResult.parse(harness_failed_result_json())
        self.assertIsNone(result.budget)
        reparsed = WorkerResult.parse(result.to_json())
        self.assertEqual(reparsed, result)
        self.assertIsNone(reparsed.budget)

    def test_telemetry_roundtrips_through_serialization(self) -> None:
        result = WorkerResult.parse(harness_failed_result_json(budget=EXHAUSTED_TELEMETRY))
        self.assertEqual(result.budget, EXHAUSTED_TELEMETRY)
        stored = WorkerResult.parse(result.to_json())
        self.assertEqual(stored, result)
        self.assertEqual(stored.budget, EXHAUSTED_TELEMETRY)

    def test_malformed_telemetry_fails_validation(self) -> None:
        for label, value in (
            ("wrong-schema", telemetry(schema_version=2)),
            ("unknown-reason", telemetry(termination_reason="vibes")),
            ("not-an-object", "fast"),
            ("explicit-null", None),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                WorkerResult.parse(harness_failed_result_json(budget=value))


class WorkerRenewablePreflightTests(unittest.TestCase):
    """A policy a route cannot honor is refused before any model launch."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def assert_preflight(self, result: WorkerResult, detail: str) -> None:
        self.assertEqual(result.status, "preflight_failed")
        self.assertEqual(result.failure, detail)
        self.assertFalse(result.model_call_started)
        self.assertIsNone(result.harness.exit_status)
        self.assertIsNone(result.budget)
        self.assertTrue((self.root / worker_evidence_path(result.model_call_id)).exists())

    def test_a_fixed_timeout_conflicts_with_a_renewable_budget(self) -> None:
        result = execute_worker(
            provider="omp",
            model_call_id="budget-preflight",
            command=WRITER_COMMAND,
            timeout=30.0,
            ctx_timeout=10.0,
            renewable_budget=RenewableBudget(0.5, 0.5, 30.0, 30.0),
            cwd=self.root,
        )
        self.assert_preflight(result, "renewable budget conflicts with fixed timeout")

    def test_a_non_budget_object_is_refused(self) -> None:
        result = execute_worker(
            provider="omp",
            model_call_id="budget-preflight",
            command=WRITER_COMMAND,
            timeout=None,
            ctx_timeout=10.0,
            renewable_budget=cast(RenewableBudget, {"initial_seconds": 1.0}),
            cwd=self.root,
        )
        self.assert_preflight(result, "renewable budget must be a RenewableBudget")

    def test_native_claude_is_refused(self) -> None:
        result = execute_worker(
            provider="claude",
            model_call_id="budget-preflight",
            command=["claude", "-p", "hi"],
            timeout=None,
            ctx_timeout=10.0,
            renewable_budget=RenewableBudget(0.5, 0.5, 30.0, 30.0),
            cwd=self.root,
        )
        self.assert_preflight(
            result,
            "renewable budgets require provider omp with the workspace-write-no-shell policy",
        )

    def test_agy_is_refused(self) -> None:
        result = execute_worker(
            provider="agy",
            model_call_id="budget-preflight",
            command=["agy", "-p", "hi"],
            timeout=None,
            ctx_timeout=10.0,
            renewable_budget=RenewableBudget(0.5, 0.5, 30.0, 30.0),
            cwd=self.root,
        )
        self.assert_preflight(
            result,
            "renewable budgets require provider omp with the workspace-write-no-shell policy",
        )

    def test_packet_omp_is_refused(self) -> None:
        result = execute_worker(
            provider="omp-packet",
            model_call_id="budget-preflight",
            command=["omp", "-p", "hi"],
            timeout=None,
            ctx_timeout=10.0,
            renewable_budget=RenewableBudget(0.5, 0.5, 30.0, 30.0),
            cwd=self.root,
        )
        self.assert_preflight(
            result,
            "renewable budgets require provider omp with the workspace-write-no-shell policy",
        )

    def test_read_only_omp_is_refused_before_the_sdk_check(self) -> None:
        command = [
            "omp",
            "-p",
            "hi",
            "--mode",
            "json",
            "--cwd",
            ".",
            "--model",
            "anthropic/claude-opus-5-5",
            "--execution-tool-policy",
            "read-only-no-shell",
        ]
        with mock.patch(
            "agent_execution.worker.require_omp_sdk",
            return_value=(self.root / "sdk", str(self.root / "bun")),
        ):
            result = execute_worker(
                provider="omp",
                model_call_id="budget-preflight",
                command=command,
                timeout=None,
                ctx_timeout=10.0,
                renewable_budget=RenewableBudget(0.5, 0.5, 30.0, 30.0),
                cwd=self.root,
            )
        self.assert_preflight(result, "renewable budgets require OMP workspace-write-no-shell")


class WorkerRunnerRefusalTests(unittest.TestCase):
    def test_custom_runner_cannot_be_bypassed_by_budget_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            custom = mock.Mock(side_effect=AssertionError("unexpected model launch"))
            result = execute_worker(
                provider="omp",
                model_call_id="runner-refusal",
                command=WRITER_COMMAND,
                timeout=None,
                ctx_timeout=10.0,
                renewable_budget=RenewableBudget(1, 1, 1, 10),
                invoke=custom,
                cwd=root,
            )
            self.assertEqual(result.status, "preflight_failed")
            self.assertFalse(result.model_call_started)
            custom.assert_not_called()
            self.assertEqual(
                WorkerResult.parse((root / worker_evidence_path("runner-refusal")).read_text()),
                result,
            )

    def test_custom_prompt_runner_refuses_before_model_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt.txt").write_text("Edit café", encoding="utf-8")
            custom = mock.Mock(side_effect=AssertionError("unexpected model launch"))
            with mock.patch.dict(os.environ, {"WEFT_PAYLOAD_DIR": str(root)}):
                result = execute_worker(
                    provider="omp",
                    model_call_id="prompt-runner-refusal",
                    command=STDIN_COMMAND,
                    timeout=None,
                    ctx_timeout=10.0,
                    renewable_budget=RenewableBudget(1, 1, 1, 10),
                    prompt_payload="prompt.txt",
                    invoke_prompt=custom,
                    cwd=root,
                )
            self.assertEqual(result.status, "preflight_failed")
            self.assertFalse(result.model_call_started)
            custom.assert_not_called()


class DrainLineDeliveryTests(unittest.TestCase):
    def test_newlines_do_not_invent_an_empty_tail_at_eof(self) -> None:
        for data, expected in (
            ("", []),
            ("\n", [""]),
            ("line\n", ["line"]),
            ("\n\n", ["", ""]),
            ("line\n\ntail", ["line", "", "tail"]),
            ("tail", ["tail"]),
        ):
            for read_size in (1, processes.DRAIN_READ_CHARS):
                with (
                    self.subTest(data=data, read_size=read_size),
                    mock.patch.object(processes, "DRAIN_READ_CHARS", read_size),
                ):
                    delivered: list[str] = []
                    sink: list[str] = []
                    processes._drain(io.StringIO(data), sink, delivered.append)
                    self.assertEqual(delivered, expected)
                    self.assertEqual("".join(sink), data)

    def test_a_short_flushed_line_is_delivered_before_eof(self) -> None:
        read_fd, write_fd = os.pipe()
        stream = io.TextIOWrapper(os.fdopen(read_fd, "rb"), encoding="utf-8")
        large = json.dumps(
            {
                "type": "tool_execution_start",
                "toolCallId": "big",
                "toolName": "execution_write",
                "args": {"path": "/w/café.txt", "content": "x" * 200_000},
            }
        )
        delivered: list[str] = []
        sink: list[str] = []
        reader = threading.Thread(
            target=processes._drain,
            args=(stream, sink, delivered.append),
            daemon=True,
        )
        reader.start()
        writer = os.fdopen(write_fd, "w", encoding="utf-8")
        try:
            writer.write(large[:100])
            writer.flush()
            writer.write(large[100:] + "\n")
            writer.flush()
            self._await(lambda: len(delivered) >= 1)
            # The writer stays open here: a read-exact-size drain would
            # withhold the short line until EOF, which is the failure mode.
            writer.write('{"short":true}\n')
            writer.flush()
            self._await(lambda: len(delivered) >= 2)
        finally:
            writer.close()
        reader.join(10.0)
        self.assertEqual(delivered, [large, '{"short":true}'])
        self.assertEqual("".join(sink), large + "\n" + '{"short":true}\n')

    def test_a_final_line_without_a_newline_is_delivered_at_eof(self) -> None:
        delivered: list[str] = []
        sink: list[str] = []
        processes._drain(io.StringIO("alpha\nbeta\ngamma"), sink, delivered.append)
        self.assertEqual(delivered, ["alpha", "beta", "gamma"])
        self.assertEqual("".join(sink), "alpha\nbeta\ngamma")

    def test_oversized_line_discards_json_suffix_but_keeps_next_line_and_eof(self) -> None:
        delivered: list[str] = []
        sink: list[str] = []
        data = 'xxxxxx{"type":"tool_execution_end"}\nvalid\nlast'
        with (
            mock.patch.object(processes, "MAX_OBSERVED_LINE_CHARS", 10),
            mock.patch.object(processes, "DRAIN_READ_CHARS", 6),
        ):
            processes._drain(io.StringIO(data), sink, delivered.append)
        self.assertEqual(delivered, ["valid", "last"])
        self.assertEqual("".join(sink), data)

    def test_split_utf8_and_newlines_preserve_decoded_output(self) -> None:
        read_fd, write_fd = os.pipe()
        stream = io.TextIOWrapper(os.fdopen(read_fd, "rb"), encoding="utf-8")
        delivered: list[str] = []
        sink: list[str] = []
        reader = threading.Thread(target=processes._drain, args=(stream, sink, delivered.append))
        reader.start()
        try:
            os.write(write_fd, b"caf\xc3")
            os.write(write_fd, b"\xa9\r\nnext")
        finally:
            os.close(write_fd)
        reader.join(10)
        self.assertEqual(delivered, ["café", "next"])
        self.assertEqual("".join(sink), "café\nnext")

    def _await(self, condition: Callable[[], bool]) -> None:
        deadline = time.monotonic() + 10.0
        while not condition():
            if time.monotonic() > deadline:
                self.fail("drain never delivered the line")
            time.sleep(0.005)


class OmpRenewableSubprocessTests(unittest.TestCase):
    """Real children, the real observer, and the real process owner."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        package = (
            self.root / "sdk" / "node_modules" / "@oh-my-pi" / "pi-coding-agent" / "package.json"
        )
        package.parent.mkdir(parents=True)
        package.write_text(json.dumps({"version": OMP_SDK_VERSION}))
        processes.take_telemetry()
        path = f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}"
        patcher = mock.patch.dict(
            os.environ,
            {"PATH": path, "AGENT_EXECUTION_OMP_SDK_ROOT": str(self.root / "sdk")},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_harness(self, body: str) -> None:
        bun = self.bin / "bun"
        bun.write_text("#!/bin/sh\n" + body)
        bun.chmod(0o755)

    def run_budgeted(
        self,
        events: list[str],
        *,
        tail: str,
        budget_policy: RenewableBudget,
    ) -> BudgetedCompletedProcess:
        self.write_harness(harness_script(events, tail))
        return run_omp_command(WRITER_COMMAND, self.snapshot, None, renewable_budget=budget_policy)

    def assert_group_cleaned_up(self) -> None:
        telemetry = processes.take_telemetry()
        self.assertIsNotNone(telemetry)
        assert telemetry is not None
        self.assertIn(telemetry.kill_kind, {"group-term", "group-kill"})
        self.assertTrue(telemetry.survivors_reaped)
        self.assertEqual(census(pgid=telemetry.pgid or -1), ())

    def test_qualifying_write_work_extends_past_the_initial_deadline(self) -> None:
        events = [
            tool_start("w1", "execution_write", {"path": "/w/file.txt", "content": "body"}),
            tool_end("w1", "execution_write", write_result("/w/file.txt")),
        ]
        with self.assertRaises(BudgetTimeoutExpired) as raised:
            self.run_budgeted(
                events,
                tail="sleep 6",
                budget_policy=RenewableBudget(0.5, 0.6, 30.0, 30.0),
            )
        error = raised.exception
        self.assertIsInstance(error, subprocess.TimeoutExpired)
        self.assertIn("tool_execution_start", error.output or "")
        stats = error.budget
        assert stats is not None
        self.assertEqual(validate_budget_stats(stats), stats)
        self.assertEqual(stats["qualifying_progress_count"], 1)
        self.assertEqual(stats["changed_file_count"], 1)
        self.assertEqual(stats["termination_reason"], "no_recent_progress")
        self.assertEqual(
            cast(list[object], stats["decisions"]),
            [
                {"at_seconds": ANY, "evidence_age_seconds": ANY, "decision": "extended"},
                {
                    "at_seconds": ANY,
                    "evidence_age_seconds": ANY,
                    "decision": "no_recent_progress",
                },
            ],
        )
        duration = cast(float, stats["duration_seconds"])
        self.assertGreaterEqual(duration, 0.9)
        self.assertLessEqual(duration, 4.0)
        self.assert_group_cleaned_up()

    def test_an_idle_attempt_stops_at_the_initial_deadline(self) -> None:
        self.write_harness(IDLE_SCRIPT)
        with self.assertRaises(BudgetTimeoutExpired) as raised:
            run_omp_command(WRITER_COMMAND, self.snapshot, None, renewable_budget=budget())
        error = raised.exception
        self.assertIn("warming up", error.output or "")
        stats = error.budget
        assert stats is not None
        self.assertEqual(stats["qualifying_progress_count"], 0)
        self.assertEqual(stats["changed_file_count"], 0)
        self.assertEqual(stats["termination_reason"], "no_recent_progress")
        self.assertEqual(
            cast(list[object], stats["decisions"]),
            [
                {
                    "at_seconds": ANY,
                    "evidence_age_seconds": None,
                    "decision": "no_recent_progress",
                }
            ],
        )
        decisions = cast(list[dict[str, object]], stats["decisions"])
        decided_at = cast(float, decisions[0]["at_seconds"])
        self.assertGreaterEqual(decided_at, 1.0)
        self.assertLessEqual(decided_at, 1.0 + SCHEDULER_TOLERANCE_SECONDS)
        duration = cast(float, stats["duration_seconds"])
        self.assertGreaterEqual(duration, decided_at)
        self.assertLessEqual(
            duration,
            1.0
            + processes.GRACE_SECONDS
            + processes.SETTLE_SECONDS
            + 2 * processes.DRAIN_SECONDS
            + SCHEDULER_TOLERANCE_SECONDS,
        )
        self.assert_group_cleaned_up()

    def test_continuous_progress_still_stops_at_the_absolute_cap(self) -> None:
        self.write_harness(CONTINUOUS_PROGRESS_SCRIPT)
        with self.assertRaises(BudgetTimeoutExpired) as raised:
            run_omp_command(
                WRITER_COMMAND,
                self.snapshot,
                None,
                renewable_budget=RenewableBudget(0.4, 0.4, 1.1, 1.2),
            )
        stats = raised.exception.budget
        assert stats is not None
        self.assertEqual(validate_budget_stats(stats), stats)
        self.assertEqual(stats["termination_reason"], "absolute_cap")
        decisions = cast(list[dict[str, object]], stats["decisions"])
        self.assertEqual(decisions[-1]["decision"], "absolute_cap")
        self.assertGreaterEqual(
            sum(1 for decision in decisions if decision["decision"] == "extended"), 1
        )
        self.assertGreaterEqual(cast(int, stats["qualifying_progress_count"]), 1)
        self.assertGreaterEqual(cast(int, stats["changed_file_count"]), 1)
        duration = cast(float, stats["duration_seconds"])
        self.assertGreaterEqual(duration, 1.1)
        self.assertLessEqual(duration, 5.0)
        self.assert_group_cleaned_up()

    def test_duplicate_completions_cannot_renew_twice(self) -> None:
        start = tool_start("w1", "execution_write", {"path": "/w/file.txt", "content": "body"})
        end = tool_end("w1", "execution_write", write_result("/w/file.txt"))
        with self.assertRaises(BudgetTimeoutExpired) as raised:
            self.run_budgeted(
                [start, end, start, end],
                tail="sleep 5",
                budget_policy=RenewableBudget(0.6, 0.6, 30.0, 30.0),
            )
        error = raised.exception
        self.assertEqual(error.output, f"{start}\n{end}\n{start}\n{end}\n")
        stats = error.budget
        assert stats is not None
        self.assertEqual(stats["qualifying_progress_count"], 1)
        self.assertEqual(stats["changed_file_count"], 1)
        self.assertEqual(stats["termination_reason"], "no_recent_progress")
        decisions = cast(list[dict[str, object]], stats["decisions"])
        self.assertEqual(
            [decision["decision"] for decision in decisions],
            ["extended", "no_recent_progress"],
        )
        for decision, deadline in zip(decisions, (0.6, 1.2), strict=True):
            decided_at = cast(float, decision["at_seconds"])
            self.assertGreaterEqual(decided_at, deadline)
            self.assertLessEqual(decided_at, deadline + SCHEDULER_TOLERANCE_SECONDS)
        duration = cast(float, stats["duration_seconds"])
        self.assertGreaterEqual(duration, cast(float, decisions[-1]["at_seconds"]))
        self.assertLessEqual(
            duration,
            1.2
            + processes.GRACE_SECONDS
            + processes.SETTLE_SECONDS
            + 2 * processes.DRAIN_SECONDS
            + SCHEDULER_TOLERANCE_SECONDS,
        )
        self.assert_group_cleaned_up()

    def test_failures_orphans_and_untracked_tools_never_qualify(self) -> None:
        events = [
            assistant_text("I will now attempt the edit."),
            tool_start("f1", "execution_write", {"path": "/w/x.txt", "content": "nope"}),
            tool_end("f1", "execution_write", write_result("/w/x.txt"), is_error=True),
            tool_start("f1", "execution_write", {"path": "/w/x.txt", "content": "nope"}),
            tool_end("f1", "execution_write", write_result("/w/x.txt")),
            tool_end("w9", "execution_write", write_result("/w/y.txt")),
            tool_start("s1", "shell", {"command": "echo hi"}),
            tool_end("s1", "shell", {"content": [{"type": "text", "text": "hi"}]}),
            tool_start("no-marker", "execution_edit", {"path": "/w/no.txt"}),
            json.dumps(
                {
                    "type": "tool_execution_end",
                    "toolCallId": "no-marker",
                    "toolName": "execution_edit",
                    "result": write_result("/w/no.txt"),
                }
            ),
            tool_start("missing-result", "execution_write", {"path": "/w/empty.txt"}),
            tool_end("missing-result", "execution_write", {}),
        ]
        with self.assertRaises(BudgetTimeoutExpired) as raised:
            self.run_budgeted(events, tail="sleep 5", budget_policy=budget())
        error = raised.exception
        self.assertIn("I will now attempt the edit", error.output or "")
        stats = error.budget
        assert stats is not None
        self.assertEqual(stats["qualifying_progress_count"], 0)
        self.assertEqual(stats["changed_file_count"], 0)
        self.assertEqual(stats["termination_reason"], "no_recent_progress")
        self.assertEqual(
            cast(list[object], stats["decisions"]),
            [
                {
                    "at_seconds": ANY,
                    "evidence_age_seconds": None,
                    "decision": "no_recent_progress",
                }
            ],
        )

    def test_identical_repeated_reads_qualify_once_and_then_stop(self) -> None:
        events = [
            tool_start("r1", "execution_read", {"path": "/src/main.py"}),
            tool_end("r1", "execution_read", read_result("/src/main.py")),
            tool_start("r2", "execution_read", {"path": "/src/main.py"}),
            tool_end("r2", "execution_read", read_result("/src/main.py")),
        ]
        with self.assertRaises(BudgetTimeoutExpired) as raised:
            self.run_budgeted(
                events,
                tail="sleep 5",
                budget_policy=RenewableBudget(0.4, 0.5, 30.0, 30.0),
            )
        stats = raised.exception.budget
        assert stats is not None
        self.assertEqual(stats["qualifying_progress_count"], 1)
        self.assertEqual(stats["changed_file_count"], 0)
        decisions = cast(list[dict[str, object]], stats["decisions"])
        self.assertEqual(
            [decision["decision"] for decision in decisions],
            ["extended", "no_recent_progress"],
        )

    def test_multiple_successful_edits_count_one_changed_file(self) -> None:
        events = [
            tool_start(
                "w1", "execution_edit", {"path": "/w/file.txt", "oldText": "a", "newText": "b"}
            ),
            tool_end("w1", "execution_edit", write_result("/w/file.txt")),
            tool_start(
                "w2", "execution_edit", {"path": "/w/file.txt", "oldText": "b", "newText": "c"}
            ),
            tool_end("w2", "execution_edit", write_result("/w/file.txt")),
        ]
        completed = self.run_budgeted(
            events, tail="exit 0", budget_policy=RenewableBudget(5, 1, 1, 6)
        )
        assert completed.budget is not None
        self.assertEqual(completed.budget["qualifying_progress_count"], 2)
        self.assertEqual(completed.budget["changed_file_count"], 1)

    def test_a_successful_exit_returns_validated_completed_telemetry(self) -> None:
        events = [
            tool_start("w1", "execution_write", {"path": "/w/file.txt", "content": "body"}),
            tool_end("w1", "execution_write", write_result("/w/file.txt")),
        ]
        completed = self.run_budgeted(
            events,
            tail="exit 0",
            budget_policy=RenewableBudget(5.0, 5.0, 30.0, 30.0),
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIsInstance(completed, BudgetedCompletedProcess)
        stats = completed.budget
        assert stats is not None
        self.assertEqual(validate_budget_stats(stats), stats)
        self.assertEqual(stats["termination_reason"], "completed")
        self.assertEqual(stats["decisions"], [])
        self.assertEqual(stats["qualifying_progress_count"], 1)
        self.assertEqual(stats["changed_file_count"], 1)
        self.assertIn("tool_execution_start", completed.stdout)
        self.assertIn("tool_execution_end", completed.stdout)

    def test_the_stdin_prompt_delivery_path_runs_under_a_budget(self) -> None:
        self.write_harness(
            harness_script(
                [
                    tool_start(
                        "w1",
                        "execution_edit",
                        {"path": "/w/a.txt", "oldText": "a", "newText": "b"},
                    ),
                    tool_end("w1", "execution_edit", write_result("/w/a.txt")),
                ],
                "exit 0",
            )
        )
        result = run_command_with_prompt(
            STDIN_COMMAND,
            self.snapshot,
            "make the edit",
            None,
            renewable_budget=RenewableBudget(5.0, 5.0, 30.0, 30.0),
        )
        self.assertEqual(result.exit_status, 0)
        self.assertIsNotNone(result.budget)
        stats = result.budget
        assert stats is not None
        self.assertEqual(stats["termination_reason"], "completed")
        self.assertEqual(stats["qualifying_progress_count"], 1)
        self.assertEqual(stats["changed_file_count"], 1)
        self.assertIn("tool_execution_start", result.stdout)

    def test_execute_worker_delivers_non_ascii_payload_to_default_budgeted_runner(self) -> None:
        prompt = "Edit café and 東京\n"
        payload = self.root / "payload"
        payload.mkdir()
        (payload / "prompt.txt").write_bytes(prompt.encode("utf-8"))
        # The SDK wrapper executes this child; only the installed SDK and
        # remote authentication probe are replaced, not worker dispatch.
        program = (
            "import sys; data=sys.stdin.buffer.read(); "
            f"assert data == {prompt.encode('utf-8')!r}, data; "
            'print(\'{"type":"unrecognized"}\', flush=True)'
        )
        self.write_harness(f"exec {shlex.quote(sys.executable)} -c {shlex.quote(program)}\n")
        fact = {"state": "available", "stale": False, "detail": ""}
        status = {
            "rows": [
                {
                    "subject": {},
                    "facts": {
                        name: dict(fact)
                        for name in ("capability", "authentication", "transport", "quota")
                    },
                }
            ]
        }
        with (
            mock.patch.dict(os.environ, {"WEFT_PAYLOAD_DIR": str(payload)}),
            mock.patch("agent_execution.execution_status.probe_status", return_value=status),
        ):
            result = execute_worker(
                provider="omp",
                model_call_id="real-stdin",
                command=STDIN_COMMAND,
                timeout=None,
                ctx_timeout=10,
                renewable_budget=RenewableBudget(3, 1, 2, 5),
                prompt_payload="prompt.txt",
                cwd=self.snapshot,
            )
        self.assertEqual(result.status, "evidence_failed")
        self.assertEqual(result.harness.exit_status, 0)
        self.assertIn("unrecognized", result.harness.stdout)
        self.assertIsNotNone(result.budget)
        self.assertEqual(
            WorkerResult.parse((self.snapshot / worker_evidence_path("real-stdin")).read_text()),
            result,
        )

    def test_hard_deadlines_are_unchanged_without_a_policy(self) -> None:
        program = "import time; print('working', flush=True); time.sleep(5)"
        with self.assertRaises(subprocess.TimeoutExpired) as raised:
            run_in_process_group([sys.executable, "-c", program], self.snapshot, "", 0.3)
        error = raised.exception
        self.assertNotIsInstance(error, BudgetTimeoutExpired)
        self.assertIn("working", error.output or "")

    def test_omp_without_a_policy_keeps_its_plain_hard_timeout(self) -> None:
        self.write_harness("sleep 5\n")
        with self.assertRaises(subprocess.TimeoutExpired) as raised:
            run_omp_command(WRITER_COMMAND, self.snapshot, 0.3)
        self.assertNotIsInstance(raised.exception, BudgetTimeoutExpired)

    def test_successful_runs_without_a_policy_carry_no_telemetry(self) -> None:
        self.write_harness("exit 0\n")
        completed = run_omp_command(WRITER_COMMAND, self.snapshot, 5.0)
        self.assertIsInstance(completed, BudgetedCompletedProcess)
        self.assertIsNone(completed.budget)

    def test_the_owner_refuses_a_budget_alongside_a_fixed_timeout(self) -> None:
        clock = BudgetClock(RenewableBudget(1.0, 1.0, 1.0, 10.0))
        with self.assertRaises(ValueError):
            run_in_process_group(
                [sys.executable, "-c", "pass"],
                self.snapshot,
                "",
                5.0,
                budget_clock=clock,
            )

    def test_the_owner_refuses_an_untyped_budget_clock(self) -> None:
        with self.assertRaises(ValueError):
            run_in_process_group(
                [sys.executable, "-c", "pass"],
                self.snapshot,
                "",
                None,
                budget_clock=cast(BudgetClock, object()),
            )
