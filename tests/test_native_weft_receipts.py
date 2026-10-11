"""Retained native results are judged against the issued command's identities."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agent_execution.command import CommandResult
from agent_execution.weft import WeftCommandRunner, WeftRetrievalOutcome
from agent_execution.worker import (
    WORKER_PROTOCOL_VERSION,
    HarnessOutcome,
    WorkerEvidence,
    WorkerResult,
)
from tests.test_claude_worker import MODEL, SESSION, command, envelope
from tests.test_weft import KEY, SOURCE, receipt


class NativeWeftReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.retained_command: str | None = None
        self.command_available = True
        self.artifact = self.worker_artifact()

    @staticmethod
    def worker_artifact(
        *, updates: dict[str, object] | None = None, omit_usage: bool = False
    ) -> str:
        body = json.loads(envelope(**(updates or {})))
        if omit_usage:
            body.pop("modelUsage")
        transcript = {
            "schema_version": 1,
            "target": "session",
            "payload_type": "session_transcript",
            "provider": "claude",
            "provider_session_id": SESSION,
            "mode": "log",
            "format": "json",
            "events": [
                {
                    "schema_version": 1,
                    "provider": "claude",
                    "provider_session_id": SESSION,
                    "event_type": "tool_call",
                    "occurred_at": "2026-10-09T00:00:00.000Z",
                    "content": {"complete": True},
                    "activity": {"facts": [{"kind": "session_cwd", "value": "/remote/project"}]},
                    "text": json.dumps(
                        {"name": "Read", "input": {"file_path": "/remote/project/review.md"}}
                    ),
                }
            ],
        }
        return WorkerResult(
            model_call_id=KEY,
            provider="claude",
            status="completed",
            worker_version="0.1.0",
            worker_protocol_version=WORKER_PROTOCOL_VERSION,
            worker_source_sha256=SOURCE,
            prompt_sha256=hashlib.sha256(b"review").hexdigest(),
            ctx_version="ctx-1",
            worker_cwd="/remote/project",
            started_at=10,
            completed_at=12,
            model_call_started=True,
            session_id=SESSION,
            harness=HarnessOutcome(0, json.dumps(body), ""),
            evidence=WorkerEvidence(
                import_receipt={"schema_version": 2, "outcome": "success"},
                transcript=transcript,
            ),
        ).to_json()

    def invoke(self, argv: list[str], cwd: Path, timeout: float | None) -> CommandResult:
        if argv[1] == "host":
            return CommandResult(
                0,
                json.dumps(
                    {
                        "kind": "host_list",
                        "version": 1,
                        "hosts": [
                            {
                                "name": "studio",
                                "capabilities": ["agent:claude", "tool:agent-execution"],
                            }
                        ],
                    }
                ),
                "",
            )
        if argv[1] == "run":
            self.retained_command = argv[-1]
            return CommandResult(0, receipt(), "")
        if argv[1] == "status":
            return CommandResult(0, "completed", "")
        if argv[1:3] == ["job", "inspect"]:
            record = {"id": "wj42", "status": "completed"}
            if self.command_available:
                assert self.retained_command is not None
                record["command"] = self.retained_command
            return CommandResult(0, json.dumps(record), "")
        if argv[1] == "artifact":
            return CommandResult(0, self.artifact, "")
        if argv[1:3] == ["job", "mark-processed"] or argv[1] == "log":
            return CommandResult(0, "", "")
        raise AssertionError(argv)

    def runner(self) -> WeftCommandRunner:
        return WeftCommandRunner(
            host="studio",
            agent="claude",
            model_call_id=KEY,
            fallback=None,
            invoke=self.invoke,
            expected_source_sha256=SOURCE,
        )

    def dispatch(self) -> dict[str, object]:
        self.artifact = self.worker_artifact()
        result = self.runner()(command(), self.root, 30.0)
        self.assertEqual(result.exit_status, 0, result.stderr)
        assert result.execution is not None
        return copy.deepcopy(result.execution)

    def test_absent_dispatch_pins_are_recovered_and_the_result_can_be_consumed(self) -> None:
        for missing in (
            ("claude_model",),
            ("claude_session_id",),
            ("claude_model", "claude_session_id"),
        ):
            with self.subTest(missing=missing):
                execution = self.dispatch()
                for field in missing:
                    execution.pop(field)
                result = self.runner().retrieve(job_id="wj42", cwd=self.root, execution=execution)
                assert isinstance(result, CommandResult), result
                self.assertEqual(result.exit_status, 0, result.stderr)
                self.assertEqual(execution["claude_model"], MODEL)
                self.assertEqual(execution["claude_session_id"], SESSION)
                result.mark_consumed()
                self.assertEqual(execution["processed"], True)

    def test_foreign_or_absent_result_identities_are_terminal_not_consumed(self) -> None:
        cases: tuple[tuple[dict[str, object], bool], ...] = (
            ({"session_id": "790268ee-8af8-4ddb-9dcd-db05429dd832"}, False),
            ({"session_id": None}, False),
            ({"modelUsage": {"claude-sonnet-4-5": {}}}, False),
            ({}, True),
        )
        for updates, omit_usage in cases:
            with self.subTest(updates=updates, omit_usage=omit_usage):
                execution = self.dispatch()
                execution.pop("claude_model")
                execution.pop("claude_session_id")
                self.artifact = self.worker_artifact(updates=updates, omit_usage=omit_usage)
                result = self.runner().retrieve(job_id="wj42", cwd=self.root, execution=execution)
                assert isinstance(result, CommandResult), result
                self.assertEqual(result.exit_status, 1)
                self.assertEqual(execution["claude_model"], MODEL)
                self.assertEqual(execution["claude_session_id"], SESSION)
                processing = execution["processing"]
                assert isinstance(processing, dict)
                self.assertEqual(processing["state"], "not_marked")
                self.assertEqual(processing["step"], "worker_result_validation")

    def test_missing_issued_command_remains_unknown_without_result_certification(self) -> None:
        execution = self.dispatch()
        execution.pop("claude_model")
        execution.pop("claude_session_id")
        self.command_available = False
        result = self.runner().retrieve(job_id="wj42", cwd=self.root, execution=execution)
        assert isinstance(result, WeftRetrievalOutcome), result
        self.assertEqual(result.status, "unknown")
        self.assertNotIn("processed", execution)

    def test_a_conflicting_retained_pin_is_not_replaced_by_the_command(self) -> None:
        execution = self.dispatch()
        execution.pop("claude_model")
        foreign = "790268ee-8af8-4ddb-9dcd-db05429dd832"
        execution["claude_session_id"] = foreign
        result = self.runner().retrieve(job_id="wj42", cwd=self.root, execution=execution)
        assert isinstance(result, WeftRetrievalOutcome), result
        self.assertEqual(result.status, "unknown")
        self.assertEqual(execution["claude_session_id"], foreign)
        self.assertNotIn("processed", execution)

    def test_offline_artifact_cannot_supply_its_own_missing_dispatch_expectation(self) -> None:
        execution = self.dispatch()
        execution.pop("claude_model")
        artifact = self.root / "retained.json"
        artifact.write_text(self.worker_artifact(updates={"modelUsage": {"claude-sonnet-4-5": {}}}))
        result = self.runner().retrieve_from_file(
            job_id="wj42",
            cwd=self.root,
            execution=execution,
            artifact_path=artifact,
        )
        assert isinstance(result, WeftRetrievalOutcome), result
        self.assertEqual(result.status, "unknown")
        self.assertNotIn("processed", execution)
        self.assertNotIn("claude_model", execution)
