"""Native evidence recovery never depends on a retained source directory."""

from __future__ import annotations

import json
import subprocess
import unittest
from dataclasses import replace
from unittest import mock

from agent_execution.native_evidence import recover_native_evidence
from agent_execution.worker import WORKER_PROTOCOL_VERSION, HarnessOutcome, WorkerResult

SESSION = "d93bfcf1-73f6-4692-af1c-1729e37e06a3"
SOURCE = "a" * 64


def original() -> WorkerResult:
    return WorkerResult(
        model_call_id="retained-native-call",
        provider="claude",
        status="evidence_failed",
        worker_version="test",
        worker_protocol_version=WORKER_PROTOCOL_VERSION,
        worker_source_sha256=SOURCE,
        prompt_sha256="b" * 64,
        ctx_version="ctx test",
        worker_cwd="/removed/source/tree",
        started_at=10.0,
        completed_at=20.0,
        model_call_started=True,
        session_id=SESSION,
        harness=HarnessOutcome(
            0,
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "retained answer",
                    "session_id": SESSION,
                    "modelUsage": {"claude-opus-5-5": {"inputTokens": 1, "outputTokens": 1}},
                }
            ),
            "",
        ),
        failure="ctx export unavailable",
    )


def transcript() -> dict[str, object]:
    return {
        "schema_version": 1,
        "target": "session",
        "payload_type": "session_transcript",
        "provider": "claude",
        "provider_session_id": SESSION,
        "mode": "log",
        "format": "json",
        "events": [
            {
                "provider": "claude",
                "provider_session_id": SESSION,
                "event_type": "tool_call",
                "occurred_at": "1970-01-01T00:00:11.000Z",
                "content": {"complete": True},
                "activity": {
                    "revision": 1,
                    "invocation": {
                        "tool": "Read",
                        "arguments": {
                            "capture_status": "present",
                            "value": {"file_path": "/removed/source/tree/plan.md"},
                        },
                    },
                    "facts": [{"kind": "session_cwd", "value": "/removed/source/tree"}],
                },
            }
        ],
    }


class NativeEvidenceTests(unittest.TestCase):
    def recover(self, worker=None):
        return recover_native_evidence(
            worker or original(),
            executor="registered.example",
            account="reviewer",
            expected_protocol_version=WORKER_PROTOCOL_VERSION,
            expected_source_sha256=SOURCE,
        )

    def test_cleaned_source_is_not_a_remote_command_precondition(self):
        worker = original()
        before = worker.to_json()
        with mock.patch(
            "agent_execution.native_evidence.subprocess.run",
            side_effect=[
                subprocess.CompletedProcess([], 0, "{}", ""),
                subprocess.CompletedProcess([], 0, json.dumps(transcript()), ""),
            ],
        ) as run:
            recovered = self.recover(worker)
        self.assertEqual(worker.to_json(), before)
        self.assertEqual(recovered.session_id, SESSION)
        for call in run.call_args_list:
            argv = call.args[0]
            self.assertEqual(argv[-2], "reviewer@registered.example")
            self.assertNotIn("cd --", argv[-1])
            self.assertNotIn(worker.worker_cwd, argv[-1])
        self.assertIn("ctx import --provider claude", run.call_args_list[0].args[0][-1])
        self.assertIn("--provider-session " + SESSION, run.call_args_list[1].args[0][-1])

    def test_transcript_identity_and_cwd_remain_mandatory(self):
        for field, value in (
            ("provider", "codex"),
            ("provider_session_id", "foreign"),
            ("events", []),
        ):
            with self.subTest(field=field):
                payload = transcript()
                payload[field] = value
                with (
                    mock.patch(
                        "agent_execution.native_evidence.subprocess.run",
                        side_effect=[
                            subprocess.CompletedProcess([], 0, "{}", ""),
                            subprocess.CompletedProcess([], 0, json.dumps(payload), ""),
                        ],
                    ),
                    self.assertRaises(ValueError),
                ):
                    self.recover()

    def test_failed_harness_and_wrong_source_do_not_observe(self):
        for worker in (
            replace(original(), harness=HarnessOutcome(1, "", "failed")),
            replace(original(), worker_source_sha256="c" * 64),
        ):
            with mock.patch("agent_execution.native_evidence.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    self.recover(worker)
                run.assert_not_called()

    def test_timeout_is_not_success(self):
        with (
            mock.patch(
                "agent_execution.native_evidence.subprocess.run",
                side_effect=subprocess.TimeoutExpired("ssh", 30),
            ),
            self.assertRaisesRegex(ValueError, "observation failed"),
        ):
            self.recover()
