"""Native worker completion, custody and evidence boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_execution.command import CommandResult
from agent_execution.credentials import claude_command_prefix
from agent_execution.identity import worker_evidence_path
from agent_execution.weft import WeftCommandRunner
from agent_execution.worker import WorkerResult, execute_worker
from tests.execution_status_fixtures import admitted_status

MODEL = "claude-opus-5-5"
SESSION = "d93bfcf1-73f6-4692-af1c-1729e37e06a3"


def command(*, packet: bool = False, prompt: str = "review") -> list[str]:
    argv = [
        *claude_command_prefix(),
        "claude",
        "-p",
        prompt,
        "--input-format",
        "text",
        "--output-format",
        "json",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--model",
        MODEL,
        "--session-id",
        SESSION,
    ]
    if packet:
        argv.extend(
            [
                "--safe-mode",
                "--tools",
                "",
                "--no-session-persistence",
                "--system-prompt",
                "Answer only from this packet.",
            ]
        )
    else:
        argv.extend(["--permission-mode", "plan", "--allowedTools", "Read,Glob,Grep"])
    return argv


def envelope(**updates: object) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "native answer",
            "session_id": SESSION,
            "modelUsage": {MODEL: {"inputTokens": 10, "outputTokens": 2}},
            **updates,
        }
    )


class ClaudeWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(
            os.environ,
            {
                "AGENT_PROVIDER_STATUS_DIR": str(self.root / "status"),
                "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.output = envelope()
        self.ctx_session = SESSION
        self.calls: list[list[str]] = []
        self.cap_basis = "subscription"
        observer = mock.patch(
            "agent_execution.execution_status.probe_status", side_effect=self.status
        )
        observer.start()
        self.addCleanup(observer.stop)

    def status(self, **arguments):
        status = admitted_status(**arguments)
        row = status["rows"][0]
        row["subject"]["effective_route"] = "anthropic"
        row["credential_basis"] = {"basis": self.cap_basis, "stale": False}
        return status

    def invoke(self, argv: list[str], cwd: Path, timeout: float | None) -> CommandResult:
        self.calls.append(argv)
        if argv[0] == "claude":
            return CommandResult(0, self.output, "")
        if argv[1:] == ["--version"]:
            return CommandResult(0, "ctx 1.0.2", "")
        if argv[1] == "import":
            return CommandResult(0, '{"schema_version":2,"outcome":"success"}', "")
        if argv[1:3] == ["show", "session"]:
            return CommandResult(
                0,
                json.dumps(
                    {
                        "schema_version": 1,
                        "target": "session",
                        "payload_type": "session_transcript",
                        "provider": "claude",
                        "provider_session_id": self.ctx_session,
                        "mode": "log",
                        "format": "json",
                        "events": [
                            {
                                "provider": "claude",
                                "provider_session_id": self.ctx_session,
                                "event_type": "tool_call",
                                "occurred_at": "2026-10-09T00:00:00.000Z",
                                "content": {"complete": True},
                                "activity": {"facts": [{"kind": "session_cwd", "value": str(cwd)}]},
                                "text": json.dumps(
                                    {"name": "Read", "input": {"file_path": str(cwd / "sample")}}
                                ),
                            }
                        ],
                    }
                ),
                "",
            )
        raise AssertionError(argv)

    def execute(
        self, *, packet: bool = False, argv: list[str] | None = None, cap: float | None = None
    ) -> WorkerResult:
        return execute_worker(
            provider="claude-packet" if packet else "claude",
            model_call_id="native-test",
            command=command(packet=packet) if argv is None else argv,
            timeout=30,
            ctx_timeout=10,
            cwd=self.root,
            invoke=self.invoke,
            which=lambda name: "/bin/ctx" if name == "ctx" else None,
            max_cost_usd=cap,
        )

    def test_grounded_exports_the_exact_pinned_session_and_round_trips(self) -> None:
        result = self.execute(cap=0)
        self.assertEqual(result.status, "completed", result.failure)
        self.assertEqual(result.session_id, SESSION)
        self.assertEqual(
            WorkerResult.parse((self.root / worker_evidence_path("native-test")).read_text()),
            result,
        )
        show = next(argv for argv in self.calls if argv[1:3] == ["show", "session"])
        self.assertEqual(show[show.index("--provider-session") + 1], SESSION)

    def test_packet_finishes_without_ctx_or_persisted_session(self) -> None:
        result = self.execute(packet=True)
        self.assertEqual(result.status, "completed", result.failure)
        self.assertEqual(result.session_id, "")
        self.assertIsNone(result.evidence)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(WorkerResult.parse(result.to_json()), result)

    def test_unknown_billing_refuses_capped_call_despite_available_login(self) -> None:
        self.cap_basis = "unobserved"
        result = self.execute(cap=0)
        self.assertEqual(result.status, "preflight_failed")
        self.assertFalse(result.model_call_started)
        self.assertIn("cost-cap-refused", result.failure)
        self.assertEqual(self.calls, [])

    def test_native_refusal_is_not_success_even_with_exit_zero(self) -> None:
        self.output = envelope(
            is_error=True, subtype="error_during_execution", result="You've hit your usage limit"
        )
        result = self.execute(packet=True)
        self.assertEqual(result.status, "harness_failed")
        self.assertTrue(result.model_call_started)
        self.assertIn("usage limit", result.failure)

    def test_mismatched_model_or_session_and_wrong_ctx_session_are_refused(self) -> None:
        for updates in (
            {"modelUsage": {"claude-sonnet-4-5": {}}},
            {"session_id": "790268ee-8af8-4ddb-9dcd-db05429dd832"},
        ):
            with self.subTest(updates=updates):
                self.output = envelope(**updates)
                self.assertEqual(self.execute().status, "evidence_failed")
        self.output = envelope()
        self.ctx_session = "790268ee-8af8-4ddb-9dcd-db05429dd832"
        self.assertEqual(self.execute().status, "evidence_failed")

    def test_unsafe_policy_is_refused_before_any_harness_call(self) -> None:
        result = self.execute(argv=command() + ["--dangerously-skip-permissions"])
        self.assertEqual(result.status, "preflight_failed")
        self.assertEqual(self.calls, [])

    def test_weft_command_preserves_stdin_model_effort_and_custody_contract(self) -> None:
        runner = WeftCommandRunner(
            host="studio", agent="claude-packet", model_call_id="native-test", fallback=None
        )
        runner.prompt_sha256 = hashlib.sha256(b"private review").hexdigest()
        import shlex

        remote = shlex.split(
            runner._remote_command(
                command(packet=True, prompt="private review") + ["--effort", "high"], self.root, 60
            )
        )
        self.assertNotIn("private review", remote)
        self.assertEqual(remote[remote.index("--harness-model") + 1], MODEL)
        self.assertEqual(remote[remote.index("--execution-transport") + 1], "weft")
        options = runner._dispatched_contract({"command": shlex.join(remote)})
        self.assertEqual(options["claude_model"], MODEL)
        self.assertEqual(options["claude_session_id"], SESSION)
