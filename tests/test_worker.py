"""Remote harness execution and ctx evidence export."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_execution.command import CommandResult
from agent_execution.identity import worker_evidence_path
from agent_execution.worker import (
    WORKER_PROTOCOL_VERSION,
    WorkerEvidence,
    WorkerResult,
    codex_session_id,
    execute_worker,
)
from agent_execution.worker_cli import main
from tests.support.omp import omp_command, omp_output


def transcript(session_id: str = "thread-42", *, cwd: str = "/remote/project") -> dict[str, object]:
    return {
        "schema_version": 1,
        "target": "session",
        "payload_type": "session_transcript",
        "provider": "codex",
        "provider_session_id": session_id,
        "mode": "log",
        "format": "json",
        "events": [
            {
                "provider": "codex",
                "provider_session_id": session_id,
                "ctx_event_id": "event-1",
                "event_type": "tool_call",
                "occurred_at": "1970-01-01T00:00:10.500Z",
                "content": {"complete": True},
                "activity": {"facts": [{"kind": "session_cwd", "value": cwd}]},
                "text": json.dumps(
                    {
                        "type": "custom_tool_call",
                        "name": "view_image",
                        "arguments": {"path": "/remote/project/figure.png"},
                    }
                ),
            }
        ],
    }


INVALID_CTX_IMPORT_RECEIPTS: tuple[tuple[dict[str, object], str], ...] = (
    ({"schema_version": True, "outcome": "success"}, "unsupported"),
    ({"schema_version": 2.0, "outcome": "success"}, "unsupported"),
    ({"schema_version": 2, "outcome": "failed"}, "did not complete"),
)


# Legacy Codex execution tests below opt in individually, with injected runners
# only, to preserve native evidence/error regression coverage. Public execution
# tests never alter the supported provider registry.


class WorkerResultTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = Path(worker_evidence_path("model-call-7"))
        self.calls: list[list[str]] = []

    def which(self, executable: str) -> str | None:
        return {"ctx": "/tools/ctx", "codex": "/tools/codex"}.get(executable)

    def write_preflight_result(self, *, model_call_id: str = "model-call-7") -> WorkerResult:
        return execute_worker(
            provider="unsupported",
            model_call_id=model_call_id,
            command=["unsupported"],
            timeout=30.0,
            ctx_timeout=10.0,
            cwd=self.root,
        )

    def test_result_identity_cannot_escape_protected_namespace(self) -> None:
        for identity in ("", "..", "../other", "/tmp/result", "a/b", "a.b", "Foo", "x" * 129):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                self.write_preflight_result(model_call_id=identity)
        self.assertFalse((self.root / ".agent-execution").exists())

    def test_cli_rejects_caller_selected_evidence_path(self) -> None:
        with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as error:
            main(
                [
                    "execute",
                    "--provider",
                    "unsupported",
                    "--model-call-id",
                    "cli-call",
                    "--evidence-out",
                    "outputs/other.json",
                    "--",
                    "unsupported",
                ]
            )
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / "outputs").exists())

    def test_cli_publishes_only_deterministic_evidence(self) -> None:
        stdout = io.StringIO()
        with (
            mock.patch("agent_execution.worker.Path.cwd", return_value=self.root),
            mock.patch("sys.stdout", new=stdout),
            mock.patch("sys.stderr", new=io.StringIO()),
        ):
            status = main(
                [
                    "execute",
                    "--provider",
                    "unsupported",
                    "--model-call-id",
                    "cli-call",
                    "--expect-protocol",
                    "2",
                    "--",
                    "unsupported",
                ]
            )
        self.assertEqual(status, 0)
        path = ".agent-execution/results/cli-call.json"
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["artifact_path"], path)
        result = WorkerResult.parse((self.root / path).read_text())
        self.assertEqual(result.model_call_id, "cli-call")
        self.assertEqual(result.worker_protocol_version, 2)
        self.assertFalse((self.root / "outputs").exists())

    def test_symlink_cannot_redirect_evidence_into_writable_files(self) -> None:
        outputs = self.root / "outputs"
        outputs.mkdir()
        (self.root / ".agent-execution").symlink_to(outputs, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.write_preflight_result()
        self.assertEqual(list(outputs.iterdir()), [])

    def test_results_for_distinct_calls_remain_independent(self) -> None:
        first = self.write_preflight_result(model_call_id="first")
        second = self.write_preflight_result(model_call_id="second")
        for result in (first, second):
            path = self.root / worker_evidence_path(result.model_call_id)
            self.assertEqual(WorkerResult.parse(path.read_text()), result)

    def test_result_publication_syncs_file_and_directory_chain(self) -> None:
        events: list[str] = []
        staging_directories: list[Path] = []
        real_fsync = os.fsync
        real_replace = os.replace
        real_mkstemp = tempfile.mkstemp

        def record_fsync(descriptor: int) -> None:
            mode = os.fstat(descriptor).st_mode
            events.append("directory" if stat.S_ISDIR(mode) else "file")
            real_fsync(descriptor)

        def record_replace(source: Path, destination: Path) -> None:
            events.append("replace")
            real_replace(source, destination)

        def record_mkstemp(*, dir: Path, prefix: str, suffix: str) -> tuple[int, str]:
            staging_directories.append(dir)
            return real_mkstemp(dir=dir, prefix=prefix, suffix=suffix)

        with (
            mock.patch("agent_execution.worker.os.fsync", side_effect=record_fsync),
            mock.patch("agent_execution.worker.os.replace", side_effect=record_replace),
            mock.patch("agent_execution.worker.tempfile.mkstemp", side_effect=record_mkstemp),
        ):
            result = self.write_preflight_result()

        destination = self.root / self.output
        self.assertEqual(staging_directories, [destination.parent.resolve()])
        self.assertEqual(events, ["file", "replace", "directory", "directory", "directory"])
        self.assertEqual(WorkerResult.parse(destination.read_text()), result)
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)

    def test_interrupted_result_replace_preserves_prior_complete_result(self) -> None:
        previous = self.write_preflight_result()
        destination = self.root / self.output

        with (
            mock.patch(
                "agent_execution.worker.os.replace",
                side_effect=OSError("replace interrupted"),
            ),
            self.assertRaisesRegex(OSError, "replace interrupted"),
        ):
            self.write_preflight_result()

        self.assertEqual(WorkerResult.parse(destination.read_text()), previous)
        self.assertEqual(list(destination.parent.glob(f".{destination.name}.*.tmp")), [])

    def test_session_identity_comes_only_from_the_structured_start_event(self) -> None:
        planted = json.dumps({"result": {"thread_id": "planted"}})
        started = json.dumps({"type": "thread.started", "thread_id": "thread-42"})
        self.assertEqual(codex_session_id(f"{planted}\n{started}"), "thread-42")
        self.assertIsNone(codex_session_id(planted))

    def test_native_codex_refuses_before_dependency_or_auth_probes(self) -> None:
        invoke = mock.Mock(side_effect=AssertionError("must not invoke a harness or probe"))
        which = mock.Mock(side_effect=AssertionError("must not inspect native dependencies"))
        result = execute_worker(
            provider="codex",
            model_call_id="model-call-7",
            command=["codex", "exec", "review"],
            timeout=30.0,
            ctx_timeout=10.0,
            invoke=invoke,
            which=which,
            cwd=self.root,
            max_cost_usd=0,
        )
        self.assertEqual(result.status, "preflight_failed")
        self.assertFalse(result.model_call_started)
        stored = WorkerResult.parse((self.root / self.output).read_text())
        self.assertIn("unsupported worker provider", stored.failure)
        self.assertEqual(stored.provider, "codex")
        invoke.assert_not_called()
        which.assert_not_called()

    def test_openai_models_execute_through_packet_omp_without_native_dependencies(self) -> None:
        selector = "openai-codex/gpt-6-astra"
        invoked: list[list[str]] = []

        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            invoked.append(command)
            return CommandResult(
                0,
                omp_output(cwd=str(cwd), selector=selector, policy="packet-only-no-tools"),
                "",
            )

        with mock.patch("agent_execution.worker.require_omp_sdk", return_value=(self.root, "bun")):
            result = execute_worker(
                provider="omp-packet",
                model_call_id="openai-via-omp",
                command=omp_command(str(self.root), selector=selector, packet=True),
                invoke=invoke,
                timeout=30.0,
                ctx_timeout=10.0,
                which=lambda name: None,
                cwd=self.root,
            )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.provider, "omp-packet")
        self.assertTrue(result.model_call_started)
        self.assertEqual(len(invoked), 1)
        self.assertNotIn("codex", invoked[0])
        stored = WorkerResult.parse(
            (self.root / worker_evidence_path(result.model_call_id)).read_text()
        )
        self.assertEqual(stored, result)

    def test_workspace_writer_admits_opus(self) -> None:
        selector = "anthropic/claude-opus-5-5"
        invoked: list[list[str]] = []

        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            invoked.append(command)
            return CommandResult(
                0,
                omp_output(cwd=str(cwd), selector=selector, policy="workspace-write-no-shell"),
                "",
            )

        with mock.patch("agent_execution.worker.require_omp_sdk", return_value=(self.root, "bun")):
            result = execute_worker(
                provider="omp",
                model_call_id="opus-writer",
                command=omp_command(str(self.root), selector=selector, writer=True),
                invoke=invoke,
                timeout=30.0,
                ctx_timeout=10.0,
                which=lambda name: None,
                cwd=self.root,
            )
        self.assertEqual(result.status, "completed", result.failure)
        self.assertEqual(len(invoked), 1)

    def test_workspace_writer_refuses_unregistered_identity_before_harness(self) -> None:
        with mock.patch("agent_execution.worker.require_omp_sdk", return_value=(self.root, "bun")):
            result = execute_worker(
                provider="omp",
                model_call_id="writer-refused",
                command=omp_command(
                    str(self.root),
                    selector="anthropic/claude-opus-5",
                    writer=True,
                ),
                invoke=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("must not invoke unregistered writer")
                ),
                timeout=30.0,
                ctx_timeout=10.0,
                which=lambda name: None,
                cwd=self.root,
            )
        self.assertEqual(result.status, "preflight_failed")
        self.assertFalse(result.model_call_started)
        self.assertIn("writer OMP selector", result.failure)

    def test_worker_refuses_metered_route_before_invoking_harness(self) -> None:
        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            self.calls.append(command)
            return CommandResult(
                0,
                omp_output(
                    cwd=str(cwd),
                    selector="zai/glm-5.3-flash",
                    policy="packet-only-no-tools",
                ),
                "",
            )

        def run(maximum: float | None) -> WorkerResult:
            with mock.patch(
                "agent_execution.worker.require_omp_sdk", return_value=(self.root, "bun")
            ):
                return execute_worker(
                    provider="omp-packet",
                    model_call_id="cost-cap-worker",
                    command=omp_command(str(self.root), selector="zai/glm-5.3-flash", packet=True),
                    timeout=30.0,
                    ctx_timeout=10.0,
                    invoke=invoke,
                    which=lambda executable: f"/tools/{executable}",
                    cwd=self.root,
                    max_cost_usd=maximum,
                )

        result = run(0)
        self.assertEqual(result.status, "preflight_failed")
        self.assertFalse(result.model_call_started)
        self.assertEqual(self.calls, [])
        self.assertIn("cost-cap-refused", result.failure)
        # Positive control: the same transport without a cap reaches the
        # harness. A missing executable cannot masquerade as budget enforcement.
        self.assertEqual(run(None).status, "completed")
        self.assertEqual(len(self.calls), 1)

    def test_protocol_mismatch_fails_before_the_model_and_still_writes_a_result(self) -> None:
        invoked = False

        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            nonlocal invoked
            invoked = True
            raise AssertionError(command)

        expected = WORKER_PROTOCOL_VERSION + 1
        result = execute_worker(
            provider="codex",
            model_call_id="model-call-7",
            command=["codex", "exec", "review"],
            timeout=30.0,
            ctx_timeout=10.0,
            expect_protocol=expected,
            invoke=invoke,
            which=self.which,
            clock=lambda: 10.0,
            cwd=self.root,
        )

        self.assertFalse(invoked)
        self.assertEqual(result.status, "preflight_failed")
        self.assertFalse(result.model_call_started)
        self.assertIn(f"conductor expects {expected}", result.failure)
        self.assertIn(f"remote worker reports {WORKER_PROTOCOL_VERSION}", result.failure)
        stored = WorkerResult.parse((self.root / self.output).read_text())
        self.assertEqual(stored, result)
        self.assertEqual(stored.worker_protocol_version, WORKER_PROTOCOL_VERSION)

    @mock.patch("agent_execution.worker.SUPPORTED_WORKER_PROVIDERS", frozenset({"codex"}))
    def test_legacy_codex_payload_and_source_are_authenticated_before_execution(self) -> None:
        prompt = "review these exact bytes"
        prompt_digest = hashlib.sha256(prompt.encode()).hexdigest()
        source_digest = "a" * 64
        payloads = self.root / "payloads"
        payloads.mkdir()
        (payloads / "execution-prompt").write_text(prompt)
        received: list[str] = []

        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            if command == ["/tools/ctx", "--version"]:
                return CommandResult(0, "ctx 1.0.2\n", "")
            if command[1] == "import":
                return CommandResult(0, json.dumps({"schema_version": 2, "outcome": "success"}), "")
            if command[1:3] == ["show", "session"]:
                return CommandResult(0, json.dumps(transcript(cwd=str(self.root.resolve()))), "")
            raise AssertionError(command)

        def invoke_prompt(
            command: list[str], cwd: Path, body: str, timeout: float | None
        ) -> CommandResult:
            received.append(body)
            self.assertEqual(command[-1], "-")
            return CommandResult(
                0,
                json.dumps({"type": "thread.started", "thread_id": "thread-42"}),
                "",
            )

        times = iter([10.0, 11.0, 12.0])
        with (
            mock.patch.dict(os.environ, {"WEFT_PAYLOAD_DIR": str(payloads)}),
            mock.patch(
                "agent_execution.worker.installed_source_sha256", return_value=source_digest
            ),
        ):
            result = execute_worker(
                provider="codex",
                model_call_id="model-call-7",
                command=["codex", "exec", "-"],
                timeout=30.0,
                ctx_timeout=10.0,
                expect_protocol=WORKER_PROTOCOL_VERSION,
                expect_source_sha256=source_digest,
                prompt_payload="execution-prompt",
                expect_prompt_sha256=prompt_digest,
                invoke=invoke,
                invoke_prompt=invoke_prompt,
                which=self.which,
                clock=lambda: next(times),
                cwd=self.root,
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(received, [prompt])
        self.assertEqual(result.worker_source_sha256, source_digest)
        self.assertEqual(result.prompt_sha256, prompt_digest)
        self.assertIsNotNone(result.worker_identity)
        assert result.worker_identity is not None
        self.assertEqual(result.worker_identity.source_sha256, source_digest)
        self.assertEqual(result.worker_identity.protocol_version, WORKER_PROTOCOL_VERSION)

    def test_source_mismatch_refuses_before_reading_the_prompt(self) -> None:
        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            raise AssertionError(command)

        with mock.patch("agent_execution.worker.installed_source_sha256", return_value="c" * 64):
            result = execute_worker(
                provider="codex",
                model_call_id="model-call-7",
                command=["codex", "exec", "-"],
                timeout=30.0,
                ctx_timeout=10.0,
                expect_source_sha256="a" * 64,
                prompt_payload="execution-prompt",
                expect_prompt_sha256="b" * 64,
                invoke=invoke,
                which=self.which,
                clock=lambda: 10.0,
                cwd=self.root,
            )

        self.assertEqual(result.status, "preflight_failed")
        self.assertFalse(result.model_call_started)
        self.assertIsNone(result.harness.exit_status)
        self.assertEqual(result.prompt_sha256, "")
        self.assertIn("source mismatch", result.failure)
        self.assertEqual(WorkerResult.parse((self.root / self.output).read_text()), result)

    def test_prompt_digest_mismatch_refuses_before_codex(self) -> None:
        payloads = self.root / "payloads"
        payloads.mkdir()
        (payloads / "execution-prompt").write_text("changed prompt")
        invoked = False

        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            nonlocal invoked
            invoked = True
            raise AssertionError(command)

        source_digest = "a" * 64
        with (
            mock.patch.dict(os.environ, {"WEFT_PAYLOAD_DIR": str(payloads)}),
            mock.patch(
                "agent_execution.worker.installed_source_sha256", return_value=source_digest
            ),
        ):
            result = execute_worker(
                provider="codex",
                model_call_id="model-call-7",
                command=["codex", "exec", "-"],
                timeout=30.0,
                ctx_timeout=10.0,
                expect_source_sha256=source_digest,
                prompt_payload="execution-prompt",
                expect_prompt_sha256="b" * 64,
                invoke=invoke,
                which=self.which,
                clock=lambda: 10.0,
                cwd=self.root,
            )

        self.assertFalse(invoked)
        self.assertEqual(result.status, "preflight_failed")
        self.assertIn("prompt payload digest mismatch", result.failure)

    @mock.patch("agent_execution.worker.SUPPORTED_WORKER_PROVIDERS", frozenset({"codex"}))
    def test_legacy_codex_success_exports_one_exact_session(self) -> None:
        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            self.calls.append(command)
            if command == ["/tools/ctx", "--version"]:
                return CommandResult(0, "ctx 1.0.2\n", "")
            if command[0] == "/tools/codex":
                return CommandResult(
                    0,
                    json.dumps({"type": "thread.started", "thread_id": "thread-42"}),
                    "",
                )
            if command[1] == "import":
                return CommandResult(0, json.dumps({"schema_version": 2, "outcome": "success"}), "")
            if command[1:3] == ["show", "session"]:
                return CommandResult(0, json.dumps(transcript(cwd=str(self.root.resolve()))), "")
            raise AssertionError(command)

        times = iter([10.0, 11.0, 12.0])
        result = execute_worker(
            provider="codex",
            model_call_id="model-call-7",
            command=["codex", "exec", "review"],
            timeout=30.0,
            ctx_timeout=10.0,
            invoke=invoke,
            which=self.which,
            clock=lambda: next(times),
            cwd=self.root,
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.session_id, "thread-42")
        self.assertTrue(result.model_call_started)
        show = next(call for call in self.calls if call[1:3] == ["show", "session"])
        self.assertEqual(show[show.index("--provider-session") + 1], "thread-42")
        self.assertEqual(show[show.index("--mode") + 1], "log")
        self.assertEqual(WorkerResult.parse((self.root / self.output).read_text()), result)

    @mock.patch("agent_execution.worker.SUPPORTED_WORKER_PROVIDERS", frozenset({"codex"}))
    def test_legacy_codex_refusal_preserves_stderr_and_reports_no_model_call(self) -> None:
        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            if command == ["/tools/ctx", "--version"]:
                return CommandResult(0, "ctx 1.0.2\n", "")
            return CommandResult(
                1,
                "",
                "Not inside a trusted directory and --skip-git-repo-check was not specified.\n",
            )

        result = execute_worker(
            provider="codex",
            model_call_id="model-call-7",
            command=["codex", "exec", "review"],
            timeout=30.0,
            ctx_timeout=10.0,
            invoke=invoke,
            which=self.which,
            cwd=self.root,
        )

        self.assertEqual(result.status, "preflight_failed")
        self.assertFalse(result.model_call_started)
        self.assertIn("Not inside a trusted directory", result.failure)
        self.assertEqual(result.harness.exit_status, 1)

    @mock.patch("agent_execution.worker.SUPPORTED_WORKER_PROVIDERS", frozenset({"codex"}))
    def test_legacy_codex_failure_preserves_the_structured_provider_error(self) -> None:
        provider_error = (
            "The 'retired-model' model is not supported when using Codex with this account."
        )

        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            if command == ["/tools/ctx", "--version"]:
                return CommandResult(0, "ctx 1.0.2\n", "")
            if command[0] == "/tools/codex":
                return CommandResult(
                    1,
                    "\n".join(
                        [
                            json.dumps({"type": "thread.started", "thread_id": "thread-42"}),
                            json.dumps(
                                {
                                    "type": "turn.failed",
                                    "error": {"message": provider_error},
                                }
                            ),
                        ]
                    ),
                    "Reading additional input from stdin...\n",
                )
            if command[1] == "import":
                return CommandResult(0, json.dumps({"schema_version": 2, "outcome": "success"}), "")
            if command[1:3] == ["show", "session"]:
                return CommandResult(0, json.dumps(transcript(cwd=str(self.root.resolve()))), "")
            raise AssertionError(command)

        times = iter([10.0, 11.0, 12.0])
        result = execute_worker(
            provider="codex",
            model_call_id="model-call-7",
            command=["codex", "exec", "review"],
            timeout=30.0,
            ctx_timeout=10.0,
            invoke=invoke,
            which=self.which,
            clock=lambda: next(times),
            cwd=self.root,
        )

        self.assertEqual(result.status, "harness_failed")
        self.assertTrue(result.model_call_started)
        self.assertIn(provider_error, result.failure)
        self.assertNotIn("Reading additional input", result.failure)

    @mock.patch("agent_execution.worker.SUPPORTED_WORKER_PROVIDERS", frozenset({"codex"}))
    def test_legacy_codex_oversized_input_remains_primary_when_session_export_fails(self) -> None:
        provider_error = "Your input exceeds the context window of this model."
        harness_stdout = "\n".join(
            (
                json.dumps({"type": "thread.started", "thread_id": "thread-oversized"}),
                json.dumps({"type": "turn.failed", "error": {"message": provider_error}}),
            )
        )
        for export_timeout in (False, True):
            with self.subTest(export_timeout=export_timeout):
                exports: list[str] = []

                def invoke(
                    command: list[str],
                    cwd: Path,
                    timeout: float | None,
                    exports: list[str] = exports,
                    export_timeout: bool = export_timeout,
                ) -> CommandResult:
                    if command == ["/tools/ctx", "--version"]:
                        return CommandResult(0, "ctx 1.0.2\n", "")
                    if command[0] == "/tools/codex":
                        return CommandResult(
                            1, harness_stdout, "Reading additional input from stdin...\n"
                        )
                    if command[1] == "import":
                        return CommandResult(
                            0, json.dumps({"schema_version": 2, "outcome": "success"}), ""
                        )
                    if command[1:3] == ["show", "session"]:
                        exports.append(command[command.index("--provider-session") + 1])
                        if export_timeout:
                            assert timeout is not None
                            raise subprocess.TimeoutExpired(command, timeout)
                        return CommandResult(1, "", "session not found")
                    raise AssertionError(command)

                result = execute_worker(
                    provider="codex",
                    model_call_id="oversized-input",
                    command=["codex", "exec", "-"],
                    timeout=30.0,
                    ctx_timeout=10.0,
                    invoke=invoke,
                    which=self.which,
                    cwd=self.root,
                )

                self.assertEqual(exports, ["thread-oversized"])
                stored = WorkerResult.parse(
                    (self.root / worker_evidence_path(result.model_call_id)).read_text()
                )
                self.assertEqual(stored, result)
                self.assertEqual(stored.status, "harness_failed")
                self.assertTrue(stored.model_call_started)
                self.assertEqual(stored.session_id, "thread-oversized")
                self.assertEqual(stored.harness.exit_status, 1)
                self.assertEqual(stored.harness.stdout, harness_stdout)
                self.assertIsNone(stored.evidence)
                self.assertIn(provider_error, stored.failure)
                self.assertIn(
                    "ctx evidence collection" if export_timeout else "session not found",
                    stored.failure,
                )
                self.assertNotIn("Reading additional input", stored.failure)

    @mock.patch("agent_execution.worker.SUPPORTED_WORKER_PROVIDERS", frozenset({"codex"}))
    def test_legacy_codex_evidence_failure_after_success_is_durable(self) -> None:
        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            if command == ["/tools/ctx", "--version"]:
                return CommandResult(0, "ctx 1.0.2", "")
            if command[0] == "/tools/codex":
                return CommandResult(
                    0,
                    json.dumps({"type": "thread.started", "thread_id": "thread-42"}),
                    "",
                )
            if command[1] == "import":
                return CommandResult(1, "", "index unavailable")
            raise AssertionError(command)

        # Three readings: worker start, harness start, completion.
        times = iter([10.0, 11.0, 12.0])
        result = execute_worker(
            provider="codex",
            model_call_id="model-call-7",
            command=["codex", "exec", "review"],
            timeout=30.0,
            ctx_timeout=10.0,
            invoke=invoke,
            which=self.which,
            clock=lambda: next(times),
            cwd=self.root,
        )

        self.assertEqual(result.status, "evidence_failed")
        self.assertTrue(result.model_call_started)
        self.assertIn("ctx import failed", result.failure)
        self.assertTrue((self.root / self.output).exists())

    @mock.patch("agent_execution.worker.SUPPORTED_WORKER_PROVIDERS", frozenset({"codex"}))
    def test_legacy_codex_ctx_timeout_is_a_durable_evidence_failure(self) -> None:
        def invoke(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            if command == ["/tools/ctx", "--version"]:
                return CommandResult(0, "ctx 1.0.2", "")
            if command[0] == "/tools/codex":
                return CommandResult(
                    0,
                    json.dumps({"type": "thread.started", "thread_id": "thread-42"}),
                    "",
                )
            assert timeout is not None
            raise subprocess.TimeoutExpired(command, timeout)

        times = iter([10.0, 11.0, 12.0])
        result = execute_worker(
            provider="codex",
            model_call_id="model-call-7",
            command=["codex", "exec", "review"],
            timeout=30.0,
            ctx_timeout=10.0,
            invoke=invoke,
            which=self.which,
            clock=lambda: next(times),
            cwd=self.root,
        )

        self.assertEqual(result.status, "evidence_failed")
        self.assertIn("ctx evidence collection", result.failure)
        self.assertTrue((self.root / self.output).exists())

    def test_worker_evidence_requires_a_successful_exact_ctx_import_receipt(self) -> None:
        for receipt, message in INVALID_CTX_IMPORT_RECEIPTS:
            with (
                self.subTest(receipt=receipt),
                self.assertRaisesRegex(ValueError, message),
            ):
                WorkerEvidence.from_dict({"import_receipt": receipt, "transcript": {}})

    @mock.patch("agent_execution.worker.SUPPORTED_WORKER_PROVIDERS", frozenset({"codex"}))
    def test_legacy_codex_refuses_invalid_import_receipt_before_export(self) -> None:
        for receipt, message in INVALID_CTX_IMPORT_RECEIPTS:
            with self.subTest(receipt=receipt):
                exported = False

                def invoke(
                    command: list[str],
                    cwd: Path,
                    timeout: float | None,
                    receipt: dict[str, object] = receipt,
                ) -> CommandResult:
                    nonlocal exported
                    if command == ["/tools/ctx", "--version"]:
                        return CommandResult(0, "ctx 1.0.2", "")
                    if command[0] == "/tools/codex":
                        return CommandResult(
                            0,
                            json.dumps({"type": "thread.started", "thread_id": "thread-42"}),
                            "",
                        )
                    if command[1] == "import":
                        return CommandResult(0, json.dumps(receipt), "")
                    if command[1:3] == ["show", "session"]:
                        exported = True
                    raise AssertionError(command)

                times = iter([10.0, 11.0, 12.0])
                result = execute_worker(
                    provider="codex",
                    model_call_id="model-call-7",
                    command=["codex", "exec", "review"],
                    timeout=30.0,
                    ctx_timeout=10.0,
                    invoke=invoke,
                    which=self.which,
                    clock=lambda times=times: next(times),
                    cwd=self.root,
                )

                self.assertEqual(result.status, "evidence_failed")
                self.assertIn(message, result.failure)
                self.assertFalse(exported)

    def test_worker_result_refuses_an_unknown_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported worker result schema"):
            WorkerResult.parse('{"schema_version":"agent-execution.worker-result/v9"}')


if __name__ == "__main__":
    unittest.main()
