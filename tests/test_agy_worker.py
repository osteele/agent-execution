"""Packet-only Antigravity CLI (`agy`) execution through the remote worker."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_execution.agy_execution import (
    AGY_DENY_HOOK_PATH,
    AGY_MAX_ARGV_PROMPT_BYTES,
    AGY_MCP_CONFIG_PATH,
    AGY_RULES_PATH,
    AGY_SETTINGS_PATH,
    AgyStatusError,
    agy_envelope,
    agy_final_text,
    build_agy_home,
    validate_agy_command,
)
from agent_execution.command import CommandResult
from agent_execution.identity import worker_evidence_path
from agent_execution.worker import WorkerResult, execute_worker

MODEL = "gemini-3.1-pro-high"
RESPONSE = "BEGIN-7f3a\nNo blocking findings.\nEND-7f3a"


def agy_command(
    *,
    prompt: str = "review this packet",
    policy: str | None = "packet-only-no-tools",
    print_timeout: str | None = "600s",
    extra: tuple[str, ...] = (),
) -> list[str]:
    command = ["agy", "-p", prompt, "--output-format", "json", "--disable-slash-commands"]
    if print_timeout is not None:
        command.extend(["--print-timeout", print_timeout])
    if policy is not None:
        command.extend(["--execution-tool-policy", policy])
    return [*command, *extra]


def envelope(**overrides: object) -> str:
    value: dict[str, object] = {
        "status": "SUCCESS",
        "response": RESPONSE,
        "num_turns": 1,
        "denied_actions": [],
    }
    value.update(overrides)
    return json.dumps(value)


#: A stand-in `agy`. It records what the child process actually sees (argv,
#: stdin, environment, and the isolated HOME's configuration), asks the
#: installed hook about a tool call, then prints the scenario's envelope.
FAKE_AGY = """#!__PYTHON__
import json
import os
import subprocess
import sys
from pathlib import Path

home = Path(os.environ["HOME"])
cwd = Path.cwd()
scenario = json.loads((cwd / "agy-scenario.json").read_text())


def text(relative):
    path = home / relative
    return path.read_text() if path.is_file() else None


hook = home / "__HOOK__"
denied = subprocess.run(
    [str(hook)], input='{"tool_name": "read_file"}', capture_output=True, text=True
)
record = {
    "argv": sys.argv[1:],
    "home": str(home),
    "stdin": sys.stdin.read(),
    "environment": sorted(os.environ),
    "settings": text("__SETTINGS__"),
    "mcp_config": text("__MCP__"),
    "rules": text("__RULES__"),
    "hook_executable": os.access(hook, os.X_OK),
    "hook_exit": denied.returncode,
    "hook_stdout": denied.stdout,
    "keychains_link": os.path.islink(home / "Library" / "Keychains"),
}
(cwd / "agy-record.json").write_text(json.dumps(record))
sys.stdout.write(scenario["stdout"])
sys.exit(scenario["exit_status"])
"""


class AgyWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.root = base / "snapshot"
        self.root.mkdir()
        self.status_dir = base / "provider-status"
        environment = mock.patch.dict(
            os.environ,
            {
                "AGENT_PROVIDER_STATUS_DIR": str(self.status_dir),
                "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
                "AGENT_PROVIDER_STATUS_SALT": "agy-test-salt",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.base = base

    def run_agy(
        self,
        stdout: str,
        *,
        exit_status: int = 0,
        command: list[str] | None = None,
        model: str = MODEL,
        max_cost_usd: float | None = None,
        model_call_id: str = "agy-call",
    ) -> tuple[WorkerResult, list[list[str]]]:
        calls: list[list[str]] = []

        def invoke(argv: list[str], cwd: Path, timeout: float | None) -> CommandResult:
            calls.append(argv)
            return CommandResult(exit_status, stdout, "")

        result = execute_worker(
            provider="agy",
            model_call_id=model_call_id,
            command=agy_command() if command is None else command,
            harness_model=model,
            timeout=30.0,
            ctx_timeout=10.0,
            max_cost_usd=max_cost_usd,
            invoke=invoke,
            which=lambda name: "/tools/agy" if name == "agy" else None,
            cwd=self.root,
        )
        return result, calls

    def stored(self, result: WorkerResult) -> WorkerResult:
        return WorkerResult.parse(
            (self.root / worker_evidence_path(result.model_call_id)).read_text()
        )

    def published(self) -> list[dict[str, object]]:
        outbox = self.status_dir / "outbox"
        return [json.loads(path.read_text()) for path in sorted(outbox.glob("*.json"))]

    def test_successful_envelope_completes_and_returns_the_response_text(self) -> None:
        result, calls = self.run_agy(envelope())
        self.assertEqual(result.status, "completed", result.failure)
        self.assertTrue(result.model_call_started)
        self.assertEqual(result.session_id, "")
        self.assertIsNone(result.evidence)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "/tools/agy")
        self.assertEqual(calls[0][calls[0].index("--model") + 1], MODEL)
        # Stdout is kept verbatim, so the conductor rebuilds the bytes the local
        # adapter would unwrap, and the response text is extracted the same way.
        self.assertEqual(result.harness.stdout, envelope())
        self.assertEqual(agy_final_text(result.harness.stdout), RESPONSE)
        self.assertEqual(self.stored(result), result)

    def test_success_is_published_under_the_antigravity_route_and_model_pool(self) -> None:
        for model, pool in (
            ("gemini-3.8-flash-high", "google-antigravity/gemini"),
            ("claude-opus-4-6-thinking", "google-antigravity/other"),
        ):
            with self.subTest(model=model):
                for path in (self.status_dir / "outbox").glob("*.json"):
                    path.unlink()
                result, _ = self.run_agy(envelope(), model=model)
                self.assertEqual(result.status, "completed", result.failure)
                events = self.published()
                self.assertEqual(len(events), 1)
                subject = events[0]["subject"]
                assert isinstance(subject, dict)
                self.assertEqual(subject["route"], "google-antigravity")
                self.assertEqual(subject["billing_pool"], pool)

    def test_non_success_status_is_a_harness_failure(self) -> None:
        for exit_status in (0, 1):
            with self.subTest(exit_status=exit_status):
                result, _ = self.run_agy(
                    envelope(status="ERROR", response="", error="quota exhausted"),
                    exit_status=exit_status,
                )
                self.assertEqual(result.status, "harness_failed")
                self.assertTrue(result.model_call_started)
                self.assertIn("'ERROR'", result.failure)
                self.assertIn("quota exhausted", result.failure)
                self.assertEqual(self.stored(result), result)
        with self.assertRaises(AgyStatusError):
            agy_envelope(envelope(status="CANCELLED"))

    def test_empty_response_is_refused(self) -> None:
        for response in ("", "   \n", None):
            with self.subTest(response=response):
                result, _ = self.run_agy(envelope(response=response))
                self.assertEqual(result.status, "evidence_failed")
                self.assertIn("empty response", result.failure)
                self.assertEqual(self.stored(result), result)

    def test_recorded_tool_use_is_refused(self) -> None:
        for field, value in (
            ("denied_actions", [{"tool": "read_file", "path": "/etc/passwd"}]),
            ("tool_calls", [{"name": "run_command"}]),
            ("stats", {"tools": {"totalCalls": 2}}),
        ):
            with self.subTest(field=field):
                result, _ = self.run_agy(envelope(**{field: value}))
                self.assertEqual(result.status, "evidence_failed")
                self.assertIn("tool use", result.failure)
        result, _ = self.run_agy(envelope(denied_actions="read_file"))
        self.assertEqual(result.status, "evidence_failed")
        self.assertIn("malformed", result.failure)

    def test_zero_or_missing_turns_are_refused(self) -> None:
        result, _ = self.run_agy(envelope(num_turns=0))
        self.assertEqual(result.status, "evidence_failed")
        self.assertIn("num_turns == 0", result.failure)
        missing = json.loads(envelope())
        del missing["num_turns"]
        result, _ = self.run_agy(json.dumps(missing))
        self.assertEqual(result.status, "evidence_failed")
        self.assertIn("num_turns", result.failure)

    def test_unparseable_output_is_refused(self) -> None:
        result, _ = self.run_agy("Signed in as someone\n" + envelope())
        self.assertEqual(result.status, "evidence_failed")
        result, _ = self.run_agy("not json", exit_status=3)
        self.assertEqual(result.status, "harness_failed")
        self.assertIn("agy exited 3", result.failure)

    def test_a_stored_completed_result_cannot_carry_a_refused_envelope(self) -> None:
        result, _ = self.run_agy(envelope())
        forged = json.loads(result.to_json())
        forged["harness"]["stdout"] = envelope(denied_actions=[{"tool": "write_file"}])
        with self.assertRaisesRegex(ValueError, "tool use"):
            WorkerResult.parse(json.dumps(forged))

    def test_non_packet_tool_policies_are_refused_before_the_harness(self) -> None:
        for policy in ("read-only-no-shell", "workspace-write-no-shell", "", None):
            with self.subTest(policy=policy):
                result, calls = self.run_agy(envelope(), command=agy_command(policy=policy))
                self.assertEqual(result.status, "preflight_failed")
                self.assertFalse(result.model_call_started)
                self.assertIn("packet-only-no-tools", result.failure)
                self.assertEqual(calls, [])

    def test_unknown_models_are_refused_before_the_harness(self) -> None:
        for model in ("gemini-2.5-pro", "gpt-oss-120b-medium", "claude-sonnet-4-6", ""):
            with self.subTest(model=model):
                result, calls = self.run_agy(envelope(), model=model)
                self.assertEqual(result.status, "preflight_failed")
                self.assertIn("admitted explicitly", result.failure)
                self.assertEqual(calls, [])

    def test_command_grammar_is_strict(self) -> None:
        refused = {
            "plan mode": agy_command(extra=("--mode", "plan", "--model", MODEL)),
            "skip permissions": agy_command(
                extra=("--dangerously-skip-permissions", "--model", MODEL)
            ),
            "no print timeout": agy_command(print_timeout=None, extra=("--model", MODEL)),
            "zero print timeout": agy_command(print_timeout="0s", extra=("--model", MODEL)),
            "minutes": agy_command(print_timeout="10m", extra=("--model", MODEL)),
            "huge timeout": agy_command(print_timeout="999999s", extra=("--model", MODEL)),
            "unknown flag": agy_command(extra=("--model", MODEL, "--yolo")),
            "positional": agy_command(extra=("--model", MODEL, "extra")),
            "equals model": agy_command(extra=(f"--model={MODEL}",)),
            "duplicate model": agy_command(extra=("--model", MODEL, "--model", MODEL)),
            "text output": [
                "agy",
                "-p",
                "brief",
                "--output-format",
                "text",
                "--print-timeout",
                "60s",
                "--disable-slash-commands",
                "--execution-tool-policy",
                "packet-only-no-tools",
                "--model",
                MODEL,
            ],
            "slash commands": [
                "agy",
                "-p",
                "brief",
                "--output-format",
                "json",
                "--print-timeout",
                "60s",
                "--execution-tool-policy",
                "packet-only-no-tools",
                "--model",
                MODEL,
            ],
            "dash brief": agy_command(
                prompt="--dangerously-skip-permissions", extra=("--model", MODEL)
            ),
        }
        for label, command in refused.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_agy_command(command)
        with self.assertRaisesRegex(ValueError, "must not begin with '-'"):
            validate_agy_command(agy_command(prompt="--help", extra=("--model", MODEL)))
        invocation = validate_agy_command(agy_command(extra=("--model", MODEL)))
        self.assertEqual(invocation.model, MODEL)
        self.assertEqual(invocation.print_timeout_seconds, 600)

    def test_a_caller_cannot_supply_home_through_an_env_prefix(self) -> None:
        result, calls = self.run_agy(
            envelope(), command=["env", "HOME=/tmp/elsewhere", *agy_command()]
        )
        self.assertEqual(result.status, "preflight_failed")
        self.assertIn("runs agy", result.failure)
        self.assertEqual(calls, [])

    def test_hard_cash_cap_refuses_the_unobservable_credential(self) -> None:
        result, calls = self.run_agy(envelope(), max_cost_usd=0)
        self.assertEqual(result.status, "preflight_failed")
        self.assertIn("cost-cap-refused", result.failure)
        self.assertIn("not_observable", result.failure)
        self.assertEqual(calls, [])
        # Positive control: without a cap the same dispatch reaches the harness.
        result, calls = self.run_agy(envelope(), model_call_id="agy-uncapped")
        self.assertEqual(result.status, "completed", result.failure)
        self.assertEqual(len(calls), 1)

    def test_payload_marker_and_argv_limits_are_enforced(self) -> None:
        payloads = self.base / "payloads"
        payloads.mkdir()
        cases = {
            "oversized": "x" * (AGY_MAX_ARGV_PROMPT_BYTES + 1),
            "leading dash": "- not a flag, but argv parsers disagree",
        }
        for label, prompt in cases.items():
            with self.subTest(label=label):
                (payloads / "execution-prompt").write_text(prompt)
                with mock.patch.dict(os.environ, {"WEFT_PAYLOAD_DIR": str(payloads)}):
                    result = execute_worker(
                        provider="agy",
                        model_call_id="agy-payload",
                        command=agy_command(prompt="-"),
                        harness_model=MODEL,
                        timeout=30.0,
                        ctx_timeout=10.0,
                        prompt_payload="execution-prompt",
                        invoke_prompt=mock.Mock(side_effect=AssertionError("must not run")),
                        which=lambda name: "/tools/agy",
                        cwd=self.root,
                    )
                self.assertEqual(result.status, "preflight_failed")
                self.assertIn("agy brief", result.failure)
        # The stdin marker without a payload has no brief at all.
        result, calls = self.run_agy(envelope(), command=agy_command(prompt="-"))
        self.assertEqual(result.status, "preflight_failed")
        self.assertIn("requires a prompt payload", result.failure)
        self.assertEqual(calls, [])


class AgyIsolatedHomeTests(unittest.TestCase):
    """A real child process: what the HOME it sees contains, and that it is gone after."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.root = base / "snapshot"
        self.root.mkdir()
        self.payloads = base / "payloads"
        self.payloads.mkdir()

        self.source_home = base / "real-home"
        config = self.source_home / ".gemini"
        (config / "antigravity").mkdir(parents=True)
        self.user_settings = {
            "selectedAuthType": "oauth-personal",
            "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"command": "/bin/true"}]}]},
            "mcpServers": {"filesystem": {"command": "mcp-filesystem", "args": ["/"]}},
        }
        (config / "settings.json").write_text(json.dumps(self.user_settings))
        (config / "antigravity" / "mcp_config.json").write_text(
            json.dumps({"mcpServers": {"browser": {"command": "mcp-browser"}}})
        )
        (config / "GEMINI.md").write_text("Always run the test suite before answering.\n")
        (self.source_home / "Library" / "Keychains").mkdir(parents=True)

        bin_dir = base / "bin"
        bin_dir.mkdir()
        self.agy = bin_dir / "agy"
        script = (
            FAKE_AGY.replace("__PYTHON__", sys.executable)
            .replace("__HOOK__", str(AGY_DENY_HOOK_PATH))
            .replace("__SETTINGS__", str(AGY_SETTINGS_PATH))
            .replace("__MCP__", str(AGY_MCP_CONFIG_PATH))
            .replace("__RULES__", str(AGY_RULES_PATH))
        )
        self.agy.write_text(script)
        self.agy.chmod(self.agy.stat().st_mode | stat.S_IXUSR)

        environment = mock.patch.dict(
            os.environ,
            {
                "HOME": str(self.source_home),
                "WEFT_PAYLOAD_DIR": str(self.payloads),
                "GEMINI_API_KEY": "must-not-reach-agy",
                "XDG_CONFIG_HOME": str(self.source_home / ".config"),
                "AGENT_PROVIDER_STATUS_DIR": str(base / "provider-status"),
                "AGENT_PROVIDER_STATUS_TRANSPORT": "none",
                "AGENT_PROVIDER_STATUS_SALT": "agy-test-salt",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def execute(self, *, stdout: str, exit_status: int, model_call_id: str) -> WorkerResult:
        prompt = "Review the packet below.\nBEGIN-7f3a ... END-7f3a\n"
        (self.payloads / "execution-prompt").write_text(prompt)
        (self.root / "agy-scenario.json").write_text(
            json.dumps({"stdout": stdout, "exit_status": exit_status})
        )
        self.prompt = prompt
        return execute_worker(
            provider="agy",
            model_call_id=model_call_id,
            command=agy_command(prompt="-"),
            harness_model=MODEL,
            timeout=60.0,
            ctx_timeout=10.0,
            prompt_payload="execution-prompt",
            expect_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            which=lambda name: str(self.agy) if name == "agy" else None,
            cwd=self.root,
        )

    def record(self) -> dict[str, object]:
        return json.loads((self.root / "agy-record.json").read_text())

    def test_the_home_copies_sign_in_and_settings_and_none_of_the_users_history(self) -> None:
        gemini = self.source_home / ".gemini"
        cli = gemini / "antigravity-cli"
        (cli / "conversations").mkdir(parents=True)
        (cli / "conversations" / "c1.pb").write_text("an earlier session")
        (cli / "brain").mkdir()
        (cli / "brain" / "notes.md").write_text("remembered")
        (cli / "antigravity-oauth-token").write_text("token")
        (gemini / "antigravity-browser-profile").mkdir()
        (gemini / "skills" / "deploy").mkdir(parents=True)
        (gemini / "skills" / "deploy" / "SKILL.md").write_text("a user skill")
        home = self.root.parent / "built-home"

        build_agy_home(home, source_home=self.source_home)

        built = home / ".gemini"
        self.assertEqual(
            (built / "antigravity-cli" / "antigravity-oauth-token").read_text(), "token"
        )
        self.assertTrue((built / "settings.json").is_file())
        for excluded in (
            "antigravity-cli/conversations",
            "antigravity-cli/brain",
            "antigravity-browser-profile",
            "skills",
        ):
            self.assertFalse((built / excluded).exists(), excluded)

    def test_child_sees_deny_all_hook_and_empty_mcp_config_and_home_is_removed(self) -> None:
        result = self.execute(stdout=envelope(), exit_status=0, model_call_id="agy-home")
        self.assertEqual(result.status, "completed", result.failure)
        self.assertEqual(agy_final_text(result.harness.stdout), RESPONSE)
        record = self.record()
        home = Path(str(record["home"]))

        # The isolated HOME is the worker's own throwaway directory, not the
        # caller's HOME, and it no longer exists once the call has returned.
        self.assertNotEqual(home, self.source_home)
        self.assertTrue(home.name.startswith("agent-execution-agy-home-"))
        self.assertFalse(home.exists())

        # Deny-all hook: installed, executable, wired as the only PreToolUse
        # hook, and actually denying when asked about a tool call.
        settings = json.loads(str(record["settings"]))
        script = shlex.quote(str(home / AGY_DENY_HOOK_PATH))
        self.assertEqual(
            settings["hooks"],
            {
                "PreToolUse": [
                    {"matcher": "*", "hooks": [{"type": "command", "command": script}]},
                ],
            },
        )
        self.assertTrue(record["hook_executable"])
        self.assertEqual(record["hook_exit"], 2)
        self.assertIn('"permissionDecision":"deny"', str(record["hook_stdout"]))
        # Non-tool settings survive the copy; MCP servers do not.
        self.assertEqual(settings["selectedAuthType"], "oauth-personal")
        self.assertEqual(settings["mcpServers"], {})
        self.assertEqual(json.loads(str(record["mcp_config"])), {"mcpServers": {}})
        rules = str(record["rules"])
        self.assertIn("Do not call any tool", rules)
        self.assertNotIn("test suite", rules)
        self.assertTrue(record["keychains_link"])

        # The brief returns to argv in place of the `-` marker; the logical
        # tool-policy flag never reaches agy; stdin carries nothing.
        self.assertEqual(
            record["argv"],
            [
                "-p",
                self.prompt,
                "--output-format",
                "json",
                "--print-timeout",
                "600s",
                "--disable-slash-commands",
                "--model",
                MODEL,
            ],
        )
        self.assertEqual(record["stdin"], "")
        environment = record["environment"]
        assert isinstance(environment, list)
        self.assertIn("HOME", environment)
        self.assertNotIn("GEMINI_API_KEY", environment)
        self.assertNotIn("XDG_CONFIG_HOME", environment)

        # The caller's real configuration is untouched.
        self.assertEqual(
            json.loads((self.source_home / ".gemini" / "settings.json").read_text()),
            self.user_settings,
        )

    def test_home_is_removed_after_a_failed_call_too(self) -> None:
        result = self.execute(
            stdout=envelope(status="ERROR", response="", error="sign-in required"),
            exit_status=1,
            model_call_id="agy-home-failed",
        )
        self.assertEqual(result.status, "harness_failed")
        self.assertIn("sign-in required", result.failure)
        self.assertFalse(Path(str(self.record()["home"])).exists())


if __name__ == "__main__":
    unittest.main()
