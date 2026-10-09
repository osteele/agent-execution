"""Native command authority, stdin transport and result identity boundaries."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_execution import claude_execution as claude
from tests.test_claude_worker import MODEL, SESSION, command, envelope


class ClaudeExecutionTests(unittest.TestCase):
    def test_unsafe_flags_and_duplicate_arguments_are_refused(self) -> None:
        for extra in (
            ["--resume", SESSION],
            ["--model", "sonnet"],
            ["--allowedTools", "Read,Bash"],
            ["--tools", "Read"],
            ["--mcp-config", '{"mcpServers":{"evil":{}}}'],
            ["--permission-mode", "acceptEdits"],
            ["--dangerously-skip-permissions"],
            ["--plugin-dir", "/tmp/plugin"],
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                claude.validate_claude_command(command() + extra)
        unsafe = command()
        unsafe.insert(1, "HOME=/tmp")
        with self.assertRaises(ValueError):
            claude.validate_claude_command(unsafe)

    def test_packet_policy_cannot_expose_tools_or_drop_persistence_guard(self) -> None:
        for flag, value in (("--tools", "Bash"), ("--system-prompt", "")):
            argv = command(packet=True)
            argv[argv.index(flag) + 1] = value
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                claude.validate_claude_command(argv)
        argv = command(packet=True)
        argv.remove("--no-session-persistence")
        with self.assertRaises(ValueError):
            claude.validate_claude_command(argv)

    def test_native_resolution_scrubs_routes_and_rejects_failed_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapper = root / "claude"
            wrapper.write_text("#!/bin/sh\nexit 1\n")
            wrapper.chmod(0o755)
            native = root / "native"
            native.write_bytes(b"\xcf\xfa\xed\xfe" + b"fixture")
            native.chmod(0o755)
            environment = {
                "PATH": str(root),
                "HOME": "/signed-in",
                "ANTHROPIC_API_KEY": "secret",
                "ANTHROPIC_AUTH_TOKEN": "token",
                "ANTHROPIC_BASE_URL": "https://proxy.invalid",
                "ANTHROPIC_MODEL": "wrong",
                "LANG": "C",
            }
            with patch.object(
                claude.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, str(native), ""),
            ):
                executable, launch = claude.native_claude_launch(environment)
            self.assertEqual(executable, str(native.resolve()))
            self.assertEqual(launch["HOME"], "/signed-in")
            self.assertFalse(any(key.startswith("ANTHROPIC_") for key in launch))
            with (
                patch.object(
                    claude.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 1, "", "failed"),
                ),
                self.assertRaises(FileNotFoundError),
            ):
                claude.native_claude_launch(environment)
            with (
                patch.object(
                    claude.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, str(wrapper), ""),
                ),
                self.assertRaises(FileNotFoundError),
            ):
                claude.native_claude_launch(environment)

    def test_explicit_profile_is_not_silently_bypassed(self) -> None:
        with self.assertRaises(ValueError):
            claude.native_claude_launch({"PATH": "/bin", "CLAUDE_PROFILE": "personal"})
        argv = command()
        argv.insert(argv.index("claude"), "CLAUDE_PROFILE=review")
        with self.assertRaises(ValueError):
            claude.validate_claude_command(argv)

    def test_launch_exposes_exact_tools_and_delivers_only_stdin_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for packet in (False, True):
                with (
                    self.subTest(packet=packet),
                    patch.object(
                        claude,
                        "native_claude_launch",
                        return_value=("/native/claude", {"PATH": "/bin"}),
                    ),
                    patch.object(
                        claude,
                        "run_in_process_group",
                        return_value=subprocess.CompletedProcess([], 0, "{}", ""),
                    ) as run,
                ):
                    claude.run_claude_command(
                        command(packet=packet, prompt="-") + ["--effort", "xhigh"],
                        Path(temporary),
                        20,
                        prompt="private payload",
                    )
                    argv = run.call_args.args[0]
                    self.assertEqual(
                        argv[argv.index("--tools") + 1], "" if packet else "Read,Glob,Grep"
                    )
                    self.assertNotIn("-", argv)
                    self.assertNotIn("private payload", argv)
                    self.assertEqual(run.call_args.args[2], "private payload")
                    self.assertEqual(argv[argv.index("--effort") + 1], "xhigh")
                    self.assertIn("--safe-mode", argv)
                    if packet:
                        self.assertEqual(
                            argv[argv.index("--system-prompt") + 1], "Answer only from this packet."
                        )
                    else:
                        self.assertIn("--restricted", argv)

    def test_result_identity_and_native_errors_are_distinct(self) -> None:
        self.assertEqual(
            claude.claude_envelope(envelope(), model=MODEL, session_id=SESSION)["result"],
            "native answer",
        )
        with self.assertRaises(ValueError):
            claude.claude_envelope(envelope(modelUsage={"claude-sonnet-4-5": {}}), model=MODEL)
        with self.assertRaises(ValueError):
            claude.claude_envelope(
                envelope(session_id="790268ee-8af8-4ddb-9dcd-db05429dd832"), session_id=SESSION
            )
        with self.assertRaises(claude.ClaudeStatusError):
            claude.claude_envelope(
                envelope(subtype="error_during_execution", is_error=True, result="quota exhausted")
            )
        with self.assertRaises(ValueError):
            claude.claude_envelope(json.dumps({"type": "result", "result": "answer"}))
