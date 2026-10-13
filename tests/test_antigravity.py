"""Antigravity is an OAuth-only OMP route with independent model-family quota."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

from agent_execution import costs
from agent_execution.credentials import CredentialBasis
from agent_execution.omp_execution import (
    GROUNDED_OMP_SELECTORS,
    GROUNDED_OMP_SELECTORS_BY_PROVIDER,
    OMP_WRITER_SELECTORS,
    omp_launch_environment,
    omp_transcript,
    validate_omp_command,
)
from tests.support.omp import omp_command, omp_output


class AntigravityTest(unittest.TestCase):
    def test_selector_registries_commands_and_transcripts(self) -> None:
        gemini = "google-antigravity/gemini-3.1-pro"
        opus = "google-antigravity/claude-opus-4-6"
        sonnet = "google-antigravity/claude-sonnet-4-6"
        self.assertEqual(GROUNDED_OMP_SELECTORS_BY_PROVIDER["google-antigravity"], gemini)
        self.assertEqual(
            {s for s in GROUNDED_OMP_SELECTORS if s.startswith("google-antigravity/")},
            {gemini, opus, sonnet},
        )
        self.assertEqual(
            {s for s in OMP_WRITER_SELECTORS if s.startswith("google-antigravity/")},
            {gemini, opus},
        )
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            for selector in (gemini, opus, sonnet):
                with self.subTest(selector=selector):
                    self.assertEqual(
                        validate_omp_command(
                            omp_command(directory, selector=selector), cwd
                        ).selector,
                        selector,
                    )
                    header = cast(
                        dict[str, object],
                        omp_transcript(omp_output(cwd=directory, selector=selector))["header"],
                    )
                    self.assertEqual(header["selector"], selector)
            for selector in (gemini, opus):
                with self.subTest(writer=selector):
                    validate_omp_command(
                        omp_command(directory, selector=selector, writer=True), cwd
                    )
                    omp_transcript(
                        omp_output(
                            cwd=directory, selector=selector, policy="workspace-write-no-shell"
                        )
                    )
            with self.assertRaisesRegex(ValueError, "writer OMP selector"):
                validate_omp_command(omp_command(directory, selector=sonnet, writer=True), cwd)
            with self.assertRaisesRegex(ValueError, "grounded OMP selector"):
                validate_omp_command(
                    omp_command(directory, selector="google-antigravity/gemini-unknown"), cwd
                )

    def test_writer_transcripts_accept_only_the_allowlist_for_their_sdk_era(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for policy, old_tools in (
                (
                    "workspace-write-no-shell",
                    [
                        "execution_read",
                        "execution_glob",
                        "execution_grep",
                        "execution_write",
                        "execution_edit",
                    ],
                ),
                (
                    "workspace-write-shell",
                    [
                        "execution_read",
                        "execution_glob",
                        "execution_grep",
                        "execution_write",
                        "execution_edit",
                        "execution_shell",
                    ],
                ),
            ):
                with self.subTest(policy=policy):
                    events = [
                        json.loads(line)
                        for line in omp_output(cwd=directory, policy=policy).splitlines()
                    ]
                    events[0]["tools"] = old_tools
                    omp_transcript("\n".join(json.dumps(event) for event in events))
                    events[0]["tools"] = old_tools[:-1] + ["unapproved"]
                    with self.assertRaisesRegex(ValueError, "allowlist"):
                        omp_transcript("\n".join(json.dumps(event) for event in events))

    def test_shell_writer_command_and_native_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            command = omp_command(directory, shell_writer=True)
            invocation = validate_omp_command(command, root)
            self.assertEqual(invocation.policy, "workspace-write-shell")

            raw = omp_output(
                cwd=str(root),
                policy="workspace-write-shell",
                shell_command="pwd",
            )
            omp_transcript(raw, policy="workspace-write-shell", cwd=str(root))

            with self.assertRaisesRegex(ValueError, "unauthorized"):
                omp_transcript(omp_output(cwd=str(root), shell_command="pwd"))

            events = [json.loads(line) for line in raw.splitlines()]
            shell_start = next(
                event for event in events if event.get("type") == "tool_execution_start"
            )
            shell_start["args"]["timeout"] = 0
            with self.assertRaisesRegex(ValueError, "timeout"):
                omp_transcript(chr(10).join(json.dumps(event) for event in events))

            events = [json.loads(line) for line in raw.splitlines()]
            shell_end = next(event for event in events if event.get("type") == "tool_execution_end")
            shell_end["result"]["details"]["command"] = "different command"
            with self.assertRaisesRegex(ValueError, "disagrees"):
                omp_transcript(chr(10).join(json.dumps(event) for event in events))

            for field in (
                "paths",
                "command",
                "exit_code",
                "timed_out",
                "stdout",
                "stderr",
                "truncated",
            ):
                with self.subTest(field=field):
                    events = [json.loads(line) for line in raw.splitlines()]
                    shell_end = next(
                        event for event in events if event.get("type") == "tool_execution_end"
                    )
                    del shell_end["result"]["details"][field]
                    with self.assertRaises(ValueError):
                        omp_transcript(chr(10).join(json.dumps(event) for event in events))

            for field, value in (
                ("exit_code", True),
                ("timed_out", 1),
                ("stdout", None),
                ("stderr", 4),
                ("truncated", 0),
                ("paths", [directory, directory]),
            ):
                with self.subTest(malformed_field=field):
                    events = [json.loads(line) for line in raw.splitlines()]
                    shell_end = next(
                        event for event in events if event.get("type") == "tool_execution_end"
                    )
                    shell_end["result"]["details"][field] = value
                    with self.assertRaises(ValueError):
                        omp_transcript(chr(10).join(json.dumps(event) for event in events))

            for timeout in (True, 901):
                with self.subTest(timeout=timeout):
                    events = [json.loads(line) for line in raw.splitlines()]
                    shell_start = next(
                        event for event in events if event.get("type") == "tool_execution_start"
                    )
                    shell_start["args"]["timeout"] = timeout
                    with self.assertRaisesRegex(ValueError, "timeout"):
                        omp_transcript(chr(10).join(json.dumps(event) for event in events))

    def test_shell_evidence_admits_sdk_truncation_meta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            raw = omp_output(
                cwd=str(root),
                policy="workspace-write-shell",
                shell_command="find large-output",
                shell_meta={
                    "truncation": {
                        "direction": "middle",
                        "truncatedBy": "middle",
                        "totalLines": 2135,
                        "totalBytes": 174227,
                        "outputLines": 453,
                        "outputBytes": 40788,
                        "maxBytes": 40960,
                        "headRange": {"start": 1, "end": 216},
                        "tailRange": {"start": 1900, "end": 2135},
                        "elidedLines": 1683,
                        "elidedBytes": 133462,
                        "artifactId": "0",
                    }
                },
            )
            omp_transcript(raw, policy="workspace-write-shell", cwd=str(root))

    def test_shell_evidence_still_rejects_unexpected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            raw = omp_output(
                cwd=str(root),
                policy="workspace-write-shell",
                shell_command="find large-output",
                shell_meta={
                    "truncation": {
                        "direction": "middle",
                        "truncatedBy": "middle",
                        "totalLines": 2135,
                        "totalBytes": 174227,
                        "outputLines": 453,
                        "outputBytes": 40788,
                    }
                },
            )
            for mutation in (
                (
                    "meta carries a non-truncation key",
                    lambda details: details.update(
                        {"meta": {**details["meta"], "limits": {"matchLimit": {"reached": True}}}}
                    ),
                ),
                (
                    "unrelated extra top-level key",
                    lambda details: details.update({"environment": {"PATH": "/usr/bin"}}),
                ),
                (
                    "truncation is not an object",
                    lambda details: details["meta"].update({"truncation": "middle"}),
                ),
                (
                    "truncation direction is not a string",
                    lambda details: details["meta"]["truncation"].update({"direction": 1}),
                ),
                (
                    "truncation count is boolean",
                    lambda details: details["meta"]["truncation"].update({"totalLines": True}),
                ),
            ):
                with self.subTest(mutation=mutation[0]):
                    events = [json.loads(line) for line in raw.splitlines()]
                    shell_end = next(
                        event for event in events if event.get("type") == "tool_execution_end"
                    )
                    mutation[1](shell_end["result"]["details"])
                    with self.assertRaises(ValueError):
                        omp_transcript(chr(10).join(json.dumps(event) for event in events))

    def test_shell_evidence_admits_runtime_sources(self) -> None:
        truncation_meta: dict[str, object] = {
            "truncation": {
                "direction": "middle",
                "truncatedBy": "middle",
                "totalLines": 2135,
                "totalBytes": 174227,
                "outputLines": 453,
                "outputBytes": 40788,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def raw_with(
                *,
                shell_meta: dict[str, object] | None = None,
                shell_runtime_sources: dict[str, str] | None = None,
            ) -> str:
                return omp_output(
                    cwd=str(root),
                    policy="workspace-write-shell",
                    shell_command="find large-output",
                    shell_meta=shell_meta,
                    shell_runtime_sources=shell_runtime_sources,
                )

            # Post-ax20 workers record runtime provenance; pre-ax20 stored
            # transcripts lack the key entirely, and both must validate.
            omp_transcript(
                raw_with(shell_runtime_sources={"python3": "system", "ruff": "mise"}),
                policy="workspace-write-shell",
                cwd=str(root),
            )
            omp_transcript(
                raw_with(shell_meta=truncation_meta, shell_runtime_sources={"bun": "homebrew"}),
                policy="workspace-write-shell",
                cwd=str(root),
            )
            raw = raw_with(shell_runtime_sources={"python3": "system"})
            for mutation in (
                (
                    "runtime_sources names an unknown tool",
                    lambda details: details.update({"runtime_sources": {"make": "system"}}),
                ),
                (
                    "runtime_sources value is not a known source",
                    lambda details: details.update({"runtime_sources": {"python3": "compiled"}}),
                ),
                (
                    "runtime_sources is not an object",
                    lambda details: details.update({"runtime_sources": "mise"}),
                ),
            ):
                with self.subTest(mutation=mutation[0]):
                    events = [json.loads(line) for line in raw.splitlines()]
                    shell_end = next(
                        event for event in events if event.get("type") == "tool_execution_end"
                    )
                    mutation[1](shell_end["result"]["details"])
                    with self.assertRaises(ValueError):
                        omp_transcript(chr(10).join(json.dumps(event) for event in events))

    def test_remote_shell_evidence_does_not_follow_requester_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            remote_root = parent / "worker-snapshot"
            raw = omp_output(
                cwd=str(remote_root), policy="workspace-write-shell", shell_command="pwd"
            )
            before = omp_transcript(raw, policy="workspace-write-shell", cwd=str(remote_root))
            requester_target = parent / "unrelated-requester-directory"
            requester_target.mkdir()
            remote_root.symlink_to(requester_target, target_is_directory=True)
            after = omp_transcript(raw, policy="workspace-write-shell", cwd=str(remote_root))
            self.assertEqual(after, before)

    def test_shell_tool_refusal_can_be_followed_by_a_completed_response(self) -> None:
        for arguments in ({"command": "pwd"}, {"command": "pwd", "timeout": 120000}):
            with self.subTest(arguments=arguments):
                events = [
                    json.loads(line)
                    for line in omp_output(
                        cwd="/snapshot", policy="workspace-write-shell", shell_command="pwd"
                    ).splitlines()
                ]
                content = [{"type": "text", "text": "Sandbox refused command setup"}]
                for event in events:
                    if event.get("type") == "tool_execution_start":
                        event["args"] = arguments
                    elif event.get("type") == "tool_execution_end":
                        event["isError"] = True
                        event["result"] = {"content": content}
                    elif event.get("message", {}).get("role") == "toolResult":
                        event["message"]["isError"] = True
                        event["message"]["content"] = content
                    elif event.get("message", {}).get("role") == "assistant":
                        for block in event["message"]["content"]:
                            if block["type"] == "toolCall":
                                block["arguments"] = arguments
                transcript = omp_transcript("\n".join(json.dumps(event) for event in events))
                self.assertEqual(transcript["events"], events)

    def test_launch_never_forwards_google_keys(self) -> None:
        launch = omp_launch_environment(
            {
                "HOME": "/tmp/home",
                "GOOGLE_API_KEY": "key",
                "GEMINI_API_KEY": "key",
                "GOOGLE_GENERATIVE_AI_API_KEY": "key",
                "ZAI_API_KEY": "coding-plan",
            }
        )
        self.assertEqual(launch["ZAI_API_KEY"], "coding-plan")
        self.assertFalse(any("GOOGLE" in key or "GEMINI" in key for key in launch))

    def test_both_adapters_probe_all_antigravity_models_and_never_price_unknown_as_free(
        self,
    ) -> None:
        self.assertNotIn("google-antigravity", costs.OMP_SUBSCRIPTION_PROVIDERS)
        for adapter in ("omp", "omp-packet"):
            for model in ("gemini-3.1-pro", "claude-opus-4-6"):
                selector = f"google-antigravity/{model}"
                profile = {"adapter": adapter, "provider": "google-antigravity", "model": selector}
                for basis, source, expected in (
                    ("subscription", "oauth", 0.0),
                    ("api_key", "api_key", None),
                    ("unobserved", None, None),
                ):
                    with self.subTest(adapter=adapter, selector=selector, source=source):
                        observation = CredentialBasis("omp", basis, source, 0.0, "fingerprint")
                        with mock.patch(
                            "agent_execution.costs.observe_credential_basis",
                            return_value=observation,
                        ) as probe:
                            estimate = costs.estimate_execution_cost(profile, refresh=True)
                        probe.assert_called_once_with(
                            "omp",
                            profile=selector,
                            state_root=None,
                            cwd=None,
                            environment=None,
                            refresh=True,
                        )
                        self.assertEqual(estimate.maximum_incremental_usd, expected)
                        self.assertEqual(estimate.probe_answered, source is not None)
