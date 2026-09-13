"""Synthetic OMP 18.1.15 native events; never presented as a live transcript."""

from __future__ import annotations

import hashlib
import json

from agent_execution.omp_execution import (
    OMP_EXECUTION_SCHEMA,
    OMP_READ_TOOLS,
    OMP_SDK_VERSION,
    OMP_WRITE_TOOLS,
)


def omp_events(
    *,
    cwd: str,
    selector: str = "anthropic/claude-opus-5",
    policy: str = "read-only-no-shell",
    prompt: str = "review",
    final: str = "final response",
    read_path: str | None = None,
) -> list[dict]:
    provider, model = selector.split("/", 1)
    events = [
        {
            "type": "execution",
            "schema_version": OMP_EXECUTION_SCHEMA,
            "sdk_version": OMP_SDK_VERSION,
            "harness": "omp",
            "selector": selector,
            "tool_policy": policy,
            "tools": list(OMP_WRITE_TOOLS)
            if policy == "workspace-write-no-shell"
            else list(OMP_READ_TOOLS)
            if policy == "read-only-no-shell"
            else [],
            "session_id": "session-1",
            "cwd": cwd,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "execution_id": "00000000-0000-0000-0000-000000000001",
        },
        {"type": "session", "version": 3, "id": "session-1", "cwd": cwd},
    ]
    if read_path:
        content = [{"type": "text", "text": "served source bytes"}]
        events.extend(
            [
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "provider": provider,
                        "model": model,
                        "stopReason": "toolUse",
                        "content": [
                            {
                                "type": "toolCall",
                                "id": "call-1",
                                "name": "execution_read",
                                "arguments": {"path": read_path, "i": "Read snapshot source"},
                                "intent": "Read snapshot source",
                            }
                        ],
                    },
                },
                {
                    "type": "tool_execution_start",
                    "toolCallId": "call-1",
                    "toolName": "execution_read",
                    "args": {"path": read_path},
                    "intent": "Read snapshot source",
                },
                {
                    "type": "tool_execution_end",
                    "toolCallId": "call-1",
                    "toolName": "execution_read",
                    "result": {"content": content, "details": {"paths": [read_path]}},
                    "isError": False,
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "toolResult",
                        "toolCallId": "call-1",
                        "toolName": "execution_read",
                        "content": content,
                        "isError": False,
                    },
                },
            ]
        )
    events.extend(
        [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "provider": provider,
                    "model": model,
                    "stopReason": "stop",
                    "content": [{"type": "text", "text": final}],
                },
            },
            {"type": "agent_end", "messages": []},
            {"type": "execution_end", "session_id": "session-1"},
        ]
    )
    return events


def omp_output(
    *,
    cwd: str,
    selector: str = "anthropic/claude-opus-5",
    policy: str = "read-only-no-shell",
    prompt: str = "review",
    final: str = "final response",
    read_path: str | None = None,
) -> str:
    return "\n".join(
        json.dumps(event)
        for event in omp_events(
            cwd=cwd,
            selector=selector,
            policy=policy,
            prompt=prompt,
            final=final,
            read_path=read_path,
        )
    )


def omp_command(
    cwd: str,
    *,
    selector: str = "anthropic/claude-opus-5",
    packet: bool = False,
    writer: bool = False,
    prompt: str = "review",
) -> list[str]:
    command = [
        "omp",
        "-p",
        prompt,
        "--mode",
        "json",
        "--cwd",
        cwd,
        "--execution-tool-policy",
        "packet-only-no-tools"
        if packet
        else "workspace-write-no-shell"
        if writer
        else "read-only-no-shell",
        "--model",
        selector,
    ]
    if packet:
        if writer:
            raise ValueError("packet and writer are mutually exclusive")
        command.extend(
            [
                "--no-session",
                "--no-tools",
                "--no-lsp",
                "--no-extensions",
                "--no-skills",
                "--no-rules",
            ]
        )
    return command
