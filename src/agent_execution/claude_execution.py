"""Policy-constrained native Claude CLI execution for remote reviews.

Logical commands are parsed before launch. The native launch is rebuilt from
that parsed identity, with process-local policy flags added here rather than
trusting caller-supplied flags to establish the safety boundary.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_execution.processes import run_in_process_group

CLAUDE_EXECUTABLE = "claude"
CLAUDE_GROUNDED_POLICY = "read-only-no-shell"
CLAUDE_PACKET_POLICY = "packet-only-no-tools"
CLAUDE_STDIN_PROMPT_MARKER = "-"
_NATIVE_RESOLVER_TIMEOUT = 10.0
_EMPTY_MCP_CONFIG = '{"mcpServers":{}}'
_MODEL_ID = re.compile(
    r"claude-(?:(?:opus|sonnet|haiku)-[0-9]+-[0-9]+(?:-[0-9]{8})?|"
    r"[0-9]+-[0-9]+-(?:opus|sonnet|haiku)-[0-9]{8}|"
    r"[0-9]-(?:opus|sonnet|haiku)-[0-9]{8})\Z"
)


def is_exact_claude_model(model: str) -> bool:
    """Recognize versioned native IDs, excluding moving aliases and provider selectors."""
    return _MODEL_ID.fullmatch(model) is not None


class ClaudeStatusError(ValueError):
    """Claude returned a native error/refusal rather than a usable result."""


@dataclass(frozen=True)
class ClaudeInvocation:
    command: list[str]
    prompt: str
    model: str
    policy: str
    session_id: str
    profile: str | None
    effort: str | None = None


def _uuid4(value: str) -> bool:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return str(parsed) == value and parsed.version == 4


def _logical_command(command: list[str]) -> list[str]:
    if not command:
        raise ValueError("Claude execution requires a command")
    args = list(command)
    if Path(args[0]).name == "env":
        # Only the exact shared credentials helper prefix is canonicalized.
        expected = ["env", "-u", "ANTHROPIC_API_KEY", "-u", "ANTHROPIC_AUTH_TOKEN"]
        if args[:5] != expected:
            raise ValueError("Claude execution refuses arbitrary env prefixes")
        index = 5
        profile = None
        if index < len(args) and args[index].startswith(
            "CLAUDE_WRAPPER_DISABLE_CONFIG_EXTRA_ARGS="
        ):
            if args[index] != "CLAUDE_WRAPPER_DISABLE_CONFIG_EXTRA_ARGS=1":
                raise ValueError("Claude execution requires wrapper extra arguments disabled")
            index += 1
        if index < len(args) and args[index].startswith("CLAUDE_PROFILE="):
            profile = args[index].partition("=")[2]
            raise ValueError("native Claude execution does not support wrapper profiles")
        if profile is not None or index == len(args):
            raise ValueError("invalid Claude env command prefix")
        if Path(args[index]).name != CLAUDE_EXECUTABLE:
            raise ValueError("Claude env prefix must select the claude harness")
        args = [CLAUDE_EXECUTABLE, *args[index + 1 :]]
    elif Path(args[0]).name != CLAUDE_EXECUTABLE:
        raise ValueError("Claude execution requires the claude harness identity")
    else:
        args[0] = CLAUDE_EXECUTABLE
    return args


def validate_claude_command(command: list[str], *, provider: str | None = None) -> ClaudeInvocation:
    """Parse the admitted logical Claude command grammar; refuse additions."""
    if provider not in (None, "claude", "claude-packet"):
        raise ValueError(f"unsupported Claude provider identity: {provider!r}")
    args = _logical_command(command)
    values: dict[str, str] = {}
    present: set[str] = set()
    value_flags = {
        "-p",
        "--input-format",
        "--output-format",
        "--mcp-config",
        "--permission-mode",
        "--allowedTools",
        "--model",
        "--session-id",
        "--tools",
        "--system-prompt",
        "--execution-tool-policy",
        "--effort",
    }
    bool_flags = {
        "--strict-mcp-config",
        "--safe-mode",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--restricted",
    }
    index = 1
    while index < len(args):
        arg = args[index]
        key = arg.split("=", 1)[0]
        if key in value_flags:
            if key in present:
                raise ValueError(f"duplicate Claude argument: {key}")
            if "=" in arg:
                value = arg.split("=", 1)[1]
            else:
                if index + 1 >= len(args):
                    raise ValueError(f"missing Claude argument value: {key}")
                index += 1
                value = args[index]
            values[key] = value
            present.add(key)
        elif key in bool_flags:
            if arg != key or key in present:
                raise ValueError(f"duplicate or malformed Claude argument: {arg}")
            present.add(key)
        else:
            raise ValueError(f"unsafe or unsupported Claude argument: {arg}")
        index += 1

    prompt = values.get("-p")
    model = values.get("--model", "")
    session_id = values.get("--session-id", "")
    if prompt is None:
        raise ValueError("Claude execution requires exactly one print prompt")
    if not prompt.strip() or "\x00" in prompt:
        raise ValueError("Claude prompt must be nonempty and contain no NUL")
    if values.get("--input-format") != "text" or values.get("--output-format") != "json":
        raise ValueError("Claude execution requires text input and JSON output")
    if "--strict-mcp-config" not in present or values.get("--mcp-config") != _EMPTY_MCP_CONFIG:
        raise ValueError("Claude execution requires an explicit empty strict MCP configuration")
    if not is_exact_claude_model(model):
        raise ValueError("Claude execution requires an exact native model ID")
    if not _uuid4(session_id):
        raise ValueError("Claude execution requires a fresh canonical UUIDv4 session ID")
    if "--resume" in args:
        raise ValueError("remote Claude session resume is refused")

    requested_policy = values.get("--execution-tool-policy")
    tool_set = values.get("--tools")
    if requested_policy is None:
        if tool_set == "":
            policy = CLAUDE_PACKET_POLICY
        elif values.get("--allowedTools") == "Read,Glob,Grep":
            policy = CLAUDE_GROUNDED_POLICY
        else:
            raise ValueError("cannot infer a safe Claude tool policy")
    else:
        policy = requested_policy
    if provider == "claude-packet":
        policy = CLAUDE_PACKET_POLICY
    elif provider == "claude":
        policy = CLAUDE_GROUNDED_POLICY
    if policy == CLAUDE_GROUNDED_POLICY:
        if tool_set is not None:
            raise ValueError("grounded Claude commands cannot declare packet tools")
        if (
            values.get("--permission-mode") != "plan"
            or values.get("--allowedTools") != "Read,Glob,Grep"
        ):
            raise ValueError("grounded Claude requires plan mode and exactly Read,Glob,Grep")
        if "--system-prompt" in present or "--no-session-persistence" in present:
            raise ValueError("grounded Claude refuses packet-only customizations")
    elif policy == CLAUDE_PACKET_POLICY:
        if tool_set != "" or not values.get("--system-prompt", "").strip():
            raise ValueError("packet Claude requires no tools and an explicit system prompt")
        if "--permission-mode" in present or "--allowedTools" in present:
            raise ValueError("packet Claude refuses tool-enabled options")
        if "--no-session-persistence" not in present:
            raise ValueError("packet Claude requires --no-session-persistence")
    else:
        raise ValueError(f"unsupported Claude execution policy: {policy!r}")
    if requested_policy not in (None, policy):
        raise ValueError("Claude policy flag conflicts with command variant")
    if "--disable-slash-commands" in present or "--restricted" in present:
        raise ValueError(
            "Claude safety flags are imposed by the native launcher, not caller supplied"
        )
    if prompt == CLAUDE_STDIN_PROMPT_MARKER:
        prompt = "-"
    effort = values.get("--effort")
    if effort is not None and effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Claude effort must be low, medium, high, xhigh, or max")
    normalized = ["claude", "-p", prompt]
    for flag, value in values.items():
        if flag != "-p":
            normalized.extend([flag, value])
    normalized.extend(sorted(present - values.keys()))
    return ClaudeInvocation(normalized, prompt, model, policy, session_id, None, effort)


def _scrub_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    source = os.environ if environment is None else environment
    if source.get("CLAUDE_PROFILE"):
        raise ValueError("native Claude execution does not support wrapper profiles")
    # Routing keys and caller-controlled model/provider overrides must not
    # displace the signed-in native identity or alter the requested model.
    blocked = {"CLAUDE_PROFILE", "CLAUDE_WRAPPER_DISABLE_CONFIG_EXTRA_ARGS"}
    launch = {
        key: value
        for key, value in source.items()
        if key not in blocked
        and not key.startswith(("ANTHROPIC_", "CLAUDE_WRAPPER_", "CLAUDE_CODE_USE_"))
        and key not in {"CLAUDE_CODE_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"}
    }
    launch["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return launch


def _executable_file(path: str | None) -> str | None:
    if not path:
        return None
    resolved = os.path.realpath(path)
    if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
        return resolved
    return None


def _is_native_binary(path: str) -> bool:
    """Only physical Mach-O/ELF executables bypass the public wrapper resolver."""
    with open(path, "rb") as stream:
        magic = stream.read(4)
    return magic in {
        b"\x7fELF",
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xce",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
        b"\xca\xfe\xba\xbf",
        b"\xbf\xba\xfe\xca",
    }


def native_claude_launch(
    environment: Mapping[str, str] | None = None, *, timeout: float = _NATIVE_RESOLVER_TIMEOUT
) -> tuple[str, dict[str, str]]:
    """Resolve a physical native executable and return its launch environment.

    The installed public launcher protocol is preferred. On ordinary installs
    the resolved command itself is accepted only when it is an executable file.
    """
    launch = _scrub_environment(environment)
    command_path = shutil.which(CLAUDE_EXECUTABLE, path=launch.get("PATH", os.defpath))
    if command_path is None:
        raise FileNotFoundError("native Claude executable is unavailable on PATH")
    command = _executable_file(command_path)
    if command is None:
        raise FileNotFoundError("Claude command did not resolve to an executable file")
    if _is_native_binary(command):
        return command, launch
    try:
        result = subprocess.run(
            [command, "wrapper", "native-binary"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=launch,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        lines = result.stdout.strip().splitlines()
        physical = _executable_file(lines[0].strip()) if len(lines) == 1 else None
        if physical is None or not _is_native_binary(physical):
            raise FileNotFoundError("Claude native-binary resolver returned an invalid executable")
        return physical, launch
    # Resolution failure must not execute a launcher whose route is unobserved.
    raise FileNotFoundError("Claude wrapper did not resolve a verified native executable")


def _native_command(executable: str, invocation: ClaudeInvocation) -> list[str]:
    command = [
        executable,
        "-p",
        "--input-format",
        "text",
        "--output-format",
        "json",
        "--strict-mcp-config",
        "--mcp-config",
        _EMPTY_MCP_CONFIG,
        "--model",
        invocation.model,
        "--session-id",
        invocation.session_id,
        "--safe-mode",
        "--disable-slash-commands",
    ]
    if invocation.policy == CLAUDE_GROUNDED_POLICY:
        command.extend(
            [
                "--permission-mode",
                "plan",
                "--allowedTools",
                "Read,Glob,Grep",
                "--restricted",
                "--tools",
                "Read,Glob,Grep",
            ]
        )
    else:
        command.extend(
            [
                "--tools",
                "",
                "--no-session-persistence",
                "--system-prompt",
                invocation.command[invocation.command.index("--system-prompt") + 1],
            ]
        )
    if invocation.effort is not None:
        command.extend(["--effort", invocation.effort])
    return command


def run_claude_command(
    command: list[str], cwd: Path, timeout: float | None, *, prompt: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the validated native command in a process group with prompt on stdin."""
    invocation = validate_claude_command(command)
    if prompt is None and invocation.prompt == CLAUDE_STDIN_PROMPT_MARKER:
        raise ValueError(
            "Claude '-' prompt marker requires its payload through the prompt argument"
        )
    executable, environment = native_claude_launch()
    payload = invocation.prompt if prompt is None else prompt
    if not payload.strip() or "\x00" in payload:
        raise ValueError("Claude prompt payload must be nonempty and contain no NUL")
    return run_in_process_group(
        _native_command(executable, invocation),
        Path(cwd),
        payload,
        timeout,
        environment=environment,
    )


def claude_envelope(
    stdout: str, *, model: str | None = None, session_id: str | None = None
) -> dict[str, object]:
    """Validate the native final JSON envelope and its exact run identities."""
    try:
        document = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Claude stdout is not a JSON envelope") from error
    if not isinstance(document, dict) or document.get("type") != "result":
        raise ValueError("Claude stdout is not a native result envelope")
    if document.get("is_error") is True or str(document.get("subtype", "")).startswith("error"):
        raise ClaudeStatusError(
            str(document.get("result") or document.get("subtype") or "Claude refused")
        )
    if document.get("subtype") not in (None, "success"):
        raise ValueError("Claude result envelope has an unknown subtype")
    result = document.get("result")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Claude result envelope has no nonempty final response")
    native_session = document.get("session_id")
    if not isinstance(native_session, str) or not _uuid4(native_session):
        raise ValueError("Claude result envelope has no valid session identity")
    if session_id is not None and native_session != session_id:
        raise ValueError("Claude result session identity does not match the requested session")
    usage = document.get("modelUsage")
    if not isinstance(usage, dict) or not usage:
        raise ValueError("Claude result envelope has no modelUsage identity")
    models = [
        key for key, value in usage.items() if isinstance(key, str) and isinstance(value, dict)
    ]
    if not models or len(models) != len(usage):
        raise ValueError("Claude result modelUsage is malformed")
    if model is not None and models != [model]:
        raise ValueError("Claude result model identity does not match the requested model")
    return document
