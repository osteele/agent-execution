"""Pinned OMP SDK launch and native event evidence, shared by both OMP policies.

The CLI's --tools flag is NOT a security allowlist in OMP 18.1.15. The shipped
SDK helper uses restrictToolNames and only its own confined filesystem tools.
Logical omp commands are validated and translated here; they never reach the
unrestricted CLI. The dependency root is provisioned only by the explicit runtime installer.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from subprocess import CompletedProcess

    from agent_execution.processes import ProcessIdentity

OMP_SDK_VERSION = "18.2.10"
OMP_EXECUTION_SCHEMA = "agent-execution.omp-execution/v1"
OMP_TRANSCRIPT_SCHEMA = "agent-execution.omp-transcript/v1"
#: Provider-qualified models admitted to the grounded SDK boundary. The same
#: registry drives requested-review selection, capability probes, command
#: validation, and transcript validation so a model upgrade cannot leave one
#: layer accepting an identity another rejects.
GROUNDED_OMP_SELECTORS_BY_PROVIDER: dict[str, str] = {
    "anthropic": "anthropic/claude-opus-5-5",
    "openai-codex": "openai-codex/gpt-6-sol",
    "zhipu-coding-plan": "zhipu-coding-plan/glm-5.3-flash",
    "kimi-code": "kimi-code/kimi-k2.5",
}
DEFAULT_GROUNDED_OMP_SELECTOR = GROUNDED_OMP_SELECTORS_BY_PROVIDER["anthropic"]
#: Provider defaults above are the automatic roster. Additional selectors are
#: admitted only when explicitly requested: Luna for economical focused work,
#: Astra for rare largest-model escalation.
GROUNDED_OMP_SELECTORS = frozenset(
    {
        *GROUNDED_OMP_SELECTORS_BY_PROVIDER.values(),
        "openai-codex/gpt-6-luna",
        "openai-codex/gpt-6-astra",
    }
)
#: Historical grounded selectors remain parseable for durable retrieval but
#: cannot pass launch validation after retirement.
GROUNDED_OMP_TRANSCRIPT_SELECTORS = frozenset(
    {*GROUNDED_OMP_SELECTORS, "anthropic/claude-opus-4-6", "anthropic/claude-opus-5"}
)
#: Automatic reviewers frozen before a selector upgrade may follow the current
#: route on retry. Caller-pinned selectors never use this map.
RETIRED_GROUNDED_OMP_SELECTOR_REPLACEMENTS: dict[str, str] = {
    "anthropic/claude-opus-4-6": DEFAULT_GROUNDED_OMP_SELECTOR,
    "anthropic/claude-opus-5": DEFAULT_GROUNDED_OMP_SELECTOR,
}
#: Provider keys that can displace a stored subscription credential. Restricted
#: OMP strips all of them. ``ZAI_API_KEY`` is deliberately absent: it is the
#: Zhipu coding-plan credential and remains in the launch environment.
OMP_TOKEN_BILLING_KEYS = (
    "ANTHROPIC_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "OPENAI_API_KEY",
)
OMP_READ_TOOLS = ("execution_read", "execution_glob", "execution_grep")
OMP_WRITE_TOOLS = (*OMP_READ_TOOLS, "execution_write", "execution_edit")
_PACKET = "packet-only-no-tools"
_GROUNDED = "read-only-no-shell"
_WRITER = "workspace-write-no-shell"
OMP_WRITER_SELECTORS = frozenset(
    {
        "kimi-code/k3",
        GROUNDED_OMP_SELECTORS_BY_PROVIDER["openai-codex"],
        "openai-codex/gpt-6-luna",
        "openai-codex/gpt-6-astra",
    }
)


def omp_sdk_root(*, environment: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environment is None else environment
    configured = env.get("AGENT_EXECUTION_OMP_SDK_ROOT")
    home = Path(env.get("HOME", str(Path.home())))
    if configured:
        configured = str(home / configured[2:]) if configured.startswith("~/") else configured
        return Path(configured).resolve()
    return home / ".local/share/agent-execution/omp-sdk" / OMP_SDK_VERSION


def require_omp_sdk(*, environment: Mapping[str, str] | None = None) -> tuple[Path, str]:
    root = omp_sdk_root(environment=environment)
    package = root / "node_modules/@oh-my-pi/pi-coding-agent/package.json"
    try:
        metadata = json.loads(package.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(
            f"OMP SDK unavailable at {root}; run agent-execution-worker install-omp"
        ) from error
    if not isinstance(metadata, dict) or metadata.get("version") != OMP_SDK_VERSION:
        raise ValueError(f"OMP SDK at {root} must be exactly {OMP_SDK_VERSION}")
    bun = shutil.which(
        "bun", path=None if environment is None else environment.get("PATH", os.defpath)
    )
    if not bun:
        raise ValueError("OMP SDK execution requires bun on PATH")
    return root, bun


def omp_auth_status_command(
    selector: str, *, environment: Mapping[str, str] | None = None
) -> list[str]:
    """Non-generating SDK credential-type probe, under the execution scrub."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", selector):
        raise ValueError("OMP auth status requires an exact provider/model selector")
    root, bun = require_omp_sdk(environment=environment)
    return [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--exec",
        str(root),
        bun,
        selector,
        "--auth-status",
        "",
    ]


@dataclass(frozen=True)
class OmpInvocation:
    selector: str
    policy: str
    prompt: str
    system_prompt: str


def validate_omp_command(
    command: list[str], cwd: Path, *, historical: bool = False
) -> OmpInvocation:
    """Validate exact grammar; historical retrieval may name retired read-only models."""
    if not command or Path(command[0]).name != "omp":
        raise ValueError("OMP execution requires the omp harness identity")
    values: dict[str, str] = {}
    booleans: set[str] = set()
    value_flags = {"--mode", "--cwd", "--model", "--system-prompt", "--execution-tool-policy"}
    boolean_flags = {
        "--no-session",
        "--no-tools",
        "--no-lsp",
        "--no-extensions",
        "--no-skills",
        "--no-rules",
    }
    index = 1
    prompt: str | None = None
    while index < len(command):
        flag = command[index]
        if flag == "-p":
            if prompt is not None or index + 1 >= len(command):
                raise ValueError("OMP command requires exactly one print prompt")
            prompt = command[index + 1]
            index += 2
        elif flag in value_flags:
            if flag in values or index + 1 >= len(command):
                raise ValueError(f"duplicate or missing OMP argument: {flag}")
            values[flag] = command[index + 1]
            index += 2
        elif flag in boolean_flags:
            if flag in booleans:
                raise ValueError(f"duplicate OMP argument: {flag}")
            booleans.add(flag)
            index += 1
        else:
            raise ValueError(f"unsafe or unsupported restricted OMP argument: {flag}")
    if prompt is None or values.get("--mode") != "json":
        raise ValueError("OMP execution requires one print prompt and JSON output")
    if "--cwd" not in values or (cwd / values["--cwd"]).resolve() != cwd.resolve():
        raise ValueError("OMP command cwd differs from execution snapshot")
    policy = values.get("--execution-tool-policy", "")
    if policy not in {_PACKET, _GROUNDED, _WRITER}:
        raise ValueError("OMP command lacks its enforced tool policy")
    selector = values.get("--model", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", selector):
        raise ValueError("OMP requires an exact provider/model selector")
    grounded_selectors = GROUNDED_OMP_TRANSCRIPT_SELECTORS if historical else GROUNDED_OMP_SELECTORS
    if policy == _GROUNDED and selector not in grounded_selectors:
        permitted = ", ".join(sorted(GROUNDED_OMP_SELECTORS))
        raise ValueError(f"grounded OMP selector must be registered: {permitted}")
    if policy == _WRITER and selector not in OMP_WRITER_SELECTORS:
        permitted = ", ".join(sorted(OMP_WRITER_SELECTORS))
        raise ValueError(f"writer OMP selector must be registered: {permitted}")
    if policy == _PACKET and booleans != boolean_flags:
        raise ValueError(
            "packet OMP must disable tools, session, extensions, skills, rules and LSP"
        )
    if policy in {_GROUNDED, _WRITER} and booleans:
        raise ValueError("tool-enabled OMP cannot masquerade as packet-only execution")
    return OmpInvocation(selector, policy, prompt, values.get("--system-prompt", ""))


def run_omp_command(
    command: list[str],
    cwd: Path,
    timeout: float | None,
    *,
    prompt: str | None = None,
    on_spawn: Callable[[ProcessIdentity], None] | None = None,
    hold_before_exec: bool = False,
    on_resource_sample: Callable[[Mapping[str, object]], None] | None = None,
) -> CompletedProcess[str]:
    from agent_execution.processes import run_in_process_group

    invocation = validate_omp_command(command, cwd)
    root, bun = require_omp_sdk()
    resolved_prompt = invocation.prompt if prompt is None else prompt
    if prompt is not None and invocation.prompt != "-":
        raise ValueError("OMP payload execution requires the stdin prompt marker")
    # Isolated Python ignores ambient PYTHONPATH and user startup hooks, then
    # scrubs the environment before Bun can honor preload options.
    # No broker secret is put on a command line.
    launch = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--exec",
        str(root),
        bun,
        invocation.selector,
        invocation.policy,
        invocation.system_prompt,
    ]
    return run_in_process_group(
        launch,
        cwd,
        resolved_prompt,
        timeout,
        on_spawn=on_spawn,
        hold_before_exec=hold_before_exec,
        on_resource_sample=on_resource_sample,
    )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"OMP {label} must be an object")
    return cast(dict[str, object], value)


def omp_transcript(
    output: str,
    *,
    selector: str | None = None,
    policy: str | None = None,
    cwd: str | None = None,
    prompt_sha256: str | None = None,
) -> dict[str, object]:
    """Validate authoritative native events, including every tool lifecycle pair."""
    events: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            events.append(_object(json.loads(line), "event"))
        except ValueError as error:
            raise ValueError("OMP output contains malformed event JSON") from error
    if len(events) < 4:
        raise ValueError("OMP output lacks a complete execution envelope")
    header = events[0]
    if header.get("type") != "execution" or header.get("schema_version") != OMP_EXECUTION_SCHEMA:
        raise ValueError("OMP output lacks restricted execution attestation")
    # Stored transcripts retain their original, previously supported SDK pin.
    if (
        header.get("sdk_version") not in ("18.1.15", OMP_SDK_VERSION)
        or header.get("harness") != "omp"
    ):
        raise ValueError("OMP execution has unsupported SDK or harness identity")
    selected = header.get("selector")
    selected_policy = header.get("tool_policy")
    if not isinstance(selected, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", selected
    ):
        raise ValueError("OMP execution lacks exact selector")
    if not isinstance(selected_policy, str) or selected_policy not in {_PACKET, _GROUNDED, _WRITER}:
        raise ValueError("OMP execution has unknown policy")
    if selected_policy == _GROUNDED and selected not in GROUNDED_OMP_TRANSCRIPT_SELECTORS:
        raise ValueError("OMP grounded inference identity was never registered")
    if selected_policy == _WRITER and selected not in OMP_WRITER_SELECTORS:
        raise ValueError("OMP writer inference identity was never registered")
    expected_tools = (
        list(OMP_READ_TOOLS)
        if selected_policy == _GROUNDED
        else list(OMP_WRITE_TOOLS)
        if selected_policy == _WRITER
        else []
    )
    if header.get("tools") != expected_tools:
        raise ValueError("OMP tool allowlist attestation mismatch")
    for key, expected in (
        ("selector", selector),
        ("tool_policy", policy),
        ("cwd", cwd),
        ("prompt_sha256", prompt_sha256),
    ):
        if expected is not None and header.get(key) != expected:
            raise ValueError(f"OMP {key} identity mismatch")
    declared_cwd = header.get("cwd")
    execution_id = header.get("execution_id")
    if not isinstance(declared_cwd, str) or not Path(declared_cwd).is_absolute():
        raise ValueError("OMP snapshot cwd is not absolute")
    if not isinstance(execution_id, str) or not re.fullmatch(r"[0-9a-f-]{36}", execution_id):
        raise ValueError("OMP execution identity is malformed")
    session_id = header.get("session_id")
    native = events[1]
    if (
        not isinstance(session_id, str)
        or not session_id
        or native.get("type") != "session"
        or native.get("id") != session_id
        or native.get("cwd") != header.get("cwd")
    ):
        raise ValueError("OMP native session identity mismatch")
    if native.get("version") != 3:
        raise ValueError("unsupported OMP native session version")
    digest = header.get("prompt_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("OMP prompt identity is malformed")
    if events[-1] != {"type": "execution_end", "session_id": session_id}:
        raise ValueError("OMP execution is incomplete")
    provider, model = selected.split("/", 1)
    calls: dict[str, dict[str, object]] = {}
    ended: dict[str, dict[str, object]] = {}
    requested: dict[str, tuple[str, object]] = {}
    returned: dict[str, dict[str, object]] = {}
    final_message: dict[str, object] | None = None
    agent_end = False
    for event in events[2:-1]:
        kind = event.get("type")
        if not isinstance(kind, str):
            raise ValueError("OMP event type is malformed")
        if kind in {"execution", "execution_end", "session"}:
            raise ValueError("OMP output contains a second execution/session")
        if kind == "message_end":
            message = _object(event.get("message"), "message")
            if message.get("role") == "assistant":
                if message.get("provider") != provider or message.get("model") != model:
                    raise ValueError("OMP assistant inference identity mismatch")
                content = message.get("content")
                if not isinstance(content, list):
                    raise ValueError("OMP assistant content is malformed")
                final_message = message
                for block in content:
                    item = _object(block, "content block")
                    if item.get("type") == "toolCall":
                        call_id = item.get("id")
                        if (
                            not isinstance(call_id, str)
                            or call_id in requested
                            or item.get("name") not in expected_tools
                        ):
                            raise ValueError(
                                "OMP assistant requested an unauthorized or duplicate tool"
                            )
                        arguments = _object(item.get("arguments"), "requested tool arguments")
                        # SDK intent tracing removes its reserved `i` field before execution.
                        execution_arguments = {
                            key: value for key, value in arguments.items() if key != "i"
                        }
                        requested[call_id] = (str(item["name"]), execution_arguments)
            elif message.get("role") == "toolResult":
                call_id = message.get("toolCallId")
                if not isinstance(call_id, str) or call_id in returned:
                    raise ValueError("OMP tool-result identity mismatch")
                returned[call_id] = message
        elif kind == "tool_execution_start":
            call_id = event.get("toolCallId")
            if (
                not isinstance(call_id, str)
                or call_id in calls
                or event.get("toolName") not in expected_tools
            ):
                raise ValueError("OMP tool execution is unauthorized or duplicated")
            _object(event.get("args"), "tool arguments")
            calls[call_id] = event
        elif kind == "tool_execution_end":
            call_id = event.get("toolCallId")
            if (
                not isinstance(call_id, str)
                or call_id not in calls
                or call_id in ended
                or event.get("toolName") != calls[call_id].get("toolName")
            ):
                raise ValueError("OMP tool lifecycle identity mismatch")
            result = _object(event.get("result"), "tool result")
            if not isinstance(result.get("content"), list):
                raise ValueError("OMP tool result lacks complete content")
            if "isError" in event and not isinstance(event["isError"], bool):
                raise ValueError("OMP tool result error flag is malformed")
            if not event.get("isError", False):
                details = _object(result.get("details"), "tool evidence")
                paths = details.get("paths")
                if not isinstance(paths, list) or any(
                    not isinstance(path, str) or not Path(path).is_absolute() for path in paths
                ):
                    raise ValueError("OMP tool evidence lacks absolute paths")
                root = Path(str(header.get("cwd")))
                if any(
                    ".." in Path(path).parts
                    or (Path(path) != root and root not in Path(path).parents)
                    for path in paths
                ):
                    raise ValueError("OMP tool evidence escapes snapshot")
            ended[call_id] = event
        elif kind == "agent_end":
            agent_end = True
    if (
        not agent_end
        or final_message is None
        or final_message.get("stopReason") not in ("stop", "length")
    ):
        raise ValueError("OMP lacks a successful terminal assistant message")
    if set(calls) != set(ended) or set(calls) != set(requested) or set(calls) != set(returned):
        raise ValueError("OMP tool evidence is incomplete")
    for call_id, call in calls.items():
        if requested[call_id] != (call.get("toolName"), call.get("args")):
            raise ValueError("OMP tool arguments disagree with assistant request")
        message = returned[call_id]
        result = cast(dict[str, object], ended[call_id]["result"])
        if message.get("toolName") != call.get("toolName") or message.get("content") != result.get(
            "content"
        ):
            raise ValueError("OMP served tool-result evidence differs from native result")
        if message.get("isError", False) != ended[call_id].get("isError", False):
            raise ValueError("OMP served tool-result error status differs")
    return {"schema_version": OMP_TRANSCRIPT_SCHEMA, "header": header, "events": events}


def validate_omp_transcript(
    raw: object,
    *,
    selector: str | None = None,
    policy: str | None = None,
    cwd: str | None = None,
    prompt_sha256: str | None = None,
) -> dict[str, object]:
    transcript = _object(raw, "transcript")
    if transcript.get("schema_version") != OMP_TRANSCRIPT_SCHEMA or not isinstance(
        transcript.get("events"), list
    ):
        raise ValueError("unsupported OMP transcript schema")
    validated = omp_transcript(
        "\n".join(json.dumps(event) for event in cast(list[object], transcript["events"])),
        selector=selector,
        policy=policy,
        cwd=cwd,
        prompt_sha256=prompt_sha256,
    )
    if transcript.get("header") != validated["header"]:
        raise ValueError("OMP transcript header differs from native stream")
    return validated


def omp_final_text(output: str) -> str:
    transcript = omp_transcript(output)
    for event in reversed(cast(list[dict[str, object]], transcript["events"])):
        if event.get("type") != "message_end":
            continue
        message = _object(event.get("message"), "message")
        if message.get("role") == "assistant":
            return "\n".join(
                str(block["text"])
                for block in cast(list[dict[str, object]], message["content"])
                if block.get("type") == "text" and isinstance(block.get("text"), str)
            )
    raise ValueError("OMP lacks a final assistant response")


def omp_launch_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the credential-minimal environment inherited by restricted OMP.

    Provider API keys are absent except ``ZAI_API_KEY``: that key is the
    Zhipu coding-plan credential rather than a metered-route override. OMP's
    stored or broker credentials remain reachable through its own auth store.
    """
    source = os.environ if environment is None else environment
    allowed = {
        "HOME",
        "PATH",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
        "OMP_AUTH_BROKER_URL",
        "OMP_AUTH_BROKER_TOKEN",
        "ZAI_API_KEY",
    }
    launch = {key: value for key, value in source.items() if key in allowed}
    launch.update({"PI_NO_TITLE": "1", "PI_NO_PTY": "1"})
    return launch


def _exec_sdk() -> None:
    _, root, bun, selector, policy, system_prompt = sys.argv[1:]
    if policy == _GROUNDED and selector not in GROUNDED_OMP_SELECTORS:
        raise SystemExit("grounded OMP selector is not registered")
    if policy == _WRITER and selector not in OMP_WRITER_SELECTORS:
        raise SystemExit("writer OMP selector is not registered")
    environment = omp_launch_environment()
    snapshot_cwd = os.getcwd()
    # Bun reads bunfig.toml before our code runs: bootstrap only in the trusted
    # SDK root, then restore the execution snapshot cwd inside the helper.
    os.chdir(root)
    os.execve(
        bun,
        [
            bun,
            "--no-env-file",
            str(Path(__file__).with_name("omp_sdk.ts")),
            root,
            selector,
            policy,
            system_prompt,
            snapshot_cwd,
        ],
        environment,
    )


if __name__ == "__main__":
    if len(sys.argv) != 7 or sys.argv[1] != "--exec":
        raise SystemExit("OMP helper is an internal restricted execution boundary")
    _exec_sdk()
