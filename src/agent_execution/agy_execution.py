"""Packet-only Antigravity CLI (`agy`) execution for the remote worker.

The only enforcement known to keep `agy` from using tools is a deny-all
`PreToolUse` hook in a throwaway HOME, so this module admits exactly one tool
policy, ``packet-only-no-tools``. A tool-enabled `agy` (grounded read-only
review, or writing) is out of scope: read-only enforcement would need its own
design, and nothing here approximates it.

- Command: ``agy -p BRIEF --output-format json --print-timeout Ns
  --disable-slash-commands --model M``. A finite print timeout is required.
  ``--mode plan`` is refused because it suppresses the final message.
  ``--dangerously-skip-permissions`` is refused because it approves writes.
  The logical command also carries ``--execution-tool-policy
  packet-only-no-tools``. As with OMP, that flag is stripped here and never
  reaches the CLI.
- Isolation: the worker builds HOME itself. It never accepts an ``env HOME=...``
  prefix from a caller. The throwaway HOME copies the sign-in and settings
  entries in ``AGY_HOME_ALLOWLIST``, never the user's history. ``agy`` reads
  MCP servers from ``.gemini/config/mcp_config.json`` and hooks from
  ``.gemini/config/hooks.json``, so the user's copies of both are left out and
  replaced: an empty MCP configuration, and a hooks file holding only the
  deny-all ``PreToolUse`` hook. The user rules are replaced with a packet-only
  rule, and ``Library/Keychains`` and ``Library/Preferences``, where the
  credential lives, are symlinked. The whole HOME is removed after the call.
- Evidence: stdout is the JSON envelope. The envelope is accepted only when
  ``status`` is ``SUCCESS``, ``response`` is non-empty, and ``num_turns`` is
  positive. A print timeout returns ``SUCCESS`` with ``num_turns`` 0 and an
  empty response, which the turn check refuses.

Containment rests on the home alone: the deny-all hook, the empty MCP
configuration, and the rules file. The envelope cannot confirm it. Probed on
agy 1.2.16 (2026-10-04), a tool that ran and a tool the hook denied produced
envelopes with the same keys (``conversation_id``, ``duration_seconds``,
``num_turns``, ``response``, ``status``, ``usage``), so nothing in stdout says
whether a tool ran.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from subprocess import CompletedProcess

AGY_EXECUTABLE = "agy"
AGY_PACKET_POLICY = "packet-only-no-tools"
#: The provider route `agy` bills against, for shared provider status.
AGY_ROUTE = "google-antigravity"
#: Antigravity keeps Gemini quota separate from the pool shared by its other
#: models (spec/provider-scheduling.allium, `antigravity_*` QuotaPools).
AGY_GEMINI_POOL = "google-antigravity/gemini"
AGY_OTHER_POOL = "google-antigravity/other"
#: Models admitted to remote packet-only execution. This is an explicit list,
#: not a projection of `agy models`. Add an identity only after it has run.
AGY_MODELS = frozenset(
    {
        "gemini-3.1-pro-high",
        "gemini-3.8-flash-high",
        "claude-opus-4-6-thinking",
    }
)
#: The prompt marker the Weft conductor writes in place of the brief
#: (`WeftCommandRunner._prompt_from_stdin`). It is not known whether `agy -p`
#: reads a prompt from stdin. The worker therefore never sends the marker to
#: `agy`; it puts the payload bytes back on argv.
AGY_STDIN_PROMPT_MARKER = "-"
#: Largest brief accepted on argv. No size limit existed in agent-execution to
#: reuse. Linux caps one argv string at MAX_ARG_STRLEN (32 pages, 128 KiB), and
#: macOS caps all argv plus environment at 1 MiB, so 128 KiB is accepted on
#: both. A larger brief would fail in exec with E2BIG; it is refused at
#: preflight instead.
AGY_MAX_ARGV_PROMPT_BYTES = 128 * 1024
#: Bound on `--print-timeout`, so "finite" means a real limit as well as a number.
AGY_MAX_PRINT_TIMEOUT_SECONDS = 24 * 60 * 60

# Throwaway-HOME layout, relative to the HOME the child sees.
AGY_CONFIG_DIR = Path(".gemini")
AGY_SETTINGS_PATH = AGY_CONFIG_DIR / "settings.json"
AGY_MCP_CONFIG_PATH = AGY_CONFIG_DIR / "config" / "mcp_config.json"
AGY_HOOKS_CONFIG_PATH = AGY_CONFIG_DIR / "config" / "hooks.json"
AGY_RULES_PATH = AGY_CONFIG_DIR / "GEMINI.md"
AGY_DENY_HOOK_PATH = AGY_CONFIG_DIR / "hooks" / "agent-execution-deny-all.sh"
#: macOS credential stores, symlinked rather than copied.
AGY_LINKED_LIBRARY_DIRS = ("Keychains", "Preferences")
#: What a reviewer home copies from the real HOME: sign-in and settings, and
#: nothing that is the user's history. Measured 2026-10-03: a home of these six
#: entries (52 KB) signs in and answers, as the whole `.gemini` (697 MB) did.
#: Everything else -- `antigravity-cli/conversations` and `brain`, the Chromium
#: browser profile, browser recordings, `skills`, `tmp` -- is excluded, which is
#: also what the other harnesses' flags do (no sessions, no memory, no skills).
AGY_HOME_ALLOWLIST: tuple[Path, ...] = (
    AGY_CONFIG_DIR / "settings.json",
    AGY_CONFIG_DIR / "installation_id",
    AGY_CONFIG_DIR / "config",
    AGY_CONFIG_DIR / "antigravity-cli" / "antigravity-oauth-token",
    AGY_CONFIG_DIR / "antigravity-cli" / "settings.json",
    AGY_CONFIG_DIR / "antigravity-cli" / "installation_id",
)

_AGY_HOOK_NAME = "agent-execution-packet-only"

_DENY_REASON = (
    "agent-execution packet-only review: every tool is disabled; answer from the brief alone"
)
#: Denies every tool call through agy's decision channel: exit status 0 with a
#: JSON ``deny`` decision on stdout. Probed on agy 1.2.16, this is a clean
#: denial that carries the reason to the model. A nonzero exit also blocked,
#: but agy reported it as a failed hook and ignored the decision, so
#: containment would depend on how agy treats a crashing hook. stdin is
#: drained first so agy never sees a broken pipe.
_AGY_DENY_SCRIPT = f"""#!/bin/sh
# Installed by agent-execution for one packet-only agy call; removed afterwards.
cat >/dev/null 2>&1
reason='{_DENY_REASON}'
printf '{{"decision":"deny","reason":"%s","hookSpecificOutput":{{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}}}\\n' "$reason" "$reason"
exit 0
"""
_AGY_RULES = """# Packet-only review (installed by agent-execution)

Everything you need is in the prompt. Do not call any tool, read any file, run
any command, browse, or contact any server. Every tool call is denied. Answer
from the prompt alone, and put your complete answer in your final message.
"""
#: Inherited environment kept for the child. Everything else is dropped,
#: including provider API keys that could displace the signed-in credential
#: and XDG/config overrides that could point `agy` back at the real HOME.
_AGY_ENVIRONMENT_ALLOWED = frozenset(
    {
        "PATH",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
        "__CF_USER_TEXT_ENCODING",
    }
)
#: Envelope fields that would record tool use, refused if a later agy emits
#: them. agy 1.2.16 emits none of them, whether a tool ran or was denied, so
#: their absence is not evidence that no tool ran (see the module docstring).
_TOOL_RECORD_FIELDS = ("denied_actions", "tool_calls", "tool_uses", "tools_used")
_PRINT_TIMEOUT = re.compile(r"[1-9][0-9]*s")
_VALUE_FLAGS = frozenset(
    {
        "-p",
        "--output-format",
        "--print-timeout",
        "--model",
        "--execution-tool-policy",
    }
)
_BOOLEAN_FLAGS = frozenset({"--disable-slash-commands"})
#: Refused by name so the diagnostic says why: plan mode suppresses the final
#: message, and skipping permissions approves writes.
_REFUSED_FLAGS = frozenset({"--mode", "--dangerously-skip-permissions"})


class AgyStatusError(ValueError):
    """The envelope reports that agy itself did not succeed (not an evidence defect)."""


@dataclass(frozen=True)
class AgyInvocation:
    model: str
    policy: str
    prompt: str
    print_timeout_seconds: int


def agy_billing_pool(model: str) -> str:
    """The Antigravity quota pool a model draws from."""
    return AGY_GEMINI_POOL if model.startswith("gemini-") else AGY_OTHER_POOL


def require_agy_argv_prompt(prompt: str) -> None:
    """Refuse a brief that cannot be passed to agy safely as the `-p` argument."""
    if not prompt.strip():
        raise ValueError("agy requires a nonempty brief")
    if "\x00" in prompt:
        raise ValueError("agy brief contains a NUL byte, which argv cannot carry")
    if prompt.startswith("-"):
        # The brief must go on argv (stdin reading is unverified), and a CLI
        # parser can read a leading dash as an option rather than as the value
        # of -p. Refuse it rather than hope.
        raise ValueError("agy brief must not begin with '-'; argv parsers read it as an option")
    size = len(prompt.encode("utf-8"))
    if size > AGY_MAX_ARGV_PROMPT_BYTES:
        raise ValueError(
            f"agy brief is {size} bytes; argv delivery admits at most "
            f"{AGY_MAX_ARGV_PROMPT_BYTES} bytes"
        )


def validate_agy_command(command: list[str]) -> AgyInvocation:
    """Validate the exact packet-only agy grammar; anything else is refused."""
    if not command or Path(command[0]).name != AGY_EXECUTABLE:
        raise ValueError("agy execution requires the agy harness identity")
    values: dict[str, str] = {}
    booleans: set[str] = set()
    index = 1
    while index < len(command):
        flag = command[index]
        if flag in _VALUE_FLAGS:
            if flag in values or index + 1 >= len(command):
                raise ValueError(f"duplicate or missing agy argument: {flag}")
            values[flag] = command[index + 1]
            index += 2
        elif flag in _BOOLEAN_FLAGS:
            if flag in booleans:
                raise ValueError(f"duplicate agy argument: {flag}")
            booleans.add(flag)
            index += 1
        elif flag.split("=", 1)[0] in _REFUSED_FLAGS:
            raise ValueError(
                f"agy {flag} is refused: plan mode suppresses the final message and "
                "skipping permissions approves writes"
            )
        else:
            raise ValueError(f"unsafe or unsupported packet-only agy argument: {flag}")
    if "-p" not in values:
        raise ValueError("agy execution requires exactly one print prompt")
    if values.get("--output-format") != "json":
        raise ValueError("agy execution requires --output-format json")
    if booleans != _BOOLEAN_FLAGS:
        raise ValueError("agy execution requires --disable-slash-commands")
    policy = values.get("--execution-tool-policy", "")
    if policy != AGY_PACKET_POLICY:
        raise ValueError(
            f"agy admits only the {AGY_PACKET_POLICY} tool policy; "
            f"refusing {policy or 'a command without one'}"
        )
    timeout_text = values.get("--print-timeout", "")
    if not _PRINT_TIMEOUT.fullmatch(timeout_text):
        raise ValueError(
            "agy execution requires a finite --print-timeout in whole seconds, such as 600s"
        )
    print_timeout = int(timeout_text[:-1])
    if print_timeout > AGY_MAX_PRINT_TIMEOUT_SECONDS:
        raise ValueError(
            f"agy --print-timeout exceeds {AGY_MAX_PRINT_TIMEOUT_SECONDS}s; "
            "refusing an effectively unbounded call"
        )
    model = values.get("--model", "")
    if model not in AGY_MODELS:
        permitted = ", ".join(sorted(AGY_MODELS))
        raise ValueError(f"agy model must be admitted explicitly ({permitted}); got {model!r}")
    prompt = values["-p"]
    if prompt != AGY_STDIN_PROMPT_MARKER:
        require_agy_argv_prompt(prompt)
    return AgyInvocation(model, policy, prompt, print_timeout)


def agy_launch_command(executable: str, invocation: AgyInvocation, prompt: str) -> list[str]:
    """The argv agy actually receives, without the logical tool-policy flag."""
    require_agy_argv_prompt(prompt)
    return [
        executable,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--print-timeout",
        f"{invocation.print_timeout_seconds}s",
        "--disable-slash-commands",
        "--model",
        invocation.model,
    ]


def _hook_settings(script: Path) -> dict[str, object]:
    command = {"type": "command", "command": shlex.quote(str(script))}
    return {"PreToolUse": [{"matcher": "*", "hooks": [command]}]}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_agy_home(home: Path, *, source_home: Path) -> None:
    """Populate a throwaway HOME for one packet-only agy call."""
    target_config = home / AGY_CONFIG_DIR
    target_config.mkdir(parents=True)
    for entry in AGY_HOME_ALLOWLIST:
        source = source_home / entry
        target = home / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        # Followed, not preserved: a copied symlink could write through into
        # the real configuration.
        if source.is_dir():
            shutil.copytree(source, target, symlinks=False, ignore_dangling_symlinks=True)
        elif source.is_file():
            shutil.copy2(source, target)

    script = home / AGY_DENY_HOOK_PATH
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_AGY_DENY_SCRIPT, encoding="utf-8")
    script.chmod(0o700)

    settings_path = home / AGY_SETTINGS_PATH
    settings: dict[str, object] = {}
    if settings_path.is_file():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("copied agy settings.json is unreadable") from error
        if not isinstance(loaded, dict):
            raise ValueError("agy settings must be a JSON object")
        settings = cast(dict[str, object], loaded)
    # Replace, never merge: a user hook that allows a tool must not survive.
    settings["hooks"] = _hook_settings(script)
    if "mcpServers" in settings:
        settings["mcpServers"] = {}
    _write_json(settings_path, settings)
    # The copied `config` directory holds the user's MCP servers and hooks;
    # both files are overwritten, never merged.
    _write_json(home / AGY_MCP_CONFIG_PATH, {"mcpServers": {}})
    deny = {"type": "command", "command": shlex.quote(str(script)), "timeout": 10}
    _write_json(
        home / AGY_HOOKS_CONFIG_PATH,
        {_AGY_HOOK_NAME: {"PreToolUse": [{"matcher": "*", "hooks": [deny]}]}},
    )
    rules = home / AGY_RULES_PATH
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text(_AGY_RULES, encoding="utf-8")

    for name in AGY_LINKED_LIBRARY_DIRS:
        source = source_home / "Library" / name
        if source.exists():
            link = home / "Library" / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(source, target_is_directory=True)


@contextmanager
def agy_isolated_home(*, source_home: Path | None = None) -> Iterator[Path]:
    """A throwaway HOME that exists only for the duration of one call."""
    source = source_home
    if source is None:
        source = Path(os.environ.get("HOME") or str(Path.home()))
    root = Path(tempfile.mkdtemp(prefix="agent-execution-agy-home-"))
    try:
        build_agy_home(root, source_home=source)
        yield root
    finally:
        # rmtree unlinks the Library symlinks without following them.
        shutil.rmtree(root, ignore_errors=True)


def agy_launch_environment(
    home: Path, environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The minimal inherited environment, with HOME set to the isolated HOME."""
    source = os.environ if environment is None else environment
    launch = {key: value for key, value in source.items() if key in _AGY_ENVIRONMENT_ALLOWED}
    launch["HOME"] = str(home)
    return launch


def run_agy_command(
    command: list[str],
    cwd: Path,
    timeout: float | None,
    *,
    prompt: str | None = None,
    source_home: Path | None = None,
) -> CompletedProcess[str]:
    """Validate, isolate, and run one packet-only agy call in its own process group."""
    from agent_execution.processes import run_in_process_group

    invocation = validate_agy_command(command)
    if prompt is not None:
        if invocation.prompt != AGY_STDIN_PROMPT_MARKER:
            raise ValueError("agy payload execution requires the stdin prompt marker")
        brief = prompt
    elif invocation.prompt == AGY_STDIN_PROMPT_MARKER:
        raise ValueError("agy stdin prompt marker requires a prompt payload")
    else:
        brief = invocation.prompt
    argv = agy_launch_command(command[0], invocation, brief)
    with agy_isolated_home(source_home=source_home) as home:
        environment = agy_launch_environment(home)
        # stdin is closed at once: the brief is on argv, and nothing may wait
        # on input.
        return run_in_process_group(argv, cwd, "", timeout, environment=environment)


def _tool_use_recorded(envelope: Mapping[str, object]) -> str | None:
    for field in _TOOL_RECORD_FIELDS:
        if field not in envelope:
            continue
        value = envelope[field]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (list, dict, int)):
            return f"agy envelope field {field} is malformed"
        if value:
            return f"agy envelope records tool use in {field}"
    stats = envelope.get("stats")
    if isinstance(stats, dict):
        tools = cast(dict[str, object], stats).get("tools")
        if isinstance(tools, dict):
            calls = cast(dict[str, object], tools).get("totalCalls")
            if isinstance(calls, int) and not isinstance(calls, bool) and calls:
                return "agy envelope records tool use in stats.tools.totalCalls"
    return None


def agy_envelope(output: str) -> dict[str, object]:
    """Parse and validate one packet-only agy JSON envelope.

    Raises AgyStatusError when agy reports a non-SUCCESS status (the harness
    failed), and ValueError for every other defect (the evidence is unusable).
    """
    try:
        decoded = cast(object, json.loads(output))
    except json.JSONDecodeError as error:
        raise ValueError("agy output is not one JSON envelope") from error
    if not isinstance(decoded, dict):
        raise ValueError("agy envelope must be a JSON object")
    envelope = cast(dict[str, object], decoded)
    status = envelope.get("status")
    if status != "SUCCESS":
        detail = ""
        for key in ("error", "message", "response"):
            value = envelope.get(key)
            if isinstance(value, str) and value.strip():
                detail = value
                break
        raise AgyStatusError(
            f"agy reported status {status!r}" + (f": {detail.strip()}" if detail else "")
        )
    response = envelope.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("agy envelope has an empty response")
    tool_use = _tool_use_recorded(envelope)
    if tool_use is not None:
        raise ValueError(f"{tool_use}; packet-only execution admits none")
    turns = envelope.get("num_turns")
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 0:
        raise ValueError("agy envelope lacks a valid num_turns")
    if turns == 0:
        raise ValueError("agy envelope reports num_turns == 0; no model turn completed")
    return envelope


def agy_final_text(output: str) -> str:
    """The response text, as `omp_final_text` returns OMP's final message."""
    return cast(str, agy_envelope(output)["response"])
