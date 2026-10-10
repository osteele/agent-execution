"""A fake native Claude install for boundary tests: real files, real resolution.

The public `claude` launcher is an executable shell script, exactly like an
ordinary install, and it answers the real wrapper protocol: `claude wrapper
native-binary` prints the single path recorded in the pointer file. The
pointed-at files carry real Mach-O magic bytes, so they pass the physical
binary check without being runnable scripts. Nothing here executes a harness:
the process boundary (`run_in_process_group`) is patched at its definition
site, never the resolver, the validator, or the launch binding.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from unittest import mock

SELECTOR = "anthropic/claude-opus-5-5"
MODEL = "claude-opus-5-5"
SESSION = "d93bfcf1-73f6-4692-af1c-1729e37e06a3"

#: Real Mach-O 64-bit magic followed by zero padding. `_is_native_binary`
#: reads four bytes; nothing else about the file is executed.
MACHO_MAGIC = b"\xcf\xfa\xed\xfe" + b"\x00" * 60

WRAPPER = """#!/bin/sh
if [ "$1" = "wrapper" ] && [ "$2" = "native-binary" ]; then
    # `read` is a shell builtin: the launch environment's PATH contains only
    # this wrapper directory, so no external utility is reachable here.
    IFS= read -r physical < "$FAKE_NATIVE_POINTER" || exit 70
    echo "$physical"
    exit 0
fi
echo "unsupported wrapper invocation" >&2
exit 64
"""


class FakeNativeInstall:
    """A test-owned native Claude install whose wrapper target can be moved.

    `point_to(path)` rewrites the pointer file, which is how a test moves the
    physical binary the real resolver selects between the worker's observation
    and its launch -- the production drift the launch binding exists to close.
    """

    def __init__(self, root: Path) -> None:
        bin_dir = root / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.wrapper = bin_dir / "claude"
        self.wrapper.write_text(WRAPPER)
        self.wrapper.chmod(0o755)
        self.pointer = root / "pointer.txt"
        self.native_a = root / "native-executable-a"
        self.native_a.write_bytes(MACHO_MAGIC)
        self.native_a.chmod(0o755)
        self.native_b = root / "native-executable-b"
        self.native_b.write_bytes(MACHO_MAGIC)
        self.native_b.chmod(0o755)
        self.point_to(self.native_a)

    def point_to(self, executable: Path) -> None:
        self.pointer.write_text(f"{executable}\n")

    def environment(self) -> dict[str, str]:
        return {
            "PATH": str(self.wrapper.parent),
            "FAKE_NATIVE_POINTER": str(self.pointer),
        }


def packet_command(prompt: str = "Answer only from this packet.") -> list[str]:
    """The admitted claude-packet logical command shape, without any prefix."""
    return [
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
        "--safe-mode",
        "--tools",
        "",
        "--no-session-persistence",
        "--system-prompt",
        prompt,
    ]


def packet_envelope(**updates: object) -> str:
    """The native result envelope a completed claude-packet run must produce."""
    import json

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


def probe_worker_surface(
    *,
    observed_state: str = "available",
    observed_basis: str = "subscription",
    help_output: str = "--safe-mode --tools --restricted",
    help_returncode: int = 0,
):
    """One real worker-surface status probe against the fake install.

    Only the process boundary is patched, at its definition site: the probe
    resolves the wrapper, verifies the physical binary, and records the launch
    binding exactly as a real worker preflight does. The fake auth observation
    reports a subscription basis so a capped launch would be admitted; tests
    that need a different basis pass `observed_state`/`observed_basis`.
    """
    from agent_execution import credentials, execution_status

    basis = credentials.CredentialBasis(
        "claude", observed_basis, "none", time.time(), "fixture-fingerprint"
    )
    observed = mock.Mock(state=observed_state, answered=True, basis=basis)
    with (
        mock.patch(
            "agent_execution.execution_status.run_in_process_group",
            return_value=subprocess.CompletedProcess(
                ["claude", "--help"], help_returncode, help_output, ""
            ),
        ),
        mock.patch("agent_execution.credentials.observe_claude_auth", return_value=observed),
    ):
        return execution_status.probe_status(
            harness="claude-packet",
            surface="worker",
            selectors=[SELECTOR],
            transport="weft",
        )


def process_boundary(module: str, side_effect: Callable[..., object]):
    """Patch `run_in_process_group` where the given module imported it."""
    return mock.patch(f"{module}.run_in_process_group", side_effect=side_effect)
