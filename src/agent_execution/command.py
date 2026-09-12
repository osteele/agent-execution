"""Command execution facts and explicit consumer acknowledgment."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_execution.worker import WorkerResult

ProgressCallback = Callable[[str, Mapping[str, object]], None]
CommandRunner = Callable[[list[str], Path, float | None], "CommandResult"]


@dataclass(frozen=True)
class CommandResult:
    exit_status: int
    stdout: str
    stderr: str
    execution: dict[str, object] | None = None
    harness: dict[str, object] | None = None
    consumed: Callable[[], None] | None = None
    worker_result: WorkerResult | None = None

    def mark_consumed(self) -> None:
        """Acknowledge only after the consumer validates and durably records output."""
        if self.consumed is not None:
            self.consumed()


def run_command(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
    """Invoke a transport CLI; harness execution belongs to the shared worker."""
    completed = subprocess.run(
        command, capture_output=True, check=False, cwd=cwd, text=True, timeout=timeout
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)
