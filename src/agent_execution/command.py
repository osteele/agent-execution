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
    refused: Callable[[], None] | None = None
    #: Versioned renewable-budget telemetry from the process owner, present
    #: only when the call ran under an ``agent_execution.budget.RenewableBudget``.
    budget: dict[str, object] | None = None

    def mark_consumed(self) -> None:
        """Acknowledge only after the consumer validates and durably records output."""
        if self.consumed is not None:
            self.consumed()

    def acknowledge_refusal(self) -> None:
        """Acknowledge a result the consumer judged and durably recorded as refused.

        Unlike `mark_consumed`, this leaves `execution["processing"]` as the
        consumer wrote it, because consumers classify failures by that state
        and step; the Weft mark is recorded on `processed` and
        `processing_error` beside it, as for a worker-side refusal.
        """
        if self.refused is not None:
            self.refused()


def run_command(command: list[str], cwd: Path, timeout: float | None) -> CommandResult:
    """Invoke a transport CLI; harness execution belongs to the shared worker."""
    completed = subprocess.run(
        command, capture_output=True, check=False, cwd=cwd, text=True, timeout=timeout
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)
