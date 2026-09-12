"""Wall, active, and boot-qualified clock readings."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

SUSPENSION_JITTER_SECONDS = 0.05


def current_boot_id() -> str | None:
    """Return the operating system's boot identity, when readable."""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "kern.bootsessionuuid"],
                capture_output=True,
                check=False,
                text=True,
            )
        except (FileNotFoundError, PermissionError, OSError):
            return None
        if result.returncode != 0:
            return None
        # kern.boottime is wall-clock based and changes under clock correction.
        # A boot-session UUID remains stable for the lifetime of the kernel.
        try:
            identity = UUID(result.stdout.strip())
        except ValueError:
            return None
        return f"macos-session:{identity}"
    if sys.platform.startswith("linux"):
        try:
            value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return f"linux:{value}" if value else None
    return None


@dataclass(frozen=True)
class ClockReading:
    wall_time: float
    monotonic_time: float
    boot_id: str | None


ClockReader = Callable[[], ClockReading]


def read_clock() -> ClockReading:
    """Read the wall and active clocks under one boot identity."""
    return ClockReading(
        wall_time=time.time(),
        monotonic_time=time.monotonic(),
        boot_id=current_boot_id(),
    )


@dataclass(frozen=True)
class ElapsedTiming:
    wall_elapsed_seconds: float
    active_elapsed_seconds: float | None
    suspended_seconds: float | None


def elapsed_timing(start: ClockReading, end: ClockReading) -> ElapsedTiming:
    """Measure wall span and boot-qualified active span between two readings."""
    wall_elapsed = end.wall_time - start.wall_time
    active_elapsed: float | None = None
    suspended: float | None = None
    if (
        start.boot_id is not None
        and start.boot_id == end.boot_id
        and end.monotonic_time >= start.monotonic_time
    ):
        active_elapsed = end.monotonic_time - start.monotonic_time
        gap = wall_elapsed - active_elapsed
        suspended = gap if gap > SUSPENSION_JITTER_SECONDS else 0.0
    return ElapsedTiming(
        wall_elapsed_seconds=wall_elapsed,
        active_elapsed_seconds=active_elapsed,
        suspended_seconds=suspended,
    )


def exclude_timing(total: ElapsedTiming, excluded: ElapsedTiming | None) -> ElapsedTiming:
    """Remove an interval that was outside the work being measured."""
    if excluded is None:
        return total
    wall_elapsed = max(0.0, total.wall_elapsed_seconds - excluded.wall_elapsed_seconds)
    active_elapsed: float | None = None
    suspended: float | None = None
    if total.active_elapsed_seconds is not None and excluded.active_elapsed_seconds is not None:
        active_elapsed = max(
            0.0,
            total.active_elapsed_seconds - excluded.active_elapsed_seconds,
        )
        gap = wall_elapsed - active_elapsed
        suspended = gap if gap > SUSPENSION_JITTER_SECONDS else 0.0
    return ElapsedTiming(
        wall_elapsed_seconds=wall_elapsed,
        active_elapsed_seconds=active_elapsed,
        suspended_seconds=suspended,
    )
