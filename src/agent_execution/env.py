"""Environment settings owned by shared execution, not consumer admission policy."""

from __future__ import annotations

import os

PREFIX = "AGENT_EXECUTION_"


def variable(suffix: str) -> str | None:
    """Read one shared execution setting."""
    return os.environ.get(name(suffix))


def name(suffix: str) -> str:
    """The full variable name for one suffix."""
    return f"{PREFIX}{suffix}"
