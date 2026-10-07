"""Deterministic credential observations for worker process-boundary tests."""

import time
from datetime import datetime, timezone

from agent_execution.execution_status import cached_status


def admitted_status(**arguments):
    status = cached_status(**arguments)
    now = time.time()
    for row in status["rows"]:
        for name in ("capability", "authentication", "transport"):
            row["facts"][name] = {
                "state": "available",
                "observed_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "expires_at": datetime.fromtimestamp(now + 600, timezone.utc).isoformat(),
                "age_seconds": 0.0,
                "stale": False,
                "source": {"tool": "fixture", "method": "external-credential-observation"},
                "detail": "Execution permitted by isolated credential fixture",
                "condition": None,
            }
        row["facts"]["quota"] = {
            "state": "unknown",
            "observed_at": None,
            "expires_at": None,
            "age_seconds": None,
            "stale": True,
            "source": {"tool": "fixture", "method": "unobserved"},
            "detail": "No quota measurement",
            "condition": None,
        }
    return status
