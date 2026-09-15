"""Publish what this worker learns about a provider route's availability.

Three systems choose provider routes and only this one observes a refusal:
agent-review cuts reviewer anchors, the coding-delegate picker chooses
delegates, and agent-execution runs the harness. Nothing carried an observation
from the component that saw it to the components that chose, so on 2026-09-15 a
review cycle froze an anchor on `zhipu-coding-plan` whose cap had days left to
run and every retry re-dispatched into the same refusal.

The contract is a file, not an API, because the picker is a standalone script
that can never import a package. This module owns the *writing* half; readers
keep their own thin loaders against the same schema.

## The file

`~/.claude/state/provider-health.json`, or `$AGENT_PROVIDER_HEALTH` when set.

    {
      "schema_version": "provider-health/v1",
      "routes": {
        "zhipu-coding-plan": {
          "condition": "quota",
          "observed_at": 1789457130,
          "cold_until": 1789700716,
          "reset_at": "2026-09-17 23:45:16",
          "window": "weekly-or-monthly",
          "observed_by": "agent-execution",
          "signature": "...(type=1310)"
        }
      }
    }

Keys are OMP provider routes (`zhipu-coding-plan`, `openai-codex`, `kimi-code`,
`anthropic`) because that is the vocabulary agent-review already speaks.
`ROUTE_ALIASES` maps shorter provider names onto them.

## What a record means

An observation, not authority. The observer can be stale, or describing its own
network rather than the provider, so a consumer may prefer a route without a
live record and may refuse one carrying it, but every consumer keeps an
override. This is the stance weft takes toward host capability observations,
for the same reason.

`cold_until` is operative. `reset_at` is retained when the provider stated one,
because it is better evidence than a cooldown we would invent: a cap measured
in days must not inherit a fallback measured in minutes.

`window` records which allowance was exhausted. It is descriptive and it
matters: a route capped on its *monthly* allowance can still report ample
weekly headroom, so a consumer reading only a weekly percentage concludes the
route is healthy while every call fails. That is exactly what `kimi-code` did.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "provider-health/v1"

#: Shorter provider names mapped onto OMP routes. Peers speak routes.
ROUTE_ALIASES = {
    "glm": "zhipu-coding-plan",
    "zai": "zhipu-coding-plan",
    "codex": "openai-codex",
    "kimi": "kimi-code",
    "claude": "anthropic",
}

#: (pattern, condition, window, cooldown seconds). None takes the cooldown from
#: a reset timestamp in group 1; -1 takes it from an hour count in group 1.
SIGNATURES: list[tuple[str, str, str | None, int | None]] = [
    (
        r"限额将在\s*(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s*重置",
        "quota",
        "weekly-or-monthly",
        None,
    ),
    (r"您已达到每周/每月使用上限", "quota", "weekly-or-monthly", 24 * 3600),
    (r"You've reached your (\d+)-hour usage limit", "quota", "session", -1),
    (r"reached your monthly (?:usage )?limit", "quota", "monthly", 7 * 24 * 3600),
    (r"provider\.auth_error:\s*403", "quota", "session", 5 * 3600),
    (r"OAuth request .*token failed", "auth", None, 24 * 3600),
    (r"Cannot connect to API", "network", None, 300),
]

_DEFAULT_COOLDOWN = 600
_MINIMUM_QUOTA_COOLDOWN = 3600


def path() -> Path:
    override = os.environ.get("AGENT_PROVIDER_HEALTH")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "state" / "provider-health.json"


def route_for(name: str) -> str:
    """Normalize a provider name onto its OMP route."""
    return ROUTE_ALIASES.get(name, name)


def seconds_until(stamp: str) -> int:
    """Seconds until a provider-stated reset, floored at an hour.

    The timestamp carries no zone. Reading it as local time is deliberate: an
    error in the other direction clears the cooldown while the cap still binds,
    which is the condition this exists to prevent.
    """
    try:
        # DTZ007 wants a zone in the format. The provider does not supply one,
        # so there is no honest %z to add; the zone is decided on the next line.
        parsed = datetime.strptime(  # noqa: DTZ007
            stamp.strip().replace("T", " "), "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return 24 * 3600
    # `.astimezone()` on a naive value interprets it in the local zone. Stated
    # here rather than left implicit, because which zone this means is the whole
    # question and getting it wrong in one direction is harmless.
    reset = parsed.astimezone()
    return max(_MINIMUM_QUOTA_COOLDOWN, int(reset.timestamp() - time.time()))


def classify(signature: str) -> tuple[str, str | None, int, str | None]:
    """Map a provider's own words onto (condition, window, cooldown, reset_at).

    Matching is on the signature, never on an exit status: an OMP harness that
    exits 0 having been refused is the normal case, and one that exits non-zero
    may have failed locally and say nothing about the provider.
    """
    for pattern, condition, window, cooldown in SIGNATURES:
        match = re.search(pattern, signature or "", re.IGNORECASE)
        if not match:
            continue
        if cooldown is None:
            stamp = match.group(1)
            return condition, window, seconds_until(stamp), stamp
        if cooldown == -1:
            hours = int(match.group(1)) if match.groups() else 5
            return condition, window, hours * 3600, None
        return condition, window, cooldown, None
    return "unknown", None, _DEFAULT_COOLDOWN, None


def load() -> dict[str, dict]:
    """Read the file. A missing or unreadable file means no observations.

    A malformed or future-schema file is treated the same way. Refusing to
    select anything because this file cannot be parsed would make an advisory
    signal load-bearing, which is precisely what it is not.
    """
    try:
        raw = json.loads(path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        return {}
    routes = raw.get("routes")
    return routes if isinstance(routes, dict) else {}


def _write(routes: dict[str, dict]) -> None:
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(f".{target.name}.tmp")
    scratch.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "routes": routes}, indent=1) + "\n"
    )
    scratch.replace(target)


def record(route: str, signature: str, *, observed_by: str = "agent-execution") -> dict:
    """Publish one observation for a route and return it. Last writer wins."""
    normalized = route_for(route)
    condition, window, cooldown, reset_at = classify(signature)
    now = int(time.time())
    entry: dict[str, object] = {
        "condition": condition,
        "observed_at": now,
        "cold_until": now + cooldown,
        "observed_by": observed_by,
        "signature": signature[:500],
    }
    if reset_at:
        entry["reset_at"] = reset_at
    if window:
        entry["window"] = window
    routes = load()
    routes[normalized] = entry
    _write(routes)
    return entry


def clear(route: str) -> None:
    """Drop a route's observation.

    A success is evidence the record is stale, and a route that recovers early
    should not stay cold until a timestamp we inferred.
    """
    routes = load()
    if routes.pop(route_for(route), None) is None:
        return
    _write(routes)
