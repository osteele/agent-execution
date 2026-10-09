"""Bounded, progress-renewable execution budgets.

A renewable budget replaces one fixed harness wall-clock deadline with a
deadline that is extended only while qualifying progress is observed. The
policy is validated data (:class:`RenewableBudget`); :class:`BudgetClock`
holds one attempt's monotonic state and is the single owner of renewal
decisions; :func:`validate_budget_stats` is the versioned boundary for the
telemetry an attempt leaves behind in a worker result.

Nothing here resumes a process or retries a call. The absolute cap always
terminates the attempt, and every duration is measured on the injected
monotonic clock rather than on any wall clock.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

BUDGET_STATS_SCHEMA_VERSION = 1

#: Why one attempt ended. ``completed`` means the harness exited inside its
#: budget; the other two are exhaustion, and distinguish an attempt that ran
#: out of time outright from one that ran out of observable progress.
TERMINATION_REASONS = frozenset({"completed", "absolute_cap", "no_recent_progress"})

#: Renewal decisions recorded per deadline expiry. ``extended`` grants an
#: extension; the other two refuse one and end the attempt.
EXTENSION_DECISIONS = frozenset({"extended", "absolute_cap", "no_recent_progress"})

#: Decisions are naturally bounded by max_seconds / extension_seconds for any
#: sane policy. The record is capped regardless, so even a degenerate policy
#: cannot grow telemetry without limit; the counters keep counting.
MAX_DECISION_RECORDS = 1024


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _positive_number(value: object) -> bool:
    """Whether value is a finite positive number. Bools are not numbers here."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and _finite_number(value)
        and value > 0
    )


def _nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and _finite_number(value)
        and value >= 0
    )


def _nonnegative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


@dataclass(frozen=True)
class RenewableBudget:
    """A validated renewable-budget policy.

    ``initial_seconds`` bounds the attempt until its first renewal decision.
    Each qualifying renewal adds ``extension_seconds``, and qualifies only
    while the newest evidence is at most ``progress_window_seconds`` old.
    ``max_seconds`` is absolute: even continuous progress ends the attempt
    there.
    """

    initial_seconds: float
    extension_seconds: float
    progress_window_seconds: float
    max_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.initial_seconds,
            self.extension_seconds,
            self.progress_window_seconds,
            self.max_seconds,
        )
        if not all(_positive_number(value) for value in values):
            raise ValueError("renewable budget values must be finite positive numbers")
        if (
            self.initial_seconds > self.max_seconds
            or self.extension_seconds > self.max_seconds
            or self.progress_window_seconds > self.max_seconds
        ):
            raise ValueError("renewable budget values cannot exceed the absolute maximum")


class BudgetClock:
    """One attempt's deadline state under a renewable policy.

    The process owner is the only caller of :meth:`remaining`; output-drain
    threads record evidence through :meth:`progress`. One lock protects the
    deadline, the evidence, and the counters, so every decision is made
    against the same state the observers are writing — never a stale copy.
    """

    def __init__(
        self,
        policy: RenewableBudget,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.clock = clock
        self._lock = threading.Lock()
        self.started = clock()
        self.deadline = self.started + policy.initial_seconds
        self._evidence_at: float | None = None
        self._decisions: list[dict[str, object]] = []
        self._termination_reason: str | None = None
        self.progress_count = 0
        self.changed_file_count = 0

    def progress(self, *, changed_files: int = 0) -> None:
        """Record one qualifying-evidence observation, made just now."""
        with self._lock:
            self.progress_count += 1
            if changed_files > 0:
                self.changed_file_count += changed_files
            now = self.clock()
            # Late observations still count, but cannot erase eligible evidence
            # while the owner is waiting to decide, or revive a refused attempt.
            if self._termination_reason is None and now <= self.deadline:
                self._evidence_at = now

    def remaining(self) -> float:
        """Seconds left on the current deadline.

        Passing a deadline grants at most one decision, and a decision
        consumes its evidence: one observation cannot renew twice. The
        absolute cap is checked first, so continuous progress still ends the
        attempt at ``max_seconds`` instead of renewing forever.
        """
        with self._lock:
            now = self.clock()
            elapsed = now - self.started
            age = None if self._evidence_at is None else now - self._evidence_at
            if self._termination_reason is not None:
                return 0.0
            if elapsed >= self.policy.max_seconds:
                return self._refuse(elapsed, age, "absolute_cap")
            if now < self.deadline:
                return self.deadline - now
            if (
                age is not None
                and self._evidence_at is not None
                and self._evidence_at <= self.deadline
                and age <= self.policy.progress_window_seconds
            ):
                self.deadline = min(
                    self.deadline + self.policy.extension_seconds,
                    self.started + self.policy.max_seconds,
                )
                self._evidence_at = None
                self._record(elapsed, age, "extended")
                if self.deadline > now:
                    return self.deadline - now
                # A long scheduling pause can consume the whole extension
                # before this decision is observed. The evidence is spent, so
                # decide again immediately rather than return an unlabeled
                # zero.
                return self._refuse(now - self.started, None, "no_recent_progress")
            return self._refuse(elapsed, age, "no_recent_progress")

    @property
    def termination_reason(self) -> str:
        """Why the attempt ended, or ``completed`` while it is inside budget."""
        with self._lock:
            return self._termination_reason or "completed"

    def stats(self) -> dict[str, object]:
        """Versioned telemetry for this attempt, valid at a result boundary."""
        with self._lock:
            return {
                "schema_version": BUDGET_STATS_SCHEMA_VERSION,
                "policy": {
                    "initial_seconds": self.policy.initial_seconds,
                    "extension_seconds": self.policy.extension_seconds,
                    "progress_window_seconds": self.policy.progress_window_seconds,
                    "max_seconds": self.policy.max_seconds,
                },
                "duration_seconds": max(0.0, self.clock() - self.started),
                "termination_reason": self._termination_reason or "completed",
                "decisions": [dict(record) for record in self._decisions],
                "qualifying_progress_count": self.progress_count,
                "changed_file_count": self.changed_file_count,
            }

    def _record(
        self,
        at_seconds: float,
        evidence_age_seconds: float | None,
        decision: str,
    ) -> None:
        if len(self._decisions) < MAX_DECISION_RECORDS:
            self._decisions.append(
                {
                    "at_seconds": at_seconds,
                    "evidence_age_seconds": evidence_age_seconds,
                    "decision": decision,
                }
            )

    def _refuse(
        self,
        at_seconds: float,
        evidence_age_seconds: float | None,
        reason: str,
    ) -> float:
        self._record(at_seconds, evidence_age_seconds, reason)
        if self._termination_reason is None:
            self._termination_reason = reason
        return 0.0


def validate_budget_stats(raw: object) -> dict[str, object]:
    """Validate versioned budget telemetry, returning it unchanged when valid.

    Stored worker results are parsed through this boundary, so telemetry with
    an unknown schema version, a non-finite duration, an out-of-order policy,
    or an untyped decision is refused rather than trusted. Results from
    before renewable budgets carry no telemetry at all and never gain
    invented values.
    """
    if not isinstance(raw, dict):
        raise ValueError("budget telemetry must be an object")
    record = cast(dict[str, object], raw)
    if set(record) != {
        "schema_version",
        "policy",
        "duration_seconds",
        "termination_reason",
        "decisions",
        "qualifying_progress_count",
        "changed_file_count",
    }:
        raise ValueError("budget telemetry has an unexpected shape")
    schema_version = record.get("schema_version")
    if type(schema_version) is not int or schema_version != BUDGET_STATS_SCHEMA_VERSION:
        raise ValueError(f"unsupported budget telemetry schema: {schema_version!r}")
    policy_raw = record.get("policy")
    if not isinstance(policy_raw, dict):
        raise ValueError("budget telemetry policy must be an object")
    policy = cast(dict[str, object], policy_raw)
    if set(policy) != {
        "initial_seconds",
        "extension_seconds",
        "progress_window_seconds",
        "max_seconds",
    }:
        raise ValueError("budget telemetry policy has an unexpected shape")
    for name in (
        "initial_seconds",
        "extension_seconds",
        "progress_window_seconds",
        "max_seconds",
    ):
        if not _positive_number(policy.get(name)):
            raise ValueError(f"budget telemetry policy {name} must be a finite positive number")
    budget = RenewableBudget(
        initial_seconds=cast(float, policy["initial_seconds"]),
        extension_seconds=cast(float, policy["extension_seconds"]),
        progress_window_seconds=cast(float, policy["progress_window_seconds"]),
        max_seconds=cast(float, policy["max_seconds"]),
    )
    if (
        budget.initial_seconds > budget.max_seconds
        or budget.extension_seconds > budget.max_seconds
        or budget.progress_window_seconds > budget.max_seconds
    ):
        raise ValueError("budget telemetry policy exceeds its absolute maximum")
    duration = record.get("duration_seconds")
    if not _nonnegative_number(duration):
        raise ValueError("budget telemetry duration_seconds must be finite and non-negative")
    duration = cast(float, duration)
    reason = record.get("termination_reason")
    if not isinstance(reason, str) or reason not in TERMINATION_REASONS:
        raise ValueError(f"unknown budget telemetry termination reason: {reason!r}")
    decisions = record.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("budget telemetry decisions must be a list")
    if len(decisions) > MAX_DECISION_RECORDS:
        raise ValueError("budget telemetry carries too many recorded decisions")
    previous_at = 0.0
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("each recorded budget decision must be an object")
        item = cast(dict[str, object], decision)
        if set(item) != {"at_seconds", "evidence_age_seconds", "decision"}:
            raise ValueError("budget decision has an unexpected shape")
        at_seconds = item.get("at_seconds")
        if not _nonnegative_number(at_seconds):
            raise ValueError("budget decision at_seconds must be finite and non-negative")
        at_seconds = cast(float, at_seconds)
        if at_seconds < previous_at or at_seconds > duration:
            raise ValueError("budget decision times must be ordered within the attempt duration")
        previous_at = at_seconds
        age = item.get("evidence_age_seconds")
        if age is not None and not _nonnegative_number(age):
            raise ValueError(
                "budget decision evidence_age_seconds must be finite, non-negative, or null"
            )
        decision_name = item.get("decision")
        if not isinstance(decision_name, str) or decision_name not in EXTENSION_DECISIONS:
            raise ValueError(f"unknown budget decision: {decision_name!r}")
    progress_count = record.get("qualifying_progress_count")
    changed_file_count = record.get("changed_file_count")
    for name, value in (
        ("qualifying_progress_count", progress_count),
        ("changed_file_count", changed_file_count),
    ):
        if not _nonnegative_integer(value):
            raise ValueError(f"budget telemetry {name} must be a non-negative integer")
    return record
