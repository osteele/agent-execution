"""Validated, exact-context observations backed by the provider event store."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from agent_execution import credentials, provider_status
from agent_execution.agy_execution import AGY_MODELS
from agent_execution.claude_execution import is_exact_claude_model
from agent_execution.omp_execution import (
    GROUNDED_OMP_SELECTORS,
    OMP_WRITER_SELECTORS,
    omp_writer_probe_command,
    require_omp_sdk,
    validate_writer_probe_evidence,
)
from agent_execution.processes import run_in_process_group

# JSON is dynamic only at this validated public boundary.
Json = dict[str, Any]
SCHEMA = "agent-execution.execution-status/v1"
FACTS = ("capability", "authentication", "quota", "generation", "transport")
TTL = 600


def _iso(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("observation timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid observation timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("observation timestamp must include a timezone")
    return parsed.timestamp()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError(f"invalid {name}")
    return value


def _object(value: object, name: str) -> Json:
    if not isinstance(value, dict):
        raise ValueError(f"invalid {name}")
    return cast(Json, value)


def _identity(expected: str | None) -> Json:
    from agent_execution.worker import installed_worker_identity

    identity = installed_worker_identity().to_dict()
    if expected is not None and identity["source_sha256"] != expected:
        raise ValueError("execution source identity differs from the requested build")
    return identity


def _selectors(harness: str, selectors: list[str] | None) -> list[str | None]:
    if selectors is None:
        if harness in {"omp", "omp-packet", "agy", "claude-packet"}:
            raise ValueError("this harness requires an exact selector")
        return [None]
    if not isinstance(selectors, list) or not selectors:
        raise ValueError("selectors must be a nonempty list")
    if any(
        not isinstance(item, str) or not re.fullmatch(provider_status.EXACT_SELECTOR_PATTERN, item)
        for item in selectors
    ):
        raise ValueError("selectors must be exact provider/model strings")
    if len(set(selectors)) != len(selectors):
        raise ValueError("selectors must be unique")
    return list(selectors)


def _policy(harness: str, surface: str, policy: str | None) -> str:
    if policy is None:
        if surface == "native":
            raise ValueError("native status requires an explicit tool policy")
        policy = (
            "workspace-write-no-shell"
            if surface == "offload-task"
            else "read-only-no-shell"
            if harness in {"omp", "claude"}
            else "packet-only-no-tools"
        )
    if policy not in provider_status.EXECUTION_TOOL_POLICIES:
        raise ValueError("unknown execution tool policy")
    return policy


def _subject(
    harness: str,
    surface: str,
    selector: str | None,
    transport: str,
    requester_host: str | None,
    requester_user: str | None,
    build: str,
    tool_policy: str | None,
    profile: str | None,
    environment: Mapping[str, str] | None,
    cwd: Path | None,
) -> Json:
    if (
        harness not in provider_status.EXECUTION_HARNESSES
        or surface not in provider_status.EXECUTION_SURFACES
    ):
        raise ValueError("unknown execution harness or surface")
    if transport not in provider_status.EXECUTION_TRANSPORTS:
        raise ValueError("unknown execution transport")
    worker_claude = surface == "worker" and harness in {"claude", "claude-packet"}
    if worker_claude and profile is not None:
        raise ValueError("native Claude workers do not support wrapper profiles")
    if worker_claude and selector is None:
        raise ValueError("Claude workers require an exact selector")
    launch_harness = "claude" if harness == "claude-packet" else harness
    if worker_claude:
        from agent_execution.claude_execution import native_claude_launch

        inherited = dict(os.environ if environment is None else environment)
        try:
            executable, launch = native_claude_launch(environment=inherited)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            executable, launch = "", inherited
        if executable:
            path = launch.get("PATH", os.defpath)
            launch = {**launch, "PATH": f"{Path(executable).parent}{os.pathsep}{path}"}
            executable_identity = credentials._executable_identity(executable)
        else:
            executable_identity = "unavailable"
        # SSH connection ports and process-local environment values must not
        # create a new subject on every observation of the same native route.
        launch_parts = [
            executable_identity,
            credentials.launch_fingerprint("claude", launch),
            f"oauth-token={'present' if launch.get('CLAUDE_CODE_OAUTH_TOKEN') else 'absent'}",
            f"simple={launch.get('CLAUDE_CODE_SIMPLE', '')}",
        ]
        fingerprint = "native-claude:" + hashlib.sha256("|".join(launch_parts).encode()).hexdigest()
    else:
        launch = credentials.harness_launch_environment(
            launch_harness, profile=profile, environment=environment
        )
        fingerprint = credentials.launch_fingerprint(launch_harness, launch)
    if launch_harness == "claude":
        fingerprint = hashlib.sha256(
            f"{fingerprint}:{(cwd or Path.cwd()).resolve()}".encode()
        ).hexdigest()
    route = selector.split("/", 1)[0] if selector else provider_status.route_for(launch_harness)
    host, user = socket.gethostname(), getpass.getuser()
    subject: Json = {
        "harness": harness,
        "surface": surface,
        "selector": selector,
        "tool_policy": _policy(harness, surface, tool_policy),
        "route": route,
        "billing_pool": provider_status.billing_pool_for(route, selector),
        "host": host,
        "os_user": user,
        "transport": transport,
        "requester_host": requester_host or host,
        "requester_user": requester_user or user,
        "execution_sha256": build,
        "profile": (
            (profile or launch.get("CLAUDE_PROFILE"))
            if harness == "claude" and not worker_claude
            else None
        ),
        "environment_fingerprint": fingerprint,
        "effective_route": (
            "anthropic"
            if worker_claude and executable
            else None
            if worker_claude or harness == "claude"
            else route
        ),
        "credential_fingerprint": provider_status.credential_fingerprint(None, host=host)[0],
        "fingerprint_scope": "host",
    }
    provider_status.validate_execution_identity(_scope(subject))
    return subject


def _scope(subject: Json) -> Json:
    return {key: subject[key] for key in provider_status.EXECUTION_IDENTITY_FIELDS}


def _unknown(name: str) -> Json:
    return {
        "state": "unknown",
        "observed_at": None,
        "expires_at": None,
        "age_seconds": None,
        "stale": True,
        "source": {"tool": "agent-execution", "method": "unobserved"},
        "detail": f"No {name} observation for this execution context",
        "condition": None,
    }


def _fact(event: Json, now: float) -> Json:
    payload = event["fact"]
    return {
        "state": payload["state"],
        "observed_at": event["observed_at"],
        "expires_at": event["expires_at"],
        "age_seconds": max(0.0, now - _timestamp(event["observed_at"])),
        "stale": _timestamp(event["expires_at"]) <= now,
        "source": event["source"],
        "detail": payload.get("detail", payload.get("reason", "Observed")),
        "condition": payload.get("condition"),
    }


def _record(
    subject: Json,
    name: str,
    state: str,
    detail: str,
    *,
    method: str,
    condition: str | None = None,
    basis: Json | None = None,
) -> Json:
    extra: Json = {"detail": detail}
    if condition is not None:
        extra["condition"] = condition
    if basis is not None:
        extra["credential_basis"] = basis
    return provider_status.observe(
        subject["route"],
        kind=name,
        state=state,
        source_tool="agent-execution",
        source_method=method,
        host=subject["host"],
        os_user=subject["os_user"],
        billing_pool=subject["billing_pool"],
        execution_identity=_scope(subject),
        ttl_seconds=TTL,
        detail=extra,
    )


def _cached_row(subject: Json, now: float) -> tuple[Json, list[str]]:
    events, diagnostics = provider_status.exact_observations(
        route=subject["route"],
        billing_pool=subject["billing_pool"],
        host=subject["host"],
        os_user=subject["os_user"],
        execution=_scope(subject),
        now=now,
    )
    if events:
        latest = max(events.values(), key=lambda event: _timestamp(event["observed_at"]))
        observed_subject = cast(Json, latest["subject"])
        subject = {
            **subject,
            **observed_subject["execution"],
            "billing_pool": observed_subject["billing_pool"],
        }
    facts = {name: _fact(events[name], now) if name in events else _unknown(name) for name in FACTS}
    basis = None
    if "credential-basis" in events:
        event = events["credential-basis"]
        basis = dict(cast(Json, event["fact"])["credential_basis"])
        observed = basis["observed_at"]
        basis.update(
            expires_at=_iso(observed + TTL),
            age_seconds=max(0, now - observed),
            stale=observed + TTL <= now,
        )
    return {"subject": subject, "facts": facts, "credential_basis": basis}, diagnostics


def _supported(subject: Json) -> bool:
    harness, surface, policy, selector = (
        subject[key] for key in ("harness", "surface", "tool_policy", "selector")
    )
    if surface == "native":
        return harness == "claude" and policy in {"read-only-no-shell", "packet-only-no-tools"}
    if surface == "offload-task":
        return (
            harness == "omp"
            and policy == "workspace-write-no-shell"
            and selector in OMP_WRITER_SELECTORS
        )
    if harness in {"claude", "claude-packet"}:
        expected_policy = "read-only-no-shell" if harness == "claude" else "packet-only-no-tools"
        return (
            selector is not None
            and selector.startswith("anthropic/")
            and is_exact_claude_model(selector.split("/", 1)[1])
            and policy == expected_policy
        )
    if harness == "omp":
        return policy == "read-only-no-shell" and selector in GROUNDED_OMP_SELECTORS
    if harness == "omp-packet":
        return policy == "packet-only-no-tools" and selector is not None
    return (
        harness == "agy"
        and policy == "packet-only-no-tools"
        and selector is not None
        and selector.split("/", 1)[1] in AGY_MODELS
        and selector.startswith("google-antigravity/")
    )


def _probe_native_claude(
    environment: Mapping[str, str] | None, cwd: Path | None, timeout: float
) -> tuple[str, str, Json]:
    """Observe installed native worker flags without starting a model turn."""
    from agent_execution.claude_execution import native_claude_launch

    try:
        executable, launch = native_claude_launch(environment=environment, timeout=timeout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return "unavailable", "Native Claude executable is unavailable", {}
    launch_path = launch.get("PATH", os.defpath)
    launch = {**launch, "PATH": f"{Path(executable).parent}{os.pathsep}{launch_path}"}
    try:
        result = run_in_process_group(
            [executable, "--help"], cwd or Path.cwd(), "", timeout, environment=launch
        )
    except (OSError, subprocess.TimeoutExpired):
        return (
            "unknown",
            "Native Claude capability help probe failed",
            {"executable": executable, "environment": launch},
        )
    if result.returncode:
        return (
            "unknown",
            "Native Claude capability help probe failed",
            {"executable": executable, "environment": launch},
        )
    help_text = result.stdout + result.stderr
    missing = [flag for flag in ("--safe-mode", "--tools", "--restricted") if flag not in help_text]
    if missing:
        return (
            "unavailable",
            "Native Claude lacks required worker options: " + ", ".join(missing),
            {"executable": executable, "environment": launch},
        )
    return (
        "available",
        "Native Claude worker options observed from the physical binary",
        {"executable": executable, "environment": launch},
    )


def _probe_omp(selector: str, environment: Mapping[str, str] | None, timeout: float) -> Json:
    command = omp_writer_probe_command([selector], environment=environment, writer_only=False)
    result = run_in_process_group(command, Path.cwd(), "", timeout, environment=environment)
    if result.returncode:
        raise ValueError("Pinned SDK model/credential probe failed")
    return validate_writer_probe_evidence(result.stdout, [selector])[0]


def _probe(
    subject: Json, environment: Mapping[str, str] | None, cwd: Path | None, timeout: float
) -> None:
    supported = _supported(subject)
    host_local = (
        subject["requester_host"] == subject["host"]
        and subject["requester_user"] == subject["os_user"]
    )
    transport = supported and (
        (subject["transport"] == "local" and host_local)
        or (subject["transport"] == "weft" and subject["surface"] in {"worker", "offload-task"})
    )
    capability = "available" if supported else "unavailable"
    auth, auth_detail = "unknown", "No registered authentication observation"
    detail = (
        "Implemented harness, selector and tool-policy combination"
        if supported
        else "Unsupported harness, surface, selector or tool-policy combination"
    )
    basis: Json | None = None
    if supported:
        harness = subject["harness"]
        if harness in {"claude", "claude-packet"} and subject["surface"] == "native":
            observed = credentials.observe_claude_auth(
                cwd=cwd, environment=environment, profile=subject["profile"], timeout=timeout
            )
            # The wrapper's auth command bypasses provider/proxy setup. Its
            # login response cannot identify the configured generation route.
            subject["effective_route"] = None
            subject["billing_pool"] = "native-claude:unobserved"
            capability = "unknown" if observed.executable else "unavailable"
            detail = (
                "Native generation route is unobserved"
                if observed.executable
                else "Native Claude executable is unavailable"
            )
            auth = "unknown"
            auth_detail = (
                f"Native sign-in is {observed.state}; generation-route authentication is unobserved"
            )
            basis = observed.basis.to_dict()
            basis.update(basis=credentials.BASIS_UNOBSERVED, reported_source=None)
        elif harness in {"claude", "claude-packet"}:
            capability, detail, launch = _probe_native_claude(environment, cwd, timeout)
            native_environment = launch.get("environment")
            if isinstance(native_environment, dict):
                try:
                    observed = credentials.observe_claude_auth(
                        cwd=cwd,
                        environment=native_environment,
                        timeout=timeout,
                        native_executable=launch["executable"],
                    )
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    auth_detail = "Native Claude authentication observation failed"
                    basis = credentials.CredentialBasis(
                        "claude", credentials.BASIS_UNOBSERVED, None, time.time(), ""
                    ).to_dict()
                else:
                    auth = observed.state
                    auth_detail = (
                        f"Native Claude authentication is {observed.state}"
                        if observed.answered
                        else "Native Claude authentication gave no readable answer"
                    )
                    basis = observed.basis.to_dict()
            else:
                auth_detail = (
                    "Native Claude executable is unavailable for authentication observation"
                )
                basis = credentials.CredentialBasis(
                    "claude", credentials.BASIS_UNOBSERVED, None, time.time(), ""
                ).to_dict()
        elif harness in {"omp", "omp-packet"}:
            try:
                require_omp_sdk(environment=environment)
            except ValueError:
                capability, detail = "unavailable", "Pinned OMP SDK or Bun is unavailable"
            else:
                try:
                    observed_omp = _probe_omp(subject["selector"], environment, timeout)
                    capability = "available" if observed_omp["model_available"] else "unavailable"
                    auth = "available" if observed_omp["credential_available"] else "unavailable"
                    detail, auth_detail = (
                        str(observed_omp["detail"]),
                        "Pinned SDK execution-eligible credentials "
                        + ("observed" if auth == "available" else "absent"),
                    )
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    capability, detail = "unknown", "Pinned SDK catalog observation failed"
                    auth_detail = "Pinned SDK credential observation failed"
                basis = credentials.observe_credential_basis(
                    "omp",
                    state_root=None,
                    cwd=cwd,
                    environment=environment,
                    profile=subject["selector"],
                    refresh=True,
                    timeout=timeout,
                ).to_dict()
        else:
            launch = credentials.harness_launch_environment(harness, environment=environment)
            capability = (
                "available"
                if shutil.which("agy", path=launch.get("PATH", os.defpath))
                else "unavailable"
            )
            basis = credentials.observe_credential_basis(
                "agy", state_root=None, cwd=cwd, environment=environment
            ).to_dict()
    _record(subject, "capability", capability, detail, method="execution-capability-probe")
    _record(subject, "authentication", auth, auth_detail, method="registered-credential-probe")
    _record(
        subject,
        "transport",
        "available" if transport else "unavailable",
        "Implemented requester/executor transport; no scheduling or network guarantee"
        if transport
        else "No implemented transport for this requester/executor context",
        method="transport-policy",
    )
    if basis is not None:
        _record(
            subject,
            "credential-basis",
            "unknown" if basis["basis"] in {"unobserved", "not_observable"} else "available",
            "Observed billing basis, independent of authentication",
            method="registered-billing-probe",
            basis=basis,
        )


def _status(
    *,
    fresh: bool,
    harness: str,
    surface: str,
    selectors: list[str] | None = None,
    transport: str = "local",
    requester_host: str | None = None,
    requester_user: str | None = None,
    expected_execution_sha256: str | None = None,
    timeout: float = 30.0,
    tool_policy: str | None = None,
    profile: str | None = None,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Json:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("probe timeout must be finite and positive")
    identity = _identity(expected_execution_sha256)
    rows, diagnostics = [], []
    subjects = [
        _subject(
            harness,
            surface,
            selector,
            transport,
            requester_host,
            requester_user,
            identity["source_sha256"],
            tool_policy,
            profile,
            environment,
            cwd,
        )
        for selector in _selectors(harness, selectors)
    ]
    for subject in subjects:
        if fresh:
            _probe(subject, environment, cwd, timeout)
    now = time.time()
    for subject in subjects:
        row, notes = _cached_row(subject, now)
        rows.append(row)
        diagnostics.extend(notes)
    return validate_status(
        {
            "schema_version": SCHEMA,
            "generated_at": _iso(now),
            "worker_identity": identity,
            "observer": {"host": socket.gethostname(), "os_user": getpass.getuser()},
            "rows": rows,
            "diagnostics": list(dict.fromkeys(diagnostics)),
        },
        expected_execution_sha256=expected_execution_sha256,
    )


def probe_status(
    *,
    harness: str,
    surface: str,
    selectors: list[str] | None = None,
    transport: str = "local",
    requester_host: str | None = None,
    requester_user: str | None = None,
    expected_execution_sha256: str | None = None,
    timeout: float = 30.0,
    tool_policy: str | None = None,
    profile: str | None = None,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Json:
    """Fresh non-generating observations on this exact executor."""
    return _status(
        fresh=True,
        harness=harness,
        surface=surface,
        selectors=selectors,
        transport=transport,
        requester_host=requester_host,
        requester_user=requester_user,
        expected_execution_sha256=expected_execution_sha256,
        timeout=timeout,
        tool_policy=tool_policy,
        profile=profile,
        environment=environment,
        cwd=cwd,
    )


def cached_status(
    *,
    harness: str,
    surface: str,
    selectors: list[str] | None = None,
    transport: str = "local",
    requester_host: str | None = None,
    requester_user: str | None = None,
    expected_execution_sha256: str | None = None,
    timeout: float = 30.0,
    tool_policy: str | None = None,
    profile: str | None = None,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Json:
    """Read immutable evidence with age; never probe credentials."""
    return _status(
        fresh=False,
        harness=harness,
        surface=surface,
        selectors=selectors,
        transport=transport,
        requester_host=requester_host,
        requester_user=requester_user,
        expected_execution_sha256=expected_execution_sha256,
        timeout=timeout,
        tool_policy=tool_policy,
        profile=profile,
        environment=environment,
        cwd=cwd,
    )


def validate_status(value: object, *, expected_execution_sha256: str | None = None) -> Json:
    """Reject malformed identity, coverage, facts and temporal claims."""
    document = _object(value, "execution-status envelope")
    if document.get("schema_version") != SCHEMA:
        raise ValueError("unsupported execution-status schema")
    now = _timestamp(document.get("generated_at"))
    from agent_execution.worker import WorkerIdentity

    identity = WorkerIdentity.from_dict(document.get("worker_identity"))
    build = identity.source_sha256
    if not build or identity.protocol_version < 1:
        raise ValueError("execution status requires an exact worker identity")
    if expected_execution_sha256 is not None and build != expected_execution_sha256:
        raise ValueError("execution source identity differs from the requested build")
    observer = _object(document.get("observer"), "observer")
    for key in ("host", "os_user"):
        _text(observer.get(key), key)
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("status lacks execution rows")
    seen = set()
    for raw in rows:
        row = _object(raw, "status row")
        if "credential_basis" not in row:
            raise ValueError("status row lacks credential billing evidence")
        subject = _object(row.get("subject"), "execution subject")
        try:
            provider_status.validate_execution_identity(_scope(subject))
        except KeyError as error:
            raise ValueError("incomplete execution subject") from error
        for key in (
            "route",
            "billing_pool",
            "host",
            "os_user",
            "credential_fingerprint",
            "fingerprint_scope",
        ):
            _text(subject.get(key), key)
        if subject["execution_sha256"] != build or any(
            subject[key] != observer[key] for key in ("host", "os_user")
        ):
            raise ValueError("execution subject differs from observed executor")
        fingerprint = subject["credential_fingerprint"]
        scope = subject["fingerprint_scope"]
        if scope not in {"host", "global"} or not (
            (scope == "host" and fingerprint == f"host:{subject['host']}:unknown")
            or re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", fingerprint)
        ):
            raise ValueError("invalid credential fingerprint identity")
        if (
            subject["harness"] in {"omp", "omp-packet", "agy", "claude-packet"}
            or subject["surface"] == "worker"
            and subject["harness"] == "claude"
        ) and subject["selector"] is None:
            raise ValueError("execution harness requires an exact selector")
        if (
            subject["surface"] == "worker"
            and subject["harness"] in {"claude", "claude-packet"}
            and (
                subject["profile"] is not None
                or subject["effective_route"] not in {None, "anthropic"}
            )
        ):
            raise ValueError("invalid native Claude worker execution identity")
        expected_route = (
            subject["selector"].split("/", 1)[0]
            if subject["selector"]
            else provider_status.route_for(subject["harness"])
        )
        if subject["route"] != expected_route:
            raise ValueError("execution route differs from selector")
        key = subject["selector"]
        if key in seen:
            raise ValueError("duplicate execution selector")
        seen.add(key)
        facts = _object(row.get("facts"), "execution facts")
        if set(facts) != set(FACTS):
            raise ValueError("incomplete execution facts")
        for name in FACTS:
            fact = _object(facts[name], name)
            if (
                not {
                    "state",
                    "observed_at",
                    "expires_at",
                    "age_seconds",
                    "stale",
                    "source",
                    "detail",
                    "condition",
                }
                <= fact.keys()
            ):
                raise ValueError("incomplete execution fact")
            if (
                not isinstance(fact["state"], str)
                or fact["state"] not in provider_status.FACT_STATES
                or type(fact.get("stale")) is not bool
            ):
                raise ValueError("invalid fact state or freshness")
            source = _object(fact.get("source"), "fact provenance")
            for field in ("tool", "method"):
                _text(source.get(field), field)
            _text(fact.get("detail"), "fact detail")
            if fact.get("condition") is not None and (
                not isinstance(fact["condition"], str)
                or fact["condition"] not in provider_status.REFUSAL_CONDITIONS
            ):
                raise ValueError("invalid provider refusal condition")
            observed, expires, age = (
                fact.get(field) for field in ("observed_at", "expires_at", "age_seconds")
            )
            if observed is None:
                if (
                    expires is not None
                    or age is not None
                    or not fact["stale"]
                    or fact["state"] != "unknown"
                ):
                    raise ValueError("unobserved fact claims current evidence")
            else:
                start, end = _timestamp(observed), _timestamp(expires)
                if (
                    end < start
                    or start > now + 1
                    or isinstance(age, bool)
                    or not isinstance(age, (float, int))
                    or not math.isfinite(age)
                    or age < 0
                    or abs(age - max(0, now - start)) > 1
                    or fact["stale"] != (end <= now)
                ):
                    raise ValueError("inconsistent observation freshness")
        if (
            subject["surface"] == "worker"
            and subject["harness"] in {"claude", "claude-packet"}
            and facts["capability"]["state"] == "available"
            and (subject["effective_route"] != "anthropic" or not _supported(subject))
        ):
            raise ValueError("available native Claude capability requires its enforced route")
        basis = row.get("credential_basis")
        if basis is not None:
            basis = _object(basis, "credential basis")
            if (
                basis.get("schema_version") != credentials.CREDENTIAL_BASIS_SCHEMA
                or not isinstance(basis.get("basis"), str)
                or basis["basis"]
                not in {
                    credentials.BASIS_SUBSCRIPTION,
                    credentials.BASIS_API_KEY,
                    credentials.BASIS_UNOBSERVED,
                    credentials.BASIS_NOT_OBSERVABLE,
                }
            ):
                raise ValueError("invalid credential billing basis")
            expected_harness = (
                "claude"
                if subject["harness"] in {"claude", "claude-packet"}
                else "omp"
                if subject["harness"] in {"omp", "omp-packet"}
                else subject["harness"]
            )
            if basis.get("harness") != expected_harness:
                raise ValueError("credential billing basis names a different harness")
            observed = basis.get("observed_at")
            age = basis.get("age_seconds")
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (float, int))
                or not math.isfinite(observed)
                or isinstance(age, bool)
                or not isinstance(age, (float, int))
                or not math.isfinite(age)
                or age < 0
                or observed > now + 1
                or _timestamp(basis.get("expires_at")) < observed
                or abs(age - max(0, now - observed)) > 2
            ):
                raise ValueError("invalid credential basis freshness")
            if type(basis.get("stale")) is not bool or basis["stale"] != (
                _timestamp(basis.get("expires_at")) <= now
            ):
                raise ValueError("inconsistent credential basis expiry")
    notes = document.get("diagnostics")
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ValueError("invalid execution diagnostics")
    return document


def record_generation(subject: Json, *, succeeded: bool, condition: str | None = None) -> None:
    """Publish an actual provider outcome for the subject observed before launch.

    Native callers pass the subject returned by probe_status. Local admission,
    spawn and evidence-validation failures must not call this producer.
    """
    provider_status.validate_execution_identity(_scope(subject))
    if subject["selector"] is None:
        raise ValueError("generation observation requires an exact selector")
    if subject["harness"] == "claude" and subject["effective_route"] is None:
        raise ValueError("native generation observation requires a verified launch route")
    if type(succeeded) is not bool or condition not in {None, *provider_status.REFUSAL_CONDITIONS}:
        raise ValueError("invalid provider generation outcome")
    if (
        subject["host"] != socket.gethostname()
        or subject["os_user"] != getpass.getuser()
        or subject["execution_sha256"] != _identity(None)["source_sha256"]
    ):
        raise ValueError("generation producer does not match execution subject")
    _record(
        subject,
        "generation",
        "available" if succeeded else "unavailable",
        "Provider generation completed"
        if succeeded
        else f"Provider reported {condition or 'unknown'} refusal",
        method="provider-generation",
        condition=None if succeeded else condition or "unknown",
    )
    if succeeded:
        _record(
            subject,
            "quota",
            "unknown",
            "Generation succeeded; remaining quota was not measured",
            method="provider-generation",
        )
    elif condition in {"quota", "auth", "network"}:
        name = {"quota": "quota", "auth": "authentication", "network": "transport"}[condition]
        _record(
            subject,
            name,
            "unavailable",
            f"Provider reported {condition} refusal",
            method="provider-refusal",
            condition=condition,
        )


def probe_remote_status(
    *,
    host: str,
    account: str = "agent",
    harness: str,
    surface: str,
    selectors: list[str] | None = None,
    transport: str = "weft",
    bin_dir: str | None = None,
    worker_executable: str = "agent-execution-worker",
    expected_execution_sha256: str | None = None,
    timeout: float = 30.0,
    tool_policy: str | None = None,
    profile: str | None = None,
    cwd: str | None = None,
) -> Json:
    target = host if "@" in host else f"{account}@{host}"
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*@[A-Za-z0-9_][A-Za-z0-9_.:-]*", target):
        raise ValueError("invalid execution SSH target")
    selected = _selectors(harness, selectors)
    policy = _policy(harness, surface, tool_policy)
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("probe timeout must be finite and positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", worker_executable):
        raise ValueError("worker executable must be a basename")
    if bin_dir is not None and not Path(bin_dir).is_absolute():
        raise ValueError("pinned worker directory must be absolute")
    executable = str(Path(bin_dir) / worker_executable) if bin_dir else worker_executable
    requester_host, requester_user = socket.gethostname(), getpass.getuser()
    command = [
        executable,
        "execution-status",
        "--harness",
        harness,
        "--surface",
        surface,
        "--transport",
        transport,
        "--tool-policy",
        policy,
        "--requester-host",
        requester_host,
        "--requester-user",
        requester_user,
        "--timeout",
        str(timeout),
        "--json",
    ]
    for selector in selected:
        if selector is not None:
            command.extend(["--selector", selector])
    for flag, value in (
        ("--expect-source-sha256", expected_execution_sha256),
        ("--profile", profile),
        ("--cwd", cwd),
    ):
        if value is not None:
            command.extend([flag, value])
    remote = (
        'export PATH="$HOME/.local/bin:$HOME/.bun/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"; '
        + shlex.join(command)
    )
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, remote],
            capture_output=True,
            text=True,
            timeout=timeout * max(1, len(selected)) * 3 + 15,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr
        detail = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
        raise ValueError(
            f"execution status SSH observation on {target} timed out after {error.timeout:g}s"
            + (f": {detail.strip()}" if detail and detail.strip() else "")
        ) from error
    except OSError as error:
        raise ValueError(f"execution status SSH observation on {target} failed: {error}") from error
    if result.returncode:
        detail = result.stderr.strip()
        raise ValueError(
            f"execution status probe on {target} exited {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    try:
        document = validate_status(
            json.loads(result.stdout), expected_execution_sha256=expected_execution_sha256
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid execution status from {target}: {error}") from error
    rows = document["rows"]
    if {row["subject"]["selector"] for row in rows} != set(selected):
        raise ValueError("execution status selector coverage differs from the request")
    for row in rows:
        subject = row["subject"]
        expected = {
            "harness": harness,
            "surface": surface,
            "transport": transport,
            "tool_policy": policy,
            "requester_host": requester_host,
            "requester_user": requester_user,
            "os_user": target.split("@", 1)[0],
        }
        if profile is not None:
            expected["profile"] = profile
        if any(subject[key] != value for key, value in expected.items()):
            raise ValueError("execution status context differs from the request")
    # Transport delay cannot extend the lifetime of the remote observations.
    now = time.time()
    if _timestamp(document["generated_at"]) > now + 1:
        raise ValueError("executor clock is ahead of the observer")
    document["generated_at"] = _iso(now)
    for row in rows:
        for fact in row["facts"].values():
            if fact["observed_at"] is not None:
                fact["age_seconds"] = max(0, now - _timestamp(fact["observed_at"]))
                fact["stale"] = _timestamp(fact["expires_at"]) <= now
        basis = row["credential_basis"]
        if basis is not None:
            basis["age_seconds"] = max(0, now - basis["observed_at"])
            basis["stale"] = _timestamp(basis["expires_at"]) <= now
    return validate_status(document, expected_execution_sha256=expected_execution_sha256)


def add_status_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "execution-status", help="Observe exact execution capabilities and credentials"
    )
    parser.add_argument("--harness", required=True, choices=provider_status.EXECUTION_HARNESSES)
    parser.add_argument("--surface", required=True, choices=provider_status.EXECUTION_SURFACES)
    parser.add_argument("--selector", action="append", dest="selectors")
    parser.add_argument(
        "--transport", default="local", choices=provider_status.EXECUTION_TRANSPORTS
    )
    parser.add_argument("--tool-policy", choices=provider_status.EXECUTION_TOOL_POLICIES)
    parser.add_argument("--requester-host")
    parser.add_argument("--requester-user")
    parser.add_argument("--expect-source-sha256", dest="expected_execution_sha256")
    parser.add_argument("--profile")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--json", action="store_true")


def run_status(args: argparse.Namespace) -> int:
    arguments = {
        name: getattr(args, name)
        for name in (
            "harness",
            "surface",
            "selectors",
            "transport",
            "tool_policy",
            "requester_host",
            "requester_user",
            "expected_execution_sha256",
            "profile",
            "cwd",
            "timeout",
        )
    }
    value = (cached_status if args.cached else probe_status)(**arguments)
    if args.json:
        print(json.dumps(value, sort_keys=True))
    else:
        for row in value["rows"]:
            print(
                f"{row['subject']['host']} {row['subject']['harness']} {row['subject']['selector'] or ''}"
            )
            for name, fact in row["facts"].items():
                print(
                    f"  {name}: {fact['state']} age={fact['age_seconds']} stale={fact['stale']}: {fact['detail']}"
                )
        for diagnostic in value["diagnostics"]:
            print(f"diagnostic: {diagnostic}")
    return 0
