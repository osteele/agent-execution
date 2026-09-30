"""Observe CLI authentication without starting a model turn.

An exported key alone does not establish the active authentication method:
stored decisions and wrapper profiles affect which credentials a CLI selects.
Status observations support marginal-cost estimates. They do not prove which
provider a later invocation will use, or whether subscription overage is enabled.

The answer is read with a registered non-generating observation: Claude's auth
status command or the OMP SDK credential helper. Native Codex is not probed.
Asking an invented subcommand of another CLI can land as a model prompt, which
is the exact failure this module exists to prevent.

Which status a harness reports depends on the environment it is launched with.
A review dispatch scrubs the exported Anthropic key sources before the CLI
starts, so the observation that matters is the one taken under that same
scrubbed launch environment - not the raw parent environment, where a key that
the review will never use makes the route look metered. `claude_command_prefix`
and `claude_launch_environment` are the shared statement of that launch
environment; the dispatch command and the probe both derive from them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

CREDENTIAL_BASIS_SCHEMA = "harness-credential-basis/v1"

#: Bumped when cached rows stop meaning what they said. A row without this
#: version was written under retired cache semantics and is re-probed.
CREDENTIAL_CACHE_VERSION = "credential-basis-cache/v3"

#: Recorded bases. `unobserved` is not a synonym for `subscription`: it means the
#: harness did not answer, and a reader must not price it as free.
BASIS_SUBSCRIPTION = "subscription"
BASIS_API_KEY = "api_key"
#: A harness that can report and did not: the probe ran and produced no answer,
#: so this row may have been billed to a key. Priced as billed.
BASIS_UNOBSERVED = "unobserved"
#: A harness with no registered status command, so nothing was asked.
#: Distinct from a registered harness whose observation failed.
BASIS_NOT_OBSERVABLE = "not_observable"
#: Recorded at dispatch, before any harness has been asked. Distinct from the
#: two absences above and from a missing field: a triager reading `None` on an
#: in-flight row cannot tell "still running, not yet observed" from "this
#: column was never written", which is the same ambiguity splitting
#: `unobserved` from `not_observable` closed one layer down.
BASIS_PENDING = "pending"
#: A route that bills local compute and no API cash.
BASIS_LOCAL = "local"
#: A direct vendor-API adapter, where the key is the route by construction.
BASIS_TOKEN_API = "token-api"

#: `apiKeySource` values, mapped at this boundary rather than stored raw. The
#: harness answers `none` for a subscription, which is an absence-shaped answer
#: to a question about which credential was used - stored raw, a later reader
#: cannot tell "OAuth" from "never asked".
_SOURCE_BASIS = {
    "none": BASIS_SUBSCRIPTION,
    "/login": BASIS_SUBSCRIPTION,
    "ANTHROPIC_API_KEY": BASIS_API_KEY,
    "ANTHROPIC_AUTH_TOKEN": BASIS_API_KEY,
    "apiKeyHelper": BASIS_API_KEY,
}

#: What a normalized `codex login status` report means. A ChatGPT-account login
#: is the subscription route; any other live login bills an API key and is
#: recorded as one rather than being absorbed into `subscription`.
_CODEX_SOURCE_BASIS = {
    "chatgpt": BASIS_SUBSCRIPTION,
    "api_key": BASIS_API_KEY,
}

#: Environment variables whose presence or rotation can change the answer. A
#: newly *approved* key changes it without changing the environment, which the
#: fingerprint cannot see, so the cache also expires.
_FINGERPRINT_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
CACHE_TTL_SECONDS = 900.0

#: The exported sources a Claude dispatch scrubs before the CLI starts. The CLI
#: is the subscription route -- a metered Anthropic stack goes through
#: AnthropicAdapter and its own api_key_env -- and an inherited key is not
#: inert: with ANTHROPIC_API_KEY exported, claude reports that the key takes
#: precedence over the claude.ai login, bills the API account, and exited 1 on
#: every review dispatched against an API account over its limit.
CLAUDE_SCRUBBED_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

#: Set on every Claude launch so a review inherits the wrapper's profile and
#: provider resolution but not configured arguments such as the interactive
#: agent-mail channel.
_CLAUDE_DISABLE_EXTRA_ARGS = "CLAUDE_WRAPPER_DISABLE_CONFIG_EXTRA_ARGS"

#: Environment variables that select which Claude configuration and credential
#: a launch reads, and therefore which answer `auth status` gives.
_CLAUDE_CACHE_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    _CLAUDE_DISABLE_EXTRA_ARGS,
    "CLAUDE_PROFILE",
    "CLAUDE_CONFIG_DIR",
    "HOME",
)

_OMP_CACHE_ENV_KEYS = (
    "HOME",
    "PATH",
    "PI_CODING_AGENT_DIR",
    "OMP_AUTH_BROKER_URL",
    "OMP_AUTH_BROKER_TOKEN",
    "AGENT_EXECUTION_OMP_SDK_ROOT",
)

_PROBE_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class CredentialBasis:
    """One harness's billing basis, as the harness reported it."""

    harness: str
    basis: str
    reported_source: str | None
    observed_at: float
    fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CREDENTIAL_BASIS_SCHEMA,
            "harness": self.harness,
            "basis": self.basis,
            "reported_source": self.reported_source,
            "observed_at": self.observed_at,
            "fingerprint": self.fingerprint,
        }


def claude_command_prefix(profile: str | None = None) -> list[str]:
    """The `env` prefix every Claude dispatch starts with.

    Scrubbing the exported key sources makes the review run under the identity
    the stack declares, and the disable flag keeps user-level configured
    arguments out of a noninteractive run. `--profile` belongs to the local
    claude-wrapper, not to the claude CLI, which rejects it outright; the
    wrapper reads CLAUDE_PROFILE as an override, and a binary without the
    wrapper ignores an environment variable it does not know.
    """
    prefix = [
        "env",
        "-u",
        CLAUDE_SCRUBBED_KEYS[0],
        "-u",
        CLAUDE_SCRUBBED_KEYS[1],
        f"{_CLAUDE_DISABLE_EXTRA_ARGS}=1",
    ]
    if profile:
        prefix.append(f"CLAUDE_PROFILE={profile}")
    return prefix


def claude_launch_environment(
    profile: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The inherited environment transformed the way a Claude dispatch launches.

    The same scrubbing as :func:`claude_command_prefix`, applied to a copy of
    the given environment (the process environment when none is given). The
    input mapping is never mutated.
    """
    source = os.environ if environment is None else environment
    launch = {name: value for name, value in source.items() if name not in CLAUDE_SCRUBBED_KEYS}
    launch[_CLAUDE_DISABLE_EXTRA_ARGS] = "1"
    if profile:
        launch["CLAUDE_PROFILE"] = profile
    return launch


def environment_fingerprint(environment: Mapping[str, str] | None = None) -> str:
    """Fingerprint the credential environment, without carrying a secret.

    Presence plus a short digest of each value, so a rotated key produces a
    different fingerprint while the fingerprint itself discloses nothing usable.
    """
    source = os.environ if environment is None else environment
    parts = []
    for name in _FINGERPRINT_KEYS:
        value = source.get(name)
        if not value:
            parts.append(f"{name}=absent")
            continue
        parts.append(f"{name}={_digest(value)}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def basis_for_source(reported: str | None) -> str:
    """Map a harness-reported credential source onto a billing basis."""
    if reported is None:
        return BASIS_UNOBSERVED
    return _SOURCE_BASIS.get(reported, BASIS_API_KEY)


def codex_basis_for_source(reported: str | None) -> str:
    """Map a normalized `codex login status` report onto a billing basis."""
    if reported is None:
        return BASIS_UNOBSERVED
    return _CODEX_SOURCE_BASIS.get(reported, BASIS_API_KEY)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _executable_identity(executable: str) -> str:
    """Invalidate observations when a binary or symlink target is replaced."""
    path = Path(executable).resolve()
    try:
        stat = path.stat()
    except OSError:
        return f"{path}:unavailable"
    return f"{path}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}"


def _resolved_status_executable(
    executable: str, environment: Mapping[str, str] | None
) -> str | None:
    """Resolve the harness binary through the launch environment's PATH.

    The probe must observe the binary a dispatch would run: a caller-supplied
    environment selects its own PATH, and resolving against the parent's
    instead could fingerprint one binary while running another.
    """
    path = None if environment is None else environment.get("PATH")
    return shutil.which(executable, path=path)


def _environment_cache_fingerprint(environment: Mapping[str, str], names: tuple[str, ...]) -> str:
    parts = []
    for name in names:
        value = environment.get(name)
        if not value:
            parts.append(f"{name}=absent")
            continue
        parts.append(f"{name}={_digest(value)}")
    return "|".join(parts)


def _cache_fingerprint(
    executable: str,
    *,
    environment: Mapping[str, str],
    cwd: Path,
    profile: str | None,
    env_keys: tuple[str, ...],
) -> str:
    """Fingerprint everything that can change what the status command answers.

    The resolved binary, the effective launch environment, the working
    directory, the selected profile, and the home/config selection. None of it
    contains a secret: every environment value is reduced to a short digest.
    """
    parts = [
        f"executable={_executable_identity(executable)}",
        f"cwd={_digest(str(cwd))}",
        f"profile={profile or ''}",
        _environment_cache_fingerprint(environment, env_keys),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _run_status_command(
    command: list[str],
    *,
    cwd: Path | None,
    environment: Mapping[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str] | None:
    """Run a non-generating status command, or return None on any failure.

    ``subprocess.run`` with a timeout kills a child that never answers, so a
    silent harness cannot hang the caller past the deadline. A missing binary
    and an expired deadline are the same no-answer as a nonzero exit.
    """
    try:
        return subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            env=dict(environment),
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _claude_status_source(status: dict[str, object]) -> str | None:
    """Extract the credential answer from a `claude auth status` object.

    A named key source wins: an exported key takes precedence over the
    subscription login, which is the precedence that makes the answer a billing
    fact. A non-empty `subscriptionType` with no key source is the measured
    subscription shape (`subscriptionType: "max"`, no `apiKeySource`); a key
    route reports `subscriptionType: null`, so its absence carries no evidence
    on its own and the answer is no answer.
    """
    reported = status.get("apiKeySource")
    if isinstance(reported, str) and reported and reported not in {"none", "/login"}:
        return reported
    subscription = status.get("subscriptionType")
    if (
        isinstance(subscription, str)
        and subscription
        and status.get("authMethod") == "claude.ai"
        and status.get("apiProvider") == "firstParty"
    ):
        return "none"
    return None


def _probe_claude_auth_status(
    executable: str,
    *,
    cwd: Path | None,
    environment: Mapping[str, str] | None,
    profile: str | None,
    timeout: float,
) -> str | None:
    """Read `claude auth status` under the scrubbed launch environment.

    Returns the reported key source when the harness names one, `"none"` when
    it reports a subscription route with no key source, and None when it gave
    no answer - a failed observation, never an assertion that no key was used.
    """
    launch = claude_launch_environment(profile=profile, environment=environment)
    completed = _run_status_command(
        [executable, "auth", "status"],
        cwd=cwd,
        environment=launch,
        timeout=timeout,
    )
    if completed is None or completed.returncode != 0:
        return None
    try:
        status = cast(dict[str, object], json.loads(completed.stdout))
    except json.JSONDecodeError:
        return None
    if not isinstance(status, dict):
        return None
    return _claude_status_source(status)


def _probe_codex_login_status(
    executable: str,
    *,
    cwd: Path | None,
    environment: Mapping[str, str] | None,
    profile: str | None,
    timeout: float,
) -> str | None:
    """Read `codex login status`.

    Returns `"chatgpt"` for a ChatGPT-account login, `"api_key"` for any other
    live login, and None when the harness reports no login or gives no answer.
    Login status observes stored authentication, not the provider configuration
    a named execution profile will select.
    """
    del profile
    source = os.environ if environment is None else environment
    completed = _run_status_command(
        [executable, "login", "status"],
        cwd=cwd,
        environment=source,
        timeout=timeout,
    )
    if completed is None or completed.returncode != 0:
        return None
    return _codex_status_source(completed.stdout + "\n" + completed.stderr)


def _codex_status_source(output: str) -> str | None:
    """Normalize `codex login status` text onto the codex source vocabulary."""
    combined = output.casefold()
    if "not logged in" in combined:
        return None
    if "logged in using chatgpt" in combined:
        return "chatgpt"
    if "logged in" in combined:
        return "api_key"
    return None


def _probe_omp_auth_status(
    executable: str,
    *,
    cwd: Path | None,
    environment: Mapping[str, str] | None,
    profile: str | None,
    timeout: float,
) -> str | None:
    """Ask the restricted SDK helper about the exact selector, without generating."""
    from agent_execution.omp_execution import omp_auth_status_command

    del executable
    if not profile or "/" not in profile:
        return None
    provider, model = profile.split("/", 1)
    launch = dict(os.environ if environment is None else environment)
    try:
        command = omp_auth_status_command(profile, environment=launch)
    except (OSError, ValueError):
        return None
    completed = _run_status_command(command, cwd=cwd, environment=launch, timeout=timeout)
    if completed is None or completed.returncode != 0:
        return None
    try:
        status = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(status, dict)
        or status.get("schema_version") != "agent-execution.omp-auth/v1"
        or status.get("provider") != provider
        or status.get("model") != model
    ):
        return None
    credential = status.get("credential_type")
    return (
        credential
        if isinstance(credential, str) and credential in {"oauth", "api_key", "missing", "unknown"}
        else None
    )


def _omp_basis_for_source(reported: str | None) -> str:
    if reported == "oauth":
        return BASIS_SUBSCRIPTION
    return BASIS_API_KEY if reported == "api_key" else BASIS_UNOBSERVED


#: The status commands this module knows. Anything else is never probed: an
#: invented auth-status subcommand on another CLI can become a model prompt,
#: which bills.
_AUTH_STATUS_PROBES = {
    "claude": _probe_claude_auth_status,
    "omp": _probe_omp_auth_status,
}

#: Per-harness source-to-basis mappings, paired with the probes above.
_SOURCE_BASIS_FOR_HARNESS = {
    "claude": basis_for_source,
    "omp": _omp_basis_for_source,
}

#: Per-harness environment selections for the cache fingerprint.
_CACHE_ENV_KEYS = {
    "claude": _CLAUDE_CACHE_ENV_KEYS,
    "omp": _OMP_CACHE_ENV_KEYS,
}


def probe_credential_source(
    executable: str,
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    profile: str | None = None,
    timeout: float | None = None,
) -> str | None:
    """Read the harness's credential answer from its own status command.

    The command is non-generating by construction: it is a registry of known
    status invocations, never a prompt, so the probe cannot start a model turn
    and cannot bill. Returns None when the harness produced no answer - a
    failed observation, never an assertion that no key was used - and None for
    a harness with no known status command, which must not be asked anything.
    """
    probe = _AUTH_STATUS_PROBES.get(Path(executable).name)
    if probe is None:
        return None
    resolved = _resolved_status_executable(
        "bun" if Path(executable).name == "omp" else executable, environment
    )
    if resolved is None:
        return None
    return probe(
        resolved,
        cwd=cwd,
        environment=environment,
        profile=profile,
        timeout=_PROBE_TIMEOUT_SECONDS if timeout is None else timeout,
    )


def _omp_cache_profile(profile: str | None, environment: Mapping[str, str]) -> str:
    """Invalidate credential observations when the SDK helper or runtime changes."""
    from agent_execution.omp_execution import omp_sdk_root

    helper = Path(__file__).with_name("omp_sdk.ts")
    root = omp_sdk_root(environment=environment)
    package = root / "node_modules/@oh-my-pi/pi-coding-agent/package.json"
    return (
        f"{profile or ''}|{_executable_identity(str(helper))}|{_executable_identity(str(package))}"
    )


def _cache_path(state_root: Path) -> Path:
    return state_root / "credential-basis.json"


def _cache_key(executable: str, profile: str | None) -> str:
    """Name one route's row: the executable plus the profile or selector it launches.

    One executable serves several routes -- `omp` serves every provider/model
    selector -- so rows are per route. A shared row would let probing one route
    evict another, leaving a later silent probe no prior evidence to fall back
    on. A bare-executable row never serves a profiled route; that route is
    re-probed rather than guessed.
    """
    return f"{executable}|{profile}" if profile else executable


def _cached_observation(
    cached: object,
    *,
    fingerprint: str,
    now: float,
    ttl: float | None,
) -> CredentialBasis | None:
    """Read a cache row written by these semantics for the same launch route.

    A row is usable when it carries the current cache version, matches the
    fingerprint, and was observed in a non-future moment. Passing a TTL also
    requires a fresh row; omitting it recovers prior direct evidence after a
    transient probe failure.
    """
    if not isinstance(cached, dict):
        return None
    row = cast(dict[str, object], cached)
    if row.get("cache_version") != CREDENTIAL_CACHE_VERSION:
        return None
    if row.get("fingerprint") != fingerprint:
        return None
    basis = row.get("basis")
    if not isinstance(basis, str):
        return None
    observed_at = row.get("observed_at")
    if isinstance(observed_at, bool) or not isinstance(observed_at, (int, float)):
        return None
    observed = float(observed_at)
    if not (0.0 <= observed <= now):
        return None
    if ttl is not None and now - observed >= ttl:
        return None
    reported = row.get("reported_source")
    return CredentialBasis(
        harness=str(row.get("harness", "")),
        basis=basis,
        reported_source=None if reported is None else str(reported),
        observed_at=observed,
        fingerprint=fingerprint,
    )


def observe_credential_basis(
    executable: str,
    *,
    state_root: Path | None,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    profile: str | None = None,
    clock: float | None = None,
    ttl: float = CACHE_TTL_SECONDS,
    refresh: bool = False,
    timeout: float | None = None,
) -> CredentialBasis:
    """Return this harness's billing basis, probing when the cache cannot serve.

    Cached on a fingerprint of the resolved binary, the effective launch
    environment, the working directory, and the profile/home config selection,
    so a changed route re-probes; expired by ``ttl`` because an approval
    decision changes the answer without changing the environment; and
    invalidated as a whole by the cache version. ``refresh=True`` forces a new
    probe before serving the result. A transient no-answer does not erase older
    direct evidence for the same route; explicit missing or changed credentials
    do.

    ``state_root=None`` probes without any cache: a worker with no database
    still gets a real observation. A harness with no known status command is
    `not_observable` and is never probed.
    """
    now = time.time() if clock is None else clock
    name = Path(executable).name
    probe = _AUTH_STATUS_PROBES.get(name)
    if probe is None:
        return CredentialBasis(
            harness=executable,
            basis=BASIS_NOT_OBSERVABLE,
            reported_source=None,
            observed_at=now,
            fingerprint="",
        )
    resolved = _resolved_status_executable("bun" if name == "omp" else executable, environment)
    if resolved is None:
        # The named harness is not installed where this launch would look for
        # it. Nobody was asked, and nothing can dispatch; the row prices as
        # billed rather than as free.
        return CredentialBasis(
            harness=executable,
            basis=BASIS_UNOBSERVED,
            reported_source=None,
            observed_at=now,
            fingerprint="",
        )
    effective_cwd = (cwd or Path.cwd()).resolve()
    if name == "claude":
        launch_view: Mapping[str, str] = claude_launch_environment(
            profile=profile, environment=environment
        )
    else:
        launch_view = os.environ if environment is None else dict(environment)
    fingerprint = _cache_fingerprint(
        resolved,
        environment=launch_view,
        cwd=effective_cwd,
        profile=_omp_cache_profile(profile, launch_view) if name == "omp" else profile,
        env_keys=_CACHE_ENV_KEYS[name],
    )
    path = None if state_root is None else _cache_path(state_root)
    cached: dict[str, object] = {}
    previous: CredentialBasis | None = None
    if path is not None and path.exists():
        try:
            loaded = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            cached = loaded
        key = _cache_key(executable, profile)
        previous = _cached_observation(cached.get(key), fingerprint=fingerprint, now=now, ttl=None)
        if not refresh:
            served = _cached_observation(cached.get(key), fingerprint=fingerprint, now=now, ttl=ttl)
            if served is not None:
                return served
    reported = probe(
        resolved,
        cwd=cwd,
        environment=environment,
        profile=profile,
        timeout=_PROBE_TIMEOUT_SECONDS if timeout is None else timeout,
    )
    if reported is None and previous is not None and previous.reported_source is not None:
        return previous
    observation = CredentialBasis(
        harness=executable,
        basis=_SOURCE_BASIS_FOR_HARNESS[name](reported),
        reported_source=reported,
        observed_at=now,
        fingerprint=fingerprint,
    )
    if path is not None:
        _write_cache_row(path, _cache_key(executable, profile), observation)
    return observation


def _write_cache_row(path: Path, key: str, observation: CredentialBasis) -> None:
    """Merge the observation into the cache file, preserving other routes' rows."""
    cached: dict[str, object] = {}
    try:
        existing = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        if isinstance(existing, dict):
            cached = existing
    except (OSError, json.JSONDecodeError):
        cached = {}
    cached[key] = {
        "cache_version": CREDENTIAL_CACHE_VERSION,
        "basis": observation.basis,
        "reported_source": observation.reported_source,
        "observed_at": observation.observed_at,
        "fingerprint": observation.fingerprint,
        "harness": observation.harness,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass
