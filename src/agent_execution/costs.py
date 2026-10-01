"""Marginal incremental-cost estimates for an execution dispatch.

The user's cash policy counts authenticated subscription usage as $0. A
subscription observed under the launch environment, an explicit OMP
subscription-plan route, or a local adapter therefore has a zero estimate and
bound. A declared billing label alone never establishes that route. Direct
token-API adapters have a declared estimate but no hard maximum; key-based and
unobserved credential routes remain unbounded. A finite cap refuses every route
whose maximum is unknown.

The estimate never relabels a raw token cost as billed cash. A subscription
review still reports token-equivalent `cost_usd`; the raw figure stays on the
record exactly as the provider produced it, and the zero here is about
marginal cash only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from agent_execution.credentials import (
    BASIS_API_KEY,
    BASIS_LOCAL,
    BASIS_NOT_OBSERVABLE,
    BASIS_SUBSCRIPTION,
    BASIS_TOKEN_API,
    BASIS_UNOBSERVED,
    CredentialBasis,
    observe_credential_basis,
)

EXECUTION_COST_ESTIMATE_SCHEMA = "agent-execution.cost-estimate/v1"

#: Explicit subscription-plan routes. Bare `zai` is deliberately absent:
#: it routes to the per-token API rather than the Zhipu coding-plan endpoint.
OMP_SUBSCRIPTION_PROVIDERS = frozenset({"openai-codex", "zhipu-coding-plan", "kimi-code"})

#: Adapters that call the vendor API with a key, where the key is the route by
#: construction and the declared worst case is the estimate.
_TOKEN_API_ADAPTERS = frozenset({"openai", "anthropic", "openrouter"})

#: CLI adapters served by the claude binary, mapped to that binary.
_CLAUDE_ADAPTERS = {"claude": "claude", "claude-packet": "claude"}

_OMP_ADAPTERS = frozenset({"omp", "omp-packet"})


@dataclass(frozen=True)
class ExecutionCostEstimate:
    """What one execution dispatch is expected to add to the cash bill.

    ``estimated_incremental_usd`` is the expected marginal cost of one call.
    ``maximum_incremental_usd`` is a bound under the user's cash policy, or None
    when no bound is established — a refusal reason for a capped caller, not a
    license to dispatch. ``reason`` names the observation and policy the figures
    rest on.

    ``probe_answered`` says whether the figures rest on an answer from a
    credential probe: True when the probe answered -- including an explicit
    ``missing`` or ``unknown`` credential -- or when it was silent and this
    route's previous direct answer was served in its place; False when no answer
    of either kind exists (timeout, failed run, unreadable or mismatched status,
    or no probe executable, with no prior answer for the route); None where no
    probe applies. False is a failed observation rather than a finding about the
    route, so a caller may wait it out; it is still never priced as free.
    """

    billing_mode: str
    cost_basis: str
    estimated_incremental_usd: float | None
    maximum_incremental_usd: float | None
    reason: str
    probe_answered: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": EXECUTION_COST_ESTIMATE_SCHEMA,
            "billing_mode": self.billing_mode,
            "cost_basis": self.cost_basis,
            "estimated_incremental_usd": self.estimated_incremental_usd,
            "maximum_incremental_usd": self.maximum_incremental_usd,
            "reason": self.reason,
            "probe_answered": self.probe_answered,
        }


def _zero_estimate(billing_mode: str, cost_basis: str, reason: str) -> ExecutionCostEstimate:
    return ExecutionCostEstimate(
        billing_mode=billing_mode,
        cost_basis=cost_basis,
        estimated_incremental_usd=0.0,
        maximum_incremental_usd=0.0,
        reason=reason,
    )


def _unknown_estimate(cost_basis: str, reason: str) -> ExecutionCostEstimate:
    return ExecutionCostEstimate(
        billing_mode="unknown",
        cost_basis=cost_basis,
        estimated_incremental_usd=None,
        maximum_incremental_usd=None,
        reason=reason,
    )


def _finite_nonnegative(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def _token_api_estimate(profile: Mapping[str, object]) -> ExecutionCostEstimate:
    declared = _finite_nonnegative(profile.get("worst_case_cost"))
    if declared is None:
        return _unknown_estimate(
            BASIS_TOKEN_API,
            "token-API route declares no usable worst_case_cost, so it has no estimate",
        )
    return ExecutionCostEstimate(
        billing_mode="token-api",
        cost_basis=BASIS_TOKEN_API,
        estimated_incremental_usd=declared,
        # A hard maximum would be invented: a retry chain can exceed any
        # declared figure, so a capped caller must refuse rather than trust it.
        maximum_incremental_usd=None,
        reason=f"token-API route billed per call, declared worst case ${declared}",
    )


def _credential_estimate(
    observation: CredentialBasis,
    *,
    probe_detail: str,
) -> ExecutionCostEstimate:
    if observation.basis == BASIS_NOT_OBSERVABLE:
        return _unknown_estimate(
            BASIS_NOT_OBSERVABLE,
            f"{probe_detail} cannot be asked about billing",
        )
    answered = observation.reported_source is not None
    if observation.basis == BASIS_SUBSCRIPTION:
        estimate = _zero_estimate(
            BASIS_SUBSCRIPTION,
            BASIS_SUBSCRIPTION,
            f"{probe_detail} reports a subscription login; "
            "user policy counts authenticated subscription usage as $0",
        )
    elif observation.basis == BASIS_API_KEY:
        estimate = _unknown_estimate(
            BASIS_API_KEY,
            f"{probe_detail} reports a key route; no trustworthy bound exists for it",
        )
    elif answered:
        estimate = _unknown_estimate(
            BASIS_UNOBSERVED,
            f"{probe_detail} reports credential {observation.reported_source!r}, "
            "which establishes no subscription; it is never priced as free",
        )
    else:
        estimate = _unknown_estimate(
            BASIS_UNOBSERVED,
            f"{probe_detail} gave no answer; an unanswered probe is never priced as free",
        )
    return replace(estimate, probe_answered=answered)


def _omp_estimate(
    profile: Mapping[str, object],
    *,
    state_root: Path | None,
    cwd: Path | None,
    environment: Mapping[str, str] | None,
    refresh: bool,
) -> ExecutionCostEstimate:
    """Use explicit plan routes or an observed OAuth basis, never a billing label."""
    selector_value = profile.get("model_argument", profile.get("model"))
    if not isinstance(selector_value, str) or not selector_value:
        return _unknown_estimate(
            BASIS_NOT_OBSERVABLE,
            "OMP stack names no model selector, so its billing route is unestablished",
        )
    selected_provider, separator, selected_model = selector_value.partition("/")
    stack_provider = str(profile.get("provider", ""))
    if (
        not separator
        or not selected_provider
        or not selected_model
        or any(character.isspace() for character in selector_value)
    ):
        return _unknown_estimate(
            BASIS_NOT_OBSERVABLE,
            f"OMP selector {selector_value!r} is not provider-qualified; "
            "billing route is unestablished",
        )
    if selected_provider != stack_provider:
        return _unknown_estimate(
            BASIS_NOT_OBSERVABLE,
            f"OMP selector {selector_value!r} disagrees with the stack's provider "
            f"{stack_provider!r}; billing route is unestablished",
        )
    adapter = str(profile.get("adapter", ""))
    if selected_provider in {"anthropic", "google-antigravity"} or adapter == "omp":
        observation = observe_credential_basis(
            "omp",
            profile=selector_value,
            state_root=state_root,
            cwd=cwd,
            environment=environment,
            refresh=refresh,
        )
        if selected_provider == "zhipu-coding-plan" and observation.basis == BASIS_API_KEY:
            return replace(
                _zero_estimate(
                    BASIS_SUBSCRIPTION,
                    BASIS_SUBSCRIPTION,
                    f"OMP selector {selector_value!r} uses its observed coding-plan credential",
                ),
                probe_answered=True,
            )
        return _credential_estimate(observation, probe_detail="restricted OMP SDK auth status")
    if selected_provider not in OMP_SUBSCRIPTION_PROVIDERS:
        return _unknown_estimate(
            BASIS_NOT_OBSERVABLE,
            f"OMP provider {selected_provider!r} is not a subscription plan; "
            "billing route is unestablished",
        )
    return _zero_estimate(
        BASIS_SUBSCRIPTION,
        BASIS_SUBSCRIPTION,
        f"OMP selector {selector_value!r} names the {selected_provider} subscription plan",
    )


def estimate_execution_cost(
    profile: Mapping[str, object],
    *,
    state_root: Path | None = None,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    refresh: bool = False,
) -> ExecutionCostEstimate:
    """Estimate the marginal cash one dispatch of this stack can bill.

    Claude is observed with `claude auth status` under the same scrubbed launch
    environment the dispatch launches with. Native Codex is not probed. A hard-cap
    caller passes ``refresh=True`` so a stale cached observation cannot
    establish a free dispatch. ``state_root=None`` skips the cache entirely,
    for a worker that has no database.

    Every route without a proven zero or a declared worst case comes back
    unknown: declared billing modes never stand in for an observation.
    """
    adapter = str(profile.get("adapter", ""))
    if adapter == "fake":
        return _zero_estimate(
            BASIS_LOCAL,
            BASIS_LOCAL,
            "the fake adapter runs locally and bills no API cash",
        )
    # The SDK's provider route determines billing; coding plans can use API keys.
    if adapter in _OMP_ADAPTERS:
        return _omp_estimate(
            profile,
            state_root=state_root,
            cwd=cwd,
            environment=environment,
            refresh=refresh,
        )
    if profile.get("api_key_env") or adapter in _TOKEN_API_ADAPTERS:
        return _token_api_estimate(profile)
    if adapter in _CLAUDE_ADAPTERS:
        wrapper_profile_value = profile.get("profile")
        wrapper_profile = (
            str(wrapper_profile_value)
            if isinstance(wrapper_profile_value, str) and wrapper_profile_value
            else None
        )
        observation = observe_credential_basis(
            _CLAUDE_ADAPTERS[adapter],
            state_root=state_root,
            cwd=cwd,
            environment=environment,
            profile=wrapper_profile,
            refresh=refresh,
        )
        return _credential_estimate(
            observation, probe_detail="claude auth status under the launch environment"
        )
    return _unknown_estimate(
        BASIS_NOT_OBSERVABLE,
        f"adapter {adapter!r} has no non-generating billing observation, "
        "so it stays unknown however it declares its billing",
    )


def validate_max_cost_usd(value: float | None) -> float | None:
    """Validate a hard cash ceiling; None means uncapped.

    Rejects booleans, non-numbers, NaN, infinities, and negatives: a cap that
    cannot be compared, or one that permits spending, is not a ceiling.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("maximum_cost_usd must be a number or None")
    maximum = float(value)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("maximum_cost_usd must be finite and nonnegative")
    return maximum


def require_cost_cap(estimate: ExecutionCostEstimate, maximum_cost_usd: float | None) -> None:
    """Refuse a dispatch whose policy-bound cost does not fit under the hard cap.

    With a finite cap, a route with no established maximum — an unknown route,
    or a token-API route whose ceiling would be invented — is refused, as is a
    route whose maximum exceeds the cap. Under the user's cash policy, an
    authenticated subscription or local route has a zero bound and passes any
    finite cap, including zero. Chains and retries of those calls therefore
    remain $0 under that policy. A None cap is uncapped and checks
    nothing.
    """
    cap = validate_max_cost_usd(maximum_cost_usd)
    if cap is None:
        return
    bound = estimate.maximum_incremental_usd
    if bound is None:
        raise ValueError(
            "refusing dispatch under a hard cash cap: no proven incremental-cost "
            f"bound for a {estimate.billing_mode}/{estimate.cost_basis} route "
            f"({estimate.reason})"
        )
    if bound > cap:
        raise ValueError(
            "refusing dispatch under a hard cash cap: worst case "
            f"${bound} exceeds the ${cap} cap ({estimate.reason})"
        )
