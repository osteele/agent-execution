# agent-execution

Constrained coding-agent execution and optional durable Weft transport.

`agent_execution` owns worker execution (`agent-execution-worker`), process
containment, raw transcript evidence, cost limits, credential observation, and
the optional OMP runtime. Its restricted OMP boundary exposes three explicit
policies only: `packet-only-no-tools`, `read-only-no-shell`, and the additive
`workspace-write-no-shell` policy with confined `execution_write` and
`execution_edit` tools. Review policy, manifests, state, findings, and
completion notifications live in `agent-review`, which consumes this package
as an installed dependency.

## Install

```sh
uv tool install .
agent-execution-worker identity
agent-execution-worker install-omp  # Optional; requires Bun >= 1.3.14
```

Consumers that embed the worker install it into their own environment
(`uv tool install --with ../agent-execution .`) and verify it at runtime with
`agent-execution-worker identity` (protocol version plus source digest).

## Requirements

- Python >= 3.10; the package is standard-library-only.
- Bun >= 1.3.14 only for the OMP adapter; `install-omp` provisions the
  lockfile-pinned SDK under `~/.local/share/agent-execution/omp-sdk/<version>`.
- `workspace-write-no-shell` admits only supported writer identities
  `anthropic/claude-opus-5-5`, `zhipu-coding-plan/glm-5.3-flash`,
  `kimi-code/k3`, and `openai-codex/gpt-6-{sol,luna,astra}`. Sol is the ordinary
  OpenAI default; Luna supports focused work, and Astra is reserved for explicit
  largest-model escalation. GLM is the economical route for bounded mechanical
  work. Grounded reviews default to
  `anthropic/claude-opus-5-5` or `openai-codex/gpt-6-sol` for those providers;
  Luna and Astra are also admitted as explicit read-only selectors. Its
  write/edit tools refuse VCS and harness metadata such as `.omp`, preserve
  existing ordinary-file permission bits on replacement, and create new files
  with conservative permissions.

## Packet-only agy (Antigravity CLI)

The worker provider `agy` runs the Antigravity CLI with the tool policy
`packet-only-no-tools` and refuses every other policy. Its logical command
must have exactly this shape:

```
agy -p BRIEF --output-format json --print-timeout Ns --disable-slash-commands \
    --execution-tool-policy packet-only-no-tools --model M
```

`--model` must be one of the models in `AGY_MODELS`: `gemini-3.1-pro-high`,
`gemini-3.8-flash-high`, or `claude-opus-4-6-thinking`. `--mode` and
`--dangerously-skip-permissions` are refused. The worker removes the policy
flag before launch.

The worker builds a throwaway HOME for each call. It copies `~/.gemini`,
installs a deny-all `PreToolUse` hook, writes an empty MCP config and a
packet-only rules file, and symlinks `Library/Keychains` and
`Library/Preferences`. The launch environment is minimal, with HOME pointing
at the throwaway directory, which is deleted after the call. A caller cannot
supply HOME.

When the Weft conductor ships the brief as a payload (`-p -`), the worker puts
it back on argv, because `agy -p -` is not known to read stdin. Briefs on argv
are limited to 128 KiB and must not begin with `-`.

The JSON envelope on stdout is the evidence. It is refused unless `status` is
`SUCCESS`, `response` is non-empty, and `num_turns` is positive, and it is
refused if it records any tool use (`denied_actions` and similar fields). Use
`agent_execution.agy_execution.agy_final_text` to get the response text.

`agy` has no auth-status command, so its credential basis is `not_observable`.
A hard `--max-cost-usd` cap therefore refuses it. Status is published under
the route `google-antigravity`, using the billing pool
`google-antigravity/gemini` for Gemini models and `google-antigravity/other`
for everything else.

Tool-enabled agy (grounded read-only review or writing) is not supported. The
deny-all hook is the only enforcement known to work.

## Local process execution

`agent_execution.processes.run_in_process_group` sends prompt strings as UTF-8
stdin bytes, independent of the host locale or text-stream encoding. Prompt
delivery shares the harness completion deadline, including when a child does
not read its input pipe.

## Worker evidence

Protocol 2 publishes each call's result at
`.agent-execution/results/<model-call-id>.json`. The shared
`agent_execution.identity.worker_evidence_path()` helper validates the identity
and returns this relative path. Identities use lowercase ASCII letters, digits,
underscores, and hyphens (1–128 characters, starting with a letter or digit).
Uppercase identities are rejected to prevent case-insensitive filesystem collisions.
Fresh execution has no `--evidence-out` flag or
Python `output` argument. Weft declares the same path with `--produces` and
`--output`, so inventory completion discovery tracks the worker result for
later artifact retrieval even when the local waiter has exited.

Write and edit tools refuse `.agent-execution` components case-insensitively.
They also protect legacy `agent-execution-worker-result.json` and
`agent-execution-worker-result-*.json` filenames. Ordinary outputs remain writable.
Worker publication rejects symlink redirects out of the protected namespace.

Retrieval compares results against the retained dispatch's
`expected_worker_protocol_version`, `worker_result_path`, source digest,
inference selector, and `omp_policy`. Missing dispatch expectations are recovered
from Weft's accepted command before fetching the artifact. Offline file ingestion
requires those expectations from the caller; result claims cannot supply them.

Historical protocol-1 read-only jobs remain retrievable at their accepted fixed
or per-call evidence paths, including protected legacy filenames. A custom
historical path outside these protected locations returns an explicit unknown
outcome, as does protocol-1 writer evidence. Current guards cannot establish
whether an older writable runtime already altered retained files. Historical
retrieval therefore preserves the old read-only evidence contract, without
claiming retrospective tamper protection. Mixed old and new writable runtimes
must not share an execution snapshot.

## Weft admission receipts

The transport normalizes `weft.run.receipt.v1` into `WeftRunReceipt`.
A validated `not_accepted` receipt permits bounded admission retries and then
local fallback only when it claims no durable job: its job ID is absent or empty,
`accepted_immediately` is false, and `deduplicated` is false or absent.
Accepted, deduplicated, and unreadable receipts never authorize local fallback.
An unreadable receipt triggers an exact-assignment job probe before the transport
reports an unknown outcome.

An optional `rejection` object supplies a nonempty string `code` and a string
`detail` containing at most 1024 UTF-8 bytes of valid Unicode. It is valid only
with `not_accepted`; `null`, missing required fields, malformed values, and
contradictory acceptance claims make the receipt unreadable. Unknown nonempty
codes remain valid for compatibility with future Weft publishers. Older
publishers may omit the object.

Normalized execution records retain the object at
`admission.attempts[].receipt.rejection`, including when a later attempt is
accepted or local fallback fails. Each attempt also retains the existing
`submission_stderr` tail as raw diagnostic evidence. The final normalized
`receipt` includes its rejection when present; serialization omits the field
when absent. Codes and human-only details explain refusals without changing
retry policy or granting fallback authority.

## Shared provider status

`agent-execution provider` records provider availability, authentication,
inventory, quota, and transport observations in `provider-status/v2`. Writers
commit immutable events under `~/.claude/state/provider-status-v2` before a
best-effort background upload to the `weft-results` R2 bucket. Unsent events
remain in the local outbox. An R2 outage never blocks model execution.

Account names and credential values are excluded. Credential identities become
HMAC fingerprints, which allow observations about one account to remain
separate from another without publishing either identity.
Fingerprints use a persistent host-local salt by default. Set the same
`AGENT_PROVIDER_STATUS_SALT` on participating hosts only when exact account
identities should produce comparable fingerprints. Observations retain their
host and OS-user provenance even when fingerprints match.
Probes read exact identities from `omp usage --json` and fingerprint them
before persistence. Display-redacted identifiers are unsuitable for account
matching because their masks depend on the other accounts in the report.
The Python `probe_omp()` API returns the observation list; raw usage reports
stay inside the probe.

```sh
agent-execution provider probe --sync --json
agent-execution provider status --refresh --json
agent-execution provider sync --json
```

The status command emits `provider-status-snapshot/v2`. Consumers should use
this CLI contract instead of reading the cache directory. Every observation
includes its host, OS user, route, billing pool, credential fingerprint,
provenance, observation time, and expiration time. Stale observations remain
visible but do not make a route unavailable.

Local reads use a compact projection of the latest raw observation for each
subject and kind. Warm reads scale with distinct subjects and diagnostics,
not accumulated event history. Host filtering, expiration, and the 30-day
retention window are evaluated on every read. Equal observation instants are
ordered by event ID.

Writers use a shared process lock and validate event-directory inventories.
With a valid projection, an uncontended write enumerates event names and
inodes without reading historical event bodies. Workers retained across a
rolling upgrade can finish using a non-locking runtime; their changes are
reconciled rather than hidden by the projection.

Before starting consumers with short timeouts, run this once on each host
to build the projection:

```sh
agent-execution provider status --json
```

The initial read, recovery from a missing, incompatible, or corrupt
`projection.json`, and reconciliation after external event-directory changes
replay the event log. These operations can exceed a consumer's normal timeout,
and other readers wait behind a rebuilding process. Updates from retained
non-locking workers can therefore cause latency spikes during an upgrade.

Cache publication requires a same-filesystem clock observation beyond the
event directories' recorded change times. If that cannot be established within
a bounded wait, or enumeration or event reads fail, the incomplete or unverified
view is not cached. Readers return readable event data with `diagnostics`.
Writers in this version require the registry lock before publishing observations.
Event files are immutable; repairs must use atomic replacement or removal,
not in-place content edits.

Sync reports malformed observations, transport failures, and local cache-write
failures in `diagnostics` while processing independent observations. Failed
uploads and failed moves into the local cache retain their outbox files for a
later sync. Remote observation keys must agree with the event's host and ID;
local filenames must agree with the event ID. Unreadable local events remain
on disk and appear in snapshot diagnostics.

## Checks

```sh
just test    # unittest suite + bun SDK-policy test
just check   # lint + typecheck + test
```

## Provenance

Extracted from `agent-review` revision `bafd59657e07`
(`packages/agent-execution`), with the worker-owned tests
(`tests/test_worker.py`, `tests/test_omp_install.py`,
`tests/omp_sdk.test.ts`) and `tests/support/omp.py`.
