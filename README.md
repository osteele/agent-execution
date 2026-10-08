# agent-execution

Constrained coding-agent execution and optional durable transport through Weft,
a job system that owns scheduling, source snapshots, and artifacts.

`agent_execution` is a Python library for consumers such as Agent Review and
Agent Offload. It owns worker execution, process containment, raw evidence,
cost limits, and observations of capabilities, credentials, quota, generation,
and transport. Consumers own their workflows: review judgments, task state,
patch application, and completion notifications.

The optional OMP (Oh My Pi) coding-agent runtime exposes three tool policies:
`packet-only-no-tools`, `read-only-no-shell`, and `workspace-write-no-shell`.
The writer policy adds confined `execution_write` and `execution_edit` tools;
it does not grant a shell. Tool confinement is not an operating-system sandbox.

- [Install and first integration](#install-and-first-integration)
- [First worker execution](#first-worker-execution)
- [Python API map](#python-api-map)
- [Execution observations](#execution-observations)
- [Worker evidence](#worker-evidence)
- [Weft transport and admission](#weft-transport-and-admission)
- [Shared provider status](#shared-provider-status)

## Install and first integration

From a consuming uv project, with this repository checked out as a sibling:

```sh
uv add ../agent-execution
uv run agent-execution-worker identity
uv run agent-execution-worker install-omp  # Required for OMP execution; Bun >= 1.3.14
```

SDK installation is optional for inspecting status, but required before running
an OMP harness. The package version is currently `0.1.0`, with worker protocol 2.
Use the protocol and source digest, not the package version alone, to identify
the code a dispatch will execute.

Import from the `agent_execution` modules below, not the package root. This
non-generating example observes the current machine without launching a model:

```python
import json

from agent_execution.execution_status import probe_status

status = probe_status(
    harness="omp",
    surface="worker",
    selectors=["anthropic/claude-opus-5-5"],
    timeout=15.0,
)
print(json.dumps(status, indent=2))
```

Save it as `status_example.py` and run `uv run python status_example.py` from
the consuming project. The output is an `agent-execution.execution-status/v1`
envelope. Inspect each fact's state, age, and staleness independently.
Unavailable capabilities or credentials are observations, not a successful
generation test. Without the pinned OMP SDK, OMP capability is unavailable.

An excerpt from one configured executor's output follows. Other envelope fields
are omitted; use the complete command output for validation. The digest and
availability observations depend on the installed build and executor:

```json
{
  "schema_version": "agent-execution.execution-status/v1",
  "worker_identity": {
    "protocol_version": 2,
    "source_sha256": "e32f15bd5044eb1a0f019f0547623705019bd2183d8bd0180f5a52e39b798809"
  },
  "rows": [{
    "subject": {"selector": "anthropic/claude-opus-5-5"},
    "facts": {
      "capability": {"state": "available", "stale": false},
      "quota": {"state": "unknown", "stale": true, "age_seconds": null}
    }
  }]
}
```

Configure the intended harness credentials on the executor. Credentials on the
caller do not establish remote authentication; use `probe_remote_status` for a
remote worker. A successful credential probe neither proves available quota
nor establishes subscription billing.

To package a consumer CLI with this dependency, run
`uv tool install --with ../agent-execution .` from the consumer checkout.
Deploy matching consumer/worker environments and retain the identities used by
in-flight jobs. `agent-execution-worker identity` reports the protocol and
installed source digest. A *paired installer* installs a consumer and its
verified Execution dependency into one immutable release, then switches their
entrypoints together. Where such an installer owns the worker command, do not
replace it with a standalone `uv tool install` of this package.

## First worker execution

This example makes a real model call with read-only tools. Install the pinned
SDK and configure execution-eligible Anthropic credentials first. The zero-dollar
cap requires an observed zero-incremental-cost billing basis; login alone is not
enough. Save as `execute_example.py` and run `uv run python execute_example.py`
from the consuming project:

```python
import json
from pathlib import Path
from uuid import uuid4

from agent_execution.identity import worker_evidence_path
from agent_execution.worker import WorkerResult, execute_worker, installed_worker_identity

call_id = f"example-{uuid4().hex}"
identity = installed_worker_identity()
command = [
    "omp", "-p", "Reply with READY.",
    "--mode", "json", "--cwd", ".",
    "--model", "anthropic/claude-opus-5-5",
    "--execution-tool-policy", "read-only-no-shell",
]
result = execute_worker(
    provider="omp",
    model_call_id=call_id,
    command=command,
    cwd=Path.cwd(),
    timeout=120.0,
    ctx_timeout=30.0,
    expect_protocol=identity.protocol_version,
    expect_source_sha256=identity.source_sha256,
    max_cost_usd=0,
)
if result.status != "completed":
    raise RuntimeError(f"{result.status}: {result.failure}")

evidence_path = Path(worker_evidence_path(call_id))
published = WorkerResult.parse(evidence_path.read_text(encoding="utf-8"))
if (published.model_call_id != call_id
        or published.worker_source_sha256 != identity.source_sha256):
    raise ValueError("Published evidence differs from retained dispatch expectations")
print(json.dumps(published.summary(artifact_path=str(evidence_path)), indent=2))
```

Trimmed output from a verified run:

```json
{
  "schema_version": "agent-execution.worker-result/v1",
  "status": "completed",
  "worker_protocol_version": 2,
  "model_call_started": true,
  "artifact_path": ".agent-execution/results/example-8d009f8c15694e6da714a299844ca4f4.json"
}
```

A successful call prints `status: "completed"` and the protected
`.agent-execution/results/<model-call-id>.json` path. The file contains the
versioned result and native OMP evidence. `preflight_failed`, `harness_failed`,
and `evidence_failed` are distinct failures, not usable successful output.
Each run above deliberately creates a new logical call; orchestration code must
retain and reconcile an existing call identity rather than rerunning it blindly.

## Requirements

- Python >= 3.10; the package is standard-library-only.
- Bun >= 1.3.14 only for the OMP adapter; `install-omp` provisions the
  lockfile-pinned SDK under `~/.local/share/agent-execution/omp-sdk/<version>`.
- `workspace-write-no-shell` admits the writer identities in
  [`OMP_WRITER_SELECTORS`](src/agent_execution/omp_execution.py):
  `anthropic/claude-opus-5-5`, `zhipu-coding-plan/glm-5.3-flash`, `kimi-code/k3`,
  `openai-codex/gpt-6-{sol,luna,astra}`, `openai-codex/gpt-6.1-sol`,
  `google-antigravity/gemini-3.1-pro`, and `google-antigravity/claude-opus-4-6`.
  `gpt-6-sol` is the ordinary OpenAI default; Luna supports focused work, and
  Astra is reserved for explicit largest-model escalation. GLM is the economical
  route for bounded mechanical work. Grounded reviews default to
  `anthropic/claude-opus-5-5` or `openai-codex/gpt-6-sol` for those providers;
  Luna, Astra, and `gpt-6.1-sol` are also admitted as explicit read-only selectors.
  The policy's write/edit tools refuse VCS and harness metadata such as `.omp`, preserve
  existing ordinary-file permission bits on replacement, and create new files
  with conservative permissions. Writes and edits accept UTF-8 output files up to
  2 MiB; read, glob, and search tool results remain capped at 512 KiB (readable
  source files may be up to 4 MiB).

## Python API map

| Module | Entry points | Consumer responsibility |
|---|---|---|
| `agent_execution.execution_status` | `probe_status`, `cached_status`, `probe_remote_status`, `validate_status` | Query the actual executor and exact context; preserve unknown and stale facts. |
| `agent_execution.worker` | `execute_worker`, `WorkerResult.parse`, `installed_worker_identity` | Supply the provider, command, unique call ID, deadlines, and retained protocol/source expectations; interpret the returned status and evidence. |
| `agent_execution.identity` | `worker_evidence_path`, `source_worker_executable` | Retain the dispatched evidence path and source-addressed executable with the request. |
| `agent_execution.weft` | `WeftCommandRunner`, `WeftRunReceipt.parse` | Persist dispatch identity, reconcile accepted or ambiguous work, and acknowledge only after durable consumption. |
| `agent_execution.command` | `CommandResult.mark_consumed`, `CommandResult.acknowledge_refusal` | Validate and durably record acceptance or refusal before acknowledging. |
| `agent_execution.processes` | `run_in_process_group` | Use this lower-level process-lifetime primitive only when worker evidence and tool-policy enforcement are not required. |

`execute_worker` admits the adapters `omp`, `omp-packet`, and `agy`, and publishes
a `WorkerResult` at its protected evidence path. Pass `timeout` for the harness.
The signature also requires `ctx_timeout`; these adapters use native evidence
rather than the `ctx` transcript helper. The recorded harness deadline is not a
total deadline for installation, queueing, or consumer processing. Preflight
and harness failures are returned as result statuses; a returned object alone
is not evidence of successful generation.

Consumers must retain dispatch expectations separately from results. Parse
versioned results and compare them with those expectations rather than trusting
a result to nominate its own source, selector, policy, or evidence path.

## Execution observations

Inspect capabilities and credentials without generating a model response:

```sh
uv run agent-execution-worker execution-status \
    --harness omp --surface offload-task \
    --selector anthropic/claude-opus-5-5 --transport weft --json
uv run agent-execution execution-status \
    --harness omp --surface offload-task --selector kimi-code/k3 --cached --json
```

`surface` is one of `native` (native-harness observation), `worker` (shared worker
execution), or `offload-task` (the writer task context). Supported combinations
of harness, selector, surface, and policy are checked together.

The command observes the machine on which it runs. `--transport` describes the
requested execution context; it does not submit a job. `--requester-host` and
`--requester-user` distinguish the caller from the executor. The default
requester is the local host/user. Both command entrypoints implement the same
`agent-execution.execution-status/v1` contract.

The envelope contains `generated_at`, `worker_identity`, the actual
`observer` host/user, one `rows` entry per exact selector, and `diagnostics`.
Each row contains:

- `subject`: harness, surface, exact selector, tool policy, requested route,
  billing pool, executor host/user, transport, requester host/user, source digest,
  profile, launch-environment fingerprint, observed effective route, and the
  credential fingerprint/scope. A configuration fingerprint is not an account
  identity. Unknown account fingerprints never join evidence across contexts.
- `facts`: separate `capability`, `authentication`, `quota`, `generation`, and
  `transport` observations. Each has `state`, `observed_at`, `expires_at`,
  `age_seconds`, `stale`, `source` (`tool`, `method`), `detail`, and an optional
  typed refusal `condition` (`quota`, `auth`, `network`, `unknown`).
- `credential_basis`: an independent `harness-credential-basis/v1` observation
  with expiry, age, and staleness, or null when absent. Authentication presence
  does not establish subscription billing.

States are `available`, `unavailable`, or `unknown`. Absent facts have null
timestamps/age and `stale: true`. Expired facts retain their observed state and
age; they do not become successful observations. There is no universal ready
flag. Login cannot clear a generation failure, and a successful generation does
not establish remaining quota. Current explicit quota refusals remain distinct
from absent quota measurements.

OMP capability uses the active pinned SDK's exact model registry; authentication
uses credentials eligible for its restricted launch. Anthropic and Antigravity
use OAuth-only credentials. The Zhipu coding-plan route can use `ZAI_API_KEY`.
Read-only Review selectors are checked independently of the writer roster.
Native Claude sign-in is probed with the scrubbed environment, selected profile,
and working directory. The wrapper's auth command bypasses generation routing,
so sign-in cannot establish the effective generation route or its billing basis.
Those native execution facts remain explicitly unobserved; they neither grant
zero-cost permission nor affect OMP facts. Native Claude is not a supported
Weft-worker transport, and native Codex worker execution is retired.

Python consumers use `agent_execution.execution_status`:

- `probe_status(*, harness, surface, selectors=None, transport="local", ... )`
  performs fresh local non-generating probes.
- `cached_status(...)` reads retained observations without probing.
- `probe_remote_status(*, host, account="agent", harness, surface, selectors=None,
  transport="weft", bin_dir=None, worker_executable="agent-execution-worker", ...)`
  queries the executor over bounded SSH. An absolute `bin_dir` selects a pinned
  build; `expected_execution_sha256` verifies it. Coverage, requester/executor
  context, schema, and freshness are validated; transport failures raise
  `ValueError`, never an empty success.
- `validate_status(value, *, expected_execution_sha256=None)` validates an
  envelope at a consumer boundary.
- `record_generation(subject, *, succeeded, condition=None)` records an actual
  provider outcome against its verified prelaunch subject. It rejects another
  executor/build and an unverified native route. Workers call it automatically;
  local preflight, packet-size, and evidence-validation failures are not failed
  generation observations.

The local APIs accept `tool_policy`, `profile`, `environment`, `cwd`,
`requester_host`, `requester_user`, `expected_execution_sha256`, and a positive
finite `timeout`. Remote queries accept profile, tool policy, and working
directory but never forward the caller's secret environment. CLI equivalents
include `--tool-policy`, `--profile`, `--cwd`, `--expect-source-sha256`, and
`--timeout`. Native queries require an explicit tool policy. OMP worker queries
default to `read-only-no-shell`; task queries use `workspace-write-no-shell`;
OMP-packet and agy use `packet-only-no-tools`.

Remote status probes require a working noninteractive SSH connection to the
executor account and the requested worker executable installed there. They
observe that environment; they do not provision it.

Immediately before launch, workers re-observe required facts for their pinned
source and actual tool policy. OMP requires current capability, authentication,
and transport; a current explicit quota refusal also blocks launch. Agy has no
registered auth-status probe: its authentication stays unknown, and its
independent hard-cash guard refuses an unobservable billing basis.

Exact observations are immutable events in the existing provider-status
registry's `events/execution/` namespace, under its shared lock. Their derived
`execution-projection.json` index can be rebuilt; it is not another authority.
This namespace prevents retained older workers from interpreting scoped facts
as route-wide availability. Consumers use the versioned API, not these files.

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

The worker builds a throwaway HOME for each call. It copies only the sign-in
and settings entries from `~/.gemini` listed in `AGY_HOME_ALLOWLIST`, excluding
conversation history, browser profiles, recordings, skills, and temporary data.
It installs a deny-all `PreToolUse` hook, writes an empty MCP config and a
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

## Weft transport and admission

Weft is optional. `agent_execution.weft.WeftCommandRunner` submits through the
Weft CLI, retrieves worker evidence, and returns a `CommandResult`. Construct it
with the execution `host`, adapter `agent`, unique `model_call_id`, and an
explicit `fallback` (`None` disables local fallback). Invoke it with
`runner(command, cwd, timeout)`. Set `project` to the consuming application's
Weft project so that its jobs remain attributable to that consumer.

Persist the dispatch identity before surrendering execution custody. An
accepted job can outlive the local waiter; a lost response is not permission
to launch the same work elsewhere. After validating and durably recording a
result, call `mark_consumed()`. If the consumer instead records a refusal, call
`acknowledge_refusal()` on the returned result or refusal outcome. Retrieval
alone does not mark a successfully retrieved result consumed.

`WeftCommandRunner` accepts `worker_executable` to pin a queued call to a retained
worker. Use `agent_execution.identity.source_worker_executable(source_digest)`
with the same digest as `expected_source_sha256`. The host's installer must bind
that command directly to a verified immutable environment and retain it while
jobs reference it; Agent Review's paired installer provides these bindings.
The default command, `agent-execution-worker`, selects the host's active worker.
Other command names and source-addressed names with a different digest are
refused before submission.
Source-addressed commands retain the source, protocol, and prompt checks, and
retrieval rejects a command name that contradicts its dispatched source digest.

### First remote call

Configure Weft on the submitting machine and a reachable Weft execution host.
The consuming project's directory must be covered by Weft's configured source
roots; an arbitrary temporary directory outside those roots cannot be pinned.
Install the pinned SDK and credentials on the execution host. For worker
bindings, consult the Install section of your Agent Review checkout's README.
Agent Offload users should follow the provisioning contract in their Offload
checkout's README and use `agent-offload task submit --help`.

Set `EXAMPLE_WEFT_HOST` to your configured Weft host name and
`EXAMPLE_WORKER_SHA256` to the source digest of its verified, retained worker.
Its protocol must match this library. The source-addressed executable must
already exist; this example does not install or replace a worker.

This small consumer archives its request and validated result in SQLite before
acknowledging the job. Save it as `weft_example.py`, then run
`uv run python weft_example.py` from the consuming project:

```python
import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4

from agent_execution.identity import source_worker_executable, worker_evidence_path
from agent_execution.weft import WeftCommandRunner
from agent_execution.worker import WORKER_PROTOCOL_VERSION

call_id = f"example-{uuid4().hex}"
source = os.environ["EXAMPLE_WORKER_SHA256"]
command = [
    "omp", "-p", "Reply with READY.",
    "--mode", "json", "--cwd", ".",
    "--model", "anthropic/claude-opus-5-5",
    "--execution-tool-policy", "read-only-no-shell",
]
runner = WeftCommandRunner(
    host=os.environ["EXAMPLE_WEFT_HOST"],
    agent="omp", model_call_id=call_id, project="execution-example",
    fallback=None, allow_queue=True, max_cost_usd=0,
    expected_source_sha256=source,
    worker_executable=source_worker_executable(source),
)
request = {
    "model_call_id": call_id, "host": runner.host, "command": command,
    "expected_source_sha256": source,
    "expected_worker_protocol_version": WORKER_PROTOCOL_VERSION,
    "worker_result_path": worker_evidence_path(call_id),
}
with sqlite3.connect("execution-example.sqlite") as db:
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS calls "
               "(id TEXT PRIMARY KEY, request TEXT, execution TEXT, result TEXT)")
    db.execute("INSERT INTO calls (id, request) VALUES (?, ?)",
               (call_id, json.dumps(request)))
    db.commit()
    try:
        result = runner(command, Path.cwd(), 300.0)
    finally:
        db.execute("UPDATE calls SET execution = ? WHERE id = ?",
                   (json.dumps(runner.last_execution), call_id))
        db.commit()
    worker = result.worker_result
    if worker is None or worker.status != "completed":
        raise RuntimeError("No validated completed worker result; retain the request")
    db.execute("UPDATE calls SET result = ? WHERE id = ?", (worker.to_json(), call_id))
    db.commit()
    result.mark_consumed()
    print(json.dumps(result.execution, indent=2))
```

The printed execution record includes the accepted receipt and acknowledgment
state. This example's consumer accepts a validated completed transcript for
archival; an application must also apply its own output-validation policy.
If waiting or acknowledgment fails, retain the database and reconcile that
call's job through `lookup_submitted_job` and `WeftCommandRunner.retrieve`.
Do not rerun the script to replace an unresolved call with a fresh identity.

### Admission receipts

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

Use the exact-context [execution observations](#execution-observations) API for
worker admission. This broader provider-status surface supports shared
inventory and status displays; route-wide availability is not launch permission.

`agent-execution provider` records provider availability, authentication,
inventory, quota, and transport observations in `provider-status/v2`. Writers
commit immutable events under `~/.claude/state/provider-status-v2`. Automatic
success/refusal records start a best-effort background push to the `weft-results`
R2 bucket when a transport is available. CLI `provider probe` and
`provider observe` synchronize in the foreground only with `--sync`; otherwise,
their events await a later sync, such as `provider sync`. Without an available
transport, events remain in the local outbox. An R2 outage does not block model
execution.

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
uv run agent-execution provider probe --sync --json
uv run agent-execution provider status --refresh --json
uv run agent-execution provider sync --json
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
uv run agent-execution provider status --json
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

## Next steps

- [Worker API](src/agent_execution/worker.py) and
  [transport API](src/agent_execution/weft.py): exact Python signatures.
- [Execution contract](spec/agent-execution.allium): worker and transport behavior.
- [Weft documentation](https://github.com/osteele/weft): host configuration,
  scheduling, snapshots, and artifact operations.
- Your consumer's installation documentation: Agent Review's README describes
  paired releases; Agent Offload's README describes task-worker provisioning.
- `uv run agent-execution-worker --help` and
  `uv run agent-execution provider --help`: installed command surfaces.

## Checks

```sh
just test    # unittest suite + bun SDK-policy test
just check   # lint + typecheck + test
```

## License

MIT. See [LICENSE](LICENSE).
