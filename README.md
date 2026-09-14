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
  `kimi-code/k3` and `openai-codex/gpt-6-astra`; read-only grounded
  selector rules are unchanged. Its write/edit tools refuse VCS and harness
  metadata such as `.omp`, preserve existing ordinary-file permission bits on
  replacement, and create new files with conservative permissions.

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
