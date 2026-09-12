# agent-execution

Constrained coding-agent execution and optional durable Weft transport.

`agent_execution` owns worker execution (`agent-execution-worker`), process
containment, raw transcript evidence, cost limits, credential observation, and
the optional OMP runtime. Review policy, manifests, state, findings, and
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
