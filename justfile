sources := "src tests"

default:
    @just --list

format:
    uv run ruff format {{sources}}

fix: format
    uv run ruff check --fix {{sources}}

lint:
    uv run ruff format --check {{sources}}
    uv run ruff check {{sources}}

typecheck:
    uv run ty check --python-version 3.10 {{sources}}

# Contention decides whether a failure here is evidence. A timeout under heavy
# load is contention rather than a finding, and its victim is whichever test
# ran first rather than a test worth labelling slow — so the load is printed
# with every run instead of being remembered at the moment a suite goes red.
# Measured 2026-09-24: one test took 21,496ms at load 272/8 cores and 646ms on
# studio. See the running-tests skill; offload with `agent-host-sync workspace run`.
load:
    @printf 'load %s on %s cores\n' "$(sysctl -n vm.loadavg | tr -d '{}' | awk '{print $2}')" "$(sysctl -n hw.ncpu)"

test: load
    uv run python -m unittest discover -s tests -v
    bun test tests/omp_sdk.test.ts

# Provision the pinned optional OMP SDK; not required by `test`.
omp-runtime:
    uv run agent-execution-worker install-omp

check: lint typecheck test
