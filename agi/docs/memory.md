# Memory subsystem

AGI threads context across tool invocations via a two-tier memory system when it
is enabled. This document covers the working-memory cache, the episodic store,
and how the orchestrator coordinates them during a run.

## Overview

- **Working memory** is a lightweight, per-run cache. It keeps the most recent
  tool episodes close to the next invocation so that intermediate results remain
  available without re-querying persistent storage.
- **Episodic memory** is a persistent log of significant tool executions. It is
  durable across runs and serves as the backing store for longer-term recalls.

Both tiers are optional and can be disabled entirely, yielding stateless tool
executions.

## Execution lifecycle

1. **Hydration.** Before executing a plan the orchestrator primes working
   memory. It pulls recent episodic entries for the relevant claims and tools
   and merges them into the cache so the upcoming tool calls receive enriched
   context.
2. **Tool execution.** Each tool invocation receives a `RunContext` containing
   the hydrated working-memory slice together with a
   `recall_from_episodic()` helper. Tools can call the helper with optional
   `tool`, `limit`, and `text_query` arguments to retrieve additional history on
   demand.
3. **Commit.** Once a tool completes, the orchestrator promotes the new episode
   back into the cache and—if it is deemed significant—appends it to episodic
   memory for future runs.

## Runtime configuration

Memory is toggled via the `AGI_ENABLE_MEMORY` environment variable:

```bash
# enable memory (default)
export AGI_ENABLE_MEMORY=1

# disable memory hydration and storage
export AGI_ENABLE_MEMORY=0
```

When disabled the orchestrator still constructs a valid `RunContext`, but the
working-memory list is empty and `recall_from_episodic()` is a no-op.

By default the episodic log is written to `<working_dir>/memory.jsonl` (with
`working_dir` defaulting to `artifacts/`) and the working cache remains in
memory. Callers may supply custom `MemoryStore` or `WorkingMemory` instances to
override these defaults while keeping the hydration pipeline intact.

## Validating behaviour

Run the focused tests below to verify the end-to-end memory flow:

```bash
pytest agi/tests/test_orchestrator.py::test_orchestrator_hydrates_working_memory \
       agi/tests/test_tools_contract.py::test_retrieval_tool_filters_memory
```

These tests confirm that working memory is hydrated before tool execution, that
episodic recalls respect filtering semantics, and that new episodes are
committed for subsequent runs.
