# Memory Subsystem

The Putnam-inspired stack threads context across tool invocations via a
two-tier memory system. Working memory offers a per-run scratchpad, while
episodic memory keeps a durable log of significant episodes that future runs
can recall.

## Overview

- **Working memory** is an in-process cache that stores the latest episodes for
  the active run. It is flushed once the run completes. Tools receive a
  serialised view via `RunContext.working_memory`.
- **Episodic memory** persists JSONL entries under `artifacts/memory.jsonl` by
  default. Tools can pull additional context on demand using
  `RunContext.recall_from_episodic(...)`, and the CLI can inspect, search, and
  summarise the log.

Both tiers can be disabled for stateless execution.

## Execution lifecycle

1. **Hydration.** Before executing a plan, the orchestrator primes working
   memory. Recent episodic entries for relevant claims/tools are merged into the
   cache so upcoming tool calls receive enriched context.
2. **Tool execution.** Each invocation receives a `RunContext` containing the
   hydrated working-memory slice plus a `recall_from_episodic()` helper. Tools
   may call the helper with `tool`, `limit`, or `text_query` filters to fetch
   additional history on demand.
3. **Commit.** After a tool completes, the orchestrator promotes the new episode
   into the cache and—if significant—appends it to episodic memory for future
   runs.

## Runtime configuration

Memory is toggled via the `AGI_ENABLE_MEMORY` environment variable:

```bash
# enable memory (default)
export AGI_ENABLE_MEMORY=1

# disable hydration/storage entirely
export AGI_ENABLE_MEMORY=0
```

When disabled the orchestrator still constructs a valid `RunContext`, but the
working-memory list is empty and `recall_from_episodic()` becomes a no-op. You
can point the orchestrator at a different episodic file by passing a custom
`MemoryStore` or setting `Orchestrator.episodic_memory_path`.

## Semantic retrieval

When `faiss-cpu` is installed the semantic search path uses a hashed vector
index (`MemoryVectorIndex`). Records that include an `embedding` payload are
added automatically, and semantic search responses expose
`vector_similarity` scores alongside lexical hits. See
`tests/test_orchestrator.py::test_orchestrator_surfaces_vector_similarity_in_memory_context`
for an end-to-end example. Without embeddings the index falls back to hashing
record text, still surfacing similarity metadata when the query and stored
records share vocabulary.

## Inspecting memory with the CLI

The Typer CLI exposes helpers for common inspection tasks:

```bash
# show the latest records (filter by type if needed)
agi-cli memory recent artifacts/memory.jsonl --type episode --limit 3

# semantic search with safety filters
agi-cli memory search artifacts/memory.jsonl "lunar" --type reflection --limit 5

# consolidate reflection insights and write back summaries
agi-cli memory reflect artifacts/memory.jsonl --goal demo --write-back

# inspect an orchestrator run directory with working/episodic context
agi-cli run inspect artifacts/run_*/ --memory artifacts/memory.jsonl --sample 2
```

## Validating behaviour

Run the focused tests below to verify the hydration/commit loop:

```bash
pytest agi/tests/test_orchestrator.py::test_orchestrator_hydrates_working_memory \
       agi/tests/test_tools_contract.py::test_retrieval_tool_filters_memory
```

End-to-end coverage ensures working memory is hydrated before execution,
episodic recalls respect filters, and significant episodes are committed for
subsequent runs. The CLI tests also validate inspection flows (`agi/tests/test_cli.py`).
