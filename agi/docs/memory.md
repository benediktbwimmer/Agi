# Memory Subsystem

The Putnam-inspired stack keeps short-term and long-term context when the
feature is enabled. Working memory offers a per-run scratchpad, while episodic
memory keeps a durable log of significant tool invocations that can be recalled
in future runs.

## Quick start

```bash
# enable memory (default behaviour)
export AGI_ENABLE_MEMORY=1

# or disable hydration/storage entirely
export AGI_ENABLE_MEMORY=0

# run the focused tests that cover the orchestration-memory loop
pytest agi/tests/test_orchestrator.py::test_orchestrator_hydrates_working_memory \
       agi/tests/test_tools_contract.py::test_retrieval_tool_filters_memory
```

With memory enabled the orchestrator hydrates the cache before each plan,
persists significant episodes, and exposes the relevant slices to tools via the
`RunContext` that is handed to every tool invocation.

## Working versus episodic memory

- **Working memory** stores the latest episodes for the active run. It is kept
  in-process and flushed when the run finishes. Tools receive a serialised view
  of this cache via `RunContext.working_memory`.
- **Episodic memory** persists JSONL entries under `artifacts/memory.jsonl` by
  default. Tools can pull additional context on demand using
  `RunContext.recall_from_episodic(...)`, and the CLI can inspect, search, and
  summarise the log.

You can point the orchestrator at a different episodic file by passing a custom
`MemoryStore` or by setting `Orchestrator.episodic_memory_path`.

## Inspecting memory with the CLI

The bundled Typer CLI exposes helpers for common inspection tasks:

```bash
# show the latest records (filter by type if needed)
agi-cli memory recent artifacts/memory.jsonl --type episode --limit 3

# semantic search with a safety filter
agi-cli memory search artifacts/memory.jsonl "lunar" --type reflection --limit 5

# consolidate reflection insights and write back summaries
agi-cli memory reflect artifacts/memory.jsonl --goal demo --write-back
```

For run-level inspection combine the manifest, working-memory snapshot, and
episodic log:

```bash
agi-cli run inspect artifacts/run_*/ --memory artifacts/memory.jsonl --sample 2
```

## Validating behaviour

The automated checks guarantee that we never regress the hydration and recall
pipeline:

- `agi/tests/test_orchestrator.py` exercises hydration, episodic writes, and
  the working-memory snapshot artefact.
- `agi/tests/test_tools_contract.py::test_retrieval_tool_filters_memory`
  ensures tools can filter both working and episodic memory.
- `agi/tests/test_cli.py` validates the CLI pathways for viewing, searching,
  reflecting, and replaying memory artefacts.

Run the full suite with `pytest` after altering memory-related code. The
tests operate with the lightweight fallback implementations of `pydantic` and
`typer`, so the behaviour is checked even when the real packages are not
available locally.
