# Putnam-Inspired Cognitive Architecture for AGI

This document adapts Peter Putnam's proposed model of human cognition into an
engineering blueprint for this repository. The aim is to ground the existing
modules—planner, critic, memory, world model, and governance—in a layered
architecture that mirrors Putnam's view of perception, cognition, and action.

## 1. Sensory Interface Layer

Putnam describes the brain as continuously ingesting multi-modal observations
and translating them into abstracted signals. In this repository, tool invocations
represent that sensory interface.

* **Tool Catalog (`agi/src/core/tools/`)** – Each tool provides a controlled
  I/O surface that the orchestrator can query for structured information.
* **Telemetry (`agi/src/core/telemetry.py`)** – Captures raw signals emitted by
  the system during runs, giving downstream components access to sensory traces.

### Engineering Implications

1. Standardise tool metadata to encode sensory modality, latency, and trust level.
2. Extend telemetry events to include provenance of inputs that triggered tool
   calls, mirroring Putnam's emphasis on contextualised perception.

## 2. Perceptual Abstraction Layer

After sensory ingestion, Putnam argues for a stage that compresses observations
into stateful representations. The AGI stack achieves this by storing structured
records in memory and updating the world model.

* **Memory Store (`agi/src/core/memory.py`)** – Append-only event log preserving
  perceptual snapshots and critiques.
* **Memory Retrieval (`agi/src/core/memory_retrieval.py`)** – Supplies context to
  planners via semantic and temporal search, acting as the abstraction filter.
* **World Model (`agi/src/core/world_model.py`)** – Maintains beliefs and
  hypotheses that summarise the current state of the environment.

### Engineering Implications

1. Implement richer encoders that transform raw tool outputs into summarised
   memory entries.
2. Introduce confidence scoring in memory retrieval to prioritise reliable
   abstractions when forming plan contexts.

### Recent Progress

* **Confidence-aware retrieval (`agi/src/core/memory_retrieval.py`)** – Semantic
  slices now annotate each match with a lexical confidence score along with
  aggregate statistics surfaced to planners and telemetry. This allows the
  orchestrator to weight contextual evidence during hypothesis formation.
* **Provenance-aware telemetry (`agi/src/core/orchestrator.py`)** – Tool events now
  include explicit references to the hypotheses and memory records that triggered
  execution, while episodic entries record sensor metadata and summarised outputs
  to tighten the perception-to-abstraction loop.
* **Context feature extraction (`agi/src/core/planner.py`)** – Planner payloads aggregate
  keywords, sensor modalities, and safety tiers from contextual memory so the workspace
  can prioritise evidence-rich signals when selecting plans.
* **Faiss vector memory (`agi/src/core/vector_index.py`)** – Memory entries are embedded
  into a Faiss-backed index, with similarity scores surfaced alongside lexical confidence
  so the workspace can rank perceptual evidence using both symbolic and geometric cues.
* **Variance-tracking world model (`agi/src/core/world_model.py`)** – Beliefs now retain
  structured evidence with per-source weights and inferred confidence intervals, allowing
  downstream planners and governance guards to reason about uncertainty and conflicting
  observations without reconstructing provenance.

## 3. Cognitive Workspace Layer

Putnam emphasises a workspace where competing hypotheses are evaluated against
stored knowledge. The planner, critic, and orchestrator collectively realise this
workspace.

* **Planner (`agi/src/core/planner.py`)** – Generates candidate plans from goals
  and hypotheses, informed by retrieved memory context.
* **Critic (`agi/src/core/critic.py`)** – Provides reflective feedback on plans,
  mirroring Putnam's evaluative mechanisms.
* **Orchestrator (`agi/src/core/orchestrator.py`)** – Coordinates planning,
  memory updates, and action execution, acting as the central workspace manager.

### Engineering Implications

1. Maintain an explicit working memory structure inside the orchestrator to track
   active hypotheses and deliberation history across replans.
2. Capture critic feedback with rationale tags so subsequent planning rounds can
   weight hypotheses by justification strength.

### Recent Progress

* **Working memory persistence (`agi/src/core/orchestrator.py`)** – Each run now
  writes a JSON snapshot of the deliberation workspace alongside the manifest,
  emits telemetry when the snapshot is recorded, and includes loader utilities
  plus summarisation helpers (`agi/src/core/reflection.py`) for reflective
  analyses. Insights are persisted into episodic memory via the executive
  (`agi/src/core/executive.py`) and surfaced through the evaluation harness
  (`agi/src/evals/harness.py`) so downstream runs and scorecards can leverage the
  Putnam-style workspace for continual learning.
* **Reflection consolidation (`agi/src/memory/reflection_job.py`)** – A scheduled
  job aggregates `reflection_insight` records, writes summaries back to memory,
  and nudges the world model with safety-weighted belief updates so persistent
  caution signals bias future planning.
* **Reflective planning bias (`agi/src/core/planner.py`)** – Planner prompts now
  include aggregated insight summaries and per-hypothesis reflective context,
  allowing deliberation tags (e.g., safety critiques) to influence plan selection
  heuristics directly.
* **Negotiation memory persistence (`agi/src/core/orchestrator.py`)** – Agent negotiation
  transcripts are captured in episodic memory with aggregate collaboration stats, and
  experience replay now highlights cross-agent negotiation patterns for future planning.

## 4. Decision and Action Layer

Putnam's architecture culminates in selection of actions that satisfy both
internal drives and external constraints. In this repository, decision-making is
anchored in the orchestrator and executed through tool calls, while safety and
policy enforce alignment.

* **Plan Execution (`agi/src/core/orchestrator.py`)** – Executes approved plan
  steps, records provenance, and handles replanning triggers.
* **Governance (`agi/src/governance/gatekeeper.py`)** – Applies policy checks to
  ensure actions remain within acceptable bounds.

### Engineering Implications

1. Augment plan execution with real-time risk assessments that incorporate
   gatekeeper feedback mid-run.
2. Provide simulation tools (see `agi/src/core/tools/sim_physics.py`) to test
   candidate actions before committing to real-world execution.

### Recent Progress

* **Runtime risk loop (`agi/src/core/orchestrator.py`)** – The orchestrator now
  performs a gatekeeper-backed risk assessment immediately before every tool
  invocation, emits telemetry events, and records the outcome in working
  memory and manifests. This realises Putnam's emphasis on contextual action
  selection by letting policies react to evolving conditions during execution.

## 5. Reflective Learning Layer

Putnam highlights continual self-improvement through reflection on outcomes.
The current stack already stores critiques and tool outcomes; this layer formalises
how those artefacts drive learning.

* **Critique Memory (`agi/src/core/orchestrator.py`)** – Persists critic feedback
  into memory entries for future runs.
* **Evals Harness (`agi/src/evals/harness.py`)** – Supplies structured tasks that
  measure capability growth over time.
* **Experience replay (`agi/src/memory/experience.py`)** – Condenses manifests and
  working-memory traces into reusable knowledge chunks appended to memory for
  subsequent planning cycles.

### Engineering Implications

1. Build periodic reflection jobs that mine memory for recurring failure modes
   and feed summary prompts into new planning sessions.
2. Use evaluation telemetry to adjust planner heuristics, closing the loop
   between performance metrics and behaviour updates.

## Putting It All Together

The following diagram summarises the Putnam-inspired layering and its mapping to
existing modules:

```
+-------------------------+
| Reflective Learning     |
| - Memory critiques      |
| - Evaluation harness    |
+-------------------------+
| Decision & Action       |
| - Orchestrator          |
| - Governance            |
+-------------------------+
| Cognitive Workspace     |
| - Planner               |
| - Critic                |
| - Orchestrator (loop)   |
+-------------------------+
| Perceptual Abstraction  |
| - Memory store/retrieval|
| - World model           |
+-------------------------+
| Sensory Interface       |
| - Tools                 |
| - Telemetry             |
+-------------------------+
```

Engineering teams can reference this hierarchy when prioritising features or
conducting design reviews. Each layer has clear responsibilities, making it
simpler to reason about new capabilities while staying aligned with Putnam's
cognitive principles.
