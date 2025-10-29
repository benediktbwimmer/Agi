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

## 5. Reflective Learning Layer

Putnam highlights continual self-improvement through reflection on outcomes.
The current stack already stores critiques and tool outcomes; this layer formalises
how those artefacts drive learning.

* **Critique Memory (`agi/src/core/orchestrator.py`)** – Persists critic feedback
  into memory entries for future runs.
* **Evals Harness (`agi/src/evals/harness.py`)** – Supplies structured tasks that
  measure capability growth over time.

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
