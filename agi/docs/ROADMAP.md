# AGI Program Roadmap

This roadmap outlines the staged development plan for the AGI stack in this
repository. It is organised into capability milestones that build on the
existing `core`, `governance`, and `evals` packages so teams can prioritise the
next engineering and research investments.

## Vision

Deliver an aligned artificial general intelligence platform that can plan,
execute, and self-reflect on complex tasks while remaining verifiably safe and
governable. The program emphasises modular components so progress can be made
incrementally without blocking critical research tracks.

## Milestone Overview

| Phase | Focus | Target Outcomes |
| --- | --- | --- |
| **Phase 0: Baseline Hardening** | Testing, traceability, observability | Reliable orchestrator runs, reproducible artefacts, actionable telemetry |
| **Phase 1: Competent Autonomy** | Richer planning/execution loop | Hierarchical planning, tool competency, world model persistence |
| **Phase 2: Reflective Learning** | Memory, model updates, metacognition | Continual learning with critique feedback, uncertainty-aware updates |
| **Phase 3: Societal Alignment** | Governance, multi-agent safety | Policy enforcement, value alignment experiments, oversight tools |
| **Phase 4: Open Ecosystem** | Extensibility, external integrations | Tool/plugin marketplace, evaluation leaderboards, community contributions |

## Phase 0 — Baseline Hardening

* **Test Coverage Expansion** – Extend unit and contract tests across
  `core` modules (planner, critic, world model) and tools to ensure deterministic
  behaviour under failure cases.
* **Run Manifest Standardisation** – Formalise the JSON schema written by the
  orchestrator so experiment logs can be parsed downstream. Introduce schema
  validation tests.
* **Operational Telemetry** – Instrument the orchestrator and `MemoryStore` with
  structured logging to surface latency, error rates, and provenance health.
* **Developer Tooling** – Provide CLI helpers for spinning up runs, inspecting
  artefacts, and replaying tool traces. Publish notebooks that demonstrate the
  current capability envelope.

## Phase 1 — Competent Autonomy

* **Hierarchical Planning** – Enhance the planner to support sub-goals and
  conditional branches while keeping compatibility with the `Plan` dataclass.
  Introduce partial plan execution and re-planning hooks in the orchestrator.
  (Initial tracing + manifest support shipped in v0.3; next step is to expand
  LLM prompts and replanning heuristics.)
* **Tool Abstractions** – Expand the tool library (simulation, retrieval, coding)
  with capability metadata and automatic safety-level negotiation.
* **Belief Tracking Persistence** – Back the `WorldModel` with durable storage so
  beliefs survive restarts and can be versioned for auditability.
* **Memory Retrieval APIs** – Provide semantic and temporal search endpoints on
  top of `MemoryStore` to support planner context retrieval.

## Phase 2 — Reflective Learning

* **Critic Feedback Loop** – Allow the critic to suggest plan amendments and
  feed critiques into memory for future planning episodes.
* **Uncertainty-Aware World Model** – Incorporate confidence intervals and
  multiple evidence types into `Belief` updates, including unit conversions and
  conflicting sources.
* **Experience Replay** – Build summarisation pipelines that condense artefacts
  into reusable knowledge chunks stored in memory.
* **Self-Evaluation Tasks** – Extend `evals` harness tasks to include
  longitudinal benchmarks that track learning progress across runs.

## Phase 3 — Societal Alignment

* **Governance Policies** – Expand the `Gatekeeper` interface with policy packs
  and experiment with dynamic risk scoring informed by execution traces.
* **Oversight Interfaces** – Develop dashboards for human-in-the-loop review of
  plans, tool calls, and belief updates, with override capabilities. (Prototype
  oversight console and interactive gatekeeper shipped in v0.5, serving manifests,
  telemetry, and a live approval queue via `agi-cli oversight serve`.)
* **Multi-Agent Coordination** – Prototype cooperative and adversarial agent
  setups to stress-test safety controls and memory isolation. (Initial
  orchestrator support for agent assignments landed alongside manifest v0.4; negotiation
  transcripts now persist into episodic memory with collaboration analytics.)
* **Value Alignment Research** – Integrate normative modelling experiments and
  alignment evaluations using external datasets.

## Phase 4 — Open Ecosystem

* **Plugin Architecture** – Define stable APIs for third-party tools and
  planners, including capability declaration, safety certification, and sandbox
  requirements.
* **Benchmark Leaderboards** – Publish public leaderboards powered by the evals
  harness to encourage reproducible comparisons.
* **Contribution Guidelines** – Document contribution pathways, coding
  standards, and governance procedures for external collaborators.
* **Deployment Readiness** – Package the system for on-prem and cloud
  deployments with infrastructure-as-code templates and compliance checklists.

## Cross-Cutting Workstreams

* **Security & Privacy** – Implement secrets management, data minimisation, and
  secure tool execution sandboxes across phases.
* **Human Experience** – Develop UX patterns for prompt design, monitoring, and
  failure handling to keep operators in control.
* **Ethics & Evaluation** – Embed ethical review checkpoints and fairness
  audits into the release process.

## Governance Model

1. **Quarterly Roadmap Reviews** – Revisit milestones, update priorities, and
   capture research learnings.
2. **Capability Gates** – Require successful evaluation runs and safety reviews
   before promoting features between phases.
3. **Documentation Debt Sprints** – Dedicate cycles to update READMEs, API docs,
   and notebooks alongside major feature deliveries.

## References

* `agi/src/core/orchestrator.py` — execution engine coordinating planning,
  tooling, memory, and world model updates.
* `agi/src/core/memory.py` — append-only memory store with indexing facilities.
* `agi/src/core/world_model.py` — belief tracking with logistic updates.
* `agi/src/governance/gatekeeper.py` — policy enforcement entrypoint.
* `agi/src/evals/harness.py` — evaluation harness for running benchmark suites.
