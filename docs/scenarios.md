# Scenario Specs (Core Gameplay)

This document defines success conditions, episode termination, and core mechanics for each scenario. These are intended to remain stable for benchmarking.

Scenario tier metadata is defined in `docs/scenario_registry.md`. The scenarios
below are `core` diagnostic tasks: they isolate specific communication and
coordination properties before richer `advanced` or `procedural` scenario packs
are added.

## Shared
- Agents act simultaneously each step.
- Actions: move, interact, pickup, drop.
- Episode ends on scenario success or `max_steps`.
- Communication is token‑bounded and can be penalized.

## A) Task Planning — “Pipeline Assembly”

**Theme:** Multiple agents must assemble a multi‑step pipeline with dependencies.  
**Core coordination:** joint planning + sequencing + spatial coordination.

**Mechanics**
- The goal is to build a device with 3–6 stages.
- Each stage requires combining objects from different rooms and assembling at a station.
- Each agent starts with partial blueprint: agent A knows stages 1–2, agent B knows stages 2–3, agent C knows stages 4–5, etc.
- Some stages require two agents to synchronize (e.g., turn keys or lift heavy object).
- A stage can be completed only when:
  - All dependencies are completed.
  - Required resources have been delivered.
  - If sync is required: at least two agents interact at the station on the same step.

**Why it’s hard**
- Requires semantic communication: “Stage 3 needs a red coil + valve, assemble at north lab.”
- Long‑horizon planning and replanning if resources are blocked or doors locked.

**Generalization knobs**
- Randomized blueprint ordering.
- Randomized resource locations.
- Varying dependency DAG depth/branching.

**Success:** all stages completed.

**Rewards:**
- Stage delivery: `reward_stage`
- Sync completion bonus: `0.5 * reward_stage` per interacting agent
- Final completion: `reward_complete`
- Wrong delivery: subtracts `pipeline_wrong_delivery_penalty` when an agent
  interacts at a station with a carried resource that cannot be accepted there.

**Training/eval events:**
- Positive progress: `picked_resource`, `delivered`, `stage_completed`,
  `sync_complete`, `pipeline_complete`
- Failure/focus labels: `pipeline_wrong_delivery`,
  `pipeline_dependency_blocked`, `pipeline_sync_wait`,
  `pipeline_pickup_miss`, `pipeline_delivery_miss`,
  `pipeline_station_stall_miss`, `pipeline_drop_miss`
- Optional BC action losses supervise correct pickup/delivery and suppress
  bad drops or wrong-station interactions during Pipeline DAgGER.
- Ready-interact action losses are available for BC/PPO ablations that need to
  encourage valid station delivery and sync-completion interactions.
- PPO delivery-progress and navigation action losses are available for
  ablations where policies learn to avoid bad station interactions but still
  fail to move carried resources to the active station or follow visible plans.
- Additional recurrent BC diagnostics can upweight trusted plan-following
  actions and structured Pipeline planner messages, making it easier to test
  whether a model has learned both "what to say" and "how to execute the plan."
- Recurrent actors can also train an experimental Pipeline plan-action head
  (`--bc-pipeline-plan-head-loss-weight`) and audit it with
  `--eval-pipeline-plan-head-threshold`; keep those scores separate from plain
  unassisted results.
- Recurrent actors can train an experimental Pipeline option head
  (`--bc-pipeline-option-loss-weight`) over high-level intents such as pickup,
  deliver, sync, and navigation-to-resource/station; audit it with
  `--eval-pipeline-option-threshold`. By default this decoder only promotes
  low-risk primitive actions; use `--eval-pipeline-option-allow-interact` for
  ablations that let options directly trigger delivery/sync `INTERACT`. Keep
  those decoded scores separate from plain unassisted results.
- Narrow `--bc-pipeline-sync-action-loss-weight` and
  `--rl-pipeline-sync-action-loss-weight` ablations label empty-agent
  rendezvous actions for sync stations once every remaining required resource
  is already carried by the team. They are useful for testing whether learned
  policies can coordinate the final synchronized interaction, but are not
  enabled in the guarded default profile yet.
- `--bc-pipeline-frontier-exploration-action-loss-weight` is an opt-in
  recurrent resource-search auxiliary for runs with exploration memory. It
  labels frontier moves only when the trusted active stage needs a resource that
  is not locally visible, so it does not reveal hidden resource positions.
- Recurrent Pipeline runs can enable `--obs-pipeline-features`, which decodes
  private hints and received planner messages into station/resource
  affordances, including held-resource target cues, without exposing
  simulator-only hidden state.
- `--obs-pipeline-progress-features` is a separate opt-in recurrent feature
  block for durable event-derived progress: completed stages, delivered and
  remaining resources, dependency availability, and sync-wait readiness.
- `--eval-pipeline-navigation-assist` is available for diagnostic recurrent
  runs that need to test whether private hints are executable; add
  `--eval-pipeline-navigation-assist-trust-messages` only for experiments that
  intentionally trust learned messages. Keep assisted results separate from
  unassisted benchmark scores.
- `--eval-pipeline-station-interact-guard` is a narrower diagnostic guard that
  suppresses station `INTERACT` attempts that trusted hints/progress state mark
  as unable to deliver, sync, or complete.
- `--rl-rollout-pipeline-station-interact-guard` applies that same narrow
  guard only while recurrent PPO collects rollouts; the guarded recurrent PPO
  sweep profile enables it by default. The same guarded profile now also turns
  on pickup/delivery/progress labels,
  plan/option/message distillation, interact-gate BC supervision, calibrated
  interact-gate threshold selection, station-guard action labels, proactive
  bad-action labels, focused DAgger replay for delivery-ready states, delivery
  misses, and wrong deliveries,
  plus wrong-delivery provenance labels for Pipeline sweep runs.
  Wrong-station recovery labels are available as explicit BC and PPO rollout
  ablation knobs, but are not part of the guarded default profile.
  Navigation movement-distillation labels are also available as an explicit
  ablation knob, but are not part of the guarded default profile.

## B) Resource Sharing — “Energy Grid”

**Theme:** Agents must maintain a shared energy grid by transporting and distributing resources.  
**Core coordination:** resource allocation + balancing tradeoffs + exploration.

**Mechanics**
- Power nodes periodically drain; if they die, mission fails.
- Resources (fuel cells) appear in unknown locations and must be delivered to nodes.
- Each agent sees only a subset of nodes’ status and a subset of resource spawns.
- Some nodes require multi‑agent activation to recharge (e.g., two switches).
- In‑env specifics:
  - Each node has an energy level that drains each step.
  - Resources are typed; a node only accepts matching types.
  - If a node falls below `sync_threshold`, a recharge requires 2 agents to interact at that node.
  - Resources can spawn stochastically on empty tiles.

**Why it’s hard**
- Requires negotiation (“I’ll take east node if you take west”).
- Requires efficient communication under uncertainty (stochastic spawn).

**Generalization knobs**
- Spawn distributions, number of nodes, time‑pressure.

**Success:** survive until `max_steps` (or use external eval horizon).

**Failure:** any node energy <= 0.

**Rewards:**
- Recharge delivery: `reward_stage`
- Failure penalty: `reward_fail`

## C) Cooperative Search — “Signal Hunt”

**Theme:** Find and decode a hidden target from distributed clues.  
**Core coordination:** shared information + semantic reasoning.

**Mechanics**
- A target artifact is hidden; agents must collect clues from different regions.
- Clues are textual and partial (“target near water + symbol X + altitude > 2”).
- Each agent’s clue is insufficient alone.
- Final confirmation requires joint action (e.g., two agents must “scan” together).
- In‑env specifics:
  - Map uses rooms/doors/occlusion by default.
  - Clue tiles provide textual hints (attribute+object, relational, riddle).
  - Decoy targets are present; scanning decoys incurs penalty.
  - Two agents must interact on the true target within `scan_window` steps.

**Why it’s hard**
- POMDP + partial textual clues makes it communication‑heavy.
- Agents must reason over semantic constraints.

**Generalization knobs**
- Different clue templates, map topologies, distractor objects.

**Success:** 2+ agents scan the true target within the window.

**Failure:** none (default). Episode ends on success or time limit.

**Rewards:**
- Completion: `reward_complete`
- Decoy scan penalty: `decoy_penalty * reward_stage`
