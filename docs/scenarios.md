# Scenario Specs (Core Gameplay)

This document defines success conditions, episode termination, and core mechanics for each scenario. These are intended to remain stable for benchmarking.

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
