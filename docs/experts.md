# Expert Planners

This document describes the centralized expert planners for each scenario, how they work, and how to use them for data collection / imitation learning. These experts are **centralized** (access to full env state) and are intended as **sanity-check and data generation policies**, not as benchmarked DTDE policies.

## Overview

Centralized experts exist for all three scenarios:

- Pipeline Assembly: `pipeline_planner_comm` (centralized planner + comm broadcast)
- Energy Grid: `energy_planner_comm` (centralized planner + comm broadcast)
- Signal Hunt: `signal_hunt_planner_comm` (centralized planner + comm broadcast)

These policies are deterministic and should solve their respective scenarios under default conditions. They can be used for:

- Solvability checks (prove that a solution exists)
- Imitation learning (generate expert trajectories)
- Regression tests for map generation and reward logic

## Scenario Experts

### Pipeline Assembly

**Policy:** `pipeline_planner_comm`  
**Where:** `syncorsink/policies/planner.py` and `syncorsink/policies/planner_comm.py`

**Logic:**
- Selects the first available stage whose dependencies are satisfied.
- Computes required resources (multiset) for that stage.
- Assigns agents to required resources by greedy shortest-path cost.
- Delivers resources to the stage station.
- Synchronizes all agents at the station for sync-required stages.

**Comm broadcast:**
`[12, stage_id, station_x, station_y, req_len, req1, req2, ...]`

### Energy Grid

**Policy:** `energy_planner_comm`  
**Where:** `syncorsink/policies/planner.py` and `syncorsink/policies/planner_comm.py`

**Logic:**
- Tracks node energy and prioritizes the lowest-energy node.
- If an agent is carrying a resource, it routes to the matching node type.
- Otherwise it navigates to the nearest resource of the target node type.

**Comm broadcast:**
`[16, node_x, node_y, node_type]`

### Signal Hunt

**Policy:** `signal_hunt_planner_comm`  
**Where:** `syncorsink/policies/planner.py` and `syncorsink/policies/planner_comm.py`

**Logic:**
- Uses the true target location (full state).
- All agents navigate to target and synchronize on scan.

**Comm broadcast:**
`[17, target_x, target_y]`

## Data Collection (Imitation Learning)

Example: generate expert rollouts with rendering (optional) to inspect behavior:

```bash
python examples/eval_run.py --scenario pipeline_assembly --episodes 5 --policy pipeline_planner_comm --render
```

For dataset generation, run multiple episodes and store step-level `(obs, action, info)` tuples in a custom collector. A simple approach:

1. Wrap the policy to log actions per step.
2. Capture `obs` and `info` for each agent.
3. Serialize to JSONL or numpy arrays for training.

If you want, we can add a dedicated `examples/collect_expert.py` that saves a dataset in a stable format.

## Notes

- These experts are centralized and not intended as DTDE baselines.
- Communication cost still applies; `*_comm` policies broadcast only when the message changes.
- For a challenging DTDE benchmark, use the region-only comm baselines in the pipeline presets.
