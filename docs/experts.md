# Expert Planners

This document describes the centralized expert planners for each scenario, how they work, and how to use them for data collection / imitation learning. These experts are **centralized** (access to full env state) and are intended as **sanity-check and data generation policies**, not as benchmarked DTDE policies.

## Overview

Centralized experts exist for all three scenarios:

- Pipeline Assembly: `pipeline_planner_comm` (centralized planner + comm broadcast)
- Energy Grid: `oracle_planner` / `energy_oracle_planner` (stateful centralized dispatcher)
- Signal Hunt: `signal_hunt_planner_comm` (centralized planner + comm broadcast)

These policies are deterministic and should solve their respective scenarios under default conditions. They can be used for:

- Solvability checks (prove that a solution exists)
- Imitation learning (generate expert trajectories)
- Regression tests for map generation and reward logic

## Scenario Experts

## Current Acceptance Status

The automated expert acceptance tests cover:

- `signal_hunt` 8x8, 16x16, and 32x32 with `signal_hunt_planner_comm`
- `energy_grid` 8x8 easy, 16x16 hard, and 32x32 hard with `energy_oracle_planner`
- `pipeline_assembly` 8x8, 16x16, 24x24, and 32x32 with `pipeline_planner_comm`
  - The 16x16 acceptance case runs a 32-episode seed sweep.
  - The 24x24 and 32x32 acceptance cases cover larger-map traffic deadlocks.

Stress sweeps beyond the acceptance window are still useful, but the known
scaled-pipeline deadlocks are now covered by tests.

### Pipeline Assembly

**Policy:** `pipeline_planner_comm`  
**Where:** `syncorsink/policies/planner.py` and `syncorsink/policies/planner_comm.py`

**Logic:**
- Selects the first available stage whose dependencies are satisfied.
- Computes required resources (multiset) for that stage.
- Assigns agents to required resources by greedy shortest-path cost.
- Delivers resources to the stage station.
- Synchronizes all agents at the station for sync-required stages.
- Resolves carrier-vs-pickup corridor conflicts so scaled maps do not deadlock
  when agents meet head-on in narrow passages.

**Comm broadcast:**
`[12, stage_id, station_x, station_y, req_len, req1, req2, ...]`

### Energy Grid

**Policy:** `oracle_planner` / `energy_oracle_planner`

**Where:** `syncorsink/policies/oracle.py`

**Logic:**
- Keeps stable per-agent assignments.
- Reserves distinct resource targets.
- Routes carriers to matching nodes by urgency and distance.
- Coordinates two matching carriers before interacting on sync-gated critical nodes.

**Comm broadcast:**
The oracle planner is a centralized solvability policy and does not need to
broadcast. `energy_planner_comm` remains available as a lighter communication
baseline, but it is not the hard-preset acceptance expert.

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
