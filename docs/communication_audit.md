# Communication Necessity Audit

This audit checks whether core scenarios expose private task information through
channels other than per-agent observations and explicit messages.

## Core Scenario Status

| Scenario | Private state channel | Shared-info audit | Status |
|---|---|---|---|
| `signal_hunt` | Per-agent `goal_hint` encodes private clue constraints; collected clues update only that agent's hint. | Shared `info` no longer returns `goal_hint_texts`, global `constraints`, or `agent_clues`. `clue_found` events do not include clue text. | Communication-required by observation contract. |
| `energy_grid` | Per-agent `goal_hint` encodes assigned nodes; `local_node_energy` shows only assigned-node energy when `energy_private_monitor=True`. | `node_critical` events are routed only to the assigned monitor. Global node energy and assignments are not in shared `info`. | Communication-required by default; symmetric ablation requires `energy_private_monitor=False`. |
| `pipeline_assembly` | Per-agent `goal_hint` encodes partial stage blueprints. | Shared `info` does not expose `hints`, `stages`, or `full_plan`. | Communication-required by observation contract. |

## Guardrails

- `tests/test_communication_audit.py` checks DTDE shared-info leakage for all
  core scenarios.
- `central_obs` appears only when `track="ctde"`.
- Private hints are available through per-agent observation fields, not shared
  `info`.

## Residual Protocol Risk

The Python policy callback receives the full `obs` dict for all agents. A
centralized hand-written policy can still read every agent's observation and
share memory internally. Official DTDE submissions should therefore use a
decentralized policy wrapper or plugin ABI that calls the same policy per agent
with only that agent's observation plus received messages. CTDE tracks may use
centralized training data, but execution-time action selection should respect
the selected track.
