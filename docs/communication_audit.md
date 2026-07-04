# Communication Necessity Audit

This audit checks whether core scenarios expose private task information through
channels other than per-agent observations and explicit messages.

## Core Scenario Status

| Scenario | Private state channel | Shared-info audit | Status |
|---|---|---|---|
| `signal_hunt` | Per-agent `goal_hint` encodes private clue constraints; collected clues update only that agent's hint. | Shared `info` no longer returns `goal_hint_texts`, global `constraints`, or `agent_clues`. `clue_found` events do not include clue text. | Communication-required by observation contract. |
| `energy_grid` | Per-agent `goal_hint` encodes assigned nodes; `local_node_energy` shows only assigned-node energy when `energy_private_monitor=True`. | `node_critical` events are routed only to the assigned monitor. Global node energy and assignments are not in shared `info`. | Communication-required by default; the small core preset is sync-gated from the first recharge. Symmetric ablation requires `energy_private_monitor=False`. |
| `pipeline_assembly` | Per-agent `goal_hint` encodes partial stage blueprints. | Shared `info` does not expose `hints`, `stages`, or `full_plan`. | Communication-required by observation contract. |

## Guardrails

- `tests/test_communication_audit.py` checks DTDE shared-info leakage for all
  core scenarios.
- `examples/communication_ablation_sweep.py` runs a behavioral
  communication-vs-no-communication sweep with the current expert policies.
- `central_obs` appears only when `track="ctde"`.
- Private hints are available through per-agent observation fields, not shared
  `info`.

## Behavioral Sweep

The leakage audit proves that private information is not exposed through shared
`info`. The behavioral sweep checks the next question: whether an explicit
communication expert solves the same case better than a local no-message
baseline.

```bash
python examples/communication_ablation_sweep.py \
  --episodes 8 \
  --map-sizes 8 16 \
  --output-json logs/communication_ablation_sweep/latest.json
```

By default the sweep runs `signal_hunt`, `energy_grid`, and
`pipeline_assembly` at 8x8 and 16x16. It records per-condition success,
return, steps, communication tokens, and a gap row for each scenario/size.
`--fail-on-weak-gap` can turn the thresholds into a CI gate, but weak gaps
should be treated as design findings during scenario development. For example,
an expert that solves a case without messages indicates that the scenario may
still be too locally observable or too easy at that size.

## External Submission Guardrail

External policies loaded through `policy_entrypoint` are decentralized by
default. The adapter passes a read-only environment view to the policy factory
and calls `act_agent(agent_id, obs, info, state)` once per agent with only that
agent's observation plus that agent's received messages/events. `central_obs`
is stripped from external execution even when the case uses CTDE metadata.

Built-in oracle/debug policies can still use centralized state and should be
reported as oracle or CTDE diagnostic baselines, not as DTDE submissions.
