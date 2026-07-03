# Scenario Registry And Tiers

SyncOrSink separates compact diagnostic scenarios from richer composite domains.
This keeps the default benchmark interpretable while leaving room for realistic
procedural task families.

Scenario metadata lives in `syncorsink/envs/scenario_registry.py`.
Procedural pack metadata lives in `syncorsink/envs/procedural.py`; see
`docs/procedural_packs.md`.

## Tiers

| Tier | Purpose | Expected stability |
|---|---|---|
| `core` | Small diagnostic scenarios that isolate a communication or coordination property. | Frozen once included in an official benchmark version. |
| `advanced` | Realistic composite domains that combine several core challenges. | Versioned, but may evolve between benchmark releases. |
| `procedural` | Scenario grammars that generate broad task distributions and OOD variants. | Versioned by grammar and seed policy. |
| `stress` | Large-scale, noisy, or adversarial variants for robustness testing. | Versioned by preset. |

## Current Core Scenarios

| Scenario | Domain | Communication role | Private information | Main diagnostic |
|---|---|---|---|---|
| `signal_hunt` | Cooperative search | Required | Complementary clues | Agents must fuse private clues and synchronize target verification. |
| `energy_grid` | Resource sharing | Required | Private node monitors | Agents must broadcast local urgency for assigned nodes and coordinate typed recharge. |
| `pipeline_assembly` | Task planning | Required | Complementary blueprints | Agents must share partial plans and execute long-horizon dependencies. |

`energy_grid` still supports the legacy symmetric-information ablation with
`energy_private_monitor=False` or `--no-energy-private-monitor`, but the default
core scenario uses private node monitoring so communication is necessary.

`signal_hunt` is disaster-adjacent because it uses search, clues, decoys, and
joint verification. It remains a `core` diagnostic scenario because it isolates
information fusion and synchronized confirmation. A future `disaster_response`
scenario should be an `advanced` composite domain that adds role asymmetry,
hazards, victim triage, supplies, route clearing, and decaying objectives.

## Inspecting The Registry

```bash
python examples/list_scenarios.py
```

Full JSON metadata:

```bash
python examples/list_scenarios.py --json
```

Filter by tier:

```bash
python examples/list_scenarios.py --tier core
```

## Adding A Scenario

When adding a new scenario:

1. Implement the scenario mechanics in `syncorsink/envs/scenarios.py` or a future scenario module.
2. Add the scenario to `SCENARIOS`.
3. Add metadata to `SCENARIO_REGISTRY`.
4. Add or update solvability checks.
5. Add scripted/oracle policies where appropriate.
6. Add tests for reset, step, success, metadata, and benchmark spec validation.
7. Add docs and benchmark manifests in a new version, for example `syncorsink_v0_2.json`.

Do not mutate an already-published official benchmark manifest to add a new
scenario. Add a new benchmark version instead.
