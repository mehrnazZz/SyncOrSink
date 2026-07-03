# Procedural Packs

Scenario packs are the bridge between fixed core scenarios and future
XLand-style procedural task generation. A pack groups scenario presets and
declares the axes that vary across those presets.

Pack metadata lives in `syncorsink/envs/procedural.py`.

## Current Packs

| Pack | Tier | Purpose |
|---|---|---|
| `core` | `core` | Stable compact diagnostic presets for the current core scenarios. |
| `core_ood` | `core_ood` | Scaled held-out presets for map, team-size, and partial-observability generalization. |

The current packs do not replace `benchmarks/syncorsink_v0_1.json`. The v0.1
manifest remains frozen. Packs are now used to generate
`benchmarks/syncorsink_v0_2.json`, the forward-compatible manifest line for
future advanced and procedural packs.

The v0.2 core packs use private node monitoring for `energy_grid`; the old
symmetric-information setting remains available only as an explicit ablation.

## Inspect Packs

```bash
python examples/list_scenario_packs.py
```

Full JSON metadata:

```bash
python examples/list_scenario_packs.py --json
```

Filter by tier:

```bash
python examples/list_scenario_packs.py --tier core_ood
```

Generate a benchmark manifest from packs:

```bash
python examples/list_scenario_packs.py \
  --benchmark core core_ood \
  --name syncorsink_v0_2 \
  --version 0.2.0 \
  --compatibility-note "Pack-generated successor to syncorsink_v0_1; covers the same core and scaled scenario surface with pack-derived case names so future packs can extend v0.2 without mutating v0.1."
```

## Pack Concepts

`ProceduralAxis`
: A named source of task variation, such as `map_size`, `agent_count`,
  `fov_preset`, information structure, hazard density, or role asymmetry.

`ProceduralPreset`
: A concrete scenario configuration that can become a benchmark case.

`ScenarioPack`
: A named group of presets plus the axes that define the pack.

## Why This Comes Before New Advanced Scenarios

Signal Hunt is intentionally a `core` cooperative-search diagnostic. It tests
private clue fusion and synchronized verification in a compact form. A future
`disaster_response` scenario should be an `advanced` pack that combines several
axes:

- victim count and urgency decay
- hazard density
- blocked routes
- role asymmetry
- supply scarcity
- communication range/noise
- map scale and room topology

Adding the pack layer first keeps advanced scenarios from becoming one-off
implementations. Every new scenario family should declare its axes, presets, and
benchmark role before it enters an official suite.
