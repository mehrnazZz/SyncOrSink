# Benchmark Versions

SyncOrSink benchmark manifests are versioned. Published manifests should not be
mutated after results have been recorded against them.

## `syncorsink_v0_1`

- Manifest: `benchmarks/syncorsink_v0_1.json`
- Status: frozen official leaderboard contract
- Source: hand-authored fixed suite
- Cases: 6
- Coverage:
  - `signal_hunt` 8x8 and 16x16
  - `energy_grid` 8x8 and 16x16 as the legacy symmetric-information ablation
  - `pipeline_assembly` 8x8 and 16x16

Use v0.1 for the current public leaderboard under
`results/syncorsink_v0_1/`.

## `syncorsink_v0_2`

- Manifest: `benchmarks/syncorsink_v0_2.json`
- Status: pack-generated foundation
- Source packs: `core`, `core_ood`
- Cases: 6
- Compatibility: covers the same core and scaled scenario surface as v0.1, but
  uses pack-derived case names, richer pack metadata, and private node
  monitoring for `energy_grid`.

v0.2 is the forward-compatible manifest line for future scenario packs. New
advanced or procedural families should extend v0.2 or later, rather than editing
v0.1.

## Generate A Manifest From Packs

```bash
python examples/list_scenario_packs.py \
  --benchmark core core_ood \
  --name syncorsink_v0_2 \
  --version 0.2.0 \
  --compatibility-note "Pack-generated successor to syncorsink_v0_1; covers the same core and scaled scenario surface with pack-derived case names so future packs can extend v0.2 without mutating v0.1."
```

The generated manifest includes:

- `source_packs`
- `source_pack_tiers`
- `scenario_coverage`
- `axes_covered`
- `case_count`
- primary metric and official score definition

## Versioning Rule

Do not mutate a published official manifest to add scenarios or change case
definitions. Create a new manifest version instead.
