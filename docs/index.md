# SyncOrSink

SyncOrSink is a communication-focused cooperative multi-agent POMDP benchmark.
It is designed to test when communication is actually useful: private clues,
partial blueprints, synchronized actions, long-horizon planning, resource
allocation, and generalization across held-out maps.

The benchmark supports MARL agents, explicit token communication, LLM/text
agents, rendered RGB observations, PettingZoo-style integration, and leaderboard
submissions through lightweight result artifacts.

## Start Here

- [Current leaderboard](leaderboard_results.md)
- [Benchmark versions](benchmark_versions.md)
- [Leaderboard protocol](leaderboard.md)
- [External policy submissions](policy_submissions.md)
- [Scenario registry and tiers](scenario_registry.md)
- [Communication necessity audit](communication_audit.md)
- [Procedural packs](procedural_packs.md)
- [Scenario descriptions](scenarios.md)
- [Experiment report](experiment_report.md)

## Official v0.1 Benchmark

The first public suite is `syncorsink_v0_1`. It evaluates three scenario
families at 8x8 and 16x16 scales. These are `core` diagnostic scenarios:

| Scenario | Communication Structure | Main Challenge |
|---|---|---|
| `signal_hunt` | Complementary private clues | Information fusion and synchronized scan |
| `energy_grid` | Legacy symmetric-information ablation | Resource sharing and time pressure |
| `pipeline_assembly` | Private blueprints | Long-horizon dependency execution |

The primary leaderboard score is:

```text
100 * weighted_mean(success_rate)
```

Return, steps, and communication tokens are reported as secondary diagnostics.

`syncorsink_v0_2` is also available as a pack-generated successor manifest built
from `core` and `core_ood` packs. In v0.2 and in the environment defaults,
`energy_grid` uses private node monitoring so communication is necessary. See
[Benchmark Versions](benchmark_versions.md).

## Install

```bash
pip install -e .
```

Optional training dependencies:

```bash
pip install -e ".[train]"
```

Optional rendering dependencies:

```bash
pip install -e ".[render]"
```

## Run A Scenario

```bash
python examples/run_signal_hunt.py
python examples/run_energy.py
python examples/run_pipeline.py
```

Human-playable and pygame visual demos:

```bash
python examples/run_human.py
python examples/run_pygame.py
```

## Run The Official Suite

```bash
python examples/benchmark_run.py \
  --spec benchmarks/syncorsink_v0_1.json \
  --policy-override random \
  --results-json results/syncorsink_v0_1/symbolic_dtde/random.json \
  --track symbolic_dtde \
  --submission-name random-v0.1 \
  --method-name Random \
  --method-type random \
  --authors "SyncOrSink Contributors"
```

Rebuild the public leaderboard:

```bash
python examples/build_leaderboard.py \
  --results results/syncorsink_v0_1 \
  --benchmark benchmarks/syncorsink_v0_1.json \
  --out-md docs/leaderboard_results.md \
  --out-csv docs/leaderboard_results.csv \
  --out-json docs/leaderboard_results.json
```

Validate generated outputs before submitting a PR:

```bash
python examples/validate_leaderboard.py
```

## Submit A Policy

External policies can be evaluated without editing SyncOrSink:

```bash
python examples/benchmark_run.py \
  --spec benchmarks/syncorsink_v0_1.json \
  --policy-entrypoint my_package.my_agent:build_policy \
  --results-json results/syncorsink_v0_1/symbolic_dtde/my_agent.json \
  --track symbolic_dtde \
  --submission-name my-agent-v0.1 \
  --method-name "My Agent" \
  --method-type "Transformer MARL" \
  --authors "First Author,Second Author"
```

See [External Policy Submissions](policy_submissions.md) for the policy API.

## Checkpoint Policy

Do not commit checkpoints to git. Public submissions should provide checkpoint
URIs in their result artifacts and store large files in Hugging Face Hub, W&B
Artifacts, GitHub Releases, or institutional object storage.
