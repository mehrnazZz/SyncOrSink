# SyncOrSink Leaderboard Protocol

SyncOrSink leaderboard submissions are evaluated against a versioned benchmark
suite. The current public leaderboard uses:

- Benchmark: `syncorsink_v0_1`
- Manifest: `benchmarks/syncorsink_v0_1.json`
- Result schema: `syncorsink.result.v0.1`
- Primary score: `100 * weighted_mean(success_rate)`

`benchmarks/syncorsink_v0_2.json` is available as a pack-generated successor
manifest, but the committed public leaderboard remains on v0.1 until v0.2
result artifacts are generated. See `docs/benchmark_versions.md`.

The primary metric is success rate because SyncOrSink scenarios have different
reward scales and horizon semantics. Return, steps, and communication tokens are
reported as secondary diagnostics.

## Tracks

Submissions must declare one track:

| Track | Intended use |
|---|---|
| `symbolic_dtde` | Decentralized execution from symbolic/grid observations only. |
| `symbolic_ctde` | Centralized training is allowed; execution remains decentralized. |
| `rgb_vision` | Policies act from rendered RGB observations. |
| `low_comm` | Communication is allowed but optimized for low token usage. |
| `no_comm` | Communication disabled or ignored. |
| `llm_text` | Language-model agents using text observations/messages. |
| `vlm_rgb` | Vision-language agents using rendered observations. |
| `sample_efficiency` | Ranking emphasizes learning with limited environment steps. |
| `ood_generalization` | Policies are trained on train/val and ranked on held-out OOD cases. |
| `human_playable` | Human or human-in-the-loop policies for demos and reference play. |

## Official v0.1 Cases

The v0.1 suite covers three communication regimes:

- `signal_hunt`: private clues, synchronized target scan, communication required.
- `energy_grid`: legacy symmetric-information ablation. The current default and
  v0.2 pack-generated case use private node monitoring where communication is
  required.
- `pipeline_assembly`: private blueprints and long-horizon dependency execution.

Each family has an 8x8 in-distribution case and a 16x16 scaled generalization
case. The test split is fixed by `syncorsink/eval/splits.py`.

## Creating A Result Artifact

The benchmark runner can emit a result artifact:

```bash
python examples/benchmark_run.py \
  --spec benchmarks/syncorsink_v0_1.json \
  --results-json results/my_method_v0_1.json \
  --track symbolic_dtde \
  --submission-name my-method-v0.1 \
  --method-name "My Comm Policy" \
  --method-type "Transformer MARL" \
  --authors "First Author,Second Author" \
  --repository https://github.com/example/my-policy \
  --checkpoint-uri https://huggingface.co/example/my-policy
```

External policies can be supplied without modifying SyncOrSink:

```bash
python examples/benchmark_run.py \
  --spec benchmarks/syncorsink_v0_1.json \
  --policy-entrypoint my_package.my_agent:build_policy \
  --policy-checkpoint /path/to/checkpoint.pt \
  --results-json results/my_method_v0_1.json \
  --track symbolic_dtde
```

See `docs/policy_submissions.md` for the policy interface.

The result JSON contains:

- `schema_version`
- `benchmark_name`
- `benchmark_version`
- `track`
- `generated_at`
- `submission`
- `cases`
- `score`

Each case includes the exact eval spec, weight, tags, seeds, and metrics:

```json
{
  "name": "signal_hunt_8x8_private_clues",
  "weight": 1.0,
  "tags": ["communication_required"],
  "spec": {"scenario": "signal_hunt", "split": "test"},
  "seeds": [],
  "metrics": {
    "episodes": 32,
    "success_rate": 0.75,
    "avg_return": 12.3,
    "avg_steps": 144.0,
    "avg_comm_tokens": 18.5
  }
}
```

## Checkpoints

Do not commit checkpoints to git. Submissions should include a checkpoint URI or
artifact reference in `submission.checkpoint_uri`. Recommended storage options:

- Hugging Face Hub
- W&B Artifacts
- GitHub Releases
- Institutional object storage

The repository should only store code, configs, docs, lightweight result JSON,
and small smoke-test fixtures.

## Validation And Scoring

Use the Python helpers for validation and scoring:

```python
from syncorsink.eval.result_schema import load_result_artifact
from syncorsink.eval.scoring import score_result_artifact

artifact = load_result_artifact("results/my_method_v0_1.json")
score = score_result_artifact(artifact)
print(score["official_score"])
```

Result artifacts that fail schema validation should not be accepted into the
public leaderboard.

## Building The Public Table

Committed result artifacts are collected from `results/syncorsink_v0_1/`.
Rebuild the Markdown table with:

```bash
python examples/build_leaderboard.py \
  --results results/syncorsink_v0_1 \
  --benchmark benchmarks/syncorsink_v0_1.json \
  --out-md docs/leaderboard_results.md \
  --out-csv docs/leaderboard_results.csv \
  --out-json docs/leaderboard_results.json
```

By default, the builder validates that every artifact matches the benchmark
name/version and includes exactly the official case set. Use `--allow-partial`
only for local smoke tests.

See `docs/leaderboard_results.md` for the current rendered table.

Validate that committed result artifacts and generated tables are in sync:

```bash
python examples/validate_leaderboard.py
```

To regenerate stale outputs locally:

```bash
python examples/validate_leaderboard.py --fix
```
