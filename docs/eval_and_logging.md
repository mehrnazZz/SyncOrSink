# Evaluation And Logging Configuration

This document is the reference for evaluation/training CLI parameters and W&B logging features, including trace capture and environment video logging.

## Scripts covered

- Evaluation:
  - `examples/eval_run.py`
  - `examples/eval_llm.py`
  - `examples/benchmark_run.py`
  - `examples/eval_from_spec.py`
- Training:
  - `examples/mappo_train.py` (`syncorsink/train/mappo.py`)
  - `examples/comm_mat_train.py` (`syncorsink/train/comm_mat.py`)

## `eval_run.py` parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--scenario` | str | `signal_hunt` | Scenario name. |
| `--episodes` | int | `10` | Number of episodes. |
| `--split` | str? | `None` | Dataset split (`train/val/test`) if used. |
| `--variant` | int | `0` | Map variant index. |
| `--policy` | str | `random` | Policy selector (`random`, `heuristic`, scripted/oracle variants, `comm_mat`, etc.). |
| `--energy-preset` | str | `hard` | Energy Grid dynamics preset (`easy`, `hard`). |
| `--render` | bool | `False` | Enable live rendering. |
| `--render-fps` | float | `10.0` | Render speed. |
| `--trace-jsonl` | str? | `None` | Write per-step trace rows to JSONL. |
| `--trace-local-obs` | bool | `False` | Include local observations in trace rows. |
| `--trace-render-ansi` | bool | `False` | Include ANSI map snapshot in trace rows. |
| `--render-split-view` | bool | `False` | Split view render (agent+god). |
| `--render-god-view` | bool | `False` | God-view render mode. |
| `--render-style` | str | `arcade_flat` | Render style (`arcade_flat`, `sprite`). |
| `--record-video` | bool | `False` | Capture RGB frames for video logging. |
| `--video-episodes` | int | `1` | Number of episodes to record. |
| `--video-fps` | int | `8` | FPS metadata for video logs. |
| `--wandb` | bool | `False` | Enable W&B logging for episode + summary stats. |
| `--wandb-project` | str | `syncorsink` | W&B project name. |
| `--wandb-run` | str? | `None` | W&B run name. |
| `--wandb-log-trace-table` | bool | `False` | Log sampled per-step traces as W&B table. |
| `--wandb-trace-max-rows` | int | `2000` | Max rows in W&B trace table. |
| `--wandb-log-trace-artifact` | bool | `False` | Upload trace JSONL as W&B artifact. |
| `--wandb-log-video` | bool | `False` | Upload recorded videos to W&B. |
| `--comm-mat-ckpt` | str? | `None` | Optional Comm-MAT checkpoint for `--policy comm_mat`. |
| `--comm-mat-stochastic` | bool | `False` | Stochastic Comm-MAT decoding (default is deterministic). |
| `--comm-mat-send-threshold` | float | `0.5` | Comm-MAT send gate threshold. |

## `eval_llm.py` parameters

### Environment/eval parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--scenario` | str | `signal_hunt` | Scenario name. |
| `--map-size` | int | `8` | Map size. |
| `--agents` | int | `3` | Number of agents. |
| `--fov-preset` | str | `easy` | FOV preset (`easy`, `medium`, `hard`). |
| `--max-steps` | int | `300` | Max steps per episode. |
| `--episodes` | int | `5` | Number of episodes. |
| `--split` | str? | `None` | Split name. |
| `--variant` | int | `0` | Map variant index. |
| `--comm-cost` | float? | `None` | Override env comm cost. |
| `--comm-len-cost` | float? | `None` | Override env comm length cost. |

### LLM/provider parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--provider` | str | `dummy` | `dummy`, `openai-chat`, `openai-responses`. |
| `--mode` | str | `tools` | LLM interaction mode (`text`, `tools`). |
| `--planner` | str | `action` | Text planner style (`action`, `executor`). |
| `--model` | str | `gpt-4o-mini` | Provider model name. |
| `--api-key-env` | str | `OPENAI_API_KEY` | API key environment variable name. |
| `--cache` | str? | `None` | Prompt cache path. |

### Trace parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--trace-jsonl` | str? | `None` | Write per-step trace rows to JSONL. |
| `--trace-local-obs` | bool | `False` | Include raw local observations in trace rows. |
| `--trace-render-ansi` | bool | `False` | Include ANSI map snapshot in trace rows. |

### Render/video parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--render-split-view` | bool | `False` | Agent+god split view rendering mode. |
| `--render-god-view` | bool | `False` | God-view rendering mode. |
| `--render-style` | str | `arcade_flat` | Visual style (`arcade_flat`, `sprite`). |
| `--record-video` | bool | `False` | Capture RGB frames for episode video. |
| `--video-episodes` | int | `1` | Number of episodes to record. |
| `--video-fps` | int | `8` | FPS metadata for W&B video export. |

### W&B-specific logging parameters (`eval_llm.py`)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--wandb` | bool | `False` | Enable W&B run. |
| `--wandb-project` | str | `syncorsink` | W&B project name. |
| `--wandb-run` | str? | `None` | W&B run name. |
| `--wandb-log-trace-table` | bool | `False` | Log sampled per-step traces as W&B Table. |
| `--wandb-trace-max-rows` | int | `2000` | Max trace rows in W&B table. |
| `--wandb-log-trace-artifact` | bool | `False` | Upload `--trace-jsonl` as W&B artifact. |
| `--wandb-log-video` | bool | `False` | Upload recorded videos to W&B. |

## Trace schema (LLM eval)

Each JSONL row from `--trace-jsonl` includes:

- `episode`, `step`
- `actions`, `rewards`, `done`, `truncated`
- `comm_tokens`, `messages_text`, `messages_with_sender`
- `goal_hint_texts`
- `llm_calls` (prompt/response and parsed actions)
- `task_metrics`, `task_events`
- optional: `obs` (if `--trace-local-obs`)
- optional: `ansi_map` (if `--trace-render-ansi`)

This supports prompt/response analysis, communication timeline inspection, and post-hoc debugging.

## W&B outputs by script

### `eval_run.py`

- Summary metrics:
  - `success_rate`, `avg_return`, `avg_steps`, `avg_comm_tokens`
  - per-agent averages
- Episode metrics:
  - `ep_return`, `ep_steps`, `ep_success`, `ep_comm_tokens`
  - per-agent return/comm
- Optional:
  - per-step trace table (`trace/steps_table`)
  - trace artifact (`eval_trace`)
  - per-episode MP4 videos (`video/episode_*`)

### `eval_llm.py`

- All eval summary + per-episode metrics above
- Optional:
  - per-step trace table (`trace/steps_table`)
  - trace artifact (`llm_trace`)
  - per-episode MP4 videos (`video/episode_*`)

### Training scripts (`mappo` and `comm_mat`)

Both support:

- `--wandb`, `--wandb-project`, `--wandb-run`
- periodic training logs (`loss`, `policy_loss`, `value_loss`, `entropy`, rollout stats)
- periodic eval logs (`eval/mean_return`, `eval/mean_steps`, `eval/success_rate`)

## Benchmark/spec configuration

### `benchmark_run.py`

Flags:

- `--spec` (required): benchmark JSON file
- `--wandb`, `--wandb-project`, `--wandb-run`

For Comm-MAT in spec cases:

- `policy: "comm_mat"`
- optional `policy_checkpoint`
- optional `comm_mat_deterministic`
- optional `comm_mat_send_threshold`

MARL benchmark/spec runners fail fast on unknown policy names. Supported non-LLM policies include `random`, `scripted`, `oracle`, `oracle_strong`, `oracle_planner`, `oracle_comm`, the `pipeline_planner_*` communication planners, `energy_planner_comm`, `signal_hunt_planner_comm`, and `comm_mat`.

### `eval_from_spec.py`

Flag:

- `--spec` (required)

Same spec keys as above are supported for Comm-MAT selection. Specs may also set `map_size`, `agents` or `num_agents`, `fov_preset`, `max_steps`, `comm_mode`, `track`, `energy_preset`, and `energy_private_monitor`.

For `energy_grid`, `energy_private_monitor` defaults to `true`. Set it to
`false` only for the legacy symmetric-information ablation.

## Recommended command patterns

LLM eval with full trace + W&B table + artifact + video:

```bash
python examples/eval_llm.py \
  --scenario signal_hunt \
  --provider openai-chat \
  --mode text \
  --planner executor \
  --model gpt-4o-mini \
  --episodes 5 \
  --trace-jsonl /tmp/syncorsink_llm_trace.jsonl \
  --trace-local-obs \
  --record-video \
  --video-episodes 2 \
  --render-split-view \
  --wandb \
  --wandb-log-trace-table \
  --wandb-log-trace-artifact \
  --wandb-log-video
```

Comm-MAT benchmark preset run:

```bash
python examples/benchmark_run.py --spec benchmarks/transformer_presets.json --wandb
```

The transformer preset expects local checkpoint artifacts:

- `checkpoints/comm_mat_pipeline.pt`
- `checkpoints/comm_mat_energy.pt`
- `checkpoints/comm_mat_signal.pt`

These checkpoint files are not tracked in the repository. Train or restore them before running `benchmarks/transformer_presets.json`.

End-to-end checkpoint smoke test:

```bash
mkdir -p checkpoints

python examples/comm_mat_train.py \
  --scenario pipeline_assembly \
  --map-size 8 \
  --agents 3 \
  --fov-preset easy \
  --updates 1 \
  --rollout-steps 32 \
  --epochs 1 \
  --minibatch 32 \
  --device cpu \
  --eval-every 0 \
  --save checkpoints/comm_mat_pipeline.pt

python examples/comm_mat_train.py \
  --scenario energy_grid \
  --map-size 8 \
  --agents 3 \
  --fov-preset easy \
  --updates 1 \
  --rollout-steps 32 \
  --epochs 1 \
  --minibatch 32 \
  --device cpu \
  --eval-every 0 \
  --save checkpoints/comm_mat_energy.pt

python examples/comm_mat_train.py \
  --scenario signal_hunt \
  --map-size 8 \
  --agents 3 \
  --fov-preset easy \
  --updates 1 \
  --rollout-steps 32 \
  --epochs 1 \
  --minibatch 32 \
  --device cpu \
  --eval-every 0 \
  --save checkpoints/comm_mat_signal.pt

python examples/benchmark_run.py --spec benchmarks/transformer_presets.json
```

This verifies train-save-load-eval plumbing only. One-update checkpoints are not meaningful baselines.

Fresh-checkout smoke checks that do not require checkpoints:

```bash
pytest tests
python examples/benchmark_run.py --spec benchmarks/pipeline_presets.json
```

Locally verified on July 2, 2026:

```text
pytest tests
18 passed, 2 warnings

python examples/benchmark_run.py --spec benchmarks/pipeline_presets.json
case pipeline_easy_expert_comm success 1.0 return 37.6
case pipeline_hard_coord success 0.0 return -0.23399999999999999
case energy_easy_expert_comm success 0.6 return -1.6800000000000002
case signal_hunt_expert_comm success 1.0 return 29.910000000000004
```

## Practical note on diagrams/charts

W&B automatically builds line charts from logged scalar series (losses, returns, success rate, comm metrics).  
Trace tables and artifacts provide step-level data for building custom diagrams (message timelines, task transitions, prompt-response flow) outside or inside W&B dashboards.
