# RL Baselines (MAPPO)

This document describes the RL-based baseline currently implemented in SyncOrSink and how it is adapted for communication-centric benchmarking.

## Implemented baseline

- Name: `MAPPO` (multi-agent PPO style)
- Training entrypoint: `examples/mappo_train.py`
- Trainer implementation: `syncorsink/train/mappo.py`
- Policy/model implementation:
  - `syncorsink/policies/mappo_policy.py`
  - `syncorsink/policies/mappo_models.py`

## Communication adaptation

The MAPPO actor supports a communication variant in addition to action selection.

- Standard action head:
  - `action_logits` over the 8 environment actions.
- Communication heads (enabled with `--comm`):
  - `send_logits`: Bernoulli gate for whether to send.
  - `token_logits`: token distribution for `comm_token_limit` positions.
  - `len_logits`: message length from `0..comm_token_limit`.
- PPO objective includes:
  - action log-prob term,
  - send-gate log-prob term,
  - message length/token log-prob terms.
- Environment-side communication settings are controlled by:
  - `--comm-token-limit`
  - `--comm-vocab-size`
  - `--comm-max-messages`
  - `--comm-len-cost`
  - `--comm-cost`

This lets MAPPO learn both task behavior and token communication behavior in the same policy.

## Training commands

Install (if needed):

```bash
pip install -e ".[train]"
```

End-to-end train/save/load/eval workbench:

```bash
python examples/train_eval_workbench.py \
  --scenario pipeline_assembly \
  --map-size 8 \
  --agents 3 \
  --fov-preset easy \
  --comm \
  --updates 20 \
  --rollout-steps 128 \
  --eval-episodes 5
```

This creates a local ignored run folder under `logs/workbench/` containing:

- `checkpoints/mappo.pt`
- `summary.json`

W&B workbench logging:

```bash
python examples/train_eval_workbench.py \
  --scenario pipeline_assembly \
  --comm \
  --wandb \
  --wandb-project syncorsink-workbench \
  --wandb-mode offline
```

Small sanity run:

```bash
python examples/mappo_train.py \
  --scenario pipeline_assembly \
  --map-size 8 \
  --agents 2 \
  --fov-preset easy \
  --comm \
  --updates 10 \
  --rollout-steps 128 \
  --epochs 2 \
  --minibatch 128
```

Longer run with checkpointing and eval:

```bash
python examples/mappo_train.py \
  --scenario pipeline_assembly \
  --map-size 8 \
  --agents 3 \
  --fov-preset easy \
  --comm \
  --updates 400 \
  --rollout-steps 256 \
  --epochs 4 \
  --minibatch 256 \
  --eval-every 10 \
  --eval-episodes 5 \
  --save checkpoints/mappo_pipeline.pt
```

W&B logging:

```bash
python examples/mappo_train.py \
  --scenario signal_hunt \
  --comm \
  --wandb \
  --wandb-project syncorsink \
  --wandb-run mappo-signal-comm
```

## Evaluation commands

Scripted eval harness:

```bash
python examples/eval_run.py --scenario pipeline_assembly --episodes 5 --policy random
```

LLM/eval traces are handled by separate scripts. For MAPPO checkpoints, use
`syncorsink.train.mappo.load_mappo_checkpoint_policy` or the train/eval
workbench above.

## Notes

- MAPPO here is currently the RL baseline with communication heads integrated into actor outputs.
- It can be used as CTDE-leaning baseline architecture while still executing decentralized actions/messages.
