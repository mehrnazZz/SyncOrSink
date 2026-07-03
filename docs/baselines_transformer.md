# Transformer Baselines (Comm-MAT)

This document describes the transformer-based baseline currently implemented in SyncOrSink and how it is adapted for communication-focused DTDE benchmarking.

## Implemented baseline

- Name: `Comm-MAT` (communication-aware multi-agent transformer)
- Model implementation:
  - `syncorsink/models/comm_mat.py`
  - classes: `CommMATConfig`, `CommMATModel`
- Policy adapter:
  - `syncorsink/policies/comm_mat_policy.py`
  - classes: `CommMATPolicyConfig`, `CommMATPolicy`
- Training entrypoint:
  - `examples/comm_mat_train.py`
  - trainer: `syncorsink/train/comm_mat.py`

## Communication adaptation

Comm-MAT is adapted to SyncOrSink communication by explicitly modeling incoming messages and generating outgoing token messages.

- Inputs per agent:
  - local observation grid tokenized from `local_grid`,
  - self features (`inventory`, `self_pos`),
  - `goal_hint`,
  - incoming message tokens (`messages_tokens`),
  - incoming sender ids (`messages_from` or `message_from`).
- Tokenization and embeddings:
  - tile embedding for map/local semantics,
  - communication token embedding,
  - sender embedding for message provenance.
- Output heads:
  - `action_logits` over 8 actions,
  - `send_logit` for send/no-send,
  - `msg_len_logits` for message length,
  - `msg_token_logits` for token content,
  - `value` head for PPO-style training.

The policy returns exactly:

```python
{agent_id: {"action": int, "message_tokens": List[int]}}
```

which matches the SyncOrSink environment API for token communication.

## Training commands

Install (if needed):

```bash
pip install -e ".[train]"
```

Small sanity run:

```bash
python examples/comm_mat_train.py \
  --scenario pipeline_assembly \
  --map-size 8 \
  --agents 2 \
  --fov-preset easy \
  --updates 10 \
  --rollout-steps 128 \
  --epochs 2 \
  --minibatch 128
```

Longer run with checkpointing and eval:

```bash
python examples/comm_mat_train.py \
  --scenario pipeline_assembly \
  --map-size 8 \
  --agents 3 \
  --fov-preset easy \
  --updates 400 \
  --rollout-steps 256 \
  --epochs 4 \
  --minibatch 256 \
  --hidden-dim 128 \
  --n-heads 4 \
  --n-layers 2 \
  --eval-every 10 \
  --eval-episodes 5 \
  --save checkpoints/comm_mat_pipeline.pt
```

W&B logging:

```bash
python examples/comm_mat_train.py \
  --scenario signal_hunt \
  --wandb \
  --wandb-project syncorsink \
  --wandb-run comm-mat-signal
```

## Evaluation commands

Evaluate Comm-MAT through the standard eval harness:

```bash
python examples/eval_run.py \
  --scenario pipeline_assembly \
  --episodes 5 \
  --policy comm_mat \
  --comm-mat-ckpt checkpoints/comm_mat_pipeline.pt
```

Optional stochastic decoding:

```bash
python examples/eval_run.py \
  --scenario pipeline_assembly \
  --episodes 5 \
  --policy comm_mat \
  --comm-mat-ckpt checkpoints/comm_mat_pipeline.pt \
  --comm-mat-stochastic
```

Control message send behavior:

```bash
python examples/eval_run.py \
  --scenario pipeline_assembly \
  --episodes 5 \
  --policy comm_mat \
  --comm-mat-ckpt checkpoints/comm_mat_pipeline.pt \
  --comm-mat-send-threshold 0.6
```

## Benchmark presets

Transformer benchmark preset file:

- `benchmarks/transformer_presets.json`

Run:

```bash
python examples/benchmark_run.py --spec benchmarks/transformer_presets.json
```

The preset references local checkpoint files under `checkpoints/`. Those artifacts are not tracked in the repo; train Comm-MAT or restore the checkpoint files before running the preset end-to-end.
