#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/core_training_sweep

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-syncorsink-core-training}"
RUN_NAME="${RUN_NAME:-recurrent_pipeline32_distill}"
WANDB_FLAG=""
if [ "$WANDB_MODE" != "disabled" ]; then
  WANDB_FLAG="--wandb"
fi

python examples/core_training_sweep.py \
  --algorithms recurrent_bc_rl \
  --benchmark-spec examples/pipeline_32_distill_spec.json \
  --benchmark-cases pipeline_32x32_easy_3agent_distill \
  --learning-profile bare \
  --seeds 0 \
  --updates 400 \
  --rollout-steps 512 \
  --eval-every 25 \
  --recurrent-ppo-profile pipeline32_distill \
  --recurrent-demo-episodes 256 \
  --recurrent-bc-epochs 10 \
  --recurrent-bc-lr 5e-4 \
  --recurrent-bc-seq-len 96 \
  --recurrent-dagger-rounds 2 \
  --recurrent-dagger-episodes 48 \
  --recurrent-dagger-oracle-action-rollin-rate 0.35 \
  --recurrent-dagger-oracle-message-rollin-rate 0.2 \
  --recurrent-rl-epochs 2 \
  --recurrent-minibatch-seqs 16 \
  --recurrent-train-map-sizes 32 \
  --recurrent-train-map-sampling-weights 32:1 \
  --recurrent-map-max-steps 32:480 \
  --recurrent-eval-map-sizes 32 \
  --recurrent-eval-seed-range 3000:24 \
  --recurrent-obs-exploration-memory \
  --recurrent-obs-memory-mode egocentric \
  --recurrent-obs-agent-id-features \
  --wandb-mode "$WANDB_MODE" \
  --wandb-project "$WANDB_PROJECT" \
  --output-dir logs/core_training_sweep \
  --run-name "$RUN_NAME" \
  $WANDB_FLAG \
  "$@"
