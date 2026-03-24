#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== Recurrent BC→RL for Pipeline Assembly ==="
echo "LSTM memory lets the policy track stage progress across steps"

nohup python examples/recurrent_train.py \
  --scenario pipeline_assembly --map-size 8 --agents 3 --fov-preset easy \
  --hidden-dim 128 \
  --demo-episodes 200 --bc-epochs 30 --bc-lr 1e-3 \
  --rl-updates 3000 --rl-lr 3e-5 --bc-kl-coeff 0.5 \
  --save checkpoints/recurrent_pipeline.pt \
  --wandb --wandb-project syncorsink --wandb-run recurrent-bc-rl-pipeline \
  > logs/recurrent_pipeline.log 2>&1 &
echo "  PID: $!"

echo ""
echo "Pipeline: oracle demos → recurrent BC (truncated BPTT) → PPO fine-tuning"
echo "Key difference: LSTM hidden state tracks stage progress across steps"
echo "Monitor: tail -f logs/recurrent_pipeline.log"
