#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== Launching Comm-MAT training ==="

nohup python examples/comm_mat_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --signal-shaping --signal-shaping-scale 0.01 \
  --updates 300 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --hidden-dim 128 --n-heads 4 --n-layers 2 \
  --eval-every 10 --eval-episodes 5 \
  --save checkpoints/comm_mat_signal_easy.pt --save-every 50 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-signal-easy \
  > logs/comm_mat.log 2>&1 &

echo "Comm-MAT PID: $!"
echo ""
echo "Monitor:"
echo "  tail -f logs/comm_mat.log"
