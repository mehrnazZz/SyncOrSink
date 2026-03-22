#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== Comm-MAT v3 — coordination shaping + comm utility ==="

nohup python examples/comm_mat_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 1.0 \
  --signal-colocation-bonus 0.5 \
  --signal-colocation-radius 2 \
  --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --hidden-dim 128 --n-heads 4 --n-layers 2 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/comm_mat_v3.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-v3 \
  > logs/comm_mat_v3.log 2>&1 &

echo "Comm-MAT v3 PID: $!"
echo "Monitor: tail -f logs/comm_mat_v3.log"
