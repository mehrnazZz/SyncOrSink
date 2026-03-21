#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== MAPPO v2 — tuned hyperparameters ==="

# DTDE, stronger shaping, more updates, lower comm cost
echo "Starting MAPPO DTDE v2..."
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode local \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_dtde_v2.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-dtde-v2 \
  > logs/dtde_v2.log 2>&1 &
echo "  PID: $!"

# CTDE, same tuning
echo "Starting MAPPO CTDE v2..."
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode central \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_ctde_v2.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-ctde-v2 \
  > logs/ctde_v2.log 2>&1 &
echo "  PID: $!"

# No-comm ablation (sanity: can agents solve it without comm?)
echo "Starting MAPPO no-comm ablation..."
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --signal-shaping --signal-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode local \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_nocomm.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-nocomm-ablation \
  > logs/nocomm.log 2>&1 &
echo "  PID: $!"

echo ""
echo "All v2 runs launched! Changes from v1:"
echo "  - 10x more updates (3000 vs 300)"
echo "  - 10x stronger shaping (0.1 vs 0.01)"
echo "  - 10x lower comm cost (0.001 vs 0.01)"
echo "  - no-comm ablation to isolate task learnability"
echo ""
echo "Monitor:"
echo "  tail -f logs/dtde_v2.log"
echo "  tail -f logs/ctde_v2.log"
echo "  tail -f logs/nocomm.log"
