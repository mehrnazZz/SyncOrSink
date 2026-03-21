#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== Launching MAPPO training runs ==="

# Session 1: DTDE (local critic) — primary DTDE baseline
echo "Starting DTDE run (background)..."
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --signal-shaping --signal-shaping-scale 0.01 \
  --updates 300 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode local \
  --eval-every 10 --eval-episodes 5 \
  --save checkpoints/mappo_dtde_signal_easy.pt --save-every 50 \
  --wandb --wandb-project syncorsink --wandb-run mappo-dtde-signal-easy \
  > logs/dtde.log 2>&1 &
DTDE_PID=$!

# Session 2: CTDE (central critic) — upper bound
echo "Starting CTDE run (background)..."
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --signal-shaping --signal-shaping-scale 0.01 \
  --updates 300 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode central \
  --eval-every 10 --eval-episodes 5 \
  --save checkpoints/mappo_ctde_signal_easy.pt --save-every 50 \
  --wandb --wandb-project syncorsink --wandb-run mappo-ctde-signal-easy \
  > logs/ctde.log 2>&1 &
CTDE_PID=$!

echo ""
echo "Both runs launched!"
echo "  DTDE PID: $DTDE_PID"
echo "  CTDE PID: $CTDE_PID"
echo ""
echo "Monitor:"
echo "  tail -f logs/dtde.log  # follow DTDE logs"
echo "  tail -f logs/ctde.log  # follow CTDE logs"
echo ""
echo "Or check wandb dashboard at https://wandb.ai"
