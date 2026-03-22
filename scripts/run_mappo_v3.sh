#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== MAPPO v3 — coordination shaping + comm utility ==="

# DTDE with coordination shaping (Part 1 + Part 2)
echo "Starting MAPPO DTDE v3..."
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 1.0 \
  --signal-colocation-bonus 0.5 \
  --signal-colocation-radius 2 \
  --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode local \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_dtde_v3.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-dtde-v3 \
  > logs/dtde_v3.log 2>&1 &
echo "  PID: $!"

# CTDE with coordination shaping (Part 1 + Part 2)
echo "Starting MAPPO CTDE v3..."
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 1.0 \
  --signal-colocation-bonus 0.5 \
  --signal-colocation-radius 2 \
  --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode central \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_ctde_v3.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-ctde-v3 \
  > logs/ctde_v3.log 2>&1 &
echo "  PID: $!"

echo ""
echo "v3 changes from v2:"
echo "  Part 1 — Coordination shaping:"
echo "    --signal-scan-bonus 1.0    (reward for interacting on true target)"
echo "    --signal-colocation-bonus 0.5 (reward when 2+ agents near target)"
echo "  Part 2 — Communication utility:"
echo "    --signal-comm-utility 0.1  (reward sender when teammate acts usefully after msg)"
echo ""
echo "Monitor:"
echo "  tail -f logs/dtde_v3.log"
echo "  tail -f logs/ctde_v3.log"
