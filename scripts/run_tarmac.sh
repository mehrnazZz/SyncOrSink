#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== TarMAC Training — all scenarios ==="
echo "Targeted Multi-Agent Communication (attention-weighted message passing)"

# Signal hunt (v4 shaping)
echo "Starting TarMAC signal_hunt..."
nohup python examples/tarmac_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 0.2 --signal-joint-scan-bonus 3.0 \
  --signal-colocation-bonus 0.5 --signal-colocation-radius 2 \
  --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --hidden-dim 128 --msg-dim 32 --key-dim 32 --n-rounds 1 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/tarmac_signal.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run tarmac-signal \
  > logs/tarmac_signal.log 2>&1 &
echo "  PID: $!"

# Energy grid
echo "Starting TarMAC energy_grid..."
nohup python examples/tarmac_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --hidden-dim 128 --msg-dim 32 --key-dim 32 --n-rounds 1 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/tarmac_energy.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run tarmac-energy \
  > logs/tarmac_energy.log 2>&1 &
echo "  PID: $!"

echo ""
echo "TarMAC key difference from Comm-MAT:"
echo "  Comm-MAT: discrete token messages through env channel"
echo "  TarMAC: continuous message vectors via learned attention"
echo "  Both learn WHAT and WHO to communicate with"
echo ""
echo "Monitor:"
echo "  tail -f logs/tarmac_signal.log"
echo "  tail -f logs/tarmac_energy.log"
