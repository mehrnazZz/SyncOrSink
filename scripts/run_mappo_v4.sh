#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== MAPPO v4 — joint scan bonus (fix shaping farming) ==="
echo "Key change: solo scan bonus reduced, large joint-scan near-miss bonus added"
echo "Co-location only fires when someone interacts (not just proximity)"

# DTDE v4
echo "Starting MAPPO DTDE v4..."
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 0.2 \
  --signal-joint-scan-bonus 3.0 \
  --signal-colocation-bonus 0.5 \
  --signal-colocation-radius 2 \
  --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode local \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_dtde_v4.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-dtde-v4 \
  > logs/dtde_v4.log 2>&1 &
echo "  PID: $!"

# CTDE v4
echo "Starting MAPPO CTDE v4..."
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 0.2 \
  --signal-joint-scan-bonus 3.0 \
  --signal-colocation-bonus 0.5 \
  --signal-colocation-radius 2 \
  --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode central \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_ctde_v4.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-ctde-v4 \
  > logs/ctde_v4.log 2>&1 &
echo "  PID: $!"

echo ""
echo "v4 reward structure:"
echo "  solo scan on target:      +0.2 (only if no joint bonus triggered)"
echo "  joint near-miss scan:     +3.0 (2+ agents scanned within window)"
echo "  co-location + interact:   +0.5 (2+ agents near target, one interacted)"
echo "  comm utility:             +0.1 (message preceded teammate's useful action)"
echo "  proximity shaping:        0.1/step (distance-based)"
echo "  comm cost:                0.001/token"
echo "  task success:             +10.0 (unchanged)"
echo ""
echo "Monitor:"
echo "  tail -f logs/dtde_v4.log"
echo "  tail -f logs/ctde_v4.log"
