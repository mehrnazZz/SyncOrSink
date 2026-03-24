#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== Energy Grid with Private Monitoring ==="
echo "Each agent only sees energy of assigned nodes"
echo "Must communicate urgency alerts to coordinate recharges"

# 16x16 hard with private monitoring — the real communication test
echo ""
echo "=== Comm-MAT with private monitoring ==="
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset hard --energy-private-monitor \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/comm_mat_energy_private.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-energy-private \
  > logs/comm_mat_energy_private.log 2>&1 &
echo "  Comm-MAT PID: $!"

echo ""
echo "=== Comm-MAT NO-COMM with private monitoring (ablation) ==="
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset hard --energy-private-monitor \
  --comm-disabled \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/comm_mat_nocomm_energy_private.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-nocomm-energy-private \
  > logs/comm_mat_nocomm_energy_private.log 2>&1 &
echo "  Comm-MAT no-comm PID: $!"

echo ""
echo "=== TarMAC with private monitoring ==="
nohup python examples/tarmac_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset hard --energy-private-monitor \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --hidden-dim 128 --msg-dim 32 --key-dim 32 --n-rounds 1 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/tarmac_energy_private.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run tarmac-energy-private \
  > logs/tarmac_energy_private.log 2>&1 &
echo "  TarMAC PID: $!"

echo ""
echo "Key hypothesis: with private monitoring, comm should finally matter"
echo "  Comm-MAT with comm > Comm-MAT no-comm (agents need to share energy alerts)"
echo "  Oracle ceiling on 16x16 hard: 85-90%"
echo ""
echo "Monitor:"
echo "  tail -f logs/comm_mat_energy_private.log"
echo "  tail -f logs/comm_mat_nocomm_energy_private.log"
echo "  tail -f logs/tarmac_energy_private.log"
