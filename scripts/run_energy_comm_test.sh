#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== Energy Grid Communication Test ==="
echo "8x8, 3 agents, 4 nodes, hard preset, private monitoring"
echo "Oracle ceiling: 95%. Each agent monitors 1-2 nodes."
echo "Key test: does communication help when agents can't see all node energy?"

# Comm-MAT WITH communication + private monitoring
echo "Starting Comm-MAT with comm..."
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset hard --energy-private-monitor \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 50 --eval-episodes 20 \
  --save checkpoints/comm_mat_energy_comm_test.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-energy-commtest \
  > logs/comm_mat_energy_commtest.log 2>&1 &
echo "  PID: $!"

# Comm-MAT WITHOUT communication + private monitoring (ablation)
echo "Starting Comm-MAT no-comm..."
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset hard --energy-private-monitor \
  --comm-disabled \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 50 --eval-episodes 20 \
  --save checkpoints/comm_mat_nocomm_energy_comm_test.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-nocomm-energy-commtest \
  > logs/comm_mat_nocomm_energy_commtest.log 2>&1 &
echo "  PID: $!"

# TarMAC (attention-based communication)
echo "Starting TarMAC..."
nohup python examples/tarmac_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset hard --energy-private-monitor \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --hidden-dim 128 --msg-dim 32 --key-dim 32 --n-rounds 1 \
  --eval-every 50 --eval-episodes 20 \
  --save checkpoints/tarmac_energy_comm_test.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run tarmac-energy-commtest \
  > logs/tarmac_energy_commtest.log 2>&1 &
echo "  PID: $!"

echo ""
echo "Setup: 3 agents, 4 nodes, hard preset (death=18), private monitoring"
echo "Agent 0 monitors 2 nodes, agents 1-2 monitor 1 each"
echo "Oracle: 95%. If comm > no-comm, communication is proven necessary."
echo ""
echo "Monitor:"
echo "  tail -f logs/comm_mat_energy_commtest.log"
echo "  tail -f logs/comm_mat_nocomm_energy_commtest.log"
echo "  tail -f logs/tarmac_energy_commtest.log"
