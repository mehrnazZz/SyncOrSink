#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== Comm-MAT training — all scenarios ==="

# Signal hunt (v4 shaping)
echo "Starting Comm-MAT signal_hunt..."
nohup python examples/comm_mat_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 0.2 --signal-joint-scan-bonus 3.0 \
  --signal-colocation-bonus 0.5 --signal-colocation-radius 2 \
  --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/comm_mat_signal.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-signal \
  > logs/comm_mat_signal.log 2>&1 &
echo "  PID: $!"

# Energy grid
echo "Starting Comm-MAT energy_grid..."
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/comm_mat_energy.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-energy \
  > logs/comm_mat_energy.log 2>&1 &
echo "  PID: $!"

# Pipeline assembly
echo "Starting Comm-MAT pipeline_assembly..."
nohup python examples/comm_mat_train.py \
  --scenario pipeline_assembly --map-size 8 --agents 3 --fov-preset easy \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --pipeline-shaping --pipeline-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/comm_mat_pipeline.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-pipeline \
  > logs/comm_mat_pipeline.log 2>&1 &
echo "  PID: $!"

echo ""
echo "Monitor:"
echo "  tail -f logs/comm_mat_signal.log"
echo "  tail -f logs/comm_mat_energy.log"
echo "  tail -f logs/comm_mat_pipeline.log"
