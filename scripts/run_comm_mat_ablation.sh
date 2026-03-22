#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== Comm-MAT Ablation: Transformer WITHOUT communication ==="
echo "Same architecture but message inputs zeroed out and comm heads disabled"
echo "Tests whether communication or transformer backbone drives performance"

# Energy grid (where Comm-MAT got 100% with comm)
echo "Starting Comm-MAT no-comm on energy_grid..."
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --comm-disabled \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/comm_mat_nocomm_energy.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-nocomm-energy \
  > logs/comm_mat_nocomm_energy.log 2>&1 &
echo "  PID: $!"

# Signal hunt (where Comm-MAT got 30% with comm)
echo "Starting Comm-MAT no-comm on signal_hunt..."
nohup python examples/comm_mat_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 0.2 --signal-joint-scan-bonus 3.0 \
  --signal-colocation-bonus 0.5 --signal-colocation-radius 2 \
  --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --comm-disabled \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/comm_mat_nocomm_signal.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-nocomm-signal \
  > logs/comm_mat_nocomm_signal.log 2>&1 &
echo "  PID: $!"

echo ""
echo "Expected results:"
echo "  If comm is key: no-comm should fail (0%) while comm version succeeds"
echo "  If backbone is key: no-comm should also succeed"
echo ""
echo "Monitor:"
echo "  tail -f logs/comm_mat_nocomm_energy.log"
echo "  tail -f logs/comm_mat_nocomm_signal.log"
