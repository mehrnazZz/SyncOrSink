#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs demos

echo "=== Scale Experiments (RunPod): Training at 16x16 ==="

# -----------------------------------------------------------------------
# Step 1: Collect oracle demos at 16x16
# -----------------------------------------------------------------------
echo "=== Collecting oracle demos at 16x16 ==="

python examples/bc_train.py collect \
  --scenario signal_hunt --map-size 16 --agents 4 --fov-preset medium \
  --energy-preset easy --episodes 100 --oracle oracle_strong \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --output demos/signal_hunt_oracle_16x16.npz

python examples/bc_train.py collect \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-preset easy --episodes 100 --oracle oracle_strong \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --output demos/energy_grid_oracle_16x16.npz

# -----------------------------------------------------------------------
# Step 2: DAgger at 16x16
# -----------------------------------------------------------------------
echo ""
echo "=== DAgger at 16x16 ==="

python examples/bc_train.py dagger \
  --demo-path demos/signal_hunt_oracle_16x16.npz \
  --scenario signal_hunt --map-size 16 --agents 4 --fov-preset medium \
  --energy-preset easy --oracle oracle_strong \
  --rounds 3 --dagger-episodes 20 --epochs 30 \
  --batch-size 256 --lr 1e-3 --hidden-dim 128 \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --save checkpoints/bc_dagger_signal_16x16.pt

python examples/bc_train.py dagger \
  --demo-path demos/energy_grid_oracle_16x16.npz \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-preset easy --oracle oracle_strong \
  --rounds 3 --dagger-episodes 20 --epochs 30 \
  --batch-size 256 --lr 1e-3 --hidden-dim 128 \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --save checkpoints/bc_dagger_energy_16x16.pt

# -----------------------------------------------------------------------
# Step 3: Comm-MAT training at 16x16
# -----------------------------------------------------------------------
echo ""
echo "=== Comm-MAT at 16x16 ==="

nohup python examples/comm_mat_train.py \
  --scenario signal_hunt --map-size 16 --agents 4 --fov-preset medium \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 0.2 --signal-joint-scan-bonus 3.0 \
  --signal-colocation-bonus 0.5 --signal-colocation-radius 3 \
  --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/comm_mat_signal_16x16.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-signal-16x16 \
  > logs/comm_mat_signal_16x16.log 2>&1 &
echo "  Comm-MAT signal PID: $!"

nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/comm_mat_energy_16x16.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-energy-16x16 \
  > logs/comm_mat_energy_16x16.log 2>&1 &
echo "  Comm-MAT energy PID: $!"

# -----------------------------------------------------------------------
# Step 4: BC→RL at 16x16
# -----------------------------------------------------------------------
echo ""
echo "=== BC→RL at 16x16 ==="

nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 16 --agents 4 --fov-preset medium \
  --comm --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 0.2 --signal-joint-scan-bonus 3.0 \
  --signal-colocation-bonus 0.5 --signal-colocation-radius 3 \
  --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 2 --minibatch 256 \
  --anneal-lr --lr 3e-5 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_signal_16x16.pt \
  --bc-kl-coeff 0.5 --bc-freeze-encoder \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_signal_16x16.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-signal-16x16 \
  > logs/bc_rl_signal_16x16.log 2>&1 &
echo "  BC→RL signal PID: $!"

nohup python examples/mappo_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 2 --minibatch 256 \
  --anneal-lr --lr 3e-5 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_energy_16x16.pt \
  --bc-kl-coeff 0.5 --bc-freeze-encoder \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_energy_16x16.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-energy-16x16 \
  > logs/bc_rl_energy_16x16.log 2>&1 &
echo "  BC→RL energy PID: $!"

echo ""
echo "Scale: 8x8 → 16x16 (4x area), 2-3 → 4 agents, easy → medium FOV"
echo ""
echo "Monitor:"
echo "  tail -f logs/comm_mat_signal_16x16.log"
echo "  tail -f logs/comm_mat_energy_16x16.log"
echo "  tail -f logs/bc_rl_signal_16x16.log"
echo "  tail -f logs/bc_rl_energy_16x16.log"
