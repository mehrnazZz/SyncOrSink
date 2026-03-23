#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs demos

echo "=== Energy Grid 16x16 HARD — the real coordination test ==="
echo "Oracle ceiling: 85-90%. This should differentiate trained methods."

# -----------------------------------------------------------------------
# Comm-MAT
# -----------------------------------------------------------------------
echo "Starting Comm-MAT 16x16 hard..."
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/comm_mat_energy_16hard.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-energy-16hard \
  > logs/comm_mat_energy_16hard.log 2>&1 &
echo "  Comm-MAT PID: $!"

# -----------------------------------------------------------------------
# TarMAC
# -----------------------------------------------------------------------
echo "Starting TarMAC 16x16 hard..."
nohup python examples/tarmac_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --hidden-dim 128 --msg-dim 32 --key-dim 32 --n-rounds 1 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/tarmac_energy_16hard.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run tarmac-energy-16hard \
  > logs/tarmac_energy_16hard.log 2>&1 &
echo "  TarMAC PID: $!"

# -----------------------------------------------------------------------
# Comm-MAT no-comm ablation (does communication matter at this difficulty?)
# -----------------------------------------------------------------------
echo "Starting Comm-MAT no-comm 16x16 hard..."
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --comm-disabled \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/comm_mat_nocomm_energy_16hard.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-nocomm-energy-16hard \
  > logs/comm_mat_nocomm_energy_16hard.log 2>&1 &
echo "  Comm-MAT no-comm PID: $!"

# -----------------------------------------------------------------------
# BC→RL (collect on hard, DAgger, then RL)
# -----------------------------------------------------------------------
echo ""
echo "Collecting oracle demos for 16x16 hard..."
python examples/bc_train.py collect \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-preset hard --episodes 100 --oracle oracle_strong \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --output demos/energy_grid_oracle_16hard.npz

echo "Running DAgger for 16x16 hard..."
python examples/bc_train.py dagger \
  --demo-path demos/energy_grid_oracle_16hard.npz \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-preset hard --oracle oracle_strong \
  --rounds 3 --dagger-episodes 20 --epochs 30 \
  --batch-size 256 --lr 1e-3 --hidden-dim 128 \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --save checkpoints/bc_dagger_energy_16hard.pt

echo "Starting BC→RL 16x16 hard..."
nohup python examples/mappo_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 2 --minibatch 256 \
  --anneal-lr --lr 3e-5 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_energy_16hard.pt \
  --bc-kl-coeff 0.5 --bc-freeze-encoder \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_energy_16hard.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-energy-16hard \
  > logs/bc_rl_energy_16hard.log 2>&1 &
echo "  BC→RL PID: $!"

echo ""
echo "Oracle ceiling: 85-90% on this setting"
echo "Key question: does communication help when difficulty is real?"
echo "  Comm-MAT with comm vs Comm-MAT no-comm will answer this"
echo ""
echo "Monitor:"
echo "  tail -f logs/comm_mat_energy_16hard.log"
echo "  tail -f logs/tarmac_energy_16hard.log"
echo "  tail -f logs/comm_mat_nocomm_energy_16hard.log"
echo "  tail -f logs/bc_rl_energy_16hard.log"
