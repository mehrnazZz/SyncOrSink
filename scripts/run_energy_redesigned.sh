#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs demos

echo "=== Energy Grid Redesigned — 16x16 with new sqrt scaling ==="
echo "Oracle ceiling: 65%. New scaling: sqrt-based energy, drain every step, 4 nodes."
echo ""

# -----------------------------------------------------------------------
# Comm-MAT with comm
# -----------------------------------------------------------------------
echo "Starting Comm-MAT 16x16 easy..."
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset easy \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/comm_mat_energy_new_easy.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-energy-new-easy \
  > logs/comm_mat_energy_new_easy.log 2>&1 &
echo "  Comm-MAT easy PID: $!"

echo "Starting Comm-MAT 16x16 hard..."
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset hard \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/comm_mat_energy_new_hard.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-energy-new-hard \
  > logs/comm_mat_energy_new_hard.log 2>&1 &
echo "  Comm-MAT hard PID: $!"

# -----------------------------------------------------------------------
# Comm-MAT NO-COMM ablation (does comm matter now?)
# -----------------------------------------------------------------------
echo "Starting Comm-MAT no-comm 16x16 easy..."
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset easy \
  --comm-disabled \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/comm_mat_nocomm_energy_new_easy.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-nocomm-energy-new-easy \
  > logs/comm_mat_nocomm_energy_new_easy.log 2>&1 &
echo "  Comm-MAT no-comm easy PID: $!"

echo "Starting Comm-MAT no-comm 16x16 hard..."
nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset hard \
  --comm-disabled \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/comm_mat_nocomm_energy_new_hard.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-nocomm-energy-new-hard \
  > logs/comm_mat_nocomm_energy_new_hard.log 2>&1 &
echo "  Comm-MAT no-comm hard PID: $!"

# -----------------------------------------------------------------------
# TarMAC
# -----------------------------------------------------------------------
echo "Starting TarMAC 16x16 easy..."
nohup python examples/tarmac_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset easy \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --hidden-dim 128 --msg-dim 32 --key-dim 32 --n-rounds 1 \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/tarmac_energy_new_easy.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run tarmac-energy-new-easy \
  > logs/tarmac_energy_new_easy.log 2>&1 &
echo "  TarMAC easy PID: $!"

# -----------------------------------------------------------------------
# BC→RL (easy preset — collect, DAgger, RL)
# -----------------------------------------------------------------------
echo ""
echo "Collecting oracle demos for 16x16 easy..."
python examples/bc_train.py collect \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-preset easy --episodes 100 --oracle oracle_strong \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --output demos/energy_grid_oracle_new_easy.npz

echo "Running DAgger for 16x16 easy..."
python examples/bc_train.py dagger \
  --demo-path demos/energy_grid_oracle_new_easy.npz \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-preset easy --oracle oracle_strong \
  --rounds 3 --dagger-episodes 20 --epochs 30 \
  --batch-size 256 --lr 1e-3 --hidden-dim 128 \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --save checkpoints/bc_dagger_energy_new_easy.pt

echo "Starting BC→RL 16x16 easy..."
nohup python examples/mappo_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-shaping --energy-shaping-scale 0.1 \
  --energy-preset easy \
  --updates 3000 --rollout-steps 512 --epochs 2 --minibatch 256 \
  --anneal-lr --lr 3e-5 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_energy_new_easy.pt \
  --bc-kl-coeff 0.5 --bc-freeze-encoder \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_energy_new_easy.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-energy-new-easy \
  > logs/bc_rl_energy_new_easy.log 2>&1 &
echo "  BC→RL easy PID: $!"

echo ""
echo "New energy scaling (16x16):"
echo "  Easy: energy=40, grace=12, drain=1/step, death=52, 4 nodes, 8 recharges"
echo "  Hard: energy=24, grace=8, drain=1/step, death=32, 4 nodes, 4 recharges"
echo "  Oracle ceiling: 65% (both easy and hard)"
echo ""
echo "Key experiment: Comm-MAT with comm vs no-comm"
echo "  If comm wins now → tighter budget makes coordination necessary"
echo ""
echo "Monitor:"
echo "  tail -f logs/comm_mat_energy_new_easy.log"
echo "  tail -f logs/comm_mat_energy_new_hard.log"
echo "  tail -f logs/comm_mat_nocomm_energy_new_easy.log"
echo "  tail -f logs/comm_mat_nocomm_energy_new_hard.log"
echo "  tail -f logs/tarmac_energy_new_easy.log"
echo "  tail -f logs/bc_rl_energy_new_easy.log"
