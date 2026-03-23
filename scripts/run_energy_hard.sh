#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs demos

echo "=== Energy Grid HARD preset — differentiating the 100% methods ==="
echo "Easy preset: every method solves it. Hard preset: who survives?"

# -----------------------------------------------------------------------
# Oracle baseline on hard (upper bound)
# -----------------------------------------------------------------------
echo ""
echo "=== Oracle on hard ==="
for policy in oracle oracle_strong; do
  echo "  $policy (8x8, 3 agents, easy FOV, HARD energy):"
  python examples/eval_run.py --scenario energy_grid --policy $policy \
    --map-size 8 --agents 3 --fov-preset easy \
    --episodes 20 --energy-preset hard 2>&1 | grep -E "success_rate|avg_steps"
done

# -----------------------------------------------------------------------
# Comm-MAT on hard
# -----------------------------------------------------------------------
echo ""
echo "=== Comm-MAT training on HARD ==="

nohup python examples/comm_mat_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/comm_mat_energy_hard.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run comm-mat-energy-hard \
  > logs/comm_mat_energy_hard.log 2>&1 &
echo "  Comm-MAT PID: $!"

# -----------------------------------------------------------------------
# TarMAC on hard
# -----------------------------------------------------------------------
echo ""
echo "=== TarMAC training on HARD ==="

nohup python examples/tarmac_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --hidden-dim 128 --msg-dim 32 --key-dim 32 --n-rounds 1 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/tarmac_energy_hard.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run tarmac-energy-hard \
  > logs/tarmac_energy_hard.log 2>&1 &
echo "  TarMAC PID: $!"

# -----------------------------------------------------------------------
# BC→RL on hard (collect demos on hard, then DAgger, then RL)
# -----------------------------------------------------------------------
echo ""
echo "=== BC→RL on HARD (collect → DAgger → RL) ==="

python examples/bc_train.py collect \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --energy-preset hard --episodes 100 --oracle oracle_strong \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --output demos/energy_grid_oracle_hard.npz

python examples/bc_train.py dagger \
  --demo-path demos/energy_grid_oracle_hard.npz \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --energy-preset hard --oracle oracle_strong \
  --rounds 3 --dagger-episodes 20 --epochs 30 \
  --batch-size 256 --lr 1e-3 --hidden-dim 128 \
  --comm-token-limit 8 --comm-vocab-size 32 \
  --save checkpoints/bc_dagger_energy_hard.pt

nohup python examples/mappo_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 2 --minibatch 256 \
  --anneal-lr --lr 3e-5 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_energy_hard.pt \
  --bc-kl-coeff 0.5 --bc-freeze-encoder \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_energy_hard.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-energy-hard \
  > logs/bc_rl_energy_hard.log 2>&1 &
echo "  BC→RL PID: $!"

echo ""
echo "Hard vs Easy energy preset:"
echo "  Energy: 16 vs 32 (half the buffer)"
echo "  Refill: 4 vs 8 (half the recovery)"
echo "  Grace: 4 vs 8 steps (half the startup time)"
echo "  Spawn: 0.15 vs 0.30 (half the resources)"
echo "  → Nodes die at step ~20 on hard vs ~40 on easy"
echo ""
echo "Monitor:"
echo "  tail -f logs/comm_mat_energy_hard.log"
echo "  tail -f logs/tarmac_energy_hard.log"
echo "  tail -f logs/bc_rl_energy_hard.log"
