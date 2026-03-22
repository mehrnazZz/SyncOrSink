#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs demos

echo "=== IRL: Train reward model → MAPPO with learned reward ==="

# Step 1: Train reward models with matching env config (comm_token_limit=8)
echo "Step 1: Training reward models..."

python examples/bc_train.py reward-model \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --episodes 100 --epochs 50 --batch-size 256 --hidden-dim 128 \
  --save checkpoints/reward_model_signal_hunt.pt

python examples/bc_train.py reward-model \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --energy-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --episodes 100 --epochs 50 --batch-size 256 --hidden-dim 128 \
  --save checkpoints/reward_model_energy_grid.pt

python examples/bc_train.py reward-model \
  --scenario pipeline_assembly --map-size 8 --agents 3 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --episodes 200 --epochs 50 --batch-size 256 --hidden-dim 128 \
  --save checkpoints/reward_model_pipeline.pt

# Step 2: MAPPO with learned reward (no hand-crafted shaping)
echo ""
echo "Step 2: Launching MAPPO with learned reward..."

# Signal hunt: learned reward only
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode local \
  --learned-reward checkpoints/reward_model_signal_hunt.pt \
  --learned-reward-weight 1.0 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_irl_signal.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-irl-signal \
  > logs/irl_signal.log 2>&1 &
echo "  Signal PID: $!"

# Energy grid: learned reward only
nohup python examples/mappo_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode local \
  --learned-reward checkpoints/reward_model_energy_grid.pt \
  --learned-reward-weight 1.0 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_irl_energy.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-irl-energy \
  > logs/irl_energy.log 2>&1 &
echo "  Energy PID: $!"

# Pipeline: learned reward only
nohup python examples/mappo_train.py \
  --scenario pipeline_assembly --map-size 8 --agents 3 --fov-preset easy \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode local \
  --learned-reward checkpoints/reward_model_pipeline.pt \
  --learned-reward-weight 1.0 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_irl_pipeline.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-irl-pipeline \
  > logs/irl_pipeline.log 2>&1 &
echo "  Pipeline PID: $!"

echo ""
echo "Reward models trained on oracle (obs, action) → reward tuples"
echo "MAPPO uses ONLY the learned reward — no hand-crafted shaping"
echo "This tests whether IRL can replace manual reward engineering"
echo ""
echo "Monitor:"
echo "  tail -f logs/irl_signal.log"
echo "  tail -f logs/irl_energy.log"
echo "  tail -f logs/irl_pipeline.log"
