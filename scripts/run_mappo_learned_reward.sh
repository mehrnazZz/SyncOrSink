#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== MAPPO with Learned Reward (IRL) ==="
echo "Uses reward model trained on oracle demonstrations instead of hand-crafted shaping"

# DTDE with learned reward (no hand-crafted shaping)
echo "Starting MAPPO DTDE + learned reward..."
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --comm-cost 0.001 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode local \
  --learned-reward checkpoints/reward_model_signal_hunt_matched.pt \
  --learned-reward-weight 1.0 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_dtde_learned_reward.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-dtde-learned-reward \
  > logs/dtde_learned_reward.log 2>&1 &
echo "  PID: $!"

# DTDE with learned reward + hand-crafted shaping (blend)
echo "Starting MAPPO DTDE + learned reward + shaping blend..."
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 0.2 \
  --signal-joint-scan-bonus 3.0 \
  --signal-colocation-bonus 0.5 \
  --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode local \
  --learned-reward checkpoints/reward_model_signal_hunt_matched.pt \
  --learned-reward-weight 0.5 \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_dtde_blend_reward.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-dtde-blend-reward \
  > logs/dtde_blend_reward.log 2>&1 &
echo "  PID: $!"

echo ""
echo "Monitor:"
echo "  tail -f logs/dtde_learned_reward.log"
echo "  tail -f logs/dtde_blend_reward.log"
