#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs

echo "=== BC→RL Warmstart ==="
echo "Initialize MAPPO actor from DAgger BC checkpoint, fine-tune with RL"

# Pipeline assembly: BC→RL (the only way to crack 0%)
echo "Starting pipeline_assembly BC→RL..."
nohup python examples/mappo_train.py \
  --scenario pipeline_assembly --map-size 8 --agents 3 --fov-preset easy \
  --pipeline-shaping --pipeline-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 1e-4 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_pipeline_assembly.pt \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_pipeline.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-pipeline \
  > logs/bc_rl_pipeline.log 2>&1 &
echo "  PID: $!"

# Signal hunt: BC→RL (improve from 55% DAgger ceiling)
echo "Starting signal_hunt BC→RL..."
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
  --anneal-lr --lr 1e-4 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_signal_hunt.pt \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_signal.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-signal \
  > logs/bc_rl_signal.log 2>&1 &
echo "  PID: $!"

# Energy grid: BC→RL
echo "Starting energy_grid BC→RL..."
nohup python examples/mappo_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 1e-4 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_energy_grid.pt \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_energy.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-energy \
  > logs/bc_rl_energy.log 2>&1 &
echo "  PID: $!"

echo ""
echo "Key design choices:"
echo "  - Lower LR (1e-4 vs 3e-4) to avoid destroying BC initialization"
echo "  - Pipeline has no comm (oracle doesn't communicate)"
echo "  - Signal hunt uses v4 shaping + comm"
echo ""
echo "Monitor:"
echo "  tail -f logs/bc_rl_pipeline.log"
echo "  tail -f logs/bc_rl_signal.log"
echo "  tail -f logs/bc_rl_energy.log"
