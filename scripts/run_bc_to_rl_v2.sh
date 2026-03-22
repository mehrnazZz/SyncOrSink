#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs demos

echo "=== BC→RL v2 — KL regularization + frozen encoder + comm ==="
echo "Fixes from v1: lower LR, KL penalty to stay near BC policy, frozen encoder"

# Step 1: Collect oracle+comm demos for pipeline_assembly
echo "Step 1: Collecting oracle+comm demos for pipeline_assembly..."
python examples/bc_train.py collect \
  --scenario pipeline_assembly --map-size 8 --agents 3 --fov-preset easy \
  --episodes 200 --oracle oracle_strong_comm \
  --output demos/pipeline_assembly_oracle_comm.npz

# Step 2: DAgger with comm for pipeline_assembly
echo "Step 2: Running DAgger with comm..."
python examples/bc_train.py dagger \
  --demo-path demos/pipeline_assembly_oracle_comm.npz \
  --scenario pipeline_assembly --map-size 8 --agents 3 --fov-preset easy \
  --oracle oracle_strong_comm \
  --rounds 3 --dagger-episodes 30 --epochs 30 \
  --batch-size 256 --lr 1e-3 --hidden-dim 128 \
  --comm --comm-token-limit 8 --comm-vocab-size 32 --comm-loss-weight 0.1 \
  --save checkpoints/bc_dagger_comm_pipeline.pt

# Step 3: BC→RL with KL regularization
echo "Step 3: Launching BC→RL training runs..."

# Pipeline assembly: BC→RL v2 (KL + frozen encoder)
nohup python examples/mappo_train.py \
  --scenario pipeline_assembly --map-size 8 --agents 3 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --pipeline-shaping --pipeline-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 2 --minibatch 256 \
  --anneal-lr --lr 3e-5 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_comm_pipeline.pt \
  --bc-kl-coeff 0.5 \
  --bc-freeze-encoder \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_v2_pipeline.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-v2-pipeline \
  > logs/bc_rl_v2_pipeline.log 2>&1 &
echo "  Pipeline PID: $!"

# Signal hunt: BC→RL v2 (KL + frozen encoder + v4 shaping)
nohup python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 --comm-cost 0.001 \
  --signal-shaping --signal-shaping-scale 0.1 \
  --signal-scan-bonus 0.2 --signal-joint-scan-bonus 3.0 \
  --signal-colocation-bonus 0.5 --signal-comm-utility 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 2 --minibatch 256 \
  --anneal-lr --lr 3e-5 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_signal_hunt.pt \
  --bc-kl-coeff 0.5 \
  --bc-freeze-encoder \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_v2_signal.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-v2-signal \
  > logs/bc_rl_v2_signal.log 2>&1 &
echo "  Signal PID: $!"

# Energy grid: BC→RL v2 (KL + frozen encoder)
nohup python examples/mappo_train.py \
  --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 2 --minibatch 256 \
  --anneal-lr --lr 3e-5 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_energy_grid.pt \
  --bc-kl-coeff 0.5 \
  --bc-freeze-encoder \
  --eval-every 50 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_v2_energy.pt --save-every 200 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-v2-energy \
  > logs/bc_rl_v2_energy.log 2>&1 &
echo "  Energy PID: $!"

echo ""
echo "v2 fixes from v1:"
echo "  - LR: 1e-4 → 3e-5 (10x lower, gentler fine-tuning)"
echo "  - KL penalty: 0.5 (prevents diverging from BC initialization)"
echo "  - Frozen encoder: only heads are fine-tuned"
echo "  - PPO epochs: 4 → 2 (less aggressive per-update change)"
echo "  - Pipeline now has comm (oracle+comm demos via DAgger)"
echo ""
echo "Monitor:"
echo "  tail -f logs/bc_rl_v2_pipeline.log"
echo "  tail -f logs/bc_rl_v2_signal.log"
echo "  tail -f logs/bc_rl_v2_energy.log"
