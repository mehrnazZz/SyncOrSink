#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints logs demos

echo "=== Scale Experiments: 16x16 maps, medium FOV, more agents ==="
echo "Tests how methods degrade with harder settings"

# -----------------------------------------------------------------------
# Step 1: Oracle baselines at scale (quick, establishes upper bounds)
# -----------------------------------------------------------------------
echo ""
echo "=== Oracle baselines at 16x16 ==="
for scenario in signal_hunt energy_grid pipeline_assembly; do
  for policy in oracle oracle_strong; do
    echo "  $scenario / $policy (16x16, 4 agents, medium FOV):"
    python examples/eval_run.py --scenario $scenario --policy $policy \
      --map-size 16 --agents 4 --fov-preset medium \
      --episodes 10 --energy-preset easy 2>&1 | grep -E "success_rate|avg_steps"
  done
done

# -----------------------------------------------------------------------
# Step 2: LLM eval at scale (gpt-oss:20b local — no API cost)
# -----------------------------------------------------------------------
echo ""
echo "=== LLM at 16x16 (gpt-oss:20b) ==="

for scenario in signal_hunt energy_grid; do
  echo "Starting $scenario LLM eval (16x16)..."
  nohup python examples/eval_llm.py \
    --scenario $scenario --map-size 16 --agents 4 --fov-preset medium \
    --episodes 3 --max-steps 300 \
    --provider litellm --mode text --planner action \
    --model ollama_chat/gpt-oss:20b \
    --energy-preset easy \
    --trace-jsonl traces/${scenario}_16x16_gptoss20b.jsonl \
    > logs/llm_${scenario}_16x16.log 2>&1 &
  echo "  PID: $!"
done

# -----------------------------------------------------------------------
# Step 3: Comm-MAT at scale (needs GPU — run on RunPod)
# -----------------------------------------------------------------------
echo ""
echo "=== Comm-MAT training at 16x16 ==="

# Signal hunt 16x16
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
echo "  Comm-MAT signal 16x16 PID: $!"

# Energy grid 16x16
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
echo "  Comm-MAT energy 16x16 PID: $!"

# -----------------------------------------------------------------------
# Step 4: BC→RL at scale (collect demos at 16x16 first)
# -----------------------------------------------------------------------
echo ""
echo "=== BC→RL at 16x16 ==="

# Collect oracle demos at 16x16
for scenario in signal_hunt energy_grid; do
  agents=4
  episodes=100
  echo "Collecting oracle demos for $scenario (16x16)..."
  python examples/bc_train.py collect \
    --scenario $scenario --map-size 16 --agents $agents --fov-preset medium \
    --energy-preset easy --episodes $episodes --oracle oracle_strong \
    --comm-token-limit 8 --comm-vocab-size 32 \
    --output demos/${scenario}_oracle_16x16.npz
done

# DAgger at 16x16
for scenario in signal_hunt energy_grid; do
  agents=4
  echo "Running DAgger for $scenario (16x16)..."
  python examples/bc_train.py dagger \
    --demo-path demos/${scenario}_oracle_16x16.npz \
    --scenario $scenario --map-size 16 --agents $agents --fov-preset medium \
    --energy-preset easy --oracle oracle_strong \
    --rounds 3 --dagger-episodes 20 --epochs 30 \
    --batch-size 256 --lr 1e-3 --hidden-dim 128 \
    --comm-token-limit 8 --comm-vocab-size 32 \
    --save checkpoints/bc_dagger_${scenario}_16x16.pt
done

# BC→RL at 16x16
echo "Launching BC→RL at 16x16..."

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
  --bc-init checkpoints/bc_dagger_signal_hunt_16x16.pt \
  --bc-kl-coeff 0.5 --bc-freeze-encoder \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_signal_16x16.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-signal-16x16 \
  > logs/bc_rl_signal_16x16.log 2>&1 &
echo "  BC→RL signal 16x16 PID: $!"

nohup python examples/mappo_train.py \
  --scenario energy_grid --map-size 16 --agents 4 --fov-preset medium \
  --energy-shaping --energy-shaping-scale 0.1 \
  --updates 3000 --rollout-steps 512 --epochs 2 --minibatch 256 \
  --anneal-lr --lr 3e-5 \
  --critic-mode local \
  --bc-init checkpoints/bc_dagger_energy_grid_16x16.pt \
  --bc-kl-coeff 0.5 --bc-freeze-encoder \
  --eval-every 100 --eval-episodes 10 \
  --save checkpoints/mappo_bc_rl_energy_16x16.pt --save-every 500 \
  --wandb --wandb-project syncorsink --wandb-run mappo-bc-rl-energy-16x16 \
  > logs/bc_rl_energy_16x16.log 2>&1 &
echo "  BC→RL energy 16x16 PID: $!"

echo ""
echo "=== Scale experiment summary ==="
echo "  Map: 8x8 → 16x16 (4x area)"
echo "  Agents: 2-3 → 4"
echo "  FOV: easy → medium (smaller view)"
echo "  Colocation radius: 2 → 3 (scaled for larger map)"
echo ""
echo "Methods tested at scale:"
echo "  Oracle (instant), LLM (local), Comm-MAT (RL), BC→RL (IL→RL)"
echo ""
echo "Monitor:"
echo "  tail -f logs/comm_mat_signal_16x16.log"
echo "  tail -f logs/comm_mat_energy_16x16.log"
echo "  tail -f logs/bc_rl_signal_16x16.log"
echo "  tail -f logs/bc_rl_energy_16x16.log"
echo "  tail -f logs/llm_signal_hunt_16x16.log"
echo "  tail -f logs/llm_energy_grid_16x16.log"
