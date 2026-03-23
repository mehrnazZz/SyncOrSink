#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p logs traces

echo "=== Scale Experiments (Local): Oracle + LLM at 16x16 ==="

# -----------------------------------------------------------------------
# Oracle baselines (instant)
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
# LLM eval (gpt-oss:20b local)
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

echo ""
echo "Monitor:"
echo "  tail -f logs/llm_signal_hunt_16x16.log"
echo "  tail -f logs/llm_energy_grid_16x16.log"
