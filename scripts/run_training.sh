#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p checkpoints

echo "=== Launching MAPPO training runs in tmux ==="

# Session 1: DTDE (local critic) — primary DTDE baseline
tmux new-session -d -s dtde "python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --signal-shaping --signal-shaping-scale 0.01 \
  --updates 300 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode local \
  --eval-every 10 --eval-episodes 5 \
  --save checkpoints/mappo_dtde_signal_easy.pt --save-every 50 \
  --wandb --wandb-project syncorsink --wandb-run mappo-dtde-signal-easy \
  2>&1 | tee logs/dtde.log; exec bash"

# Session 2: CTDE (central critic) — upper bound
tmux new-session -d -s ctde "python examples/mappo_train.py \
  --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy \
  --comm --comm-token-limit 8 --comm-vocab-size 32 \
  --signal-shaping --signal-shaping-scale 0.01 \
  --updates 300 --rollout-steps 512 --epochs 4 --minibatch 256 \
  --anneal-lr --lr 3e-4 \
  --critic-mode central \
  --eval-every 10 --eval-episodes 5 \
  --save checkpoints/mappo_ctde_signal_easy.pt --save-every 50 \
  --wandb --wandb-project syncorsink --wandb-run mappo-ctde-signal-easy \
  2>&1 | tee logs/ctde.log; exec bash"

echo "Both runs launched!"
echo ""
echo "Monitor:"
echo "  tmux attach -t dtde    # watch DTDE run"
echo "  tmux attach -t ctde    # watch CTDE run"
echo "  tail -f logs/dtde.log  # follow DTDE logs"
echo "  tail -f logs/ctde.log  # follow CTDE logs"
echo ""
echo "Or check wandb dashboard at https://wandb.ai"
