#!/bin/bash
set -e

echo "=== SyncOrSink RunPod Setup ==="

# Clone repo
cd /workspace
if [ ! -d "SyncOrSink" ]; then
    git clone https://github.com/mehrnazZz/SyncOrSink.git
fi
cd SyncOrSink

# Install
pip install -e ".[train]" 2>&1 | tail -3
pip install wandb 2>&1 | tail -1

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# Quick smoke test
python -c "
from syncorsink.train.mappo import train_mappo, MAPPOConfig
cfg = MAPPOConfig(updates=2, rollout_steps=32, epochs=2, minibatch=32, eval_every=0, comm=True, comm_token_limit=8, comm_vocab_size=32, max_steps=50, device='auto')
train_mappo(cfg)
print('Smoke test passed!')
"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Run: wandb login"
echo "  2. Run: mkdir -p /workspace/SyncOrSink/checkpoints"
echo "  3. Run: bash scripts/run_training.sh"
