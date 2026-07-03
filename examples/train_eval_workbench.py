from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.train.workbench import TrainEvalWorkbenchConfig, run_train_eval_workbench


def main():
    parser = argparse.ArgumentParser(description="Train-save-load-eval workbench for SyncOrSink baselines")
    parser.add_argument("--algorithm", default="mappo", choices=["mappo"])
    parser.add_argument("--scenario", default="signal_hunt")
    parser.add_argument("--map-size", type=int, default=8)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--fov-preset", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--energy-preset", default="easy", choices=["easy", "hard"])
    parser.add_argument("--comm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--comm-token-limit", type=int, default=4)
    parser.add_argument("--comm-vocab-size", type=int, default=8)
    parser.add_argument("--comm-max-messages", type=int, default=4)
    parser.add_argument("--comm-cost", type=float, default=0.01)
    parser.add_argument("--comm-len-cost", type=float, default=0.0)
    parser.add_argument("--energy-private-monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pipeline-shaping", action="store_true")
    parser.add_argument("--pipeline-shaping-scale", type=float, default=0.01)
    parser.add_argument("--energy-shaping", action="store_true")
    parser.add_argument("--energy-shaping-scale", type=float, default=0.01)
    parser.add_argument("--signal-shaping", action="store_true")
    parser.add_argument("--signal-shaping-scale", type=float, default=0.01)
    parser.add_argument("--signal-scan-bonus", type=float, default=0.0)
    parser.add_argument("--signal-joint-scan-bonus", type=float, default=0.0)
    parser.add_argument("--signal-colocation-bonus", type=float, default=0.0)
    parser.add_argument("--signal-colocation-radius", type=int, default=2)
    parser.add_argument("--signal-comm-utility", type=float, default=0.0)
    parser.add_argument("--critic-mode", default="central", choices=["local", "central"])
    parser.add_argument("--shared-actor", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-eval-every", type=int, default=0)
    parser.add_argument("--train-eval-episodes", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--output-dir", default="logs/workbench")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink-workbench")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    result = run_train_eval_workbench(TrainEvalWorkbenchConfig(**vars(args)))
    print(json.dumps({
        "status": result["status"],
        "run_dir": result["run_dir"],
        "checkpoint_path": result["checkpoint_path"],
        "summary_path": result["summary_path"],
        "eval": result["eval"],
        "wandb": result["wandb"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
