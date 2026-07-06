from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.train.curriculum import BCRLCurriculumConfig, run_bc_rl_curriculum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run expert demos -> BC/DAgger -> MAPPO BC-to-RL curriculum"
    )
    parser.add_argument("--scenario", default="signal_hunt",
                        choices=["signal_hunt", "energy_grid", "pipeline_assembly"])
    parser.add_argument("--map-size", type=int, default=8)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--fov-preset", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--energy-preset", default="easy", choices=["easy", "hard"])
    parser.add_argument("--energy-private-monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-exploration-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-exploration-age", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--comm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--comm-token-limit", type=int, default=8)
    parser.add_argument("--comm-vocab-size", type=int, default=32)
    parser.add_argument("--comm-max-messages", type=int, default=8)
    parser.add_argument("--comm-cost", type=float, default=0.001)
    parser.add_argument("--comm-len-cost", type=float, default=0.0)
    parser.add_argument("--comm-send-target", type=float, default=0.25)
    parser.add_argument("--comm-send-target-coeff", type=float, default=0.05)

    parser.add_argument("--demo-episodes", type=int, default=None)
    parser.add_argument("--oracle", default="oracle_strong_comm",
                        choices=[
                            "oracle",
                            "oracle_strong",
                            "oracle_comm",
                            "oracle_strong_comm",
                            "signal_hint_comm",
                        ])
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-episodes", type=int, default=None)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-comm-loss-weight", type=float, default=0.1)
    parser.add_argument("--bc-comm-send-pos-weight", type=float, default=-1.0,
                        help="Positive class weight for BC send BCE; negative auto-balances demos")
    parser.add_argument("--bc-two-phase", action="store_true")
    parser.add_argument("--bc-phase1-epochs", type=int, default=30)
    parser.add_argument("--bc-phase2-epochs", type=int, default=20)
    parser.add_argument("--bc-phase2-lr", type=float, default=3e-4)

    parser.add_argument("--rl-updates", type=int, default=3000)
    parser.add_argument("--rl-rollout-steps", type=int, default=512)
    parser.add_argument("--rl-epochs", type=int, default=2)
    parser.add_argument("--rl-minibatch", type=int, default=256)
    parser.add_argument("--rl-lr", type=float, default=3e-5)
    parser.add_argument("--rl-entropy", type=float, default=0.01)
    parser.add_argument("--rl-anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-kl-coeff", type=float, default=0.5)
    parser.add_argument("--bc-freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--critic-mode", default="local", choices=["local", "central"])
    parser.add_argument("--shared-actor", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--train-eval-every", type=int, default=50)
    parser.add_argument("--train-eval-episodes", type=int, default=10)

    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--eval-stochastic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-action-mode", default="sample", choices=["argmax", "sample"],
                        help="Decode mode for non-deterministic MAPPO evaluation")
    parser.add_argument("--eval-action-temperature", type=float, default=1.0)
    parser.add_argument("--eval-send-mode", default="threshold", choices=["threshold", "sample"],
                        help="Communication send-gate decode mode for non-deterministic MAPPO evaluation")
    parser.add_argument("--eval-send-threshold", type=float, default=0.25)
    parser.add_argument("--eval-token-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--eval-token-temperature", type=float, default=1.0)
    parser.add_argument("--eval-length-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--eval-length-temperature", type=float, default=1.0)

    parser.add_argument("--output-dir", default="logs/curriculum")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink-curriculum")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])

    args = parser.parse_args(argv)
    result = run_bc_rl_curriculum(BCRLCurriculumConfig(**vars(args)))
    print(json.dumps({
        "status": result["status"],
        "run_dir": result["run_dir"],
        "summary_path": result["summary_path"],
        "demo": result.get("demo"),
        "bc_diagnostics": result.get("bc_diagnostics"),
        "eval_bc": (result.get("eval_bc") or {}).get("summary"),
        "eval_rl_deterministic": (result.get("eval_rl_deterministic") or {}).get("summary"),
        "eval_rl_stochastic": (result.get("eval_rl_stochastic") or {}).get("summary"),
        "eval_rl_stochastic_decode": (result.get("eval_rl_stochastic") or {}).get("decode"),
        "best_eval_checkpoint": result.get("best_eval_checkpoint"),
        "eval_rl_best_deterministic": (result.get("eval_rl_best_deterministic") or {}).get("summary"),
        "eval_rl_best_stochastic": (result.get("eval_rl_best_stochastic") or {}).get("summary"),
        "eval_rl_best_stochastic_decode": (result.get("eval_rl_best_stochastic") or {}).get("decode"),
        "stages": result["stages"],
        "wandb_summary": result.get("wandb_summary"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
