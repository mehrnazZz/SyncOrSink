from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.eval.decoding_sweep import (
    MAPPODecodingSweepConfig,
    run_mappo_decoding_sweep,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep inference-time decoding settings for MAPPO checkpoints"
    )
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="MAPPO checkpoint path; repeat for multiple checkpoints")
    parser.add_argument("--checkpoint-label", action="append", default=None,
                        help="Label matching each --checkpoint; repeat in the same order")

    parser.add_argument("--scenario", default="signal_hunt",
                        choices=["signal_hunt", "energy_grid", "pipeline_assembly"])
    parser.add_argument("--map-size", type=int, default=8)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--fov-preset", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--energy-preset", default="easy", choices=["easy", "hard"])
    parser.add_argument("--energy-private-monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-exploration-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-exploration-age", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--comm-token-limit", type=int, default=8)
    parser.add_argument("--comm-vocab-size", type=int, default=32)
    parser.add_argument("--comm-max-messages", type=int, default=8)
    parser.add_argument("--comm-cost", type=float, default=0.001)
    parser.add_argument("--comm-len-cost", type=float, default=0.0)

    parser.add_argument("--pipeline-shaping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pipeline-shaping-scale", type=float, default=0.01)
    parser.add_argument("--energy-shaping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--energy-shaping-scale", type=float, default=0.01)
    parser.add_argument("--signal-shaping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--signal-shaping-scale", type=float, default=0.1)
    parser.add_argument("--signal-scan-bonus", type=float, default=0.2)
    parser.add_argument("--signal-joint-scan-bonus", type=float, default=3.0)
    parser.add_argument("--signal-colocation-bonus", type=float, default=0.5)
    parser.add_argument("--signal-colocation-radius", type=int, default=2)
    parser.add_argument("--signal-comm-utility", type=float, default=0.1)

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])

    parser.add_argument("--action-modes", nargs="+", default=["argmax", "sample"],
                        choices=["argmax", "sample"])
    parser.add_argument("--action-temperatures", nargs="+", type=float, default=[1.0])
    parser.add_argument("--send-modes", nargs="+", default=["threshold"],
                        choices=["threshold", "sample"])
    parser.add_argument("--send-thresholds", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--token-modes", nargs="+", default=["argmax", "sample"],
                        choices=["argmax", "sample"])
    parser.add_argument("--token-temperatures", nargs="+", type=float, default=[1.0])
    parser.add_argument("--length-modes", nargs="+", default=["argmax", "sample"],
                        choices=["argmax", "sample"])
    parser.add_argument("--length-temperatures", nargs="+", type=float, default=[1.0])

    parser.add_argument("--output-dir", default="logs/decoding_sweep")
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink-curriculum")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])

    args = parser.parse_args(argv)
    cfg = MAPPODecodingSweepConfig(
        checkpoints=args.checkpoint,
        checkpoint_labels=args.checkpoint_label,
        scenario=args.scenario,
        map_size=args.map_size,
        agents=args.agents,
        fov_preset=args.fov_preset,
        max_steps=args.max_steps,
        energy_preset=args.energy_preset,
        energy_private_monitor=args.energy_private_monitor,
        obs_exploration_memory=args.obs_exploration_memory,
        obs_exploration_age=args.obs_exploration_age,
        comm_token_limit=args.comm_token_limit,
        comm_vocab_size=args.comm_vocab_size,
        comm_max_messages=args.comm_max_messages,
        comm_cost=args.comm_cost,
        comm_len_cost=args.comm_len_cost,
        pipeline_shaping=args.pipeline_shaping,
        pipeline_shaping_scale=args.pipeline_shaping_scale,
        energy_shaping=args.energy_shaping,
        energy_shaping_scale=args.energy_shaping_scale,
        signal_shaping=args.signal_shaping,
        signal_shaping_scale=args.signal_shaping_scale,
        signal_scan_bonus=args.signal_scan_bonus,
        signal_joint_scan_bonus=args.signal_joint_scan_bonus,
        signal_colocation_bonus=args.signal_colocation_bonus,
        signal_colocation_radius=args.signal_colocation_radius,
        signal_comm_utility=args.signal_comm_utility,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        action_modes=tuple(args.action_modes),
        action_temperatures=tuple(args.action_temperatures),
        send_modes=tuple(args.send_modes),
        send_thresholds=tuple(args.send_thresholds),
        token_modes=tuple(args.token_modes),
        token_temperatures=tuple(args.token_temperatures),
        length_modes=tuple(args.length_modes),
        length_temperatures=tuple(args.length_temperatures),
        output_dir=args.output_dir,
        run_name=args.run_name,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        wandb_mode=args.wandb_mode,
    )
    result = run_mappo_decoding_sweep(cfg)
    best = result.get("best_row") or {}
    print(json.dumps({
        "status": result["status"],
        "run_dir": result["run_dir"],
        "summary_path": result["summary_path"],
        "csv_path": result["csv_path"],
        "checkpoint_count": result["checkpoint_count"],
        "combo_count": result["combo_count"],
        "best": {
            "checkpoint_label": best.get("checkpoint_label"),
            "rank": best.get("rank"),
            "summary": best.get("summary"),
            "action_mode": best.get("action_mode"),
            "send_mode": best.get("send_mode"),
            "send_threshold": best.get("send_threshold"),
            "token_mode": best.get("token_mode"),
            "length_mode": best.get("length_mode"),
        },
        "wandb": result.get("wandb"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
