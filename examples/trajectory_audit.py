from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkConfig
from syncorsink.eval.trajectory_audit import (
    AuditPolicySpec,
    MAPPODecodeConfig,
    make_bc_checkpoint_policy_factory,
    make_mappo_checkpoint_policy_factory,
    make_oracle_policy_factory,
    make_recurrent_checkpoint_policy_factory,
    recurrent_checkpoint_env_config,
    run_trajectory_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit policy trajectories to diagnose SyncOrSink failure modes"
    )
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

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--signal-trace", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--oracle", action="append", default=[],
                        choices=[
                            "oracle",
                            "oracle_strong",
                            "oracle_comm",
                            "oracle_strong_comm",
                            "signal_hint_comm",
                        ])
    parser.add_argument("--oracle-label", action="append", default=None)
    parser.add_argument("--bc-checkpoint", action="append", default=[])
    parser.add_argument("--bc-label", action="append", default=None)
    parser.add_argument("--mappo-checkpoint", action="append", default=[])
    parser.add_argument("--mappo-label", action="append", default=None)
    parser.add_argument("--recurrent-checkpoint", action="append", default=[])
    parser.add_argument("--recurrent-label", action="append", default=None)
    parser.add_argument("--recurrent-send-threshold", type=float, default=0.25)
    parser.add_argument("--recurrent-signal-scan-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--recurrent-signal-scan-gate-suppress",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--recurrent-signal-target-validity-threshold", type=float, default=None)
    parser.add_argument("--recurrent-signal-target-decision-threshold", type=float, default=None)
    parser.add_argument(
        "--recurrent-signal-target-decision-suppress",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-signal-scan-sync-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-signal-scan-sync-force-first",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-signal-scan-broadcast-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-signal-exact-target-message-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-signal-exact-target-navigation-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--recurrent-signal-exact-target-memory-steps", type=int, default=None)
    parser.add_argument(
        "--recurrent-signal-scan-refresh-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--recurrent-signal-scan-refresh-threshold", type=float, default=None)

    parser.add_argument("--mappo-deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mappo-action-mode", default="sample", choices=["argmax", "sample"])
    parser.add_argument("--mappo-action-temperature", type=float, default=1.0)
    parser.add_argument("--mappo-send-mode", default="threshold", choices=["threshold", "sample"])
    parser.add_argument("--mappo-send-threshold", type=float, default=0.25)
    parser.add_argument("--mappo-token-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--mappo-token-temperature", type=float, default=1.0)
    parser.add_argument("--mappo-length-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--mappo-length-temperature", type=float, default=1.0)

    parser.add_argument("--output-dir", default="logs/trajectory_audit")
    parser.add_argument("--run-name", default=None)

    args = parser.parse_args(argv)
    env_config = SyncOrSinkConfig(
        scenario=args.scenario,
        map_size=args.map_size,
        num_agents=args.agents,
        fov_preset=args.fov_preset,
        max_steps=args.max_steps,
        energy_preset=args.energy_preset,
        energy_private_monitor=args.energy_private_monitor,
        obs_exploration_memory=args.obs_exploration_memory,
        obs_exploration_age=args.obs_exploration_age,
        comm_token_limit=args.comm_token_limit,
        token_vocab_size=args.comm_vocab_size,
        max_messages=args.comm_max_messages,
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
    )
    policy_specs = _policy_specs(args, env_config)
    if not policy_specs:
        parser.error(
            "provide at least one --oracle, --bc-checkpoint, --mappo-checkpoint, "
            "or --recurrent-checkpoint"
        )
    result = run_trajectory_audit(
        env_config,
        policy_specs,
        episodes=args.episodes,
        seed=args.seed,
        include_signal_trace=args.signal_trace,
    )
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(json.dumps({
        "status": result["status"],
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "policies": [
            {
                "label": policy["label"],
                "summary": policy["summary"],
                "failure_type_counts": policy["diagnostics"]["failure_type_counts"],
                "signal": policy["diagnostics"].get("signal"),
                "signal_lifecycle": policy["diagnostics"].get("signal_lifecycle"),
            }
            for policy in result["policies"]
        ],
    }, indent=2, sort_keys=True))
    return 0


def _policy_specs(args, env_config: SyncOrSinkConfig) -> list[AuditPolicySpec]:
    specs: list[AuditPolicySpec] = []
    oracle_labels = _labels(args.oracle_label, args.oracle, "oracle")
    for label, oracle_type in zip(oracle_labels, args.oracle):
        specs.append(AuditPolicySpec(
            label=label,
            factory=make_oracle_policy_factory(args.scenario, oracle_type),
            metadata={"kind": "oracle", "oracle_type": oracle_type},
        ))

    bc_labels = _labels(args.bc_label, args.bc_checkpoint, "bc")
    for label, checkpoint in zip(bc_labels, args.bc_checkpoint):
        specs.append(AuditPolicySpec(
            label=label,
            factory=make_bc_checkpoint_policy_factory(checkpoint, deterministic=True, device=args.device),
            metadata={"kind": "bc", "checkpoint": checkpoint},
        ))

    decode = MAPPODecodeConfig(
        deterministic=args.mappo_deterministic,
        action_mode=args.mappo_action_mode,
        action_temperature=args.mappo_action_temperature,
        send_mode=args.mappo_send_mode,
        send_threshold=args.mappo_send_threshold,
        token_mode=args.mappo_token_mode,
        token_temperature=args.mappo_token_temperature,
        length_mode=args.mappo_length_mode,
        length_temperature=args.mappo_length_temperature,
    )
    mappo_labels = _labels(args.mappo_label, args.mappo_checkpoint, "mappo")
    for label, checkpoint in zip(mappo_labels, args.mappo_checkpoint):
        specs.append(AuditPolicySpec(
            label=label,
            factory=make_mappo_checkpoint_policy_factory(
                checkpoint,
                decode=decode,
                device=args.device,
                sample_seed=args.seed,
            ),
            metadata={
                "kind": "mappo",
                "checkpoint": checkpoint,
                "decode": asdict(decode),
            },
        ))

    recurrent_labels = _labels(args.recurrent_label, args.recurrent_checkpoint, "recurrent")
    for label, checkpoint in zip(recurrent_labels, args.recurrent_checkpoint):
        specs.append(AuditPolicySpec(
            label=label,
            factory=make_recurrent_checkpoint_policy_factory(
                checkpoint,
                device=args.device,
                eval_send_threshold=args.recurrent_send_threshold,
                eval_signal_scan_gate_threshold=args.recurrent_signal_scan_gate_threshold,
                eval_signal_scan_gate_suppress=args.recurrent_signal_scan_gate_suppress,
                eval_signal_target_validity_threshold=args.recurrent_signal_target_validity_threshold,
                eval_signal_target_decision_threshold=args.recurrent_signal_target_decision_threshold,
                eval_signal_target_decision_suppress=args.recurrent_signal_target_decision_suppress,
                eval_signal_scan_sync_assist=args.recurrent_signal_scan_sync_assist,
                eval_signal_scan_sync_force_first=args.recurrent_signal_scan_sync_force_first,
                eval_signal_scan_broadcast_assist=args.recurrent_signal_scan_broadcast_assist,
                eval_signal_exact_target_message_guard=args.recurrent_signal_exact_target_message_guard,
                eval_signal_exact_target_navigation_assist=(
                    args.recurrent_signal_exact_target_navigation_assist
                ),
                eval_signal_exact_target_memory_steps=args.recurrent_signal_exact_target_memory_steps,
                eval_signal_scan_refresh_assist=args.recurrent_signal_scan_refresh_assist,
                eval_signal_scan_refresh_threshold=args.recurrent_signal_scan_refresh_threshold,
            ),
            env_config=recurrent_checkpoint_env_config(checkpoint, env_config),
            metadata={
                "kind": "recurrent",
                "checkpoint": checkpoint,
                "eval_send_threshold": args.recurrent_send_threshold,
                "eval_signal_scan_gate_threshold": args.recurrent_signal_scan_gate_threshold,
                "eval_signal_scan_gate_suppress": args.recurrent_signal_scan_gate_suppress,
                "eval_signal_target_validity_threshold": args.recurrent_signal_target_validity_threshold,
                "eval_signal_target_decision_threshold": args.recurrent_signal_target_decision_threshold,
                "eval_signal_target_decision_suppress": args.recurrent_signal_target_decision_suppress,
                "eval_signal_scan_sync_assist": args.recurrent_signal_scan_sync_assist,
                "eval_signal_scan_sync_force_first": args.recurrent_signal_scan_sync_force_first,
                "eval_signal_scan_broadcast_assist": args.recurrent_signal_scan_broadcast_assist,
                "eval_signal_exact_target_message_guard": args.recurrent_signal_exact_target_message_guard,
                "eval_signal_exact_target_navigation_assist": (
                    args.recurrent_signal_exact_target_navigation_assist
                ),
                "eval_signal_exact_target_memory_steps": args.recurrent_signal_exact_target_memory_steps,
                "eval_signal_scan_refresh_assist": args.recurrent_signal_scan_refresh_assist,
                "eval_signal_scan_refresh_threshold": args.recurrent_signal_scan_refresh_threshold,
            },
        ))
    return specs


def _labels(labels: list[str] | None, values: list[str], prefix: str) -> list[str]:
    if labels is not None and len(labels) != len(values):
        raise ValueError(f"{prefix} labels must match {prefix} values")
    if labels is not None:
        return labels
    return [f"{prefix}_{idx}" for idx, _ in enumerate(values)]


def _run_dir(args) -> Path:
    if args.run_name:
        name = args.run_name
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{args.scenario}_{args.map_size}x{args.map_size}_audit_seed{args.seed}_{stamp}"
    return Path(args.output_dir) / name


if __name__ == "__main__":
    raise SystemExit(main())
