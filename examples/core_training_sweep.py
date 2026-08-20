from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from json import JSONDecodeError
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ScenarioCase:
    scenario: str
    map_size: int
    agents: int
    fov_preset: str
    max_steps: int
    energy_preset: str | None = None
    benchmark_case: str | None = None
    energy_private_monitor: bool | None = None


@dataclass
class RunRecord:
    algorithm: str
    scenario: str
    seed: int
    run_dir: str
    checkpoint_path: str
    command: list[str]
    status: str
    returncode: int | None
    elapsed_sec: float
    checkpoint_exists: bool
    stdout_path: str
    stderr_path: str
    stdout_tail: list[str]
    stderr_tail: list[str]
    eval_metrics: dict[str, float] | None
    final_eval_metrics: dict[str, float] | None
    best_eval_metrics: dict[str, float] | None
    best_checkpoint_path: str | None
    best_checkpoint_exists: bool
    wandb: dict


DEFAULT_CASES: dict[str, ScenarioCase] = {
    "signal_hunt": ScenarioCase(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=60,
    ),
    "energy_grid": ScenarioCase(
        scenario="energy_grid",
        map_size=8,
        agents=3,
        fov_preset="easy",
        max_steps=80,
        energy_preset="easy",
        energy_private_monitor=True,
    ),
    "pipeline_assembly": ScenarioCase(
        scenario="pipeline_assembly",
        map_size=8,
        agents=3,
        fov_preset="easy",
        max_steps=80,
    ),
}

TRAIN_SCRIPTS = {
    "mappo": "examples/mappo_train.py",
    "comm_mat": "examples/comm_mat_train.py",
    "tarmac": "examples/tarmac_train.py",
    "recurrent_bc_rl": "examples/recurrent_train.py",
}

SCENARIO_SHAPING_ARGS = {
    "signal_hunt": [
        "--signal-shaping",
        "--signal-shaping-scale",
        "0.05",
        "--signal-scan-bonus",
        "0.05",
        "--signal-joint-scan-bonus",
        "1.0",
        "--signal-colocation-bonus",
        "0.25",
        "--signal-comm-utility",
        "0.05",
    ],
    "energy_grid": [
        "--energy-shaping",
        "--energy-shaping-scale",
        "0.05",
    ],
    "pipeline_assembly": [
        "--pipeline-shaping",
        "--pipeline-shaping-scale",
        "0.05",
    ],
}


RECURRENT_AUTO_ORACLES = {
    "signal_hunt": "signal_hint_comm",
    "energy_grid": "planner_comm",
    "pipeline_assembly": "planner_comm",
}

RECURRENT_DEFAULT_DAGGER_FOCUS_EVENTS = (
    "missed_delivery,missed_target_scan,solo_target_interact,target_interact_miss,"
    "target_pursuit_miss,target_decoy_drift_miss,target_discovery_miss,"
    "frontier_exploration_miss,target_handoff_miss,movement_stall_miss,"
    "pipeline_wrong_delivery,pipeline_dependency_blocked,"
    "pipeline_sync_wait,pipeline_pickup_miss,pipeline_delivery_miss,pipeline_station_stall_miss,"
    "pipeline_drop_miss,pipeline_bad_pickup,pipeline_wrong_delivery_root_pickup"
)

RECURRENT_SIGNAL_SPECIALIST_PRESETS = frozenset({"specialist", "large_map"})
RECURRENT_SIGNAL_LARGE_MAP_FOCUS_EVENTS = (
    "visible_clue_miss",
    "decoy_scan",
    "rejected_target_scan",
)
RECURRENT_SIGNAL_LARGE_MAP_REPLAY_EVENT_WEIGHTS = {
    "visible_clue_miss": 4.0,
    "decoy_scan": 4.0,
    "rejected_target_scan": 4.0,
}
RECURRENT_SIGNAL_LARGE_MAP_REPLAY_PRIORITY_EVENTS = (
    "visible_clue_miss",
    "decoy_scan",
    "rejected_target_scan",
)

RECURRENT_PPO_PROFILES = {
    "standard": {
        "recurrent_rl_lr": 3e-5,
        "recurrent_clip": 0.2,
        "recurrent_entropy_coeff": 0.01,
        "recurrent_max_grad_norm": 0.5,
        "recurrent_bc_kl_coeff": 0.5,
        "recurrent_bc_comm_kl_coeff": 0.5,
        "recurrent_bc_pipeline_pickup_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_delivery_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_delivery_progress_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_navigation_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_frontier_exploration_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_frontier_exploration_min_map_size": 8,
        "recurrent_bc_pipeline_sync_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_ready_interact_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_station_guard_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_wrong_station_recovery_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_pickup_gate_loss_weight": 0.0,
        "recurrent_bc_pipeline_pickup_gate_pos_weight": 1.0,
        "recurrent_bc_pipeline_pickup_gate_neg_weight": 1.0,
        "recurrent_bc_pipeline_plan_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_plan_head_loss_weight": 0.0,
        "recurrent_bc_pipeline_option_loss_weight": 0.0,
        "recurrent_bc_pipeline_message_loss_weight": 0.0,
        "recurrent_bc_pipeline_send_gate_loss_weight": 0.0,
        "recurrent_bc_pipeline_send_gate_pos_weight": 1.0,
        "recurrent_bc_pipeline_send_gate_neg_weight": 1.0,
        "recurrent_bc_pipeline_interact_gate_loss_weight": 0.0,
        "recurrent_bc_pipeline_interact_gate_pos_weight": 1.0,
        "recurrent_bc_pipeline_interact_gate_neg_weight": 1.0,
        "recurrent_bc_calibrate_pipeline_interact_gate_threshold": False,
        "recurrent_bc_pipeline_interact_gate_threshold_target_rate": -1.0,
        "recurrent_bc_pipeline_bad_pickup_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_bad_drop_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_bad_interact_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_bad_action_margin_loss_weight": 0.0,
        "recurrent_bc_pipeline_bad_action_margin": 1.0,
        "recurrent_bc_pipeline_proactive_bad_action_labels": False,
        "recurrent_dagger_focus_replay": False,
        "recurrent_dagger_retrain_from_scratch": True,
        "recurrent_dagger_pipeline_wrong_delivery_provenance_labels": False,
        "recurrent_dagger_pipeline_wrong_delivery_provenance_weight": -1.0,
        "recurrent_dagger_replay_pre_steps": 2,
        "recurrent_dagger_replay_post_steps": 2,
        "recurrent_dagger_replay_weight": 1.0,
        "recurrent_dagger_positive_replay_events": "",
        "recurrent_dagger_replay_event_weights": "",
        "recurrent_dagger_replay_event_caps": "",
        "recurrent_dagger_replay_success_only_events": "",
        "recurrent_dagger_replay_priority_events": "",
        "recurrent_dagger_replay_balance_positive_events": "",
        "recurrent_dagger_replay_balance_negative_events": "",
        "recurrent_dagger_replay_max_negative_per_positive": -1.0,
        "recurrent_dagger_max_replay_snippets_per_episode": 4,
        "recurrent_dagger_max_failed_parent_replay_snippets_per_episode": -1,
        "recurrent_dagger_failed_parent_replay_weight_scale": 1.0,
        "recurrent_dagger_expert_max_replay_snippets_per_episode": -1,
        "recurrent_pipeline_assisted_rollout_episodes": 0,
        "recurrent_pipeline_assisted_rollout_seed_base": 20000,
        "recurrent_pipeline_assisted_rollout_max_steps_per_episode": 0,
        "recurrent_pipeline_assisted_rollout_weight": 1.0,
        "recurrent_pipeline_assisted_rollout_success_only": False,
        "recurrent_pipeline_assisted_rollout_navigation_assist": True,
        "recurrent_pipeline_assisted_rollout_navigation_assist_trust_messages": True,
        "recurrent_pipeline_assisted_rollout_station_interact_guard": True,
        "recurrent_pipeline_assisted_rollout_bc_epochs": -1,
        "recurrent_rl_balanced_rollouts": False,
        "recurrent_rl_rollout_eval_decoding": False,
        "recurrent_rl_rollout_pipeline_navigation_assist": False,
        "recurrent_rl_rollout_pipeline_navigation_assist_trust_messages": False,
        "recurrent_rl_rollout_pipeline_station_interact_guard": False,
        "recurrent_rl_rollout_pipeline_interact_gate_promote": False,
        "recurrent_rl_eval_decoding_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_assisted_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_interact_gate_loss_weight": 0.0,
        "recurrent_rl_pipeline_interact_gate_pos_weight": 1.0,
        "recurrent_rl_pipeline_interact_gate_neg_weight": 1.0,
        "recurrent_rl_pipeline_pickup_gate_loss_weight": 0.0,
        "recurrent_rl_pipeline_pickup_gate_pos_weight": 1.0,
        "recurrent_rl_pipeline_pickup_gate_neg_weight": 1.0,
        "recurrent_rl_pipeline_delivery_progress_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_navigation_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_sync_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_ready_interact_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_station_guard_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_wrong_station_recovery_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_plan_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_plan_head_loss_weight": 0.0,
        "recurrent_rl_pipeline_option_loss_weight": 0.0,
        "recurrent_rl_pipeline_bad_pickup_penalty": 0.0,
        "recurrent_rl_pipeline_bad_interact_penalty": 0.0,
        "recurrent_rl_pipeline_unneeded_drop_bonus": 0.0,
        "recurrent_rl_early_stop_eval_patience": 0,
        "recurrent_eval_pipeline_interact_gate_threshold": -1.0,
        "recurrent_eval_pipeline_event_head_threshold": -1.0,
        "recurrent_eval_pipeline_navigation_head_threshold": -1.0,
    },
    "guarded": {
        "recurrent_rl_lr": 1e-5,
        "recurrent_clip": 0.1,
        "recurrent_entropy_coeff": 0.0,
        "recurrent_max_grad_norm": 0.25,
        "recurrent_bc_kl_coeff": 2.0,
        "recurrent_bc_comm_kl_coeff": 2.0,
        "recurrent_bc_pipeline_pickup_action_loss_weight": 1.0,
        "recurrent_bc_pipeline_delivery_action_loss_weight": 2.0,
        "recurrent_bc_pipeline_delivery_progress_action_loss_weight": 2.0,
        "recurrent_bc_pipeline_navigation_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_frontier_exploration_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_frontier_exploration_min_map_size": 8,
        "recurrent_bc_pipeline_sync_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_ready_interact_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_station_guard_action_loss_weight": 1.0,
        "recurrent_bc_pipeline_wrong_station_recovery_action_loss_weight": 0.0,
        "recurrent_bc_pipeline_pickup_gate_loss_weight": 1.0,
        "recurrent_bc_pipeline_pickup_gate_pos_weight": 2.0,
        "recurrent_bc_pipeline_pickup_gate_neg_weight": 1.5,
        "recurrent_bc_pipeline_plan_action_loss_weight": 1.0,
        "recurrent_bc_pipeline_plan_head_loss_weight": 1.0,
        "recurrent_bc_pipeline_option_loss_weight": 0.75,
        "recurrent_bc_pipeline_message_loss_weight": 1.0,
        "recurrent_bc_pipeline_send_gate_loss_weight": 1.0,
        "recurrent_bc_pipeline_send_gate_pos_weight": 3.0,
        "recurrent_bc_pipeline_send_gate_neg_weight": 1.0,
        "recurrent_bc_pipeline_interact_gate_loss_weight": 2.0,
        "recurrent_bc_pipeline_interact_gate_pos_weight": 3.0,
        "recurrent_bc_pipeline_interact_gate_neg_weight": 2.5,
        "recurrent_bc_calibrate_pipeline_interact_gate_threshold": True,
        "recurrent_bc_pipeline_interact_gate_threshold_target_rate": 0.33,
        "recurrent_bc_pipeline_bad_pickup_action_loss_weight": 0.5,
        "recurrent_bc_pipeline_bad_drop_action_loss_weight": 0.5,
        "recurrent_bc_pipeline_bad_interact_action_loss_weight": 1.0,
        "recurrent_bc_pipeline_bad_action_margin_loss_weight": 0.0,
        "recurrent_bc_pipeline_bad_action_margin": 1.0,
        "recurrent_bc_pipeline_proactive_bad_action_labels": True,
        "recurrent_dagger_focus_replay": True,
        "recurrent_dagger_retrain_from_scratch": True,
        "recurrent_dagger_pipeline_wrong_delivery_provenance_labels": True,
        "recurrent_dagger_pipeline_wrong_delivery_provenance_weight": 3.0,
        "recurrent_dagger_replay_pre_steps": 2,
        "recurrent_dagger_replay_post_steps": 2,
        "recurrent_dagger_replay_weight": 1.0,
        "recurrent_dagger_positive_replay_events": (
            "target_handoff,pipeline_delivery_ready,delivered,stage_completed"
        ),
        "recurrent_dagger_replay_event_weights": (
            "pipeline_delivery_ready:4.0,pipeline_delivery_miss:4.0,"
            "pipeline_station_stall_miss:3.0,"
            "pipeline_sync_wait:4.0,"
            "frontier_exploration_miss:4.0,target_discovery_miss:4.0,"
            "target_decoy_drift_miss:4.0,target_pursuit_miss:3.0,"
            "target_handoff_miss:4.0,target_handoff:3.0,"
            "pipeline_wrong_delivery:3.0,pipeline_wrong_delivery_root_pickup:3.0,"
            "delivered:2.0,stage_completed:2.0"
        ),
        "recurrent_dagger_replay_event_caps": "",
        "recurrent_dagger_replay_success_only_events": "delivered,stage_completed",
        "recurrent_dagger_replay_priority_events": (
            "frontier_exploration_miss,target_discovery_miss,target_decoy_drift_miss,"
            "target_handoff_miss,target_handoff,"
            "pipeline_delivery_miss,pipeline_delivery_ready,pipeline_wrong_delivery,"
            "pipeline_wrong_delivery_root_pickup,pipeline_sync_wait"
        ),
        "recurrent_dagger_replay_balance_positive_events": "",
        "recurrent_dagger_replay_balance_negative_events": "",
        "recurrent_dagger_replay_max_negative_per_positive": -1.0,
        "recurrent_dagger_max_replay_snippets_per_episode": 8,
        "recurrent_dagger_max_failed_parent_replay_snippets_per_episode": 4,
        "recurrent_dagger_failed_parent_replay_weight_scale": 1.0,
        "recurrent_dagger_expert_max_replay_snippets_per_episode": -1,
        "recurrent_pipeline_assisted_rollout_episodes": 0,
        "recurrent_pipeline_assisted_rollout_seed_base": 20000,
        "recurrent_pipeline_assisted_rollout_max_steps_per_episode": 0,
        "recurrent_pipeline_assisted_rollout_weight": 1.0,
        "recurrent_pipeline_assisted_rollout_success_only": False,
        "recurrent_pipeline_assisted_rollout_navigation_assist": True,
        "recurrent_pipeline_assisted_rollout_navigation_assist_trust_messages": True,
        "recurrent_pipeline_assisted_rollout_station_interact_guard": True,
        "recurrent_pipeline_assisted_rollout_bc_epochs": -1,
        "recurrent_rl_balanced_rollouts": True,
        "recurrent_rl_rollout_eval_decoding": True,
        "recurrent_rl_rollout_pipeline_navigation_assist": False,
        "recurrent_rl_rollout_pipeline_navigation_assist_trust_messages": False,
        "recurrent_rl_rollout_pipeline_station_interact_guard": True,
        "recurrent_rl_rollout_pipeline_interact_gate_promote": False,
        "recurrent_rl_eval_decoding_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_assisted_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_interact_gate_loss_weight": 0.0,
        "recurrent_rl_pipeline_interact_gate_pos_weight": 1.0,
        "recurrent_rl_pipeline_interact_gate_neg_weight": 1.0,
        "recurrent_rl_pipeline_pickup_gate_loss_weight": 0.0,
        "recurrent_rl_pipeline_pickup_gate_pos_weight": 1.0,
        "recurrent_rl_pipeline_pickup_gate_neg_weight": 1.0,
        "recurrent_rl_pipeline_delivery_progress_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_navigation_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_sync_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_ready_interact_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_station_guard_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_wrong_station_recovery_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_plan_action_loss_weight": 0.0,
        "recurrent_rl_pipeline_plan_head_loss_weight": 0.0,
        "recurrent_rl_pipeline_option_loss_weight": 0.0,
        "recurrent_rl_pipeline_bad_pickup_penalty": 0.1,
        "recurrent_rl_pipeline_bad_interact_penalty": 0.1,
        "recurrent_rl_pipeline_unneeded_drop_bonus": 0.05,
        "recurrent_rl_early_stop_eval_patience": 4,
        "recurrent_eval_pipeline_interact_gate_threshold": -1.0,
        "recurrent_eval_pipeline_event_head_threshold": -1.0,
        "recurrent_eval_pipeline_navigation_head_threshold": -1.0,
    },
}

RECURRENT_PPO_PROFILES["pipeline32_distill"] = {
    **RECURRENT_PPO_PROFILES["guarded"],
    "recurrent_bc_pipeline_navigation_action_loss_weight": 0.75,
    "recurrent_bc_pipeline_frontier_exploration_action_loss_weight": 0.5,
    "recurrent_bc_pipeline_frontier_exploration_min_map_size": 16,
    "recurrent_bc_pipeline_sync_action_loss_weight": 1.0,
    "recurrent_bc_pipeline_ready_interact_action_loss_weight": 1.0,
    "recurrent_bc_pipeline_wrong_station_recovery_action_loss_weight": 1.0,
    "recurrent_dagger_max_replay_snippets_per_episode": 12,
    "recurrent_dagger_max_failed_parent_replay_snippets_per_episode": 8,
    "recurrent_pipeline_assisted_rollout_episodes": 64,
    "recurrent_pipeline_assisted_rollout_seed_base": 40000,
    "recurrent_pipeline_assisted_rollout_weight": 2.0,
    "recurrent_pipeline_assisted_rollout_success_only": True,
    "recurrent_pipeline_assisted_rollout_bc_epochs": 4,
    "recurrent_rl_rollout_pipeline_navigation_assist": True,
    "recurrent_rl_rollout_pipeline_navigation_assist_trust_messages": True,
    "recurrent_rl_rollout_pipeline_interact_gate_promote": True,
    "recurrent_rl_eval_decoding_action_loss_weight": 0.1,
    "recurrent_rl_pipeline_assisted_action_loss_weight": 0.4,
    "recurrent_rl_pipeline_delivery_progress_action_loss_weight": 0.1,
    "recurrent_rl_pipeline_navigation_action_loss_weight": 0.1,
    "recurrent_rl_pipeline_sync_action_loss_weight": 0.1,
    "recurrent_rl_pipeline_ready_interact_action_loss_weight": 0.1,
    "recurrent_rl_pipeline_station_guard_action_loss_weight": 0.1,
    "recurrent_rl_pipeline_wrong_station_recovery_action_loss_weight": 0.1,
    "recurrent_rl_early_stop_eval_patience": 8,
}


def _expand_contiguous_seed_range(spec: str) -> str:
    match = re.fullmatch(r"(\d+):(\d+)", spec.strip())
    if match is None:
        raise ValueError(
            "seed range entries must use START:COUNT, for example 3000:40"
        )
    start = int(match.group(1))
    count = int(match.group(2))
    if count <= 0:
        raise ValueError("seed range COUNT must be positive")
    return ",".join(str(start + offset) for offset in range(count))


def _safe_run_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    safe = safe.strip("._-")
    return safe or "case"


def _benchmark_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return ROOT / path


def _case_from_benchmark_case(benchmark_case) -> ScenarioCase:
    spec = dict(benchmark_case.spec)
    required = ("scenario", "map_size", "agents", "fov_preset", "max_steps")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(
            f"benchmark case {benchmark_case.name!r} missing required fields: {missing}"
        )
    return ScenarioCase(
        scenario=str(spec["scenario"]),
        map_size=int(spec["map_size"]),
        agents=int(spec["agents"]),
        fov_preset=str(spec["fov_preset"]),
        max_steps=int(spec["max_steps"]),
        energy_preset=(
            str(spec["energy_preset"]) if spec.get("energy_preset") is not None else None
        ),
        benchmark_case=str(benchmark_case.name),
        energy_private_monitor=(
            bool(spec["energy_private_monitor"])
            if spec.get("energy_private_monitor") is not None
            else None
        ),
    )


def _cases_from_args(args) -> list[ScenarioCase]:
    if not args.benchmark_spec:
        return [DEFAULT_CASES[name] for name in args.scenarios]

    from syncorsink.eval.benchmark_spec import load_benchmark

    benchmark = load_benchmark(str(_benchmark_path(args.benchmark_spec)))
    by_name = {case.name: case for case in benchmark.cases}
    selected_names = args.benchmark_cases or list(by_name)
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise ValueError(
            f"unknown benchmark cases for {args.benchmark_spec}: {missing}; "
            f"available cases: {sorted(by_name)}"
        )
    return [_case_from_benchmark_case(by_name[name]) for name in selected_names]


def _expand_seed_range(spec: str) -> str:
    """Expand compact audit seed panels into the trainer's seed-list format."""
    spec = spec.strip()
    if not spec:
        return ""
    entries = []
    for raw_entry in re.split(r"[;+]", spec):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" in entry:
            map_size, range_spec = entry.split("=", 1)
            map_size = map_size.strip()
            if not map_size.isdigit() or int(map_size) <= 0:
                raise ValueError(
                    "map-specific seed ranges must use MAP_SIZE=START:COUNT, "
                    "for example 16=13000:40"
                )
            entries.append(f"{int(map_size)}:{_expand_contiguous_seed_range(range_spec)}")
        else:
            entries.append(_expand_contiguous_seed_range(entry))
    if not entries:
        raise ValueError("seed range cannot be empty")
    return "+".join(entries)


def build_command(
    *,
    algorithm: str,
    case: ScenarioCase,
    checkpoint_path: Path,
    args,
    run_name: str,
    seed: int,
) -> list[str]:
    if algorithm not in TRAIN_SCRIPTS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if algorithm == "recurrent_bc_rl":
        return _build_recurrent_command(
            case=case,
            checkpoint_path=checkpoint_path,
            args=args,
            run_name=run_name,
            seed=seed,
        )

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / TRAIN_SCRIPTS[algorithm]),
        "--scenario",
        case.scenario,
        "--map-size",
        str(case.map_size),
        "--agents",
        str(case.agents),
        "--fov-preset",
        case.fov_preset,
        "--max-steps",
        str(case.max_steps),
        "--updates",
        str(args.updates),
        "--rollout-steps",
        str(args.rollout_steps),
        "--epochs",
        str(args.epochs),
        "--minibatch",
        str(args.minibatch),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--eval-every",
        str(args.eval_every),
        "--eval-episodes",
        str(args.eval_episodes),
        "--save",
        str(checkpoint_path),
        "--save-every",
        str(max(1, args.updates)),
        "--wandb-project",
        args.wandb_project,
        "--wandb-run",
        run_name,
    ]
    if case.energy_preset is not None:
        cmd.extend(["--energy-preset", case.energy_preset])
    if case.energy_private_monitor is not None:
        if case.energy_private_monitor:
            cmd.append("--energy-private-monitor")
        else:
            cmd.append("--no-energy-private-monitor")
    if algorithm == "mappo":
        cmd.extend([
            "--comm",
            "--critic-mode",
            args.mappo_critic_mode,
            "--backbone",
            args.mappo_backbone,
            "--eval-action-mode",
            args.mappo_eval_action_mode,
            "--eval-action-temperature",
            str(args.mappo_eval_action_temperature),
            "--eval-send-mode",
            args.mappo_eval_send_mode,
            "--eval-send-threshold",
            str(args.mappo_eval_send_threshold),
            "--eval-token-mode",
            args.mappo_eval_token_mode,
            "--eval-token-temperature",
            str(args.mappo_eval_token_temperature),
            "--eval-length-mode",
            args.mappo_eval_length_mode,
            "--eval-length-temperature",
            str(args.mappo_eval_length_temperature),
        ])
        if args.mappo_shared_actor:
            cmd.append("--shared-actor")
        if args.mappo_obs_exploration_memory:
            cmd.append("--obs-exploration-memory")
        if args.mappo_obs_exploration_age:
            cmd.append("--obs-exploration-age")
    cmd.extend(_learning_profile_args(args.learning_profile, algorithm, case))
    if args.wandb:
        cmd.append("--wandb")
    return cmd


def _resolve_recurrent_signal_obs_memory_mode(args) -> str:
    if args.recurrent_obs_memory_mode == "auto":
        return "egocentric"
    return args.recurrent_obs_memory_mode


def _csv_items(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _append_csv_items(value: str | None, additions: Iterable[str]) -> str:
    items = _csv_items(value)
    seen = set(items)
    for item in additions:
        item = str(item).strip()
        if item and item not in seen:
            items.append(item)
            seen.add(item)
    return ",".join(items)


def _weighted_event_name(item: str) -> str:
    return item.split(":", 1)[0].strip()


def _append_weighted_csv_items(
    value: str | None,
    additions: dict[str, float],
) -> str:
    items = _csv_items(value)
    seen = {_weighted_event_name(item) for item in items}
    for event, weight in additions.items():
        event = str(event).strip()
        if event and event not in seen:
            items.append(f"{event}:{float(weight):.1f}")
            seen.add(event)
    return ",".join(items)


def _build_recurrent_command(
    *,
    case: ScenarioCase,
    checkpoint_path: Path,
    args,
    run_name: str,
    seed: int,
) -> list[str]:
    rl_updates = args.recurrent_rl_updates
    if rl_updates is None:
        rl_updates = args.updates
    eval_episodes = args.recurrent_eval_episodes
    if eval_episodes is None:
        eval_episodes = args.eval_episodes
    rl_eval_episodes = args.recurrent_rl_eval_episodes
    if rl_eval_episodes is None:
        rl_eval_episodes = eval_episodes
    signal_preset = (
        args.recurrent_signal_preset
        if case.scenario == "signal_hunt"
        else "minimal"
    )
    signal_specialist = signal_preset in RECURRENT_SIGNAL_SPECIALIST_PRESETS
    signal_large_map = signal_preset == "large_map"
    recurrent_obs_exploration_age = bool(args.recurrent_obs_exploration_age)
    initial_target_broadcast_labels = args.recurrent_dagger_initial_target_broadcast_labels
    if initial_target_broadcast_labels is None:
        initial_target_broadcast_labels = signal_specialist
    target_handoff_requires_exact_target = (
        args.recurrent_dagger_target_handoff_requires_exact_target
    )
    if target_handoff_requires_exact_target is None:
        target_handoff_requires_exact_target = False
    bc_comm_send_pos_weight = float(args.recurrent_bc_comm_send_pos_weight)
    if signal_specialist and bc_comm_send_pos_weight == 0.0:
        bc_comm_send_pos_weight = -1.0
    bc_signal_initial_message_weight = args.recurrent_bc_signal_initial_message_weight
    if bc_signal_initial_message_weight is None:
        bc_signal_initial_message_weight = 4.0 if signal_specialist else 1.0
    bc_signal_initial_message_loss_weight = (
        args.recurrent_bc_signal_initial_message_loss_weight
    )
    if bc_signal_initial_message_loss_weight is None:
        bc_signal_initial_message_loss_weight = 4.0 if signal_specialist else 0.0
    bc_signal_constraint_message_loss_weight = (
        args.recurrent_bc_signal_constraint_message_loss_weight
    )
    if bc_signal_constraint_message_loss_weight is None:
        bc_signal_constraint_message_loss_weight = 4.0 if signal_specialist else 0.0
    bc_signal_target_aux_weight = args.recurrent_bc_signal_target_aux_weight
    if bc_signal_target_aux_weight is None:
        bc_signal_target_aux_weight = 0.25 if signal_specialist else 0.0
    bc_signal_target_hypothesis_loss_weight = (
        args.recurrent_bc_signal_target_hypothesis_loss_weight
    )
    if bc_signal_target_hypothesis_loss_weight is None:
        bc_signal_target_hypothesis_loss_weight = 0.0
    bc_signal_target_hypothesis_commit_loss_weight = (
        args.recurrent_bc_signal_target_hypothesis_commit_loss_weight
    )
    if bc_signal_target_hypothesis_commit_loss_weight is None:
        bc_signal_target_hypothesis_commit_loss_weight = 1.0
    bc_signal_target_hypothesis_ambiguity_loss_weight = (
        args.recurrent_bc_signal_target_hypothesis_ambiguity_loss_weight
    )
    if bc_signal_target_hypothesis_ambiguity_loss_weight is None:
        bc_signal_target_hypothesis_ambiguity_loss_weight = 1.0
    bc_signal_target_hypothesis_xy_loss_weight = (
        args.recurrent_bc_signal_target_hypothesis_xy_loss_weight
    )
    if bc_signal_target_hypothesis_xy_loss_weight is None:
        bc_signal_target_hypothesis_xy_loss_weight = 1.0
    bc_signal_target_hypothesis_min_map_size = (
        args.recurrent_bc_signal_target_hypothesis_min_map_size
    )
    if bc_signal_target_hypothesis_min_map_size is None:
        bc_signal_target_hypothesis_min_map_size = 16
    bc_signal_target_pursuit_action_weight = (
        args.recurrent_bc_signal_target_pursuit_action_weight
    )
    if bc_signal_target_pursuit_action_weight is None:
        bc_signal_target_pursuit_action_weight = 0.4 if signal_specialist else 0.0
    bc_signal_target_pursuit_max_agents = (
        args.recurrent_bc_signal_target_pursuit_max_agents
    )
    if bc_signal_target_pursuit_max_agents is None:
        bc_signal_target_pursuit_max_agents = 0
    bc_signal_sync_response_action_loss_weight = (
        args.recurrent_bc_signal_sync_response_action_loss_weight
    )
    if bc_signal_sync_response_action_loss_weight is None:
        bc_signal_sync_response_action_loss_weight = 0.2 if signal_specialist else 0.0
    bc_signal_active_scan_response_action_weight = (
        args.recurrent_bc_signal_active_scan_response_action_weight
    )
    if bc_signal_active_scan_response_action_weight is None:
        bc_signal_active_scan_response_action_weight = 0.0
    bc_signal_active_scan_response_min_map_size = (
        args.recurrent_bc_signal_active_scan_response_min_map_size
    )
    if bc_signal_active_scan_response_min_map_size is None:
        bc_signal_active_scan_response_min_map_size = 16
    bc_signal_active_scan_response_max_agents = (
        args.recurrent_bc_signal_active_scan_response_max_agents
    )
    if bc_signal_active_scan_response_max_agents is None:
        bc_signal_active_scan_response_max_agents = 1
    bc_signal_scan_bridge_action_weight = (
        args.recurrent_bc_signal_scan_bridge_action_weight
    )
    if bc_signal_scan_bridge_action_weight is None:
        bc_signal_scan_bridge_action_weight = 0.0
    bc_signal_scan_bridge_min_map_size = (
        args.recurrent_bc_signal_scan_bridge_min_map_size
    )
    if bc_signal_scan_bridge_min_map_size is None:
        bc_signal_scan_bridge_min_map_size = 16
    bc_signal_scan_bridge_remaining_threshold = (
        args.recurrent_bc_signal_scan_bridge_remaining_threshold
    )
    if bc_signal_scan_bridge_remaining_threshold is None:
        bc_signal_scan_bridge_remaining_threshold = 0.5
    bc_signal_scan_bridge_max_teammate_distance = (
        args.recurrent_bc_signal_scan_bridge_max_teammate_distance
    )
    if bc_signal_scan_bridge_max_teammate_distance is None:
        bc_signal_scan_bridge_max_teammate_distance = 6
    bc_signal_target_match_action_weight = (
        args.recurrent_bc_signal_target_match_action_weight
    )
    if bc_signal_target_match_action_weight is None:
        bc_signal_target_match_action_weight = 0.4 if signal_specialist else 0.0
    bc_signal_first_target_scan_action_weight = (
        args.recurrent_bc_signal_first_target_scan_action_weight
    )
    if bc_signal_first_target_scan_action_weight is None:
        bc_signal_first_target_scan_action_weight = 0.8 if signal_specialist else 0.0
    bc_signal_refresh_target_scan_action_weight = (
        args.recurrent_bc_signal_refresh_target_scan_action_weight
    )
    if bc_signal_refresh_target_scan_action_weight is None:
        bc_signal_refresh_target_scan_action_weight = 0.3 if signal_specialist else 0.0
    bc_signal_joint_target_scan_action_weight = (
        args.recurrent_bc_signal_joint_target_scan_action_weight
    )
    if bc_signal_joint_target_scan_action_weight is None:
        bc_signal_joint_target_scan_action_weight = 0.5 if signal_specialist else 0.0
    bc_signal_target_opportunity_action_weight = (
        args.recurrent_bc_signal_target_opportunity_action_weight
    )
    if bc_signal_target_opportunity_action_weight is None:
        bc_signal_target_opportunity_action_weight = 0.4 if signal_specialist else 0.0
    bc_signal_redundant_target_wait_action_loss_weight = (
        args.recurrent_bc_signal_redundant_target_wait_action_loss_weight
    )
    if bc_signal_redundant_target_wait_action_loss_weight is None:
        bc_signal_redundant_target_wait_action_loss_weight = 0.0
    bc_signal_scan_decision_loss_weight = args.recurrent_bc_signal_scan_decision_loss_weight
    if bc_signal_scan_decision_loss_weight is None:
        bc_signal_scan_decision_loss_weight = 1.0 if signal_specialist else 0.0
    bc_signal_scan_decision_pos_weight = args.recurrent_bc_signal_scan_decision_pos_weight
    if bc_signal_scan_decision_pos_weight is None:
        bc_signal_scan_decision_pos_weight = 2.0 if signal_specialist else 1.0
    bc_signal_scan_decision_neg_weight = args.recurrent_bc_signal_scan_decision_neg_weight
    if bc_signal_scan_decision_neg_weight is None:
        bc_signal_scan_decision_neg_weight = 3.0 if signal_specialist else 1.0
    bc_signal_scan_gate_loss_weight = args.recurrent_bc_signal_scan_gate_loss_weight
    if bc_signal_scan_gate_loss_weight is None:
        bc_signal_scan_gate_loss_weight = 1.0 if signal_specialist else 0.0
    bc_signal_scan_gate_pos_weight = args.recurrent_bc_signal_scan_gate_pos_weight
    if bc_signal_scan_gate_pos_weight is None:
        bc_signal_scan_gate_pos_weight = 2.0 if signal_specialist else 1.0
    bc_signal_scan_gate_neg_weight = args.recurrent_bc_signal_scan_gate_neg_weight
    if bc_signal_scan_gate_neg_weight is None:
        bc_signal_scan_gate_neg_weight = 3.0 if signal_specialist else 1.0
    bc_signal_target_validity_loss_weight = (
        args.recurrent_bc_signal_target_validity_loss_weight
    )
    if bc_signal_target_validity_loss_weight is None:
        bc_signal_target_validity_loss_weight = 1.0 if signal_specialist else 0.0
    bc_signal_target_validity_pos_weight = (
        args.recurrent_bc_signal_target_validity_pos_weight
    )
    if bc_signal_target_validity_pos_weight is None:
        bc_signal_target_validity_pos_weight = 2.0 if signal_specialist else 1.0
    bc_signal_target_validity_neg_weight = (
        args.recurrent_bc_signal_target_validity_neg_weight
    )
    if bc_signal_target_validity_neg_weight is None:
        bc_signal_target_validity_neg_weight = 3.0 if signal_specialist else 1.0
    bc_signal_target_decision_loss_weight = (
        args.recurrent_bc_signal_target_decision_loss_weight
    )
    if bc_signal_target_decision_loss_weight is None:
        bc_signal_target_decision_loss_weight = 1.0 if signal_specialist else 0.0
    bc_signal_target_decision_pos_weight = (
        args.recurrent_bc_signal_target_decision_pos_weight
    )
    if bc_signal_target_decision_pos_weight is None:
        bc_signal_target_decision_pos_weight = 2.0 if signal_specialist else 1.0
    bc_signal_target_decision_neg_weight = (
        args.recurrent_bc_signal_target_decision_neg_weight
    )
    if bc_signal_target_decision_neg_weight is None:
        bc_signal_target_decision_neg_weight = 3.0 if signal_specialist else 1.0
    signal_scan_head_overrides = any(
        value is not None
        for value in (
            args.recurrent_bc_signal_scan_decision_loss_weight,
            args.recurrent_bc_signal_scan_decision_pos_weight,
            args.recurrent_bc_signal_scan_decision_neg_weight,
            args.recurrent_bc_signal_scan_gate_loss_weight,
            args.recurrent_bc_signal_scan_gate_pos_weight,
            args.recurrent_bc_signal_scan_gate_neg_weight,
            args.recurrent_bc_signal_target_validity_loss_weight,
            args.recurrent_bc_signal_target_validity_pos_weight,
            args.recurrent_bc_signal_target_validity_neg_weight,
            args.recurrent_bc_signal_target_decision_loss_weight,
            args.recurrent_bc_signal_target_decision_pos_weight,
            args.recurrent_bc_signal_target_decision_neg_weight,
        )
    )
    bc_signal_decoy_drift_action_loss_weight = (
        args.recurrent_bc_signal_decoy_drift_action_loss_weight
    )
    if bc_signal_decoy_drift_action_loss_weight is None:
        bc_signal_decoy_drift_action_loss_weight = 0.1 if signal_large_map else 0.0
    bc_signal_decoy_scan_action_loss_weight = (
        args.recurrent_bc_signal_decoy_scan_action_loss_weight
    )
    if bc_signal_decoy_scan_action_loss_weight is None:
        bc_signal_decoy_scan_action_loss_weight = 0.25 if signal_large_map else 0.0
    bc_signal_rejected_target_drift_action_loss_weight = (
        args.recurrent_bc_signal_rejected_target_drift_action_loss_weight
    )
    if bc_signal_rejected_target_drift_action_loss_weight is None:
        bc_signal_rejected_target_drift_action_loss_weight = 0.0
    bc_signal_clue_interact_action_weight = (
        args.recurrent_bc_signal_clue_interact_action_weight
    )
    if bc_signal_clue_interact_action_weight is None:
        bc_signal_clue_interact_action_weight = 0.0
    bc_signal_clue_interact_min_map_size = (
        args.recurrent_bc_signal_clue_interact_min_map_size
    )
    if bc_signal_clue_interact_min_map_size is None:
        bc_signal_clue_interact_min_map_size = 16
    bc_signal_visible_clue_action_weight = (
        args.recurrent_bc_signal_visible_clue_action_weight
    )
    if bc_signal_visible_clue_action_weight is None:
        bc_signal_visible_clue_action_weight = 0.25 if signal_large_map else 0.0
    bc_signal_visible_clue_min_map_size = (
        args.recurrent_bc_signal_visible_clue_min_map_size
    )
    if bc_signal_visible_clue_min_map_size is None:
        bc_signal_visible_clue_min_map_size = 32 if signal_large_map else 16
    bc_signal_evidence_sweep_action_weight = (
        args.recurrent_bc_signal_evidence_sweep_action_weight
    )
    if bc_signal_evidence_sweep_action_weight is None:
        bc_signal_evidence_sweep_action_weight = 0.0
    bc_signal_evidence_sweep_min_map_size = (
        args.recurrent_bc_signal_evidence_sweep_min_map_size
    )
    if bc_signal_evidence_sweep_min_map_size is None:
        bc_signal_evidence_sweep_min_map_size = 16
    bc_signal_frontier_exploration_action_weight = (
        args.recurrent_bc_signal_frontier_exploration_action_weight
    )
    if bc_signal_frontier_exploration_action_weight is None:
        if signal_large_map:
            bc_signal_frontier_exploration_action_weight = 0.25
        else:
            bc_signal_frontier_exploration_action_weight = 0.25 if signal_specialist else 0.0
    bc_signal_frontier_exploration_min_map_size = (
        args.recurrent_bc_signal_frontier_exploration_min_map_size
    )
    if bc_signal_frontier_exploration_min_map_size is None:
        bc_signal_frontier_exploration_min_map_size = 16
    bc_comm_loss_weight = args.recurrent_bc_comm_loss_weight
    if bc_comm_loss_weight is None:
        bc_comm_loss_weight = 1.0 if signal_specialist else 0.1
    initial_exact_message_copy_assist = (
        args.recurrent_eval_signal_initial_exact_message_copy_assist
    )
    if initial_exact_message_copy_assist is None:
        initial_exact_message_copy_assist = signal_specialist
    exact_target_message_copy_assist = (
        args.recurrent_eval_signal_exact_target_message_copy_assist
    )
    if exact_target_message_copy_assist is None:
        exact_target_message_copy_assist = signal_large_map
    constraint_message_copy_assist = (
        args.recurrent_eval_signal_constraint_message_copy_assist
    )
    if constraint_message_copy_assist is None:
        constraint_message_copy_assist = signal_specialist
    constraint_message_guard = args.recurrent_eval_signal_constraint_message_guard
    if constraint_message_guard is None:
        constraint_message_guard = signal_large_map
    recurrent_obs_agent_id_features = args.recurrent_obs_agent_id_features
    if recurrent_obs_agent_id_features is None:
        recurrent_obs_agent_id_features = signal_specialist
    eval_signal_target_scan_threshold = args.recurrent_eval_signal_target_scan_threshold
    if eval_signal_target_scan_threshold is None and signal_specialist:
        eval_signal_target_scan_threshold = 0.0
    eval_signal_target_scan_lock = args.recurrent_eval_signal_target_scan_lock
    if eval_signal_target_scan_lock is None:
        eval_signal_target_scan_lock = False
    eval_signal_exact_target_scan_lock = (
        args.recurrent_eval_signal_exact_target_scan_lock
    )
    if eval_signal_exact_target_scan_lock is None:
        eval_signal_exact_target_scan_lock = False
    eval_signal_compatible_target_scan_assist = (
        args.recurrent_eval_signal_compatible_target_scan_assist
    )
    eval_signal_negative_memory_scan_guard = (
        args.recurrent_eval_signal_negative_memory_scan_guard
    )
    eval_signal_target_probe_assist = args.recurrent_eval_signal_target_probe_assist
    eval_signal_evidence_sweep_assist = args.recurrent_eval_signal_evidence_sweep_assist
    if eval_signal_evidence_sweep_assist is None:
        eval_signal_evidence_sweep_assist = signal_large_map
    dagger_focus_events = args.recurrent_dagger_focus_events
    dagger_replay_event_weights = args.recurrent_dagger_replay_event_weights
    dagger_replay_priority_events = args.recurrent_dagger_replay_priority_events
    if signal_large_map:
        dagger_focus_events = _append_csv_items(
            dagger_focus_events,
            RECURRENT_SIGNAL_LARGE_MAP_FOCUS_EVENTS,
        )
        dagger_replay_event_weights = _append_weighted_csv_items(
            dagger_replay_event_weights,
            RECURRENT_SIGNAL_LARGE_MAP_REPLAY_EVENT_WEIGHTS,
        )
        dagger_replay_priority_events = _append_csv_items(
            dagger_replay_priority_events,
            RECURRENT_SIGNAL_LARGE_MAP_REPLAY_PRIORITY_EVENTS,
        )
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / TRAIN_SCRIPTS["recurrent_bc_rl"]),
        "--scenario",
        case.scenario,
        "--map-size",
        str(case.map_size),
        "--agents",
        str(case.agents),
        "--fov-preset",
        case.fov_preset,
        "--max-steps",
        str(case.max_steps),
        "--oracle",
        _resolve_recurrent_oracle(args, case),
        "--demo-episodes",
        str(args.recurrent_demo_episodes),
        "--bc-epochs",
        str(args.recurrent_bc_epochs),
        "--bc-lr",
        str(args.recurrent_bc_lr),
        "--bc-seq-len",
        str(args.recurrent_bc_seq_len),
        "--bc-action-class-balance-max-weight",
        str(args.recurrent_bc_action_class_balance_max_weight),
        "--bc-event-action-weight",
        str(args.recurrent_bc_event_action_weight),
        "--bc-event-action-events",
        args.recurrent_bc_event_action_events,
        "--bc-comm-loss-weight",
        str(bc_comm_loss_weight),
        "--bc-pipeline-pickup-action-loss-weight",
        str(args.recurrent_bc_pipeline_pickup_action_loss_weight),
        "--bc-pipeline-delivery-action-loss-weight",
        str(args.recurrent_bc_pipeline_delivery_action_loss_weight),
        "--bc-pipeline-delivery-progress-action-loss-weight",
        str(args.recurrent_bc_pipeline_delivery_progress_action_loss_weight),
        "--bc-pipeline-navigation-action-loss-weight",
        str(args.recurrent_bc_pipeline_navigation_action_loss_weight),
        "--bc-pipeline-frontier-exploration-action-loss-weight",
        str(args.recurrent_bc_pipeline_frontier_exploration_action_loss_weight),
        "--bc-pipeline-frontier-exploration-min-map-size",
        str(args.recurrent_bc_pipeline_frontier_exploration_min_map_size),
        "--bc-pipeline-sync-action-loss-weight",
        str(args.recurrent_bc_pipeline_sync_action_loss_weight),
        "--bc-pipeline-ready-interact-action-loss-weight",
        str(args.recurrent_bc_pipeline_ready_interact_action_loss_weight),
        "--bc-pipeline-station-guard-action-loss-weight",
        str(args.recurrent_bc_pipeline_station_guard_action_loss_weight),
        "--bc-pipeline-wrong-station-recovery-action-loss-weight",
        str(args.recurrent_bc_pipeline_wrong_station_recovery_action_loss_weight),
        "--bc-pipeline-pickup-gate-loss-weight",
        str(args.recurrent_bc_pipeline_pickup_gate_loss_weight),
        "--bc-pipeline-pickup-gate-pos-weight",
        str(args.recurrent_bc_pipeline_pickup_gate_pos_weight),
        "--bc-pipeline-pickup-gate-neg-weight",
        str(args.recurrent_bc_pipeline_pickup_gate_neg_weight),
        "--bc-pipeline-plan-action-loss-weight",
        str(args.recurrent_bc_pipeline_plan_action_loss_weight),
        "--bc-pipeline-plan-head-loss-weight",
        str(args.recurrent_bc_pipeline_plan_head_loss_weight),
        "--bc-pipeline-option-loss-weight",
        str(args.recurrent_bc_pipeline_option_loss_weight),
        "--bc-pipeline-message-loss-weight",
        str(args.recurrent_bc_pipeline_message_loss_weight),
        "--bc-pipeline-send-gate-loss-weight",
        str(args.recurrent_bc_pipeline_send_gate_loss_weight),
        "--bc-pipeline-send-gate-pos-weight",
        str(args.recurrent_bc_pipeline_send_gate_pos_weight),
        "--bc-pipeline-send-gate-neg-weight",
        str(args.recurrent_bc_pipeline_send_gate_neg_weight),
        "--bc-pipeline-interact-gate-loss-weight",
        str(args.recurrent_bc_pipeline_interact_gate_loss_weight),
        "--bc-pipeline-interact-gate-pos-weight",
        str(args.recurrent_bc_pipeline_interact_gate_pos_weight),
        "--bc-pipeline-interact-gate-neg-weight",
        str(args.recurrent_bc_pipeline_interact_gate_neg_weight),
        "--bc-pipeline-interact-gate-threshold-target-rate",
        str(args.recurrent_bc_pipeline_interact_gate_threshold_target_rate),
        "--bc-pipeline-bad-pickup-action-loss-weight",
        str(args.recurrent_bc_pipeline_bad_pickup_action_loss_weight),
        "--bc-pipeline-bad-drop-action-loss-weight",
        str(args.recurrent_bc_pipeline_bad_drop_action_loss_weight),
        "--bc-pipeline-bad-interact-action-loss-weight",
        str(args.recurrent_bc_pipeline_bad_interact_action_loss_weight),
        "--bc-pipeline-bad-action-margin-loss-weight",
        str(args.recurrent_bc_pipeline_bad_action_margin_loss_weight),
        "--bc-pipeline-bad-action-margin",
        str(args.recurrent_bc_pipeline_bad_action_margin),
        "--bc-signal-initial-message-weight",
        str(bc_signal_initial_message_weight),
        "--bc-signal-initial-message-loss-weight",
        str(bc_signal_initial_message_loss_weight),
        "--bc-signal-constraint-message-loss-weight",
        str(bc_signal_constraint_message_loss_weight),
        "--bc-signal-target-aux-weight",
        str(bc_signal_target_aux_weight),
        "--bc-signal-target-hypothesis-loss-weight",
        str(bc_signal_target_hypothesis_loss_weight),
        "--bc-signal-target-hypothesis-commit-loss-weight",
        str(bc_signal_target_hypothesis_commit_loss_weight),
        "--bc-signal-target-hypothesis-ambiguity-loss-weight",
        str(bc_signal_target_hypothesis_ambiguity_loss_weight),
        "--bc-signal-target-hypothesis-xy-loss-weight",
        str(bc_signal_target_hypothesis_xy_loss_weight),
        "--bc-signal-target-hypothesis-min-map-size",
        str(bc_signal_target_hypothesis_min_map_size),
        "--bc-signal-target-pursuit-action-weight",
        str(bc_signal_target_pursuit_action_weight),
        "--bc-signal-target-pursuit-max-agents",
        str(bc_signal_target_pursuit_max_agents),
        "--bc-signal-sync-response-action-loss-weight",
        str(bc_signal_sync_response_action_loss_weight),
        "--bc-signal-active-scan-response-action-weight",
        str(bc_signal_active_scan_response_action_weight),
        "--bc-signal-active-scan-response-min-map-size",
        str(bc_signal_active_scan_response_min_map_size),
        "--bc-signal-active-scan-response-max-agents",
        str(bc_signal_active_scan_response_max_agents),
        "--bc-signal-scan-bridge-action-weight",
        str(bc_signal_scan_bridge_action_weight),
        "--bc-signal-scan-bridge-min-map-size",
        str(bc_signal_scan_bridge_min_map_size),
        "--bc-signal-scan-bridge-remaining-threshold",
        str(bc_signal_scan_bridge_remaining_threshold),
        "--bc-signal-scan-bridge-max-teammate-distance",
        str(bc_signal_scan_bridge_max_teammate_distance),
        "--bc-signal-target-match-action-weight",
        str(bc_signal_target_match_action_weight),
        "--bc-signal-first-target-scan-action-weight",
        str(bc_signal_first_target_scan_action_weight),
        "--bc-signal-refresh-target-scan-action-weight",
        str(bc_signal_refresh_target_scan_action_weight),
        "--bc-signal-joint-target-scan-action-weight",
        str(bc_signal_joint_target_scan_action_weight),
        "--bc-signal-target-opportunity-action-weight",
        str(bc_signal_target_opportunity_action_weight),
        "--bc-signal-redundant-target-wait-action-loss-weight",
        str(bc_signal_redundant_target_wait_action_loss_weight),
        "--bc-signal-decoy-drift-action-loss-weight",
        str(bc_signal_decoy_drift_action_loss_weight),
        "--bc-signal-decoy-scan-action-loss-weight",
        str(bc_signal_decoy_scan_action_loss_weight),
        "--bc-signal-rejected-target-drift-action-loss-weight",
        str(bc_signal_rejected_target_drift_action_loss_weight),
        "--bc-signal-clue-interact-action-weight",
        str(bc_signal_clue_interact_action_weight),
        "--bc-signal-clue-interact-min-map-size",
        str(bc_signal_clue_interact_min_map_size),
        "--bc-signal-visible-clue-action-weight",
        str(bc_signal_visible_clue_action_weight),
        "--bc-signal-visible-clue-min-map-size",
        str(bc_signal_visible_clue_min_map_size),
        "--bc-signal-evidence-sweep-action-weight",
        str(bc_signal_evidence_sweep_action_weight),
        "--bc-signal-evidence-sweep-min-map-size",
        str(bc_signal_evidence_sweep_min_map_size),
        "--bc-signal-frontier-exploration-action-weight",
        str(bc_signal_frontier_exploration_action_weight),
        "--bc-signal-frontier-exploration-min-map-size",
        str(bc_signal_frontier_exploration_min_map_size),
        "--pipeline-required-per-stage-min",
        str(args.recurrent_pipeline_required_per_stage_min),
        "--pipeline-required-per-stage-max",
        str(args.recurrent_pipeline_required_per_stage_max),
        "--pipeline-sync-probability",
        str(args.recurrent_pipeline_sync_probability),
        "--pipeline-dependency-probability",
        str(args.recurrent_pipeline_dependency_probability),
        "--pipeline-wrong-delivery-penalty",
        str(args.recurrent_pipeline_wrong_delivery_penalty),
        "--bc-comm-send-pos-weight",
        str(bc_comm_send_pos_weight),
        "--bc-comm-send-rate-penalty-weight",
        str(args.recurrent_bc_comm_send_rate_penalty_weight),
        "--bc-comm-send-rate-target",
        str(args.recurrent_bc_comm_send_rate_target),
        "--bc-send-threshold-target-rate",
        str(args.recurrent_bc_send_threshold_target_rate),
        "--dagger-rounds",
        str(args.recurrent_dagger_rounds),
        "--dagger-episodes",
        str(args.recurrent_dagger_episodes),
        "--dagger-failed-effective-ratio-cap",
        str(args.recurrent_dagger_failed_effective_ratio_cap),
        "--dagger-oracle-action-rollin-rate",
        str(args.recurrent_dagger_oracle_action_rollin_rate),
        "--dagger-oracle-message-rollin-rate",
        str(args.recurrent_dagger_oracle_message_rollin_rate),
        "--dagger-focus-events",
        dagger_focus_events,
        "--dagger-focus-error-weight",
        str(args.recurrent_dagger_focus_error_weight),
        "--dagger-focus-recovery-weight",
        str(args.recurrent_dagger_focus_recovery_weight),
        "--dagger-focus-window",
        str(args.recurrent_dagger_focus_window),
        "--dagger-target-interact-focus-weight",
        str(args.recurrent_dagger_target_interact_focus_weight),
        "--dagger-target-discovery-min-map-size",
        str(args.recurrent_dagger_target_discovery_min_map_size),
        "--dagger-target-discovery-focus-weight",
        str(args.recurrent_dagger_target_discovery_focus_weight),
        "--dagger-movement-stall-min-map-size",
        str(args.recurrent_dagger_movement_stall_min_map_size),
        "--dagger-movement-stall-window",
        str(args.recurrent_dagger_movement_stall_window),
        "--dagger-movement-stall-focus-weight",
        str(args.recurrent_dagger_movement_stall_focus_weight),
        "--dagger-target-decoy-drift-focus-weight",
        str(args.recurrent_dagger_target_decoy_drift_focus_weight),
        "--dagger-replay-pre-steps",
        str(args.recurrent_dagger_replay_pre_steps),
        "--dagger-replay-post-steps",
        str(args.recurrent_dagger_replay_post_steps),
        "--dagger-replay-weight",
        str(args.recurrent_dagger_replay_weight),
        "--dagger-positive-replay-events",
        args.recurrent_dagger_positive_replay_events,
        "--dagger-replay-event-weights",
        dagger_replay_event_weights,
        "--dagger-replay-event-caps",
        args.recurrent_dagger_replay_event_caps,
        "--dagger-replay-success-only-events",
        args.recurrent_dagger_replay_success_only_events,
        "--dagger-replay-priority-events",
        dagger_replay_priority_events,
        "--dagger-replay-balance-positive-events",
        args.recurrent_dagger_replay_balance_positive_events,
        "--dagger-replay-balance-negative-events",
        args.recurrent_dagger_replay_balance_negative_events,
        "--dagger-replay-max-negative-per-positive",
        str(args.recurrent_dagger_replay_max_negative_per_positive),
        "--dagger-max-replay-snippets-per-episode",
        str(args.recurrent_dagger_max_replay_snippets_per_episode),
        "--dagger-max-failed-parent-replay-snippets-per-episode",
        str(args.recurrent_dagger_max_failed_parent_replay_snippets_per_episode),
        "--dagger-failed-parent-replay-weight-scale",
        str(args.recurrent_dagger_failed_parent_replay_weight_scale),
        "--dagger-expert-max-replay-snippets-per-episode",
        str(args.recurrent_dagger_expert_max_replay_snippets_per_episode),
        "--dagger-signal-target-rendezvous-min-map-size",
        str(args.recurrent_dagger_signal_target_rendezvous_min_map_size),
        "--dagger-signal-target-rendezvous-max-agents",
        str(args.recurrent_dagger_signal_target_rendezvous_max_agents),
        "--rl-updates",
        str(rl_updates),
        "--rl-early-stop-eval-patience",
        str(args.recurrent_rl_early_stop_eval_patience),
        "--rollout-steps",
        str(args.rollout_steps),
        "--rl-epochs",
        str(args.recurrent_rl_epochs),
        "--minibatch-seqs",
        str(args.recurrent_minibatch_seqs),
        "--rl-lr",
        str(args.recurrent_rl_lr),
        "--clip",
        str(args.recurrent_clip),
        "--entropy-coeff",
        str(args.recurrent_entropy_coeff),
        "--max-grad-norm",
        str(args.recurrent_max_grad_norm),
        "--bc-kl-coeff",
        str(args.recurrent_bc_kl_coeff),
        "--bc-comm-kl-coeff",
        str(args.recurrent_bc_comm_kl_coeff),
        "--rl-eval-decoding-action-loss-weight",
        str(args.recurrent_rl_eval_decoding_action_loss_weight),
        "--rl-pipeline-assisted-action-loss-weight",
        str(args.recurrent_rl_pipeline_assisted_action_loss_weight),
        "--rl-pipeline-interact-gate-loss-weight",
        str(args.recurrent_rl_pipeline_interact_gate_loss_weight),
        "--rl-pipeline-interact-gate-pos-weight",
        str(args.recurrent_rl_pipeline_interact_gate_pos_weight),
        "--rl-pipeline-interact-gate-neg-weight",
        str(args.recurrent_rl_pipeline_interact_gate_neg_weight),
        "--rl-pipeline-pickup-gate-loss-weight",
        str(args.recurrent_rl_pipeline_pickup_gate_loss_weight),
        "--rl-pipeline-pickup-gate-pos-weight",
        str(args.recurrent_rl_pipeline_pickup_gate_pos_weight),
        "--rl-pipeline-pickup-gate-neg-weight",
        str(args.recurrent_rl_pipeline_pickup_gate_neg_weight),
        "--rl-pipeline-delivery-progress-action-loss-weight",
        str(args.recurrent_rl_pipeline_delivery_progress_action_loss_weight),
        "--rl-pipeline-navigation-action-loss-weight",
        str(args.recurrent_rl_pipeline_navigation_action_loss_weight),
        "--rl-pipeline-sync-action-loss-weight",
        str(args.recurrent_rl_pipeline_sync_action_loss_weight),
        "--rl-pipeline-ready-interact-action-loss-weight",
        str(args.recurrent_rl_pipeline_ready_interact_action_loss_weight),
        "--rl-pipeline-station-guard-action-loss-weight",
        str(args.recurrent_rl_pipeline_station_guard_action_loss_weight),
        "--rl-pipeline-wrong-station-recovery-action-loss-weight",
        str(args.recurrent_rl_pipeline_wrong_station_recovery_action_loss_weight),
        "--rl-pipeline-plan-action-loss-weight",
        str(args.recurrent_rl_pipeline_plan_action_loss_weight),
        "--rl-pipeline-plan-head-loss-weight",
        str(args.recurrent_rl_pipeline_plan_head_loss_weight),
        "--rl-pipeline-option-loss-weight",
        str(args.recurrent_rl_pipeline_option_loss_weight),
        "--rl-pipeline-bad-pickup-penalty",
        str(args.recurrent_rl_pipeline_bad_pickup_penalty),
        "--rl-pipeline-bad-interact-penalty",
        str(args.recurrent_rl_pipeline_bad_interact_penalty),
        "--rl-pipeline-unneeded-drop-bonus",
        str(args.recurrent_rl_pipeline_unneeded_drop_bonus),
        "--rl-eval-every",
        str(args.eval_every),
        "--rl-eval-episodes",
        str(rl_eval_episodes),
        "--eval-episodes",
        str(eval_episodes),
        "--eval-seed-count",
        str(args.recurrent_eval_seed_count),
        "--eval-send-threshold",
        str(args.recurrent_eval_send_threshold),
        "--hidden-dim",
        str(args.recurrent_hidden_dim),
        "--recurrent-backbone",
        args.recurrent_backbone,
        "--comm-token-limit",
        str(args.recurrent_comm_token_limit),
        "--comm-vocab-size",
        str(args.recurrent_comm_vocab_size),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--save",
        str(checkpoint_path),
        "--wandb-project",
        args.wandb_project,
        "--wandb-run",
        run_name,
    ]
    if case.energy_preset is not None:
        cmd.extend(["--energy-preset", case.energy_preset])
    if args.recurrent_bc_action_class_balance:
        cmd.append("--bc-action-class-balance")
    if args.recurrent_bc_calibrate_send_threshold:
        cmd.append("--bc-calibrate-send-threshold")
    if args.recurrent_pipeline_stage_count is not None:
        cmd.extend(["--pipeline-stage-count", str(args.recurrent_pipeline_stage_count)])
    if args.recurrent_train_map_sizes:
        cmd.extend(["--train-map-sizes", args.recurrent_train_map_sizes])
    if args.recurrent_train_map_sampling_weights:
        cmd.extend(["--train-map-sampling-weights", args.recurrent_train_map_sampling_weights])
    if args.recurrent_map_max_steps:
        cmd.extend(["--map-max-steps", args.recurrent_map_max_steps])
    if args.recurrent_eval_map_sizes:
        cmd.extend(["--eval-map-sizes", args.recurrent_eval_map_sizes])
    if args.recurrent_eval_seed_list:
        cmd.extend(["--eval-seed-list", args.recurrent_eval_seed_list])
    if case.scenario == "signal_hunt" and eval_signal_target_scan_threshold is not None:
        cmd.extend([
            "--eval-signal-target-scan-threshold",
            str(eval_signal_target_scan_threshold),
        ])
    if args.recurrent_bc_signal_target_pursuit_trust_exact_memory:
        cmd.append("--bc-signal-target-pursuit-trust-exact-memory")
    if case.scenario == "signal_hunt" and eval_signal_target_scan_lock:
        cmd.append("--eval-signal-target-scan-lock")
    if case.scenario == "signal_hunt" and eval_signal_exact_target_scan_lock:
        cmd.append("--eval-signal-exact-target-scan-lock")
    if case.scenario == "signal_hunt" and eval_signal_compatible_target_scan_assist:
        cmd.append("--eval-signal-compatible-target-scan-assist")
        cmd.extend([
            "--eval-signal-compatible-target-scan-min-strength",
            str(args.recurrent_eval_signal_compatible_target_scan_min_strength),
        ])
    if case.scenario == "signal_hunt" and eval_signal_negative_memory_scan_guard:
        cmd.append("--eval-signal-negative-memory-scan-guard")
    if case.scenario == "signal_hunt" and eval_signal_target_probe_assist:
        cmd.append("--eval-signal-target-probe-assist")
    if case.scenario == "signal_hunt" and eval_signal_evidence_sweep_assist:
        cmd.append("--eval-signal-evidence-sweep-assist")
        cmd.extend([
            "--eval-signal-evidence-sweep-min-step",
            str(args.recurrent_eval_signal_evidence_sweep_min_step),
        ])
    if case.scenario == "signal_hunt" and args.recurrent_eval_signal_frontier_exploration_assist:
        cmd.append("--eval-signal-frontier-exploration-assist")
    if case.scenario == "signal_hunt" and args.recurrent_eval_signal_scan_refresh_assist:
        cmd.append("--eval-signal-scan-refresh-assist")
        cmd.extend([
            "--eval-signal-scan-refresh-threshold",
            str(args.recurrent_eval_signal_scan_refresh_threshold),
        ])
    if case.scenario == "signal_hunt" and constraint_message_copy_assist:
        cmd.append("--eval-signal-constraint-message-copy-assist")
    if case.scenario == "signal_hunt" and constraint_message_guard:
        cmd.append("--eval-signal-constraint-message-guard")
    if args.recurrent_dagger_seed_list:
        cmd.extend(["--dagger-seed-list", args.recurrent_dagger_seed_list])
    if (
        case.scenario == "pipeline_assembly"
        and int(args.recurrent_pipeline_assisted_rollout_episodes) > 0
    ):
        cmd.extend([
            "--pipeline-assisted-rollout-episodes",
            str(args.recurrent_pipeline_assisted_rollout_episodes),
            "--pipeline-assisted-rollout-seed-base",
            str(args.recurrent_pipeline_assisted_rollout_seed_base),
            "--pipeline-assisted-rollout-max-steps-per-episode",
            str(args.recurrent_pipeline_assisted_rollout_max_steps_per_episode),
            "--pipeline-assisted-rollout-weight",
            str(args.recurrent_pipeline_assisted_rollout_weight),
            "--pipeline-assisted-rollout-bc-epochs",
            str(args.recurrent_pipeline_assisted_rollout_bc_epochs),
        ])
        if args.recurrent_pipeline_assisted_rollout_seed_list:
            cmd.extend([
                "--pipeline-assisted-rollout-seed-list",
                args.recurrent_pipeline_assisted_rollout_seed_list,
            ])
        if args.recurrent_pipeline_assisted_rollout_success_only:
            cmd.append("--pipeline-assisted-rollout-success-only")
        if not args.recurrent_pipeline_assisted_rollout_navigation_assist:
            cmd.append("--no-pipeline-assisted-rollout-navigation-assist")
        if not args.recurrent_pipeline_assisted_rollout_navigation_assist_trust_messages:
            cmd.append("--no-pipeline-assisted-rollout-navigation-assist-trust-messages")
        if not args.recurrent_pipeline_assisted_rollout_station_interact_guard:
            cmd.append("--no-pipeline-assisted-rollout-station-interact-guard")
    recurrent_init = _resolve_recurrent_init(
        args,
        case=case,
        seed=seed,
        run_name=run_name,
    )
    if args.recurrent_skip_bc:
        cmd.append("--skip-bc")
    if recurrent_init:
        cmd.extend(["--recurrent-init", recurrent_init])
    if args.recurrent_init_for_dagger:
        cmd.append("--recurrent-init-for-dagger")
    if args.recurrent_init_allow_obs_dim_mismatch:
        cmd.append("--recurrent-init-allow-obs-dim-mismatch")
    if args.recurrent_rl_balanced_rollouts:
        cmd.append("--rl-balanced-rollouts")
    if args.recurrent_rl_rollout_eval_decoding:
        cmd.append("--rl-rollout-eval-decoding")
    if args.recurrent_rl_eval_use_eval_seeds:
        cmd.append("--rl-eval-use-eval-seeds")
    if args.recurrent_rl_rollout_pipeline_navigation_assist and case.scenario == "pipeline_assembly":
        cmd.append("--rl-rollout-pipeline-navigation-assist")
    if (
        args.recurrent_rl_rollout_pipeline_navigation_assist_trust_messages
        and case.scenario == "pipeline_assembly"
    ):
        cmd.append("--rl-rollout-pipeline-navigation-assist-trust-messages")
    if (
        args.recurrent_rl_rollout_pipeline_station_interact_guard
        and case.scenario == "pipeline_assembly"
    ):
        cmd.append("--rl-rollout-pipeline-station-interact-guard")
    if (
        args.recurrent_rl_rollout_pipeline_interact_gate_promote
        and case.scenario == "pipeline_assembly"
    ):
        cmd.append("--rl-rollout-pipeline-interact-gate-promote")
    if not args.recurrent_rl_restore_best:
        cmd.append("--no-rl-restore-best")
    if not args.recurrent_rl_save_best:
        cmd.append("--no-rl-save-best")
    if args.recurrent_comm:
        cmd.append("--comm")
    if args.recurrent_calibrate_send_threshold:
        cmd.append("--bc-calibrate-send-threshold")
    if args.recurrent_obs_exploration_memory:
        cmd.append("--obs-exploration-memory")
        if args.recurrent_obs_exploration_age:
            cmd.append("--obs-exploration-age")
    if args.recurrent_obs_memory_mode != "auto" and not signal_specialist:
        cmd.extend([
            "--obs-memory-mode",
            args.recurrent_obs_memory_mode,
            "--obs-memory-radius",
            str(args.recurrent_obs_memory_radius),
        ])
    recurrent_pipeline_feedback = (
        case.scenario == "pipeline_assembly"
        and (
            args.recurrent_obs_pipeline_feedback
            or args.recurrent_obs_pipeline_shared_feedback
        )
    )
    recurrent_signal_feedback = (
        case.scenario == "signal_hunt"
        and (
            args.recurrent_obs_signal_sync_feedback
            or args.recurrent_obs_signal_scan_state
        )
    )
    if args.recurrent_obs_feedback or recurrent_pipeline_feedback or recurrent_signal_feedback:
        cmd.append("--obs-feedback")
    if args.recurrent_obs_normalize_tokens:
        cmd.append("--obs-normalize-tokens")
    if args.recurrent_obs_navigation_features:
        cmd.append("--obs-navigation-features")
    if args.recurrent_obs_signal_features:
        cmd.append("--obs-signal-features")
    if args.recurrent_obs_signal_target_match_features:
        cmd.append("--obs-signal-target-match-features")
    if args.recurrent_obs_signal_confidence_features:
        cmd.append("--obs-signal-confidence-features")
    if args.recurrent_obs_signal_sector_features:
        cmd.append("--obs-signal-sector-features")
    if args.recurrent_obs_signal_sync_feedback:
        cmd.append("--obs-signal-sync-feedback")
    if args.recurrent_obs_signal_scan_state:
        cmd.append("--obs-signal-scan-state")
    if recurrent_obs_agent_id_features:
        cmd.append("--obs-agent-id-features")
    if (
        args.recurrent_bc_pipeline_proactive_bad_action_labels
        and case.scenario == "pipeline_assembly"
    ):
        cmd.append("--bc-pipeline-proactive-bad-action-labels")
    if args.recurrent_bc_signal_constraint_frontier_bias and case.scenario == "signal_hunt":
        cmd.append("--bc-signal-constraint-frontier-bias")
    if (
        args.recurrent_bc_calibrate_pipeline_interact_gate_threshold
        and case.scenario == "pipeline_assembly"
    ):
        cmd.append("--bc-calibrate-pipeline-interact-gate-threshold")
    if args.recurrent_dagger_focus_replay:
        cmd.append("--dagger-focus-replay")
    if not args.recurrent_dagger_retrain_from_scratch:
        cmd.append("--no-dagger-retrain-from-scratch")
    if not args.recurrent_dagger_restore_best:
        cmd.append("--no-dagger-restore-best")
    if initial_target_broadcast_labels and case.scenario == "signal_hunt":
        cmd.append("--dagger-initial-target-broadcast-labels")
    if target_handoff_requires_exact_target and case.scenario == "signal_hunt":
        cmd.append("--dagger-target-handoff-requires-exact-target")
    if args.recurrent_dagger_signal_target_rendezvous_labels and case.scenario == "signal_hunt":
        cmd.append("--dagger-signal-target-rendezvous-labels")
    if (
        args.recurrent_dagger_pipeline_wrong_delivery_provenance_labels
        and case.scenario == "pipeline_assembly"
    ):
        cmd.extend([
            "--dagger-pipeline-wrong-delivery-provenance-labels",
            "--dagger-pipeline-wrong-delivery-provenance-weight",
            str(args.recurrent_dagger_pipeline_wrong_delivery_provenance_weight),
        ])
    if args.recurrent_obs_pipeline_features and case.scenario == "pipeline_assembly":
        cmd.append("--obs-pipeline-features")
    if args.recurrent_obs_pipeline_progress_features and case.scenario == "pipeline_assembly":
        cmd.append("--obs-pipeline-progress-features")
    if recurrent_pipeline_feedback:
        cmd.append("--obs-pipeline-feedback")
        if args.recurrent_obs_pipeline_feedback_metadata:
            cmd.append("--obs-pipeline-feedback-metadata")
        else:
            cmd.append("--no-obs-pipeline-feedback-metadata")
        if args.recurrent_obs_pipeline_shared_feedback:
            cmd.append("--obs-pipeline-shared-feedback")
    if args.recurrent_eval_pipeline_navigation_assist and case.scenario == "pipeline_assembly":
        cmd.append("--eval-pipeline-navigation-assist")
    if args.recurrent_eval_pipeline_navigation_assist_trust_messages and case.scenario == "pipeline_assembly":
        cmd.append("--eval-pipeline-navigation-assist-trust-messages")
    if args.recurrent_eval_pipeline_station_interact_guard and case.scenario == "pipeline_assembly":
        cmd.append("--eval-pipeline-station-interact-guard")
    if (
        float(args.recurrent_eval_pipeline_interact_gate_threshold) >= 0.0
        and case.scenario == "pipeline_assembly"
    ):
        cmd.extend([
            "--eval-pipeline-interact-gate-threshold",
            str(args.recurrent_eval_pipeline_interact_gate_threshold),
        ])
    if args.recurrent_eval_pipeline_interact_gate_promote and case.scenario == "pipeline_assembly":
        cmd.append("--eval-pipeline-interact-gate-promote")
    if (
        float(args.recurrent_eval_pipeline_event_head_threshold) >= 0.0
        and case.scenario == "pipeline_assembly"
    ):
        cmd.extend([
            "--eval-pipeline-event-head-threshold",
            str(args.recurrent_eval_pipeline_event_head_threshold),
        ])
    if (
        float(args.recurrent_eval_pipeline_navigation_head_threshold) >= 0.0
        and case.scenario == "pipeline_assembly"
    ):
        cmd.extend([
            "--eval-pipeline-navigation-head-threshold",
            str(args.recurrent_eval_pipeline_navigation_head_threshold),
        ])
    if (
        float(args.recurrent_eval_pipeline_plan_head_threshold) >= 0.0
        and case.scenario == "pipeline_assembly"
    ):
        cmd.extend([
            "--eval-pipeline-plan-head-threshold",
            str(args.recurrent_eval_pipeline_plan_head_threshold),
        ])
    if (
        float(args.recurrent_eval_pipeline_option_threshold) >= 0.0
        and case.scenario == "pipeline_assembly"
    ):
        cmd.extend([
            "--eval-pipeline-option-threshold",
            str(args.recurrent_eval_pipeline_option_threshold),
        ])
    if args.recurrent_eval_pipeline_option_allow_interact and case.scenario == "pipeline_assembly":
        cmd.append("--eval-pipeline-option-allow-interact")
    if signal_specialist or (case.scenario == "signal_hunt" and signal_scan_head_overrides):
        cmd.extend([
            "--bc-signal-scan-decision-loss-weight",
            str(bc_signal_scan_decision_loss_weight),
            "--bc-signal-scan-decision-pos-weight",
            str(bc_signal_scan_decision_pos_weight),
            "--bc-signal-scan-decision-neg-weight",
            str(bc_signal_scan_decision_neg_weight),
            "--bc-signal-scan-gate-loss-weight",
            str(bc_signal_scan_gate_loss_weight),
            "--bc-signal-scan-gate-pos-weight",
            str(bc_signal_scan_gate_pos_weight),
            "--bc-signal-scan-gate-neg-weight",
            str(bc_signal_scan_gate_neg_weight),
            "--bc-signal-target-validity-loss-weight",
            str(bc_signal_target_validity_loss_weight),
            "--bc-signal-target-validity-pos-weight",
            str(bc_signal_target_validity_pos_weight),
            "--bc-signal-target-validity-neg-weight",
            str(bc_signal_target_validity_neg_weight),
            "--bc-signal-target-decision-loss-weight",
            str(bc_signal_target_decision_loss_weight),
            "--bc-signal-target-decision-pos-weight",
            str(bc_signal_target_decision_pos_weight),
            "--bc-signal-target-decision-neg-weight",
            str(bc_signal_target_decision_neg_weight),
            "--bc-signal-ambiguous-target-decision-min-map-size",
            str(args.recurrent_bc_signal_ambiguous_target_decision_min_map_size),
            "--bc-signal-ambiguous-target-search-min-map-size",
            str(args.recurrent_bc_signal_ambiguous_target_search_min_map_size),
        ])
    if args.recurrent_bc_signal_ambiguous_target_decision_negatives and case.scenario == "signal_hunt":
        cmd.append("--bc-signal-ambiguous-target-decision-negatives")
    if args.recurrent_bc_signal_ambiguous_target_search_labels and case.scenario == "signal_hunt":
        cmd.append("--bc-signal-ambiguous-target-search-labels")
    if signal_specialist:
        cmd.extend([
            "--obs-exploration-memory",
            "--obs-feedback",
            "--obs-normalize-tokens",
            "--obs-memory-mode",
            _resolve_recurrent_signal_obs_memory_mode(args),
            "--obs-memory-radius",
            str(args.recurrent_obs_memory_radius),
            "--obs-navigation-features",
            "--obs-signal-features",
            "--obs-signal-negative-memory",
            "--obs-signal-inferred-target-features",
            "--obs-signal-target-match-features",
            "--obs-signal-sync-feedback",
            "--obs-signal-scan-state",
            "--eval-signal-scan-sync-assist",
            "--eval-signal-scan-broadcast-assist",
            "--eval-signal-exact-target-message-guard",
            "--eval-signal-exact-target-navigation-assist",
            "--eval-signal-exact-target-memory-steps",
            str(args.recurrent_eval_signal_exact_target_memory_steps),
            "--eval-signal-scan-gate-threshold",
            "0.4",
            "--eval-signal-scan-gate-suppress",
            "--eval-signal-target-validity-threshold",
            "0.4",
            "--eval-signal-target-decision-threshold",
            "0.4",
        ])
        if recurrent_obs_exploration_age:
            cmd.append("--obs-exploration-age")
    if case.scenario == "signal_hunt" and initial_exact_message_copy_assist:
        cmd.append("--eval-signal-initial-exact-message-copy-assist")
    if case.scenario == "signal_hunt" and exact_target_message_copy_assist:
        cmd.append("--eval-signal-exact-target-message-copy-assist")
    cmd.extend(_learning_profile_args(args.learning_profile, "recurrent_bc_rl", case))
    if args.wandb:
        cmd.append("--wandb")
    return cmd


def _learning_profile_args(profile: str, algorithm: str, case: ScenarioCase) -> list[str]:
    if profile == "bare":
        return []

    cmd = list(SCENARIO_SHAPING_ARGS.get(case.scenario, []))
    if profile == "shaped":
        return cmd
    if profile != "comm_curriculum":
        raise ValueError(f"Unknown learning profile: {profile}")

    if algorithm in {"mappo", "comm_mat"}:
        cmd.extend([
            "--comm-cost",
            "0.0",
            "--comm-send-target",
            "0.25",
            "--comm-send-target-coeff",
            "0.05",
        ])
    elif algorithm == "tarmac":
        cmd.extend(["--attn-entropy-coeff", "0.01"])
    return cmd


def _resolve_recurrent_init(args, *, case: ScenarioCase, seed: int, run_name: str) -> str:
    if args.recurrent_init_template:
        return args.recurrent_init_template.format(
            seed=seed,
            scenario=case.scenario,
            map_size=case.map_size,
            agents=case.agents,
            algorithm="recurrent_bc_rl",
            run_name=run_name,
        )
    return args.recurrent_init or ""


def _resolve_recurrent_oracle(args, case: ScenarioCase) -> str:
    if args.recurrent_oracle == "auto":
        return RECURRENT_AUTO_ORACLES[case.scenario]
    return args.recurrent_oracle


def _apply_recurrent_ppo_profile(args):
    profile = RECURRENT_PPO_PROFILES[args.recurrent_ppo_profile]
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def run_suite(args) -> dict:
    suite_dir = _suite_dir(args)
    suite_dir.mkdir(parents=True, exist_ok=True)
    cases = _cases_from_args(args)
    records: list[RunRecord] = []

    for algorithm in args.algorithms:
        for case in cases:
            for seed in args.seeds:
                case_name = case.benchmark_case or f"{case.scenario}_{case.map_size}x{case.map_size}"
                run_name = f"{algorithm}_{_safe_run_name(case_name)}_seed{seed}"
                run_dir = suite_dir / run_name
                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"{algorithm}.pt"
                stdout_path = run_dir / "stdout.log"
                stderr_path = run_dir / "stderr.log"
                cmd = build_command(
                    algorithm=algorithm,
                    case=case,
                    checkpoint_path=checkpoint_path,
                    args=args,
                    run_name=run_name,
                    seed=seed,
                )
                record = _run_one(
                    cmd=cmd,
                    algorithm=algorithm,
                    case=case,
                    seed=seed,
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    dry_run=args.dry_run,
                    wandb_mode=args.wandb_mode,
                    strict_wandb=args.strict_wandb,
                )
                records.append(record)
                _write_json(run_dir / "run_summary.json", asdict(record))
                print(_format_record(record), flush=True)
                if args.fail_fast and record.status == "failed":
                    break
            if args.fail_fast and records and records[-1].status == "failed":
                break
        if args.fail_fast and records and records[-1].status == "failed":
            break

    payload = {
        "suite": "core_training_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir),
        "dry_run": bool(args.dry_run),
        "config": {
            "algorithms": args.algorithms,
            "scenarios": args.scenarios,
            "benchmark_spec": args.benchmark_spec,
            "benchmark_cases": args.benchmark_cases,
            "updates": args.updates,
            "rollout_steps": args.rollout_steps,
            "epochs": args.epochs,
            "minibatch": args.minibatch,
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
            "recurrent_eval_episodes": args.recurrent_eval_episodes,
            "device": args.device,
            "seeds": args.seeds,
            "learning_profile": args.learning_profile,
            "mappo_backbone": args.mappo_backbone,
            "wandb": args.wandb,
            "wandb_mode": args.wandb_mode,
            "wandb_project": args.wandb_project,
            "strict_wandb": args.strict_wandb,
            "mappo_critic_mode": args.mappo_critic_mode,
            "mappo_shared_actor": args.mappo_shared_actor,
            "mappo_obs_exploration_memory": args.mappo_obs_exploration_memory,
            "mappo_obs_exploration_age": args.mappo_obs_exploration_age,
            "mappo_eval_action_mode": args.mappo_eval_action_mode,
            "mappo_eval_action_temperature": args.mappo_eval_action_temperature,
            "mappo_eval_send_mode": args.mappo_eval_send_mode,
            "mappo_eval_send_threshold": args.mappo_eval_send_threshold,
            "mappo_eval_token_mode": args.mappo_eval_token_mode,
            "mappo_eval_token_temperature": args.mappo_eval_token_temperature,
            "mappo_eval_length_mode": args.mappo_eval_length_mode,
            "mappo_eval_length_temperature": args.mappo_eval_length_temperature,
            "recurrent_oracle": args.recurrent_oracle,
            "recurrent_resolved_oracles": {
                case.scenario: _resolve_recurrent_oracle(args, case)
                for case in cases
            },
            "recurrent_signal_preset": args.recurrent_signal_preset,
            "recurrent_demo_episodes": args.recurrent_demo_episodes,
            "recurrent_bc_epochs": args.recurrent_bc_epochs,
            "recurrent_bc_lr": args.recurrent_bc_lr,
            "recurrent_bc_seq_len": args.recurrent_bc_seq_len,
            "recurrent_bc_action_class_balance": args.recurrent_bc_action_class_balance,
            "recurrent_bc_action_class_balance_max_weight": args.recurrent_bc_action_class_balance_max_weight,
            "recurrent_bc_event_action_weight": args.recurrent_bc_event_action_weight,
            "recurrent_bc_event_action_events": args.recurrent_bc_event_action_events,
            "recurrent_bc_comm_loss_weight": args.recurrent_bc_comm_loss_weight,
            "recurrent_bc_pipeline_pickup_action_loss_weight": (
                args.recurrent_bc_pipeline_pickup_action_loss_weight
            ),
            "recurrent_bc_pipeline_delivery_action_loss_weight": (
                args.recurrent_bc_pipeline_delivery_action_loss_weight
            ),
            "recurrent_bc_pipeline_delivery_progress_action_loss_weight": (
                args.recurrent_bc_pipeline_delivery_progress_action_loss_weight
            ),
            "recurrent_bc_pipeline_navigation_action_loss_weight": (
                args.recurrent_bc_pipeline_navigation_action_loss_weight
            ),
            "recurrent_bc_pipeline_frontier_exploration_action_loss_weight": (
                args.recurrent_bc_pipeline_frontier_exploration_action_loss_weight
            ),
            "recurrent_bc_pipeline_frontier_exploration_min_map_size": (
                args.recurrent_bc_pipeline_frontier_exploration_min_map_size
            ),
            "recurrent_bc_pipeline_sync_action_loss_weight": (
                args.recurrent_bc_pipeline_sync_action_loss_weight
            ),
            "recurrent_bc_pipeline_ready_interact_action_loss_weight": (
                args.recurrent_bc_pipeline_ready_interact_action_loss_weight
            ),
            "recurrent_bc_pipeline_station_guard_action_loss_weight": (
                args.recurrent_bc_pipeline_station_guard_action_loss_weight
            ),
            "recurrent_bc_pipeline_wrong_station_recovery_action_loss_weight": (
                args.recurrent_bc_pipeline_wrong_station_recovery_action_loss_weight
            ),
            "recurrent_bc_pipeline_pickup_gate_loss_weight": (
                args.recurrent_bc_pipeline_pickup_gate_loss_weight
            ),
            "recurrent_bc_pipeline_pickup_gate_pos_weight": (
                args.recurrent_bc_pipeline_pickup_gate_pos_weight
            ),
            "recurrent_bc_pipeline_pickup_gate_neg_weight": (
                args.recurrent_bc_pipeline_pickup_gate_neg_weight
            ),
            "recurrent_bc_pipeline_plan_action_loss_weight": (
                args.recurrent_bc_pipeline_plan_action_loss_weight
            ),
            "recurrent_bc_pipeline_plan_head_loss_weight": (
                args.recurrent_bc_pipeline_plan_head_loss_weight
            ),
            "recurrent_bc_pipeline_option_loss_weight": (
                args.recurrent_bc_pipeline_option_loss_weight
            ),
            "recurrent_bc_pipeline_message_loss_weight": (
                args.recurrent_bc_pipeline_message_loss_weight
            ),
            "recurrent_bc_pipeline_send_gate_loss_weight": (
                args.recurrent_bc_pipeline_send_gate_loss_weight
            ),
            "recurrent_bc_pipeline_send_gate_pos_weight": (
                args.recurrent_bc_pipeline_send_gate_pos_weight
            ),
            "recurrent_bc_pipeline_send_gate_neg_weight": (
                args.recurrent_bc_pipeline_send_gate_neg_weight
            ),
            "recurrent_bc_pipeline_interact_gate_loss_weight": (
                args.recurrent_bc_pipeline_interact_gate_loss_weight
            ),
            "recurrent_bc_pipeline_interact_gate_pos_weight": (
                args.recurrent_bc_pipeline_interact_gate_pos_weight
            ),
            "recurrent_bc_pipeline_interact_gate_neg_weight": (
                args.recurrent_bc_pipeline_interact_gate_neg_weight
            ),
            "recurrent_bc_calibrate_pipeline_interact_gate_threshold": (
                args.recurrent_bc_calibrate_pipeline_interact_gate_threshold
            ),
            "recurrent_bc_pipeline_interact_gate_threshold_target_rate": (
                args.recurrent_bc_pipeline_interact_gate_threshold_target_rate
            ),
            "recurrent_bc_pipeline_bad_pickup_action_loss_weight": (
                args.recurrent_bc_pipeline_bad_pickup_action_loss_weight
            ),
            "recurrent_bc_pipeline_bad_drop_action_loss_weight": (
                args.recurrent_bc_pipeline_bad_drop_action_loss_weight
            ),
            "recurrent_bc_pipeline_bad_interact_action_loss_weight": (
                args.recurrent_bc_pipeline_bad_interact_action_loss_weight
            ),
            "recurrent_bc_pipeline_bad_action_margin_loss_weight": (
                args.recurrent_bc_pipeline_bad_action_margin_loss_weight
            ),
            "recurrent_bc_pipeline_bad_action_margin": (
                args.recurrent_bc_pipeline_bad_action_margin
            ),
            "recurrent_bc_pipeline_proactive_bad_action_labels": (
                args.recurrent_bc_pipeline_proactive_bad_action_labels
            ),
            "recurrent_pipeline_stage_count": args.recurrent_pipeline_stage_count,
            "recurrent_pipeline_required_per_stage_min": args.recurrent_pipeline_required_per_stage_min,
            "recurrent_pipeline_required_per_stage_max": args.recurrent_pipeline_required_per_stage_max,
            "recurrent_pipeline_sync_probability": args.recurrent_pipeline_sync_probability,
            "recurrent_pipeline_dependency_probability": args.recurrent_pipeline_dependency_probability,
            "recurrent_pipeline_wrong_delivery_penalty": args.recurrent_pipeline_wrong_delivery_penalty,
            "recurrent_obs_pipeline_features": args.recurrent_obs_pipeline_features,
            "recurrent_obs_pipeline_feedback": args.recurrent_obs_pipeline_feedback,
            "recurrent_obs_pipeline_feedback_metadata": (
                args.recurrent_obs_pipeline_feedback_metadata
            ),
            "recurrent_obs_pipeline_progress_features": (
                args.recurrent_obs_pipeline_progress_features
            ),
            "recurrent_obs_pipeline_shared_feedback": (
                args.recurrent_obs_pipeline_shared_feedback
            ),
            "recurrent_eval_pipeline_navigation_assist": args.recurrent_eval_pipeline_navigation_assist,
            "recurrent_eval_pipeline_navigation_assist_trust_messages": (
                args.recurrent_eval_pipeline_navigation_assist_trust_messages
            ),
            "recurrent_eval_pipeline_station_interact_guard": (
                args.recurrent_eval_pipeline_station_interact_guard
            ),
            "recurrent_eval_pipeline_interact_gate_threshold": (
                args.recurrent_eval_pipeline_interact_gate_threshold
            ),
            "recurrent_eval_pipeline_interact_gate_promote": (
                args.recurrent_eval_pipeline_interact_gate_promote
            ),
            "recurrent_eval_pipeline_event_head_threshold": (
                args.recurrent_eval_pipeline_event_head_threshold
            ),
            "recurrent_eval_pipeline_navigation_head_threshold": (
                args.recurrent_eval_pipeline_navigation_head_threshold
            ),
            "recurrent_eval_pipeline_plan_head_threshold": (
                args.recurrent_eval_pipeline_plan_head_threshold
            ),
            "recurrent_eval_pipeline_option_threshold": (
                args.recurrent_eval_pipeline_option_threshold
            ),
            "recurrent_eval_pipeline_option_allow_interact": (
                args.recurrent_eval_pipeline_option_allow_interact
            ),
            "recurrent_bc_calibrate_send_threshold": args.recurrent_bc_calibrate_send_threshold,
            "recurrent_bc_send_threshold_target_rate": args.recurrent_bc_send_threshold_target_rate,
            "recurrent_bc_comm_send_pos_weight": args.recurrent_bc_comm_send_pos_weight,
            "recurrent_bc_comm_send_rate_penalty_weight": args.recurrent_bc_comm_send_rate_penalty_weight,
            "recurrent_bc_comm_send_rate_target": args.recurrent_bc_comm_send_rate_target,
            "recurrent_bc_signal_initial_message_weight": (
                args.recurrent_bc_signal_initial_message_weight
            ),
            "recurrent_bc_signal_initial_message_loss_weight": (
                args.recurrent_bc_signal_initial_message_loss_weight
            ),
            "recurrent_bc_signal_constraint_message_loss_weight": (
                args.recurrent_bc_signal_constraint_message_loss_weight
            ),
            "recurrent_bc_signal_target_aux_weight": (
                args.recurrent_bc_signal_target_aux_weight
            ),
            "recurrent_bc_signal_target_hypothesis_loss_weight": (
                args.recurrent_bc_signal_target_hypothesis_loss_weight
            ),
            "recurrent_bc_signal_target_hypothesis_commit_loss_weight": (
                args.recurrent_bc_signal_target_hypothesis_commit_loss_weight
            ),
            "recurrent_bc_signal_target_hypothesis_ambiguity_loss_weight": (
                args.recurrent_bc_signal_target_hypothesis_ambiguity_loss_weight
            ),
            "recurrent_bc_signal_target_hypothesis_xy_loss_weight": (
                args.recurrent_bc_signal_target_hypothesis_xy_loss_weight
            ),
            "recurrent_bc_signal_target_hypothesis_min_map_size": (
                args.recurrent_bc_signal_target_hypothesis_min_map_size
            ),
            "recurrent_bc_signal_target_pursuit_action_weight": (
                args.recurrent_bc_signal_target_pursuit_action_weight
            ),
            "recurrent_bc_signal_target_pursuit_trust_exact_memory": (
                bool(args.recurrent_bc_signal_target_pursuit_trust_exact_memory)
            ),
            "recurrent_bc_signal_target_pursuit_max_agents": (
                args.recurrent_bc_signal_target_pursuit_max_agents
            ),
            "recurrent_bc_signal_sync_response_action_loss_weight": (
                args.recurrent_bc_signal_sync_response_action_loss_weight
            ),
            "recurrent_bc_signal_active_scan_response_action_weight": (
                args.recurrent_bc_signal_active_scan_response_action_weight
            ),
            "recurrent_bc_signal_active_scan_response_min_map_size": (
                args.recurrent_bc_signal_active_scan_response_min_map_size
            ),
            "recurrent_bc_signal_active_scan_response_max_agents": (
                args.recurrent_bc_signal_active_scan_response_max_agents
            ),
            "recurrent_bc_signal_scan_bridge_action_weight": (
                args.recurrent_bc_signal_scan_bridge_action_weight
            ),
            "recurrent_bc_signal_scan_bridge_min_map_size": (
                args.recurrent_bc_signal_scan_bridge_min_map_size
            ),
            "recurrent_bc_signal_scan_bridge_remaining_threshold": (
                args.recurrent_bc_signal_scan_bridge_remaining_threshold
            ),
            "recurrent_bc_signal_scan_bridge_max_teammate_distance": (
                args.recurrent_bc_signal_scan_bridge_max_teammate_distance
            ),
            "recurrent_bc_signal_target_match_action_weight": (
                args.recurrent_bc_signal_target_match_action_weight
            ),
            "recurrent_bc_signal_first_target_scan_action_weight": (
                args.recurrent_bc_signal_first_target_scan_action_weight
            ),
            "recurrent_bc_signal_refresh_target_scan_action_weight": (
                args.recurrent_bc_signal_refresh_target_scan_action_weight
            ),
            "recurrent_bc_signal_joint_target_scan_action_weight": (
                args.recurrent_bc_signal_joint_target_scan_action_weight
            ),
            "recurrent_bc_signal_target_opportunity_action_weight": (
                args.recurrent_bc_signal_target_opportunity_action_weight
            ),
            "recurrent_bc_signal_redundant_target_wait_action_loss_weight": (
                args.recurrent_bc_signal_redundant_target_wait_action_loss_weight
            ),
            "recurrent_bc_signal_scan_decision_loss_weight": (
                args.recurrent_bc_signal_scan_decision_loss_weight
            ),
            "recurrent_bc_signal_scan_decision_pos_weight": (
                args.recurrent_bc_signal_scan_decision_pos_weight
            ),
            "recurrent_bc_signal_scan_decision_neg_weight": (
                args.recurrent_bc_signal_scan_decision_neg_weight
            ),
            "recurrent_bc_signal_scan_gate_loss_weight": (
                args.recurrent_bc_signal_scan_gate_loss_weight
            ),
            "recurrent_bc_signal_scan_gate_pos_weight": (
                args.recurrent_bc_signal_scan_gate_pos_weight
            ),
            "recurrent_bc_signal_scan_gate_neg_weight": (
                args.recurrent_bc_signal_scan_gate_neg_weight
            ),
            "recurrent_bc_signal_target_validity_loss_weight": (
                args.recurrent_bc_signal_target_validity_loss_weight
            ),
            "recurrent_bc_signal_target_validity_pos_weight": (
                args.recurrent_bc_signal_target_validity_pos_weight
            ),
            "recurrent_bc_signal_target_validity_neg_weight": (
                args.recurrent_bc_signal_target_validity_neg_weight
            ),
            "recurrent_bc_signal_target_decision_loss_weight": (
                args.recurrent_bc_signal_target_decision_loss_weight
            ),
            "recurrent_bc_signal_target_decision_pos_weight": (
                args.recurrent_bc_signal_target_decision_pos_weight
            ),
            "recurrent_bc_signal_target_decision_neg_weight": (
                args.recurrent_bc_signal_target_decision_neg_weight
            ),
            "recurrent_bc_signal_ambiguous_target_decision_negatives": (
                bool(args.recurrent_bc_signal_ambiguous_target_decision_negatives)
            ),
            "recurrent_bc_signal_ambiguous_target_decision_min_map_size": (
                args.recurrent_bc_signal_ambiguous_target_decision_min_map_size
            ),
            "recurrent_bc_signal_ambiguous_target_search_labels": (
                bool(args.recurrent_bc_signal_ambiguous_target_search_labels)
            ),
            "recurrent_bc_signal_ambiguous_target_search_min_map_size": (
                args.recurrent_bc_signal_ambiguous_target_search_min_map_size
            ),
            "recurrent_bc_signal_constraint_frontier_bias": (
                bool(args.recurrent_bc_signal_constraint_frontier_bias)
            ),
            "recurrent_bc_signal_decoy_drift_action_loss_weight": (
                args.recurrent_bc_signal_decoy_drift_action_loss_weight
            ),
            "recurrent_bc_signal_decoy_scan_action_loss_weight": (
                args.recurrent_bc_signal_decoy_scan_action_loss_weight
            ),
            "recurrent_bc_signal_rejected_target_drift_action_loss_weight": (
                args.recurrent_bc_signal_rejected_target_drift_action_loss_weight
            ),
            "recurrent_bc_signal_clue_interact_action_weight": (
                args.recurrent_bc_signal_clue_interact_action_weight
            ),
            "recurrent_bc_signal_clue_interact_min_map_size": (
                args.recurrent_bc_signal_clue_interact_min_map_size
            ),
            "recurrent_bc_signal_visible_clue_action_weight": (
                args.recurrent_bc_signal_visible_clue_action_weight
            ),
            "recurrent_bc_signal_visible_clue_min_map_size": (
                args.recurrent_bc_signal_visible_clue_min_map_size
            ),
            "recurrent_bc_signal_evidence_sweep_action_weight": (
                args.recurrent_bc_signal_evidence_sweep_action_weight
            ),
            "recurrent_bc_signal_evidence_sweep_min_map_size": (
                args.recurrent_bc_signal_evidence_sweep_min_map_size
            ),
            "recurrent_bc_signal_frontier_exploration_action_weight": (
                args.recurrent_bc_signal_frontier_exploration_action_weight
            ),
            "recurrent_bc_signal_frontier_exploration_min_map_size": (
                args.recurrent_bc_signal_frontier_exploration_min_map_size
            ),
            "recurrent_dagger_rounds": args.recurrent_dagger_rounds,
            "recurrent_dagger_episodes": args.recurrent_dagger_episodes,
            "recurrent_dagger_failed_effective_ratio_cap": args.recurrent_dagger_failed_effective_ratio_cap,
            "recurrent_dagger_oracle_action_rollin_rate": args.recurrent_dagger_oracle_action_rollin_rate,
            "recurrent_dagger_oracle_message_rollin_rate": args.recurrent_dagger_oracle_message_rollin_rate,
            "recurrent_dagger_initial_target_broadcast_labels": (
                args.recurrent_dagger_initial_target_broadcast_labels
            ),
            "recurrent_dagger_target_handoff_requires_exact_target": (
                args.recurrent_dagger_target_handoff_requires_exact_target
            ),
            "recurrent_dagger_signal_target_rendezvous_labels": bool(
                args.recurrent_dagger_signal_target_rendezvous_labels
            ),
            "recurrent_dagger_signal_target_rendezvous_min_map_size": (
                args.recurrent_dagger_signal_target_rendezvous_min_map_size
            ),
            "recurrent_dagger_signal_target_rendezvous_max_agents": (
                args.recurrent_dagger_signal_target_rendezvous_max_agents
            ),
            "recurrent_dagger_focus_events": args.recurrent_dagger_focus_events,
            "recurrent_dagger_focus_error_weight": args.recurrent_dagger_focus_error_weight,
            "recurrent_dagger_focus_recovery_weight": args.recurrent_dagger_focus_recovery_weight,
            "recurrent_dagger_focus_window": args.recurrent_dagger_focus_window,
            "recurrent_dagger_target_interact_focus_weight": (
                args.recurrent_dagger_target_interact_focus_weight
            ),
            "recurrent_dagger_target_discovery_min_map_size": (
                args.recurrent_dagger_target_discovery_min_map_size
            ),
            "recurrent_dagger_target_discovery_focus_weight": (
                args.recurrent_dagger_target_discovery_focus_weight
            ),
            "recurrent_dagger_movement_stall_min_map_size": (
                args.recurrent_dagger_movement_stall_min_map_size
            ),
            "recurrent_dagger_movement_stall_window": (
                args.recurrent_dagger_movement_stall_window
            ),
            "recurrent_dagger_movement_stall_focus_weight": (
                args.recurrent_dagger_movement_stall_focus_weight
            ),
            "recurrent_dagger_target_decoy_drift_focus_weight": (
                args.recurrent_dagger_target_decoy_drift_focus_weight
            ),
            "recurrent_dagger_focus_replay": args.recurrent_dagger_focus_replay,
            "recurrent_dagger_retrain_from_scratch": args.recurrent_dagger_retrain_from_scratch,
            "recurrent_dagger_restore_best": args.recurrent_dagger_restore_best,
            "recurrent_dagger_pipeline_wrong_delivery_provenance_labels": (
                args.recurrent_dagger_pipeline_wrong_delivery_provenance_labels
            ),
            "recurrent_dagger_pipeline_wrong_delivery_provenance_weight": (
                args.recurrent_dagger_pipeline_wrong_delivery_provenance_weight
            ),
            "recurrent_dagger_replay_pre_steps": args.recurrent_dagger_replay_pre_steps,
            "recurrent_dagger_replay_post_steps": args.recurrent_dagger_replay_post_steps,
            "recurrent_dagger_replay_weight": args.recurrent_dagger_replay_weight,
            "recurrent_dagger_positive_replay_events": args.recurrent_dagger_positive_replay_events,
            "recurrent_dagger_replay_event_weights": args.recurrent_dagger_replay_event_weights,
            "recurrent_dagger_replay_event_caps": args.recurrent_dagger_replay_event_caps,
            "recurrent_dagger_replay_success_only_events": (
                args.recurrent_dagger_replay_success_only_events
            ),
            "recurrent_dagger_replay_priority_events": args.recurrent_dagger_replay_priority_events,
            "recurrent_dagger_replay_balance_positive_events": (
                args.recurrent_dagger_replay_balance_positive_events
            ),
            "recurrent_dagger_replay_balance_negative_events": (
                args.recurrent_dagger_replay_balance_negative_events
            ),
            "recurrent_dagger_replay_max_negative_per_positive": (
                args.recurrent_dagger_replay_max_negative_per_positive
            ),
            "recurrent_dagger_max_replay_snippets_per_episode": (
                args.recurrent_dagger_max_replay_snippets_per_episode
            ),
            "recurrent_dagger_max_failed_parent_replay_snippets_per_episode": (
                args.recurrent_dagger_max_failed_parent_replay_snippets_per_episode
            ),
            "recurrent_dagger_failed_parent_replay_weight_scale": (
                args.recurrent_dagger_failed_parent_replay_weight_scale
            ),
            "recurrent_dagger_expert_max_replay_snippets_per_episode": (
                args.recurrent_dagger_expert_max_replay_snippets_per_episode
            ),
            "recurrent_pipeline_assisted_rollout_episodes": (
                args.recurrent_pipeline_assisted_rollout_episodes
            ),
            "recurrent_pipeline_assisted_rollout_seed_base": (
                args.recurrent_pipeline_assisted_rollout_seed_base
            ),
            "recurrent_pipeline_assisted_rollout_seed_list": (
                args.recurrent_pipeline_assisted_rollout_seed_list
            ),
            "recurrent_pipeline_assisted_rollout_max_steps_per_episode": (
                args.recurrent_pipeline_assisted_rollout_max_steps_per_episode
            ),
            "recurrent_pipeline_assisted_rollout_weight": (
                args.recurrent_pipeline_assisted_rollout_weight
            ),
            "recurrent_pipeline_assisted_rollout_success_only": (
                args.recurrent_pipeline_assisted_rollout_success_only
            ),
            "recurrent_pipeline_assisted_rollout_navigation_assist": (
                args.recurrent_pipeline_assisted_rollout_navigation_assist
            ),
            "recurrent_pipeline_assisted_rollout_navigation_assist_trust_messages": (
                args.recurrent_pipeline_assisted_rollout_navigation_assist_trust_messages
            ),
            "recurrent_pipeline_assisted_rollout_station_interact_guard": (
                args.recurrent_pipeline_assisted_rollout_station_interact_guard
            ),
            "recurrent_pipeline_assisted_rollout_bc_epochs": (
                args.recurrent_pipeline_assisted_rollout_bc_epochs
            ),
            "recurrent_rl_updates": args.recurrent_rl_updates,
            "recurrent_rl_early_stop_eval_patience": args.recurrent_rl_early_stop_eval_patience,
            "recurrent_rl_eval_episodes": args.recurrent_rl_eval_episodes,
            "recurrent_rl_eval_use_eval_seeds": args.recurrent_rl_eval_use_eval_seeds,
            "recurrent_ppo_profile": args.recurrent_ppo_profile,
            "recurrent_rl_epochs": args.recurrent_rl_epochs,
            "recurrent_minibatch_seqs": args.recurrent_minibatch_seqs,
            "recurrent_rl_lr": args.recurrent_rl_lr,
            "recurrent_clip": args.recurrent_clip,
            "recurrent_entropy_coeff": args.recurrent_entropy_coeff,
            "recurrent_max_grad_norm": args.recurrent_max_grad_norm,
            "recurrent_bc_kl_coeff": args.recurrent_bc_kl_coeff,
            "recurrent_bc_comm_kl_coeff": args.recurrent_bc_comm_kl_coeff,
            "recurrent_rl_balanced_rollouts": args.recurrent_rl_balanced_rollouts,
            "recurrent_rl_rollout_eval_decoding": args.recurrent_rl_rollout_eval_decoding,
            "recurrent_rl_rollout_pipeline_navigation_assist": (
                args.recurrent_rl_rollout_pipeline_navigation_assist
            ),
            "recurrent_rl_rollout_pipeline_navigation_assist_trust_messages": (
                args.recurrent_rl_rollout_pipeline_navigation_assist_trust_messages
            ),
            "recurrent_rl_rollout_pipeline_station_interact_guard": (
                args.recurrent_rl_rollout_pipeline_station_interact_guard
            ),
            "recurrent_rl_rollout_pipeline_interact_gate_promote": (
                args.recurrent_rl_rollout_pipeline_interact_gate_promote
            ),
            "recurrent_rl_eval_decoding_action_loss_weight": (
                args.recurrent_rl_eval_decoding_action_loss_weight
            ),
            "recurrent_rl_pipeline_assisted_action_loss_weight": (
                args.recurrent_rl_pipeline_assisted_action_loss_weight
            ),
            "recurrent_rl_pipeline_interact_gate_loss_weight": (
                args.recurrent_rl_pipeline_interact_gate_loss_weight
            ),
            "recurrent_rl_pipeline_interact_gate_pos_weight": (
                args.recurrent_rl_pipeline_interact_gate_pos_weight
            ),
            "recurrent_rl_pipeline_interact_gate_neg_weight": (
                args.recurrent_rl_pipeline_interact_gate_neg_weight
            ),
            "recurrent_rl_pipeline_pickup_gate_loss_weight": (
                args.recurrent_rl_pipeline_pickup_gate_loss_weight
            ),
            "recurrent_rl_pipeline_pickup_gate_pos_weight": (
                args.recurrent_rl_pipeline_pickup_gate_pos_weight
            ),
            "recurrent_rl_pipeline_pickup_gate_neg_weight": (
                args.recurrent_rl_pipeline_pickup_gate_neg_weight
            ),
            "recurrent_rl_pipeline_delivery_progress_action_loss_weight": (
                args.recurrent_rl_pipeline_delivery_progress_action_loss_weight
            ),
            "recurrent_rl_pipeline_navigation_action_loss_weight": (
                args.recurrent_rl_pipeline_navigation_action_loss_weight
            ),
            "recurrent_rl_pipeline_sync_action_loss_weight": (
                args.recurrent_rl_pipeline_sync_action_loss_weight
            ),
            "recurrent_rl_pipeline_ready_interact_action_loss_weight": (
                args.recurrent_rl_pipeline_ready_interact_action_loss_weight
            ),
            "recurrent_rl_pipeline_station_guard_action_loss_weight": (
                args.recurrent_rl_pipeline_station_guard_action_loss_weight
            ),
            "recurrent_rl_pipeline_wrong_station_recovery_action_loss_weight": (
                args.recurrent_rl_pipeline_wrong_station_recovery_action_loss_weight
            ),
            "recurrent_rl_pipeline_plan_action_loss_weight": (
                args.recurrent_rl_pipeline_plan_action_loss_weight
            ),
            "recurrent_rl_pipeline_plan_head_loss_weight": (
                args.recurrent_rl_pipeline_plan_head_loss_weight
            ),
            "recurrent_rl_pipeline_option_loss_weight": (
                args.recurrent_rl_pipeline_option_loss_weight
            ),
            "recurrent_rl_pipeline_bad_pickup_penalty": args.recurrent_rl_pipeline_bad_pickup_penalty,
            "recurrent_rl_pipeline_bad_interact_penalty": args.recurrent_rl_pipeline_bad_interact_penalty,
            "recurrent_rl_pipeline_unneeded_drop_bonus": args.recurrent_rl_pipeline_unneeded_drop_bonus,
            "recurrent_rl_restore_best": args.recurrent_rl_restore_best,
            "recurrent_rl_save_best": args.recurrent_rl_save_best,
            "recurrent_train_map_sizes": args.recurrent_train_map_sizes,
            "recurrent_train_map_sampling_weights": args.recurrent_train_map_sampling_weights,
            "recurrent_map_max_steps": args.recurrent_map_max_steps,
            "recurrent_eval_map_sizes": args.recurrent_eval_map_sizes,
            "recurrent_eval_seed_count": args.recurrent_eval_seed_count,
            "recurrent_eval_seed_range": args.recurrent_eval_seed_range,
            "recurrent_eval_seed_list": args.recurrent_eval_seed_list,
            "recurrent_dagger_seed_list": args.recurrent_dagger_seed_list,
            "recurrent_skip_bc": args.recurrent_skip_bc,
            "recurrent_init": args.recurrent_init,
            "recurrent_init_template": args.recurrent_init_template,
            "recurrent_init_for_dagger": args.recurrent_init_for_dagger,
            "recurrent_init_allow_obs_dim_mismatch": args.recurrent_init_allow_obs_dim_mismatch,
            "recurrent_comm": args.recurrent_comm,
            "recurrent_comm_token_limit": args.recurrent_comm_token_limit,
            "recurrent_comm_vocab_size": args.recurrent_comm_vocab_size,
            "recurrent_hidden_dim": args.recurrent_hidden_dim,
            "recurrent_backbone": args.recurrent_backbone,
            "recurrent_obs_exploration_memory": args.recurrent_obs_exploration_memory,
            "recurrent_obs_exploration_age": args.recurrent_obs_exploration_age,
            "recurrent_obs_feedback": args.recurrent_obs_feedback,
            "recurrent_obs_normalize_tokens": args.recurrent_obs_normalize_tokens,
            "recurrent_obs_navigation_features": args.recurrent_obs_navigation_features,
            "recurrent_obs_signal_features": args.recurrent_obs_signal_features,
            "recurrent_obs_signal_target_match_features": (
                args.recurrent_obs_signal_target_match_features
            ),
            "recurrent_obs_signal_confidence_features": (
                args.recurrent_obs_signal_confidence_features
            ),
            "recurrent_obs_signal_sector_features": (
                args.recurrent_obs_signal_sector_features
            ),
            "recurrent_obs_signal_sync_feedback": args.recurrent_obs_signal_sync_feedback,
            "recurrent_obs_signal_scan_state": args.recurrent_obs_signal_scan_state,
            "recurrent_eval_send_threshold": args.recurrent_eval_send_threshold,
            "recurrent_calibrate_send_threshold": args.recurrent_calibrate_send_threshold,
            "recurrent_obs_memory_mode": args.recurrent_obs_memory_mode,
            "recurrent_obs_memory_radius": args.recurrent_obs_memory_radius,
            "recurrent_eval_signal_exact_target_memory_steps": (
                args.recurrent_eval_signal_exact_target_memory_steps
            ),
            "recurrent_eval_signal_target_scan_lock": (
                args.recurrent_eval_signal_target_scan_lock
            ),
            "recurrent_eval_signal_exact_target_scan_lock": (
                args.recurrent_eval_signal_exact_target_scan_lock
            ),
            "recurrent_eval_signal_compatible_target_scan_assist": (
                bool(args.recurrent_eval_signal_compatible_target_scan_assist)
            ),
            "recurrent_eval_signal_compatible_target_scan_min_strength": (
                args.recurrent_eval_signal_compatible_target_scan_min_strength
            ),
            "recurrent_eval_signal_negative_memory_scan_guard": (
                bool(args.recurrent_eval_signal_negative_memory_scan_guard)
            ),
            "recurrent_eval_signal_target_probe_assist": (
                bool(args.recurrent_eval_signal_target_probe_assist)
            ),
            "recurrent_eval_signal_evidence_sweep_assist": (
                bool(args.recurrent_eval_signal_evidence_sweep_assist)
                if args.recurrent_eval_signal_evidence_sweep_assist is not None
                else args.recurrent_signal_preset == "large_map"
            ),
            "recurrent_eval_signal_evidence_sweep_min_step": (
                int(args.recurrent_eval_signal_evidence_sweep_min_step)
            ),
            "recurrent_eval_signal_frontier_exploration_assist": (
                bool(args.recurrent_eval_signal_frontier_exploration_assist)
            ),
            "recurrent_eval_signal_scan_refresh_assist": (
                bool(args.recurrent_eval_signal_scan_refresh_assist)
            ),
            "recurrent_eval_signal_scan_refresh_threshold": (
                args.recurrent_eval_signal_scan_refresh_threshold
            ),
            "recurrent_eval_signal_initial_exact_message_copy_assist": (
                args.recurrent_eval_signal_initial_exact_message_copy_assist
            ),
            "recurrent_eval_signal_exact_target_message_copy_assist": (
                bool(args.recurrent_eval_signal_exact_target_message_copy_assist)
                if args.recurrent_eval_signal_exact_target_message_copy_assist is not None
                else args.recurrent_signal_preset == "large_map"
            ),
            "recurrent_eval_signal_constraint_message_copy_assist": (
                args.recurrent_eval_signal_constraint_message_copy_assist
            ),
            "recurrent_eval_signal_constraint_message_guard": (
                bool(args.recurrent_eval_signal_constraint_message_guard)
                if args.recurrent_eval_signal_constraint_message_guard is not None
                else args.recurrent_signal_preset == "large_map"
            ),
        },
        "cases": [asdict(case) for case in cases],
        "runs": [asdict(record) for record in records],
        "aggregate": _aggregate_records(records),
        "overall": {
            "total": len(records),
            "complete": sum(record.status == "complete" for record in records),
            "failed": sum(record.status == "failed" for record in records),
            "dry_run": sum(record.status == "dry_run" for record in records),
        },
    }
    _write_json(suite_dir / "suite_summary.json", payload)
    if args.output_json:
        _write_json(Path(args.output_json), payload)
    if payload["overall"]["failed"] > 0:
        return payload
    return payload


def _run_one(
    *,
    cmd: list[str],
    algorithm: str,
    case: ScenarioCase,
    seed: int,
    run_dir: Path,
    checkpoint_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    dry_run: bool,
    wandb_mode: str,
    strict_wandb: bool,
) -> RunRecord:
    start = time.time()
    wandb_requested = "--wandb" in cmd
    best_checkpoint_path = _best_checkpoint_path(algorithm, checkpoint_path)
    if dry_run:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return RunRecord(
            algorithm=algorithm,
            scenario=case.scenario,
            seed=seed,
            run_dir=str(run_dir),
            checkpoint_path=str(checkpoint_path),
            command=cmd,
            status="dry_run",
            returncode=None,
            elapsed_sec=0.0,
            checkpoint_exists=checkpoint_path.exists(),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            stdout_tail=[],
            stderr_tail=[],
            eval_metrics=None,
            final_eval_metrics=None,
            best_eval_metrics=None,
            best_checkpoint_path=str(best_checkpoint_path) if best_checkpoint_path is not None else None,
            best_checkpoint_exists=best_checkpoint_path.exists() if best_checkpoint_path is not None else False,
            wandb=_wandb_record(requested=wandb_requested, mode=wandb_mode, status="dry_run", error_lines=[]),
        )

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = wandb_mode
    env.setdefault("WANDB_SILENT", "true")
    wandb_dir = _prepare_wandb_dirs(run_dir)
    env["WANDB_DIR"] = str(wandb_dir)
    env["WANDB_DATA_DIR"] = str(wandb_dir / "data")
    env["WANDB_ARTIFACT_DIR"] = str(wandb_dir / "artifacts")
    env["WANDB_CACHE_DIR"] = str(wandb_dir / "cache")
    env["WANDB_CONFIG_DIR"] = str(wandb_dir / "config")
    env["TMPDIR"] = str(wandb_dir / "tmp")

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, check=False)

    stdout_tail = _tail_lines(stdout_path)
    stderr_tail = _tail_lines(stderr_path)
    wandb = _parse_wandb_record(
        stdout_path,
        stderr_path,
        requested=wandb_requested,
        mode=wandb_mode,
        run_dir=run_dir,
    )
    status = "complete" if proc.returncode == 0 and checkpoint_path.exists() else "failed"
    if strict_wandb and wandb.get("status") == "failed":
        status = "failed"
    eval_metrics = _parse_eval_metrics(
        algorithm,
        stdout_path,
        stdout_tail,
        checkpoint_path=checkpoint_path,
    )
    final_eval_metrics = None
    best_eval_metrics = None
    if algorithm == "recurrent_bc_rl":
        recurrent_evals = _parse_recurrent_checkpoint_evals(checkpoint_path)
        final_eval_metrics = recurrent_evals.get("final_eval")
        best_eval_metrics = recurrent_evals.get("best_eval")
    return RunRecord(
        algorithm=algorithm,
        scenario=case.scenario,
        seed=seed,
        run_dir=str(run_dir),
        checkpoint_path=str(checkpoint_path),
        command=cmd,
        status=status,
        returncode=proc.returncode,
        elapsed_sec=round(time.time() - start, 3),
        checkpoint_exists=checkpoint_path.exists(),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        eval_metrics=eval_metrics,
        final_eval_metrics=final_eval_metrics,
        best_eval_metrics=best_eval_metrics,
        best_checkpoint_path=str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        best_checkpoint_exists=best_checkpoint_path.exists() if best_checkpoint_path is not None else False,
        wandb=wandb,
    )


def _parse_eval_metrics(
    algorithm: str,
    stdout_path: Path,
    stdout_tail: Iterable[str],
    *,
    checkpoint_path: Path | None = None,
) -> dict[str, float] | None:
    if algorithm == "recurrent_bc_rl":
        checkpoint_metrics = _parse_recurrent_checkpoint_eval(checkpoint_path)
        if checkpoint_metrics is not None:
            return checkpoint_metrics
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        return _parse_recurrent_stdout_eval(stdout)
    return _parse_last_eval(stdout_tail)


def _parse_last_eval(lines: Iterable[str]) -> dict[str, float] | None:
    for line in reversed(list(lines)):
        if "eval |" not in line:
            continue
        metrics: dict[str, float] = {}
        for part in line.split("|"):
            part = part.strip()
            if part.startswith("ret "):
                metrics["return"] = float(part.split()[1])
            elif part.startswith("steps "):
                metrics["steps"] = float(part.split()[1])
            elif part.startswith("success "):
                metrics["success_rate"] = float(part.split()[1])
        return metrics or None
    return None


def _parse_recurrent_checkpoint_eval(checkpoint_path: Path | None) -> dict[str, float] | None:
    evals, restored_best = _parse_recurrent_checkpoint_eval_data(checkpoint_path)
    if restored_best:
        metrics = evals.get("best_eval")
        if metrics is not None:
            return metrics
    for key in ("eval_recurrent_policy", "final_eval", "best_eval"):
        metrics = evals.get(key)
        if metrics is not None:
            return metrics
    return None


def _parse_recurrent_checkpoint_evals(checkpoint_path: Path | None) -> dict[str, dict[str, float]]:
    evals, _ = _parse_recurrent_checkpoint_eval_data(checkpoint_path)
    return evals


def _parse_recurrent_checkpoint_eval_data(
    checkpoint_path: Path | None,
) -> tuple[dict[str, dict[str, float]], bool]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}, False
    try:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return {}, False
    if not isinstance(ckpt, dict):
        return {}, False
    evals: dict[str, dict[str, float]] = {}
    for key in ("eval_recurrent_policy", "final_eval", "best_eval"):
        metrics = _metrics_from_recurrent_eval(ckpt.get(key))
        if metrics is not None:
            evals[key] = metrics
    return evals, bool(ckpt.get("restored_best", False))


def _parse_recurrent_stdout_eval(stdout: str) -> dict[str, float] | None:
    latest: dict[str, float] | None = None
    for obj in _iter_json_objects(stdout):
        if not isinstance(obj, dict):
            continue
        for key in ("eval_recurrent_bc", "eval_recurrent_init"):
            metrics = _metrics_from_recurrent_eval(obj.get(key))
            if metrics is not None:
                latest = metrics
        for key in ("recurrent_dagger", "recurrent_dagger_initial"):
            row = obj.get(key)
            if isinstance(row, dict):
                metrics = _metrics_from_recurrent_eval(row.get("eval"))
                if metrics is not None:
                    latest = metrics
        metrics = _metrics_from_recurrent_eval(obj.get("recurrent_rl_eval"))
        if metrics is not None:
            latest = metrics
    return latest


def _iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            return
        try:
            obj, end = decoder.raw_decode(text[start:])
        except JSONDecodeError:
            idx = start + 1
            continue
        yield obj
        idx = start + end


def _metrics_from_recurrent_eval(eval_result) -> dict[str, float] | None:
    if not isinstance(eval_result, dict):
        return None
    try:
        return {
            "success_rate": float(eval_result["success_rate"]),
            "return": float(eval_result["avg_return"]),
            "steps": float(eval_result["avg_steps"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _aggregate_records(records: list[RunRecord]) -> list[dict]:
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.algorithm, record.scenario), []).append(record)

    aggregate = []
    for (algorithm, scenario), group in sorted(groups.items()):
        eval_records = [record.eval_metrics for record in group if record.eval_metrics is not None]
        final_eval_records = [
            record.final_eval_metrics for record in group if record.final_eval_metrics is not None
        ]
        best_eval_records = [
            record.best_eval_metrics for record in group if record.best_eval_metrics is not None
        ]
        aggregate.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "seeds": sorted(record.seed for record in group),
                "runs": len(group),
                "complete": sum(record.status == "complete" for record in group),
                "failed": sum(record.status == "failed" for record in group),
                "dry_run": sum(record.status == "dry_run" for record in group),
                "checkpoint_count": sum(record.checkpoint_exists for record in group),
                "wandb_requested": sum(bool(record.wandb.get("requested")) for record in group),
                "wandb_failed": sum(record.wandb.get("status") == "failed" for record in group),
                "mean_eval_success_rate": _mean_metric(eval_records, "success_rate"),
                "mean_eval_return": _mean_metric(eval_records, "return"),
                "mean_eval_steps": _mean_metric(eval_records, "steps"),
                "mean_final_eval_success_rate": _mean_metric(final_eval_records, "success_rate"),
                "mean_final_eval_return": _mean_metric(final_eval_records, "return"),
                "mean_final_eval_steps": _mean_metric(final_eval_records, "steps"),
                "mean_best_eval_success_rate": _mean_metric(best_eval_records, "success_rate"),
                "mean_best_eval_return": _mean_metric(best_eval_records, "return"),
                "mean_best_eval_steps": _mean_metric(best_eval_records, "steps"),
            }
        )
    return aggregate


def _mean_metric(metrics: list[dict[str, float]], key: str) -> float | None:
    values = [metric[key] for metric in metrics if key in metric]
    if not values:
        return None
    return float(sum(values) / len(values))


def _prepare_wandb_dirs(run_dir: Path) -> Path:
    wandb_dir = run_dir / "wandb"
    for name in ("data", "artifacts", "cache", "config", "tmp"):
        (wandb_dir / name).mkdir(parents=True, exist_ok=True)
    return wandb_dir


def _best_checkpoint_path(algorithm: str, checkpoint_path: Path) -> Path | None:
    if algorithm != "recurrent_bc_rl":
        return None
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")


def _parse_wandb_record(
    stdout_path: Path,
    stderr_path: Path,
    *,
    requested: bool,
    mode: str,
    run_dir: Path | None = None,
) -> dict:
    if not requested:
        return _wandb_record(requested=False, mode=mode, status="not_requested", error_lines=[])
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    lines = [line for line in [*stdout.splitlines(), *stderr.splitlines()] if "wandb" in line.lower()]
    failed = [
        line
        for line in lines
        if "wandb init failed" in line.lower()
        or "wandb-core exited" in line.lower()
        or "serve() returned error" in line.lower()
        or "wandb log failed, disabling wandb" in line.lower()
        or "wandb scalar retry failed" in line.lower()
    ]
    if failed:
        return _wandb_record(requested=True, mode=mode, status="failed", error_lines=failed[-5:])
    if mode == "disabled":
        status = "disabled"
    else:
        status = "initialized"
    return _wandb_record(
        requested=True,
        mode=mode,
        status=status,
        error_lines=[],
        **_find_wandb_run_info(run_dir),
    )


def _wandb_record(
    *,
    requested: bool,
    mode: str,
    status: str,
    error_lines: list[str],
    **extra,
) -> dict:
    record = {
        "requested": requested,
        "mode": mode,
        "status": status,
        "error_lines": error_lines,
    }
    record.update({key: value for key, value in extra.items() if value is not None})
    return record


def _find_wandb_run_info(run_dir: Path | None) -> dict[str, str]:
    if run_dir is None:
        return {}
    wandb_root = Path(run_dir) / "wandb" / "wandb"
    if not wandb_root.exists():
        return {}
    run_dirs = sorted(path for path in wandb_root.glob("run-*-*") if path.is_dir())
    if not run_dirs:
        return {}
    run_dir_path = run_dirs[-1]
    run_id = run_dir_path.name.rsplit("-", 1)[-1]
    info = {
        "run_id": run_id,
        "local_run_dir": str(run_dir_path),
    }
    debug_log = run_dir_path / "logs" / "debug.log"
    if debug_log.exists():
        text = debug_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"finishing run ([^\s/]+)/([^\s/]+)/([^\s]+)", text)
        if match:
            entity, project, parsed_run_id = match.groups()
            info["run_id"] = parsed_run_id
            info["run_path"] = f"{entity}/{project}/{parsed_run_id}"
            info["url"] = f"https://wandb.ai/{entity}/{project}/runs/{parsed_run_id}"
    return info


def _tail_lines(path: Path, limit: int = 20) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]


def _suite_dir(args) -> Path:
    if args.run_name:
        name = args.run_name
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"core_training_{stamp}"
    return Path(args.output_dir) / name


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_record(record: RunRecord) -> str:
    metrics = ""
    if record.eval_metrics:
        metrics = (
            f" eval_success={record.eval_metrics.get('success_rate', 0.0):.2f}"
            f" eval_return={record.eval_metrics.get('return', 0.0):.2f}"
        )
    if record.final_eval_metrics and record.final_eval_metrics != record.eval_metrics:
        metrics += f" final_success={record.final_eval_metrics.get('success_rate', 0.0):.2f}"
    if record.best_eval_metrics and record.best_eval_metrics != record.eval_metrics:
        metrics += f" best_success={record.best_eval_metrics.get('success_rate', 0.0):.2f}"
    return (
        f"{record.status:8s} {record.algorithm:8s} {record.scenario:18s} seed={record.seed:<3d} "
        f"elapsed={record.elapsed_sec:.1f}s ckpt={int(record.checkpoint_exists)}{metrics}"
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run core learned-policy training sweeps.")
    parser.add_argument("--algorithms", nargs="+", default=["mappo", "comm_mat", "tarmac"], choices=sorted(TRAIN_SCRIPTS))
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_CASES), choices=sorted(DEFAULT_CASES))
    parser.add_argument(
        "--benchmark-spec",
        default="",
        help="Optional benchmark manifest path; when set, cases are read from the manifest instead of --scenarios",
    )
    parser.add_argument(
        "--benchmark-cases",
        nargs="+",
        default=None,
        help="Subset of benchmark case names to run from --benchmark-spec; defaults to all manifest cases",
    )
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=None, help="Single-seed alias for --seeds")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--learning-profile",
        default="bare",
        choices=["bare", "shaped", "comm_curriculum"],
        help="Training aids to apply before benchmark evaluation; bare leaves trainers unchanged",
    )
    parser.add_argument("--mappo-critic-mode", default="central", choices=["local", "central"])
    parser.add_argument("--mappo-shared-actor", action="store_true")
    parser.add_argument("--mappo-backbone", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--mappo-obs-exploration-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mappo-obs-exploration-age", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mappo-eval-action-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--mappo-eval-action-temperature", type=float, default=1.0)
    parser.add_argument("--mappo-eval-send-mode", default="threshold", choices=["threshold", "sample"])
    parser.add_argument("--mappo-eval-send-threshold", type=float, default=0.5)
    parser.add_argument("--mappo-eval-token-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--mappo-eval-token-temperature", type=float, default=1.0)
    parser.add_argument("--mappo-eval-length-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--mappo-eval-length-temperature", type=float, default=1.0)
    parser.add_argument(
        "--recurrent-oracle",
        default="auto",
        choices=[
            "auto",
            "oracle_strong",
            "oracle_strong_comm",
            "signal_hint_comm",
            "planner_comm",
            "energy_planner_comm",
            "pipeline_planner_comm",
            "signal_hunt_planner_comm",
        ],
        help=(
            "Oracle used for recurrent demos. auto uses signal_hint_comm for Signal Hunt "
            "and scenario planner communication teachers for Energy Grid/Pipeline."
        ),
    )
    parser.add_argument(
        "--recurrent-signal-preset",
        default="specialist",
        choices=["minimal", "specialist", "large_map"],
        help=(
            "Signal Hunt recurrent defaults. large_map inherits specialist and adds "
            "32x32-oriented clue-discovery and decoy-suppression training signals."
        ),
    )
    parser.add_argument("--recurrent-demo-episodes", type=int, default=20)
    parser.add_argument("--recurrent-bc-epochs", type=int, default=1)
    parser.add_argument("--recurrent-bc-lr", type=float, default=1e-3)
    parser.add_argument("--recurrent-bc-seq-len", type=int, default=32)
    parser.add_argument(
        "--recurrent-bc-action-class-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--recurrent-bc-action-class-balance-max-weight", type=float, default=5.0)
    parser.add_argument("--recurrent-bc-event-action-weight", type=float, default=2.0)
    parser.add_argument(
        "--recurrent-bc-event-action-events",
        default="picked_resource,dropped_resource,delivered,stage_completed,sync_complete,"
        "recharged,joint_target_scan",
    )
    parser.add_argument(
        "--recurrent-bc-comm-loss-weight",
        type=float,
        default=None,
        help="Overall recurrent BC communication loss weight; defaults to 1.0 for Signal specialist",
    )
    parser.add_argument("--recurrent-bc-pipeline-pickup-action-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-delivery-action-loss-weight", type=float, default=None)
    parser.add_argument(
        "--recurrent-bc-pipeline-delivery-progress-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recurrent-bc-pipeline-navigation-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recurrent-bc-pipeline-frontier-exploration-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recurrent-bc-pipeline-frontier-exploration-min-map-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--recurrent-bc-pipeline-sync-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recurrent-bc-pipeline-ready-interact-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recurrent-bc-pipeline-station-guard-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recurrent-bc-pipeline-wrong-station-recovery-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument("--recurrent-bc-pipeline-pickup-gate-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-pickup-gate-pos-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-pickup-gate-neg-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-plan-action-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-plan-head-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-option-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-message-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-send-gate-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-send-gate-pos-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-send-gate-neg-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-interact-gate-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-interact-gate-pos-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-interact-gate-neg-weight", type=float, default=None)
    parser.add_argument(
        "--recurrent-bc-calibrate-pipeline-interact-gate-threshold",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-bc-pipeline-interact-gate-threshold-target-rate",
        type=float,
        default=None,
    )
    parser.add_argument("--recurrent-bc-pipeline-bad-pickup-action-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-bad-drop-action-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-bad-interact-action-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-bad-action-margin-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-bc-pipeline-bad-action-margin", type=float, default=None)
    parser.add_argument(
        "--recurrent-bc-pipeline-proactive-bad-action-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--recurrent-pipeline-stage-count", type=int, default=None)
    parser.add_argument("--recurrent-pipeline-required-per-stage-min", type=int, default=1)
    parser.add_argument("--recurrent-pipeline-required-per-stage-max", type=int, default=2)
    parser.add_argument("--recurrent-pipeline-sync-probability", type=float, default=0.5)
    parser.add_argument("--recurrent-pipeline-dependency-probability", type=float, default=0.7)
    parser.add_argument("--recurrent-pipeline-wrong-delivery-penalty", type=float, default=0.25)
    parser.add_argument("--recurrent-obs-pipeline-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--recurrent-bc-calibrate-send-threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--recurrent-bc-send-threshold-target-rate", type=float, default=-1.0)
    parser.add_argument("--recurrent-bc-comm-send-pos-weight", type=float, default=0.0)
    parser.add_argument("--recurrent-bc-comm-send-rate-penalty-weight", type=float, default=0.0)
    parser.add_argument("--recurrent-bc-comm-send-rate-target", type=float, default=-1.0)
    parser.add_argument(
        "--recurrent-bc-signal-initial-message-weight",
        type=float,
        default=None,
        help="Signal specialist BC weight for step-0 private-hint message labels; defaults to 4.0 for specialist",
    )
    parser.add_argument(
        "--recurrent-bc-signal-initial-message-loss-weight",
        type=float,
        default=None,
        help="Signal specialist auxiliary token/length loss for step-0 private-hint messages",
    )
    parser.add_argument(
        "--recurrent-bc-signal-constraint-message-loss-weight",
        type=float,
        default=None,
        help="Signal specialist auxiliary token/length loss for non-initial clue/constraint messages",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-aux-weight",
        type=float,
        default=None,
        help=(
            "Signal specialist auxiliary loss for predicting fused exact target location from clues; "
            "defaults to 0.25 for the specialist preset"
        ),
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-hypothesis-loss-weight",
        type=float,
        default=None,
        help=(
            "Signal specialist auxiliary loss for target hypothesis commit, "
            "ambiguity, and coordinate predictions"
        ),
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-hypothesis-commit-loss-weight",
        type=float,
        default=None,
        help="Component multiplier for target-hypothesis commit supervision",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-hypothesis-ambiguity-loss-weight",
        type=float,
        default=None,
        help="Component multiplier for target-hypothesis ambiguity supervision",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-hypothesis-xy-loss-weight",
        type=float,
        default=None,
        help="Component multiplier for target-hypothesis coordinate supervision",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-hypothesis-min-map-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-pursuit-action-weight",
        type=float,
        default=None,
        help="Signal specialist action loss for moving toward observation-safe target candidates",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-pursuit-trust-exact-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in Signal ablation: let target-pursuit labels follow trusted exact "
            "target messages retained in scan-state memory."
        ),
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-pursuit-max-agents",
        type=int,
        default=None,
        help=(
            "Optional cap on how many closest agents receive Signal target-pursuit "
            "action labels; 0 keeps all eligible agents."
        ),
    )
    parser.add_argument(
        "--recurrent-bc-signal-sync-response-action-loss-weight",
        type=float,
        default=None,
        help="Signal specialist action loss for responding to teammate target scans",
    )
    parser.add_argument(
        "--recurrent-bc-signal-active-scan-response-action-weight",
        type=float,
        default=None,
        help=(
            "Signal ablation action loss for trusted target-informed agents to "
            "join/scan while a teammate target scan is active."
        ),
    )
    parser.add_argument(
        "--recurrent-bc-signal-active-scan-response-min-map-size",
        type=int,
        default=None,
        help="Minimum map size for Signal active-scan response labels in recurrent sweep runs",
    )
    parser.add_argument(
        "--recurrent-bc-signal-active-scan-response-max-agents",
        type=int,
        default=None,
        help="Maximum closest active-scan responders labeled per step; 0 keeps all eligible agents",
    )
    parser.add_argument(
        "--recurrent-bc-signal-scan-bridge-action-weight",
        type=float,
        default=None,
        help=(
            "Signal ablation action loss for refreshing/bridging a trusted self "
            "target scan while an exact-informed teammate is nearby."
        ),
    )
    parser.add_argument(
        "--recurrent-bc-signal-scan-bridge-min-map-size",
        type=int,
        default=None,
        help="Minimum map size for Signal scan-bridge labels in recurrent sweep runs",
    )
    parser.add_argument(
        "--recurrent-bc-signal-scan-bridge-remaining-threshold",
        type=float,
        default=None,
        help="Maximum normalized self scan-window remaining value for scan-bridge labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-scan-bridge-max-teammate-distance",
        type=int,
        default=None,
        help="Maximum Manhattan distance to an exact-informed teammate for scan-bridge labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-match-action-weight",
        type=float,
        default=None,
        help="Signal specialist action loss for matching target-directed oracle moves",
    )
    parser.add_argument(
        "--recurrent-bc-signal-first-target-scan-action-weight",
        type=float,
        default=None,
        help="Signal specialist positive action loss for first true-target scan labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-refresh-target-scan-action-weight",
        type=float,
        default=None,
        help="Signal specialist positive action loss for refresh true-target scan labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-joint-target-scan-action-weight",
        type=float,
        default=None,
        help="Signal specialist positive action loss for joint-completion target scans",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-opportunity-action-weight",
        type=float,
        default=None,
        help="Signal specialist positive action loss for observation-safe target scan opportunities",
    )
    parser.add_argument(
        "--recurrent-bc-signal-redundant-target-wait-action-loss-weight",
        type=float,
        default=None,
        help="Signal specialist action loss for waiting after an active target scan",
    )
    parser.add_argument(
        "--recurrent-bc-signal-constraint-frontier-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in Signal ablation: bias frontier exploration labels toward "
            "cells compatible with known target constraints."
        ),
    )
    parser.add_argument(
        "--recurrent-bc-signal-scan-decision-loss-weight",
        type=float,
        default=None,
        help="Signal specialist auxiliary loss for scan/no-scan decisions",
    )
    parser.add_argument(
        "--recurrent-bc-signal-scan-decision-pos-weight",
        type=float,
        default=None,
        help="Positive class weight for Signal scan-decision auxiliary labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-scan-decision-neg-weight",
        type=float,
        default=None,
        help="Negative class weight for Signal scan-decision auxiliary labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-scan-gate-loss-weight",
        type=float,
        default=None,
        help="Signal specialist auxiliary loss for suppressing unsafe target scans",
    )
    parser.add_argument(
        "--recurrent-bc-signal-scan-gate-pos-weight",
        type=float,
        default=None,
        help="Positive class weight for Signal scan-gate auxiliary labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-scan-gate-neg-weight",
        type=float,
        default=None,
        help="Negative class weight for Signal scan-gate auxiliary labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-validity-loss-weight",
        type=float,
        default=None,
        help="Signal specialist auxiliary loss for true-vs-rejected target validity",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-validity-pos-weight",
        type=float,
        default=None,
        help="Positive class weight for Signal target-validity auxiliary labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-validity-neg-weight",
        type=float,
        default=None,
        help="Negative class weight for Signal target-validity auxiliary labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-decision-loss-weight",
        type=float,
        default=None,
        help="Signal specialist auxiliary loss for target-center interact decisions",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-decision-pos-weight",
        type=float,
        default=None,
        help="Positive class weight for Signal target-decision auxiliary labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-target-decision-neg-weight",
        type=float,
        default=None,
        help="Negative class weight for Signal target-decision auxiliary labels",
    )
    parser.add_argument(
        "--recurrent-bc-signal-ambiguous-target-decision-negatives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Signal ablation: target-decision labels treat locally ambiguous true-target "
            "scans as negative until constraints identify a unique target."
        ),
    )
    parser.add_argument(
        "--recurrent-bc-signal-ambiguous-target-decision-min-map-size",
        type=int,
        default=16,
        help="Minimum map size for ambiguous target-decision negative labels.",
    )
    parser.add_argument(
        "--recurrent-bc-signal-ambiguous-target-search-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Signal ablation: keep visible-clue/frontier search action labels active "
            "when the agent is standing on a locally ambiguous target."
        ),
    )
    parser.add_argument(
        "--recurrent-bc-signal-ambiguous-target-search-min-map-size",
        type=int,
        default=16,
        help="Minimum map size for ambiguous target search labels.",
    )
    parser.add_argument(
        "--recurrent-bc-signal-decoy-drift-action-loss-weight",
        type=float,
        default=None,
        help="Signal specialist bad-action loss for drifting toward known decoy targets",
    )
    parser.add_argument(
        "--recurrent-bc-signal-decoy-scan-action-loss-weight",
        type=float,
        default=None,
        help="Signal specialist bad-action loss for scanning known decoy targets",
    )
    parser.add_argument(
        "--recurrent-bc-signal-rejected-target-drift-action-loss-weight",
        type=float,
        default=None,
        help="Signal specialist bad-action loss for moving toward rejected target candidates",
    )
    parser.add_argument(
        "--recurrent-bc-signal-clue-interact-action-weight",
        type=float,
        default=None,
        help="Signal specialist action loss for interacting on unclaimed center clues",
    )
    parser.add_argument(
        "--recurrent-bc-signal-clue-interact-min-map-size",
        type=int,
        default=None,
        help="Minimum map size for Signal center-clue interact labels in recurrent sweep runs",
    )
    parser.add_argument(
        "--recurrent-bc-signal-visible-clue-action-weight",
        type=float,
        default=None,
        help="Signal specialist action loss for moving toward/interacting with visible clues",
    )
    parser.add_argument(
        "--recurrent-bc-signal-visible-clue-min-map-size",
        type=int,
        default=None,
        help="Minimum map size for Signal visible-clue labels in recurrent sweep runs",
    )
    parser.add_argument(
        "--recurrent-bc-signal-evidence-sweep-action-weight",
        type=float,
        default=None,
        help="Signal large-map action loss for sweeping assigned sectors before evidence is found",
    )
    parser.add_argument(
        "--recurrent-bc-signal-evidence-sweep-min-map-size",
        type=int,
        default=None,
        help="Minimum map size for Signal evidence-sweep labels in recurrent sweep runs",
    )
    parser.add_argument(
        "--recurrent-bc-signal-frontier-exploration-action-weight",
        type=float,
        default=None,
        help="Signal specialist action loss for exploring frontiers when no clue/target is visible",
    )
    parser.add_argument(
        "--recurrent-bc-signal-frontier-exploration-min-map-size",
        type=int,
        default=None,
        help="Minimum map size for Signal frontier-exploration labels in recurrent sweep runs",
    )
    parser.add_argument("--recurrent-dagger-rounds", type=int, default=0)
    parser.add_argument("--recurrent-dagger-episodes", type=int, default=20)
    parser.add_argument("--recurrent-dagger-failed-effective-ratio-cap", type=float, default=0.25)
    parser.add_argument("--recurrent-dagger-oracle-action-rollin-rate", type=float, default=0.25)
    parser.add_argument("--recurrent-dagger-oracle-message-rollin-rate", type=float, default=0.0)
    parser.add_argument(
        "--recurrent-dagger-initial-target-broadcast-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "For Signal Hunt recurrent training, label step-0 agents with unambiguous private exact "
            "target hints to broadcast [26, x, y]. Defaults on for the specialist preset."
        ),
    )
    parser.add_argument(
        "--recurrent-dagger-target-handoff-requires-exact-target",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Signal recurrent ablation: only label target-handoff joins when the responder "
            "has trusted exact target evidence."
        ),
    )
    parser.add_argument(
        "--recurrent-dagger-signal-target-rendezvous-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Signal recurrent ablation: route the closest target-informed pair to the "
            "target and make an early scanner wait until a partner can complete."
        ),
    )
    parser.add_argument("--recurrent-dagger-signal-target-rendezvous-min-map-size", type=int, default=16)
    parser.add_argument("--recurrent-dagger-signal-target-rendezvous-max-agents", type=int, default=2)
    parser.add_argument(
        "--recurrent-dagger-focus-events",
        default=RECURRENT_DEFAULT_DAGGER_FOCUS_EVENTS,
    )
    parser.add_argument("--recurrent-dagger-focus-error-weight", type=float, default=3.0)
    parser.add_argument("--recurrent-dagger-focus-recovery-weight", type=float, default=2.0)
    parser.add_argument("--recurrent-dagger-focus-window", type=int, default=1)
    parser.add_argument("--recurrent-dagger-target-interact-focus-weight", type=float, default=5.0)
    parser.add_argument("--recurrent-dagger-target-discovery-min-map-size", type=int, default=16)
    parser.add_argument("--recurrent-dagger-target-discovery-focus-weight", type=float, default=3.0)
    parser.add_argument("--recurrent-dagger-movement-stall-min-map-size", type=int, default=16)
    parser.add_argument("--recurrent-dagger-movement-stall-window", type=int, default=6)
    parser.add_argument("--recurrent-dagger-movement-stall-focus-weight", type=float, default=4.0)
    parser.add_argument("--recurrent-dagger-target-decoy-drift-focus-weight", type=float, default=5.0)
    parser.add_argument(
        "--recurrent-dagger-focus-replay",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-dagger-retrain-from-scratch",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-dagger-restore-best",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--recurrent-dagger-pipeline-wrong-delivery-provenance-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-dagger-pipeline-wrong-delivery-provenance-weight",
        type=float,
        default=None,
    )
    parser.add_argument("--recurrent-dagger-replay-pre-steps", type=int, default=None)
    parser.add_argument("--recurrent-dagger-replay-post-steps", type=int, default=None)
    parser.add_argument("--recurrent-dagger-replay-weight", type=float, default=None)
    parser.add_argument("--recurrent-dagger-positive-replay-events", default=None)
    parser.add_argument("--recurrent-dagger-replay-event-weights", default=None)
    parser.add_argument("--recurrent-dagger-replay-event-caps", default=None)
    parser.add_argument("--recurrent-dagger-replay-success-only-events", default=None)
    parser.add_argument("--recurrent-dagger-replay-priority-events", default=None)
    parser.add_argument("--recurrent-dagger-replay-balance-positive-events", default=None)
    parser.add_argument("--recurrent-dagger-replay-balance-negative-events", default=None)
    parser.add_argument("--recurrent-dagger-replay-max-negative-per-positive", type=float, default=None)
    parser.add_argument("--recurrent-dagger-max-replay-snippets-per-episode", type=int, default=None)
    parser.add_argument(
        "--recurrent-dagger-max-failed-parent-replay-snippets-per-episode",
        type=int,
        default=None,
    )
    parser.add_argument("--recurrent-dagger-failed-parent-replay-weight-scale", type=float, default=None)
    parser.add_argument("--recurrent-dagger-expert-max-replay-snippets-per-episode", type=int, default=None)
    parser.add_argument(
        "--recurrent-eval-episodes",
        type=int,
        default=None,
        help="Override final recurrent eval episodes; defaults to the shared --eval-episodes value",
    )
    parser.add_argument(
        "--recurrent-rl-updates",
        type=int,
        default=None,
        help="Override recurrent PPO updates; defaults to the shared --updates value",
    )
    parser.add_argument(
        "--recurrent-rl-eval-episodes",
        type=int,
        default=None,
        help="Override recurrent PPO best-checkpoint eval episodes; defaults to recurrent final eval episodes",
    )
    parser.add_argument(
        "--recurrent-rl-eval-use-eval-seeds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the recurrent final eval seed panel for PPO best-checkpoint selection",
    )
    parser.add_argument(
        "--recurrent-ppo-profile",
        default="guarded",
        choices=sorted(RECURRENT_PPO_PROFILES),
        help="Named recurrent PPO tuning profile; explicit --recurrent-* PPO flags override it.",
    )
    parser.add_argument("--recurrent-rl-epochs", type=int, default=2)
    parser.add_argument("--recurrent-minibatch-seqs", type=int, default=8)
    parser.add_argument("--recurrent-rl-early-stop-eval-patience", type=int, default=None)
    parser.add_argument("--recurrent-rl-lr", type=float, default=None)
    parser.add_argument("--recurrent-clip", type=float, default=None)
    parser.add_argument("--recurrent-entropy-coeff", type=float, default=None)
    parser.add_argument("--recurrent-max-grad-norm", type=float, default=None)
    parser.add_argument("--recurrent-bc-kl-coeff", type=float, default=None)
    parser.add_argument("--recurrent-bc-comm-kl-coeff", type=float, default=None)
    parser.add_argument("--recurrent-rl-balanced-rollouts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--recurrent-rl-rollout-eval-decoding", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--recurrent-rl-rollout-pipeline-navigation-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-rl-rollout-pipeline-navigation-assist-trust-messages",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-rl-rollout-pipeline-station-interact-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-rl-rollout-pipeline-interact-gate-promote",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--recurrent-rl-eval-decoding-action-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-rl-pipeline-assisted-action-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-rl-pipeline-interact-gate-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-rl-pipeline-interact-gate-pos-weight", type=float, default=None)
    parser.add_argument("--recurrent-rl-pipeline-interact-gate-neg-weight", type=float, default=None)
    parser.add_argument("--recurrent-rl-pipeline-pickup-gate-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-rl-pipeline-pickup-gate-pos-weight", type=float, default=None)
    parser.add_argument("--recurrent-rl-pipeline-pickup-gate-neg-weight", type=float, default=None)
    parser.add_argument(
        "--recurrent-rl-pipeline-delivery-progress-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recurrent-rl-pipeline-navigation-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument("--recurrent-rl-pipeline-sync-action-loss-weight", type=float, default=None)
    parser.add_argument("--recurrent-rl-pipeline-ready-interact-action-loss-weight", type=float, default=None)
    parser.add_argument(
        "--recurrent-rl-pipeline-station-guard-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recurrent-rl-pipeline-wrong-station-recovery-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recurrent-rl-pipeline-plan-action-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recurrent-rl-pipeline-plan-head-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recurrent-rl-pipeline-option-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument("--recurrent-rl-pipeline-bad-pickup-penalty", type=float, default=None)
    parser.add_argument("--recurrent-rl-pipeline-bad-interact-penalty", type=float, default=None)
    parser.add_argument("--recurrent-rl-pipeline-unneeded-drop-bonus", type=float, default=None)
    parser.add_argument("--recurrent-rl-restore-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recurrent-rl-save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recurrent-train-map-sizes", default="")
    parser.add_argument("--recurrent-train-map-sampling-weights", default="")
    parser.add_argument("--recurrent-map-max-steps", default="")
    parser.add_argument("--recurrent-eval-map-sizes", default="")
    parser.add_argument("--recurrent-eval-seed-count", type=int, default=1)
    parser.add_argument(
        "--recurrent-eval-seed-range",
        default="",
        help=(
            "Compact recurrent eval seed panel. Use START:COUNT, for example 3000:40, "
            "or MAP_SIZE=START:COUNT entries joined by '+'."
        ),
    )
    parser.add_argument("--recurrent-eval-seed-list", default="")
    parser.add_argument("--recurrent-dagger-seed-list", default="")
    parser.add_argument("--recurrent-pipeline-assisted-rollout-episodes", type=int, default=None)
    parser.add_argument("--recurrent-pipeline-assisted-rollout-seed-base", type=int, default=None)
    parser.add_argument("--recurrent-pipeline-assisted-rollout-seed-list", default="")
    parser.add_argument(
        "--recurrent-pipeline-assisted-rollout-max-steps-per-episode",
        type=int,
        default=None,
    )
    parser.add_argument("--recurrent-pipeline-assisted-rollout-weight", type=float, default=None)
    parser.add_argument(
        "--recurrent-pipeline-assisted-rollout-success-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-pipeline-assisted-rollout-navigation-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-pipeline-assisted-rollout-navigation-assist-trust-messages",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recurrent-pipeline-assisted-rollout-station-interact-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--recurrent-pipeline-assisted-rollout-bc-epochs", type=int, default=None)
    parser.add_argument("--recurrent-skip-bc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-init", default=None)
    parser.add_argument(
        "--recurrent-init-template",
        default="",
        help=(
            "Seed-specific recurrent init path template. Supports {seed}, {scenario}, "
            "{map_size}, {agents}, {algorithm}, and {run_name}."
        ),
    )
    parser.add_argument("--recurrent-init-for-dagger", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-init-allow-obs-dim-mismatch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-comm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recurrent-comm-token-limit", type=int, default=8)
    parser.add_argument("--recurrent-comm-vocab-size", type=int, default=32)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--recurrent-backbone",
        choices=["mlp", "residual_mlp", "local_cnn"],
        default="mlp",
        help="Recurrent actor encoder backbone.",
    )
    parser.add_argument("--recurrent-obs-exploration-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-obs-exploration-age", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-obs-feedback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-obs-normalize-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-obs-navigation-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-obs-signal-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--recurrent-obs-signal-target-match-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--recurrent-obs-signal-confidence-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--recurrent-obs-signal-sector-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--recurrent-obs-signal-sync-feedback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-obs-signal-scan-state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--recurrent-obs-agent-id-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Append recurrent-only agent identity/search-role features. "
            "Defaults on for Signal Hunt specialist runs."
        ),
    )
    parser.add_argument("--recurrent-obs-pipeline-feedback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--recurrent-obs-pipeline-feedback-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--recurrent-obs-pipeline-progress-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--recurrent-obs-pipeline-shared-feedback",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--recurrent-eval-send-threshold", type=float, default=0.25)
    parser.add_argument("--recurrent-calibrate-send-threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-obs-memory-mode", default="auto", choices=["auto", "full", "egocentric"])
    parser.add_argument("--recurrent-obs-memory-radius", type=int, default=4)
    parser.add_argument(
        "--recurrent-eval-signal-target-scan-threshold",
        type=float,
        default=None,
        help=(
            "Signal specialist decode calibration: force interact on an observation-safe "
            "center true target when the model assigns at least this interact probability. "
            "Defaults to 0.0 for Signal specialist runs; set -1 to disable."
        ),
    )
    parser.add_argument("--recurrent-eval-signal-exact-target-memory-steps", type=int, default=32)
    parser.add_argument(
        "--recurrent-eval-signal-target-scan-lock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep observation-safe Signal target scans from being suppressed by learned "
            "scan/validity/decision gates. Opt-in; broad candidates can over-scan decoys."
        ),
    )
    parser.add_argument(
        "--recurrent-eval-signal-exact-target-scan-lock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Keep exact, trusted Signal target scans from being suppressed by learned "
            "scan/validity/decision gates without locking weaker target guesses."
        ),
    )
    parser.add_argument(
        "--recurrent-eval-signal-compatible-target-scan-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in Signal ablation: scan visible center target tiles that satisfy shared "
            "constraints even when the constraints do not infer a unique target yet."
        ),
    )
    parser.add_argument(
        "--recurrent-eval-signal-compatible-target-scan-min-strength",
        type=int,
        default=3,
        help="Minimum Signal clue constraint strength for compatible target scan assist.",
    )
    parser.add_argument(
        "--recurrent-eval-signal-negative-memory-scan-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in Signal ablation: suppress target scans on visible center target tiles "
            "remembered as decoy scans."
        ),
    )
    parser.add_argument(
        "--recurrent-eval-signal-target-probe-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in Signal ablation: probe-scan visible center targets that are not "
            "rejected, not remembered as decoys, and not already under own active scan."
        ),
    )
    parser.add_argument(
        "--recurrent-eval-signal-initial-exact-message-copy-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Canonicalize Signal specialist step-0 exact-target messages from private exact hints",
    )
    parser.add_argument(
        "--recurrent-eval-signal-exact-target-message-copy-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Opt-in Signal ablation: broadcast a canonical exact-target message from "
            "any single trusted Signal exact target during evaluation. Defaults on "
            "for the large_map Signal preset."
        ),
    )
    parser.add_argument(
        "--recurrent-eval-signal-constraint-message-copy-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Canonicalize sent Signal clue messages from the agent's private/collected "
            "structured goal-hint constraints. Defaults on for Signal specialist runs."
        ),
    )
    parser.add_argument(
        "--recurrent-eval-signal-constraint-message-guard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Opt-in Signal ablation: drop learned structured constraint messages unless "
            "the sender currently observes a supporting Signal constraint. Defaults on "
            "for the large_map Signal preset."
        ),
    )
    parser.add_argument(
        "--recurrent-eval-signal-evidence-sweep-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Signal eval rescue: during eval, use exploration-memory evidence sweeps "
            "after the rescue step. Defaults on for the large_map Signal preset."
        ),
    )
    parser.add_argument(
        "--recurrent-eval-signal-evidence-sweep-min-step",
        type=int,
        default=40,
        help="First eval step where recurrent Signal evidence-sweep rescue may override actions.",
    )
    parser.add_argument(
        "--recurrent-eval-signal-frontier-exploration-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in Signal ablation: during eval, use exploration-memory frontiers "
            "for agents without visible clue/target or unique target evidence."
        ),
    )
    parser.add_argument(
        "--recurrent-eval-signal-scan-refresh-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in Signal ablation: refresh an expiring own target scan from "
            "scan-state feedback when no teammate scan is active."
        ),
    )
    parser.add_argument(
        "--recurrent-eval-signal-scan-refresh-threshold",
        type=float,
        default=0.5,
        help="Remaining-scan-window threshold for Signal scan-refresh assist.",
    )
    parser.add_argument("--recurrent-eval-pipeline-navigation-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--recurrent-eval-pipeline-navigation-assist-trust-messages",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--recurrent-eval-pipeline-station-interact-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--recurrent-eval-pipeline-interact-gate-threshold", type=float, default=None)
    parser.add_argument(
        "--recurrent-eval-pipeline-interact-gate-promote",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--recurrent-eval-pipeline-event-head-threshold", type=float, default=-1.0)
    parser.add_argument("--recurrent-eval-pipeline-navigation-head-threshold", type=float, default=-1.0)
    parser.add_argument("--recurrent-eval-pipeline-plan-head-threshold", type=float, default=-1.0)
    parser.add_argument("--recurrent-eval-pipeline-option-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--recurrent-eval-pipeline-option-allow-interact",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", default="syncorsink-core-training")
    parser.add_argument("--strict-wandb", action="store_true", help="Fail a run if W&B was requested but did not initialize")
    parser.add_argument("--output-dir", default="logs/core_training_sweep")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds is None:
        args.seeds = [0 if args.seed is None else args.seed]
    elif args.seed is not None and args.seed not in args.seeds:
        args.seeds = sorted([*args.seeds, args.seed])
    args.seeds = sorted(set(args.seeds))
    if args.recurrent_eval_seed_range:
        if args.recurrent_eval_seed_list:
            parser.error("--recurrent-eval-seed-range cannot be combined with --recurrent-eval-seed-list")
        try:
            args.recurrent_eval_seed_list = _expand_seed_range(args.recurrent_eval_seed_range)
        except ValueError as exc:
            parser.error(str(exc))
        if args.recurrent_eval_episodes is None:
            args.recurrent_eval_episodes = 1
        if args.recurrent_rl_eval_episodes is None:
            args.recurrent_rl_eval_episodes = 1
        if args.recurrent_rl_eval_use_eval_seeds is None:
            args.recurrent_rl_eval_use_eval_seeds = True
    if args.recurrent_rl_eval_use_eval_seeds is None:
        args.recurrent_rl_eval_use_eval_seeds = False
    return _apply_recurrent_ppo_profile(args)


def main(argv: list[str] | None = None) -> int:
    payload = run_suite(parse_args(argv))
    return 1 if payload["overall"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
