"""Staged recurrent BC/DAgger curriculum for Signal Hunt.

This runner is intentionally trainer-facing: it creates local demos,
checkpoints, and summaries under logs/ and does not define benchmark artifacts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from syncorsink.train.mappo import resolve_device
from syncorsink.train.recurrent_bc_rl import (
    RecurrentConfig,
    _init_recurrent_wandb,
    _map_diagnostics_wandb_payload,
    _wandb_log,
    collect_episode_demos,
    evaluate_recurrent_policy_multi_seed,
    load_recurrent_actor_checkpoint,
    train_recurrent_bc_dagger,
)


DEFAULT_EVAL_SEND_THRESHOLD = 0.25


@dataclass
class RecurrentCurriculumConfig:
    scenario: str = "signal_hunt"
    agents: int = 2
    fov_preset: str = "easy"
    stage_map_suites: str = "8;8,16;8,16,32"
    max_steps_by_map: str = "8:60,16:120,32:240"
    train_map_sampling_weights: str = ""
    promotion_success_threshold: float = 0.8
    stop_on_unmet_mastery: bool = True
    carry_model_between_stages: bool = True

    # Observation and communication defaults used by the strongest recurrent Signal Hunt runs.
    oracle_type: str = "signal_hint_comm"
    obs_exploration_memory: bool = True
    obs_exploration_age: bool = False
    obs_feedback: bool = True
    obs_normalize_tokens: bool = True
    obs_memory_mode: str = "egocentric"
    obs_memory_radius: int = 4
    obs_navigation_features: bool = True
    obs_signal_features: bool = True
    obs_signal_sync_feedback: bool = True
    obs_signal_scan_state: bool = True
    obs_signal_negative_memory: bool = False
    obs_signal_negative_memory_window: int = 64
    obs_signal_inferred_target_features: bool = False
    obs_signal_target_match_features: bool = True
    comm: bool = True
    comm_token_limit: int = 8
    comm_vocab_size: int = 32
    comm_max_messages: int = 8
    comm_cost: float = 0.01
    comm_len_cost: float = 0.0

    hidden_dim: int = 128
    demo_episodes: int = 60
    bc_epochs: int = 3
    bc_lr: float = 1e-4
    bc_seq_len: int = 32
    bc_eval_every_epochs: int = 0
    bc_eval_episodes: int = 0
    bc_eval_seed_count: int = 1
    bc_restore_best_eval_epoch: bool = False
    bc_comm_loss_weight: float = 0.1
    bc_comm_send_pos_weight: float = 5.0
    bc_calibrate_send_threshold: bool = True
    bc_send_threshold_target_rate: float = -1.0
    bc_signal_redundant_target_interact_weight: float = 1.0
    bc_signal_target_pursuit_weight: float = 1.0
    bc_signal_target_pursuit_action_weight: float = 0.0
    bc_signal_sync_response_weight: float = 1.0
    bc_signal_sync_response_action_loss_weight: float = 0.0
    bc_signal_target_aux_weight: float = 0.0
    bc_signal_target_match_action_weight: float = 0.0
    bc_signal_first_target_scan_action_weight: float = 0.0
    bc_signal_refresh_target_scan_action_weight: float = 0.0
    bc_signal_joint_target_scan_action_weight: float = 0.0
    bc_signal_target_opportunity_action_weight: float = 0.0
    bc_signal_redundant_target_wait_action_loss_weight: float = 0.0
    bc_signal_scan_decision_loss_weight: float = 0.0
    bc_signal_scan_decision_pos_weight: float = 1.0
    bc_signal_scan_decision_neg_weight: float = 1.0
    bc_signal_scan_gate_loss_weight: float = 0.0
    bc_signal_scan_gate_pos_weight: float = 1.0
    bc_signal_scan_gate_neg_weight: float = 1.0
    bc_signal_target_validity_loss_weight: float = 0.0
    bc_signal_target_validity_pos_weight: float = 1.0
    bc_signal_target_validity_neg_weight: float = 1.0
    bc_signal_target_decision_loss_weight: float = 0.0
    bc_signal_target_decision_pos_weight: float = 1.0
    bc_signal_target_decision_neg_weight: float = 1.0
    bc_signal_rejected_target_interact_loss_weight: float = 0.05
    bc_signal_rejected_target_interact_action_loss_weight: float = 0.0
    bc_signal_bad_redundant_target_interact_loss_weight: float = 0.05
    bc_signal_decoy_drift_action_loss_weight: float = 0.25
    bc_signal_decoy_scan_action_loss_weight: float = 0.1
    bc_signal_rejected_target_drift_action_loss_weight: float = 0.0

    dagger_rounds: int = 1
    dagger_episodes: int = 16
    dagger_retrain_from_scratch: bool = False
    dagger_failed_episode_weight: float = 0.25
    dagger_focus_error_weight: float = 3.0
    dagger_focus_recovery_weight: float = 2.0
    dagger_focus_window: int = 1
    dagger_focus_replay: bool = True
    dagger_oracle_message_rollin_rate: float = 0.0
    dagger_target_scan_broadcast_labels: bool = False
    dagger_redundant_target_wait_labels: bool = False
    dagger_target_discovery_min_map_size: int = 16
    dagger_target_discovery_focus_weight: float = 3.0
    dagger_movement_stall_min_map_size: int = 16
    dagger_movement_stall_window: int = 6
    dagger_movement_stall_focus_weight: float = 4.0
    dagger_solo_target_team_weight: float = 1.0
    dagger_solo_target_team_success_only: bool = False
    dagger_positive_target_pursuit_min_map_size: int = 16
    dagger_positive_replay_events: str = ""
    dagger_replay_event_weights: str = ""
    dagger_replay_event_caps: str = ""
    dagger_replay_success_only_events: str = ""
    dagger_replay_priority_events: str = ""
    dagger_replay_balance_positive_events: str = ""
    dagger_replay_balance_negative_events: str = ""
    dagger_replay_max_negative_per_positive: float = -1.0
    dagger_replay_pre_steps: int = 2
    dagger_replay_post_steps: int = 2
    dagger_replay_weight: float = 1.0
    dagger_max_replay_snippets_per_episode: int = 4
    dagger_expert_max_replay_snippets_per_episode: int = -1

    eval_episodes: int = 12
    eval_seed: int = 3000
    eval_seed_count: int = 2
    eval_send_threshold: float | None = None
    eval_signal_target_scan_threshold: float = -1.0
    eval_signal_scan_gate_threshold: float = -1.0
    eval_signal_scan_gate_suppress: bool = False
    eval_signal_target_validity_threshold: float = -1.0
    eval_signal_target_decision_threshold: float = -1.0
    eval_signal_target_decision_suppress: bool = True
    eval_signal_scan_sync_assist: bool = False
    eval_signal_scan_sync_force_first: bool = False
    eval_signal_scan_broadcast_assist: bool = False
    eval_signal_exact_target_message_guard: bool = False
    eval_signal_exact_target_navigation_assist: bool = False
    eval_signal_exact_target_memory_steps: int = 0
    eval_signal_scan_refresh_assist: bool = False
    eval_signal_scan_refresh_threshold: float = 0.5

    output_dir: str = "logs/recurrent_curriculum"
    run_name: str | None = None
    initial_recurrent_checkpoint: str | None = None
    seed: int = 0
    device: str = "auto"
    dry_run: bool = False
    wandb: bool = False
    wandb_project: str = "syncorsink"
    wandb_run: str | None = None


def run_recurrent_curriculum(cfg: RecurrentCurriculumConfig) -> dict[str, Any]:
    stage_suites = _parse_stage_map_suites(cfg.stage_map_suites)
    max_steps = _parse_max_steps_by_map(cfg.max_steps_by_map)
    run_dir = _make_run_dir(cfg)
    summary_path = run_dir / "summary.json"
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "status": "dry_run" if cfg.dry_run else "running",
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "config": asdict(cfg),
        "planned_stages": [
            _planned_stage_row(idx, suite, max_steps, cfg, checkpoints_dir)
            for idx, suite in enumerate(stage_suites)
        ],
        "stages": [],
    }
    _write_json(summary_path, result)
    if cfg.dry_run:
        return result

    device = resolve_device(cfg.device)
    model = None
    current_threshold = _resolve_initial_eval_send_threshold(cfg)
    status = "complete"

    for stage_idx, suite in enumerate(stage_suites):
        has_initial_model = model is not None or (
            stage_idx == 0 and bool(cfg.initial_recurrent_checkpoint)
        )
        stage_cfg = _stage_recurrent_config(
            cfg,
            stage_idx=stage_idx,
            suite=suite,
            max_steps=max_steps,
            checkpoint_path=_stage_checkpoint_path(checkpoints_dir, stage_idx, suite),
            eval_send_threshold=current_threshold,
            has_initial_model=has_initial_model,
        )
        if stage_idx == 0 and model is None and cfg.initial_recurrent_checkpoint:
            model = load_recurrent_actor_checkpoint(
                cfg.initial_recurrent_checkpoint,
                stage_cfg,
                device,
            )
        stage_run = _init_recurrent_wandb(stage_cfg) if cfg.wandb else None
        demos = collect_episode_demos(stage_cfg)
        model, history, all_episodes, best_round = train_recurrent_bc_dagger(
            stage_cfg,
            demos,
            device,
            wandb_run=stage_run,
            initial_model=model if cfg.carry_model_between_stages else None,
        )
        eval_result = (best_round or {}).get("eval")
        if eval_result is None:
            eval_result = evaluate_recurrent_policy_multi_seed(
                stage_cfg,
                model,
                device,
                seed_count=max(1, int(stage_cfg.eval_seed_count)),
            )
        current_threshold = float(stage_cfg.eval_send_threshold)
        mastery = _mastery_row(eval_result, cfg.promotion_success_threshold)
        checkpoint_path = _stage_checkpoint_path(checkpoints_dir, stage_idx, suite)
        _save_stage_checkpoint(
            checkpoint_path,
            model=model,
            stage_cfg=stage_cfg,
            curriculum_cfg=cfg,
            stage_idx=stage_idx,
            suite=suite,
            eval_result=eval_result,
            history=history,
            best_round=best_round,
        )
        stage_row = {
            "stage_index": stage_idx,
            "stage_name": _stage_name(suite),
            "train_map_sizes": list(suite),
            "checkpoint": str(checkpoint_path),
            "demo_episodes": len(demos),
            "dataset_episodes": len(all_episodes),
            "dagger_history": history,
            "best_round": best_round,
            "initial_recurrent_checkpoint": cfg.initial_recurrent_checkpoint if stage_idx == 0 else None,
            "eval": eval_result,
            "mastery": mastery,
            "calibrated_send_threshold": current_threshold,
        }
        result["stages"].append(stage_row)
        result["status"] = status
        _write_json(summary_path, result)
        if stage_run is not None:
            _wandb_log(
                stage_run,
                _stage_wandb_payload(stage_row),
                context="recurrent curriculum stage log",
            )
            stage_run.finish()
        if not mastery["passed"] and cfg.stop_on_unmet_mastery and stage_idx < len(stage_suites) - 1:
            status = "stopped_unmet_mastery"
            break

    result["status"] = status
    _write_json(summary_path, result)
    return result


def _parse_stage_map_suites(raw_value: str) -> list[tuple[int, ...]]:
    raw = str(raw_value or "").strip()
    if not raw:
        raise ValueError("stage_map_suites must contain at least one stage")
    suites: list[tuple[int, ...]] = []
    for raw_stage in raw.split(";"):
        raw_stage = raw_stage.strip()
        if not raw_stage:
            continue
        suite = tuple(int(item.strip()) for item in raw_stage.split(",") if item.strip())
        if not suite or any(size <= 0 for size in suite):
            raise ValueError(f"invalid stage map suite: {raw_stage!r}")
        suites.append(suite)
    if not suites:
        raise ValueError("stage_map_suites must contain at least one valid stage")
    return suites


def _checkpoint_eval_send_threshold(path: str | Path) -> float | None:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(checkpoint, dict):
        return None
    candidates = [
        (checkpoint.get("config") or {}).get("eval_send_threshold"),
        (checkpoint.get("best_dagger_round") or {}).get("eval_send_threshold"),
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_initial_eval_send_threshold(cfg: RecurrentCurriculumConfig) -> float:
    if cfg.eval_send_threshold is not None:
        return float(cfg.eval_send_threshold)
    if cfg.initial_recurrent_checkpoint:
        inherited = _checkpoint_eval_send_threshold(cfg.initial_recurrent_checkpoint)
        if inherited is not None:
            return float(inherited)
    return float(DEFAULT_EVAL_SEND_THRESHOLD)


def _parse_max_steps_by_map(raw_value: str) -> dict[int, int]:
    raw = str(raw_value or "").strip()
    if not raw:
        return {}
    values: dict[int, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"max_steps_by_map entries must be map_size:max_steps pairs, got {item!r}")
        raw_size, raw_steps = item.split(":", 1)
        size = int(raw_size.strip())
        steps = int(raw_steps.strip())
        if size <= 0 or steps <= 0:
            raise ValueError(f"max_steps_by_map values must be positive, got {item!r}")
        values[size] = steps
    return values


def _stage_max_steps(suite: tuple[int, ...], max_steps: dict[int, int]) -> int:
    return int(max_steps.get(int(suite[0]), max(60, int(suite[0]) * 8)))


def _map_max_steps_string(max_steps: dict[int, int]) -> str:
    return ",".join(f"{size}:{steps}" for size, steps in sorted(max_steps.items()))


def _stage_name(suite: tuple[int, ...]) -> str:
    return "maps_" + "_".join(str(size) for size in suite)


def _make_run_dir(cfg: RecurrentCurriculumConfig) -> Path:
    if cfg.run_name:
        name = cfg.run_name
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{cfg.scenario}_recurrent_curriculum_seed{cfg.seed}_{stamp}"
    return Path(cfg.output_dir) / name


def _stage_checkpoint_path(checkpoints_dir: Path, stage_idx: int, suite: tuple[int, ...]) -> Path:
    return checkpoints_dir / f"stage{stage_idx}_{_stage_name(suite)}.pt"


def _planned_stage_row(
    stage_idx: int,
    suite: tuple[int, ...],
    max_steps: dict[int, int],
    cfg: RecurrentCurriculumConfig,
    checkpoints_dir: Path,
) -> dict[str, Any]:
    return {
        "stage_index": int(stage_idx),
        "stage_name": _stage_name(suite),
        "train_map_sizes": list(suite),
        "eval_map_sizes": list(suite),
        "max_steps": {str(size): int(max_steps.get(size, _stage_max_steps((size,), max_steps))) for size in suite},
        "promotion_success_threshold": float(cfg.promotion_success_threshold),
        "checkpoint": str(_stage_checkpoint_path(checkpoints_dir, stage_idx, suite)),
    }


def _stage_recurrent_config(
    cfg: RecurrentCurriculumConfig,
    *,
    stage_idx: int,
    suite: tuple[int, ...],
    max_steps: dict[int, int],
    checkpoint_path: Path,
    eval_send_threshold: float,
    has_initial_model: bool,
) -> RecurrentConfig:
    run_base = cfg.wandb_run or cfg.run_name or "recurrent_curriculum"
    stage_wandb_run = f"{run_base}-stage{stage_idx}-{_stage_name(suite)}"
    return RecurrentConfig(
        scenario=cfg.scenario,
        map_size=int(suite[0]),
        train_map_sizes=",".join(str(size) for size in suite),
        train_map_sampling_weights=cfg.train_map_sampling_weights,
        map_max_steps=_map_max_steps_string(max_steps),
        agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        max_steps=_stage_max_steps(suite, max_steps),
        oracle_type=cfg.oracle_type,
        obs_exploration_memory=cfg.obs_exploration_memory,
        obs_exploration_age=cfg.obs_exploration_age,
        obs_feedback=cfg.obs_feedback,
        obs_normalize_tokens=cfg.obs_normalize_tokens,
        obs_memory_mode=cfg.obs_memory_mode,
        obs_memory_radius=cfg.obs_memory_radius,
        obs_navigation_features=cfg.obs_navigation_features,
        obs_signal_features=cfg.obs_signal_features,
        obs_signal_sync_feedback=cfg.obs_signal_sync_feedback,
        obs_signal_scan_state=cfg.obs_signal_scan_state,
        obs_signal_negative_memory=cfg.obs_signal_negative_memory,
        obs_signal_negative_memory_window=cfg.obs_signal_negative_memory_window,
        obs_signal_inferred_target_features=cfg.obs_signal_inferred_target_features,
        obs_signal_target_match_features=cfg.obs_signal_target_match_features,
        hidden_dim=cfg.hidden_dim,
        comm=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
        comm_max_messages=cfg.comm_max_messages,
        comm_cost=cfg.comm_cost,
        comm_len_cost=cfg.comm_len_cost,
        demo_episodes=cfg.demo_episodes,
        bc_epochs=cfg.bc_epochs,
        bc_lr=cfg.bc_lr,
        bc_seq_len=cfg.bc_seq_len,
        bc_eval_every_epochs=cfg.bc_eval_every_epochs,
        bc_eval_episodes=cfg.bc_eval_episodes,
        bc_eval_seed_count=cfg.bc_eval_seed_count,
        bc_restore_best_eval_epoch=cfg.bc_restore_best_eval_epoch,
        bc_comm_loss_weight=cfg.bc_comm_loss_weight,
        bc_comm_send_pos_weight=cfg.bc_comm_send_pos_weight,
        bc_calibrate_send_threshold=cfg.bc_calibrate_send_threshold,
        bc_send_threshold_target_rate=cfg.bc_send_threshold_target_rate,
        bc_signal_redundant_target_interact_weight=cfg.bc_signal_redundant_target_interact_weight,
        bc_signal_target_pursuit_weight=cfg.bc_signal_target_pursuit_weight,
        bc_signal_target_pursuit_action_weight=cfg.bc_signal_target_pursuit_action_weight,
        bc_signal_sync_response_weight=cfg.bc_signal_sync_response_weight,
        bc_signal_sync_response_action_loss_weight=cfg.bc_signal_sync_response_action_loss_weight,
        bc_signal_target_aux_weight=cfg.bc_signal_target_aux_weight,
        bc_signal_target_match_action_weight=cfg.bc_signal_target_match_action_weight,
        bc_signal_first_target_scan_action_weight=cfg.bc_signal_first_target_scan_action_weight,
        bc_signal_refresh_target_scan_action_weight=cfg.bc_signal_refresh_target_scan_action_weight,
        bc_signal_joint_target_scan_action_weight=cfg.bc_signal_joint_target_scan_action_weight,
        bc_signal_target_opportunity_action_weight=cfg.bc_signal_target_opportunity_action_weight,
        bc_signal_redundant_target_wait_action_loss_weight=cfg.bc_signal_redundant_target_wait_action_loss_weight,
        bc_signal_scan_decision_loss_weight=cfg.bc_signal_scan_decision_loss_weight,
        bc_signal_scan_decision_pos_weight=cfg.bc_signal_scan_decision_pos_weight,
        bc_signal_scan_decision_neg_weight=cfg.bc_signal_scan_decision_neg_weight,
        bc_signal_scan_gate_loss_weight=cfg.bc_signal_scan_gate_loss_weight,
        bc_signal_scan_gate_pos_weight=cfg.bc_signal_scan_gate_pos_weight,
        bc_signal_scan_gate_neg_weight=cfg.bc_signal_scan_gate_neg_weight,
        bc_signal_target_validity_loss_weight=cfg.bc_signal_target_validity_loss_weight,
        bc_signal_target_validity_pos_weight=cfg.bc_signal_target_validity_pos_weight,
        bc_signal_target_validity_neg_weight=cfg.bc_signal_target_validity_neg_weight,
        bc_signal_target_decision_loss_weight=cfg.bc_signal_target_decision_loss_weight,
        bc_signal_target_decision_pos_weight=cfg.bc_signal_target_decision_pos_weight,
        bc_signal_target_decision_neg_weight=cfg.bc_signal_target_decision_neg_weight,
        bc_signal_rejected_target_interact_loss_weight=cfg.bc_signal_rejected_target_interact_loss_weight,
        bc_signal_rejected_target_interact_action_loss_weight=(
            cfg.bc_signal_rejected_target_interact_action_loss_weight
        ),
        bc_signal_bad_redundant_target_interact_loss_weight=cfg.bc_signal_bad_redundant_target_interact_loss_weight,
        bc_signal_decoy_drift_action_loss_weight=cfg.bc_signal_decoy_drift_action_loss_weight,
        bc_signal_decoy_scan_action_loss_weight=cfg.bc_signal_decoy_scan_action_loss_weight,
        bc_signal_rejected_target_drift_action_loss_weight=(
            cfg.bc_signal_rejected_target_drift_action_loss_weight
        ),
        dagger_rounds=cfg.dagger_rounds,
        dagger_episodes=cfg.dagger_episodes,
        dagger_retrain_from_scratch=(
            False if has_initial_model and cfg.carry_model_between_stages else cfg.dagger_retrain_from_scratch
        ),
        dagger_failed_episode_weight=cfg.dagger_failed_episode_weight,
        dagger_focus_error_weight=cfg.dagger_focus_error_weight,
        dagger_focus_recovery_weight=cfg.dagger_focus_recovery_weight,
        dagger_focus_window=cfg.dagger_focus_window,
        dagger_focus_replay=cfg.dagger_focus_replay,
        dagger_oracle_message_rollin_rate=cfg.dagger_oracle_message_rollin_rate,
        dagger_target_scan_broadcast_labels=cfg.dagger_target_scan_broadcast_labels,
        dagger_redundant_target_wait_labels=cfg.dagger_redundant_target_wait_labels,
        dagger_target_discovery_min_map_size=cfg.dagger_target_discovery_min_map_size,
        dagger_target_discovery_focus_weight=cfg.dagger_target_discovery_focus_weight,
        dagger_movement_stall_min_map_size=cfg.dagger_movement_stall_min_map_size,
        dagger_movement_stall_window=cfg.dagger_movement_stall_window,
        dagger_movement_stall_focus_weight=cfg.dagger_movement_stall_focus_weight,
        dagger_solo_target_team_weight=cfg.dagger_solo_target_team_weight,
        dagger_solo_target_team_success_only=cfg.dagger_solo_target_team_success_only,
        dagger_positive_target_pursuit_min_map_size=cfg.dagger_positive_target_pursuit_min_map_size,
        dagger_positive_replay_events=cfg.dagger_positive_replay_events,
        dagger_replay_event_weights=cfg.dagger_replay_event_weights,
        dagger_replay_event_caps=cfg.dagger_replay_event_caps,
        dagger_replay_success_only_events=cfg.dagger_replay_success_only_events,
        dagger_replay_priority_events=cfg.dagger_replay_priority_events,
        dagger_replay_balance_positive_events=cfg.dagger_replay_balance_positive_events,
        dagger_replay_balance_negative_events=cfg.dagger_replay_balance_negative_events,
        dagger_replay_max_negative_per_positive=cfg.dagger_replay_max_negative_per_positive,
        dagger_replay_pre_steps=cfg.dagger_replay_pre_steps,
        dagger_replay_post_steps=cfg.dagger_replay_post_steps,
        dagger_replay_weight=cfg.dagger_replay_weight,
        dagger_max_replay_snippets_per_episode=cfg.dagger_max_replay_snippets_per_episode,
        dagger_expert_max_replay_snippets_per_episode=cfg.dagger_expert_max_replay_snippets_per_episode,
        rl_updates=0,
        eval_episodes=cfg.eval_episodes,
        eval_seed=int(cfg.eval_seed) + stage_idx * 100000,
        eval_seed_count=cfg.eval_seed_count,
        eval_map_sizes=",".join(str(size) for size in suite),
        eval_send_threshold=eval_send_threshold,
        eval_signal_target_scan_threshold=cfg.eval_signal_target_scan_threshold,
        eval_signal_scan_gate_threshold=cfg.eval_signal_scan_gate_threshold,
        eval_signal_scan_gate_suppress=cfg.eval_signal_scan_gate_suppress,
        eval_signal_target_validity_threshold=cfg.eval_signal_target_validity_threshold,
        eval_signal_target_decision_threshold=cfg.eval_signal_target_decision_threshold,
        eval_signal_target_decision_suppress=cfg.eval_signal_target_decision_suppress,
        eval_signal_scan_sync_assist=cfg.eval_signal_scan_sync_assist,
        eval_signal_scan_sync_force_first=cfg.eval_signal_scan_sync_force_first,
        eval_signal_scan_broadcast_assist=cfg.eval_signal_scan_broadcast_assist,
        eval_signal_exact_target_message_guard=cfg.eval_signal_exact_target_message_guard,
        eval_signal_exact_target_navigation_assist=cfg.eval_signal_exact_target_navigation_assist,
        eval_signal_exact_target_memory_steps=cfg.eval_signal_exact_target_memory_steps,
        eval_signal_scan_refresh_assist=cfg.eval_signal_scan_refresh_assist,
        eval_signal_scan_refresh_threshold=cfg.eval_signal_scan_refresh_threshold,
        save=str(checkpoint_path),
        seed=int(cfg.seed) + stage_idx,
        device=cfg.device,
        wandb=cfg.wandb,
        wandb_project=cfg.wandb_project,
        wandb_run=stage_wandb_run,
    )


def _mastery_row(eval_result: dict[str, Any], threshold: float) -> dict[str, Any]:
    success_rate = float(eval_result.get("success_rate", 0.0))
    signal = eval_result.get("signal") or {}
    return {
        "metric": "success_rate",
        "value": success_rate,
        "threshold": float(threshold),
        "passed": bool(success_rate >= float(threshold)),
        "avg_wrong_target_scans": float(signal.get("avg_wrong_target_scans", 0.0)),
        "avg_redundant_target_scans": float(signal.get("avg_redundant_target_scans", 0.0)),
        "avg_reached_true_target": float(signal.get("avg_reached_true_target", 0.0)),
    }


def _save_stage_checkpoint(
    path: Path,
    *,
    model,
    stage_cfg: RecurrentConfig,
    curriculum_cfg: RecurrentCurriculumConfig,
    stage_idx: int,
    suite: tuple[int, ...],
    eval_result: dict[str, Any],
    history: list[dict[str, Any]],
    best_round: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": "recurrent_bc_dagger_curriculum",
            "model": model.state_dict(),
            "config": vars(stage_cfg),
            "curriculum_config": asdict(curriculum_cfg),
            "stage_index": int(stage_idx),
            "stage_map_sizes": list(suite),
            "eval_recurrent_policy": eval_result,
            "dagger_history": history,
            "best_dagger_round": best_round,
        },
        path,
    )


def _stage_wandb_payload(stage_row: dict[str, Any]) -> dict[str, float | int]:
    eval_result = stage_row.get("eval") or {}
    mastery = stage_row.get("mastery") or {}
    payload: dict[str, float | int] = {
        "curriculum/stage_index": int(stage_row.get("stage_index", 0)),
        "curriculum/mastery_success_rate": float(mastery.get("value", 0.0)),
        "curriculum/mastery_threshold": float(mastery.get("threshold", 0.0)),
        "curriculum/mastery_passed": int(bool(mastery.get("passed", False))),
        "curriculum/calibrated_send_threshold": float(stage_row.get("calibrated_send_threshold", 0.0)),
        "curriculum/eval_success_rate": float(eval_result.get("success_rate", 0.0)),
        "curriculum/eval_avg_return": float(eval_result.get("avg_return", 0.0)),
        "curriculum/eval_avg_steps": float(eval_result.get("avg_steps", 0.0)),
        "curriculum/eval_avg_comm_tokens": float(eval_result.get("avg_comm_tokens", 0.0)),
    }
    signal = eval_result.get("signal") or {}
    for key in ("avg_wrong_target_scans", "avg_redundant_target_scans", "avg_reached_true_target"):
        if key in signal:
            payload[f"curriculum/signal/{key}"] = float(signal[key])
    payload.update(_map_diagnostics_wandb_payload("curriculum/dataset", _stage_dataset_diagnostics(stage_row)))
    return payload


def _stage_dataset_diagnostics(stage_row: dict[str, Any]) -> dict[str, dict]:
    best_round = stage_row.get("best_round") or {}
    return best_round.get("dataset_map_diagnostics") or {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged recurrent Signal Hunt curriculum")
    parser.add_argument("--stage-map-suites", default="8;8,16;8,16,32")
    parser.add_argument("--max-steps-by-map", default="8:60,16:120,32:240")
    parser.add_argument("--train-map-sampling-weights", default="")
    parser.add_argument("--promotion-success-threshold", type=float, default=0.8)
    parser.add_argument("--stop-on-unmet-mastery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--carry-model-between-stages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demo-episodes", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-lr", type=float, default=1e-4)
    parser.add_argument("--bc-eval-every-epochs", type=int, default=0)
    parser.add_argument("--bc-eval-episodes", type=int, default=0)
    parser.add_argument("--bc-eval-seed-count", type=int, default=1)
    parser.add_argument("--bc-restore-best-eval-epoch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-signal-negative-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-exploration-age", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-signal-negative-memory-window", type=int, default=64)
    parser.add_argument("--obs-signal-inferred-target-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-signal-target-match-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-signal-redundant-target-interact-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-target-pursuit-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-target-pursuit-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-sync-response-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-sync-response-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-target-aux-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-target-match-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-first-target-scan-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-refresh-target-scan-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-joint-target-scan-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-target-opportunity-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-redundant-target-wait-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-scan-decision-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-scan-decision-pos-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-scan-decision-neg-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-scan-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-scan-gate-pos-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-scan-gate-neg-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-target-validity-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-target-validity-pos-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-target-validity-neg-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-target-decision-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-target-decision-pos-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-target-decision-neg-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-rejected-target-interact-loss-weight", type=float, default=0.05)
    parser.add_argument("--bc-signal-rejected-target-interact-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-bad-redundant-target-interact-loss-weight", type=float, default=0.05)
    parser.add_argument("--bc-signal-decoy-drift-action-loss-weight", type=float, default=0.25)
    parser.add_argument("--bc-signal-decoy-scan-action-loss-weight", type=float, default=0.1)
    parser.add_argument("--bc-signal-rejected-target-drift-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--dagger-episodes", type=int, default=16)
    parser.add_argument("--dagger-focus-error-weight", type=float, default=3.0)
    parser.add_argument("--dagger-focus-recovery-weight", type=float, default=2.0)
    parser.add_argument("--dagger-focus-window", type=int, default=1)
    parser.add_argument("--dagger-oracle-message-rollin-rate", type=float, default=0.0)
    parser.add_argument("--dagger-target-scan-broadcast-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dagger-redundant-target-wait-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dagger-target-discovery-min-map-size", type=int, default=16)
    parser.add_argument("--dagger-target-discovery-focus-weight", type=float, default=3.0)
    parser.add_argument("--dagger-movement-stall-min-map-size", type=int, default=16)
    parser.add_argument("--dagger-movement-stall-window", type=int, default=6)
    parser.add_argument("--dagger-movement-stall-focus-weight", type=float, default=4.0)
    parser.add_argument("--dagger-solo-target-team-weight", type=float, default=1.0)
    parser.add_argument("--dagger-solo-target-team-success-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dagger-positive-target-pursuit-min-map-size", type=int, default=16)
    parser.add_argument("--dagger-positive-replay-events", default="")
    parser.add_argument("--dagger-replay-event-weights", default="")
    parser.add_argument("--dagger-replay-event-caps", default="")
    parser.add_argument("--dagger-replay-success-only-events", default="")
    parser.add_argument("--dagger-replay-priority-events", default="")
    parser.add_argument("--dagger-replay-balance-positive-events", default="")
    parser.add_argument("--dagger-replay-balance-negative-events", default="")
    parser.add_argument("--dagger-replay-max-negative-per-positive", type=float, default=-1.0)
    parser.add_argument("--dagger-replay-pre-steps", type=int, default=2)
    parser.add_argument("--dagger-replay-post-steps", type=int, default=2)
    parser.add_argument("--dagger-replay-weight", type=float, default=1.0)
    parser.add_argument("--dagger-max-replay-snippets-per-episode", type=int, default=4)
    parser.add_argument("--dagger-expert-max-replay-snippets-per-episode", type=int, default=-1)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--eval-seed-count", type=int, default=2)
    parser.add_argument("--eval-send-threshold", type=float, default=None)
    parser.add_argument("--eval-signal-target-scan-threshold", type=float, default=-1.0)
    parser.add_argument("--eval-signal-scan-gate-threshold", type=float, default=-1.0)
    parser.add_argument("--eval-signal-scan-gate-suppress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-signal-target-validity-threshold", type=float, default=-1.0)
    parser.add_argument("--eval-signal-target-decision-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--eval-signal-target-decision-suppress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-signal-scan-sync-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-signal-scan-sync-force-first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-signal-scan-broadcast-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-signal-exact-target-message-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--eval-signal-exact-target-navigation-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eval-signal-exact-target-memory-steps", type=int, default=0)
    parser.add_argument("--eval-signal-scan-refresh-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-signal-scan-refresh-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="logs/recurrent_curriculum")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--initial-recurrent-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
    args = parser.parse_args()

    cfg = RecurrentCurriculumConfig(
        stage_map_suites=args.stage_map_suites,
        max_steps_by_map=args.max_steps_by_map,
        train_map_sampling_weights=args.train_map_sampling_weights,
        promotion_success_threshold=args.promotion_success_threshold,
        stop_on_unmet_mastery=args.stop_on_unmet_mastery,
        carry_model_between_stages=args.carry_model_between_stages,
        demo_episodes=args.demo_episodes,
        hidden_dim=args.hidden_dim,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
        bc_eval_every_epochs=args.bc_eval_every_epochs,
        bc_eval_episodes=args.bc_eval_episodes,
        bc_eval_seed_count=args.bc_eval_seed_count,
        bc_restore_best_eval_epoch=args.bc_restore_best_eval_epoch,
        obs_exploration_age=args.obs_exploration_age,
        obs_signal_negative_memory=args.obs_signal_negative_memory,
        obs_signal_negative_memory_window=args.obs_signal_negative_memory_window,
        obs_signal_inferred_target_features=args.obs_signal_inferred_target_features,
        obs_signal_target_match_features=args.obs_signal_target_match_features,
        bc_signal_redundant_target_interact_weight=args.bc_signal_redundant_target_interact_weight,
        bc_signal_target_pursuit_weight=args.bc_signal_target_pursuit_weight,
        bc_signal_target_pursuit_action_weight=args.bc_signal_target_pursuit_action_weight,
        bc_signal_sync_response_weight=args.bc_signal_sync_response_weight,
        bc_signal_sync_response_action_loss_weight=args.bc_signal_sync_response_action_loss_weight,
        bc_signal_target_aux_weight=args.bc_signal_target_aux_weight,
        bc_signal_target_match_action_weight=args.bc_signal_target_match_action_weight,
        bc_signal_first_target_scan_action_weight=args.bc_signal_first_target_scan_action_weight,
        bc_signal_refresh_target_scan_action_weight=args.bc_signal_refresh_target_scan_action_weight,
        bc_signal_joint_target_scan_action_weight=args.bc_signal_joint_target_scan_action_weight,
        bc_signal_target_opportunity_action_weight=args.bc_signal_target_opportunity_action_weight,
        bc_signal_redundant_target_wait_action_loss_weight=args.bc_signal_redundant_target_wait_action_loss_weight,
        bc_signal_scan_decision_loss_weight=args.bc_signal_scan_decision_loss_weight,
        bc_signal_scan_decision_pos_weight=args.bc_signal_scan_decision_pos_weight,
        bc_signal_scan_decision_neg_weight=args.bc_signal_scan_decision_neg_weight,
        bc_signal_scan_gate_loss_weight=args.bc_signal_scan_gate_loss_weight,
        bc_signal_scan_gate_pos_weight=args.bc_signal_scan_gate_pos_weight,
        bc_signal_scan_gate_neg_weight=args.bc_signal_scan_gate_neg_weight,
        bc_signal_target_validity_loss_weight=args.bc_signal_target_validity_loss_weight,
        bc_signal_target_validity_pos_weight=args.bc_signal_target_validity_pos_weight,
        bc_signal_target_validity_neg_weight=args.bc_signal_target_validity_neg_weight,
        bc_signal_target_decision_loss_weight=args.bc_signal_target_decision_loss_weight,
        bc_signal_target_decision_pos_weight=args.bc_signal_target_decision_pos_weight,
        bc_signal_target_decision_neg_weight=args.bc_signal_target_decision_neg_weight,
        bc_signal_rejected_target_interact_loss_weight=args.bc_signal_rejected_target_interact_loss_weight,
        bc_signal_rejected_target_interact_action_loss_weight=(
            args.bc_signal_rejected_target_interact_action_loss_weight
        ),
        bc_signal_bad_redundant_target_interact_loss_weight=args.bc_signal_bad_redundant_target_interact_loss_weight,
        bc_signal_decoy_drift_action_loss_weight=args.bc_signal_decoy_drift_action_loss_weight,
        bc_signal_decoy_scan_action_loss_weight=args.bc_signal_decoy_scan_action_loss_weight,
        bc_signal_rejected_target_drift_action_loss_weight=args.bc_signal_rejected_target_drift_action_loss_weight,
        dagger_rounds=args.dagger_rounds,
        dagger_episodes=args.dagger_episodes,
        dagger_focus_error_weight=args.dagger_focus_error_weight,
        dagger_focus_recovery_weight=args.dagger_focus_recovery_weight,
        dagger_focus_window=args.dagger_focus_window,
        dagger_oracle_message_rollin_rate=args.dagger_oracle_message_rollin_rate,
        dagger_target_scan_broadcast_labels=args.dagger_target_scan_broadcast_labels,
        dagger_redundant_target_wait_labels=args.dagger_redundant_target_wait_labels,
        dagger_target_discovery_min_map_size=args.dagger_target_discovery_min_map_size,
        dagger_target_discovery_focus_weight=args.dagger_target_discovery_focus_weight,
        dagger_movement_stall_min_map_size=args.dagger_movement_stall_min_map_size,
        dagger_movement_stall_window=args.dagger_movement_stall_window,
        dagger_movement_stall_focus_weight=args.dagger_movement_stall_focus_weight,
        dagger_solo_target_team_weight=args.dagger_solo_target_team_weight,
        dagger_solo_target_team_success_only=args.dagger_solo_target_team_success_only,
        dagger_positive_target_pursuit_min_map_size=args.dagger_positive_target_pursuit_min_map_size,
        dagger_positive_replay_events=args.dagger_positive_replay_events,
        dagger_replay_event_weights=args.dagger_replay_event_weights,
        dagger_replay_event_caps=args.dagger_replay_event_caps,
        dagger_replay_success_only_events=args.dagger_replay_success_only_events,
        dagger_replay_priority_events=args.dagger_replay_priority_events,
        dagger_replay_balance_positive_events=args.dagger_replay_balance_positive_events,
        dagger_replay_balance_negative_events=args.dagger_replay_balance_negative_events,
        dagger_replay_max_negative_per_positive=args.dagger_replay_max_negative_per_positive,
        dagger_replay_pre_steps=args.dagger_replay_pre_steps,
        dagger_replay_post_steps=args.dagger_replay_post_steps,
        dagger_replay_weight=args.dagger_replay_weight,
        dagger_max_replay_snippets_per_episode=args.dagger_max_replay_snippets_per_episode,
        dagger_expert_max_replay_snippets_per_episode=args.dagger_expert_max_replay_snippets_per_episode,
        eval_episodes=args.eval_episodes,
        eval_seed_count=args.eval_seed_count,
        eval_send_threshold=args.eval_send_threshold,
        eval_signal_target_scan_threshold=args.eval_signal_target_scan_threshold,
        eval_signal_scan_gate_threshold=args.eval_signal_scan_gate_threshold,
        eval_signal_scan_gate_suppress=args.eval_signal_scan_gate_suppress,
        eval_signal_target_validity_threshold=args.eval_signal_target_validity_threshold,
        eval_signal_target_decision_threshold=args.eval_signal_target_decision_threshold,
        eval_signal_target_decision_suppress=args.eval_signal_target_decision_suppress,
        eval_signal_scan_sync_assist=args.eval_signal_scan_sync_assist,
        eval_signal_scan_sync_force_first=args.eval_signal_scan_sync_force_first,
        eval_signal_scan_broadcast_assist=args.eval_signal_scan_broadcast_assist,
        eval_signal_exact_target_message_guard=args.eval_signal_exact_target_message_guard,
        eval_signal_exact_target_navigation_assist=args.eval_signal_exact_target_navigation_assist,
        eval_signal_exact_target_memory_steps=args.eval_signal_exact_target_memory_steps,
        eval_signal_scan_refresh_assist=args.eval_signal_scan_refresh_assist,
        eval_signal_scan_refresh_threshold=args.eval_signal_scan_refresh_threshold,
        output_dir=args.output_dir,
        run_name=args.run_name,
        initial_recurrent_checkpoint=args.initial_recurrent_checkpoint,
        seed=args.seed,
        device=args.device,
        dry_run=args.dry_run,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
    )
    result = run_recurrent_curriculum(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
