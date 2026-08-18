"""Staged recurrent BC/DAgger curriculum for SyncOrSink scenarios.

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
    DEFAULT_DAGGER_FOCUS_EVENTS,
    RecurrentConfig,
    _init_recurrent_wandb,
    _map_diagnostics_wandb_payload,
    _parse_map_sampling_weights,
    _wandb_log,
    collect_episode_demos,
    evaluate_recurrent_policy_multi_seed,
    load_recurrent_actor_checkpoint,
    train_pipeline_assisted_rollout_bc_stage,
    train_recurrent_bc_dagger,
    train_recurrent_rl,
)


DEFAULT_EVAL_SEND_THRESHOLD = 0.25


@dataclass
class RecurrentCurriculumConfig:
    scenario: str = "signal_hunt"
    agents: int = 2
    fov_preset: str = "easy"
    stage_map_suites: str = "8;8,16;8,16,32"
    max_steps_by_map: str = "8:80,16:160,32:500"
    train_map_sampling_weights: str = ""
    promotion_success_threshold: float = 0.8
    stop_on_unmet_mastery: bool = True
    carry_model_between_stages: bool = True

    # Observation and communication defaults remain tuned for recurrent Signal Hunt;
    # override scenario/oracle/channel fields for Pipeline and Energy curricula.
    oracle_type: str = "signal_hint_comm"
    obs_exploration_memory: bool = True
    obs_exploration_age: bool = False
    obs_feedback: bool = True
    obs_normalize_tokens: bool = True
    obs_memory_mode: str = "egocentric"
    obs_memory_radius: int = 4
    obs_navigation_features: bool = True
    obs_pipeline_features: bool = True
    obs_pipeline_feedback: bool = True
    obs_pipeline_feedback_metadata: bool = True
    obs_pipeline_progress_features: bool = False
    obs_pipeline_shared_feedback: bool = False
    obs_signal_features: bool = True
    obs_signal_sync_feedback: bool = True
    obs_signal_scan_state: bool = True
    obs_signal_negative_memory: bool = False
    obs_signal_negative_memory_window: int = 64
    obs_signal_inferred_target_features: bool = False
    obs_signal_target_match_features: bool = True
    obs_signal_confidence_features: bool = False
    obs_signal_sector_features: bool = False
    comm: bool = True
    comm_token_limit: int = 8
    comm_vocab_size: int = 32
    comm_max_messages: int = 8
    comm_cost: float = 0.01
    comm_len_cost: float = 0.0
    pipeline_stage_count: int | None = None
    pipeline_required_per_stage_min: int = 1
    pipeline_required_per_stage_max: int = 2
    pipeline_sync_probability: float = 0.5
    pipeline_dependency_probability: float = 0.7
    pipeline_wrong_delivery_penalty: float = 0.25
    pipeline_stage_count_schedule: str = ""
    pipeline_required_per_stage_min_schedule: str = ""
    pipeline_required_per_stage_max_schedule: str = ""
    pipeline_sync_probability_schedule: str = ""
    pipeline_dependency_probability_schedule: str = ""

    hidden_dim: int = 128
    recurrent_backbone: str = "mlp"
    demo_episodes: int = 60
    bc_epochs: int = 3
    bc_lr: float = 1e-4
    bc_seq_len: int = 32
    bc_eval_every_epochs: int = 0
    bc_eval_episodes: int = 0
    bc_eval_seed_count: int = 1
    bc_restore_best_eval_epoch: bool = False
    bc_equal_episode_weight: bool = True
    bc_action_class_balance: bool = False
    bc_action_class_balance_max_weight: float = 5.0
    bc_event_action_weight: float = 0.0
    bc_event_action_events: str = (
        "picked_resource,dropped_resource,delivered,stage_completed,sync_complete,"
        "recharged,joint_target_scan"
    )
    bc_comm_loss_weight: float = 1.0
    bc_comm_send_pos_weight: float = 5.0
    bc_comm_send_loss_weight: float = 1.0
    bc_comm_length_loss_weight: float = 1.0
    bc_comm_token_loss_weight: float = 1.0
    bc_comm_send_rate_penalty_weight: float = 0.0
    bc_comm_send_rate_target: float = -1.0
    bc_calibrate_send_threshold: bool = True
    bc_send_threshold_target_rate: float = -1.0
    bc_calibrate_pipeline_interact_gate_threshold: bool = False
    bc_pipeline_interact_gate_threshold_target_rate: float = -1.0
    bc_signal_redundant_target_interact_weight: float = 1.0
    bc_signal_target_pursuit_weight: float = 1.0
    bc_signal_target_pursuit_action_weight: float = 0.0
    bc_signal_target_pursuit_trust_exact_memory: bool = False
    bc_signal_target_pursuit_max_agents: int = 0
    bc_signal_constraint_frontier_bias: bool = False
    bc_signal_initial_message_weight: float = 4.0
    bc_signal_initial_message_loss_weight: float = 4.0
    bc_signal_constraint_message_loss_weight: float = 4.0
    bc_signal_sync_response_weight: float = 1.0
    bc_signal_sync_response_action_loss_weight: float = 0.0
    bc_signal_active_scan_response_action_weight: float = 0.0
    bc_signal_active_scan_response_min_map_size: int = 16
    bc_signal_active_scan_response_max_agents: int = 1
    bc_signal_scan_bridge_action_weight: float = 0.0
    bc_signal_scan_bridge_min_map_size: int = 16
    bc_signal_scan_bridge_remaining_threshold: float = 0.5
    bc_signal_scan_bridge_max_teammate_distance: int = 6
    bc_signal_target_aux_weight: float = 0.0
    bc_signal_target_match_action_weight: float = 0.0
    bc_signal_first_target_scan_action_weight: float = 0.0
    bc_signal_refresh_target_scan_action_weight: float = 0.0
    bc_signal_joint_target_scan_action_weight: float = 0.0
    bc_signal_target_opportunity_action_weight: float = 0.0
    bc_signal_redundant_target_wait_action_loss_weight: float = 0.0
    bc_signal_scan_decision_loss_weight: float = 1.0
    bc_signal_scan_decision_pos_weight: float = 2.0
    bc_signal_scan_decision_neg_weight: float = 3.0
    bc_signal_scan_gate_loss_weight: float = 1.0
    bc_signal_scan_gate_pos_weight: float = 2.0
    bc_signal_scan_gate_neg_weight: float = 3.0
    bc_signal_target_validity_loss_weight: float = 1.0
    bc_signal_target_validity_pos_weight: float = 2.0
    bc_signal_target_validity_neg_weight: float = 3.0
    bc_signal_target_decision_loss_weight: float = 1.0
    bc_signal_target_decision_pos_weight: float = 2.0
    bc_signal_target_decision_neg_weight: float = 3.0
    bc_signal_ambiguous_target_decision_negatives: bool = False
    bc_signal_ambiguous_target_decision_min_map_size: int = 16
    bc_signal_ambiguous_target_search_labels: bool = False
    bc_signal_ambiguous_target_search_min_map_size: int = 16
    bc_signal_target_hypothesis_loss_weight: float = 0.0
    bc_signal_target_hypothesis_commit_loss_weight: float = 1.0
    bc_signal_target_hypothesis_ambiguity_loss_weight: float = 1.0
    bc_signal_target_hypothesis_xy_loss_weight: float = 1.0
    bc_signal_target_hypothesis_min_map_size: int = 16
    bc_signal_rejected_target_interact_loss_weight: float = 0.05
    bc_signal_rejected_target_interact_action_loss_weight: float = 0.0
    bc_signal_bad_redundant_target_interact_loss_weight: float = 0.05
    bc_signal_decoy_drift_action_loss_weight: float = 0.25
    bc_signal_decoy_scan_action_loss_weight: float = 0.1
    bc_signal_rejected_target_drift_action_loss_weight: float = 0.0
    bc_signal_clue_interact_action_weight: float = 0.0
    bc_signal_clue_interact_min_map_size: int = 16
    bc_signal_evidence_sweep_action_weight: float = 0.0
    bc_signal_evidence_sweep_min_map_size: int = 16
    bc_signal_frontier_exploration_action_weight: float = 0.0
    bc_signal_frontier_exploration_min_map_size: int = 16
    bc_pipeline_pickup_action_loss_weight: float = 0.0
    bc_pipeline_delivery_action_loss_weight: float = 0.0
    bc_pipeline_delivery_progress_action_loss_weight: float = 0.0
    bc_pipeline_navigation_action_loss_weight: float = 0.0
    bc_pipeline_frontier_exploration_action_loss_weight: float = 0.0
    bc_pipeline_frontier_exploration_min_map_size: int = 8
    bc_pipeline_sync_action_loss_weight: float = 0.0
    bc_pipeline_ready_interact_action_loss_weight: float = 0.0
    bc_pipeline_station_guard_action_loss_weight: float = 0.0
    bc_pipeline_pickup_gate_loss_weight: float = 0.0
    bc_pipeline_pickup_gate_pos_weight: float = 1.0
    bc_pipeline_pickup_gate_neg_weight: float = 1.0
    bc_pipeline_plan_action_loss_weight: float = 0.0
    bc_pipeline_plan_head_loss_weight: float = 0.0
    bc_pipeline_option_loss_weight: float = 0.0
    bc_pipeline_message_loss_weight: float = 0.0
    bc_pipeline_send_gate_loss_weight: float = 0.0
    bc_pipeline_send_gate_pos_weight: float = 1.0
    bc_pipeline_send_gate_neg_weight: float = 1.0
    bc_pipeline_interact_gate_loss_weight: float = 0.0
    bc_pipeline_interact_gate_pos_weight: float = 1.0
    bc_pipeline_interact_gate_neg_weight: float = 1.0
    bc_pipeline_bad_pickup_action_loss_weight: float = 0.0
    bc_pipeline_bad_drop_action_loss_weight: float = 0.0
    bc_pipeline_bad_interact_action_loss_weight: float = 0.0
    bc_pipeline_proactive_bad_action_labels: bool = False

    dagger_rounds: int = 1
    dagger_episodes: int = 16
    dagger_seed_base: int = 10000
    dagger_seed_stride: int = 1000
    dagger_seed_list: str = ""
    dagger_retrain_from_scratch: bool = False
    dagger_failed_episode_weight: float = 0.25
    dagger_focus_events: str = DEFAULT_DAGGER_FOCUS_EVENTS
    dagger_focus_error_weight: float = 3.0
    dagger_focus_recovery_weight: float = 2.0
    dagger_focus_window: int = 1
    dagger_focus_replay: bool = True
    dagger_pipeline_wrong_delivery_provenance_labels: bool = False
    dagger_pipeline_wrong_delivery_provenance_weight: float = -1.0
    dagger_oracle_message_rollin_rate: float = 0.0
    dagger_oracle_action_rollin_rate: float = 0.0
    dagger_initial_target_broadcast_labels: bool = True
    dagger_target_scan_broadcast_labels: bool = False
    dagger_target_handoff_requires_exact_target: bool = False
    dagger_signal_target_rendezvous_labels: bool = False
    dagger_signal_target_rendezvous_min_map_size: int = 16
    dagger_signal_target_rendezvous_max_agents: int = 2
    dagger_redundant_target_wait_labels: bool = False
    dagger_target_discovery_min_map_size: int = 16
    dagger_target_discovery_focus_weight: float = 3.0
    dagger_movement_stall_min_map_size: int = 16
    dagger_movement_stall_window: int = 6
    dagger_movement_stall_focus_weight: float = 4.0
    dagger_solo_target_team_weight: float = 1.0
    dagger_solo_target_team_success_only: bool = False
    dagger_restore_best: bool = True
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
    dagger_max_failed_parent_replay_snippets_per_episode: int = -1
    dagger_failed_parent_replay_weight_scale: float = 1.0
    dagger_expert_max_replay_snippets_per_episode: int = -1

    pipeline_assisted_rollout_episodes: int = 0
    pipeline_assisted_rollout_seed_base: int = 20000
    pipeline_assisted_rollout_seed_list: str = ""
    pipeline_assisted_rollout_max_steps_per_episode: int = 0
    pipeline_assisted_rollout_weight: float = 1.0
    pipeline_assisted_rollout_success_only: bool = False
    pipeline_assisted_rollout_navigation_assist: bool = True
    pipeline_assisted_rollout_navigation_assist_trust_messages: bool = True
    pipeline_assisted_rollout_station_interact_guard: bool = True
    pipeline_assisted_rollout_bc_epochs: int = -1

    rl_updates: int = 0
    rl_updates_schedule: str = ""
    rl_early_stop_eval_patience: int = 0
    rollout_steps: int = 256
    rl_balanced_rollouts: bool = False
    rl_rollout_map_steps: str = ""
    rl_rollout_eval_decoding: bool = False
    rl_rollout_pipeline_navigation_assist: bool = False
    rl_rollout_pipeline_navigation_assist_trust_messages: bool = False
    rl_rollout_pipeline_station_interact_guard: bool = False
    rl_rollout_pipeline_interact_gate_promote: bool = False
    rl_eval_decoding_action_loss_weight: float = 0.0
    rl_pipeline_assisted_action_loss_weight: float = 0.0
    rl_pipeline_interact_gate_loss_weight: float = 0.0
    rl_pipeline_interact_gate_pos_weight: float = 1.0
    rl_pipeline_interact_gate_neg_weight: float = 1.0
    rl_pipeline_pickup_gate_loss_weight: float = 0.0
    rl_pipeline_pickup_gate_pos_weight: float = 1.0
    rl_pipeline_pickup_gate_neg_weight: float = 1.0
    rl_pipeline_delivery_progress_action_loss_weight: float = 0.0
    rl_pipeline_navigation_action_loss_weight: float = 0.0
    rl_pipeline_sync_action_loss_weight: float = 0.0
    rl_pipeline_ready_interact_action_loss_weight: float = 0.0
    rl_pipeline_station_guard_action_loss_weight: float = 0.0
    rl_pipeline_wrong_station_recovery_action_loss_weight: float = 0.0
    rl_pipeline_plan_action_loss_weight: float = 0.0
    rl_pipeline_plan_head_loss_weight: float = 0.0
    rl_pipeline_option_loss_weight: float = 0.0
    rl_redundant_target_scan_penalty: float = 0.0
    rl_wrong_target_scan_penalty: float = 0.0
    rl_pipeline_bad_pickup_penalty: float = 0.0
    rl_pipeline_bad_interact_penalty: float = 0.0
    rl_pipeline_unneeded_drop_bonus: float = 0.0
    rl_epochs: int = 2
    minibatch_seqs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    entropy_coeff: float = 0.01
    rl_lr: float = 3e-5
    max_grad_norm: float = 0.5
    bc_kl_coeff: float = 0.5
    bc_comm_kl_coeff: float = 0.5
    rl_eval_every: int = 5
    rl_eval_episodes: int = 20
    rl_eval_use_eval_seeds: bool = True
    rl_eval_seed: int = 10000
    rl_eval_seed_stage_stride: int = 100000
    rl_eval_seed_count: int = 1
    rl_eval_seed_list: str = ""
    rl_restore_best: bool = True
    rl_save_best: bool = True
    rl_best_save: str | None = None

    eval_episodes: int = 12
    eval_seed: int = 3000
    eval_seed_count: int = 2
    eval_send_threshold: float | None = None
    eval_signal_target_scan_threshold: float = -1.0
    eval_signal_scan_gate_threshold: float = 0.4
    eval_signal_scan_gate_suppress: bool = True
    eval_signal_target_validity_threshold: float = 0.4
    eval_signal_target_decision_threshold: float = 0.4
    eval_signal_target_decision_suppress: bool = True
    eval_signal_exact_target_scan_lock: bool = False
    eval_signal_compatible_target_scan_assist: bool = False
    eval_signal_compatible_target_scan_min_strength: int = 3
    eval_signal_negative_memory_scan_guard: bool = False
    eval_signal_target_probe_assist: bool = False
    eval_signal_scan_sync_assist: bool = False
    eval_signal_scan_sync_force_first: bool = False
    eval_signal_scan_broadcast_assist: bool = False
    eval_signal_constraint_message_copy_assist: bool = False
    eval_signal_constraint_message_guard: bool = False
    eval_signal_exact_target_message_guard: bool = False
    eval_signal_initial_exact_message_copy_assist: bool = True
    eval_signal_exact_target_message_copy_assist: bool = False
    eval_signal_exact_target_navigation_assist: bool = False
    eval_signal_exact_target_memory_steps: int = 0
    eval_signal_scan_refresh_assist: bool = False
    eval_signal_scan_refresh_threshold: float = 0.5
    eval_signal_evidence_sweep_assist: bool = False
    eval_signal_evidence_sweep_min_step: int = 40
    eval_signal_frontier_exploration_assist: bool = False
    eval_pipeline_navigation_assist: bool = False
    eval_pipeline_navigation_assist_trust_messages: bool = False
    eval_pipeline_station_interact_guard: bool = False
    eval_pipeline_plan_broadcast_assist: bool = False
    eval_pipeline_pickup_gate_suppress: bool = False
    eval_pipeline_frontier_exploration_assist: bool = False
    eval_pipeline_interact_gate_threshold: float = -1.0
    eval_pipeline_interact_gate_promote: bool = False
    eval_pipeline_event_head_threshold: float = -1.0
    eval_pipeline_navigation_head_threshold: float = -1.0
    eval_pipeline_plan_head_threshold: float = -1.0
    eval_pipeline_option_threshold: float = -1.0
    eval_pipeline_option_allow_interact: bool = False

    output_dir: str = "logs/recurrent_curriculum"
    run_name: str | None = None
    initial_recurrent_checkpoint: str | None = None
    recurrent_init_allow_obs_dim_mismatch: bool = False
    seed: int = 0
    device: str = "auto"
    dry_run: bool = False
    wandb: bool = False
    wandb_project: str = "syncorsink"
    wandb_run: str | None = None


def run_recurrent_curriculum(cfg: RecurrentCurriculumConfig) -> dict[str, Any]:
    stage_suites = _parse_stage_map_suites(cfg.stage_map_suites)
    max_steps = _parse_max_steps_by_map(cfg.max_steps_by_map)
    _validate_curriculum_map_sampling_weights(cfg.train_map_sampling_weights, stage_suites)
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
        bc_eval_result = (best_round or {}).get("eval")
        assisted_rollout_summary = None
        if int(stage_cfg.pipeline_assisted_rollout_episodes) > 0:
            model, all_episodes, assisted_rollout_summary, assisted_eval_result = (
                train_pipeline_assisted_rollout_bc_stage(
                    stage_cfg,
                    model,
                    all_episodes,
                    device,
                    wandb_run=stage_run,
                )
            )
            if assisted_eval_result is not None:
                bc_eval_result = assisted_eval_result
        if bc_eval_result is None:
            bc_eval_result = evaluate_recurrent_policy_multi_seed(
                stage_cfg,
                model,
                device,
                seed_count=max(1, int(stage_cfg.eval_seed_count)),
            )
        eval_result = bc_eval_result
        if int(stage_cfg.rl_updates) > 0:
            model = train_recurrent_rl(stage_cfg, model, device, wandb_run=stage_run)
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
            assisted_rollout_summary=assisted_rollout_summary,
        )
        stage_row = {
            "stage_index": stage_idx,
            "stage_name": _stage_name(suite),
            "train_map_sizes": list(suite),
            "max_steps": {
                str(size): int(max_steps.get(size, _stage_max_steps((size,), max_steps)))
                for size in suite
            },
            "pipeline": _stage_pipeline_settings(cfg, stage_idx),
            "checkpoint": str(checkpoint_path),
            "demo_episodes": len(demos),
            "dataset_episodes": len(all_episodes),
            "pipeline_assisted_rollout": assisted_rollout_summary,
            "dagger_history": history,
            "best_round": best_round,
            "bc_eval": bc_eval_result,
            "rl_updates": int(stage_cfg.rl_updates),
            "rl_best_checkpoint": _stage_rl_best_checkpoint(stage_cfg),
            "initial_recurrent_checkpoint": cfg.initial_recurrent_checkpoint if stage_idx == 0 else None,
            "eval": eval_result,
            "mastery": mastery,
            "rl_early_stop_eval_patience": int(stage_cfg.rl_early_stop_eval_patience),
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


def _stage_rl_best_checkpoint(stage_cfg: RecurrentConfig) -> str | None:
    if not stage_cfg.rl_save_best:
        return None
    if stage_cfg.rl_best_save:
        return str(stage_cfg.rl_best_save)
    if not stage_cfg.save:
        return None
    path = Path(stage_cfg.save)
    suffix = path.suffix or ".pt"
    return str(path.with_name(f"{path.stem}_best{suffix}"))


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
        "pipeline": _stage_pipeline_settings(cfg, stage_idx),
        "rl_updates": _stage_rl_updates(cfg, stage_idx),
        "rl_early_stop_eval_patience": int(cfg.rl_early_stop_eval_patience),
        "rl_eval_use_eval_seeds": bool(cfg.rl_eval_use_eval_seeds),
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
    pipeline = _stage_pipeline_settings(cfg, stage_idx)
    return RecurrentConfig(
        scenario=cfg.scenario,
        map_size=int(suite[0]),
        train_map_sizes=",".join(str(size) for size in suite),
        train_map_sampling_weights=_stage_train_map_sampling_weights(
            cfg.train_map_sampling_weights,
            suite,
        ),
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
        obs_pipeline_features=cfg.obs_pipeline_features,
        obs_pipeline_feedback=cfg.obs_pipeline_feedback,
        obs_pipeline_feedback_metadata=cfg.obs_pipeline_feedback_metadata,
        obs_pipeline_progress_features=cfg.obs_pipeline_progress_features,
        obs_pipeline_shared_feedback=cfg.obs_pipeline_shared_feedback,
        obs_signal_features=cfg.obs_signal_features,
        obs_signal_sync_feedback=cfg.obs_signal_sync_feedback,
        obs_signal_scan_state=cfg.obs_signal_scan_state,
        obs_signal_negative_memory=cfg.obs_signal_negative_memory,
        obs_signal_negative_memory_window=cfg.obs_signal_negative_memory_window,
        obs_signal_inferred_target_features=cfg.obs_signal_inferred_target_features,
        obs_signal_target_match_features=cfg.obs_signal_target_match_features,
        obs_signal_confidence_features=cfg.obs_signal_confidence_features,
        obs_signal_sector_features=cfg.obs_signal_sector_features,
        hidden_dim=cfg.hidden_dim,
        recurrent_backbone=cfg.recurrent_backbone,
        comm=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
        comm_max_messages=cfg.comm_max_messages,
        comm_cost=cfg.comm_cost,
        comm_len_cost=cfg.comm_len_cost,
        pipeline_stage_count=pipeline["stage_count"],
        pipeline_required_per_stage_min=pipeline["required_per_stage_min"],
        pipeline_required_per_stage_max=pipeline["required_per_stage_max"],
        pipeline_sync_probability=pipeline["sync_probability"],
        pipeline_dependency_probability=pipeline["dependency_probability"],
        pipeline_wrong_delivery_penalty=cfg.pipeline_wrong_delivery_penalty,
        demo_episodes=cfg.demo_episodes,
        bc_epochs=cfg.bc_epochs,
        bc_lr=cfg.bc_lr,
        bc_seq_len=cfg.bc_seq_len,
        bc_eval_every_epochs=cfg.bc_eval_every_epochs,
        bc_eval_episodes=cfg.bc_eval_episodes,
        bc_eval_seed_count=cfg.bc_eval_seed_count,
        bc_restore_best_eval_epoch=cfg.bc_restore_best_eval_epoch,
        bc_equal_episode_weight=cfg.bc_equal_episode_weight,
        bc_action_class_balance=cfg.bc_action_class_balance,
        bc_action_class_balance_max_weight=cfg.bc_action_class_balance_max_weight,
        bc_event_action_weight=cfg.bc_event_action_weight,
        bc_event_action_events=cfg.bc_event_action_events,
        bc_comm_loss_weight=cfg.bc_comm_loss_weight,
        bc_comm_send_pos_weight=cfg.bc_comm_send_pos_weight,
        bc_comm_send_loss_weight=cfg.bc_comm_send_loss_weight,
        bc_comm_length_loss_weight=cfg.bc_comm_length_loss_weight,
        bc_comm_token_loss_weight=cfg.bc_comm_token_loss_weight,
        bc_comm_send_rate_penalty_weight=cfg.bc_comm_send_rate_penalty_weight,
        bc_comm_send_rate_target=cfg.bc_comm_send_rate_target,
        bc_calibrate_send_threshold=cfg.bc_calibrate_send_threshold,
        bc_send_threshold_target_rate=cfg.bc_send_threshold_target_rate,
        bc_calibrate_pipeline_interact_gate_threshold=(
            cfg.bc_calibrate_pipeline_interact_gate_threshold
        ),
        bc_pipeline_interact_gate_threshold_target_rate=(
            cfg.bc_pipeline_interact_gate_threshold_target_rate
        ),
        bc_signal_redundant_target_interact_weight=cfg.bc_signal_redundant_target_interact_weight,
        bc_signal_target_pursuit_weight=cfg.bc_signal_target_pursuit_weight,
        bc_signal_target_pursuit_action_weight=cfg.bc_signal_target_pursuit_action_weight,
        bc_signal_target_pursuit_trust_exact_memory=(
            cfg.bc_signal_target_pursuit_trust_exact_memory
        ),
        bc_signal_target_pursuit_max_agents=cfg.bc_signal_target_pursuit_max_agents,
        bc_signal_constraint_frontier_bias=cfg.bc_signal_constraint_frontier_bias,
        bc_signal_initial_message_weight=cfg.bc_signal_initial_message_weight,
        bc_signal_initial_message_loss_weight=cfg.bc_signal_initial_message_loss_weight,
        bc_signal_constraint_message_loss_weight=cfg.bc_signal_constraint_message_loss_weight,
        bc_signal_sync_response_weight=cfg.bc_signal_sync_response_weight,
        bc_signal_sync_response_action_loss_weight=cfg.bc_signal_sync_response_action_loss_weight,
        bc_signal_active_scan_response_action_weight=(
            cfg.bc_signal_active_scan_response_action_weight
        ),
        bc_signal_active_scan_response_min_map_size=(
            cfg.bc_signal_active_scan_response_min_map_size
        ),
        bc_signal_active_scan_response_max_agents=(
            cfg.bc_signal_active_scan_response_max_agents
        ),
        bc_signal_scan_bridge_action_weight=cfg.bc_signal_scan_bridge_action_weight,
        bc_signal_scan_bridge_min_map_size=cfg.bc_signal_scan_bridge_min_map_size,
        bc_signal_scan_bridge_remaining_threshold=(
            cfg.bc_signal_scan_bridge_remaining_threshold
        ),
        bc_signal_scan_bridge_max_teammate_distance=(
            cfg.bc_signal_scan_bridge_max_teammate_distance
        ),
        bc_signal_target_aux_weight=cfg.bc_signal_target_aux_weight,
        bc_signal_target_hypothesis_loss_weight=(
            cfg.bc_signal_target_hypothesis_loss_weight
        ),
        bc_signal_target_hypothesis_commit_loss_weight=(
            cfg.bc_signal_target_hypothesis_commit_loss_weight
        ),
        bc_signal_target_hypothesis_ambiguity_loss_weight=(
            cfg.bc_signal_target_hypothesis_ambiguity_loss_weight
        ),
        bc_signal_target_hypothesis_xy_loss_weight=(
            cfg.bc_signal_target_hypothesis_xy_loss_weight
        ),
        bc_signal_target_hypothesis_min_map_size=(
            cfg.bc_signal_target_hypothesis_min_map_size
        ),
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
        bc_signal_ambiguous_target_decision_negatives=(
            cfg.bc_signal_ambiguous_target_decision_negatives
        ),
        bc_signal_ambiguous_target_decision_min_map_size=(
            cfg.bc_signal_ambiguous_target_decision_min_map_size
        ),
        bc_signal_ambiguous_target_search_labels=(
            cfg.bc_signal_ambiguous_target_search_labels
        ),
        bc_signal_ambiguous_target_search_min_map_size=(
            cfg.bc_signal_ambiguous_target_search_min_map_size
        ),
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
        bc_signal_clue_interact_action_weight=(
            cfg.bc_signal_clue_interact_action_weight
        ),
        bc_signal_clue_interact_min_map_size=cfg.bc_signal_clue_interact_min_map_size,
        bc_signal_evidence_sweep_action_weight=(
            cfg.bc_signal_evidence_sweep_action_weight
        ),
        bc_signal_evidence_sweep_min_map_size=(
            cfg.bc_signal_evidence_sweep_min_map_size
        ),
        bc_signal_frontier_exploration_action_weight=(
            cfg.bc_signal_frontier_exploration_action_weight
        ),
        bc_signal_frontier_exploration_min_map_size=(
            cfg.bc_signal_frontier_exploration_min_map_size
        ),
        bc_pipeline_pickup_action_loss_weight=cfg.bc_pipeline_pickup_action_loss_weight,
        bc_pipeline_delivery_action_loss_weight=cfg.bc_pipeline_delivery_action_loss_weight,
        bc_pipeline_delivery_progress_action_loss_weight=(
            cfg.bc_pipeline_delivery_progress_action_loss_weight
        ),
        bc_pipeline_navigation_action_loss_weight=(
            cfg.bc_pipeline_navigation_action_loss_weight
        ),
        bc_pipeline_frontier_exploration_action_loss_weight=(
            cfg.bc_pipeline_frontier_exploration_action_loss_weight
        ),
        bc_pipeline_frontier_exploration_min_map_size=(
            cfg.bc_pipeline_frontier_exploration_min_map_size
        ),
        bc_pipeline_sync_action_loss_weight=cfg.bc_pipeline_sync_action_loss_weight,
        bc_pipeline_ready_interact_action_loss_weight=(
            cfg.bc_pipeline_ready_interact_action_loss_weight
        ),
        bc_pipeline_station_guard_action_loss_weight=(
            cfg.bc_pipeline_station_guard_action_loss_weight
        ),
        bc_pipeline_pickup_gate_loss_weight=cfg.bc_pipeline_pickup_gate_loss_weight,
        bc_pipeline_pickup_gate_pos_weight=cfg.bc_pipeline_pickup_gate_pos_weight,
        bc_pipeline_pickup_gate_neg_weight=cfg.bc_pipeline_pickup_gate_neg_weight,
        bc_pipeline_plan_action_loss_weight=cfg.bc_pipeline_plan_action_loss_weight,
        bc_pipeline_plan_head_loss_weight=cfg.bc_pipeline_plan_head_loss_weight,
        bc_pipeline_option_loss_weight=cfg.bc_pipeline_option_loss_weight,
        bc_pipeline_message_loss_weight=cfg.bc_pipeline_message_loss_weight,
        bc_pipeline_send_gate_loss_weight=cfg.bc_pipeline_send_gate_loss_weight,
        bc_pipeline_send_gate_pos_weight=cfg.bc_pipeline_send_gate_pos_weight,
        bc_pipeline_send_gate_neg_weight=cfg.bc_pipeline_send_gate_neg_weight,
        bc_pipeline_interact_gate_loss_weight=cfg.bc_pipeline_interact_gate_loss_weight,
        bc_pipeline_interact_gate_pos_weight=cfg.bc_pipeline_interact_gate_pos_weight,
        bc_pipeline_interact_gate_neg_weight=cfg.bc_pipeline_interact_gate_neg_weight,
        bc_pipeline_bad_pickup_action_loss_weight=cfg.bc_pipeline_bad_pickup_action_loss_weight,
        bc_pipeline_bad_drop_action_loss_weight=cfg.bc_pipeline_bad_drop_action_loss_weight,
        bc_pipeline_bad_interact_action_loss_weight=cfg.bc_pipeline_bad_interact_action_loss_weight,
        bc_pipeline_proactive_bad_action_labels=cfg.bc_pipeline_proactive_bad_action_labels,
        dagger_rounds=cfg.dagger_rounds,
        dagger_episodes=cfg.dagger_episodes,
        dagger_seed_base=cfg.dagger_seed_base,
        dagger_seed_stride=cfg.dagger_seed_stride,
        dagger_seed_list=cfg.dagger_seed_list,
        dagger_retrain_from_scratch=(
            False if has_initial_model and cfg.carry_model_between_stages else cfg.dagger_retrain_from_scratch
        ),
        dagger_failed_episode_weight=cfg.dagger_failed_episode_weight,
        dagger_focus_events=cfg.dagger_focus_events,
        dagger_focus_error_weight=cfg.dagger_focus_error_weight,
        dagger_focus_recovery_weight=cfg.dagger_focus_recovery_weight,
        dagger_focus_window=cfg.dagger_focus_window,
        dagger_focus_replay=cfg.dagger_focus_replay,
        dagger_pipeline_wrong_delivery_provenance_labels=(
            cfg.dagger_pipeline_wrong_delivery_provenance_labels
        ),
        dagger_pipeline_wrong_delivery_provenance_weight=(
            cfg.dagger_pipeline_wrong_delivery_provenance_weight
        ),
        dagger_oracle_message_rollin_rate=cfg.dagger_oracle_message_rollin_rate,
        dagger_oracle_action_rollin_rate=cfg.dagger_oracle_action_rollin_rate,
        dagger_initial_target_broadcast_labels=cfg.dagger_initial_target_broadcast_labels,
        dagger_target_scan_broadcast_labels=cfg.dagger_target_scan_broadcast_labels,
        dagger_target_handoff_requires_exact_target=(
            cfg.dagger_target_handoff_requires_exact_target
        ),
        dagger_signal_target_rendezvous_labels=cfg.dagger_signal_target_rendezvous_labels,
        dagger_signal_target_rendezvous_min_map_size=(
            cfg.dagger_signal_target_rendezvous_min_map_size
        ),
        dagger_signal_target_rendezvous_max_agents=(
            cfg.dagger_signal_target_rendezvous_max_agents
        ),
        dagger_redundant_target_wait_labels=cfg.dagger_redundant_target_wait_labels,
        dagger_target_discovery_min_map_size=cfg.dagger_target_discovery_min_map_size,
        dagger_target_discovery_focus_weight=cfg.dagger_target_discovery_focus_weight,
        dagger_movement_stall_min_map_size=cfg.dagger_movement_stall_min_map_size,
        dagger_movement_stall_window=cfg.dagger_movement_stall_window,
        dagger_movement_stall_focus_weight=cfg.dagger_movement_stall_focus_weight,
        dagger_solo_target_team_weight=cfg.dagger_solo_target_team_weight,
        dagger_solo_target_team_success_only=cfg.dagger_solo_target_team_success_only,
        dagger_restore_best=cfg.dagger_restore_best,
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
        dagger_max_failed_parent_replay_snippets_per_episode=(
            cfg.dagger_max_failed_parent_replay_snippets_per_episode
        ),
        dagger_failed_parent_replay_weight_scale=cfg.dagger_failed_parent_replay_weight_scale,
        dagger_expert_max_replay_snippets_per_episode=cfg.dagger_expert_max_replay_snippets_per_episode,
        pipeline_assisted_rollout_episodes=cfg.pipeline_assisted_rollout_episodes,
        pipeline_assisted_rollout_seed_base=cfg.pipeline_assisted_rollout_seed_base,
        pipeline_assisted_rollout_seed_list=cfg.pipeline_assisted_rollout_seed_list,
        pipeline_assisted_rollout_max_steps_per_episode=(
            cfg.pipeline_assisted_rollout_max_steps_per_episode
        ),
        pipeline_assisted_rollout_weight=cfg.pipeline_assisted_rollout_weight,
        pipeline_assisted_rollout_success_only=cfg.pipeline_assisted_rollout_success_only,
        pipeline_assisted_rollout_navigation_assist=(
            cfg.pipeline_assisted_rollout_navigation_assist
        ),
        pipeline_assisted_rollout_navigation_assist_trust_messages=(
            cfg.pipeline_assisted_rollout_navigation_assist_trust_messages
        ),
        pipeline_assisted_rollout_station_interact_guard=(
            cfg.pipeline_assisted_rollout_station_interact_guard
        ),
        pipeline_assisted_rollout_bc_epochs=cfg.pipeline_assisted_rollout_bc_epochs,
        rl_updates=_stage_rl_updates(cfg, stage_idx),
        rl_early_stop_eval_patience=cfg.rl_early_stop_eval_patience,
        rollout_steps=cfg.rollout_steps,
        rl_balanced_rollouts=cfg.rl_balanced_rollouts,
        rl_rollout_map_steps=cfg.rl_rollout_map_steps,
        rl_rollout_eval_decoding=cfg.rl_rollout_eval_decoding,
        rl_rollout_pipeline_navigation_assist=cfg.rl_rollout_pipeline_navigation_assist,
        rl_rollout_pipeline_navigation_assist_trust_messages=(
            cfg.rl_rollout_pipeline_navigation_assist_trust_messages
        ),
        rl_rollout_pipeline_station_interact_guard=cfg.rl_rollout_pipeline_station_interact_guard,
        rl_rollout_pipeline_interact_gate_promote=cfg.rl_rollout_pipeline_interact_gate_promote,
        rl_eval_decoding_action_loss_weight=cfg.rl_eval_decoding_action_loss_weight,
        rl_pipeline_assisted_action_loss_weight=cfg.rl_pipeline_assisted_action_loss_weight,
        rl_pipeline_interact_gate_loss_weight=cfg.rl_pipeline_interact_gate_loss_weight,
        rl_pipeline_interact_gate_pos_weight=cfg.rl_pipeline_interact_gate_pos_weight,
        rl_pipeline_interact_gate_neg_weight=cfg.rl_pipeline_interact_gate_neg_weight,
        rl_pipeline_pickup_gate_loss_weight=cfg.rl_pipeline_pickup_gate_loss_weight,
        rl_pipeline_pickup_gate_pos_weight=cfg.rl_pipeline_pickup_gate_pos_weight,
        rl_pipeline_pickup_gate_neg_weight=cfg.rl_pipeline_pickup_gate_neg_weight,
        rl_pipeline_delivery_progress_action_loss_weight=(
            cfg.rl_pipeline_delivery_progress_action_loss_weight
        ),
        rl_pipeline_navigation_action_loss_weight=(
            cfg.rl_pipeline_navigation_action_loss_weight
        ),
        rl_pipeline_sync_action_loss_weight=cfg.rl_pipeline_sync_action_loss_weight,
        rl_pipeline_ready_interact_action_loss_weight=(
            cfg.rl_pipeline_ready_interact_action_loss_weight
        ),
        rl_pipeline_station_guard_action_loss_weight=(
            cfg.rl_pipeline_station_guard_action_loss_weight
        ),
        rl_pipeline_wrong_station_recovery_action_loss_weight=(
            cfg.rl_pipeline_wrong_station_recovery_action_loss_weight
        ),
        rl_pipeline_plan_action_loss_weight=cfg.rl_pipeline_plan_action_loss_weight,
        rl_pipeline_plan_head_loss_weight=cfg.rl_pipeline_plan_head_loss_weight,
        rl_pipeline_option_loss_weight=cfg.rl_pipeline_option_loss_weight,
        rl_redundant_target_scan_penalty=cfg.rl_redundant_target_scan_penalty,
        rl_wrong_target_scan_penalty=cfg.rl_wrong_target_scan_penalty,
        rl_pipeline_bad_pickup_penalty=cfg.rl_pipeline_bad_pickup_penalty,
        rl_pipeline_bad_interact_penalty=cfg.rl_pipeline_bad_interact_penalty,
        rl_pipeline_unneeded_drop_bonus=cfg.rl_pipeline_unneeded_drop_bonus,
        rl_epochs=cfg.rl_epochs,
        minibatch_seqs=cfg.minibatch_seqs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip=cfg.clip,
        value_clip=cfg.value_clip,
        entropy_coeff=cfg.entropy_coeff,
        rl_lr=cfg.rl_lr,
        max_grad_norm=cfg.max_grad_norm,
        bc_kl_coeff=cfg.bc_kl_coeff,
        bc_comm_kl_coeff=cfg.bc_comm_kl_coeff,
        rl_eval_every=cfg.rl_eval_every,
        rl_eval_episodes=cfg.rl_eval_episodes,
        rl_eval_use_eval_seeds=cfg.rl_eval_use_eval_seeds,
        rl_eval_seed=int(cfg.rl_eval_seed) + stage_idx * int(cfg.rl_eval_seed_stage_stride),
        rl_eval_seed_count=cfg.rl_eval_seed_count,
        rl_eval_seed_list=cfg.rl_eval_seed_list,
        rl_restore_best=cfg.rl_restore_best,
        rl_save_best=cfg.rl_save_best,
        rl_best_save=cfg.rl_best_save,
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
        eval_signal_exact_target_scan_lock=cfg.eval_signal_exact_target_scan_lock,
        eval_signal_compatible_target_scan_assist=cfg.eval_signal_compatible_target_scan_assist,
        eval_signal_compatible_target_scan_min_strength=cfg.eval_signal_compatible_target_scan_min_strength,
        eval_signal_negative_memory_scan_guard=cfg.eval_signal_negative_memory_scan_guard,
        eval_signal_target_probe_assist=cfg.eval_signal_target_probe_assist,
        eval_signal_scan_sync_assist=cfg.eval_signal_scan_sync_assist,
        eval_signal_scan_sync_force_first=cfg.eval_signal_scan_sync_force_first,
        eval_signal_scan_broadcast_assist=cfg.eval_signal_scan_broadcast_assist,
        eval_signal_constraint_message_copy_assist=cfg.eval_signal_constraint_message_copy_assist,
        eval_signal_constraint_message_guard=cfg.eval_signal_constraint_message_guard,
        eval_signal_exact_target_message_guard=cfg.eval_signal_exact_target_message_guard,
        eval_signal_initial_exact_message_copy_assist=(
            cfg.eval_signal_initial_exact_message_copy_assist
        ),
        eval_signal_exact_target_message_copy_assist=(
            cfg.eval_signal_exact_target_message_copy_assist
        ),
        eval_signal_exact_target_navigation_assist=cfg.eval_signal_exact_target_navigation_assist,
        eval_signal_exact_target_memory_steps=cfg.eval_signal_exact_target_memory_steps,
        eval_signal_scan_refresh_assist=cfg.eval_signal_scan_refresh_assist,
        eval_signal_scan_refresh_threshold=cfg.eval_signal_scan_refresh_threshold,
        eval_signal_evidence_sweep_assist=cfg.eval_signal_evidence_sweep_assist,
        eval_signal_evidence_sweep_min_step=cfg.eval_signal_evidence_sweep_min_step,
        eval_signal_frontier_exploration_assist=cfg.eval_signal_frontier_exploration_assist,
        eval_pipeline_navigation_assist=cfg.eval_pipeline_navigation_assist,
        eval_pipeline_navigation_assist_trust_messages=cfg.eval_pipeline_navigation_assist_trust_messages,
        eval_pipeline_station_interact_guard=cfg.eval_pipeline_station_interact_guard,
        eval_pipeline_plan_broadcast_assist=cfg.eval_pipeline_plan_broadcast_assist,
        eval_pipeline_pickup_gate_suppress=cfg.eval_pipeline_pickup_gate_suppress,
        eval_pipeline_frontier_exploration_assist=cfg.eval_pipeline_frontier_exploration_assist,
        eval_pipeline_interact_gate_threshold=cfg.eval_pipeline_interact_gate_threshold,
        eval_pipeline_interact_gate_promote=cfg.eval_pipeline_interact_gate_promote,
        eval_pipeline_event_head_threshold=cfg.eval_pipeline_event_head_threshold,
        eval_pipeline_navigation_head_threshold=cfg.eval_pipeline_navigation_head_threshold,
        eval_pipeline_plan_head_threshold=cfg.eval_pipeline_plan_head_threshold,
        eval_pipeline_option_threshold=cfg.eval_pipeline_option_threshold,
        eval_pipeline_option_allow_interact=cfg.eval_pipeline_option_allow_interact,
        save=str(checkpoint_path),
        recurrent_init_allow_obs_dim_mismatch=cfg.recurrent_init_allow_obs_dim_mismatch,
        seed=int(cfg.seed) + stage_idx,
        device=cfg.device,
        wandb=cfg.wandb,
        wandb_project=cfg.wandb_project,
        wandb_run=stage_wandb_run,
    )


def _stage_pipeline_settings(cfg: RecurrentCurriculumConfig, stage_idx: int) -> dict[str, Any]:
    return {
        "stage_count": _stage_optional_int_schedule_value(
            cfg.pipeline_stage_count_schedule,
            stage_idx,
            cfg.pipeline_stage_count,
        ),
        "required_per_stage_min": _stage_int_schedule_value(
            cfg.pipeline_required_per_stage_min_schedule,
            stage_idx,
            cfg.pipeline_required_per_stage_min,
        ),
        "required_per_stage_max": _stage_int_schedule_value(
            cfg.pipeline_required_per_stage_max_schedule,
            stage_idx,
            cfg.pipeline_required_per_stage_max,
        ),
        "sync_probability": _stage_float_schedule_value(
            cfg.pipeline_sync_probability_schedule,
            stage_idx,
            cfg.pipeline_sync_probability,
        ),
        "dependency_probability": _stage_float_schedule_value(
            cfg.pipeline_dependency_probability_schedule,
            stage_idx,
            cfg.pipeline_dependency_probability,
        ),
    }


def _validate_curriculum_map_sampling_weights(
    raw_value: str,
    stage_suites: list[tuple[int, ...]],
) -> None:
    weights = _parse_map_sampling_weights(
        raw_value,
        field_name="train_map_sampling_weights",
    )
    if not weights:
        return
    planned_maps = {int(size) for suite in stage_suites for size in suite}
    unknown_maps = sorted(set(weights) - planned_maps)
    if unknown_maps:
        raise ValueError(
            "train_map_sampling_weights contains map sizes not present in any curriculum stage: "
            f"{unknown_maps}"
        )


def _stage_train_map_sampling_weights(raw_value: str, suite: tuple[int, ...]) -> str:
    weights = _parse_map_sampling_weights(
        raw_value,
        field_name="train_map_sampling_weights",
    )
    if not weights:
        return ""
    entries = [
        f"{int(size)}:{int(weights[int(size)])}"
        for size in suite
        if int(size) in weights
    ]
    return ",".join(entries)


def _stage_rl_updates(cfg: RecurrentCurriculumConfig, stage_idx: int) -> int:
    return _stage_int_schedule_value(cfg.rl_updates_schedule, stage_idx, cfg.rl_updates)


def _stage_schedule_items(raw_value: str) -> list[str]:
    return [
        item.strip()
        for item in str(raw_value or "").replace(";", ",").split(",")
        if item.strip()
    ]


def _stage_schedule_value(
    raw_value: str,
    stage_idx: int,
    default: Any,
    caster,
    *,
    optional_none: bool = False,
) -> Any:
    items = _stage_schedule_items(raw_value)
    if not items:
        return default
    value = items[min(int(stage_idx), len(items) - 1)]
    if optional_none and value.lower() in {"none", "null", "default"}:
        return None
    return caster(value)


def _stage_optional_int_schedule_value(raw_value: str, stage_idx: int, default: int | None) -> int | None:
    return _stage_schedule_value(raw_value, stage_idx, default, int, optional_none=True)


def _stage_int_schedule_value(raw_value: str, stage_idx: int, default: int) -> int:
    return int(_stage_schedule_value(raw_value, stage_idx, default, int))


def _stage_float_schedule_value(raw_value: str, stage_idx: int, default: float) -> float:
    return float(_stage_schedule_value(raw_value, stage_idx, default, float))


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
    assisted_rollout_summary: dict[str, Any] | None = None,
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
            "pipeline_assisted_rollout": assisted_rollout_summary,
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
    parser = argparse.ArgumentParser(description="Staged recurrent SyncOrSink curriculum")
    parser.add_argument("--scenario", default="signal_hunt")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--fov-preset", choices=["hard", "medium", "easy"], default="easy")
    parser.add_argument("--oracle-type", default="signal_hint_comm")
    parser.add_argument("--stage-map-suites", default="8;8,16;8,16,32")
    parser.add_argument("--max-steps-by-map", default=RecurrentCurriculumConfig.max_steps_by_map)
    parser.add_argument("--train-map-sampling-weights", default="")
    parser.add_argument("--promotion-success-threshold", type=float, default=0.8)
    parser.add_argument("--stop-on-unmet-mastery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--carry-model-between-stages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demo-episodes", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--recurrent-backbone",
        choices=["mlp", "residual_mlp", "local_cnn"],
        default=RecurrentCurriculumConfig.recurrent_backbone,
        help="Recurrent actor encoder backbone for every stage.",
    )
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-lr", type=float, default=1e-4)
    parser.add_argument("--bc-seq-len", type=int, default=32)
    parser.add_argument("--bc-eval-every-epochs", type=int, default=0)
    parser.add_argument("--bc-eval-episodes", type=int, default=0)
    parser.add_argument("--bc-eval-seed-count", type=int, default=1)
    parser.add_argument("--bc-restore-best-eval-epoch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bc-equal-episode-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-action-class-balance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bc-action-class-balance-max-weight", type=float, default=5.0)
    parser.add_argument("--bc-event-action-weight", type=float, default=0.0)
    parser.add_argument(
        "--bc-event-action-events",
        default=RecurrentCurriculumConfig.bc_event_action_events,
    )
    parser.add_argument("--bc-comm-loss-weight", type=float, default=1.0)
    parser.add_argument("--bc-comm-send-pos-weight", type=float, default=5.0)
    parser.add_argument("--bc-comm-send-loss-weight", type=float, default=1.0)
    parser.add_argument("--bc-comm-length-loss-weight", type=float, default=1.0)
    parser.add_argument("--bc-comm-token-loss-weight", type=float, default=1.0)
    parser.add_argument("--bc-comm-send-rate-penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--bc-comm-send-rate-target",
        type=float,
        default=-1.0,
        help="Target send probability for BC send-rate penalty; negative matches the batch label rate",
    )
    parser.add_argument("--bc-calibrate-send-threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bc-send-threshold-target-rate", type=float, default=-1.0)
    parser.add_argument(
        "--bc-calibrate-pipeline-interact-gate-threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--bc-pipeline-interact-gate-threshold-target-rate", type=float, default=-1.0)
    parser.add_argument("--obs-exploration-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-signal-negative-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-exploration-age", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-normalize-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-memory-mode", choices=["full", "egocentric"], default="egocentric")
    parser.add_argument("--obs-memory-radius", type=int, default=4)
    parser.add_argument("--obs-navigation-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-pipeline-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-pipeline-feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-pipeline-feedback-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-pipeline-progress-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-pipeline-shared-feedback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-signal-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-signal-sync-feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-signal-scan-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-signal-negative-memory-window", type=int, default=64)
    parser.add_argument("--obs-signal-inferred-target-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-signal-target-match-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs-signal-confidence-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--obs-signal-sector-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--comm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--comm-token-limit", type=int, default=8)
    parser.add_argument("--comm-vocab-size", type=int, default=32)
    parser.add_argument("--comm-max-messages", type=int, default=8)
    parser.add_argument("--comm-cost", type=float, default=0.01)
    parser.add_argument("--comm-len-cost", type=float, default=0.0)
    parser.add_argument("--bc-signal-redundant-target-interact-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-target-pursuit-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-target-pursuit-action-weight", type=float, default=0.0)
    parser.add_argument(
        "--bc-signal-target-pursuit-trust-exact-memory",
        action=argparse.BooleanOptionalAction,
        default=RecurrentCurriculumConfig.bc_signal_target_pursuit_trust_exact_memory,
        help=(
            "Allow Signal target-pursuit action labels to use trusted exact target "
            "messages retained in scan-state memory."
        ),
    )
    parser.add_argument(
        "--bc-signal-target-pursuit-max-agents",
        type=int,
        default=RecurrentCurriculumConfig.bc_signal_target_pursuit_max_agents,
        help=(
            "Optional cap on how many closest agents receive Signal target-pursuit "
            "action labels at a step; 0 keeps all eligible agents."
        ),
    )
    parser.add_argument(
        "--bc-signal-constraint-frontier-bias",
        action=argparse.BooleanOptionalAction,
        default=RecurrentCurriculumConfig.bc_signal_constraint_frontier_bias,
        help=(
            "Bias Signal frontier action labels toward cells compatible with "
            "currently known target constraints."
        ),
    )
    parser.add_argument("--bc-signal-initial-message-weight", type=float, default=4.0)
    parser.add_argument("--bc-signal-initial-message-loss-weight", type=float, default=4.0)
    parser.add_argument("--bc-signal-constraint-message-loss-weight", type=float, default=4.0)
    parser.add_argument("--bc-signal-sync-response-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-sync-response-action-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--bc-signal-active-scan-response-action-weight",
        type=float,
        default=RecurrentCurriculumConfig.bc_signal_active_scan_response_action_weight,
        help=(
            "Opt-in Signal Hunt action loss for trusted target-informed agents to "
            "join/scan while a teammate target scan is active."
        ),
    )
    parser.add_argument(
        "--bc-signal-active-scan-response-min-map-size",
        type=int,
        default=RecurrentCurriculumConfig.bc_signal_active_scan_response_min_map_size,
    )
    parser.add_argument(
        "--bc-signal-active-scan-response-max-agents",
        type=int,
        default=RecurrentCurriculumConfig.bc_signal_active_scan_response_max_agents,
    )
    parser.add_argument(
        "--bc-signal-scan-bridge-action-weight",
        type=float,
        default=RecurrentCurriculumConfig.bc_signal_scan_bridge_action_weight,
    )
    parser.add_argument(
        "--bc-signal-scan-bridge-min-map-size",
        type=int,
        default=RecurrentCurriculumConfig.bc_signal_scan_bridge_min_map_size,
    )
    parser.add_argument(
        "--bc-signal-scan-bridge-remaining-threshold",
        type=float,
        default=RecurrentCurriculumConfig.bc_signal_scan_bridge_remaining_threshold,
    )
    parser.add_argument(
        "--bc-signal-scan-bridge-max-teammate-distance",
        type=int,
        default=RecurrentCurriculumConfig.bc_signal_scan_bridge_max_teammate_distance,
    )
    parser.add_argument("--bc-signal-target-aux-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-target-hypothesis-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--bc-signal-target-hypothesis-commit-loss-weight",
        type=float,
        default=RecurrentCurriculumConfig.bc_signal_target_hypothesis_commit_loss_weight,
    )
    parser.add_argument(
        "--bc-signal-target-hypothesis-ambiguity-loss-weight",
        type=float,
        default=RecurrentCurriculumConfig.bc_signal_target_hypothesis_ambiguity_loss_weight,
    )
    parser.add_argument(
        "--bc-signal-target-hypothesis-xy-loss-weight",
        type=float,
        default=RecurrentCurriculumConfig.bc_signal_target_hypothesis_xy_loss_weight,
    )
    parser.add_argument("--bc-signal-target-hypothesis-min-map-size", type=int, default=16)
    parser.add_argument("--bc-signal-target-match-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-first-target-scan-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-refresh-target-scan-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-joint-target-scan-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-target-opportunity-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-redundant-target-wait-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-scan-decision-loss-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-scan-decision-pos-weight", type=float, default=2.0)
    parser.add_argument("--bc-signal-scan-decision-neg-weight", type=float, default=3.0)
    parser.add_argument("--bc-signal-scan-gate-loss-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-scan-gate-pos-weight", type=float, default=2.0)
    parser.add_argument("--bc-signal-scan-gate-neg-weight", type=float, default=3.0)
    parser.add_argument("--bc-signal-target-validity-loss-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-target-validity-pos-weight", type=float, default=2.0)
    parser.add_argument("--bc-signal-target-validity-neg-weight", type=float, default=3.0)
    parser.add_argument("--bc-signal-target-decision-loss-weight", type=float, default=1.0)
    parser.add_argument("--bc-signal-target-decision-pos-weight", type=float, default=2.0)
    parser.add_argument("--bc-signal-target-decision-neg-weight", type=float, default=3.0)
    parser.add_argument(
        "--bc-signal-ambiguous-target-decision-negatives",
        action="store_true",
        default=False,
        help=(
            "Opt-in Signal ablation: label true target scans as negative target "
            "decisions when local constraints still allow multiple target hypotheses."
        ),
    )
    parser.add_argument("--bc-signal-ambiguous-target-decision-min-map-size", type=int, default=16)
    parser.add_argument(
        "--bc-signal-ambiguous-target-search-labels",
        action="store_true",
        default=False,
        help=(
            "Opt-in Signal ablation: keep clue/frontier search labels active "
            "while standing on locally ambiguous target hypotheses."
        ),
    )
    parser.add_argument("--bc-signal-ambiguous-target-search-min-map-size", type=int, default=16)
    parser.add_argument("--bc-signal-rejected-target-interact-loss-weight", type=float, default=0.05)
    parser.add_argument("--bc-signal-rejected-target-interact-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-bad-redundant-target-interact-loss-weight", type=float, default=0.05)
    parser.add_argument("--bc-signal-decoy-drift-action-loss-weight", type=float, default=0.25)
    parser.add_argument("--bc-signal-decoy-scan-action-loss-weight", type=float, default=0.1)
    parser.add_argument("--bc-signal-rejected-target-drift-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-clue-interact-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-clue-interact-min-map-size", type=int, default=16)
    parser.add_argument("--bc-signal-evidence-sweep-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-evidence-sweep-min-map-size", type=int, default=16)
    parser.add_argument("--bc-signal-frontier-exploration-action-weight", type=float, default=0.0)
    parser.add_argument("--bc-signal-frontier-exploration-min-map-size", type=int, default=16)
    parser.add_argument("--bc-pipeline-pickup-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-delivery-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-delivery-progress-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-navigation-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-frontier-exploration-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-frontier-exploration-min-map-size", type=int, default=8)
    parser.add_argument("--bc-pipeline-sync-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-ready-interact-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-station-guard-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-pickup-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-pickup-gate-pos-weight", type=float, default=1.0)
    parser.add_argument("--bc-pipeline-pickup-gate-neg-weight", type=float, default=1.0)
    parser.add_argument("--bc-pipeline-plan-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-plan-head-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-option-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-message-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-send-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-send-gate-pos-weight", type=float, default=1.0)
    parser.add_argument("--bc-pipeline-send-gate-neg-weight", type=float, default=1.0)
    parser.add_argument("--bc-pipeline-interact-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-interact-gate-pos-weight", type=float, default=1.0)
    parser.add_argument("--bc-pipeline-interact-gate-neg-weight", type=float, default=1.0)
    parser.add_argument("--bc-pipeline-bad-pickup-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-bad-drop-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--bc-pipeline-bad-interact-action-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--bc-pipeline-proactive-bad-action-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--pipeline-stage-count", type=int, default=None)
    parser.add_argument("--pipeline-required-per-stage-min", type=int, default=1)
    parser.add_argument("--pipeline-required-per-stage-max", type=int, default=2)
    parser.add_argument("--pipeline-sync-probability", type=float, default=0.5)
    parser.add_argument("--pipeline-dependency-probability", type=float, default=0.7)
    parser.add_argument("--pipeline-wrong-delivery-penalty", type=float, default=0.25)
    parser.add_argument(
        "--pipeline-stage-count-schedule",
        default="",
        help=(
            "Comma- or semicolon-separated per-curriculum-stage override. "
            "Use none/null/default to keep procedural default stage counts."
        ),
    )
    parser.add_argument("--pipeline-required-per-stage-min-schedule", default="")
    parser.add_argument("--pipeline-required-per-stage-max-schedule", default="")
    parser.add_argument("--pipeline-sync-probability-schedule", default="")
    parser.add_argument("--pipeline-dependency-probability-schedule", default="")
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--dagger-episodes", type=int, default=16)
    parser.add_argument("--dagger-seed-base", type=int, default=10000)
    parser.add_argument("--dagger-seed-stride", type=int, default=1000)
    parser.add_argument(
        "--dagger-seed-list",
        default="",
        help=(
            "Optional comma-separated environment reset seeds for DAgger collection; "
            "when set, episodes cycle through this explicit list"
        ),
    )
    parser.add_argument("--dagger-focus-error-weight", type=float, default=3.0)
    parser.add_argument(
        "--dagger-focus-events",
        default=RecurrentCurriculumConfig.dagger_focus_events,
        help="Comma-separated trajectory event labels to upweight during DAgger correction.",
    )
    parser.add_argument("--dagger-focus-recovery-weight", type=float, default=2.0)
    parser.add_argument("--dagger-focus-window", type=int, default=1)
    parser.add_argument(
        "--dagger-pipeline-wrong-delivery-provenance-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dagger-pipeline-wrong-delivery-provenance-weight",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--dagger-oracle-message-rollin-rate", type=float, default=0.0)
    parser.add_argument("--dagger-oracle-action-rollin-rate", type=float, default=0.0)
    parser.add_argument(
        "--dagger-initial-target-broadcast-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Label step-0 Signal Hunt agents with unambiguous private exact target hints to broadcast [26, x, y]",
    )
    parser.add_argument("--dagger-target-scan-broadcast-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--dagger-target-handoff-requires-exact-target",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dagger-signal-target-rendezvous-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dagger-signal-target-rendezvous-min-map-size", type=int, default=16)
    parser.add_argument("--dagger-signal-target-rendezvous-max-agents", type=int, default=2)
    parser.add_argument("--dagger-redundant-target-wait-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dagger-target-discovery-min-map-size", type=int, default=16)
    parser.add_argument("--dagger-target-discovery-focus-weight", type=float, default=3.0)
    parser.add_argument("--dagger-movement-stall-min-map-size", type=int, default=16)
    parser.add_argument("--dagger-movement-stall-window", type=int, default=6)
    parser.add_argument("--dagger-movement-stall-focus-weight", type=float, default=4.0)
    parser.add_argument("--dagger-solo-target-team-weight", type=float, default=1.0)
    parser.add_argument("--dagger-solo-target-team-success-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dagger-restore-best", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--dagger-max-failed-parent-replay-snippets-per-episode", type=int, default=-1)
    parser.add_argument("--dagger-failed-parent-replay-weight-scale", type=float, default=1.0)
    parser.add_argument("--dagger-expert-max-replay-snippets-per-episode", type=int, default=-1)
    parser.add_argument("--pipeline-assisted-rollout-episodes", type=int, default=0)
    parser.add_argument("--pipeline-assisted-rollout-seed-base", type=int, default=20000)
    parser.add_argument("--pipeline-assisted-rollout-seed-list", default="")
    parser.add_argument("--pipeline-assisted-rollout-max-steps-per-episode", type=int, default=0)
    parser.add_argument("--pipeline-assisted-rollout-weight", type=float, default=1.0)
    parser.add_argument(
        "--pipeline-assisted-rollout-success-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pipeline-assisted-rollout-navigation-assist",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pipeline-assisted-rollout-navigation-assist-trust-messages",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pipeline-assisted-rollout-station-interact-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pipeline-assisted-rollout-bc-epochs", type=int, default=-1)
    parser.add_argument("--rl-updates", type=int, default=0)
    parser.add_argument(
        "--rl-updates-schedule",
        default="",
        help=(
            "Comma- or semicolon-separated per-curriculum-stage PPO update override. "
            "For example, 0,0,60 skips PPO on the first two stages and fine-tunes the third."
        ),
    )
    parser.add_argument(
        "--rl-early-stop-eval-patience",
        type=int,
        default=0,
        help=(
            "Stop recurrent PPO within a stage after this many eval checkpoints fail to improve. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--rl-balanced-rollouts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rl-rollout-map-steps", default="")
    parser.add_argument("--rl-rollout-eval-decoding", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rl-rollout-pipeline-navigation-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rl-rollout-pipeline-navigation-assist-trust-messages",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--rl-rollout-pipeline-station-interact-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--rl-rollout-pipeline-interact-gate-promote",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--rl-eval-decoding-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--rl-pipeline-assisted-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--rl-pipeline-interact-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--rl-pipeline-interact-gate-pos-weight", type=float, default=1.0)
    parser.add_argument("--rl-pipeline-interact-gate-neg-weight", type=float, default=1.0)
    parser.add_argument("--rl-pipeline-pickup-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--rl-pipeline-pickup-gate-pos-weight", type=float, default=1.0)
    parser.add_argument("--rl-pipeline-pickup-gate-neg-weight", type=float, default=1.0)
    parser.add_argument(
        "--rl-pipeline-delivery-progress-action-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--rl-pipeline-navigation-action-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--rl-pipeline-sync-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--rl-pipeline-ready-interact-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--rl-pipeline-station-guard-action-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--rl-pipeline-wrong-station-recovery-action-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--rl-pipeline-plan-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--rl-pipeline-plan-head-loss-weight", type=float, default=0.0)
    parser.add_argument("--rl-pipeline-option-loss-weight", type=float, default=0.0)
    parser.add_argument("--rl-redundant-target-scan-penalty", type=float, default=0.0)
    parser.add_argument("--rl-wrong-target-scan-penalty", type=float, default=0.0)
    parser.add_argument("--rl-pipeline-bad-pickup-penalty", type=float, default=0.0)
    parser.add_argument("--rl-pipeline-bad-interact-penalty", type=float, default=0.0)
    parser.add_argument("--rl-pipeline-unneeded-drop-bonus", type=float, default=0.0)
    parser.add_argument("--rl-epochs", type=int, default=2)
    parser.add_argument("--minibatch-seqs", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--rl-lr", type=float, default=3e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--bc-kl-coeff", type=float, default=0.5)
    parser.add_argument("--bc-comm-kl-coeff", type=float, default=0.5)
    parser.add_argument("--rl-eval-every", type=int, default=5)
    parser.add_argument("--rl-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--rl-eval-use-eval-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use each curriculum stage's main eval seed/list for PPO best-checkpoint selection. "
            "Disable to use --rl-eval-seed/--rl-eval-seed-list instead."
        ),
    )
    parser.add_argument("--rl-eval-seed", type=int, default=10000)
    parser.add_argument(
        "--rl-eval-seed-stage-stride",
        type=int,
        default=100000,
        help=(
            "Added to --rl-eval-seed for each curriculum stage. Use 0 to reuse the same PPO "
            "eval seed base on every stage."
        ),
    )
    parser.add_argument("--rl-eval-seed-count", type=int, default=1)
    parser.add_argument("--rl-eval-seed-list", default="")
    parser.add_argument("--rl-restore-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl-save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl-best-save", default=None)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--eval-seed", type=int, default=3000)
    parser.add_argument("--eval-seed-count", type=int, default=2)
    parser.add_argument("--eval-send-threshold", type=float, default=None)
    parser.add_argument("--eval-signal-target-scan-threshold", type=float, default=-1.0)
    parser.add_argument("--eval-signal-scan-gate-threshold", type=float, default=0.4)
    parser.add_argument("--eval-signal-scan-gate-suppress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-signal-target-validity-threshold", type=float, default=0.4)
    parser.add_argument("--eval-signal-target-decision-threshold", type=float, default=0.4)
    parser.add_argument(
        "--eval-signal-target-decision-suppress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-signal-exact-target-scan-lock", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--eval-signal-compatible-target-scan-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eval-signal-compatible-target-scan-min-strength", type=int, default=3)
    parser.add_argument(
        "--eval-signal-negative-memory-scan-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eval-signal-target-probe-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eval-signal-scan-sync-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-signal-scan-sync-force-first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-signal-scan-broadcast-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-signal-constraint-message-copy-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-signal-constraint-message-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-signal-exact-target-message-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--eval-signal-initial-exact-message-copy-assist",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--eval-signal-exact-target-message-copy-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eval-signal-exact-target-navigation-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eval-signal-exact-target-memory-steps", type=int, default=0)
    parser.add_argument("--eval-signal-scan-refresh-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-signal-scan-refresh-threshold", type=float, default=0.5)
    parser.add_argument(
        "--eval-signal-evidence-sweep-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eval-signal-evidence-sweep-min-step", type=int, default=40)
    parser.add_argument(
        "--eval-signal-frontier-exploration-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eval-pipeline-navigation-assist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--eval-pipeline-navigation-assist-trust-messages",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eval-pipeline-station-interact-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eval-pipeline-pickup-gate-suppress",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eval-pipeline-plan-broadcast-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--eval-pipeline-frontier-exploration-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eval-pipeline-interact-gate-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--eval-pipeline-interact-gate-promote",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eval-pipeline-event-head-threshold", type=float, default=-1.0)
    parser.add_argument("--eval-pipeline-navigation-head-threshold", type=float, default=-1.0)
    parser.add_argument("--eval-pipeline-plan-head-threshold", type=float, default=-1.0)
    parser.add_argument("--eval-pipeline-option-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--eval-pipeline-option-allow-interact",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-dir", default="logs/recurrent_curriculum")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--initial-recurrent-checkpoint", default=None)
    parser.add_argument(
        "--recurrent-init-allow-obs-dim-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
    args = parser.parse_args()

    cfg = RecurrentCurriculumConfig(
        scenario=args.scenario,
        agents=args.agents,
        fov_preset=args.fov_preset,
        oracle_type=args.oracle_type,
        stage_map_suites=args.stage_map_suites,
        max_steps_by_map=args.max_steps_by_map,
        train_map_sampling_weights=args.train_map_sampling_weights,
        promotion_success_threshold=args.promotion_success_threshold,
        stop_on_unmet_mastery=args.stop_on_unmet_mastery,
        carry_model_between_stages=args.carry_model_between_stages,
        demo_episodes=args.demo_episodes,
        hidden_dim=args.hidden_dim,
        recurrent_backbone=args.recurrent_backbone,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
        bc_seq_len=args.bc_seq_len,
        bc_eval_every_epochs=args.bc_eval_every_epochs,
        bc_eval_episodes=args.bc_eval_episodes,
        bc_eval_seed_count=args.bc_eval_seed_count,
        bc_restore_best_eval_epoch=args.bc_restore_best_eval_epoch,
        bc_equal_episode_weight=args.bc_equal_episode_weight,
        bc_action_class_balance=args.bc_action_class_balance,
        bc_action_class_balance_max_weight=args.bc_action_class_balance_max_weight,
        bc_event_action_weight=args.bc_event_action_weight,
        bc_event_action_events=args.bc_event_action_events,
        bc_comm_loss_weight=args.bc_comm_loss_weight,
        bc_comm_send_pos_weight=args.bc_comm_send_pos_weight,
        bc_comm_send_loss_weight=args.bc_comm_send_loss_weight,
        bc_comm_length_loss_weight=args.bc_comm_length_loss_weight,
        bc_comm_token_loss_weight=args.bc_comm_token_loss_weight,
        bc_comm_send_rate_penalty_weight=args.bc_comm_send_rate_penalty_weight,
        bc_comm_send_rate_target=args.bc_comm_send_rate_target,
        bc_calibrate_send_threshold=args.bc_calibrate_send_threshold,
        bc_send_threshold_target_rate=args.bc_send_threshold_target_rate,
        bc_calibrate_pipeline_interact_gate_threshold=(
            args.bc_calibrate_pipeline_interact_gate_threshold
        ),
        bc_pipeline_interact_gate_threshold_target_rate=(
            args.bc_pipeline_interact_gate_threshold_target_rate
        ),
        obs_exploration_memory=args.obs_exploration_memory,
        obs_exploration_age=args.obs_exploration_age,
        obs_feedback=args.obs_feedback,
        obs_normalize_tokens=args.obs_normalize_tokens,
        obs_memory_mode=args.obs_memory_mode,
        obs_memory_radius=args.obs_memory_radius,
        obs_navigation_features=args.obs_navigation_features,
        obs_pipeline_features=args.obs_pipeline_features,
        obs_pipeline_feedback=args.obs_pipeline_feedback,
        obs_pipeline_feedback_metadata=args.obs_pipeline_feedback_metadata,
        obs_pipeline_progress_features=args.obs_pipeline_progress_features,
        obs_pipeline_shared_feedback=args.obs_pipeline_shared_feedback,
        obs_signal_features=args.obs_signal_features,
        obs_signal_sync_feedback=args.obs_signal_sync_feedback,
        obs_signal_scan_state=args.obs_signal_scan_state,
        obs_signal_negative_memory=args.obs_signal_negative_memory,
        obs_signal_negative_memory_window=args.obs_signal_negative_memory_window,
        obs_signal_inferred_target_features=args.obs_signal_inferred_target_features,
        obs_signal_target_match_features=args.obs_signal_target_match_features,
        obs_signal_confidence_features=args.obs_signal_confidence_features,
        obs_signal_sector_features=args.obs_signal_sector_features,
        comm=args.comm,
        comm_token_limit=args.comm_token_limit,
        comm_vocab_size=args.comm_vocab_size,
        comm_max_messages=args.comm_max_messages,
        comm_cost=args.comm_cost,
        comm_len_cost=args.comm_len_cost,
        bc_signal_redundant_target_interact_weight=args.bc_signal_redundant_target_interact_weight,
        bc_signal_target_pursuit_weight=args.bc_signal_target_pursuit_weight,
        bc_signal_target_pursuit_action_weight=args.bc_signal_target_pursuit_action_weight,
        bc_signal_target_pursuit_trust_exact_memory=(
            args.bc_signal_target_pursuit_trust_exact_memory
        ),
        bc_signal_target_pursuit_max_agents=args.bc_signal_target_pursuit_max_agents,
        bc_signal_constraint_frontier_bias=args.bc_signal_constraint_frontier_bias,
        bc_signal_initial_message_weight=args.bc_signal_initial_message_weight,
        bc_signal_initial_message_loss_weight=args.bc_signal_initial_message_loss_weight,
        bc_signal_constraint_message_loss_weight=args.bc_signal_constraint_message_loss_weight,
        bc_signal_sync_response_weight=args.bc_signal_sync_response_weight,
        bc_signal_sync_response_action_loss_weight=args.bc_signal_sync_response_action_loss_weight,
        bc_signal_active_scan_response_action_weight=(
            args.bc_signal_active_scan_response_action_weight
        ),
        bc_signal_active_scan_response_min_map_size=(
            args.bc_signal_active_scan_response_min_map_size
        ),
        bc_signal_active_scan_response_max_agents=(
            args.bc_signal_active_scan_response_max_agents
        ),
        bc_signal_scan_bridge_action_weight=args.bc_signal_scan_bridge_action_weight,
        bc_signal_scan_bridge_min_map_size=args.bc_signal_scan_bridge_min_map_size,
        bc_signal_scan_bridge_remaining_threshold=(
            args.bc_signal_scan_bridge_remaining_threshold
        ),
        bc_signal_scan_bridge_max_teammate_distance=(
            args.bc_signal_scan_bridge_max_teammate_distance
        ),
        bc_signal_target_aux_weight=args.bc_signal_target_aux_weight,
        bc_signal_target_hypothesis_loss_weight=(
            args.bc_signal_target_hypothesis_loss_weight
        ),
        bc_signal_target_hypothesis_commit_loss_weight=(
            args.bc_signal_target_hypothesis_commit_loss_weight
        ),
        bc_signal_target_hypothesis_ambiguity_loss_weight=(
            args.bc_signal_target_hypothesis_ambiguity_loss_weight
        ),
        bc_signal_target_hypothesis_xy_loss_weight=(
            args.bc_signal_target_hypothesis_xy_loss_weight
        ),
        bc_signal_target_hypothesis_min_map_size=(
            args.bc_signal_target_hypothesis_min_map_size
        ),
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
        bc_signal_ambiguous_target_decision_negatives=(
            args.bc_signal_ambiguous_target_decision_negatives
        ),
        bc_signal_ambiguous_target_decision_min_map_size=(
            args.bc_signal_ambiguous_target_decision_min_map_size
        ),
        bc_signal_ambiguous_target_search_labels=(
            args.bc_signal_ambiguous_target_search_labels
        ),
        bc_signal_ambiguous_target_search_min_map_size=(
            args.bc_signal_ambiguous_target_search_min_map_size
        ),
        bc_signal_rejected_target_interact_loss_weight=args.bc_signal_rejected_target_interact_loss_weight,
        bc_signal_rejected_target_interact_action_loss_weight=(
            args.bc_signal_rejected_target_interact_action_loss_weight
        ),
        bc_signal_bad_redundant_target_interact_loss_weight=args.bc_signal_bad_redundant_target_interact_loss_weight,
        bc_signal_decoy_drift_action_loss_weight=args.bc_signal_decoy_drift_action_loss_weight,
        bc_signal_decoy_scan_action_loss_weight=args.bc_signal_decoy_scan_action_loss_weight,
        bc_signal_rejected_target_drift_action_loss_weight=args.bc_signal_rejected_target_drift_action_loss_weight,
        bc_signal_clue_interact_action_weight=args.bc_signal_clue_interact_action_weight,
        bc_signal_clue_interact_min_map_size=args.bc_signal_clue_interact_min_map_size,
        bc_signal_evidence_sweep_action_weight=(
            args.bc_signal_evidence_sweep_action_weight
        ),
        bc_signal_evidence_sweep_min_map_size=(
            args.bc_signal_evidence_sweep_min_map_size
        ),
        bc_signal_frontier_exploration_action_weight=(
            args.bc_signal_frontier_exploration_action_weight
        ),
        bc_signal_frontier_exploration_min_map_size=(
            args.bc_signal_frontier_exploration_min_map_size
        ),
        bc_pipeline_pickup_action_loss_weight=args.bc_pipeline_pickup_action_loss_weight,
        bc_pipeline_delivery_action_loss_weight=args.bc_pipeline_delivery_action_loss_weight,
        bc_pipeline_delivery_progress_action_loss_weight=(
            args.bc_pipeline_delivery_progress_action_loss_weight
        ),
        bc_pipeline_navigation_action_loss_weight=(
            args.bc_pipeline_navigation_action_loss_weight
        ),
        bc_pipeline_frontier_exploration_action_loss_weight=(
            args.bc_pipeline_frontier_exploration_action_loss_weight
        ),
        bc_pipeline_frontier_exploration_min_map_size=(
            args.bc_pipeline_frontier_exploration_min_map_size
        ),
        bc_pipeline_sync_action_loss_weight=args.bc_pipeline_sync_action_loss_weight,
        bc_pipeline_ready_interact_action_loss_weight=(
            args.bc_pipeline_ready_interact_action_loss_weight
        ),
        bc_pipeline_station_guard_action_loss_weight=(
            args.bc_pipeline_station_guard_action_loss_weight
        ),
        bc_pipeline_pickup_gate_loss_weight=args.bc_pipeline_pickup_gate_loss_weight,
        bc_pipeline_pickup_gate_pos_weight=args.bc_pipeline_pickup_gate_pos_weight,
        bc_pipeline_pickup_gate_neg_weight=args.bc_pipeline_pickup_gate_neg_weight,
        bc_pipeline_plan_action_loss_weight=args.bc_pipeline_plan_action_loss_weight,
        bc_pipeline_plan_head_loss_weight=args.bc_pipeline_plan_head_loss_weight,
        bc_pipeline_option_loss_weight=args.bc_pipeline_option_loss_weight,
        bc_pipeline_message_loss_weight=args.bc_pipeline_message_loss_weight,
        bc_pipeline_send_gate_loss_weight=args.bc_pipeline_send_gate_loss_weight,
        bc_pipeline_send_gate_pos_weight=args.bc_pipeline_send_gate_pos_weight,
        bc_pipeline_send_gate_neg_weight=args.bc_pipeline_send_gate_neg_weight,
        bc_pipeline_interact_gate_loss_weight=args.bc_pipeline_interact_gate_loss_weight,
        bc_pipeline_interact_gate_pos_weight=args.bc_pipeline_interact_gate_pos_weight,
        bc_pipeline_interact_gate_neg_weight=args.bc_pipeline_interact_gate_neg_weight,
        bc_pipeline_bad_pickup_action_loss_weight=args.bc_pipeline_bad_pickup_action_loss_weight,
        bc_pipeline_bad_drop_action_loss_weight=args.bc_pipeline_bad_drop_action_loss_weight,
        bc_pipeline_bad_interact_action_loss_weight=args.bc_pipeline_bad_interact_action_loss_weight,
        bc_pipeline_proactive_bad_action_labels=args.bc_pipeline_proactive_bad_action_labels,
        pipeline_stage_count=args.pipeline_stage_count,
        pipeline_required_per_stage_min=args.pipeline_required_per_stage_min,
        pipeline_required_per_stage_max=args.pipeline_required_per_stage_max,
        pipeline_sync_probability=args.pipeline_sync_probability,
        pipeline_dependency_probability=args.pipeline_dependency_probability,
        pipeline_wrong_delivery_penalty=args.pipeline_wrong_delivery_penalty,
        pipeline_stage_count_schedule=args.pipeline_stage_count_schedule,
        pipeline_required_per_stage_min_schedule=args.pipeline_required_per_stage_min_schedule,
        pipeline_required_per_stage_max_schedule=args.pipeline_required_per_stage_max_schedule,
        pipeline_sync_probability_schedule=args.pipeline_sync_probability_schedule,
        pipeline_dependency_probability_schedule=args.pipeline_dependency_probability_schedule,
        dagger_rounds=args.dagger_rounds,
        dagger_episodes=args.dagger_episodes,
        dagger_seed_base=args.dagger_seed_base,
        dagger_seed_stride=args.dagger_seed_stride,
        dagger_seed_list=args.dagger_seed_list,
        dagger_focus_events=args.dagger_focus_events,
        dagger_focus_error_weight=args.dagger_focus_error_weight,
        dagger_focus_recovery_weight=args.dagger_focus_recovery_weight,
        dagger_focus_window=args.dagger_focus_window,
        dagger_pipeline_wrong_delivery_provenance_labels=(
            args.dagger_pipeline_wrong_delivery_provenance_labels
        ),
        dagger_pipeline_wrong_delivery_provenance_weight=(
            args.dagger_pipeline_wrong_delivery_provenance_weight
        ),
        dagger_oracle_message_rollin_rate=args.dagger_oracle_message_rollin_rate,
        dagger_oracle_action_rollin_rate=args.dagger_oracle_action_rollin_rate,
        dagger_initial_target_broadcast_labels=args.dagger_initial_target_broadcast_labels,
        dagger_target_scan_broadcast_labels=args.dagger_target_scan_broadcast_labels,
        dagger_target_handoff_requires_exact_target=(
            args.dagger_target_handoff_requires_exact_target
        ),
        dagger_signal_target_rendezvous_labels=args.dagger_signal_target_rendezvous_labels,
        dagger_signal_target_rendezvous_min_map_size=(
            args.dagger_signal_target_rendezvous_min_map_size
        ),
        dagger_signal_target_rendezvous_max_agents=(
            args.dagger_signal_target_rendezvous_max_agents
        ),
        dagger_redundant_target_wait_labels=args.dagger_redundant_target_wait_labels,
        dagger_target_discovery_min_map_size=args.dagger_target_discovery_min_map_size,
        dagger_target_discovery_focus_weight=args.dagger_target_discovery_focus_weight,
        dagger_movement_stall_min_map_size=args.dagger_movement_stall_min_map_size,
        dagger_movement_stall_window=args.dagger_movement_stall_window,
        dagger_movement_stall_focus_weight=args.dagger_movement_stall_focus_weight,
        dagger_solo_target_team_weight=args.dagger_solo_target_team_weight,
        dagger_solo_target_team_success_only=args.dagger_solo_target_team_success_only,
        dagger_restore_best=args.dagger_restore_best,
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
        dagger_max_failed_parent_replay_snippets_per_episode=(
            args.dagger_max_failed_parent_replay_snippets_per_episode
        ),
        dagger_failed_parent_replay_weight_scale=args.dagger_failed_parent_replay_weight_scale,
        dagger_expert_max_replay_snippets_per_episode=args.dagger_expert_max_replay_snippets_per_episode,
        pipeline_assisted_rollout_episodes=args.pipeline_assisted_rollout_episodes,
        pipeline_assisted_rollout_seed_base=args.pipeline_assisted_rollout_seed_base,
        pipeline_assisted_rollout_seed_list=args.pipeline_assisted_rollout_seed_list,
        pipeline_assisted_rollout_max_steps_per_episode=(
            args.pipeline_assisted_rollout_max_steps_per_episode
        ),
        pipeline_assisted_rollout_weight=args.pipeline_assisted_rollout_weight,
        pipeline_assisted_rollout_success_only=args.pipeline_assisted_rollout_success_only,
        pipeline_assisted_rollout_navigation_assist=(
            args.pipeline_assisted_rollout_navigation_assist
        ),
        pipeline_assisted_rollout_navigation_assist_trust_messages=(
            args.pipeline_assisted_rollout_navigation_assist_trust_messages
        ),
        pipeline_assisted_rollout_station_interact_guard=(
            args.pipeline_assisted_rollout_station_interact_guard
        ),
        pipeline_assisted_rollout_bc_epochs=args.pipeline_assisted_rollout_bc_epochs,
        rl_updates=args.rl_updates,
        rl_updates_schedule=args.rl_updates_schedule,
        rl_early_stop_eval_patience=args.rl_early_stop_eval_patience,
        rollout_steps=args.rollout_steps,
        rl_balanced_rollouts=args.rl_balanced_rollouts,
        rl_rollout_map_steps=args.rl_rollout_map_steps,
        rl_rollout_eval_decoding=args.rl_rollout_eval_decoding,
        rl_rollout_pipeline_navigation_assist=args.rl_rollout_pipeline_navigation_assist,
        rl_rollout_pipeline_navigation_assist_trust_messages=(
            args.rl_rollout_pipeline_navigation_assist_trust_messages
        ),
        rl_rollout_pipeline_station_interact_guard=args.rl_rollout_pipeline_station_interact_guard,
        rl_rollout_pipeline_interact_gate_promote=args.rl_rollout_pipeline_interact_gate_promote,
        rl_eval_decoding_action_loss_weight=args.rl_eval_decoding_action_loss_weight,
        rl_pipeline_assisted_action_loss_weight=(
            args.rl_pipeline_assisted_action_loss_weight
        ),
        rl_pipeline_interact_gate_loss_weight=args.rl_pipeline_interact_gate_loss_weight,
        rl_pipeline_interact_gate_pos_weight=args.rl_pipeline_interact_gate_pos_weight,
        rl_pipeline_interact_gate_neg_weight=args.rl_pipeline_interact_gate_neg_weight,
        rl_pipeline_pickup_gate_loss_weight=args.rl_pipeline_pickup_gate_loss_weight,
        rl_pipeline_pickup_gate_pos_weight=args.rl_pipeline_pickup_gate_pos_weight,
        rl_pipeline_pickup_gate_neg_weight=args.rl_pipeline_pickup_gate_neg_weight,
        rl_pipeline_delivery_progress_action_loss_weight=(
            args.rl_pipeline_delivery_progress_action_loss_weight
        ),
        rl_pipeline_navigation_action_loss_weight=(
            args.rl_pipeline_navigation_action_loss_weight
        ),
        rl_pipeline_sync_action_loss_weight=args.rl_pipeline_sync_action_loss_weight,
        rl_pipeline_ready_interact_action_loss_weight=(
            args.rl_pipeline_ready_interact_action_loss_weight
        ),
        rl_pipeline_station_guard_action_loss_weight=(
            args.rl_pipeline_station_guard_action_loss_weight
        ),
        rl_pipeline_wrong_station_recovery_action_loss_weight=(
            args.rl_pipeline_wrong_station_recovery_action_loss_weight
        ),
        rl_pipeline_plan_action_loss_weight=args.rl_pipeline_plan_action_loss_weight,
        rl_pipeline_plan_head_loss_weight=args.rl_pipeline_plan_head_loss_weight,
        rl_pipeline_option_loss_weight=args.rl_pipeline_option_loss_weight,
        rl_redundant_target_scan_penalty=args.rl_redundant_target_scan_penalty,
        rl_wrong_target_scan_penalty=args.rl_wrong_target_scan_penalty,
        rl_pipeline_bad_pickup_penalty=args.rl_pipeline_bad_pickup_penalty,
        rl_pipeline_bad_interact_penalty=args.rl_pipeline_bad_interact_penalty,
        rl_pipeline_unneeded_drop_bonus=args.rl_pipeline_unneeded_drop_bonus,
        rl_epochs=args.rl_epochs,
        minibatch_seqs=args.minibatch_seqs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        value_clip=args.value_clip,
        entropy_coeff=args.entropy_coeff,
        rl_lr=args.rl_lr,
        max_grad_norm=args.max_grad_norm,
        bc_kl_coeff=args.bc_kl_coeff,
        bc_comm_kl_coeff=args.bc_comm_kl_coeff,
        rl_eval_every=args.rl_eval_every,
        rl_eval_episodes=args.rl_eval_episodes,
        rl_eval_use_eval_seeds=args.rl_eval_use_eval_seeds,
        rl_eval_seed=args.rl_eval_seed,
        rl_eval_seed_stage_stride=args.rl_eval_seed_stage_stride,
        rl_eval_seed_count=args.rl_eval_seed_count,
        rl_eval_seed_list=args.rl_eval_seed_list,
        rl_restore_best=args.rl_restore_best,
        rl_save_best=args.rl_save_best,
        rl_best_save=args.rl_best_save,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        eval_seed_count=args.eval_seed_count,
        eval_send_threshold=args.eval_send_threshold,
        eval_signal_target_scan_threshold=args.eval_signal_target_scan_threshold,
        eval_signal_scan_gate_threshold=args.eval_signal_scan_gate_threshold,
        eval_signal_scan_gate_suppress=args.eval_signal_scan_gate_suppress,
        eval_signal_target_validity_threshold=args.eval_signal_target_validity_threshold,
        eval_signal_target_decision_threshold=args.eval_signal_target_decision_threshold,
        eval_signal_target_decision_suppress=args.eval_signal_target_decision_suppress,
        eval_signal_exact_target_scan_lock=args.eval_signal_exact_target_scan_lock,
        eval_signal_compatible_target_scan_assist=args.eval_signal_compatible_target_scan_assist,
        eval_signal_compatible_target_scan_min_strength=(
            args.eval_signal_compatible_target_scan_min_strength
        ),
        eval_signal_negative_memory_scan_guard=args.eval_signal_negative_memory_scan_guard,
        eval_signal_target_probe_assist=args.eval_signal_target_probe_assist,
        eval_signal_scan_sync_assist=args.eval_signal_scan_sync_assist,
        eval_signal_scan_sync_force_first=args.eval_signal_scan_sync_force_first,
        eval_signal_scan_broadcast_assist=args.eval_signal_scan_broadcast_assist,
        eval_signal_constraint_message_copy_assist=args.eval_signal_constraint_message_copy_assist,
        eval_signal_constraint_message_guard=args.eval_signal_constraint_message_guard,
        eval_signal_exact_target_message_guard=args.eval_signal_exact_target_message_guard,
        eval_signal_initial_exact_message_copy_assist=(
            args.eval_signal_initial_exact_message_copy_assist
        ),
        eval_signal_exact_target_message_copy_assist=(
            args.eval_signal_exact_target_message_copy_assist
        ),
        eval_signal_exact_target_navigation_assist=args.eval_signal_exact_target_navigation_assist,
        eval_signal_exact_target_memory_steps=args.eval_signal_exact_target_memory_steps,
        eval_signal_scan_refresh_assist=args.eval_signal_scan_refresh_assist,
        eval_signal_scan_refresh_threshold=args.eval_signal_scan_refresh_threshold,
        eval_signal_evidence_sweep_assist=args.eval_signal_evidence_sweep_assist,
        eval_signal_evidence_sweep_min_step=args.eval_signal_evidence_sweep_min_step,
        eval_signal_frontier_exploration_assist=args.eval_signal_frontier_exploration_assist,
        eval_pipeline_navigation_assist=args.eval_pipeline_navigation_assist,
        eval_pipeline_navigation_assist_trust_messages=args.eval_pipeline_navigation_assist_trust_messages,
        eval_pipeline_station_interact_guard=args.eval_pipeline_station_interact_guard,
        eval_pipeline_plan_broadcast_assist=args.eval_pipeline_plan_broadcast_assist,
        eval_pipeline_pickup_gate_suppress=args.eval_pipeline_pickup_gate_suppress,
        eval_pipeline_frontier_exploration_assist=args.eval_pipeline_frontier_exploration_assist,
        eval_pipeline_interact_gate_threshold=args.eval_pipeline_interact_gate_threshold,
        eval_pipeline_interact_gate_promote=args.eval_pipeline_interact_gate_promote,
        eval_pipeline_event_head_threshold=args.eval_pipeline_event_head_threshold,
        eval_pipeline_navigation_head_threshold=args.eval_pipeline_navigation_head_threshold,
        eval_pipeline_plan_head_threshold=args.eval_pipeline_plan_head_threshold,
        eval_pipeline_option_threshold=args.eval_pipeline_option_threshold,
        eval_pipeline_option_allow_interact=args.eval_pipeline_option_allow_interact,
        output_dir=args.output_dir,
        run_name=args.run_name,
        initial_recurrent_checkpoint=args.initial_recurrent_checkpoint,
        recurrent_init_allow_obs_dim_mismatch=args.recurrent_init_allow_obs_dim_mismatch,
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
