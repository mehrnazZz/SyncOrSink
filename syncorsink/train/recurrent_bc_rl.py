"""Recurrent BC→RL for pipeline assembly.

Uses LSTM-augmented actor to maintain state across steps within an episode.
This lets the policy track which stages are complete, what resources have
been delivered, and coordinate multi-step dependency chains.

Pipeline:
  1. Collect oracle demos with step-by-step hidden state
  2. Train recurrent BC via truncated BPTT
  3. Fine-tune with PPO (KL-regularized)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import warnings
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.envs.maps import (
    TILE_BEACON,
    TILE_CLUE,
    TILE_DOOR,
    TILE_NODE,
    TILE_RESOURCE,
    TILE_STATION,
    TILE_TARGET,
    TILE_WATER,
    TILE_WALL,
)
from syncorsink.eval.metrics import EpisodeStats, summarize
from syncorsink.eval.success import episode_success
from syncorsink.train.mappo import (
    action_mask_from_flat_obs,
    flatten_obs,
    mask_action_logits,
    resolve_device,
)
from syncorsink.policies.pathing import move_action_from_delta, shortest_path
from syncorsink.train.seed import set_global_seeds
from syncorsink.policies.mappo_models import MAPPORecurrentActor, MAPPOCritic


@dataclass
class RecurrentConfig:
    # environment
    scenario: str = "pipeline_assembly"
    map_size: int = 8
    train_map_sizes: str = ""
    train_map_sampling_weights: str = ""
    map_max_steps: str = ""
    agents: int = 3
    fov_preset: str = "easy"
    max_steps: int = 300
    energy_preset: str = "easy"
    oracle_type: str = "oracle_strong"
    obs_exploration_memory: bool = False
    obs_exploration_age: bool = False
    obs_feedback: bool = False
    obs_normalize_tokens: bool = False
    obs_memory_mode: str = "full"  # full | egocentric
    obs_memory_radius: int = 4
    obs_navigation_features: bool = False
    obs_signal_features: bool = False
    obs_signal_sync_feedback: bool = False
    obs_signal_scan_state: bool = False
    obs_signal_negative_memory: bool = False
    obs_signal_negative_memory_window: int = 64
    obs_signal_inferred_target_features: bool = False
    obs_signal_target_match_features: bool = False
    signal_decoy_count: Optional[int] = None
    decoy_penalty: float = 0.5
    scan_window: int = 3
    # shaping
    pipeline_shaping: bool = True
    pipeline_shaping_scale: float = 0.1
    energy_shaping: bool = False
    energy_shaping_scale: float = 0.01
    signal_shaping: bool = False
    signal_shaping_scale: float = 0.01
    signal_scan_bonus: float = 0.0
    signal_joint_scan_bonus: float = 0.0
    signal_colocation_bonus: float = 0.0
    signal_colocation_radius: int = 2
    signal_comm_utility: float = 0.0
    signal_target_visit_bonus: float = 0.0
    signal_decoy_visit_penalty: float = 0.0
    signal_unique_target_scan_bonus: float = 0.0
    # architecture
    hidden_dim: int = 128
    comm: bool = False
    comm_token_limit: int = 8
    comm_vocab_size: int = 32
    comm_max_messages: int = 8
    comm_len_cost: float = 0.0
    comm_cost: float = 0.01
    # BC
    demo_episodes: int = 200
    bc_epochs: int = 30
    bc_lr: float = 1e-3
    bc_seq_len: int = 32  # truncated BPTT sequence length
    bc_equal_episode_weight: bool = True
    bc_comm_loss_weight: float = 0.1
    bc_comm_send_pos_weight: float = 0.0  # negative = auto-balance send positives
    bc_comm_send_loss_weight: float = 1.0
    bc_comm_length_loss_weight: float = 1.0
    bc_comm_token_loss_weight: float = 1.0
    bc_comm_send_rate_penalty_weight: float = 0.0
    bc_comm_send_rate_target: float = -1.0  # negative = match current batch send-label rate
    bc_calibrate_send_threshold: bool = False
    bc_send_threshold_target_rate: float = -1.0  # negative = match dataset send-label rate
    bc_eval_every_epochs: int = 0
    bc_eval_episodes: int = 0  # <=0 uses eval_episodes
    bc_eval_seed_count: int = 1
    bc_restore_best_eval_epoch: bool = False
    bc_signal_target_interact_weight: float = 1.0
    bc_signal_redundant_target_interact_weight: float = 1.0
    bc_signal_target_pursuit_weight: float = 1.0
    bc_signal_target_pursuit_action_weight: float = 0.0
    bc_signal_sync_response_weight: float = 1.0
    bc_signal_sync_response_action_loss_weight: float = 0.0
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
    bc_signal_target_aux_weight: float = 0.0
    bc_signal_rejected_target_interact_loss_weight: float = 0.0
    bc_signal_rejected_target_interact_action_loss_weight: float = 0.0
    bc_signal_bad_redundant_target_interact_loss_weight: float = 0.0
    bc_signal_decoy_drift_action_loss_weight: float = 0.0
    bc_signal_decoy_scan_action_loss_weight: float = 0.0
    bc_signal_rejected_target_drift_action_loss_weight: float = 0.0
    dagger_rounds: int = 0
    dagger_episodes: int = 20
    dagger_seed_base: int = 10000
    dagger_seed_stride: int = 1000
    dagger_seed_list: str = ""
    dagger_retrain_from_scratch: bool = True
    dagger_max_steps_per_episode: int = 0
    dagger_success_episode_weight: float = 1.0
    dagger_failed_episode_weight: float = 0.25
    dagger_focus_events: str = (
        "decoy_scan,solo_target_scan,rejected_target_scan,"
        "bad_redundant_target_scan,target_interact_miss,target_pursuit_miss,"
        "target_decoy_drift_miss,target_discovery_miss,target_handoff_miss,"
        "movement_stall_miss"
    )
    dagger_focus_error_weight: float = 3.0
    dagger_focus_recovery_weight: float = 2.0
    dagger_focus_window: int = 1
    dagger_target_interact_focus_weight: float = 5.0
    dagger_target_discovery_min_map_size: int = 16
    dagger_target_discovery_focus_weight: float = 3.0
    dagger_movement_stall_min_map_size: int = 16
    dagger_movement_stall_window: int = 6
    dagger_movement_stall_focus_weight: float = 4.0
    dagger_target_decoy_drift_focus_weight: float = 5.0
    dagger_solo_target_team_weight: float = 1.0
    dagger_early_stop_patience: int = 0
    dagger_focus_replay: bool = False
    dagger_replay_pre_steps: int = 2
    dagger_replay_post_steps: int = 2
    dagger_replay_weight: float = 1.0
    dagger_positive_replay_events: str = ""
    dagger_replay_event_weights: str = ""
    dagger_replay_event_caps: str = ""
    dagger_replay_success_only_events: str = ""
    dagger_replay_priority_events: str = ""
    dagger_replay_balance_positive_events: str = ""
    dagger_replay_balance_negative_events: str = ""
    dagger_replay_max_negative_per_positive: float = -1.0
    dagger_max_replay_snippets_per_episode: int = 4
    dagger_expert_max_replay_snippets_per_episode: int = -1
    dagger_solo_target_team_success_only: bool = False
    dagger_positive_target_pursuit_min_map_size: int = 16
    dagger_redundant_target_wait_labels: bool = False
    dagger_target_scan_broadcast_labels: bool = False
    dagger_oracle_message_rollin_rate: float = 0.0
    # RL
    rl_updates: int = 3000
    rollout_steps: int = 256
    rl_balanced_rollouts: bool = False
    rl_rollout_map_steps: str = ""
    rl_redundant_target_scan_penalty: float = 0.0
    rl_wrong_target_scan_penalty: float = 0.0
    rl_epochs: int = 2
    minibatch_seqs: int = 8  # number of sequences per minibatch
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
    rl_eval_seed: int = 10000
    rl_eval_seed_count: int = 1
    rl_restore_best: bool = True
    rl_save_best: bool = True
    rl_best_save: Optional[str] = None
    recurrent_init: Optional[str] = None
    recurrent_init_for_dagger: bool = False
    recurrent_init_allow_obs_dim_mismatch: bool = False
    # device
    device: str = "auto"
    seed: Optional[int] = 0
    # output
    save: Optional[str] = None
    eval_episodes: int = 100
    eval_seed: int = 3000
    eval_seed_count: int = 1
    eval_map_sizes: str = ""
    eval_send_threshold: float = 0.25
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
    wandb: bool = False
    wandb_project: str = "syncorsink"
    wandb_run: Optional[str] = None


def _build_env(cfg: RecurrentConfig):
    return SyncOrSinkEnv(SyncOrSinkConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        num_agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        max_steps=cfg.max_steps,
        energy_preset=cfg.energy_preset,
        pipeline_shaping=cfg.pipeline_shaping,
        pipeline_shaping_scale=cfg.pipeline_shaping_scale,
        energy_shaping=cfg.energy_shaping,
        energy_shaping_scale=cfg.energy_shaping_scale,
        signal_decoy_count=cfg.signal_decoy_count,
        decoy_penalty=cfg.decoy_penalty,
        scan_window=cfg.scan_window,
        signal_shaping=cfg.signal_shaping,
        signal_shaping_scale=cfg.signal_shaping_scale,
        signal_scan_bonus=cfg.signal_scan_bonus,
        signal_joint_scan_bonus=cfg.signal_joint_scan_bonus,
        signal_colocation_bonus=cfg.signal_colocation_bonus,
        signal_colocation_radius=cfg.signal_colocation_radius,
        signal_comm_utility=cfg.signal_comm_utility,
        signal_target_visit_bonus=cfg.signal_target_visit_bonus,
        signal_decoy_visit_penalty=cfg.signal_decoy_visit_penalty,
        signal_unique_target_scan_bonus=cfg.signal_unique_target_scan_bonus,
        comm_token_limit=cfg.comm_token_limit,
        token_vocab_size=cfg.comm_vocab_size,
        max_messages=cfg.comm_max_messages,
        comm_len_cost=cfg.comm_len_cost,
        comm_cost=cfg.comm_cost,
        obs_exploration_memory=cfg.obs_exploration_memory,
        obs_exploration_age=cfg.obs_exploration_age,
    ))


def _parse_map_sizes(raw_value: str, default_size: int, *, field_name: str) -> list[int]:
    raw = str(raw_value or "").strip()
    if not raw:
        return [int(default_size)]
    sizes = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        size = int(item)
        if size <= 0:
            raise ValueError(f"{field_name} must contain positive integers, got {size}")
        sizes.append(size)
    return sizes or [int(default_size)]


def _parse_seed_list(raw_value: str, *, field_name: str) -> list[int]:
    raw = str(raw_value or "").strip()
    if not raw:
        return []
    seeds: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        seed = int(item)
        if seed < 0:
            raise ValueError(f"{field_name} must contain non-negative integers, got {seed}")
        seeds.append(seed)
    return seeds


def _parse_map_step_overrides(raw_value: str, *, field_name: str) -> dict[int, int]:
    raw = str(raw_value or "").strip()
    if not raw:
        return {}
    overrides: dict[int, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{field_name} entries must be map_size:max_steps pairs, got {item!r}")
        raw_size, raw_steps = item.split(":", 1)
        map_size = int(raw_size.strip())
        max_steps = int(raw_steps.strip())
        if map_size <= 0 or max_steps <= 0:
            raise ValueError(f"{field_name} entries must contain positive integers, got {item!r}")
        overrides[map_size] = max_steps
    return overrides


def _parse_map_sampling_weights(raw_value: str, *, field_name: str) -> dict[int, int]:
    raw = str(raw_value or "").strip()
    if not raw:
        return {}
    weights: dict[int, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{field_name} entries must be map_size:weight pairs, got {item!r}")
        raw_size, raw_weight = item.split(":", 1)
        map_size = int(raw_size.strip())
        weight = int(raw_weight.strip())
        if map_size <= 0 or weight <= 0:
            raise ValueError(f"{field_name} entries must contain positive integers, got {item!r}")
        weights[map_size] = weight
    return weights


def _training_map_sizes(cfg: RecurrentConfig) -> list[int]:
    return _parse_map_sizes(cfg.train_map_sizes, int(cfg.map_size), field_name="train_map_sizes")


def _training_map_schedule(cfg: RecurrentConfig) -> list[int]:
    sizes = _training_map_sizes(cfg)
    weights = _parse_map_sampling_weights(
        cfg.train_map_sampling_weights,
        field_name="train_map_sampling_weights",
    )
    if not weights:
        return sizes
    unknown_maps = sorted(set(weights) - set(sizes))
    if unknown_maps:
        raise ValueError(
            "train_map_sampling_weights contains map sizes not present in train_map_sizes: "
            f"{unknown_maps}"
        )
    schedule: list[int] = []
    for size in sizes:
        schedule.extend([int(size)] * int(weights.get(int(size), 1)))
    return schedule or sizes


def _eval_map_sizes(cfg: RecurrentConfig) -> list[int]:
    return _parse_map_sizes(cfg.eval_map_sizes, int(cfg.map_size), field_name="eval_map_sizes")


def _max_steps_for_map_size(cfg: RecurrentConfig, map_size: int) -> int:
    overrides = _parse_map_step_overrides(cfg.map_max_steps, field_name="map_max_steps")
    return int(overrides.get(int(map_size), int(cfg.max_steps)))


def _cfg_for_map_size(cfg: RecurrentConfig, map_size: int) -> RecurrentConfig:
    map_size = int(map_size)
    max_steps = _max_steps_for_map_size(cfg, map_size)
    if map_size == int(cfg.map_size) and max_steps == int(cfg.max_steps):
        return cfg
    return replace(cfg, map_size=map_size, max_steps=max_steps)


def _cfg_for_training_episode(cfg: RecurrentConfig, episode_idx: int) -> RecurrentConfig:
    sizes = _training_map_schedule(cfg)
    map_size = sizes[int(episode_idx) % len(sizes)]
    return _cfg_for_map_size(cfg, map_size)


def _build_training_env(cfg: RecurrentConfig, episode_idx: int) -> tuple[SyncOrSinkEnv, RecurrentConfig]:
    episode_cfg = _cfg_for_training_episode(cfg, episode_idx)
    return _build_env(episode_cfg), episode_cfg


def _warn_if_signal_hint_comm_channel_is_too_small(cfg: RecurrentConfig) -> None:
    if cfg.scenario != "signal_hunt" or cfg.oracle_type != "signal_hint_comm":
        return
    try:
        map_sizes = set(_training_map_sizes(cfg)) | set(_eval_map_sizes(cfg)) | {int(cfg.map_size)}
    except Exception:
        map_sizes = {int(cfg.map_size)}
    required_vocab = max(32, max(map_sizes))
    required_limit = 6
    if int(cfg.comm_token_limit) >= required_limit and int(cfg.comm_vocab_size) >= required_vocab:
        return
    warnings.warn(
        "signal_hint_comm uses structured clue messages; use comm_token_limit >= "
        f"{required_limit} and comm_vocab_size >= {required_vocab} for this map suite. "
        f"Current channel is comm_token_limit={cfg.comm_token_limit}, "
        f"comm_vocab_size={cfg.comm_vocab_size}, which will clip or alias oracle messages.",
        UserWarning,
        stacklevel=2,
    )


def _scale_nonnegative(values, denominator: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    denom = max(1.0, float(denominator))
    return np.where(arr >= 0.0, arr / denom, -1.0).astype(np.float32)


def _observed_map_size(obs_agent: dict, cfg: RecurrentConfig) -> int:
    for key in ("explored_mask", "explored_age"):
        arr = obs_agent.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim >= 2:
            return int(max(arr.shape[-2], arr.shape[-1]))
    return int(cfg.map_size)


def _egocentric_memory_patch(values, self_pos, radius: int, fill_value: float) -> np.ndarray:
    arr = np.asarray(values)
    radius = max(0, int(radius))
    size = radius * 2 + 1
    patch = np.full((size, size), fill_value, dtype=arr.dtype)
    if arr.ndim < 2:
        return patch
    pos = np.asarray(self_pos, dtype=np.int64).reshape(-1)
    if pos.size < 2:
        return patch
    cx, cy = int(pos[0]), int(pos[1])
    height, width = int(arr.shape[-2]), int(arr.shape[-1])
    for py, gy in enumerate(range(cy - radius, cy + radius + 1)):
        if gy < 0 or gy >= height:
            continue
        for px, gx in enumerate(range(cx - radius, cx + radius + 1)):
            if gx < 0 or gx >= width:
                continue
            patch[py, px] = arr[gy, gx]
    return patch


def _project_recurrent_memory(obs_agent: dict, cfg: RecurrentConfig) -> dict:
    if not cfg.obs_exploration_memory or cfg.obs_memory_mode == "full":
        return obs_agent
    if cfg.obs_memory_mode != "egocentric":
        raise ValueError(f"unsupported recurrent obs_memory_mode={cfg.obs_memory_mode!r}")
    projected = dict(obs_agent)
    self_pos = obs_agent.get("self_pos", np.zeros((2,), dtype=np.int16))
    if "explored_mask" in obs_agent:
        projected["explored_mask"] = _egocentric_memory_patch(
            obs_agent["explored_mask"],
            self_pos,
            cfg.obs_memory_radius,
            fill_value=0,
        )
    if cfg.obs_exploration_age and "explored_age" in obs_agent:
        projected["explored_age"] = _egocentric_memory_patch(
            obs_agent["explored_age"],
            self_pos,
            cfg.obs_memory_radius,
            fill_value=-1,
        )
    return projected


def _recurrent_local_grid_ids(local_grid) -> np.ndarray:
    arr = np.asarray(local_grid)
    if arr.ndim == 3:
        return np.argmax(arr, axis=0).astype(np.int16)
    return arr.astype(np.int16)


def _direction_features(dx: float, dy: float, denom: float, present: bool) -> list[float]:
    if not present:
        return [0.0, 0.0, 0.0, 1.0]
    denom = max(1.0, float(denom))
    dist = abs(float(dx)) + abs(float(dy))
    return [
        1.0,
        float(dx) / denom,
        float(dy) / denom,
        min(1.0, dist / denom),
    ]


def _nearest_visible_tile_features(obs_agent: dict, tile_id: int) -> list[float]:
    local_ids = _recurrent_local_grid_ids(obs_agent.get("local_grid", np.zeros((1, 1), dtype=np.int16)))
    if local_ids.ndim != 2:
        return [0.0, 0.0, 0.0, 1.0]
    h, w = local_ids.shape
    cx, cy = w // 2, h // 2
    best = None
    best_dist = float("inf")
    for y in range(h):
        for x in range(w):
            if int(local_ids[y, x]) != int(tile_id):
                continue
            dx, dy = x - cx, y - cy
            dist = abs(dx) + abs(dy)
            if dist < best_dist:
                best = (dx, dy)
                best_dist = dist
    radius = max(cx, cy, 1)
    if best is None:
        return _direction_features(0.0, 0.0, radius, False)
    return _direction_features(float(best[0]), float(best[1]), radius, True)


def _frontier_features(obs_agent: dict, cfg: RecurrentConfig, observed_map_size: int) -> list[float]:
    explored = obs_agent.get("explored_mask")
    if explored is None:
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    mask = np.asarray(explored).astype(bool)
    if mask.ndim != 2 or mask.size == 0:
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    pos = np.asarray(obs_agent.get("self_pos", np.zeros((2,), dtype=np.int16)), dtype=np.int64).reshape(-1)
    if pos.size < 2:
        return [0.0, 0.0, 0.0, 1.0, float(mask.mean())]
    sx, sy = int(pos[0]), int(pos[1])
    height, width = mask.shape
    best = None
    best_dist = float("inf")
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            is_frontier = False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not mask[ny, nx]:
                    is_frontier = True
                    break
            if not is_frontier:
                continue
            dx, dy = x - sx, y - sy
            dist = abs(dx) + abs(dy)
            if dist < best_dist:
                best = (dx, dy)
                best_dist = dist
    denom = max(1.0, float(observed_map_size - 1))
    direction = _direction_features(
        float(best[0]) if best is not None else 0.0,
        float(best[1]) if best is not None else 0.0,
        denom,
        best is not None,
    )
    return [*direction, float(mask.mean())]


def _navigation_features(obs_agent: dict, cfg: RecurrentConfig, observed_map_size: int) -> np.ndarray:
    if not cfg.obs_navigation_features:
        return np.zeros((0,), dtype=np.float32)
    parts: list[float] = []
    for tile_id in (TILE_CLUE, TILE_TARGET, TILE_RESOURCE, TILE_NODE, TILE_STATION):
        parts.extend(_nearest_visible_tile_features(obs_agent, tile_id))
    parts.extend(_frontier_features(obs_agent, cfg, observed_map_size))
    return np.asarray(parts, dtype=np.float32)


_SIGNAL_SEGMENT_LENGTHS = {21: 5, 22: 6, 23: 4, 24: 2, 25: 2, 26: 3}


def _signal_segments(raw) -> list[list[int]]:
    if raw is None:
        return []
    tokens = []
    for value in np.asarray(raw).reshape(-1):
        try:
            tokens.append(int(value))
        except (TypeError, ValueError):
            break
    segments: list[list[int]] = []
    i = 0
    while i < len(tokens):
        code = int(tokens[i])
        if code < 0:
            break
        length = _SIGNAL_SEGMENT_LENGTHS.get(code)
        if length is None or i + length > len(tokens):
            break
        segment = tokens[i: i + length]
        segments.append(segment)
        i += length
    return segments


def _signal_targets_from_segments(
    segments: list[list[int]],
    observed_map_size: int,
) -> list[tuple[int, int]]:
    targets: list[tuple[int, int]] = []
    for segment in segments:
        tx: int | None = None
        ty: int | None = None
        if segment[0] == 22:
            tx = int(segment[2]) + int(segment[4])
            ty = int(segment[3]) + int(segment[5])
        elif segment[0] == 26:
            tx = int(segment[1])
            ty = int(segment[2])
        if tx is None or ty is None:
            continue
        if 0 <= tx < observed_map_size and 0 <= ty < observed_map_size:
            targets.append((tx, ty))
    return targets


def _signal_targets_from_tokens(raw, observed_map_size: int) -> list[tuple[int, int]]:
    return _signal_targets_from_segments(_signal_segments(raw), observed_map_size)


def _message_token_rows(raw) -> list[np.ndarray]:
    rows = np.asarray(raw if raw is not None else np.zeros((0, 0), dtype=np.int16))
    if rows.ndim == 1:
        return [rows]
    if rows.ndim >= 2 and rows.shape[-1] > 0:
        return [row for row in rows.reshape(-1, rows.shape[-1])]
    return []


def _signal_segments_from_observation(obs_agent: dict) -> tuple[list[list[int]], list[list[int]]]:
    own_segments = _signal_segments(obs_agent.get("goal_hint"))
    message_segments: list[list[int]] = []
    for row in _message_token_rows(obs_agent.get("messages_tokens")):
        message_segments.extend(_signal_segments(row))
    return own_segments, message_segments


def _signal_targets_from_observation(
    obs_agent: dict,
    observed_map_size: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    own_segments, message_segments = _signal_segments_from_observation(obs_agent)
    return (
        _signal_targets_from_segments(own_segments, observed_map_size),
        _signal_targets_from_segments(message_segments, observed_map_size),
    )


def _signal_observation_allows_target(
    obs_agent: dict,
    target: tuple[int, int],
    observed_map_size: int,
) -> bool:
    own_segments, message_segments = _signal_segments_from_observation(obs_agent)
    segments = [*own_segments, *message_segments]
    exact_targets = set(_signal_targets_from_segments(segments, observed_map_size))
    if exact_targets:
        return target in exact_targets

    has_constraint = False
    near_constraints: list[tuple[int, int, int]] = []
    parity: int | None = None
    quadrant: int | None = None
    quadrant_size: int | None = None
    x_parity: int | None = None
    y_parity: int | None = None
    for segment in segments:
        code = int(segment[0])
        if code == 21:
            ox, oy = int(segment[2]), int(segment[3])
            dist = max(0, int(segment[4]))
            if 0 <= ox < observed_map_size and 0 <= oy < observed_map_size:
                near_constraints.append((ox, oy, dist))
                has_constraint = True
        elif code == 23:
            parity = int(segment[1]) if int(segment[1]) in (0, 1) else None
            quadrant = int(segment[2]) if 0 <= int(segment[2]) <= 3 else None
            quadrant_size = max(1, int(segment[3]))
            has_constraint = True
        elif code == 24:
            x_parity = int(segment[1]) if int(segment[1]) in (0, 1) else None
            has_constraint = True
        elif code == 25:
            y_parity = int(segment[1]) if int(segment[1]) in (0, 1) else None
            has_constraint = True

    if not has_constraint:
        return False

    tx, ty = int(target[0]), int(target[1])
    if parity is not None and (tx + ty) % 2 != parity:
        return False
    if x_parity is not None and tx % 2 != x_parity:
        return False
    if y_parity is not None and ty % 2 != y_parity:
        return False
    if quadrant is not None:
        half = float(quadrant_size or observed_map_size) / 2.0
        in_quadrant = (
            (quadrant == 0 and tx < half and ty < half)
            or (quadrant == 1 and tx >= half and ty < half)
            or (quadrant == 2 and tx < half and ty >= half)
            or (quadrant == 3 and tx >= half and ty >= half)
        )
        if not in_quadrant:
            return False
    for ox, oy, dist in near_constraints:
        if abs(tx - ox) + abs(ty - oy) > dist:
            return False
    return True


def _signal_constraint_features(
    obs_agent: dict,
    self_pos: np.ndarray,
    observed_map_size: int,
) -> np.ndarray:
    own_segments, message_segments = _signal_segments_from_observation(obs_agent)
    segments = [*own_segments, *message_segments]
    constraint_codes = {21, 23, 24, 25}
    own_has_constraint = any(segment and int(segment[0]) in constraint_codes for segment in own_segments)
    message_has_constraint = any(segment and int(segment[0]) in constraint_codes for segment in message_segments)

    near_constraints: list[tuple[int, int, int, int]] = []
    parity: int | None = None
    quadrant: int | None = None
    quadrant_size: int | None = None
    x_parity: int | None = None
    y_parity: int | None = None

    for segment in segments:
        code = int(segment[0])
        if code == 21:
            obj = int(segment[1])
            ox, oy = int(segment[2]), int(segment[3])
            dist = max(0, int(segment[4]))
            if 0 <= ox < observed_map_size and 0 <= oy < observed_map_size:
                near_constraints.append((obj, ox, oy, dist))
        elif code == 23:
            parity = int(segment[1]) if int(segment[1]) in (0, 1) else None
            quadrant = int(segment[2]) if 0 <= int(segment[2]) <= 3 else None
            quadrant_size = max(1, int(segment[3]))
        elif code == 24:
            x_parity = int(segment[1]) if int(segment[1]) in (0, 1) else None
        elif code == 25:
            y_parity = int(segment[1]) if int(segment[1]) in (0, 1) else None

    sx, sy = int(self_pos[0]), int(self_pos[1])
    nearest_near = None
    if near_constraints:
        nearest_near = min(
            near_constraints,
            key=lambda item: (abs(item[1] - sx) + abs(item[2] - sy), item[2], item[1]),
        )

    parts: list[float] = []
    if nearest_near is None:
        parts.extend(_direction_features(0.0, 0.0, observed_map_size - 1, False))
        parts.extend([0.0, 0.0, 0.0, 0.0])
    else:
        obj, ox, oy, dist = nearest_near
        parts.extend(_direction_features(float(ox - sx), float(oy - sy), observed_map_size - 1, True))
        parts.append(min(1.0, float(dist) / max(1.0, float(observed_map_size - 1))))
        parts.extend([
            1.0 if obj == TILE_WATER else 0.0,
            1.0 if obj == TILE_BEACON else 0.0,
            1.0 if obj not in (TILE_WATER, TILE_BEACON) else 0.0,
        ])

    parts.extend([
        1.0 if parity is not None else 0.0,
        float(parity) if parity is not None else 0.0,
        1.0 if quadrant is not None else 0.0,
    ])
    for q in range(4):
        parts.append(1.0 if quadrant == q else 0.0)
    parts.append(min(1.0, float(quadrant_size or 0) / max(1.0, float(observed_map_size))))
    parts.extend([
        1.0 if x_parity is not None else 0.0,
        float(x_parity) if x_parity is not None else 0.0,
        1.0 if y_parity is not None else 0.0,
        float(y_parity) if y_parity is not None else 0.0,
        1.0 if own_has_constraint else 0.0,
        1.0 if message_has_constraint else 0.0,
    ])
    return np.asarray(parts, dtype=np.float32)


def _signal_constraint_strength_from_segments(segments: list[list[int]]) -> int:
    strength = 0
    has_near = False
    for segment in segments:
        if not segment:
            continue
        code = int(segment[0])
        if code == 21:
            has_near = True
        elif code == 23:
            if len(segment) >= 2 and int(segment[1]) in (0, 1):
                strength += 1
            if len(segment) >= 3 and 0 <= int(segment[2]) <= 3:
                strength += 1
        elif code == 24 and len(segment) >= 2 and int(segment[1]) in (0, 1):
            strength += 1
        elif code == 25 and len(segment) >= 2 and int(segment[1]) in (0, 1):
            strength += 1
    return strength + (1 if has_near else 0)


def _signal_inferred_constraint_targets(
    obs_agent: dict,
    observed_map_size: int,
) -> list[tuple[int, int]]:
    own_segments, message_segments = _signal_segments_from_observation(obs_agent)
    segments = [*own_segments, *message_segments]
    if _signal_targets_from_segments(segments, observed_map_size):
        return []
    if _signal_constraint_strength_from_segments(segments) < 3:
        return []
    candidates = [
        (x, y)
        for y in range(int(observed_map_size))
        for x in range(int(observed_map_size))
        if _signal_observation_allows_target(obs_agent, (x, y), observed_map_size)
    ]
    if len(candidates) <= max(8, int(observed_map_size) * 4):
        return candidates
    return []


def _signal_inferred_target_features(
    obs_agent: dict,
    self_pos: np.ndarray,
    observed_map_size: int,
) -> np.ndarray:
    candidates = _signal_inferred_constraint_targets(obs_agent, observed_map_size)
    nearest = _nearest_signal_target(candidates, self_pos)
    parts = _target_direction_group(nearest, self_pos, observed_map_size)
    count_scale = max(1.0, float(int(observed_map_size) * 4))
    sx, sy = int(self_pos[0]), int(self_pos[1])
    parts.extend([
        min(1.0, float(len(candidates)) / count_scale),
        1.0 if nearest is not None and nearest == (sx, sy) else 0.0,
    ])
    return np.asarray(parts, dtype=np.float32)


def _nearest_signal_target(
    targets: list[tuple[int, int]],
    self_pos: np.ndarray,
) -> tuple[int, int] | None:
    if not targets:
        return None
    sx, sy = int(self_pos[0]), int(self_pos[1])
    return min(targets, key=lambda pos: (abs(pos[0] - sx) + abs(pos[1] - sy), pos[1], pos[0]))


def _target_direction_group(
    target: tuple[int, int] | None,
    self_pos: np.ndarray,
    observed_map_size: int,
) -> list[float]:
    if target is None:
        return _direction_features(0.0, 0.0, observed_map_size - 1, False)
    sx, sy = int(self_pos[0]), int(self_pos[1])
    return _direction_features(
        float(int(target[0]) - sx),
        float(int(target[1]) - sy),
        observed_map_size - 1,
        True,
    )


def _visible_signal_targets(
    obs_agent: dict,
    self_pos: np.ndarray,
    observed_map_size: int,
) -> list[tuple[int, int]]:
    local_ids = _recurrent_local_grid_ids(obs_agent.get("local_grid", np.zeros((1, 1), dtype=np.int16)))
    if local_ids.ndim != 2:
        return []
    sx, sy = int(self_pos[0]), int(self_pos[1])
    h, w = local_ids.shape
    cx, cy = w // 2, h // 2
    targets: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if int(local_ids[y, x]) != TILE_TARGET:
                continue
            gx, gy = sx + (x - cx), sy + (y - cy)
            if 0 <= gx < observed_map_size and 0 <= gy < observed_map_size:
                targets.append((gx, gy))
    return targets


def _signal_observation_has_target_information(obs_agent: dict) -> bool:
    own_segments, message_segments = _signal_segments_from_observation(obs_agent)
    return any(segment and 21 <= int(segment[0]) <= 26 for segment in [*own_segments, *message_segments])


def _signal_visible_target_match_features(
    obs_agent: dict,
    self_pos: np.ndarray,
    observed_map_size: int,
) -> np.ndarray:
    visible_targets = _visible_signal_targets(obs_agent, self_pos, observed_map_size)
    has_target_info = _signal_observation_has_target_information(obs_agent)
    allowed = [
        target for target in visible_targets
        if _signal_observation_allows_target(obs_agent, target, observed_map_size)
    ]
    rejected = [
        target for target in visible_targets
        if has_target_info and target not in set(allowed)
    ]
    nearest_allowed = _nearest_signal_target(allowed, self_pos)
    nearest_rejected = _nearest_signal_target(rejected, self_pos)
    sx, sy = int(self_pos[0]), int(self_pos[1])
    center_is_visible_target = (sx, sy) in set(visible_targets)
    center_allowed = center_is_visible_target and (sx, sy) in set(allowed)
    center_rejected = center_is_visible_target and has_target_info and not center_allowed
    denom = max(1.0, float(len(visible_targets) or 1))
    parts: list[float] = []
    parts.extend(_target_direction_group(nearest_allowed, self_pos, observed_map_size))
    parts.extend(_target_direction_group(nearest_rejected, self_pos, observed_map_size))
    parts.extend([
        1.0 if center_allowed else 0.0,
        1.0 if center_rejected else 0.0,
        min(1.0, float(len(visible_targets)) / 4.0),
        float(len(allowed)) / denom,
        float(len(rejected)) / denom,
        1.0 if has_target_info else 0.0,
    ])
    return np.asarray(parts, dtype=np.float32)


def _signal_center_rejected_target(obs_agent: dict, observed_map_size: int) -> bool:
    if not _signal_observation_has_target_information(obs_agent):
        return False
    local_ids = _recurrent_local_grid_ids(obs_agent.get("local_grid", np.zeros((1, 1), dtype=np.int16)))
    if local_ids.ndim != 2 or local_ids.size == 0:
        return False
    cy, cx = local_ids.shape[0] // 2, local_ids.shape[1] // 2
    if int(local_ids[cy, cx]) != TILE_TARGET:
        return False
    self_pos = np.asarray(obs_agent.get("self_pos", np.zeros((2,), dtype=np.int16)), dtype=np.int64).reshape(-1)
    if self_pos.size < 2:
        return False
    pos = (int(self_pos[0]), int(self_pos[1]))
    return not _signal_observation_allows_target(obs_agent, pos, observed_map_size)


def _signal_rejected_target_mask(obs: dict, observed_map_size: int, num_agents: int) -> np.ndarray:
    return np.asarray(
        [
            1.0 if _signal_center_rejected_target(obs.get(aid, {}), observed_map_size) else 0.0
            for aid in range(num_agents)
        ],
        dtype=np.float32,
    )


def _clear_true_target_rejected_mask(env: SyncOrSinkEnv, mask: np.ndarray) -> np.ndarray:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return mask
    target = env.scenario_state.data.get("target")
    if target is None:
        return mask
    true_target = (int(target[0]), int(target[1]))
    corrected = np.asarray(mask, dtype=np.float32).copy()
    for aid in range(min(env.num_agents, corrected.shape[0])):
        if tuple(env.agent_positions[int(aid)]) == true_target:
            corrected[int(aid)] = 0.0
    return corrected


def _recent_signal_scan_age(env: SyncOrSinkEnv, agent_id: int, *, next_step: int) -> int | None:
    scan_log = env.scenario_state.data.get("scan_log") or {}
    last_scan = scan_log.get(int(agent_id), scan_log.get(str(int(agent_id))))
    if last_scan is None:
        return None
    return int(next_step) - int(last_scan)


def _teammate_can_join_target_by_distance(env: SyncOrSinkEnv, *, scanner_id: int, target: tuple[int, int]) -> bool:
    scan_window = int(env.scenario_state.data.get("scan_window", getattr(env.config, "scan_window", 3)))
    for aid in range(env.num_agents):
        if int(aid) == int(scanner_id):
            continue
        pos = env.agent_positions[int(aid)]
        dist = abs(int(target[0]) - int(pos[0])) + abs(int(target[1]) - int(pos[1]))
        if dist + 1 <= scan_window:
            return True
    return False


def _signal_bad_redundant_target_scan_agents(env: SyncOrSinkEnv, obs: dict | None = None) -> list[int]:
    del obs
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    target = env.scenario_state.data.get("target")
    if target is None:
        return []
    target = tuple(target)
    scan_window = int(env.scenario_state.data.get("scan_window", getattr(env.config, "scan_window", 3)))
    next_step = int(env.steps) + 1
    bad_agents: list[int] = []
    for aid in range(env.num_agents):
        if tuple(env.agent_positions[int(aid)]) != target:
            continue
        self_age = _recent_signal_scan_age(env, int(aid), next_step=next_step)
        if self_age is None or self_age < 0 or self_age > scan_window:
            continue
        teammate_recent = False
        for other_id in range(env.num_agents):
            if int(other_id) == int(aid):
                continue
            other_age = _recent_signal_scan_age(env, int(other_id), next_step=next_step)
            if other_age is not None and 0 <= other_age <= scan_window:
                teammate_recent = True
                break
        if teammate_recent:
            continue
        if _teammate_can_join_target_by_distance(env, scanner_id=int(aid), target=target):
            continue
        bad_agents.append(int(aid))
    return bad_agents


def _signal_bad_redundant_target_mask(env: SyncOrSinkEnv, obs: dict) -> np.ndarray:
    del obs
    bad_agents = set(_signal_bad_redundant_target_scan_agents(env))
    return np.asarray(
        [1.0 if int(aid) in bad_agents else 0.0 for aid in range(env.num_agents)],
        dtype=np.float32,
    )


def _signal_coordination_features(obs_agent: dict, cfg: RecurrentConfig, observed_map_size: int) -> np.ndarray:
    if not cfg.obs_signal_features or cfg.scenario != "signal_hunt":
        return np.zeros((0,), dtype=np.float32)
    pos = np.asarray(obs_agent.get("self_pos", np.zeros((2,), dtype=np.int16)), dtype=np.int64).reshape(-1)
    if pos.size < 2:
        pos = np.zeros((2,), dtype=np.int64)
    own_targets, message_targets = _signal_targets_from_observation(obs_agent, observed_map_size)
    own_target = _nearest_signal_target(own_targets, pos)
    message_target = _nearest_signal_target(message_targets, pos)
    combined_target = _nearest_signal_target([*own_targets, *message_targets], pos)
    constraint_features = _signal_constraint_features(obs_agent, pos, observed_map_size)

    local_ids = _recurrent_local_grid_ids(obs_agent.get("local_grid", np.zeros((1, 1), dtype=np.int16)))
    center_is_target = 0.0
    if local_ids.ndim == 2 and local_ids.size:
        center_is_target = 1.0 if int(local_ids[local_ids.shape[0] // 2, local_ids.shape[1] // 2]) == TILE_TARGET else 0.0
    combined_at_self = 0.0
    if combined_target is not None:
        combined_at_self = 1.0 if (int(combined_target[0]) == int(pos[0]) and int(combined_target[1]) == int(pos[1])) else 0.0

    parts: list[float] = []
    parts.extend(_target_direction_group(own_target, pos, observed_map_size))
    parts.extend(_target_direction_group(message_target, pos, observed_map_size))
    parts.extend(_target_direction_group(combined_target, pos, observed_map_size))
    parts.extend([
        center_is_target,
        combined_at_self,
        1.0 if own_targets else 0.0,
        1.0 if message_targets else 0.0,
    ])
    parts.extend(constraint_features.tolist())
    if cfg.obs_signal_inferred_target_features:
        parts.extend(
            _signal_inferred_target_features(obs_agent, pos, observed_map_size).tolist()
        )
    if cfg.obs_signal_target_match_features:
        parts.extend(
            _signal_visible_target_match_features(obs_agent, pos, observed_map_size).tolist()
        )
    return np.asarray(parts, dtype=np.float32)


def _normalize_recurrent_obs_agent(
    obs_agent: dict,
    cfg: RecurrentConfig,
    *,
    observed_map_size: int | None = None,
) -> dict:
    if not cfg.obs_normalize_tokens:
        return obs_agent
    current_map_size = int(observed_map_size or _observed_map_size(obs_agent, cfg))
    map_denom = max(float(current_map_size - 1), 1.0)
    token_denom = max(float(cfg.comm_vocab_size - 1), map_denom, 31.0)
    norm = dict(obs_agent)
    norm["self_pos"] = _scale_nonnegative(
        obs_agent.get("self_pos", np.zeros((2,), dtype=np.float32)),
        map_denom,
    )
    norm["local_resource_types"] = _scale_nonnegative(
        obs_agent.get("local_resource_types", np.zeros((1, 1), dtype=np.float32)),
        10.0,
    )
    norm["local_node_types"] = _scale_nonnegative(
        obs_agent.get("local_node_types", np.zeros((1, 1), dtype=np.float32)),
        10.0,
    )
    norm["local_node_energy"] = _scale_nonnegative(
        obs_agent.get("local_node_energy", np.zeros((1, 1), dtype=np.float32)),
        50.0,
    )
    norm["messages_tokens"] = _scale_nonnegative(
        obs_agent.get("messages_tokens", np.zeros((1, 1), dtype=np.float32)),
        token_denom,
    )
    norm["message_from"] = _scale_nonnegative(
        obs_agent.get("message_from", np.zeros((1,), dtype=np.float32)),
        cfg.agents - 1,
    )
    norm["goal_hint"] = _scale_nonnegative(
        obs_agent.get("goal_hint", np.zeros((1,), dtype=np.float32)),
        token_denom,
    )
    if cfg.obs_exploration_age and "explored_age" in obs_agent:
        norm["explored_age"] = _scale_nonnegative(obs_agent["explored_age"], cfg.max_steps)
    return norm


def _feedback_dim(cfg: RecurrentConfig) -> int:
    return (
        12
        + (4 if cfg.obs_signal_sync_feedback else 0)
        + (4 if cfg.obs_signal_scan_state else 0)
        + (8 if cfg.obs_signal_negative_memory else 0)
    )


def _flatten_recurrent_obs(obs_agent: dict, cfg: RecurrentConfig, feedback: np.ndarray | None = None) -> np.ndarray:
    observed_map_size = _observed_map_size(obs_agent, cfg)
    navigation = _navigation_features(obs_agent, cfg, observed_map_size)
    signal_features = _signal_coordination_features(obs_agent, cfg, observed_map_size)
    obs_agent = _project_recurrent_memory(obs_agent, cfg)
    obs_agent = _normalize_recurrent_obs_agent(
        obs_agent,
        cfg,
        observed_map_size=observed_map_size,
    )
    flat = flatten_obs(
        obs_agent,
        include_exploration_memory=cfg.obs_exploration_memory,
        include_exploration_age=cfg.obs_exploration_age,
    )
    extras = []
    if cfg.obs_feedback:
        if feedback is None:
            feedback = np.zeros((_feedback_dim(cfg),), dtype=np.float32)
        feedback = np.asarray(feedback, dtype=np.float32).reshape(-1)
        extras.append(feedback)
    if navigation.size > 0:
        extras.append(navigation)
    if signal_features.size > 0:
        extras.append(signal_features)
    if extras:
        flat = np.concatenate([flat[:-8], *extras, flat[-8:]], axis=0)
    return flat


def _build_recurrent_obs_batch(
    obs: dict,
    num_agents: int,
    cfg: RecurrentConfig,
    feedback: np.ndarray | None = None,
) -> np.ndarray:
    if feedback is None:
        feedback = np.zeros((num_agents, _feedback_dim(cfg)), dtype=np.float32) if cfg.obs_feedback else None
    if feedback is not None:
        feedback = np.asarray(feedback, dtype=np.float32).reshape(num_agents, -1)
    return np.stack([
        _flatten_recurrent_obs(
            obs[aid],
            cfg,
            feedback=feedback[aid] if feedback is not None else None,
        )
        for aid in range(num_agents)
    ]).astype(np.float32)


def _event_flags_for_agent(info: dict, agent_id: int) -> tuple[float, float]:
    events = (info or {}).get("events", {})
    if not isinstance(events, dict):
        return 0.0, 0.0
    agent_events = events.get(agent_id, events.get(str(agent_id), []))
    if not isinstance(agent_events, list):
        return 0.0, 0.0
    names = {event.get("event") for event in agent_events if isinstance(event, dict)}
    return (
        1.0 if "decoy_scan" in names else 0.0,
        1.0 if "clue_found" in names else 0.0,
    )


def _event_names_by_agent(info: dict | None, num_agents: int) -> dict[int, set[str]]:
    names = {aid: set() for aid in range(num_agents)}
    events = (info or {}).get("events", {})
    if not isinstance(events, dict):
        return names
    for aid in range(num_agents):
        agent_events = events.get(aid, events.get(str(aid), []))
        if not isinstance(agent_events, list):
            continue
        for event in agent_events:
            if isinstance(event, dict) and event.get("event"):
                names[aid].add(str(event["event"]))
    return names


def _signal_sync_feedback_flags(info: dict | None, agent_id: int, num_agents: int) -> list[float]:
    names_by_agent = _event_names_by_agent(info, num_agents)
    self_names = names_by_agent.get(int(agent_id), set())
    teammate_target_scans = sum(
        1
        for other_id, names in names_by_agent.items()
        if other_id != int(agent_id) and "target_scan" in names
    )
    total_target_scans = teammate_target_scans + (1 if "target_scan" in self_names else 0)
    any_joint = any("joint_target_scan" in names for names in names_by_agent.values())
    return [
        1.0 if "target_scan" in self_names else 0.0,
        1.0 if teammate_target_scans > 0 else 0.0,
        float(total_target_scans) / max(1.0, float(num_agents)),
        1.0 if any_joint else 0.0,
    ]


def _scan_log_value(scan_log: dict, agent_id: int) -> int | None:
    raw = scan_log.get(int(agent_id), scan_log.get(str(int(agent_id))))
    if raw is None:
        return None
    return int(raw)


def _signal_scan_state_feedback_flags(
    cfg: RecurrentConfig,
    agent_id: int,
    num_agents: int,
    *,
    env: SyncOrSinkEnv | None = None,
    scan_state: dict | None = None,
) -> list[float]:
    if cfg.scenario != "signal_hunt":
        return [0.0, 0.0, 0.0, 0.0]
    if env is not None:
        state = getattr(getattr(env, "scenario_state", None), "data", {}) or {}
        scan_log = state.get("scan_log") or {}
        scan_window = int(state.get("scan_window", getattr(cfg, "scan_window", 3)))
        current_step = int(getattr(env, "steps", 0))
    elif scan_state is not None:
        scan_log = scan_state.get("scan_log") or {}
        scan_window = int(scan_state.get("scan_window", getattr(cfg, "scan_window", 3)))
        current_step = int(scan_state.get("step", 0))
    else:
        return [0.0, 0.0, 0.0, 0.0]

    def remaining_fraction(other_id: int) -> float:
        last_scan = _scan_log_value(scan_log, int(other_id))
        if last_scan is None:
            return 0.0
        age = current_step - int(last_scan)
        if age < 0 or age > scan_window:
            return 0.0
        return float(scan_window - age + 1) / max(1.0, float(scan_window + 1))

    self_remaining = remaining_fraction(int(agent_id))
    teammate_remaining = max(
        (remaining_fraction(other_id) for other_id in range(num_agents) if int(other_id) != int(agent_id)),
        default=0.0,
    )
    return [
        1.0 if self_remaining > 0.0 else 0.0,
        1.0 if teammate_remaining > 0.0 else 0.0,
        self_remaining,
        teammate_remaining,
    ]


def _signal_xy(raw) -> tuple[int, int] | None:
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 2:
        return None
    return int(arr[0]), int(arr[1])


def _obs_agent_for_feedback(obs: dict | None, agent_id: int) -> dict | None:
    if not isinstance(obs, dict):
        return None
    if "self_pos" in obs:
        return obs
    obs_agent = obs.get(int(agent_id), obs.get(str(int(agent_id))))
    return obs_agent if isinstance(obs_agent, dict) else None


def _normalize_signal_negative_target_log(raw) -> list[dict]:
    entries: list[dict] = []
    if raw is None:
        return entries
    if isinstance(raw, dict):
        raw_iterable = [raw] if any(key in raw for key in ("pos", "target", "target_pos")) else raw.values()
    else:
        raw_iterable = raw
    try:
        iterator = iter(raw_iterable)
    except TypeError:
        return entries
    for item in iterator:
        if isinstance(item, dict):
            pos = _signal_xy(item.get("pos", item.get("target", item.get("target_pos"))))
            if pos is None:
                continue
            try:
                step = int(item.get("step", 0))
            except (TypeError, ValueError):
                step = 0
            try:
                entry_agent = int(item.get("agent_id", item.get("agent", -1)))
            except (TypeError, ValueError):
                entry_agent = -1
            entries.append({"agent_id": entry_agent, "pos": pos, "step": step})
            continue
        pos = _signal_xy(item)
        if pos is not None:
            entries.append({"agent_id": -1, "pos": pos, "step": 0})
    return entries


def _signal_negative_memory_feedback_flags(
    cfg: RecurrentConfig,
    agent_id: int,
    num_agents: int,
    *,
    env: SyncOrSinkEnv | None = None,
    scan_state: dict | None = None,
    obs: dict | None = None,
) -> list[float]:
    del num_agents
    if cfg.scenario != "signal_hunt":
        return [0.0] * 8
    if env is not None:
        state = getattr(getattr(env, "scenario_state", None), "data", {}) or {}
        raw_log = state.get("negative_target_log") or []
        current_step = int(getattr(env, "steps", 0))
    elif scan_state is not None:
        raw_log = scan_state.get("negative_target_log") or []
        current_step = int(scan_state.get("step", 0))
    else:
        return [0.0] * 8

    window = max(1, int(getattr(cfg, "obs_signal_negative_memory_window", 64)))
    active_entries = []
    for entry in _normalize_signal_negative_target_log(raw_log):
        age = current_step - int(entry.get("step", 0))
        if 0 <= age <= window:
            active_entries.append((int(entry.get("agent_id", -1)), tuple(entry["pos"]), int(age)))
    if not active_entries:
        return [0.0] * 8

    agent_id = int(agent_id)
    obs_agent = _obs_agent_for_feedback(obs, agent_id)
    if env is not None:
        self_pos = tuple(env.agent_positions[agent_id])
        observed_map_size = int(getattr(env, "map_size", cfg.map_size))
    elif obs_agent is not None:
        parsed_pos = _signal_xy(obs_agent.get("self_pos"))
        self_pos = parsed_pos if parsed_pos is not None else None
        observed_map_size = _observed_map_size(obs_agent, cfg)
    else:
        self_pos = None
        observed_map_size = int(cfg.map_size)

    self_recent = any(entry_agent == agent_id for entry_agent, _pos, _age in active_entries)
    teammate_recent = any(entry_agent >= 0 and entry_agent != agent_id for entry_agent, _pos, _age in active_entries)
    self_at_negative = 0.0
    any_at_negative = 0.0
    nearest_pos: tuple[int, int] | None = None
    if self_pos is not None:
        self_at_negative = 1.0 if any(
            pos == self_pos and entry_agent == agent_id
            for entry_agent, pos, _age in active_entries
        ) else 0.0
        any_at_negative = 1.0 if any(pos == self_pos for _entry_agent, pos, _age in active_entries) else 0.0
        nearest_pos = min(
            (pos for _entry_agent, pos, _age in active_entries),
            key=lambda pos: (abs(pos[0] - self_pos[0]) + abs(pos[1] - self_pos[1]), pos[1], pos[0]),
        )

    nearest_visible_negative = 0.0
    dx_norm = 0.0
    dy_norm = 0.0
    if nearest_pos is not None and self_pos is not None:
        denom = max(1.0, float(observed_map_size - 1))
        dx_norm = float(nearest_pos[0] - self_pos[0]) / denom
        dy_norm = float(nearest_pos[1] - self_pos[1]) / denom
        if obs_agent is not None:
            visible = set(_visible_signal_targets(obs_agent, np.asarray(self_pos, dtype=np.int64), observed_map_size))
            nearest_visible_negative = 1.0 if nearest_pos in visible else 0.0

    max_remaining = max(
        (float(window - age + 1) / max(1.0, float(window + 1)) for _entry_agent, _pos, age in active_entries),
        default=0.0,
    )
    return [
        1.0 if self_recent else 0.0,
        1.0 if teammate_recent else 0.0,
        self_at_negative,
        any_at_negative,
        nearest_visible_negative,
        dx_norm,
        dy_norm,
        max_remaining,
    ]


def _feedback_matrix(
    cfg: RecurrentConfig,
    num_agents: int,
    *,
    prev_actions: dict[int, int] | None = None,
    prev_msg_lens: dict[int, int] | None = None,
    info: dict | None = None,
    env: SyncOrSinkEnv | None = None,
    scan_state: dict | None = None,
    obs: dict | None = None,
) -> np.ndarray | None:
    if not cfg.obs_feedback:
        return None
    prev_actions = prev_actions or {}
    prev_msg_lens = prev_msg_lens or {}
    rows = np.zeros((num_agents, _feedback_dim(cfg)), dtype=np.float32)
    for aid in range(num_agents):
        action_id = prev_actions.get(aid)
        if action_id is not None and 0 <= int(action_id) < 8:
            rows[aid, int(action_id)] = 1.0
        msg_len = max(0, int(prev_msg_lens.get(aid, 0)))
        rows[aid, 8] = 1.0 if msg_len > 0 else 0.0
        rows[aid, 9] = msg_len / max(1, cfg.comm_token_limit)
        decoy, clue = _event_flags_for_agent(info or {}, aid)
        rows[aid, 10] = decoy
        rows[aid, 11] = clue
        if cfg.obs_signal_sync_feedback:
            rows[aid, 12:16] = np.asarray(
                _signal_sync_feedback_flags(info, aid, num_agents),
                dtype=np.float32,
            )
        if cfg.obs_signal_scan_state:
            offset = 12 + (4 if cfg.obs_signal_sync_feedback else 0)
            rows[aid, offset:offset + 4] = np.asarray(
                _signal_scan_state_feedback_flags(
                    cfg,
                    aid,
                    num_agents,
                    env=env,
                    scan_state=scan_state,
                ),
                dtype=np.float32,
            )
        if cfg.obs_signal_negative_memory:
            offset = (
                12
                + (4 if cfg.obs_signal_sync_feedback else 0)
                + (4 if cfg.obs_signal_scan_state else 0)
            )
            rows[aid, offset:offset + 8] = np.asarray(
                _signal_negative_memory_feedback_flags(
                    cfg,
                    aid,
                    num_agents,
                    env=env,
                    scan_state=scan_state,
                    obs=obs,
                ),
                dtype=np.float32,
            )
    return rows


def _message_lengths(actions: dict[int, dict]) -> dict[int, int]:
    return {
        int(aid): len(action.get("message_tokens") or [])
        for aid, action in actions.items()
    }


def _initial_signal_scan_state(cfg: RecurrentConfig) -> dict:
    return {
        "scan_log": {},
        "scan_pos": {},
        "scan_broadcast_log": {},
        "scan_window": int(cfg.scan_window),
        "negative_target_log": [],
        "negative_memory_window": int(cfg.obs_signal_negative_memory_window),
        "step": 0,
    }


def _signal_positions_from_obs(obs: dict) -> dict[int, tuple[int, int]]:
    positions: dict[int, tuple[int, int]] = {}
    if not isinstance(obs, dict):
        return positions
    for raw_aid, obs_agent in obs.items():
        try:
            aid = int(raw_aid)
        except (TypeError, ValueError):
            continue
        if not isinstance(obs_agent, dict):
            continue
        pos = _signal_xy(obs_agent.get("self_pos"))
        if pos is not None:
            positions[aid] = pos
    return positions


def _update_signal_scan_state_from_info(
    cfg: RecurrentConfig,
    scan_state: dict,
    info: dict | None,
    num_agents: int,
    prev_positions: Mapping[int, tuple[int, int]],
    *,
    has_policy_step: bool,
) -> bool:
    if cfg.scenario != "signal_hunt" or not (
        cfg.obs_signal_scan_state or cfg.obs_signal_negative_memory
    ):
        return True
    if not has_policy_step:
        return True
    scan_state["step"] = int(scan_state.get("step", 0)) + 1
    for aid, names in _event_names_by_agent(info or {}, num_agents).items():
        if cfg.obs_signal_scan_state and "target_scan" in names:
            scan_state.setdefault("scan_log", {})[int(aid)] = int(scan_state["step"])
            pos = prev_positions.get(int(aid))
            if pos is not None:
                scan_state.setdefault("scan_pos", {})[int(aid)] = tuple(pos)
        if cfg.obs_signal_negative_memory and "decoy_scan" in names:
            pos = prev_positions.get(int(aid))
            if pos is not None:
                scan_state.setdefault("negative_target_log", []).append({
                    "agent_id": int(aid),
                    "pos": tuple(pos),
                    "step": int(scan_state["step"]),
                })
    if cfg.obs_signal_negative_memory:
        window = max(1, int(cfg.obs_signal_negative_memory_window))
        current_step = int(scan_state.get("step", 0))
        scan_state["negative_target_log"] = [
            entry
            for entry in _normalize_signal_negative_target_log(
                scan_state.get("negative_target_log") or []
            )
            if 0 <= current_step - int(entry.get("step", 0)) <= window
        ]
    return True


def _mix_oracle_rollin_messages(
    model_actions: dict[int, dict],
    oracle_actions: dict[int, dict],
    rate: float,
    rng: np.random.Generator,
) -> tuple[dict[int, dict], int, int]:
    rate = min(1.0, max(0.0, float(rate)))
    mixed: dict[int, dict] = {}
    replaced_agents = 0
    replaced_tokens = 0
    for aid, model_action in model_actions.items():
        aid_int = int(aid)
        action = dict(model_action)
        action["message_tokens"] = [int(t) for t in action.get("message_tokens", [])]
        if rate > 0.0 and float(rng.random()) < rate:
            oracle_tokens = [
                int(t)
                for t in oracle_actions.get(aid_int, oracle_actions.get(aid, {})).get(
                    "message_tokens",
                    [],
                )
            ]
            action["message_tokens"] = oracle_tokens
            replaced_agents += 1
            replaced_tokens += len(oracle_tokens)
        mixed[aid_int] = action
    return mixed, replaced_agents, replaced_tokens


def _signal_target_interact_agents(env: SyncOrSinkEnv, actions: dict[int, dict]) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    target = env.scenario_state.data.get("target")
    if target is None:
        return []
    return [
        int(aid)
        for aid, action in actions.items()
        if int(action.get("action", -1)) == env.ACTION_INTERACT
        and tuple(env.agent_positions[int(aid)]) == tuple(target)
    ]


def _signal_rejected_target_interact_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    actions: dict[int, dict],
) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    true_target = env.scenario_state.data.get("target")
    true_target_pos = tuple(true_target) if true_target is not None else None
    agents: list[int] = []
    for aid, action in actions.items():
        aid = int(aid)
        if int(action.get("action", -1)) != env.ACTION_INTERACT:
            continue
        if true_target_pos is not None and tuple(env.agent_positions[aid]) == true_target_pos:
            continue
        obs_agent = obs.get(aid)
        if obs_agent is None:
            continue
        if _signal_center_rejected_target(obs_agent, int(env.map_size)):
            agents.append(aid)
    return agents


def _signal_bad_redundant_target_interact_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    actions: dict[int, dict],
) -> list[int]:
    del obs
    bad_agents = set(_signal_bad_redundant_target_scan_agents(env))
    if not bad_agents:
        return []
    return [
        int(aid)
        for aid, action in actions.items()
        if int(aid) in bad_agents and int(action.get("action", -1)) == env.ACTION_INTERACT
    ]


def _redundant_target_scan_agents(env: SyncOrSinkEnv, actions: dict[int, dict]) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    scan_log = env.scenario_state.data.get("scan_log") or {}
    scan_window = int(env.scenario_state.data.get("scan_window", getattr(env.config, "scan_window", 3)))
    next_step = int(env.steps) + 1
    redundant = []
    for aid in _signal_target_interact_agents(env, actions):
        last_scan = scan_log.get(int(aid), scan_log.get(str(int(aid))))
        if last_scan is None:
            continue
        if next_step - int(last_scan) <= scan_window:
            redundant.append(int(aid))
    return redundant


_SIGNAL_TARGET_SCAN_KIND_UNKNOWN = -1
_SIGNAL_TARGET_SCAN_KIND_FIRST = 0
_SIGNAL_TARGET_SCAN_KIND_REFRESH = 1
_SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION = 2
_SIGNAL_TARGET_SCAN_KIND_REDUNDANT_ACTIVE = 3
_SIGNAL_TARGET_SCAN_KIND_NAMES = (
    "first",
    "refresh",
    "joint_completion",
    "redundant_active",
)


def _signal_target_scan_kind(env: SyncOrSinkEnv, agent_id: int, *, next_step: int | None = None) -> int:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return _SIGNAL_TARGET_SCAN_KIND_UNKNOWN
    target = env.scenario_state.data.get("target")
    if target is None:
        return _SIGNAL_TARGET_SCAN_KIND_UNKNOWN
    agent_id = int(agent_id)
    if tuple(env.agent_positions[agent_id]) != tuple(target):
        return _SIGNAL_TARGET_SCAN_KIND_UNKNOWN
    if next_step is None:
        next_step = int(env.steps) + 1
    scan_window = int(env.scenario_state.data.get("scan_window", getattr(env.config, "scan_window", 3)))
    for other_id in range(env.num_agents):
        if int(other_id) == agent_id:
            continue
        other_age = _recent_signal_scan_age(env, int(other_id), next_step=int(next_step))
        if other_age is not None and 0 <= int(other_age) <= scan_window:
            return _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION
    self_age = _recent_signal_scan_age(env, agent_id, next_step=int(next_step))
    if self_age is None:
        return _SIGNAL_TARGET_SCAN_KIND_FIRST
    if int(self_age) >= scan_window:
        return _SIGNAL_TARGET_SCAN_KIND_REFRESH
    return _SIGNAL_TARGET_SCAN_KIND_REDUNDANT_ACTIVE


def _signal_target_scan_label_mask(env: SyncOrSinkEnv, actions: dict[int, dict]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((env.num_agents,), dtype=np.float32)
    kind_ids = np.full((env.num_agents,), _SIGNAL_TARGET_SCAN_KIND_UNKNOWN, dtype=np.int64)
    for aid in _signal_target_interact_agents(env, actions):
        kind = _signal_target_scan_kind(env, int(aid))
        if kind < 0:
            continue
        mask[int(aid)] = 1.0
        kind_ids[int(aid)] = int(kind)
    return mask, kind_ids


def _signal_target_scan_opportunity_label_mask(
    env: SyncOrSinkEnv,
    obs: dict,
    cfg: RecurrentConfig,
    feedback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((env.num_agents,), dtype=np.float32)
    kind_ids = np.full((env.num_agents,), _SIGNAL_TARGET_SCAN_KIND_UNKNOWN, dtype=np.int64)
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return mask, kind_ids
    target = env.scenario_state.data.get("target")
    if target is None:
        return mask, kind_ids
    true_target = (int(target[0]), int(target[1]))
    for aid in range(env.num_agents):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        if not _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_INTERACT):
            continue
        pos_arr = np.asarray(
            obs_agent.get("self_pos", env.agent_positions[aid]),
            dtype=np.int64,
        ).reshape(-1)
        if pos_arr.size < 2:
            continue
        pos = (int(pos_arr[0]), int(pos_arr[1]))
        if pos != true_target:
            continue
        kind = _signal_target_scan_kind(env, int(aid))
        if kind not in (
            _SIGNAL_TARGET_SCAN_KIND_FIRST,
            _SIGNAL_TARGET_SCAN_KIND_REFRESH,
            _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION,
        ):
            continue
        if not _signal_center_target_scan_decoding_candidate(obs_agent, cfg):
            observed_map_size = _observed_map_size(obs_agent, cfg)
            if (
                kind != _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION
                or not _signal_feedback_teammate_scan_active(cfg, feedback, int(aid), env.num_agents)
                or not _signal_center_visible_target_tile(obs_agent, cfg)
                or _signal_center_rejected_target(obs_agent, observed_map_size)
            ):
                continue
        mask[int(aid)] = 1.0
        kind_ids[int(aid)] = int(kind)
    return mask, kind_ids


def _signal_feedback_teammate_scan_active(
    cfg: RecurrentConfig,
    feedback: np.ndarray | None,
    agent_id: int,
    num_agents: int,
) -> bool:
    if feedback is None or not (cfg.obs_feedback and cfg.obs_signal_scan_state):
        return False
    try:
        feedback_arr = np.asarray(feedback, dtype=np.float32).reshape(int(num_agents), -1)
    except (TypeError, ValueError):
        return False
    scan_offset = 12 + (4 if cfg.obs_signal_sync_feedback else 0)
    agent_id = int(agent_id)
    if agent_id < 0 or agent_id >= feedback_arr.shape[0] or feedback_arr.shape[1] <= scan_offset + 1:
        return False
    return float(feedback_arr[agent_id, scan_offset + 1]) > 0.0


def _apply_redundant_target_scan_penalty(
    rewards: dict[int, float],
    redundant_agents: list[int],
    penalty: float,
) -> tuple[int, float]:
    penalty = max(0.0, float(penalty))
    if penalty <= 0.0 or not redundant_agents:
        return 0, 0.0

    applied = 0
    for aid in redundant_agents:
        aid = int(aid)
        if aid not in rewards:
            continue
        rewards[aid] = float(rewards[aid]) - penalty
        applied += 1
    return applied, penalty * applied


def _wrong_target_scan_agents(info: dict | None, num_agents: int) -> list[int]:
    return [
        int(aid)
        for aid, names in _event_names_by_agent(info, num_agents).items()
        if "decoy_scan" in names
    ]


def _apply_wrong_target_scan_penalty(
    rewards: dict[int, float],
    wrong_agents: list[int],
    penalty: float,
) -> tuple[int, float]:
    penalty = max(0.0, float(penalty))
    if penalty <= 0.0 or not wrong_agents:
        return 0, 0.0

    applied = 0
    for aid in wrong_agents:
        aid = int(aid)
        if aid not in rewards:
            continue
        rewards[aid] = float(rewards[aid]) - penalty
        applied += 1
    return applied, penalty * applied


def _move_delta_for_action(env: SyncOrSinkEnv, action_id: int) -> tuple[int, int] | None:
    return {
        env.ACTION_UP: (0, -1),
        env.ACTION_DOWN: (0, 1),
        env.ACTION_LEFT: (-1, 0),
        env.ACTION_RIGHT: (1, 0),
    }.get(int(action_id))


def _signal_action_moves_toward_target(
    env: SyncOrSinkEnv,
    *,
    pos: tuple[int, int],
    target: tuple[int, int],
    action_id: int,
    rejected_targets: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
) -> bool:
    if int(action_id) == env.ACTION_INTERACT:
        return tuple(pos) == tuple(target)
    delta = _move_delta_for_action(env, int(action_id))
    if delta is None:
        return False
    nx, ny = int(pos[0]) + int(delta[0]), int(pos[1]) + int(delta[1])
    if not (0 <= nx < env.map_size and 0 <= ny < env.map_size):
        return False
    if int(env.grid[ny, nx]) in (1, TILE_DOOR):
        return False
    if (nx, ny) in set(rejected_targets):
        return False
    current_dist = abs(int(target[0]) - int(pos[0])) + abs(int(target[1]) - int(pos[1]))
    next_dist = abs(int(target[0]) - int(nx)) + abs(int(target[1]) - int(ny))
    return next_dist < current_dist


def _signal_target_decoding_candidate(
    obs_agent: dict,
    cfg: RecurrentConfig,
    target: tuple[int, int],
) -> bool:
    if cfg.scenario != "signal_hunt" or not isinstance(obs_agent, dict):
        return False
    if not _signal_observation_has_target_information(obs_agent):
        return False
    pos_arr = np.asarray(
        obs_agent.get("self_pos", np.zeros((2,), dtype=np.int16)),
        dtype=np.int64,
    ).reshape(-1)
    if pos_arr.size < 2:
        return False
    observed_map_size = _observed_map_size(obs_agent, cfg)
    target = (int(target[0]), int(target[1]))
    if not _signal_observation_allows_target(obs_agent, target, observed_map_size):
        return False

    own_segments, message_segments = _signal_segments_from_observation(obs_agent)
    exact_targets = set(_signal_targets_from_segments([*own_segments, *message_segments], observed_map_size))
    if target in exact_targets:
        return True

    visible_targets = set(_visible_signal_targets(obs_agent, pos_arr, observed_map_size))
    if target in visible_targets:
        allowed_visible_targets = {
            visible_target
            for visible_target in visible_targets
            if _signal_observation_allows_target(obs_agent, visible_target, observed_map_size)
        }
        if allowed_visible_targets == {target}:
            return True

    inferred_targets = set(_signal_inferred_constraint_targets(obs_agent, observed_map_size))
    return inferred_targets == {target}


def _signal_target_pursuit_action_label_mask(
    env: SyncOrSinkEnv,
    obs: dict,
    cfg: RecurrentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((env.num_agents,), dtype=np.float32)
    action_ids = np.full((env.num_agents,), -1, dtype=np.int64)
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return mask, action_ids
    target = env.scenario_state.data.get("target")
    if target is None:
        return mask, action_ids
    true_target = (int(target[0]), int(target[1]))
    observed_map_size = int(env.map_size)
    for aid in range(env.num_agents):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        pos_arr = np.asarray(
            obs_agent.get("self_pos", env.agent_positions[aid]),
            dtype=np.int64,
        ).reshape(-1)
        if pos_arr.size < 2:
            continue
        pos = (int(pos_arr[0]), int(pos_arr[1]))
        if pos == true_target:
            continue
        if not _signal_target_decoding_candidate(obs_agent, cfg, true_target):
            continue
        visible_targets = _visible_signal_targets(obs_agent, pos_arr, observed_map_size)
        rejected_targets = {
            visible_target
            for visible_target in visible_targets
            if visible_target != true_target
            and not _signal_observation_allows_target(obs_agent, visible_target, observed_map_size)
        }
        dx, dy, path_target = shortest_path(
            env.grid,
            pos,
            [true_target],
            blocked_positions=rejected_targets,
        )
        if path_target != true_target:
            continue
        action_id = int(move_action_from_delta(dx, dy, env))
        if action_id == int(env.ACTION_STAY):
            continue
        if not _action_allowed_from_obs(obs_agent, action_id):
            continue
        mask[aid] = 1.0
        action_ids[aid] = action_id
    return mask, action_ids


def _signal_target_match_action_label_mask(
    env: SyncOrSinkEnv,
    obs: dict,
    actions: dict[int, dict],
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((env.num_agents,), dtype=np.float32)
    action_ids = np.full((env.num_agents,), -1, dtype=np.int64)
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return mask, action_ids
    target = env.scenario_state.data.get("target")
    if target is None:
        return mask, action_ids
    true_target = (int(target[0]), int(target[1]))
    observed_map_size = int(env.map_size)
    for aid in range(env.num_agents):
        obs_agent = obs.get(aid)
        if obs_agent is None or not _signal_observation_has_target_information(obs_agent):
            continue
        pos_arr = np.asarray(obs_agent.get("self_pos", env.agent_positions[aid]), dtype=np.int64).reshape(-1)
        if pos_arr.size < 2:
            continue
        pos = (int(pos_arr[0]), int(pos_arr[1]))
        visible_targets = _visible_signal_targets(obs_agent, pos_arr, observed_map_size)
        if true_target not in visible_targets:
            continue
        if not _signal_observation_allows_target(obs_agent, true_target, observed_map_size):
            continue
        rejected_targets = [
            visible_target
            for visible_target in visible_targets
            if visible_target != true_target
            and not _signal_observation_allows_target(obs_agent, visible_target, observed_map_size)
        ]
        if not rejected_targets:
            continue
        action = int(actions.get(aid, {}).get("action", -1))
        if not _signal_action_moves_toward_target(
            env,
            pos=pos,
            target=true_target,
            action_id=action,
            rejected_targets=rejected_targets,
        ):
            continue
        mask[aid] = 1.0
        action_ids[aid] = action
    return mask, action_ids


def _signal_target_pursuit_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    actions: dict[int, dict],
) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    target = env.scenario_state.data.get("target")
    if target is None:
        return []
    target = tuple(target)
    agents: list[int] = []
    for aid, action in actions.items():
        aid = int(aid)
        delta = _move_delta_for_action(env, int(action.get("action", -1)))
        if delta is None:
            continue
        obs_agent = obs.get(aid)
        if obs_agent is None:
            continue
        if not _signal_observation_allows_target(
            obs_agent,
            target,
            observed_map_size=int(env.map_size),
        ):
            continue
        x, y = env.agent_positions[aid]
        nx, ny = x + delta[0], y + delta[1]
        if not (0 <= nx < env.map_size and 0 <= ny < env.map_size):
            continue
        if int(env.grid[ny, nx]) in (1, TILE_DOOR):
            continue
        current_dist = abs(int(target[0]) - int(x)) + abs(int(target[1]) - int(y))
        next_dist = abs(int(target[0]) - int(nx)) + abs(int(target[1]) - int(ny))
        if next_dist < current_dist:
            agents.append(aid)
    return agents


def _signal_positive_target_pursuit_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    actions: dict[int, dict],
    *,
    min_map_size: int = 16,
) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    if int(env.map_size) < int(min_map_size):
        return []
    return _signal_target_pursuit_agents(env, obs, actions)


def _signal_sync_response_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    actions: dict[int, dict],
    feedback: np.ndarray | None,
    cfg: RecurrentConfig | None = None,
) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt" or feedback is None:
        return []
    feedback_arr = np.asarray(feedback, dtype=np.float32)
    if feedback_arr.ndim != 2 or feedback_arr.shape[1] < 14:
        return []
    target_interactors = set(_signal_target_interact_agents(env, actions))
    target_pursuers = set(_signal_target_pursuit_agents(env, obs, actions))
    responders = []
    for aid in range(min(env.num_agents, feedback_arr.shape[0])):
        row = feedback_arr[aid]
        teammate_scanned_now = float(row[13]) > 0.0
        self_scan_active = False
        teammate_scan_active = False
        if cfg is not None and bool(getattr(cfg, "obs_signal_scan_state", False)):
            scan_offset = 12 + (4 if bool(getattr(cfg, "obs_signal_sync_feedback", False)) else 0)
            if row.shape[0] > scan_offset + 1:
                self_scan_active = float(row[scan_offset]) > 0.0
                teammate_scan_active = float(row[scan_offset + 1]) > 0.0
        if (
            (teammate_scanned_now or teammate_scan_active)
            and not self_scan_active
            and (aid in target_interactors or aid in target_pursuers)
        ):
            responders.append(aid)
    return responders


def _signal_sync_response_action_label_mask(
    env: SyncOrSinkEnv,
    obs: dict,
    actions: dict[int, dict],
    feedback: np.ndarray | None,
    cfg: RecurrentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((env.num_agents,), dtype=np.float32)
    action_ids = np.full((env.num_agents,), -1, dtype=np.int64)
    for aid in _signal_sync_response_agents(env, obs, actions, feedback, cfg=cfg):
        action_id = int(actions.get(int(aid), {}).get("action", -1))
        obs_agent = obs.get(int(aid), obs.get(str(int(aid)))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        if not _action_allowed_from_obs(obs_agent, action_id):
            continue
        mask[int(aid)] = 1.0
        action_ids[int(aid)] = action_id
    return mask, action_ids


def _signal_target_pursuit_miss_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    oracle_actions: dict[int, dict],
    model_actions: dict[int, dict],
) -> list[int]:
    oracle_pursuers = set(_signal_target_pursuit_agents(env, obs, oracle_actions))
    if not oracle_pursuers:
        return []
    model_pursuers = set(_signal_target_pursuit_agents(env, obs, model_actions))
    return sorted(int(aid) for aid in oracle_pursuers if int(aid) not in model_pursuers)


def _signal_target_interact_miss_agents(
    env: SyncOrSinkEnv,
    oracle_actions: dict[int, dict],
    model_actions: dict[int, dict],
) -> list[int]:
    oracle_interactors = set(_signal_target_interact_agents(env, oracle_actions))
    if not oracle_interactors:
        return []
    model_interactors = set(_signal_target_interact_agents(env, model_actions))
    return sorted(int(aid) for aid in oracle_interactors if int(aid) not in model_interactors)


def _signal_decoy_pursuit_agents(
    env: SyncOrSinkEnv,
    actions: dict[int, dict],
) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    decoys = [tuple(pos) for pos in env.scenario_state.data.get("decoys", [])]
    if not decoys:
        return []
    agents: list[int] = []
    for aid, action in actions.items():
        aid = int(aid)
        delta = _move_delta_for_action(env, int(action.get("action", -1)))
        if delta is None:
            continue
        x, y = env.agent_positions[aid]
        nx, ny = x + delta[0], y + delta[1]
        if not (0 <= nx < env.map_size and 0 <= ny < env.map_size):
            continue
        if int(env.grid[ny, nx]) in (1, TILE_DOOR):
            continue
        current_dist = min(abs(int(dx) - int(x)) + abs(int(dy) - int(y)) for dx, dy in decoys)
        next_dist = min(abs(int(dx) - int(nx)) + abs(int(dy) - int(ny)) for dx, dy in decoys)
        if next_dist < current_dist:
            agents.append(aid)
    return agents


def _signal_rejected_target_drift_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    actions: dict[int, dict],
) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    agents: list[int] = []
    for aid, action in actions.items():
        aid = int(aid)
        delta = _move_delta_for_action(env, int(action.get("action", -1)))
        if delta is None:
            continue
        obs_agent = obs.get(aid)
        if obs_agent is None:
            continue
        if not _signal_observation_has_target_information(obs_agent):
            continue
        pos_arr = np.asarray(obs_agent.get("self_pos", env.agent_positions[aid]), dtype=np.int64).reshape(-1)
        if pos_arr.size < 2:
            continue
        pos = (int(pos_arr[0]), int(pos_arr[1]))
        nx, ny = pos[0] + int(delta[0]), pos[1] + int(delta[1])
        if not (0 <= nx < env.map_size and 0 <= ny < env.map_size):
            continue
        if int(env.grid[ny, nx]) in (1, TILE_DOOR):
            continue
        visible_targets = _visible_signal_targets(obs_agent, pos_arr, int(env.map_size))
        if not visible_targets:
            continue
        allowed_targets = [
            target
            for target in visible_targets
            if _signal_observation_allows_target(obs_agent, target, int(env.map_size))
        ]
        allowed_set = set(allowed_targets)
        rejected_targets = [target for target in visible_targets if target not in allowed_set]
        if not rejected_targets:
            continue

        def nearest_distance(targets: list[tuple[int, int]], point: tuple[int, int]) -> int | None:
            if not targets:
                return None
            return min(abs(int(tx) - point[0]) + abs(int(ty) - point[1]) for tx, ty in targets)

        current_rejected_dist = nearest_distance(rejected_targets, pos)
        next_rejected_dist = nearest_distance(rejected_targets, (nx, ny))
        if current_rejected_dist is None or next_rejected_dist is None:
            continue
        if next_rejected_dist >= current_rejected_dist:
            continue
        if allowed_targets:
            current_allowed_dist = nearest_distance(allowed_targets, pos)
            next_allowed_dist = nearest_distance(allowed_targets, (nx, ny))
            if (
                current_allowed_dist is not None
                and next_allowed_dist is not None
                and next_allowed_dist < current_allowed_dist
                and next_allowed_dist <= next_rejected_dist
            ):
                continue
        agents.append(aid)
    return agents


def _signal_target_decoy_drift_miss_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    oracle_actions: dict[int, dict],
    model_actions: dict[int, dict],
    *,
    min_map_size: int = 16,
) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    if int(env.map_size) < int(min_map_size):
        return []
    oracle_pursuers = set(_signal_target_pursuit_agents(env, obs, oracle_actions))
    if not oracle_pursuers:
        return []
    model_pursuers = set(_signal_target_pursuit_agents(env, obs, model_actions))
    decoy_pursuers = set(_signal_decoy_pursuit_agents(env, model_actions))
    return sorted(
        int(aid)
        for aid in oracle_pursuers
        if int(aid) in decoy_pursuers and int(aid) not in model_pursuers
    )


def _signal_target_discovery_miss_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    oracle_actions: dict[int, dict],
    model_actions: dict[int, dict],
    *,
    min_map_size: int = 16,
) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    if int(env.map_size) < int(min_map_size):
        return []
    oracle_pursuers = set(_signal_target_pursuit_agents(env, obs, oracle_actions))
    if not oracle_pursuers:
        return []
    model_pursuers = set(_signal_target_pursuit_agents(env, obs, model_actions))
    decoy_drifters = set(_signal_target_decoy_drift_miss_agents(
        env,
        obs,
        oracle_actions,
        model_actions,
        min_map_size=min_map_size,
    ))
    missed = set(int(aid) for aid in oracle_pursuers if int(aid) not in model_pursuers)
    return sorted(missed - decoy_drifters)


def _signal_movement_stall_miss_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    oracle_actions: dict[int, dict],
    model_actions: dict[int, dict],
    position_history: Mapping[int, list[tuple[int, int]]],
    *,
    min_map_size: int = 16,
    window: int = 6,
) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    if int(env.map_size) < int(min_map_size) or int(window) < 3:
        return []
    target = env.scenario_state.data.get("target")
    movement_actions = {
        env.ACTION_UP,
        env.ACTION_DOWN,
        env.ACTION_LEFT,
        env.ACTION_RIGHT,
        env.ACTION_STAY,
    }
    stalled: list[int] = []
    for aid in range(env.num_agents):
        oracle_action = int(oracle_actions.get(aid, {}).get("action", -1))
        model_action = int(model_actions.get(aid, {}).get("action", -1))
        if oracle_action == model_action or model_action not in movement_actions:
            continue
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        pos = _signal_xy(obs_agent.get("self_pos")) if isinstance(obs_agent, dict) else None
        if pos is None:
            pos = tuple(env.agent_positions[int(aid)])
        if target is not None and tuple(pos) == tuple(target):
            continue
        recent = [*position_history.get(int(aid), []), tuple(pos)][-int(window):]
        if len(recent) < int(window):
            continue
        if len(set(recent)) <= 2:
            stalled.append(int(aid))
    return stalled


def _signal_target_handoff_miss_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    oracle_actions: dict[int, dict],
    model_actions: dict[int, dict],
    feedback: np.ndarray | None,
    cfg: RecurrentConfig | None = None,
) -> list[int]:
    missed: set[int] = set()
    oracle_responders = set(_signal_sync_response_agents(env, obs, oracle_actions, feedback, cfg=cfg))
    if oracle_responders:
        model_responders = set(_signal_sync_response_agents(env, obs, model_actions, feedback, cfg=cfg))
        missed.update(
            int(aid)
            for aid in oracle_responders
            if int(aid) not in model_responders
        )

    model_target_interactors = set(_signal_target_interact_agents(env, model_actions))
    if len(model_target_interactors) == 1:
        oracle_joiners = set(_signal_target_pursuit_agents(env, obs, oracle_actions))
        oracle_joiners.update(_signal_target_interact_agents(env, oracle_actions))
        model_joiners = set(_signal_target_pursuit_agents(env, obs, model_actions))
        model_joiners.update(model_target_interactors)
        missed.update(
            int(aid)
            for aid in oracle_joiners
            if int(aid) not in model_target_interactors and int(aid) not in model_joiners
        )
    return sorted(missed)


def _apply_signal_target_interact_weight(
    weights: np.ndarray,
    *,
    env: SyncOrSinkEnv,
    cfg: RecurrentConfig,
    actions: dict[int, dict],
) -> np.ndarray:
    target_weight = float(getattr(cfg, "bc_signal_target_interact_weight", 1.0))
    redundant_weight = float(getattr(cfg, "bc_signal_redundant_target_interact_weight", 1.0))
    if target_weight <= 1.0 and redundant_weight <= 1.0:
        return weights
    weighted = np.asarray(weights, dtype=np.float32).copy()
    redundant_agents = set(_redundant_target_scan_agents(env, actions))
    for aid in _signal_target_interact_agents(env, actions):
        step_weight = redundant_weight if int(aid) in redundant_agents else target_weight
        weighted[int(aid)] = max(float(weighted[int(aid)]), step_weight)
    return weighted


def _apply_signal_target_pursuit_weight(
    weights: np.ndarray,
    *,
    env: SyncOrSinkEnv,
    cfg: RecurrentConfig,
    obs: dict,
    actions: dict[int, dict],
) -> np.ndarray:
    target_weight = float(getattr(cfg, "bc_signal_target_pursuit_weight", 1.0))
    if target_weight <= 1.0:
        return weights
    weighted = np.asarray(weights, dtype=np.float32).copy()
    for aid in _signal_target_pursuit_agents(env, obs, actions):
        weighted[int(aid)] = max(float(weighted[int(aid)]), target_weight)
    return weighted


def _apply_signal_sync_response_weight(
    weights: np.ndarray,
    *,
    env: SyncOrSinkEnv,
    cfg: RecurrentConfig,
    obs: dict,
    actions: dict[int, dict],
    feedback: np.ndarray | None,
) -> np.ndarray:
    target_weight = float(getattr(cfg, "bc_signal_sync_response_weight", 1.0))
    if target_weight <= 1.0 or not cfg.obs_signal_sync_feedback:
        return weights
    weighted = np.asarray(weights, dtype=np.float32).copy()
    for aid in _signal_sync_response_agents(env, obs, actions, feedback, cfg=cfg):
        weighted[int(aid)] = max(float(weighted[int(aid)]), target_weight)
    return weighted


def _signal_target_join_action(env: SyncOrSinkEnv, agent_id: int) -> int:
    target = env.scenario_state.data.get("target")
    if target is None:
        return env.ACTION_STAY
    target = tuple(target)
    agent_id = int(agent_id)
    pos = tuple(env.agent_positions[agent_id])
    if pos == target:
        return env.ACTION_INTERACT
    blocked_positions = {
        tuple(other_pos)
        for other_id, other_pos in enumerate(env.agent_positions)
        if int(other_id) != agent_id and tuple(other_pos) != target
    }
    dx, dy, reached = shortest_path(
        env.grid,
        pos,
        {target},
        blocked_positions=blocked_positions,
    )
    if reached is None:
        return env.ACTION_STAY
    return move_action_from_delta(dx, dy, env)


def _signal_feedback_target_handoff_agents(
    cfg: RecurrentConfig,
    env: SyncOrSinkEnv,
    feedback: np.ndarray | None,
    obs: dict | None = None,
) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    if feedback is None:
        return []
    feedback_arr = np.asarray(feedback, dtype=np.float32)
    if feedback_arr.ndim != 2 or feedback_arr.shape[0] < env.num_agents:
        return []
    agents: list[int] = []
    sync_offset = 12
    scan_offset = 12 + (4 if cfg.obs_signal_sync_feedback else 0)
    target = env.scenario_state.data.get("target")
    true_target = (int(target[0]), int(target[1])) if target is not None else None
    for aid in range(env.num_agents):
        row = feedback_arr[int(aid)]
        teammate_scan_now = (
            cfg.obs_signal_sync_feedback
            and row.shape[0] > sync_offset + 1
            and float(row[sync_offset + 1]) > 0.0
        )
        self_scan_active = False
        teammate_scan_active = False
        if cfg.obs_signal_scan_state and row.shape[0] > scan_offset + 1:
            self_scan_active = float(row[scan_offset]) > 0.0
            teammate_scan_active = float(row[scan_offset + 1]) > 0.0
        if (teammate_scan_now or teammate_scan_active) and not self_scan_active:
            if obs is not None:
                obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
                if not isinstance(obs_agent, dict) or true_target is None:
                    continue
                if not _signal_target_decoding_candidate(obs_agent, cfg, true_target):
                    continue
            agents.append(int(aid))
    return agents


def _apply_signal_target_handoff_overrides(
    cfg: RecurrentConfig,
    env: SyncOrSinkEnv,
    actions: dict[int, dict],
    feedback: np.ndarray | None,
    obs: dict | None = None,
) -> tuple[dict[int, dict], list[int]]:
    handoff_agents = _signal_feedback_target_handoff_agents(cfg, env, feedback, obs=obs)
    if not handoff_agents:
        return actions, []
    corrected = {int(aid): dict(action) for aid, action in actions.items()}
    for aid in handoff_agents:
        action = corrected.setdefault(int(aid), {"message_tokens": []})
        action["action"] = _signal_target_join_action(env, int(aid))
        action.setdefault("message_tokens", [])
    return corrected, handoff_agents


def _signal_exact_target_message(cfg: RecurrentConfig, env: SyncOrSinkEnv) -> list[int]:
    target = env.scenario_state.data.get("target")
    if target is None:
        return []
    return _signal_exact_target_message_for_pos(cfg, tuple(target))


def _signal_exact_target_message_for_pos(
    cfg: RecurrentConfig,
    pos: tuple[int, int] | None,
) -> list[int]:
    if pos is None or not cfg.comm or int(cfg.comm_token_limit) < 3 or int(cfg.comm_vocab_size) <= 26:
        return []
    tx, ty = int(pos[0]), int(pos[1])
    if not (0 <= tx < int(cfg.comm_vocab_size) and 0 <= ty < int(cfg.comm_vocab_size)):
        return []
    return [26, tx, ty]


def _signal_target_scan_broadcaster_agents(
    cfg: RecurrentConfig,
    env: SyncOrSinkEnv,
    feedback: np.ndarray | None,
    info: dict | None = None,
) -> list[int]:
    del feedback
    if not cfg.dagger_target_scan_broadcast_labels:
        return []
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    target = env.scenario_state.data.get("target")
    if target is None or not _signal_exact_target_message(cfg, env):
        return []
    names_by_agent = _event_names_by_agent(info or {}, env.num_agents)
    broadcasters: list[int] = []
    for aid in range(env.num_agents):
        if tuple(env.agent_positions[int(aid)]) != tuple(target):
            continue
        if "first_target_scan" in names_by_agent.get(int(aid), set()):
            broadcasters.append(int(aid))
    return broadcasters


def _apply_signal_target_scan_broadcast_overrides(
    cfg: RecurrentConfig,
    env: SyncOrSinkEnv,
    actions: dict[int, dict],
    feedback: np.ndarray | None,
    info: dict | None = None,
) -> tuple[dict[int, dict], list[int]]:
    broadcasters = _signal_target_scan_broadcaster_agents(cfg, env, feedback, info=info)
    if not broadcasters:
        return actions, []
    message = _signal_exact_target_message(cfg, env)
    if not message:
        return actions, []
    corrected = {int(aid): dict(action) for aid, action in actions.items()}
    for aid in broadcasters:
        action = corrected.setdefault(int(aid), {"action": env.ACTION_STAY})
        action["message_tokens"] = list(message)
    return corrected, broadcasters


def _apply_signal_redundant_target_wait_overrides(
    env: SyncOrSinkEnv,
    actions: dict[int, dict],
) -> tuple[dict[int, dict], list[int]]:
    scan_window = int(env.scenario_state.data.get("scan_window", getattr(env.config, "scan_window", 3)))
    next_step = int(env.steps) + 1
    wait_agents = []
    for aid in _redundant_target_scan_agents(env, actions):
        age = _recent_signal_scan_age(env, int(aid), next_step=next_step)
        if age is not None and 0 <= int(age) < scan_window:
            wait_agents.append(int(aid))
    if not wait_agents:
        return actions, []
    corrected = {int(aid): dict(action) for aid, action in actions.items()}
    label_agents: list[int] = []
    for aid in sorted(wait_agents):
        action = corrected.get(int(aid))
        if action is None or int(action.get("action", -1)) != env.ACTION_INTERACT:
            continue
        action["action"] = env.ACTION_STAY
        action.setdefault("message_tokens", [])
        label_agents.append(int(aid))
    return corrected, label_agents


def _signal_redundant_target_wait_action_label_mask(
    env: SyncOrSinkEnv,
    obs: dict,
    actions: dict[int, dict],
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((env.num_agents,), dtype=np.float32)
    action_ids = np.full((env.num_agents,), -1, dtype=np.int64)
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return mask, action_ids
    target = env.scenario_state.data.get("target")
    if target is None:
        return mask, action_ids
    scan_window = int(env.scenario_state.data.get("scan_window", getattr(env.config, "scan_window", 3)))
    next_step = int(env.steps) + 1
    for aid in range(env.num_agents):
        if tuple(env.agent_positions[int(aid)]) != tuple(target):
            continue
        action_id = int(actions.get(int(aid), {}).get("action", -1))
        if action_id != int(env.ACTION_STAY):
            continue
        age = _recent_signal_scan_age(env, int(aid), next_step=next_step)
        if age is None or not (0 <= int(age) < scan_window):
            continue
        obs_agent = obs.get(int(aid), obs.get(str(int(aid)))) if isinstance(obs, dict) else None
        if isinstance(obs_agent, dict) and not _action_allowed_from_obs(obs_agent, env.ACTION_STAY):
            continue
        mask[int(aid)] = 1.0
        action_ids[int(aid)] = int(env.ACTION_STAY)
    return mask, action_ids


def _make_oracle_policy(env: SyncOrSinkEnv, cfg: RecurrentConfig):
    from syncorsink.policies.comm_wrapper import wrap_oracle_with_comm
    from syncorsink.policies.local_oracle import local_signal_policy
    from syncorsink.policies.oracle import (
        energy_oracle_strong,
        pipeline_oracle_strong,
        signal_hunt_oracle_strong,
    )

    if cfg.oracle_type == "signal_hint_comm":
        return local_signal_policy(env)
    oracle_map = {
        "signal_hunt": signal_hunt_oracle_strong,
        "energy_grid": energy_oracle_strong,
        "pipeline_assembly": pipeline_oracle_strong,
    }
    base = oracle_map[cfg.scenario](env)
    return wrap_oracle_with_comm(base, env) if cfg.oracle_type.endswith("_comm") else base


def _new_episode_sequence() -> dict:
    return {
        "obs": [],
        "actions": [],
        "msg_tokens": [],
        "msg_lens": [],
        "step_weights": [],
        "signal_rejected_target_mask": [],
        "signal_bad_redundant_target_mask": [],
        "signal_target_scan_action_mask": [],
        "signal_target_scan_kind_id": [],
        "signal_target_opportunity_action_mask": [],
        "signal_target_opportunity_kind_id": [],
        "signal_redundant_target_wait_action_mask": [],
        "signal_redundant_target_wait_action_id": [],
        "signal_target_pursuit_action_mask": [],
        "signal_target_pursuit_action_id": [],
        "signal_sync_response_action_mask": [],
        "signal_sync_response_action_id": [],
        "signal_target_match_action_mask": [],
        "signal_target_match_action_id": [],
        "signal_target_validity_mask": [],
        "signal_target_validity_label": [],
        "signal_target_decision_mask": [],
        "signal_target_decision_label": [],
        "signal_target_aux_mask": [],
        "signal_target_aux_xy": [],
        "signal_decoy_drift_action_mask": [],
        "signal_decoy_drift_action_id": [],
        "signal_decoy_scan_action_mask": [],
        "signal_decoy_scan_action_id": [],
        "signal_rejected_target_drift_action_mask": [],
        "signal_rejected_target_drift_action_id": [],
    }


def _signal_target_aux_label(env: SyncOrSinkEnv) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((env.num_agents,), dtype=np.float32)
    xy = np.zeros((env.num_agents, 2), dtype=np.float32)
    if env.config.scenario != "signal_hunt" or env.scenario_state is None:
        return mask, xy
    target = env.scenario_state.data.get("target")
    if target is None:
        return mask, xy
    denom = max(1.0, float(env.map_size - 1))
    target_xy = np.array([float(target[0]) / denom, float(target[1]) / denom], dtype=np.float32)
    mask[:] = 1.0
    xy[:, :] = target_xy
    return mask, xy


def _signal_target_validity_label(env: SyncOrSinkEnv, obs: dict) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((env.num_agents,), dtype=np.float32)
    labels = np.zeros((env.num_agents,), dtype=np.float32)
    if getattr(env.config, "scenario", None) != "signal_hunt" or env.scenario_state is None:
        return mask, labels
    target = env.scenario_state.data.get("target")
    if target is None:
        return mask, labels
    true_target = (int(target[0]), int(target[1]))
    observed_map_size = int(env.map_size)
    for aid in range(env.num_agents):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        if not _signal_observation_has_target_information(obs_agent):
            continue
        pos_arr = np.asarray(obs_agent.get("self_pos", env.agent_positions[aid]), dtype=np.int64).reshape(-1)
        if pos_arr.size < 2:
            continue
        pos = (int(pos_arr[0]), int(pos_arr[1]))
        if pos not in set(_visible_signal_targets(obs_agent, pos_arr, observed_map_size)):
            continue
        mask[aid] = 1.0
        labels[aid] = 1.0 if pos == true_target else 0.0
    return mask, labels


def _signal_target_decision_label_mask(
    env: SyncOrSinkEnv,
    obs: dict,
    cfg: RecurrentConfig,
    actions: dict[int, dict],
    *,
    target_scan_mask: np.ndarray | None = None,
    target_scan_kind_id: np.ndarray | None = None,
    target_opportunity_mask: np.ndarray | None = None,
    target_opportunity_kind_id: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((env.num_agents,), dtype=np.float32)
    labels = np.zeros((env.num_agents,), dtype=np.float32)
    if getattr(env.config, "scenario", None) != "signal_hunt" or env.scenario_state is None:
        return mask, labels
    target = env.scenario_state.data.get("target")
    if target is None:
        return mask, labels
    true_target = (int(target[0]), int(target[1]))
    positive_kinds = {
        _SIGNAL_TARGET_SCAN_KIND_FIRST,
        _SIGNAL_TARGET_SCAN_KIND_REFRESH,
        _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION,
    }
    if target_scan_mask is None or target_scan_kind_id is None:
        target_scan_mask, target_scan_kind_id = _signal_target_scan_label_mask(env, actions)
    if target_opportunity_mask is None or target_opportunity_kind_id is None:
        target_opportunity_mask = np.zeros((env.num_agents,), dtype=np.float32)
        target_opportunity_kind_id = np.full(
            (env.num_agents,),
            _SIGNAL_TARGET_SCAN_KIND_UNKNOWN,
            dtype=np.int64,
        )

    for aid in range(env.num_agents):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        if not _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_INTERACT):
            continue
        pos_arr = np.asarray(
            obs_agent.get("self_pos", env.agent_positions[aid]),
            dtype=np.int64,
        ).reshape(-1)
        if pos_arr.size < 2:
            continue
        pos = (int(pos_arr[0]), int(pos_arr[1]))
        if not _signal_center_visible_target_tile(obs_agent, cfg):
            continue

        scan_kind = int(target_scan_kind_id[int(aid)])
        opportunity_kind = int(target_opportunity_kind_id[int(aid)])
        positive = (
            (float(target_scan_mask[int(aid)]) > 0.0 and scan_kind in positive_kinds)
            or (
                float(target_opportunity_mask[int(aid)]) > 0.0
                and opportunity_kind in positive_kinds
            )
        )
        if positive:
            mask[int(aid)] = 1.0
            labels[int(aid)] = 1.0
            continue

        observed_map_size = _observed_map_size(obs_agent, cfg)
        redundant_kind = _signal_target_scan_kind(env, int(aid)) == _SIGNAL_TARGET_SCAN_KIND_REDUNDANT_ACTIVE
        rejected = _signal_center_rejected_target(obs_agent, observed_map_size)
        negative = pos != true_target or redundant_kind or (rejected and pos != true_target)
        if negative:
            mask[int(aid)] = 1.0
            labels[int(aid)] = 0.0
    return mask, labels


def _append_labeled_step(
    ep_data: dict,
    obs: dict,
    actions: dict[int, dict],
    env: SyncOrSinkEnv,
    cfg: RecurrentConfig,
    feedback: np.ndarray | None = None,
    step_weight: float | np.ndarray = 1.0,
) -> None:
    weights = np.asarray(step_weight, dtype=np.float32)
    if weights.ndim == 0:
        weights = np.full((env.num_agents,), float(weights.item()), dtype=np.float32)
    weights = weights.reshape(env.num_agents)
    weights = _apply_signal_target_interact_weight(weights, env=env, cfg=cfg, actions=actions)
    weights = _apply_signal_target_pursuit_weight(weights, env=env, cfg=cfg, obs=obs, actions=actions)
    weights = _apply_signal_sync_response_weight(
        weights,
        env=env,
        cfg=cfg,
        obs=obs,
        actions=actions,
        feedback=feedback,
    )
    rejected_target_mask = _clear_true_target_rejected_mask(
        env,
        _signal_rejected_target_mask(obs, int(env.map_size), env.num_agents),
    )
    bad_redundant_target_mask = _signal_bad_redundant_target_mask(env, obs)
    target_scan_mask, target_scan_kind_id = _signal_target_scan_label_mask(env, actions)
    target_opportunity_mask, target_opportunity_kind_id = _signal_target_scan_opportunity_label_mask(
        env,
        obs,
        cfg,
        feedback=feedback,
    )
    redundant_wait_action_mask, redundant_wait_action_id = _signal_redundant_target_wait_action_label_mask(
        env,
        obs,
        actions,
    )
    target_pursuit_action_mask, target_pursuit_action_id = _signal_target_pursuit_action_label_mask(
        env,
        obs,
        cfg,
    )
    sync_response_action_mask, sync_response_action_id = _signal_sync_response_action_label_mask(
        env,
        obs,
        actions,
        feedback,
        cfg=cfg,
    )
    target_match_action_mask, target_match_action_id = _signal_target_match_action_label_mask(
        env,
        obs,
        actions,
    )
    target_validity_mask, target_validity_label = _signal_target_validity_label(env, obs)
    target_decision_mask, target_decision_label = _signal_target_decision_label_mask(
        env,
        obs,
        cfg,
        actions,
        target_scan_mask=target_scan_mask,
        target_scan_kind_id=target_scan_kind_id,
        target_opportunity_mask=target_opportunity_mask,
        target_opportunity_kind_id=target_opportunity_kind_id,
    )
    target_aux_mask, target_aux_xy = _signal_target_aux_label(env)
    for aid in range(env.num_agents):
        fb = feedback[aid] if feedback is not None else None
        ep_data["obs"].append(_flatten_recurrent_obs(obs[aid], cfg, fb))
        ep_data["actions"].append(int(actions[aid]["action"]))
        raw_tokens = actions[aid].get("message_tokens", [])
        msg_tokens = np.zeros((cfg.comm_token_limit,), dtype=np.int64)
        msg_len = min(len(raw_tokens), cfg.comm_token_limit)
        if msg_len > 0:
            msg_tokens[:msg_len] = [int(t) for t in raw_tokens[:msg_len]]
        ep_data["msg_tokens"].append(msg_tokens)
        ep_data["msg_lens"].append(msg_len)
        ep_data["step_weights"].append(float(weights[aid]))
        ep_data["signal_rejected_target_mask"].append(float(rejected_target_mask[aid]))
        ep_data["signal_bad_redundant_target_mask"].append(float(bad_redundant_target_mask[aid]))
        ep_data["signal_target_scan_action_mask"].append(float(target_scan_mask[aid]))
        ep_data["signal_target_scan_kind_id"].append(int(target_scan_kind_id[aid]))
        ep_data["signal_target_opportunity_action_mask"].append(float(target_opportunity_mask[aid]))
        ep_data["signal_target_opportunity_kind_id"].append(int(target_opportunity_kind_id[aid]))
        ep_data["signal_redundant_target_wait_action_mask"].append(float(redundant_wait_action_mask[aid]))
        ep_data["signal_redundant_target_wait_action_id"].append(int(redundant_wait_action_id[aid]))
        ep_data["signal_target_pursuit_action_mask"].append(float(target_pursuit_action_mask[aid]))
        ep_data["signal_target_pursuit_action_id"].append(int(target_pursuit_action_id[aid]))
        ep_data["signal_sync_response_action_mask"].append(float(sync_response_action_mask[aid]))
        ep_data["signal_sync_response_action_id"].append(int(sync_response_action_id[aid]))
        ep_data["signal_target_match_action_mask"].append(float(target_match_action_mask[aid]))
        ep_data["signal_target_match_action_id"].append(int(target_match_action_id[aid]))
        ep_data["signal_target_validity_mask"].append(float(target_validity_mask[aid]))
        ep_data["signal_target_validity_label"].append(float(target_validity_label[aid]))
        ep_data["signal_target_decision_mask"].append(float(target_decision_mask[aid]))
        ep_data["signal_target_decision_label"].append(float(target_decision_label[aid]))
        ep_data["signal_target_aux_mask"].append(float(target_aux_mask[aid]))
        ep_data["signal_target_aux_xy"].append(target_aux_xy[aid].astype(np.float32, copy=True))
        ep_data["signal_decoy_drift_action_mask"].append(0.0)
        ep_data["signal_decoy_drift_action_id"].append(-1)
        ep_data["signal_decoy_scan_action_mask"].append(0.0)
        ep_data["signal_decoy_scan_action_id"].append(-1)
        ep_data["signal_rejected_target_drift_action_mask"].append(0.0)
        ep_data["signal_rejected_target_drift_action_id"].append(-1)


def _finalize_episode_sequence(
    ep_data: dict,
    env: SyncOrSinkEnv,
    cfg: RecurrentConfig,
    **metadata,
) -> dict:
    obs_dim = len(ep_data["obs"][0])
    episode = {
        "obs": np.stack(ep_data["obs"]).reshape(-1, env.num_agents, obs_dim),
        "actions": np.array(ep_data["actions"], dtype=np.int64).reshape(-1, env.num_agents),
        "msg_tokens": np.stack(ep_data["msg_tokens"]).reshape(
            -1, env.num_agents, cfg.comm_token_limit
        ),
        "msg_lens": np.array(ep_data["msg_lens"], dtype=np.int64).reshape(
            -1, env.num_agents
        ),
        "step_weights": np.array(ep_data["step_weights"], dtype=np.float32).reshape(
            -1, env.num_agents
        ),
        "signal_rejected_target_mask": np.array(
            ep_data.get("signal_rejected_target_mask", []),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_bad_redundant_target_mask": np.array(
            ep_data.get("signal_bad_redundant_target_mask", []),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_target_scan_action_mask": np.array(
            ep_data.get("signal_target_scan_action_mask", []),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_target_scan_kind_id": np.array(
            ep_data.get("signal_target_scan_kind_id", []),
            dtype=np.int64,
        ).reshape(-1, env.num_agents),
        "signal_target_opportunity_action_mask": np.array(
            ep_data.get("signal_target_opportunity_action_mask", [0.0 for _ in ep_data["actions"]]),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_target_opportunity_kind_id": np.array(
            ep_data.get(
                "signal_target_opportunity_kind_id",
                [_SIGNAL_TARGET_SCAN_KIND_UNKNOWN for _ in ep_data["actions"]],
            ),
            dtype=np.int64,
        ).reshape(-1, env.num_agents),
        "signal_redundant_target_wait_action_mask": np.array(
            ep_data.get("signal_redundant_target_wait_action_mask", [0.0 for _ in ep_data["actions"]]),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_redundant_target_wait_action_id": np.array(
            ep_data.get("signal_redundant_target_wait_action_id", [-1 for _ in ep_data["actions"]]),
            dtype=np.int64,
        ).reshape(-1, env.num_agents),
        "signal_target_pursuit_action_mask": np.array(
            ep_data.get("signal_target_pursuit_action_mask", [0.0 for _ in ep_data["actions"]]),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_target_pursuit_action_id": np.array(
            ep_data.get("signal_target_pursuit_action_id", [-1 for _ in ep_data["actions"]]),
            dtype=np.int64,
        ).reshape(-1, env.num_agents),
        "signal_sync_response_action_mask": np.array(
            ep_data.get("signal_sync_response_action_mask", [0.0 for _ in ep_data["actions"]]),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_sync_response_action_id": np.array(
            ep_data.get("signal_sync_response_action_id", [-1 for _ in ep_data["actions"]]),
            dtype=np.int64,
        ).reshape(-1, env.num_agents),
        "signal_target_match_action_mask": np.array(
            ep_data.get("signal_target_match_action_mask", []),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_target_match_action_id": np.array(
            ep_data.get("signal_target_match_action_id", []),
            dtype=np.int64,
        ).reshape(-1, env.num_agents),
        "signal_target_validity_mask": np.array(
            ep_data.get("signal_target_validity_mask", [0.0 for _ in ep_data["actions"]]),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_target_validity_label": np.array(
            ep_data.get("signal_target_validity_label", [0.0 for _ in ep_data["actions"]]),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_target_decision_mask": np.array(
            ep_data.get("signal_target_decision_mask", [0.0 for _ in ep_data["actions"]]),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_target_decision_label": np.array(
            ep_data.get("signal_target_decision_label", [0.0 for _ in ep_data["actions"]]),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_target_aux_mask": np.array(
            ep_data.get("signal_target_aux_mask", [0.0 for _ in ep_data["actions"]]),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_target_aux_xy": np.stack(
            ep_data.get(
                "signal_target_aux_xy",
                [np.zeros((2,), dtype=np.float32) for _ in ep_data["actions"]],
            )
        ).astype(np.float32).reshape(-1, env.num_agents, 2),
        "signal_decoy_drift_action_mask": np.array(
            ep_data.get("signal_decoy_drift_action_mask", []),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_decoy_drift_action_id": np.array(
            ep_data.get("signal_decoy_drift_action_id", []),
            dtype=np.int64,
        ).reshape(-1, env.num_agents),
        "signal_decoy_scan_action_mask": np.array(
            ep_data.get("signal_decoy_scan_action_mask", []),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_decoy_scan_action_id": np.array(
            ep_data.get("signal_decoy_scan_action_id", []),
            dtype=np.int64,
        ).reshape(-1, env.num_agents),
        "signal_rejected_target_drift_action_mask": np.array(
            ep_data.get("signal_rejected_target_drift_action_mask", []),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_rejected_target_drift_action_id": np.array(
            ep_data.get("signal_rejected_target_drift_action_id", []),
            dtype=np.int64,
        ).reshape(-1, env.num_agents),
    }
    episode.update(metadata)
    return episode


def _episode_count_transitions(episodes) -> int:
    return int(sum(ep["obs"].shape[0] * ep["obs"].shape[1] for ep in episodes))


def _episode_count_label_mask(episodes, key: str) -> int:
    total = 0
    for ep in episodes:
        if key not in ep:
            continue
        total += int(np.asarray(ep[key], dtype=np.float32).sum())
    return total


def _episode_training_weight(ep_data: dict) -> float:
    return max(0.0, float(ep_data.get("weight", 1.0)))


def _episode_count_effective_transitions(episodes) -> float:
    total = 0.0
    for ep in episodes:
        episode_weight = _episode_training_weight(ep)
        if "step_weights" in ep:
            total += float(np.asarray(ep["step_weights"], dtype=np.float32).sum()) * episode_weight
        else:
            total += ep["obs"].shape[0] * ep["obs"].shape[1] * episode_weight
    return float(total)


def _episode_source_counts(episodes) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ep in episodes:
        source = str(ep.get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def _episode_map_size_counts(episodes) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ep in episodes:
        if "map_size" not in ep:
            continue
        size = str(int(ep["map_size"]))
        counts[size] = counts.get(size, 0) + 1
    return counts


def _episode_map_size_diagnostics(episodes) -> dict[str, dict]:
    diagnostics: dict[str, dict] = {}
    for ep in episodes:
        if "map_size" not in ep:
            continue
        map_size = str(int(ep["map_size"]))
        row = diagnostics.setdefault(
            map_size,
            {
                "episodes": 0,
                "transitions": 0,
                "effective_transitions": 0.0,
                "success_episodes": 0,
                "failed_episodes": 0,
                "capped_episodes": 0,
                "replay_episodes": 0,
                "sources": {},
                "replay_trigger_events": {},
            },
        )
        source = str(ep.get("source", "unknown"))
        row["episodes"] += 1
        row["transitions"] += int(ep["obs"].shape[0] * ep["obs"].shape[1])
        row["effective_transitions"] += float(_episode_count_effective_transitions([ep]))
        row["sources"][source] = int(row["sources"].get(source, 0)) + 1
        if bool(ep.get("success", False)):
            row["success_episodes"] += 1
        else:
            row["failed_episodes"] += 1
        if bool(ep.get("capped", False)):
            row["capped_episodes"] += 1
        if source.endswith("_replay"):
            row["replay_episodes"] += 1
            trigger = str(ep.get("trigger_event", "unknown"))
            row["replay_trigger_events"][trigger] = int(row["replay_trigger_events"].get(trigger, 0)) + 1
    for row in diagnostics.values():
        row["success_rate"] = row["success_episodes"] / row["episodes"] if row["episodes"] else 0.0
    return diagnostics


def _map_diagnostics_wandb_payload(prefix: str, diagnostics: dict[str, dict]) -> dict:
    payload: dict[str, int | float] = {}
    for map_size, row in sorted(diagnostics.items(), key=lambda item: int(item[0])):
        map_prefix = f"{prefix}/map_{map_size}"
        for key in (
            "episodes",
            "transitions",
            "effective_transitions",
            "success_episodes",
            "failed_episodes",
            "capped_episodes",
            "replay_episodes",
            "success_rate",
        ):
            if key in row:
                payload[f"{map_prefix}/{key}"] = row[key]
        for source, count in sorted((row.get("sources") or {}).items()):
            payload[f"{map_prefix}/source_{source}_episodes"] = int(count)
        for event, count in sorted((row.get("replay_trigger_events") or {}).items()):
            payload[f"{map_prefix}/replay_trigger_{event}"] = int(count)
    return payload


def _slice_recurrent_episode(episode: dict, start: int, end: int, **metadata) -> dict:
    sliced = {
        "obs": episode["obs"][start:end].copy(),
        "actions": episode["actions"][start:end].copy(),
        "msg_tokens": episode["msg_tokens"][start:end].copy(),
        "msg_lens": episode["msg_lens"][start:end].copy(),
    }
    if "step_weights" in episode:
        sliced["step_weights"] = episode["step_weights"][start:end].copy()
    else:
        sliced["step_weights"] = np.ones_like(episode["actions"][start:end], dtype=np.float32)
    if "signal_rejected_target_mask" in episode:
        sliced["signal_rejected_target_mask"] = episode["signal_rejected_target_mask"][start:end].copy()
    else:
        sliced["signal_rejected_target_mask"] = np.zeros_like(episode["actions"][start:end], dtype=np.float32)
    if "signal_bad_redundant_target_mask" in episode:
        sliced["signal_bad_redundant_target_mask"] = episode["signal_bad_redundant_target_mask"][start:end].copy()
    else:
        sliced["signal_bad_redundant_target_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_target_scan_action_mask" in episode:
        sliced["signal_target_scan_action_mask"] = episode["signal_target_scan_action_mask"][start:end].copy()
    else:
        sliced["signal_target_scan_action_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_target_scan_kind_id" in episode:
        sliced["signal_target_scan_kind_id"] = episode["signal_target_scan_kind_id"][start:end].copy()
    else:
        sliced["signal_target_scan_kind_id"] = np.full_like(
            episode["actions"][start:end],
            _SIGNAL_TARGET_SCAN_KIND_UNKNOWN,
            dtype=np.int64,
        )
    if "signal_target_opportunity_action_mask" in episode:
        sliced["signal_target_opportunity_action_mask"] = episode[
            "signal_target_opportunity_action_mask"
        ][start:end].copy()
    else:
        sliced["signal_target_opportunity_action_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_target_opportunity_kind_id" in episode:
        sliced["signal_target_opportunity_kind_id"] = episode[
            "signal_target_opportunity_kind_id"
        ][start:end].copy()
    else:
        sliced["signal_target_opportunity_kind_id"] = np.full_like(
            episode["actions"][start:end],
            _SIGNAL_TARGET_SCAN_KIND_UNKNOWN,
            dtype=np.int64,
        )
    if "signal_redundant_target_wait_action_mask" in episode:
        sliced["signal_redundant_target_wait_action_mask"] = episode[
            "signal_redundant_target_wait_action_mask"
        ][start:end].copy()
    else:
        sliced["signal_redundant_target_wait_action_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_redundant_target_wait_action_id" in episode:
        sliced["signal_redundant_target_wait_action_id"] = episode[
            "signal_redundant_target_wait_action_id"
        ][start:end].copy()
    else:
        sliced["signal_redundant_target_wait_action_id"] = np.full_like(
            episode["actions"][start:end],
            -1,
            dtype=np.int64,
        )
    if "signal_target_pursuit_action_mask" in episode:
        sliced["signal_target_pursuit_action_mask"] = episode[
            "signal_target_pursuit_action_mask"
        ][start:end].copy()
    else:
        sliced["signal_target_pursuit_action_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_target_pursuit_action_id" in episode:
        sliced["signal_target_pursuit_action_id"] = episode[
            "signal_target_pursuit_action_id"
        ][start:end].copy()
    else:
        sliced["signal_target_pursuit_action_id"] = np.full_like(
            episode["actions"][start:end],
            -1,
            dtype=np.int64,
        )
    if "signal_sync_response_action_mask" in episode:
        sliced["signal_sync_response_action_mask"] = episode["signal_sync_response_action_mask"][
            start:end
        ].copy()
    else:
        sliced["signal_sync_response_action_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_sync_response_action_id" in episode:
        sliced["signal_sync_response_action_id"] = episode["signal_sync_response_action_id"][start:end].copy()
    else:
        sliced["signal_sync_response_action_id"] = np.full_like(
            episode["actions"][start:end],
            -1,
            dtype=np.int64,
        )
    if "signal_target_match_action_mask" in episode:
        sliced["signal_target_match_action_mask"] = episode["signal_target_match_action_mask"][start:end].copy()
    else:
        sliced["signal_target_match_action_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_target_match_action_id" in episode:
        sliced["signal_target_match_action_id"] = episode["signal_target_match_action_id"][start:end].copy()
    else:
        sliced["signal_target_match_action_id"] = np.full_like(
            episode["actions"][start:end],
            -1,
            dtype=np.int64,
        )
    if "signal_target_validity_mask" in episode:
        sliced["signal_target_validity_mask"] = episode["signal_target_validity_mask"][start:end].copy()
    else:
        sliced["signal_target_validity_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_target_validity_label" in episode:
        sliced["signal_target_validity_label"] = episode["signal_target_validity_label"][start:end].copy()
    else:
        sliced["signal_target_validity_label"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_target_decision_mask" in episode:
        sliced["signal_target_decision_mask"] = episode["signal_target_decision_mask"][start:end].copy()
    else:
        sliced["signal_target_decision_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_target_decision_label" in episode:
        sliced["signal_target_decision_label"] = episode["signal_target_decision_label"][start:end].copy()
    else:
        sliced["signal_target_decision_label"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_target_aux_mask" in episode:
        sliced["signal_target_aux_mask"] = episode["signal_target_aux_mask"][start:end].copy()
    else:
        sliced["signal_target_aux_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_target_aux_xy" in episode:
        sliced["signal_target_aux_xy"] = episode["signal_target_aux_xy"][start:end].copy()
    else:
        base = episode["actions"][start:end]
        sliced["signal_target_aux_xy"] = np.zeros((*base.shape, 2), dtype=np.float32)
    if "signal_decoy_drift_action_mask" in episode:
        sliced["signal_decoy_drift_action_mask"] = episode["signal_decoy_drift_action_mask"][start:end].copy()
    else:
        sliced["signal_decoy_drift_action_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_decoy_drift_action_id" in episode:
        sliced["signal_decoy_drift_action_id"] = episode["signal_decoy_drift_action_id"][start:end].copy()
    else:
        sliced["signal_decoy_drift_action_id"] = np.full_like(
            episode["actions"][start:end],
            -1,
            dtype=np.int64,
        )
    if "signal_decoy_scan_action_mask" in episode:
        sliced["signal_decoy_scan_action_mask"] = episode["signal_decoy_scan_action_mask"][start:end].copy()
    else:
        sliced["signal_decoy_scan_action_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_decoy_scan_action_id" in episode:
        sliced["signal_decoy_scan_action_id"] = episode["signal_decoy_scan_action_id"][start:end].copy()
    else:
        sliced["signal_decoy_scan_action_id"] = np.full_like(
            episode["actions"][start:end],
            -1,
            dtype=np.int64,
        )
    if "signal_rejected_target_drift_action_mask" in episode:
        sliced["signal_rejected_target_drift_action_mask"] = episode[
            "signal_rejected_target_drift_action_mask"
        ][start:end].copy()
    else:
        sliced["signal_rejected_target_drift_action_mask"] = np.zeros_like(
            episode["actions"][start:end],
            dtype=np.float32,
        )
    if "signal_rejected_target_drift_action_id" in episode:
        sliced["signal_rejected_target_drift_action_id"] = episode[
            "signal_rejected_target_drift_action_id"
        ][start:end].copy()
    else:
        sliced["signal_rejected_target_drift_action_id"] = np.full_like(
            episode["actions"][start:end],
            -1,
            dtype=np.int64,
        )
    sliced.update(metadata)
    return sliced


def _focus_replay_episodes(
    episode: dict,
    focus_records: list[dict],
    cfg: RecurrentConfig,
    *,
    source: str = "dagger_focus_replay",
) -> list[dict]:
    if not cfg.dagger_focus_replay or not focus_records:
        return []
    max_snippets = int(cfg.dagger_max_replay_snippets_per_episode)
    if source == "expert_positive_replay":
        expert_max = int(getattr(cfg, "dagger_expert_max_replay_snippets_per_episode", -1))
        if expert_max >= 0:
            max_snippets = expert_max
    max_snippets = max(0, max_snippets)
    if max_snippets <= 0:
        return []

    total_steps = int(episode["obs"].shape[0])
    pre_steps = max(0, int(cfg.dagger_replay_pre_steps))
    post_steps = max(0, int(cfg.dagger_replay_post_steps))
    replay_weight = max(0.0, float(cfg.dagger_replay_weight))
    event_weights = _parse_event_float_overrides(
        cfg.dagger_replay_event_weights,
        field_name="dagger_replay_event_weights",
    )
    if replay_weight <= 0.0 and not any(weight > 0.0 for weight in event_weights.values()):
        return []
    event_caps = _parse_event_int_overrides(
        cfg.dagger_replay_event_caps,
        field_name="dagger_replay_event_caps",
    )
    success_only_events = _dagger_replay_success_only_events(cfg)
    priority_events = _dagger_replay_priority_events(cfg)
    parent_success = bool(episode.get("success", False))
    balance_positive_events = _dagger_replay_balance_positive_events(cfg)
    balance_negative_events = _dagger_replay_balance_negative_events(cfg)
    balance_ratio = float(getattr(cfg, "dagger_replay_max_negative_per_positive", -1.0))
    balance_enabled = (
        source == "dagger_focus_replay"
        and balance_ratio >= 0.0
        and bool(balance_positive_events)
        and bool(balance_negative_events)
    )

    snippets = []
    candidates = []
    seen_windows = set()
    event_counts: dict[str, int] = {}
    sorted_records = sorted(
        focus_records,
        key=lambda item: (
            0 if str(item["event"]) in priority_events else 1,
            int(item["step"]),
            str(item["event"]),
        ),
    )
    for record_index, record in enumerate(sorted_records):
        event = str(record["event"])
        if event in success_only_events and not parent_success:
            continue
        event_cap = event_caps.get(event)
        if event_cap is not None and event_counts.get(event, 0) >= event_cap:
            continue
        record_weight = float(event_weights.get(event, replay_weight))
        if record_weight <= 0.0:
            continue
        step = int(record["step"])
        start = max(0, step - pre_steps)
        end = min(total_steps, step + post_steps + 1)
        if end <= start:
            continue
        key = (start, end, event)
        if key in seen_windows:
            continue
        seen_windows.add(key)
        metadata = dict(
            source=source,
            round=episode.get("round"),
            seed=episode.get("seed"),
            parent_success=episode.get("success"),
            parent_capped=episode.get("capped"),
            success=episode.get("success", False),
            capped=episode.get("capped", False),
            weight=record_weight,
            trigger_event=event,
            trigger_kind=str(record.get("kind", "focus")),
            trigger_step=step,
            trigger_agents=[int(aid) for aid in record.get("agents", [])],
            replay_start_step=start,
            replay_end_step=end,
            steps=end - start,
        )
        if "map_size" in episode:
            metadata["map_size"] = episode["map_size"]
        snippet = _slice_recurrent_episode(episode, start, end, **metadata)
        event_counts[event] = event_counts.get(event, 0) + 1
        if balance_enabled:
            candidates.append((record_index, event, snippet))
            continue
        snippets.append(snippet)
        if len(snippets) >= max_snippets:
            break
    if balance_enabled:
        snippets = _balanced_focus_replay_candidates(
            candidates,
            positive_events=balance_positive_events,
            negative_events=balance_negative_events,
            priority_events=priority_events,
            max_negative_per_positive=balance_ratio,
            max_snippets=max_snippets,
        )
    return snippets


def _balanced_focus_replay_candidates(
    candidates: list[tuple[int, str, dict]],
    *,
    positive_events: set[str],
    negative_events: set[str],
    priority_events: set[str],
    max_negative_per_positive: float,
    max_snippets: int,
) -> list[dict]:
    if not candidates or max_snippets <= 0:
        return []
    max_negative_per_positive = max(0.0, float(max_negative_per_positive))
    positive_count = sum(1 for _idx, event, _snippet in candidates if event in positive_events)
    negative_limit = int(math.ceil(float(positive_count) * max_negative_per_positive))
    negative_kept = 0
    filtered: list[tuple[int, int, dict]] = []
    for record_index, event, snippet in candidates:
        if event in negative_events:
            if negative_kept >= negative_limit:
                continue
            negative_kept += 1
        if event in priority_events:
            priority = 0
        elif event in positive_events:
            priority = 1
        elif event in negative_events:
            priority = 3
        else:
            priority = 2
        filtered.append((priority, record_index, snippet))
    filtered.sort(key=lambda item: (item[0], item[1]))
    return [snippet for _priority, _record_index, snippet in filtered[:max_snippets]]


def _dagger_episode_weight(cfg: RecurrentConfig, success: bool) -> float:
    if success:
        return max(0.0, float(cfg.dagger_success_episode_weight))
    return max(0.0, float(cfg.dagger_failed_episode_weight))


def _dagger_focus_events(cfg: RecurrentConfig) -> set[str]:
    return {
        item.strip()
        for item in str(cfg.dagger_focus_events or "").split(",")
        if item.strip()
    }


def _dagger_positive_replay_events(cfg: RecurrentConfig) -> set[str]:
    return {
        item.strip()
        for item in str(cfg.dagger_positive_replay_events or "").split(",")
        if item.strip()
    }


def _dagger_collection_seed(cfg: RecurrentConfig, round_idx: int, episode_idx: int) -> int:
    ordinal = max(0, int(round_idx)) * max(0, int(cfg.dagger_episodes)) + max(0, int(episode_idx))
    explicit_seeds = _parse_seed_list(cfg.dagger_seed_list, field_name="dagger_seed_list")
    if explicit_seeds:
        return int(explicit_seeds[ordinal % len(explicit_seeds)])
    return int(cfg.dagger_seed_base) + max(0, int(round_idx)) * int(cfg.dagger_seed_stride) + int(episode_idx)


def _dagger_replay_success_only_events(cfg: RecurrentConfig) -> set[str]:
    return {
        item.strip()
        for item in str(cfg.dagger_replay_success_only_events or "").split(",")
        if item.strip()
    }


def _dagger_replay_priority_events(cfg: RecurrentConfig) -> set[str]:
    return {
        item.strip()
        for item in str(cfg.dagger_replay_priority_events or "").split(",")
        if item.strip()
    }


def _dagger_replay_balance_positive_events(cfg: RecurrentConfig) -> set[str]:
    return {
        item.strip()
        for item in str(cfg.dagger_replay_balance_positive_events or "").split(",")
        if item.strip()
    }


def _dagger_replay_balance_negative_events(cfg: RecurrentConfig) -> set[str]:
    return {
        item.strip()
        for item in str(cfg.dagger_replay_balance_negative_events or "").split(",")
        if item.strip()
    }


def _parse_event_float_overrides(raw_value: str, *, field_name: str) -> dict[str, float]:
    raw = str(raw_value or "").strip()
    if not raw:
        return {}
    overrides: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{field_name} entries must be event:value pairs, got {item!r}")
        raw_event, raw_weight = item.split(":", 1)
        event = raw_event.strip()
        weight = float(raw_weight.strip())
        if not event or weight < 0.0:
            raise ValueError(f"{field_name} entries must contain an event and nonnegative value, got {item!r}")
        overrides[event] = weight
    return overrides


def _parse_event_int_overrides(raw_value: str, *, field_name: str) -> dict[str, int]:
    raw = str(raw_value or "").strip()
    if not raw:
        return {}
    overrides: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{field_name} entries must be event:value pairs, got {item!r}")
        raw_event, raw_value_part = item.split(":", 1)
        event = raw_event.strip()
        value = int(raw_value_part.strip())
        if not event or value < 0:
            raise ValueError(f"{field_name} entries must contain an event and nonnegative integer, got {item!r}")
        overrides[event] = value
    return overrides


def _scale_latest_agent_weights(
    ep_data: dict,
    *,
    num_agents: int,
    agent_ids,
    weight: float,
) -> int:
    if not ep_data.get("step_weights"):
        return 0
    scaled = 0
    start = len(ep_data["step_weights"]) - num_agents
    for aid in agent_ids:
        idx = start + int(aid)
        if start <= idx < len(ep_data["step_weights"]):
            old_weight = float(ep_data["step_weights"][idx])
            ep_data["step_weights"][idx] = max(old_weight, float(weight))
            scaled += int(float(ep_data["step_weights"][idx]) > old_weight)
    return scaled


def _label_latest_signal_decoy_drift_actions(
    ep_data: dict,
    *,
    num_agents: int,
    agent_ids,
    model_actions: dict[int, dict],
) -> int:
    action_masks = ep_data.get("signal_decoy_drift_action_mask")
    action_ids = ep_data.get("signal_decoy_drift_action_id")
    if not action_masks or action_ids is None:
        return 0
    labeled = 0
    start = len(action_masks) - num_agents
    for aid in agent_ids:
        idx = start + int(aid)
        if start <= idx < len(action_masks):
            action = int(model_actions[int(aid)].get("action", SyncOrSinkEnv.ACTION_STAY))
            action_masks[idx] = 1.0
            action_ids[idx] = action
            labeled += 1
    return labeled


def _label_latest_signal_decoy_scan_actions(
    ep_data: dict,
    *,
    num_agents: int,
    agent_ids,
    model_actions: dict[int, dict],
) -> int:
    action_masks = ep_data.get("signal_decoy_scan_action_mask")
    action_ids = ep_data.get("signal_decoy_scan_action_id")
    if not action_masks or action_ids is None:
        return 0
    labeled = 0
    start = len(action_masks) - num_agents
    for aid in agent_ids:
        idx = start + int(aid)
        if start <= idx < len(action_masks):
            action = int(model_actions[int(aid)].get("action", SyncOrSinkEnv.ACTION_STAY))
            action_masks[idx] = 1.0
            action_ids[idx] = action
            labeled += 1
    return labeled


def _label_latest_signal_rejected_target_drift_actions(
    ep_data: dict,
    *,
    num_agents: int,
    agent_ids,
    model_actions: dict[int, dict],
) -> int:
    action_masks = ep_data.get("signal_rejected_target_drift_action_mask")
    action_ids = ep_data.get("signal_rejected_target_drift_action_id")
    if not action_masks or action_ids is None:
        return 0
    labeled = 0
    start = len(action_masks) - num_agents
    for aid in agent_ids:
        idx = start + int(aid)
        if start <= idx < len(action_masks):
            action = int(model_actions[int(aid)].get("action", SyncOrSinkEnv.ACTION_STAY))
            action_masks[idx] = 1.0
            action_ids[idx] = action
            labeled += 1
    return labeled


def _scale_agent_weights_at_step(
    ep_data: dict,
    *,
    num_agents: int,
    step: int,
    agent_ids,
    weight: float,
) -> int:
    if not ep_data.get("step_weights"):
        return 0
    scaled = 0
    start = int(step) * int(num_agents)
    for aid in agent_ids:
        idx = start + int(aid)
        if 0 <= idx < len(ep_data["step_weights"]):
            old_weight = float(ep_data["step_weights"][idx])
            ep_data["step_weights"][idx] = max(old_weight, float(weight))
            scaled += int(float(ep_data["step_weights"][idx]) > old_weight)
    return scaled


def _solo_target_teammate_agents(num_agents: int, solo_target_agents) -> list[int]:
    solo_set = {int(aid) for aid in solo_target_agents}
    if not solo_set:
        return []
    return [aid for aid in range(int(num_agents)) if aid not in solo_set]


def _scale_solo_target_team_weights(
    ep_data: dict,
    *,
    num_agents: int,
    solo_target_agents,
    weight: float,
) -> tuple[int, list[int]]:
    teammate_agents = _solo_target_teammate_agents(num_agents, solo_target_agents)
    if float(weight) <= 1.0 or not teammate_agents:
        return 0, teammate_agents
    return (
        _scale_latest_agent_weights(
            ep_data,
            num_agents=num_agents,
            agent_ids=teammate_agents,
            weight=float(weight),
        ),
        teammate_agents,
    )


def _apply_deferred_solo_target_team_weights(
    ep_data: dict,
    records: list[dict],
    *,
    num_agents: int,
    focus_window: int,
) -> int:
    updates = 0
    max_offset = max(0, int(focus_window))
    for record in records:
        step = int(record.get("step", 0))
        weight = float(record.get("weight", 1.0))
        agent_ids = [int(aid) for aid in record.get("agents", [])]
        for offset in range(max_offset + 1):
            updates += _scale_agent_weights_at_step(
                ep_data,
                num_agents=num_agents,
                step=step + offset,
                agent_ids=agent_ids,
                weight=weight,
            )
    return updates


def _solo_target_interactors(env: SyncOrSinkEnv, actions: dict[int, dict]) -> list[int]:
    interactors = _signal_target_interact_agents(env, actions)
    return interactors if len(interactors) == 1 else []


def _teammate_can_join_target_scan(
    env: SyncOrSinkEnv,
    *,
    scanner_id: int,
    obs: dict,
    actions: dict[int, dict],
) -> bool:
    target = env.scenario_state.data.get("target")
    if target is None:
        return False
    target = tuple(target)
    scan_log = env.scenario_state.data.get("scan_log") or {}
    scan_window = int(env.scenario_state.data.get("scan_window", getattr(env.config, "scan_window", 3)))
    next_step = int(env.steps) + 1
    target_pursuers = set(_signal_target_pursuit_agents(env, obs, actions))
    for aid in range(env.num_agents):
        if int(aid) == int(scanner_id):
            continue
        last_scan = scan_log.get(int(aid), scan_log.get(str(int(aid))))
        if last_scan is not None and next_step - int(last_scan) <= scan_window:
            return True
        if int(aid) in target_pursuers:
            return True
        pos = env.agent_positions[int(aid)]
        dist = abs(int(target[0]) - int(pos[0])) + abs(int(target[1]) - int(pos[1]))
        if dist + 1 <= scan_window:
            return True
    return False


def _split_solo_target_scan_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    actions: dict[int, dict],
) -> tuple[list[int], list[int]]:
    valid_hold_agents: list[int] = []
    bad_loop_agents: list[int] = []
    redundant_agents = set(_redundant_target_scan_agents(env, actions))
    for aid in _solo_target_interactors(env, actions):
        aid = int(aid)
        if aid not in redundant_agents or _teammate_can_join_target_scan(
            env,
            scanner_id=aid,
            obs=obs,
            actions=actions,
        ):
            valid_hold_agents.append(aid)
        else:
            bad_loop_agents.append(aid)
    return valid_hold_agents, bad_loop_agents


def collect_episode_demos(cfg: RecurrentConfig):
    """Collect oracle demonstrations as full episodes (not shuffled transitions)."""
    set_global_seeds(cfg.seed)
    _warn_if_signal_hint_comm_channel_is_too_small(cfg)

    episodes = []
    successful_demos = 0
    replay_demos = 0
    positive_replay_events = _dagger_positive_replay_events(cfg)
    for ep in range(cfg.demo_episodes):
        env, episode_cfg = _build_training_env(cfg, ep)
        oracle_fn = _make_oracle_policy(env, episode_cfg)
        obs, info = env.reset(seed=ep)
        ep_data = _new_episode_sequence()
        done, truncated = False, False
        step = 0
        prev_actions: dict[int, int] = {}
        prev_msg_lens: dict[int, int] = {}
        prev_info: dict = {}
        positive_records: list[dict] = []
        while not (done or truncated):
            feedback = _feedback_matrix(
                episode_cfg,
                env.num_agents,
                prev_actions=prev_actions,
                prev_msg_lens=prev_msg_lens,
                info=prev_info,
                env=env,
                obs=obs,
            )
            actions = oracle_fn(obs, info, {"step": step})
            actions, _broadcast_label_agents = _apply_signal_target_scan_broadcast_overrides(
                episode_cfg,
                env,
                actions,
                feedback,
                info=prev_info,
            )
            actions, handoff_label_agents = _apply_signal_target_handoff_overrides(
                episode_cfg,
                env,
                actions,
                feedback,
                obs=obs,
            )
            if episode_cfg.dagger_redundant_target_wait_labels:
                actions, _redundant_wait_label_agents = _apply_signal_redundant_target_wait_overrides(
                    env,
                    actions,
                )
            _append_labeled_step(ep_data, obs, actions, env, episode_cfg, feedback=feedback)
            if "target_handoff" in positive_replay_events and handoff_label_agents:
                for aid in handoff_label_agents:
                    positive_records.append({
                        "event": "target_handoff",
                        "step": step,
                        "agents": [aid],
                        "kind": "positive",
                    })
            if "target_pursuit" in positive_replay_events:
                target_pursuit_agents = _signal_positive_target_pursuit_agents(
                    env,
                    obs,
                    actions,
                    min_map_size=episode_cfg.dagger_positive_target_pursuit_min_map_size,
                )
                for aid in target_pursuit_agents:
                    positive_records.append({
                        "event": "target_pursuit",
                        "step": step,
                        "agents": [aid],
                        "kind": "positive",
                    })
            obs, rewards, done, truncated, info = env.step(actions)
            event_names = _event_names_by_agent(info or {}, env.num_agents)
            for aid, names in event_names.items():
                for name in sorted(names & positive_replay_events):
                    positive_records.append({
                        "event": name,
                        "step": step,
                        "agents": [aid],
                        "kind": "positive",
                    })
            prev_actions = {aid: int(action["action"]) for aid, action in actions.items()}
            prev_msg_lens = _message_lengths(actions)
            prev_info = info or {}
            step += 1

        success = episode_success(cfg.scenario, done, info)
        if success:
            successful_demos += 1
            base_episode = _finalize_episode_sequence(
                ep_data,
                env,
                episode_cfg,
                source="expert",
                seed=ep,
                map_size=episode_cfg.map_size,
                success=True,
                capped=False,
                weight=1.0,
                steps=step,
            )
            episodes.append(base_episode)
            replay_snippets = _focus_replay_episodes(
                base_episode,
                positive_records,
                episode_cfg,
                source="expert_positive_replay",
            )
            episodes.extend(replay_snippets)
            replay_demos += len(replay_snippets)

        if (ep + 1) % 50 == 0:
            print(
                f"  collected {ep + 1}/{cfg.demo_episodes}, "
                f"{successful_demos} successful demos, {replay_demos} replay snippets"
            )

    print(
        f"Collected {successful_demos} successful demos, "
        f"{replay_demos} replay snippets, {len(episodes)} training episodes"
    )
    return episodes


def _recurrent_comm_send_pos_weight(episodes, cfg: RecurrentConfig, device):
    if cfg.bc_comm_send_pos_weight > 0:
        return torch.tensor(float(cfg.bc_comm_send_pos_weight), dtype=torch.float32, device=device)
    if cfg.bc_comm_send_pos_weight < 0:
        msg_lens = np.concatenate([ep["msg_lens"].reshape(-1) for ep in episodes])
        positives = int((msg_lens > 0).sum())
        negatives = int((msg_lens <= 0).sum())
        if positives > 0 and negatives > 0:
            return torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    return None


def _weighted_mean(loss: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return loss.mean()
    weights = weights.to(loss.device, dtype=loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-8)


def _categorical_reference_kl(reference_logits: torch.Tensor, current_logits: torch.Tensor) -> torch.Tensor:
    reference_logprobs = torch.log_softmax(reference_logits, dim=-1)
    current_logprobs = torch.log_softmax(current_logits, dim=-1)
    return (reference_logprobs.exp() * (reference_logprobs - current_logprobs)).sum(dim=-1)


def _bernoulli_reference_kl(reference_logits: torch.Tensor, current_logits: torch.Tensor) -> torch.Tensor:
    reference_prob = torch.sigmoid(reference_logits)
    return (
        reference_prob * (nn.functional.logsigmoid(reference_logits) - nn.functional.logsigmoid(current_logits))
        + (1.0 - reference_prob)
        * (nn.functional.logsigmoid(-reference_logits) - nn.functional.logsigmoid(-current_logits))
    )


def _recurrent_comm_reference_kl(
    send_logits: torch.Tensor,
    token_logits: torch.Tensor,
    len_logits: torch.Tensor,
    reference_send_logits: torch.Tensor,
    reference_token_logits: torch.Tensor,
    reference_len_logits: torch.Tensor,
) -> torch.Tensor:
    send_kl = _bernoulli_reference_kl(
        reference_send_logits.squeeze(-1),
        send_logits.squeeze(-1),
    ).mean()
    token_kl = _categorical_reference_kl(reference_token_logits, token_logits).mean()
    len_kl = _categorical_reference_kl(reference_len_logits, len_logits).mean()
    return send_kl + token_kl + len_kl


def _recurrent_comm_loss_components(
    send_logits,
    token_logits,
    len_logits,
    msg_tokens,
    msg_lens,
    send_pos_weight=None,
    sample_weight=None,
    send_loss_weight: float = 1.0,
    length_loss_weight: float = 1.0,
    token_loss_weight: float = 1.0,
    send_rate_penalty_weight: float = 0.0,
    send_rate_target: float = -1.0,
):
    send_target = (msg_lens > 0).float()
    send_loss = nn.functional.binary_cross_entropy_with_logits(
        send_logits.squeeze(-1),
        send_target,
        pos_weight=send_pos_weight,
        reduction="none",
    )
    send_loss = _weighted_mean(send_loss, sample_weight)
    len_positive = (msg_lens > 0).float()
    if len_positive.sum() > 0:
        len_loss_vec = nn.functional.cross_entropy(len_logits, msg_lens, reduction="none")
        len_weights = len_positive if sample_weight is None else sample_weight * len_positive
        len_loss = _weighted_mean(len_loss_vec, len_weights)
    else:
        len_loss = torch.tensor(0.0, device=msg_lens.device)
    token_limit = token_logits.shape[1]
    t_mask = (torch.arange(token_limit, device=msg_lens.device)[None, :] < msg_lens[:, None]).float()
    if t_mask.sum() > 0:
        tok_loss_matrix = nn.functional.cross_entropy(
            token_logits.reshape(-1, token_logits.shape[-1]),
            msg_tokens.reshape(-1),
            reduction="none",
        ).reshape(token_logits.shape[0], token_limit)
        if sample_weight is not None:
            token_den = (t_mask * sample_weight[:, None]).sum().clamp_min(1e-8)
            tok_loss = (tok_loss_matrix * t_mask * sample_weight[:, None]).sum() / token_den
        else:
            tok_loss = (tok_loss_matrix * t_mask).sum() / t_mask.sum()
    else:
        tok_loss = torch.tensor(0.0, device=msg_lens.device)
    send_rate_loss = torch.tensor(0.0, device=msg_lens.device)
    send_rate_penalty_weight = max(0.0, float(send_rate_penalty_weight))
    if send_rate_penalty_weight > 0.0:
        send_prob = torch.sigmoid(send_logits.squeeze(-1))
        pred_rate = _weighted_mean(send_prob, sample_weight)
        if float(send_rate_target) >= 0.0:
            target_rate = torch.tensor(
                min(1.0, max(0.0, float(send_rate_target))),
                dtype=send_prob.dtype,
                device=send_prob.device,
            )
        else:
            target_rate = _weighted_mean(send_target, sample_weight)
        send_rate_loss = (pred_rate - target_rate.detach()).pow(2)
    weighted_send = max(0.0, float(send_loss_weight)) * send_loss
    weighted_length = max(0.0, float(length_loss_weight)) * len_loss
    weighted_token = max(0.0, float(token_loss_weight)) * tok_loss
    weighted_send_rate = send_rate_penalty_weight * send_rate_loss
    return {
        "total": weighted_send + weighted_length + weighted_token + weighted_send_rate,
        "send": send_loss,
        "length": len_loss,
        "token": tok_loss,
        "send_rate": send_rate_loss,
    }


def _recurrent_comm_loss(
    send_logits,
    token_logits,
    len_logits,
    msg_tokens,
    msg_lens,
    send_pos_weight=None,
    sample_weight=None,
    send_loss_weight: float = 1.0,
    length_loss_weight: float = 1.0,
    token_loss_weight: float = 1.0,
    send_rate_penalty_weight: float = 0.0,
    send_rate_target: float = -1.0,
):
    return _recurrent_comm_loss_components(
        send_logits,
        token_logits,
        len_logits,
        msg_tokens,
        msg_lens,
        send_pos_weight=send_pos_weight,
        sample_weight=sample_weight,
        send_loss_weight=send_loss_weight,
        length_loss_weight=length_loss_weight,
        token_loss_weight=token_loss_weight,
        send_rate_penalty_weight=send_rate_penalty_weight,
        send_rate_target=send_rate_target,
    )["total"]


def _signal_rejected_target_interact_loss(
    logits: torch.Tensor,
    rejected_target_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    rejected_target_mask = rejected_target_mask.float().reshape(-1)
    if sample_weight is not None:
        weights = rejected_target_mask * sample_weight.float().reshape(-1)
    else:
        weights = rejected_target_mask
    if float(weights.detach().sum().item()) <= 0.0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    interact_logits = logits[:, SyncOrSinkEnv.ACTION_INTERACT]
    loss_vec = nn.functional.softplus(interact_logits)
    return _weighted_mean(loss_vec, weights)


def _signal_rejected_target_interact_action_loss(
    logits: torch.Tensor,
    rejected_target_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bad_action_id = torch.full(
        rejected_target_mask.reshape(-1).shape,
        int(SyncOrSinkEnv.ACTION_INTERACT),
        dtype=torch.long,
        device=logits.device,
    )
    return _signal_bad_action_loss(
        logits,
        bad_action_id,
        rejected_target_mask,
        sample_weight=sample_weight,
    )


def _signal_bad_action_loss(
    logits: torch.Tensor,
    bad_action_id: torch.Tensor,
    bad_action_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bad_action_mask = bad_action_mask.float().reshape(-1)
    if sample_weight is not None:
        weights = bad_action_mask * sample_weight.float().reshape(-1)
    else:
        weights = bad_action_mask
    if float(weights.detach().sum().item()) <= 0.0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    action_ids = bad_action_id.long().reshape(-1).clamp(0, logits.shape[-1] - 1)
    bad_logits = logits.gather(1, action_ids.unsqueeze(1)).squeeze(1)
    other_logits = logits.masked_fill(
        nn.functional.one_hot(action_ids, num_classes=logits.shape[-1]).bool(),
        torch.finfo(logits.dtype).min,
    )
    other_logsumexp = torch.logsumexp(other_logits, dim=-1)
    loss_vec = nn.functional.softplus(bad_logits - other_logsumexp)
    return _weighted_mean(loss_vec, weights)


def _signal_bad_redundant_target_interact_loss(
    logits: torch.Tensor,
    bad_redundant_target_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return _signal_rejected_target_interact_loss(
        logits,
        bad_redundant_target_mask,
        sample_weight=sample_weight,
    )


def _signal_decoy_drift_action_loss(
    logits: torch.Tensor,
    decoy_drift_action_id: torch.Tensor,
    decoy_drift_action_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return _signal_bad_action_loss(
        logits,
        decoy_drift_action_id,
        decoy_drift_action_mask,
        sample_weight=sample_weight,
    )


def _signal_target_match_action_loss(
    logits: torch.Tensor,
    target_match_action_id: torch.Tensor,
    target_match_action_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    target_match_action_mask = target_match_action_mask.float().reshape(-1)
    if sample_weight is not None:
        weights = target_match_action_mask * sample_weight.float().reshape(-1)
    else:
        weights = target_match_action_mask
    if float(weights.detach().sum().item()) <= 0.0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    action_ids = target_match_action_id.long().reshape(-1).clamp(0, logits.shape[-1] - 1)
    loss_vec = nn.functional.cross_entropy(logits, action_ids, reduction="none")
    return _weighted_mean(loss_vec, weights)


def _signal_target_scan_action_loss(
    logits: torch.Tensor,
    target_scan_action_mask: torch.Tensor,
    target_scan_kind_id: torch.Tensor,
    *,
    first_weight: float = 0.0,
    refresh_weight: float = 0.0,
    joint_weight: float = 0.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    first_weight = max(0.0, float(first_weight))
    refresh_weight = max(0.0, float(refresh_weight))
    joint_weight = max(0.0, float(joint_weight))
    if first_weight <= 0.0 and refresh_weight <= 0.0 and joint_weight <= 0.0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    mask = target_scan_action_mask.float().reshape(-1)
    kind = target_scan_kind_id.long().reshape(-1)
    weights = torch.zeros_like(mask)
    if first_weight > 0.0:
        weights = weights + first_weight * (kind == _SIGNAL_TARGET_SCAN_KIND_FIRST).float()
    if refresh_weight > 0.0:
        weights = weights + refresh_weight * (kind == _SIGNAL_TARGET_SCAN_KIND_REFRESH).float()
    if joint_weight > 0.0:
        weights = weights + joint_weight * (kind == _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION).float()
    weights = weights * mask
    if sample_weight is not None:
        weights = weights * sample_weight.float().reshape(-1)
    if float(weights.detach().sum().item()) <= 0.0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    action_ids = torch.full(
        (logits.shape[0],),
        int(SyncOrSinkEnv.ACTION_INTERACT),
        dtype=torch.long,
        device=logits.device,
    )
    loss_vec = nn.functional.cross_entropy(logits, action_ids, reduction="none")
    return _weighted_mean(loss_vec, weights)


def _signal_scan_decision_loss(
    logits: torch.Tensor,
    positive_scan_mask: torch.Tensor,
    negative_scan_mask: torch.Tensor,
    *,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    positive_scan_mask = positive_scan_mask.float().reshape(-1).clamp(0.0, 1.0)
    negative_scan_mask = negative_scan_mask.float().reshape(-1).clamp(0.0, 1.0)
    negative_scan_mask = negative_scan_mask * (1.0 - positive_scan_mask)
    positive_weight = max(0.0, float(positive_weight))
    negative_weight = max(0.0, float(negative_weight))
    weights = positive_scan_mask * positive_weight + negative_scan_mask * negative_weight
    if sample_weight is not None:
        weights = weights * sample_weight.float().reshape(-1)
    if float(weights.detach().sum().item()) <= 0.0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    targets = positive_scan_mask
    interact_logits = logits[:, SyncOrSinkEnv.ACTION_INTERACT]
    loss_vec = nn.functional.binary_cross_entropy_with_logits(
        interact_logits,
        targets,
        reduction="none",
    )
    return _weighted_mean(loss_vec, weights)


def _signal_scan_gate_loss(
    scan_gate_logits: torch.Tensor,
    positive_scan_mask: torch.Tensor,
    negative_scan_mask: torch.Tensor,
    *,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    positive_scan_mask = positive_scan_mask.float().reshape(-1).clamp(0.0, 1.0)
    negative_scan_mask = negative_scan_mask.float().reshape(-1).clamp(0.0, 1.0)
    negative_scan_mask = negative_scan_mask * (1.0 - positive_scan_mask)
    positive_weight = max(0.0, float(positive_weight))
    negative_weight = max(0.0, float(negative_weight))
    weights = positive_scan_mask * positive_weight + negative_scan_mask * negative_weight
    if sample_weight is not None:
        weights = weights * sample_weight.float().reshape(-1)
    if float(weights.detach().sum().item()) <= 0.0:
        return torch.tensor(0.0, dtype=scan_gate_logits.dtype, device=scan_gate_logits.device)
    loss_vec = nn.functional.binary_cross_entropy_with_logits(
        scan_gate_logits.reshape(-1),
        positive_scan_mask,
        reduction="none",
    )
    return _weighted_mean(loss_vec, weights)


def _signal_target_validity_loss(
    validity_logits: torch.Tensor,
    validity_mask: torch.Tensor,
    validity_label: torch.Tensor,
    *,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    validity_mask = validity_mask.float().reshape(-1).clamp(0.0, 1.0)
    validity_label = validity_label.float().reshape(-1).clamp(0.0, 1.0)
    positive_weight = max(0.0, float(positive_weight))
    negative_weight = max(0.0, float(negative_weight))
    weights = validity_mask * (
        validity_label * positive_weight + (1.0 - validity_label) * negative_weight
    )
    if sample_weight is not None:
        weights = weights * sample_weight.float().reshape(-1)
    if float(weights.detach().sum().item()) <= 0.0:
        return torch.tensor(0.0, dtype=validity_logits.dtype, device=validity_logits.device)
    loss_vec = nn.functional.binary_cross_entropy_with_logits(
        validity_logits.reshape(-1),
        validity_label,
        reduction="none",
    )
    return _weighted_mean(loss_vec, weights)


def _signal_target_decision_loss(
    decision_logits: torch.Tensor,
    decision_mask: torch.Tensor,
    decision_label: torch.Tensor,
    *,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return _signal_target_validity_loss(
        decision_logits,
        decision_mask,
        decision_label,
        positive_weight=positive_weight,
        negative_weight=negative_weight,
        sample_weight=sample_weight,
    )


def _signal_target_aux_loss(
    pred: torch.Tensor,
    target_mask: torch.Tensor,
    target_xy: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    target_mask = target_mask.float().reshape(-1)
    target_xy = target_xy.float().reshape(-1, 2)
    if sample_weight is not None:
        weights = sample_weight.float().reshape(-1)
    else:
        weights = torch.ones_like(target_mask)
    present_loss_vec = nn.functional.binary_cross_entropy_with_logits(
        pred[:, 0],
        target_mask,
        reduction="none",
    )
    present_loss = _weighted_mean(present_loss_vec, weights)
    xy_weights = weights * target_mask
    if float(xy_weights.detach().sum().item()) <= 0.0:
        xy_loss = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
    else:
        pred_xy = torch.sigmoid(pred[:, 1:3])
        xy_loss = _weighted_mean(((pred_xy - target_xy) ** 2).mean(dim=-1), xy_weights)
    return present_loss + xy_loss


def _send_threshold_for_target_rate(probs, target_rate: float) -> float:
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if probs.size == 0:
        return 1.0
    target_rate = min(1.0, max(0.0, float(target_rate)))
    if target_rate <= 0.0:
        return 1.0
    if target_rate >= 1.0:
        return 0.0
    return float(np.quantile(probs, 1.0 - target_rate))


def _calibrate_recurrent_send_threshold(
    cfg: RecurrentConfig,
    model: MAPPORecurrentActor,
    episodes,
    device,
) -> dict:
    if not cfg.comm:
        return {}
    probs: list[float] = []
    labels: list[float] = []
    model.eval()
    with torch.no_grad():
        for ep_data in episodes:
            obs_seq = torch.tensor(ep_data["obs"], dtype=torch.float32, device=device)
            msg_len_seq = torch.tensor(ep_data["msg_lens"], dtype=torch.long, device=device)
            hidden = model.init_hidden(obs_seq.shape[1], device)
            for t in range(obs_seq.shape[0]):
                _logits, send_logits, _token_logits, _len_logits, hidden = model(obs_seq[t], hidden)
                probs.extend(torch.sigmoid(send_logits.squeeze(-1)).detach().cpu().tolist())
                labels.extend((msg_len_seq[t] > 0).float().detach().cpu().tolist())
    if not probs:
        return {}
    probs_arr = np.asarray(probs, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)
    label_rate = float(labels_arr.mean()) if labels_arr.size else 0.0
    target_rate = (
        float(cfg.bc_send_threshold_target_rate)
        if float(cfg.bc_send_threshold_target_rate) >= 0.0
        else label_rate
    )
    threshold = _send_threshold_for_target_rate(probs_arr, target_rate)
    pred_rate = float((probs_arr > threshold).mean()) if probs_arr.size else 0.0
    old_threshold = float(cfg.eval_send_threshold)
    cfg.eval_send_threshold = float(threshold)
    return {
        "old_threshold": old_threshold,
        "threshold": float(threshold),
        "target_rate": float(min(1.0, max(0.0, target_rate))),
        "label_rate": label_rate,
        "pred_rate": pred_rate,
        "mean_prob": float(probs_arr.mean()),
        "max_prob": float(probs_arr.max()),
    }


def train_recurrent_bc(
    cfg: RecurrentConfig,
    episodes,
    device,
    model: MAPPORecurrentActor | None = None,
    *,
    wandb_run=None,
    log_prefix: str = "bc",
    log_context: dict | None = None,
):
    """Train recurrent BC via truncated BPTT on episode sequences."""
    set_global_seeds(cfg.seed)
    obs_dims = {int(ep["obs"].shape[-1]) for ep in episodes}
    if len(obs_dims) != 1:
        raise ValueError(
            "recurrent training episodes have different observation dimensions; "
            "use --obs-memory-mode egocentric for multi-map exploration memory"
        )
    obs_dim = episodes[0]["obs"].shape[-1]
    if model is None:
        model = MAPPORecurrentActor(
            obs_dim=obs_dim, action_dim=8, hidden_dim=cfg.hidden_dim,
            comm_enabled=cfg.comm,
            comm_token_limit=cfg.comm_token_limit,
            comm_vocab_size=cfg.comm_vocab_size,
        ).to(device)
    trainable_params = list(model.parameters())
    optimizer = optim.Adam(trainable_params, lr=cfg.bc_lr)
    send_pos_weight = _recurrent_comm_send_pos_weight(episodes, cfg, device) if cfg.comm else None
    bc_eval_every = max(0, int(getattr(cfg, "bc_eval_every_epochs", 0)))
    bc_eval_seed_count = max(1, int(getattr(cfg, "bc_eval_seed_count", 1)))
    bc_eval_episodes = int(getattr(cfg, "bc_eval_episodes", 0))
    bc_restore_best = bool(getattr(cfg, "bc_restore_best_eval_epoch", False))
    best_epoch_score = None
    best_epoch_state = None
    best_epoch_threshold = None
    best_epoch_row = None

    for epoch in range(cfg.bc_epochs):
        np.random.shuffle(episodes)
        total_loss, total_correct, total_count = 0.0, 0, 0
        total_comm_loss = 0.0
        comm_loss_steps = 0
        comm_send_loss_sum = 0.0
        comm_len_loss_sum = 0.0
        comm_token_loss_sum = 0.0
        comm_send_rate_loss_sum = 0.0
        comm_agents = 0
        comm_send_label_count = 0
        comm_pred_send_count = 0
        comm_send_prob_sum = 0.0
        comm_pred_positive_len_count = 0
        comm_pred_positive_len_on_label_positive_count = 0
        comm_label_positive_count = 0
        comm_true_len_sum = 0.0
        comm_pred_len_sum = 0.0
        rejected_target_loss_sum = 0.0
        rejected_target_loss_steps = 0
        rejected_target_action_loss_sum = 0.0
        rejected_target_action_loss_steps = 0
        rejected_target_count = 0
        rejected_target_pred_interact_count = 0
        rejected_target_interact_prob_sum = 0.0
        bad_redundant_target_loss_sum = 0.0
        bad_redundant_target_loss_steps = 0
        bad_redundant_target_count = 0
        bad_redundant_target_pred_interact_count = 0
        bad_redundant_target_interact_prob_sum = 0.0
        target_scan_label_count = 0
        target_scan_pred_interact_count = 0
        target_scan_interact_prob_sum = 0.0
        target_scan_kind_counts = [0 for _ in _SIGNAL_TARGET_SCAN_KIND_NAMES]
        target_scan_kind_pred_interact_counts = [0 for _ in _SIGNAL_TARGET_SCAN_KIND_NAMES]
        target_scan_kind_interact_prob_sums = [0.0 for _ in _SIGNAL_TARGET_SCAN_KIND_NAMES]
        target_scan_action_loss_sum = 0.0
        target_scan_action_loss_steps = 0
        target_opportunity_label_count = 0
        target_opportunity_pred_interact_count = 0
        target_opportunity_interact_prob_sum = 0.0
        target_opportunity_kind_counts = [0 for _ in _SIGNAL_TARGET_SCAN_KIND_NAMES]
        target_opportunity_kind_pred_interact_counts = [
            0 for _ in _SIGNAL_TARGET_SCAN_KIND_NAMES
        ]
        target_opportunity_kind_interact_prob_sums = [
            0.0 for _ in _SIGNAL_TARGET_SCAN_KIND_NAMES
        ]
        target_opportunity_action_loss_sum = 0.0
        target_opportunity_action_loss_steps = 0
        redundant_wait_action_loss_sum = 0.0
        redundant_wait_action_loss_steps = 0
        redundant_wait_action_count = 0
        redundant_wait_pred_action_count = 0
        redundant_wait_action_prob_sum = 0.0
        target_pursuit_action_loss_sum = 0.0
        target_pursuit_action_loss_steps = 0
        target_pursuit_action_count = 0
        target_pursuit_pred_action_count = 0
        target_pursuit_action_prob_sum = 0.0
        sync_response_action_loss_sum = 0.0
        sync_response_action_loss_steps = 0
        sync_response_action_count = 0
        sync_response_pred_action_count = 0
        sync_response_action_prob_sum = 0.0
        scan_decision_loss_sum = 0.0
        scan_decision_loss_steps = 0
        scan_decision_positive_count = 0
        scan_decision_negative_count = 0
        scan_decision_positive_pred_count = 0
        scan_decision_negative_pred_count = 0
        scan_gate_loss_sum = 0.0
        scan_gate_loss_steps = 0
        scan_gate_positive_count = 0
        scan_gate_negative_count = 0
        scan_gate_positive_pred_count = 0
        scan_gate_negative_pred_count = 0
        scan_gate_positive_prob_sum = 0.0
        scan_gate_negative_prob_sum = 0.0
        target_match_action_loss_sum = 0.0
        target_match_action_loss_steps = 0
        target_match_action_count = 0
        target_match_pred_action_count = 0
        target_match_action_prob_sum = 0.0
        target_validity_loss_sum = 0.0
        target_validity_loss_steps = 0
        target_validity_positive_count = 0
        target_validity_negative_count = 0
        target_validity_positive_pred_count = 0
        target_validity_negative_pred_count = 0
        target_validity_positive_prob_sum = 0.0
        target_validity_negative_prob_sum = 0.0
        target_decision_loss_sum = 0.0
        target_decision_loss_steps = 0
        target_decision_positive_count = 0
        target_decision_negative_count = 0
        target_decision_positive_pred_count = 0
        target_decision_negative_pred_count = 0
        target_decision_positive_prob_sum = 0.0
        target_decision_negative_prob_sum = 0.0
        target_aux_loss_sum = 0.0
        target_aux_steps = 0
        target_aux_count = 0
        target_aux_present_correct = 0
        target_aux_present_total = 0
        target_aux_xy_l1_sum = 0.0
        decoy_drift_action_loss_sum = 0.0
        decoy_drift_action_loss_steps = 0
        decoy_drift_action_count = 0
        decoy_drift_pred_bad_action_count = 0
        decoy_drift_bad_action_prob_sum = 0.0
        decoy_scan_action_loss_sum = 0.0
        decoy_scan_action_loss_steps = 0
        decoy_scan_action_count = 0
        decoy_scan_pred_bad_action_count = 0
        decoy_scan_bad_action_prob_sum = 0.0
        rejected_target_drift_action_loss_sum = 0.0
        rejected_target_drift_action_loss_steps = 0
        rejected_target_drift_action_count = 0
        rejected_target_drift_pred_bad_action_count = 0
        rejected_target_drift_bad_action_prob_sum = 0.0
        loss_den = 0.0
        chunks = 0

        for ep_data in episodes:
            obs_seq = torch.tensor(ep_data["obs"], dtype=torch.float32, device=device)
            act_seq = torch.tensor(ep_data["actions"], dtype=torch.long, device=device)
            msg_seq = torch.tensor(ep_data["msg_tokens"], dtype=torch.long, device=device)
            msg_len_seq = torch.tensor(ep_data["msg_lens"], dtype=torch.long, device=device)
            if "step_weights" in ep_data:
                step_weight_seq = torch.tensor(ep_data["step_weights"], dtype=torch.float32, device=device)
            else:
                step_weight_seq = torch.ones_like(act_seq, dtype=torch.float32, device=device)
            if "signal_rejected_target_mask" in ep_data:
                rejected_target_seq = torch.tensor(
                    ep_data["signal_rejected_target_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                rejected_target_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_bad_redundant_target_mask" in ep_data:
                bad_redundant_target_seq = torch.tensor(
                    ep_data["signal_bad_redundant_target_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                bad_redundant_target_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_target_scan_action_mask" in ep_data:
                target_scan_action_mask_seq = torch.tensor(
                    ep_data["signal_target_scan_action_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                target_scan_action_mask_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_target_scan_kind_id" in ep_data:
                target_scan_kind_id_seq = torch.tensor(
                    ep_data["signal_target_scan_kind_id"],
                    dtype=torch.long,
                    device=device,
                )
            else:
                target_scan_kind_id_seq = torch.full_like(
                    act_seq,
                    _SIGNAL_TARGET_SCAN_KIND_UNKNOWN,
                    dtype=torch.long,
                    device=device,
                )
            if "signal_target_opportunity_action_mask" in ep_data:
                target_opportunity_action_mask_seq = torch.tensor(
                    ep_data["signal_target_opportunity_action_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                target_opportunity_action_mask_seq = torch.zeros_like(
                    act_seq,
                    dtype=torch.float32,
                    device=device,
                )
            if "signal_target_opportunity_kind_id" in ep_data:
                target_opportunity_kind_id_seq = torch.tensor(
                    ep_data["signal_target_opportunity_kind_id"],
                    dtype=torch.long,
                    device=device,
                )
            else:
                target_opportunity_kind_id_seq = torch.full_like(
                    act_seq,
                    _SIGNAL_TARGET_SCAN_KIND_UNKNOWN,
                    dtype=torch.long,
                    device=device,
                )
            if "signal_redundant_target_wait_action_mask" in ep_data:
                redundant_wait_action_mask_seq = torch.tensor(
                    ep_data["signal_redundant_target_wait_action_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                redundant_wait_action_mask_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_redundant_target_wait_action_id" in ep_data:
                redundant_wait_action_id_seq = torch.tensor(
                    ep_data["signal_redundant_target_wait_action_id"],
                    dtype=torch.long,
                    device=device,
                )
            else:
                redundant_wait_action_id_seq = torch.full_like(act_seq, -1, dtype=torch.long, device=device)
            if "signal_target_pursuit_action_mask" in ep_data:
                target_pursuit_action_mask_seq = torch.tensor(
                    ep_data["signal_target_pursuit_action_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                target_pursuit_action_mask_seq = torch.zeros_like(
                    act_seq,
                    dtype=torch.float32,
                    device=device,
                )
            if "signal_target_pursuit_action_id" in ep_data:
                target_pursuit_action_id_seq = torch.tensor(
                    ep_data["signal_target_pursuit_action_id"],
                    dtype=torch.long,
                    device=device,
                )
            else:
                target_pursuit_action_id_seq = torch.full_like(
                    act_seq,
                    -1,
                    dtype=torch.long,
                    device=device,
                )
            if "signal_sync_response_action_mask" in ep_data:
                sync_response_action_mask_seq = torch.tensor(
                    ep_data["signal_sync_response_action_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                sync_response_action_mask_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_sync_response_action_id" in ep_data:
                sync_response_action_id_seq = torch.tensor(
                    ep_data["signal_sync_response_action_id"],
                    dtype=torch.long,
                    device=device,
                )
            else:
                sync_response_action_id_seq = torch.full_like(act_seq, -1, dtype=torch.long, device=device)
            if "signal_target_match_action_mask" in ep_data:
                target_match_action_mask_seq = torch.tensor(
                    ep_data["signal_target_match_action_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                target_match_action_mask_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_target_match_action_id" in ep_data:
                target_match_action_id_seq = torch.tensor(
                    ep_data["signal_target_match_action_id"],
                    dtype=torch.long,
                    device=device,
                )
            else:
                target_match_action_id_seq = torch.full_like(act_seq, -1, dtype=torch.long, device=device)
            if "signal_target_validity_mask" in ep_data:
                target_validity_mask_seq = torch.tensor(
                    ep_data["signal_target_validity_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                target_validity_mask_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_target_validity_label" in ep_data:
                target_validity_label_seq = torch.tensor(
                    ep_data["signal_target_validity_label"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                target_validity_label_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_target_decision_mask" in ep_data:
                target_decision_mask_seq = torch.tensor(
                    ep_data["signal_target_decision_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                target_decision_mask_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_target_decision_label" in ep_data:
                target_decision_label_seq = torch.tensor(
                    ep_data["signal_target_decision_label"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                target_decision_label_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_target_aux_mask" in ep_data:
                target_aux_mask_seq = torch.tensor(
                    ep_data["signal_target_aux_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                target_aux_mask_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_target_aux_xy" in ep_data:
                target_aux_xy_seq = torch.tensor(
                    ep_data["signal_target_aux_xy"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                target_aux_xy_seq = torch.zeros(
                    (*act_seq.shape, 2),
                    dtype=torch.float32,
                    device=device,
                )
            if "signal_decoy_drift_action_mask" in ep_data:
                decoy_drift_action_mask_seq = torch.tensor(
                    ep_data["signal_decoy_drift_action_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                decoy_drift_action_mask_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_decoy_drift_action_id" in ep_data:
                decoy_drift_action_id_seq = torch.tensor(
                    ep_data["signal_decoy_drift_action_id"],
                    dtype=torch.long,
                    device=device,
                )
            else:
                decoy_drift_action_id_seq = torch.full_like(act_seq, -1, dtype=torch.long, device=device)
            if "signal_decoy_scan_action_mask" in ep_data:
                decoy_scan_action_mask_seq = torch.tensor(
                    ep_data["signal_decoy_scan_action_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                decoy_scan_action_mask_seq = torch.zeros_like(act_seq, dtype=torch.float32, device=device)
            if "signal_decoy_scan_action_id" in ep_data:
                decoy_scan_action_id_seq = torch.tensor(
                    ep_data["signal_decoy_scan_action_id"],
                    dtype=torch.long,
                    device=device,
                )
            else:
                decoy_scan_action_id_seq = torch.full_like(act_seq, -1, dtype=torch.long, device=device)
            if "signal_rejected_target_drift_action_mask" in ep_data:
                rejected_target_drift_action_mask_seq = torch.tensor(
                    ep_data["signal_rejected_target_drift_action_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                rejected_target_drift_action_mask_seq = torch.zeros_like(
                    act_seq,
                    dtype=torch.float32,
                    device=device,
                )
            if "signal_rejected_target_drift_action_id" in ep_data:
                rejected_target_drift_action_id_seq = torch.tensor(
                    ep_data["signal_rejected_target_drift_action_id"],
                    dtype=torch.long,
                    device=device,
                )
            else:
                rejected_target_drift_action_id_seq = torch.full_like(
                    act_seq,
                    -1,
                    dtype=torch.long,
                    device=device,
                )
            T = obs_seq.shape[0]
            N = obs_seq.shape[1]
            episode_weight = _episode_training_weight(ep_data)
            if episode_weight <= 0.0:
                continue
            episode_chunk_losses = []

            # Process in truncated BPTT chunks
            hidden = model.init_hidden(N, device)
            for t_start in range(0, T, cfg.bc_seq_len):
                t_end = min(t_start + cfg.bc_seq_len, T)
                chunk_loss = 0.0
                chunk_correct = 0
                chunk_count = 0

                # Detach hidden state at chunk boundaries
                hidden = (hidden[0].detach(), hidden[1].detach())

                for t in range(t_start, t_end):
                    if cfg.comm:
                        logits, send_logits, token_logits, len_logits, hidden = model(obs_seq[t], hidden)
                    else:
                        logits, hidden = model(obs_seq[t], hidden)
                    sample_weight = step_weight_seq[t]
                    action_loss = nn.functional.cross_entropy(logits, act_seq[t], reduction="none")
                    loss = _weighted_mean(action_loss, sample_weight)
                    rejected_target_mask = rejected_target_seq[t]
                    target_scan_action_mask = target_scan_action_mask_seq[t]
                    target_scan_kind_ids = target_scan_kind_id_seq[t]
                    target_opportunity_action_mask = target_opportunity_action_mask_seq[t]
                    target_opportunity_kind_ids = target_opportunity_kind_id_seq[t]
                    positive_target_scan_mask = torch.maximum(
                        target_scan_action_mask.float(),
                        target_opportunity_action_mask.float(),
                    ).clamp(0.0, 1.0)
                    rejected_loss_mask = rejected_target_mask * (1.0 - positive_target_scan_mask)
                    rejected_weight = float(cfg.bc_signal_rejected_target_interact_loss_weight)
                    if rejected_weight > 0.0:
                        rejected_loss = _signal_rejected_target_interact_loss(
                            logits,
                            rejected_loss_mask,
                            sample_weight=sample_weight,
                        )
                        loss = loss + rejected_weight * rejected_loss
                        rejected_target_loss_sum += float(rejected_loss.item())
                        rejected_target_loss_steps += 1
                    rejected_action_weight = float(
                        cfg.bc_signal_rejected_target_interact_action_loss_weight
                    )
                    if rejected_action_weight > 0.0:
                        rejected_action_loss = _signal_rejected_target_interact_action_loss(
                            logits,
                            rejected_loss_mask,
                            sample_weight=sample_weight,
                        )
                        loss = loss + rejected_action_weight * rejected_action_loss
                        rejected_target_action_loss_sum += float(rejected_action_loss.item())
                        rejected_target_action_loss_steps += 1
                    with torch.no_grad():
                        rejected_bool = rejected_loss_mask > 0.0
                        rejected_count = int(rejected_bool.sum().item())
                        if rejected_count > 0:
                            rejected_target_count += rejected_count
                            rejected_target_pred_interact_count += int(
                                (logits.argmax(dim=-1) == SyncOrSinkEnv.ACTION_INTERACT)[rejected_bool].sum().item()
                            )
                            rejected_target_interact_prob_sum += float(
                                torch.softmax(logits, dim=-1)[rejected_bool, SyncOrSinkEnv.ACTION_INTERACT]
                                .sum()
                                .item()
                            )
                    bad_redundant_target_mask = bad_redundant_target_seq[t]
                    bad_redundant_weight = float(cfg.bc_signal_bad_redundant_target_interact_loss_weight)
                    if bad_redundant_weight > 0.0:
                        bad_redundant_loss = _signal_bad_redundant_target_interact_loss(
                            logits,
                            bad_redundant_target_mask,
                            sample_weight=sample_weight,
                        )
                        loss = loss + bad_redundant_weight * bad_redundant_loss
                        bad_redundant_target_loss_sum += float(bad_redundant_loss.item())
                        bad_redundant_target_loss_steps += 1
                    with torch.no_grad():
                        bad_redundant_bool = bad_redundant_target_mask > 0.0
                        bad_redundant_count = int(bad_redundant_bool.sum().item())
                        if bad_redundant_count > 0:
                            bad_redundant_target_count += bad_redundant_count
                            bad_redundant_target_pred_interact_count += int(
                                (logits.argmax(dim=-1) == SyncOrSinkEnv.ACTION_INTERACT)[bad_redundant_bool]
                                .sum()
                                .item()
                            )
                            bad_redundant_target_interact_prob_sum += float(
                                torch.softmax(logits, dim=-1)[bad_redundant_bool, SyncOrSinkEnv.ACTION_INTERACT]
                                .sum()
                                .item()
                            )
                    target_scan_action_mask = target_scan_action_mask_seq[t]
                    target_scan_kind_ids = target_scan_kind_id_seq[t]
                    with torch.no_grad():
                        target_scan_bool = target_scan_action_mask > 0.0
                        target_scan_count = int(target_scan_bool.sum().item())
                        if target_scan_count > 0:
                            target_scan_label_count += target_scan_count
                            pred_interact = logits.argmax(dim=-1) == SyncOrSinkEnv.ACTION_INTERACT
                            interact_probs = torch.softmax(logits, dim=-1)[:, SyncOrSinkEnv.ACTION_INTERACT]
                            target_scan_pred_interact_count += int(
                                pred_interact[target_scan_bool].sum().item()
                            )
                            target_scan_interact_prob_sum += float(
                                interact_probs[target_scan_bool].sum().item()
                            )
                            clipped_kind_ids = target_scan_kind_ids.clamp(0, len(_SIGNAL_TARGET_SCAN_KIND_NAMES) - 1)
                            for kind_idx in range(len(_SIGNAL_TARGET_SCAN_KIND_NAMES)):
                                kind_bool = target_scan_bool & (clipped_kind_ids == int(kind_idx))
                                kind_count = int(kind_bool.sum().item())
                                if kind_count <= 0:
                                    continue
                                target_scan_kind_counts[kind_idx] += kind_count
                                target_scan_kind_pred_interact_counts[kind_idx] += int(
                                    pred_interact[kind_bool].sum().item()
                                )
                                target_scan_kind_interact_prob_sums[kind_idx] += float(
                                    interact_probs[kind_bool].sum().item()
                                )
                    first_scan_action_weight = float(cfg.bc_signal_first_target_scan_action_weight)
                    refresh_scan_action_weight = float(cfg.bc_signal_refresh_target_scan_action_weight)
                    joint_scan_action_weight = float(cfg.bc_signal_joint_target_scan_action_weight)
                    if (
                        first_scan_action_weight > 0.0
                        or refresh_scan_action_weight > 0.0
                        or joint_scan_action_weight > 0.0
                    ):
                        target_scan_action_loss = _signal_target_scan_action_loss(
                            logits,
                            target_scan_action_mask,
                            target_scan_kind_ids,
                            first_weight=first_scan_action_weight,
                            refresh_weight=refresh_scan_action_weight,
                            joint_weight=joint_scan_action_weight,
                            sample_weight=sample_weight,
                        )
                        loss = loss + target_scan_action_loss
                        target_scan_action_loss_sum += float(target_scan_action_loss.item())
                        target_scan_action_loss_steps += 1
                    target_opportunity_action_mask = target_opportunity_action_mask_seq[t]
                    target_opportunity_kind_ids = target_opportunity_kind_id_seq[t]
                    with torch.no_grad():
                        target_opportunity_bool = target_opportunity_action_mask > 0.0
                        target_opportunity_count = int(target_opportunity_bool.sum().item())
                        if target_opportunity_count > 0:
                            target_opportunity_label_count += target_opportunity_count
                            pred_interact = logits.argmax(dim=-1) == SyncOrSinkEnv.ACTION_INTERACT
                            interact_probs = torch.softmax(logits, dim=-1)[:, SyncOrSinkEnv.ACTION_INTERACT]
                            target_opportunity_pred_interact_count += int(
                                pred_interact[target_opportunity_bool].sum().item()
                            )
                            target_opportunity_interact_prob_sum += float(
                                interact_probs[target_opportunity_bool].sum().item()
                            )
                            clipped_kind_ids = target_opportunity_kind_ids.clamp(
                                0,
                                len(_SIGNAL_TARGET_SCAN_KIND_NAMES) - 1,
                            )
                            for kind_idx in range(len(_SIGNAL_TARGET_SCAN_KIND_NAMES)):
                                kind_bool = target_opportunity_bool & (clipped_kind_ids == int(kind_idx))
                                kind_count = int(kind_bool.sum().item())
                                if kind_count <= 0:
                                    continue
                                target_opportunity_kind_counts[kind_idx] += kind_count
                                target_opportunity_kind_pred_interact_counts[kind_idx] += int(
                                    pred_interact[kind_bool].sum().item()
                                )
                                target_opportunity_kind_interact_prob_sums[kind_idx] += float(
                                    interact_probs[kind_bool].sum().item()
                                )
                    opportunity_action_weight = float(cfg.bc_signal_target_opportunity_action_weight)
                    if opportunity_action_weight > 0.0:
                        target_opportunity_action_loss = _signal_target_scan_action_loss(
                            logits,
                            target_opportunity_action_mask,
                            target_opportunity_kind_ids,
                            first_weight=opportunity_action_weight,
                            refresh_weight=opportunity_action_weight,
                            joint_weight=opportunity_action_weight,
                            sample_weight=sample_weight,
                        )
                        loss = loss + target_opportunity_action_loss
                        target_opportunity_action_loss_sum += float(
                            target_opportunity_action_loss.item()
                        )
                        target_opportunity_action_loss_steps += 1
                    redundant_wait_action_mask = redundant_wait_action_mask_seq[t]
                    redundant_wait_action_id = redundant_wait_action_id_seq[t]
                    redundant_wait_action_weight = float(
                        cfg.bc_signal_redundant_target_wait_action_loss_weight
                    )
                    if redundant_wait_action_weight > 0.0:
                        redundant_wait_loss = _signal_target_match_action_loss(
                            logits,
                            redundant_wait_action_id,
                            redundant_wait_action_mask,
                            sample_weight=sample_weight,
                        )
                        loss = loss + redundant_wait_action_weight * redundant_wait_loss
                        redundant_wait_action_loss_sum += float(redundant_wait_loss.item())
                        redundant_wait_action_loss_steps += 1
                    with torch.no_grad():
                        redundant_wait_bool = redundant_wait_action_mask > 0.0
                        redundant_wait_count = int(redundant_wait_bool.sum().item())
                        if redundant_wait_count > 0:
                            redundant_wait_action_count += redundant_wait_count
                            wait_action_ids = redundant_wait_action_id.clamp(0, logits.shape[-1] - 1)
                            redundant_wait_pred_action_count += int(
                                (logits.argmax(dim=-1) == wait_action_ids)[redundant_wait_bool]
                                .sum()
                                .item()
                            )
                            redundant_wait_action_prob_sum += float(
                                torch.softmax(logits, dim=-1)
                                .gather(1, wait_action_ids.unsqueeze(1))
                                .squeeze(1)[redundant_wait_bool]
                                .sum()
                                .item()
                            )
                    target_pursuit_action_mask = target_pursuit_action_mask_seq[t]
                    target_pursuit_action_id = target_pursuit_action_id_seq[t]
                    target_pursuit_action_weight = float(cfg.bc_signal_target_pursuit_action_weight)
                    if target_pursuit_action_weight > 0.0:
                        target_pursuit_loss = _signal_target_match_action_loss(
                            logits,
                            target_pursuit_action_id,
                            target_pursuit_action_mask,
                            sample_weight=sample_weight,
                        )
                        loss = loss + target_pursuit_action_weight * target_pursuit_loss
                        target_pursuit_action_loss_sum += float(target_pursuit_loss.item())
                        target_pursuit_action_loss_steps += 1
                    with torch.no_grad():
                        target_pursuit_bool = target_pursuit_action_mask > 0.0
                        target_pursuit_count = int(target_pursuit_bool.sum().item())
                        if target_pursuit_count > 0:
                            target_pursuit_action_count += target_pursuit_count
                            target_action_ids = target_pursuit_action_id.clamp(0, logits.shape[-1] - 1)
                            target_pursuit_pred_action_count += int(
                                (logits.argmax(dim=-1) == target_action_ids)[target_pursuit_bool]
                                .sum()
                                .item()
                            )
                            target_pursuit_action_prob_sum += float(
                                torch.softmax(logits, dim=-1)
                                .gather(1, target_action_ids.unsqueeze(1))
                                .squeeze(1)[target_pursuit_bool]
                                .sum()
                                .item()
                            )
                    sync_response_action_mask = sync_response_action_mask_seq[t]
                    sync_response_action_id = sync_response_action_id_seq[t]
                    sync_response_action_weight = float(
                        cfg.bc_signal_sync_response_action_loss_weight
                    )
                    if sync_response_action_weight > 0.0:
                        sync_response_loss = _signal_target_match_action_loss(
                            logits,
                            sync_response_action_id,
                            sync_response_action_mask,
                            sample_weight=sample_weight,
                        )
                        loss = loss + sync_response_action_weight * sync_response_loss
                        sync_response_action_loss_sum += float(sync_response_loss.item())
                        sync_response_action_loss_steps += 1
                    with torch.no_grad():
                        sync_response_bool = sync_response_action_mask > 0.0
                        sync_response_count = int(sync_response_bool.sum().item())
                        if sync_response_count > 0:
                            sync_response_action_count += sync_response_count
                            sync_action_ids = sync_response_action_id.clamp(0, logits.shape[-1] - 1)
                            sync_response_pred_action_count += int(
                                (logits.argmax(dim=-1) == sync_action_ids)[sync_response_bool]
                                .sum()
                                .item()
                            )
                            sync_response_action_prob_sum += float(
                                torch.softmax(logits, dim=-1)
                                .gather(1, sync_action_ids.unsqueeze(1))
                                .squeeze(1)[sync_response_bool]
                                .sum()
                                .item()
                            )
                    target_match_action_mask = target_match_action_mask_seq[t]
                    target_match_action_id = target_match_action_id_seq[t]
                    target_match_action_weight = float(cfg.bc_signal_target_match_action_weight)
                    if target_match_action_weight > 0.0:
                        target_match_loss = _signal_target_match_action_loss(
                            logits,
                            target_match_action_id,
                            target_match_action_mask,
                            sample_weight=sample_weight,
                        )
                        loss = loss + target_match_action_weight * target_match_loss
                        target_match_action_loss_sum += float(target_match_loss.item())
                        target_match_action_loss_steps += 1
                    with torch.no_grad():
                        target_match_bool = target_match_action_mask > 0.0
                        target_match_count = int(target_match_bool.sum().item())
                        if target_match_count > 0:
                            target_match_action_count += target_match_count
                            target_action_ids = target_match_action_id.clamp(0, logits.shape[-1] - 1)
                            target_match_pred_action_count += int(
                                (logits.argmax(dim=-1) == target_action_ids)[target_match_bool]
                                .sum()
                                .item()
                            )
                            target_match_action_prob_sum += float(
                                torch.softmax(logits, dim=-1)
                                .gather(1, target_action_ids.unsqueeze(1))
                                .squeeze(1)[target_match_bool]
                                .sum()
                                .item()
                            )
                    target_validity_mask = target_validity_mask_seq[t]
                    target_validity_label = target_validity_label_seq[t]
                    target_validity_weight = float(cfg.bc_signal_target_validity_loss_weight)
                    if target_validity_weight > 0.0 and hasattr(model, "signal_target_validity"):
                        target_validity_logits = model.signal_target_validity(hidden[0])
                        target_validity_loss = _signal_target_validity_loss(
                            target_validity_logits,
                            target_validity_mask,
                            target_validity_label,
                            positive_weight=cfg.bc_signal_target_validity_pos_weight,
                            negative_weight=cfg.bc_signal_target_validity_neg_weight,
                            sample_weight=sample_weight,
                        )
                        loss = loss + target_validity_weight * target_validity_loss
                        target_validity_loss_sum += float(target_validity_loss.item())
                        target_validity_loss_steps += 1
                        with torch.no_grad():
                            validity_bool = target_validity_mask > 0.0
                            positive_bool = validity_bool & (target_validity_label >= 0.5)
                            negative_bool = validity_bool & (target_validity_label < 0.5)
                            validity_prob = torch.sigmoid(target_validity_logits.reshape(-1))
                            pred_positive = validity_prob >= 0.5
                            target_validity_positive_count += int(positive_bool.sum().item())
                            target_validity_negative_count += int(negative_bool.sum().item())
                            target_validity_positive_pred_count += int(
                                pred_positive[positive_bool].sum().item()
                            )
                            target_validity_negative_pred_count += int(
                                pred_positive[negative_bool].sum().item()
                            )
                            target_validity_positive_prob_sum += float(
                                validity_prob[positive_bool].sum().item()
                            )
                            target_validity_negative_prob_sum += float(
                                validity_prob[negative_bool].sum().item()
                            )
                    target_decision_mask = target_decision_mask_seq[t]
                    target_decision_label = target_decision_label_seq[t]
                    target_decision_weight = float(cfg.bc_signal_target_decision_loss_weight)
                    if target_decision_weight > 0.0 and hasattr(model, "signal_target_decision"):
                        target_decision_logits = model.signal_target_decision(hidden[0])
                        target_decision_loss = _signal_target_decision_loss(
                            target_decision_logits,
                            target_decision_mask,
                            target_decision_label,
                            positive_weight=cfg.bc_signal_target_decision_pos_weight,
                            negative_weight=cfg.bc_signal_target_decision_neg_weight,
                            sample_weight=sample_weight,
                        )
                        loss = loss + target_decision_weight * target_decision_loss
                        target_decision_loss_sum += float(target_decision_loss.item())
                        target_decision_loss_steps += 1
                        with torch.no_grad():
                            decision_bool = target_decision_mask > 0.0
                            positive_bool = decision_bool & (target_decision_label >= 0.5)
                            negative_bool = decision_bool & (target_decision_label < 0.5)
                            decision_prob = torch.sigmoid(target_decision_logits.reshape(-1))
                            pred_positive = decision_prob >= 0.5
                            target_decision_positive_count += int(positive_bool.sum().item())
                            target_decision_negative_count += int(negative_bool.sum().item())
                            target_decision_positive_pred_count += int(
                                pred_positive[positive_bool].sum().item()
                            )
                            target_decision_negative_pred_count += int(
                                pred_positive[negative_bool].sum().item()
                            )
                            target_decision_positive_prob_sum += float(
                                decision_prob[positive_bool].sum().item()
                            )
                            target_decision_negative_prob_sum += float(
                                decision_prob[negative_bool].sum().item()
                            )
                    target_aux_weight = float(cfg.bc_signal_target_aux_weight)
                    if target_aux_weight > 0.0 and hasattr(model, "signal_target_aux"):
                        target_aux_mask = target_aux_mask_seq[t]
                        target_aux_xy = target_aux_xy_seq[t]
                        target_aux_pred = model.signal_target_aux(hidden[0])
                        target_aux_loss = _signal_target_aux_loss(
                            target_aux_pred,
                            target_aux_mask,
                            target_aux_xy,
                            sample_weight=sample_weight,
                        )
                        loss = loss + target_aux_weight * target_aux_loss
                        target_aux_loss_sum += float(target_aux_loss.item())
                        target_aux_steps += 1
                        with torch.no_grad():
                            target_aux_bool = target_aux_mask > 0.0
                            target_aux_label_count = int(target_aux_bool.sum().item())
                            target_aux_count += target_aux_label_count
                            present_pred = torch.sigmoid(target_aux_pred[:, 0]) > 0.5
                            target_aux_present_correct += int(
                                (present_pred == target_aux_bool).sum().item()
                            )
                            target_aux_present_total += int(target_aux_bool.numel())
                            if target_aux_label_count > 0:
                                pred_xy = torch.sigmoid(target_aux_pred[:, 1:3])
                                target_aux_xy_l1_sum += float(
                                    torch.abs(pred_xy - target_aux_xy)
                                    .mean(dim=-1)[target_aux_bool]
                                    .sum()
                                    .item()
                                )
                    decoy_drift_action_mask = decoy_drift_action_mask_seq[t]
                    decoy_drift_action_id = decoy_drift_action_id_seq[t]
                    decoy_drift_action_weight = float(cfg.bc_signal_decoy_drift_action_loss_weight)
                    if decoy_drift_action_weight > 0.0:
                        decoy_drift_loss = _signal_decoy_drift_action_loss(
                            logits,
                            decoy_drift_action_id,
                            decoy_drift_action_mask,
                            sample_weight=sample_weight,
                        )
                        loss = loss + decoy_drift_action_weight * decoy_drift_loss
                        decoy_drift_action_loss_sum += float(decoy_drift_loss.item())
                        decoy_drift_action_loss_steps += 1
                    with torch.no_grad():
                        decoy_drift_bool = decoy_drift_action_mask > 0.0
                        decoy_drift_count = int(decoy_drift_bool.sum().item())
                        if decoy_drift_count > 0:
                            decoy_drift_action_count += decoy_drift_count
                            bad_action_ids = decoy_drift_action_id.clamp(0, logits.shape[-1] - 1)
                            decoy_drift_pred_bad_action_count += int(
                                (logits.argmax(dim=-1) == bad_action_ids)[decoy_drift_bool].sum().item()
                            )
                            decoy_drift_bad_action_prob_sum += float(
                                torch.softmax(logits, dim=-1)
                                .gather(1, bad_action_ids.unsqueeze(1))
                                .squeeze(1)[decoy_drift_bool]
                                .sum()
                                .item()
                            )
                    decoy_scan_action_mask = decoy_scan_action_mask_seq[t]
                    decoy_scan_action_id = decoy_scan_action_id_seq[t]
                    decoy_scan_action_weight = float(cfg.bc_signal_decoy_scan_action_loss_weight)
                    if decoy_scan_action_weight > 0.0:
                        decoy_scan_loss = _signal_decoy_drift_action_loss(
                            logits,
                            decoy_scan_action_id,
                            decoy_scan_action_mask,
                            sample_weight=sample_weight,
                        )
                        loss = loss + decoy_scan_action_weight * decoy_scan_loss
                        decoy_scan_action_loss_sum += float(decoy_scan_loss.item())
                        decoy_scan_action_loss_steps += 1
                    with torch.no_grad():
                        decoy_scan_bool = decoy_scan_action_mask > 0.0
                        decoy_scan_count = int(decoy_scan_bool.sum().item())
                        if decoy_scan_count > 0:
                            decoy_scan_action_count += decoy_scan_count
                            bad_action_ids = decoy_scan_action_id.clamp(0, logits.shape[-1] - 1)
                            decoy_scan_pred_bad_action_count += int(
                                (logits.argmax(dim=-1) == bad_action_ids)[decoy_scan_bool].sum().item()
                            )
                            decoy_scan_bad_action_prob_sum += float(
                                torch.softmax(logits, dim=-1)
                                .gather(1, bad_action_ids.unsqueeze(1))
                                .squeeze(1)[decoy_scan_bool]
                                .sum()
                                .item()
                            )
                    positive_scan_mask = target_scan_action_mask * (
                        (target_scan_kind_ids == _SIGNAL_TARGET_SCAN_KIND_FIRST)
                        | (target_scan_kind_ids == _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION)
                        | (target_scan_kind_ids == _SIGNAL_TARGET_SCAN_KIND_REFRESH)
                    ).float()
                    opportunity_positive_scan_mask = target_opportunity_action_mask * (
                        (target_opportunity_kind_ids == _SIGNAL_TARGET_SCAN_KIND_FIRST)
                        | (target_opportunity_kind_ids == _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION)
                        | (target_opportunity_kind_ids == _SIGNAL_TARGET_SCAN_KIND_REFRESH)
                    ).float()
                    if float(cfg.bc_signal_target_opportunity_action_weight) > 0.0:
                        positive_scan_mask = torch.maximum(
                            positive_scan_mask,
                            opportunity_positive_scan_mask,
                        )
                    redundant_active_scan_mask = target_scan_action_mask * (
                        target_scan_kind_ids == _SIGNAL_TARGET_SCAN_KIND_REDUNDANT_ACTIVE
                    ).float()
                    negative_scan_mask = torch.maximum(
                        torch.maximum(decoy_scan_action_mask, redundant_active_scan_mask),
                        torch.maximum(rejected_target_mask, bad_redundant_target_mask),
                    )
                    scan_decision_weight = float(cfg.bc_signal_scan_decision_loss_weight)
                    if scan_decision_weight > 0.0:
                        scan_decision_loss = _signal_scan_decision_loss(
                            logits,
                            positive_scan_mask,
                            negative_scan_mask,
                            positive_weight=cfg.bc_signal_scan_decision_pos_weight,
                            negative_weight=cfg.bc_signal_scan_decision_neg_weight,
                            sample_weight=sample_weight,
                        )
                        loss = loss + scan_decision_weight * scan_decision_loss
                        scan_decision_loss_sum += float(scan_decision_loss.item())
                        scan_decision_loss_steps += 1
                        with torch.no_grad():
                            positive_bool = positive_scan_mask > 0.0
                            negative_bool = (negative_scan_mask > 0.0) & ~positive_bool
                            pred_interact = logits.argmax(dim=-1) == SyncOrSinkEnv.ACTION_INTERACT
                            scan_decision_positive_count += int(positive_bool.sum().item())
                            scan_decision_negative_count += int(negative_bool.sum().item())
                            scan_decision_positive_pred_count += int(
                                pred_interact[positive_bool].sum().item()
                            )
                            scan_decision_negative_pred_count += int(
                                pred_interact[negative_bool].sum().item()
                            )
                    scan_gate_weight = float(cfg.bc_signal_scan_gate_loss_weight)
                    if scan_gate_weight > 0.0 and hasattr(model, "signal_scan_gate"):
                        scan_gate_logits = model.signal_scan_gate(hidden[0])
                        scan_gate_loss = _signal_scan_gate_loss(
                            scan_gate_logits,
                            positive_scan_mask,
                            negative_scan_mask,
                            positive_weight=cfg.bc_signal_scan_gate_pos_weight,
                            negative_weight=cfg.bc_signal_scan_gate_neg_weight,
                            sample_weight=sample_weight,
                        )
                        loss = loss + scan_gate_weight * scan_gate_loss
                        scan_gate_loss_sum += float(scan_gate_loss.item())
                        scan_gate_loss_steps += 1
                        with torch.no_grad():
                            positive_bool = positive_scan_mask > 0.0
                            negative_bool = (negative_scan_mask > 0.0) & ~positive_bool
                            gate_prob = torch.sigmoid(scan_gate_logits.reshape(-1))
                            pred_scan = gate_prob >= 0.5
                            scan_gate_positive_count += int(positive_bool.sum().item())
                            scan_gate_negative_count += int(negative_bool.sum().item())
                            scan_gate_positive_pred_count += int(pred_scan[positive_bool].sum().item())
                            scan_gate_negative_pred_count += int(pred_scan[negative_bool].sum().item())
                            scan_gate_positive_prob_sum += float(gate_prob[positive_bool].sum().item())
                            scan_gate_negative_prob_sum += float(gate_prob[negative_bool].sum().item())
                    rejected_target_drift_action_mask = rejected_target_drift_action_mask_seq[t]
                    rejected_target_drift_action_id = rejected_target_drift_action_id_seq[t]
                    rejected_target_drift_action_weight = float(
                        cfg.bc_signal_rejected_target_drift_action_loss_weight
                    )
                    if rejected_target_drift_action_weight > 0.0:
                        rejected_target_drift_loss = _signal_decoy_drift_action_loss(
                            logits,
                            rejected_target_drift_action_id,
                            rejected_target_drift_action_mask,
                            sample_weight=sample_weight,
                        )
                        loss = loss + rejected_target_drift_action_weight * rejected_target_drift_loss
                        rejected_target_drift_action_loss_sum += float(rejected_target_drift_loss.item())
                        rejected_target_drift_action_loss_steps += 1
                    with torch.no_grad():
                        rejected_target_drift_bool = rejected_target_drift_action_mask > 0.0
                        rejected_target_drift_count = int(rejected_target_drift_bool.sum().item())
                        if rejected_target_drift_count > 0:
                            rejected_target_drift_action_count += rejected_target_drift_count
                            bad_action_ids = rejected_target_drift_action_id.clamp(0, logits.shape[-1] - 1)
                            rejected_target_drift_pred_bad_action_count += int(
                                (logits.argmax(dim=-1) == bad_action_ids)[rejected_target_drift_bool]
                                .sum()
                                .item()
                            )
                            rejected_target_drift_bad_action_prob_sum += float(
                                torch.softmax(logits, dim=-1)
                                .gather(1, bad_action_ids.unsqueeze(1))
                                .squeeze(1)[rejected_target_drift_bool]
                                .sum()
                                .item()
                            )
                    if cfg.comm and cfg.bc_comm_loss_weight > 0:
                        comm_components = _recurrent_comm_loss_components(
                            send_logits,
                            token_logits,
                            len_logits,
                            msg_seq[t],
                            msg_len_seq[t],
                            send_pos_weight=send_pos_weight,
                            sample_weight=sample_weight,
                            send_loss_weight=cfg.bc_comm_send_loss_weight,
                            length_loss_weight=cfg.bc_comm_length_loss_weight,
                            token_loss_weight=cfg.bc_comm_token_loss_weight,
                            send_rate_penalty_weight=cfg.bc_comm_send_rate_penalty_weight,
                            send_rate_target=cfg.bc_comm_send_rate_target,
                        )
                        comm_loss = comm_components["total"]
                        loss = loss + cfg.bc_comm_loss_weight * comm_loss
                        total_comm_loss += comm_loss.item()
                        comm_loss_steps += 1
                        comm_send_loss_sum += float(comm_components["send"].item())
                        comm_len_loss_sum += float(comm_components["length"].item())
                        comm_token_loss_sum += float(comm_components["token"].item())
                        comm_send_rate_loss_sum += float(comm_components["send_rate"].item())
                        with torch.no_grad():
                            send_target = msg_len_seq[t] > 0
                            send_prob = torch.sigmoid(send_logits.squeeze(-1))
                            pred_send = send_prob > float(cfg.eval_send_threshold)
                            pred_len = len_logits.argmax(dim=-1)
                            pred_positive_len = pred_len > 0
                            comm_agents += int(send_target.numel())
                            comm_send_label_count += int(send_target.sum().item())
                            comm_pred_send_count += int(pred_send.sum().item())
                            comm_send_prob_sum += float(send_prob.sum().item())
                            comm_pred_positive_len_count += int(pred_positive_len.sum().item())
                            comm_pred_positive_len_on_label_positive_count += int(
                                (pred_positive_len & send_target).sum().item()
                            )
                            comm_label_positive_count += int(send_target.sum().item())
                            comm_true_len_sum += float(msg_len_seq[t].sum().item())
                            comm_pred_len_sum += float(pred_len.sum().item())
                    chunk_loss += loss
                    chunk_correct += (logits.argmax(dim=-1) == act_seq[t]).sum().item()
                    chunk_count += N

                chunk_loss = chunk_loss / (t_end - t_start)
                if cfg.bc_equal_episode_weight:
                    episode_chunk_losses.append(chunk_loss)
                else:
                    optimizer.zero_grad()
                    (chunk_loss * episode_weight).backward()
                    nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    total_loss += chunk_loss.item() * episode_weight
                    loss_den += episode_weight

                total_correct += chunk_correct
                total_count += chunk_count
                chunks += 1

            if cfg.bc_equal_episode_weight and episode_chunk_losses:
                episode_loss = torch.stack(episode_chunk_losses).mean()
                optimizer.zero_grad()
                (episode_loss * episode_weight).backward()
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                total_loss += episode_loss.item() * episode_weight
                loss_den += episode_weight

        acc = total_correct / total_count if total_count > 0 else 0
        avg_comm = total_comm_loss / chunks if cfg.comm and chunks > 0 else 0.0
        comm_loss_den = max(comm_loss_steps, 1)
        comm_agent_den = max(comm_agents, 1)
        comm_label_pos_den = max(comm_label_positive_count, 1)
        comm_pred_send_rate = comm_pred_send_count / comm_agent_den if cfg.comm else 0.0
        comm_send_label_rate = comm_send_label_count / comm_agent_den if cfg.comm else 0.0
        comm_pred_positive_len_rate = comm_pred_positive_len_count / comm_agent_den if cfg.comm else 0.0
        comm_pred_positive_len_on_label_positive_rate = (
            comm_pred_positive_len_on_label_positive_count / comm_label_pos_den
            if cfg.comm and comm_label_positive_count > 0
            else 0.0
        )
        comm_mean_send_prob = comm_send_prob_sum / comm_agent_den if cfg.comm else 0.0
        comm_mean_true_len = comm_true_len_sum / comm_agent_den if cfg.comm else 0.0
        comm_mean_pred_len = comm_pred_len_sum / comm_agent_den if cfg.comm else 0.0
        rejected_target_den = max(rejected_target_count, 1)
        rejected_target_pred_interact_rate = rejected_target_pred_interact_count / rejected_target_den
        rejected_target_mean_interact_prob = rejected_target_interact_prob_sum / rejected_target_den
        rejected_target_loss_mean = rejected_target_loss_sum / max(rejected_target_loss_steps, 1)
        rejected_target_action_loss_mean = (
            rejected_target_action_loss_sum / max(rejected_target_action_loss_steps, 1)
        )
        bad_redundant_target_den = max(bad_redundant_target_count, 1)
        bad_redundant_target_pred_interact_rate = (
            bad_redundant_target_pred_interact_count / bad_redundant_target_den
        )
        bad_redundant_target_mean_interact_prob = (
            bad_redundant_target_interact_prob_sum / bad_redundant_target_den
        )
        bad_redundant_target_loss_mean = (
            bad_redundant_target_loss_sum / max(bad_redundant_target_loss_steps, 1)
        )
        target_scan_den = max(target_scan_label_count, 1)
        target_scan_pred_interact_rate = target_scan_pred_interact_count / target_scan_den
        target_scan_mean_interact_prob = target_scan_interact_prob_sum / target_scan_den
        target_scan_kind_rates = {}
        target_scan_kind_probs = {}
        for kind_idx, kind_name in enumerate(_SIGNAL_TARGET_SCAN_KIND_NAMES):
            kind_den = max(target_scan_kind_counts[kind_idx], 1)
            target_scan_kind_rates[kind_name] = (
                target_scan_kind_pred_interact_counts[kind_idx] / kind_den
            )
            target_scan_kind_probs[kind_name] = (
                target_scan_kind_interact_prob_sums[kind_idx] / kind_den
            )
        target_scan_action_loss_mean = (
            target_scan_action_loss_sum / max(target_scan_action_loss_steps, 1)
        )
        target_opportunity_den = max(target_opportunity_label_count, 1)
        target_opportunity_pred_interact_rate = (
            target_opportunity_pred_interact_count / target_opportunity_den
        )
        target_opportunity_mean_interact_prob = (
            target_opportunity_interact_prob_sum / target_opportunity_den
        )
        target_opportunity_kind_rates = {}
        target_opportunity_kind_probs = {}
        for kind_idx, kind_name in enumerate(_SIGNAL_TARGET_SCAN_KIND_NAMES):
            kind_den = max(target_opportunity_kind_counts[kind_idx], 1)
            target_opportunity_kind_rates[kind_name] = (
                target_opportunity_kind_pred_interact_counts[kind_idx] / kind_den
            )
            target_opportunity_kind_probs[kind_name] = (
                target_opportunity_kind_interact_prob_sums[kind_idx] / kind_den
            )
        target_opportunity_action_loss_mean = (
            target_opportunity_action_loss_sum / max(target_opportunity_action_loss_steps, 1)
        )
        redundant_wait_action_den = max(redundant_wait_action_count, 1)
        redundant_wait_pred_action_rate = (
            redundant_wait_pred_action_count / redundant_wait_action_den
        )
        redundant_wait_mean_action_prob = (
            redundant_wait_action_prob_sum / redundant_wait_action_den
        )
        redundant_wait_action_loss_mean = (
            redundant_wait_action_loss_sum / max(redundant_wait_action_loss_steps, 1)
        )
        target_pursuit_action_den = max(target_pursuit_action_count, 1)
        target_pursuit_pred_action_rate = (
            target_pursuit_pred_action_count / target_pursuit_action_den
        )
        target_pursuit_mean_action_prob = (
            target_pursuit_action_prob_sum / target_pursuit_action_den
        )
        target_pursuit_action_loss_mean = (
            target_pursuit_action_loss_sum / max(target_pursuit_action_loss_steps, 1)
        )
        sync_response_action_den = max(sync_response_action_count, 1)
        sync_response_pred_action_rate = (
            sync_response_pred_action_count / sync_response_action_den
        )
        sync_response_mean_action_prob = (
            sync_response_action_prob_sum / sync_response_action_den
        )
        sync_response_action_loss_mean = (
            sync_response_action_loss_sum / max(sync_response_action_loss_steps, 1)
        )
        scan_decision_loss_mean = scan_decision_loss_sum / max(scan_decision_loss_steps, 1)
        scan_decision_positive_rate = (
            scan_decision_positive_pred_count / max(scan_decision_positive_count, 1)
        )
        scan_decision_negative_rate = (
            scan_decision_negative_pred_count / max(scan_decision_negative_count, 1)
        )
        scan_gate_loss_mean = scan_gate_loss_sum / max(scan_gate_loss_steps, 1)
        scan_gate_positive_rate = scan_gate_positive_pred_count / max(scan_gate_positive_count, 1)
        scan_gate_negative_rate = scan_gate_negative_pred_count / max(scan_gate_negative_count, 1)
        scan_gate_positive_mean_prob = scan_gate_positive_prob_sum / max(scan_gate_positive_count, 1)
        scan_gate_negative_mean_prob = scan_gate_negative_prob_sum / max(scan_gate_negative_count, 1)
        target_match_action_den = max(target_match_action_count, 1)
        target_match_pred_action_rate = target_match_pred_action_count / target_match_action_den
        target_match_mean_action_prob = target_match_action_prob_sum / target_match_action_den
        target_match_action_loss_mean = (
            target_match_action_loss_sum / max(target_match_action_loss_steps, 1)
        )
        target_validity_loss_mean = target_validity_loss_sum / max(target_validity_loss_steps, 1)
        target_validity_positive_pred_rate = (
            target_validity_positive_pred_count / max(target_validity_positive_count, 1)
        )
        target_validity_negative_pred_rate = (
            target_validity_negative_pred_count / max(target_validity_negative_count, 1)
        )
        target_validity_positive_mean_prob = (
            target_validity_positive_prob_sum / max(target_validity_positive_count, 1)
        )
        target_validity_negative_mean_prob = (
            target_validity_negative_prob_sum / max(target_validity_negative_count, 1)
        )
        target_decision_loss_mean = target_decision_loss_sum / max(target_decision_loss_steps, 1)
        target_decision_positive_pred_rate = (
            target_decision_positive_pred_count / max(target_decision_positive_count, 1)
        )
        target_decision_negative_pred_rate = (
            target_decision_negative_pred_count / max(target_decision_negative_count, 1)
        )
        target_decision_positive_mean_prob = (
            target_decision_positive_prob_sum / max(target_decision_positive_count, 1)
        )
        target_decision_negative_mean_prob = (
            target_decision_negative_prob_sum / max(target_decision_negative_count, 1)
        )
        target_aux_loss_mean = target_aux_loss_sum / max(target_aux_steps, 1)
        target_aux_present_acc = target_aux_present_correct / max(target_aux_present_total, 1)
        target_aux_xy_l1 = target_aux_xy_l1_sum / max(target_aux_count, 1)
        decoy_drift_action_den = max(decoy_drift_action_count, 1)
        decoy_drift_pred_bad_action_rate = (
            decoy_drift_pred_bad_action_count / decoy_drift_action_den
        )
        decoy_drift_mean_bad_action_prob = (
            decoy_drift_bad_action_prob_sum / decoy_drift_action_den
        )
        decoy_drift_action_loss_mean = (
            decoy_drift_action_loss_sum / max(decoy_drift_action_loss_steps, 1)
        )
        decoy_scan_action_den = max(decoy_scan_action_count, 1)
        decoy_scan_pred_bad_action_rate = (
            decoy_scan_pred_bad_action_count / decoy_scan_action_den
        )
        decoy_scan_mean_bad_action_prob = (
            decoy_scan_bad_action_prob_sum / decoy_scan_action_den
        )
        decoy_scan_action_loss_mean = (
            decoy_scan_action_loss_sum / max(decoy_scan_action_loss_steps, 1)
        )
        rejected_target_drift_action_den = max(rejected_target_drift_action_count, 1)
        rejected_target_drift_pred_bad_action_rate = (
            rejected_target_drift_pred_bad_action_count / rejected_target_drift_action_den
        )
        rejected_target_drift_mean_bad_action_prob = (
            rejected_target_drift_bad_action_prob_sum / rejected_target_drift_action_den
        )
        rejected_target_drift_action_loss_mean = (
            rejected_target_drift_action_loss_sum / max(rejected_target_drift_action_loss_steps, 1)
        )
        print(
            f"[BC] epoch {epoch:3d} | loss {total_loss / max(loss_den, 1e-8):.4f} | "
            f"comm {avg_comm:.4f} | acc {acc:.3f} | "
            f"send {comm_send_label_rate:.3f}->{comm_pred_send_rate:.3f} | "
            f"len+ {comm_pred_positive_len_rate:.3f} | "
            f"rej_int {rejected_target_pred_interact_rate:.3f} | "
            f"rej_act_loss {rejected_target_action_loss_mean:.3f} | "
            f"bad_red_int {bad_redundant_target_pred_interact_rate:.3f} | "
            f"scan_int {target_scan_pred_interact_rate:.3f} | "
            f"opp_int {target_opportunity_pred_interact_rate:.3f} | "
            f"wait_act {redundant_wait_pred_action_rate:.3f} | "
            f"scan_loss {target_scan_action_loss_mean:.3f} | "
            f"scan_dec {scan_decision_positive_rate:.3f}/{scan_decision_negative_rate:.3f} | "
            f"scan_gate {scan_gate_positive_rate:.3f}/{scan_gate_negative_rate:.3f} | "
            f"pursuit_act {target_pursuit_pred_action_rate:.3f} | "
            f"sync_act {sync_response_pred_action_rate:.3f} | "
            f"match_act {target_match_pred_action_rate:.3f} | "
            f"valid {target_validity_positive_pred_rate:.3f}/{target_validity_negative_pred_rate:.3f} | "
            f"decision {target_decision_positive_pred_rate:.3f}/{target_decision_negative_pred_rate:.3f} | "
            f"target_aux {target_aux_xy_l1:.3f} | "
            f"decoy_bad {decoy_drift_pred_bad_action_rate:.3f} | "
            f"decoy_scan_bad {decoy_scan_pred_bad_action_rate:.3f} | "
            f"rej_drift_bad {rejected_target_drift_pred_bad_action_rate:.3f}"
        )
        if wandb_run is not None:
            send_pos_weight_value = (
                None if send_pos_weight is None else float(send_pos_weight.detach().cpu().item())
            )
            _wandb_log(
                wandb_run,
                {
                    f"{log_prefix}/epoch": epoch,
                    f"{log_prefix}/loss": float(total_loss / max(loss_den, 1e-8)),
                    f"{log_prefix}/comm_loss": float(avg_comm),
                    f"{log_prefix}/comm_loss_per_step": float(total_comm_loss / comm_loss_den),
                    f"{log_prefix}/comm_send_loss": float(comm_send_loss_sum / comm_loss_den),
                    f"{log_prefix}/comm_length_loss": float(comm_len_loss_sum / comm_loss_den),
                    f"{log_prefix}/comm_token_loss": float(comm_token_loss_sum / comm_loss_den),
                    f"{log_prefix}/comm_send_rate_loss": float(comm_send_rate_loss_sum / comm_loss_den),
                    f"{log_prefix}/comm_send_label_rate": float(comm_send_label_rate),
                    f"{log_prefix}/comm_pred_send_rate": float(comm_pred_send_rate),
                    f"{log_prefix}/comm_mean_send_prob": float(comm_mean_send_prob),
                    f"{log_prefix}/comm_pred_positive_len_rate": float(comm_pred_positive_len_rate),
                    f"{log_prefix}/comm_pred_positive_len_on_label_positive_rate": float(
                        comm_pred_positive_len_on_label_positive_rate
                    ),
                    f"{log_prefix}/comm_mean_true_len": float(comm_mean_true_len),
                    f"{log_prefix}/comm_mean_pred_len": float(comm_mean_pred_len),
                    f"{log_prefix}/comm_send_pos_weight": float(send_pos_weight_value or 0.0),
                    f"{log_prefix}/comm_send_loss_weight": float(cfg.bc_comm_send_loss_weight),
                    f"{log_prefix}/comm_length_loss_weight": float(cfg.bc_comm_length_loss_weight),
                    f"{log_prefix}/comm_token_loss_weight": float(cfg.bc_comm_token_loss_weight),
                    f"{log_prefix}/comm_send_rate_penalty_weight": float(
                        cfg.bc_comm_send_rate_penalty_weight
                    ),
                    f"{log_prefix}/comm_send_rate_target": float(cfg.bc_comm_send_rate_target),
                    f"{log_prefix}/signal_rejected_target_count": int(rejected_target_count),
                    f"{log_prefix}/signal_rejected_target_interact_loss": float(rejected_target_loss_mean),
                    f"{log_prefix}/signal_rejected_target_interact_action_loss": float(
                        rejected_target_action_loss_mean
                    ),
                    f"{log_prefix}/signal_rejected_target_pred_interact_rate": float(
                        rejected_target_pred_interact_rate
                    ),
                    f"{log_prefix}/signal_rejected_target_mean_interact_prob": float(
                        rejected_target_mean_interact_prob
                    ),
                    f"{log_prefix}/signal_rejected_target_interact_loss_weight": float(
                        cfg.bc_signal_rejected_target_interact_loss_weight
                    ),
                    f"{log_prefix}/signal_rejected_target_interact_action_loss_weight": float(
                        cfg.bc_signal_rejected_target_interact_action_loss_weight
                    ),
                    f"{log_prefix}/signal_bad_redundant_target_count": int(bad_redundant_target_count),
                    f"{log_prefix}/signal_bad_redundant_target_interact_loss": float(
                        bad_redundant_target_loss_mean
                    ),
                    f"{log_prefix}/signal_bad_redundant_target_pred_interact_rate": float(
                        bad_redundant_target_pred_interact_rate
                    ),
                    f"{log_prefix}/signal_bad_redundant_target_mean_interact_prob": float(
                        bad_redundant_target_mean_interact_prob
                    ),
                    f"{log_prefix}/signal_bad_redundant_target_interact_loss_weight": float(
                        cfg.bc_signal_bad_redundant_target_interact_loss_weight
                    ),
                    f"{log_prefix}/signal_target_scan_label_count": int(target_scan_label_count),
                    f"{log_prefix}/signal_target_scan_pred_interact_rate": float(
                        target_scan_pred_interact_rate
                    ),
                    f"{log_prefix}/signal_target_scan_mean_interact_prob": float(
                        target_scan_mean_interact_prob
                    ),
                    f"{log_prefix}/signal_target_scan_action_loss": float(
                        target_scan_action_loss_mean
                    ),
                    f"{log_prefix}/signal_first_target_scan_action_weight": float(
                        cfg.bc_signal_first_target_scan_action_weight
                    ),
                    f"{log_prefix}/signal_refresh_target_scan_action_weight": float(
                        cfg.bc_signal_refresh_target_scan_action_weight
                    ),
                    f"{log_prefix}/signal_joint_target_scan_action_weight": float(
                        cfg.bc_signal_joint_target_scan_action_weight
                    ),
                    f"{log_prefix}/signal_target_opportunity_label_count": int(
                        target_opportunity_label_count
                    ),
                    f"{log_prefix}/signal_target_opportunity_pred_interact_rate": float(
                        target_opportunity_pred_interact_rate
                    ),
                    f"{log_prefix}/signal_target_opportunity_mean_interact_prob": float(
                        target_opportunity_mean_interact_prob
                    ),
                    f"{log_prefix}/signal_target_opportunity_action_loss": float(
                        target_opportunity_action_loss_mean
                    ),
                    f"{log_prefix}/signal_target_opportunity_action_weight": float(
                        cfg.bc_signal_target_opportunity_action_weight
                    ),
                    f"{log_prefix}/signal_redundant_target_wait_action_count": int(
                        redundant_wait_action_count
                    ),
                    f"{log_prefix}/signal_redundant_target_wait_action_loss": float(
                        redundant_wait_action_loss_mean
                    ),
                    f"{log_prefix}/signal_redundant_target_wait_pred_action_rate": float(
                        redundant_wait_pred_action_rate
                    ),
                    f"{log_prefix}/signal_redundant_target_wait_mean_action_prob": float(
                        redundant_wait_mean_action_prob
                    ),
                    f"{log_prefix}/signal_redundant_target_wait_action_loss_weight": float(
                        cfg.bc_signal_redundant_target_wait_action_loss_weight
                    ),
                    f"{log_prefix}/signal_scan_decision_loss": float(scan_decision_loss_mean),
                    f"{log_prefix}/signal_scan_decision_positive_count": int(
                        scan_decision_positive_count
                    ),
                    f"{log_prefix}/signal_scan_decision_negative_count": int(
                        scan_decision_negative_count
                    ),
                    f"{log_prefix}/signal_scan_decision_positive_pred_interact_rate": float(
                        scan_decision_positive_rate
                    ),
                    f"{log_prefix}/signal_scan_decision_negative_pred_interact_rate": float(
                        scan_decision_negative_rate
                    ),
                    f"{log_prefix}/signal_scan_decision_loss_weight": float(
                        cfg.bc_signal_scan_decision_loss_weight
                    ),
                    f"{log_prefix}/signal_scan_decision_pos_weight": float(
                        cfg.bc_signal_scan_decision_pos_weight
                    ),
                    f"{log_prefix}/signal_scan_decision_neg_weight": float(
                        cfg.bc_signal_scan_decision_neg_weight
                    ),
                    f"{log_prefix}/signal_scan_gate_loss": float(scan_gate_loss_mean),
                    f"{log_prefix}/signal_scan_gate_positive_count": int(scan_gate_positive_count),
                    f"{log_prefix}/signal_scan_gate_negative_count": int(scan_gate_negative_count),
                    f"{log_prefix}/signal_scan_gate_positive_pred_rate": float(
                        scan_gate_positive_rate
                    ),
                    f"{log_prefix}/signal_scan_gate_negative_pred_rate": float(
                        scan_gate_negative_rate
                    ),
                    f"{log_prefix}/signal_scan_gate_positive_mean_prob": float(
                        scan_gate_positive_mean_prob
                    ),
                    f"{log_prefix}/signal_scan_gate_negative_mean_prob": float(
                        scan_gate_negative_mean_prob
                    ),
                    f"{log_prefix}/signal_scan_gate_loss_weight": float(
                        cfg.bc_signal_scan_gate_loss_weight
                    ),
                    f"{log_prefix}/signal_scan_gate_pos_weight": float(
                        cfg.bc_signal_scan_gate_pos_weight
                    ),
                    f"{log_prefix}/signal_scan_gate_neg_weight": float(
                        cfg.bc_signal_scan_gate_neg_weight
                    ),
                    **{
                        f"{log_prefix}/signal_target_scan_{kind_name}_count": int(
                            target_scan_kind_counts[kind_idx]
                        )
                        for kind_idx, kind_name in enumerate(_SIGNAL_TARGET_SCAN_KIND_NAMES)
                    },
                    **{
                        f"{log_prefix}/signal_target_scan_{kind_name}_pred_interact_rate": float(
                            target_scan_kind_rates[kind_name]
                        )
                        for kind_name in _SIGNAL_TARGET_SCAN_KIND_NAMES
                    },
                    **{
                        f"{log_prefix}/signal_target_scan_{kind_name}_mean_interact_prob": float(
                            target_scan_kind_probs[kind_name]
                        )
                        for kind_name in _SIGNAL_TARGET_SCAN_KIND_NAMES
                    },
                    **{
                        f"{log_prefix}/signal_target_opportunity_{kind_name}_count": int(
                            target_opportunity_kind_counts[kind_idx]
                        )
                        for kind_idx, kind_name in enumerate(_SIGNAL_TARGET_SCAN_KIND_NAMES)
                    },
                    **{
                        f"{log_prefix}/signal_target_opportunity_{kind_name}_pred_interact_rate": float(
                            target_opportunity_kind_rates[kind_name]
                        )
                        for kind_name in _SIGNAL_TARGET_SCAN_KIND_NAMES
                    },
                    **{
                        f"{log_prefix}/signal_target_opportunity_{kind_name}_mean_interact_prob": float(
                            target_opportunity_kind_probs[kind_name]
                        )
                        for kind_name in _SIGNAL_TARGET_SCAN_KIND_NAMES
                    },
                    f"{log_prefix}/signal_target_pursuit_action_count": int(
                        target_pursuit_action_count
                    ),
                    f"{log_prefix}/signal_target_pursuit_action_loss": float(
                        target_pursuit_action_loss_mean
                    ),
                    f"{log_prefix}/signal_target_pursuit_pred_action_rate": float(
                        target_pursuit_pred_action_rate
                    ),
                    f"{log_prefix}/signal_target_pursuit_mean_action_prob": float(
                        target_pursuit_mean_action_prob
                    ),
                    f"{log_prefix}/signal_target_pursuit_action_weight": float(
                        cfg.bc_signal_target_pursuit_action_weight
                    ),
                    f"{log_prefix}/signal_sync_response_action_count": int(
                        sync_response_action_count
                    ),
                    f"{log_prefix}/signal_sync_response_action_loss": float(
                        sync_response_action_loss_mean
                    ),
                    f"{log_prefix}/signal_sync_response_pred_action_rate": float(
                        sync_response_pred_action_rate
                    ),
                    f"{log_prefix}/signal_sync_response_mean_action_prob": float(
                        sync_response_mean_action_prob
                    ),
                    f"{log_prefix}/signal_sync_response_action_loss_weight": float(
                        cfg.bc_signal_sync_response_action_loss_weight
                    ),
                    f"{log_prefix}/signal_target_match_action_count": int(target_match_action_count),
                    f"{log_prefix}/signal_target_match_action_loss": float(target_match_action_loss_mean),
                    f"{log_prefix}/signal_target_match_pred_action_rate": float(
                        target_match_pred_action_rate
                    ),
                    f"{log_prefix}/signal_target_match_mean_action_prob": float(
                        target_match_mean_action_prob
                    ),
                    f"{log_prefix}/signal_target_match_action_weight": float(
                        cfg.bc_signal_target_match_action_weight
                    ),
                    f"{log_prefix}/signal_target_validity_loss": float(target_validity_loss_mean),
                    f"{log_prefix}/signal_target_validity_positive_count": int(
                        target_validity_positive_count
                    ),
                    f"{log_prefix}/signal_target_validity_negative_count": int(
                        target_validity_negative_count
                    ),
                    f"{log_prefix}/signal_target_validity_positive_pred_rate": float(
                        target_validity_positive_pred_rate
                    ),
                    f"{log_prefix}/signal_target_validity_negative_pred_rate": float(
                        target_validity_negative_pred_rate
                    ),
                    f"{log_prefix}/signal_target_validity_positive_mean_prob": float(
                        target_validity_positive_mean_prob
                    ),
                    f"{log_prefix}/signal_target_validity_negative_mean_prob": float(
                        target_validity_negative_mean_prob
                    ),
                    f"{log_prefix}/signal_target_validity_loss_weight": float(
                        cfg.bc_signal_target_validity_loss_weight
                    ),
                    f"{log_prefix}/signal_target_validity_pos_weight": float(
                        cfg.bc_signal_target_validity_pos_weight
                    ),
                    f"{log_prefix}/signal_target_validity_neg_weight": float(
                        cfg.bc_signal_target_validity_neg_weight
                    ),
                    f"{log_prefix}/signal_target_decision_loss": float(target_decision_loss_mean),
                    f"{log_prefix}/signal_target_decision_positive_count": int(
                        target_decision_positive_count
                    ),
                    f"{log_prefix}/signal_target_decision_negative_count": int(
                        target_decision_negative_count
                    ),
                    f"{log_prefix}/signal_target_decision_positive_pred_rate": float(
                        target_decision_positive_pred_rate
                    ),
                    f"{log_prefix}/signal_target_decision_negative_pred_rate": float(
                        target_decision_negative_pred_rate
                    ),
                    f"{log_prefix}/signal_target_decision_positive_mean_prob": float(
                        target_decision_positive_mean_prob
                    ),
                    f"{log_prefix}/signal_target_decision_negative_mean_prob": float(
                        target_decision_negative_mean_prob
                    ),
                    f"{log_prefix}/signal_target_decision_loss_weight": float(
                        cfg.bc_signal_target_decision_loss_weight
                    ),
                    f"{log_prefix}/signal_target_decision_pos_weight": float(
                        cfg.bc_signal_target_decision_pos_weight
                    ),
                    f"{log_prefix}/signal_target_decision_neg_weight": float(
                        cfg.bc_signal_target_decision_neg_weight
                    ),
                    f"{log_prefix}/signal_target_aux_count": int(target_aux_count),
                    f"{log_prefix}/signal_target_aux_loss": float(target_aux_loss_mean),
                    f"{log_prefix}/signal_target_aux_present_acc": float(target_aux_present_acc),
                    f"{log_prefix}/signal_target_aux_xy_l1": float(target_aux_xy_l1),
                    f"{log_prefix}/signal_target_aux_weight": float(
                        cfg.bc_signal_target_aux_weight
                    ),
                    f"{log_prefix}/signal_decoy_drift_action_count": int(decoy_drift_action_count),
                    f"{log_prefix}/signal_decoy_drift_action_loss": float(decoy_drift_action_loss_mean),
                    f"{log_prefix}/signal_decoy_drift_pred_bad_action_rate": float(
                        decoy_drift_pred_bad_action_rate
                    ),
                    f"{log_prefix}/signal_decoy_drift_mean_bad_action_prob": float(
                        decoy_drift_mean_bad_action_prob
                    ),
                    f"{log_prefix}/signal_decoy_drift_action_loss_weight": float(
                        cfg.bc_signal_decoy_drift_action_loss_weight
                    ),
                    f"{log_prefix}/signal_decoy_scan_action_count": int(decoy_scan_action_count),
                    f"{log_prefix}/signal_decoy_scan_action_loss": float(decoy_scan_action_loss_mean),
                    f"{log_prefix}/signal_decoy_scan_pred_bad_action_rate": float(
                        decoy_scan_pred_bad_action_rate
                    ),
                    f"{log_prefix}/signal_decoy_scan_mean_bad_action_prob": float(
                        decoy_scan_mean_bad_action_prob
                    ),
                    f"{log_prefix}/signal_decoy_scan_action_loss_weight": float(
                        cfg.bc_signal_decoy_scan_action_loss_weight
                    ),
                    f"{log_prefix}/signal_rejected_target_drift_action_count": int(
                        rejected_target_drift_action_count
                    ),
                    f"{log_prefix}/signal_rejected_target_drift_action_loss": float(
                        rejected_target_drift_action_loss_mean
                    ),
                    f"{log_prefix}/signal_rejected_target_drift_pred_bad_action_rate": float(
                        rejected_target_drift_pred_bad_action_rate
                    ),
                    f"{log_prefix}/signal_rejected_target_drift_mean_bad_action_prob": float(
                        rejected_target_drift_mean_bad_action_prob
                    ),
                    f"{log_prefix}/signal_rejected_target_drift_action_loss_weight": float(
                        cfg.bc_signal_rejected_target_drift_action_loss_weight
                    ),
                    f"{log_prefix}/action_acc": float(acc),
                    f"{log_prefix}/lr": float(optimizer.param_groups[0]["lr"]),
                    f"{log_prefix}/chunks": int(chunks),
                    f"{log_prefix}/dataset_episodes": int(len(episodes)),
                    f"{log_prefix}/dataset_transitions": int(_episode_count_transitions(episodes)),
                    **dict(log_context or {}),
                },
                context=f"{log_prefix} log",
            )
        if bc_eval_every > 0 and (epoch + 1) % bc_eval_every == 0:
            calibration = {}
            if cfg.comm and cfg.bc_calibrate_send_threshold:
                calibration = _calibrate_recurrent_send_threshold(cfg, model, episodes, device)
            eval_cfg = cfg
            if bc_eval_episodes > 0:
                eval_cfg = replace(cfg, eval_episodes=bc_eval_episodes)
            eval_result = evaluate_recurrent_policy_multi_seed(
                eval_cfg,
                model,
                device,
                seed_count=bc_eval_seed_count,
            )
            eval_score = _recurrent_eval_score(eval_result)
            is_best_epoch = best_epoch_score is None or eval_score > best_epoch_score
            row = {
                "epoch": int(epoch),
                "eval": eval_result,
                "eval_score": eval_score,
                "eval_send_threshold": float(cfg.eval_send_threshold),
                "is_best_epoch": bool(is_best_epoch),
            }
            if calibration:
                row["calibration"] = calibration
            print(json.dumps({f"{log_prefix}_epoch_eval": row}, indent=2, sort_keys=True))
            if is_best_epoch:
                best_epoch_score = eval_score
                best_epoch_state = copy.deepcopy(model.state_dict())
                best_epoch_threshold = float(cfg.eval_send_threshold)
                best_epoch_row = dict(row)
            if wandb_run is not None:
                _wandb_log(
                    wandb_run,
                    {
                        **_recurrent_eval_wandb_payload(
                            eval_result,
                            update=epoch,
                            is_best=is_best_epoch,
                            best_eval=(best_epoch_row or {}).get("eval") if best_epoch_row else None,
                            prefix=f"{log_prefix}/epoch_eval",
                        ),
                        f"{log_prefix}/epoch_eval_epoch": int(epoch),
                        f"{log_prefix}/epoch_eval_send_threshold": float(cfg.eval_send_threshold),
                        f"{log_prefix}/epoch_eval_restore_best_enabled": int(bc_restore_best),
                        **dict(log_context or {}),
                    },
                    context=f"{log_prefix} epoch eval log",
                )

    if bc_restore_best and best_epoch_state is not None:
        model.load_state_dict(best_epoch_state)
        if best_epoch_threshold is not None:
            cfg.eval_send_threshold = float(best_epoch_threshold)
        print(json.dumps({
            f"{log_prefix}_restore_best_epoch": {
                "epoch": (best_epoch_row or {}).get("epoch"),
                "eval_score": (best_epoch_row or {}).get("eval_score"),
                "eval_send_threshold": float(cfg.eval_send_threshold),
            }
        }, indent=2, sort_keys=True))

    if cfg.comm and cfg.bc_calibrate_send_threshold:
        calibration = _calibrate_recurrent_send_threshold(cfg, model, episodes, device)
        if calibration:
            print(
                f"[BC] calibrated send threshold {calibration['old_threshold']:.3f} -> "
                f"{calibration['threshold']:.3f} | label_rate {calibration['label_rate']:.3f} | "
                f"pred_rate {calibration['pred_rate']:.3f}"
            )
            if wandb_run is not None:
                _wandb_log(
                    wandb_run,
                    {
                        f"{log_prefix}/calibrated_send_threshold": float(calibration["threshold"]),
                        f"{log_prefix}/calibrated_send_threshold_old": float(calibration["old_threshold"]),
                        f"{log_prefix}/calibrated_send_label_rate": float(calibration["label_rate"]),
                        f"{log_prefix}/calibrated_send_target_rate": float(calibration["target_rate"]),
                        f"{log_prefix}/calibrated_send_pred_rate": float(calibration["pred_rate"]),
                        f"{log_prefix}/calibrated_send_mean_prob": float(calibration["mean_prob"]),
                        f"{log_prefix}/calibrated_send_max_prob": float(calibration["max_prob"]),
                        **dict(log_context or {}),
                    },
                    context=f"{log_prefix} calibration log",
                )

    return model


def collect_recurrent_dagger_episodes(
    cfg: RecurrentConfig,
    model: MAPPORecurrentActor,
    device,
    *,
    round_idx: int,
) -> tuple[list[dict], dict]:
    """Collect model-visited sequences labeled by the oracle."""
    model.eval()
    episodes = []
    successes = 0
    capped_episodes = 0
    total_steps = 0
    stored_steps = 0
    base_episodes = 0
    replay_episodes = 0
    focus_event_counts: dict[str, int] = {}
    non_focus_event_counts: dict[str, int] = {}
    focused_state_updates = 0
    solo_target_team_state_updates = 0
    recovery_state_updates = 0
    replay_trigger_counts: dict[str, int] = {}
    deferred_solo_target_team_records = 0
    dropped_solo_target_team_records = 0
    decoy_drift_action_labels = 0
    decoy_scan_action_labels = 0
    rejected_target_drift_action_labels = 0
    target_handoff_action_labels = 0
    target_scan_broadcast_labels = 0
    redundant_target_wait_action_labels = 0
    oracle_message_rollin_steps = 0
    oracle_message_rollin_agents = 0
    oracle_message_rollin_tokens = 0
    target_interact_miss_kind_counts = {name: 0 for name in _SIGNAL_TARGET_SCAN_KIND_NAMES}
    focus_events = _dagger_focus_events(cfg)
    positive_replay_events = _dagger_positive_replay_events(cfg)
    positive_replay_event_counts: dict[str, int] = {}
    oracle_message_rollin_rate = min(1.0, max(0.0, float(cfg.dagger_oracle_message_rollin_rate)))

    for ep in range(cfg.dagger_episodes):
        seed = _dagger_collection_seed(cfg, round_idx, ep)
        rng = np.random.default_rng(seed + 7919)
        env, episode_cfg = _build_training_env(cfg, round_idx * cfg.dagger_episodes + ep)
        oracle_fn = _make_oracle_policy(env, episode_cfg)
        obs, info = env.reset(seed=seed)
        hidden = model.init_hidden(env.num_agents, device)
        ep_data = _new_episode_sequence()
        done = False
        truncated = False
        step = 0
        last_info = {}
        prev_actions: dict[int, int] = {}
        prev_msg_lens: dict[int, int] = {}
        scan_state = _initial_signal_scan_state(episode_cfg)
        has_policy_step = False
        prev_positions: dict[int, tuple[int, int]] = {}
        model_position_history: dict[int, list[tuple[int, int]]] = {aid: [] for aid in range(env.num_agents)}
        max_collect_steps = int(cfg.dagger_max_steps_per_episode)
        recovery_remaining = {aid: 0 for aid in range(env.num_agents)}
        focus_records: list[dict] = []
        deferred_team_records: list[dict] = []

        while not (done or truncated) and (max_collect_steps <= 0 or step < max_collect_steps):
            has_policy_step = _update_signal_scan_state_from_info(
                episode_cfg,
                scan_state,
                last_info,
                env.num_agents,
                prev_positions,
                has_policy_step=has_policy_step,
            )
            feedback = _feedback_matrix(
                episode_cfg,
                env.num_agents,
                prev_actions=prev_actions,
                prev_msg_lens=prev_msg_lens,
                info=last_info,
                env=env,
                obs=obs,
            )
            oracle_actions = oracle_fn(obs, info, {"step": step})
            oracle_actions, broadcast_label_agents = _apply_signal_target_scan_broadcast_overrides(
                episode_cfg,
                env,
                oracle_actions,
                feedback,
                info=last_info,
            )
            target_scan_broadcast_labels += len(broadcast_label_agents)
            oracle_actions, handoff_label_agents = _apply_signal_target_handoff_overrides(
                episode_cfg,
                env,
                oracle_actions,
                feedback,
                obs=obs,
            )
            target_handoff_action_labels += len(handoff_label_agents)
            if "target_handoff" in positive_replay_events and handoff_label_agents:
                positive_replay_event_counts["target_handoff"] = (
                    positive_replay_event_counts.get("target_handoff", 0)
                    + len(handoff_label_agents)
                )
                for aid in handoff_label_agents:
                    focus_records.append({
                        "event": "target_handoff",
                        "step": step,
                        "agents": [aid],
                        "kind": "positive",
                    })
            if episode_cfg.dagger_redundant_target_wait_labels:
                oracle_actions, redundant_wait_label_agents = _apply_signal_redundant_target_wait_overrides(
                    env,
                    oracle_actions,
                )
                redundant_target_wait_action_labels += len(redundant_wait_label_agents)
            step_weights = np.ones((env.num_agents,), dtype=np.float32)
            if "target_pursuit" in positive_replay_events:
                target_pursuit_agents = _signal_positive_target_pursuit_agents(
                    env,
                    obs,
                    oracle_actions,
                    min_map_size=episode_cfg.dagger_positive_target_pursuit_min_map_size,
                )
                if target_pursuit_agents:
                    positive_replay_event_counts["target_pursuit"] = (
                        positive_replay_event_counts.get("target_pursuit", 0)
                        + len(target_pursuit_agents)
                    )
                    for aid in target_pursuit_agents:
                        focus_records.append({
                            "event": "target_pursuit",
                            "step": step,
                            "agents": [aid],
                            "kind": "positive",
                        })
            for aid, remaining in recovery_remaining.items():
                if remaining > 0:
                    step_weights[aid] = max(step_weights[aid], float(cfg.dagger_focus_recovery_weight))
                    recovery_state_updates += 1
                    recovery_remaining[aid] = remaining - 1
            _append_labeled_step(
                ep_data,
                obs,
                oracle_actions,
                env,
                episode_cfg,
                feedback=feedback,
                step_weight=step_weights,
            )
            model_actions, hidden = _decode_recurrent_actions(
                episode_cfg,
                model,
                obs,
                hidden,
                device,
                feedback=feedback,
                scan_state=scan_state,
            )
            target_interact_miss_agents = _signal_target_interact_miss_agents(
                env,
                oracle_actions,
                model_actions,
            )
            if target_interact_miss_agents and "target_interact_miss" in focus_events:
                for aid in target_interact_miss_agents:
                    kind = _signal_target_scan_kind(env, int(aid))
                    if 0 <= int(kind) < len(_SIGNAL_TARGET_SCAN_KIND_NAMES):
                        kind_name = _SIGNAL_TARGET_SCAN_KIND_NAMES[int(kind)]
                        target_interact_miss_kind_counts[kind_name] = (
                            target_interact_miss_kind_counts.get(kind_name, 0) + 1
                        )
                focus_event_counts["target_interact_miss"] = (
                    focus_event_counts.get("target_interact_miss", 0)
                    + len(target_interact_miss_agents)
                )
                focus_records.append({
                    "event": "target_interact_miss",
                    "step": step,
                    "agents": list(target_interact_miss_agents),
                    "kind": "focus",
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=target_interact_miss_agents,
                    weight=max(
                        float(cfg.dagger_focus_error_weight),
                        float(cfg.dagger_target_interact_focus_weight),
                    ),
                )
                for aid in target_interact_miss_agents:
                    recovery_remaining[aid] = max(
                        recovery_remaining[aid],
                        int(cfg.dagger_focus_window),
                    )
            target_pursuit_miss_agents = _signal_target_pursuit_miss_agents(
                env,
                obs,
                oracle_actions,
                model_actions,
            )
            if target_pursuit_miss_agents and "target_pursuit_miss" in focus_events:
                focus_event_counts["target_pursuit_miss"] = (
                    focus_event_counts.get("target_pursuit_miss", 0) + len(target_pursuit_miss_agents)
                )
                focus_records.append({
                    "event": "target_pursuit_miss",
                    "step": step,
                    "agents": list(target_pursuit_miss_agents),
                    "kind": "focus",
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=target_pursuit_miss_agents,
                    weight=cfg.dagger_focus_error_weight,
                )
                for aid in target_pursuit_miss_agents:
                    recovery_remaining[aid] = max(
                        recovery_remaining[aid],
                        int(cfg.dagger_focus_window),
                    )
            target_decoy_drift_miss_agents = _signal_target_decoy_drift_miss_agents(
                env,
                obs,
                oracle_actions,
                model_actions,
                min_map_size=cfg.dagger_target_discovery_min_map_size,
            )
            if target_decoy_drift_miss_agents and "target_decoy_drift_miss" in focus_events:
                focus_event_counts["target_decoy_drift_miss"] = (
                    focus_event_counts.get("target_decoy_drift_miss", 0)
                    + len(target_decoy_drift_miss_agents)
                )
                focus_records.append({
                    "event": "target_decoy_drift_miss",
                    "step": step,
                    "agents": list(target_decoy_drift_miss_agents),
                    "kind": "focus",
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=target_decoy_drift_miss_agents,
                    weight=max(
                        float(cfg.dagger_focus_error_weight),
                        float(cfg.dagger_target_decoy_drift_focus_weight),
                    ),
                )
                decoy_drift_action_labels += _label_latest_signal_decoy_drift_actions(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=target_decoy_drift_miss_agents,
                    model_actions=model_actions,
                )
                for aid in target_decoy_drift_miss_agents:
                    recovery_remaining[aid] = max(
                        recovery_remaining[aid],
                        int(cfg.dagger_focus_window),
                    )
            rejected_target_drift_agents = _signal_rejected_target_drift_agents(
                env,
                obs,
                model_actions,
            )
            if rejected_target_drift_agents:
                rejected_target_drift_action_labels += _label_latest_signal_rejected_target_drift_actions(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=rejected_target_drift_agents,
                    model_actions=model_actions,
                )
            if rejected_target_drift_agents and "rejected_target_drift" in focus_events:
                focus_event_counts["rejected_target_drift"] = (
                    focus_event_counts.get("rejected_target_drift", 0)
                    + len(rejected_target_drift_agents)
                )
                focus_records.append({
                    "event": "rejected_target_drift",
                    "step": step,
                    "agents": list(rejected_target_drift_agents),
                    "kind": "focus",
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=rejected_target_drift_agents,
                    weight=cfg.dagger_focus_error_weight,
                )
                for aid in rejected_target_drift_agents:
                    recovery_remaining[aid] = max(
                        recovery_remaining[aid],
                        int(cfg.dagger_focus_window),
                    )
            target_discovery_miss_agents = _signal_target_discovery_miss_agents(
                env,
                obs,
                oracle_actions,
                model_actions,
                min_map_size=cfg.dagger_target_discovery_min_map_size,
            )
            if target_discovery_miss_agents and "target_discovery_miss" in focus_events:
                focus_event_counts["target_discovery_miss"] = (
                    focus_event_counts.get("target_discovery_miss", 0)
                    + len(target_discovery_miss_agents)
                )
                focus_records.append({
                    "event": "target_discovery_miss",
                    "step": step,
                    "agents": list(target_discovery_miss_agents),
                    "kind": "focus",
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=target_discovery_miss_agents,
                    weight=max(
                        float(cfg.dagger_focus_error_weight),
                        float(cfg.dagger_target_discovery_focus_weight),
                    ),
                )
                for aid in target_discovery_miss_agents:
                    recovery_remaining[aid] = max(
                        recovery_remaining[aid],
                        int(cfg.dagger_focus_window),
                    )
            movement_stall_miss_agents = _signal_movement_stall_miss_agents(
                env,
                obs,
                oracle_actions,
                model_actions,
                model_position_history,
                min_map_size=cfg.dagger_movement_stall_min_map_size,
                window=cfg.dagger_movement_stall_window,
            )
            if movement_stall_miss_agents and "movement_stall_miss" in focus_events:
                focus_event_counts["movement_stall_miss"] = (
                    focus_event_counts.get("movement_stall_miss", 0)
                    + len(movement_stall_miss_agents)
                )
                focus_records.append({
                    "event": "movement_stall_miss",
                    "step": step,
                    "agents": list(movement_stall_miss_agents),
                    "kind": "focus",
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=movement_stall_miss_agents,
                    weight=max(
                        float(cfg.dagger_focus_error_weight),
                        float(cfg.dagger_movement_stall_focus_weight),
                    ),
                )
                for aid in movement_stall_miss_agents:
                    recovery_remaining[aid] = max(
                        recovery_remaining[aid],
                        int(cfg.dagger_focus_window),
                    )
            target_handoff_miss_agents = _signal_target_handoff_miss_agents(
                env,
                obs,
                oracle_actions,
                model_actions,
                feedback,
                cfg=episode_cfg,
            )
            if target_handoff_miss_agents and "target_handoff_miss" in focus_events:
                focus_event_counts["target_handoff_miss"] = (
                    focus_event_counts.get("target_handoff_miss", 0) + len(target_handoff_miss_agents)
                )
                focus_records.append({
                    "event": "target_handoff_miss",
                    "step": step,
                    "agents": list(target_handoff_miss_agents),
                    "kind": "focus",
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=target_handoff_miss_agents,
                    weight=cfg.dagger_focus_error_weight,
                )
                for aid in target_handoff_miss_agents:
                    recovery_remaining[aid] = max(
                        recovery_remaining[aid],
                        int(cfg.dagger_focus_window),
                    )
            rejected_target_agents = _signal_rejected_target_interact_agents(env, obs, model_actions)
            if rejected_target_agents and "rejected_target_scan" in focus_events:
                focus_event_counts["rejected_target_scan"] = (
                    focus_event_counts.get("rejected_target_scan", 0) + len(rejected_target_agents)
                )
                focus_records.append({
                    "event": "rejected_target_scan",
                    "step": step,
                    "agents": list(rejected_target_agents),
                    "kind": "focus",
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=rejected_target_agents,
                    weight=cfg.dagger_focus_error_weight,
                )
                for aid in rejected_target_agents:
                    recovery_remaining[aid] = max(
                        recovery_remaining[aid],
                        int(cfg.dagger_focus_window),
                    )
            bad_redundant_target_agents = _signal_bad_redundant_target_interact_agents(env, obs, model_actions)
            if bad_redundant_target_agents and "bad_redundant_target_scan" in focus_events:
                focus_event_counts["bad_redundant_target_scan"] = (
                    focus_event_counts.get("bad_redundant_target_scan", 0) + len(bad_redundant_target_agents)
                )
                focus_records.append({
                    "event": "bad_redundant_target_scan",
                    "step": step,
                    "agents": list(bad_redundant_target_agents),
                    "kind": "focus",
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=bad_redundant_target_agents,
                    weight=cfg.dagger_focus_error_weight,
                )
                for aid in bad_redundant_target_agents:
                    recovery_remaining[aid] = max(
                        recovery_remaining[aid],
                        int(cfg.dagger_focus_window),
                    )
            redundant_target_agents = _redundant_target_scan_agents(env, model_actions)
            if redundant_target_agents and "redundant_target_scan" in focus_events:
                focus_event_counts["redundant_target_scan"] = (
                    focus_event_counts.get("redundant_target_scan", 0) + len(redundant_target_agents)
                )
                focus_records.append({
                    "event": "redundant_target_scan",
                    "step": step,
                    "agents": list(redundant_target_agents),
                    "kind": "focus",
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=redundant_target_agents,
                    weight=cfg.dagger_focus_error_weight,
                )
                for aid in redundant_target_agents:
                    recovery_remaining[aid] = max(
                        recovery_remaining[aid],
                        int(cfg.dagger_focus_window),
                    )
            valid_solo_target_agents, solo_target_agents = _split_solo_target_scan_agents(
                env,
                obs,
                model_actions,
            )
            if valid_solo_target_agents:
                non_focus_event_counts["valid_solo_target_scan"] = (
                    non_focus_event_counts.get("valid_solo_target_scan", 0)
                    + len(valid_solo_target_agents)
                )
                if "valid_solo_target_scan" in positive_replay_events:
                    positive_replay_event_counts["valid_solo_target_scan"] = (
                        positive_replay_event_counts.get("valid_solo_target_scan", 0)
                        + len(valid_solo_target_agents)
                    )
                    focus_records.append({
                        "event": "valid_solo_target_scan",
                        "step": step,
                        "agents": list(valid_solo_target_agents),
                        "kind": "positive",
                    })
                team_weight = float(getattr(cfg, "dagger_solo_target_team_weight", 1.0))
                teammate_agents = _solo_target_teammate_agents(env.num_agents, valid_solo_target_agents)
                if team_weight > 1.0 and teammate_agents and cfg.dagger_solo_target_team_success_only:
                    deferred_team_records.append({
                        "step": step,
                        "agents": teammate_agents,
                        "weight": team_weight,
                    })
                else:
                    team_updates, teammate_agents = _scale_solo_target_team_weights(
                        ep_data,
                        num_agents=env.num_agents,
                        solo_target_agents=valid_solo_target_agents,
                        weight=team_weight,
                    )
                    solo_target_team_state_updates += team_updates
                if team_weight > 1.0 and not cfg.dagger_solo_target_team_success_only:
                    for aid in teammate_agents:
                        recovery_remaining[aid] = max(
                            recovery_remaining[aid],
                            int(cfg.dagger_focus_window),
                        )
            if "redundant_target_scan" in focus_events:
                redundant_set = set(int(aid) for aid in redundant_target_agents)
                solo_target_agents = [aid for aid in solo_target_agents if int(aid) not in redundant_set]
            if solo_target_agents and "solo_target_scan" in focus_events:
                focus_event_counts["solo_target_scan"] = focus_event_counts.get("solo_target_scan", 0) + len(solo_target_agents)
                focus_records.append({
                    "event": "solo_target_scan",
                    "step": step,
                    "agents": list(solo_target_agents),
                    "kind": "focus",
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=solo_target_agents,
                    weight=cfg.dagger_focus_error_weight,
                )
                team_weight = float(getattr(cfg, "dagger_solo_target_team_weight", 1.0))
                teammate_agents = _solo_target_teammate_agents(env.num_agents, solo_target_agents)
                if team_weight > 1.0 and teammate_agents and cfg.dagger_solo_target_team_success_only:
                    deferred_team_records.append({
                        "step": step,
                        "agents": teammate_agents,
                        "weight": team_weight,
                    })
                else:
                    team_updates, teammate_agents = _scale_solo_target_team_weights(
                        ep_data,
                        num_agents=env.num_agents,
                        solo_target_agents=solo_target_agents,
                        weight=team_weight,
                    )
                    solo_target_team_state_updates += team_updates
                if team_weight > 1.0 and not cfg.dagger_solo_target_team_success_only:
                    for aid in teammate_agents:
                        recovery_remaining[aid] = max(
                            recovery_remaining[aid],
                            int(cfg.dagger_focus_window),
                        )
                for aid in solo_target_agents:
                    recovery_remaining[aid] = max(
                        recovery_remaining[aid],
                        int(cfg.dagger_focus_window),
                    )
            rollout_actions, replaced_agents, replaced_tokens = _mix_oracle_rollin_messages(
                model_actions,
                oracle_actions,
                oracle_message_rollin_rate,
                rng,
            )
            if replaced_agents > 0:
                oracle_message_rollin_steps += 1
                oracle_message_rollin_agents += int(replaced_agents)
                oracle_message_rollin_tokens += int(replaced_tokens)
            prev_positions = _signal_positions_from_obs(obs)
            history_window = max(1, int(cfg.dagger_movement_stall_window))
            for aid, pos in prev_positions.items():
                history = model_position_history.setdefault(int(aid), [])
                history.append(tuple(pos))
                if len(history) > history_window:
                    del history[:-history_window]
            obs, _rewards, done, truncated, info = env.step(rollout_actions)
            last_info = info or {}
            event_names = _event_names_by_agent(last_info, env.num_agents)
            decoy_scan_agents = [
                int(aid)
                for aid, names in event_names.items()
                if "decoy_scan" in names
            ]
            if decoy_scan_agents:
                decoy_scan_action_labels += _label_latest_signal_decoy_scan_actions(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=decoy_scan_agents,
                    model_actions=model_actions,
                )
            focused_agents = []
            for aid, names in event_names.items():
                positive_matched = sorted((names & positive_replay_events) - focus_events)
                for name in positive_matched:
                    positive_replay_event_counts[name] = positive_replay_event_counts.get(name, 0) + 1
                    focus_records.append({
                        "event": name,
                        "step": step,
                        "agents": [aid],
                        "kind": "positive",
                    })
                matched = sorted(names & focus_events)
                if not matched:
                    continue
                focused_agents.append(aid)
                for name in matched:
                    focus_event_counts[name] = focus_event_counts.get(name, 0) + 1
                    focus_records.append({
                        "event": name,
                        "step": step,
                        "agents": [aid],
                        "kind": "focus",
                    })
                recovery_remaining[aid] = max(
                    recovery_remaining[aid],
                    int(cfg.dagger_focus_window),
                )
            if focused_agents:
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=focused_agents,
                    weight=cfg.dagger_focus_error_weight,
                )
            prev_actions = {aid: int(action["action"]) for aid, action in rollout_actions.items()}
            prev_msg_lens = _message_lengths(rollout_actions)
            step += 1

        capped = max_collect_steps > 0 and step >= max_collect_steps and not (done or truncated)
        success = episode_success(cfg.scenario, done, last_info)
        deferred_solo_target_team_records += len(deferred_team_records)
        if cfg.dagger_solo_target_team_success_only and deferred_team_records:
            if success:
                solo_target_team_state_updates += _apply_deferred_solo_target_team_weights(
                    ep_data,
                    deferred_team_records,
                    num_agents=env.num_agents,
                    focus_window=cfg.dagger_focus_window,
                )
            else:
                dropped_solo_target_team_records += len(deferred_team_records)
        if ep_data["obs"]:
            base_episode = _finalize_episode_sequence(
                ep_data,
                env,
                episode_cfg,
                source="dagger",
                round=round_idx,
                seed=seed,
                map_size=episode_cfg.map_size,
                success=success,
                capped=capped,
                weight=_dagger_episode_weight(cfg, success),
                steps=step,
            )
            episodes.append(base_episode)
            base_episodes += 1
            replay_snippets = _focus_replay_episodes(base_episode, focus_records, episode_cfg)
            for snippet in replay_snippets:
                episodes.append(snippet)
                replay_episodes += 1
                event = str(snippet.get("trigger_event", "unknown"))
                replay_trigger_counts[event] = replay_trigger_counts.get(event, 0) + 1
        total_steps += step
        stored_steps += len(ep_data["obs"]) // max(1, env.num_agents)
        if capped:
            capped_episodes += 1
        if success:
            successes += 1

    model.train()
    summary = {
        "episodes": len(episodes),
        "base_episodes": base_episodes,
        "replay_episodes": replay_episodes,
        "success_episodes": successes,
        "failed_episodes": cfg.dagger_episodes - successes,
        "capped_episodes": capped_episodes,
        "model_success_rate": successes / cfg.dagger_episodes if cfg.dagger_episodes else 0.0,
        "avg_steps": total_steps / cfg.dagger_episodes if cfg.dagger_episodes else 0.0,
        "avg_stored_steps": stored_steps / base_episodes if base_episodes else 0.0,
        "transitions": _episode_count_transitions(episodes),
        "effective_transitions": _episode_count_effective_transitions(episodes),
        "failed_episode_weight": float(cfg.dagger_failed_episode_weight),
        "success_episode_weight": float(cfg.dagger_success_episode_weight),
        "seed_base": int(cfg.dagger_seed_base),
        "seed_stride": int(cfg.dagger_seed_stride),
        "seed_list": _parse_seed_list(cfg.dagger_seed_list, field_name="dagger_seed_list"),
        "focus_events": focus_event_counts,
        "non_focus_events": non_focus_event_counts,
        "positive_replay_events": positive_replay_event_counts,
        "focused_state_updates": focused_state_updates,
        "solo_target_team_state_updates": solo_target_team_state_updates,
        "solo_target_team_deferred_records": deferred_solo_target_team_records,
        "solo_target_team_dropped_records": dropped_solo_target_team_records,
        "solo_target_team_success_only": bool(cfg.dagger_solo_target_team_success_only),
        "target_match_action_labels": _episode_count_label_mask(
            episodes,
            "signal_target_match_action_mask",
        ),
        "target_opportunity_action_labels": _episode_count_label_mask(
            episodes,
            "signal_target_opportunity_action_mask",
        ),
        "redundant_target_wait_action_aux_labels": _episode_count_label_mask(
            episodes,
            "signal_redundant_target_wait_action_mask",
        ),
        "target_pursuit_action_labels": _episode_count_label_mask(
            episodes,
            "signal_target_pursuit_action_mask",
        ),
        "sync_response_action_labels": _episode_count_label_mask(
            episodes,
            "signal_sync_response_action_mask",
        ),
        "decoy_drift_action_labels": decoy_drift_action_labels,
        "decoy_scan_action_labels": decoy_scan_action_labels,
        "rejected_target_drift_action_labels": rejected_target_drift_action_labels,
        "target_handoff_action_labels": target_handoff_action_labels,
        "target_scan_broadcast_labels": target_scan_broadcast_labels,
        "redundant_target_wait_action_labels": redundant_target_wait_action_labels,
        "oracle_message_rollin_rate": float(oracle_message_rollin_rate),
        "oracle_message_rollin_steps": int(oracle_message_rollin_steps),
        "oracle_message_rollin_agents": int(oracle_message_rollin_agents),
        "oracle_message_rollin_tokens": int(oracle_message_rollin_tokens),
        "target_interact_miss_kinds": target_interact_miss_kind_counts,
        "recovery_state_updates": recovery_state_updates,
        "focus_error_weight": float(cfg.dagger_focus_error_weight),
        "movement_stall_focus_weight": float(cfg.dagger_movement_stall_focus_weight),
        "movement_stall_window": int(cfg.dagger_movement_stall_window),
        "solo_target_team_weight": float(cfg.dagger_solo_target_team_weight),
        "focus_recovery_weight": float(cfg.dagger_focus_recovery_weight),
        "focus_replay_enabled": bool(cfg.dagger_focus_replay),
        "map_sizes": _episode_map_size_counts(episodes),
        "map_diagnostics": _episode_map_size_diagnostics(episodes),
        "replay_trigger_events": replay_trigger_counts,
        "replay_transitions": _episode_count_transitions(
            [ep for ep in episodes if ep.get("source") == "dagger_focus_replay"]
        ),
        "replay_effective_transitions": _episode_count_effective_transitions(
            [ep for ep in episodes if ep.get("source") == "dagger_focus_replay"]
        ),
    }
    return episodes, summary


def train_recurrent_bc_dagger(
    cfg: RecurrentConfig,
    episodes,
    device,
    *,
    wandb_run=None,
    initial_model: MAPPORecurrentActor | None = None,
):
    """Run recurrent DAgger over full episode sequences."""
    import copy

    all_episodes = list(episodes)
    history = []
    model = initial_model
    best_state = None
    best_row = None
    best_score = None
    best_eval_send_threshold = None
    non_improving_rounds = 0
    eval_seed_count = max(1, int(cfg.eval_seed_count))
    if initial_model is not None:
        initial_eval = evaluate_recurrent_policy_multi_seed(
            cfg,
            initial_model,
            device,
            seed_count=eval_seed_count,
        )
        best_score = _recurrent_eval_score(initial_eval)
        best_state = copy.deepcopy(initial_model.state_dict())
        best_eval_send_threshold = float(cfg.eval_send_threshold)
        best_row = {
            "round": -1,
            "phase": "recurrent_init",
            "dataset_episodes": len(all_episodes),
            "dataset_transitions": _episode_count_transitions(all_episodes),
            "dataset_effective_transitions": _episode_count_effective_transitions(all_episodes),
            "dataset_sources": _episode_source_counts(all_episodes),
            "dataset_map_sizes": _episode_map_size_counts(all_episodes),
            "dataset_map_diagnostics": _episode_map_size_diagnostics(all_episodes),
            "retrain_from_scratch": False,
            "started_from_recurrent_init": True,
            "eval": initial_eval,
            "eval_score": best_score,
            "eval_send_threshold": best_eval_send_threshold,
        }
        print(json.dumps({"recurrent_dagger_initial": best_row}, indent=2, sort_keys=True))

    for round_idx in range(cfg.dagger_rounds + 1):
        print(
            f"\n=== Recurrent DAgger round {round_idx} | "
            f"episodes {len(all_episodes)} | transitions {_episode_count_transitions(all_episodes)} | "
            f"effective {_episode_count_effective_transitions(all_episodes):.1f} ==="
        )
        starts_from_initial_model = bool(round_idx == 0 and initial_model is not None)
        if starts_from_initial_model:
            start_model = initial_model
        else:
            start_model = None if cfg.dagger_retrain_from_scratch else model
        if wandb_run is None:
            model = train_recurrent_bc(cfg, all_episodes, device, model=start_model)
        else:
            model = train_recurrent_bc(
                cfg,
                all_episodes,
                device,
                model=start_model,
                wandb_run=wandb_run,
                log_prefix="bc",
                log_context={
                    "dagger/round": int(round_idx),
                    "dagger/dataset_episodes": int(len(all_episodes)),
                    "dagger/dataset_transitions": int(_episode_count_transitions(all_episodes)),
                    "dagger/dataset_effective_transitions": float(_episode_count_effective_transitions(all_episodes)),
                },
            )
        eval_result = evaluate_recurrent_policy_multi_seed(
            cfg,
            model,
            device,
            seed_count=eval_seed_count,
        )
        eval_score = _recurrent_eval_score(eval_result)
        is_best = best_score is None or eval_score > best_score
        round_row = {
            "round": round_idx,
            "dataset_episodes": len(all_episodes),
            "dataset_transitions": _episode_count_transitions(all_episodes),
            "dataset_effective_transitions": _episode_count_effective_transitions(all_episodes),
            "dataset_sources": _episode_source_counts(all_episodes),
            "dataset_map_sizes": _episode_map_size_counts(all_episodes),
            "dataset_map_diagnostics": _episode_map_size_diagnostics(all_episodes),
            "retrain_from_scratch": start_model is None,
            "started_from_recurrent_init": starts_from_initial_model,
            "eval": eval_result,
            "eval_score": eval_score,
            "eval_send_threshold": float(cfg.eval_send_threshold),
        }
        if is_best:
            best_score = eval_score
            best_state = copy.deepcopy(model.state_dict())
            best_eval_send_threshold = float(cfg.eval_send_threshold)
            non_improving_rounds = 0
        else:
            non_improving_rounds += 1
        early_stop_patience = max(0, int(cfg.dagger_early_stop_patience))
        should_early_stop = early_stop_patience > 0 and non_improving_rounds >= early_stop_patience
        round_row["non_improving_rounds"] = non_improving_rounds
        round_row["early_stop"] = bool(should_early_stop)
        if is_best:
            best_row = dict(round_row)
        print(json.dumps({"recurrent_dagger": round_row}, indent=2, sort_keys=True))
        if wandb_run is not None:
            _wandb_log(
                wandb_run,
                {
                    **_recurrent_eval_wandb_payload(
                        eval_result,
                        update=round_idx,
                        is_best=is_best,
                        best_eval=(best_row or {}).get("eval") if best_row else None,
                        prefix="dagger/eval",
                    ),
                    "dagger/round": int(round_idx),
                    "dagger/dataset_episodes": int(len(all_episodes)),
                    "dagger/dataset_transitions": int(_episode_count_transitions(all_episodes)),
                    "dagger/dataset_effective_transitions": float(_episode_count_effective_transitions(all_episodes)),
                    **_map_diagnostics_wandb_payload(
                        "dagger/dataset",
                        _episode_map_size_diagnostics(all_episodes),
                    ),
                    "dagger/non_improving_rounds": int(non_improving_rounds),
                    "dagger/early_stop": int(should_early_stop),
                },
                context="dagger eval log",
            )

        if should_early_stop:
            print(json.dumps({
                "recurrent_dagger_early_stop": {
                    "round": round_idx,
                    "patience": early_stop_patience,
                    "best_round": (best_row or {}).get("round"),
                    "best_score": best_score,
                }
            }, indent=2, sort_keys=True))
        if round_idx < cfg.dagger_rounds and not should_early_stop:
            new_episodes, collect_summary = collect_recurrent_dagger_episodes(
                cfg,
                model,
                device,
                round_idx=round_idx,
            )
            all_episodes.extend(new_episodes)
            round_row["collect"] = collect_summary
            print(json.dumps({"dagger_collect": collect_summary}, indent=2, sort_keys=True))
            if wandb_run is not None:
                _wandb_log(
                    wandb_run,
                    {
                        "dagger/collect_round": int(round_idx),
                        "dagger/collect_model_success_rate": float(collect_summary["model_success_rate"]),
                        "dagger/collect_avg_steps": float(collect_summary["avg_steps"]),
                        "dagger/collect_transitions": int(collect_summary["transitions"]),
                        "dagger/collect_effective_transitions": float(collect_summary["effective_transitions"]),
                        "dagger/collect_focused_state_updates": int(collect_summary["focused_state_updates"]),
                        "dagger/collect_solo_target_team_state_updates": int(collect_summary["solo_target_team_state_updates"]),
                        "dagger/collect_solo_target_team_deferred_records": int(
                            collect_summary.get("solo_target_team_deferred_records", 0)
                        ),
                        "dagger/collect_solo_target_team_dropped_records": int(
                            collect_summary.get("solo_target_team_dropped_records", 0)
                        ),
                        "dagger/collect_solo_target_team_success_only": int(
                            bool(collect_summary.get("solo_target_team_success_only", False))
                        ),
                        "dagger/collect_target_match_action_labels": int(
                            collect_summary.get("target_match_action_labels", 0)
                        ),
                        "dagger/collect_target_opportunity_action_labels": int(
                            collect_summary.get("target_opportunity_action_labels", 0)
                        ),
                        "dagger/collect_redundant_target_wait_action_aux_labels": int(
                            collect_summary.get("redundant_target_wait_action_aux_labels", 0)
                        ),
                        "dagger/collect_target_pursuit_action_labels": int(
                            collect_summary.get("target_pursuit_action_labels", 0)
                        ),
                        "dagger/collect_sync_response_action_labels": int(
                            collect_summary.get("sync_response_action_labels", 0)
                        ),
                        "dagger/collect_decoy_drift_action_labels": int(
                            collect_summary.get("decoy_drift_action_labels", 0)
                        ),
                        "dagger/collect_decoy_scan_action_labels": int(
                            collect_summary.get("decoy_scan_action_labels", 0)
                        ),
                        "dagger/collect_rejected_target_drift_action_labels": int(
                            collect_summary.get("rejected_target_drift_action_labels", 0)
                        ),
                        "dagger/collect_target_handoff_action_labels": int(
                            collect_summary.get("target_handoff_action_labels", 0)
                        ),
                        "dagger/collect_target_scan_broadcast_labels": int(
                            collect_summary.get("target_scan_broadcast_labels", 0)
                        ),
                        "dagger/collect_redundant_target_wait_action_labels": int(
                            collect_summary.get("redundant_target_wait_action_labels", 0)
                        ),
                        "dagger/collect_oracle_message_rollin_rate": float(
                            collect_summary.get("oracle_message_rollin_rate", 0.0)
                        ),
                        "dagger/collect_oracle_message_rollin_steps": int(
                            collect_summary.get("oracle_message_rollin_steps", 0)
                        ),
                        "dagger/collect_oracle_message_rollin_agents": int(
                            collect_summary.get("oracle_message_rollin_agents", 0)
                        ),
                        "dagger/collect_oracle_message_rollin_tokens": int(
                            collect_summary.get("oracle_message_rollin_tokens", 0)
                        ),
                        **{
                            f"dagger/collect_target_interact_miss_{kind_name}": int(count)
                            for kind_name, count in sorted(
                                (collect_summary.get("target_interact_miss_kinds") or {}).items()
                            )
                        },
                        "dagger/collect_recovery_state_updates": int(collect_summary["recovery_state_updates"]),
                        "dagger/collect_replay_episodes": int(collect_summary["replay_episodes"]),
                        **{
                            f"dagger/collect_positive_replay_event_{event}": int(count)
                            for event, count in sorted((collect_summary.get("positive_replay_events") or {}).items())
                        },
                        **{
                            f"dagger/collect_non_focus_event_{event}": int(count)
                            for event, count in sorted((collect_summary.get("non_focus_events") or {}).items())
                        },
                        **{
                            f"dagger/collect_replay_trigger_{event}": int(count)
                            for event, count in sorted((collect_summary.get("replay_trigger_events") or {}).items())
                        },
                        **_map_diagnostics_wandb_payload(
                            "dagger/collect",
                            collect_summary.get("map_diagnostics") or {},
                        ),
                    },
                    context="dagger collect log",
                )

        history.append(round_row)
        if should_early_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_eval_send_threshold is not None:
        cfg.eval_send_threshold = float(best_eval_send_threshold)

    return model, history, all_episodes, best_row


def _signal_center_target_scan_decoding_candidate(obs_agent: dict, cfg: RecurrentConfig) -> bool:
    if cfg.scenario != "signal_hunt" or not isinstance(obs_agent, dict):
        return False
    pos_arr = np.asarray(obs_agent.get("self_pos", np.zeros((2,), dtype=np.int16)), dtype=np.int64).reshape(-1)
    if pos_arr.size < 2:
        return False
    observed_map_size = _observed_map_size(obs_agent, cfg)
    pos = (int(pos_arr[0]), int(pos_arr[1]))
    visible_targets = set(_visible_signal_targets(obs_agent, pos_arr, observed_map_size))
    if pos not in visible_targets:
        return False
    if not _signal_observation_has_target_information(obs_agent):
        return False
    if not _signal_observation_allows_target(obs_agent, pos, observed_map_size):
        return False

    own_segments, message_segments = _signal_segments_from_observation(obs_agent)
    exact_targets = set(_signal_targets_from_segments([*own_segments, *message_segments], observed_map_size))
    if pos in exact_targets:
        return True

    allowed_visible_targets = {
        target
        for target in visible_targets
        if _signal_observation_allows_target(obs_agent, target, observed_map_size)
    }
    if allowed_visible_targets == {pos}:
        return True

    inferred_targets = set(_signal_inferred_constraint_targets(obs_agent, observed_map_size))
    return inferred_targets == {pos}


def _signal_center_visible_target_tile(obs_agent: dict, cfg: RecurrentConfig) -> bool:
    if cfg.scenario != "signal_hunt" or not isinstance(obs_agent, dict):
        return False
    pos_arr = np.asarray(obs_agent.get("self_pos", np.zeros((2,), dtype=np.int16)), dtype=np.int64).reshape(-1)
    if pos_arr.size < 2:
        return False
    observed_map_size = _observed_map_size(obs_agent, cfg)
    pos = (int(pos_arr[0]), int(pos_arr[1]))
    return pos in set(_visible_signal_targets(obs_agent, pos_arr, observed_map_size))


def _action_allowed_from_obs(obs_agent: dict, action_id: int) -> bool:
    raw_mask = obs_agent.get("action_mask") if isinstance(obs_agent, dict) else None
    if raw_mask is None:
        return True
    try:
        mask = np.asarray(raw_mask, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return True
    action_id = int(action_id)
    return action_id < mask.size and float(mask[action_id]) > 0.0


def _apply_signal_target_scan_decoding(
    cfg: RecurrentConfig,
    obs: dict,
    logits: torch.Tensor,
    acts: torch.Tensor,
) -> torch.Tensor:
    threshold = float(getattr(cfg, "eval_signal_target_scan_threshold", -1.0))
    if cfg.scenario != "signal_hunt" or threshold < 0.0:
        return acts
    threshold = min(1.0, max(0.0, threshold))
    interact_probs = torch.softmax(logits, dim=-1)[:, SyncOrSinkEnv.ACTION_INTERACT]
    corrected = acts.clone()
    for aid in range(int(acts.shape[0])):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        if not _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_INTERACT):
            continue
        if float(interact_probs[aid].detach().cpu().item()) < threshold:
            continue
        if _signal_center_target_scan_decoding_candidate(obs_agent, cfg):
            corrected[aid] = int(SyncOrSinkEnv.ACTION_INTERACT)
    return corrected


def _apply_signal_scan_gate_decoding(
    cfg: RecurrentConfig,
    obs: dict,
    acts: torch.Tensor,
    scan_gate_logits: torch.Tensor | None,
) -> torch.Tensor:
    threshold = float(getattr(cfg, "eval_signal_scan_gate_threshold", -1.0))
    if cfg.scenario != "signal_hunt" or threshold < 0.0 or scan_gate_logits is None:
        return acts
    threshold = min(1.0, max(0.0, threshold))
    scan_probs = torch.sigmoid(scan_gate_logits.reshape(-1))
    corrected = acts.clone()
    for aid in range(int(acts.shape[0])):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        if not _signal_center_visible_target_tile(obs_agent, cfg):
            continue
        if not _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_INTERACT):
            continue
        observed_map_size = _observed_map_size(obs_agent, cfg)
        candidate = _signal_center_target_scan_decoding_candidate(obs_agent, cfg)
        rejected = _signal_center_rejected_target(obs_agent, observed_map_size)
        prob = float(scan_probs[aid].detach().cpu().item())
        if prob >= threshold and candidate:
            corrected[aid] = int(SyncOrSinkEnv.ACTION_INTERACT)
        elif (
            bool(getattr(cfg, "eval_signal_scan_gate_suppress", True))
            and int(corrected[aid].item()) == int(SyncOrSinkEnv.ACTION_INTERACT)
            and (prob < threshold or rejected)
            and _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_STAY)
        ):
            corrected[aid] = int(SyncOrSinkEnv.ACTION_STAY)
    return corrected


def _apply_signal_target_validity_decoding(
    cfg: RecurrentConfig,
    obs: dict,
    acts: torch.Tensor,
    target_validity_logits: torch.Tensor | None,
) -> torch.Tensor:
    threshold = float(getattr(cfg, "eval_signal_target_validity_threshold", -1.0))
    if cfg.scenario != "signal_hunt" or threshold < 0.0 or target_validity_logits is None:
        return acts
    threshold = min(1.0, max(0.0, threshold))
    validity_probs = torch.sigmoid(target_validity_logits.reshape(-1))
    corrected = acts.clone()
    for aid in range(int(acts.shape[0])):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        if int(corrected[aid].item()) != int(SyncOrSinkEnv.ACTION_INTERACT):
            continue
        if not _signal_center_visible_target_tile(obs_agent, cfg):
            continue
        if not _signal_observation_has_target_information(obs_agent):
            continue
        if not _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_STAY):
            continue
        prob = float(validity_probs[aid].detach().cpu().item())
        if prob < threshold:
            corrected[aid] = int(SyncOrSinkEnv.ACTION_STAY)
    return corrected


def _apply_signal_target_decision_decoding(
    cfg: RecurrentConfig,
    obs: dict,
    acts: torch.Tensor,
    target_decision_logits: torch.Tensor | None,
) -> torch.Tensor:
    threshold = float(getattr(cfg, "eval_signal_target_decision_threshold", -1.0))
    if cfg.scenario != "signal_hunt" or threshold < 0.0 or target_decision_logits is None:
        return acts
    threshold = min(1.0, max(0.0, threshold))
    decision_probs = torch.sigmoid(target_decision_logits.reshape(-1))
    corrected = acts.clone()
    for aid in range(int(acts.shape[0])):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        if not _signal_center_visible_target_tile(obs_agent, cfg):
            continue
        prob = float(decision_probs[aid].detach().cpu().item())
        if prob >= threshold and _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_INTERACT):
            corrected[aid] = int(SyncOrSinkEnv.ACTION_INTERACT)
        elif (
            bool(getattr(cfg, "eval_signal_target_decision_suppress", True))
            and int(corrected[aid].item()) == int(SyncOrSinkEnv.ACTION_INTERACT)
            and prob < threshold
            and _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_STAY)
        ):
            corrected[aid] = int(SyncOrSinkEnv.ACTION_STAY)
    return corrected


def _signal_teammate_scan_position_match(
    cfg: RecurrentConfig,
    obs_agent: dict,
    agent_id: int,
    num_agents: int,
    scan_state: Mapping[str, Any] | None,
) -> bool:
    if scan_state is None:
        return False
    self_pos = _signal_xy(obs_agent.get("self_pos"))
    if self_pos is None:
        return False
    scan_positions = scan_state.get("scan_pos") or {}
    if not isinstance(scan_positions, Mapping):
        return False
    scan_log = scan_state.get("scan_log") or {}
    if not isinstance(scan_log, Mapping):
        return False
    current_step = int(scan_state.get("step", 0))
    scan_window = int(scan_state.get("scan_window", getattr(cfg, "scan_window", 3)))
    for other_id in range(int(num_agents)):
        if int(other_id) == int(agent_id):
            continue
        last_scan = _scan_log_value(dict(scan_log), int(other_id))
        if last_scan is None:
            continue
        age = current_step - int(last_scan)
        if age < 0 or age > scan_window:
            continue
        other_pos = _signal_xy(scan_positions.get(int(other_id), scan_positions.get(str(int(other_id)))))
        if other_pos == self_pos:
            return True
    return False


def _apply_signal_scan_sync_decoding(
    cfg: RecurrentConfig,
    obs: dict,
    acts: torch.Tensor,
    feedback: np.ndarray | None,
    scan_state: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    if (
        cfg.scenario != "signal_hunt"
        or not bool(getattr(cfg, "eval_signal_scan_sync_assist", False))
        or feedback is None
        or not (cfg.obs_feedback and cfg.obs_signal_scan_state)
    ):
        return acts
    try:
        feedback_arr = np.asarray(feedback, dtype=np.float32).reshape(int(acts.shape[0]), -1)
    except (TypeError, ValueError):
        return acts
    scan_offset = 12 + (4 if cfg.obs_signal_sync_feedback else 0)
    if feedback_arr.shape[1] <= scan_offset + 3:
        return acts

    corrected = acts.clone()
    for aid in range(int(acts.shape[0])):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        observed_map_size = _observed_map_size(obs_agent, cfg)
        visible_target = _signal_center_visible_target_tile(obs_agent, cfg)
        if not visible_target:
            continue
        row = feedback_arr[int(aid)]
        self_active = float(row[scan_offset]) > 0.0
        teammate_active = float(row[scan_offset + 1]) > 0.0
        rejected = _signal_center_rejected_target(obs_agent, observed_map_size)
        teammate_position_match = _signal_teammate_scan_position_match(
            cfg,
            obs_agent,
            int(aid),
            int(acts.shape[0]),
            scan_state,
        )
        if rejected and not (self_active or (teammate_active and teammate_position_match)):
            continue
        candidate = _signal_center_target_scan_decoding_candidate(obs_agent, cfg)
        if teammate_active and _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_INTERACT):
            corrected[aid] = int(SyncOrSinkEnv.ACTION_INTERACT)
        elif (
            self_active
            and not teammate_active
            and _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_STAY)
        ):
            corrected[aid] = int(SyncOrSinkEnv.ACTION_STAY)
        elif (
            bool(getattr(cfg, "eval_signal_scan_sync_force_first", False))
            and candidate
            and not self_active
            and not teammate_active
            and _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_INTERACT)
        ):
            corrected[aid] = int(SyncOrSinkEnv.ACTION_INTERACT)
    return corrected


def _apply_signal_scan_refresh_decoding(
    cfg: RecurrentConfig,
    obs: dict,
    acts: torch.Tensor,
    feedback: np.ndarray | None,
) -> torch.Tensor:
    if (
        cfg.scenario != "signal_hunt"
        or not bool(getattr(cfg, "eval_signal_scan_refresh_assist", False))
        or feedback is None
        or not (cfg.obs_feedback and cfg.obs_signal_scan_state)
    ):
        return acts
    try:
        feedback_arr = np.asarray(feedback, dtype=np.float32).reshape(int(acts.shape[0]), -1)
    except (TypeError, ValueError):
        return acts
    scan_offset = 12 + (4 if cfg.obs_signal_sync_feedback else 0)
    if feedback_arr.shape[1] <= scan_offset + 3:
        return acts

    threshold = min(1.0, max(0.0, float(getattr(cfg, "eval_signal_scan_refresh_threshold", 0.5))))
    corrected = acts.clone()
    for aid in range(int(acts.shape[0])):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        row = feedback_arr[int(aid)]
        self_active = float(row[scan_offset]) > 0.0
        teammate_active = float(row[scan_offset + 1]) > 0.0
        self_remaining = float(row[scan_offset + 2])
        observed_map_size = _observed_map_size(obs_agent, cfg)
        if (
            _signal_center_rejected_target(obs_agent, observed_map_size)
            or not _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_INTERACT)
        ):
            continue
        candidate = _signal_center_target_scan_decoding_candidate(obs_agent, cfg)
        visible_target = _signal_center_visible_target_tile(obs_agent, cfg)
        if not candidate and not (self_active and visible_target):
            continue
        if self_active and not teammate_active and 0.0 < self_remaining <= threshold:
            corrected[aid] = int(SyncOrSinkEnv.ACTION_INTERACT)
    return corrected


def _signal_active_scan_position_message(
    cfg: RecurrentConfig,
    scan_state: Mapping[str, Any] | None,
    agent_id: int,
) -> list[int]:
    if (
        cfg.scenario != "signal_hunt"
        or not bool(getattr(cfg, "eval_signal_scan_broadcast_assist", False))
        or not (cfg.comm and cfg.obs_signal_scan_state)
        or scan_state is None
    ):
        return []
    scan_log = scan_state.get("scan_log") or {}
    scan_positions = scan_state.get("scan_pos") or {}
    if not isinstance(scan_log, Mapping) or not isinstance(scan_positions, Mapping):
        return []
    last_scan = _scan_log_value(dict(scan_log), int(agent_id))
    if last_scan is None:
        return []
    current_step = int(scan_state.get("step", 0))
    scan_window = int(scan_state.get("scan_window", getattr(cfg, "scan_window", 3)))
    age = current_step - int(last_scan)
    if age < 0 or age > scan_window:
        return []
    scan_pos = _signal_xy(scan_positions.get(int(agent_id), scan_positions.get(str(int(agent_id)))))
    message = _signal_exact_target_message_for_pos(cfg, scan_pos)
    if not message:
        return []
    if not isinstance(scan_state, MutableMapping):
        return message
    broadcast_log = scan_state.get("scan_broadcast_log")
    if not isinstance(broadcast_log, MutableMapping):
        broadcast_log = {}
        scan_state["scan_broadcast_log"] = broadcast_log
    previous = broadcast_log.get(int(agent_id), broadcast_log.get(str(int(agent_id))))
    previous_step = None
    previous_pos = None
    if isinstance(previous, Mapping):
        raw_step = previous.get("scan_step", previous.get("step"))
        if raw_step is not None:
            try:
                previous_step = int(raw_step)
            except (TypeError, ValueError):
                previous_step = None
        previous_pos = _signal_xy(previous.get("pos"))
    if previous_step == int(last_scan) and previous_pos == scan_pos:
        return []
    broadcast_log[int(agent_id)] = {
        "scan_step": int(last_scan),
        "pos": [int(scan_pos[0]), int(scan_pos[1])],
    }
    return message


def _apply_signal_scan_broadcast_assist(
    cfg: RecurrentConfig,
    actions: dict[int, dict],
    scan_state: Mapping[str, Any] | None,
) -> dict[int, dict]:
    corrected = {int(aid): dict(action) for aid, action in actions.items()}
    if scan_state is None:
        return corrected
    for aid in sorted(corrected):
        message = _signal_active_scan_position_message(cfg, scan_state, int(aid))
        if not message:
            continue
        corrected[int(aid)]["message_tokens"] = message
    return corrected


def _signal_exact_targets_from_message(
    tokens: list[int] | tuple[int, ...] | np.ndarray,
    observed_map_size: int,
) -> list[tuple[int, int]]:
    exact_segments = [
        segment
        for segment in _signal_segments(tokens)
        if segment and int(segment[0]) == 26
    ]
    return _signal_targets_from_segments(exact_segments, int(observed_map_size))


def _signal_exact_targets_from_observation(obs_agent: dict, observed_map_size: int) -> set[tuple[int, int]]:
    own_segments, message_segments = _signal_segments_from_observation(obs_agent)
    exact_segments = [
        segment
        for segment in [*own_segments, *message_segments]
        if segment and int(segment[0]) == 26
    ]
    return set(_signal_targets_from_segments(exact_segments, int(observed_map_size)))


def _signal_exact_target_memory_steps(cfg: RecurrentConfig) -> int:
    return max(0, int(getattr(cfg, "eval_signal_exact_target_memory_steps", 0)))


def _signal_exact_target_memory_targets(
    cfg: RecurrentConfig,
    scan_state: Mapping[str, Any] | None,
    agent_id: int,
) -> set[tuple[int, int]]:
    ttl = _signal_exact_target_memory_steps(cfg)
    if ttl <= 0 or scan_state is None:
        return set()
    memory = scan_state.get("exact_target_memory") if isinstance(scan_state, Mapping) else None
    if not isinstance(memory, Mapping):
        return set()
    row = memory.get(int(agent_id), memory.get(str(int(agent_id))))
    if not isinstance(row, Mapping):
        return set()
    pos = _signal_xy(row.get("pos"))
    if pos is None:
        return set()
    try:
        memory_step = int(row.get("step", -10**9))
    except (TypeError, ValueError):
        return set()
    current_step = int(scan_state.get("step", 0))
    if current_step - memory_step > ttl:
        return set()
    return {pos}


def _signal_base_trusted_exact_message_targets(
    cfg: RecurrentConfig,
    obs_agent: dict,
    scan_state: Mapping[str, Any] | None,
) -> set[tuple[int, int]]:
    observed_map_size = _observed_map_size(obs_agent, cfg)
    own_segments, _message_segments = _signal_segments_from_observation(obs_agent)
    own_exact_segments = [
        segment
        for segment in own_segments
        if segment and int(segment[0]) == 26
    ]
    trusted = set(_signal_targets_from_segments(own_exact_segments, observed_map_size))
    trusted.update(_signal_active_scan_positions(cfg, scan_state))
    return trusted


def _update_signal_exact_target_memory(
    cfg: RecurrentConfig,
    obs: dict,
    scan_state: Mapping[str, Any] | None,
) -> None:
    if _signal_exact_target_memory_steps(cfg) <= 0 or not isinstance(scan_state, MutableMapping):
        return
    memory = scan_state.get("exact_target_memory")
    if not isinstance(memory, MutableMapping):
        memory = {}
        scan_state["exact_target_memory"] = memory
    current_step = int(scan_state.get("step", 0))
    for raw_aid, obs_agent in (obs or {}).items():
        if not isinstance(obs_agent, dict):
            continue
        try:
            aid = int(raw_aid)
        except (TypeError, ValueError):
            continue
        observed_map_size = _observed_map_size(obs_agent, cfg)
        exact_targets = _signal_exact_targets_from_observation(obs_agent, observed_map_size)
        trusted_targets = exact_targets & _signal_base_trusted_exact_message_targets(cfg, obs_agent, scan_state)
        if len(trusted_targets) != 1:
            continue
        target = next(iter(trusted_targets))
        memory[aid] = {"pos": [int(target[0]), int(target[1])], "step": current_step}


_SIGNAL_NAV_ACTION_DELTAS = (
    (int(SyncOrSinkEnv.ACTION_UP), 0, -1),
    (int(SyncOrSinkEnv.ACTION_DOWN), 0, 1),
    (int(SyncOrSinkEnv.ACTION_LEFT), -1, 0),
    (int(SyncOrSinkEnv.ACTION_RIGHT), 1, 0),
)


def _signal_navigation_action_order(pos: tuple[int, int], target: tuple[int, int]) -> list[int]:
    x, y = int(pos[0]), int(pos[1])
    tx, ty = int(target[0]), int(target[1])
    dx, dy = tx - x, ty - y
    ordered: list[int] = []
    if abs(dx) >= abs(dy):
        if dx > 0:
            ordered.append(int(SyncOrSinkEnv.ACTION_RIGHT))
        elif dx < 0:
            ordered.append(int(SyncOrSinkEnv.ACTION_LEFT))
        if dy > 0:
            ordered.append(int(SyncOrSinkEnv.ACTION_DOWN))
        elif dy < 0:
            ordered.append(int(SyncOrSinkEnv.ACTION_UP))
    else:
        if dy > 0:
            ordered.append(int(SyncOrSinkEnv.ACTION_DOWN))
        elif dy < 0:
            ordered.append(int(SyncOrSinkEnv.ACTION_UP))
        if dx > 0:
            ordered.append(int(SyncOrSinkEnv.ACTION_RIGHT))
        elif dx < 0:
            ordered.append(int(SyncOrSinkEnv.ACTION_LEFT))

    base_rank = {
        int(SyncOrSinkEnv.ACTION_UP): 0,
        int(SyncOrSinkEnv.ACTION_DOWN): 1,
        int(SyncOrSinkEnv.ACTION_LEFT): 2,
        int(SyncOrSinkEnv.ACTION_RIGHT): 3,
    }
    action_deltas = {
        int(action_id): (int(mx), int(my))
        for action_id, mx, my in _SIGNAL_NAV_ACTION_DELTAS
    }
    remaining = [
        int(action_id)
        for action_id, _mx, _my in _SIGNAL_NAV_ACTION_DELTAS
        if int(action_id) not in ordered
    ]
    remaining.sort(
        key=lambda action_id: (
            abs(tx - (x + action_deltas[int(action_id)][0]))
            + abs(ty - (y + action_deltas[int(action_id)][1])),
            base_rank[int(action_id)],
        )
    )
    ordered.extend(remaining)
    return ordered


def _signal_local_navigation_action_from_obs(
    obs_agent: dict,
    target: tuple[int, int],
    *,
    blocked_actions: set[int] | None = None,
) -> int | None:
    blocked = {int(action_id) for action_id in (blocked_actions or set())}
    local_ids = _recurrent_local_grid_ids(obs_agent.get("local_grid", np.zeros((1, 1), dtype=np.int16)))
    if local_ids.ndim != 2 or local_ids.size == 0:
        return None
    pos_arr = np.asarray(obs_agent.get("self_pos", np.zeros((2,), dtype=np.int16)), dtype=np.int64).reshape(-1)
    if pos_arr.size < 2:
        return None
    sx, sy = int(pos_arr[0]), int(pos_arr[1])
    tx, ty = int(target[0]), int(target[1])
    h, w = int(local_ids.shape[0]), int(local_ids.shape[1])
    cx, cy = w // 2, h // 2
    target_lx = cx + (tx - sx)
    target_ly = cy + (ty - sy)
    current_dist = abs(tx - sx) + abs(ty - sy)
    if current_dist <= 0:
        return int(SyncOrSinkEnv.ACTION_INTERACT)

    action_order = _signal_navigation_action_order((sx, sy), target)
    action_rank = {int(action_id): rank for rank, action_id in enumerate(action_order)}
    ordered_deltas = sorted(
        _SIGNAL_NAV_ACTION_DELTAS,
        key=lambda row: action_rank.get(int(row[0]), len(action_rank)),
    )
    visited = {(cx, cy)}
    queue: list[tuple[int, int, int | None, int]] = [(cx, cy, None, 0)]
    index = 0
    best: tuple[tuple[int, int, int, int], int] | None = None

    while index < len(queue):
        lx, ly, first_action, depth = queue[index]
        index += 1
        gx = sx + (int(lx) - cx)
        gy = sy + (int(ly) - cy)
        dist = abs(tx - gx) + abs(ty - gy)
        if first_action is not None:
            reached_projection = int(lx) == int(target_lx) and int(ly) == int(target_ly)
            score = (
                dist,
                0 if reached_projection else 1,
                int(depth),
                action_rank.get(int(first_action), len(action_rank)),
            )
            if best is None or score < best[0]:
                best = (score, int(first_action))

        for action_id, mx, my in ordered_deltas:
            nx, ny = int(lx) + int(mx), int(ly) + int(my)
            if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in visited:
                continue
            tile_id = int(local_ids[ny, nx])
            if tile_id in (int(TILE_WALL), int(TILE_DOOR)):
                continue
            next_first = int(first_action) if first_action is not None else int(action_id)
            if first_action is None and next_first in blocked:
                continue
            if first_action is None and not _action_allowed_from_obs(obs_agent, next_first):
                continue
            visited.add((nx, ny))
            queue.append((nx, ny, next_first, int(depth) + 1))

    if best is None:
        return None
    target_projection_reached = best[0][0] == 0
    if target_projection_reached or int(best[0][0]) < current_dist:
        return int(best[1])
    return None


def _signal_navigation_action_from_obs(
    obs_agent: dict,
    target: tuple[int, int],
    *,
    blocked_actions: set[int] | None = None,
) -> int | None:
    blocked = {int(action_id) for action_id in (blocked_actions or set())}
    pos_arr = np.asarray(obs_agent.get("self_pos", np.zeros((2,), dtype=np.int16)), dtype=np.int64).reshape(-1)
    if pos_arr.size < 2:
        return None
    x, y = int(pos_arr[0]), int(pos_arr[1])
    tx, ty = int(target[0]), int(target[1])
    if (x, y) == (tx, ty):
        return int(SyncOrSinkEnv.ACTION_INTERACT)
    local_action = _signal_local_navigation_action_from_obs(
        obs_agent,
        target,
        blocked_actions=blocked,
    )
    if local_action is not None and _action_allowed_from_obs(obs_agent, int(local_action)):
        return int(local_action)
    for action_id in _signal_navigation_action_order((x, y), target):
        if int(action_id) in blocked:
            continue
        if _action_allowed_from_obs(obs_agent, action_id):
            return int(action_id)
    return None


def _signal_navigation_destination(pos: tuple[int, int], action_id: int) -> tuple[int, int] | None:
    for candidate_action, dx, dy in _SIGNAL_NAV_ACTION_DELTAS:
        if int(candidate_action) == int(action_id):
            return int(pos[0]) + int(dx), int(pos[1]) + int(dy)
    return None


def _signal_memory_adjusted_navigation_action(
    obs_agent: dict,
    target: tuple[int, int],
    action_id: int,
    scan_state: Mapping[str, Any] | None,
    agent_id: int,
) -> int:
    pos = _signal_xy(obs_agent.get("self_pos"))
    if pos is None or scan_state is None or not isinstance(scan_state, MutableMapping):
        return int(action_id)
    nav_memory = scan_state.get("exact_target_navigation")
    if not isinstance(nav_memory, MutableMapping):
        nav_memory = {}
        scan_state["exact_target_navigation"] = nav_memory
    row = nav_memory.get(int(agent_id), nav_memory.get(str(int(agent_id))))
    adjusted = int(action_id)
    dest = _signal_navigation_destination(pos, int(action_id))
    previous_pos = None
    previous_target_matches = False
    if dest is not None and isinstance(row, Mapping):
        previous_target = _signal_xy(row.get("target"))
        row_previous_pos = _signal_xy(row.get("pos"))
        if previous_target == tuple(target) and row_previous_pos is not None:
            previous_pos = row_previous_pos
            previous_target_matches = True
    if previous_pos is None:
        prev_positions = scan_state.get("prev_positions")
        if isinstance(prev_positions, Mapping):
            previous_pos = _signal_xy(prev_positions.get(int(agent_id), prev_positions.get(str(int(agent_id)))))
    if dest is not None and previous_pos == dest and dest != tuple(target):
        order = _signal_navigation_action_order(pos, target)
        primary_blocked = bool(order) and not _action_allowed_from_obs(obs_agent, int(order[0]))
        if previous_target_matches or primary_blocked:
            alternative = _signal_navigation_action_from_obs(
                obs_agent,
                target,
                blocked_actions={int(action_id)},
            )
            if alternative is not None:
                adjusted = int(alternative)
    nav_memory[int(agent_id)] = {
        "target": [int(target[0]), int(target[1])],
        "pos": [int(pos[0]), int(pos[1])],
        "action": int(adjusted),
        "step": int(scan_state.get("step", 0)),
    }
    return int(adjusted)


def _signal_scan_activity_at_position(
    cfg: RecurrentConfig,
    scan_state: Mapping[str, Any] | None,
    *,
    agent_id: int,
    num_agents: int,
    pos: tuple[int, int],
) -> tuple[bool, bool]:
    if scan_state is None:
        return False, False
    scan_log = scan_state.get("scan_log") or {}
    scan_positions = scan_state.get("scan_pos") or {}
    if not isinstance(scan_log, Mapping) or not isinstance(scan_positions, Mapping):
        return False, False
    current_step = int(scan_state.get("step", 0))
    scan_window = int(scan_state.get("scan_window", getattr(cfg, "scan_window", 3)))
    self_active = False
    teammate_active = False
    for other_id in range(int(num_agents)):
        last_scan = _scan_log_value(dict(scan_log), int(other_id))
        if last_scan is None:
            continue
        age = current_step - int(last_scan)
        if age < 0 or age > scan_window:
            continue
        scan_pos = _signal_xy(scan_positions.get(int(other_id), scan_positions.get(str(int(other_id)))))
        if scan_pos != tuple(pos):
            continue
        if int(other_id) == int(agent_id):
            self_active = True
        else:
            teammate_active = True
    return self_active, teammate_active


def _signal_active_scan_positions(
    cfg: RecurrentConfig,
    scan_state: Mapping[str, Any] | None,
) -> set[tuple[int, int]]:
    if scan_state is None:
        return set()
    scan_log = scan_state.get("scan_log") or {}
    scan_positions = scan_state.get("scan_pos") or {}
    if not isinstance(scan_log, Mapping) or not isinstance(scan_positions, Mapping):
        return set()
    current_step = int(scan_state.get("step", 0))
    scan_window = int(scan_state.get("scan_window", getattr(cfg, "scan_window", 3)))
    agent_ids: set[int] = set()
    for raw_agent_id in set(scan_log.keys()) | set(scan_positions.keys()):
        try:
            agent_ids.add(int(raw_agent_id))
        except (TypeError, ValueError):
            continue
    active_positions: set[tuple[int, int]] = set()
    for aid in agent_ids:
        last_scan = _scan_log_value(dict(scan_log), int(aid))
        if last_scan is None:
            continue
        age = current_step - int(last_scan)
        if age < 0 or age > scan_window:
            continue
        scan_pos = _signal_xy(scan_positions.get(int(aid), scan_positions.get(str(int(aid)))))
        if scan_pos is not None:
            active_positions.add(scan_pos)
    return active_positions


def _signal_trusted_exact_message_targets(
    cfg: RecurrentConfig,
    obs_agent: dict,
    scan_state: Mapping[str, Any] | None,
    agent_id: int | None = None,
) -> set[tuple[int, int]]:
    trusted = _signal_base_trusted_exact_message_targets(cfg, obs_agent, scan_state)
    if agent_id is not None:
        trusted.update(_signal_exact_target_memory_targets(cfg, scan_state, int(agent_id)))
    return trusted


def _apply_signal_exact_target_navigation_assist(
    cfg: RecurrentConfig,
    obs: dict,
    acts: torch.Tensor,
    scan_state: Mapping[str, Any] | None,
) -> torch.Tensor:
    if (
        cfg.scenario != "signal_hunt"
        or not bool(getattr(cfg, "eval_signal_exact_target_navigation_assist", False))
        or not cfg.comm
    ):
        return acts
    _update_signal_exact_target_memory(cfg, obs, scan_state)
    corrected = acts.clone()
    for aid in range(int(acts.shape[0])):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        observed_map_size = _observed_map_size(obs_agent, cfg)
        exact_targets = _signal_exact_targets_from_observation(obs_agent, observed_map_size)
        exact_targets.update(_signal_exact_target_memory_targets(cfg, scan_state, int(aid)))
        trusted_targets = _signal_trusted_exact_message_targets(cfg, obs_agent, scan_state, agent_id=int(aid))
        candidate_targets = sorted(exact_targets & trusted_targets)
        if len(candidate_targets) != 1:
            continue
        target = candidate_targets[0]
        action_id = _signal_navigation_action_from_obs(obs_agent, target)
        if action_id is None or not _action_allowed_from_obs(obs_agent, action_id):
            continue
        if int(action_id) != int(SyncOrSinkEnv.ACTION_INTERACT):
            action_id = _signal_memory_adjusted_navigation_action(
                obs_agent,
                target,
                int(action_id),
                scan_state,
                int(aid),
            )
            if not _action_allowed_from_obs(obs_agent, int(action_id)):
                continue
        if int(action_id) == int(SyncOrSinkEnv.ACTION_INTERACT):
            self_active, teammate_active = _signal_scan_activity_at_position(
                cfg,
                scan_state,
                agent_id=int(aid),
                num_agents=int(acts.shape[0]),
                pos=target,
            )
            if (
                self_active
                and not teammate_active
                and _action_allowed_from_obs(obs_agent, SyncOrSinkEnv.ACTION_STAY)
            ):
                corrected[aid] = int(SyncOrSinkEnv.ACTION_STAY)
                continue
        corrected[aid] = int(action_id)
    return corrected


def _apply_signal_exact_target_message_guard(
    cfg: RecurrentConfig,
    obs: dict,
    actions: dict[int, dict],
    scan_state: Mapping[str, Any] | None,
) -> dict[int, dict]:
    corrected = {int(aid): dict(action) for aid, action in actions.items()}
    if (
        cfg.scenario != "signal_hunt"
        or not bool(getattr(cfg, "eval_signal_exact_target_message_guard", False))
        or not cfg.comm
    ):
        return corrected
    _update_signal_exact_target_memory(cfg, obs, scan_state)
    for aid in sorted(corrected):
        obs_agent = obs.get(aid, obs.get(str(aid))) if isinstance(obs, dict) else None
        if not isinstance(obs_agent, dict):
            continue
        tokens = corrected[int(aid)].get("message_tokens", [])
        observed_map_size = _observed_map_size(obs_agent, cfg)
        exact_targets = set(_signal_exact_targets_from_message(tokens, observed_map_size))
        if not exact_targets:
            continue
        trusted_targets = _signal_trusted_exact_message_targets(
            cfg,
            obs_agent,
            scan_state,
            agent_id=int(aid),
        )
        if exact_targets.issubset(trusted_targets):
            continue
        corrected[int(aid)]["message_tokens"] = []
    return corrected


def _decode_recurrent_actions(
    cfg: RecurrentConfig,
    model,
    obs,
    hidden,
    device,
    feedback: np.ndarray | None = None,
    scan_state: Mapping[str, Any] | None = None,
):
    env_agents = len(obs)
    obs_batch = _build_recurrent_obs_batch(obs, env_agents, cfg, feedback=feedback)
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        if cfg.comm:
            logits, send_logits, token_logits, len_logits, hidden = model(obs_tensor, hidden)
        else:
            logits, hidden = model(obs_tensor, hidden)
    logits = mask_action_logits(logits, action_mask_from_flat_obs(obs_tensor))
    acts = torch.argmax(logits, dim=-1)
    acts = _apply_signal_target_scan_decoding(cfg, obs, logits, acts)
    scan_gate_logits = None
    if hasattr(model, "signal_scan_gate"):
        with torch.no_grad():
            scan_gate_logits = model.signal_scan_gate(hidden[0])
    acts = _apply_signal_scan_gate_decoding(cfg, obs, acts, scan_gate_logits)
    target_decision_logits = None
    if hasattr(model, "signal_target_decision"):
        with torch.no_grad():
            target_decision_logits = model.signal_target_decision(hidden[0])
    acts = _apply_signal_target_decision_decoding(cfg, obs, acts, target_decision_logits)
    acts = _apply_signal_scan_sync_decoding(cfg, obs, acts, feedback, scan_state=scan_state)
    acts = _apply_signal_scan_refresh_decoding(cfg, obs, acts, feedback)
    acts = _apply_signal_exact_target_navigation_assist(cfg, obs, acts, scan_state)
    target_validity_logits = None
    if hasattr(model, "signal_target_validity"):
        with torch.no_grad():
            target_validity_logits = model.signal_target_validity(hidden[0])
    acts = _apply_signal_target_validity_decoding(cfg, obs, acts, target_validity_logits)

    if not cfg.comm:
        return {
            aid: {"action": int(acts[aid].item()), "message_tokens": []}
            for aid in range(env_agents)
        }, hidden

    send = (torch.sigmoid(send_logits.squeeze(-1)) > cfg.eval_send_threshold).long()
    token_samples = torch.argmax(token_logits, dim=-1)
    len_samples = torch.argmax(len_logits, dim=-1)
    actions = {}
    for aid in range(env_agents):
        msg_len = int(len_samples[aid].item()) if int(send[aid].item()) == 1 else 0
        if msg_len > 0:
            tokens = token_samples[aid][:msg_len].detach().cpu().tolist()
        else:
            tokens = []
        actions[aid] = {
            "action": int(acts[aid].item()),
            "message_tokens": [int(t) for t in tokens],
        }
    actions = _apply_signal_scan_broadcast_assist(cfg, actions, scan_state)
    actions = _apply_signal_exact_target_message_guard(cfg, obs, actions, scan_state)
    return actions, hidden


_SIGNAL_EVAL_METRIC_KEYS = (
    "decoy_scans",
    "clues_found",
    "target_scans",
    "solo_target_scans",
    "redundant_target_scans",
    "unique_target_scanners",
    "target_tile_visits",
    "true_target_visits",
    "decoy_target_visits",
    "true_target_unscanned_visits",
    "decoy_target_unscanned_visits",
    "target_first_scan_opportunities",
    "target_first_scan_misses",
    "target_refresh_scan_opportunities",
    "target_refresh_scan_misses",
    "target_joint_completion_opportunities",
    "target_joint_completion_misses",
    "target_redundant_active_scan_opportunities",
    "target_redundant_active_scans",
    "wrong_target_scans",
    "target_scan_obs_candidate",
    "target_scan_obs_rejected",
    "wrong_scan_obs_candidate",
    "wrong_scan_obs_rejected",
    "wrong_scan_obs_uncertain",
    "reached_any_target",
    "reached_true_target",
    "reached_decoy_target",
    "no_target_reached",
    "true_target_reached_without_scan",
    "decoy_target_reached_without_scan",
    "wrong_target_scanned",
)


def _summarize_signal_eval_rows(signal_rows: list[dict[str, float]]) -> dict[str, float]:
    if not signal_rows:
        return {}
    keys = list(_SIGNAL_EVAL_METRIC_KEYS)
    for row in signal_rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return {
        f"avg_{key}": float(np.mean([float(row.get(key, 0.0)) for row in signal_rows]))
        for key in keys
    }


def _evaluate_recurrent_policy_single_map(cfg: RecurrentConfig, model, device) -> dict:
    env = _build_env(cfg)
    model.eval()
    stats: list[EpisodeStats] = []
    signal_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for ep in range(cfg.eval_episodes):
            obs, info = env.reset(seed=cfg.eval_seed + ep)
            hidden = model.init_hidden(env.num_agents, device)
            done = False
            truncated = False
            total_reward = 0.0
            steps = 0
            comm_tokens = 0
            per_agent_reward = {i: 0.0 for i in range(env.num_agents)}
            per_agent_comm = {i: 0 for i in range(env.num_agents)}
            last_info = {}
            prev_actions: dict[int, int] = {}
            prev_msg_lens: dict[int, int] = {}
            scan_state = _initial_signal_scan_state(cfg)
            has_policy_step = False
            prev_positions: dict[int, tuple[int, int]] = {}
            signal_ep = {
                "decoy_scans": 0.0,
                "clues_found": 0.0,
                "target_scans": 0.0,
                "solo_target_scans": 0.0,
                "redundant_target_scans": 0.0,
                "unique_target_scanners": 0.0,
                "target_tile_visits": 0.0,
                "true_target_visits": 0.0,
                "decoy_target_visits": 0.0,
                "true_target_unscanned_visits": 0.0,
                "decoy_target_unscanned_visits": 0.0,
                "target_first_scan_opportunities": 0.0,
                "target_first_scan_misses": 0.0,
                "target_refresh_scan_opportunities": 0.0,
                "target_refresh_scan_misses": 0.0,
                "target_joint_completion_opportunities": 0.0,
                "target_joint_completion_misses": 0.0,
                "target_redundant_active_scan_opportunities": 0.0,
                "target_redundant_active_scans": 0.0,
                "wrong_target_scans": 0.0,
                "target_scan_obs_candidate": 0.0,
                "target_scan_obs_rejected": 0.0,
                "wrong_scan_obs_candidate": 0.0,
                "wrong_scan_obs_rejected": 0.0,
                "wrong_scan_obs_uncertain": 0.0,
            }
            target_scanners = set()
            target_scan_steps: dict[int, int] = {}
            reached_true_target = False
            reached_decoy_target = False

            while not (done or truncated):
                has_policy_step = _update_signal_scan_state_from_info(
                    cfg,
                    scan_state,
                    last_info,
                    env.num_agents,
                    prev_positions,
                    has_policy_step=has_policy_step,
                )
                feedback = _feedback_matrix(
                    cfg,
                    env.num_agents,
                    prev_actions=prev_actions,
                    prev_msg_lens=prev_msg_lens,
                    info=last_info,
                    env=env,
                    obs=obs,
                )
                actions, hidden = _decode_recurrent_actions(
                    cfg,
                    model,
                    obs,
                    hidden,
                    device,
                    feedback=feedback,
                    scan_state=scan_state,
                )
                if cfg.scenario == "signal_hunt":
                    target = env.scenario_state.data.get("target")
                    if target is not None:
                        target_pos = tuple(target)
                        decoys = {tuple(pos) for pos in env.scenario_state.data.get("decoys", [])}
                        for aid, action in actions.items():
                            action_id = int(action.get("action", -1))
                            pos = tuple(env.agent_positions[int(aid)])
                            obs_agent = obs.get(int(aid), obs.get(str(int(aid)), {}))
                            on_true_target = pos == target_pos
                            on_decoy_target = pos in decoys
                            if on_true_target or on_decoy_target:
                                signal_ep["target_tile_visits"] += 1.0
                            if on_true_target:
                                reached_true_target = True
                                signal_ep["true_target_visits"] += 1.0
                                scan_kind = _signal_target_scan_kind(env, int(aid))
                                if scan_kind == _SIGNAL_TARGET_SCAN_KIND_FIRST:
                                    signal_ep["target_first_scan_opportunities"] += 1.0
                                    if action_id != env.ACTION_INTERACT:
                                        signal_ep["target_first_scan_misses"] += 1.0
                                elif scan_kind == _SIGNAL_TARGET_SCAN_KIND_REFRESH:
                                    signal_ep["target_refresh_scan_opportunities"] += 1.0
                                    if action_id != env.ACTION_INTERACT:
                                        signal_ep["target_refresh_scan_misses"] += 1.0
                                elif scan_kind == _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION:
                                    signal_ep["target_joint_completion_opportunities"] += 1.0
                                    if action_id != env.ACTION_INTERACT:
                                        signal_ep["target_joint_completion_misses"] += 1.0
                                elif scan_kind == _SIGNAL_TARGET_SCAN_KIND_REDUNDANT_ACTIVE:
                                    signal_ep["target_redundant_active_scan_opportunities"] += 1.0
                                    if action_id == env.ACTION_INTERACT:
                                        signal_ep["target_redundant_active_scans"] += 1.0
                                if action_id != env.ACTION_INTERACT:
                                    signal_ep["true_target_unscanned_visits"] += 1.0
                            if on_decoy_target:
                                reached_decoy_target = True
                                signal_ep["decoy_target_visits"] += 1.0
                                if action_id == env.ACTION_INTERACT:
                                    signal_ep["wrong_target_scans"] += 1.0
                                    if _signal_center_rejected_target(obs_agent, int(env.map_size)):
                                        signal_ep["wrong_scan_obs_rejected"] += 1.0
                                    elif _signal_center_target_scan_decoding_candidate(obs_agent, cfg):
                                        signal_ep["wrong_scan_obs_candidate"] += 1.0
                                    else:
                                        signal_ep["wrong_scan_obs_uncertain"] += 1.0
                                else:
                                    signal_ep["decoy_target_unscanned_visits"] += 1.0
                        target_interactors = [
                            int(aid)
                            for aid, action in actions.items()
                            if (
                                int(action.get("action", -1)) == env.ACTION_INTERACT
                                and tuple(env.agent_positions[int(aid)]) == tuple(target)
                            )
                        ]
                        if len(target_interactors) == 1:
                            signal_ep["solo_target_scans"] += 1.0
                        next_step = int(env.steps) + 1
                        scan_window = int(
                            env.scenario_state.data.get(
                                "scan_window",
                                getattr(env.config, "scan_window", cfg.scan_window),
                            )
                        )
                        for aid in target_interactors:
                            last_scan = target_scan_steps.get(int(aid))
                            if last_scan is not None and next_step - int(last_scan) <= scan_window:
                                signal_ep["redundant_target_scans"] += 1.0
                        for aid, action in actions.items():
                            if (
                                int(action.get("action", -1)) == env.ACTION_INTERACT
                                and tuple(env.agent_positions[int(aid)]) == tuple(target)
                            ):
                                obs_agent = obs.get(int(aid), obs.get(str(int(aid)), {}))
                                signal_ep["target_scans"] += 1.0
                                if _signal_center_target_scan_decoding_candidate(obs_agent, cfg):
                                    signal_ep["target_scan_obs_candidate"] += 1.0
                                if _signal_center_rejected_target(obs_agent, int(env.map_size)):
                                    signal_ep["target_scan_obs_rejected"] += 1.0
                                target_scanners.add(int(aid))
                                target_scan_steps[int(aid)] = next_step
                prev_positions = _signal_positions_from_obs(obs)
                obs, rewards, done, truncated, info = env.step(actions)
                last_info = info or {}
                if cfg.scenario == "signal_hunt":
                    for names in _event_names_by_agent(last_info, env.num_agents).values():
                        if "decoy_scan" in names:
                            signal_ep["decoy_scans"] += 1.0
                        if "clue_found" in names:
                            signal_ep["clues_found"] += 1.0
                prev_actions = {aid: int(action["action"]) for aid, action in actions.items()}
                prev_msg_lens = _message_lengths(actions)
                total_reward += sum(rewards.values())
                steps += 1
                for aid, reward in rewards.items():
                    per_agent_reward[aid] += reward
                if "comm_tokens" in last_info:
                    comm_tokens += sum(last_info["comm_tokens"].values())
                    for aid, count in last_info["comm_tokens"].items():
                        per_agent_comm[aid] += count

            stats.append(EpisodeStats(
                total_reward=total_reward,
                steps=steps,
                success=episode_success(cfg.scenario, done, last_info),
                comm_tokens=comm_tokens,
                per_agent_reward=per_agent_reward,
                per_agent_comm=per_agent_comm,
            ))
            if cfg.scenario == "signal_hunt":
                signal_ep["unique_target_scanners"] = float(len(target_scanners))
                reached_any_target = reached_true_target or reached_decoy_target
                signal_ep["reached_any_target"] = 1.0 if reached_any_target else 0.0
                signal_ep["reached_true_target"] = 1.0 if reached_true_target else 0.0
                signal_ep["reached_decoy_target"] = 1.0 if reached_decoy_target else 0.0
                signal_ep["no_target_reached"] = 0.0 if reached_any_target else 1.0
                signal_ep["true_target_reached_without_scan"] = (
                    1.0 if reached_true_target and signal_ep["target_scans"] <= 0.0 else 0.0
                )
                signal_ep["decoy_target_reached_without_scan"] = (
                    1.0 if reached_decoy_target and signal_ep["wrong_target_scans"] <= 0.0 else 0.0
                )
                signal_ep["wrong_target_scanned"] = 1.0 if signal_ep["wrong_target_scans"] > 0.0 else 0.0
                signal_rows.append(signal_ep)
    model.train()
    summary = summarize(stats)
    result = {
        "episodes": summary.episodes,
        "success_rate": summary.success_rate,
        "avg_return": summary.avg_return,
        "avg_steps": summary.avg_steps,
        "avg_comm_tokens": summary.avg_comm_tokens,
        "avg_agent_reward": summary.avg_agent_reward,
        "avg_agent_comm": summary.avg_agent_comm,
    }
    if signal_rows:
        result["signal"] = _summarize_signal_eval_rows(signal_rows)
    return result


def _weighted_eval_average(rows: list[dict], key: str, total_episodes: int) -> float:
    if total_episodes <= 0:
        return 0.0
    return float(sum(float(row.get(key, 0.0)) * int(row.get("episodes", 0)) for row in rows) / total_episodes)


def _weighted_eval_dict(rows: list[dict], key: str, total_episodes: int) -> dict:
    if total_episodes <= 0:
        return {}
    item_keys = set()
    for row in rows:
        item_keys.update((row.get(key) or {}).keys())
    return {
        item: float(
            sum(float((row.get(key) or {}).get(item, 0.0)) * int(row.get("episodes", 0)) for row in rows)
            / total_episodes
        )
        for item in sorted(item_keys, key=lambda value: int(value) if str(value).isdigit() else str(value))
    }


def _aggregate_recurrent_metric_rows(rows: list[dict]) -> dict:
    total_episodes = int(sum(int(row.get("episodes", 0)) for row in rows))
    result = {
        "episodes": total_episodes,
        "success_rate": _weighted_eval_average(rows, "success_rate", total_episodes),
        "avg_return": _weighted_eval_average(rows, "avg_return", total_episodes),
        "avg_steps": _weighted_eval_average(rows, "avg_steps", total_episodes),
        "avg_comm_tokens": _weighted_eval_average(rows, "avg_comm_tokens", total_episodes),
        "avg_agent_reward": _weighted_eval_dict(rows, "avg_agent_reward", total_episodes),
        "avg_agent_comm": _weighted_eval_dict(rows, "avg_agent_comm", total_episodes),
    }
    signal_rows = [row.get("signal") or {} for row in rows if row.get("signal")]
    if signal_rows:
        result["signal"] = {
            key: _weighted_eval_average(
                [{"episodes": row.get("episodes", 0), key: (row.get("signal") or {}).get(key, 0.0)} for row in rows],
                key,
                total_episodes,
            )
            for key in sorted({key for signal in signal_rows for key in signal.keys()})
        }
    return result


def _aggregate_recurrent_eval_rows(rows: list[dict]) -> dict:
    result = _aggregate_recurrent_metric_rows(rows)
    result["eval_map_sizes"] = {str(row["map_size"]): row for row in rows}
    return result


def evaluate_recurrent_policy(cfg: RecurrentConfig, model, device) -> dict:
    map_sizes = _eval_map_sizes(cfg)
    if len(map_sizes) == 1:
        return _evaluate_recurrent_policy_single_map(_cfg_for_map_size(cfg, int(map_sizes[0])), model, device)
    rows = []
    for idx, map_size in enumerate(map_sizes):
        row_cfg = replace(
            _cfg_for_map_size(cfg, int(map_size)),
            eval_seed=int(cfg.eval_seed) + idx * 10000,
        )
        row = _evaluate_recurrent_policy_single_map(row_cfg, model, device)
        row["map_size"] = int(map_size)
        rows.append(row)
    return _aggregate_recurrent_eval_rows(rows)


def evaluate_recurrent_policy_multi_seed(
    cfg: RecurrentConfig,
    model,
    device,
    *,
    seed_count: int,
) -> dict:
    seed_count = max(1, int(seed_count))
    if seed_count == 1:
        result = evaluate_recurrent_policy(cfg, model, device)
        result["eval_seed_count"] = 1
        result["eval_seeds"] = [int(cfg.eval_seed)]
        return result

    rows = []
    seed_stride = max(10000, int(cfg.eval_episodes) * max(1, len(_eval_map_sizes(cfg))) * 10)
    for seed_idx in range(seed_count):
        seed = int(cfg.eval_seed) + seed_idx * seed_stride
        row = evaluate_recurrent_policy(replace(cfg, eval_seed=seed), model, device)
        row["eval_seed"] = seed
        rows.append(row)

    result = _aggregate_recurrent_metric_rows(rows)
    result["eval_seed_count"] = seed_count
    result["eval_seeds"] = [int(row["eval_seed"]) for row in rows]
    map_sizes = sorted(
        {str(size) for row in rows for size in (row.get("eval_map_sizes") or {}).keys()},
        key=lambda value: int(value) if value.isdigit() else value,
    )
    if map_sizes:
        result["eval_map_sizes"] = {}
        for map_size in map_sizes:
            map_rows = [
                row["eval_map_sizes"][map_size]
                for row in rows
                if map_size in (row.get("eval_map_sizes") or {})
            ]
            if map_rows:
                map_result = _aggregate_recurrent_metric_rows(map_rows)
                map_result["map_size"] = int(map_size) if map_size.isdigit() else map_size
                result["eval_map_sizes"][map_size] = map_result
    return result


def _recurrent_eval_score(result: dict) -> tuple[float, ...]:
    signal = result.get("signal") or {}
    return (
        float(result.get("success_rate", 0.0)),
        -float(signal.get("avg_decoy_scans", 0.0)),
        -float(signal.get("avg_redundant_target_scans", 0.0)),
        float(result.get("avg_return", 0.0)),
        -float(result.get("avg_steps", 0.0)),
    )


def _recurrent_eval_wandb_payload(
    result: dict,
    *,
    update: int,
    is_best: bool,
    best_eval: dict | None = None,
    prefix: str = "eval",
) -> dict:
    prefix = str(prefix).strip("/")
    payload = {
        f"{prefix}/success_rate": float(result["success_rate"]),
        f"{prefix}/mean_return": float(result["avg_return"]),
        f"{prefix}/mean_steps": float(result["avg_steps"]),
        f"{prefix}/mean_comm_tokens": float(result.get("avg_comm_tokens", 0.0)),
        f"{prefix}/episodes": int(result.get("episodes", 0)),
        f"{prefix}/seed_count": int(result.get("eval_seed_count", 1)),
        f"{prefix}/update": int(update),
        f"{prefix}/is_best": int(is_best),
    }
    if best_eval is not None:
        payload[f"{prefix}/best_success_rate"] = float(best_eval.get("success_rate", 0.0))
        payload[f"{prefix}/best_mean_return"] = float(best_eval.get("avg_return", 0.0))
        payload[f"{prefix}/best_update"] = int(best_eval.get("update", update))
    signal = result.get("signal") or {}
    for key, value in signal.items():
        payload[f"{prefix}/signal/{key}"] = float(value)
    for raw_size, row in (result.get("eval_map_sizes") or {}).items():
        map_size = str(raw_size)
        map_prefix = f"{prefix}/map_{map_size}"
        payload[f"{map_prefix}/success_rate"] = float(row.get("success_rate", 0.0))
        payload[f"{map_prefix}/mean_return"] = float(row.get("avg_return", 0.0))
        payload[f"{map_prefix}/mean_steps"] = float(row.get("avg_steps", 0.0))
        payload[f"{map_prefix}/mean_comm_tokens"] = float(row.get("avg_comm_tokens", 0.0))
        for key, value in (row.get("signal") or {}).items():
            payload[f"{map_prefix}/signal/{key}"] = float(value)
    return payload


def _init_recurrent_wandb(cfg: RecurrentConfig):
    if not cfg.wandb:
        return None
    try:
        import wandb

        return wandb.init(project=cfg.wandb_project, name=cfg.wandb_run, config=vars(cfg))
    except Exception as exc:
        print(f"wandb init failed, continuing without wandb: {exc}")
        return None


def _wandb_log(wandb_run, payload: dict, *, context: str):
    if wandb_run is None:
        return
    try:
        wandb_run.log(payload)
    except Exception as exc:
        print(f"wandb {context} failed: {exc}")


def _recurrent_rl_best_path(cfg: RecurrentConfig) -> str | None:
    if not cfg.rl_save_best:
        return None
    if cfg.rl_best_save:
        return cfg.rl_best_save
    if not cfg.save:
        return None
    path = Path(cfg.save)
    suffix = path.suffix or ".pt"
    return str(path.with_name(f"{path.stem}_best{suffix}"))


def _save_recurrent_rl_checkpoint(
    path: str | Path,
    *,
    model: MAPPORecurrentActor,
    critic: MAPPOCritic | None,
    cfg: RecurrentConfig,
    obs_dim: int,
    best_eval: dict | None,
    final_eval: dict | None,
    restored_best: bool,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "recurrent_bc_rl",
        "model": model.state_dict(),
        "config": vars(cfg),
        "obs_dim": obs_dim,
        "best_eval": best_eval,
        "final_eval": final_eval,
        "restored_best": restored_best,
    }
    if critic is not None:
        payload["critic"] = critic.state_dict()
    torch.save(payload, path)


class RecurrentCheckpointPolicy:
    def __init__(self, model: MAPPORecurrentActor, cfg: RecurrentConfig, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.hidden = None
        self.prev_actions: dict[int, int] = {}
        self.prev_msg_lens: dict[int, int] = {}
        self.prev_positions: dict[int, tuple[int, int]] = {}
        self.scan_state = self._initial_scan_state()
        self._has_policy_step = False
        self.model.eval()

    def _initial_scan_state(self) -> dict:
        return _initial_signal_scan_state(self.cfg)

    def reset(self, *args, **kwargs):
        del args, kwargs
        self.hidden = None
        self.prev_actions = {}
        self.prev_msg_lens = {}
        self.prev_positions = {}
        self.scan_state = self._initial_scan_state()
        self._has_policy_step = False
        return None

    def metadata(self) -> dict:
        return {
            "algorithm": "recurrent_bc",
            "comm": self.cfg.comm,
            "hidden_dim": self.cfg.hidden_dim,
            "eval_send_threshold": self.cfg.eval_send_threshold,
            "eval_signal_target_scan_threshold": self.cfg.eval_signal_target_scan_threshold,
            "eval_signal_scan_gate_threshold": self.cfg.eval_signal_scan_gate_threshold,
            "eval_signal_scan_gate_suppress": self.cfg.eval_signal_scan_gate_suppress,
            "eval_signal_target_validity_threshold": self.cfg.eval_signal_target_validity_threshold,
            "eval_signal_target_decision_threshold": self.cfg.eval_signal_target_decision_threshold,
            "eval_signal_target_decision_suppress": self.cfg.eval_signal_target_decision_suppress,
            "eval_signal_scan_sync_assist": self.cfg.eval_signal_scan_sync_assist,
            "eval_signal_scan_sync_force_first": self.cfg.eval_signal_scan_sync_force_first,
            "eval_signal_scan_broadcast_assist": self.cfg.eval_signal_scan_broadcast_assist,
            "eval_signal_exact_target_message_guard": self.cfg.eval_signal_exact_target_message_guard,
            "eval_signal_exact_target_navigation_assist": self.cfg.eval_signal_exact_target_navigation_assist,
            "eval_signal_exact_target_memory_steps": self.cfg.eval_signal_exact_target_memory_steps,
            "eval_signal_scan_refresh_assist": self.cfg.eval_signal_scan_refresh_assist,
            "eval_signal_scan_refresh_threshold": self.cfg.eval_signal_scan_refresh_threshold,
            "obs_signal_scan_state": self.cfg.obs_signal_scan_state,
            "obs_signal_negative_memory": self.cfg.obs_signal_negative_memory,
        }

    def _update_scan_state_from_info(self, info: dict, num_agents: int) -> None:
        self._has_policy_step = _update_signal_scan_state_from_info(
            self.cfg,
            self.scan_state,
            info,
            num_agents,
            self.prev_positions,
            has_policy_step=self._has_policy_step,
        )

    def __call__(self, obs: dict, info: dict, state: dict) -> dict[int, dict]:
        del state
        if self.hidden is None:
            self.hidden = self.model.init_hidden(len(obs), self.device)
        self._update_scan_state_from_info(info or {}, len(obs))
        if isinstance(self.scan_state, MutableMapping):
            self.scan_state["prev_positions"] = {
                int(aid): [int(pos[0]), int(pos[1])]
                for aid, pos in self.prev_positions.items()
            }
        feedback = _feedback_matrix(
            self.cfg,
            len(obs),
            prev_actions=self.prev_actions,
            prev_msg_lens=self.prev_msg_lens,
            info=info,
            scan_state=self.scan_state,
            obs=obs,
        )
        actions, self.hidden = _decode_recurrent_actions(
            self.cfg,
            self.model,
            obs,
            self.hidden,
            self.device,
            feedback=feedback,
            scan_state=self.scan_state,
        )
        self.prev_actions = {aid: int(action["action"]) for aid, action in actions.items()}
        self.prev_msg_lens = _message_lengths(actions)
        self.prev_positions = _signal_positions_from_obs(obs)
        return actions


def load_recurrent_checkpoint_policy(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    eval_send_threshold: float | None = None,
    eval_signal_scan_gate_threshold: float | None = None,
    eval_signal_scan_gate_suppress: bool | None = None,
    eval_signal_target_validity_threshold: float | None = None,
    eval_signal_target_decision_threshold: float | None = None,
    eval_signal_target_decision_suppress: bool | None = None,
    eval_signal_scan_sync_assist: bool | None = None,
    eval_signal_scan_sync_force_first: bool | None = None,
    eval_signal_scan_broadcast_assist: bool | None = None,
    eval_signal_exact_target_message_guard: bool | None = None,
    eval_signal_exact_target_navigation_assist: bool | None = None,
    eval_signal_exact_target_memory_steps: int | None = None,
    eval_signal_scan_refresh_assist: bool | None = None,
    eval_signal_scan_refresh_threshold: float | None = None,
) -> RecurrentCheckpointPolicy:
    device = resolve_device(str(device)) if isinstance(device, str) else device
    ckpt = torch.load(Path(path), map_location="cpu")
    raw_cfg = ckpt.get("config", {})
    allowed = {field.name for field in fields(RecurrentConfig)}
    cfg = RecurrentConfig(**{key: value for key, value in raw_cfg.items() if key in allowed})
    if eval_send_threshold is not None:
        cfg.eval_send_threshold = float(eval_send_threshold)
    if eval_signal_scan_gate_threshold is not None:
        cfg.eval_signal_scan_gate_threshold = float(eval_signal_scan_gate_threshold)
    if eval_signal_scan_gate_suppress is not None:
        cfg.eval_signal_scan_gate_suppress = bool(eval_signal_scan_gate_suppress)
    if eval_signal_target_validity_threshold is not None:
        cfg.eval_signal_target_validity_threshold = float(eval_signal_target_validity_threshold)
    if eval_signal_target_decision_threshold is not None:
        cfg.eval_signal_target_decision_threshold = float(eval_signal_target_decision_threshold)
    if eval_signal_target_decision_suppress is not None:
        cfg.eval_signal_target_decision_suppress = bool(eval_signal_target_decision_suppress)
    if eval_signal_scan_sync_assist is not None:
        cfg.eval_signal_scan_sync_assist = bool(eval_signal_scan_sync_assist)
    if eval_signal_scan_sync_force_first is not None:
        cfg.eval_signal_scan_sync_force_first = bool(eval_signal_scan_sync_force_first)
    if eval_signal_scan_broadcast_assist is not None:
        cfg.eval_signal_scan_broadcast_assist = bool(eval_signal_scan_broadcast_assist)
    if eval_signal_exact_target_message_guard is not None:
        cfg.eval_signal_exact_target_message_guard = bool(eval_signal_exact_target_message_guard)
    if eval_signal_exact_target_navigation_assist is not None:
        cfg.eval_signal_exact_target_navigation_assist = bool(eval_signal_exact_target_navigation_assist)
    if eval_signal_exact_target_memory_steps is not None:
        cfg.eval_signal_exact_target_memory_steps = int(eval_signal_exact_target_memory_steps)
    if eval_signal_scan_refresh_assist is not None:
        cfg.eval_signal_scan_refresh_assist = bool(eval_signal_scan_refresh_assist)
    if eval_signal_scan_refresh_threshold is not None:
        cfg.eval_signal_scan_refresh_threshold = float(eval_signal_scan_refresh_threshold)
    state = ckpt.get("model", ckpt)
    first_weight = state.get("encoder.net.0.weight")
    if first_weight is None:
        raise ValueError(f"checkpoint {path} does not contain recurrent actor encoder weights")
    obs_dim = int(first_weight.shape[1])
    model = MAPPORecurrentActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    ).to(device)
    _load_recurrent_actor_state(model, state)
    return RecurrentCheckpointPolicy(model, cfg, device)


def _recurrent_training_obs_shape(cfg: RecurrentConfig) -> tuple[int, int]:
    env, sample_cfg = _build_training_env(cfg, 0)
    num_agents = env.num_agents
    sample_obs, _ = env.reset(seed=0)
    obs_dim = _build_recurrent_obs_batch(
        sample_obs,
        num_agents,
        sample_cfg,
        feedback=_feedback_matrix(sample_cfg, num_agents),
    ).shape[1]
    for idx, _size in enumerate(_training_map_sizes(cfg)):
        check_env, check_cfg = _build_training_env(cfg, idx)
        check_obs, _ = check_env.reset(seed=idx)
        check_dim = _build_recurrent_obs_batch(
            check_obs,
            check_env.num_agents,
            check_cfg,
            feedback=_feedback_matrix(check_cfg, check_env.num_agents),
        ).shape[1]
        if check_dim != obs_dim:
            raise ValueError(
                "train_map_sizes produce different recurrent observation dimensions; "
                "use --obs-memory-mode egocentric when training with full exploration memory"
            )
    return obs_dim, num_agents


def _expand_recurrent_actor_input_state(
    state: dict,
    *,
    checkpoint_obs_dim: int,
    expected_obs_dim: int,
) -> dict:
    if expected_obs_dim <= checkpoint_obs_dim:
        raise ValueError(
            f"cannot expand recurrent actor obs_dim from {checkpoint_obs_dim} to {expected_obs_dim}"
        )
    first_weight = state.get("encoder.net.0.weight")
    if first_weight is None:
        raise ValueError("checkpoint does not contain recurrent actor encoder weights")
    if int(first_weight.shape[1]) != int(checkpoint_obs_dim):
        raise ValueError(
            "checkpoint encoder input width does not match checkpoint_obs_dim: "
            f"{int(first_weight.shape[1])} != {int(checkpoint_obs_dim)}"
        )
    if checkpoint_obs_dim < 8:
        raise ValueError("cannot expand recurrent actor checkpoint with fewer than 8 action-mask columns")

    expanded = dict(state)
    extra_dim = int(expected_obs_dim) - int(checkpoint_obs_dim)
    pad = torch.zeros(
        (first_weight.shape[0], extra_dim),
        dtype=first_weight.dtype,
        device=first_weight.device,
    )
    expanded["encoder.net.0.weight"] = torch.cat(
        [first_weight[:, :-8], pad, first_weight[:, -8:]],
        dim=1,
    )
    return expanded


def _load_recurrent_actor_state(model: MAPPORecurrentActor, state: dict) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = (
        "signal_scan_gate.",
        "signal_target_validity.",
        "signal_target_decision.",
        "signal_target_aux.",
    )
    missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "recurrent actor checkpoint state mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def load_recurrent_actor_checkpoint(
    path: str | Path,
    cfg: RecurrentConfig,
    device: str | torch.device = "cpu",
) -> MAPPORecurrentActor:
    """Load a recurrent actor checkpoint for further BC/RL training."""
    device = resolve_device(str(device)) if isinstance(device, str) else device
    ckpt = torch.load(Path(path), map_location="cpu")
    raw_cfg = ckpt.get("config", {})
    allowed = {field.name for field in fields(RecurrentConfig)}
    checkpoint_cfg = RecurrentConfig(**{key: value for key, value in raw_cfg.items() if key in allowed})
    for key in ("hidden_dim", "comm", "comm_token_limit", "comm_vocab_size"):
        if getattr(checkpoint_cfg, key) != getattr(cfg, key):
            raise ValueError(
                f"recurrent init checkpoint {path} has {key}={getattr(checkpoint_cfg, key)!r}, "
                f"but current config has {key}={getattr(cfg, key)!r}"
            )
    state = ckpt.get("model", ckpt)
    first_weight = state.get("encoder.net.0.weight")
    if first_weight is None:
        raise ValueError(f"checkpoint {path} does not contain recurrent actor encoder weights")
    checkpoint_obs_dim = int(first_weight.shape[1])
    expected_obs_dim, _num_agents = _recurrent_training_obs_shape(cfg)
    if checkpoint_obs_dim != expected_obs_dim:
        if cfg.recurrent_init_allow_obs_dim_mismatch and expected_obs_dim > checkpoint_obs_dim:
            state = _expand_recurrent_actor_input_state(
                state,
                checkpoint_obs_dim=checkpoint_obs_dim,
                expected_obs_dim=expected_obs_dim,
            )
            print(
                "Expanded recurrent init checkpoint input width "
                f"from {checkpoint_obs_dim} to {expected_obs_dim}; "
                "new observation columns are zero-initialized before the action-mask tail."
            )
            checkpoint_obs_dim = expected_obs_dim
        else:
            raise ValueError(
                f"recurrent init checkpoint {path} obs_dim={checkpoint_obs_dim} does not match "
                f"current training obs_dim={expected_obs_dim}; check map/observation flags"
            )
    model = MAPPORecurrentActor(
        obs_dim=checkpoint_obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    ).to(device)
    _load_recurrent_actor_state(model, state)
    return model


def _balanced_step_counts(total_steps: int, bucket_count: int) -> list[int]:
    total_steps = max(0, int(total_steps))
    bucket_count = max(1, int(bucket_count))
    base = total_steps // bucket_count
    remainder = total_steps % bucket_count
    return [base + (1 if idx < remainder else 0) for idx in range(bucket_count)]


def _balanced_rollout_step_counts_for_maps(cfg: RecurrentConfig, map_sizes: list[int]) -> list[int]:
    map_sizes = [int(map_size) for map_size in map_sizes]
    if not map_sizes:
        return []
    default_counts = _balanced_step_counts(cfg.rollout_steps, len(map_sizes))
    overrides = _parse_map_step_overrides(cfg.rl_rollout_map_steps, field_name="rl_rollout_map_steps")
    if not overrides:
        return default_counts
    unknown_maps = sorted(set(overrides) - set(map_sizes))
    if unknown_maps:
        raise ValueError(
            "rl_rollout_map_steps contains map sizes not present in train_map_sizes: "
            f"{unknown_maps}"
        )
    return [int(overrides.get(map_size, default_counts[idx])) for idx, map_size in enumerate(map_sizes)]


def _bootstrap_recurrent_value(
    cfg: RecurrentConfig,
    critic: MAPPOCritic,
    obs: dict,
    device,
    *,
    prev_actions: dict[int, int],
    prev_msg_lens: dict[int, int],
    info: dict,
    env: SyncOrSinkEnv | None = None,
) -> torch.Tensor:
    feedback = _feedback_matrix(
        cfg,
        len(obs),
        prev_actions=prev_actions,
        prev_msg_lens=prev_msg_lens,
        info=info,
        env=env,
        obs=obs,
    )
    obs_tensor = torch.tensor(
        _build_recurrent_obs_batch(obs, len(obs), cfg, feedback=feedback),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        return critic(obs_tensor).cpu()


def _collect_recurrent_rl_rollout(
    cfg: RecurrentConfig,
    model: MAPPORecurrentActor,
    critic: MAPPOCritic,
    device,
    *,
    update: int,
    num_agents: int,
) -> dict:
    obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
    rew_buf, done_buf, reset_after_buf, bootstrap_val_buf = [], [], [], []
    send_buf, token_buf, len_buf = [], [], []
    hidden_buf = []
    ep_returns, ep_steps = [], []
    ep_comm = []
    comm_send_counts = 0
    comm_total_steps = 0
    comm_len_sum = 0.0
    comm_len_count = 0
    comm_token_entropy_sum = 0.0
    action_hist = np.zeros(8, dtype=np.int64)
    map_step_counts: dict[str, int] = {}
    total_collected_steps = 0
    redundant_target_scan_count = 0
    redundant_target_scan_penalty_sum = 0.0
    wrong_target_scan_count = 0
    wrong_target_scan_penalty_sum = 0.0

    train_map_sizes = _training_map_sizes(cfg)
    balanced = bool(cfg.rl_balanced_rollouts and len(train_map_sizes) > 1)
    if balanced:
        step_counts = _balanced_rollout_step_counts_for_maps(cfg, train_map_sizes)
        segments = [
            {
                "mode": "fixed_map",
                "map_size": int(map_size),
                "steps": int(steps),
                "seed": int(update) * 100000 + idx * 10000,
            }
            for idx, (map_size, steps) in enumerate(zip(train_map_sizes, step_counts))
            if int(steps) > 0
        ]
    else:
        segments = [{
            "mode": "cycling_maps",
            "episode_idx": int(update),
            "steps": int(cfg.rollout_steps),
            "seed": int(update),
        }]

    for segment_idx, segment in enumerate(segments):
        if segment["mode"] == "fixed_map":
            active_cfg = _cfg_for_map_size(cfg, int(segment["map_size"]))
            env = _build_env(active_cfg)
            rollout_episode_idx = segment_idx
        else:
            rollout_episode_idx = int(segment["episode_idx"])
            env, active_cfg = _build_training_env(cfg, rollout_episode_idx)

        obs, _info = env.reset(seed=int(segment["seed"]))
        hidden = model.init_hidden(num_agents, device)
        prev_actions: dict[int, int] = {}
        prev_msg_lens: dict[int, int] = {}
        last_info: dict = {}
        ep_ret, ep_step = 0.0, 0
        ep_comm_tokens = 0
        resets_in_segment = 0

        for local_t in range(int(segment["steps"])):
            feedback = _feedback_matrix(
                active_cfg,
                num_agents,
                prev_actions=prev_actions,
                prev_msg_lens=prev_msg_lens,
                info=last_info,
                env=env,
                obs=obs,
            )
            obs_batch = _build_recurrent_obs_batch(obs, num_agents, active_cfg, feedback=feedback)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            action_mask = action_mask_from_flat_obs(obs_tensor)

            with torch.no_grad():
                if cfg.comm:
                    logits, send_logits, token_logits, len_logits, new_hidden = model(obs_tensor, hidden)
                else:
                    logits, new_hidden = model(obs_tensor, hidden)
                v = critic(obs_tensor)
            logits = mask_action_logits(logits, action_mask)

            if cfg.comm:
                action_dist = torch.distributions.Categorical(logits=logits)
                send_dist = torch.distributions.Bernoulli(logits=send_logits.squeeze(-1))
                token_dist = torch.distributions.Categorical(logits=token_logits)
                len_dist = torch.distributions.Categorical(logits=len_logits)

                acts = action_dist.sample()
                send = send_dist.sample()
                token_samples = token_dist.sample()
                len_samples = len_dist.sample()

                token_mask = (
                    torch.arange(cfg.comm_token_limit, device=device)[None, :] < len_samples[:, None]
                ).float()
                logp_tokens = (token_dist.log_prob(token_samples) * token_mask).sum(dim=-1)
                logp_len = len_dist.log_prob(len_samples)
                logp = (
                    action_dist.log_prob(acts)
                    + send_dist.log_prob(send)
                    + (logp_len + logp_tokens) * send
                )

                actions = {}
                for aid in range(num_agents):
                    if int(send[aid].item()) == 1 and int(len_samples[aid].item()) > 0:
                        msg = token_samples[aid][: int(len_samples[aid].item())].detach().cpu().tolist()
                    else:
                        msg = []
                    actions[aid] = {
                        "action": int(acts[aid].item()),
                        "message_tokens": [int(token) for token in msg],
                    }

                send_buf.append(send.cpu())
                token_buf.append(token_samples.cpu())
                len_buf.append(len_samples.cpu())
                comm_total_steps += num_agents
                comm_send_counts += int(send.sum().item())
                if int(send.sum().item()) > 0:
                    comm_len_sum += float(len_samples[send.bool()].sum().item())
                    comm_len_count += int(send.sum().item())
                comm_token_entropy_sum += float(token_dist.entropy().mean().item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logp = dist.log_prob(acts)
                actions = {aid: {"action": int(acts[aid].item()), "message_tokens": []} for aid in range(num_agents)}

            redundant_target_agents = _redundant_target_scan_agents(env, actions)
            next_obs, rewards, done, truncated, info = env.step(actions)
            next_info = info or {}
            redundant_target_scan_count += len(redundant_target_agents)
            _, penalty_sum = _apply_redundant_target_scan_penalty(
                rewards,
                redundant_target_agents,
                cfg.rl_redundant_target_scan_penalty,
            )
            redundant_target_scan_penalty_sum += penalty_sum
            wrong_target_agents = _wrong_target_scan_agents(next_info, num_agents)
            wrong_target_scan_count += len(wrong_target_agents)
            _, penalty_sum = _apply_wrong_target_scan_penalty(
                rewards,
                wrong_target_agents,
                cfg.rl_wrong_target_scan_penalty,
            )
            wrong_target_scan_penalty_sum += penalty_sum
            next_prev_actions = {aid: int(action["action"]) for aid, action in actions.items()}
            next_prev_msg_lens = _message_lengths(actions)
            segment_end = local_t == int(segment["steps"]) - 1
            reset_after = bool(done or truncated or segment_end)
            if segment_end and not (done or truncated):
                bootstrap_v = _bootstrap_recurrent_value(
                    active_cfg,
                    critic,
                    next_obs,
                    device,
                    prev_actions=next_prev_actions,
                    prev_msg_lens=next_prev_msg_lens,
                    info=next_info,
                    env=env,
                )
            else:
                bootstrap_v = torch.zeros((num_agents,), dtype=torch.float32)

            obs_buf.append(obs_tensor.cpu())
            act_buf.append(acts.cpu())
            logp_buf.append(logp.cpu())
            val_buf.append(v.cpu())
            hidden_buf.append((hidden[0].cpu(), hidden[1].cpu()))
            rew_buf.append(torch.tensor([rewards[i] for i in range(num_agents)], dtype=torch.float32))
            done_buf.append(torch.tensor([float(done or truncated)] * num_agents, dtype=torch.float32))
            reset_after_buf.append(reset_after)
            bootstrap_val_buf.append(bootstrap_v)

            for action_id in acts.detach().cpu().tolist():
                action_hist[int(action_id)] += 1
            if "comm_tokens" in next_info:
                ep_comm_tokens += sum(next_info["comm_tokens"].values())

            ep_ret += sum(rewards.values())
            ep_step += 1
            total_collected_steps += 1
            map_key = str(active_cfg.map_size)
            map_step_counts[map_key] = map_step_counts.get(map_key, 0) + 1

            if done or truncated:
                ep_returns.append(ep_ret)
                ep_steps.append(ep_step)
                ep_comm.append(ep_comm_tokens)
                resets_in_segment += 1
                if not segment_end:
                    if segment["mode"] == "fixed_map":
                        reset_seed = int(segment["seed"]) + resets_in_segment
                    else:
                        rollout_episode_idx += 1
                        env, active_cfg = _build_training_env(cfg, rollout_episode_idx)
                        reset_seed = int(update) * max(1, int(cfg.rollout_steps)) + total_collected_steps
                    obs, _info = env.reset(seed=reset_seed)
                    hidden = model.init_hidden(num_agents, device)
                    prev_actions = {}
                    prev_msg_lens = {}
                    last_info = {}
                    ep_ret, ep_step = 0.0, 0
                    ep_comm_tokens = 0
            else:
                hidden = new_hidden
                obs = next_obs
                last_info = next_info
                prev_actions = next_prev_actions
                prev_msg_lens = next_prev_msg_lens

    return {
        "obs_buf": obs_buf,
        "act_buf": act_buf,
        "logp_buf": logp_buf,
        "val_buf": val_buf,
        "rew_buf": rew_buf,
        "done_buf": done_buf,
        "reset_after_buf": reset_after_buf,
        "bootstrap_val_buf": bootstrap_val_buf,
        "send_buf": send_buf,
        "token_buf": token_buf,
        "len_buf": len_buf,
        "hidden_buf": hidden_buf,
        "ep_returns": ep_returns,
        "ep_steps": ep_steps,
        "ep_comm": ep_comm,
        "comm_send_counts": comm_send_counts,
        "comm_total_steps": comm_total_steps,
        "comm_len_sum": comm_len_sum,
        "comm_len_count": comm_len_count,
        "comm_token_entropy_sum": comm_token_entropy_sum,
        "action_hist": action_hist,
        "map_step_counts": map_step_counts,
        "balanced": balanced,
        "redundant_target_scan_count": redundant_target_scan_count,
        "redundant_target_scan_penalty_sum": redundant_target_scan_penalty_sum,
        "wrong_target_scan_count": wrong_target_scan_count,
        "wrong_target_scan_penalty_sum": wrong_target_scan_penalty_sum,
    }


def train_recurrent_rl(cfg: RecurrentConfig, model, device, *, wandb_run=None):
    """Fine-tune recurrent policy with PPO, carrying hidden state across steps."""
    import copy

    if cfg.rl_updates <= 0:
        return model

    set_global_seeds(cfg.seed)
    obs_dim, N = _recurrent_training_obs_shape(cfg)

    critic = MAPPOCritic(obs_dim, hidden_dim=cfg.hidden_dim).to(device)
    model.train()

    # Frozen BC reference for KL
    bc_ref = copy.deepcopy(model)
    bc_ref.eval()
    for p in bc_ref.parameters():
        p.requires_grad = False

    params = list(model.parameters()) + list(critic.parameters())
    optimizer = optim.Adam(params, lr=cfg.rl_lr, eps=1e-5)

    owns_wandb_run = False
    if wandb_run is None and cfg.wandb:
        wandb_run = _init_recurrent_wandb(cfg)
        owns_wandb_run = wandb_run is not None

    eval_cfg = replace(
        cfg,
        eval_episodes=max(1, int(cfg.rl_eval_episodes)),
        eval_seed=int(cfg.rl_eval_seed),
    )
    rl_eval_seed_count = max(1, int(cfg.rl_eval_seed_count))
    best_path = _recurrent_rl_best_path(cfg)
    initial_eval = evaluate_recurrent_policy_multi_seed(
        eval_cfg,
        model,
        device,
        seed_count=rl_eval_seed_count,
    )
    best_eval = {"update": -1, "phase": "initial", **initial_eval}
    final_eval = dict(best_eval)
    best_score = _recurrent_eval_score(initial_eval)
    best_state = copy.deepcopy(model.state_dict())
    print(json.dumps({"recurrent_rl_eval": best_eval}, indent=2, sort_keys=True))
    if best_path is not None:
        _save_recurrent_rl_checkpoint(
            best_path,
            model=model,
            critic=critic,
            cfg=cfg,
            obs_dim=obs_dim,
            best_eval=best_eval,
            final_eval=final_eval,
            restored_best=True,
        )
        print(f"Saved best recurrent RL checkpoint to {best_path}")
    if wandb_run is not None:
        eval_log = _recurrent_eval_wandb_payload(
            initial_eval,
            update=-1,
            is_best=True,
            best_eval=best_eval,
        )
        _wandb_log(wandb_run, eval_log, context="initial eval log")

    for update in range(cfg.rl_updates):
        # LR annealing
        frac = 1.0 - update / max(cfg.rl_updates, 1)
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.rl_lr * frac

        rollout = _collect_recurrent_rl_rollout(
            cfg,
            model,
            critic,
            device,
            update=update,
            num_agents=N,
        )
        obs_buf = rollout["obs_buf"]
        act_buf = rollout["act_buf"]
        logp_buf = rollout["logp_buf"]
        val_buf = rollout["val_buf"]
        rew_buf = rollout["rew_buf"]
        done_buf = rollout["done_buf"]
        reset_after_buf = rollout["reset_after_buf"]
        bootstrap_val_buf = rollout["bootstrap_val_buf"]
        send_buf = rollout["send_buf"]
        token_buf = rollout["token_buf"]
        len_buf = rollout["len_buf"]
        hidden_buf = rollout["hidden_buf"]
        ep_returns = rollout["ep_returns"]
        ep_steps = rollout["ep_steps"]
        ep_comm = rollout["ep_comm"]
        comm_send_counts = rollout["comm_send_counts"]
        comm_total_steps = rollout["comm_total_steps"]
        comm_len_sum = rollout["comm_len_sum"]
        comm_len_count = rollout["comm_len_count"]
        comm_token_entropy_sum = rollout["comm_token_entropy_sum"]
        action_hist = rollout["action_hist"]
        map_step_counts = rollout["map_step_counts"]
        redundant_target_scan_count = int(rollout.get("redundant_target_scan_count", 0))
        redundant_target_scan_penalty_sum = float(rollout.get("redundant_target_scan_penalty_sum", 0.0))
        wrong_target_scan_count = int(rollout.get("wrong_target_scan_count", 0))
        wrong_target_scan_penalty_sum = float(rollout.get("wrong_target_scan_penalty_sum", 0.0))

        # GAE
        values = torch.stack(val_buf)
        rewards_t = torch.stack(rew_buf)
        dones_t = torch.stack(done_buf)
        reset_after_t = torch.tensor(reset_after_buf, dtype=torch.bool)
        bootstrap_values = torch.stack(bootstrap_val_buf)

        advantages = torch.zeros_like(rewards_t)
        gae = torch.zeros(N)
        for t in reversed(range(len(obs_buf))):
            if bool(reset_after_t[t].item()):
                next_v = bootstrap_values[t]
                next_gae = torch.zeros(N)
            else:
                next_v = values[t + 1]
                next_gae = gae
            delta = rewards_t[t] + cfg.gamma * next_v * (1.0 - dones_t[t]) - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - dones_t[t]) * next_gae
            advantages[t] = gae
        returns = advantages + values

        # PPO update — process in sequential chunks to maintain hidden state
        T = len(obs_buf)
        redundant_target_scan_rate = redundant_target_scan_count / max(1, T * N)
        wrong_target_scan_rate = wrong_target_scan_count / max(1, T * N)
        adv_flat = advantages.reshape(T * N).to(device)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        ret_flat = returns.reshape(T * N).to(device)
        val_old_flat = values.reshape(T * N).to(device)
        if cfg.comm:
            send_t = torch.stack(send_buf).to(device)
            token_t = torch.stack(token_buf).to(device)
            len_t = torch.stack(len_buf).to(device)

        for epoch in range(cfg.rl_epochs):
            # Replay the sequence with current model
            hidden_replay = (hidden_buf[0][0].to(device), hidden_buf[0][1].to(device))
            bc_hidden_replay = bc_ref.init_hidden(N, device)
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_kl = 0.0
            total_comm_kl = 0.0
            total_entropy = 0.0
            total_steps = 0

            for t in range(T):
                obs_t = obs_buf[t].to(device)
                act_t = act_buf[t].to(device)
                idx = t * N

                # Reset hidden at episode and artificial rollout-segment boundaries.
                if t > 0 and bool(reset_after_t[t - 1].item()):
                    hidden_replay = model.init_hidden(N, device)
                    bc_hidden_replay = bc_ref.init_hidden(N, device)

                action_mask = action_mask_from_flat_obs(obs_t)
                if cfg.comm:
                    logits, send_logits, token_logits, len_logits, hidden_replay = model(obs_t, hidden_replay)
                    logits = mask_action_logits(logits, action_mask)

                    action_dist = torch.distributions.Categorical(logits=logits)
                    send_dist = torch.distributions.Bernoulli(logits=send_logits.squeeze(-1))
                    token_dist = torch.distributions.Categorical(logits=token_logits)
                    len_dist = torch.distributions.Categorical(logits=len_logits)

                    step_send = send_t[t]
                    step_tokens = token_t[t]
                    step_lens = len_t[t]
                    token_mask = (
                        torch.arange(cfg.comm_token_limit, device=device)[None, :] < step_lens[:, None]
                    ).float()
                    new_logp_tokens = (token_dist.log_prob(step_tokens) * token_mask).sum(dim=-1)
                    new_logp_len = len_dist.log_prob(step_lens)
                    new_logp = (
                        action_dist.log_prob(act_t)
                        + send_dist.log_prob(step_send)
                        + (new_logp_len + new_logp_tokens) * step_send
                    )
                    entropy = (
                        action_dist.entropy().mean()
                        + send_dist.entropy().mean()
                        + token_dist.entropy().mean()
                        + len_dist.entropy().mean()
                    )
                else:
                    logits, hidden_replay = model(obs_t, hidden_replay)
                    logits = mask_action_logits(logits, action_mask)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(act_t)
                    entropy = dist.entropy().mean()

                # KL toward BC reference
                with torch.no_grad():
                    if cfg.comm:
                        bc_logits, bc_send_logits, bc_token_logits, bc_len_logits, bc_hidden_replay = bc_ref(
                            obs_t,
                            bc_hidden_replay,
                        )
                    else:
                        bc_logits, bc_hidden_replay = bc_ref(obs_t, bc_hidden_replay)
                    bc_hidden_replay = (
                        bc_hidden_replay[0].detach(),
                        bc_hidden_replay[1].detach(),
                    )
                bc_logits = mask_action_logits(bc_logits, action_mask)
                bc_logprobs = torch.log_softmax(bc_logits, dim=-1)
                bc_probs = bc_logprobs.exp()
                current_logprobs = torch.log_softmax(logits, dim=-1)
                kl = (bc_probs * (bc_logprobs - current_logprobs)).sum(dim=-1).mean()
                if cfg.comm:
                    comm_kl = _recurrent_comm_reference_kl(
                        send_logits,
                        token_logits,
                        len_logits,
                        bc_send_logits,
                        bc_token_logits,
                        bc_len_logits,
                    )
                else:
                    comm_kl = torch.tensor(0.0, dtype=torch.float32, device=device)

                # PPO loss
                old_logp = logp_buf[t].to(device)
                ratio = (new_logp - old_logp).exp()
                adv = adv_flat[idx:idx + N]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                new_v = critic(obs_t)
                v_old = val_old_flat[idx:idx + N]
                v_clipped = v_old + torch.clamp(new_v - v_old, -cfg.value_clip, cfg.value_clip)
                value_loss = 0.5 * torch.max(
                    (ret_flat[idx:idx + N] - new_v).pow(2),
                    (ret_flat[idx:idx + N] - v_clipped).pow(2),
                ).mean()

                loss = (
                    policy_loss
                    + value_loss
                    - cfg.entropy_coeff * entropy
                    + cfg.bc_kl_coeff * kl
                    + cfg.bc_comm_kl_coeff * comm_kl
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()

                # Detach hidden for next step
                hidden_replay = (hidden_replay[0].detach(), hidden_replay[1].detach())

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl += kl.item()
                total_comm_kl += comm_kl.item()
                total_entropy += entropy.item()
                total_steps += 1

        denom = max(total_steps, 1)
        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        mean_len = float(np.mean(ep_steps)) if ep_steps else 0.0
        mean_comm = float(np.mean(ep_comm)) if ep_comm else 0.0
        comm_send_rate = comm_send_counts / comm_total_steps if comm_total_steps else 0.0
        comm_mean_len = comm_len_sum / comm_len_count if comm_len_count else 0.0
        comm_token_entropy = comm_token_entropy_sum / T if T else 0.0
        print(
            f"update {update:4d} | pi {total_policy_loss / denom:.3f} | "
            f"v {total_value_loss / denom:.3f} | kl {total_kl / denom:.4f} | "
            f"comm_kl {total_comm_kl / denom:.4f} | "
            f"ent {total_entropy / denom:.3f} | ret {mean_ret:.2f} | len {mean_len:.1f} | "
            f"red_scan {redundant_target_scan_count} | wrong_scan {wrong_target_scan_count}"
        )

        if wandb_run is not None:
            lr = float(optimizer.param_groups[0]["lr"])
            log_payload = {
                "policy_loss": total_policy_loss / denom,
                "value_loss": total_value_loss / denom,
                "kl": total_kl / denom,
                "comm_kl": total_comm_kl / denom,
                "entropy": total_entropy / denom,
                "lr": lr,
                "train/policy_loss": total_policy_loss / denom,
                "train/value_loss": total_value_loss / denom,
                "train/kl": total_kl / denom,
                "train/action_kl": total_kl / denom,
                "train/comm_kl": total_comm_kl / denom,
                "train/entropy": total_entropy / denom,
                "train/lr": lr,
                "rollout/episodes": len(ep_returns),
                "rollout/mean_ep_return": mean_ret,
                "rollout/mean_ep_len": mean_len,
                "rollout/mean_ep_comm_tokens": mean_comm,
                "rollout/comm_send_rate": comm_send_rate,
                "rollout/comm_mean_len": comm_mean_len,
                "rollout/comm_token_entropy": comm_token_entropy,
                "rollout/redundant_target_scans": redundant_target_scan_count,
                "rollout/redundant_target_scan_rate": redundant_target_scan_rate,
                "rollout/redundant_target_scan_penalty": redundant_target_scan_penalty_sum,
                "rollout/wrong_target_scans": wrong_target_scan_count,
                "rollout/wrong_target_scan_rate": wrong_target_scan_rate,
                "rollout/wrong_target_scan_penalty": wrong_target_scan_penalty_sum,
                "rollout/balanced": int(bool(rollout["balanced"])),
                "rollout/steps": int(T),
                "update": update,
            }
            for map_size, count in sorted(map_step_counts.items(), key=lambda item: int(item[0])):
                log_payload[f"rollout/map_{map_size}/steps"] = int(count)
            for i in range(8):
                log_payload[f"rollout/action_hist_{i}"] = int(action_hist[i])
            _wandb_log(wandb_run, log_payload, context="train log")

        should_eval = update == cfg.rl_updates - 1 or (
            cfg.rl_eval_every > 0 and (update + 1) % cfg.rl_eval_every == 0
        )
        if should_eval:
            eval_result = evaluate_recurrent_policy_multi_seed(
                eval_cfg,
                model,
                device,
                seed_count=rl_eval_seed_count,
            )
            eval_row = {"update": update, "phase": "ppo", **eval_result}
            eval_score = _recurrent_eval_score(eval_result)
            is_best = eval_score > best_score
            if is_best:
                best_score = eval_score
                best_eval = dict(eval_row)
                best_state = copy.deepcopy(model.state_dict())
                if best_path is not None:
                    _save_recurrent_rl_checkpoint(
                        best_path,
                        model=model,
                        critic=critic,
                        cfg=cfg,
                        obs_dim=obs_dim,
                        best_eval=best_eval,
                        final_eval=eval_row,
                        restored_best=True,
                    )
                    print(f"Saved best recurrent RL checkpoint to {best_path}")
            final_eval = dict(eval_row)
            print(json.dumps({"recurrent_rl_eval": {**eval_row, "is_best": is_best}}, indent=2, sort_keys=True))
            if wandb_run is not None:
                eval_log = _recurrent_eval_wandb_payload(
                    eval_result,
                    update=update,
                    is_best=is_best,
                    best_eval=best_eval,
                )
                _wandb_log(wandb_run, eval_log, context="eval log")

    restored_best = False
    if cfg.rl_restore_best and best_state is not None:
        model.load_state_dict(best_state)
        restored_best = True
        print(json.dumps({"recurrent_rl_restored_best": best_eval}, indent=2, sort_keys=True))
    if best_path is not None and best_state is not None:
        current_state = None
        if not restored_best:
            current_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
        _save_recurrent_rl_checkpoint(
            best_path,
            model=model,
            critic=critic,
            cfg=cfg,
            obs_dim=obs_dim,
            best_eval=best_eval,
            final_eval=final_eval,
            restored_best=True,
        )
        if current_state is not None:
            model.load_state_dict(current_state)
        print(f"Saved best recurrent RL checkpoint to {best_path}")
    if cfg.save:
        _save_recurrent_rl_checkpoint(
            cfg.save,
            model=model,
            critic=critic,
            cfg=cfg,
            obs_dim=obs_dim,
            best_eval=best_eval,
            final_eval=final_eval,
            restored_best=restored_best,
        )
        print(f"Saved to {cfg.save}")
    if owns_wandb_run and wandb_run is not None:
        wandb_run.finish()
    return model


def main():
    p = argparse.ArgumentParser(description="Recurrent BC→RL for SyncOrSink scenarios")
    p.add_argument("--scenario", default="pipeline_assembly")
    p.add_argument("--map-size", type=int, default=8)
    p.add_argument("--train-map-sizes", default="")
    p.add_argument(
        "--train-map-sampling-weights",
        default="",
        help=(
            "Optional comma-separated integer map_size:weight schedule for training collection, "
            "for example 8:1,16:1,32:3"
        ),
    )
    p.add_argument(
        "--map-max-steps",
        default="",
        help="Optional comma-separated map_size:max_steps overrides, e.g. 8:60,16:120,32:240",
    )
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--fov-preset", default="easy")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--energy-preset", default="easy")
    p.add_argument("--signal-decoy-count", type=int, default=None)
    p.add_argument("--decoy-penalty", type=float, default=0.5)
    p.add_argument("--scan-window", type=int, default=3)
    p.add_argument("--oracle", default="oracle_strong",
                   choices=["oracle_strong", "oracle_strong_comm", "signal_hint_comm"])
    p.add_argument("--obs-exploration-memory", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obs-exploration-age", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obs-feedback", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obs-normalize-tokens", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obs-memory-mode", choices=["full", "egocentric"], default="full")
    p.add_argument("--obs-memory-radius", type=int, default=4)
    p.add_argument("--obs-navigation-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obs-signal-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obs-signal-sync-feedback", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obs-signal-scan-state", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obs-signal-negative-memory", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obs-signal-negative-memory-window", type=int, default=64)
    p.add_argument("--obs-signal-inferred-target-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obs-signal-target-match-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pipeline-shaping", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pipeline-shaping-scale", type=float, default=0.1)
    p.add_argument("--energy-shaping", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--energy-shaping-scale", type=float, default=0.01)
    p.add_argument("--signal-shaping", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--signal-shaping-scale", type=float, default=0.01)
    p.add_argument("--signal-scan-bonus", type=float, default=0.0)
    p.add_argument("--signal-joint-scan-bonus", type=float, default=0.0)
    p.add_argument("--signal-colocation-bonus", type=float, default=0.0)
    p.add_argument("--signal-colocation-radius", type=int, default=2)
    p.add_argument("--signal-comm-utility", type=float, default=0.0)
    p.add_argument("--signal-target-visit-bonus", type=float, default=0.0)
    p.add_argument("--signal-decoy-visit-penalty", type=float, default=0.0)
    p.add_argument("--signal-unique-target-scan-bonus", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--comm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--comm-token-limit", type=int, default=8)
    p.add_argument("--comm-vocab-size", type=int, default=32)
    p.add_argument("--comm-max-messages", type=int, default=8)
    p.add_argument("--comm-len-cost", type=float, default=0.0)
    p.add_argument("--comm-cost", type=float, default=0.01)
    p.add_argument("--demo-episodes", type=int, default=200)
    p.add_argument("--bc-epochs", type=int, default=30)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    p.add_argument("--bc-seq-len", type=int, default=32)
    p.add_argument("--bc-eval-every-epochs", type=int, default=0)
    p.add_argument("--bc-eval-episodes", type=int, default=0)
    p.add_argument("--bc-eval-seed-count", type=int, default=1)
    p.add_argument("--bc-restore-best-eval-epoch", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--bc-equal-episode-weight", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bc-comm-loss-weight", type=float, default=0.1)
    p.add_argument("--bc-comm-send-pos-weight", type=float, default=0.0)
    p.add_argument("--bc-comm-send-loss-weight", type=float, default=1.0)
    p.add_argument("--bc-comm-length-loss-weight", type=float, default=1.0)
    p.add_argument("--bc-comm-token-loss-weight", type=float, default=1.0)
    p.add_argument("--bc-comm-send-rate-penalty-weight", type=float, default=0.0)
    p.add_argument(
        "--bc-comm-send-rate-target",
        type=float,
        default=-1.0,
        help="Target send probability for BC send-rate penalty; negative matches the batch label rate",
    )
    p.add_argument("--bc-calibrate-send-threshold", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--bc-send-threshold-target-rate",
        type=float,
        default=-1.0,
        help="Target send rate for post-BC threshold calibration; negative matches dataset send-label rate",
    )
    p.add_argument("--bc-signal-target-interact-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-redundant-target-interact-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-target-pursuit-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-target-pursuit-action-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-sync-response-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-sync-response-action-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-target-match-action-weight", type=float, default=0.0)
    p.add_argument(
        "--bc-signal-first-target-scan-action-weight",
        type=float,
        default=0.0,
        help="Extra positive interact loss weight for first true-target scan labels",
    )
    p.add_argument(
        "--bc-signal-refresh-target-scan-action-weight",
        type=float,
        default=0.0,
        help="Extra positive interact loss weight for refresh true-target scan labels",
    )
    p.add_argument(
        "--bc-signal-joint-target-scan-action-weight",
        type=float,
        default=0.0,
        help="Extra positive interact loss weight for joint-completion true-target scan labels",
    )
    p.add_argument(
        "--bc-signal-target-opportunity-action-weight",
        type=float,
        default=0.0,
        help="Opt-in positive interact loss weight for observation-safe true-target scan opportunities",
    )
    p.add_argument("--bc-signal-redundant-target-wait-action-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-scan-decision-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-scan-decision-pos-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-scan-decision-neg-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-scan-gate-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-scan-gate-pos-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-scan-gate-neg-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-target-validity-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-target-validity-pos-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-target-validity-neg-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-target-decision-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-target-decision-pos-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-target-decision-neg-weight", type=float, default=1.0)
    p.add_argument(
        "--bc-signal-target-aux-weight",
        type=float,
        default=0.0,
        help="Opt-in BC auxiliary loss that predicts the Signal Hunt true-target coordinate from recurrent state",
    )
    p.add_argument("--bc-signal-rejected-target-interact-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-rejected-target-interact-action-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-bad-redundant-target-interact-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-decoy-drift-action-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-decoy-scan-action-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-rejected-target-drift-action-loss-weight", type=float, default=0.0)
    p.add_argument("--dagger-rounds", type=int, default=0)
    p.add_argument("--dagger-episodes", type=int, default=20)
    p.add_argument("--dagger-seed-base", type=int, default=10000)
    p.add_argument("--dagger-seed-stride", type=int, default=1000)
    p.add_argument(
        "--dagger-seed-list",
        default="",
        help=(
            "Optional comma-separated environment reset seeds for DAgger collection; "
            "when set, episodes cycle through this explicit list"
        ),
    )
    p.add_argument("--dagger-retrain-from-scratch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dagger-max-steps-per-episode", type=int, default=0)
    p.add_argument("--dagger-success-episode-weight", type=float, default=1.0)
    p.add_argument("--dagger-failed-episode-weight", type=float, default=0.25)
    p.add_argument(
        "--dagger-focus-events",
        default=(
            "decoy_scan,solo_target_scan,rejected_target_scan,"
            "bad_redundant_target_scan,target_interact_miss,target_pursuit_miss,"
            "target_decoy_drift_miss,target_discovery_miss,target_handoff_miss,"
            "movement_stall_miss"
        ),
    )
    p.add_argument("--dagger-focus-error-weight", type=float, default=3.0)
    p.add_argument("--dagger-focus-recovery-weight", type=float, default=2.0)
    p.add_argument("--dagger-focus-window", type=int, default=1)
    p.add_argument("--dagger-target-interact-focus-weight", type=float, default=5.0)
    p.add_argument("--dagger-target-discovery-min-map-size", type=int, default=16)
    p.add_argument("--dagger-target-discovery-focus-weight", type=float, default=3.0)
    p.add_argument("--dagger-movement-stall-min-map-size", type=int, default=16)
    p.add_argument("--dagger-movement-stall-window", type=int, default=6)
    p.add_argument("--dagger-movement-stall-focus-weight", type=float, default=4.0)
    p.add_argument("--dagger-target-decoy-drift-focus-weight", type=float, default=5.0)
    p.add_argument("--dagger-solo-target-team-weight", type=float, default=1.0)
    p.add_argument("--dagger-early-stop-patience", type=int, default=0)
    p.add_argument("--dagger-focus-replay", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dagger-replay-pre-steps", type=int, default=2)
    p.add_argument("--dagger-replay-post-steps", type=int, default=2)
    p.add_argument("--dagger-replay-weight", type=float, default=1.0)
    p.add_argument(
        "--dagger-positive-replay-events",
        default="",
        help="Comma-separated event names to replay as positive examples, e.g. joint_target_scan",
    )
    p.add_argument(
        "--dagger-replay-event-weights",
        default="",
        help="Optional comma-separated event:weight overrides for replay snippets",
    )
    p.add_argument(
        "--dagger-replay-event-caps",
        default="",
        help="Optional comma-separated event:max_snippets_per_episode replay caps",
    )
    p.add_argument(
        "--dagger-replay-success-only-events",
        default="",
        help="Comma-separated event names whose replay snippets are kept only from successful parent rollouts",
    )
    p.add_argument(
        "--dagger-replay-priority-events",
        default="",
        help="Comma-separated replay event names to prioritize before ordinary snippets",
    )
    p.add_argument(
        "--dagger-replay-balance-positive-events",
        default="",
        help="Positive replay events that anchor opt-in positive/negative replay balancing",
    )
    p.add_argument(
        "--dagger-replay-balance-negative-events",
        default="",
        help="Negative replay events capped by --dagger-replay-max-negative-per-positive",
    )
    p.add_argument(
        "--dagger-replay-max-negative-per-positive",
        type=float,
        default=-1.0,
        help="If >=0, cap balanced negative replay snippets to ceil(positive_count * this value)",
    )
    p.add_argument("--dagger-max-replay-snippets-per-episode", type=int, default=4)
    p.add_argument(
        "--dagger-expert-max-replay-snippets-per-episode",
        type=int,
        default=-1,
        help=(
            "Optional cap for expert_positive_replay snippets only; negative reuses "
            "--dagger-max-replay-snippets-per-episode"
        ),
    )
    p.add_argument(
        "--dagger-solo-target-team-success-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Defer solo-target teammate upweights and apply them only to successful parent rollouts",
    )
    p.add_argument("--dagger-positive-target-pursuit-min-map-size", type=int, default=16)
    p.add_argument(
        "--dagger-redundant-target-wait-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convert repeated true-target scan labels into wait labels while the previous scan is still active",
    )
    p.add_argument(
        "--dagger-target-scan-broadcast-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Label confirmed true-target scanners to broadcast exact target coordinates for teammate handoff",
    )
    p.add_argument(
        "--dagger-oracle-message-rollin-rate",
        type=float,
        default=0.0,
        help=(
            "During DAgger collection, replace each model message with the oracle message at this "
            "per-agent probability while keeping model physical actions"
        ),
    )
    p.add_argument("--rl-updates", type=int, default=3000)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--rl-balanced-rollouts", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--rl-rollout-map-steps",
        default="",
        help=(
            "Optional comma-separated map_size:rollout_steps overrides for balanced recurrent PPO, "
            "for example 8:64,16:128,32:256"
        ),
    )
    p.add_argument("--rl-redundant-target-scan-penalty", type=float, default=0.0)
    p.add_argument("--rl-wrong-target-scan-penalty", type=float, default=0.0)
    p.add_argument("--rl-epochs", type=int, default=2)
    p.add_argument("--minibatch-seqs", type=int, default=8)
    p.add_argument("--rl-lr", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--bc-kl-coeff", type=float, default=0.5)
    p.add_argument("--bc-comm-kl-coeff", type=float, default=0.5)
    p.add_argument("--rl-eval-every", type=int, default=5)
    p.add_argument("--rl-eval-episodes", type=int, default=20)
    p.add_argument("--rl-eval-seed", type=int, default=10000)
    p.add_argument("--rl-eval-seed-count", type=int, default=1)
    p.add_argument("--rl-restore-best", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rl-save-best", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rl-best-save", default=None)
    p.add_argument(
        "--recurrent-init",
        default=None,
        help="Optional recurrent actor checkpoint to load before RL, skipping demo/BC/DAgger collection",
    )
    p.add_argument(
        "--recurrent-init-for-dagger",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use --recurrent-init as the starting model for BC/DAgger instead of skipping to RL",
    )
    p.add_argument(
        "--recurrent-init-allow-obs-dim-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow loading a smaller recurrent-init observation width by zero-padding newly added "
            "columns before the action-mask tail"
        ),
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed Python, NumPy, and Torch RNGs (default: 0)")
    p.add_argument("--save", default=None)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=3000)
    p.add_argument("--eval-seed-count", type=int, default=1)
    p.add_argument("--eval-map-sizes", default="")
    p.add_argument("--eval-send-threshold", type=float, default=0.25)
    p.add_argument(
        "--eval-signal-target-scan-threshold",
        type=float,
        default=-1.0,
        help=(
            "Optional Signal Hunt decode calibration: if >=0, force interact on an observation-safe "
            "center target when the interact probability reaches this threshold"
        ),
    )
    p.add_argument(
        "--eval-signal-scan-gate-threshold",
        type=float,
        default=-1.0,
        help="Optional Signal Hunt scan-gate threshold; if >=0, the recurrent scan gate controls center-target interact",
    )
    p.add_argument(
        "--eval-signal-scan-gate-suppress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When the scan gate is enabled, suppress center-target interact below the gate threshold",
    )
    p.add_argument(
        "--eval-signal-target-validity-threshold",
        type=float,
        default=-1.0,
        help="Optional Signal Hunt target-validity threshold; if >=0, suppress center-target interact below it",
    )
    p.add_argument(
        "--eval-signal-target-decision-threshold",
        type=float,
        default=-1.0,
        help="Optional Signal Hunt unified target-decision threshold; if >=0, control center-target interact",
    )
    p.add_argument(
        "--eval-signal-target-decision-suppress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When target-decision decoding is enabled, suppress center-target interact below the threshold",
    )
    p.add_argument(
        "--eval-signal-scan-sync-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use scan-state feedback to join teammate scans and wait while own true-target scan is active",
    )
    p.add_argument(
        "--eval-signal-scan-sync-force-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With scan-sync assist enabled, force first/refresh scans on compatible center target candidates",
    )
    p.add_argument(
        "--eval-signal-scan-broadcast-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use scan-state feedback to broadcast the remembered true-target scan position",
    )
    p.add_argument(
        "--eval-signal-exact-target-message-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop exact Signal Hunt target messages unless backed by an own exact hint or active scan state",
    )
    p.add_argument(
        "--eval-signal-exact-target-navigation-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use trusted exact Signal Hunt target messages to force greedy target navigation/interact",
    )
    p.add_argument(
        "--eval-signal-exact-target-memory-steps",
        type=int,
        default=0,
        help="Steps to retain trusted exact Signal Hunt target messages for opt-in navigation/message assist",
    )
    p.add_argument(
        "--eval-signal-scan-refresh-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use scan-state feedback to refresh an own true-target scan as it is about to expire",
    )
    p.add_argument(
        "--eval-signal-scan-refresh-threshold",
        type=float,
        default=0.5,
        help="Remaining scan-window fraction at or below which refresh assist forces interact",
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="syncorsink")
    p.add_argument("--wandb-run", default=None)
    args = p.parse_args()

    cfg = RecurrentConfig(
        scenario=args.scenario,
        map_size=args.map_size,
        train_map_sizes=args.train_map_sizes,
        train_map_sampling_weights=args.train_map_sampling_weights,
        map_max_steps=args.map_max_steps,
        agents=args.agents,
        fov_preset=args.fov_preset,
        max_steps=args.max_steps,
        energy_preset=args.energy_preset,
        signal_decoy_count=args.signal_decoy_count,
        decoy_penalty=args.decoy_penalty,
        scan_window=args.scan_window,
        oracle_type=args.oracle,
        obs_exploration_memory=args.obs_exploration_memory,
        obs_exploration_age=args.obs_exploration_age,
        obs_feedback=args.obs_feedback,
        obs_normalize_tokens=args.obs_normalize_tokens,
        obs_memory_mode=args.obs_memory_mode,
        obs_memory_radius=args.obs_memory_radius,
        obs_navigation_features=args.obs_navigation_features,
        obs_signal_features=args.obs_signal_features,
        obs_signal_sync_feedback=args.obs_signal_sync_feedback,
        obs_signal_scan_state=args.obs_signal_scan_state,
        obs_signal_negative_memory=args.obs_signal_negative_memory,
        obs_signal_negative_memory_window=args.obs_signal_negative_memory_window,
        obs_signal_inferred_target_features=args.obs_signal_inferred_target_features,
        obs_signal_target_match_features=args.obs_signal_target_match_features,
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
        signal_target_visit_bonus=args.signal_target_visit_bonus,
        signal_decoy_visit_penalty=args.signal_decoy_visit_penalty,
        signal_unique_target_scan_bonus=args.signal_unique_target_scan_bonus,
        hidden_dim=args.hidden_dim,
        comm=args.comm,
        comm_token_limit=args.comm_token_limit,
        comm_vocab_size=args.comm_vocab_size,
        comm_max_messages=args.comm_max_messages,
        comm_len_cost=args.comm_len_cost,
        comm_cost=args.comm_cost,
        demo_episodes=args.demo_episodes,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
        bc_seq_len=args.bc_seq_len,
        bc_eval_every_epochs=args.bc_eval_every_epochs,
        bc_eval_episodes=args.bc_eval_episodes,
        bc_eval_seed_count=args.bc_eval_seed_count,
        bc_restore_best_eval_epoch=args.bc_restore_best_eval_epoch,
        bc_equal_episode_weight=args.bc_equal_episode_weight,
        bc_comm_loss_weight=args.bc_comm_loss_weight,
        bc_comm_send_pos_weight=args.bc_comm_send_pos_weight,
        bc_comm_send_loss_weight=args.bc_comm_send_loss_weight,
        bc_comm_length_loss_weight=args.bc_comm_length_loss_weight,
        bc_comm_token_loss_weight=args.bc_comm_token_loss_weight,
        bc_comm_send_rate_penalty_weight=args.bc_comm_send_rate_penalty_weight,
        bc_comm_send_rate_target=args.bc_comm_send_rate_target,
        bc_calibrate_send_threshold=args.bc_calibrate_send_threshold,
        bc_send_threshold_target_rate=args.bc_send_threshold_target_rate,
        bc_signal_target_interact_weight=args.bc_signal_target_interact_weight,
        bc_signal_redundant_target_interact_weight=args.bc_signal_redundant_target_interact_weight,
        bc_signal_target_pursuit_weight=args.bc_signal_target_pursuit_weight,
        bc_signal_target_pursuit_action_weight=args.bc_signal_target_pursuit_action_weight,
        bc_signal_sync_response_weight=args.bc_signal_sync_response_weight,
        bc_signal_sync_response_action_loss_weight=args.bc_signal_sync_response_action_loss_weight,
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
        bc_signal_target_aux_weight=args.bc_signal_target_aux_weight,
        bc_signal_rejected_target_interact_loss_weight=args.bc_signal_rejected_target_interact_loss_weight,
        bc_signal_rejected_target_interact_action_loss_weight=(
            args.bc_signal_rejected_target_interact_action_loss_weight
        ),
        bc_signal_bad_redundant_target_interact_loss_weight=(
            args.bc_signal_bad_redundant_target_interact_loss_weight
        ),
        bc_signal_decoy_drift_action_loss_weight=args.bc_signal_decoy_drift_action_loss_weight,
        bc_signal_decoy_scan_action_loss_weight=args.bc_signal_decoy_scan_action_loss_weight,
        bc_signal_rejected_target_drift_action_loss_weight=(
            args.bc_signal_rejected_target_drift_action_loss_weight
        ),
        dagger_rounds=args.dagger_rounds,
        dagger_episodes=args.dagger_episodes,
        dagger_seed_base=args.dagger_seed_base,
        dagger_seed_stride=args.dagger_seed_stride,
        dagger_seed_list=args.dagger_seed_list,
        dagger_retrain_from_scratch=args.dagger_retrain_from_scratch,
        dagger_max_steps_per_episode=args.dagger_max_steps_per_episode,
        dagger_success_episode_weight=args.dagger_success_episode_weight,
        dagger_failed_episode_weight=args.dagger_failed_episode_weight,
        dagger_focus_events=args.dagger_focus_events,
        dagger_focus_error_weight=args.dagger_focus_error_weight,
        dagger_focus_recovery_weight=args.dagger_focus_recovery_weight,
        dagger_focus_window=args.dagger_focus_window,
        dagger_target_interact_focus_weight=args.dagger_target_interact_focus_weight,
        dagger_target_discovery_min_map_size=args.dagger_target_discovery_min_map_size,
        dagger_target_discovery_focus_weight=args.dagger_target_discovery_focus_weight,
        dagger_movement_stall_min_map_size=args.dagger_movement_stall_min_map_size,
        dagger_movement_stall_window=args.dagger_movement_stall_window,
        dagger_movement_stall_focus_weight=args.dagger_movement_stall_focus_weight,
        dagger_target_decoy_drift_focus_weight=args.dagger_target_decoy_drift_focus_weight,
        dagger_solo_target_team_weight=args.dagger_solo_target_team_weight,
        dagger_early_stop_patience=args.dagger_early_stop_patience,
        dagger_focus_replay=args.dagger_focus_replay,
        dagger_replay_pre_steps=args.dagger_replay_pre_steps,
        dagger_replay_post_steps=args.dagger_replay_post_steps,
        dagger_replay_weight=args.dagger_replay_weight,
        dagger_positive_replay_events=args.dagger_positive_replay_events,
        dagger_replay_event_weights=args.dagger_replay_event_weights,
        dagger_replay_event_caps=args.dagger_replay_event_caps,
        dagger_replay_success_only_events=args.dagger_replay_success_only_events,
        dagger_replay_priority_events=args.dagger_replay_priority_events,
        dagger_replay_balance_positive_events=args.dagger_replay_balance_positive_events,
        dagger_replay_balance_negative_events=args.dagger_replay_balance_negative_events,
        dagger_replay_max_negative_per_positive=args.dagger_replay_max_negative_per_positive,
        dagger_max_replay_snippets_per_episode=args.dagger_max_replay_snippets_per_episode,
        dagger_expert_max_replay_snippets_per_episode=args.dagger_expert_max_replay_snippets_per_episode,
        dagger_solo_target_team_success_only=args.dagger_solo_target_team_success_only,
        dagger_positive_target_pursuit_min_map_size=args.dagger_positive_target_pursuit_min_map_size,
        dagger_redundant_target_wait_labels=args.dagger_redundant_target_wait_labels,
        dagger_target_scan_broadcast_labels=args.dagger_target_scan_broadcast_labels,
        dagger_oracle_message_rollin_rate=args.dagger_oracle_message_rollin_rate,
        rl_updates=args.rl_updates,
        rollout_steps=args.rollout_steps,
        rl_balanced_rollouts=args.rl_balanced_rollouts,
        rl_rollout_map_steps=args.rl_rollout_map_steps,
        rl_redundant_target_scan_penalty=args.rl_redundant_target_scan_penalty,
        rl_wrong_target_scan_penalty=args.rl_wrong_target_scan_penalty,
        rl_epochs=args.rl_epochs,
        minibatch_seqs=args.minibatch_seqs,
        rl_lr=args.rl_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        entropy_coeff=args.entropy_coeff,
        max_grad_norm=args.max_grad_norm,
        bc_kl_coeff=args.bc_kl_coeff,
        bc_comm_kl_coeff=args.bc_comm_kl_coeff,
        rl_eval_every=args.rl_eval_every,
        rl_eval_episodes=args.rl_eval_episodes,
        rl_eval_seed=args.rl_eval_seed,
        rl_eval_seed_count=args.rl_eval_seed_count,
        rl_restore_best=args.rl_restore_best,
        rl_save_best=args.rl_save_best,
        rl_best_save=args.rl_best_save,
        recurrent_init=args.recurrent_init,
        recurrent_init_for_dagger=args.recurrent_init_for_dagger,
        recurrent_init_allow_obs_dim_mismatch=args.recurrent_init_allow_obs_dim_mismatch,
        device=args.device,
        seed=args.seed,
        save=args.save,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        eval_seed_count=args.eval_seed_count,
        eval_map_sizes=args.eval_map_sizes,
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
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
    )

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")
    wandb_run = _init_recurrent_wandb(cfg)

    dagger_history = []
    best_dagger_row = None
    initial_dagger_model = None
    if cfg.recurrent_init:
        print("=== Step 1: Loading recurrent init checkpoint ===")
        model = load_recurrent_actor_checkpoint(cfg.recurrent_init, cfg, device)
        initial_dagger_model = model if cfg.recurrent_init_for_dagger else None
        print(f"Loaded recurrent init checkpoint from {cfg.recurrent_init}")

        print("\n=== Step 2: Evaluating recurrent init checkpoint ===")
        eval_result = evaluate_recurrent_policy_multi_seed(
            cfg,
            model,
            device,
            seed_count=max(1, int(cfg.eval_seed_count)),
        )
        print(json.dumps({"eval_recurrent_init": eval_result}, indent=2, sort_keys=True))
        if wandb_run is not None:
            _wandb_log(
                wandb_run,
                _recurrent_eval_wandb_payload(
                    eval_result,
                    update=0,
                    is_best=True,
                    best_eval=eval_result,
                    prefix="init/eval",
                ),
                context="init eval log",
            )
    if not cfg.recurrent_init or cfg.recurrent_init_for_dagger:
        step_prefix = "Step 3" if cfg.recurrent_init else "Step 1"
        print(f"=== {step_prefix}: Collecting oracle demos ===")
        episodes = collect_episode_demos(cfg)
        if wandb_run is not None:
            demo_source_counts = _episode_source_counts(episodes)
            demo_replay_episodes = sum(
                int(count)
                for source, count in demo_source_counts.items()
                if str(source).endswith("_replay")
            )
            _wandb_log(
                wandb_run,
                {
                    "demo/episodes": int(len(episodes)),
                    "demo/base_episodes": int(demo_source_counts.get("expert", 0)),
                    "demo/replay_episodes": int(demo_replay_episodes),
                    "demo/transitions": int(_episode_count_transitions(episodes)),
                    "demo/effective_transitions": float(_episode_count_effective_transitions(episodes)),
                    "demo/target_match_action_labels": int(
                        _episode_count_label_mask(episodes, "signal_target_match_action_mask")
                    ),
                    "demo/target_opportunity_action_labels": int(
                        _episode_count_label_mask(
                            episodes,
                            "signal_target_opportunity_action_mask",
                        )
                    ),
                    "demo/redundant_target_wait_action_aux_labels": int(
                        _episode_count_label_mask(
                            episodes,
                            "signal_redundant_target_wait_action_mask",
                        )
                    ),
                    "demo/target_pursuit_action_labels": int(
                        _episode_count_label_mask(
                            episodes,
                            "signal_target_pursuit_action_mask",
                        )
                    ),
                    "demo/sync_response_action_labels": int(
                        _episode_count_label_mask(
                            episodes,
                            "signal_sync_response_action_mask",
                        )
                    ),
                    "demo/target_aux_labels": int(
                        _episode_count_label_mask(episodes, "signal_target_aux_mask")
                    ),
                    **{
                        f"demo/map_{map_size}/episodes": int(count)
                        for map_size, count in _episode_map_size_counts(episodes).items()
                    },
                    **{
                        f"demo/source_{source}/episodes": int(count)
                        for source, count in demo_source_counts.items()
                    },
                    **_map_diagnostics_wandb_payload(
                        "demo",
                        _episode_map_size_diagnostics(episodes),
                    ),
                },
                context="demo log",
            )

        if cfg.dagger_rounds > 0:
            train_step = "Step 4" if cfg.recurrent_init else "Step 2"
            print(f"\n=== {train_step}: Training recurrent DAgger ===")
            model, dagger_history, episodes, best_dagger_row = train_recurrent_bc_dagger(
                cfg,
                episodes,
                device,
                wandb_run=wandb_run,
                initial_model=initial_dagger_model,
            )
            eval_result = best_dagger_row["eval"]
        else:
            train_step = "Step 4" if cfg.recurrent_init else "Step 2"
            print(f"\n=== {train_step}: Training recurrent BC ===")
            model = train_recurrent_bc(
                cfg,
                episodes,
                device,
                model=initial_dagger_model,
                wandb_run=wandb_run,
            )

            print(f"\n=== {train_step}b: Evaluating recurrent BC ===")
            eval_result = evaluate_recurrent_policy_multi_seed(
                cfg,
                model,
                device,
                seed_count=max(1, int(cfg.eval_seed_count)),
            )
            print(json.dumps({"eval_recurrent_bc": eval_result}, indent=2, sort_keys=True))
            if wandb_run is not None:
                _wandb_log(
                    wandb_run,
                    _recurrent_eval_wandb_payload(
                        eval_result,
                        update=0,
                        is_best=True,
                        best_eval=eval_result,
                        prefix="bc/eval",
                    ),
                    context="bc eval log",
                )

    if cfg.save and cfg.rl_updates <= 0:
        os.makedirs(os.path.dirname(cfg.save) or ".", exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "config": vars(cfg),
            "eval_recurrent_policy": eval_result,
            "dagger_history": dagger_history,
            "best_dagger_round": best_dagger_row,
            "recurrent_init": cfg.recurrent_init,
        }, cfg.save)
        print(f"Saved to {cfg.save}")

    if cfg.rl_updates > 0:
        print("\n=== Step 3: RL fine-tuning ===")
        train_recurrent_rl(cfg, model, device, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
