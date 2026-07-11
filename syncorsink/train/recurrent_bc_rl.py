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
import json
import os
import warnings
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional

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
)
from syncorsink.eval.metrics import EpisodeStats, summarize
from syncorsink.eval.success import episode_success
from syncorsink.train.mappo import (
    action_mask_from_flat_obs,
    flatten_obs,
    mask_action_logits,
    resolve_device,
)
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
    bc_signal_target_interact_weight: float = 1.0
    bc_signal_redundant_target_interact_weight: float = 1.0
    bc_signal_target_pursuit_weight: float = 1.0
    bc_signal_sync_response_weight: float = 1.0
    bc_signal_rejected_target_interact_loss_weight: float = 0.0
    bc_signal_bad_redundant_target_interact_loss_weight: float = 0.0
    bc_signal_decoy_drift_action_loss_weight: float = 0.0
    dagger_rounds: int = 0
    dagger_episodes: int = 20
    dagger_retrain_from_scratch: bool = True
    dagger_max_steps_per_episode: int = 0
    dagger_success_episode_weight: float = 1.0
    dagger_failed_episode_weight: float = 0.25
    dagger_focus_events: str = (
        "decoy_scan,solo_target_scan,rejected_target_scan,"
        "bad_redundant_target_scan,target_interact_miss,target_pursuit_miss,"
        "target_decoy_drift_miss,target_discovery_miss,target_handoff_miss"
    )
    dagger_focus_error_weight: float = 3.0
    dagger_focus_recovery_weight: float = 2.0
    dagger_focus_window: int = 1
    dagger_target_interact_focus_weight: float = 5.0
    dagger_target_discovery_min_map_size: int = 16
    dagger_target_discovery_focus_weight: float = 3.0
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
    dagger_max_replay_snippets_per_episode: int = 4
    dagger_solo_target_team_success_only: bool = False
    # RL
    rl_updates: int = 3000
    rollout_steps: int = 256
    rl_balanced_rollouts: bool = False
    rl_rollout_map_steps: str = ""
    rl_redundant_target_scan_penalty: float = 0.0
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


def _feedback_matrix(
    cfg: RecurrentConfig,
    num_agents: int,
    *,
    prev_actions: dict[int, int] | None = None,
    prev_msg_lens: dict[int, int] | None = None,
    info: dict | None = None,
    env: SyncOrSinkEnv | None = None,
    scan_state: dict | None = None,
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
    return rows


def _message_lengths(actions: dict[int, dict]) -> dict[int, int]:
    return {
        int(aid): len(action.get("message_tokens") or [])
        for aid, action in actions.items()
    }


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
    agents: list[int] = []
    for aid, action in actions.items():
        aid = int(aid)
        if int(action.get("action", -1)) != env.ACTION_INTERACT:
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


def _move_delta_for_action(env: SyncOrSinkEnv, action_id: int) -> tuple[int, int] | None:
    return {
        env.ACTION_UP: (0, -1),
        env.ACTION_DOWN: (0, 1),
        env.ACTION_LEFT: (-1, 0),
        env.ACTION_RIGHT: (1, 0),
    }.get(int(action_id))


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


def _signal_sync_response_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    actions: dict[int, dict],
    feedback: np.ndarray | None,
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
        teammate_scanned = float(feedback_arr[aid, 13]) > 0.0
        if teammate_scanned and (aid in target_interactors or aid in target_pursuers):
            responders.append(aid)
    return responders


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


def _signal_target_handoff_miss_agents(
    env: SyncOrSinkEnv,
    obs: dict,
    oracle_actions: dict[int, dict],
    model_actions: dict[int, dict],
    feedback: np.ndarray | None,
) -> list[int]:
    oracle_responders = set(_signal_sync_response_agents(env, obs, oracle_actions, feedback))
    if not oracle_responders:
        return []
    model_responders = set(_signal_sync_response_agents(env, obs, model_actions, feedback))
    return sorted(int(aid) for aid in oracle_responders if int(aid) not in model_responders)


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
    for aid in _signal_sync_response_agents(env, obs, actions, feedback):
        weighted[int(aid)] = max(float(weighted[int(aid)]), target_weight)
    return weighted


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
        "signal_decoy_drift_action_mask": [],
        "signal_decoy_drift_action_id": [],
    }


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
    rejected_target_mask = _signal_rejected_target_mask(obs, int(env.map_size), env.num_agents)
    bad_redundant_target_mask = _signal_bad_redundant_target_mask(env, obs)
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
        ep_data["signal_decoy_drift_action_mask"].append(0.0)
        ep_data["signal_decoy_drift_action_id"].append(-1)


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
        "signal_decoy_drift_action_mask": np.array(
            ep_data.get("signal_decoy_drift_action_mask", []),
            dtype=np.float32,
        ).reshape(-1, env.num_agents),
        "signal_decoy_drift_action_id": np.array(
            ep_data.get("signal_decoy_drift_action_id", []),
            dtype=np.int64,
        ).reshape(-1, env.num_agents),
    }
    episode.update(metadata)
    return episode


def _episode_count_transitions(episodes) -> int:
    return int(sum(ep["obs"].shape[0] * ep["obs"].shape[1] for ep in episodes))


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
    max_snippets = max(0, int(cfg.dagger_max_replay_snippets_per_episode))
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
    parent_success = bool(episode.get("success", False))

    snippets = []
    seen_windows = set()
    event_counts: dict[str, int] = {}
    for record in sorted(focus_records, key=lambda item: (int(item["step"]), str(item["event"]))):
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
        snippets.append(_slice_recurrent_episode(episode, start, end, **metadata))
        event_counts[event] = event_counts.get(event, 0) + 1
        if len(snippets) >= max_snippets:
            break
    return snippets


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


def _dagger_replay_success_only_events(cfg: RecurrentConfig) -> set[str]:
    return {
        item.strip()
        for item in str(cfg.dagger_replay_success_only_events or "").split(",")
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
            )
            actions = oracle_fn(obs, info, {"step": step})
            _append_labeled_step(ep_data, obs, actions, env, episode_cfg, feedback=feedback)
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
    decoy_drift_action_mask = decoy_drift_action_mask.float().reshape(-1)
    if sample_weight is not None:
        weights = decoy_drift_action_mask * sample_weight.float().reshape(-1)
    else:
        weights = decoy_drift_action_mask
    if float(weights.detach().sum().item()) <= 0.0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    action_ids = decoy_drift_action_id.long().reshape(-1).clamp(0, logits.shape[-1] - 1)
    bad_logits = logits.gather(1, action_ids.unsqueeze(1)).squeeze(1)
    other_logits = logits.masked_fill(
        nn.functional.one_hot(action_ids, num_classes=logits.shape[-1]).bool(),
        torch.finfo(logits.dtype).min,
    )
    other_logsumexp = torch.logsumexp(other_logits, dim=-1)
    loss_vec = nn.functional.softplus(bad_logits - other_logsumexp)
    return _weighted_mean(loss_vec, weights)


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
    optimizer = optim.Adam(model.parameters(), lr=cfg.bc_lr)
    send_pos_weight = _recurrent_comm_send_pos_weight(episodes, cfg, device) if cfg.comm else None

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
        rejected_target_count = 0
        rejected_target_pred_interact_count = 0
        rejected_target_interact_prob_sum = 0.0
        bad_redundant_target_loss_sum = 0.0
        bad_redundant_target_loss_steps = 0
        bad_redundant_target_count = 0
        bad_redundant_target_pred_interact_count = 0
        bad_redundant_target_interact_prob_sum = 0.0
        decoy_drift_action_loss_sum = 0.0
        decoy_drift_action_loss_steps = 0
        decoy_drift_action_count = 0
        decoy_drift_pred_bad_action_count = 0
        decoy_drift_bad_action_prob_sum = 0.0
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
                    rejected_weight = float(cfg.bc_signal_rejected_target_interact_loss_weight)
                    if rejected_weight > 0.0:
                        rejected_loss = _signal_rejected_target_interact_loss(
                            logits,
                            rejected_target_mask,
                            sample_weight=sample_weight,
                        )
                        loss = loss + rejected_weight * rejected_loss
                        rejected_target_loss_sum += float(rejected_loss.item())
                        rejected_target_loss_steps += 1
                    with torch.no_grad():
                        rejected_bool = rejected_target_mask > 0.0
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
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        print(
            f"[BC] epoch {epoch:3d} | loss {total_loss / max(loss_den, 1e-8):.4f} | "
            f"comm {avg_comm:.4f} | acc {acc:.3f} | "
            f"send {comm_send_label_rate:.3f}->{comm_pred_send_rate:.3f} | "
            f"len+ {comm_pred_positive_len_rate:.3f} | "
            f"rej_int {rejected_target_pred_interact_rate:.3f} | "
            f"bad_red_int {bad_redundant_target_pred_interact_rate:.3f} | "
            f"decoy_bad {decoy_drift_pred_bad_action_rate:.3f}"
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
                    f"{log_prefix}/signal_rejected_target_pred_interact_rate": float(
                        rejected_target_pred_interact_rate
                    ),
                    f"{log_prefix}/signal_rejected_target_mean_interact_prob": float(
                        rejected_target_mean_interact_prob
                    ),
                    f"{log_prefix}/signal_rejected_target_interact_loss_weight": float(
                        cfg.bc_signal_rejected_target_interact_loss_weight
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
                    f"{log_prefix}/action_acc": float(acc),
                    f"{log_prefix}/lr": float(optimizer.param_groups[0]["lr"]),
                    f"{log_prefix}/chunks": int(chunks),
                    f"{log_prefix}/dataset_episodes": int(len(episodes)),
                    f"{log_prefix}/dataset_transitions": int(_episode_count_transitions(episodes)),
                    **dict(log_context or {}),
                },
                context=f"{log_prefix} log",
            )

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
    focus_events = _dagger_focus_events(cfg)
    positive_replay_events = _dagger_positive_replay_events(cfg)
    positive_replay_event_counts: dict[str, int] = {}

    for ep in range(cfg.dagger_episodes):
        seed = 10000 + round_idx * 1000 + ep
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
        max_collect_steps = int(cfg.dagger_max_steps_per_episode)
        recovery_remaining = {aid: 0 for aid in range(env.num_agents)}
        focus_records: list[dict] = []
        deferred_team_records: list[dict] = []

        while not (done or truncated) and (max_collect_steps <= 0 or step < max_collect_steps):
            feedback = _feedback_matrix(
                episode_cfg,
                env.num_agents,
                prev_actions=prev_actions,
                prev_msg_lens=prev_msg_lens,
                info=last_info,
                env=env,
            )
            oracle_actions = oracle_fn(obs, info, {"step": step})
            step_weights = np.ones((env.num_agents,), dtype=np.float32)
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
            )
            target_interact_miss_agents = _signal_target_interact_miss_agents(
                env,
                oracle_actions,
                model_actions,
            )
            if target_interact_miss_agents and "target_interact_miss" in focus_events:
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
            target_handoff_miss_agents = _signal_target_handoff_miss_agents(
                env,
                obs,
                oracle_actions,
                model_actions,
                feedback,
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
            obs, _rewards, done, truncated, info = env.step(model_actions)
            last_info = info or {}
            event_names = _event_names_by_agent(last_info, env.num_agents)
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
            prev_actions = {aid: int(action["action"]) for aid, action in model_actions.items()}
            prev_msg_lens = _message_lengths(model_actions)
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
        "focus_events": focus_event_counts,
        "non_focus_events": non_focus_event_counts,
        "positive_replay_events": positive_replay_event_counts,
        "focused_state_updates": focused_state_updates,
        "solo_target_team_state_updates": solo_target_team_state_updates,
        "solo_target_team_deferred_records": deferred_solo_target_team_records,
        "solo_target_team_dropped_records": dropped_solo_target_team_records,
        "solo_target_team_success_only": bool(cfg.dagger_solo_target_team_success_only),
        "decoy_drift_action_labels": decoy_drift_action_labels,
        "recovery_state_updates": recovery_state_updates,
        "focus_error_weight": float(cfg.dagger_focus_error_weight),
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
        }
        if is_best:
            best_score = eval_score
            best_state = copy.deepcopy(model.state_dict())
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
                        "dagger/collect_decoy_drift_action_labels": int(
                            collect_summary.get("decoy_drift_action_labels", 0)
                        ),
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

    return model, history, all_episodes, best_row


def _decode_recurrent_actions(cfg: RecurrentConfig, model, obs, hidden, device, feedback: np.ndarray | None = None):
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
    "wrong_target_scans",
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
                "wrong_target_scans": 0.0,
            }
            target_scanners = set()
            target_scan_steps: dict[int, int] = {}
            reached_true_target = False
            reached_decoy_target = False

            while not (done or truncated):
                feedback = _feedback_matrix(
                    cfg,
                    env.num_agents,
                    prev_actions=prev_actions,
                    prev_msg_lens=prev_msg_lens,
                    info=last_info,
                    env=env,
                )
                actions, hidden = _decode_recurrent_actions(
                    cfg,
                    model,
                    obs,
                    hidden,
                    device,
                    feedback=feedback,
                )
                if cfg.scenario == "signal_hunt":
                    target = env.scenario_state.data.get("target")
                    if target is not None:
                        target_pos = tuple(target)
                        decoys = {tuple(pos) for pos in env.scenario_state.data.get("decoys", [])}
                        for aid, action in actions.items():
                            action_id = int(action.get("action", -1))
                            pos = tuple(env.agent_positions[int(aid)])
                            on_true_target = pos == target_pos
                            on_decoy_target = pos in decoys
                            if on_true_target or on_decoy_target:
                                signal_ep["target_tile_visits"] += 1.0
                            if on_true_target:
                                reached_true_target = True
                                signal_ep["true_target_visits"] += 1.0
                                if action_id != env.ACTION_INTERACT:
                                    signal_ep["true_target_unscanned_visits"] += 1.0
                            if on_decoy_target:
                                reached_decoy_target = True
                                signal_ep["decoy_target_visits"] += 1.0
                                if action_id == env.ACTION_INTERACT:
                                    signal_ep["wrong_target_scans"] += 1.0
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
                                signal_ep["target_scans"] += 1.0
                                target_scanners.add(int(aid))
                                target_scan_steps[int(aid)] = next_step
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
        self.scan_state = {"scan_log": {}, "scan_window": int(cfg.scan_window), "step": 0}
        self._has_policy_step = False
        self.model.eval()

    def reset(self, *args, **kwargs):
        del args, kwargs
        self.hidden = None
        self.prev_actions = {}
        self.prev_msg_lens = {}
        self.scan_state = {"scan_log": {}, "scan_window": int(self.cfg.scan_window), "step": 0}
        self._has_policy_step = False
        return None

    def metadata(self) -> dict:
        return {
            "algorithm": "recurrent_bc",
            "comm": self.cfg.comm,
            "hidden_dim": self.cfg.hidden_dim,
            "eval_send_threshold": self.cfg.eval_send_threshold,
            "obs_signal_scan_state": self.cfg.obs_signal_scan_state,
        }

    def _update_scan_state_from_info(self, info: dict, num_agents: int) -> None:
        if self.cfg.scenario != "signal_hunt" or not self.cfg.obs_signal_scan_state:
            return
        if not self._has_policy_step:
            self._has_policy_step = True
            return
        self.scan_state["step"] = int(self.scan_state.get("step", 0)) + 1
        for aid, names in _event_names_by_agent(info or {}, num_agents).items():
            if "target_scan" in names:
                self.scan_state.setdefault("scan_log", {})[int(aid)] = int(self.scan_state["step"])

    def __call__(self, obs: dict, info: dict, state: dict) -> dict[int, dict]:
        del state
        if self.hidden is None:
            self.hidden = self.model.init_hidden(len(obs), self.device)
        self._update_scan_state_from_info(info or {}, len(obs))
        feedback = _feedback_matrix(
            self.cfg,
            len(obs),
            prev_actions=self.prev_actions,
            prev_msg_lens=self.prev_msg_lens,
            info=info,
            scan_state=self.scan_state,
        )
        actions, self.hidden = _decode_recurrent_actions(
            self.cfg,
            self.model,
            obs,
            self.hidden,
            self.device,
            feedback=feedback,
        )
        self.prev_actions = {aid: int(action["action"]) for aid, action in actions.items()}
        self.prev_msg_lens = _message_lengths(actions)
        return actions


def load_recurrent_checkpoint_policy(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    eval_send_threshold: float | None = None,
) -> RecurrentCheckpointPolicy:
    device = resolve_device(str(device)) if isinstance(device, str) else device
    ckpt = torch.load(Path(path), map_location="cpu")
    raw_cfg = ckpt.get("config", {})
    allowed = {field.name for field in fields(RecurrentConfig)}
    cfg = RecurrentConfig(**{key: value for key, value in raw_cfg.items() if key in allowed})
    if eval_send_threshold is not None:
        cfg.eval_send_threshold = float(eval_send_threshold)
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
    model.load_state_dict(state)
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
    model.load_state_dict(state)
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
            redundant_target_scan_count += len(redundant_target_agents)
            _, penalty_sum = _apply_redundant_target_scan_penalty(
                rewards,
                redundant_target_agents,
                cfg.rl_redundant_target_scan_penalty,
            )
            redundant_target_scan_penalty_sum += penalty_sum
            next_info = info or {}
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
            f"red_scan {redundant_target_scan_count}"
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
    p.add_argument("--bc-signal-sync-response-weight", type=float, default=1.0)
    p.add_argument("--bc-signal-rejected-target-interact-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-bad-redundant-target-interact-loss-weight", type=float, default=0.0)
    p.add_argument("--bc-signal-decoy-drift-action-loss-weight", type=float, default=0.0)
    p.add_argument("--dagger-rounds", type=int, default=0)
    p.add_argument("--dagger-episodes", type=int, default=20)
    p.add_argument("--dagger-retrain-from-scratch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dagger-max-steps-per-episode", type=int, default=0)
    p.add_argument("--dagger-success-episode-weight", type=float, default=1.0)
    p.add_argument("--dagger-failed-episode-weight", type=float, default=0.25)
    p.add_argument(
        "--dagger-focus-events",
        default=(
            "decoy_scan,solo_target_scan,rejected_target_scan,"
            "bad_redundant_target_scan,target_interact_miss,target_pursuit_miss,"
            "target_decoy_drift_miss,target_discovery_miss,target_handoff_miss"
        ),
    )
    p.add_argument("--dagger-focus-error-weight", type=float, default=3.0)
    p.add_argument("--dagger-focus-recovery-weight", type=float, default=2.0)
    p.add_argument("--dagger-focus-window", type=int, default=1)
    p.add_argument("--dagger-target-interact-focus-weight", type=float, default=5.0)
    p.add_argument("--dagger-target-discovery-min-map-size", type=int, default=16)
    p.add_argument("--dagger-target-discovery-focus-weight", type=float, default=3.0)
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
    p.add_argument("--dagger-max-replay-snippets-per-episode", type=int, default=4)
    p.add_argument(
        "--dagger-solo-target-team-success-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Defer solo-target teammate upweights and apply them only to successful parent rollouts",
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
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed Python, NumPy, and Torch RNGs (default: 0)")
    p.add_argument("--save", default=None)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=3000)
    p.add_argument("--eval-seed-count", type=int, default=1)
    p.add_argument("--eval-map-sizes", default="")
    p.add_argument("--eval-send-threshold", type=float, default=0.25)
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
        bc_signal_sync_response_weight=args.bc_signal_sync_response_weight,
        bc_signal_rejected_target_interact_loss_weight=args.bc_signal_rejected_target_interact_loss_weight,
        bc_signal_bad_redundant_target_interact_loss_weight=(
            args.bc_signal_bad_redundant_target_interact_loss_weight
        ),
        bc_signal_decoy_drift_action_loss_weight=args.bc_signal_decoy_drift_action_loss_weight,
        dagger_rounds=args.dagger_rounds,
        dagger_episodes=args.dagger_episodes,
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
        dagger_max_replay_snippets_per_episode=args.dagger_max_replay_snippets_per_episode,
        dagger_solo_target_team_success_only=args.dagger_solo_target_team_success_only,
        rl_updates=args.rl_updates,
        rollout_steps=args.rollout_steps,
        rl_balanced_rollouts=args.rl_balanced_rollouts,
        rl_rollout_map_steps=args.rl_rollout_map_steps,
        rl_redundant_target_scan_penalty=args.rl_redundant_target_scan_penalty,
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
        device=args.device,
        seed=args.seed,
        save=args.save,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        eval_seed_count=args.eval_seed_count,
        eval_map_sizes=args.eval_map_sizes,
        eval_send_threshold=args.eval_send_threshold,
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
