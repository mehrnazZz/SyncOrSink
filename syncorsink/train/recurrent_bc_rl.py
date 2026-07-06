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
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
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
    agents: int = 3
    fov_preset: str = "easy"
    max_steps: int = 300
    energy_preset: str = "easy"
    oracle_type: str = "oracle_strong"
    obs_exploration_memory: bool = False
    obs_exploration_age: bool = False
    obs_feedback: bool = False
    obs_normalize_tokens: bool = False
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
    dagger_rounds: int = 0
    dagger_episodes: int = 20
    dagger_retrain_from_scratch: bool = True
    dagger_max_steps_per_episode: int = 0
    dagger_success_episode_weight: float = 1.0
    dagger_failed_episode_weight: float = 0.25
    dagger_focus_events: str = "decoy_scan,solo_target_scan"
    dagger_focus_error_weight: float = 3.0
    dagger_focus_recovery_weight: float = 2.0
    dagger_focus_window: int = 1
    dagger_focus_replay: bool = False
    dagger_replay_pre_steps: int = 2
    dagger_replay_post_steps: int = 2
    dagger_replay_weight: float = 1.0
    dagger_max_replay_snippets_per_episode: int = 4
    # RL
    rl_updates: int = 3000
    rollout_steps: int = 256
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
    rl_eval_every: int = 5
    rl_eval_episodes: int = 20
    rl_eval_seed: int = 10000
    rl_restore_best: bool = True
    rl_save_best: bool = True
    rl_best_save: Optional[str] = None
    # device
    device: str = "auto"
    seed: Optional[int] = 0
    # output
    save: Optional[str] = None
    eval_episodes: int = 100
    eval_seed: int = 3000
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


def _scale_nonnegative(values, denominator: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    denom = max(1.0, float(denominator))
    return np.where(arr >= 0.0, arr / denom, -1.0).astype(np.float32)


def _normalize_recurrent_obs_agent(obs_agent: dict, cfg: RecurrentConfig) -> dict:
    if not cfg.obs_normalize_tokens:
        return obs_agent
    token_denom = max(float(cfg.comm_vocab_size - 1), float(cfg.map_size - 1), 1.0)
    norm = dict(obs_agent)
    norm["self_pos"] = _scale_nonnegative(
        obs_agent.get("self_pos", np.zeros((2,), dtype=np.float32)),
        cfg.map_size - 1,
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


def _flatten_recurrent_obs(obs_agent: dict, cfg: RecurrentConfig, feedback: np.ndarray | None = None) -> np.ndarray:
    obs_agent = _normalize_recurrent_obs_agent(obs_agent, cfg)
    flat = flatten_obs(
        obs_agent,
        include_exploration_memory=cfg.obs_exploration_memory,
        include_exploration_age=cfg.obs_exploration_age,
    )
    if cfg.obs_feedback:
        if feedback is None:
            feedback = np.zeros((12,), dtype=np.float32)
        feedback = np.asarray(feedback, dtype=np.float32).reshape(-1)
        flat = np.concatenate([flat[:-8], feedback, flat[-8:]], axis=0)
    return flat


def _build_recurrent_obs_batch(
    obs: dict,
    num_agents: int,
    cfg: RecurrentConfig,
    feedback: np.ndarray | None = None,
) -> np.ndarray:
    if feedback is None:
        feedback = np.zeros((num_agents, 12), dtype=np.float32) if cfg.obs_feedback else None
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


def _feedback_matrix(
    cfg: RecurrentConfig,
    num_agents: int,
    *,
    prev_actions: dict[int, int] | None = None,
    prev_msg_lens: dict[int, int] | None = None,
    info: dict | None = None,
) -> np.ndarray | None:
    if not cfg.obs_feedback:
        return None
    prev_actions = prev_actions or {}
    prev_msg_lens = prev_msg_lens or {}
    rows = np.zeros((num_agents, 12), dtype=np.float32)
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
    return rows


def _message_lengths(actions: dict[int, dict]) -> dict[int, int]:
    return {
        int(aid): len(action.get("message_tokens") or [])
        for aid, action in actions.items()
    }


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
    return {"obs": [], "actions": [], "msg_tokens": [], "msg_lens": [], "step_weights": []}


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
    sliced.update(metadata)
    return sliced


def _focus_replay_episodes(
    episode: dict,
    focus_records: list[dict],
    cfg: RecurrentConfig,
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
    if replay_weight <= 0.0:
        return []

    snippets = []
    seen_windows = set()
    for record in sorted(focus_records, key=lambda item: (int(item["step"]), str(item["event"]))):
        step = int(record["step"])
        start = max(0, step - pre_steps)
        end = min(total_steps, step + post_steps + 1)
        if end <= start:
            continue
        key = (start, end, str(record["event"]))
        if key in seen_windows:
            continue
        seen_windows.add(key)
        snippets.append(_slice_recurrent_episode(
            episode,
            start,
            end,
            source="dagger_focus_replay",
            round=episode.get("round"),
            seed=episode.get("seed"),
            parent_success=episode.get("success"),
            parent_capped=episode.get("capped"),
            success=episode.get("success", False),
            capped=episode.get("capped", False),
            weight=replay_weight,
            trigger_event=str(record["event"]),
            trigger_step=step,
            trigger_agents=[int(aid) for aid in record.get("agents", [])],
            replay_start_step=start,
            replay_end_step=end,
            steps=end - start,
        ))
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


def _solo_target_interactors(env: SyncOrSinkEnv, actions: dict[int, dict]) -> list[int]:
    if getattr(env.config, "scenario", None) != "signal_hunt":
        return []
    target = env.scenario_state.data.get("target")
    if target is None:
        return []
    interactors = [
        int(aid)
        for aid, action in actions.items()
        if int(action.get("action", -1)) == env.ACTION_INTERACT
        and tuple(env.agent_positions[int(aid)]) == tuple(target)
    ]
    return interactors if len(interactors) == 1 else []


def collect_episode_demos(cfg: RecurrentConfig):
    """Collect oracle demonstrations as full episodes (not shuffled transitions)."""
    set_global_seeds(cfg.seed)
    env = _build_env(cfg)
    oracle_fn = _make_oracle_policy(env, cfg)

    episodes = []
    for ep in range(cfg.demo_episodes):
        obs, info = env.reset(seed=ep)
        ep_data = _new_episode_sequence()
        done, truncated = False, False
        step = 0
        prev_actions: dict[int, int] = {}
        prev_msg_lens: dict[int, int] = {}
        prev_info: dict = {}
        while not (done or truncated):
            feedback = _feedback_matrix(
                cfg,
                env.num_agents,
                prev_actions=prev_actions,
                prev_msg_lens=prev_msg_lens,
                info=prev_info,
            )
            actions = oracle_fn(obs, info, {"step": step})
            _append_labeled_step(ep_data, obs, actions, env, cfg, feedback=feedback)
            obs, rewards, done, truncated, info = env.step(actions)
            prev_actions = {aid: int(action["action"]) for aid, action in actions.items()}
            prev_msg_lens = _message_lengths(actions)
            prev_info = info or {}
            step += 1

        success = episode_success(cfg.scenario, done, info)
        if success:
            episodes.append(_finalize_episode_sequence(
                ep_data,
                env,
                cfg,
                source="expert",
                seed=ep,
                success=True,
                capped=False,
                weight=1.0,
                steps=step,
            ))

        if (ep + 1) % 50 == 0:
            print(f"  collected {ep + 1}/{cfg.demo_episodes}, {len(episodes)} successful")

    print(f"Collected {len(episodes)} successful episodes")
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


def _recurrent_comm_loss(
    send_logits,
    token_logits,
    len_logits,
    msg_tokens,
    msg_lens,
    send_pos_weight=None,
    sample_weight=None,
):
    send_target = (msg_lens > 0).float()
    send_loss = nn.functional.binary_cross_entropy_with_logits(
        send_logits.squeeze(-1),
        send_target,
        pos_weight=send_pos_weight,
        reduction="none",
    )
    send_loss = _weighted_mean(send_loss, sample_weight)
    len_loss = nn.functional.cross_entropy(len_logits, msg_lens, reduction="none")
    len_loss = _weighted_mean(len_loss, sample_weight)
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
    return send_loss + len_loss + tok_loss


def train_recurrent_bc(cfg: RecurrentConfig, episodes, device, model: MAPPORecurrentActor | None = None):
    """Train recurrent BC via truncated BPTT on episode sequences."""
    set_global_seeds(cfg.seed)
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
                    if cfg.comm and cfg.bc_comm_loss_weight > 0:
                        comm_loss = _recurrent_comm_loss(
                            send_logits,
                            token_logits,
                            len_logits,
                            msg_seq[t],
                            msg_len_seq[t],
                            send_pos_weight=send_pos_weight,
                            sample_weight=sample_weight,
                        )
                        loss = loss + cfg.bc_comm_loss_weight * comm_loss
                        total_comm_loss += comm_loss.item()
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
        print(
            f"[BC] epoch {epoch:3d} | loss {total_loss / max(loss_den, 1e-8):.4f} | "
            f"comm {avg_comm:.4f} | acc {acc:.3f}"
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
    env = _build_env(cfg)
    oracle_fn = _make_oracle_policy(env, cfg)
    model.eval()
    episodes = []
    successes = 0
    capped_episodes = 0
    total_steps = 0
    stored_steps = 0
    base_episodes = 0
    replay_episodes = 0
    focus_event_counts: dict[str, int] = {}
    focused_state_updates = 0
    recovery_state_updates = 0
    replay_trigger_counts: dict[str, int] = {}
    focus_events = _dagger_focus_events(cfg)

    for ep in range(cfg.dagger_episodes):
        seed = 10000 + round_idx * 1000 + ep
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

        while not (done or truncated) and (max_collect_steps <= 0 or step < max_collect_steps):
            feedback = _feedback_matrix(
                cfg,
                env.num_agents,
                prev_actions=prev_actions,
                prev_msg_lens=prev_msg_lens,
                info=last_info,
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
                cfg,
                feedback=feedback,
                step_weight=step_weights,
            )
            model_actions, hidden = _decode_recurrent_actions(
                cfg,
                model,
                obs,
                hidden,
                device,
                feedback=feedback,
            )
            solo_target_agents = _solo_target_interactors(env, model_actions)
            if solo_target_agents and "solo_target_scan" in focus_events:
                focus_event_counts["solo_target_scan"] = focus_event_counts.get("solo_target_scan", 0) + len(solo_target_agents)
                focus_records.append({
                    "event": "solo_target_scan",
                    "step": step,
                    "agents": list(solo_target_agents),
                })
                focused_state_updates += _scale_latest_agent_weights(
                    ep_data,
                    num_agents=env.num_agents,
                    agent_ids=solo_target_agents,
                    weight=cfg.dagger_focus_error_weight,
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
        if ep_data["obs"]:
            base_episode = _finalize_episode_sequence(
                ep_data,
                env,
                cfg,
                source="dagger",
                round=round_idx,
                seed=seed,
                success=success,
                capped=capped,
                weight=_dagger_episode_weight(cfg, success),
                steps=step,
            )
            episodes.append(base_episode)
            base_episodes += 1
            replay_snippets = _focus_replay_episodes(base_episode, focus_records, cfg)
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
        "focused_state_updates": focused_state_updates,
        "recovery_state_updates": recovery_state_updates,
        "focus_error_weight": float(cfg.dagger_focus_error_weight),
        "focus_recovery_weight": float(cfg.dagger_focus_recovery_weight),
        "focus_replay_enabled": bool(cfg.dagger_focus_replay),
        "replay_trigger_events": replay_trigger_counts,
        "replay_transitions": _episode_count_transitions(
            [ep for ep in episodes if ep.get("source") == "dagger_focus_replay"]
        ),
        "replay_effective_transitions": _episode_count_effective_transitions(
            [ep for ep in episodes if ep.get("source") == "dagger_focus_replay"]
        ),
    }
    return episodes, summary


def train_recurrent_bc_dagger(cfg: RecurrentConfig, episodes, device):
    """Run recurrent DAgger over full episode sequences."""
    import copy

    all_episodes = list(episodes)
    history = []
    model = None
    best_state = None
    best_row = None
    best_score = None

    for round_idx in range(cfg.dagger_rounds + 1):
        print(
            f"\n=== Recurrent DAgger round {round_idx} | "
            f"episodes {len(all_episodes)} | transitions {_episode_count_transitions(all_episodes)} | "
            f"effective {_episode_count_effective_transitions(all_episodes):.1f} ==="
        )
        start_model = None if cfg.dagger_retrain_from_scratch else model
        model = train_recurrent_bc(cfg, all_episodes, device, model=start_model)
        eval_result = evaluate_recurrent_policy(cfg, model, device)
        eval_score = _recurrent_eval_score(eval_result)
        round_row = {
            "round": round_idx,
            "dataset_episodes": len(all_episodes),
            "dataset_transitions": _episode_count_transitions(all_episodes),
            "dataset_effective_transitions": _episode_count_effective_transitions(all_episodes),
            "dataset_sources": _episode_source_counts(all_episodes),
            "retrain_from_scratch": cfg.dagger_retrain_from_scratch or round_idx == 0,
            "eval": eval_result,
            "eval_score": eval_score,
        }
        print(json.dumps({"recurrent_dagger": round_row}, indent=2, sort_keys=True))
        if best_score is None or eval_score > best_score:
            best_score = eval_score
            best_row = dict(round_row)
            best_state = copy.deepcopy(model.state_dict())

        if round_idx < cfg.dagger_rounds:
            new_episodes, collect_summary = collect_recurrent_dagger_episodes(
                cfg,
                model,
                device,
                round_idx=round_idx,
            )
            all_episodes.extend(new_episodes)
            round_row["collect"] = collect_summary
            print(json.dumps({"dagger_collect": collect_summary}, indent=2, sort_keys=True))

        history.append(round_row)

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


def evaluate_recurrent_policy(cfg: RecurrentConfig, model, device) -> dict:
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
                "unique_target_scanners": 0.0,
            }
            target_scanners = set()

            while not (done or truncated):
                feedback = _feedback_matrix(
                    cfg,
                    env.num_agents,
                    prev_actions=prev_actions,
                    prev_msg_lens=prev_msg_lens,
                    info=last_info,
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
                        for aid, action in actions.items():
                            if (
                                int(action.get("action", -1)) == env.ACTION_INTERACT
                                and tuple(env.agent_positions[int(aid)]) == tuple(target)
                            ):
                                signal_ep["target_scans"] += 1.0
                                target_scanners.add(int(aid))
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
        result["signal"] = {
            "avg_decoy_scans": float(np.mean([row["decoy_scans"] for row in signal_rows])),
            "avg_clues_found": float(np.mean([row["clues_found"] for row in signal_rows])),
            "avg_target_scans": float(np.mean([row["target_scans"] for row in signal_rows])),
            "avg_unique_target_scanners": float(np.mean([row["unique_target_scanners"] for row in signal_rows])),
        }
    return result


def _recurrent_eval_score(result: dict) -> tuple[float, float, float, float]:
    signal = result.get("signal") or {}
    return (
        float(result.get("success_rate", 0.0)),
        -float(signal.get("avg_decoy_scans", 0.0)),
        float(result.get("avg_return", 0.0)),
        -float(result.get("avg_steps", 0.0)),
    )


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
        self.model.eval()

    def reset(self, *args, **kwargs):
        del args, kwargs
        self.hidden = None
        self.prev_actions = {}
        self.prev_msg_lens = {}
        return None

    def metadata(self) -> dict:
        return {
            "algorithm": "recurrent_bc",
            "comm": self.cfg.comm,
            "hidden_dim": self.cfg.hidden_dim,
            "eval_send_threshold": self.cfg.eval_send_threshold,
        }

    def __call__(self, obs: dict, info: dict, state: dict) -> dict[int, dict]:
        del state
        if self.hidden is None:
            self.hidden = self.model.init_hidden(len(obs), self.device)
        feedback = _feedback_matrix(
            self.cfg,
            len(obs),
            prev_actions=self.prev_actions,
            prev_msg_lens=self.prev_msg_lens,
            info=info,
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


def train_recurrent_rl(cfg: RecurrentConfig, model, device):
    """Fine-tune recurrent policy with PPO, carrying hidden state across steps."""
    import copy

    if cfg.rl_updates <= 0:
        return model

    set_global_seeds(cfg.seed)
    env = _build_env(cfg)
    N = env.num_agents
    sample_obs, _ = env.reset(seed=0)
    obs_dim = _build_recurrent_obs_batch(
        sample_obs,
        N,
        cfg,
        feedback=_feedback_matrix(cfg, N),
    ).shape[1]

    critic = MAPPOCritic(obs_dim, hidden_dim=cfg.hidden_dim).to(device)
    model.train()

    # Frozen BC reference for KL
    bc_ref = copy.deepcopy(model)
    bc_ref.eval()
    for p in bc_ref.parameters():
        p.requires_grad = False

    params = list(model.parameters()) + list(critic.parameters())
    optimizer = optim.Adam(params, lr=cfg.rl_lr, eps=1e-5)

    wandb_run = None
    if cfg.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run, config=vars(cfg))
        except Exception:
            pass

    eval_cfg = replace(
        cfg,
        eval_episodes=max(1, int(cfg.rl_eval_episodes)),
        eval_seed=int(cfg.rl_eval_seed),
    )
    best_path = _recurrent_rl_best_path(cfg)
    initial_eval = evaluate_recurrent_policy(eval_cfg, model, device)
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
        try:
            eval_log = {
                "eval/success_rate": float(initial_eval["success_rate"]),
                "eval/mean_return": float(initial_eval["avg_return"]),
                "eval/mean_steps": float(initial_eval["avg_steps"]),
                "eval/update": -1,
                "eval/is_best": 1,
                "eval/best_success_rate": float(best_eval["success_rate"]),
            }
            signal = initial_eval.get("signal") or {}
            for key, value in signal.items():
                eval_log[f"eval/signal/{key}"] = float(value)
            wandb_run.log(eval_log)
        except Exception as exc:
            print(f"wandb initial eval log failed, disabling wandb for this run: {exc}")
            try:
                wandb_run.finish()
            except Exception:
                pass
            wandb_run = None

    for update in range(cfg.rl_updates):
        # LR annealing
        frac = 1.0 - update / max(cfg.rl_updates, 1)
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.rl_lr * frac

        # Rollout with hidden state
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf = [], []
        send_buf, token_buf, len_buf = [], [], []
        hidden_buf = []
        ep_returns, ep_steps = [], []
        ep_comm = []
        ep_ret, ep_step = 0.0, 0
        ep_comm_tokens = 0
        comm_send_counts = 0
        comm_total_steps = 0
        comm_len_sum = 0.0
        comm_len_count = 0
        comm_token_entropy_sum = 0.0
        action_hist = np.zeros(8, dtype=np.int64)

        obs, _info = env.reset(seed=update)
        hidden = model.init_hidden(N, device)
        prev_actions: dict[int, int] = {}
        prev_msg_lens: dict[int, int] = {}
        last_info: dict = {}

        for t in range(cfg.rollout_steps):
            feedback = _feedback_matrix(
                cfg,
                N,
                prev_actions=prev_actions,
                prev_msg_lens=prev_msg_lens,
                info=last_info,
            )
            obs_batch = _build_recurrent_obs_batch(obs, N, cfg, feedback=feedback)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            action_mask = action_mask_from_flat_obs(obs_tensor)

            with torch.no_grad():
                if cfg.comm:
                    logits, send_logits, token_logits, len_logits, new_hidden = model(obs_tensor, hidden)
                else:
                    logits, new_hidden = model(obs_tensor, hidden)
                v = critic(obs_tensor)
            logits = mask_action_logits(logits, action_mask_from_flat_obs(obs_tensor))

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
                for aid in range(N):
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
                comm_total_steps += N
                comm_send_counts += int(send.sum().item())
                if int(send.sum().item()) > 0:
                    comm_len_sum += float(len_samples[send.bool()].sum().item())
                    comm_len_count += int(send.sum().item())
                comm_token_entropy_sum += float(token_dist.entropy().mean().item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logp = dist.log_prob(acts)
                actions = {aid: {"action": int(acts[aid].item()), "message_tokens": []} for aid in range(N)}
            next_obs, rewards, done, truncated, info = env.step(actions)

            obs_buf.append(obs_tensor.cpu())
            act_buf.append(acts.cpu())
            logp_buf.append(logp.cpu())
            val_buf.append(v.cpu())
            hidden_buf.append((hidden[0].cpu(), hidden[1].cpu()))
            rew_buf.append(torch.tensor([rewards[i] for i in range(N)], dtype=torch.float32))
            done_buf.append(torch.tensor([float(done or truncated)] * N, dtype=torch.float32))

            for action_id in acts.detach().cpu().tolist():
                action_hist[int(action_id)] += 1
            if "comm_tokens" in (info or {}):
                ep_comm_tokens += sum(info["comm_tokens"].values())

            hidden = new_hidden
            obs = next_obs
            last_info = info or {}
            prev_actions = {aid: int(action["action"]) for aid, action in actions.items()}
            prev_msg_lens = _message_lengths(actions)
            ep_ret += sum(rewards.values())
            ep_step += 1

            if done or truncated:
                ep_returns.append(ep_ret)
                ep_steps.append(ep_step)
                ep_comm.append(ep_comm_tokens)
                obs, _info = env.reset(seed=update * cfg.rollout_steps + t + 1)
                hidden = model.init_hidden(N, device)
                prev_actions = {}
                prev_msg_lens = {}
                last_info = {}
                ep_ret, ep_step = 0.0, 0
                ep_comm_tokens = 0

        # GAE
        values = torch.stack(val_buf)
        rewards_t = torch.stack(rew_buf)
        dones_t = torch.stack(done_buf)

        with torch.no_grad():
            feedback = _feedback_matrix(
                cfg,
                N,
                prev_actions=prev_actions,
                prev_msg_lens=prev_msg_lens,
                info=last_info,
            )
            last_obs = torch.tensor(
                _build_recurrent_obs_batch(obs, N, cfg, feedback=feedback),
                dtype=torch.float32,
                device=device,
            )
            last_v = critic(last_obs).cpu()

        advantages = torch.zeros_like(rewards_t)
        gae = torch.zeros(N)
        for t in reversed(range(cfg.rollout_steps)):
            next_v = last_v if t == cfg.rollout_steps - 1 else values[t + 1]
            delta = rewards_t[t] + cfg.gamma * next_v * (1.0 - dones_t[t]) - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - dones_t[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        # PPO update — process in sequential chunks to maintain hidden state
        T = cfg.rollout_steps
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
            total_entropy = 0.0
            total_steps = 0

            for t in range(T):
                obs_t = obs_buf[t].to(device)
                act_t = act_buf[t].to(device)
                idx = t * N

                # Reset hidden at episode boundaries
                if t > 0 and bool(dones_t[t - 1].any().item()):
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
                        bc_logits, _bc_send, _bc_tokens, _bc_len, bc_hidden_replay = bc_ref(
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

                loss = policy_loss + value_loss - cfg.entropy_coeff * entropy + cfg.bc_kl_coeff * kl

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()

                # Detach hidden for next step
                hidden_replay = (hidden_replay[0].detach(), hidden_replay[1].detach())

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl += kl.item()
                total_entropy += entropy.item()
                total_steps += 1

        denom = max(total_steps, 1)
        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        mean_len = float(np.mean(ep_steps)) if ep_steps else 0.0
        mean_comm = float(np.mean(ep_comm)) if ep_comm else 0.0
        comm_send_rate = comm_send_counts / comm_total_steps if comm_total_steps else 0.0
        comm_mean_len = comm_len_sum / comm_len_count if comm_len_count else 0.0
        comm_token_entropy = comm_token_entropy_sum / cfg.rollout_steps if cfg.rollout_steps else 0.0
        print(
            f"update {update:4d} | pi {total_policy_loss / denom:.3f} | "
            f"v {total_value_loss / denom:.3f} | kl {total_kl / denom:.4f} | "
            f"ent {total_entropy / denom:.3f} | ret {mean_ret:.2f} | len {mean_len:.1f}"
        )

        if wandb_run is not None:
            try:
                log_payload = {
                    "policy_loss": total_policy_loss / denom,
                    "value_loss": total_value_loss / denom,
                    "kl": total_kl / denom,
                    "entropy": total_entropy / denom,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "rollout/episodes": len(ep_returns),
                    "rollout/mean_ep_return": mean_ret,
                    "rollout/mean_ep_len": mean_len,
                    "rollout/mean_ep_comm_tokens": mean_comm,
                    "rollout/comm_send_rate": comm_send_rate,
                    "rollout/comm_mean_len": comm_mean_len,
                    "rollout/comm_token_entropy": comm_token_entropy,
                    "update": update,
                }
                for i in range(8):
                    log_payload[f"rollout/action_hist_{i}"] = int(action_hist[i])
                wandb_run.log(log_payload)
            except Exception as exc:
                print(f"wandb log failed, disabling wandb for this run: {exc}")
                try:
                    wandb_run.finish()
                except Exception:
                    pass
                wandb_run = None

        should_eval = update == cfg.rl_updates - 1 or (
            cfg.rl_eval_every > 0 and (update + 1) % cfg.rl_eval_every == 0
        )
        if should_eval:
            eval_result = evaluate_recurrent_policy(eval_cfg, model, device)
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
                try:
                    eval_log = {
                        "eval/success_rate": float(eval_result["success_rate"]),
                        "eval/mean_return": float(eval_result["avg_return"]),
                        "eval/mean_steps": float(eval_result["avg_steps"]),
                        "eval/update": update,
                        "eval/is_best": int(is_best),
                        "eval/best_success_rate": float(best_eval["success_rate"]),
                    }
                    signal = eval_result.get("signal") or {}
                    for key, value in signal.items():
                        eval_log[f"eval/signal/{key}"] = float(value)
                    wandb_run.log(eval_log)
                except Exception as exc:
                    print(f"wandb eval log failed, disabling wandb for this run: {exc}")
                    try:
                        wandb_run.finish()
                    except Exception:
                        pass
                    wandb_run = None

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
    if wandb_run is not None:
        wandb_run.finish()
    return model


def main():
    p = argparse.ArgumentParser(description="Recurrent BC→RL for SyncOrSink scenarios")
    p.add_argument("--scenario", default="pipeline_assembly")
    p.add_argument("--map-size", type=int, default=8)
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
    p.add_argument("--dagger-rounds", type=int, default=0)
    p.add_argument("--dagger-episodes", type=int, default=20)
    p.add_argument("--dagger-retrain-from-scratch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dagger-max-steps-per-episode", type=int, default=0)
    p.add_argument("--dagger-success-episode-weight", type=float, default=1.0)
    p.add_argument("--dagger-failed-episode-weight", type=float, default=0.25)
    p.add_argument("--dagger-focus-events", default="decoy_scan,solo_target_scan")
    p.add_argument("--dagger-focus-error-weight", type=float, default=3.0)
    p.add_argument("--dagger-focus-recovery-weight", type=float, default=2.0)
    p.add_argument("--dagger-focus-window", type=int, default=1)
    p.add_argument("--dagger-focus-replay", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dagger-replay-pre-steps", type=int, default=2)
    p.add_argument("--dagger-replay-post-steps", type=int, default=2)
    p.add_argument("--dagger-replay-weight", type=float, default=1.0)
    p.add_argument("--dagger-max-replay-snippets-per-episode", type=int, default=4)
    p.add_argument("--rl-updates", type=int, default=3000)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--rl-epochs", type=int, default=2)
    p.add_argument("--minibatch-seqs", type=int, default=8)
    p.add_argument("--rl-lr", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--bc-kl-coeff", type=float, default=0.5)
    p.add_argument("--rl-eval-every", type=int, default=5)
    p.add_argument("--rl-eval-episodes", type=int, default=20)
    p.add_argument("--rl-eval-seed", type=int, default=10000)
    p.add_argument("--rl-restore-best", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rl-save-best", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rl-best-save", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed Python, NumPy, and Torch RNGs (default: 0)")
    p.add_argument("--save", default=None)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=3000)
    p.add_argument("--eval-send-threshold", type=float, default=0.25)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="syncorsink")
    p.add_argument("--wandb-run", default=None)
    args = p.parse_args()

    cfg = RecurrentConfig(
        scenario=args.scenario,
        map_size=args.map_size,
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
        dagger_focus_replay=args.dagger_focus_replay,
        dagger_replay_pre_steps=args.dagger_replay_pre_steps,
        dagger_replay_post_steps=args.dagger_replay_post_steps,
        dagger_replay_weight=args.dagger_replay_weight,
        dagger_max_replay_snippets_per_episode=args.dagger_max_replay_snippets_per_episode,
        rl_updates=args.rl_updates,
        rollout_steps=args.rollout_steps,
        rl_epochs=args.rl_epochs,
        minibatch_seqs=args.minibatch_seqs,
        rl_lr=args.rl_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        entropy_coeff=args.entropy_coeff,
        max_grad_norm=args.max_grad_norm,
        bc_kl_coeff=args.bc_kl_coeff,
        rl_eval_every=args.rl_eval_every,
        rl_eval_episodes=args.rl_eval_episodes,
        rl_eval_seed=args.rl_eval_seed,
        rl_restore_best=args.rl_restore_best,
        rl_save_best=args.rl_save_best,
        rl_best_save=args.rl_best_save,
        device=args.device,
        seed=args.seed,
        save=args.save,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        eval_send_threshold=args.eval_send_threshold,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
    )

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    print("=== Step 1: Collecting oracle demos ===")
    episodes = collect_episode_demos(cfg)

    dagger_history = []
    if cfg.dagger_rounds > 0:
        print("\n=== Step 2: Training recurrent DAgger ===")
        model, dagger_history, episodes, best_dagger_row = train_recurrent_bc_dagger(cfg, episodes, device)
        eval_result = best_dagger_row["eval"]
    else:
        best_dagger_row = None
        print("\n=== Step 2: Training recurrent BC ===")
        model = train_recurrent_bc(cfg, episodes, device)

        print("\n=== Step 2b: Evaluating recurrent BC ===")
        eval_result = evaluate_recurrent_policy(cfg, model, device)
        print(json.dumps({"eval_recurrent_bc": eval_result}, indent=2, sort_keys=True))

    if cfg.save and cfg.rl_updates <= 0:
        os.makedirs(os.path.dirname(cfg.save) or ".", exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "config": vars(cfg),
            "eval_recurrent_bc": eval_result,
            "dagger_history": dagger_history,
            "best_dagger_round": best_dagger_row,
        }, cfg.save)
        print(f"Saved to {cfg.save}")

    if cfg.rl_updates > 0:
        print("\n=== Step 3: RL fine-tuning ===")
        train_recurrent_rl(cfg, model, device)


if __name__ == "__main__":
    main()
