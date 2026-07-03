from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.eval.success import episode_success
from syncorsink.policies.mappo_models import MAPPOActor, MAPPOCritic
from syncorsink.train.seed import set_global_seeds


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MAPPOConfig:
    # environment
    scenario: str = "signal_hunt"
    map_size: int = 8
    agents: int = 3
    fov_preset: str = "medium"
    comm: bool = False
    comm_token_limit: int = 24
    comm_vocab_size: int = 256
    comm_max_messages: int = 8
    comm_len_cost: float = 0.0
    comm_cost: float = 0.01
    max_steps: int = 300
    energy_preset: str = "hard"
    energy_private_monitor: bool = False
    # reward shaping
    pipeline_shaping: bool = False
    pipeline_shaping_scale: float = 0.01
    energy_shaping: bool = False
    energy_shaping_scale: float = 0.01
    signal_shaping: bool = False
    signal_shaping_scale: float = 0.01
    signal_scan_bonus: float = 0.0
    signal_joint_scan_bonus: float = 0.0
    signal_colocation_bonus: float = 0.0
    signal_colocation_radius: int = 2
    signal_comm_utility: float = 0.0
    # PPO hyperparameters
    updates: int = 20
    rollout_steps: int = 256
    epochs: int = 4
    minibatch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    entropy: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    # architecture
    shared_actor: bool = False
    backbone: str = "mlp"
    hidden_dim: int = 128
    critic_mode: str = "local"  # "local" (DTDE) or "central" (CTDE)
    learned_reward: Optional[str] = None  # path to reward model checkpoint (replaces shaping)
    learned_reward_weight: float = 1.0    # blend weight for learned reward
    bc_init: Optional[str] = None         # path to BC checkpoint to initialize actor weights
    bc_kl_coeff: float = 0.0              # KL penalty toward frozen BC policy (0 = disabled)
    bc_freeze_encoder: bool = False       # freeze encoder layers, only fine-tune heads
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    seed: Optional[int] = 0
    # logging
    wandb: bool = False
    wandb_project: str = "syncorsink"
    wandb_run: Optional[str] = None
    # checkpointing
    save: Optional[str] = None
    load: Optional[str] = None
    save_every: int = 5
    eval_every: int = 10
    eval_episodes: int = 5


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _flatten_array(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def flatten_obs(obs_agent: dict) -> np.ndarray:
    parts = [
        _flatten_array(obs_agent.get("local_grid")),
        _flatten_array(obs_agent.get("inventory", np.array([0], dtype=np.float32))),
        _flatten_array(obs_agent.get("self_pos", np.array([0, 0], dtype=np.float32))),
        _flatten_array(obs_agent.get("local_resource_types", np.zeros((1, 1), dtype=np.float32))),
        _flatten_array(obs_agent.get("local_node_types", np.zeros((1, 1), dtype=np.float32))),
        _flatten_array(obs_agent.get("local_node_energy", np.zeros((1, 1), dtype=np.float32))),
        _flatten_array(obs_agent.get("messages_tokens", np.zeros((1, 1), dtype=np.float32))),
        _flatten_array(obs_agent.get("message_from", np.zeros((1,), dtype=np.float32))),
        _flatten_array(obs_agent.get("goal_hint", np.zeros((1,), dtype=np.float32))),
        _flatten_array(obs_agent.get("action_mask", np.ones((8,), dtype=np.float32))),
    ]
    return np.concatenate(parts, axis=0)


def build_batch_obs(obs: dict, num_agents: int) -> np.ndarray:
    """Return (num_agents, obs_dim) array of flattened per-agent observations."""
    return np.stack([flatten_obs(obs[aid]) for aid in range(num_agents)])


def action_mask_from_flat_obs(obs_tensor: torch.Tensor, action_dim: int = 8) -> torch.Tensor:
    """Extract the flattened action mask appended by ``flatten_obs``."""
    return obs_tensor[..., -action_dim:].float()


def mask_action_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Set invalid action logits low enough that Categorical/argmax ignore them."""
    valid = action_mask > 0
    if valid.ndim != logits.ndim or valid.shape[-1] != logits.shape[-1]:
        raise ValueError(
            f"action_mask shape {tuple(valid.shape)} incompatible with logits {tuple(logits.shape)}"
        )
    all_invalid = valid.sum(dim=-1, keepdim=True) == 0
    valid = torch.where(all_invalid, torch.ones_like(valid, dtype=torch.bool), valid)
    return logits.masked_fill(~valid, -1e9)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, actors, critic, optimizer, step):
    torch.save(
        {
            "actors": [a.state_dict() for a in actors],
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(path, actors, critic, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    for a, sd in zip(actors, ckpt["actors"]):
        a.load_state_dict(sd)
    critic.load_state_dict(ckpt["critic"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Rollout helpers — batched actor forward
# ---------------------------------------------------------------------------

def _actor_forward_batched(actors, obs_tensor, num_agents, shared_actor, comm):
    """Run actor forward for all agents, return batched outputs.

    Args:
        actors: list of MAPPOActor modules
        obs_tensor: (num_agents, obs_dim)
        num_agents: int
        shared_actor: bool
        comm: bool

    Returns (comm=False): action_logits (N, action_dim)
    Returns (comm=True): (action_logits, send_logits, token_logits, len_logits)
    """
    if shared_actor:
        return actors[0](obs_tensor)

    # Per-agent actors: group indices by actor, forward each, reassemble
    if comm:
        action_parts = [None] * num_agents
        send_parts = [None] * num_agents
        token_parts = [None] * num_agents
        len_parts = [None] * num_agents
        for aid in range(num_agents):
            a_logits, s_logits, t_logits, l_logits = actors[aid](obs_tensor[aid:aid + 1])
            action_parts[aid] = a_logits
            send_parts[aid] = s_logits
            token_parts[aid] = t_logits
            len_parts[aid] = l_logits
        return (
            torch.cat(action_parts, dim=0),
            torch.cat(send_parts, dim=0),
            torch.cat(token_parts, dim=0),
            torch.cat(len_parts, dim=0),
        )
    else:
        parts = [actors[aid](obs_tensor[aid:aid + 1]) for aid in range(num_agents)]
        return torch.cat(parts, dim=0)


def _actor_forward_minibatch(actors, obs_batch, agent_ids, shared_actor, comm):
    """Run actor forward for a minibatch where each row may belong to a different agent.

    Args:
        actors: list of MAPPOActor modules
        obs_batch: (B, obs_dim)
        agent_ids: (B,) int tensor — which agent each row belongs to
        shared_actor: bool
        comm: bool
    """
    if shared_actor:
        return actors[0](obs_batch)

    # Group by agent, forward each, scatter back
    B = obs_batch.shape[0]
    unique_aids = agent_ids.unique()

    if comm:
        action_out = torch.zeros(B, actors[0].policy.linear.out_features, device=obs_batch.device)
        send_out = torch.zeros(B, 1, device=obs_batch.device)
        token_out = torch.zeros(B, actors[0].comm_token_limit, actors[0].comm_vocab_size, device=obs_batch.device)
        len_out = torch.zeros(B, actors[0].comm_token_limit + 1, device=obs_batch.device)
        for aid in unique_aids:
            mask = agent_ids == aid
            a_logits, s_logits, t_logits, l_logits = actors[aid.item()](obs_batch[mask])
            action_out[mask] = a_logits
            send_out[mask] = s_logits
            token_out[mask] = t_logits
            len_out[mask] = l_logits
        return action_out, send_out, token_out, len_out
    else:
        action_out = torch.zeros(B, actors[0].policy.linear.out_features, device=obs_batch.device)
        for aid in unique_aids:
            mask = agent_ids == aid
            action_out[mask] = actors[aid.item()](obs_batch[mask])
        return action_out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_mappo(cfg: MAPPOConfig):
    set_global_seeds(cfg.seed)
    env_config = SyncOrSinkConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        num_agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        comm_token_limit=cfg.comm_token_limit,
        token_vocab_size=cfg.comm_vocab_size,
        max_messages=cfg.comm_max_messages,
        comm_len_cost=cfg.comm_len_cost,
        comm_cost=cfg.comm_cost,
        energy_preset=cfg.energy_preset,
        energy_private_monitor=cfg.energy_private_monitor,
        pipeline_shaping=cfg.pipeline_shaping,
        pipeline_shaping_scale=cfg.pipeline_shaping_scale,
        energy_shaping=cfg.energy_shaping,
        energy_shaping_scale=cfg.energy_shaping_scale,
        signal_shaping=cfg.signal_shaping,
        signal_shaping_scale=cfg.signal_shaping_scale,
        signal_scan_bonus=cfg.signal_scan_bonus,
        signal_joint_scan_bonus=cfg.signal_joint_scan_bonus,
        signal_colocation_bonus=cfg.signal_colocation_bonus,
        signal_colocation_radius=cfg.signal_colocation_radius,
        signal_comm_utility=cfg.signal_comm_utility,
        max_steps=cfg.max_steps,
    )
    env = SyncOrSinkEnv(env_config)
    N = env.num_agents

    # --- Device ---
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    # Determine observation dimensions
    sample_obs, _ = env.reset(seed=0)
    obs_dim = flatten_obs(sample_obs[0]).shape[0]
    action_dim = 8

    # --- Learned reward model (optional, replaces/augments hand-crafted shaping) ---
    reward_model = None
    if cfg.learned_reward:
        from syncorsink.train.bc import RewardNet
        ckpt = torch.load(cfg.learned_reward, map_location="cpu")
        reward_model = RewardNet(ckpt["obs_dim"], hidden_dim=ckpt["hidden_dim"]).to(device)
        reward_model.load_state_dict(ckpt["model"])
        reward_model.eval()
        print(f"Loaded learned reward model from {cfg.learned_reward}")

    # --- Build actors ---
    actor_kwargs = dict(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=cfg.hidden_dim,
        backbone=cfg.backbone,
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    )
    if cfg.shared_actor:
        actors = [MAPPOActor(**actor_kwargs).to(device)]
    else:
        actors = [MAPPOActor(**actor_kwargs).to(device) for _ in range(N)]

    # --- BC warmstart: initialize actor weights from BC checkpoint ---
    bc_ref_actors = None  # frozen BC policy for KL regularization
    if cfg.bc_init:
        bc_ckpt = torch.load(cfg.bc_init, map_location="cpu")
        bc_state = bc_ckpt["model"]
        for actor in actors:
            actor_state = actor.state_dict()
            loaded = 0
            for key in bc_state:
                if key in actor_state and bc_state[key].shape == actor_state[key].shape:
                    actor_state[key] = bc_state[key]
                    loaded += 1
            actor.load_state_dict(actor_state)
        print(f"BC warmstart: loaded {loaded}/{len(bc_state)} params from {cfg.bc_init}")

        # Freeze encoder if requested (only fine-tune policy/comm heads)
        if cfg.bc_freeze_encoder:
            for actor in actors:
                for name, param in actor.named_parameters():
                    if "encoder" in name:
                        param.requires_grad = False
            print("BC warmstart: encoder layers frozen")

        # Create frozen BC reference for KL penalty
        if cfg.bc_kl_coeff > 0:
            import copy
            bc_ref_actors = []
            for actor in actors:
                ref = copy.deepcopy(actor)
                ref.eval()
                for p in ref.parameters():
                    p.requires_grad = False
                bc_ref_actors.append(ref)
            print(f"BC warmstart: KL penalty enabled (coeff={cfg.bc_kl_coeff})")

    # --- Build critic ---
    if cfg.critic_mode == "central":
        critic_input_dim = obs_dim * N
    else:
        critic_input_dim = obs_dim
    critic = MAPPOCritic(critic_input_dim, hidden_dim=cfg.hidden_dim).to(device)

    # --- Optimizer ---
    params = list(critic.parameters())
    for actor in actors:
        params += list(actor.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr, eps=1e-5)

    start_update = 0
    if cfg.load:
        start_update = load_checkpoint(cfg.load, actors, critic, optimizer)

    # --- W&B ---
    wandb_run = None
    if cfg.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project, name=cfg.wandb_run, config=vars(cfg)
            )
        except Exception as exc:
            print(f"wandb init failed, continuing without wandb: {exc}")

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------
    global_step = 0
    for update in range(start_update, cfg.updates):
        # LR annealing
        if cfg.anneal_lr:
            frac = 1.0 - update / cfg.updates
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * frac

        # ---- Rollout collection ----
        # Buffers: all tensors are (T, N, ...) where T=rollout_steps, N=num_agents
        obs_buf = []          # (T, N, obs_dim)
        act_buf = []          # (T, N)
        logp_buf = []         # (T, N)
        rew_buf = []          # (T, N)
        done_buf = []         # (T, N)
        val_buf = []          # (T, N)
        # Comm-specific buffers
        send_buf = []         # (T, N)
        token_buf = []        # (T, N, comm_token_limit)
        len_buf = []          # (T, N)

        # Episode tracking
        action_hist = np.zeros(8, dtype=np.int64)
        ep_returns = []
        ep_steps = []
        ep_comm = []
        comm_send_counts = 0
        comm_total_steps = 0
        comm_len_sum = 0.0
        comm_len_count = 0
        comm_len_samples = []
        comm_token_entropy_sum = 0.0

        obs, _ = env.reset(seed=update)
        ep_return = 0.0
        ep_step = 0
        ep_comm_tokens = 0

        for t in range(cfg.rollout_steps):
            obs_batch = build_batch_obs(obs, N)                 # (N, obs_dim)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            action_mask = action_mask_from_flat_obs(obs_tensor, action_dim)

            # --- Critic value ---
            with torch.no_grad():
                if cfg.critic_mode == "central":
                    critic_input = obs_tensor.reshape(1, -1)        # (1, N*obs_dim)
                    v = critic(critic_input).expand(N)              # (N,)
                else:
                    v = critic(obs_tensor)                          # (N,)

            # --- Actor forward ---
            with torch.no_grad():
                actor_out = _actor_forward_batched(
                    actors, obs_tensor, N, cfg.shared_actor, cfg.comm
                )

            if cfg.comm:
                logits, send_logits, token_logits, len_logits = actor_out
                logits = mask_action_logits(logits, action_mask)

                action_dist = torch.distributions.Categorical(logits=logits)
                send_dist = torch.distributions.Bernoulli(logits=send_logits.squeeze(-1))
                token_dist = torch.distributions.Categorical(logits=token_logits)
                len_dist = torch.distributions.Categorical(logits=len_logits)

                acts = action_dist.sample()
                send = send_dist.sample()
                token_samples = token_dist.sample()
                len_samples = len_dist.sample()

                # Joint log-prob: action + send_gate + (len + tokens) * send
                logp_action = action_dist.log_prob(acts)
                logp_send = send_dist.log_prob(send)
                token_mask = (
                    torch.arange(cfg.comm_token_limit, device=device)[None, :] < len_samples[:, None]
                ).float()
                logp_tokens = (token_dist.log_prob(token_samples) * token_mask).sum(dim=-1)
                logp_len = len_dist.log_prob(len_samples)
                logp = logp_action + logp_send + (logp_len + logp_tokens) * send

                # Build env actions
                actions = {}
                for aid in range(N):
                    if int(send[aid].item()) == 1 and int(len_samples[aid].item()) > 0:
                        msg_tokens = token_samples[aid][: int(len_samples[aid].item())].tolist()
                    else:
                        msg_tokens = []
                    actions[aid] = {"action": int(acts[aid].item()), "message_tokens": msg_tokens}

                send_buf.append(send.cpu())
                token_buf.append(token_samples.cpu())
                len_buf.append(len_samples.cpu())

                # Comm stats
                comm_total_steps += N
                comm_send_counts += int(send.sum().item())
                if int(send.sum().item()) > 0:
                    comm_len_sum += float(len_samples[send.bool()].sum().item())
                    comm_len_count += int(send.sum().item())
                    comm_len_samples.extend(len_samples[send.bool()].tolist())
                comm_token_entropy_sum += float(token_dist.entropy().mean().item())
            else:
                logits = actor_out
                logits = mask_action_logits(logits, action_mask)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logp = dist.log_prob(acts)
                actions = {
                    aid: {"action": int(acts[aid].item()), "message_tokens": []}
                    for aid in range(N)
                }

            # --- Step environment ---
            next_obs, rewards, done, truncated, info = env.step(actions)

            for a in acts.tolist():
                action_hist[a] += 1
            if "comm_tokens" in info:
                ep_comm_tokens += sum(info["comm_tokens"].values())

            # Augment rewards with learned reward model if available
            if reward_model is not None:
                with torch.no_grad():
                    # Rebuild obs without comm-dependent fields to match reward model's obs_dim
                    reward_obs = torch.tensor(
                        build_batch_obs(obs, N), dtype=torch.float32, device=device
                    )
                    # Truncate or pad to match reward model's expected input dim
                    rm_dim = reward_model.net[0].in_features - 16  # subtract action embed dim
                    if reward_obs.shape[1] > rm_dim:
                        reward_obs = reward_obs[:, :rm_dim]
                    elif reward_obs.shape[1] < rm_dim:
                        pad = torch.zeros(N, rm_dim - reward_obs.shape[1], device=device)
                        reward_obs = torch.cat([reward_obs, pad], dim=1)
                    learned_r = reward_model(
                        reward_obs,
                        acts.to(device) if acts.device != device else acts,
                    ).cpu()
                for i in range(N):
                    rewards[i] += cfg.learned_reward_weight * learned_r[i].item()

            # Store to buffers (keep on CPU to avoid GPU memory pressure during rollout)
            obs_buf.append(obs_tensor.cpu())
            act_buf.append(acts.cpu())
            logp_buf.append(logp.cpu())
            val_buf.append(v.cpu())
            rew_buf.append(
                torch.tensor([rewards[i] for i in range(N)], dtype=torch.float32)
            )
            done_buf.append(
                torch.tensor([float(done or truncated)] * N, dtype=torch.float32)
            )

            obs = next_obs
            ep_return += sum(rewards.values())
            ep_step += 1
            global_step += 1

            if done or truncated:
                ep_returns.append(ep_return)
                ep_steps.append(ep_step)
                ep_comm.append(ep_comm_tokens)
                obs, _ = env.reset(seed=update * cfg.rollout_steps + t + 1)
                ep_return = 0.0
                ep_step = 0
                ep_comm_tokens = 0

        # ---- Compute GAE per agent ----
        # Stack buffers: (T, N)
        values = torch.stack(val_buf)       # (T, N)
        rewards_t = torch.stack(rew_buf)    # (T, N)
        dones_t = torch.stack(done_buf)     # (T, N)

        # Bootstrap value for last step
        with torch.no_grad():
            last_obs_batch = build_batch_obs(obs, N)
            last_obs_tensor = torch.tensor(last_obs_batch, dtype=torch.float32, device=device)
            if cfg.critic_mode == "central":
                last_v = critic(last_obs_tensor.reshape(1, -1)).expand(N).cpu()
            else:
                last_v = critic(last_obs_tensor).cpu()

        advantages = torch.zeros_like(rewards_t)    # (T, N)
        gae = torch.zeros(N)
        for t in reversed(range(cfg.rollout_steps)):
            next_v = last_v if t == cfg.rollout_steps - 1 else values[t + 1]
            delta = rewards_t[t] + cfg.gamma * next_v * (1.0 - dones_t[t]) - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - dones_t[t]) * gae
            advantages[t] = gae
        returns = advantages + values  # (T, N)

        # ---- Flatten for minibatch PPO ----
        # Reshape from (T, N, ...) to (T*N, ...) and move to device
        T = cfg.rollout_steps
        obs_b = torch.stack(obs_buf).reshape(T * N, -1).to(device)
        act_b = torch.stack(act_buf).reshape(T * N).to(device)
        logp_old = torch.stack(logp_buf).reshape(T * N).to(device)
        adv_b = advantages.reshape(T * N).to(device)
        ret_b = returns.reshape(T * N).to(device)
        val_old = values.reshape(T * N).to(device)
        # Agent IDs for per-agent actor routing
        agent_ids = torch.arange(N).unsqueeze(0).expand(T, N).reshape(T * N).to(device)

        if cfg.comm:
            send_b = torch.stack(send_buf).reshape(T * N).to(device)
            token_b = torch.stack(token_buf).reshape(T * N, cfg.comm_token_limit).to(device)
            len_b = torch.stack(len_buf).reshape(T * N).to(device)

        # Critic input for minibatch
        if cfg.critic_mode == "central":
            joint_obs_per_step = torch.stack(obs_buf).reshape(T, N * obs_dim)
            critic_obs_b = joint_obs_per_step.unsqueeze(1).expand(T, N, N * obs_dim).reshape(T * N, N * obs_dim).to(device)
        else:
            critic_obs_b = obs_b

        # Normalize advantages
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # ---- PPO epochs ----
        total = T * N
        idx = np.arange(total)

        for epoch in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, total, cfg.minibatch):
                mb = idx[start: start + cfg.minibatch]
                # --- Actor forward ---
                actor_out = _actor_forward_minibatch(
                    actors, obs_b[mb], agent_ids[mb], cfg.shared_actor, cfg.comm
                )
                action_mask = action_mask_from_flat_obs(obs_b[mb], action_dim)

                if cfg.comm:
                    logits, send_logits, token_logits, len_logits = actor_out
                    logits = mask_action_logits(logits, action_mask)

                    action_dist = torch.distributions.Categorical(logits=logits)
                    send_dist = torch.distributions.Bernoulli(logits=send_logits.squeeze(-1))
                    token_dist = torch.distributions.Categorical(logits=token_logits)
                    len_dist = torch.distributions.Categorical(logits=len_logits)

                    new_logp_action = action_dist.log_prob(act_b[mb])
                    new_logp_send = send_dist.log_prob(send_b[mb])
                    t_mask = (
                        torch.arange(cfg.comm_token_limit, device=device)[None, :] < len_b[mb][:, None]
                    ).float()
                    new_logp_tokens = (token_dist.log_prob(token_b[mb]) * t_mask).sum(dim=-1)
                    new_logp_len = len_dist.log_prob(len_b[mb])
                    new_logp = (
                        new_logp_action
                        + new_logp_send
                        + (new_logp_len + new_logp_tokens) * send_b[mb]
                    )
                    entropy = (
                        action_dist.entropy().mean()
                        + send_dist.entropy().mean()
                        + token_dist.entropy().mean()
                        + len_dist.entropy().mean()
                    )
                else:
                    logits = actor_out
                    logits = mask_action_logits(logits, action_mask)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(act_b[mb])
                    entropy = dist.entropy().mean()

                # --- Policy loss (clipped surrogate) ---
                ratio = (new_logp - logp_old[mb]).exp()
                surr1 = ratio * adv_b[mb]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv_b[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value loss (clipped) ---
                new_values = critic(critic_obs_b[mb])
                v_clipped = val_old[mb] + torch.clamp(
                    new_values - val_old[mb], -cfg.value_clip, cfg.value_clip
                )
                value_loss = 0.5 * torch.max(
                    (ret_b[mb] - new_values).pow(2),
                    (ret_b[mb] - v_clipped).pow(2),
                ).mean()

                # --- KL penalty toward frozen BC policy ---
                kl_loss = torch.tensor(0.0, device=device)
                if bc_ref_actors is not None and cfg.bc_kl_coeff > 0:
                    with torch.no_grad():
                        bc_out = _actor_forward_minibatch(
                            bc_ref_actors, obs_b[mb], agent_ids[mb],
                            cfg.shared_actor, cfg.comm,
                        )
                    if cfg.comm:
                        bc_logits = bc_out[0]
                    else:
                        bc_logits = bc_out
                    bc_logits = mask_action_logits(bc_logits, action_mask)
                    bc_logprobs = torch.log_softmax(bc_logits, dim=-1)
                    bc_probs = bc_logprobs.exp()
                    current_logprobs = torch.log_softmax(logits, dim=-1)
                    kl_loss = (bc_probs * (bc_logprobs - current_logprobs)).sum(dim=-1).mean()

                # --- Total loss ---
                loss = policy_loss + value_loss - cfg.entropy * entropy + cfg.bc_kl_coeff * kl_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for pg in optimizer.param_groups for p in pg["params"]],
                    cfg.max_grad_norm,
                )
                optimizer.step()

        # ---- Logging ----
        comm_send_rate = (comm_send_counts / comm_total_steps) if comm_total_steps else 0.0
        comm_mean_len = (comm_len_sum / comm_len_count) if comm_len_count else 0.0
        comm_token_ent = (comm_token_entropy_sum / cfg.rollout_steps) if cfg.rollout_steps else 0.0

        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        mean_len = float(np.mean(ep_steps)) if ep_steps else 0.0
        print(
            f"update {update:4d} | loss {loss.item():.3f} | "
            f"pi {policy_loss.item():.3f} | v {value_loss.item():.3f} | "
            f"ent {entropy.item():.3f} | ret {mean_ret:.2f} | len {mean_len:.1f}"
        )

        if wandb_run is not None:
            log_payload = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "update": update,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "rollout/episodes": len(ep_returns),
                "rollout/mean_ep_return": mean_ret,
                "rollout/mean_ep_len": mean_len,
                "rollout/mean_ep_comm_tokens": float(np.mean(ep_comm)) if ep_comm else 0.0,
                "rollout/comm_send_rate": comm_send_rate,
                "rollout/comm_mean_len": comm_mean_len,
                "rollout/comm_token_entropy": comm_token_ent,
            }
            for i in range(8):
                log_payload[f"rollout/action_hist_{i}"] = int(action_hist[i])
            if comm_len_samples:
                try:
                    import wandb
                    log_payload["rollout/comm_len_hist"] = wandb.Histogram(comm_len_samples)
                except Exception:
                    pass
            wandb_run.log(log_payload)

        # ---- Periodic evaluation ----
        if cfg.eval_every > 0 and (update + 1) % cfg.eval_every == 0:
            eval_returns, eval_steps_list, eval_success = [], [], []
            eval_obs, _ = env.reset(seed=10000 + update)
            for ep in range(cfg.eval_episodes):
                ep_done = False
                ep_trunc = False
                steps = 0
                total_reward = 0.0
                info = {}
                while not (ep_done or ep_trunc):
                    obs_batch = build_batch_obs(eval_obs, N)
                    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                    action_mask = action_mask_from_flat_obs(obs_tensor, action_dim)
                    with torch.no_grad():
                        actor_out = _actor_forward_batched(
                            actors, obs_tensor, N, cfg.shared_actor, cfg.comm
                        )
                    if cfg.comm:
                        logits, send_logits, token_logits, len_logits = actor_out
                        logits = mask_action_logits(logits, action_mask)
                        acts = torch.argmax(logits, dim=-1)
                        send = (torch.sigmoid(send_logits.squeeze(-1)) > 0.5).long()
                        token_samples = torch.argmax(token_logits, dim=-1)
                        len_samples = torch.argmax(len_logits, dim=-1)
                        actions = {}
                        for aid in range(N):
                            if int(send[aid].item()) == 1 and int(len_samples[aid].item()) > 0:
                                msg = token_samples[aid][: int(len_samples[aid].item())].tolist()
                            else:
                                msg = []
                            actions[aid] = {"action": int(acts[aid].item()), "message_tokens": msg}
                    else:
                        logits = actor_out
                        logits = mask_action_logits(logits, action_mask)
                        acts = torch.argmax(logits, dim=-1)
                        actions = {
                            aid: {"action": int(acts[aid].item()), "message_tokens": []}
                            for aid in range(N)
                        }
                    eval_obs, rewards, ep_done, ep_trunc, info = env.step(actions)
                    total_reward += sum(rewards.values())
                    steps += 1
                eval_returns.append(total_reward)
                eval_steps_list.append(steps)
                success = episode_success(cfg.scenario, ep_done, info)
                eval_success.append(1.0 if success else 0.0)
                eval_obs, _ = env.reset(seed=10000 + update + ep + 1)

            print(
                f"  eval | ret {np.mean(eval_returns):.2f} | "
                f"steps {np.mean(eval_steps_list):.1f} | "
                f"success {np.mean(eval_success):.2f}"
            )
            if wandb_run is not None:
                wandb_run.log({
                    "eval/mean_return": float(np.mean(eval_returns)),
                    "eval/mean_steps": float(np.mean(eval_steps_list)),
                    "eval/success_rate": float(np.mean(eval_success)),
                    "eval/update": update,
                })

        # ---- Checkpointing ----
        if cfg.save and (update + 1) % cfg.save_every == 0:
            save_checkpoint(cfg.save, actors, critic, optimizer, update + 1)

    if cfg.save:
        save_checkpoint(cfg.save, actors, critic, optimizer, cfg.updates)
    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAPPO training for SyncOrSink")
    # environment
    parser.add_argument("--scenario", default="signal_hunt")
    parser.add_argument("--map-size", type=int, default=8)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--fov-preset", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--comm", action="store_true")
    parser.add_argument("--comm-token-limit", type=int, default=24)
    parser.add_argument("--comm-vocab-size", type=int, default=256)
    parser.add_argument("--comm-max-messages", type=int, default=8)
    parser.add_argument("--comm-len-cost", type=float, default=0.0)
    parser.add_argument("--comm-cost", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--energy-preset", default="hard")
    parser.add_argument("--energy-private-monitor", action="store_true")
    # reward shaping
    parser.add_argument("--pipeline-shaping", action="store_true")
    parser.add_argument("--pipeline-shaping-scale", type=float, default=0.01)
    parser.add_argument("--energy-shaping", action="store_true")
    parser.add_argument("--energy-shaping-scale", type=float, default=0.01)
    parser.add_argument("--signal-shaping", action="store_true")
    parser.add_argument("--signal-shaping-scale", type=float, default=0.01)
    parser.add_argument("--signal-scan-bonus", type=float, default=0.0,
                        help="Small bonus for solo scan on target")
    parser.add_argument("--signal-joint-scan-bonus", type=float, default=0.0,
                        help="Large bonus when partner also scanned within window (near-miss)")
    parser.add_argument("--signal-colocation-bonus", type=float, default=0.0,
                        help="Bonus when 2+ agents near target and at least one interacts")
    parser.add_argument("--signal-colocation-radius", type=int, default=2)
    parser.add_argument("--signal-comm-utility", type=float, default=0.0,
                        help="Bonus for sending message that precedes teammate's useful action")
    # PPO
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--anneal-lr", action="store_true")
    # architecture
    parser.add_argument("--shared-actor", action="store_true")
    parser.add_argument("--backbone", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--critic-mode", default="local", choices=["local", "central"],
                        help="local=DTDE (default), central=CTDE")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Device for training (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed Python, NumPy, and Torch RNGs (default: 0)")
    parser.add_argument("--learned-reward", default=None,
                        help="Path to learned reward model checkpoint (replaces/augments shaping)")
    parser.add_argument("--learned-reward-weight", type=float, default=1.0,
                        help="Weight for learned reward signal")
    parser.add_argument("--bc-init", default=None,
                        help="Path to BC checkpoint to initialize actor weights (IL→RL warmstart)")
    parser.add_argument("--bc-kl-coeff", type=float, default=0.0,
                        help="KL penalty toward frozen BC policy (prevents RL from destroying BC init)")
    parser.add_argument("--bc-freeze-encoder", action="store_true",
                        help="Freeze encoder layers, only fine-tune heads")
    # logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
    # checkpointing
    parser.add_argument("--save", default=None, help="Checkpoint path")
    parser.add_argument("--load", default=None, help="Checkpoint path")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    cfg = MAPPOConfig(
        scenario=args.scenario,
        map_size=args.map_size,
        agents=args.agents,
        fov_preset=args.fov_preset,
        comm=args.comm,
        comm_token_limit=args.comm_token_limit,
        comm_vocab_size=args.comm_vocab_size,
        comm_max_messages=args.comm_max_messages,
        comm_len_cost=args.comm_len_cost,
        comm_cost=args.comm_cost,
        energy_preset=args.energy_preset,
        energy_private_monitor=args.energy_private_monitor,
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
        max_steps=args.max_steps,
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch=args.minibatch,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        value_clip=args.value_clip,
        entropy=args.entropy,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        anneal_lr=args.anneal_lr,
        shared_actor=args.shared_actor,
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        critic_mode=args.critic_mode,
        seed=args.seed,
        learned_reward=args.learned_reward,
        learned_reward_weight=args.learned_reward_weight,
        bc_init=args.bc_init,
        bc_kl_coeff=args.bc_kl_coeff,
        bc_freeze_encoder=args.bc_freeze_encoder,
        device=args.device,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        save=args.save,
        load=args.load,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )
    train_mappo(cfg)


if __name__ == "__main__":
    main()
