"""Behavioral Cloning from oracle demonstrations.

Pipeline:
  1. collect_demos() — run oracle policy, save (obs, action, comm) tuples
  2. train_bc() — supervised learning on collected data
  3. Evaluate via eval_run.py --policy il
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.train.mappo import flatten_obs, resolve_device
from syncorsink.policies.mappo_models import MAPPOActor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BCConfig:
    # environment
    scenario: str = "signal_hunt"
    map_size: int = 8
    agents: int = 2
    fov_preset: str = "easy"
    max_steps: int = 300
    energy_preset: str = "easy"
    # data collection
    demo_episodes: int = 100
    oracle_type: str = "oracle_strong"  # "oracle" or "oracle_strong"
    demo_path: str = "demos/signal_hunt_oracle.npz"
    # training
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 128
    # communication
    comm: bool = True
    comm_token_limit: int = 8
    comm_vocab_size: int = 32
    comm_loss_weight: float = 0.1  # low to avoid diluting action learning
    # two-phase training
    two_phase: bool = False
    phase1_epochs: int = 30   # action-only epochs
    phase2_epochs: int = 20   # comm fine-tuning epochs
    phase2_lr: float = 3e-4   # lower LR for comm fine-tuning
    # DAgger
    dagger_rounds: int = 0    # 0 = vanilla BC, >0 = DAgger iterations
    dagger_episodes: int = 20  # episodes per DAgger round
    # device
    device: str = "auto"
    # output
    save: Optional[str] = None
    wandb: bool = False
    wandb_project: str = "syncorsink"
    wandb_run: Optional[str] = None


# ---------------------------------------------------------------------------
# Step 1: Collect demonstrations
# ---------------------------------------------------------------------------

def collect_demos(cfg: BCConfig):
    """Run oracle policy and collect (obs, action) pairs from successful episodes."""
    from syncorsink.policies.oracle import (
        pipeline_oracle, pipeline_oracle_strong,
        energy_oracle, energy_oracle_strong,
        signal_hunt_oracle, signal_hunt_oracle_strong,
    )
    from syncorsink.policies.comm_wrapper import wrap_oracle_with_comm

    env_cfg = SyncOrSinkConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        num_agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        max_steps=cfg.max_steps,
        energy_preset=cfg.energy_preset,
        comm_token_limit=cfg.comm_token_limit,
        token_vocab_size=cfg.comm_vocab_size,
    )
    env = SyncOrSinkEnv(env_cfg)

    oracle_map = {
        "signal_hunt": {"oracle": signal_hunt_oracle, "oracle_strong": signal_hunt_oracle_strong},
        "energy_grid": {"oracle": energy_oracle, "oracle_strong": energy_oracle_strong},
        "pipeline_assembly": {"oracle": pipeline_oracle, "oracle_strong": pipeline_oracle_strong},
    }
    base_type = cfg.oracle_type.replace("_comm", "")
    oracle_fn = oracle_map[cfg.scenario][base_type](env)
    if cfg.oracle_type.endswith("_comm"):
        oracle_fn = wrap_oracle_with_comm(oracle_fn, env)

    all_obs = []
    all_actions = []
    all_msgs = []
    success_count = 0
    total_transitions = 0

    for ep in range(cfg.demo_episodes):
        obs, info = env.reset(seed=ep)
        ep_obs, ep_actions, ep_msgs = [], [], []
        done, truncated = False, False
        step = 0

        while not (done or truncated):
            actions = oracle_fn(obs, info, {"step": step})

            # Store per-agent transitions
            for aid in range(env.num_agents):
                flat = flatten_obs(obs[aid])
                act = int(actions[aid]["action"])
                msg_tokens = actions[aid].get("message_tokens", [])
                ep_obs.append(flat)
                ep_actions.append(act)
                ep_msgs.append(msg_tokens)

            obs, rewards, done, truncated, info = env.step(actions)
            step += 1

        # Only keep successful episodes for clean demonstrations
        # For energy_grid, check info["success"]; for others, check done
        if cfg.scenario == "energy_grid":
            success = bool(info.get("success", False))
        else:
            success = bool(done)

        if success:
            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)
            all_msgs.extend(ep_msgs)
            success_count += 1
            total_transitions += len(ep_obs)

        if (ep + 1) % 20 == 0:
            print(f"  collected {ep + 1}/{cfg.demo_episodes} episodes, "
                  f"{success_count} successful, {total_transitions} transitions")

    if success_count == 0:
        print("WARNING: no successful episodes collected!")
        return

    obs_arr = np.stack(all_obs)
    act_arr = np.array(all_actions, dtype=np.int64)

    # Pad message tokens to fixed length
    padded_msgs = np.zeros((len(all_msgs), cfg.comm_token_limit), dtype=np.int64)
    msg_lens = np.zeros(len(all_msgs), dtype=np.int64)
    for i, tokens in enumerate(all_msgs):
        L = min(len(tokens), cfg.comm_token_limit)
        if L > 0:
            padded_msgs[i, :L] = tokens[:L]
        msg_lens[i] = L

    os.makedirs(os.path.dirname(cfg.demo_path) or ".", exist_ok=True)
    np.savez_compressed(
        cfg.demo_path,
        obs=obs_arr,
        actions=act_arr,
        msg_tokens=padded_msgs,
        msg_lens=msg_lens,
    )
    print(f"Saved {total_transitions} transitions from {success_count}/{cfg.demo_episodes} "
          f"successful episodes to {cfg.demo_path}")
    print(f"  obs shape: {obs_arr.shape}, action distribution: {np.bincount(act_arr, minlength=8)}")


def collect_llm_demos(cfg: BCConfig, trace_path: str):
    """Extract (obs, action, comm) demonstrations from LLM trace JSONL files.

    Only keeps transitions from successful episodes. Replays the environment
    to get matching observations since traces don't store raw obs arrays.
    """
    import json

    env_cfg = SyncOrSinkConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        num_agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        max_steps=cfg.max_steps,
        energy_preset=cfg.energy_preset,
        comm_mode="text",
    )
    env = SyncOrSinkEnv(env_cfg)

    ACTION_MAP = {
        "up": 0, "down": 1, "left": 2, "right": 3,
        "stay": 4, "interact": 5, "pickup": 6, "drop": 7,
    }

    # Parse trace file — group by episode
    episodes = {}
    with open(trace_path, "r") as f:
        for line in f:
            row = json.loads(line)
            ep = row["episode"]
            episodes.setdefault(ep, []).append(row)

    all_obs, all_actions, all_msgs = [], [], []
    success_count = 0
    total_transitions = 0

    for ep_idx in sorted(episodes.keys()):
        steps = episodes[ep_idx]
        # Check if episode was successful
        last_step = steps[-1]
        if not last_step.get("done", False):
            continue

        # Replay environment to get observations
        obs, info = env.reset(seed=ep_idx)
        ep_obs, ep_actions, ep_msgs = [], [], []

        for step_data in steps:
            actions_raw = step_data.get("actions", {})

            for aid in range(env.num_agents):
                aid_str = str(aid)
                if aid_str not in actions_raw:
                    continue
                agent_action = actions_raw[aid_str]

                # Parse action
                act = agent_action.get("action", 4)
                if isinstance(act, str):
                    act = ACTION_MAP.get(act.lower(), 4)
                act = int(act)

                # Parse message tokens from text
                msg_text = agent_action.get("message_text", "")
                # For LLM demos, we store the text as token IDs (simple hash)
                msg_tokens = []
                if msg_text and isinstance(msg_text, str) and msg_text.strip():
                    words = msg_text.strip().split()[:cfg.comm_token_limit]
                    msg_tokens = [hash(w) % cfg.comm_vocab_size for w in words]

                flat = flatten_obs(obs[aid])
                ep_obs.append(flat)
                ep_actions.append(act)
                ep_msgs.append(msg_tokens)

            # Step environment with parsed actions
            env_actions = {}
            for aid in range(env.num_agents):
                aid_str = str(aid)
                if aid_str in actions_raw:
                    agent_action = actions_raw[aid_str]
                    act = agent_action.get("action", 4)
                    if isinstance(act, str):
                        act = ACTION_MAP.get(act.lower(), 4)
                    msg_text = agent_action.get("message_text", "")
                    env_actions[aid] = {
                        "action": int(act),
                        "message_text": msg_text if msg_text else None,
                    }
                else:
                    env_actions[aid] = {"action": 4}

            obs, rewards, done, truncated, info = env.step(env_actions)
            if done or truncated:
                break

        all_obs.extend(ep_obs)
        all_actions.extend(ep_actions)
        all_msgs.extend(ep_msgs)
        success_count += 1
        total_transitions += len(ep_obs)

    if success_count == 0:
        print("WARNING: no successful episodes found in trace!")
        return

    obs_arr = np.stack(all_obs)
    act_arr = np.array(all_actions, dtype=np.int64)

    padded_msgs = np.zeros((len(all_msgs), cfg.comm_token_limit), dtype=np.int64)
    msg_lens = np.zeros(len(all_msgs), dtype=np.int64)
    for i, tokens in enumerate(all_msgs):
        L = min(len(tokens), cfg.comm_token_limit)
        if L > 0:
            padded_msgs[i, :L] = tokens[:L]
        msg_lens[i] = L

    os.makedirs(os.path.dirname(cfg.demo_path) or ".", exist_ok=True)
    np.savez_compressed(
        cfg.demo_path,
        obs=obs_arr,
        actions=act_arr,
        msg_tokens=padded_msgs,
        msg_lens=msg_lens,
    )
    print(f"Saved {total_transitions} transitions from {success_count} successful "
          f"LLM episodes to {cfg.demo_path}")
    print(f"  obs shape: {obs_arr.shape}, action distribution: {np.bincount(act_arr, minlength=8)}")
    msg_rate = (msg_lens > 0).sum() / len(msg_lens) if len(msg_lens) > 0 else 0
    print(f"  comm rate: {msg_rate:.2%} of transitions have messages")


# ---------------------------------------------------------------------------
# Step 2: Train BC
# ---------------------------------------------------------------------------

def _build_bc_model(obs_dim: int, cfg: BCConfig, device):
    return MAPPOActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        backbone="mlp",
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    ).to(device)


def _compute_comm_loss(model, obs_b, msg_all, msg_lens, mb, cfg, device):
    """Compute communication loss (send + length + token content)."""
    _, send_logits, token_logits, len_logits = model(obs_b)
    msg_b = msg_all[mb].to(device)
    mlen_b = msg_lens[mb].to(device)
    send_target = (mlen_b > 0).float()

    send_loss = nn.functional.binary_cross_entropy_with_logits(
        send_logits.squeeze(-1), send_target
    )
    len_loss = nn.functional.cross_entropy(len_logits, mlen_b)

    t_mask = (torch.arange(cfg.comm_token_limit, device=device)[None, :] < mlen_b[:, None]).float()
    if t_mask.sum() > 0:
        tok_loss = (nn.functional.cross_entropy(
            token_logits.reshape(-1, cfg.comm_vocab_size),
            msg_b.reshape(-1),
            reduction="none",
        ).reshape(token_logits.shape[0], -1) * t_mask).sum() / t_mask.sum()
    else:
        tok_loss = torch.tensor(0.0, device=device)

    return send_loss + len_loss + tok_loss


def _run_bc_epochs(model, optimizer, obs_all, act_all, msg_all, msg_lens,
                   cfg, device, epochs, phase_name, comm_weight,
                   wandb_run=None, epoch_offset=0):
    """Core BC training loop used by both standard and two-phase training."""
    action_loss_fn = nn.CrossEntropyLoss()
    N = obs_all.shape[0]
    idx = np.arange(N)

    for epoch in range(epochs):
        np.random.shuffle(idx)
        total_loss = 0.0
        total_action_loss = 0.0
        total_comm_loss = 0.0
        action_correct = 0
        batches = 0

        for start in range(0, N, cfg.batch_size):
            mb = idx[start:start + cfg.batch_size]
            obs_b = obs_all[mb].to(device)
            act_b = act_all[mb].to(device)

            if cfg.comm:
                logits, _, _, _ = model(obs_b)
                a_loss = action_loss_fn(logits, act_b)

                if comm_weight > 0:
                    comm_loss = _compute_comm_loss(model, obs_b, msg_all, msg_lens, mb, cfg, device)
                    loss = a_loss + comm_weight * comm_loss
                    total_comm_loss += comm_loss.item()
                else:
                    loss = a_loss
            else:
                logits = model(obs_b)
                a_loss = action_loss_fn(logits, act_b)
                loss = a_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_action_loss += a_loss.item()
            action_correct += (logits.argmax(dim=-1) == act_b).sum().item()
            batches += 1

        acc = action_correct / N
        avg_loss = total_loss / batches
        avg_a_loss = total_action_loss / batches
        avg_c_loss = total_comm_loss / batches if cfg.comm and comm_weight > 0 else 0.0
        global_epoch = epoch_offset + epoch

        print(f"[{phase_name}] epoch {global_epoch:3d} | loss {avg_loss:.4f} | "
              f"a_loss {avg_a_loss:.4f} | c_loss {avg_c_loss:.4f} | acc {acc:.3f}")

        if wandb_run is not None:
            wandb_run.log({
                "epoch": global_epoch,
                "phase": phase_name,
                "loss": avg_loss,
                "action_loss": avg_a_loss,
                "comm_loss": avg_c_loss,
                "action_accuracy": acc,
            })

    return acc


def _save_bc_model(model, obs_dim, cfg):
    if cfg.save:
        os.makedirs(os.path.dirname(cfg.save) or ".", exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "obs_dim": obs_dim,
            "hidden_dim": cfg.hidden_dim,
            "comm": cfg.comm,
            "comm_token_limit": cfg.comm_token_limit,
            "comm_vocab_size": cfg.comm_vocab_size,
        }, cfg.save)
        print(f"Saved model to {cfg.save}")


def train_bc(cfg: BCConfig):
    """Train a policy network via behavioral cloning on collected demonstrations.

    Supports two-phase training when cfg.two_phase=True:
      Phase 1: Train action head only (comm_weight=0), comm heads frozen.
      Phase 2: Fine-tune comm heads with lower LR, action head frozen.
    """
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    # Load demonstrations
    data = np.load(cfg.demo_path)
    obs_all = torch.tensor(data["obs"], dtype=torch.float32)
    act_all = torch.tensor(data["actions"], dtype=torch.long)
    msg_all = torch.tensor(data["msg_tokens"], dtype=torch.long)
    msg_lens = torch.tensor(data["msg_lens"], dtype=torch.long)
    N = obs_all.shape[0]
    obs_dim = obs_all.shape[1]
    print(f"Loaded {N} transitions, obs_dim={obs_dim}")

    model = _build_bc_model(obs_dim, cfg, device)

    # W&B
    wandb_run = None
    if cfg.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run, config=vars(cfg))
        except Exception as exc:
            print(f"wandb init failed: {exc}")

    if cfg.two_phase and cfg.comm:
        # --- Phase 1: action-only (freeze comm heads) ---
        print(f"=== Phase 1: action-only ({cfg.phase1_epochs} epochs) ===")
        for name, param in model.named_parameters():
            if "comm_" in name:
                param.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
        _run_bc_epochs(model, optimizer, obs_all, act_all, msg_all, msg_lens,
                       cfg, device, cfg.phase1_epochs, "phase1-action", comm_weight=0.0,
                       wandb_run=wandb_run)

        # --- Phase 2: comm fine-tuning (freeze encoder+action, unfreeze comm) ---
        print(f"=== Phase 2: comm fine-tuning ({cfg.phase2_epochs} epochs) ===")
        for name, param in model.named_parameters():
            if "comm_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.phase2_lr)
        _run_bc_epochs(model, optimizer, obs_all, act_all, msg_all, msg_lens,
                       cfg, device, cfg.phase2_epochs, "phase2-comm", comm_weight=1.0,
                       wandb_run=wandb_run, epoch_offset=cfg.phase1_epochs)

        # Unfreeze everything for saving
        for param in model.parameters():
            param.requires_grad = True
    else:
        # --- Standard single-phase training ---
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        _run_bc_epochs(model, optimizer, obs_all, act_all, msg_all, msg_lens,
                       cfg, device, cfg.epochs, "train", comm_weight=cfg.comm_loss_weight,
                       wandb_run=wandb_run)

    _save_bc_model(model, obs_dim, cfg)
    if wandb_run is not None:
        wandb_run.finish()
    return model


# ---------------------------------------------------------------------------
# Step 3: DAgger (Dataset Aggregation)
# ---------------------------------------------------------------------------

def train_bc_dagger(cfg: BCConfig):
    """Behavioral Cloning with DAgger: iteratively collect on-policy data and retrain.

    Round 0: Train on initial oracle demos (standard BC).
    Round 1..N: Run current policy, collect visited states, query oracle
    for actions at those states, add to dataset, retrain from scratch.
    """
    from syncorsink.policies.oracle import (
        pipeline_oracle, pipeline_oracle_strong,
        energy_oracle, energy_oracle_strong,
        signal_hunt_oracle, signal_hunt_oracle_strong,
    )
    from syncorsink.policies.comm_wrapper import wrap_oracle_with_comm

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    # Setup environment and oracle
    env_cfg = SyncOrSinkConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        num_agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        max_steps=cfg.max_steps,
        energy_preset=cfg.energy_preset,
        comm_token_limit=cfg.comm_token_limit,
        token_vocab_size=cfg.comm_vocab_size,
    )
    env = SyncOrSinkEnv(env_cfg)

    oracle_map = {
        "signal_hunt": {"oracle": signal_hunt_oracle, "oracle_strong": signal_hunt_oracle_strong},
        "energy_grid": {"oracle": energy_oracle, "oracle_strong": energy_oracle_strong},
        "pipeline_assembly": {"oracle": pipeline_oracle, "oracle_strong": pipeline_oracle_strong},
    }
    base_type = cfg.oracle_type.replace("_comm", "")
    oracle_fn = oracle_map[cfg.scenario][base_type](env)
    if cfg.oracle_type.endswith("_comm"):
        oracle_fn = wrap_oracle_with_comm(oracle_fn, env)

    # Load initial demos
    data = np.load(cfg.demo_path)
    obs_list = [data["obs"]]
    act_list = [data["actions"]]
    msg_list = [data["msg_tokens"]]
    mlen_list = [data["msg_lens"]]
    obs_dim = data["obs"].shape[1]
    print(f"DAgger: loaded {data['obs'].shape[0]} initial transitions")

    # W&B
    wandb_run = None
    if cfg.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run, config=vars(cfg))
        except Exception as exc:
            print(f"wandb init failed: {exc}")

    model = None
    for rnd in range(cfg.dagger_rounds + 1):
        # Aggregate dataset
        obs_all = torch.tensor(np.concatenate(obs_list), dtype=torch.float32)
        act_all = torch.tensor(np.concatenate(act_list), dtype=torch.long)
        msg_all = torch.tensor(np.concatenate(msg_list), dtype=torch.long)
        msg_lens_all = torch.tensor(np.concatenate(mlen_list), dtype=torch.long)
        N = obs_all.shape[0]
        print(f"\n=== DAgger round {rnd} | dataset: {N} transitions ===")

        # Train from scratch each round (clean optimization landscape)
        model = _build_bc_model(obs_dim, cfg, device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        _run_bc_epochs(model, optimizer, obs_all, act_all, msg_all, msg_lens_all,
                       cfg, device, cfg.epochs, f"dagger-r{rnd}",
                       comm_weight=cfg.comm_loss_weight if cfg.comm else 0.0,
                       wandb_run=wandb_run, epoch_offset=rnd * cfg.epochs)

        # Collect on-policy data using current model, labeled by oracle
        if rnd < cfg.dagger_rounds:
            print(f"  Collecting {cfg.dagger_episodes} on-policy episodes...")
            new_obs, new_acts, new_msgs, new_mlens = [], [], [], []
            model.eval()

            for ep in range(cfg.dagger_episodes):
                obs, info = env.reset(seed=10000 + rnd * 1000 + ep)
                done, truncated = False, False
                step = 0

                while not (done or truncated):
                    # Get current policy's action (to determine visited states)
                    # But label with oracle's action (DAgger key insight)
                    oracle_actions = oracle_fn(obs, info, {"step": step})

                    for aid in range(env.num_agents):
                        flat = flatten_obs(obs[aid])
                        act = int(oracle_actions[aid]["action"])
                        msg_tokens = oracle_actions[aid].get("message_tokens", [])
                        new_obs.append(flat)
                        new_acts.append(act)
                        new_msgs.append(msg_tokens)

                    # Step with model's actions (not oracle) to visit model's distribution
                    import torch as _torch
                    model_actions = {}
                    for aid in range(env.num_agents):
                        flat_t = _torch.tensor(flatten_obs(obs[aid]), dtype=_torch.float32, device=device).unsqueeze(0)
                        with _torch.no_grad():
                            out = model(flat_t)
                        if cfg.comm:
                            logits = out[0]
                        else:
                            logits = out
                        act = int(_torch.argmax(logits, dim=-1).item())
                        model_actions[aid] = {"action": act, "message_tokens": []}

                    obs, rewards, done, truncated, info = env.step(model_actions)
                    step += 1

            model.train()

            # Pad and store new data
            padded = np.zeros((len(new_msgs), cfg.comm_token_limit), dtype=np.int64)
            mlens = np.zeros(len(new_msgs), dtype=np.int64)
            for i, tokens in enumerate(new_msgs):
                L = min(len(tokens), cfg.comm_token_limit)
                if L > 0:
                    padded[i, :L] = tokens[:L]
                mlens[i] = L

            obs_list.append(np.stack(new_obs))
            act_list.append(np.array(new_acts, dtype=np.int64))
            msg_list.append(padded)
            mlen_list.append(mlens)
            print(f"  Added {len(new_obs)} transitions (total: {N + len(new_obs)})")

    _save_bc_model(model, obs_dim, cfg)
    if wandb_run is not None:
        wandb_run.finish()
    return model


# ---------------------------------------------------------------------------
# Step 4: Cooperative Reward Regression (Simple IRL)
# ---------------------------------------------------------------------------

class RewardNet(nn.Module):
    """Small network that predicts reward from (obs, action)."""
    def __init__(self, obs_dim: int, action_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, 16)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a_emb = self.action_embed(action)
        x = torch.cat([obs, a_emb], dim=-1)
        return self.net(x).squeeze(-1)


def train_reward_model(cfg: BCConfig):
    """Train a reward model R(obs, action) → reward on oracle demonstrations.

    Collects (obs, action, reward) from oracle rollouts and trains a small
    network to predict the per-step reward. The learned reward can then
    replace hand-crafted shaping in MAPPO training.
    """
    from syncorsink.policies.oracle import (
        pipeline_oracle_strong, energy_oracle_strong, signal_hunt_oracle_strong,
    )

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    env_kwargs = dict(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        num_agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        max_steps=cfg.max_steps,
        energy_preset=cfg.energy_preset,
    )
    if cfg.comm:
        env_kwargs["comm_token_limit"] = cfg.comm_token_limit
        env_kwargs["token_vocab_size"] = cfg.comm_vocab_size
        env_kwargs["max_messages"] = 8
    env_cfg = SyncOrSinkConfig(**env_kwargs)
    env = SyncOrSinkEnv(env_cfg)

    oracle_map = {
        "signal_hunt": signal_hunt_oracle_strong,
        "energy_grid": energy_oracle_strong,
        "pipeline_assembly": pipeline_oracle_strong,
    }
    oracle_fn = oracle_map[cfg.scenario](env)

    # Collect (obs, action, reward) from oracle rollouts
    all_obs, all_acts, all_rewards = [], [], []
    for ep in range(cfg.demo_episodes):
        obs, info = env.reset(seed=ep)
        done, truncated = False, False
        step = 0
        while not (done or truncated):
            actions = oracle_fn(obs, info, {"step": step})
            next_obs, rewards, done, truncated, info = env.step(actions)
            for aid in range(env.num_agents):
                all_obs.append(flatten_obs(obs[aid]))
                all_acts.append(int(actions[aid]["action"]))
                all_rewards.append(float(rewards[aid]))
            obs = next_obs
            step += 1

    obs_t = torch.tensor(np.stack(all_obs), dtype=torch.float32)
    act_t = torch.tensor(all_acts, dtype=torch.long)
    rew_t = torch.tensor(all_rewards, dtype=torch.float32)
    N = obs_t.shape[0]
    obs_dim = obs_t.shape[1]
    print(f"Collected {N} (obs, action, reward) tuples for reward regression")
    print(f"  reward stats: mean={rew_t.mean():.4f} std={rew_t.std():.4f} "
          f"min={rew_t.min():.4f} max={rew_t.max():.4f}")

    # Train reward network
    reward_net = RewardNet(obs_dim, hidden_dim=cfg.hidden_dim).to(device)
    optimizer = optim.Adam(reward_net.parameters(), lr=cfg.lr)

    idx = np.arange(N)
    for epoch in range(cfg.epochs):
        np.random.shuffle(idx)
        total_loss = 0.0
        batches = 0
        for start in range(0, N, cfg.batch_size):
            mb = idx[start:start + cfg.batch_size]
            pred = reward_net(obs_t[mb].to(device), act_t[mb].to(device))
            loss = nn.functional.mse_loss(pred, rew_t[mb].to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(reward_net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:3d} | reward_mse {total_loss / batches:.6f}")

    # Save
    save_path = cfg.save or "checkpoints/reward_model.pt"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "model": reward_net.state_dict(),
        "obs_dim": obs_dim,
        "hidden_dim": cfg.hidden_dim,
    }, save_path)
    print(f"Saved reward model to {save_path}")
    return reward_net


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Behavioral Cloning from oracle demonstrations")
    sub = parser.add_subparsers(dest="command")

    # --- collect ---
    collect_p = sub.add_parser("collect", help="Collect oracle demonstrations")
    collect_p.add_argument("--scenario", default="signal_hunt")
    collect_p.add_argument("--map-size", type=int, default=8)
    collect_p.add_argument("--agents", type=int, default=2)
    collect_p.add_argument("--fov-preset", default="easy")
    collect_p.add_argument("--max-steps", type=int, default=300)
    collect_p.add_argument("--energy-preset", default="easy")
    collect_p.add_argument("--episodes", type=int, default=100)
    collect_p.add_argument("--oracle", default="oracle_strong",
                           choices=["oracle", "oracle_strong", "oracle_comm", "oracle_strong_comm"])
    collect_p.add_argument("--comm-token-limit", type=int, default=8)
    collect_p.add_argument("--comm-vocab-size", type=int, default=32)
    collect_p.add_argument("--output", default="demos/signal_hunt_oracle.npz")

    # --- collect-llm ---
    collect_llm_p = sub.add_parser("collect-llm", help="Extract demonstrations from LLM trace JSONL")
    collect_llm_p.add_argument("--trace", required=True, help="Path to trace JSONL file")
    collect_llm_p.add_argument("--scenario", default="signal_hunt")
    collect_llm_p.add_argument("--map-size", type=int, default=8)
    collect_llm_p.add_argument("--agents", type=int, default=2)
    collect_llm_p.add_argument("--fov-preset", default="easy")
    collect_llm_p.add_argument("--max-steps", type=int, default=300)
    collect_llm_p.add_argument("--energy-preset", default="easy")
    collect_llm_p.add_argument("--comm-token-limit", type=int, default=8)
    collect_llm_p.add_argument("--comm-vocab-size", type=int, default=32)
    collect_llm_p.add_argument("--output", default="demos/signal_hunt_llm.npz")

    # --- train ---
    train_p = sub.add_parser("train", help="Train BC policy on demonstrations")
    train_p.add_argument("--demo-path", required=True, help="Path to .npz demo file")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch-size", type=int, default=256)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--hidden-dim", type=int, default=128)
    train_p.add_argument("--comm", action="store_true")
    train_p.add_argument("--comm-token-limit", type=int, default=8)
    train_p.add_argument("--comm-vocab-size", type=int, default=32)
    train_p.add_argument("--comm-loss-weight", type=float, default=0.1)
    train_p.add_argument("--two-phase", action="store_true", help="Two-phase training: action then comm")
    train_p.add_argument("--phase1-epochs", type=int, default=30)
    train_p.add_argument("--phase2-epochs", type=int, default=20)
    train_p.add_argument("--phase2-lr", type=float, default=3e-4)
    train_p.add_argument("--device", default="auto")
    train_p.add_argument("--save", default=None, help="Path to save trained model")
    train_p.add_argument("--wandb", action="store_true")
    train_p.add_argument("--wandb-project", default="syncorsink")
    train_p.add_argument("--wandb-run", default=None)

    # --- dagger ---
    dagger_p = sub.add_parser("dagger", help="Train BC with DAgger (iterative on-policy collection)")
    dagger_p.add_argument("--demo-path", required=True, help="Path to initial .npz demo file")
    dagger_p.add_argument("--scenario", default="signal_hunt")
    dagger_p.add_argument("--map-size", type=int, default=8)
    dagger_p.add_argument("--agents", type=int, default=2)
    dagger_p.add_argument("--fov-preset", default="easy")
    dagger_p.add_argument("--max-steps", type=int, default=300)
    dagger_p.add_argument("--energy-preset", default="easy")
    dagger_p.add_argument("--oracle", default="oracle_strong",
                          choices=["oracle", "oracle_strong", "oracle_comm", "oracle_strong_comm"])
    dagger_p.add_argument("--rounds", type=int, default=3, help="Number of DAgger rounds")
    dagger_p.add_argument("--dagger-episodes", type=int, default=20, help="On-policy episodes per round")
    dagger_p.add_argument("--epochs", type=int, default=30)
    dagger_p.add_argument("--batch-size", type=int, default=256)
    dagger_p.add_argument("--lr", type=float, default=1e-3)
    dagger_p.add_argument("--hidden-dim", type=int, default=128)
    dagger_p.add_argument("--comm", action="store_true")
    dagger_p.add_argument("--comm-token-limit", type=int, default=8)
    dagger_p.add_argument("--comm-vocab-size", type=int, default=32)
    dagger_p.add_argument("--comm-loss-weight", type=float, default=0.1)
    dagger_p.add_argument("--device", default="auto")
    dagger_p.add_argument("--save", default=None)
    dagger_p.add_argument("--wandb", action="store_true")
    dagger_p.add_argument("--wandb-project", default="syncorsink")
    dagger_p.add_argument("--wandb-run", default=None)

    # --- reward-model ---
    reward_p = sub.add_parser("reward-model", help="Train reward regression model (simple IRL)")
    reward_p.add_argument("--scenario", default="signal_hunt")
    reward_p.add_argument("--map-size", type=int, default=8)
    reward_p.add_argument("--agents", type=int, default=2)
    reward_p.add_argument("--fov-preset", default="easy")
    reward_p.add_argument("--max-steps", type=int, default=300)
    reward_p.add_argument("--energy-preset", default="easy")
    reward_p.add_argument("--episodes", type=int, default=100)
    reward_p.add_argument("--epochs", type=int, default=50)
    reward_p.add_argument("--batch-size", type=int, default=256)
    reward_p.add_argument("--lr", type=float, default=1e-3)
    reward_p.add_argument("--hidden-dim", type=int, default=128)
    reward_p.add_argument("--comm", action="store_true")
    reward_p.add_argument("--comm-token-limit", type=int, default=8)
    reward_p.add_argument("--comm-vocab-size", type=int, default=32)
    reward_p.add_argument("--device", default="auto")
    reward_p.add_argument("--save", default=None)

    args = parser.parse_args()

    if args.command == "collect":
        cfg = BCConfig(
            scenario=args.scenario,
            map_size=args.map_size,
            agents=args.agents,
            fov_preset=args.fov_preset,
            max_steps=args.max_steps,
            energy_preset=args.energy_preset,
            demo_episodes=args.episodes,
            oracle_type=args.oracle,
            comm_token_limit=args.comm_token_limit,
            comm_vocab_size=args.comm_vocab_size,
            demo_path=args.output,
        )
        collect_demos(cfg)

    elif args.command == "collect-llm":
        cfg = BCConfig(
            scenario=args.scenario,
            map_size=args.map_size,
            agents=args.agents,
            fov_preset=args.fov_preset,
            max_steps=args.max_steps,
            energy_preset=args.energy_preset,
            comm_token_limit=args.comm_token_limit,
            comm_vocab_size=args.comm_vocab_size,
            demo_path=args.output,
        )
        collect_llm_demos(cfg, args.trace)

    elif args.command == "train":
        cfg = BCConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            comm=args.comm,
            comm_token_limit=args.comm_token_limit,
            comm_vocab_size=args.comm_vocab_size,
            comm_loss_weight=args.comm_loss_weight,
            two_phase=args.two_phase,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            phase2_lr=args.phase2_lr,
            device=args.device,
            demo_path=args.demo_path,
            save=args.save,
            wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run=args.wandb_run,
        )
        train_bc(cfg)

    elif args.command == "dagger":
        cfg = BCConfig(
            scenario=args.scenario,
            map_size=args.map_size,
            agents=args.agents,
            fov_preset=args.fov_preset,
            max_steps=args.max_steps,
            energy_preset=args.energy_preset,
            oracle_type=args.oracle,
            demo_path=args.demo_path,
            dagger_rounds=args.rounds,
            dagger_episodes=args.dagger_episodes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            comm=args.comm,
            comm_token_limit=args.comm_token_limit,
            comm_vocab_size=args.comm_vocab_size,
            comm_loss_weight=args.comm_loss_weight,
            device=args.device,
            save=args.save,
            wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run=args.wandb_run,
        )
        train_bc_dagger(cfg)

    elif args.command == "reward-model":
        cfg = BCConfig(
            scenario=args.scenario,
            map_size=args.map_size,
            agents=args.agents,
            fov_preset=args.fov_preset,
            max_steps=args.max_steps,
            energy_preset=args.energy_preset,
            demo_episodes=args.episodes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            comm=args.comm,
            comm_token_limit=args.comm_token_limit,
            comm_vocab_size=args.comm_vocab_size,
            device=args.device,
            save=args.save,
        )
        train_reward_model(cfg)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
