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
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.eval.success import episode_success
from syncorsink.train.mappo import flatten_obs, build_batch_obs, resolve_device
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
    # shaping
    pipeline_shaping: bool = True
    pipeline_shaping_scale: float = 0.1
    # architecture
    hidden_dim: int = 128
    comm: bool = False
    comm_token_limit: int = 8
    comm_vocab_size: int = 32
    # BC
    demo_episodes: int = 200
    bc_epochs: int = 30
    bc_lr: float = 1e-3
    bc_seq_len: int = 32  # truncated BPTT sequence length
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
    # device
    device: str = "auto"
    # output
    save: Optional[str] = None
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
        comm_token_limit=cfg.comm_token_limit,
        token_vocab_size=cfg.comm_vocab_size,
    ))


def collect_episode_demos(cfg: RecurrentConfig):
    """Collect oracle demonstrations as full episodes (not shuffled transitions)."""
    from syncorsink.policies.oracle import pipeline_oracle_strong

    env = _build_env(cfg)
    oracle_fn = pipeline_oracle_strong(env)

    episodes = []
    for ep in range(cfg.demo_episodes):
        obs, info = env.reset(seed=ep)
        ep_data = {"obs": [], "actions": []}
        done, truncated = False, False
        step = 0
        while not (done or truncated):
            actions = oracle_fn(obs, info, {"step": step})
            for aid in range(env.num_agents):
                ep_data["obs"].append(flatten_obs(obs[aid]))
                ep_data["actions"].append(int(actions[aid]["action"]))
            obs, rewards, done, truncated, info = env.step(actions)
            step += 1

        success = bool(done) if cfg.scenario != "energy_grid" else bool(info.get("success", False))
        if success:
            ep_data["obs"] = np.stack(ep_data["obs"]).reshape(-1, env.num_agents,
                                                                flatten_obs(obs[0]).shape[0])
            ep_data["actions"] = np.array(ep_data["actions"]).reshape(-1, env.num_agents)
            episodes.append(ep_data)

        if (ep + 1) % 50 == 0:
            print(f"  collected {ep + 1}/{cfg.demo_episodes}, {len(episodes)} successful")

    print(f"Collected {len(episodes)} successful episodes")
    return episodes


def train_recurrent_bc(cfg: RecurrentConfig, episodes, device):
    """Train recurrent BC via truncated BPTT on episode sequences."""
    obs_dim = episodes[0]["obs"].shape[-1]
    model = MAPPORecurrentActor(
        obs_dim=obs_dim, action_dim=8, hidden_dim=cfg.hidden_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.bc_lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.bc_epochs):
        np.random.shuffle(episodes)
        total_loss, total_correct, total_count = 0.0, 0, 0

        for ep_data in episodes:
            obs_seq = torch.tensor(ep_data["obs"], dtype=torch.float32, device=device)
            act_seq = torch.tensor(ep_data["actions"], dtype=torch.long, device=device)
            T = obs_seq.shape[0]
            N = obs_seq.shape[1]

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
                    logits, hidden = model(obs_seq[t], hidden)
                    loss = loss_fn(logits, act_seq[t])
                    chunk_loss += loss
                    chunk_correct += (logits.argmax(dim=-1) == act_seq[t]).sum().item()
                    chunk_count += N

                chunk_loss = chunk_loss / (t_end - t_start)
                optimizer.zero_grad()
                chunk_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += chunk_loss.item()
                total_correct += chunk_correct
                total_count += chunk_count

        acc = total_correct / total_count if total_count > 0 else 0
        print(f"[BC] epoch {epoch:3d} | loss {total_loss / len(episodes):.4f} | acc {acc:.3f}")

    return model


def train_recurrent_rl(cfg: RecurrentConfig, model, device):
    """Fine-tune recurrent policy with PPO, carrying hidden state across steps."""
    import copy

    env = _build_env(cfg)
    N = env.num_agents
    obs_dim = flatten_obs(env.reset(seed=0)[0][0]).shape[0]

    critic = MAPPOCritic(obs_dim, hidden_dim=cfg.hidden_dim).to(device)

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

    for update in range(cfg.rl_updates):
        # LR annealing
        frac = 1.0 - update / cfg.rl_updates
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.rl_lr * frac

        # Rollout with hidden state
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf = [], []
        hidden_buf = []
        ep_returns, ep_steps = [], []
        ep_ret, ep_step = 0.0, 0

        obs, _ = env.reset(seed=update)
        hidden = model.init_hidden(N, device)

        for t in range(cfg.rollout_steps):
            obs_batch = build_batch_obs(obs, N)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits, new_hidden = model(obs_tensor, hidden)
                v = critic(obs_tensor)

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

            hidden = new_hidden
            obs = next_obs
            ep_ret += sum(rewards.values())
            ep_step += 1

            if done or truncated:
                ep_returns.append(ep_ret)
                ep_steps.append(ep_step)
                obs, _ = env.reset(seed=update * cfg.rollout_steps + t + 1)
                hidden = model.init_hidden(N, device)
                ep_ret, ep_step = 0.0, 0

        # GAE
        values = torch.stack(val_buf)
        rewards_t = torch.stack(rew_buf)
        dones_t = torch.stack(done_buf)

        with torch.no_grad():
            last_obs = torch.tensor(build_batch_obs(obs, N), dtype=torch.float32, device=device)
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

        for epoch in range(cfg.rl_epochs):
            # Replay the sequence with current model
            hidden_replay = (hidden_buf[0][0].to(device), hidden_buf[0][1].to(device))
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_kl = 0.0
            total_entropy = 0.0

            for t in range(T):
                obs_t = obs_buf[t].to(device)
                act_t = act_buf[t].to(device)
                idx = t * N

                # Reset hidden at episode boundaries
                if t > 0 and dones_t[t - 1].any():
                    hidden_replay = model.init_hidden(N, device)

                logits, hidden_replay = model(obs_t, hidden_replay)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t)
                entropy = dist.entropy().mean()

                # KL toward BC reference
                with torch.no_grad():
                    bc_h = (hidden_buf[t][0].to(device), hidden_buf[t][1].to(device))
                    bc_logits, _ = bc_ref(obs_t, bc_h)
                bc_probs = torch.softmax(bc_logits, dim=-1)
                current_logprobs = torch.log_softmax(logits, dim=-1)
                kl = (bc_probs * (bc_probs.log() - current_logprobs)).sum(dim=-1).mean()

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

        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        mean_len = float(np.mean(ep_steps)) if ep_steps else 0.0
        print(
            f"update {update:4d} | pi {total_policy_loss / T:.3f} | "
            f"v {total_value_loss / T:.3f} | kl {total_kl / T:.4f} | "
            f"ent {total_entropy / T:.3f} | ret {mean_ret:.2f} | len {mean_len:.1f}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "policy_loss": total_policy_loss / T,
                "value_loss": total_value_loss / T,
                "kl": total_kl / T,
                "entropy": total_entropy / T,
                "rollout/mean_ep_return": mean_ret,
                "rollout/mean_ep_len": mean_len,
                "update": update,
            })

        # Eval
        if (update + 1) % 50 == 0:
            model.eval()
            eval_success = []
            for ep in range(10):
                eval_obs, _ = env.reset(seed=10000 + update + ep)
                h_eval = model.init_hidden(N, device)
                ep_done, ep_trunc = False, False
                info = {}
                while not (ep_done or ep_trunc):
                    obs_t = torch.tensor(build_batch_obs(eval_obs, N), dtype=torch.float32, device=device)
                    with torch.no_grad():
                        logits, h_eval = model(obs_t, h_eval)
                    acts = torch.argmax(logits, dim=-1)
                    actions = {aid: {"action": int(acts[aid].item()), "message_tokens": []} for aid in range(N)}
                    eval_obs, _, ep_done, ep_trunc, info = env.step(actions)
                success = episode_success(cfg.scenario, ep_done, info)
                eval_success.append(1.0 if success else 0.0)
            sr = np.mean(eval_success)
            print(f"  eval | success {sr:.2f}")
            model.train()
            if wandb_run is not None:
                wandb_run.log({"eval/success_rate": sr, "eval/update": update})

    if cfg.save:
        os.makedirs(os.path.dirname(cfg.save) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "config": vars(cfg)}, cfg.save)
        print(f"Saved to {cfg.save}")
    if wandb_run is not None:
        wandb_run.finish()


def main():
    p = argparse.ArgumentParser(description="Recurrent BC→RL for pipeline assembly")
    p.add_argument("--scenario", default="pipeline_assembly")
    p.add_argument("--map-size", type=int, default=8)
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--fov-preset", default="easy")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--demo-episodes", type=int, default=200)
    p.add_argument("--bc-epochs", type=int, default=30)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    p.add_argument("--rl-updates", type=int, default=3000)
    p.add_argument("--rl-lr", type=float, default=3e-5)
    p.add_argument("--bc-kl-coeff", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    p.add_argument("--save", default=None)
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
        hidden_dim=args.hidden_dim,
        demo_episodes=args.demo_episodes,
        bc_epochs=args.bc_epochs,
        bc_lr=args.bc_lr,
        rl_updates=args.rl_updates,
        rl_lr=args.rl_lr,
        bc_kl_coeff=args.bc_kl_coeff,
        device=args.device,
        save=args.save,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
    )

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    print("=== Step 1: Collecting oracle demos ===")
    episodes = collect_episode_demos(cfg)

    print("\n=== Step 2: Training recurrent BC ===")
    model = train_recurrent_bc(cfg, episodes, device)

    print("\n=== Step 3: RL fine-tuning ===")
    train_recurrent_rl(cfg, model, device)


if __name__ == "__main__":
    main()
