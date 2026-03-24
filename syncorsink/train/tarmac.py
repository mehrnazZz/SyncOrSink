"""TarMAC training loop using PPO.

TarMAC agents communicate via attention-weighted messages. Unlike MAPPO/Comm-MAT
where communication goes through the environment's message channel, TarMAC
communication is internal to the model — all agents share a forward pass
and messages are passed as continuous vectors via attention.

The env still receives discrete actions but no explicit message tokens.
Communication is learned end-to-end through the policy gradient.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.train.mappo import flatten_obs, build_batch_obs, resolve_device
from syncorsink.models.tarmac import TarMACConfig, TarMACModel


@dataclass
class TarMACTrainConfig:
    # environment
    scenario: str = "signal_hunt"
    map_size: int = 8
    agents: int = 2
    fov_preset: str = "easy"
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
    # PPO
    updates: int = 3000
    rollout_steps: int = 512
    epochs: int = 4
    minibatch: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    entropy_coeff: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    # TarMAC architecture
    hidden_dim: int = 128
    msg_dim: int = 32
    key_dim: int = 32
    n_rounds: int = 1
    # attention entropy bonus (encourages diverse communication patterns)
    attn_entropy_coeff: float = 0.001
    # device
    device: str = "auto"
    # logging
    wandb: bool = False
    wandb_project: str = "syncorsink"
    wandb_run: Optional[str] = None
    # checkpointing
    save: Optional[str] = None
    load: Optional[str] = None
    save_every: int = 200
    eval_every: int = 50
    eval_episodes: int = 10


def train_tarmac(cfg: TarMACTrainConfig):
    env_config = SyncOrSinkConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        num_agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        max_steps=cfg.max_steps,
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
    )
    env = SyncOrSinkEnv(env_config)
    N = env.num_agents

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    sample_obs, _ = env.reset(seed=0)
    obs_dim = flatten_obs(sample_obs[0]).shape[0]

    model_cfg = TarMACConfig(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        msg_dim=cfg.msg_dim,
        key_dim=cfg.key_dim,
        n_rounds=cfg.n_rounds,
        value_head=True,
    )
    model = TarMACModel(model_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    start_update = 0
    if cfg.load:
        ckpt = torch.load(cfg.load, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt.get("step", 0)

    wandb_run = None
    if cfg.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run, config=vars(cfg))
        except Exception as exc:
            print(f"wandb init failed: {exc}")

    global_step = 0
    for update in range(start_update, cfg.updates):
        if cfg.anneal_lr:
            frac = 1.0 - update / cfg.updates
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * frac

        # Rollout buffers
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf = [], []
        attn_entropy_buf = []

        ep_returns, ep_steps = [], []
        ep_ret, ep_step = 0.0, 0
        action_hist = np.zeros(8, dtype=np.int64)

        obs, _ = env.reset(seed=update)
        for t in range(cfg.rollout_steps):
            obs_batch = build_batch_obs(obs, N)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)

            with torch.no_grad():
                out = model(obs_tensor)

            logits = out["action_logits"]
            dist = torch.distributions.Categorical(logits=logits)
            acts = dist.sample()
            logp = dist.log_prob(acts)

            # Attention entropy (how spread out is communication)
            attn = out["attention"]  # (N, N)
            attn_ent = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean()

            actions = {
                aid: {"action": int(acts[aid].item()), "message_tokens": []}
                for aid in range(N)
            }

            next_obs, rewards, done, truncated, info = env.step(actions)

            for a in acts.tolist():
                action_hist[a] += 1

            obs_buf.append(obs_tensor.cpu())
            act_buf.append(acts.cpu())
            logp_buf.append(logp.cpu())
            val_buf.append(out["value"].cpu())
            rew_buf.append(torch.tensor([rewards[i] for i in range(N)], dtype=torch.float32))
            done_buf.append(torch.tensor([float(done or truncated)] * N, dtype=torch.float32))
            attn_entropy_buf.append(attn_ent.cpu().item())

            obs = next_obs
            ep_ret += sum(rewards.values())
            ep_step += 1
            global_step += 1

            if done or truncated:
                ep_returns.append(ep_ret)
                ep_steps.append(ep_step)
                obs, _ = env.reset(seed=update * cfg.rollout_steps + t + 1)
                ep_ret, ep_step = 0.0, 0

        # GAE
        values = torch.stack(val_buf)
        rewards_t = torch.stack(rew_buf)
        dones_t = torch.stack(done_buf)

        with torch.no_grad():
            last_obs = torch.tensor(build_batch_obs(obs, N), dtype=torch.float32, device=device)
            last_v = model(last_obs)["value"].cpu()

        advantages = torch.zeros_like(rewards_t)
        gae = torch.zeros(N)
        for t in reversed(range(cfg.rollout_steps)):
            next_v = last_v if t == cfg.rollout_steps - 1 else values[t + 1]
            delta = rewards_t[t] + cfg.gamma * next_v * (1.0 - dones_t[t]) - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - dones_t[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        # Flatten
        T = cfg.rollout_steps
        obs_b = torch.stack(obs_buf).reshape(T * N, -1).to(device)
        act_b = torch.stack(act_buf).reshape(T * N).to(device)
        logp_old = torch.stack(logp_buf).reshape(T * N).to(device)
        adv_b = advantages.reshape(T * N).to(device)
        ret_b = returns.reshape(T * N).to(device)
        val_old = values.reshape(T * N).to(device)

        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # PPO epochs
        # TarMAC needs all N agents together for communication, so minibatches
        # are over timesteps, not individual agent transitions
        total_timesteps = T
        ts_idx = np.arange(total_timesteps)
        mb_timesteps = max(1, cfg.minibatch // N)

        for epoch in range(cfg.epochs):
            np.random.shuffle(ts_idx)
            for start in range(0, total_timesteps, mb_timesteps):
                ts_mb = ts_idx[start:start + mb_timesteps]

                # For each timestep in minibatch, run full N-agent forward
                all_logits = []
                all_values = []
                all_attn_ent = []
                for t_idx in ts_mb:
                    t_obs = obs_b[t_idx * N:(t_idx + 1) * N]
                    out = model(t_obs)
                    all_logits.append(out["action_logits"])
                    all_values.append(out["value"])
                    attn = out["attention"]
                    all_attn_ent.append(-(attn * (attn + 1e-8).log()).sum(dim=-1).mean())

                logits = torch.cat(all_logits, dim=0)
                new_values = torch.cat(all_values, dim=0)
                mean_attn_ent = torch.stack(all_attn_ent).mean()

                # Flat indices for this minibatch
                flat_idx = []
                for t_idx in ts_mb:
                    flat_idx.extend(range(t_idx * N, (t_idx + 1) * N))
                flat_idx = torch.tensor(flat_idx, dtype=torch.long, device=device)

                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_b[flat_idx])
                entropy = dist.entropy().mean()

                ratio = (new_logp - logp_old[flat_idx]).exp()
                surr1 = ratio * adv_b[flat_idx]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv_b[flat_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = val_old[flat_idx] + torch.clamp(
                    new_values - val_old[flat_idx], -cfg.value_clip, cfg.value_clip
                )
                value_loss = 0.5 * torch.max(
                    (ret_b[flat_idx] - new_values).pow(2),
                    (ret_b[flat_idx] - v_clipped).pow(2),
                ).mean()

                loss = (policy_loss + value_loss
                        - cfg.entropy_coeff * entropy
                        - cfg.attn_entropy_coeff * mean_attn_ent)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        # Logging
        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        mean_len = float(np.mean(ep_steps)) if ep_steps else 0.0
        mean_attn_ent = float(np.mean(attn_entropy_buf))
        print(
            f"update {update:4d} | loss {loss.item():.3f} | "
            f"pi {policy_loss.item():.3f} | v {value_loss.item():.3f} | "
            f"ent {entropy.item():.3f} | attn_ent {mean_attn_ent:.3f} | "
            f"ret {mean_ret:.2f} | len {mean_len:.1f}"
        )

        if wandb_run is not None:
            log_payload = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "attn_entropy": mean_attn_ent,
                "update": update,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "rollout/episodes": len(ep_returns),
                "rollout/mean_ep_return": mean_ret,
                "rollout/mean_ep_len": mean_len,
            }
            for i in range(8):
                log_payload[f"rollout/action_hist_{i}"] = int(action_hist[i])
            wandb_run.log(log_payload)

        # Eval
        if cfg.eval_every > 0 and (update + 1) % cfg.eval_every == 0:
            model.eval()
            eval_returns, eval_steps_list, eval_success = [], [], []
            eval_obs, _ = env.reset(seed=10000 + update)
            for ep in range(cfg.eval_episodes):
                ep_done, ep_trunc = False, False
                total_reward, steps = 0.0, 0
                while not (ep_done or ep_trunc):
                    obs_batch = build_batch_obs(eval_obs, N)
                    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        out = model(obs_tensor)
                    acts = torch.argmax(out["action_logits"], dim=-1)
                    actions = {
                        aid: {"action": int(acts[aid].item()), "message_tokens": []}
                        for aid in range(N)
                    }
                    eval_obs, rewards, ep_done, ep_trunc, info = env.step(actions)
                    total_reward += sum(rewards.values())
                    steps += 1
                eval_returns.append(total_reward)
                eval_steps_list.append(steps)
                eval_success.append(1.0 if ep_done else 0.0)
                eval_obs, _ = env.reset(seed=10000 + update + ep + 1)

            model.train()
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

        if cfg.save and (update + 1) % cfg.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": update + 1,
                "config": vars(cfg),
            }, cfg.save)

    if cfg.save:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": cfg.updates,
            "config": vars(cfg),
        }, cfg.save)
    if wandb_run is not None:
        wandb_run.finish()


def main():
    p = argparse.ArgumentParser(description="TarMAC training for SyncOrSink")
    p.add_argument("--scenario", default="signal_hunt")
    p.add_argument("--map-size", type=int, default=8)
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--fov-preset", default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--energy-preset", default="hard")
    p.add_argument("--energy-private-monitor", action="store_true")
    # shaping
    p.add_argument("--pipeline-shaping", action="store_true")
    p.add_argument("--pipeline-shaping-scale", type=float, default=0.01)
    p.add_argument("--energy-shaping", action="store_true")
    p.add_argument("--energy-shaping-scale", type=float, default=0.01)
    p.add_argument("--signal-shaping", action="store_true")
    p.add_argument("--signal-shaping-scale", type=float, default=0.01)
    p.add_argument("--signal-scan-bonus", type=float, default=0.0)
    p.add_argument("--signal-joint-scan-bonus", type=float, default=0.0)
    p.add_argument("--signal-colocation-bonus", type=float, default=0.0)
    p.add_argument("--signal-colocation-radius", type=int, default=2)
    p.add_argument("--signal-comm-utility", type=float, default=0.0)
    # PPO
    p.add_argument("--updates", type=int, default=3000)
    p.add_argument("--rollout-steps", type=int, default=512)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value-clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--anneal-lr", action="store_true")
    # TarMAC
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--msg-dim", type=int, default=32)
    p.add_argument("--key-dim", type=int, default=32)
    p.add_argument("--n-rounds", type=int, default=1)
    p.add_argument("--attn-entropy-coeff", type=float, default=0.001)
    # device
    p.add_argument("--device", default="auto")
    # logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="syncorsink")
    p.add_argument("--wandb-run", default=None)
    # checkpointing
    p.add_argument("--save", default=None)
    p.add_argument("--load", default=None)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=10)
    args = p.parse_args()

    cfg = TarMACTrainConfig(
        scenario=args.scenario,
        map_size=args.map_size,
        agents=args.agents,
        fov_preset=args.fov_preset,
        max_steps=args.max_steps,
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
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch=args.minibatch,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        value_clip=args.value_clip,
        entropy_coeff=args.entropy,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        anneal_lr=args.anneal_lr,
        hidden_dim=args.hidden_dim,
        msg_dim=args.msg_dim,
        key_dim=args.key_dim,
        n_rounds=args.n_rounds,
        attn_entropy_coeff=args.attn_entropy_coeff,
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
    train_tarmac(cfg)


if __name__ == "__main__":
    main()
