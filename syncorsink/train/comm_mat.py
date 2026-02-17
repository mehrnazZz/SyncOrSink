from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
from syncorsink.models.comm_mat import CommMATConfig, CommMATModel


@dataclass
class CommMATTrainConfig:
    scenario: str = "pipeline_assembly"
    map_size: int = 8
    agents: int = 3
    fov_preset: str = "easy"
    max_steps: int = 300
    comm_token_limit: int = 24
    comm_vocab_size: int = 256
    comm_max_messages: int = 8
    comm_len_cost: float = 0.0
    comm_cost: float = 0.01
    pipeline_shaping: bool = False
    pipeline_shaping_scale: float = 0.01
    energy_shaping: bool = False
    energy_shaping_scale: float = 0.01
    signal_shaping: bool = False
    signal_shaping_scale: float = 0.01
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
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    goal_hint_dim: int = 32
    send_threshold: float = 0.5
    deterministic_eval: bool = True
    wandb: bool = False
    wandb_project: str = "syncorsink"
    wandb_run: Optional[str] = None
    save: Optional[str] = None
    load: Optional[str] = None
    save_every: int = 5
    eval_every: int = 10
    eval_episodes: int = 5


def _to_grid_ids(local_grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(local_grid)
    if arr.ndim == 3:
        return arr.argmax(axis=0).astype(np.int64)
    return arr.astype(np.int64)


def _recv_from(obs_agent: dict, max_messages: int) -> np.ndarray:
    recv_from = obs_agent.get("messages_from")
    if recv_from is None:
        recv_from = obs_agent.get("message_from")
    if recv_from is None:
        return np.full((max_messages,), -1, dtype=np.int64)
    recv_from = np.asarray(recv_from, dtype=np.int64).reshape(-1)
    if recv_from.shape[0] < max_messages:
        pad = np.full((max_messages - recv_from.shape[0],), -1, dtype=np.int64)
        recv_from = np.concatenate([recv_from, pad], axis=0)
    return recv_from[:max_messages]


def _recv_tokens(obs_agent: dict, max_messages: int, token_limit: int) -> np.ndarray:
    toks = np.asarray(
        obs_agent.get("messages_tokens", np.zeros((max_messages, token_limit), dtype=np.int64)),
        dtype=np.int64,
    )
    if toks.ndim == 1:
        toks = toks.reshape(1, -1)
    out = np.zeros((max_messages, token_limit), dtype=np.int64)
    h = min(max_messages, toks.shape[0])
    w = min(token_limit, toks.shape[1])
    out[:h, :w] = toks[:h, :w]
    return out


def _build_agent_batch(obs: dict, cfg: CommMATTrainConfig):
    agent_ids = sorted(obs.keys())
    n = len(agent_ids)
    grid = []
    inv = []
    pos = []
    hint = []
    rtok = []
    rfrom = []
    amask = []
    hint_dim = cfg.goal_hint_dim
    for aid in agent_ids:
        oa = obs[aid]
        grid.append(_to_grid_ids(oa["local_grid"]))
        inv.append(np.asarray(oa.get("inventory", np.array([0], dtype=np.float32)), dtype=np.float32).reshape(1))
        pos.append(np.asarray(oa.get("self_pos", np.array([0, 0], dtype=np.float32)), dtype=np.float32).reshape(2))
        h = np.asarray(oa.get("goal_hint", np.zeros((hint_dim,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        hint.append(h)
        rtok.append(_recv_tokens(oa, cfg.comm_max_messages, cfg.comm_token_limit))
        rfrom.append(_recv_from(oa, cfg.comm_max_messages))
        amask.append(np.asarray(oa.get("action_mask", np.ones((8,), dtype=np.float32)), dtype=np.float32))
    hint_w = max(max(x.shape[0] for x in hint), hint_dim)
    hint_arr = np.zeros((n, hint_w), dtype=np.float32)
    for i, h in enumerate(hint):
        hint_arr[i, : h.shape[0]] = h
    return (
        np.stack(grid),
        np.stack(inv),
        np.stack(pos),
        hint_arr,
        np.stack(rtok),
        np.stack(rfrom),
        np.stack(amask),
    )


def _build_model(env: SyncOrSinkEnv, cfg: CommMATTrainConfig) -> CommMATModel:
    sample_obs, _ = env.reset(seed=0)
    grid, _, _, hint, _, _, _ = _build_agent_batch(sample_obs, cfg)
    model_cfg = CommMATConfig(
        action_dim=8,
        tile_vocab_size=16,
        comm_vocab_size=cfg.comm_vocab_size,
        comm_token_limit=cfg.comm_token_limit,
        max_messages=cfg.comm_max_messages,
        max_agents=max(16, env.num_agents + 1),
        goal_hint_dim=max(cfg.goal_hint_dim, hint.shape[1]),
        hidden_dim=cfg.hidden_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
    )
    return CommMATModel(model_cfg)


def save_checkpoint(path: str, model: CommMATModel, optimizer, step: int):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)


def load_checkpoint(path: str, model: CommMATModel, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))


def train_comm_mat(cfg: CommMATTrainConfig):
    env_cfg = SyncOrSinkConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        num_agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        max_steps=cfg.max_steps,
        comm_token_limit=cfg.comm_token_limit,
        token_vocab_size=cfg.comm_vocab_size,
        max_messages=cfg.comm_max_messages,
        comm_len_cost=cfg.comm_len_cost,
        comm_cost=cfg.comm_cost,
        pipeline_shaping=cfg.pipeline_shaping,
        pipeline_shaping_scale=cfg.pipeline_shaping_scale,
        energy_shaping=cfg.energy_shaping,
        energy_shaping_scale=cfg.energy_shaping_scale,
        signal_shaping=cfg.signal_shaping,
        signal_shaping_scale=cfg.signal_shaping_scale,
    )
    env = SyncOrSinkEnv(env_cfg)

    model = _build_model(env, cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    device = torch.device("cpu")
    model.to(device)
    model.train()

    start_update = 0
    if cfg.load:
        start_update = load_checkpoint(cfg.load, model, optimizer)

    wandb_run = None
    if cfg.wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run, config=vars(cfg))
        except Exception as exc:
            print(f"wandb init failed, continuing without wandb: {exc}")

    for update in range(start_update, cfg.updates):
        grid_buf = []
        inv_buf = []
        pos_buf = []
        hint_buf = []
        rtok_buf = []
        rfrom_buf = []
        mask_buf = []
        act_buf = []
        send_buf = []
        len_buf = []
        tok_buf = []
        logp_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []

        obs, _ = env.reset(seed=update)
        ep_returns, ep_steps, ep_comm = [], [], []
        ep_ret = 0.0
        ep_step = 0
        ep_comm_tokens = 0
        action_hist = np.zeros(8, dtype=np.int64)

        for t in range(cfg.rollout_steps):
            grid, inv, pos, hint, rtok, rfrom, amask = _build_agent_batch(obs, cfg)
            grid_t = torch.tensor(grid, dtype=torch.long, device=device)
            inv_t = torch.tensor(inv, dtype=torch.float32, device=device)
            pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
            hint_t = torch.tensor(hint, dtype=torch.float32, device=device)
            rtok_t = torch.tensor(rtok, dtype=torch.long, device=device)
            rfrom_t = torch.tensor(rfrom, dtype=torch.long, device=device)
            amask_t = torch.tensor(amask, dtype=torch.float32, device=device)

            out = model(grid_t, inv_t, pos_t, hint_t, rtok_t, rfrom_t)
            logits = out["action_logits"].masked_fill((amask_t <= 0).bool(), -1e9)
            action_dist = torch.distributions.Categorical(logits=logits)
            send_dist = torch.distributions.Bernoulli(logits=out["send_logit"])
            len_dist = torch.distributions.Categorical(logits=out["msg_len_logits"])
            tok_dist = torch.distributions.Categorical(logits=out["msg_token_logits"])

            acts = action_dist.sample()
            send = send_dist.sample()
            lens = len_dist.sample()
            toks = tok_dist.sample()

            logp_action = action_dist.log_prob(acts)
            logp_send = send_dist.log_prob(send)
            token_mask = (
                torch.arange(cfg.comm_token_limit, device=device)[None, :] < lens[:, None]
            ).float()
            logp_tokens = (tok_dist.log_prob(toks) * token_mask).sum(dim=-1)
            logp_len = len_dist.log_prob(lens)
            logp = logp_action + logp_send + (logp_len + logp_tokens) * send

            actions = {}
            for i, aid in enumerate(sorted(obs.keys())):
                action_hist[int(acts[i].item())] += 1
                if int(send[i].item()) == 1 and int(lens[i].item()) > 0:
                    L = min(int(lens[i].item()), cfg.comm_token_limit)
                    msg_tokens = toks[i, :L].detach().cpu().tolist()
                else:
                    msg_tokens = []
                actions[aid] = {"action": int(acts[i].item()), "message_tokens": [int(x) for x in msg_tokens]}

            next_obs, rewards, done, truncated, info = env.step(actions)
            if "comm_tokens" in info:
                ep_comm_tokens += sum(info["comm_tokens"].values())

            grid_buf.append(grid_t.detach())
            inv_buf.append(inv_t.detach())
            pos_buf.append(pos_t.detach())
            hint_buf.append(hint_t.detach())
            rtok_buf.append(rtok_t.detach())
            rfrom_buf.append(rfrom_t.detach())
            mask_buf.append(amask_t.detach())
            act_buf.append(acts.detach())
            send_buf.append(send.detach())
            len_buf.append(lens.detach())
            tok_buf.append(toks.detach())
            logp_buf.append(logp.detach())
            val_buf.append(out["value"].detach())
            rew_buf.append(torch.tensor([rewards[i] for i in sorted(obs.keys())], dtype=torch.float32))
            done_buf.append(torch.tensor([done or truncated] * env.num_agents, dtype=torch.float32))

            obs = next_obs
            ep_ret += float(sum(rewards.values()))
            ep_step += 1
            if done or truncated:
                ep_returns.append(ep_ret)
                ep_steps.append(ep_step)
                ep_comm.append(ep_comm_tokens)
                obs, _ = env.reset(seed=update + t + 1)
                ep_ret = 0.0
                ep_step = 0
                ep_comm_tokens = 0

        values = torch.stack(val_buf)  # (T,N)
        rewards = torch.stack(rew_buf)  # (T,N)
        dones = torch.stack(done_buf)  # (T,N)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.shape[1])
        for t in reversed(range(cfg.rollout_steps)):
            next_value = values[t + 1] if t + 1 < cfg.rollout_steps else torch.zeros_like(values[t])
            delta = rewards[t] + cfg.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        grid_b = torch.cat(grid_buf, dim=0)
        inv_b = torch.cat(inv_buf, dim=0)
        pos_b = torch.cat(pos_buf, dim=0)
        hint_b = torch.cat(hint_buf, dim=0)
        rtok_b = torch.cat(rtok_buf, dim=0)
        rfrom_b = torch.cat(rfrom_buf, dim=0)
        mask_b = torch.cat(mask_buf, dim=0)
        act_b = torch.cat(act_buf, dim=0)
        send_b = torch.cat(send_buf, dim=0)
        len_b = torch.cat(len_buf, dim=0)
        tok_b = torch.cat(tok_buf, dim=0)
        logp_b = torch.cat(logp_buf, dim=0)
        val_old_b = values.reshape(-1)
        adv_b = advantages.reshape(-1)
        ret_b = returns.reshape(-1)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        total = grid_b.shape[0]
        idx = np.arange(total)
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, total, cfg.minibatch):
                mb = idx[start : start + cfg.minibatch]
                out = model(
                    grid_b[mb],
                    inv_b[mb],
                    pos_b[mb],
                    hint_b[mb],
                    rtok_b[mb],
                    rfrom_b[mb],
                )
                logits = out["action_logits"].masked_fill((mask_b[mb] <= 0).bool(), -1e9)
                action_dist = torch.distributions.Categorical(logits=logits)
                send_dist = torch.distributions.Bernoulli(logits=out["send_logit"])
                len_dist = torch.distributions.Categorical(logits=out["msg_len_logits"])
                tok_dist = torch.distributions.Categorical(logits=out["msg_token_logits"])

                new_logp_action = action_dist.log_prob(act_b[mb])
                new_logp_send = send_dist.log_prob(send_b[mb])
                token_mask = (
                    torch.arange(cfg.comm_token_limit, device=device)[None, :] < len_b[mb][:, None]
                ).float()
                new_logp_tokens = (tok_dist.log_prob(tok_b[mb]) * token_mask).sum(dim=-1)
                new_logp_len = len_dist.log_prob(len_b[mb])
                new_logp = new_logp_action + new_logp_send + (new_logp_len + new_logp_tokens) * send_b[mb]

                entropy = (
                    action_dist.entropy().mean()
                    + send_dist.entropy().mean()
                    + len_dist.entropy().mean()
                    + tok_dist.entropy().mean()
                )

                ratio = (new_logp - logp_b[mb]).exp()
                surr1 = ratio * adv_b[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_b[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                v_new = out["value"]
                v_old = val_old_b[mb]
                v_clipped = v_old + torch.clamp(v_new - v_old, -cfg.value_clip, cfg.value_clip)
                value_loss = 0.5 * torch.max((ret_b[mb] - v_new).pow(2), (ret_b[mb] - v_clipped).pow(2)).mean()
                loss = policy_loss + value_loss - cfg.entropy * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if wandb_run is not None:
            wandb_run.log(
                {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "update": update,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "rollout/episodes": len(ep_returns),
                    "rollout/mean_ep_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
                    "rollout/mean_ep_len": float(np.mean(ep_steps)) if ep_steps else 0.0,
                    "rollout/mean_ep_comm_tokens": float(np.mean(ep_comm)) if ep_comm else 0.0,
                    "rollout/action_hist_0": int(action_hist[0]),
                    "rollout/action_hist_1": int(action_hist[1]),
                    "rollout/action_hist_2": int(action_hist[2]),
                    "rollout/action_hist_3": int(action_hist[3]),
                    "rollout/action_hist_4": int(action_hist[4]),
                    "rollout/action_hist_5": int(action_hist[5]),
                    "rollout/action_hist_6": int(action_hist[6]),
                    "rollout/action_hist_7": int(action_hist[7]),
                }
            )
        print(f"update {update} loss {float(loss.item()):.3f}")

        if cfg.eval_every > 0 and (update + 1) % cfg.eval_every == 0:
            eval_returns = []
            eval_steps = []
            eval_success = []
            eval_obs, _ = env.reset(seed=10000 + update)
            for ep in range(cfg.eval_episodes):
                done = False
                truncated = False
                total_reward = 0.0
                steps = 0
                while not (done or truncated):
                    grid, inv, pos, hint, rtok, rfrom, amask = _build_agent_batch(eval_obs, cfg)
                    with torch.no_grad():
                        out = model(
                            torch.tensor(grid, dtype=torch.long, device=device),
                            torch.tensor(inv, dtype=torch.float32, device=device),
                            torch.tensor(pos, dtype=torch.float32, device=device),
                            torch.tensor(hint, dtype=torch.float32, device=device),
                            torch.tensor(rtok, dtype=torch.long, device=device),
                            torch.tensor(rfrom, dtype=torch.long, device=device),
                        )
                    logits = out["action_logits"]
                    logits = logits.masked_fill((torch.tensor(amask, dtype=torch.float32, device=device) <= 0).bool(), -1e9)
                    acts = torch.argmax(logits, dim=-1)
                    send = (torch.sigmoid(out["send_logit"]) > cfg.send_threshold).to(torch.int64)
                    lens = torch.argmax(out["msg_len_logits"], dim=-1)
                    toks = torch.argmax(out["msg_token_logits"], dim=-1)

                    actions = {}
                    for i, aid in enumerate(sorted(eval_obs.keys())):
                        if int(send[i].item()) == 1 and int(lens[i].item()) > 0:
                            L = min(int(lens[i].item()), cfg.comm_token_limit)
                            msg_tokens = toks[i, :L].detach().cpu().tolist()
                        else:
                            msg_tokens = []
                        actions[aid] = {"action": int(acts[i].item()), "message_tokens": [int(x) for x in msg_tokens]}

                    eval_obs, rewards, done, truncated, info = env.step(actions)
                    total_reward += float(sum(rewards.values()))
                    steps += 1
                eval_returns.append(total_reward)
                eval_steps.append(steps)
                eval_success.append(1.0 if done else 0.0)
                eval_obs, _ = env.reset(seed=10000 + update + ep + 1)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/mean_return": float(np.mean(eval_returns)),
                        "eval/mean_steps": float(np.mean(eval_steps)),
                        "eval/success_rate": float(np.mean(eval_success)),
                        "eval/update": update,
                    }
                )

        if cfg.save and (update + 1) % cfg.save_every == 0:
            save_checkpoint(cfg.save, model, optimizer, update + 1)

    if cfg.save:
        save_checkpoint(cfg.save, model, optimizer, cfg.updates)
    if wandb_run is not None:
        wandb_run.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="pipeline_assembly")
    p.add_argument("--map-size", type=int, default=8)
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--fov-preset", default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--comm-token-limit", type=int, default=24)
    p.add_argument("--comm-vocab-size", type=int, default=256)
    p.add_argument("--comm-max-messages", type=int, default=8)
    p.add_argument("--comm-len-cost", type=float, default=0.0)
    p.add_argument("--comm-cost", type=float, default=0.01)
    p.add_argument("--pipeline-shaping", action="store_true")
    p.add_argument("--pipeline-shaping-scale", type=float, default=0.01)
    p.add_argument("--energy-shaping", action="store_true")
    p.add_argument("--energy-shaping-scale", type=float, default=0.01)
    p.add_argument("--signal-shaping", action="store_true")
    p.add_argument("--signal-shaping-scale", type=float, default=0.01)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value-clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--goal-hint-dim", type=int, default=32)
    p.add_argument("--send-threshold", type=float, default=0.5)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="syncorsink")
    p.add_argument("--wandb-run", default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--load", default=None)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=5)
    args = p.parse_args()

    cfg = CommMATTrainConfig(
        scenario=args.scenario,
        map_size=args.map_size,
        agents=args.agents,
        fov_preset=args.fov_preset,
        max_steps=args.max_steps,
        comm_token_limit=args.comm_token_limit,
        comm_vocab_size=args.comm_vocab_size,
        comm_max_messages=args.comm_max_messages,
        comm_len_cost=args.comm_len_cost,
        comm_cost=args.comm_cost,
        pipeline_shaping=args.pipeline_shaping,
        pipeline_shaping_scale=args.pipeline_shaping_scale,
        energy_shaping=args.energy_shaping,
        energy_shaping_scale=args.energy_shaping_scale,
        signal_shaping=args.signal_shaping,
        signal_shaping_scale=args.signal_shaping_scale,
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
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        goal_hint_dim=args.goal_hint_dim,
        send_threshold=args.send_threshold,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        save=args.save,
        load=args.load,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )
    train_comm_mat(cfg)


if __name__ == "__main__":
    main()
