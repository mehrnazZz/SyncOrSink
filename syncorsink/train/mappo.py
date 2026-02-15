from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.policies.mappo_models import MAPPOActor, MAPPOCentralValue


@dataclass
class MAPPOConfig:
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
    shared_actor: bool = False
    backbone: str = "mlp"
    wandb: bool = False
    wandb_project: str = "syncorsink"
    wandb_run: Optional[str] = None
    save: Optional[str] = None
    load: Optional[str] = None
    save_every: int = 5
    eval_every: int = 10
    eval_episodes: int = 5


def _flatten_array(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def flatten_obs(obs_agent):
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
    ]
    return np.concatenate(parts, axis=0)


def build_batch_obs(obs, num_agents):
    obs_list = []
    for aid in range(num_agents):
        obs_list.append(flatten_obs(obs[aid]))
    return np.stack(obs_list)


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


def train_mappo(cfg: MAPPOConfig):
    config = SyncOrSinkConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        num_agents=cfg.agents,
        fov_preset=cfg.fov_preset,
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
        max_steps=cfg.max_steps,
    )
    env = SyncOrSinkEnv(config)

    sample_obs, _ = env.reset(seed=0)
    obs_dim = flatten_obs(sample_obs[0]).shape[0]
    joint_obs_dim = obs_dim * env.num_agents
    action_dim = 8

    if cfg.shared_actor:
        actors = [
            MAPPOActor(
                obs_dim,
                action_dim,
                backbone=cfg.backbone,
                comm_enabled=cfg.comm,
                comm_token_limit=cfg.comm_token_limit,
                comm_vocab_size=cfg.comm_vocab_size,
            )
        ]
    else:
        actors = [
            MAPPOActor(
                obs_dim,
                action_dim,
                backbone=cfg.backbone,
                comm_enabled=cfg.comm,
                comm_token_limit=cfg.comm_token_limit,
                comm_vocab_size=cfg.comm_vocab_size,
            )
            for _ in range(env.num_agents)
        ]
    critic = MAPPOCentralValue(joint_obs_dim)

    params = list(critic.parameters())
    for actor in actors:
        params += list(actor.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr)

    start_update = 0
    if cfg.load:
        start_update = load_checkpoint(cfg.load, actors, critic, optimizer)

    wandb_run = None
    if cfg.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run, config=vars(cfg))
        except Exception as exc:
            print(f"wandb init failed, continuing without wandb: {exc}")
            wandb_run = None

    global_step = 0
    for update in range(start_update, cfg.updates):
        obs_buf = []
        act_buf = []
        send_buf = []
        token_buf = []
        len_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        joint_obs_buf = []
        action_hist = np.zeros(8, dtype=np.int64)
        ep_returns = []
        ep_steps = []
        ep_comm = []
        comm_send_counts = 0
        comm_total_steps = 0
        comm_len_sum = 0.0
        comm_len_count = 0
        comm_len_samples = []
        comm_token_entropy = 0.0

        obs, _ = env.reset(seed=update)
        ep_return = 0.0
        ep_step = 0
        ep_comm_tokens = 0
        for t in range(cfg.rollout_steps):
            obs_batch = build_batch_obs(obs, env.num_agents)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)
            joint_obs = torch.tensor(obs_batch.reshape(-1), dtype=torch.float32)

            if cfg.comm:
                logits_list = []
                send_list = []
                token_list = []
                len_list = []
                for aid in range(env.num_agents):
                    actor = actors[0] if cfg.shared_actor else actors[aid]
                    action_logits, send_logits, token_logits, len_logits = actor(obs_tensor[aid : aid + 1])
                    logits_list.append(action_logits)
                    send_list.append(send_logits)
                    token_list.append(token_logits)
                    len_list.append(len_logits)
                logits = torch.cat(logits_list, dim=0)
                send_logits = torch.cat(send_list, dim=0)
                token_logits = torch.cat(token_list, dim=0)
                len_logits = torch.cat(len_list, dim=0)

                action_dist = torch.distributions.Categorical(logits=logits)
                send_dist = torch.distributions.Bernoulli(logits=send_logits.squeeze(-1))
                token_dist = torch.distributions.Categorical(logits=token_logits)
                len_dist = torch.distributions.Categorical(logits=len_logits)

                acts = action_dist.sample()
                send = send_dist.sample()
                token_samples = token_dist.sample()
                len_samples = len_dist.sample()

                logp_action = action_dist.log_prob(acts)
                logp_send = send_dist.log_prob(send)
                token_mask = (torch.arange(cfg.comm_token_limit)[None, :].to(len_samples.device) < len_samples[:, None]).float()
                logp_tokens = (token_dist.log_prob(token_samples) * token_mask).sum(dim=-1)
                logp_len = len_dist.log_prob(len_samples)
                logp = logp_action + logp_send + (logp_len + logp_tokens) * send

                actions = {}
                for aid in range(env.num_agents):
                    if int(send[aid].item()) == 1 and int(len_samples[aid].item()) > 0:
                        msg_tokens = token_samples[aid][: int(len_samples[aid].item())].tolist()
                    else:
                        msg_tokens = []
                    actions[aid] = {"action": int(acts[aid].item()), "message_tokens": msg_tokens}
                send_buf.append(send.detach())
                token_buf.append(token_samples.detach())
                len_buf.append(len_samples.detach())

                comm_total_steps += env.num_agents
                comm_send_counts += int(send.sum().item())
                if int(send.sum().item()) > 0:
                    comm_len_sum += float(len_samples[send.bool()].sum().item())
                    comm_len_count += int(send.sum().item())
                    comm_len_samples.extend(len_samples[send.bool()].tolist())
                comm_token_entropy += float(token_dist.entropy().mean().item())
            else:
                logits_list = []
                for aid in range(env.num_agents):
                    actor = actors[0] if cfg.shared_actor else actors[aid]
                    logits_list.append(actor(obs_tensor[aid : aid + 1]))
                logits = torch.cat(logits_list, dim=0)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logp = dist.log_prob(acts)

                actions = {aid: {"action": int(acts[aid].item()), "message_tokens": []} for aid in range(env.num_agents)}
            next_obs, rewards, done, truncated, info = env.step(actions)
            for a in acts.tolist():
                action_hist[a] += 1
            if "comm_tokens" in info:
                ep_comm_tokens += sum(info["comm_tokens"].values())

            obs_buf.append(obs_tensor)
            joint_obs_buf.append(joint_obs)
            act_buf.append(acts)
            logp_buf.append(logp.detach())
            val_buf.append(critic(joint_obs).detach())
            rew_buf.append(torch.tensor([rewards[i] for i in range(env.num_agents)], dtype=torch.float32))
            done_buf.append(torch.tensor([done or truncated] * env.num_agents, dtype=torch.float32))

            obs = next_obs
            ep_return += sum(rewards.values())
            ep_step += 1
            global_step += 1
            if done or truncated:
                ep_returns.append(ep_return)
                ep_steps.append(ep_step)
                ep_comm.append(ep_comm_tokens)
                obs, _ = env.reset(seed=update + t + 1)
                ep_return = 0.0
                ep_step = 0
                ep_comm_tokens = 0

        values = torch.stack(val_buf)
        rewards = torch.stack(rew_buf)
        dones = torch.stack(done_buf)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.shape[1])
        for t in reversed(range(cfg.rollout_steps)):
            next_value = values[t + 1] if t + 1 < cfg.rollout_steps else torch.zeros_like(values[t])
            delta = rewards[t].mean() + cfg.gamma * next_value * (1.0 - dones[t].mean()) - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - dones[t].mean()) * gae
            advantages[t] = gae
        returns = advantages + values.unsqueeze(-1)

        obs_b = torch.cat(obs_buf, dim=0)
        act_b = torch.cat(act_buf, dim=0)
        logp_b = torch.cat(logp_buf, dim=0)
        if cfg.comm:
            send_b = torch.stack(send_buf, dim=0).reshape(-1)
            token_b = torch.stack(token_buf, dim=0).reshape(-1, cfg.comm_token_limit)
            len_b = torch.stack(len_buf, dim=0).reshape(-1)
        adv_b = advantages.flatten()
        ret_b = returns.flatten()
        joint_obs_b = torch.stack(joint_obs_buf).repeat_interleave(env.num_agents, dim=0)

        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        total = obs_b.shape[0]
        idx = np.arange(total)
        for epoch in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, total, cfg.minibatch):
                mb = idx[start : start + cfg.minibatch]
                if cfg.comm:
                    if cfg.shared_actor:
                        logits, send_logits, token_logits, len_logits = actors[0](obs_b[mb])
                    else:
                        logits_list = []
                        send_list = []
                        token_list = []
                        len_list = []
                        for idx_i in mb:
                            aid = idx_i % env.num_agents
                            action_logits, send_logits, token_logits, len_logits = actors[aid](obs_b[idx_i : idx_i + 1])
                            logits_list.append(action_logits)
                            send_list.append(send_logits)
                            token_list.append(token_logits)
                            len_list.append(len_logits)
                        logits = torch.cat(logits_list, dim=0)
                        send_logits = torch.cat(send_list, dim=0)
                        token_logits = torch.cat(token_list, dim=0)
                        len_logits = torch.cat(len_list, dim=0)

                    action_dist = torch.distributions.Categorical(logits=logits)
                    send_dist = torch.distributions.Bernoulli(logits=send_logits.squeeze(-1))
                    token_dist = torch.distributions.Categorical(logits=token_logits)
                    len_dist = torch.distributions.Categorical(logits=len_logits)

                    new_logp_action = action_dist.log_prob(act_b[mb])
                    new_logp_send = send_dist.log_prob(send_b[mb])
                    token_mask = (torch.arange(cfg.comm_token_limit)[None, :].to(token_b.device) < len_b[mb][:, None]).float()
                    new_logp_tokens = (token_dist.log_prob(token_b[mb]) * token_mask).sum(dim=-1)
                    new_logp_len = len_dist.log_prob(len_b[mb])
                    new_logp = new_logp_action + new_logp_send + (new_logp_len + new_logp_tokens) * send_b[mb]
                    entropy = (
                        action_dist.entropy().mean()
                        + send_dist.entropy().mean()
                        + token_dist.entropy().mean()
                        + len_dist.entropy().mean()
                    )
                else:
                    if cfg.shared_actor:
                        logits = actors[0](obs_b[mb])
                    else:
                        logits_list = []
                        for idx_i in mb:
                            aid = idx_i % env.num_agents
                            logits_list.append(actors[aid](obs_b[idx_i : idx_i + 1]))
                        logits = torch.cat(logits_list, dim=0)

                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(act_b[mb])
                    entropy = dist.entropy().mean()
                ratio = (new_logp - logp_b[mb]).exp()
                surr1 = ratio * adv_b[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_b[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                values = critic(joint_obs_b[mb])
                v_old = torch.stack(val_buf).repeat_interleave(env.num_agents, dim=0)[mb]
                v_clipped = v_old + torch.clamp(values - v_old, -cfg.value_clip, cfg.value_clip)
                value_loss = 0.5 * torch.max((ret_b[mb] - values).pow(2), (ret_b[mb] - v_clipped).pow(2)).mean()
                loss = policy_loss + value_loss - cfg.entropy * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if wandb_run is not None:
            comm_send_rate = (comm_send_counts / comm_total_steps) if comm_total_steps else 0.0
            comm_mean_len = (comm_len_sum / comm_len_count) if comm_len_count else 0.0
            comm_token_entropy = (comm_token_entropy / cfg.rollout_steps) if cfg.rollout_steps else 0.0
            log_payload = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "update": update,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "rollout/action_hist_0": int(action_hist[0]),
                "rollout/action_hist_1": int(action_hist[1]),
                "rollout/action_hist_2": int(action_hist[2]),
                "rollout/action_hist_3": int(action_hist[3]),
                "rollout/action_hist_4": int(action_hist[4]),
                "rollout/action_hist_5": int(action_hist[5]),
                "rollout/action_hist_6": int(action_hist[6]),
                "rollout/action_hist_7": int(action_hist[7]),
                "rollout/episodes": len(ep_returns),
                "rollout/mean_ep_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
                "rollout/mean_ep_len": float(np.mean(ep_steps)) if ep_steps else 0.0,
                "rollout/mean_ep_comm_tokens": float(np.mean(ep_comm)) if ep_comm else 0.0,
                "rollout/comm_send_rate": comm_send_rate,
                "rollout/comm_mean_len": comm_mean_len,
                "rollout/comm_token_entropy": comm_token_entropy,
            }
            if comm_len_samples:
                try:
                    import wandb
                    log_payload["rollout/comm_len_hist"] = wandb.Histogram(comm_len_samples)
                except Exception:
                    pass
            wandb_run.log(
                log_payload
            )
        print(f"update {update} loss {loss.item():.3f}")

        # periodic evaluation
        if cfg.eval_every > 0 and (update + 1) % cfg.eval_every == 0:
            eval_returns = []
            eval_steps = []
            eval_success = []
            eval_obs, _ = env.reset(seed=10000 + update)
            for ep in range(cfg.eval_episodes):
                done = False
                truncated = False
                steps = 0
                total_reward = 0.0
                while not (done or truncated):
                    obs_batch = build_batch_obs(eval_obs, env.num_agents)
                    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)
                    if cfg.comm:
                        logits_list = []
                        send_list = []
                        token_list = []
                        len_list = []
                        for aid in range(env.num_agents):
                            actor = actors[0] if cfg.shared_actor else actors[aid]
                            action_logits, send_logits, token_logits, len_logits = actor(obs_tensor[aid:aid+1])
                            logits_list.append(action_logits)
                            send_list.append(send_logits)
                            token_list.append(token_logits)
                            len_list.append(len_logits)
                        logits = torch.cat(logits_list, dim=0)
                        send_logits = torch.cat(send_list, dim=0)
                        token_logits = torch.cat(token_list, dim=0)
                        len_logits = torch.cat(len_list, dim=0)
                        acts = torch.argmax(logits, dim=-1)
                        send = (torch.sigmoid(send_logits.squeeze(-1)) > 0.5).to(torch.int64)
                        token_samples = torch.argmax(token_logits, dim=-1)
                        len_samples = torch.argmax(len_logits, dim=-1)
                        actions = {}
                        for aid in range(env.num_agents):
                            if int(send[aid].item()) == 1 and int(len_samples[aid].item()) > 0:
                                msg_tokens = token_samples[aid][: int(len_samples[aid].item())].tolist()
                            else:
                                msg_tokens = []
                            actions[aid] = {"action": int(acts[aid].item()), "message_tokens": msg_tokens}
                    else:
                        logits_list = []
                        for aid in range(env.num_agents):
                            actor = actors[0] if cfg.shared_actor else actors[aid]
                            logits_list.append(actor(obs_tensor[aid:aid+1]))
                        logits = torch.cat(logits_list, dim=0)
                        acts = torch.argmax(logits, dim=-1)
                        actions = {aid: {"action": int(acts[aid].item()), "message_tokens": []} for aid in range(env.num_agents)}
                    eval_obs, rewards, done, truncated, info = env.step(actions)
                    total_reward += sum(rewards.values())
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
            save_checkpoint(cfg.save, actors, critic, optimizer, update + 1)

    if cfg.save:
        save_checkpoint(cfg.save, actors, critic, optimizer, cfg.updates)

    if wandb_run is not None:
        wandb_run.finish()


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--pipeline-shaping", action="store_true")
    parser.add_argument("--pipeline-shaping-scale", type=float, default=0.01)
    parser.add_argument("--energy-shaping", action="store_true")
    parser.add_argument("--energy-shaping-scale", type=float, default=0.01)
    parser.add_argument("--signal-shaping", action="store_true")
    parser.add_argument("--signal-shaping-scale", type=float, default=0.01)
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
    parser.add_argument("--shared-actor", action="store_true")
    parser.add_argument("--backbone", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
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
        pipeline_shaping=args.pipeline_shaping,
        pipeline_shaping_scale=args.pipeline_shaping_scale,
        energy_shaping=args.energy_shaping,
        energy_shaping_scale=args.energy_shaping_scale,
        signal_shaping=args.signal_shaping,
        signal_shaping_scale=args.signal_shaping_scale,
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
        shared_actor=args.shared_actor,
        backbone=args.backbone,
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
