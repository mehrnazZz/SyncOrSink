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

def train_bc(cfg: BCConfig):
    """Train a policy network via behavioral cloning on collected demonstrations."""
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

    # Build model (same architecture as MAPPO actor)
    model = MAPPOActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        backbone="mlp",
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    action_loss_fn = nn.CrossEntropyLoss()

    # W&B
    wandb_run = None
    if cfg.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run, config=vars(cfg))
        except Exception as exc:
            print(f"wandb init failed: {exc}")

    # Training loop
    idx = np.arange(N)
    for epoch in range(cfg.epochs):
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
                logits, send_logits, token_logits, len_logits = model(obs_b)

                # Action loss
                a_loss = action_loss_fn(logits, act_b)

                # Comm losses: send gate, token content, message length
                # For oracle demos without comm, msg_lens will be 0 → send=0
                msg_b = msg_all[mb].to(device)
                mlen_b = msg_lens[mb].to(device)
                send_target = (mlen_b > 0).float()

                send_loss = nn.functional.binary_cross_entropy_with_logits(
                    send_logits.squeeze(-1), send_target
                )
                len_loss = nn.functional.cross_entropy(len_logits, mlen_b)

                # Token loss only for positions within message length
                t_mask = (torch.arange(cfg.comm_token_limit, device=device)[None, :] < mlen_b[:, None]).float()
                if t_mask.sum() > 0:
                    tok_loss = (nn.functional.cross_entropy(
                        token_logits.reshape(-1, cfg.comm_vocab_size),
                        msg_b.reshape(-1),
                        reduction="none",
                    ).reshape(token_logits.shape[0], -1) * t_mask).sum() / t_mask.sum()
                else:
                    tok_loss = torch.tensor(0.0, device=device)

                comm_loss = send_loss + len_loss + tok_loss
                loss = a_loss + 0.5 * comm_loss
                total_comm_loss += comm_loss.item()
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
        avg_c_loss = total_comm_loss / batches if cfg.comm else 0.0

        print(f"epoch {epoch:3d} | loss {avg_loss:.4f} | action_loss {avg_a_loss:.4f} | "
              f"comm_loss {avg_c_loss:.4f} | action_acc {acc:.3f}")

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "loss": avg_loss,
                "action_loss": avg_a_loss,
                "comm_loss": avg_c_loss,
                "action_accuracy": acc,
            })

    # Save
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

    if wandb_run is not None:
        wandb_run.finish()

    return model


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
    train_p.add_argument("--device", default="auto")
    train_p.add_argument("--save", default=None, help="Path to save trained model")
    train_p.add_argument("--wandb", action="store_true")
    train_p.add_argument("--wandb-project", default="syncorsink")
    train_p.add_argument("--wandb-run", default=None)

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
            device=args.device,
            demo_path=args.demo_path,
            save=args.save,
            wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run=args.wandb_run,
        )
        train_bc(cfg)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
