from __future__ import annotations

from typing import Callable, Dict, Any, List
import time

from .metrics import EpisodeStats, EvalSummary, summarize


PolicyFn = Callable[[dict, dict, dict], Dict[int, dict]]


def run_episodes(
    env,
    policy: PolicyFn,
    episodes: int = 10,
    seed: int | None = None,
    per_episode_cb: callable | None = None,
    per_step_cb: callable | None = None,
    render: bool = False,
    render_fps: float = 10.0,
):
    results: List[EpisodeStats] = []
    base_seed = seed
    for ep in range(episodes):
        ep_seed = None if base_seed is None else base_seed + ep
        obs, info = env.reset(seed=ep_seed)
        done = False
        truncated = False
        steps = 0
        total_reward = 0.0
        comm_tokens = 0
        per_agent_reward = {i: 0.0 for i in range(env.num_agents)}
        per_agent_comm = {i: 0 for i in range(env.num_agents)}
        last_info: dict[str, Any] = {}
        while not (done or truncated):
            prev_obs = obs
            prev_info = info
            actions = policy(obs, info, {"step": steps})
            obs, rewards, done, truncated, info = env.step(actions)
            last_info = info or {}
            steps += 1
            total_reward += sum(rewards.values())
            for aid, r in rewards.items():
                per_agent_reward[aid] += r
            if "comm_tokens" in info:
                comm_tokens += sum(info["comm_tokens"].values())
                for aid, c in info["comm_tokens"].items():
                    per_agent_comm[aid] += c
            if per_step_cb is not None:
                per_step_cb(ep, steps - 1, prev_obs, prev_info, actions, rewards, done, truncated, info)
            if render:
                env.render()
                if render_fps > 0:
                    time.sleep(1.0 / render_fps)
        if getattr(env, "config", None) is not None and env.config.scenario == "energy_grid":
            # EnergyGrid: success if recharge target hit; failure if depleted early.
            success = bool(last_info.get("success", False))
        else:
            success = bool(done)
        ep_stats = EpisodeStats(
            total_reward=total_reward,
            steps=steps,
            success=success,
            comm_tokens=comm_tokens,
            per_agent_reward=per_agent_reward,
            per_agent_comm=per_agent_comm,
        )
        results.append(ep_stats)
        if per_episode_cb is not None:
            per_episode_cb(ep, ep_stats)
    return summarize(results), results
