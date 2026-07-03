from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List
import hashlib
import json

from .metrics import EpisodeStats
from .success import episode_success
from syncorsink.policies.submission import reset_policy


@dataclass
class LLMRunConfig:
    max_tokens: int = 128
    message_budget: int = 24
    cache: bool = True


def run_llm_episodes(
    env,
    policy,
    episodes: int = 10,
    seed: int | None = None,
    per_episode_cb=None,
    per_step_cb=None,
):
    results: List[EpisodeStats] = []
    base_seed = seed
    for ep in range(episodes):
        ep_seed = None if base_seed is None else base_seed + ep
        obs, info = env.reset(seed=ep_seed)
        reset_policy(policy, episode=ep, seed=ep_seed)
        done = False
        truncated = False
        steps = 0
        policy_state = {"episode": ep, "step": 0}
        total_reward = 0.0
        comm_tokens = 0
        per_agent_reward = {i: 0.0 for i in range(env.num_agents)}
        per_agent_comm = {i: 0 for i in range(env.num_agents)}

        while not (done or truncated):
            policy_state["step"] = steps
            policy_state["llm_calls"] = []
            prev_obs = obs
            prev_info = info
            actions = policy(obs, info, policy_state)
            obs, rewards, done, truncated, info = env.step(actions)
            steps += 1
            total_reward += sum(rewards.values())
            for aid, r in rewards.items():
                per_agent_reward[aid] += r
            if "comm_tokens" in info:
                comm_tokens += sum(info["comm_tokens"].values())
                for aid, c in info["comm_tokens"].items():
                    per_agent_comm[aid] += c
            if per_step_cb is not None:
                per_step_cb(ep, steps - 1, prev_obs, prev_info, actions, rewards, done, truncated, info, policy_state)

        scenario = getattr(getattr(env, "config", None), "scenario", None)
        success_flag = episode_success(scenario, done, info)
        ep_stats = EpisodeStats(
            total_reward=total_reward,
            steps=steps,
            success=success_flag,
            comm_tokens=comm_tokens,
            per_agent_reward=per_agent_reward,
            per_agent_comm=per_agent_comm,
        )
        results.append(ep_stats)
        if per_episode_cb is not None:
            per_episode_cb(ep, ep_stats)

    return results
