from __future__ import annotations

from .base import SyncOrSinkConfig, SyncOrSinkEnv


class SyncOrSinkVector:
    """
    Lightweight vector wrapper that returns python lists of per-env outputs.
    This avoids Gym vector concatenation, which doesn't support dict-of-agents.
    """

    def __init__(self, num_envs: int, config: SyncOrSinkConfig | None = None):
        self.num_envs = num_envs
        self.envs = [SyncOrSinkEnv(config=config) for _ in range(num_envs)]

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs_batch = []
        info_batch = []
        base_seed = seed if seed is not None else None
        for i, env in enumerate(self.envs):
            s = None if base_seed is None else base_seed + i
            obs, info = env.reset(seed=s, options=options)
            obs_batch.append(obs)
            info_batch.append(info)
        return obs_batch, info_batch

    def step(self, actions):
        obs_batch = []
        reward_batch = []
        term_batch = []
        trunc_batch = []
        info_batch = []
        for env, action in zip(self.envs, actions):
            obs, rewards, done, truncated, info = env.step(action)
            obs_batch.append(obs)
            reward_batch.append(rewards)
            term_batch.append(done)
            trunc_batch.append(truncated)
            info_batch.append(info)
        return obs_batch, reward_batch, term_batch, trunc_batch, info_batch

    def close(self):
        for env in self.envs:
            env.close()
        return None
