from __future__ import annotations

from typing import Any

try:
    from pettingzoo.utils import ParallelEnv
except Exception as exc:  # pragma: no cover
    raise ImportError("PettingZoo is required for this wrapper. Install with `pip install -e .[pettingzoo]`.") from exc

from .base import SyncOrSinkConfig, SyncOrSinkEnv


class SyncOrSinkParallel(ParallelEnv):
    metadata = {"name": "SyncOrSinkParallel", "render_modes": ["ansi"]}

    def __init__(self, config: SyncOrSinkConfig | None = None, render_mode: str | None = None):
        self._env = SyncOrSinkEnv(config=config, render_mode=render_mode)
        self.possible_agents = [f"agent_{i}" for i in range(self._env.num_agents)]
        self.agents = list(self.possible_agents)
        self.render_mode = render_mode

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self._env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        return self._to_pz_obs(obs), self._to_pz_info(info)

    def step(self, actions: dict[str, Any]):
        sos_actions = {}
        for agent_name, action in actions.items():
            idx = int(agent_name.split("_")[-1])
            sos_actions[idx] = action
        obs, rewards, done, truncated, info = self._env.step(sos_actions)

        terminations = {name: done for name in self.agents}
        truncations = {name: truncated for name in self.agents}
        rewards = {f"agent_{i}": rewards[i] for i in range(self._env.num_agents)}
        infos = self._to_pz_info(info)
        observations = self._to_pz_obs(obs)

        if done or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self._env.render()

    def close(self):
        return None

    def observation_space(self, agent: str):
        return self._env.observation_space

    def action_space(self, agent: str):
        return self._env.action_space

    def _to_pz_obs(self, obs: dict[int, dict]) -> dict[str, dict]:
        return {f"agent_{i}": obs[i] for i in range(self._env.num_agents)}

    def _to_pz_info(self, info: dict) -> dict[str, dict]:
        # replicate shared info for all agents
        return {f"agent_{i}": info for i in range(self._env.num_agents)}
