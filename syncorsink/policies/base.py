from __future__ import annotations

from typing import Dict, Any


class BasePolicy:
    def reset(self, *args, **kwargs):
        return None

    def load_checkpoint(self, path: str):
        return None

    def metadata(self) -> Dict[str, Any]:
        return {}

    def act(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        raise NotImplementedError

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        return self.act(obs, info, state)
