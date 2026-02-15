from __future__ import annotations

from typing import Dict, Any


class BasePolicy:
    def reset(self):
        return None

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        raise NotImplementedError
