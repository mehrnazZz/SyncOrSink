from __future__ import annotations

from typing import Dict, Any

from .base import BasePolicy
from .registry import register


@register("vlm")
class VLMPolicy(BasePolicy):
    """
    Adapter for a VLM-style policy. Expects predict(image, text=None).
    The env should be set to render_mode="rgb_array".
    """

    def __init__(self, model, text_prompt_fn=None):
        self.model = model
        self.text_prompt_fn = text_prompt_fn

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        # Assume obs includes a shared rgb frame in info if provided externally.
        frame = info.get("rgb_frame")
        for agent_id in obs.keys():
            text = self.text_prompt_fn(obs, info, agent_id) if self.text_prompt_fn else None
            action = self.model.predict(frame, text)
            if isinstance(action, dict):
                actions[agent_id] = action
            else:
                actions[agent_id] = {"action": int(action), "message_tokens": []}
        return actions
