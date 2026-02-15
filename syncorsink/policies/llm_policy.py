from __future__ import annotations

from typing import Dict

from .base import BasePolicy
from .registry import register
from syncorsink.llm.policy import LLMPolicy


@register("llm")
class LLMPolicyAdapter(BasePolicy):
    def __init__(self, llm_call, **kwargs):
        self.policy = LLMPolicy(llm_call, **kwargs)

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        return self.policy(obs, info, state)
