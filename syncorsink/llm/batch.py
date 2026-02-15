from __future__ import annotations

from typing import Callable, List


def batch_llm_call(prompts: List[str], llm_call: Callable[[str], str]) -> List[str]:
    # default: sequential batching
    return [llm_call(p) for p in prompts]
