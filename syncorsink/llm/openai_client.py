from __future__ import annotations

from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .tools import openai_tools_schema


class OpenAIToolCaller:
    """
    Thin adapter around an OpenAI client to perform tool-calling in batch.
    This expects a client with `chat.completions.create` compatible signature.
    """

    def __init__(self, client, model: str, tools: List[Dict[str, Any]] | None = None):
        self.client = client
        self.model = model
        self.tools = tools or openai_tools_schema()

    def call(self, prompt: str) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=self.tools,
            tool_choice="auto",
        )
        message = resp.choices[0].message
        return message

    def batch(self, prompts: List[str], max_workers: int = 8) -> List[Dict[str, Any]]:
        # Parallel batching using threads to overlap network calls.
        results: List[Dict[str, Any]] = [None for _ in prompts]  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {ex.submit(self.call, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(future_map):
                idx = future_map[fut]
                results[idx] = fut.result()
        return results
