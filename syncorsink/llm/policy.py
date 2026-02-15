from __future__ import annotations

import json
from typing import Callable, Dict, Any, List

from .batch import batch_llm_call
from .cache import PromptCache


ACTION_NAMES = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "stay": 4,
    "interact": 5,
    "pickup": 6,
    "drop": 7,
}


def grid_to_ascii(local_grid) -> str:
    mapping = {
        0: ".",
        1: "#",
        2: "R",
        3: "S",
        4: "N",
        5: "C",
        6: "T",
        7: "W",
        8: "B",
        9: "D",
        10: "?",
    }
    lines = []
    for row in local_grid:
        lines.append("".join(mapping.get(int(v), "?") for v in row))
    return "\n".join(lines)


def default_prompt(obs: dict, info: dict, agent_id: int) -> str:
    local = obs[agent_id]["local_grid"]
    inventory = int(obs[agent_id]["inventory"][0])
    hints = info.get("goal_hint_texts", {}).get(agent_id)
    messages = info.get("messages_text", {}).get(agent_id, [])
    prompt = [
        "You are an agent in a cooperative gridworld.",
        "Choose an action and an optional message.",
        "Return JSON: {\"action\": int or name, \"message_text\": str}.",
        "Valid actions: up, down, left, right, stay, interact, pickup, drop.",
        f"Inventory: {inventory}",
        "Local view:",
        grid_to_ascii(local),
    ]
    if hints:
        prompt.append(f"Hint: {hints}")
    if messages:
        prompt.append("Messages: " + " | ".join(m for m in messages if m))
    return "\n".join(prompt)


class LLMPolicy:
    def __init__(
        self,
        llm_call: Callable[[str], str],
        prompt_fn: Callable[[dict, dict, int], str] | None = None,
        default_action: int = 4,
        postprocess: Callable[[dict], dict] | None = None,
        cache: PromptCache | None = None,
        batch_fn: Callable[[List[str], Callable[[str], str]], List[str]] | None = None,
    ):
        self.llm_call = llm_call
        self.prompt_fn = prompt_fn or default_prompt
        self.default_action = default_action
        self.postprocess = postprocess
        self.cache = cache
        self.batch_fn = batch_fn or batch_llm_call

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        prompts = []
        order = []
        for agent_id in obs.keys():
            prompt = self.prompt_fn(obs, info, agent_id)
            order.append(agent_id)
            prompts.append(prompt)

        responses = []
        for p in prompts:
            if self.cache:
                cached = self.cache.get(p)
                if cached is not None:
                    responses.append(cached)
                    continue
            responses.append(None)

        missing_idx = [i for i, r in enumerate(responses) if r is None]
        if missing_idx:
            to_call = [prompts[i] for i in missing_idx]
            called = self.batch_fn(to_call, self.llm_call)
            for idx, resp in zip(missing_idx, called):
                responses[idx] = resp
                if self.cache:
                    self.cache.set(prompts[idx], resp)

        for agent_id, raw in zip(order, responses):
            action = self._parse_response(raw)
            if self.postprocess:
                action = self.postprocess(action)
            actions[agent_id] = action
        return actions

    def _parse_response(self, text: str) -> dict:
        # JSON path
        try:
            data = json.loads(text)
            act = data.get("action", self.default_action)
            action_id = self._action_to_id(act)
            message_text = data.get("message_text")
            return {"action": action_id, "message_text": message_text}
        except Exception:
            pass
        # heuristic parse
        for token in str(text).replace("{", " ").replace("}", " ").split():
            token = token.strip().lower()
            if token.isdigit():
                return {"action": int(token), "message_text": None}
            if token in ACTION_NAMES:
                return {"action": ACTION_NAMES[token], "message_text": None}
        return {"action": self.default_action, "message_text": None}

    def _action_to_id(self, act) -> int:
        if isinstance(act, int):
            return act
        if isinstance(act, str):
            return ACTION_NAMES.get(act.lower(), self.default_action)
        return self.default_action
