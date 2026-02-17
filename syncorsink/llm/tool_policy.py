from __future__ import annotations

from typing import Callable, Dict, Any

from .policy import default_prompt, stabilize_agent_action, update_agent_memory
from .tools import parse_tool_calls, parse_openai_tool_calls, tool_action_to_env
from .responses_adapter import parse_responses_tool_calls


class ToolCallingPolicy:
    """
    LLM policy that expects tool calls.

    The LLM should return a list of tool calls, e.g.:
    [
      {"name": "move", "arguments": {"direction": "up"}},
      {"name": "message", "arguments": {"text": "heading north"}}
    ]
    """

    def __init__(
        self,
        llm_call: Callable[[str], Any],
        prompt_fn: Callable[..., str] | None = None,
        default_action: int = 4,
    ):
        self.llm_call = llm_call
        self.prompt_fn = prompt_fn or default_prompt
        self.default_action = default_action

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        for agent_id in obs.keys():
            update_agent_memory(obs[agent_id], info, agent_id, state)
            try:
                prompt = self.prompt_fn(obs, info, agent_id, state)
            except TypeError:
                prompt = self.prompt_fn(obs, info, agent_id)
            raw = self.llm_call(prompt)
            if isinstance(raw, dict) and "tool_calls" in raw:
                calls = parse_openai_tool_calls(raw)
            elif isinstance(raw, dict) and "output" in raw:
                calls = parse_responses_tool_calls(raw)
            else:
                calls = parse_tool_calls(raw)
            action = {"action": self.default_action, "message_text": None}
            for call in calls:
                result = tool_action_to_env(call, default_action=self.default_action)
                # last tool wins for action, but message can be set by message tool
                if result.get("message_text"):
                    action["message_text"] = result["message_text"]
                if "action" in result and result["action"] is not None:
                    action["action"] = result["action"]
            model_action = dict(action)
            action = stabilize_agent_action(action, obs[agent_id], agent_id, state)
            actions[agent_id] = action
            if isinstance(state, dict):
                trace = state.setdefault("llm_calls", [])
                trace.append(
                    {
                        "agent_id": int(agent_id),
                        "mode": "tools",
                        "prompt": prompt,
                        "raw_response": raw,
                        "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in calls],
                        "model_action": model_action,
                        "parsed_action": action,
                    }
                )
        return actions
