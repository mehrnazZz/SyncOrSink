from __future__ import annotations

from typing import Any, Dict, List
import json

from .tools import ToolCall, tool_action_to_env


def parse_responses_tool_calls(response: Dict[str, Any]) -> List[ToolCall]:
    """
    Parse tool calls from the OpenAI Responses API output structure.
    Expected shape (simplified):
      {"output": [{"type": "tool_call", "name": "move", "arguments": "{...}"}, ...]}
    """
    calls: List[ToolCall] = []
    for item in response.get("output", []):
        if item.get("type") != "tool_call":
            continue
        name = item.get("name")
        args = item.get("arguments", "{}")
        try:
            arguments = json.loads(args) if isinstance(args, str) else args
        except Exception:
            arguments = {}
        if name:
            calls.append(ToolCall(name=name, arguments=arguments))
    return calls


def response_tool_calls_to_action(response: Dict[str, Any], default_action: int = 4) -> dict:
    calls = parse_responses_tool_calls(response)
    action = {"action": default_action, "message_text": None}
    for call in calls:
        result = tool_action_to_env(call, default_action=default_action)
        if result.get("message_text"):
            action["message_text"] = result["message_text"]
        if "action" in result and result["action"] is not None:
            action["action"] = result["action"]
    return action
