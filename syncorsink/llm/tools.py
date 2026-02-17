from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import json

from .policy import ACTION_NAMES


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


def parse_tool_calls(payload: Any) -> List[ToolCall]:
    """
    Parse tool calls from a provider-agnostic payload.
    Accepts a list of dicts: {"name": str, "arguments": {...}}
    """
    calls = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "name" in item:
                calls.append(ToolCall(name=item["name"], arguments=item.get("arguments", {})))
    return calls


def parse_openai_tool_calls(message: Dict[str, Any]) -> List[ToolCall]:
    """
    Parse OpenAI Chat Completions-style tool calls.
    Expects message like: {"tool_calls": [{"type":"function","function":{"name":"move","arguments":"{...}"}}]}
    """
    calls: List[ToolCall] = []
    if not isinstance(message, dict):
        return calls
    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        if call.get("type") != "function":
            continue
        fn = call.get("function", {})
        name = fn.get("name")
        args = fn.get("arguments", "{}")
        try:
            arguments = json.loads(args) if isinstance(args, str) else args
        except Exception:
            arguments = {}
        if name:
            calls.append(ToolCall(name=name, arguments=arguments))
    return calls


def openai_tools_schema() -> List[Dict[str, Any]]:
    """
    OpenAI function tool schema for SyncOrSink actions.
    Compatible with Chat Completions tools[].
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Move the agent in a cardinal direction.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right", "stay"],
                        }
                    },
                    "required": ["direction"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "interact",
                "description": "Interact with the tile you are on.",
                "strict": True,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "pickup",
                "description": "Pick up a resource from your current tile.",
                "strict": True,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "drop",
                "description": "Drop the item you are carrying on your current tile.",
                "strict": True,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "message",
                "description": "Send a message to other agents.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def tool_action_to_env(call: ToolCall, default_action: int = 4) -> dict:
    if call.name == "move":
        direction = str(call.arguments.get("direction", "stay")).lower()
        return {"action": ACTION_NAMES.get(direction, default_action), "message_text": None}
    if call.name == "interact":
        return {"action": ACTION_NAMES["interact"], "message_text": None}
    if call.name == "pickup":
        return {"action": ACTION_NAMES["pickup"], "message_text": None}
    if call.name == "drop":
        return {"action": ACTION_NAMES["drop"], "message_text": None}
    if call.name == "message":
        text = str(call.arguments.get("text", ""))
        return {"action": default_action, "message_text": text}
    return {"action": default_action, "message_text": None}
