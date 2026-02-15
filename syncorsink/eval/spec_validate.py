from __future__ import annotations

from typing import Any, Dict

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["scenario", "mode"],
    "properties": {
        "scenario": {"type": "string"},
        "mode": {"type": "string", "enum": ["marl", "llm"]},
        "split": {"type": ["string", "null"]},
        "episodes": {"type": "integer", "minimum": 1},
        "map_variant": {"type": "integer", "minimum": 0},
        "policy": {"type": "string"},
        "map_size": {"type": "integer"},
        "fov_preset": {"type": "string"},
        "comm_mode": {"type": "string"},
        "track": {"type": "string", "enum": ["dtde", "ctde"]},
    },
    "additionalProperties": True,
}


def validate_spec(data: Dict[str, Any]) -> None:
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=SCHEMA)
        return
    except ImportError:
        pass
    _manual_validate(data)


def _manual_validate(data: Dict[str, Any]) -> None:
    if "scenario" not in data or not isinstance(data["scenario"], str):
        raise ValueError("spec.scenario must be a string")
    if data.get("mode") not in ("marl", "llm"):
        raise ValueError("spec.mode must be 'marl' or 'llm'")
    if "episodes" in data and (not isinstance(data["episodes"], int) or data["episodes"] < 1):
        raise ValueError("spec.episodes must be int >= 1")
    if "map_variant" in data and (not isinstance(data["map_variant"], int) or data["map_variant"] < 0):
        raise ValueError("spec.map_variant must be int >= 0")
    if "map_size" in data and (not isinstance(data["map_size"], int) or data["map_size"] <= 0):
        raise ValueError("spec.map_size must be int > 0")
    if "comm_mode" in data and data["comm_mode"] not in ("tokens", "text"):
        raise ValueError("spec.comm_mode must be 'tokens' or 'text'")
    if "track" in data and data["track"] not in ("dtde", "ctde"):
        raise ValueError("spec.track must be 'dtde' or 'ctde'")
