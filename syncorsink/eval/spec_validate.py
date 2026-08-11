from __future__ import annotations

from typing import Any, Dict

from syncorsink.envs.scenario_registry import scenario_names

_SCENARIO_NAMES = scenario_names()

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["scenario", "mode"],
    "properties": {
        "scenario": {"type": "string", "enum": _SCENARIO_NAMES},
        "mode": {"type": "string", "enum": ["marl", "llm"]},
        "split": {"type": ["string", "null"]},
        "episodes": {"type": "integer", "minimum": 1},
        "map_variant": {"type": "integer", "minimum": 0},
        "policy": {"type": "string"},
        "policy_entrypoint": {"type": ["string", "null"]},
        "policy_kwargs": {"type": "object"},
        "map_size": {"type": "integer", "minimum": 1},
        "agents": {"type": "integer", "minimum": 1},
        "num_agents": {"type": "integer", "minimum": 1},
        "max_steps": {"type": "integer", "minimum": 1},
        "fov_preset": {"type": "string"},
        "comm_mode": {"type": "string"},
        "track": {"type": "string", "enum": ["dtde", "ctde"]},
        "pipeline_stage_count": {"type": ["integer", "null"], "minimum": 1},
        "pipeline_required_per_stage_min": {"type": "integer", "minimum": 1},
        "pipeline_required_per_stage_max": {"type": "integer", "minimum": 1},
        "pipeline_sync_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "pipeline_dependency_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "pipeline_wrong_delivery_penalty": {"type": "number", "minimum": 0.0},
        "energy_preset": {"type": "string"},
        "energy_private_monitor": {"type": "boolean"},
        "policy_checkpoint": {"type": ["string", "null"]},
        "comm_mat_deterministic": {"type": "boolean"},
        "comm_mat_send_threshold": {"type": "number"},
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
    if data["scenario"] not in _SCENARIO_NAMES:
        raise ValueError(f"spec.scenario must be one of {_SCENARIO_NAMES}")
    if data.get("mode") not in ("marl", "llm"):
        raise ValueError("spec.mode must be 'marl' or 'llm'")
    if "episodes" in data and (not isinstance(data["episodes"], int) or data["episodes"] < 1):
        raise ValueError("spec.episodes must be int >= 1")
    if "map_variant" in data and (not isinstance(data["map_variant"], int) or data["map_variant"] < 0):
        raise ValueError("spec.map_variant must be int >= 0")
    if "map_size" in data and (not isinstance(data["map_size"], int) or data["map_size"] <= 0):
        raise ValueError("spec.map_size must be int > 0")
    if "agents" in data and (not isinstance(data["agents"], int) or data["agents"] <= 0):
        raise ValueError("spec.agents must be int > 0")
    if "num_agents" in data and (not isinstance(data["num_agents"], int) or data["num_agents"] <= 0):
        raise ValueError("spec.num_agents must be int > 0")
    if "max_steps" in data and (not isinstance(data["max_steps"], int) or data["max_steps"] <= 0):
        raise ValueError("spec.max_steps must be int > 0")
    if "comm_mode" in data and data["comm_mode"] not in ("tokens", "text"):
        raise ValueError("spec.comm_mode must be 'tokens' or 'text'")
    if "track" in data and data["track"] not in ("dtde", "ctde"):
        raise ValueError("spec.track must be 'dtde' or 'ctde'")
    if (
        "pipeline_stage_count" in data
        and data["pipeline_stage_count"] is not None
        and (not isinstance(data["pipeline_stage_count"], int) or data["pipeline_stage_count"] <= 0)
    ):
        raise ValueError("spec.pipeline_stage_count must be null or int > 0")
    for key in ("pipeline_required_per_stage_min", "pipeline_required_per_stage_max"):
        if key in data and (not isinstance(data[key], int) or data[key] <= 0):
            raise ValueError(f"spec.{key} must be int > 0")
    for key in ("pipeline_sync_probability", "pipeline_dependency_probability"):
        if key in data and (not isinstance(data[key], (int, float)) or not 0.0 <= float(data[key]) <= 1.0):
            raise ValueError(f"spec.{key} must be a number in [0, 1]")
    if (
        "pipeline_wrong_delivery_penalty" in data
        and (
            not isinstance(data["pipeline_wrong_delivery_penalty"], (int, float))
            or float(data["pipeline_wrong_delivery_penalty"]) < 0.0
        )
    ):
        raise ValueError("spec.pipeline_wrong_delivery_penalty must be a number >= 0")
    if "energy_private_monitor" in data and not isinstance(data["energy_private_monitor"], bool):
        raise ValueError("spec.energy_private_monitor must be boolean")
    if "policy_entrypoint" in data and data["policy_entrypoint"] is not None and not isinstance(data["policy_entrypoint"], str):
        raise ValueError("spec.policy_entrypoint must be a string or null")
    if "policy_kwargs" in data and not isinstance(data["policy_kwargs"], dict):
        raise ValueError("spec.policy_kwargs must be an object")
