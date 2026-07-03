from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


SCENARIO_TIERS = ("core", "advanced", "procedural", "stress")


@dataclass(frozen=True)
class ScenarioPreset:
    name: str
    map_size: int
    num_agents: int
    fov_preset: str
    max_steps: int = 300
    split: str | None = "test"
    comm_mode: str = "tokens"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> dict[str, Any]:
        spec = {
            "map_size": self.map_size,
            "agents": self.num_agents,
            "fov_preset": self.fov_preset,
            "max_steps": self.max_steps,
            "split": self.split,
            "comm_mode": self.comm_mode,
        }
        spec.update(self.extra)
        return spec


@dataclass(frozen=True)
class ScenarioMetadata:
    name: str
    display_name: str
    tier: str
    domain: str
    summary: str
    communication_role: str
    private_information: str
    coordination_types: tuple[str, ...]
    generalization_axes: tuple[str, ...]
    benchmark_tags: tuple[str, ...]
    supported_comm_modes: tuple[str, ...] = ("tokens", "text")
    supported_tracks: tuple[str, ...] = ("dtde", "ctde")
    supports_rgb: bool = True
    supports_text: bool = True
    supports_tokens: bool = True
    oracle_available: bool = True
    scripted_available: bool = True
    solvability_check_available: bool = True
    default_presets: tuple[ScenarioPreset, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCENARIO_REGISTRY: dict[str, ScenarioMetadata] = {
    "pipeline_assembly": ScenarioMetadata(
        name="pipeline_assembly",
        display_name="Pipeline Assembly",
        tier="core",
        domain="task_planning",
        summary="Assemble a multi-stage artifact from partial blueprints, typed resources, dependencies, and sync steps.",
        communication_role="required",
        private_information="complementary_blueprints",
        coordination_types=("long_horizon_planning", "resource_delivery", "synchronized_action", "sequencing"),
        generalization_axes=("dependency_graph", "resource_locations", "station_layout", "map_scale"),
        benchmark_tags=("communication_required", "private_information", "long_horizon", "sequential_dependencies"),
        default_presets=(
            ScenarioPreset("core_8x8", map_size=8, num_agents=3, fov_preset="easy"),
            ScenarioPreset("scaled_16x16", map_size=16, num_agents=4, fov_preset="medium"),
        ),
    ),
    "energy_grid": ScenarioMetadata(
        name="energy_grid",
        display_name="Energy Grid",
        tier="core",
        domain="resource_sharing",
        summary="Maintain draining typed nodes when only assigned monitors know each node's urgency.",
        communication_role="required",
        private_information="private_node_monitoring",
        coordination_types=("resource_allocation", "routing", "time_pressure", "synchronized_recharge"),
        generalization_axes=("node_count", "resource_spawns", "energy_budget", "map_scale"),
        benchmark_tags=("communication_required", "private_information", "resource_allocation", "synchronized_recharge"),
        default_presets=(
            ScenarioPreset("core_8x8_easy", map_size=8, num_agents=3, fov_preset="easy", extra={"energy_preset": "easy", "energy_private_monitor": True}),
            ScenarioPreset("scaled_16x16_hard", map_size=16, num_agents=4, fov_preset="medium", extra={"energy_preset": "hard", "energy_private_monitor": True}),
        ),
    ),
    "signal_hunt": ScenarioMetadata(
        name="signal_hunt",
        display_name="Signal Hunt",
        tier="core",
        domain="cooperative_search",
        summary="Fuse private clues to identify a true target among decoys and coordinate a synchronized scan.",
        communication_role="required",
        private_information="complementary_clues",
        coordination_types=("information_fusion", "cooperative_search", "synchronized_action", "decoy_avoidance"),
        generalization_axes=("clue_templates", "target_constraints", "decoy_count", "map_scale", "room_topology"),
        benchmark_tags=("communication_required", "private_information", "synchronized_action"),
        default_presets=(
            ScenarioPreset("core_8x8", map_size=8, num_agents=2, fov_preset="easy"),
            ScenarioPreset("scaled_16x16", map_size=16, num_agents=4, fov_preset="medium"),
        ),
    ),
}


def get_scenario_metadata(name: str) -> ScenarioMetadata:
    try:
        return SCENARIO_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario metadata: {name}") from exc


def list_scenario_metadata(tier: str | None = None) -> list[ScenarioMetadata]:
    if tier is not None and tier not in SCENARIO_TIERS:
        raise ValueError(f"Unknown scenario tier: {tier}")
    scenarios = SCENARIO_REGISTRY.values()
    if tier is not None:
        scenarios = [meta for meta in scenarios if meta.tier == tier]
    return sorted(scenarios, key=lambda meta: (meta.tier, meta.name))


def scenario_names(tier: str | None = None) -> list[str]:
    return [meta.name for meta in list_scenario_metadata(tier=tier)]


def scenario_registry_as_dict(tier: str | None = None) -> dict[str, dict[str, Any]]:
    return {meta.name: meta.to_dict() for meta in list_scenario_metadata(tier=tier)}


def scenario_table_rows(tier: str | None = None) -> list[dict[str, str]]:
    rows = []
    for meta in list_scenario_metadata(tier=tier):
        rows.append(
            {
                "name": meta.name,
                "tier": meta.tier,
                "domain": meta.domain,
                "communication_role": meta.communication_role,
                "private_information": meta.private_information,
                "coordination_types": ", ".join(meta.coordination_types),
            }
        )
    return rows


def scenario_tags(name: str) -> tuple[str, ...]:
    return get_scenario_metadata(name).benchmark_tags


def validate_registered_scenarios(implemented_names: Iterable[str]) -> None:
    implemented = set(implemented_names)
    registered = set(SCENARIO_REGISTRY)
    missing_metadata = sorted(implemented - registered)
    missing_implementation = sorted(registered - implemented)
    if missing_metadata:
        raise ValueError(f"Implemented scenarios missing metadata: {missing_metadata}")
    if missing_implementation:
        raise ValueError(f"Registered scenarios missing implementation: {missing_implementation}")
    for meta in SCENARIO_REGISTRY.values():
        if meta.tier not in SCENARIO_TIERS:
            raise ValueError(f"Scenario {meta.name} has unknown tier: {meta.tier}")
        if not meta.coordination_types:
            raise ValueError(f"Scenario {meta.name} must declare at least one coordination type")
        if not meta.generalization_axes:
            raise ValueError(f"Scenario {meta.name} must declare at least one generalization axis")
