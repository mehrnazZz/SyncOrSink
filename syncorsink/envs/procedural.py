from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from syncorsink.eval.spec_validate import validate_spec

from .scenario_registry import get_scenario_metadata, scenario_names


PACK_TIERS = ("core", "core_ood", "advanced", "procedural", "stress")


@dataclass(frozen=True)
class ProceduralAxis:
    name: str
    description: str
    values: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProceduralPreset:
    name: str
    scenario: str
    split: str
    map_size: int
    num_agents: int
    fov_preset: str
    max_steps: int = 300
    comm_mode: str = "tokens"
    track: str = "dtde"
    episodes: int = 32
    tags: tuple[str, ...] = ()
    weight: float = 1.0
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_spec(self, *, mode: str = "marl") -> dict[str, Any]:
        spec = {
            "scenario": self.scenario,
            "mode": mode,
            "track": self.track,
            "split": self.split,
            "episodes": self.episodes,
            "map_size": self.map_size,
            "agents": self.num_agents,
            "fov_preset": self.fov_preset,
            "max_steps": self.max_steps,
            "comm_mode": self.comm_mode,
        }
        spec.update(self.overrides)
        return spec

    def to_case(self, *, mode: str = "marl") -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": float(self.weight),
            "tags": list(self.tags),
            "spec": self.to_spec(mode=mode),
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["spec"] = self.to_spec()
        return data


@dataclass(frozen=True)
class ScenarioPack:
    name: str
    tier: str
    description: str
    presets: tuple[ProceduralPreset, ...]
    axes: tuple[ProceduralAxis, ...] = ()
    version: str = "0.1"

    def to_benchmark(
        self,
        *,
        name: str | None = None,
        version: str | None = None,
        description: str | None = None,
        mode: str = "marl",
    ) -> dict[str, Any]:
        return {
            "name": name or f"syncorsink_{self.name}",
            "version": version or self.version,
            "description": description or self.description,
            "metadata": {
                "source_pack": self.name,
                "pack_tier": self.tier,
                "primary_metric": "mean_success_rate",
                "official_score": "100 * weighted_mean(success_rate)",
            },
            "cases": [preset.to_case(mode=mode) for preset in self.presets],
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCENARIO_PACKS: dict[str, ScenarioPack] = {
    "core": ScenarioPack(
        name="core",
        tier="core",
        version="0.1",
        description="Stable compact diagnostic presets for the current core scenarios.",
        axes=(
            ProceduralAxis("scenario", "Core diagnostic scenario family.", tuple(scenario_names(tier="core"))),
            ProceduralAxis("map_size", "Canonical small-map benchmark scale.", (8,)),
            ProceduralAxis("information_structure", "Whether communication is required or a control variable.", ("required", "control")),
        ),
        presets=(
            ProceduralPreset(
                name="signal_hunt_core_8x8",
                scenario="signal_hunt",
                split="test",
                map_size=8,
                num_agents=2,
                fov_preset="easy",
                tags=("communication_required", "private_information", "synchronized_action", "in_distribution"),
            ),
            ProceduralPreset(
                name="energy_grid_core_8x8_easy",
                scenario="energy_grid",
                split="test",
                map_size=8,
                num_agents=3,
                fov_preset="easy",
                tags=("communication_control", "symmetric_information", "resource_allocation", "in_distribution"),
                overrides={"energy_preset": "easy"},
            ),
            ProceduralPreset(
                name="pipeline_core_8x8",
                scenario="pipeline_assembly",
                split="test",
                map_size=8,
                num_agents=3,
                fov_preset="easy",
                tags=("communication_required", "private_information", "long_horizon", "sequential_dependencies", "in_distribution"),
            ),
        ),
    ),
    "core_ood": ScenarioPack(
        name="core_ood",
        tier="core_ood",
        version="0.1",
        description="Scaled core presets for held-out map and coordination generalization.",
        axes=(
            ProceduralAxis("map_size", "Held-out spatial scale.", (16,)),
            ProceduralAxis("fov_preset", "More constrained partial observability than core 8x8.", ("medium",)),
            ProceduralAxis("agent_count", "Larger teams for scaled coordination.", (4,)),
        ),
        presets=(
            ProceduralPreset(
                name="signal_hunt_ood_16x16",
                scenario="signal_hunt",
                split="test",
                map_size=16,
                num_agents=4,
                fov_preset="medium",
                tags=("communication_required", "private_information", "scale_generalization", "ood"),
            ),
            ProceduralPreset(
                name="energy_grid_ood_16x16_hard",
                scenario="energy_grid",
                split="test",
                map_size=16,
                num_agents=4,
                fov_preset="medium",
                tags=("communication_control", "symmetric_information", "scale_generalization", "resource_allocation", "ood"),
                overrides={"energy_preset": "hard"},
            ),
            ProceduralPreset(
                name="pipeline_ood_16x16",
                scenario="pipeline_assembly",
                split="test",
                map_size=16,
                num_agents=4,
                fov_preset="medium",
                tags=("communication_required", "private_information", "long_horizon", "sequential_dependencies", "scale_generalization", "ood"),
            ),
        ),
    ),
}


def get_scenario_pack(name: str) -> ScenarioPack:
    try:
        return SCENARIO_PACKS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario pack: {name}") from exc


def list_scenario_packs(tier: str | None = None) -> list[ScenarioPack]:
    if tier is not None and tier not in PACK_TIERS:
        raise ValueError(f"Unknown pack tier: {tier}")
    packs = SCENARIO_PACKS.values()
    if tier is not None:
        packs = [pack for pack in packs if pack.tier == tier]
    return sorted(packs, key=lambda pack: (pack.tier, pack.name))


def scenario_pack_names(tier: str | None = None) -> list[str]:
    return [pack.name for pack in list_scenario_packs(tier=tier)]


def scenario_packs_as_dict(tier: str | None = None) -> dict[str, dict[str, Any]]:
    return {pack.name: pack.to_dict() for pack in list_scenario_packs(tier=tier)}


def scenario_pack_table_rows(tier: str | None = None) -> list[dict[str, str]]:
    rows = []
    for pack in list_scenario_packs(tier=tier):
        scenarios = sorted({preset.scenario for preset in pack.presets})
        rows.append(
            {
                "name": pack.name,
                "tier": pack.tier,
                "version": pack.version,
                "presets": str(len(pack.presets)),
                "scenarios": ", ".join(scenarios),
                "axes": ", ".join(axis.name for axis in pack.axes),
            }
        )
    return rows


def pack_benchmark_manifest(
    pack_names: Iterable[str],
    *,
    name: str,
    version: str,
    description: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    source_packs: list[str] = []
    source_pack_tiers: list[str] = []
    scenario_coverage: set[str] = set()
    axes_covered: set[str] = set()
    for pack_name in pack_names:
        pack = get_scenario_pack(pack_name)
        source_packs.append(pack.name)
        source_pack_tiers.append(pack.tier)
        axes_covered.update(axis.name for axis in pack.axes)
        scenario_coverage.update(preset.scenario for preset in pack.presets)
        cases.extend(preset.to_case() for preset in pack.presets)
    metadata = {
        "source_packs": source_packs,
        "source_pack_tiers": sorted(set(source_pack_tiers)),
        "scenario_coverage": sorted(scenario_coverage),
        "axes_covered": sorted(axes_covered),
        "case_count": len(cases),
        "primary_metric": "mean_success_rate",
        "official_score": "100 * weighted_mean(success_rate)",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    manifest = {
        "name": name,
        "version": version,
        "description": description,
        "metadata": metadata,
        "cases": cases,
    }
    validate_pack_benchmark_manifest(manifest)
    return manifest


def validate_scenario_pack(pack: ScenarioPack) -> None:
    if pack.tier not in PACK_TIERS:
        raise ValueError(f"Unknown pack tier for {pack.name}: {pack.tier}")
    if not pack.presets:
        raise ValueError(f"Scenario pack {pack.name} must contain at least one preset")
    seen = set()
    for preset in pack.presets:
        if preset.name in seen:
            raise ValueError(f"Duplicate preset in pack {pack.name}: {preset.name}")
        seen.add(preset.name)
        meta = get_scenario_metadata(preset.scenario)
        if preset.comm_mode not in meta.supported_comm_modes:
            raise ValueError(f"Preset {preset.name} uses unsupported comm mode: {preset.comm_mode}")
        if preset.track not in meta.supported_tracks:
            raise ValueError(f"Preset {preset.name} uses unsupported track: {preset.track}")
        if preset.map_size <= 0 or preset.num_agents <= 0 or preset.max_steps <= 0:
            raise ValueError(f"Preset {preset.name} has invalid numeric settings")
        validate_spec(preset.to_spec())


def validate_all_scenario_packs() -> None:
    for pack in SCENARIO_PACKS.values():
        validate_scenario_pack(pack)


def validate_pack_benchmark_manifest(manifest: dict[str, Any]) -> None:
    names = set()
    for case in manifest.get("cases", []):
        name = case.get("name")
        if name in names:
            raise ValueError(f"Duplicate benchmark case generated from packs: {name}")
        names.add(name)
        validate_spec(case["spec"])
