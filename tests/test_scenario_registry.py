import json
import subprocess
import sys

import pytest


def test_scenario_registry_matches_implemented_scenarios():
    from syncorsink.envs.scenario_registry import (
        SCENARIO_REGISTRY,
        get_scenario_metadata,
        scenario_names,
        validate_registered_scenarios,
    )
    from syncorsink.envs.scenarios import SCENARIOS

    validate_registered_scenarios(SCENARIOS.keys())

    assert set(scenario_names()) == set(SCENARIOS)
    assert all(meta.tier == "core" for meta in SCENARIO_REGISTRY.values())
    assert get_scenario_metadata("signal_hunt").communication_role == "required"
    assert get_scenario_metadata("energy_grid").communication_role == "required"


def test_scenario_metadata_is_json_serializable():
    from syncorsink.envs.scenario_registry import scenario_registry_as_dict

    data = scenario_registry_as_dict()
    encoded = json.dumps(data, sort_keys=True)

    assert "pipeline_assembly" in encoded
    assert data["signal_hunt"]["tier"] == "core"
    assert data["pipeline_assembly"]["default_presets"][0]["name"] == "core_8x8"


def test_benchmark_cases_reference_registered_scenarios_and_tags():
    from syncorsink.envs.scenario_registry import get_scenario_metadata
    from syncorsink.eval.benchmark_spec import load_benchmark

    bench = load_benchmark("benchmarks/syncorsink_v0_1.json")

    for case in bench.cases:
        meta = get_scenario_metadata(case.spec["scenario"])
        assert meta.tier == "core"
        legacy_energy_control = case.spec["scenario"] == "energy_grid" and case.spec.get("energy_private_monitor") is False
        if legacy_energy_control:
            assert "communication_control" in case.tags
        else:
            assert set(case.tags) & set(meta.benchmark_tags)


def test_spec_validation_rejects_unregistered_scenario():
    from syncorsink.eval.spec_validate import validate_spec

    with pytest.raises(Exception):
        validate_spec({"scenario": "disaster_response", "mode": "marl"})


def test_list_scenarios_cli_json():
    result = subprocess.run(
        [sys.executable, "examples/list_scenarios.py", "--tier", "core", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)

    assert set(data) == {"energy_grid", "pipeline_assembly", "signal_hunt"}
