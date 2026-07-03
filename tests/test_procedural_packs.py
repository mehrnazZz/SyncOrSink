import json
import subprocess
import sys


def test_scenario_packs_validate_and_reference_registered_scenarios():
    from syncorsink.envs.procedural import SCENARIO_PACKS, validate_all_scenario_packs
    from syncorsink.envs.scenario_registry import scenario_names

    validate_all_scenario_packs()

    registered = set(scenario_names())
    assert set(SCENARIO_PACKS) == {"core", "core_ood"}
    for pack in SCENARIO_PACKS.values():
        assert pack.presets
        for preset in pack.presets:
            assert preset.scenario in registered
            assert preset.to_spec()["scenario"] == preset.scenario


def test_pack_benchmark_manifest_validates_and_loads(tmp_path):
    from syncorsink.envs.procedural import pack_benchmark_manifest
    from syncorsink.eval.benchmark_spec import load_benchmark

    manifest = pack_benchmark_manifest(
        ["core", "core_ood"],
        name="syncorsink_generated_test",
        version="test",
        description="Generated test manifest",
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_benchmark(str(path))

    assert loaded.name == "syncorsink_generated_test"
    assert loaded.version == "test"
    assert len(loaded.cases) == 6
    assert loaded.metadata["source_packs"] == ["core", "core_ood"]
    assert loaded.metadata["source_pack_tiers"] == ["core", "core_ood"]
    assert loaded.metadata["scenario_coverage"] == ["energy_grid", "pipeline_assembly", "signal_hunt"]
    assert loaded.metadata["case_count"] == 6


def test_scenario_pack_cli_json_and_manifest():
    json_result = subprocess.run(
        [sys.executable, "examples/list_scenario_packs.py", "--tier", "core", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(json_result.stdout)

    assert set(data) == {"core"}
    assert data["core"]["tier"] == "core"

    manifest_result = subprocess.run(
        [
            sys.executable,
            "examples/list_scenario_packs.py",
            "--benchmark",
            "core",
            "core_ood",
            "--name",
            "syncorsink_cli_test",
            "--version",
            "test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(manifest_result.stdout)

    assert manifest["name"] == "syncorsink_cli_test"
    assert manifest["metadata"]["source_packs"] == ["core", "core_ood"]
    assert len(manifest["cases"]) == 6

    manifest_with_note = subprocess.check_output(
        [
            "python",
            "examples/list_scenario_packs.py",
            "--benchmark",
            "core",
            "--name",
            "syncorsink_cli_note_test",
            "--version",
            "test",
            "--compatibility-note",
            "compatibility note",
        ],
        text=True,
    )
    note_manifest = json.loads(manifest_with_note)
    assert note_manifest["metadata"]["compatibility_note"] == "compatibility note"


def test_syncorsink_v0_2_is_generated_from_core_packs():
    from syncorsink.envs.procedural import pack_benchmark_manifest
    from syncorsink.eval.benchmark_spec import load_benchmark

    compatibility_note = (
        "Pack-generated successor to syncorsink_v0_1; covers the same core and scaled scenario surface "
        "with pack-derived case names so future packs can extend v0.2 without mutating v0.1."
    )
    expected = pack_benchmark_manifest(
        ["core", "core_ood"],
        name="syncorsink_v0_2",
        version="0.2.0",
        description="Generated from scenario packs: core, core_ood",
        extra_metadata={"compatibility_note": compatibility_note},
    )
    with open("benchmarks/syncorsink_v0_2.json", "r", encoding="utf-8") as f:
        actual = json.load(f)
    loaded = load_benchmark("benchmarks/syncorsink_v0_2.json")

    assert actual == expected
    assert loaded.name == "syncorsink_v0_2"
    assert loaded.version == "0.2.0"
    assert loaded.metadata["axes_covered"] == [
        "agent_count",
        "fov_preset",
        "information_structure",
        "map_size",
        "scenario",
    ]


def test_syncorsink_v0_1_case_names_remain_frozen():
    from syncorsink.eval.benchmark_spec import load_benchmark

    bench = load_benchmark("benchmarks/syncorsink_v0_1.json")

    assert [case.name for case in bench.cases] == [
        "signal_hunt_8x8_private_clues",
        "signal_hunt_16x16_scaled_search",
        "energy_grid_8x8_symmetric_info",
        "energy_grid_16x16_hard_resource_sharing",
        "pipeline_8x8_private_blueprints",
        "pipeline_16x16_scaled_dependencies",
    ]
