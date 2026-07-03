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
