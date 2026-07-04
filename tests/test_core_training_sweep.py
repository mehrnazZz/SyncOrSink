import json


def test_core_training_sweep_dry_run_writes_manifest(tmp_path):
    from examples.core_training_sweep import parse_args, run_suite

    output_json = tmp_path / "summary.json"
    args = parse_args([
        "--algorithms",
        "mappo",
        "comm_mat",
        "--scenarios",
        "energy_grid",
        "--updates",
        "1",
        "--rollout-steps",
        "8",
        "--epochs",
        "1",
        "--minibatch",
        "8",
        "--eval-every",
        "1",
        "--eval-episodes",
        "1",
        "--output-dir",
        str(tmp_path / "runs"),
        "--output-json",
        str(output_json),
        "--run-name",
        "dry",
        "--dry-run",
    ])

    payload = run_suite(args)
    saved = json.loads(output_json.read_text(encoding="utf-8"))

    assert saved == payload
    assert payload["overall"] == {"complete": 0, "dry_run": 2, "failed": 0, "total": 2}
    assert {run["algorithm"] for run in payload["runs"]} == {"mappo", "comm_mat"}
    assert all(run["scenario"] == "energy_grid" for run in payload["runs"])
    assert all("--energy-preset" in run["command"] for run in payload["runs"])
    assert "--comm" in payload["runs"][0]["command"]
    assert (tmp_path / "runs" / "dry" / "suite_summary.json").exists()


def test_core_training_sweep_default_cases_are_core_8x8():
    from examples.core_training_sweep import DEFAULT_CASES

    assert set(DEFAULT_CASES) == {"signal_hunt", "energy_grid", "pipeline_assembly"}
    assert all(case.map_size == 8 for case in DEFAULT_CASES.values())
    assert DEFAULT_CASES["energy_grid"].energy_preset == "easy"
