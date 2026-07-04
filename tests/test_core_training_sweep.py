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
        "--seeds",
        "0",
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
    assert payload["overall"] == {"complete": 0, "dry_run": 4, "failed": 0, "total": 4}
    assert {run["algorithm"] for run in payload["runs"]} == {"mappo", "comm_mat"}
    assert all(run["scenario"] == "energy_grid" for run in payload["runs"])
    assert {run["seed"] for run in payload["runs"]} == {0, 1}
    assert all("--energy-preset" in run["command"] for run in payload["runs"])
    assert "--comm" in payload["runs"][0]["command"]
    assert all(run["wandb"]["status"] == "dry_run" for run in payload["runs"])
    assert len(payload["aggregate"]) == 2
    assert all(group["seeds"] == [0, 1] for group in payload["aggregate"])
    assert all(group["wandb_failed"] == 0 for group in payload["aggregate"])
    assert (tmp_path / "runs" / "dry" / "suite_summary.json").exists()


def test_core_training_sweep_default_cases_are_core_8x8():
    from examples.core_training_sweep import DEFAULT_CASES

    assert set(DEFAULT_CASES) == {"signal_hunt", "energy_grid", "pipeline_assembly"}
    assert all(case.map_size == 8 for case in DEFAULT_CASES.values())
    assert DEFAULT_CASES["energy_grid"].energy_preset == "easy"


def test_core_training_sweep_seed_alias_merges_with_seeds():
    from examples.core_training_sweep import parse_args

    args = parse_args(["--seed", "2", "--seeds", "0", "2", "1"])

    assert args.seeds == [0, 1, 2]


def test_core_training_sweep_parses_wandb_failures(tmp_path):
    from examples.core_training_sweep import _parse_wandb_record

    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("wandb init failed, continuing without wandb: wandb-core exited with code 1\n", encoding="utf-8")
    stderr.write_text("ERROR main: Serve() returned error\n", encoding="utf-8")

    record = _parse_wandb_record(stdout, stderr, requested=True, mode="offline")

    assert record["requested"] is True
    assert record["mode"] == "offline"
    assert record["status"] == "failed"
    assert record["error_lines"]
