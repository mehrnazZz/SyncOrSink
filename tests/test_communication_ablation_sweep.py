import json


def test_communication_ablation_sweep_smoke_writes_gap_artifact(tmp_path):
    from examples.communication_ablation_sweep import parse_args, run_sweep

    out = tmp_path / "communication_sweep.json"
    args = parse_args([
        "--scenarios",
        "signal_hunt",
        "--map-sizes",
        "8",
        "--episodes",
        "1",
        "--seed",
        "0",
        "--output-json",
        str(out),
    ])

    payload = run_sweep(args)
    saved = json.loads(out.read_text(encoding="utf-8"))

    assert saved == payload
    assert payload["suite"] == "communication_ablation_sweep"
    assert [row["condition"] for row in payload["rows"]] == ["comm_expert", "no_comm_local"]
    assert payload["gaps"][0]["scenario"] == "signal_hunt"
    assert payload["gaps"][0]["map_size"] == 8
    assert payload["gaps"][0]["comm_avg_tokens"] > 0.0
    assert payload["gaps"][0]["no_comm_avg_tokens"] == 0.0
    assert payload["overall"]["all_pass_threshold"] is True


def test_communication_ablation_sweep_include_32_deduplicates_map_sizes():
    from examples.communication_ablation_sweep import parse_args

    args = parse_args(["--map-sizes", "16", "8", "16", "--include-32"])

    assert args.map_sizes == [8, 16, 32]
