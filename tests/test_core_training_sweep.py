import json

import torch


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


def test_core_training_sweep_comm_curriculum_profile_adds_training_aids(tmp_path):
    from examples.core_training_sweep import parse_args, run_suite

    args = parse_args([
        "--algorithms",
        "mappo",
        "comm_mat",
        "tarmac",
        "--scenarios",
        "signal_hunt",
        "energy_grid",
        "pipeline_assembly",
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
        "--learning-profile",
        "comm_curriculum",
        "--mappo-backbone",
        "transformer",
        "--mappo-shared-actor",
        "--mappo-obs-exploration-memory",
        "--mappo-obs-exploration-age",
        "--mappo-eval-action-mode",
        "sample",
        "--mappo-eval-send-mode",
        "sample",
        "--mappo-eval-send-threshold",
        "0.25",
        "--output-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "curriculum",
        "--dry-run",
    ])

    payload = run_suite(args)
    commands = {
        (run["algorithm"], run["scenario"]): run["command"]
        for run in payload["runs"]
    }

    assert payload["config"]["learning_profile"] == "comm_curriculum"
    assert payload["config"]["mappo_backbone"] == "transformer"
    assert payload["config"]["mappo_shared_actor"] is True
    assert payload["config"]["mappo_obs_exploration_memory"] is True
    assert payload["config"]["mappo_obs_exploration_age"] is True
    assert payload["config"]["mappo_eval_action_mode"] == "sample"
    assert payload["config"]["mappo_eval_send_mode"] == "sample"
    assert payload["config"]["mappo_eval_send_threshold"] == 0.25
    assert "--signal-shaping" in commands[("mappo", "signal_hunt")]
    assert "--energy-shaping" in commands[("comm_mat", "energy_grid")]
    assert "--pipeline-shaping" in commands[("tarmac", "pipeline_assembly")]
    assert "--comm-send-target" in commands[("mappo", "signal_hunt")]
    assert "--comm-send-target" in commands[("comm_mat", "pipeline_assembly")]
    assert "--comm-send-target" not in commands[("tarmac", "signal_hunt")]
    assert "--attn-entropy-coeff" in commands[("tarmac", "signal_hunt")]
    assert commands[("mappo", "signal_hunt")].count("--comm-cost") == 1
    assert "--backbone" in commands[("mappo", "signal_hunt")]
    assert "--shared-actor" in commands[("mappo", "signal_hunt")]
    assert "--obs-exploration-memory" in commands[("mappo", "signal_hunt")]
    assert "--obs-exploration-age" in commands[("mappo", "signal_hunt")]
    assert "--eval-action-mode" in commands[("mappo", "signal_hunt")]
    assert "--eval-send-mode" in commands[("mappo", "signal_hunt")]
    assert "--eval-send-threshold" in commands[("mappo", "signal_hunt")]
    assert "--backbone" not in commands[("comm_mat", "signal_hunt")]
    assert "--obs-exploration-memory" not in commands[("comm_mat", "signal_hunt")]
    assert "--eval-action-mode" not in commands[("tarmac", "signal_hunt")]


def test_core_training_sweep_recurrent_defaults_are_guarded_and_scenario_aware(tmp_path):
    from examples.core_training_sweep import parse_args, run_suite

    args = parse_args([
        "--algorithms",
        "recurrent_bc_rl",
        "--scenarios",
        "signal_hunt",
        "energy_grid",
        "pipeline_assembly",
        "--updates",
        "1",
        "--rollout-steps",
        "8",
        "--eval-every",
        "1",
        "--eval-episodes",
        "1",
        "--seeds",
        "0",
        "--output-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "recurrent-defaults",
        "--dry-run",
    ])

    payload = run_suite(args)
    commands = {run["scenario"]: run["command"] for run in payload["runs"]}

    assert payload["config"]["recurrent_oracle"] == "auto"
    assert payload["config"]["recurrent_resolved_oracles"] == {
        "energy_grid": "planner_comm",
        "pipeline_assembly": "planner_comm",
        "signal_hunt": "signal_hint_comm",
    }
    assert payload["config"]["recurrent_ppo_profile"] == "guarded"
    assert payload["config"]["recurrent_rl_lr"] == 1e-5
    assert payload["config"]["recurrent_clip"] == 0.1
    assert payload["config"]["recurrent_entropy_coeff"] == 0.0
    assert payload["config"]["recurrent_bc_action_class_balance"] is True
    assert payload["config"]["recurrent_bc_action_class_balance_max_weight"] == 5.0
    assert payload["config"]["recurrent_bc_event_action_weight"] == 2.0
    assert payload["config"]["recurrent_bc_event_action_events"] == (
        "delivered,sync_complete,recharged,joint_target_scan"
    )
    assert payload["config"]["recurrent_pipeline_stage_count"] is None
    assert payload["config"]["recurrent_pipeline_required_per_stage_min"] == 1
    assert payload["config"]["recurrent_pipeline_required_per_stage_max"] == 2
    assert payload["config"]["recurrent_pipeline_sync_probability"] == 0.5
    assert payload["config"]["recurrent_pipeline_dependency_probability"] == 0.7
    assert payload["config"]["recurrent_bc_kl_coeff"] == 2.0
    assert payload["config"]["recurrent_bc_comm_kl_coeff"] == 2.0
    assert payload["config"]["recurrent_dagger_failed_effective_ratio_cap"] == 0.25
    assert payload["config"]["recurrent_dagger_oracle_action_rollin_rate"] == 0.25
    assert payload["config"]["recurrent_dagger_oracle_message_rollin_rate"] == 0.0
    assert payload["config"]["recurrent_rl_balanced_rollouts"] is True
    assert payload["config"]["recurrent_rl_rollout_eval_decoding"] is True

    assert commands["signal_hunt"][commands["signal_hunt"].index("--oracle") + 1] == "signal_hint_comm"
    assert commands["energy_grid"][commands["energy_grid"].index("--oracle") + 1] == "planner_comm"
    assert (
        commands["pipeline_assembly"][commands["pipeline_assembly"].index("--oracle") + 1]
        == "planner_comm"
    )
    assert "--bc-action-class-balance" in commands["energy_grid"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-action-class-balance-max-weight") + 1
    ] == "5.0"
    assert commands["energy_grid"][
        commands["energy_grid"].index("--bc-event-action-weight") + 1
    ] == "2.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-event-action-events") + 1
    ] == "delivered,sync_complete,recharged,joint_target_scan"
    assert "--pipeline-stage-count" not in commands["pipeline_assembly"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--pipeline-required-per-stage-min") + 1
    ] == "1"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--pipeline-required-per-stage-max") + 1
    ] == "2"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--pipeline-sync-probability") + 1
    ] == "0.5"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--pipeline-dependency-probability") + 1
    ] == "0.7"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-failed-effective-ratio-cap") + 1
    ] == "0.25"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-oracle-action-rollin-rate") + 1
    ] == "0.25"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-oracle-message-rollin-rate") + 1
    ] == "0.0"
    assert "--rl-balanced-rollouts" in commands["energy_grid"]
    assert "--rl-rollout-eval-decoding" in commands["pipeline_assembly"]


def test_core_training_sweep_recurrent_dry_run_command_and_eval_parser(tmp_path):
    from examples.core_training_sweep import (
        _parse_eval_metrics,
        _parse_recurrent_checkpoint_evals,
        parse_args,
        run_suite,
    )

    args = parse_args([
        "--algorithms",
        "recurrent_bc_rl",
        "--scenarios",
        "signal_hunt",
        "--updates",
        "7",
        "--rollout-steps",
        "16",
        "--eval-every",
        "2",
        "--eval-episodes",
        "3",
        "--seeds",
        "0",
        "--learning-profile",
        "comm_curriculum",
        "--recurrent-demo-episodes",
        "4",
        "--recurrent-bc-epochs",
        "2",
        "--recurrent-bc-action-class-balance-max-weight",
        "4.0",
        "--recurrent-bc-event-action-weight",
        "6.0",
        "--recurrent-bc-event-action-events",
        "delivered,sync_complete",
        "--recurrent-pipeline-stage-count",
        "2",
        "--recurrent-pipeline-required-per-stage-min",
        "1",
        "--recurrent-pipeline-required-per-stage-max",
        "1",
        "--recurrent-pipeline-sync-probability",
        "0.0",
        "--recurrent-pipeline-dependency-probability",
        "0.0",
        "--recurrent-bc-calibrate-send-threshold",
        "--recurrent-bc-send-threshold-target-rate",
        "0.15",
        "--recurrent-bc-comm-send-rate-penalty-weight",
        "0.25",
        "--recurrent-bc-comm-send-rate-target",
        "0.15",
        "--recurrent-dagger-rounds",
        "1",
        "--recurrent-dagger-episodes",
        "5",
        "--recurrent-dagger-failed-effective-ratio-cap",
        "0.5",
        "--recurrent-dagger-oracle-action-rollin-rate",
        "0.4",
        "--recurrent-dagger-oracle-message-rollin-rate",
        "0.3",
        "--recurrent-rl-updates",
        "0",
        "--recurrent-rl-lr",
        "1e-5",
        "--recurrent-clip",
        "0.1",
        "--recurrent-entropy-coeff",
        "0.0",
        "--recurrent-max-grad-norm",
        "0.25",
        "--recurrent-bc-kl-coeff",
        "1.0",
        "--recurrent-bc-comm-kl-coeff",
        "1.5",
        "--recurrent-rl-balanced-rollouts",
        "--recurrent-rl-rollout-eval-decoding",
        "--no-recurrent-rl-restore-best",
        "--recurrent-train-map-sizes",
        "8,16",
        "--recurrent-map-max-steps",
        "8:60,16:120",
        "--recurrent-eval-map-sizes",
        "8,16",
        "--recurrent-eval-seed-list",
        "8:3000+16:13000",
        "--recurrent-dagger-seed-list",
        "8:3000+16:13000",
        "--recurrent-init-template",
        "logs/{scenario}/seed{seed}/{algorithm}_{map_size}x.pt",
        "--recurrent-init-for-dagger",
        "--recurrent-calibrate-send-threshold",
        "--output-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "recurrent",
        "--dry-run",
    ])

    payload = run_suite(args)
    command = payload["runs"][0]["command"]

    assert payload["overall"] == {"complete": 0, "dry_run": 1, "failed": 0, "total": 1}
    assert payload["config"]["recurrent_oracle"] == "auto"
    assert payload["config"]["recurrent_resolved_oracles"] == {"signal_hunt": "signal_hint_comm"}
    assert payload["config"]["recurrent_ppo_profile"] == "guarded"
    assert payload["config"]["recurrent_signal_preset"] == "specialist"
    assert payload["config"]["recurrent_bc_action_class_balance"] is True
    assert payload["config"]["recurrent_bc_action_class_balance_max_weight"] == 4.0
    assert payload["config"]["recurrent_bc_event_action_weight"] == 6.0
    assert payload["config"]["recurrent_bc_event_action_events"] == "delivered,sync_complete"
    assert payload["config"]["recurrent_pipeline_stage_count"] == 2
    assert payload["config"]["recurrent_pipeline_required_per_stage_min"] == 1
    assert payload["config"]["recurrent_pipeline_required_per_stage_max"] == 1
    assert payload["config"]["recurrent_pipeline_sync_probability"] == 0.0
    assert payload["config"]["recurrent_pipeline_dependency_probability"] == 0.0
    assert payload["config"]["recurrent_bc_calibrate_send_threshold"] is True
    assert payload["config"]["recurrent_bc_send_threshold_target_rate"] == 0.15
    assert payload["config"]["recurrent_bc_comm_send_rate_penalty_weight"] == 0.25
    assert payload["config"]["recurrent_bc_comm_send_rate_target"] == 0.15
    assert payload["config"]["recurrent_dagger_failed_effective_ratio_cap"] == 0.5
    assert payload["config"]["recurrent_dagger_oracle_action_rollin_rate"] == 0.4
    assert payload["config"]["recurrent_dagger_oracle_message_rollin_rate"] == 0.3
    assert "--updates" not in command
    assert "--epochs" not in command
    assert "--save-every" not in command
    assert "--rl-updates" in command
    assert command[command.index("--rl-updates") + 1] == "0"
    assert command[command.index("--rl-lr") + 1] == "1e-05"
    assert command[command.index("--clip") + 1] == "0.1"
    assert command[command.index("--entropy-coeff") + 1] == "0.0"
    assert command[command.index("--max-grad-norm") + 1] == "0.25"
    assert command[command.index("--bc-kl-coeff") + 1] == "1.0"
    assert command[command.index("--bc-comm-kl-coeff") + 1] == "1.5"
    assert "--rl-balanced-rollouts" in command
    assert "--rl-rollout-eval-decoding" in command
    assert "--no-rl-restore-best" in command
    assert "--demo-episodes" in command
    assert command[command.index("--demo-episodes") + 1] == "4"
    assert "--bc-action-class-balance" in command
    assert command[command.index("--bc-action-class-balance-max-weight") + 1] == "4.0"
    assert command[command.index("--bc-event-action-weight") + 1] == "6.0"
    assert command[command.index("--bc-event-action-events") + 1] == "delivered,sync_complete"
    assert command[command.index("--pipeline-stage-count") + 1] == "2"
    assert command[command.index("--pipeline-required-per-stage-min") + 1] == "1"
    assert command[command.index("--pipeline-required-per-stage-max") + 1] == "1"
    assert command[command.index("--pipeline-sync-probability") + 1] == "0.0"
    assert command[command.index("--pipeline-dependency-probability") + 1] == "0.0"
    assert "--bc-calibrate-send-threshold" in command
    assert command[command.index("--bc-send-threshold-target-rate") + 1] == "0.15"
    assert command[command.index("--bc-comm-send-rate-penalty-weight") + 1] == "0.25"
    assert command[command.index("--bc-comm-send-rate-target") + 1] == "0.15"
    assert "--dagger-rounds" in command
    assert command[command.index("--dagger-rounds") + 1] == "1"
    assert command[command.index("--dagger-failed-effective-ratio-cap") + 1] == "0.5"
    assert command[command.index("--dagger-oracle-action-rollin-rate") + 1] == "0.4"
    assert command[command.index("--dagger-oracle-message-rollin-rate") + 1] == "0.3"
    assert "--train-map-sizes" in command
    assert "--eval-map-sizes" in command
    assert "--eval-seed-list" in command
    assert "--dagger-seed-list" in command
    assert "--recurrent-init" in command
    assert command[command.index("--recurrent-init") + 1] == (
        "logs/signal_hunt/seed0/recurrent_bc_rl_8x.pt"
    )
    assert "--recurrent-init-for-dagger" in command
    assert "--obs-exploration-memory" in command
    assert "--obs-signal-scan-state" in command
    assert "--eval-signal-exact-target-navigation-assist" in command
    assert "--signal-shaping" in command
    assert "--comm-send-target" not in command

    stdout = tmp_path / "stdout.log"
    stdout.write_text(
        'noise\n{"eval_recurrent_bc": {"success_rate": 0.5, "avg_return": 1.25, "avg_steps": 42.0}}\n',
        encoding="utf-8",
    )
    assert _parse_eval_metrics("recurrent_bc_rl", stdout, [], checkpoint_path=None) == {
        "success_rate": 0.5,
        "return": 1.25,
        "steps": 42.0,
    }

    rl_stdout = tmp_path / "rl_stdout.log"
    rl_stdout.write_text(
        '{"recurrent_rl_eval": {"success_rate": 0.75, "avg_return": 3.5, "avg_steps": 12.0}}\n',
        encoding="utf-8",
    )
    assert _parse_eval_metrics("recurrent_bc_rl", rl_stdout, [], checkpoint_path=None) == {
        "success_rate": 0.75,
        "return": 3.5,
        "steps": 12.0,
    }

    rl_checkpoint = tmp_path / "recurrent_rl.pt"
    torch.save(
        {
            "final_eval": {
                "success_rate": 0.25,
                "avg_return": -4.0,
                "avg_steps": 55.0,
            },
            "best_eval": {
                "success_rate": 0.8,
                "avg_return": 6.0,
                "avg_steps": 14.0,
            },
            "restored_best": True,
        },
        rl_checkpoint,
    )
    assert _parse_eval_metrics("recurrent_bc_rl", rl_stdout, [], checkpoint_path=rl_checkpoint) == {
        "success_rate": 0.8,
        "return": 6.0,
        "steps": 14.0,
    }
    assert _parse_recurrent_checkpoint_evals(rl_checkpoint) == {
        "final_eval": {
            "success_rate": 0.25,
            "return": -4.0,
            "steps": 55.0,
        },
        "best_eval": {
            "success_rate": 0.8,
            "return": 6.0,
            "steps": 14.0,
        },
    }

    final_checkpoint = tmp_path / "recurrent_rl_final.pt"
    torch.save(
        {
            "final_eval": {
                "success_rate": 0.25,
                "avg_return": -4.0,
                "avg_steps": 55.0,
            },
            "best_eval": {
                "success_rate": 0.8,
                "avg_return": 6.0,
                "avg_steps": 14.0,
            },
            "restored_best": False,
        },
        final_checkpoint,
    )
    assert _parse_eval_metrics("recurrent_bc_rl", rl_stdout, [], checkpoint_path=final_checkpoint) == {
        "success_rate": 0.25,
        "return": -4.0,
        "steps": 55.0,
    }


def test_core_training_sweep_parses_wandb_failures(tmp_path):
    from examples.core_training_sweep import _parse_wandb_record

    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text(
        "wandb init failed, continuing without wandb: wandb-core exited with code 1\n"
        "wandb log failed, disabling wandb for this run: optional backend unavailable\n",
        encoding="utf-8",
    )
    stderr.write_text("ERROR main: Serve() returned error\n", encoding="utf-8")

    record = _parse_wandb_record(stdout, stderr, requested=True, mode="offline")

    assert record["requested"] is True
    assert record["mode"] == "offline"
    assert record["status"] == "failed"
    assert record["error_lines"]


def test_core_training_sweep_parses_wandb_run_url(tmp_path):
    from examples.core_training_sweep import _parse_wandb_record

    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    run_logs = tmp_path / "wandb" / "wandb" / "run-20260726_120000-abc123" / "logs"
    run_logs.mkdir(parents=True)
    (run_logs / "debug.log").write_text(
        "2026 INFO finishing run orion8/syncorsink-core-training/abc123\n",
        encoding="utf-8",
    )

    record = _parse_wandb_record(
        stdout,
        stderr,
        requested=True,
        mode="online",
        run_dir=tmp_path,
    )

    assert record["status"] == "initialized"
    assert record["run_id"] == "abc123"
    assert record["run_path"] == "orion8/syncorsink-core-training/abc123"
    assert record["url"] == "https://wandb.ai/orion8/syncorsink-core-training/runs/abc123"
