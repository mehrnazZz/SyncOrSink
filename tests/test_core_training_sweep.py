import json

import pytest
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


def test_core_training_sweep_can_run_official_benchmark_cases(tmp_path):
    from examples.core_training_sweep import parse_args, run_suite

    args = parse_args([
        "--algorithms",
        "recurrent_bc_rl",
        "--benchmark-spec",
        "benchmarks/syncorsink_v0_1.json",
        "--benchmark-cases",
        "signal_hunt_16x16_scaled_search",
        "--seeds",
        "0",
        "--recurrent-demo-episodes",
        "1",
        "--recurrent-bc-epochs",
        "1",
        "--recurrent-dagger-rounds",
        "0",
        "--recurrent-rl-updates",
        "0",
        "--output-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "official",
        "--dry-run",
    ])

    payload = run_suite(args)
    run = payload["runs"][0]
    command = run["command"]

    assert payload["overall"] == {"complete": 0, "dry_run": 1, "failed": 0, "total": 1}
    assert payload["config"]["benchmark_spec"] == "benchmarks/syncorsink_v0_1.json"
    assert payload["config"]["benchmark_cases"] == ["signal_hunt_16x16_scaled_search"]
    assert payload["cases"][0]["benchmark_case"] == "signal_hunt_16x16_scaled_search"
    assert payload["cases"][0]["map_size"] == 16
    assert payload["cases"][0]["agents"] == 4
    assert payload["cases"][0]["fov_preset"] == "medium"
    assert payload["cases"][0]["max_steps"] == 300
    assert "recurrent_bc_rl_signal_hunt_16x16_scaled_search_seed0" in run["run_dir"]
    assert command[command.index("--map-size") + 1] == "16"
    assert command[command.index("--agents") + 1] == "4"
    assert command[command.index("--fov-preset") + 1] == "medium"
    assert command[command.index("--max-steps") + 1] == "300"
    assert command[command.index("--bc-signal-target-aux-weight") + 1] == "0.25"
    assert command[command.index("--bc-signal-target-pursuit-action-weight") + 1] == "0.4"
    assert "--bc-signal-target-pursuit-trust-exact-memory" not in command
    assert command[command.index("--bc-signal-target-match-action-weight") + 1] == "0.4"
    assert command[command.index("--bc-signal-scan-bridge-action-weight") + 1] == "0.0"
    assert command[command.index("--bc-signal-scan-bridge-min-map-size") + 1] == "16"
    assert command[command.index("--bc-signal-scan-bridge-remaining-threshold") + 1] == "0.5"
    assert command[command.index("--bc-signal-scan-bridge-max-teammate-distance") + 1] == "6"
    assert command[command.index("--bc-signal-first-target-scan-action-weight") + 1] == "0.8"
    assert command[command.index("--bc-signal-visible-clue-action-weight") + 1] == "0.0"
    assert command[command.index("--bc-signal-frontier-exploration-action-weight") + 1] == "0.25"
    assert "--eval-signal-target-scan-lock" not in command
    assert "--eval-signal-exact-target-scan-lock" not in command
    assert "--eval-signal-compatible-target-scan-assist" not in command
    assert "--eval-signal-negative-memory-scan-guard" not in command
    assert "--eval-signal-target-probe-assist" not in command
    assert "--eval-signal-frontier-exploration-assist" not in command
    assert "--eval-signal-scan-refresh-assist" not in command
    assert "--eval-signal-constraint-message-copy-assist" in command
    assert "--eval-signal-constraint-message-guard" not in command


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
        "picked_resource,dropped_resource,delivered,stage_completed,sync_complete,"
        "recharged,joint_target_scan"
    )
    assert payload["config"]["recurrent_bc_pipeline_pickup_action_loss_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_delivery_action_loss_weight"] == 2.0
    assert payload["config"]["recurrent_bc_pipeline_delivery_progress_action_loss_weight"] == 2.0
    assert payload["config"]["recurrent_bc_pipeline_navigation_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_sync_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_ready_interact_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_station_guard_action_loss_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_wrong_station_recovery_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_rl_pipeline_delivery_progress_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_rl_pipeline_navigation_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_rl_pipeline_sync_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_rl_pipeline_plan_head_loss_weight"] == 0.0
    assert payload["config"]["recurrent_rl_pipeline_option_loss_weight"] == 0.0
    assert payload["config"]["recurrent_backbone"] == "mlp"
    assert payload["config"]["recurrent_bc_pipeline_pickup_gate_loss_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_pickup_gate_pos_weight"] == 2.0
    assert payload["config"]["recurrent_bc_pipeline_pickup_gate_neg_weight"] == 1.5
    assert payload["config"]["recurrent_bc_pipeline_plan_action_loss_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_plan_head_loss_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_option_loss_weight"] == 0.75
    assert payload["config"]["recurrent_bc_pipeline_message_loss_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_send_gate_loss_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_send_gate_pos_weight"] == 3.0
    assert payload["config"]["recurrent_bc_pipeline_send_gate_neg_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_loss_weight"] == 2.0
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_pos_weight"] == 3.0
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_neg_weight"] == 2.5
    assert payload["config"]["recurrent_bc_calibrate_pipeline_interact_gate_threshold"] is True
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_threshold_target_rate"] == 0.33
    assert payload["config"]["recurrent_bc_pipeline_bad_pickup_action_loss_weight"] == 0.5
    assert payload["config"]["recurrent_bc_pipeline_bad_drop_action_loss_weight"] == 0.5
    assert payload["config"]["recurrent_bc_pipeline_bad_interact_action_loss_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_bad_action_margin_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_bad_action_margin"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_proactive_bad_action_labels"] is True
    assert payload["config"]["recurrent_pipeline_stage_count"] is None
    assert payload["config"]["recurrent_pipeline_required_per_stage_min"] == 1
    assert payload["config"]["recurrent_pipeline_required_per_stage_max"] == 2
    assert payload["config"]["recurrent_pipeline_sync_probability"] == 0.5
    assert payload["config"]["recurrent_pipeline_dependency_probability"] == 0.7
    assert payload["config"]["recurrent_pipeline_wrong_delivery_penalty"] == 0.25
    assert payload["config"]["recurrent_obs_pipeline_features"] is True
    assert payload["config"]["recurrent_eval_pipeline_navigation_assist"] is False
    assert payload["config"]["recurrent_eval_pipeline_navigation_assist_trust_messages"] is False
    assert payload["config"]["recurrent_eval_pipeline_station_interact_guard"] is False
    assert payload["config"]["recurrent_eval_pipeline_interact_gate_threshold"] == -1.0
    assert payload["config"]["recurrent_eval_pipeline_event_head_threshold"] == -1.0
    assert payload["config"]["recurrent_eval_pipeline_plan_head_threshold"] == -1.0
    assert payload["config"]["recurrent_eval_pipeline_navigation_head_threshold"] == -1.0
    assert payload["config"]["recurrent_eval_pipeline_option_threshold"] == -1.0
    assert payload["config"]["recurrent_eval_pipeline_option_allow_interact"] is False
    assert payload["config"]["recurrent_bc_kl_coeff"] == 2.0
    assert payload["config"]["recurrent_bc_comm_kl_coeff"] == 2.0
    assert payload["config"]["recurrent_bc_comm_send_pos_weight"] == 0.0
    signal_send_pos_idx = commands["signal_hunt"].index("--bc-comm-send-pos-weight")
    energy_send_pos_idx = commands["energy_grid"].index("--bc-comm-send-pos-weight")
    pipeline_send_pos_idx = commands["pipeline_assembly"].index("--bc-comm-send-pos-weight")
    signal_initial_msg_idx = commands["signal_hunt"].index("--bc-signal-initial-message-weight")
    energy_initial_msg_idx = commands["energy_grid"].index("--bc-signal-initial-message-weight")
    pipeline_initial_msg_idx = commands["pipeline_assembly"].index("--bc-signal-initial-message-weight")
    signal_initial_msg_loss_idx = commands["signal_hunt"].index(
        "--bc-signal-initial-message-loss-weight"
    )
    signal_constraint_msg_loss_idx = commands["signal_hunt"].index(
        "--bc-signal-constraint-message-loss-weight"
    )
    signal_target_aux_idx = commands["signal_hunt"].index(
        "--bc-signal-target-aux-weight"
    )
    signal_target_hypothesis_idx = commands["signal_hunt"].index(
        "--bc-signal-target-hypothesis-loss-weight"
    )
    signal_target_hypothesis_min_idx = commands["signal_hunt"].index(
        "--bc-signal-target-hypothesis-min-map-size"
    )
    energy_target_aux_idx = commands["energy_grid"].index(
        "--bc-signal-target-aux-weight"
    )
    energy_target_hypothesis_idx = commands["energy_grid"].index(
        "--bc-signal-target-hypothesis-loss-weight"
    )
    pipeline_target_aux_idx = commands["pipeline_assembly"].index(
        "--bc-signal-target-aux-weight"
    )
    pipeline_target_hypothesis_idx = commands["pipeline_assembly"].index(
        "--bc-signal-target-hypothesis-loss-weight"
    )
    signal_target_pursuit_idx = commands["signal_hunt"].index(
        "--bc-signal-target-pursuit-action-weight"
    )
    signal_sync_response_idx = commands["signal_hunt"].index(
        "--bc-signal-sync-response-action-loss-weight"
    )
    signal_active_scan_response_idx = commands["signal_hunt"].index(
        "--bc-signal-active-scan-response-action-weight"
    )
    signal_active_scan_response_min_idx = commands["signal_hunt"].index(
        "--bc-signal-active-scan-response-min-map-size"
    )
    signal_active_scan_response_max_idx = commands["signal_hunt"].index(
        "--bc-signal-active-scan-response-max-agents"
    )
    signal_scan_bridge_idx = commands["signal_hunt"].index(
        "--bc-signal-scan-bridge-action-weight"
    )
    signal_scan_bridge_min_idx = commands["signal_hunt"].index(
        "--bc-signal-scan-bridge-min-map-size"
    )
    signal_scan_bridge_remaining_idx = commands["signal_hunt"].index(
        "--bc-signal-scan-bridge-remaining-threshold"
    )
    signal_scan_bridge_distance_idx = commands["signal_hunt"].index(
        "--bc-signal-scan-bridge-max-teammate-distance"
    )
    signal_target_match_idx = commands["signal_hunt"].index(
        "--bc-signal-target-match-action-weight"
    )
    signal_first_scan_idx = commands["signal_hunt"].index(
        "--bc-signal-first-target-scan-action-weight"
    )
    signal_refresh_scan_idx = commands["signal_hunt"].index(
        "--bc-signal-refresh-target-scan-action-weight"
    )
    signal_joint_scan_idx = commands["signal_hunt"].index(
        "--bc-signal-joint-target-scan-action-weight"
    )
    signal_target_opportunity_idx = commands["signal_hunt"].index(
        "--bc-signal-target-opportunity-action-weight"
    )
    signal_redundant_wait_idx = commands["signal_hunt"].index(
        "--bc-signal-redundant-target-wait-action-loss-weight"
    )
    energy_initial_msg_loss_idx = commands["energy_grid"].index(
        "--bc-signal-initial-message-loss-weight"
    )
    pipeline_initial_msg_loss_idx = commands["pipeline_assembly"].index(
        "--bc-signal-initial-message-loss-weight"
    )
    energy_constraint_msg_loss_idx = commands["energy_grid"].index(
        "--bc-signal-constraint-message-loss-weight"
    )
    pipeline_constraint_msg_loss_idx = commands["pipeline_assembly"].index(
        "--bc-signal-constraint-message-loss-weight"
    )
    signal_comm_loss_idx = commands["signal_hunt"].index("--bc-comm-loss-weight")
    energy_comm_loss_idx = commands["energy_grid"].index("--bc-comm-loss-weight")
    pipeline_comm_loss_idx = commands["pipeline_assembly"].index("--bc-comm-loss-weight")
    signal_decoy_scan_idx = commands["signal_hunt"].index(
        "--bc-signal-decoy-scan-action-loss-weight"
    )
    energy_decoy_scan_idx = commands["energy_grid"].index(
        "--bc-signal-decoy-scan-action-loss-weight"
    )
    pipeline_decoy_scan_idx = commands["pipeline_assembly"].index(
        "--bc-signal-decoy-scan-action-loss-weight"
    )
    signal_decoy_drift_idx = commands["signal_hunt"].index(
        "--bc-signal-decoy-drift-action-loss-weight"
    )
    signal_rejected_drift_idx = commands["signal_hunt"].index(
        "--bc-signal-rejected-target-drift-action-loss-weight"
    )
    signal_visible_clue_idx = commands["signal_hunt"].index(
        "--bc-signal-visible-clue-action-weight"
    )
    signal_visible_clue_min_idx = commands["signal_hunt"].index(
        "--bc-signal-visible-clue-min-map-size"
    )
    signal_clue_interact_idx = commands["signal_hunt"].index(
        "--bc-signal-clue-interact-action-weight"
    )
    signal_clue_interact_min_idx = commands["signal_hunt"].index(
        "--bc-signal-clue-interact-min-map-size"
    )
    signal_evidence_sweep_idx = commands["signal_hunt"].index(
        "--bc-signal-evidence-sweep-action-weight"
    )
    signal_evidence_sweep_min_idx = commands["signal_hunt"].index(
        "--bc-signal-evidence-sweep-min-map-size"
    )
    signal_frontier_idx = commands["signal_hunt"].index(
        "--bc-signal-frontier-exploration-action-weight"
    )
    signal_frontier_min_idx = commands["signal_hunt"].index(
        "--bc-signal-frontier-exploration-min-map-size"
    )
    signal_focus_events_idx = commands["signal_hunt"].index("--dagger-focus-events")
    energy_focus_events_idx = commands["energy_grid"].index("--dagger-focus-events")
    pipeline_focus_events_idx = commands["pipeline_assembly"].index("--dagger-focus-events")
    signal_positive_replay_idx = commands["signal_hunt"].index(
        "--dagger-positive-replay-events"
    )
    signal_replay_weights_idx = commands["signal_hunt"].index("--dagger-replay-event-weights")
    signal_replay_priority_idx = commands["signal_hunt"].index(
        "--dagger-replay-priority-events"
    )
    signal_target_discovery_min_idx = commands["signal_hunt"].index(
        "--dagger-target-discovery-min-map-size"
    )
    energy_target_discovery_min_idx = commands["energy_grid"].index(
        "--dagger-target-discovery-min-map-size"
    )
    signal_movement_stall_min_idx = commands["signal_hunt"].index(
        "--dagger-movement-stall-min-map-size"
    )
    signal_movement_stall_window_idx = commands["signal_hunt"].index(
        "--dagger-movement-stall-window"
    )
    assert commands["signal_hunt"][signal_send_pos_idx + 1] == "-1.0"
    assert commands["energy_grid"][energy_send_pos_idx + 1] == "0.0"
    assert commands["pipeline_assembly"][pipeline_send_pos_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_initial_msg_idx + 1] == "4.0"
    assert commands["energy_grid"][energy_initial_msg_idx + 1] == "1.0"
    assert commands["pipeline_assembly"][pipeline_initial_msg_idx + 1] == "1.0"
    assert commands["signal_hunt"][signal_initial_msg_loss_idx + 1] == "4.0"
    assert commands["energy_grid"][energy_initial_msg_loss_idx + 1] == "0.0"
    assert commands["pipeline_assembly"][pipeline_initial_msg_loss_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_constraint_msg_loss_idx + 1] == "4.0"
    assert commands["energy_grid"][energy_constraint_msg_loss_idx + 1] == "0.0"
    assert commands["pipeline_assembly"][pipeline_constraint_msg_loss_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_target_aux_idx + 1] == "0.25"
    assert commands["signal_hunt"][signal_target_hypothesis_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_target_hypothesis_min_idx + 1] == "16"
    assert commands["energy_grid"][energy_target_aux_idx + 1] == "0.0"
    assert commands["energy_grid"][energy_target_hypothesis_idx + 1] == "0.0"
    assert commands["pipeline_assembly"][pipeline_target_aux_idx + 1] == "0.0"
    assert commands["pipeline_assembly"][pipeline_target_hypothesis_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_target_pursuit_idx + 1] == "0.4"
    assert commands["signal_hunt"][signal_sync_response_idx + 1] == "0.2"
    assert commands["signal_hunt"][signal_active_scan_response_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_active_scan_response_min_idx + 1] == "16"
    assert commands["signal_hunt"][signal_active_scan_response_max_idx + 1] == "1"
    assert commands["signal_hunt"][signal_scan_bridge_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_scan_bridge_min_idx + 1] == "16"
    assert commands["signal_hunt"][signal_scan_bridge_remaining_idx + 1] == "0.5"
    assert commands["signal_hunt"][signal_scan_bridge_distance_idx + 1] == "6"
    assert commands["signal_hunt"][signal_target_match_idx + 1] == "0.4"
    assert commands["signal_hunt"][signal_first_scan_idx + 1] == "0.8"
    assert commands["signal_hunt"][signal_refresh_scan_idx + 1] == "0.3"
    assert commands["signal_hunt"][signal_joint_scan_idx + 1] == "0.5"
    assert commands["signal_hunt"][signal_target_opportunity_idx + 1] == "0.4"
    assert commands["signal_hunt"][signal_redundant_wait_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_comm_loss_idx + 1] == "1.0"
    assert commands["energy_grid"][energy_comm_loss_idx + 1] == "0.1"
    assert commands["pipeline_assembly"][pipeline_comm_loss_idx + 1] == "0.1"
    assert commands["signal_hunt"][signal_decoy_scan_idx + 1] == "0.0"
    assert commands["energy_grid"][energy_decoy_scan_idx + 1] == "0.0"
    assert commands["pipeline_assembly"][pipeline_decoy_scan_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_decoy_drift_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_rejected_drift_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_clue_interact_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_clue_interact_min_idx + 1] == "16"
    assert commands["signal_hunt"][signal_visible_clue_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_visible_clue_min_idx + 1] == "16"
    assert commands["signal_hunt"][signal_evidence_sweep_idx + 1] == "0.0"
    assert commands["signal_hunt"][signal_evidence_sweep_min_idx + 1] == "16"
    assert commands["signal_hunt"][signal_frontier_idx + 1] == "0.25"
    assert commands["signal_hunt"][signal_frontier_min_idx + 1] == "16"
    signal_focus_events = commands["signal_hunt"][signal_focus_events_idx + 1]
    energy_focus_events = commands["energy_grid"][energy_focus_events_idx + 1]
    pipeline_focus_events = commands["pipeline_assembly"][pipeline_focus_events_idx + 1]
    assert "missed_target_scan" in signal_focus_events
    assert "target_interact_miss" in signal_focus_events
    assert "target_pursuit_miss" in signal_focus_events
    assert "target_discovery_miss" in signal_focus_events
    assert "target_decoy_drift_miss" in signal_focus_events
    assert "visible_clue_miss" not in signal_focus_events
    assert "frontier_exploration_miss" in signal_focus_events
    assert "target_handoff_miss" in signal_focus_events
    assert "pipeline_wrong_delivery" in signal_focus_events
    assert "decoy_scan" not in signal_focus_events
    assert "pipeline_wrong_delivery" in energy_focus_events
    assert "pipeline_wrong_delivery" in pipeline_focus_events
    assert "pipeline_delivery_ready" in commands["signal_hunt"][
        signal_positive_replay_idx + 1
    ]
    assert "target_handoff" in commands["signal_hunt"][signal_positive_replay_idx + 1]
    assert "target_pursuit" not in commands["signal_hunt"][signal_positive_replay_idx + 1]
    assert "pipeline_delivery_miss:4.0" in commands["signal_hunt"][
        signal_replay_weights_idx + 1
    ]
    assert "target_discovery_miss:4.0" in commands["signal_hunt"][
        signal_replay_weights_idx + 1
    ]
    assert "frontier_exploration_miss:4.0" in commands["signal_hunt"][
        signal_replay_weights_idx + 1
    ]
    assert "target_decoy_drift_miss:4.0" in commands["signal_hunt"][
        signal_replay_weights_idx + 1
    ]
    assert "target_handoff_miss:4.0" in commands["signal_hunt"][
        signal_replay_weights_idx + 1
    ]
    assert "target_handoff:3.0" in commands["signal_hunt"][
        signal_replay_weights_idx + 1
    ]
    assert "pipeline_delivery_miss" in commands["signal_hunt"][
        signal_replay_priority_idx + 1
    ]
    assert "target_discovery_miss" in commands["signal_hunt"][
        signal_replay_priority_idx + 1
    ]
    assert "frontier_exploration_miss" in commands["signal_hunt"][
        signal_replay_priority_idx + 1
    ]
    assert "target_handoff_miss" in commands["signal_hunt"][
        signal_replay_priority_idx + 1
    ]
    assert commands["signal_hunt"][signal_target_discovery_min_idx + 1] == "16"
    assert commands["energy_grid"][energy_target_discovery_min_idx + 1] == "16"
    assert commands["signal_hunt"][signal_movement_stall_min_idx + 1] == "16"
    assert commands["signal_hunt"][signal_movement_stall_window_idx + 1] == "6"
    assert "--dagger-initial-target-broadcast-labels" in commands["signal_hunt"]
    assert "--dagger-target-handoff-requires-exact-target" not in commands["signal_hunt"]
    assert "--eval-signal-initial-exact-message-copy-assist" in commands["signal_hunt"]
    assert "--obs-agent-id-features" in commands["signal_hunt"]
    assert "--bc-signal-scan-decision-loss-weight" in commands["signal_hunt"]
    assert commands["signal_hunt"][
        commands["signal_hunt"].index("--bc-signal-scan-decision-neg-weight") + 1
    ] == "3.0"
    assert commands["signal_hunt"][
        commands["signal_hunt"].index("--bc-signal-scan-gate-loss-weight") + 1
    ] == "1.0"
    assert commands["signal_hunt"][
        commands["signal_hunt"].index("--bc-signal-target-validity-loss-weight") + 1
    ] == "1.0"
    assert commands["signal_hunt"][
        commands["signal_hunt"].index("--bc-signal-target-decision-loss-weight") + 1
    ] == "1.0"
    signal_scan_gate_threshold_idx = commands["signal_hunt"].index(
        "--eval-signal-scan-gate-threshold"
    )
    signal_target_scan_threshold_idx = commands["signal_hunt"].index(
        "--eval-signal-target-scan-threshold"
    )
    signal_target_validity_threshold_idx = commands["signal_hunt"].index(
        "--eval-signal-target-validity-threshold"
    )
    signal_target_decision_threshold_idx = commands["signal_hunt"].index(
        "--eval-signal-target-decision-threshold"
    )
    assert commands["signal_hunt"][signal_scan_gate_threshold_idx + 1] == "0.4"
    assert commands["signal_hunt"][signal_target_scan_threshold_idx + 1] == "0.0"
    assert "--eval-signal-scan-gate-suppress" in commands["signal_hunt"]
    assert "--eval-signal-target-scan-lock" not in commands["signal_hunt"]
    assert "--eval-signal-exact-target-scan-lock" not in commands["signal_hunt"]
    assert "--eval-signal-compatible-target-scan-assist" not in commands["signal_hunt"]
    assert "--eval-signal-constraint-message-copy-assist" in commands["signal_hunt"]
    assert "--eval-signal-constraint-message-guard" not in commands["signal_hunt"]
    assert commands["signal_hunt"][signal_target_validity_threshold_idx + 1] == "0.4"
    assert commands["signal_hunt"][signal_target_decision_threshold_idx + 1] == "0.4"
    assert "--dagger-initial-target-broadcast-labels" not in commands["energy_grid"]
    assert "--dagger-initial-target-broadcast-labels" not in commands["pipeline_assembly"]
    assert "--dagger-target-handoff-requires-exact-target" not in commands["energy_grid"]
    assert "--dagger-target-handoff-requires-exact-target" not in commands["pipeline_assembly"]
    assert "--eval-signal-initial-exact-message-copy-assist" not in commands["energy_grid"]
    assert "--eval-signal-initial-exact-message-copy-assist" not in commands["pipeline_assembly"]
    assert "--obs-agent-id-features" not in commands["energy_grid"]
    assert "--obs-agent-id-features" not in commands["pipeline_assembly"]
    assert "--eval-signal-target-scan-threshold" not in commands["energy_grid"]
    assert "--eval-signal-target-scan-threshold" not in commands["pipeline_assembly"]
    assert "--eval-signal-target-scan-lock" not in commands["energy_grid"]
    assert "--eval-signal-target-scan-lock" not in commands["pipeline_assembly"]
    assert "--eval-signal-exact-target-scan-lock" not in commands["energy_grid"]
    assert "--eval-signal-exact-target-scan-lock" not in commands["pipeline_assembly"]
    assert "--eval-signal-compatible-target-scan-assist" not in commands["energy_grid"]
    assert "--eval-signal-compatible-target-scan-assist" not in commands["pipeline_assembly"]
    assert "--eval-signal-constraint-message-copy-assist" not in commands["energy_grid"]
    assert "--eval-signal-constraint-message-copy-assist" not in commands["pipeline_assembly"]
    assert "--eval-signal-constraint-message-guard" not in commands["energy_grid"]
    assert "--eval-signal-constraint-message-guard" not in commands["pipeline_assembly"]
    assert "--bc-signal-scan-decision-loss-weight" not in commands["energy_grid"]
    assert "--bc-signal-scan-decision-loss-weight" not in commands["pipeline_assembly"]
    assert payload["config"]["recurrent_dagger_failed_effective_ratio_cap"] == 0.25
    assert payload["config"]["recurrent_dagger_oracle_action_rollin_rate"] == 0.25
    assert payload["config"]["recurrent_dagger_oracle_message_rollin_rate"] == 0.0
    assert "pipeline_wrong_delivery" in payload["config"]["recurrent_dagger_focus_events"]
    assert "pipeline_sync_wait" in payload["config"]["recurrent_dagger_focus_events"]
    assert payload["config"]["recurrent_dagger_focus_error_weight"] == 3.0
    assert payload["config"]["recurrent_dagger_focus_recovery_weight"] == 2.0
    assert payload["config"]["recurrent_dagger_focus_window"] == 1
    assert payload["config"]["recurrent_dagger_focus_replay"] is True
    assert payload["config"]["recurrent_dagger_retrain_from_scratch"] is True
    assert payload["config"]["recurrent_dagger_restore_best"] is True
    assert payload["config"]["recurrent_dagger_pipeline_wrong_delivery_provenance_labels"] is True
    assert payload["config"]["recurrent_dagger_pipeline_wrong_delivery_provenance_weight"] == 3.0
    assert payload["config"]["recurrent_dagger_replay_pre_steps"] == 2
    assert payload["config"]["recurrent_dagger_replay_post_steps"] == 2
    assert payload["config"]["recurrent_dagger_replay_weight"] == 1.0
    assert payload["config"]["recurrent_dagger_positive_replay_events"] == (
        "target_handoff,pipeline_delivery_ready,delivered,stage_completed"
    )
    assert payload["config"]["recurrent_dagger_replay_event_weights"] == (
        "pipeline_delivery_ready:4.0,pipeline_delivery_miss:4.0,"
        "pipeline_station_stall_miss:3.0,"
        "pipeline_sync_wait:4.0,"
        "frontier_exploration_miss:4.0,target_discovery_miss:4.0,"
        "target_decoy_drift_miss:4.0,target_pursuit_miss:3.0,"
        "target_handoff_miss:4.0,target_handoff:3.0,"
        "pipeline_wrong_delivery:3.0,pipeline_wrong_delivery_root_pickup:3.0,"
        "delivered:2.0,stage_completed:2.0"
    )
    assert payload["config"]["recurrent_dagger_replay_event_caps"] == ""
    assert payload["config"]["recurrent_dagger_replay_success_only_events"] == "delivered,stage_completed"
    assert payload["config"]["recurrent_dagger_replay_priority_events"] == (
        "frontier_exploration_miss,target_discovery_miss,target_decoy_drift_miss,"
        "target_handoff_miss,target_handoff,"
        "pipeline_delivery_miss,pipeline_delivery_ready,pipeline_wrong_delivery,"
        "pipeline_wrong_delivery_root_pickup,pipeline_sync_wait"
    )
    assert payload["config"]["recurrent_dagger_replay_balance_positive_events"] == ""
    assert payload["config"]["recurrent_dagger_replay_balance_negative_events"] == ""
    assert payload["config"]["recurrent_dagger_replay_max_negative_per_positive"] == -1.0
    assert payload["config"]["recurrent_dagger_max_replay_snippets_per_episode"] == 8
    assert payload["config"]["recurrent_dagger_max_failed_parent_replay_snippets_per_episode"] == 4
    assert payload["config"]["recurrent_dagger_failed_parent_replay_weight_scale"] == 1.0
    assert payload["config"]["recurrent_dagger_expert_max_replay_snippets_per_episode"] == -1
    assert payload["config"]["recurrent_rl_early_stop_eval_patience"] == 4
    assert payload["config"]["recurrent_rl_balanced_rollouts"] is True
    assert payload["config"]["recurrent_rl_rollout_eval_decoding"] is True
    assert payload["config"]["recurrent_rl_rollout_pipeline_navigation_assist"] is False
    assert payload["config"]["recurrent_rl_rollout_pipeline_navigation_assist_trust_messages"] is False
    assert payload["config"]["recurrent_rl_rollout_pipeline_station_interact_guard"] is True
    assert payload["config"]["recurrent_rl_pipeline_bad_pickup_penalty"] == 0.1
    assert payload["config"]["recurrent_rl_pipeline_bad_interact_penalty"] == 0.1
    assert payload["config"]["recurrent_rl_pipeline_unneeded_drop_bonus"] == 0.05
    assert payload["config"]["recurrent_obs_memory_mode"] == "auto"

    assert commands["signal_hunt"][commands["signal_hunt"].index("--oracle") + 1] == "signal_hint_comm"
    assert commands["energy_grid"][commands["energy_grid"].index("--oracle") + 1] == "planner_comm"
    assert (
        commands["pipeline_assembly"][commands["pipeline_assembly"].index("--oracle") + 1]
        == "planner_comm"
    )
    assert "--bc-action-class-balance" in commands["energy_grid"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--rl-pipeline-bad-pickup-penalty") + 1
    ] == "0.1"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--rl-pipeline-bad-interact-penalty") + 1
    ] == "0.1"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--rl-pipeline-unneeded-drop-bonus") + 1
    ] == "0.05"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-action-class-balance-max-weight") + 1
    ] == "5.0"
    assert commands["energy_grid"][
        commands["energy_grid"].index("--bc-event-action-weight") + 1
    ] == "2.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-event-action-events") + 1
    ] == (
        "picked_resource,dropped_resource,delivered,stage_completed,sync_complete,"
        "recharged,joint_target_scan"
    )
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-pickup-action-loss-weight") + 1
    ] == "1.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-delivery-action-loss-weight") + 1
    ] == "2.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-delivery-progress-action-loss-weight") + 1
    ] == "2.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-navigation-action-loss-weight") + 1
    ] == "0.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-sync-action-loss-weight") + 1
    ] == "0.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-station-guard-action-loss-weight") + 1
    ] == "1.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index(
            "--bc-pipeline-wrong-station-recovery-action-loss-weight"
        ) + 1
    ] == "0.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-pickup-gate-loss-weight") + 1
    ] == "1.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-pickup-gate-pos-weight") + 1
    ] == "2.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-pickup-gate-neg-weight") + 1
    ] == "1.5"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-plan-action-loss-weight") + 1
    ] == "1.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-plan-head-loss-weight") + 1
    ] == "1.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-option-loss-weight") + 1
    ] == "0.75"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-message-loss-weight") + 1
    ] == "1.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-send-gate-loss-weight") + 1
    ] == "1.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-send-gate-pos-weight") + 1
    ] == "3.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-send-gate-neg-weight") + 1
    ] == "1.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-interact-gate-loss-weight") + 1
    ] == "2.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-interact-gate-pos-weight") + 1
    ] == "3.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-interact-gate-neg-weight") + 1
    ] == "2.5"
    assert "--bc-calibrate-pipeline-interact-gate-threshold" in commands["pipeline_assembly"]
    assert "--bc-calibrate-pipeline-interact-gate-threshold" not in commands["signal_hunt"]
    assert "--bc-calibrate-pipeline-interact-gate-threshold" not in commands["energy_grid"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-interact-gate-threshold-target-rate") + 1
    ] == "0.33"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-bad-pickup-action-loss-weight") + 1
    ] == "0.5"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-bad-drop-action-loss-weight") + 1
    ] == "0.5"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-bad-interact-action-loss-weight") + 1
    ] == "1.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-bad-action-margin-loss-weight") + 1
    ] == "0.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--bc-pipeline-bad-action-margin") + 1
    ] == "1.0"
    assert "--bc-pipeline-proactive-bad-action-labels" in commands["pipeline_assembly"]
    assert "--bc-pipeline-proactive-bad-action-labels" not in commands["signal_hunt"]
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
    assert "--obs-pipeline-features" in commands["pipeline_assembly"]
    assert "--eval-pipeline-navigation-assist" not in commands["pipeline_assembly"]
    assert "--eval-pipeline-navigation-assist-trust-messages" not in commands["pipeline_assembly"]
    assert "--eval-pipeline-event-head-threshold" not in commands["pipeline_assembly"]
    assert "--eval-pipeline-navigation-head-threshold" not in commands["pipeline_assembly"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-failed-effective-ratio-cap") + 1
    ] == "0.25"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-oracle-action-rollin-rate") + 1
    ] == "0.25"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-oracle-message-rollin-rate") + 1
    ] == "0.0"
    assert "--dagger-focus-replay" in commands["pipeline_assembly"]
    assert "--no-dagger-retrain-from-scratch" not in commands["pipeline_assembly"]
    assert "--dagger-pipeline-wrong-delivery-provenance-labels" in commands["pipeline_assembly"]
    assert "--dagger-pipeline-wrong-delivery-provenance-labels" not in commands["signal_hunt"]
    assert "--dagger-pipeline-wrong-delivery-provenance-labels" not in commands["energy_grid"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-pipeline-wrong-delivery-provenance-weight") + 1
    ] == "3.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-replay-pre-steps") + 1
    ] == "2"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-replay-post-steps") + 1
    ] == "2"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-replay-weight") + 1
    ] == "1.0"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-positive-replay-events") + 1
    ] == "target_handoff,pipeline_delivery_ready,delivered,stage_completed"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-replay-event-weights") + 1
    ] == payload["config"]["recurrent_dagger_replay_event_weights"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-replay-success-only-events") + 1
    ] == "delivered,stage_completed"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-replay-priority-events") + 1
    ] == payload["config"]["recurrent_dagger_replay_priority_events"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-max-replay-snippets-per-episode") + 1
    ] == "8"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--dagger-max-failed-parent-replay-snippets-per-episode") + 1
    ] == "4"
    assert "--rl-balanced-rollouts" in commands["energy_grid"]
    assert "--rl-rollout-eval-decoding" in commands["pipeline_assembly"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--rl-early-stop-eval-patience") + 1
    ] == "4"
    assert "--rl-rollout-pipeline-navigation-assist" not in commands["pipeline_assembly"]
    assert "--rl-rollout-pipeline-navigation-assist-trust-messages" not in commands["pipeline_assembly"]
    assert "--rl-rollout-pipeline-station-interact-guard" in commands["pipeline_assembly"]
    assert "--rl-rollout-pipeline-station-interact-guard" not in commands["signal_hunt"]
    assert commands["signal_hunt"][commands["signal_hunt"].index("--obs-memory-mode") + 1] == "egocentric"
    assert "--obs-memory-mode" not in commands["pipeline_assembly"]


def test_core_training_sweep_recurrent_large_map_signal_preset_targets_32x_failures(tmp_path):
    from examples.core_training_sweep import parse_args, run_suite

    args = parse_args([
        "--algorithms",
        "recurrent_bc_rl",
        "--scenarios",
        "signal_hunt",
        "energy_grid",
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
        "--recurrent-signal-preset",
        "large_map",
        "--output-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "recurrent-large-map",
        "--dry-run",
    ])

    payload = run_suite(args)
    commands = {run["scenario"]: run["command"] for run in payload["runs"]}

    def arg_value(command, option):
        return command[command.index(option) + 1]

    signal = commands["signal_hunt"]
    energy = commands["energy_grid"]
    assert payload["config"]["recurrent_signal_preset"] == "large_map"
    assert payload["config"]["recurrent_eval_signal_constraint_message_guard"] is True
    assert payload["config"]["recurrent_eval_signal_exact_target_message_copy_assist"] is True
    assert payload["config"]["recurrent_obs_signal_confidence_features"] is False
    assert payload["config"]["recurrent_obs_signal_sector_features"] is False
    assert arg_value(signal, "--bc-signal-decoy-drift-action-loss-weight") == "0.1"
    assert arg_value(signal, "--bc-signal-decoy-scan-action-loss-weight") == "0.25"
    assert arg_value(signal, "--bc-signal-rejected-target-drift-action-loss-weight") == "0.0"
    assert arg_value(signal, "--bc-signal-clue-interact-action-weight") == "0.0"
    assert arg_value(signal, "--bc-signal-clue-interact-min-map-size") == "16"
    assert arg_value(signal, "--bc-signal-visible-clue-action-weight") == "0.25"
    assert arg_value(signal, "--bc-signal-visible-clue-min-map-size") == "32"
    assert arg_value(signal, "--bc-signal-evidence-sweep-action-weight") == "0.0"
    assert arg_value(signal, "--bc-signal-evidence-sweep-min-map-size") == "16"
    assert arg_value(signal, "--bc-signal-frontier-exploration-action-weight") == "0.25"
    assert arg_value(signal, "--bc-signal-frontier-exploration-min-map-size") == "16"
    assert arg_value(signal, "--bc-signal-target-pursuit-max-agents") == "0"
    assert "--bc-signal-ambiguous-target-decision-negatives" not in signal
    assert arg_value(signal, "--bc-signal-ambiguous-target-decision-min-map-size") == "16"
    assert "--bc-signal-ambiguous-target-search-labels" not in signal
    assert arg_value(signal, "--bc-signal-ambiguous-target-search-min-map-size") == "16"
    assert arg_value(signal, "--bc-comm-loss-weight") == "1.0"
    assert "--bc-signal-constraint-frontier-bias" not in signal
    assert "--dagger-signal-target-rendezvous-labels" not in signal
    assert arg_value(signal, "--dagger-signal-target-rendezvous-min-map-size") == "16"
    assert arg_value(signal, "--dagger-signal-target-rendezvous-max-agents") == "2"
    assert "--obs-exploration-age" not in signal
    assert "--obs-signal-confidence-features" not in signal
    assert "--obs-signal-sector-features" not in signal
    assert "--eval-signal-constraint-message-copy-assist" in signal
    assert "--eval-signal-constraint-message-guard" in signal
    assert "--eval-signal-exact-target-message-copy-assist" in signal

    signal_focus_events = arg_value(signal, "--dagger-focus-events")
    signal_replay_weights = arg_value(signal, "--dagger-replay-event-weights")
    signal_replay_priority = arg_value(signal, "--dagger-replay-priority-events")
    for event in (
        "visible_clue_miss",
        "decoy_scan",
        "rejected_target_scan",
    ):
        assert event in signal_focus_events
        assert event in signal_replay_priority
        assert f"{event}:4.0" in signal_replay_weights
    assert "evidence_sweep_miss" not in signal_focus_events
    assert "evidence_sweep_miss" not in signal_replay_priority
    assert "evidence_sweep_miss:4.0" not in signal_replay_weights

    energy_focus_events = arg_value(energy, "--dagger-focus-events")
    energy_replay_weights = arg_value(energy, "--dagger-replay-event-weights")
    assert "visible_clue_miss" not in energy_focus_events
    assert "decoy_scan" not in energy_focus_events
    assert "visible_clue_miss:4.0" not in energy_replay_weights
    assert "decoy_scan:4.0" not in energy_replay_weights
    assert arg_value(energy, "--bc-signal-decoy-scan-action-loss-weight") == "0.0"


def test_core_training_sweep_recurrent_pipeline_assist_flag_is_pipeline_only(tmp_path):
    from examples.core_training_sweep import parse_args, run_suite

    args = parse_args([
        "--algorithms",
        "recurrent_bc_rl",
        "--scenarios",
        "signal_hunt",
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
        "--recurrent-eval-pipeline-navigation-assist",
        "--recurrent-eval-pipeline-navigation-assist-trust-messages",
        "--recurrent-eval-pipeline-station-interact-guard",
        "--recurrent-eval-pipeline-interact-gate-threshold",
        "0.55",
        "--recurrent-eval-pipeline-interact-gate-promote",
        "--recurrent-eval-pipeline-event-head-threshold",
        "0.57",
        "--recurrent-eval-pipeline-plan-head-threshold",
        "0.61",
        "--recurrent-eval-pipeline-navigation-head-threshold",
        "0.63",
        "--recurrent-eval-pipeline-option-threshold",
        "0.62",
        "--recurrent-eval-pipeline-option-allow-interact",
        "--recurrent-rl-rollout-pipeline-navigation-assist",
        "--recurrent-rl-rollout-pipeline-navigation-assist-trust-messages",
        "--recurrent-rl-rollout-pipeline-station-interact-guard",
        "--recurrent-rl-rollout-pipeline-interact-gate-promote",
        "--recurrent-pipeline-assisted-rollout-episodes",
        "5",
        "--recurrent-pipeline-assisted-rollout-seed-base",
        "4100",
        "--recurrent-pipeline-assisted-rollout-seed-list",
        "8:4100,4101",
        "--recurrent-pipeline-assisted-rollout-max-steps-per-episode",
        "12",
        "--recurrent-pipeline-assisted-rollout-weight",
        "2.5",
        "--recurrent-pipeline-assisted-rollout-success-only",
        "--no-recurrent-pipeline-assisted-rollout-navigation-assist",
        "--no-recurrent-pipeline-assisted-rollout-navigation-assist-trust-messages",
        "--no-recurrent-pipeline-assisted-rollout-station-interact-guard",
        "--recurrent-pipeline-assisted-rollout-bc-epochs",
        "3",
        "--recurrent-obs-exploration-memory",
        "--recurrent-obs-exploration-age",
        "--recurrent-obs-feedback",
        "--recurrent-obs-normalize-tokens",
        "--recurrent-obs-navigation-features",
        "--recurrent-obs-signal-features",
        "--recurrent-obs-signal-target-match-features",
        "--recurrent-obs-signal-confidence-features",
        "--recurrent-obs-signal-sector-features",
        "--recurrent-obs-signal-sync-feedback",
        "--recurrent-obs-signal-scan-state",
        "--recurrent-obs-pipeline-feedback",
        "--recurrent-obs-pipeline-progress-features",
        "--recurrent-obs-pipeline-shared-feedback",
        "--recurrent-obs-memory-mode",
        "egocentric",
        "--output-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "recurrent-pipeline-assist",
        "--dry-run",
    ])

    payload = run_suite(args)
    commands = {run["scenario"]: run["command"] for run in payload["runs"]}

    assert payload["config"]["recurrent_eval_pipeline_navigation_assist"] is True
    assert payload["config"]["recurrent_eval_pipeline_navigation_assist_trust_messages"] is True
    assert payload["config"]["recurrent_eval_pipeline_station_interact_guard"] is True
    assert payload["config"]["recurrent_eval_pipeline_interact_gate_threshold"] == 0.55
    assert payload["config"]["recurrent_eval_pipeline_interact_gate_promote"] is True
    assert payload["config"]["recurrent_eval_pipeline_event_head_threshold"] == 0.57
    assert payload["config"]["recurrent_eval_pipeline_plan_head_threshold"] == 0.61
    assert payload["config"]["recurrent_eval_pipeline_navigation_head_threshold"] == 0.63
    assert payload["config"]["recurrent_eval_pipeline_option_threshold"] == 0.62
    assert payload["config"]["recurrent_eval_pipeline_option_allow_interact"] is True
    assert payload["config"]["recurrent_rl_rollout_pipeline_navigation_assist"] is True
    assert payload["config"]["recurrent_rl_rollout_pipeline_navigation_assist_trust_messages"] is True
    assert payload["config"]["recurrent_rl_rollout_pipeline_station_interact_guard"] is True
    assert payload["config"]["recurrent_rl_rollout_pipeline_interact_gate_promote"] is True
    assert payload["config"]["recurrent_pipeline_assisted_rollout_episodes"] == 5
    assert payload["config"]["recurrent_pipeline_assisted_rollout_seed_base"] == 4100
    assert payload["config"]["recurrent_pipeline_assisted_rollout_seed_list"] == "8:4100,4101"
    assert payload["config"]["recurrent_pipeline_assisted_rollout_max_steps_per_episode"] == 12
    assert payload["config"]["recurrent_pipeline_assisted_rollout_weight"] == 2.5
    assert payload["config"]["recurrent_pipeline_assisted_rollout_success_only"] is True
    assert payload["config"]["recurrent_pipeline_assisted_rollout_navigation_assist"] is False
    assert (
        payload["config"]["recurrent_pipeline_assisted_rollout_navigation_assist_trust_messages"]
        is False
    )
    assert payload["config"]["recurrent_pipeline_assisted_rollout_station_interact_guard"] is False
    assert payload["config"]["recurrent_pipeline_assisted_rollout_bc_epochs"] == 3
    assert payload["config"]["recurrent_obs_exploration_memory"] is True
    assert payload["config"]["recurrent_obs_exploration_age"] is True
    assert payload["config"]["recurrent_obs_feedback"] is True
    assert payload["config"]["recurrent_obs_normalize_tokens"] is True
    assert payload["config"]["recurrent_obs_navigation_features"] is True
    assert payload["config"]["recurrent_obs_signal_features"] is True
    assert payload["config"]["recurrent_obs_signal_target_match_features"] is True
    assert payload["config"]["recurrent_obs_signal_confidence_features"] is True
    assert payload["config"]["recurrent_obs_signal_sector_features"] is True
    assert payload["config"]["recurrent_obs_signal_sync_feedback"] is True
    assert payload["config"]["recurrent_obs_signal_scan_state"] is True
    assert payload["config"]["recurrent_obs_pipeline_feedback"] is True
    assert payload["config"]["recurrent_obs_pipeline_feedback_metadata"] is True
    assert payload["config"]["recurrent_obs_pipeline_progress_features"] is True
    assert payload["config"]["recurrent_obs_pipeline_shared_feedback"] is True
    assert payload["config"]["recurrent_obs_memory_mode"] == "egocentric"
    assert "--eval-pipeline-navigation-assist" in commands["pipeline_assembly"]
    assert "--eval-pipeline-navigation-assist-trust-messages" in commands["pipeline_assembly"]
    assert "--eval-pipeline-station-interact-guard" in commands["pipeline_assembly"]
    assert "--eval-pipeline-interact-gate-threshold" in commands["pipeline_assembly"]
    assert "--eval-pipeline-interact-gate-promote" in commands["pipeline_assembly"]
    assert "--eval-pipeline-event-head-threshold" in commands["pipeline_assembly"]
    assert "--eval-pipeline-plan-head-threshold" in commands["pipeline_assembly"]
    assert "--eval-pipeline-navigation-head-threshold" in commands["pipeline_assembly"]
    assert "--eval-pipeline-option-threshold" in commands["pipeline_assembly"]
    assert "--eval-pipeline-option-allow-interact" in commands["pipeline_assembly"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--eval-pipeline-interact-gate-threshold") + 1
    ] == "0.55"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--eval-pipeline-event-head-threshold") + 1
    ] == "0.57"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--eval-pipeline-plan-head-threshold") + 1
    ] == "0.61"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--eval-pipeline-navigation-head-threshold") + 1
    ] == "0.63"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--eval-pipeline-option-threshold") + 1
    ] == "0.62"
    assert "--rl-rollout-pipeline-navigation-assist" in commands["pipeline_assembly"]
    assert "--rl-rollout-pipeline-navigation-assist-trust-messages" in commands["pipeline_assembly"]
    assert "--rl-rollout-pipeline-station-interact-guard" in commands["pipeline_assembly"]
    assert "--rl-rollout-pipeline-interact-gate-promote" in commands["pipeline_assembly"]
    assert "--pipeline-assisted-rollout-episodes" in commands["pipeline_assembly"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--pipeline-assisted-rollout-episodes") + 1
    ] == "5"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--pipeline-assisted-rollout-seed-base") + 1
    ] == "4100"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--pipeline-assisted-rollout-seed-list") + 1
    ] == "8:4100,4101"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--pipeline-assisted-rollout-max-steps-per-episode") + 1
    ] == "12"
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--pipeline-assisted-rollout-weight") + 1
    ] == "2.5"
    assert "--pipeline-assisted-rollout-success-only" in commands["pipeline_assembly"]
    assert "--no-pipeline-assisted-rollout-navigation-assist" in commands["pipeline_assembly"]
    assert (
        "--no-pipeline-assisted-rollout-navigation-assist-trust-messages"
        in commands["pipeline_assembly"]
    )
    assert "--no-pipeline-assisted-rollout-station-interact-guard" in commands["pipeline_assembly"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--pipeline-assisted-rollout-bc-epochs") + 1
    ] == "3"
    assert "--pipeline-wrong-delivery-penalty" in commands["pipeline_assembly"]
    assert commands["pipeline_assembly"][
        commands["pipeline_assembly"].index("--pipeline-wrong-delivery-penalty") + 1
    ] == "0.25"
    assert "--obs-exploration-memory" in commands["pipeline_assembly"]
    assert "--obs-feedback" in commands["pipeline_assembly"]
    assert "--obs-normalize-tokens" in commands["pipeline_assembly"]
    assert "--obs-navigation-features" in commands["pipeline_assembly"]
    assert "--obs-signal-features" in commands["pipeline_assembly"]
    assert "--obs-signal-target-match-features" in commands["pipeline_assembly"]
    assert "--obs-signal-confidence-features" in commands["pipeline_assembly"]
    assert "--obs-signal-sector-features" in commands["pipeline_assembly"]
    assert "--obs-signal-sync-feedback" in commands["pipeline_assembly"]
    assert "--obs-signal-scan-state" in commands["pipeline_assembly"]
    assert "--obs-pipeline-feedback" in commands["pipeline_assembly"]
    assert "--obs-pipeline-feedback-metadata" in commands["pipeline_assembly"]
    assert "--obs-pipeline-progress-features" in commands["pipeline_assembly"]
    assert "--obs-pipeline-shared-feedback" in commands["pipeline_assembly"]
    assert commands["pipeline_assembly"][commands["pipeline_assembly"].index("--obs-memory-mode") + 1] == "egocentric"
    assert commands["signal_hunt"].count("--obs-memory-mode") == 1
    assert "--obs-signal-confidence-features" in commands["signal_hunt"]
    assert "--obs-signal-sector-features" in commands["signal_hunt"]
    assert "--eval-pipeline-navigation-assist" not in commands["signal_hunt"]
    assert "--eval-pipeline-navigation-assist-trust-messages" not in commands["signal_hunt"]
    assert "--eval-pipeline-station-interact-guard" not in commands["signal_hunt"]
    assert "--eval-pipeline-option-allow-interact" not in commands["signal_hunt"]
    assert "--eval-pipeline-interact-gate-threshold" not in commands["signal_hunt"]
    assert "--eval-pipeline-interact-gate-promote" not in commands["signal_hunt"]
    assert "--eval-pipeline-event-head-threshold" not in commands["signal_hunt"]
    assert "--eval-pipeline-plan-head-threshold" not in commands["signal_hunt"]
    assert "--eval-pipeline-navigation-head-threshold" not in commands["signal_hunt"]
    assert "--eval-pipeline-option-threshold" not in commands["signal_hunt"]
    assert "--rl-rollout-pipeline-navigation-assist" not in commands["signal_hunt"]
    assert "--rl-rollout-pipeline-navigation-assist-trust-messages" not in commands["signal_hunt"]
    assert "--rl-rollout-pipeline-station-interact-guard" not in commands["signal_hunt"]
    assert "--rl-rollout-pipeline-interact-gate-promote" not in commands["signal_hunt"]
    assert "--pipeline-assisted-rollout-episodes" not in commands["signal_hunt"]
    assert "--obs-pipeline-feedback" not in commands["signal_hunt"]
    assert "--obs-pipeline-feedback-metadata" not in commands["signal_hunt"]
    assert "--obs-pipeline-progress-features" not in commands["signal_hunt"]
    assert "--obs-pipeline-shared-feedback" not in commands["signal_hunt"]


def test_core_training_sweep_recurrent_standard_profile_keeps_pipeline_gate_off(tmp_path):
    from examples.core_training_sweep import parse_args, run_suite

    args = parse_args([
        "--algorithms",
        "recurrent_bc_rl",
        "--scenarios",
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
        "--recurrent-ppo-profile",
        "standard",
        "--output-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "recurrent-standard",
        "--dry-run",
    ])

    payload = run_suite(args)
    command = payload["runs"][0]["command"]

    assert payload["config"]["recurrent_ppo_profile"] == "standard"
    assert payload["config"]["recurrent_bc_pipeline_pickup_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_delivery_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_delivery_progress_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_navigation_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_sync_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_ready_interact_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_station_guard_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_wrong_station_recovery_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_pickup_gate_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_pickup_gate_pos_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_pickup_gate_neg_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_plan_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_plan_head_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_option_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_message_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_send_gate_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_send_gate_pos_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_send_gate_neg_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_pos_weight"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_neg_weight"] == 1.0
    assert payload["config"]["recurrent_bc_calibrate_pipeline_interact_gate_threshold"] is False
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_threshold_target_rate"] == -1.0
    assert payload["config"]["recurrent_bc_pipeline_bad_pickup_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_bad_drop_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_bad_interact_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_bad_action_margin_loss_weight"] == 0.0
    assert payload["config"]["recurrent_bc_pipeline_bad_action_margin"] == 1.0
    assert payload["config"]["recurrent_bc_pipeline_proactive_bad_action_labels"] is False
    assert payload["config"]["recurrent_obs_pipeline_progress_features"] is False
    assert payload["config"]["recurrent_pipeline_wrong_delivery_penalty"] == 0.25
    assert payload["config"]["recurrent_rl_pipeline_delivery_progress_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_rl_pipeline_navigation_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_rl_pipeline_sync_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_rl_pipeline_assisted_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_rl_pipeline_ready_interact_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_rl_pipeline_wrong_station_recovery_action_loss_weight"] == 0.0
    assert payload["config"]["recurrent_dagger_focus_replay"] is False
    assert payload["config"]["recurrent_dagger_retrain_from_scratch"] is True
    assert payload["config"]["recurrent_dagger_pipeline_wrong_delivery_provenance_labels"] is False
    assert payload["config"]["recurrent_dagger_pipeline_wrong_delivery_provenance_weight"] == -1.0
    assert payload["config"]["recurrent_dagger_replay_pre_steps"] == 2
    assert payload["config"]["recurrent_dagger_replay_post_steps"] == 2
    assert payload["config"]["recurrent_dagger_replay_weight"] == 1.0
    assert payload["config"]["recurrent_dagger_positive_replay_events"] == ""
    assert payload["config"]["recurrent_dagger_replay_event_weights"] == ""
    assert payload["config"]["recurrent_dagger_replay_event_caps"] == ""
    assert payload["config"]["recurrent_dagger_replay_success_only_events"] == ""
    assert payload["config"]["recurrent_dagger_replay_priority_events"] == ""
    assert payload["config"]["recurrent_dagger_replay_balance_positive_events"] == ""
    assert payload["config"]["recurrent_dagger_replay_balance_negative_events"] == ""
    assert payload["config"]["recurrent_dagger_replay_max_negative_per_positive"] == -1.0
    assert payload["config"]["recurrent_dagger_max_replay_snippets_per_episode"] == 4
    assert payload["config"]["recurrent_dagger_max_failed_parent_replay_snippets_per_episode"] == -1
    assert payload["config"]["recurrent_dagger_failed_parent_replay_weight_scale"] == 1.0
    assert payload["config"]["recurrent_dagger_expert_max_replay_snippets_per_episode"] == -1
    assert payload["config"]["recurrent_eval_pipeline_interact_gate_threshold"] == -1.0
    assert payload["config"]["recurrent_eval_pipeline_event_head_threshold"] == -1.0
    assert "--bc-calibrate-pipeline-interact-gate-threshold" not in command
    assert "--bc-pipeline-proactive-bad-action-labels" not in command
    assert "--eval-pipeline-interact-gate-threshold" not in command
    assert "--dagger-focus-replay" not in command
    assert "--no-dagger-retrain-from-scratch" not in command
    assert "--dagger-pipeline-wrong-delivery-provenance-labels" not in command
    assert command[command.index("--dagger-positive-replay-events") + 1] == ""
    assert command[command.index("--dagger-max-replay-snippets-per-episode") + 1] == "4"


def test_core_training_sweep_pipeline_feedback_implies_parent_feedback(tmp_path):
    from examples.core_training_sweep import parse_args, run_suite

    args = parse_args([
        "--algorithms",
        "recurrent_bc_rl",
        "--scenarios",
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
        "--recurrent-obs-pipeline-shared-feedback",
        "--output-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "recurrent-pipeline-feedback",
        "--dry-run",
    ])

    payload = run_suite(args)
    command = payload["runs"][0]["command"]

    assert "--obs-feedback" in command
    assert "--obs-pipeline-feedback" in command
    assert "--obs-pipeline-feedback-metadata" in command
    assert "--obs-pipeline-shared-feedback" in command


def test_core_training_sweep_recurrent_eval_seed_range_sets_audit_panel_defaults(tmp_path):
    from examples.core_training_sweep import parse_args, run_suite

    args = parse_args([
        "--algorithms",
        "recurrent_bc_rl",
        "--scenarios",
        "pipeline_assembly",
        "--updates",
        "6",
        "--rollout-steps",
        "16",
        "--eval-every",
        "1",
        "--eval-episodes",
        "40",
        "--seeds",
        "0",
        "--recurrent-eval-seed-range",
        "3000:4",
        "--output-dir",
        str(tmp_path / "runs"),
        "--run-name",
        "recurrent-audit-panel",
        "--dry-run",
    ])

    payload = run_suite(args)
    command = payload["runs"][0]["command"]

    assert payload["config"]["eval_episodes"] == 40
    assert payload["config"]["recurrent_eval_episodes"] == 1
    assert payload["config"]["recurrent_rl_eval_episodes"] == 1
    assert payload["config"]["recurrent_rl_eval_use_eval_seeds"] is True
    assert payload["config"]["recurrent_eval_seed_range"] == "3000:4"
    assert payload["config"]["recurrent_eval_seed_list"] == "3000,3001,3002,3003"
    assert command[command.index("--eval-episodes") + 1] == "1"
    assert command[command.index("--rl-eval-episodes") + 1] == "1"
    assert command[command.index("--eval-seed-list") + 1] == "3000,3001,3002,3003"
    assert "--rl-eval-use-eval-seeds" in command


def test_core_training_sweep_recurrent_eval_seed_range_validation():
    from examples.core_training_sweep import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--recurrent-eval-seed-range", "3000"])
    with pytest.raises(SystemExit):
        parse_args(["--recurrent-eval-seed-range", "3000:0"])
    with pytest.raises(SystemExit):
        parse_args([
            "--recurrent-eval-seed-range",
            "3000:4",
            "--recurrent-eval-seed-list",
            "3000,3001,3002,3003",
        ])


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
        "--recurrent-bc-pipeline-pickup-action-loss-weight",
        "0.25",
        "--recurrent-bc-pipeline-delivery-action-loss-weight",
        "0.5",
        "--recurrent-bc-pipeline-delivery-progress-action-loss-weight",
        "0.8",
        "--recurrent-bc-pipeline-navigation-action-loss-weight",
        "1.2",
        "--recurrent-bc-pipeline-frontier-exploration-action-loss-weight",
        "0.7",
        "--recurrent-bc-pipeline-frontier-exploration-min-map-size",
        "8",
            "--recurrent-bc-pipeline-sync-action-loss-weight",
            "1.3",
            "--recurrent-bc-pipeline-ready-interact-action-loss-weight",
            "1.45",
            "--recurrent-bc-pipeline-station-guard-action-loss-weight",
            "1.1",
        "--recurrent-bc-pipeline-wrong-station-recovery-action-loss-weight",
        "1.6",
        "--recurrent-bc-pipeline-pickup-gate-loss-weight",
        "0.9",
        "--recurrent-bc-pipeline-pickup-gate-pos-weight",
        "2.2",
        "--recurrent-bc-pipeline-pickup-gate-neg-weight",
        "1.3",
        "--recurrent-bc-pipeline-plan-action-loss-weight",
        "1.75",
        "--recurrent-bc-pipeline-plan-head-loss-weight",
        "1.9",
        "--recurrent-bc-pipeline-option-loss-weight",
        "2.1",
        "--recurrent-bc-pipeline-message-loss-weight",
        "2.25",
        "--recurrent-bc-pipeline-send-gate-loss-weight",
        "1.5",
        "--recurrent-bc-pipeline-send-gate-pos-weight",
        "2.0",
        "--recurrent-bc-pipeline-send-gate-neg-weight",
        "1.25",
        "--recurrent-bc-pipeline-interact-gate-loss-weight",
        "1.4",
        "--recurrent-bc-pipeline-interact-gate-pos-weight",
        "2.5",
        "--recurrent-bc-pipeline-interact-gate-neg-weight",
        "1.1",
        "--recurrent-bc-calibrate-pipeline-interact-gate-threshold",
        "--recurrent-bc-pipeline-interact-gate-threshold-target-rate",
        "0.33",
        "--recurrent-bc-pipeline-bad-pickup-action-loss-weight",
        "0.6",
        "--recurrent-bc-pipeline-bad-drop-action-loss-weight",
        "0.75",
        "--recurrent-bc-pipeline-bad-interact-action-loss-weight",
        "1.25",
        "--recurrent-bc-pipeline-bad-action-margin-loss-weight",
        "1.35",
        "--recurrent-bc-pipeline-bad-action-margin",
        "0.8",
        "--recurrent-bc-pipeline-proactive-bad-action-labels",
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
        "--recurrent-eval-pipeline-interact-gate-threshold",
        "0.42",
        "--recurrent-eval-pipeline-event-head-threshold",
        "0.44",
        "--recurrent-eval-signal-target-scan-lock",
        "--recurrent-eval-signal-exact-target-scan-lock",
        "--recurrent-eval-signal-compatible-target-scan-assist",
        "--recurrent-eval-signal-compatible-target-scan-min-strength",
        "4",
        "--recurrent-eval-signal-negative-memory-scan-guard",
        "--recurrent-eval-signal-target-probe-assist",
        "--recurrent-eval-signal-constraint-message-copy-assist",
        "--recurrent-eval-signal-constraint-message-guard",
        "--recurrent-eval-signal-exact-target-message-copy-assist",
        "--recurrent-eval-signal-frontier-exploration-assist",
        "--recurrent-eval-signal-scan-refresh-assist",
        "--recurrent-eval-signal-scan-refresh-threshold",
        "0.35",
        "--no-recurrent-obs-pipeline-features",
        "--recurrent-bc-calibrate-send-threshold",
        "--recurrent-bc-send-threshold-target-rate",
        "0.15",
        "--recurrent-bc-comm-send-pos-weight",
        "-1",
        "--recurrent-bc-comm-send-rate-penalty-weight",
        "0.25",
        "--recurrent-bc-comm-send-rate-target",
        "0.15",
        "--recurrent-bc-signal-target-aux-weight",
        "0.45",
        "--recurrent-bc-signal-target-hypothesis-loss-weight",
        "0.55",
        "--recurrent-bc-signal-target-hypothesis-commit-loss-weight",
        "0.25",
        "--recurrent-bc-signal-target-hypothesis-ambiguity-loss-weight",
        "0.5",
        "--recurrent-bc-signal-target-hypothesis-xy-loss-weight",
        "1.75",
        "--recurrent-bc-signal-target-hypothesis-min-map-size",
        "8",
        "--recurrent-bc-signal-constraint-message-loss-weight",
        "1.25",
        "--recurrent-bc-signal-target-pursuit-action-weight",
        "0.6",
        "--recurrent-bc-signal-target-pursuit-trust-exact-memory",
        "--recurrent-bc-signal-target-pursuit-max-agents",
        "1",
        "--recurrent-bc-signal-sync-response-action-loss-weight",
        "0.7",
        "--recurrent-bc-signal-active-scan-response-action-weight",
        "0.65",
        "--recurrent-bc-signal-active-scan-response-min-map-size",
        "8",
        "--recurrent-bc-signal-active-scan-response-max-agents",
        "2",
        "--recurrent-bc-signal-scan-bridge-action-weight",
        "0.44",
        "--recurrent-bc-signal-scan-bridge-min-map-size",
        "8",
        "--recurrent-bc-signal-scan-bridge-remaining-threshold",
        "0.25",
        "--recurrent-bc-signal-scan-bridge-max-teammate-distance",
        "5",
        "--recurrent-bc-signal-target-match-action-weight",
        "0.8",
        "--recurrent-bc-signal-first-target-scan-action-weight",
        "0.9",
        "--recurrent-bc-signal-refresh-target-scan-action-weight",
        "0.4",
        "--recurrent-bc-signal-joint-target-scan-action-weight",
        "0.5",
        "--recurrent-bc-signal-target-opportunity-action-weight",
        "0.3",
        "--recurrent-bc-signal-redundant-target-wait-action-loss-weight",
        "0.2",
        "--recurrent-bc-signal-constraint-frontier-bias",
        "--recurrent-bc-signal-scan-decision-loss-weight",
        "1.4",
        "--recurrent-bc-signal-scan-decision-pos-weight",
        "2.4",
        "--recurrent-bc-signal-scan-decision-neg-weight",
        "3.4",
        "--recurrent-bc-signal-scan-gate-loss-weight",
        "1.5",
        "--recurrent-bc-signal-scan-gate-pos-weight",
        "2.5",
        "--recurrent-bc-signal-scan-gate-neg-weight",
        "3.5",
        "--recurrent-bc-signal-target-validity-loss-weight",
        "1.6",
        "--recurrent-bc-signal-target-validity-pos-weight",
        "2.6",
        "--recurrent-bc-signal-target-validity-neg-weight",
        "3.6",
        "--recurrent-bc-signal-target-decision-loss-weight",
        "1.7",
        "--recurrent-bc-signal-target-decision-pos-weight",
        "2.7",
        "--recurrent-bc-signal-target-decision-neg-weight",
        "3.7",
        "--recurrent-bc-signal-ambiguous-target-decision-negatives",
        "--recurrent-bc-signal-ambiguous-target-decision-min-map-size",
        "8",
        "--recurrent-bc-signal-ambiguous-target-search-labels",
        "--recurrent-bc-signal-ambiguous-target-search-min-map-size",
        "8",
        "--recurrent-bc-signal-clue-interact-action-weight",
        "0.42",
        "--recurrent-bc-signal-clue-interact-min-map-size",
        "8",
        "--recurrent-bc-signal-visible-clue-action-weight",
        "0.35",
        "--recurrent-bc-signal-visible-clue-min-map-size",
        "8",
        "--recurrent-bc-signal-evidence-sweep-action-weight",
        "0.45",
        "--recurrent-bc-signal-evidence-sweep-min-map-size",
        "8",
        "--recurrent-bc-signal-frontier-exploration-action-weight",
        "0.55",
        "--recurrent-bc-signal-frontier-exploration-min-map-size",
        "8",
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
        "--recurrent-dagger-target-handoff-requires-exact-target",
        "--recurrent-dagger-signal-target-rendezvous-labels",
        "--recurrent-dagger-signal-target-rendezvous-min-map-size",
        "16",
        "--recurrent-dagger-signal-target-rendezvous-max-agents",
        "2",
        "--recurrent-dagger-focus-events",
        "pipeline_wrong_delivery,pipeline_bad_pickup",
        "--recurrent-dagger-focus-error-weight",
        "4.5",
        "--recurrent-dagger-focus-recovery-weight",
        "2.5",
        "--recurrent-dagger-focus-window",
        "3",
        "--recurrent-dagger-focus-replay",
        "--no-recurrent-dagger-retrain-from-scratch",
        "--no-recurrent-dagger-restore-best",
        "--recurrent-dagger-pipeline-wrong-delivery-provenance-labels",
        "--recurrent-dagger-pipeline-wrong-delivery-provenance-weight",
        "1.75",
        "--recurrent-dagger-replay-pre-steps",
        "1",
        "--recurrent-dagger-replay-post-steps",
        "4",
        "--recurrent-dagger-replay-weight",
        "2.25",
        "--recurrent-dagger-positive-replay-events",
        "delivered,stage_completed",
        "--recurrent-dagger-replay-event-weights",
        "pipeline_wrong_delivery:5.0,delivered:2.0",
        "--recurrent-dagger-replay-event-caps",
        "pipeline_wrong_delivery:2",
        "--recurrent-dagger-replay-success-only-events",
        "delivered",
        "--recurrent-dagger-replay-priority-events",
        "pipeline_wrong_delivery",
        "--recurrent-dagger-replay-balance-positive-events",
        "delivered",
        "--recurrent-dagger-replay-balance-negative-events",
        "pipeline_wrong_delivery",
        "--recurrent-dagger-replay-max-negative-per-positive",
        "1.5",
        "--recurrent-dagger-max-replay-snippets-per-episode",
        "7",
        "--recurrent-dagger-max-failed-parent-replay-snippets-per-episode",
        "3",
        "--recurrent-dagger-failed-parent-replay-weight-scale",
        "0.5",
        "--recurrent-dagger-expert-max-replay-snippets-per-episode",
        "2",
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
        "--recurrent-rl-pipeline-bad-pickup-penalty",
        "0.2",
        "--recurrent-rl-pipeline-bad-interact-penalty",
        "0.15",
        "--recurrent-rl-pipeline-unneeded-drop-bonus",
        "0.075",
        "--recurrent-rl-eval-decoding-action-loss-weight",
        "0.35",
        "--recurrent-rl-pipeline-assisted-action-loss-weight",
        "0.92",
        "--recurrent-rl-pipeline-interact-gate-loss-weight",
        "0.55",
        "--recurrent-rl-pipeline-interact-gate-pos-weight",
        "2.0",
        "--recurrent-rl-pipeline-interact-gate-neg-weight",
        "3.0",
        "--recurrent-rl-pipeline-pickup-gate-loss-weight",
        "0.75",
        "--recurrent-rl-pipeline-pickup-gate-pos-weight",
        "2.5",
        "--recurrent-rl-pipeline-pickup-gate-neg-weight",
        "3.5",
        "--recurrent-rl-pipeline-delivery-progress-action-loss-weight",
        "0.6",
        "--recurrent-rl-pipeline-navigation-action-loss-weight",
        "0.7",
        "--recurrent-rl-pipeline-sync-action-loss-weight",
        "0.7",
        "--recurrent-rl-pipeline-ready-interact-action-loss-weight",
        "0.95",
        "--recurrent-rl-pipeline-station-guard-action-loss-weight",
        "0.45",
        "--recurrent-rl-pipeline-wrong-station-recovery-action-loss-weight",
        "0.85",
        "--recurrent-rl-pipeline-plan-action-loss-weight",
        "0.65",
        "--recurrent-rl-pipeline-plan-head-loss-weight",
        "0.72",
        "--recurrent-rl-pipeline-option-loss-weight",
        "0.82",
        "--recurrent-rl-early-stop-eval-patience",
        "2",
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
        "--recurrent-skip-bc",
        "--recurrent-init-template",
        "logs/{scenario}/seed{seed}/{algorithm}_{map_size}x.pt",
        "--recurrent-init-for-dagger",
        "--recurrent-backbone",
        "residual_mlp",
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
    assert payload["config"]["recurrent_obs_exploration_age"] is False
    assert payload["config"]["recurrent_bc_action_class_balance"] is True
    assert payload["config"]["recurrent_bc_action_class_balance_max_weight"] == 4.0
    assert payload["config"]["recurrent_bc_event_action_weight"] == 6.0
    assert payload["config"]["recurrent_bc_event_action_events"] == "delivered,sync_complete"
    assert payload["config"]["recurrent_bc_pipeline_pickup_action_loss_weight"] == 0.25
    assert payload["config"]["recurrent_bc_pipeline_delivery_action_loss_weight"] == 0.5
    assert payload["config"]["recurrent_bc_pipeline_delivery_progress_action_loss_weight"] == 0.8
    assert payload["config"]["recurrent_bc_pipeline_navigation_action_loss_weight"] == 1.2
    assert payload["config"]["recurrent_bc_pipeline_frontier_exploration_action_loss_weight"] == 0.7
    assert payload["config"]["recurrent_bc_pipeline_frontier_exploration_min_map_size"] == 8
    assert payload["config"]["recurrent_bc_pipeline_sync_action_loss_weight"] == 1.3
    assert payload["config"]["recurrent_bc_pipeline_ready_interact_action_loss_weight"] == 1.45
    assert payload["config"]["recurrent_bc_pipeline_station_guard_action_loss_weight"] == 1.1
    assert payload["config"]["recurrent_bc_pipeline_wrong_station_recovery_action_loss_weight"] == 1.6
    assert payload["config"]["recurrent_bc_pipeline_pickup_gate_loss_weight"] == 0.9
    assert payload["config"]["recurrent_bc_pipeline_pickup_gate_pos_weight"] == 2.2
    assert payload["config"]["recurrent_bc_pipeline_pickup_gate_neg_weight"] == 1.3
    assert payload["config"]["recurrent_bc_pipeline_plan_action_loss_weight"] == 1.75
    assert payload["config"]["recurrent_bc_pipeline_plan_head_loss_weight"] == 1.9
    assert payload["config"]["recurrent_bc_pipeline_option_loss_weight"] == 2.1
    assert payload["config"]["recurrent_bc_pipeline_message_loss_weight"] == 2.25
    assert payload["config"]["recurrent_bc_pipeline_send_gate_loss_weight"] == 1.5
    assert payload["config"]["recurrent_bc_pipeline_send_gate_pos_weight"] == 2.0
    assert payload["config"]["recurrent_bc_pipeline_send_gate_neg_weight"] == 1.25
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_loss_weight"] == 1.4
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_pos_weight"] == 2.5
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_neg_weight"] == 1.1
    assert payload["config"]["recurrent_bc_calibrate_pipeline_interact_gate_threshold"] is True
    assert payload["config"]["recurrent_bc_pipeline_interact_gate_threshold_target_rate"] == 0.33
    assert payload["config"]["recurrent_bc_pipeline_bad_pickup_action_loss_weight"] == 0.6
    assert payload["config"]["recurrent_bc_pipeline_bad_drop_action_loss_weight"] == 0.75
    assert payload["config"]["recurrent_bc_pipeline_bad_interact_action_loss_weight"] == 1.25
    assert payload["config"]["recurrent_bc_pipeline_bad_action_margin_loss_weight"] == 1.35
    assert payload["config"]["recurrent_bc_pipeline_bad_action_margin"] == 0.8
    assert payload["config"]["recurrent_bc_pipeline_proactive_bad_action_labels"] is True
    assert payload["config"]["recurrent_pipeline_stage_count"] == 2
    assert payload["config"]["recurrent_pipeline_required_per_stage_min"] == 1
    assert payload["config"]["recurrent_pipeline_required_per_stage_max"] == 1
    assert payload["config"]["recurrent_pipeline_sync_probability"] == 0.0
    assert payload["config"]["recurrent_pipeline_dependency_probability"] == 0.0
    assert payload["config"]["recurrent_obs_pipeline_features"] is False
    assert payload["config"]["recurrent_eval_pipeline_navigation_assist"] is False
    assert payload["config"]["recurrent_eval_pipeline_navigation_assist_trust_messages"] is False
    assert payload["config"]["recurrent_eval_pipeline_station_interact_guard"] is False
    assert payload["config"]["recurrent_eval_pipeline_interact_gate_threshold"] == 0.42
    assert payload["config"]["recurrent_eval_pipeline_event_head_threshold"] == 0.44
    assert payload["config"]["recurrent_eval_pipeline_plan_head_threshold"] == -1.0
    assert payload["config"]["recurrent_eval_pipeline_navigation_head_threshold"] == -1.0
    assert payload["config"]["recurrent_eval_pipeline_option_threshold"] == -1.0
    assert payload["config"]["recurrent_eval_pipeline_option_allow_interact"] is False
    assert payload["config"]["recurrent_eval_signal_target_scan_lock"] is True
    assert payload["config"]["recurrent_eval_signal_exact_target_scan_lock"] is True
    assert payload["config"]["recurrent_eval_signal_compatible_target_scan_assist"] is True
    assert payload["config"]["recurrent_eval_signal_compatible_target_scan_min_strength"] == 4
    assert payload["config"]["recurrent_eval_signal_negative_memory_scan_guard"] is True
    assert payload["config"]["recurrent_eval_signal_target_probe_assist"] is True
    assert payload["config"]["recurrent_eval_signal_frontier_exploration_assist"] is True
    assert payload["config"]["recurrent_eval_signal_scan_refresh_assist"] is True
    assert payload["config"]["recurrent_eval_signal_scan_refresh_threshold"] == 0.35
    assert payload["config"]["recurrent_eval_signal_constraint_message_copy_assist"] is True
    assert payload["config"]["recurrent_eval_signal_constraint_message_guard"] is True
    assert payload["config"]["recurrent_eval_signal_exact_target_message_copy_assist"] is True
    assert payload["config"]["recurrent_bc_calibrate_send_threshold"] is True
    assert payload["config"]["recurrent_bc_send_threshold_target_rate"] == 0.15
    assert payload["config"]["recurrent_bc_comm_send_pos_weight"] == -1
    assert payload["config"]["recurrent_bc_comm_send_rate_penalty_weight"] == 0.25
    assert payload["config"]["recurrent_bc_comm_send_rate_target"] == 0.15
    assert payload["config"]["recurrent_bc_signal_target_aux_weight"] == 0.45
    assert payload["config"]["recurrent_bc_signal_target_hypothesis_loss_weight"] == 0.55
    assert payload["config"]["recurrent_bc_signal_target_hypothesis_commit_loss_weight"] == 0.25
    assert payload["config"]["recurrent_bc_signal_target_hypothesis_ambiguity_loss_weight"] == 0.5
    assert payload["config"]["recurrent_bc_signal_target_hypothesis_xy_loss_weight"] == 1.75
    assert payload["config"]["recurrent_bc_signal_target_hypothesis_min_map_size"] == 8
    assert payload["config"]["recurrent_bc_signal_constraint_message_loss_weight"] == 1.25
    assert payload["config"]["recurrent_bc_signal_target_pursuit_action_weight"] == 0.6
    assert payload["config"]["recurrent_bc_signal_target_pursuit_trust_exact_memory"] is True
    assert payload["config"]["recurrent_bc_signal_target_pursuit_max_agents"] == 1
    assert payload["config"]["recurrent_bc_signal_sync_response_action_loss_weight"] == 0.7
    assert payload["config"]["recurrent_bc_signal_active_scan_response_action_weight"] == 0.65
    assert payload["config"]["recurrent_bc_signal_active_scan_response_min_map_size"] == 8
    assert payload["config"]["recurrent_bc_signal_active_scan_response_max_agents"] == 2
    assert payload["config"]["recurrent_bc_signal_scan_bridge_action_weight"] == 0.44
    assert payload["config"]["recurrent_bc_signal_scan_bridge_min_map_size"] == 8
    assert payload["config"]["recurrent_bc_signal_scan_bridge_remaining_threshold"] == 0.25
    assert payload["config"]["recurrent_bc_signal_scan_bridge_max_teammate_distance"] == 5
    assert payload["config"]["recurrent_bc_signal_target_match_action_weight"] == 0.8
    assert payload["config"]["recurrent_bc_signal_first_target_scan_action_weight"] == 0.9
    assert payload["config"]["recurrent_bc_signal_refresh_target_scan_action_weight"] == 0.4
    assert payload["config"]["recurrent_bc_signal_joint_target_scan_action_weight"] == 0.5
    assert payload["config"]["recurrent_bc_signal_target_opportunity_action_weight"] == 0.3
    assert payload[
        "config"
    ]["recurrent_bc_signal_redundant_target_wait_action_loss_weight"] == 0.2
    assert payload["config"]["recurrent_bc_signal_constraint_frontier_bias"] is True
    assert payload["config"]["recurrent_bc_signal_scan_decision_loss_weight"] == 1.4
    assert payload["config"]["recurrent_bc_signal_scan_decision_pos_weight"] == 2.4
    assert payload["config"]["recurrent_bc_signal_scan_decision_neg_weight"] == 3.4
    assert payload["config"]["recurrent_bc_signal_scan_gate_loss_weight"] == 1.5
    assert payload["config"]["recurrent_bc_signal_scan_gate_pos_weight"] == 2.5
    assert payload["config"]["recurrent_bc_signal_scan_gate_neg_weight"] == 3.5
    assert payload["config"]["recurrent_bc_signal_target_validity_loss_weight"] == 1.6
    assert payload["config"]["recurrent_bc_signal_target_validity_pos_weight"] == 2.6
    assert payload["config"]["recurrent_bc_signal_target_validity_neg_weight"] == 3.6
    assert payload["config"]["recurrent_bc_signal_target_decision_loss_weight"] == 1.7
    assert payload["config"]["recurrent_bc_signal_target_decision_pos_weight"] == 2.7
    assert payload["config"]["recurrent_bc_signal_target_decision_neg_weight"] == 3.7
    assert command[command.index("--bc-signal-target-hypothesis-loss-weight") + 1] == "0.55"
    assert command[command.index("--bc-signal-target-hypothesis-commit-loss-weight") + 1] == "0.25"
    assert command[command.index("--bc-signal-target-hypothesis-ambiguity-loss-weight") + 1] == "0.5"
    assert command[command.index("--bc-signal-target-hypothesis-xy-loss-weight") + 1] == "1.75"
    assert command[command.index("--bc-signal-target-hypothesis-min-map-size") + 1] == "8"
    assert command[command.index("--bc-signal-scan-decision-loss-weight") + 1] == "1.4"
    assert command[command.index("--bc-signal-scan-decision-pos-weight") + 1] == "2.4"
    assert command[command.index("--bc-signal-scan-decision-neg-weight") + 1] == "3.4"
    assert command[command.index("--bc-signal-scan-gate-loss-weight") + 1] == "1.5"
    assert command[command.index("--bc-signal-scan-gate-pos-weight") + 1] == "2.5"
    assert command[command.index("--bc-signal-scan-gate-neg-weight") + 1] == "3.5"
    assert command[command.index("--bc-signal-target-validity-loss-weight") + 1] == "1.6"
    assert command[command.index("--bc-signal-target-validity-pos-weight") + 1] == "2.6"
    assert command[command.index("--bc-signal-target-validity-neg-weight") + 1] == "3.6"
    assert command[command.index("--bc-signal-target-decision-loss-weight") + 1] == "1.7"
    assert command[command.index("--bc-signal-target-decision-pos-weight") + 1] == "2.7"
    assert command[command.index("--bc-signal-target-decision-neg-weight") + 1] == "3.7"
    assert "--bc-signal-ambiguous-target-decision-negatives" in command
    assert command[command.index("--bc-signal-ambiguous-target-decision-min-map-size") + 1] == "8"
    assert "--bc-signal-ambiguous-target-search-labels" in command
    assert command[command.index("--bc-signal-ambiguous-target-search-min-map-size") + 1] == "8"
    assert command[command.index("--bc-signal-target-pursuit-max-agents") + 1] == "1"
    assert "--bc-signal-constraint-frontier-bias" in command
    assert command[command.index("--bc-signal-constraint-message-loss-weight") + 1] == "1.25"
    assert "--dagger-signal-target-rendezvous-labels" in command
    assert command[command.index("--dagger-signal-target-rendezvous-min-map-size") + 1] == "16"
    assert command[command.index("--dagger-signal-target-rendezvous-max-agents") + 1] == "2"
    assert payload["config"]["recurrent_bc_signal_clue_interact_action_weight"] == 0.42
    assert payload["config"]["recurrent_bc_signal_clue_interact_min_map_size"] == 8
    assert command[command.index("--bc-signal-clue-interact-action-weight") + 1] == "0.42"
    assert command[command.index("--bc-signal-clue-interact-min-map-size") + 1] == "8"
    assert payload["config"]["recurrent_bc_signal_visible_clue_action_weight"] == 0.35
    assert payload["config"]["recurrent_bc_signal_visible_clue_min_map_size"] == 8
    assert payload["config"]["recurrent_bc_signal_evidence_sweep_action_weight"] == 0.45
    assert payload["config"]["recurrent_bc_signal_evidence_sweep_min_map_size"] == 8
    assert payload["config"]["recurrent_bc_signal_frontier_exploration_action_weight"] == 0.55
    assert payload["config"]["recurrent_bc_signal_frontier_exploration_min_map_size"] == 8
    assert command[command.index("--bc-signal-evidence-sweep-action-weight") + 1] == "0.45"
    assert command[command.index("--bc-signal-evidence-sweep-min-map-size") + 1] == "8"
    assert payload["config"]["recurrent_bc_signal_ambiguous_target_decision_negatives"] is True
    assert payload["config"]["recurrent_bc_signal_ambiguous_target_decision_min_map_size"] == 8
    assert payload["config"]["recurrent_bc_signal_ambiguous_target_search_labels"] is True
    assert payload["config"]["recurrent_bc_signal_ambiguous_target_search_min_map_size"] == 8
    assert payload["config"]["recurrent_dagger_failed_effective_ratio_cap"] == 0.5
    assert payload["config"]["recurrent_dagger_oracle_action_rollin_rate"] == 0.4
    assert payload["config"]["recurrent_dagger_oracle_message_rollin_rate"] == 0.3
    assert payload["config"]["recurrent_dagger_target_handoff_requires_exact_target"] is True
    assert payload["config"]["recurrent_dagger_signal_target_rendezvous_labels"] is True
    assert payload["config"]["recurrent_dagger_signal_target_rendezvous_min_map_size"] == 16
    assert payload["config"]["recurrent_dagger_signal_target_rendezvous_max_agents"] == 2
    assert payload["config"]["recurrent_dagger_focus_events"] == "pipeline_wrong_delivery,pipeline_bad_pickup"
    assert payload["config"]["recurrent_dagger_focus_error_weight"] == 4.5
    assert payload["config"]["recurrent_dagger_focus_recovery_weight"] == 2.5
    assert payload["config"]["recurrent_dagger_focus_window"] == 3
    assert payload["config"]["recurrent_dagger_focus_replay"] is True
    assert payload["config"]["recurrent_dagger_retrain_from_scratch"] is False
    assert payload["config"]["recurrent_dagger_restore_best"] is False
    assert payload["config"]["recurrent_dagger_pipeline_wrong_delivery_provenance_labels"] is True
    assert payload["config"]["recurrent_dagger_pipeline_wrong_delivery_provenance_weight"] == 1.75
    assert payload["config"]["recurrent_dagger_replay_pre_steps"] == 1
    assert payload["config"]["recurrent_dagger_replay_post_steps"] == 4
    assert payload["config"]["recurrent_dagger_replay_weight"] == 2.25
    assert payload["config"]["recurrent_dagger_positive_replay_events"] == "delivered,stage_completed"
    assert payload["config"]["recurrent_dagger_replay_event_weights"] == (
        "pipeline_wrong_delivery:5.0,delivered:2.0"
    )
    assert payload["config"]["recurrent_dagger_replay_event_caps"] == "pipeline_wrong_delivery:2"
    assert payload["config"]["recurrent_dagger_replay_success_only_events"] == "delivered"
    assert payload["config"]["recurrent_dagger_replay_priority_events"] == "pipeline_wrong_delivery"
    assert payload["config"]["recurrent_dagger_replay_balance_positive_events"] == "delivered"
    assert payload["config"]["recurrent_dagger_replay_balance_negative_events"] == "pipeline_wrong_delivery"
    assert payload["config"]["recurrent_dagger_replay_max_negative_per_positive"] == 1.5
    assert payload["config"]["recurrent_dagger_max_replay_snippets_per_episode"] == 7
    assert payload["config"]["recurrent_dagger_max_failed_parent_replay_snippets_per_episode"] == 3
    assert payload["config"]["recurrent_dagger_failed_parent_replay_weight_scale"] == 0.5
    assert payload["config"]["recurrent_dagger_expert_max_replay_snippets_per_episode"] == 2
    assert payload["config"]["recurrent_rl_early_stop_eval_patience"] == 2
    assert payload["config"]["recurrent_rl_rollout_pipeline_navigation_assist"] is False
    assert payload["config"]["recurrent_rl_rollout_pipeline_navigation_assist_trust_messages"] is False
    assert payload["config"]["recurrent_rl_rollout_pipeline_station_interact_guard"] is True
    assert payload["config"]["recurrent_rl_eval_decoding_action_loss_weight"] == 0.35
    assert payload["config"]["recurrent_rl_pipeline_assisted_action_loss_weight"] == 0.92
    assert payload["config"]["recurrent_rl_pipeline_interact_gate_loss_weight"] == 0.55
    assert payload["config"]["recurrent_rl_pipeline_interact_gate_pos_weight"] == 2.0
    assert payload["config"]["recurrent_rl_pipeline_interact_gate_neg_weight"] == 3.0
    assert payload["config"]["recurrent_rl_pipeline_pickup_gate_loss_weight"] == 0.75
    assert payload["config"]["recurrent_rl_pipeline_pickup_gate_pos_weight"] == 2.5
    assert payload["config"]["recurrent_rl_pipeline_pickup_gate_neg_weight"] == 3.5
    assert payload["config"]["recurrent_rl_pipeline_delivery_progress_action_loss_weight"] == 0.6
    assert payload["config"]["recurrent_rl_pipeline_navigation_action_loss_weight"] == 0.7
    assert payload["config"]["recurrent_rl_pipeline_sync_action_loss_weight"] == 0.7
    assert payload["config"]["recurrent_rl_pipeline_ready_interact_action_loss_weight"] == 0.95
    assert payload["config"]["recurrent_rl_pipeline_station_guard_action_loss_weight"] == 0.45
    assert payload["config"]["recurrent_rl_pipeline_wrong_station_recovery_action_loss_weight"] == 0.85
    assert payload["config"]["recurrent_rl_pipeline_plan_action_loss_weight"] == 0.65
    assert payload["config"]["recurrent_rl_pipeline_plan_head_loss_weight"] == 0.72
    assert payload["config"]["recurrent_rl_pipeline_option_loss_weight"] == 0.82
    assert payload["config"]["recurrent_rl_pipeline_bad_interact_penalty"] == 0.15
    assert payload["config"]["recurrent_rl_pipeline_bad_pickup_penalty"] == 0.2
    assert payload["config"]["recurrent_rl_pipeline_unneeded_drop_bonus"] == 0.075
    assert payload["config"]["recurrent_backbone"] == "residual_mlp"
    assert "--updates" not in command
    assert "--epochs" not in command
    assert "--save-every" not in command
    assert "--rl-updates" in command
    assert command[command.index("--rl-updates") + 1] == "0"
    assert command[command.index("--rl-early-stop-eval-patience") + 1] == "2"
    assert command[command.index("--rl-lr") + 1] == "1e-05"
    assert command[command.index("--clip") + 1] == "0.1"
    assert command[command.index("--entropy-coeff") + 1] == "0.0"
    assert command[command.index("--max-grad-norm") + 1] == "0.25"
    assert command[command.index("--bc-kl-coeff") + 1] == "1.0"
    assert command[command.index("--bc-comm-kl-coeff") + 1] == "1.5"
    assert command[command.index("--rl-pipeline-bad-pickup-penalty") + 1] == "0.2"
    assert command[command.index("--rl-pipeline-bad-interact-penalty") + 1] == "0.15"
    assert command[command.index("--rl-pipeline-unneeded-drop-bonus") + 1] == "0.075"
    assert command[command.index("--rl-eval-decoding-action-loss-weight") + 1] == "0.35"
    assert command[command.index("--rl-pipeline-assisted-action-loss-weight") + 1] == "0.92"
    assert command[command.index("--rl-pipeline-interact-gate-loss-weight") + 1] == "0.55"
    assert command[command.index("--rl-pipeline-interact-gate-pos-weight") + 1] == "2.0"
    assert command[command.index("--rl-pipeline-interact-gate-neg-weight") + 1] == "3.0"
    assert command[command.index("--rl-pipeline-pickup-gate-loss-weight") + 1] == "0.75"
    assert command[command.index("--rl-pipeline-pickup-gate-pos-weight") + 1] == "2.5"
    assert command[command.index("--rl-pipeline-pickup-gate-neg-weight") + 1] == "3.5"
    assert command[
        command.index("--rl-pipeline-delivery-progress-action-loss-weight") + 1
    ] == "0.6"
    assert command[command.index("--rl-pipeline-navigation-action-loss-weight") + 1] == "0.7"
    assert command[command.index("--rl-pipeline-sync-action-loss-weight") + 1] == "0.7"
    assert command[
        command.index("--rl-pipeline-ready-interact-action-loss-weight") + 1
    ] == "0.95"
    assert command[
        command.index("--rl-pipeline-station-guard-action-loss-weight") + 1
    ] == "0.45"
    assert command[
        command.index("--rl-pipeline-wrong-station-recovery-action-loss-weight") + 1
    ] == "0.85"
    assert command[command.index("--rl-pipeline-plan-action-loss-weight") + 1] == "0.65"
    assert command[command.index("--rl-pipeline-plan-head-loss-weight") + 1] == "0.72"
    assert command[command.index("--rl-pipeline-option-loss-weight") + 1] == "0.82"
    assert command[command.index("--recurrent-backbone") + 1] == "residual_mlp"
    assert command[
        command.index("--bc-pipeline-frontier-exploration-action-loss-weight") + 1
    ] == "0.7"
    assert command[
        command.index("--bc-pipeline-frontier-exploration-min-map-size") + 1
    ] == "8"
    assert command[command.index("--bc-signal-target-aux-weight") + 1] == "0.45"
    assert command[command.index("--bc-signal-target-pursuit-action-weight") + 1] == "0.6"
    assert "--bc-signal-target-pursuit-trust-exact-memory" in command
    assert "--eval-signal-frontier-exploration-assist" in command
    assert command[
        command.index("--bc-signal-sync-response-action-loss-weight") + 1
    ] == "0.7"
    assert command[
        command.index("--bc-signal-active-scan-response-action-weight") + 1
    ] == "0.65"
    assert command[
        command.index("--bc-signal-active-scan-response-min-map-size") + 1
    ] == "8"
    assert command[
        command.index("--bc-signal-active-scan-response-max-agents") + 1
    ] == "2"
    assert command[
        command.index("--bc-signal-scan-bridge-action-weight") + 1
    ] == "0.44"
    assert command[
        command.index("--bc-signal-scan-bridge-min-map-size") + 1
    ] == "8"
    assert command[
        command.index("--bc-signal-scan-bridge-remaining-threshold") + 1
    ] == "0.25"
    assert command[
        command.index("--bc-signal-scan-bridge-max-teammate-distance") + 1
    ] == "5"
    assert command[command.index("--bc-signal-target-match-action-weight") + 1] == "0.8"
    assert command[
        command.index("--bc-signal-first-target-scan-action-weight") + 1
    ] == "0.9"
    assert command[
        command.index("--bc-signal-refresh-target-scan-action-weight") + 1
    ] == "0.4"
    assert command[
        command.index("--bc-signal-joint-target-scan-action-weight") + 1
    ] == "0.5"
    assert command[
        command.index("--bc-signal-target-opportunity-action-weight") + 1
    ] == "0.3"
    assert command[
        command.index("--bc-signal-redundant-target-wait-action-loss-weight") + 1
    ] == "0.2"
    assert command[
        command.index("--bc-signal-frontier-exploration-action-weight") + 1
    ] == "0.55"
    assert command[
        command.index("--bc-signal-frontier-exploration-min-map-size") + 1
    ] == "8"
    assert "--eval-signal-target-scan-lock" in command
    assert "--eval-signal-exact-target-scan-lock" in command
    assert "--eval-signal-compatible-target-scan-assist" in command
    assert command[
        command.index("--eval-signal-compatible-target-scan-min-strength") + 1
    ] == "4"
    assert "--eval-signal-negative-memory-scan-guard" in command
    assert "--eval-signal-target-probe-assist" in command
    assert "--eval-signal-constraint-message-copy-assist" in command
    assert "--eval-signal-constraint-message-guard" in command
    assert "--eval-signal-exact-target-message-copy-assist" in command
    assert "--eval-signal-scan-refresh-assist" in command
    assert command[command.index("--eval-signal-scan-refresh-threshold") + 1] == "0.35"
    assert "--dagger-target-handoff-requires-exact-target" in command
    assert command[
        command.index("--bc-pipeline-ready-interact-action-loss-weight") + 1
    ] == "1.45"
    assert "--rl-balanced-rollouts" in command
    assert "--rl-rollout-eval-decoding" in command
    assert "--no-rl-restore-best" in command
    assert "--demo-episodes" in command
    assert command[command.index("--demo-episodes") + 1] == "4"
    assert "--bc-action-class-balance" in command
    assert command[command.index("--bc-action-class-balance-max-weight") + 1] == "4.0"
    assert command[command.index("--bc-event-action-weight") + 1] == "6.0"
    assert command[command.index("--bc-event-action-events") + 1] == "delivered,sync_complete"
    assert command[command.index("--bc-pipeline-pickup-action-loss-weight") + 1] == "0.25"
    assert command[command.index("--bc-pipeline-delivery-action-loss-weight") + 1] == "0.5"
    assert command[command.index("--bc-pipeline-delivery-progress-action-loss-weight") + 1] == "0.8"
    assert command[command.index("--bc-pipeline-navigation-action-loss-weight") + 1] == "1.2"
    assert command[command.index("--bc-pipeline-sync-action-loss-weight") + 1] == "1.3"
    assert command[command.index("--bc-pipeline-station-guard-action-loss-weight") + 1] == "1.1"
    assert (
        command[
            command.index("--bc-pipeline-wrong-station-recovery-action-loss-weight") + 1
        ]
        == "1.6"
    )
    assert command[command.index("--bc-pipeline-pickup-gate-loss-weight") + 1] == "0.9"
    assert command[command.index("--bc-pipeline-pickup-gate-pos-weight") + 1] == "2.2"
    assert command[command.index("--bc-pipeline-pickup-gate-neg-weight") + 1] == "1.3"
    assert command[command.index("--bc-pipeline-plan-action-loss-weight") + 1] == "1.75"
    assert command[command.index("--bc-pipeline-plan-head-loss-weight") + 1] == "1.9"
    assert command[command.index("--bc-pipeline-option-loss-weight") + 1] == "2.1"
    assert command[command.index("--bc-pipeline-message-loss-weight") + 1] == "2.25"
    assert command[command.index("--bc-pipeline-send-gate-loss-weight") + 1] == "1.5"
    assert command[command.index("--bc-pipeline-send-gate-pos-weight") + 1] == "2.0"
    assert command[command.index("--bc-pipeline-send-gate-neg-weight") + 1] == "1.25"
    assert command[command.index("--bc-pipeline-interact-gate-loss-weight") + 1] == "1.4"
    assert command[command.index("--bc-pipeline-interact-gate-pos-weight") + 1] == "2.5"
    assert command[command.index("--bc-pipeline-interact-gate-neg-weight") + 1] == "1.1"
    assert "--bc-calibrate-pipeline-interact-gate-threshold" not in command
    assert command[command.index("--bc-pipeline-interact-gate-threshold-target-rate") + 1] == "0.33"
    assert command[command.index("--bc-pipeline-bad-pickup-action-loss-weight") + 1] == "0.6"
    assert command[command.index("--bc-pipeline-bad-drop-action-loss-weight") + 1] == "0.75"
    assert command[command.index("--bc-pipeline-bad-interact-action-loss-weight") + 1] == "1.25"
    assert command[command.index("--bc-pipeline-bad-action-margin-loss-weight") + 1] == "1.35"
    assert command[command.index("--bc-pipeline-bad-action-margin") + 1] == "0.8"
    assert "--bc-pipeline-proactive-bad-action-labels" not in command
    assert command[command.index("--pipeline-stage-count") + 1] == "2"
    assert command[command.index("--pipeline-required-per-stage-min") + 1] == "1"
    assert command[command.index("--pipeline-required-per-stage-max") + 1] == "1"
    assert command[command.index("--pipeline-sync-probability") + 1] == "0.0"
    assert command[command.index("--pipeline-dependency-probability") + 1] == "0.0"
    assert "--eval-pipeline-interact-gate-threshold" not in command
    assert "--eval-pipeline-event-head-threshold" not in command
    assert "--bc-calibrate-send-threshold" in command
    assert command[command.index("--bc-send-threshold-target-rate") + 1] == "0.15"
    assert command[command.index("--bc-comm-send-pos-weight") + 1] == "-1.0"
    assert command[command.index("--bc-comm-send-rate-penalty-weight") + 1] == "0.25"
    assert command[command.index("--bc-comm-send-rate-target") + 1] == "0.15"
    assert "--dagger-rounds" in command
    assert command[command.index("--dagger-rounds") + 1] == "1"
    assert command[command.index("--dagger-failed-effective-ratio-cap") + 1] == "0.5"
    assert command[command.index("--dagger-oracle-action-rollin-rate") + 1] == "0.4"
    assert command[command.index("--dagger-oracle-message-rollin-rate") + 1] == "0.3"
    assert command[command.index("--dagger-focus-events") + 1] == "pipeline_wrong_delivery,pipeline_bad_pickup"
    assert command[command.index("--dagger-focus-error-weight") + 1] == "4.5"
    assert command[command.index("--dagger-focus-recovery-weight") + 1] == "2.5"
    assert command[command.index("--dagger-focus-window") + 1] == "3"
    assert "--dagger-focus-replay" in command
    assert "--no-dagger-retrain-from-scratch" in command
    assert "--no-dagger-restore-best" in command
    assert "--dagger-pipeline-wrong-delivery-provenance-labels" not in command
    assert "--dagger-pipeline-wrong-delivery-provenance-weight" not in command
    assert command[command.index("--dagger-replay-pre-steps") + 1] == "1"
    assert command[command.index("--dagger-replay-post-steps") + 1] == "4"
    assert command[command.index("--dagger-replay-weight") + 1] == "2.25"
    assert command[command.index("--dagger-positive-replay-events") + 1] == "delivered,stage_completed"
    assert command[command.index("--dagger-replay-event-weights") + 1] == (
        "pipeline_wrong_delivery:5.0,delivered:2.0"
    )
    assert command[command.index("--dagger-replay-event-caps") + 1] == "pipeline_wrong_delivery:2"
    assert command[command.index("--dagger-replay-success-only-events") + 1] == "delivered"
    assert command[command.index("--dagger-replay-priority-events") + 1] == "pipeline_wrong_delivery"
    assert command[command.index("--dagger-replay-balance-positive-events") + 1] == "delivered"
    assert command[command.index("--dagger-replay-balance-negative-events") + 1] == "pipeline_wrong_delivery"
    assert command[command.index("--dagger-replay-max-negative-per-positive") + 1] == "1.5"
    assert command[command.index("--dagger-max-replay-snippets-per-episode") + 1] == "7"
    assert command[command.index("--dagger-max-failed-parent-replay-snippets-per-episode") + 1] == "3"
    assert command[command.index("--dagger-failed-parent-replay-weight-scale") + 1] == "0.5"
    assert command[command.index("--dagger-expert-max-replay-snippets-per-episode") + 1] == "2"
    assert "--train-map-sizes" in command
    assert "--eval-map-sizes" in command
    assert "--eval-seed-list" in command
    assert "--dagger-seed-list" in command
    assert "--skip-bc" in command
    assert "--recurrent-init" in command
    assert payload["config"]["recurrent_skip_bc"] is True
    assert command[command.index("--recurrent-init") + 1] == (
        "logs/signal_hunt/seed0/recurrent_bc_rl_8x.pt"
    )
    assert "--recurrent-init-for-dagger" in command
    assert "--obs-exploration-memory" in command
    assert "--obs-signal-scan-state" in command
    assert "--eval-signal-exact-target-navigation-assist" in command
    assert "--eval-signal-initial-exact-message-copy-assist" in command
    assert "--bc-signal-scan-gate-loss-weight" in command
    assert "--bc-signal-decoy-scan-action-loss-weight" in command
    assert "--eval-signal-target-decision-threshold" in command
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
