"""Smoke tests for training pipelines — verify no crashes or shape mismatches."""
import numpy as np
import torch
import pytest


def test_mappo_dtde_no_comm():
    from syncorsink.train.mappo import train_mappo, MAPPOConfig
    cfg = MAPPOConfig(
        updates=2, rollout_steps=16, epochs=1, minibatch=16,
        eval_every=0, max_steps=20, device="cpu", agents=2,
        critic_mode="local", comm=False,
    )
    train_mappo(cfg)


def test_mappo_ctde_with_comm():
    from syncorsink.train.mappo import train_mappo, MAPPOConfig
    cfg = MAPPOConfig(
        updates=2, rollout_steps=16, epochs=1, minibatch=16,
        eval_every=2, eval_episodes=1, max_steps=20, device="cpu", agents=2,
        critic_mode="central", comm=True, comm_token_limit=4, comm_vocab_size=8,
        comm_send_target=0.25, comm_send_target_coeff=0.01,
        eval_action_mode="sample", eval_send_threshold=0.25,
    )
    train_mappo(cfg)


def test_mappo_wandb_log_retries_scalar_subset():
    from syncorsink.train.mappo import _safe_wandb_log

    class FakeRun:
        def __init__(self):
            self.logged = []
            self.finished = False

        def log(self, payload):
            self.logged.append(dict(payload))
            if "histogram" in payload:
                raise RuntimeError("optional histogram backend unavailable")

        def finish(self):
            self.finished = True

    run = FakeRun()
    result = _safe_wandb_log(run, {"loss": 1.0, "histogram": object()})

    assert result is run
    assert len(run.logged) == 2
    assert run.logged[0]["loss"] == 1.0
    assert "histogram" in run.logged[0]
    assert run.logged[1] == {"loss": 1.0}
    assert run.finished is False


def test_mappo_train_save_load_eval_workbench(tmp_path):
    from syncorsink.train.workbench import TrainEvalWorkbenchConfig, run_train_eval_workbench

    cfg = TrainEvalWorkbenchConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=20,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=8,
        comm_send_target=0.25,
        comm_send_target_coeff=0.01,
        obs_exploration_memory=True,
        obs_exploration_age=True,
        updates=1,
        rollout_steps=8,
        epochs=1,
        minibatch=8,
        eval_send_threshold=0.25,
        eval_episodes=1,
        output_dir=str(tmp_path / "workbench"),
        run_name="smoke",
        wandb=True,
        wandb_mode="disabled",
        device="cpu",
    )

    result = run_train_eval_workbench(cfg)
    checkpoint = tmp_path / "workbench" / "smoke" / "checkpoints" / "mappo.pt"
    summary = tmp_path / "workbench" / "smoke" / "summary.json"

    assert checkpoint.exists()
    assert summary.exists()
    assert result["eval"]["episodes"] == 1
    assert result["checkpoint_path"] == str(checkpoint)
    assert "wandb" in result
    assert result["policy_metadata"]["send_threshold"] == pytest.approx(0.25)

    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["algorithm"] == "mappo"
    assert payload["config"]["comm"] is True
    assert payload["config"]["comm_send_target"] == pytest.approx(0.25)
    assert payload["config"]["obs_exploration_memory"] is True
    assert payload["config"]["obs_exploration_age"] is True
    assert payload["obs_dim"] > 0


def test_mappo_decoding_sweep_smoke(tmp_path):
    from syncorsink.eval.decoding_sweep import (
        MAPPODecodingSweepConfig,
        run_mappo_decoding_sweep,
    )
    from syncorsink.train.mappo import MAPPOConfig, train_mappo

    checkpoint = tmp_path / "mappo.pt"
    train_mappo(MAPPOConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=20,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=8,
        comm_max_messages=4,
        hidden_dim=32,
        updates=1,
        rollout_steps=8,
        epochs=1,
        minibatch=8,
        eval_every=0,
        save=str(checkpoint),
        save_every=1,
        device="cpu",
    ))

    result = run_mappo_decoding_sweep(MAPPODecodingSweepConfig(
        checkpoints=[str(checkpoint)],
        checkpoint_labels=["tiny"],
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=20,
        comm_token_limit=4,
        comm_vocab_size=8,
        comm_max_messages=4,
        episodes=1,
        seed=123,
        action_modes=("argmax",),
        action_temperatures=(1.0,),
        send_modes=("threshold",),
        send_thresholds=(0.25, 0.5),
        token_modes=("argmax",),
        token_temperatures=(1.0,),
        length_modes=("argmax",),
        length_temperatures=(1.0,),
        output_dir=str(tmp_path / "sweep"),
        run_name="smoke",
        device="cpu",
    ))

    summary_path = tmp_path / "sweep" / "smoke" / "summary.json"
    csv_path = tmp_path / "sweep" / "smoke" / "results.csv"
    assert result["status"] == "complete"
    assert result["combo_count"] == 2
    assert len(result["rows"]) == 2
    assert result["best_row"]["rank"] == 1
    assert result["rows"][0]["summary"]["episodes"] == 1
    assert summary_path.exists()
    assert csv_path.exists()


def test_mappo_action_mask_helpers():
    from syncorsink.train.mappo import action_mask_from_flat_obs, mask_action_logits

    flat_obs = torch.tensor([
        [9.0, 8.0, 1, 0, 1, 0, 0, 0, 0, 1],
        [7.0, 6.0, 0, 0, 0, 0, 1, 0, 0, 0],
    ])
    mask = action_mask_from_flat_obs(flat_obs, action_dim=8)

    assert torch.equal(mask, flat_obs[:, -8:])

    logits = torch.tensor([
        [0.0, 100.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0],
        [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0],
    ])
    masked_logits = mask_action_logits(logits, mask)
    dist = torch.distributions.Categorical(logits=masked_logits)
    samples = [int(dist.sample()[0].item()) for _ in range(50)]

    assert set(samples).issubset({0, 2, 7})
    assert int(torch.argmax(masked_logits[0]).item()) == 2
    assert int(torch.argmax(masked_logits[1]).item()) == 4
    assert torch.isfinite(dist.entropy()).all()


def test_flatten_obs_optional_exploration_memory_keeps_action_mask_tail():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.train.mappo import action_mask_from_flat_obs, flatten_obs

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
        obs_exploration_memory=True,
        obs_exploration_age=True,
    ))
    obs, _ = env.reset(seed=0)

    base = flatten_obs(obs[0])
    with_memory = flatten_obs(obs[0], include_exploration_memory=True)
    with_age = flatten_obs(
        obs[0],
        include_exploration_memory=True,
        include_exploration_age=True,
    )

    assert with_memory.shape[0] == base.shape[0] + 64
    assert with_age.shape[0] == base.shape[0] + 128
    assert torch.equal(
        action_mask_from_flat_obs(torch.tensor(with_age).unsqueeze(0))[0],
        torch.tensor(obs[0]["action_mask"], dtype=torch.float32),
    )


def test_mappo_categorical_sampling_uses_local_generator():
    from syncorsink.train.mappo import _select_categorical

    logits = torch.tensor([[0.1, 1.0, -0.3], [0.5, -0.1, 0.2]], dtype=torch.float32)
    gen_a = torch.Generator(device="cpu")
    gen_b = torch.Generator(device="cpu")
    gen_a.manual_seed(123)
    gen_b.manual_seed(123)

    sample_a = _select_categorical(logits, mode="sample", generator=gen_a)
    sample_b = _select_categorical(logits, mode="sample", generator=gen_b)

    torch.testing.assert_close(sample_a, sample_b)


def test_mappo_action_mask_all_invalid_fallback():
    from syncorsink.train.mappo import mask_action_logits

    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.zeros_like(logits)

    masked_logits = mask_action_logits(logits, mask)

    assert torch.equal(masked_logits, logits)


def test_recurrent_bad_action_margin_loss_requires_alternative_winner():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import _signal_bad_action_margin_loss

    bad_action = int(SyncOrSinkEnv.ACTION_INTERACT)
    bad_ids = torch.tensor([bad_action, bad_action, bad_action])
    mask = torch.tensor([1.0, 1.0, 0.0])
    logits = torch.zeros((3, 8), dtype=torch.float32)
    logits[0, bad_action] = 2.0
    logits[0, int(SyncOrSinkEnv.ACTION_UP)] = 0.0
    logits[1, bad_action] = 0.0
    logits[1, int(SyncOrSinkEnv.ACTION_UP)] = 2.0
    logits[2, bad_action] = 5.0

    loss = _signal_bad_action_margin_loss(logits, bad_ids, mask, margin=1.0)

    assert loss.item() == pytest.approx(1.5)


def test_set_global_seeds_reproducible():
    import random

    from syncorsink.train.seed import set_global_seeds

    set_global_seeds(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_global_seeds(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_comm_mat_training():
    from syncorsink.train.comm_mat import train_comm_mat, CommMATTrainConfig
    cfg = CommMATTrainConfig(
        updates=2, rollout_steps=16, epochs=1, minibatch=16,
        eval_every=0, max_steps=20, device="cpu", agents=2,
        comm_token_limit=4, comm_vocab_size=8,
        comm_send_target=0.25, comm_send_target_coeff=0.01,
    )
    train_comm_mat(cfg)


def test_bc_collect_and_train(tmp_path):
    from syncorsink.train.bc import collect_demos, train_bc, BCConfig
    demo_path = str(tmp_path / "demos.npz")
    cfg = BCConfig(
        scenario="signal_hunt", map_size=8, agents=2, fov_preset="easy",
        demo_episodes=5, oracle_type="oracle_strong", demo_path=demo_path,
        max_steps=50,
    )
    collect_demos(cfg)

    model_path = str(tmp_path / "bc.pt")
    cfg = BCConfig(
        demo_path=demo_path, epochs=3, batch_size=16, lr=1e-3,
        hidden_dim=32, comm=False, device="cpu", save=model_path,
    )
    model = train_bc(cfg)
    assert model is not None
    assert (tmp_path / "bc.pt").exists()


def test_signal_hint_comm_expert_acceptance_and_demo_collection(tmp_path):
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.eval.runner import run_episodes
    from syncorsink.policies.local_oracle import local_signal_policy
    from syncorsink.train.bc import BCConfig, collect_demos

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=60,
        comm_token_limit=8,
        token_vocab_size=32,
        max_messages=8,
    ))
    summary, _ = run_episodes(env, local_signal_policy(env), episodes=16, seed=0)
    assert summary.success_rate == 1.0
    assert summary.avg_steps < 10.0
    assert summary.avg_comm_tokens > 0.0

    demo_path = str(tmp_path / "signal_hint_demos.npz")
    collect_demos(BCConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=60,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        demo_episodes=8,
        oracle_type="signal_hint_comm",
        demo_path=demo_path,
        seed=0,
    ))
    data = np.load(demo_path)
    assert data["obs"].shape[0] > 0
    assert np.count_nonzero(data["msg_lens"]) > 0


def test_signal_hint_comm_oracle_does_not_share_private_hints_without_message():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.policies.local_oracle import local_signal_policy

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=60,
        comm_token_limit=8,
        token_vocab_size=32,
        max_messages=8,
    ))
    env.reset(seed=0)
    target = tuple(env.scenario_state.data["target"])
    anchor = (target[0] - 1, target[1]) if target[0] > 0 else (target[0] + 1, target[1])
    env.scenario_state.data["agent_hint_specs"][0] = {
        "type": "x_parity",
        "value": target[0] % 2,
    }
    env.scenario_state.data["agent_hint_specs"][1] = {
        "type": "offset",
        "object": "beacon",
        "pos": anchor,
        "dx": target[0] - anchor[0],
        "dy": target[1] - anchor[1],
    }

    policy = local_signal_policy(env)
    obs = env._build_observations()
    first_actions = policy(obs, {}, {"step": 0})
    exact_message = [26, target[0], target[1]]

    assert first_actions[1]["message_tokens"] == exact_message

    obs_without_message = env._build_observations()
    no_inbox_actions = policy(obs_without_message, {}, {"step": 1})
    assert no_inbox_actions[0]["message_tokens"] != exact_message

    obs_with_message, _rewards, _done, _truncated, info = env.step(first_actions)
    inbox_actions = policy(obs_with_message, info, {"step": env.steps})
    assert inbox_actions[0]["message_tokens"] == exact_message


def test_recurrent_oracle_factory_supports_scenario_planner_comm():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.eval.runner import run_episodes
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _make_oracle_policy

    cases = [
        (
            "energy_grid",
            "energy_planner_comm",
            {"num_agents": 3, "energy_preset": "easy", "max_steps": 180},
        ),
        (
            "pipeline_assembly",
            "pipeline_planner_comm",
            {"num_agents": 3, "max_steps": 180},
        ),
    ]

    for scenario, explicit_oracle, kwargs in cases:
        env_config = SyncOrSinkConfig(
            scenario=scenario,
            map_size=8,
            fov_preset="easy",
            comm_token_limit=8,
            token_vocab_size=32,
            max_messages=8,
            **kwargs,
        )
        env = SyncOrSinkEnv(env_config)
        cfg = RecurrentConfig(
            scenario=scenario,
            map_size=8,
            agents=env_config.num_agents,
            fov_preset="easy",
            max_steps=env_config.max_steps,
            energy_preset=env_config.energy_preset,
            oracle_type="planner_comm",
            comm=True,
        )

        summary, _episodes = run_episodes(env, _make_oracle_policy(env, cfg), episodes=4, seed=0)

        assert summary.success_rate == 1.0

        explicit_cfg = RecurrentConfig(**{**vars(cfg), "oracle_type": explicit_oracle})
        explicit_env = SyncOrSinkEnv(env_config)
        explicit_summary, _episodes = run_episodes(
            explicit_env,
            _make_oracle_policy(explicit_env, explicit_cfg),
            episodes=4,
            seed=0,
        )
        assert explicit_summary.success_rate == 1.0


def test_bc_dagger(tmp_path):
    from syncorsink.train.bc import collect_demos, train_bc_dagger, BCConfig
    demo_path = str(tmp_path / "demos.npz")
    cfg = BCConfig(
        scenario="signal_hunt", map_size=8, agents=2, fov_preset="easy",
        demo_episodes=5, oracle_type="oracle_strong", demo_path=demo_path,
        max_steps=50,
    )
    collect_demos(cfg)

    model_path = str(tmp_path / "dagger.pt")
    cfg = BCConfig(
        scenario="signal_hunt", map_size=8, agents=2, fov_preset="easy",
        demo_path=demo_path, dagger_rounds=1, dagger_episodes=3,
        epochs=3, batch_size=16, lr=1e-3, hidden_dim=32,
        comm=False, device="cpu", save=model_path, max_steps=50,
    )
    model = train_bc_dagger(cfg)
    assert model is not None


def test_bc_rl_curriculum_dry_run(tmp_path):
    from syncorsink.train.curriculum import BCRLCurriculumConfig, run_bc_rl_curriculum

    result = run_bc_rl_curriculum(BCRLCurriculumConfig(
        scenario="energy_grid",
        output_dir=str(tmp_path),
        run_name="dry",
        dry_run=True,
    ))

    assert result["status"] == "dry_run"
    assert result["config"]["agents"] == 3
    assert result["config"]["oracle"] == "oracle_strong_comm"
    assert result["stages"][0]["name"] == "collect_demos"
    assert result["stages"][1]["name"] == "dagger"
    assert (tmp_path / "dry" / "summary.json").exists()


def test_recurrent_curriculum_dry_run(tmp_path):
    from syncorsink.train.recurrent_curriculum import (
        RecurrentCurriculumConfig,
        _checkpoint_eval_send_threshold,
        _resolve_initial_eval_send_threshold,
        _stage_recurrent_config,
        run_recurrent_curriculum,
    )

    cfg = RecurrentCurriculumConfig(
        stage_map_suites="8;8,16",
        max_steps_by_map="8:60,16:120",
        train_map_sampling_weights="8:1,16:3",
        promotion_success_threshold=0.75,
        obs_exploration_age=True,
        obs_pipeline_shared_feedback=True,
        obs_pipeline_progress_features=True,
        obs_signal_negative_memory=True,
        obs_signal_negative_memory_window=12,
        obs_signal_inferred_target_features=True,
        obs_signal_confidence_features=True,
        obs_signal_sector_features=True,
        bc_signal_redundant_target_interact_weight=1.5,
        hidden_dim=96,
        recurrent_backbone="residual_mlp",
        bc_eval_every_epochs=2,
        bc_eval_episodes=3,
        bc_eval_seed_count=2,
        bc_restore_best_eval_epoch=True,
        bc_action_class_balance=True,
        bc_action_class_balance_max_weight=7.0,
        bc_event_action_weight=2.5,
        bc_event_action_events="picked_resource,delivered",
        bc_comm_send_loss_weight=1.7,
        bc_comm_length_loss_weight=0.8,
        bc_comm_token_loss_weight=1.9,
        bc_comm_send_rate_penalty_weight=0.35,
        bc_comm_send_rate_target=0.12,
        bc_signal_target_pursuit_weight=2.0,
        bc_signal_target_pursuit_action_weight=0.9,
        bc_signal_target_pursuit_trust_exact_memory=True,
        bc_signal_target_pursuit_max_agents=1,
        bc_signal_constraint_frontier_bias=True,
        bc_signal_initial_message_weight=4.5,
        bc_signal_initial_message_loss_weight=3.5,
        bc_signal_constraint_message_loss_weight=2.75,
        bc_signal_sync_response_weight=2.5,
        bc_signal_sync_response_action_loss_weight=1.25,
        bc_signal_active_scan_response_action_weight=0.95,
        bc_signal_active_scan_response_min_map_size=12,
        bc_signal_active_scan_response_max_agents=2,
        bc_signal_scan_bridge_action_weight=0.55,
        bc_signal_scan_bridge_min_map_size=12,
        bc_signal_scan_bridge_remaining_threshold=0.35,
        bc_signal_scan_bridge_max_teammate_distance=5,
        bc_signal_clue_interact_action_weight=0.85,
        bc_signal_clue_interact_min_map_size=12,
        bc_signal_target_match_action_weight=1.75,
        bc_signal_target_opportunity_action_weight=0.8,
        bc_signal_redundant_target_wait_action_loss_weight=1.4,
        bc_signal_rejected_target_interact_action_loss_weight=0.7,
        bc_signal_target_validity_loss_weight=0.6,
        bc_signal_target_validity_pos_weight=2.0,
        bc_signal_target_validity_neg_weight=1.5,
        bc_signal_target_decision_loss_weight=0.4,
        bc_signal_target_decision_pos_weight=2.5,
        bc_signal_target_decision_neg_weight=1.75,
        bc_signal_target_hypothesis_loss_weight=0.45,
        bc_signal_target_hypothesis_commit_loss_weight=0.25,
        bc_signal_target_hypothesis_ambiguity_loss_weight=0.5,
        bc_signal_target_hypothesis_xy_loss_weight=1.75,
        bc_signal_target_hypothesis_min_map_size=12,
        bc_signal_ambiguous_target_decision_negatives=True,
        bc_signal_ambiguous_target_decision_min_map_size=12,
        bc_signal_ambiguous_target_search_labels=True,
        bc_signal_ambiguous_target_search_min_map_size=12,
        bc_signal_evidence_sweep_action_weight=0.75,
        bc_signal_evidence_sweep_min_map_size=12,
        bc_signal_frontier_exploration_action_weight=0.65,
        bc_signal_frontier_exploration_min_map_size=12,
        bc_pipeline_delivery_progress_action_loss_weight=0.8,
        bc_pipeline_navigation_action_loss_weight=1.2,
        bc_pipeline_frontier_exploration_action_loss_weight=0.55,
        bc_pipeline_frontier_exploration_min_map_size=8,
        bc_pipeline_sync_action_loss_weight=1.3,
        bc_pipeline_ready_interact_action_loss_weight=1.15,
        bc_pipeline_station_guard_action_loss_weight=1.05,
        bc_pipeline_pickup_gate_loss_weight=0.9,
        bc_pipeline_pickup_gate_pos_weight=2.2,
        bc_pipeline_pickup_gate_neg_weight=1.3,
        bc_pipeline_plan_action_loss_weight=1.1,
        bc_pipeline_message_loss_weight=1.3,
        bc_pipeline_send_gate_loss_weight=0.9,
        bc_pipeline_send_gate_pos_weight=2.0,
        bc_pipeline_send_gate_neg_weight=1.25,
        bc_pipeline_interact_gate_loss_weight=0.7,
        bc_pipeline_interact_gate_pos_weight=2.5,
        bc_pipeline_interact_gate_neg_weight=1.1,
        bc_calibrate_pipeline_interact_gate_threshold=True,
        bc_pipeline_interact_gate_threshold_target_rate=0.33,
        bc_pipeline_proactive_bad_action_labels=True,
        pipeline_stage_count=2,
        pipeline_required_per_stage_min=1,
        pipeline_required_per_stage_max=1,
        pipeline_sync_probability=0.0,
        pipeline_dependency_probability=0.25,
        pipeline_wrong_delivery_penalty=0.75,
        eval_signal_target_validity_threshold=0.55,
        eval_signal_target_decision_threshold=0.6,
        eval_signal_target_decision_suppress=False,
        eval_signal_negative_memory_scan_guard=True,
        eval_signal_target_probe_assist=True,
        eval_signal_scan_broadcast_assist=True,
        eval_signal_exact_target_message_guard=True,
        eval_signal_initial_exact_message_copy_assist=True,
        eval_signal_exact_target_navigation_assist=True,
        eval_signal_exact_target_memory_steps=24,
        eval_signal_scan_refresh_assist=True,
        eval_signal_scan_refresh_threshold=0.5,
        eval_signal_frontier_exploration_assist=True,
        eval_pipeline_navigation_assist=True,
        eval_pipeline_navigation_assist_trust_messages=True,
        eval_pipeline_interact_gate_threshold=0.42,
        eval_pipeline_event_head_threshold=0.41,
        eval_pipeline_navigation_head_threshold=0.43,
        dagger_focus_events="pipeline_pickup_miss,pipeline_delivery_miss",
        dagger_focus_error_weight=4.0,
        dagger_focus_recovery_weight=2.5,
        dagger_focus_window=3,
        dagger_oracle_action_rollin_rate=0.35,
        dagger_seed_base=3000,
        dagger_seed_stride=17,
        dagger_seed_list="3002,3003,3020",
        dagger_target_discovery_min_map_size=8,
        dagger_target_discovery_focus_weight=4.25,
        dagger_movement_stall_min_map_size=8,
        dagger_movement_stall_window=4,
        dagger_movement_stall_focus_weight=5.5,
        dagger_solo_target_team_weight=2.25,
        dagger_solo_target_team_success_only=True,
        dagger_restore_best=False,
        dagger_positive_target_pursuit_min_map_size=8,
        dagger_signal_target_rendezvous_labels=True,
        dagger_signal_target_rendezvous_min_map_size=12,
        dagger_signal_target_rendezvous_max_agents=3,
        dagger_replay_priority_events="movement_stall_miss",
        dagger_replay_balance_positive_events="first_target_scan,joint_target_scan",
        dagger_replay_balance_negative_events="decoy_scan,rejected_target_scan",
        dagger_replay_max_negative_per_positive=0.5,
        dagger_max_failed_parent_replay_snippets_per_episode=2,
        dagger_failed_parent_replay_weight_scale=0.25,
        dagger_expert_max_replay_snippets_per_episode=3,
        pipeline_assisted_rollout_episodes=4,
        pipeline_assisted_rollout_seed_base=5100,
        pipeline_assisted_rollout_seed_list="8:5100,5101",
        pipeline_assisted_rollout_max_steps_per_episode=14,
        pipeline_assisted_rollout_weight=2.75,
        pipeline_assisted_rollout_success_only=True,
        pipeline_assisted_rollout_navigation_assist=False,
        pipeline_assisted_rollout_navigation_assist_trust_messages=False,
        pipeline_assisted_rollout_station_interact_guard=True,
        pipeline_assisted_rollout_bc_epochs=2,
        rl_updates=3,
        rl_updates_schedule="3,5",
        rl_early_stop_eval_patience=2,
        rl_eval_decoding_action_loss_weight=0.35,
        rl_pipeline_assisted_action_loss_weight=0.88,
        rl_pipeline_interact_gate_loss_weight=0.55,
        rl_pipeline_interact_gate_pos_weight=2.0,
        rl_pipeline_interact_gate_neg_weight=3.0,
        rl_pipeline_pickup_gate_loss_weight=0.75,
        rl_pipeline_pickup_gate_pos_weight=2.5,
        rl_pipeline_pickup_gate_neg_weight=3.5,
        rl_pipeline_delivery_progress_action_loss_weight=0.6,
        rl_pipeline_navigation_action_loss_weight=0.7,
        rl_pipeline_sync_action_loss_weight=0.7,
        rl_pipeline_ready_interact_action_loss_weight=0.95,
        rl_pipeline_station_guard_action_loss_weight=0.45,
        rl_pipeline_wrong_station_recovery_action_loss_weight=0.85,
        rl_pipeline_plan_action_loss_weight=0.65,
        rl_pipeline_plan_head_loss_weight=0.72,
        rl_pipeline_option_loss_weight=0.82,
        rl_pipeline_bad_pickup_penalty=0.2,
        rl_pipeline_bad_interact_penalty=0.15,
        rl_pipeline_unneeded_drop_bonus=0.075,
        rollout_steps=40,
        rl_epochs=1,
        minibatch_seqs=4,
        rl_lr=2e-5,
        rl_eval_every=2,
        rl_eval_episodes=5,
        rl_eval_seed=7000,
        rl_eval_seed_stage_stride=250,
        rl_eval_seed_count=2,
        rl_eval_seed_list="7000,7001",
        rl_restore_best=False,
        rl_save_best=False,
        output_dir=str(tmp_path),
        run_name="recurrent_dry",
        initial_recurrent_checkpoint="logs/recurrent_curriculum/example.pt",
        dry_run=True,
    )
    result = run_recurrent_curriculum(cfg)

    assert result["status"] == "dry_run"
    assert result["config"]["bc_signal_target_match_action_weight"] == pytest.approx(1.75)
    assert result["config"]["bc_signal_target_opportunity_action_weight"] == pytest.approx(0.8)
    assert result["config"]["bc_signal_target_pursuit_action_weight"] == pytest.approx(0.9)
    assert result["config"]["bc_signal_target_pursuit_trust_exact_memory"] is True
    assert result["config"]["bc_signal_target_pursuit_max_agents"] == 1
    assert result["config"]["bc_signal_constraint_frontier_bias"] is True
    assert result["config"]["bc_signal_initial_message_weight"] == pytest.approx(4.5)
    assert result["config"]["bc_signal_initial_message_loss_weight"] == pytest.approx(3.5)
    assert result["config"]["bc_signal_constraint_message_loss_weight"] == pytest.approx(2.75)
    assert result["config"]["bc_signal_sync_response_action_loss_weight"] == pytest.approx(1.25)
    assert result["config"]["bc_signal_active_scan_response_action_weight"] == pytest.approx(0.95)
    assert result["config"]["bc_signal_active_scan_response_min_map_size"] == 12
    assert result["config"]["bc_signal_active_scan_response_max_agents"] == 2
    assert result["config"]["bc_signal_scan_bridge_action_weight"] == pytest.approx(0.55)
    assert result["config"]["bc_signal_scan_bridge_min_map_size"] == 12
    assert result["config"]["bc_signal_scan_bridge_remaining_threshold"] == pytest.approx(0.35)
    assert result["config"]["bc_signal_scan_bridge_max_teammate_distance"] == 5
    assert result["config"]["bc_signal_clue_interact_action_weight"] == pytest.approx(0.85)
    assert result["config"]["bc_signal_clue_interact_min_map_size"] == 12
    assert result["config"]["bc_signal_redundant_target_wait_action_loss_weight"] == pytest.approx(1.4)
    assert result["config"]["bc_signal_rejected_target_interact_action_loss_weight"] == pytest.approx(0.7)
    assert result["config"]["bc_signal_target_validity_loss_weight"] == pytest.approx(0.6)
    assert result["config"]["bc_signal_target_decision_loss_weight"] == pytest.approx(0.4)
    assert result["config"]["bc_signal_target_hypothesis_loss_weight"] == pytest.approx(0.45)
    assert result["config"]["bc_signal_target_hypothesis_commit_loss_weight"] == pytest.approx(0.25)
    assert result["config"]["bc_signal_target_hypothesis_ambiguity_loss_weight"] == pytest.approx(0.5)
    assert result["config"]["bc_signal_target_hypothesis_xy_loss_weight"] == pytest.approx(1.75)
    assert result["config"]["bc_signal_target_hypothesis_min_map_size"] == 12
    assert result["config"]["bc_signal_evidence_sweep_action_weight"] == pytest.approx(0.75)
    assert result["config"]["bc_signal_evidence_sweep_min_map_size"] == 12
    assert result["config"]["bc_signal_frontier_exploration_action_weight"] == pytest.approx(0.65)
    assert result["config"]["bc_signal_frontier_exploration_min_map_size"] == 12
    assert result["config"]["bc_pipeline_delivery_progress_action_loss_weight"] == pytest.approx(0.8)
    assert result["config"]["bc_pipeline_navigation_action_loss_weight"] == pytest.approx(1.2)
    assert result["config"]["bc_pipeline_frontier_exploration_action_loss_weight"] == pytest.approx(0.55)
    assert result["config"]["bc_pipeline_frontier_exploration_min_map_size"] == 8
    assert result["config"]["bc_pipeline_sync_action_loss_weight"] == pytest.approx(1.3)
    assert result["config"]["bc_pipeline_ready_interact_action_loss_weight"] == pytest.approx(1.15)
    assert result["config"]["bc_pipeline_station_guard_action_loss_weight"] == pytest.approx(1.05)
    assert result["config"]["bc_pipeline_pickup_gate_loss_weight"] == pytest.approx(0.9)
    assert result["config"]["bc_pipeline_pickup_gate_pos_weight"] == pytest.approx(2.2)
    assert result["config"]["bc_pipeline_pickup_gate_neg_weight"] == pytest.approx(1.3)
    assert result["config"]["bc_pipeline_plan_action_loss_weight"] == pytest.approx(1.1)
    assert result["config"]["bc_pipeline_message_loss_weight"] == pytest.approx(1.3)
    assert result["config"]["bc_pipeline_send_gate_loss_weight"] == pytest.approx(0.9)
    assert result["config"]["bc_pipeline_send_gate_pos_weight"] == pytest.approx(2.0)
    assert result["config"]["bc_pipeline_send_gate_neg_weight"] == pytest.approx(1.25)
    assert result["config"]["bc_pipeline_interact_gate_loss_weight"] == pytest.approx(0.7)
    assert result["config"]["bc_pipeline_interact_gate_pos_weight"] == pytest.approx(2.5)
    assert result["config"]["bc_pipeline_interact_gate_neg_weight"] == pytest.approx(1.1)
    assert result["config"]["bc_calibrate_pipeline_interact_gate_threshold"] is True
    assert result["config"]["bc_pipeline_interact_gate_threshold_target_rate"] == pytest.approx(0.33)
    assert result["config"]["bc_pipeline_proactive_bad_action_labels"] is True
    assert result["config"]["pipeline_stage_count"] == 2
    assert result["config"]["pipeline_required_per_stage_min"] == 1
    assert result["config"]["pipeline_required_per_stage_max"] == 1
    assert result["config"]["pipeline_sync_probability"] == pytest.approx(0.0)
    assert result["config"]["pipeline_dependency_probability"] == pytest.approx(0.25)
    assert result["config"]["pipeline_wrong_delivery_penalty"] == pytest.approx(0.75)
    assert result["config"]["eval_signal_target_validity_threshold"] == pytest.approx(0.55)
    assert result["config"]["eval_signal_target_decision_threshold"] == pytest.approx(0.6)
    assert result["config"]["eval_signal_target_decision_suppress"] is False
    assert result["config"]["eval_signal_negative_memory_scan_guard"] is True
    assert result["config"]["eval_signal_target_probe_assist"] is True
    assert result["config"]["eval_signal_scan_broadcast_assist"] is True
    assert result["config"]["eval_signal_exact_target_message_guard"] is True
    assert result["config"]["eval_signal_initial_exact_message_copy_assist"] is True
    assert result["config"]["eval_signal_exact_target_navigation_assist"] is True
    assert result["config"]["eval_signal_exact_target_memory_steps"] == 24
    assert result["config"]["eval_signal_scan_refresh_assist"] is True
    assert result["config"]["eval_signal_scan_refresh_threshold"] == pytest.approx(0.5)
    assert result["config"]["eval_signal_frontier_exploration_assist"] is True
    assert result["config"]["eval_pipeline_navigation_assist"] is True
    assert result["config"]["eval_pipeline_navigation_assist_trust_messages"] is True
    assert result["config"]["eval_pipeline_interact_gate_threshold"] == pytest.approx(0.42)
    assert result["config"]["eval_pipeline_event_head_threshold"] == pytest.approx(0.41)
    assert result["config"]["eval_pipeline_navigation_head_threshold"] == pytest.approx(0.43)
    assert result["config"]["initial_recurrent_checkpoint"] == "logs/recurrent_curriculum/example.pt"
    assert result["config"]["hidden_dim"] == 96
    assert result["config"]["recurrent_backbone"] == "residual_mlp"
    assert result["config"]["bc_eval_every_epochs"] == 2
    assert result["config"]["bc_eval_episodes"] == 3
    assert result["config"]["bc_eval_seed_count"] == 2
    assert result["config"]["bc_restore_best_eval_epoch"] is True
    assert result["config"]["bc_action_class_balance"] is True
    assert result["config"]["bc_action_class_balance_max_weight"] == pytest.approx(7.0)
    assert result["config"]["bc_event_action_weight"] == pytest.approx(2.5)
    assert result["config"]["bc_event_action_events"] == "picked_resource,delivered"
    assert result["config"]["bc_comm_send_loss_weight"] == pytest.approx(1.7)
    assert result["config"]["bc_comm_length_loss_weight"] == pytest.approx(0.8)
    assert result["config"]["bc_comm_token_loss_weight"] == pytest.approx(1.9)
    assert result["config"]["bc_comm_send_rate_penalty_weight"] == pytest.approx(0.35)
    assert result["config"]["bc_comm_send_rate_target"] == pytest.approx(0.12)
    assert result["config"]["train_map_sampling_weights"] == "8:1,16:3"
    assert result["config"]["obs_exploration_age"] is True
    assert result["config"]["obs_pipeline_shared_feedback"] is True
    assert result["config"]["obs_pipeline_progress_features"] is True
    assert result["config"]["obs_signal_negative_memory"] is True
    assert result["config"]["obs_signal_sector_features"] is True
    assert result["config"]["dagger_solo_target_team_weight"] == pytest.approx(2.25)
    assert result["config"]["dagger_focus_events"] == (
        "pipeline_pickup_miss,pipeline_delivery_miss"
    )
    assert result["config"]["dagger_focus_error_weight"] == pytest.approx(4.0)
    assert result["config"]["dagger_focus_recovery_weight"] == pytest.approx(2.5)
    assert result["config"]["dagger_focus_window"] == 3
    assert result["config"]["dagger_oracle_action_rollin_rate"] == pytest.approx(0.35)
    assert result["config"]["dagger_initial_target_broadcast_labels"] is True
    assert result["config"]["dagger_seed_base"] == 3000
    assert result["config"]["dagger_seed_stride"] == 17
    assert result["config"]["dagger_seed_list"] == "3002,3003,3020"
    assert result["config"]["dagger_movement_stall_min_map_size"] == 8
    assert result["config"]["dagger_movement_stall_window"] == 4
    assert result["config"]["dagger_movement_stall_focus_weight"] == pytest.approx(5.5)
    assert result["config"]["dagger_restore_best"] is False
    assert result["config"]["dagger_positive_target_pursuit_min_map_size"] == 8
    assert result["config"]["dagger_replay_priority_events"] == "movement_stall_miss"
    assert result["config"]["dagger_replay_balance_positive_events"] == (
        "first_target_scan,joint_target_scan"
    )
    assert result["config"]["dagger_replay_balance_negative_events"] == (
        "decoy_scan,rejected_target_scan"
    )
    assert result["config"]["dagger_replay_max_negative_per_positive"] == pytest.approx(0.5)
    assert result["config"]["dagger_max_failed_parent_replay_snippets_per_episode"] == 2
    assert result["config"]["dagger_failed_parent_replay_weight_scale"] == pytest.approx(0.25)
    assert result["config"]["dagger_expert_max_replay_snippets_per_episode"] == 3
    assert result["config"]["pipeline_assisted_rollout_episodes"] == 4
    assert result["config"]["pipeline_assisted_rollout_seed_base"] == 5100
    assert result["config"]["pipeline_assisted_rollout_seed_list"] == "8:5100,5101"
    assert result["config"]["pipeline_assisted_rollout_max_steps_per_episode"] == 14
    assert result["config"]["pipeline_assisted_rollout_weight"] == pytest.approx(2.75)
    assert result["config"]["pipeline_assisted_rollout_success_only"] is True
    assert result["config"]["dagger_signal_target_rendezvous_labels"] is True
    assert result["config"]["dagger_signal_target_rendezvous_min_map_size"] == 12
    assert result["config"]["dagger_signal_target_rendezvous_max_agents"] == 3
    assert result["config"]["bc_signal_ambiguous_target_decision_negatives"] is True
    assert result["config"]["bc_signal_ambiguous_target_decision_min_map_size"] == 12
    assert result["config"]["bc_signal_ambiguous_target_search_labels"] is True
    assert result["config"]["bc_signal_ambiguous_target_search_min_map_size"] == 12
    assert result["config"]["pipeline_assisted_rollout_navigation_assist"] is False
    assert result["config"]["pipeline_assisted_rollout_navigation_assist_trust_messages"] is False
    assert result["config"]["pipeline_assisted_rollout_station_interact_guard"] is True
    assert result["config"]["pipeline_assisted_rollout_bc_epochs"] == 2
    assert result["config"]["rl_updates"] == 3
    assert result["config"]["rl_updates_schedule"] == "3,5"
    assert result["config"]["rl_early_stop_eval_patience"] == 2
    assert result["config"]["rl_eval_decoding_action_loss_weight"] == pytest.approx(0.35)
    assert result["config"]["rl_pipeline_assisted_action_loss_weight"] == pytest.approx(0.88)
    assert result["config"]["rl_pipeline_interact_gate_loss_weight"] == pytest.approx(0.55)
    assert result["config"]["rl_pipeline_interact_gate_pos_weight"] == pytest.approx(2.0)
    assert result["config"]["rl_pipeline_interact_gate_neg_weight"] == pytest.approx(3.0)
    assert result["config"]["rl_pipeline_pickup_gate_loss_weight"] == pytest.approx(0.75)
    assert result["config"]["rl_pipeline_pickup_gate_pos_weight"] == pytest.approx(2.5)
    assert result["config"]["rl_pipeline_pickup_gate_neg_weight"] == pytest.approx(3.5)
    assert result["config"]["rl_pipeline_delivery_progress_action_loss_weight"] == pytest.approx(0.6)
    assert result["config"]["rl_pipeline_navigation_action_loss_weight"] == pytest.approx(0.7)
    assert result["config"]["rl_pipeline_sync_action_loss_weight"] == pytest.approx(0.7)
    assert result["config"]["rl_pipeline_ready_interact_action_loss_weight"] == pytest.approx(0.95)
    assert result["config"]["rl_pipeline_station_guard_action_loss_weight"] == pytest.approx(0.45)
    assert result["config"]["rl_pipeline_wrong_station_recovery_action_loss_weight"] == pytest.approx(0.85)
    assert result["config"]["rl_pipeline_plan_action_loss_weight"] == pytest.approx(0.65)
    assert result["config"]["rl_pipeline_plan_head_loss_weight"] == pytest.approx(0.72)
    assert result["config"]["rl_pipeline_option_loss_weight"] == pytest.approx(0.82)
    assert result["config"]["rl_pipeline_bad_pickup_penalty"] == pytest.approx(0.2)
    assert result["config"]["rl_pipeline_bad_interact_penalty"] == pytest.approx(0.15)
    assert result["config"]["rl_pipeline_unneeded_drop_bonus"] == pytest.approx(0.075)
    assert result["config"]["rollout_steps"] == 40
    assert result["config"]["rl_epochs"] == 1
    assert result["config"]["minibatch_seqs"] == 4
    assert result["config"]["rl_lr"] == pytest.approx(2e-5)
    assert result["config"]["rl_eval_every"] == 2
    assert result["config"]["rl_eval_episodes"] == 5
    assert result["config"]["rl_eval_use_eval_seeds"] is True
    assert result["config"]["rl_eval_seed"] == 7000
    assert result["config"]["rl_eval_seed_stage_stride"] == 250
    assert result["config"]["rl_eval_seed_count"] == 2
    assert result["config"]["rl_eval_seed_list"] == "7000,7001"
    assert result["config"]["rl_restore_best"] is False
    assert result["config"]["rl_save_best"] is False
    assert result["planned_stages"][0]["train_map_sizes"] == [8]
    assert result["planned_stages"][0]["rl_updates"] == 3
    assert result["planned_stages"][1]["train_map_sizes"] == [8, 16]
    assert result["planned_stages"][1]["rl_updates"] == 5
    assert result["planned_stages"][1]["max_steps"] == {"8": 60, "16": 120}
    assert result["planned_stages"][0]["promotion_success_threshold"] == pytest.approx(0.75)
    assert result["planned_stages"][0]["checkpoint"].endswith("stage0_maps_8.pt")
    assert (tmp_path / "recurrent_dry" / "summary.json").exists()
    stage_cfg = _stage_recurrent_config(
        cfg,
        stage_idx=0,
        suite=(8,),
        max_steps={8: 60, 16: 120},
        checkpoint_path=tmp_path / "stage0_maps_8.pt",
        eval_send_threshold=0.25,
        has_initial_model=False,
    )
    assert stage_cfg.obs_signal_negative_memory is True
    assert stage_cfg.dagger_oracle_action_rollin_rate == pytest.approx(0.35)
    assert stage_cfg.train_map_sampling_weights == "8:1"
    mixed_stage_cfg = _stage_recurrent_config(
        cfg,
        stage_idx=1,
        suite=(8, 16),
        max_steps={8: 60, 16: 120},
        checkpoint_path=tmp_path / "stage1_maps_8_16.pt",
        eval_send_threshold=0.25,
        has_initial_model=True,
    )
    assert mixed_stage_cfg.train_map_sampling_weights == "8:1,16:3"
    assert stage_cfg.obs_exploration_age is True
    assert stage_cfg.obs_pipeline_shared_feedback is True
    assert stage_cfg.obs_pipeline_progress_features is True
    assert stage_cfg.obs_signal_negative_memory_window == 12
    assert stage_cfg.obs_signal_inferred_target_features is True
    assert stage_cfg.obs_signal_confidence_features is True
    assert stage_cfg.obs_signal_sector_features is True
    assert stage_cfg.eval_signal_negative_memory_scan_guard is True
    assert stage_cfg.eval_signal_target_probe_assist is True
    assert stage_cfg.hidden_dim == 96
    assert stage_cfg.recurrent_backbone == "residual_mlp"
    assert stage_cfg.bc_eval_every_epochs == 2
    assert stage_cfg.bc_eval_episodes == 3
    assert stage_cfg.bc_eval_seed_count == 2
    assert stage_cfg.bc_restore_best_eval_epoch is True
    assert stage_cfg.bc_action_class_balance is True
    assert stage_cfg.bc_action_class_balance_max_weight == pytest.approx(7.0)
    assert stage_cfg.bc_event_action_weight == pytest.approx(2.5)
    assert stage_cfg.bc_event_action_events == "picked_resource,delivered"
    assert stage_cfg.bc_comm_send_loss_weight == pytest.approx(1.7)
    assert stage_cfg.bc_comm_length_loss_weight == pytest.approx(0.8)
    assert stage_cfg.bc_comm_token_loss_weight == pytest.approx(1.9)
    assert stage_cfg.bc_comm_send_rate_penalty_weight == pytest.approx(0.35)
    assert stage_cfg.bc_comm_send_rate_target == pytest.approx(0.12)
    assert stage_cfg.bc_signal_redundant_target_interact_weight == pytest.approx(1.5)
    assert stage_cfg.bc_signal_target_pursuit_weight == pytest.approx(2.0)
    assert stage_cfg.bc_signal_target_pursuit_action_weight == pytest.approx(0.9)
    assert stage_cfg.bc_signal_target_pursuit_trust_exact_memory is True
    assert stage_cfg.bc_signal_target_pursuit_max_agents == 1
    assert stage_cfg.bc_signal_constraint_frontier_bias is True
    assert stage_cfg.bc_signal_initial_message_weight == pytest.approx(4.5)
    assert stage_cfg.bc_signal_initial_message_loss_weight == pytest.approx(3.5)
    assert stage_cfg.bc_signal_constraint_message_loss_weight == pytest.approx(2.75)
    assert stage_cfg.bc_signal_sync_response_weight == pytest.approx(2.5)
    assert stage_cfg.bc_signal_sync_response_action_loss_weight == pytest.approx(1.25)
    assert stage_cfg.bc_signal_active_scan_response_action_weight == pytest.approx(0.95)
    assert stage_cfg.bc_signal_active_scan_response_min_map_size == 12
    assert stage_cfg.bc_signal_active_scan_response_max_agents == 2
    assert stage_cfg.bc_signal_scan_bridge_action_weight == pytest.approx(0.55)
    assert stage_cfg.bc_signal_scan_bridge_min_map_size == 12
    assert stage_cfg.bc_signal_scan_bridge_remaining_threshold == pytest.approx(0.35)
    assert stage_cfg.bc_signal_scan_bridge_max_teammate_distance == 5
    assert stage_cfg.bc_signal_clue_interact_action_weight == pytest.approx(0.85)
    assert stage_cfg.bc_signal_clue_interact_min_map_size == 12
    assert stage_cfg.bc_signal_target_match_action_weight == pytest.approx(1.75)
    assert stage_cfg.bc_signal_target_opportunity_action_weight == pytest.approx(0.8)
    assert stage_cfg.bc_signal_redundant_target_wait_action_loss_weight == pytest.approx(1.4)
    assert stage_cfg.bc_signal_rejected_target_interact_action_loss_weight == pytest.approx(0.7)
    assert stage_cfg.bc_signal_target_validity_loss_weight == pytest.approx(0.6)
    assert stage_cfg.bc_signal_target_validity_pos_weight == pytest.approx(2.0)
    assert stage_cfg.bc_signal_target_validity_neg_weight == pytest.approx(1.5)
    assert stage_cfg.bc_signal_target_decision_loss_weight == pytest.approx(0.4)
    assert stage_cfg.bc_signal_target_decision_pos_weight == pytest.approx(2.5)
    assert stage_cfg.bc_signal_target_decision_neg_weight == pytest.approx(1.75)
    assert stage_cfg.bc_signal_target_hypothesis_loss_weight == pytest.approx(0.45)
    assert stage_cfg.bc_signal_target_hypothesis_commit_loss_weight == pytest.approx(0.25)
    assert stage_cfg.bc_signal_target_hypothesis_ambiguity_loss_weight == pytest.approx(0.5)
    assert stage_cfg.bc_signal_target_hypothesis_xy_loss_weight == pytest.approx(1.75)
    assert stage_cfg.bc_signal_target_hypothesis_min_map_size == 12
    assert stage_cfg.bc_signal_evidence_sweep_action_weight == pytest.approx(0.75)
    assert stage_cfg.bc_signal_evidence_sweep_min_map_size == 12
    assert stage_cfg.bc_signal_frontier_exploration_action_weight == pytest.approx(0.65)
    assert stage_cfg.bc_signal_frontier_exploration_min_map_size == 12
    assert stage_cfg.bc_pipeline_frontier_exploration_action_loss_weight == pytest.approx(0.55)
    assert stage_cfg.bc_pipeline_frontier_exploration_min_map_size == 8
    assert stage_cfg.bc_pipeline_ready_interact_action_loss_weight == pytest.approx(1.15)
    assert stage_cfg.bc_pipeline_plan_action_loss_weight == pytest.approx(1.1)
    assert stage_cfg.bc_pipeline_message_loss_weight == pytest.approx(1.3)
    assert stage_cfg.bc_pipeline_send_gate_loss_weight == pytest.approx(0.9)
    assert stage_cfg.bc_pipeline_send_gate_pos_weight == pytest.approx(2.0)
    assert stage_cfg.bc_pipeline_send_gate_neg_weight == pytest.approx(1.25)
    assert stage_cfg.bc_pipeline_interact_gate_loss_weight == pytest.approx(0.7)
    assert stage_cfg.bc_pipeline_interact_gate_pos_weight == pytest.approx(2.5)
    assert stage_cfg.bc_pipeline_interact_gate_neg_weight == pytest.approx(1.1)
    assert stage_cfg.bc_calibrate_pipeline_interact_gate_threshold is True
    assert stage_cfg.bc_pipeline_interact_gate_threshold_target_rate == pytest.approx(0.33)
    assert stage_cfg.bc_pipeline_proactive_bad_action_labels is True
    assert stage_cfg.pipeline_stage_count == 2
    assert stage_cfg.pipeline_required_per_stage_min == 1
    assert stage_cfg.pipeline_required_per_stage_max == 1
    assert stage_cfg.pipeline_sync_probability == pytest.approx(0.0)
    assert stage_cfg.pipeline_dependency_probability == pytest.approx(0.25)
    assert stage_cfg.pipeline_wrong_delivery_penalty == pytest.approx(0.75)
    assert stage_cfg.eval_signal_target_validity_threshold == pytest.approx(0.55)
    assert stage_cfg.eval_signal_target_decision_threshold == pytest.approx(0.6)
    assert stage_cfg.eval_signal_target_decision_suppress is False
    assert stage_cfg.eval_signal_scan_broadcast_assist is True
    assert stage_cfg.eval_signal_exact_target_message_guard is True
    assert stage_cfg.eval_signal_initial_exact_message_copy_assist is True
    assert stage_cfg.eval_signal_exact_target_navigation_assist is True
    assert stage_cfg.eval_signal_exact_target_memory_steps == 24
    assert stage_cfg.eval_signal_scan_refresh_assist is True
    assert stage_cfg.eval_signal_scan_refresh_threshold == pytest.approx(0.5)
    assert stage_cfg.eval_signal_frontier_exploration_assist is True
    assert stage_cfg.eval_pipeline_navigation_assist is True
    assert stage_cfg.eval_pipeline_navigation_assist_trust_messages is True
    assert stage_cfg.eval_pipeline_interact_gate_threshold == pytest.approx(0.42)
    assert stage_cfg.eval_pipeline_event_head_threshold == pytest.approx(0.41)
    assert stage_cfg.eval_pipeline_navigation_head_threshold == pytest.approx(0.43)
    assert stage_cfg.rl_rollout_pipeline_navigation_assist is False
    assert stage_cfg.rl_rollout_pipeline_navigation_assist_trust_messages is False
    assert stage_cfg.rl_rollout_pipeline_station_interact_guard is False
    assert stage_cfg.rl_eval_decoding_action_loss_weight == pytest.approx(0.35)
    assert stage_cfg.rl_pipeline_assisted_action_loss_weight == pytest.approx(0.88)
    assert stage_cfg.rl_pipeline_interact_gate_loss_weight == pytest.approx(0.55)
    assert stage_cfg.rl_pipeline_interact_gate_pos_weight == pytest.approx(2.0)
    assert stage_cfg.rl_pipeline_interact_gate_neg_weight == pytest.approx(3.0)
    assert stage_cfg.rl_pipeline_pickup_gate_loss_weight == pytest.approx(0.75)
    assert stage_cfg.rl_pipeline_pickup_gate_pos_weight == pytest.approx(2.5)
    assert stage_cfg.rl_pipeline_pickup_gate_neg_weight == pytest.approx(3.5)
    assert stage_cfg.rl_pipeline_delivery_progress_action_loss_weight == pytest.approx(0.6)
    assert stage_cfg.rl_pipeline_navigation_action_loss_weight == pytest.approx(0.7)
    assert stage_cfg.rl_pipeline_sync_action_loss_weight == pytest.approx(0.7)
    assert stage_cfg.rl_pipeline_ready_interact_action_loss_weight == pytest.approx(0.95)
    assert stage_cfg.rl_pipeline_station_guard_action_loss_weight == pytest.approx(0.45)
    assert stage_cfg.rl_pipeline_wrong_station_recovery_action_loss_weight == pytest.approx(0.85)
    assert stage_cfg.rl_pipeline_plan_action_loss_weight == pytest.approx(0.65)
    assert stage_cfg.rl_pipeline_plan_head_loss_weight == pytest.approx(0.72)
    assert stage_cfg.rl_pipeline_option_loss_weight == pytest.approx(0.82)
    assert stage_cfg.rl_pipeline_bad_pickup_penalty == pytest.approx(0.2)
    assert stage_cfg.rl_pipeline_bad_interact_penalty == pytest.approx(0.15)
    assert stage_cfg.rl_pipeline_unneeded_drop_bonus == pytest.approx(0.075)
    assert stage_cfg.dagger_focus_events == "pipeline_pickup_miss,pipeline_delivery_miss"
    assert stage_cfg.dagger_focus_error_weight == pytest.approx(4.0)
    assert stage_cfg.dagger_focus_recovery_weight == pytest.approx(2.5)
    assert stage_cfg.dagger_focus_window == 3
    assert stage_cfg.dagger_initial_target_broadcast_labels is True
    assert stage_cfg.dagger_seed_base == 3000
    assert stage_cfg.dagger_seed_stride == 17
    assert stage_cfg.dagger_seed_list == "3002,3003,3020"
    assert stage_cfg.dagger_target_discovery_min_map_size == 8
    assert stage_cfg.dagger_target_discovery_focus_weight == pytest.approx(4.25)
    assert stage_cfg.dagger_movement_stall_min_map_size == 8
    assert stage_cfg.dagger_movement_stall_window == 4
    assert stage_cfg.dagger_movement_stall_focus_weight == pytest.approx(5.5)
    assert stage_cfg.dagger_solo_target_team_weight == pytest.approx(2.25)
    assert stage_cfg.dagger_solo_target_team_success_only is True
    assert stage_cfg.dagger_restore_best is False
    assert stage_cfg.dagger_positive_target_pursuit_min_map_size == 8
    assert stage_cfg.dagger_signal_target_rendezvous_labels is True
    assert stage_cfg.dagger_signal_target_rendezvous_min_map_size == 12
    assert stage_cfg.dagger_signal_target_rendezvous_max_agents == 3
    assert stage_cfg.bc_signal_ambiguous_target_decision_negatives is True
    assert stage_cfg.bc_signal_ambiguous_target_decision_min_map_size == 12
    assert stage_cfg.bc_signal_ambiguous_target_search_labels is True
    assert stage_cfg.bc_signal_ambiguous_target_search_min_map_size == 12
    assert stage_cfg.dagger_replay_priority_events == "movement_stall_miss"
    assert stage_cfg.dagger_replay_balance_positive_events == "first_target_scan,joint_target_scan"
    assert stage_cfg.dagger_replay_balance_negative_events == "decoy_scan,rejected_target_scan"
    assert stage_cfg.dagger_replay_max_negative_per_positive == pytest.approx(0.5)
    assert stage_cfg.dagger_max_failed_parent_replay_snippets_per_episode == 2
    assert stage_cfg.dagger_failed_parent_replay_weight_scale == pytest.approx(0.25)
    assert stage_cfg.dagger_expert_max_replay_snippets_per_episode == 3
    assert stage_cfg.pipeline_assisted_rollout_episodes == 4
    assert stage_cfg.pipeline_assisted_rollout_seed_base == 5100
    assert stage_cfg.pipeline_assisted_rollout_seed_list == "8:5100,5101"
    assert stage_cfg.pipeline_assisted_rollout_max_steps_per_episode == 14
    assert stage_cfg.pipeline_assisted_rollout_weight == pytest.approx(2.75)
    assert stage_cfg.pipeline_assisted_rollout_success_only is True
    assert stage_cfg.pipeline_assisted_rollout_navigation_assist is False
    assert stage_cfg.pipeline_assisted_rollout_navigation_assist_trust_messages is False
    assert stage_cfg.pipeline_assisted_rollout_station_interact_guard is True
    assert stage_cfg.pipeline_assisted_rollout_bc_epochs == 2
    assert stage_cfg.rl_updates == 3
    assert stage_cfg.rl_early_stop_eval_patience == 2
    assert stage_cfg.rollout_steps == 40
    assert stage_cfg.rl_epochs == 1
    assert stage_cfg.minibatch_seqs == 4
    assert stage_cfg.rl_lr == pytest.approx(2e-5)
    assert stage_cfg.rl_eval_every == 2
    assert stage_cfg.rl_eval_episodes == 5
    assert stage_cfg.rl_eval_use_eval_seeds is True
    assert stage_cfg.rl_eval_seed == 7000
    assert stage_cfg.rl_eval_seed_count == 2
    assert stage_cfg.rl_eval_seed_list == "7000,7001"
    assert stage_cfg.rl_restore_best is False
    assert stage_cfg.rl_save_best is False

    threshold_checkpoint = tmp_path / "threshold.pt"
    torch.save({"config": {"eval_send_threshold": 0.73}}, threshold_checkpoint)
    inherit_cfg = RecurrentCurriculumConfig(initial_recurrent_checkpoint=str(threshold_checkpoint))
    assert _checkpoint_eval_send_threshold(threshold_checkpoint) == pytest.approx(0.73)
    assert _resolve_initial_eval_send_threshold(inherit_cfg) == pytest.approx(0.73)
    override_cfg = RecurrentCurriculumConfig(
        initial_recurrent_checkpoint=str(threshold_checkpoint),
        eval_send_threshold=0.41,
    )
    assert _resolve_initial_eval_send_threshold(override_cfg) == pytest.approx(0.41)

    bad_weight_cfg = RecurrentCurriculumConfig(
        stage_map_suites="8;8,16",
        train_map_sampling_weights="64:2",
        output_dir=str(tmp_path / "bad_weights"),
        dry_run=True,
    )
    with pytest.raises(ValueError, match="not present in any curriculum stage"):
        run_recurrent_curriculum(bad_weight_cfg)


def test_recurrent_curriculum_defaults_use_pipeline_feasible_map_budgets():
    from syncorsink.train.recurrent_curriculum import (
        RecurrentCurriculumConfig,
        _parse_max_steps_by_map,
        _stage_max_steps,
    )

    cfg = RecurrentCurriculumConfig()
    max_steps = _parse_max_steps_by_map(cfg.max_steps_by_map)

    assert max_steps == {8: 80, 16: 160, 32: 500}
    assert _stage_max_steps((8,), max_steps) == 80
    assert _stage_max_steps((16,), max_steps) == 160
    assert _stage_max_steps((32,), max_steps) == 500


def test_recurrent_curriculum_passes_initial_obs_dim_expansion_flag(tmp_path):
    from syncorsink.train.recurrent_curriculum import (
        RecurrentCurriculumConfig,
        _stage_recurrent_config,
    )

    cfg = RecurrentCurriculumConfig(
        recurrent_init_allow_obs_dim_mismatch=True,
        output_dir=str(tmp_path),
        run_name="expand_init",
    )

    stage_cfg = _stage_recurrent_config(
        cfg,
        stage_idx=0,
        suite=(8, 16),
        max_steps={8: 80, 16: 160},
        checkpoint_path=tmp_path / "stage0_maps_8_16.pt",
        eval_send_threshold=0.25,
        has_initial_model=True,
    )

    assert stage_cfg.recurrent_init_allow_obs_dim_mismatch is True


def test_recurrent_init_inherits_eval_send_threshold(tmp_path):
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _checkpoint_eval_send_threshold,
        _inherit_recurrent_init_observation_config,
        _inherit_recurrent_init_eval_send_threshold,
    )

    threshold_checkpoint = tmp_path / "threshold.pt"
    torch.save(
        {
            "config": {
                "eval_send_threshold": 0.73,
                "obs_memory_mode": "egocentric",
                "obs_memory_radius": 2,
                "obs_signal_negative_memory": True,
                "obs_signal_sector_features": True,
            }
        },
        threshold_checkpoint,
    )

    inherit_cfg = RecurrentConfig(recurrent_init=str(threshold_checkpoint))
    inherited_obs = _inherit_recurrent_init_observation_config(inherit_cfg)
    assert inherited_obs == {
        "obs_memory_mode": "egocentric",
        "obs_memory_radius": 2,
        "obs_signal_negative_memory": True,
        "obs_signal_sector_features": True,
    }
    assert inherit_cfg.obs_memory_mode == "egocentric"
    assert inherit_cfg.obs_memory_radius == 2
    assert inherit_cfg.obs_signal_negative_memory is True
    assert inherit_cfg.obs_signal_sector_features is True
    assert _checkpoint_eval_send_threshold(threshold_checkpoint) == pytest.approx(0.73)
    assert _inherit_recurrent_init_eval_send_threshold(inherit_cfg) == pytest.approx(0.73)
    assert inherit_cfg.eval_send_threshold == pytest.approx(0.73)

    override_cfg = RecurrentConfig(
        recurrent_init=str(threshold_checkpoint),
        obs_memory_radius=7,
        obs_signal_negative_memory=True,
        obs_signal_sector_features=True,
        eval_send_threshold=0.41,
    )
    inherited_override_obs = _inherit_recurrent_init_observation_config(override_cfg)
    assert inherited_override_obs == {"obs_memory_mode": "egocentric"}
    assert override_cfg.obs_memory_mode == "egocentric"
    assert override_cfg.obs_memory_radius == 7
    assert override_cfg.obs_signal_negative_memory is True
    assert override_cfg.obs_signal_sector_features is True
    assert _inherit_recurrent_init_eval_send_threshold(override_cfg) is None
    assert override_cfg.eval_send_threshold == pytest.approx(0.41)


def test_recurrent_curriculum_cli_accepts_pipeline_core_args(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from syncorsink.train import recurrent_curriculum

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recurrent_curriculum.py",
            "--scenario",
            "pipeline_assembly",
            "--agents",
            "3",
            "--fov-preset",
            "easy",
            "--oracle-type",
            "planner_comm",
            "--stage-map-suites",
            "8",
            "--max-steps-by-map",
            "8:40",
            "--demo-episodes",
            "2",
            "--bc-epochs",
            "1",
            "--bc-seq-len",
            "12",
            "--bc-action-class-balance",
            "--bc-action-class-balance-max-weight",
            "8",
            "--bc-event-action-weight",
            "3",
            "--bc-event-action-events",
            "picked_resource,delivered",
            "--comm-token-limit",
            "6",
            "--comm-vocab-size",
            "48",
            "--bc-pipeline-delivery-progress-action-loss-weight",
            "0.8",
            "--bc-pipeline-navigation-action-loss-weight",
            "1.2",
            "--bc-pipeline-frontier-exploration-action-loss-weight",
            "0.55",
            "--bc-pipeline-frontier-exploration-min-map-size",
            "8",
            "--bc-pipeline-sync-action-loss-weight",
            "1.3",
            "--bc-pipeline-station-guard-action-loss-weight",
            "1.05",
            "--bc-pipeline-pickup-gate-loss-weight",
            "0.9",
            "--bc-pipeline-pickup-gate-pos-weight",
            "2.2",
            "--bc-pipeline-pickup-gate-neg-weight",
            "1.3",
            "--bc-pipeline-plan-action-loss-weight",
            "1.4",
            "--bc-pipeline-message-loss-weight",
            "1.6",
            "--bc-pipeline-send-gate-loss-weight",
            "0.8",
            "--bc-pipeline-send-gate-pos-weight",
            "1.7",
            "--bc-pipeline-send-gate-neg-weight",
            "1.2",
            "--bc-pipeline-interact-gate-loss-weight",
            "0.7",
            "--bc-pipeline-interact-gate-pos-weight",
            "2.4",
            "--bc-pipeline-interact-gate-neg-weight",
            "1.1",
            "--bc-pipeline-ready-interact-action-loss-weight",
            "1.15",
            "--bc-calibrate-pipeline-interact-gate-threshold",
            "--bc-pipeline-interact-gate-threshold-target-rate",
            "0.33",
            "--bc-pipeline-proactive-bad-action-labels",
            "--pipeline-stage-count",
            "1",
            "--pipeline-required-per-stage-max",
            "1",
            "--pipeline-sync-probability",
            "0",
            "--pipeline-dependency-probability",
            "0",
            "--pipeline-stage-count-schedule",
            "1,2",
            "--pipeline-required-per-stage-max-schedule",
            "1,1",
            "--pipeline-sync-probability-schedule",
            "0,0.25",
            "--rl-updates",
            "2",
            "--rl-updates-schedule",
            "0,2",
            "--rl-early-stop-eval-patience",
            "1",
            "--rollout-steps",
            "24",
            "--rl-rollout-eval-decoding",
            "--rl-rollout-pipeline-navigation-assist",
            "--rl-rollout-pipeline-navigation-assist-trust-messages",
            "--rl-rollout-pipeline-station-interact-guard",
            "--pipeline-assisted-rollout-episodes",
            "4",
            "--pipeline-assisted-rollout-seed-base",
            "5100",
            "--pipeline-assisted-rollout-seed-list",
            "8:5100,5101",
            "--pipeline-assisted-rollout-max-steps-per-episode",
            "14",
            "--pipeline-assisted-rollout-weight",
            "2.75",
            "--pipeline-assisted-rollout-success-only",
            "--no-pipeline-assisted-rollout-navigation-assist",
            "--no-pipeline-assisted-rollout-navigation-assist-trust-messages",
            "--pipeline-assisted-rollout-station-interact-guard",
            "--pipeline-assisted-rollout-bc-epochs",
            "2",
            "--no-dagger-restore-best",
            "--rl-eval-decoding-action-loss-weight",
            "0.45",
            "--rl-pipeline-assisted-action-loss-weight",
            "0.66",
            "--rl-pipeline-interact-gate-loss-weight",
            "0.55",
            "--rl-pipeline-interact-gate-pos-weight",
            "2.0",
            "--rl-pipeline-interact-gate-neg-weight",
            "3.0",
            "--rl-pipeline-pickup-gate-loss-weight",
            "0.75",
            "--rl-pipeline-pickup-gate-pos-weight",
            "2.5",
            "--rl-pipeline-pickup-gate-neg-weight",
            "3.5",
            "--rl-pipeline-delivery-progress-action-loss-weight",
            "0.6",
            "--rl-pipeline-navigation-action-loss-weight",
            "0.7",
            "--rl-pipeline-sync-action-loss-weight",
            "0.7",
            "--rl-pipeline-ready-interact-action-loss-weight",
            "0.95",
            "--rl-pipeline-station-guard-action-loss-weight",
            "0.55",
            "--rl-pipeline-wrong-station-recovery-action-loss-weight",
            "0.85",
            "--rl-pipeline-plan-action-loss-weight",
            "0.65",
            "--rl-pipeline-plan-head-loss-weight",
            "0.72",
            "--rl-pipeline-option-loss-weight",
            "0.82",
            "--rl-pipeline-bad-pickup-penalty",
            "0.2",
            "--rl-pipeline-bad-interact-penalty",
            "0.15",
            "--rl-pipeline-unneeded-drop-bonus",
            "0.075",
            "--rl-eval-every",
            "1",
            "--rl-eval-episodes",
            "3",
            "--no-rl-eval-use-eval-seeds",
            "--rl-eval-seed",
            "4321",
            "--rl-eval-seed-stage-stride",
            "123",
            "--eval-seed",
            "1234",
            "--eval-seed-count",
            "1",
            "--eval-pipeline-navigation-assist",
            "--eval-pipeline-navigation-assist-trust-messages",
            "--eval-pipeline-interact-gate-threshold",
            "0.42",
            "--eval-pipeline-event-head-threshold",
            "0.41",
            "--eval-pipeline-navigation-head-threshold",
            "0.43",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "pipeline_cli",
            "--dry-run",
        ],
    )

    recurrent_curriculum.main()
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "dry_run"
    assert result["config"]["scenario"] == "pipeline_assembly"
    assert result["config"]["agents"] == 3
    assert result["config"]["fov_preset"] == "easy"
    assert result["config"]["oracle_type"] == "planner_comm"
    assert result["config"]["bc_seq_len"] == 12
    assert result["config"]["bc_action_class_balance"] is True
    assert result["config"]["bc_action_class_balance_max_weight"] == pytest.approx(8.0)
    assert result["config"]["bc_event_action_weight"] == pytest.approx(3.0)
    assert result["config"]["bc_event_action_events"] == "picked_resource,delivered"
    assert result["config"]["comm_token_limit"] == 6
    assert result["config"]["comm_vocab_size"] == 48
    assert result["config"]["obs_pipeline_features"] is True
    assert result["config"]["bc_pipeline_delivery_progress_action_loss_weight"] == pytest.approx(0.8)
    assert result["config"]["bc_pipeline_navigation_action_loss_weight"] == pytest.approx(1.2)
    assert result["config"]["bc_pipeline_frontier_exploration_action_loss_weight"] == pytest.approx(0.55)
    assert result["config"]["bc_pipeline_frontier_exploration_min_map_size"] == 8
    assert result["config"]["bc_pipeline_sync_action_loss_weight"] == pytest.approx(1.3)
    assert result["config"]["bc_pipeline_ready_interact_action_loss_weight"] == pytest.approx(1.15)
    assert result["config"]["bc_pipeline_station_guard_action_loss_weight"] == pytest.approx(1.05)
    assert result["config"]["bc_pipeline_pickup_gate_loss_weight"] == pytest.approx(0.9)
    assert result["config"]["bc_pipeline_pickup_gate_pos_weight"] == pytest.approx(2.2)
    assert result["config"]["bc_pipeline_pickup_gate_neg_weight"] == pytest.approx(1.3)
    assert result["config"]["bc_pipeline_plan_action_loss_weight"] == pytest.approx(1.4)
    assert result["config"]["bc_pipeline_message_loss_weight"] == pytest.approx(1.6)
    assert result["config"]["bc_pipeline_send_gate_loss_weight"] == pytest.approx(0.8)
    assert result["config"]["bc_pipeline_send_gate_pos_weight"] == pytest.approx(1.7)
    assert result["config"]["bc_pipeline_send_gate_neg_weight"] == pytest.approx(1.2)
    assert result["config"]["bc_pipeline_interact_gate_loss_weight"] == pytest.approx(0.7)
    assert result["config"]["bc_pipeline_interact_gate_pos_weight"] == pytest.approx(2.4)
    assert result["config"]["bc_pipeline_interact_gate_neg_weight"] == pytest.approx(1.1)
    assert result["config"]["bc_calibrate_pipeline_interact_gate_threshold"] is True
    assert result["config"]["bc_pipeline_interact_gate_threshold_target_rate"] == pytest.approx(0.33)
    assert result["config"]["bc_pipeline_proactive_bad_action_labels"] is True
    assert result["config"]["eval_pipeline_navigation_assist"] is True
    assert result["config"]["eval_pipeline_navigation_assist_trust_messages"] is True
    assert result["config"]["eval_pipeline_interact_gate_threshold"] == pytest.approx(0.42)
    assert result["config"]["eval_pipeline_event_head_threshold"] == pytest.approx(0.41)
    assert result["config"]["eval_pipeline_navigation_head_threshold"] == pytest.approx(0.43)
    assert result["config"]["rl_rollout_eval_decoding"] is True
    assert result["config"]["rl_rollout_pipeline_navigation_assist"] is True
    assert result["config"]["rl_rollout_pipeline_navigation_assist_trust_messages"] is True
    assert result["config"]["rl_rollout_pipeline_station_interact_guard"] is True
    assert result["config"]["pipeline_assisted_rollout_episodes"] == 4
    assert result["config"]["pipeline_assisted_rollout_seed_base"] == 5100
    assert result["config"]["pipeline_assisted_rollout_seed_list"] == "8:5100,5101"
    assert result["config"]["pipeline_assisted_rollout_max_steps_per_episode"] == 14
    assert result["config"]["pipeline_assisted_rollout_weight"] == pytest.approx(2.75)
    assert result["config"]["pipeline_assisted_rollout_success_only"] is True
    assert result["config"]["pipeline_assisted_rollout_navigation_assist"] is False
    assert result["config"]["pipeline_assisted_rollout_navigation_assist_trust_messages"] is False
    assert result["config"]["pipeline_assisted_rollout_station_interact_guard"] is True
    assert result["config"]["pipeline_assisted_rollout_bc_epochs"] == 2
    assert result["config"]["rl_eval_decoding_action_loss_weight"] == pytest.approx(0.45)
    assert result["config"]["rl_pipeline_assisted_action_loss_weight"] == pytest.approx(0.66)
    assert result["config"]["rl_pipeline_interact_gate_loss_weight"] == pytest.approx(0.55)
    assert result["config"]["rl_pipeline_interact_gate_pos_weight"] == pytest.approx(2.0)
    assert result["config"]["rl_pipeline_interact_gate_neg_weight"] == pytest.approx(3.0)
    assert result["config"]["rl_pipeline_pickup_gate_loss_weight"] == pytest.approx(0.75)
    assert result["config"]["rl_pipeline_pickup_gate_pos_weight"] == pytest.approx(2.5)
    assert result["config"]["rl_pipeline_pickup_gate_neg_weight"] == pytest.approx(3.5)
    assert result["config"]["rl_pipeline_delivery_progress_action_loss_weight"] == pytest.approx(0.6)
    assert result["config"]["rl_pipeline_navigation_action_loss_weight"] == pytest.approx(0.7)
    assert result["config"]["rl_pipeline_sync_action_loss_weight"] == pytest.approx(0.7)
    assert result["config"]["rl_pipeline_ready_interact_action_loss_weight"] == pytest.approx(0.95)
    assert result["config"]["rl_pipeline_station_guard_action_loss_weight"] == pytest.approx(0.55)
    assert result["config"]["rl_pipeline_wrong_station_recovery_action_loss_weight"] == pytest.approx(0.85)
    assert result["config"]["rl_pipeline_plan_action_loss_weight"] == pytest.approx(0.65)
    assert result["config"]["rl_pipeline_plan_head_loss_weight"] == pytest.approx(0.72)
    assert result["config"]["rl_pipeline_option_loss_weight"] == pytest.approx(0.82)
    assert result["config"]["dagger_restore_best"] is False
    assert result["config"]["rl_pipeline_bad_pickup_penalty"] == pytest.approx(0.2)
    assert result["config"]["rl_pipeline_bad_interact_penalty"] == pytest.approx(0.15)
    assert result["config"]["rl_pipeline_unneeded_drop_bonus"] == pytest.approx(0.075)
    assert result["config"]["pipeline_stage_count"] == 1
    assert result["config"]["pipeline_required_per_stage_max"] == 1
    assert result["config"]["pipeline_sync_probability"] == pytest.approx(0.0)
    assert result["config"]["pipeline_dependency_probability"] == pytest.approx(0.0)
    assert result["config"]["pipeline_stage_count_schedule"] == "1,2"
    assert result["planned_stages"][0]["pipeline"]["stage_count"] == 1
    assert result["planned_stages"][0]["pipeline"]["required_per_stage_max"] == 1
    assert result["planned_stages"][0]["pipeline"]["sync_probability"] == pytest.approx(0.0)
    assert result["config"]["rl_updates"] == 2
    assert result["config"]["rl_updates_schedule"] == "0,2"
    assert result["config"]["rl_early_stop_eval_patience"] == 1
    assert result["config"]["rollout_steps"] == 24
    assert result["config"]["rl_eval_every"] == 1
    assert result["config"]["rl_eval_episodes"] == 3
    assert result["config"]["rl_eval_use_eval_seeds"] is False
    assert result["config"]["rl_eval_seed"] == 4321
    assert result["config"]["rl_eval_seed_stage_stride"] == 123
    assert result["planned_stages"][0]["rl_updates"] == 0
    assert result["planned_stages"][0]["rl_eval_use_eval_seeds"] is False
    assert result["config"]["eval_seed"] == 1234
    assert result["planned_stages"][0]["train_map_sizes"] == [8]


def test_recurrent_curriculum_cli_signal_defaults_enable_decision_heads(
    tmp_path,
    monkeypatch,
    capsys,
):
    import json
    import sys

    from syncorsink.train import recurrent_curriculum

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recurrent_curriculum.py",
            "--stage-map-suites",
            "8",
            "--max-steps-by-map",
            "8:40",
            "--demo-episodes",
            "1",
            "--bc-epochs",
            "1",
            "--dagger-rounds",
            "0",
            "--rl-updates",
            "0",
            "--eval-episodes",
            "1",
            "--eval-seed-count",
            "1",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "signal_cli_defaults",
            "--dry-run",
        ],
    )

    recurrent_curriculum.main()
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "dry_run"
    assert result["config"]["scenario"] == "signal_hunt"
    assert result["config"]["bc_signal_scan_decision_loss_weight"] == pytest.approx(1.0)
    assert result["config"]["bc_signal_scan_decision_pos_weight"] == pytest.approx(2.0)
    assert result["config"]["bc_signal_scan_decision_neg_weight"] == pytest.approx(3.0)
    assert result["config"]["bc_signal_scan_gate_loss_weight"] == pytest.approx(1.0)
    assert result["config"]["bc_signal_scan_gate_pos_weight"] == pytest.approx(2.0)
    assert result["config"]["bc_signal_scan_gate_neg_weight"] == pytest.approx(3.0)
    assert result["config"]["bc_signal_target_validity_loss_weight"] == pytest.approx(1.0)
    assert result["config"]["bc_signal_target_validity_pos_weight"] == pytest.approx(2.0)
    assert result["config"]["bc_signal_target_validity_neg_weight"] == pytest.approx(3.0)
    assert result["config"]["bc_signal_target_decision_loss_weight"] == pytest.approx(1.0)
    assert result["config"]["bc_signal_target_decision_pos_weight"] == pytest.approx(2.0)
    assert result["config"]["bc_signal_target_decision_neg_weight"] == pytest.approx(3.0)
    assert result["config"]["bc_signal_decoy_drift_action_loss_weight"] == pytest.approx(0.25)
    assert result["config"]["bc_signal_decoy_scan_action_loss_weight"] == pytest.approx(0.1)
    assert result["config"]["bc_signal_rejected_target_drift_action_loss_weight"] == pytest.approx(0.0)
    assert result["config"]["eval_signal_scan_gate_threshold"] == pytest.approx(0.4)
    assert result["config"]["eval_signal_scan_gate_suppress"] is True
    assert result["config"]["eval_signal_target_validity_threshold"] == pytest.approx(0.4)
    assert result["config"]["eval_signal_target_decision_threshold"] == pytest.approx(0.4)
    assert result["config"]["eval_signal_initial_exact_message_copy_assist"] is True


def test_recurrent_curriculum_pipeline_schedules_override_stage_defaults(tmp_path):
    from syncorsink.train.recurrent_curriculum import (
        RecurrentCurriculumConfig,
        _stage_recurrent_config,
        run_recurrent_curriculum,
    )

    cfg = RecurrentCurriculumConfig(
        scenario="pipeline_assembly",
        agents=3,
        oracle_type="planner_comm",
        stage_map_suites="8;8",
        max_steps_by_map="8:60",
        pipeline_stage_count=4,
        pipeline_required_per_stage_min=1,
        pipeline_required_per_stage_max=2,
        pipeline_sync_probability=0.5,
        pipeline_dependency_probability=0.7,
        pipeline_stage_count_schedule="1,2",
        pipeline_required_per_stage_min_schedule="1,1",
        pipeline_required_per_stage_max_schedule="1,1",
        pipeline_sync_probability_schedule="0,0.25",
        pipeline_dependency_probability_schedule="0,0.5",
        rl_updates=9,
        rl_updates_schedule="0,4",
        rl_early_stop_eval_patience=1,
        rl_eval_use_eval_seeds=True,
        rl_eval_seed=5000,
        rl_eval_seed_stage_stride=100,
        rl_eval_decoding_action_loss_weight=0.35,
        rl_pipeline_interact_gate_loss_weight=0.55,
        rl_pipeline_interact_gate_pos_weight=2.0,
        rl_pipeline_interact_gate_neg_weight=3.0,
        rl_pipeline_pickup_gate_loss_weight=0.75,
        rl_pipeline_pickup_gate_pos_weight=2.5,
        rl_pipeline_pickup_gate_neg_weight=3.5,
        rl_pipeline_delivery_progress_action_loss_weight=0.6,
        rl_pipeline_navigation_action_loss_weight=0.7,
        rl_pipeline_sync_action_loss_weight=0.7,
        rl_pipeline_ready_interact_action_loss_weight=0.95,
        rl_pipeline_station_guard_action_loss_weight=0.45,
        rl_pipeline_wrong_station_recovery_action_loss_weight=0.85,
        rl_pipeline_plan_action_loss_weight=0.65,
        rl_pipeline_plan_head_loss_weight=0.72,
        rl_pipeline_option_loss_weight=0.82,
        rl_pipeline_bad_pickup_penalty=0.2,
        rl_pipeline_bad_interact_penalty=0.15,
        rl_pipeline_unneeded_drop_bonus=0.075,
        bc_pipeline_pickup_action_loss_weight=0.25,
        bc_pipeline_delivery_action_loss_weight=0.5,
        bc_pipeline_delivery_progress_action_loss_weight=0.8,
        bc_pipeline_navigation_action_loss_weight=1.2,
        bc_pipeline_frontier_exploration_action_loss_weight=0.55,
        bc_pipeline_frontier_exploration_min_map_size=8,
        bc_pipeline_sync_action_loss_weight=1.3,
        bc_pipeline_ready_interact_action_loss_weight=1.15,
        bc_pipeline_station_guard_action_loss_weight=1.05,
        bc_pipeline_pickup_gate_loss_weight=0.9,
        bc_pipeline_pickup_gate_pos_weight=2.2,
        bc_pipeline_pickup_gate_neg_weight=1.3,
        bc_pipeline_plan_action_loss_weight=1.75,
        bc_pipeline_message_loss_weight=2.25,
        bc_pipeline_send_gate_loss_weight=1.5,
        bc_pipeline_send_gate_pos_weight=2.25,
        bc_pipeline_send_gate_neg_weight=1.5,
        bc_pipeline_interact_gate_loss_weight=1.25,
        bc_pipeline_interact_gate_pos_weight=2.75,
        bc_pipeline_interact_gate_neg_weight=1.25,
        bc_calibrate_pipeline_interact_gate_threshold=True,
        bc_pipeline_interact_gate_threshold_target_rate=0.4,
        bc_pipeline_bad_pickup_action_loss_weight=0.6,
        bc_pipeline_bad_drop_action_loss_weight=0.75,
        bc_pipeline_bad_interact_action_loss_weight=1.25,
        eval_pipeline_interact_gate_threshold=0.37,
        eval_pipeline_event_head_threshold=0.36,
        eval_pipeline_navigation_head_threshold=0.38,
        dagger_pipeline_wrong_delivery_provenance_labels=True,
        dagger_pipeline_wrong_delivery_provenance_weight=1.5,
        output_dir=str(tmp_path),
        run_name="pipeline_schedule",
        dry_run=True,
    )
    result = run_recurrent_curriculum(cfg)

    assert result["planned_stages"][0]["pipeline"] == {
        "stage_count": 1,
        "required_per_stage_min": 1,
        "required_per_stage_max": 1,
        "sync_probability": 0.0,
        "dependency_probability": 0.0,
    }
    assert result["planned_stages"][1]["pipeline"] == {
        "stage_count": 2,
        "required_per_stage_min": 1,
        "required_per_stage_max": 1,
        "sync_probability": 0.25,
        "dependency_probability": 0.5,
    }
    assert result["planned_stages"][0]["rl_updates"] == 0
    assert result["planned_stages"][1]["rl_updates"] == 4
    assert result["planned_stages"][1]["rl_eval_use_eval_seeds"] is True

    stage_cfg = _stage_recurrent_config(
        cfg,
        stage_idx=1,
        suite=(8,),
        max_steps={8: 60},
        checkpoint_path=tmp_path / "stage1.pt",
        eval_send_threshold=0.25,
        has_initial_model=True,
    )
    assert stage_cfg.scenario == "pipeline_assembly"
    assert stage_cfg.agents == 3
    assert stage_cfg.oracle_type == "planner_comm"
    assert stage_cfg.pipeline_stage_count == 2
    assert stage_cfg.pipeline_required_per_stage_max == 1
    assert stage_cfg.pipeline_sync_probability == pytest.approx(0.25)
    assert stage_cfg.pipeline_dependency_probability == pytest.approx(0.5)
    assert stage_cfg.rl_updates == 4
    assert stage_cfg.rl_early_stop_eval_patience == 1
    assert stage_cfg.rl_eval_use_eval_seeds is True
    assert stage_cfg.rl_eval_seed == 5100
    assert stage_cfg.rl_eval_decoding_action_loss_weight == pytest.approx(0.35)
    assert stage_cfg.rl_pipeline_interact_gate_loss_weight == pytest.approx(0.55)
    assert stage_cfg.rl_pipeline_interact_gate_pos_weight == pytest.approx(2.0)
    assert stage_cfg.rl_pipeline_interact_gate_neg_weight == pytest.approx(3.0)
    assert stage_cfg.rl_pipeline_pickup_gate_loss_weight == pytest.approx(0.75)
    assert stage_cfg.rl_pipeline_pickup_gate_pos_weight == pytest.approx(2.5)
    assert stage_cfg.rl_pipeline_pickup_gate_neg_weight == pytest.approx(3.5)
    assert stage_cfg.rl_pipeline_delivery_progress_action_loss_weight == pytest.approx(0.6)
    assert stage_cfg.rl_pipeline_navigation_action_loss_weight == pytest.approx(0.7)
    assert stage_cfg.rl_pipeline_sync_action_loss_weight == pytest.approx(0.7)
    assert stage_cfg.rl_pipeline_ready_interact_action_loss_weight == pytest.approx(0.95)
    assert stage_cfg.rl_pipeline_station_guard_action_loss_weight == pytest.approx(0.45)
    assert stage_cfg.rl_pipeline_wrong_station_recovery_action_loss_weight == pytest.approx(0.85)
    assert stage_cfg.rl_pipeline_plan_action_loss_weight == pytest.approx(0.65)
    assert stage_cfg.rl_pipeline_plan_head_loss_weight == pytest.approx(0.72)
    assert stage_cfg.rl_pipeline_option_loss_weight == pytest.approx(0.82)
    assert stage_cfg.rl_pipeline_bad_pickup_penalty == pytest.approx(0.2)
    assert stage_cfg.rl_pipeline_bad_interact_penalty == pytest.approx(0.15)
    assert stage_cfg.rl_pipeline_unneeded_drop_bonus == pytest.approx(0.075)
    assert stage_cfg.obs_pipeline_features is True
    assert stage_cfg.bc_pipeline_pickup_action_loss_weight == pytest.approx(0.25)
    assert stage_cfg.bc_pipeline_delivery_action_loss_weight == pytest.approx(0.5)
    assert stage_cfg.bc_pipeline_delivery_progress_action_loss_weight == pytest.approx(0.8)
    assert stage_cfg.bc_pipeline_navigation_action_loss_weight == pytest.approx(1.2)
    assert stage_cfg.bc_pipeline_frontier_exploration_action_loss_weight == pytest.approx(0.55)
    assert stage_cfg.bc_pipeline_frontier_exploration_min_map_size == 8
    assert stage_cfg.bc_pipeline_sync_action_loss_weight == pytest.approx(1.3)
    assert stage_cfg.bc_pipeline_ready_interact_action_loss_weight == pytest.approx(1.15)
    assert stage_cfg.bc_pipeline_station_guard_action_loss_weight == pytest.approx(1.05)
    assert stage_cfg.bc_pipeline_pickup_gate_loss_weight == pytest.approx(0.9)
    assert stage_cfg.bc_pipeline_pickup_gate_pos_weight == pytest.approx(2.2)
    assert stage_cfg.bc_pipeline_pickup_gate_neg_weight == pytest.approx(1.3)
    assert stage_cfg.bc_pipeline_plan_action_loss_weight == pytest.approx(1.75)
    assert stage_cfg.bc_pipeline_message_loss_weight == pytest.approx(2.25)
    assert stage_cfg.bc_pipeline_send_gate_loss_weight == pytest.approx(1.5)
    assert stage_cfg.bc_pipeline_send_gate_pos_weight == pytest.approx(2.25)
    assert stage_cfg.bc_pipeline_send_gate_neg_weight == pytest.approx(1.5)
    assert stage_cfg.bc_pipeline_interact_gate_loss_weight == pytest.approx(1.25)
    assert stage_cfg.bc_pipeline_interact_gate_pos_weight == pytest.approx(2.75)
    assert stage_cfg.bc_pipeline_interact_gate_neg_weight == pytest.approx(1.25)
    assert stage_cfg.bc_calibrate_pipeline_interact_gate_threshold is True
    assert stage_cfg.bc_pipeline_interact_gate_threshold_target_rate == pytest.approx(0.4)
    assert stage_cfg.bc_pipeline_bad_pickup_action_loss_weight == pytest.approx(0.6)
    assert stage_cfg.bc_pipeline_bad_drop_action_loss_weight == pytest.approx(0.75)
    assert stage_cfg.bc_pipeline_bad_interact_action_loss_weight == pytest.approx(1.25)
    assert stage_cfg.eval_pipeline_interact_gate_threshold == pytest.approx(0.37)
    assert stage_cfg.eval_pipeline_event_head_threshold == pytest.approx(0.36)
    assert stage_cfg.eval_pipeline_navigation_head_threshold == pytest.approx(0.38)
    assert stage_cfg.dagger_pipeline_wrong_delivery_provenance_labels is True
    assert stage_cfg.dagger_pipeline_wrong_delivery_provenance_weight == pytest.approx(1.5)


def test_recurrent_dagger_collection_seed_schedule():
    from syncorsink.train.recurrent_bc_rl import (
        PIPELINE_WRONG_DELIVERY_ROOT_PICKUP_EVENT,
        RecurrentConfig,
        _dagger_collection_map_ordinal,
        _dagger_collection_seed,
        _pipeline_assisted_rollout_collection_seed,
    )

    default_cfg = RecurrentConfig(dagger_episodes=3)
    assert [_dagger_collection_seed(default_cfg, 0, ep) for ep in range(3)] == [10000, 10001, 10002]
    assert [_dagger_collection_seed(default_cfg, 1, ep) for ep in range(3)] == [11000, 11001, 11002]

    offset_cfg = RecurrentConfig(
        dagger_episodes=2,
        dagger_seed_base=3000,
        dagger_seed_stride=17,
    )
    assert [_dagger_collection_seed(offset_cfg, 0, ep) for ep in range(2)] == [3000, 3001]
    assert [_dagger_collection_seed(offset_cfg, 1, ep) for ep in range(2)] == [3017, 3018]

    explicit_cfg = RecurrentConfig(
        dagger_episodes=4,
        dagger_seed_list="3002,3003,3020",
    )
    assert [_dagger_collection_seed(explicit_cfg, 0, ep) for ep in range(4)] == [
        3002,
        3003,
        3020,
        3002,
    ]
    assert [_dagger_collection_seed(explicit_cfg, 1, ep) for ep in range(4)] == [
        3003,
        3020,
        3002,
        3003,
    ]

    mixed_cfg = RecurrentConfig(
        map_size=16,
        train_map_sizes="16,32",
        dagger_episodes=4,
        dagger_seed_list="16:160,161+32:320,321,322",
    )
    assert [_dagger_collection_seed(mixed_cfg, 0, ep) for ep in range(4)] == [
        160,
        320,
        161,
        321,
    ]
    assert [_dagger_collection_seed(mixed_cfg, 1, ep) for ep in range(4)] == [
        160,
        322,
        161,
        320,
    ]
    assert [_dagger_collection_map_ordinal(mixed_cfg, ep, 32) for ep in range(1, 8, 2)] == [
        0,
        1,
        2,
        3,
    ]

    invalid_cfg = RecurrentConfig(dagger_seed_list="3002,-1")
    with pytest.raises(ValueError, match="dagger_seed_list"):
        _dagger_collection_seed(invalid_cfg, 0, 0)

    missing_map_cfg = RecurrentConfig(
        map_size=16,
        train_map_sizes="16,32",
        dagger_episodes=2,
        dagger_seed_list="16:160,161",
    )
    with pytest.raises(ValueError, match="map_size=32"):
        _dagger_collection_seed(missing_map_cfg, 0, 1)

    assisted_cfg = RecurrentConfig(
        map_size=16,
        train_map_sizes="16,32",
        pipeline_assisted_rollout_seed_base=7000,
        pipeline_assisted_rollout_seed_list="16:1600+32:3200,3201",
    )
    assert [
        _pipeline_assisted_rollout_collection_seed(assisted_cfg, ep)
        for ep in range(4)
    ] == [1600, 3200, 1600, 3201]


def test_pipeline_assisted_rollout_collection_smoke():
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.mappo import resolve_device
    from syncorsink.train.recurrent_bc_rl import (
        PIPELINE_OPTION_COUNT,
        RecurrentConfig,
        _recurrent_fov_radius,
        _recurrent_training_obs_shape,
        collect_pipeline_assisted_rollout_episodes,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=3,
        fov_preset="easy",
        max_steps=20,
        oracle_type="planner_comm",
        obs_pipeline_features=True,
        obs_feedback=True,
        obs_pipeline_feedback=True,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=32,
        hidden_dim=16,
        pipeline_assisted_rollout_episodes=1,
        pipeline_assisted_rollout_max_steps_per_episode=6,
        pipeline_assisted_rollout_weight=1.5,
        pipeline_assisted_rollout_seed_base=123,
    )
    device = resolve_device("cpu")
    model = MAPPORecurrentActor(
        obs_dim=_recurrent_training_obs_shape(cfg)[0],
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        backbone=cfg.recurrent_backbone,
        fov_radius=_recurrent_fov_radius(cfg),
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
        pipeline_option_dim=PIPELINE_OPTION_COUNT,
    ).to(device)

    episodes, summary = collect_pipeline_assisted_rollout_episodes(cfg, model, device)

    assert summary["attempted_episodes"] == 1
    assert summary["episodes"] == 1
    assert summary["transitions"] > 0
    assert summary["navigation_assist"] is True
    assert summary["station_interact_guard"] is True
    episode = episodes[0]
    assert episode["source"] == "pipeline_assisted_rollout"
    assert episode["seed"] == 123
    assert episode["weight"] == pytest.approx(1.5)
    assert episode["obs"].shape[1] == cfg.agents
    assert episode["actions"].shape[:2] == episode["msg_lens"].shape


def test_recurrent_skip_bc_stage_requires_checkpoint_and_no_dagger():
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _should_run_recurrent_bc_stage

    assert _should_run_recurrent_bc_stage(RecurrentConfig()) is True
    assert _should_run_recurrent_bc_stage(
        RecurrentConfig(recurrent_init="checkpoint.pt", recurrent_init_for_dagger=False)
    ) is False
    assert _should_run_recurrent_bc_stage(
        RecurrentConfig(
            recurrent_init="checkpoint.pt",
            recurrent_init_for_dagger=True,
            skip_bc=True,
            rl_updates=1,
        )
    ) is False

    with pytest.raises(ValueError, match="requires --recurrent-init"):
        _should_run_recurrent_bc_stage(RecurrentConfig(skip_bc=True))

    with pytest.raises(ValueError, match="--dagger-rounds"):
        _should_run_recurrent_bc_stage(
            RecurrentConfig(
                recurrent_init="checkpoint.pt",
                skip_bc=True,
                dagger_rounds=1,
            )
        )


def test_recurrent_rollout_eval_decoding_updates_actions_and_comm_tensors():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_CLUE, TILE_TARGET
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_recurrent_rollout_eval_decoding,
        _comm_tensors_from_actions,
        _feedback_dim,
        _initial_signal_scan_state,
    )

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
    ))
    obs, _ = env.reset(seed=0)
    target_pos = tuple(env.agent_positions[0])
    obs[0]["self_pos"] = np.asarray(target_pos, dtype=np.int16)
    obs[0]["local_grid"][obs[0]["local_grid"].shape[0] // 2, obs[0]["local_grid"].shape[1] // 2] = TILE_TARGET
    obs[0]["action_mask"] = np.ones((8,), dtype=np.float32)

    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=32,
        obs_feedback=True,
        obs_signal_scan_state=True,
        rl_rollout_eval_decoding=True,
        eval_signal_scan_sync_assist=True,
        eval_signal_scan_broadcast_assist=True,
    )
    feedback = np.zeros((2, _feedback_dim(cfg)), dtype=np.float32)
    scan_offset = 12
    feedback[0, scan_offset] = 1.0
    feedback[0, scan_offset + 2] = 1.0
    scan_state = _initial_signal_scan_state(cfg)
    scan_state["scan_log"] = {0: 0}
    scan_state["scan_pos"] = {0: target_pos}
    scan_state["step"] = 0
    logits = torch.zeros((2, 8), dtype=torch.float32)
    acts = torch.tensor([env.ACTION_INTERACT, env.ACTION_STAY], dtype=torch.long)
    actions = {
        0: {"action": env.ACTION_INTERACT, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    }
    hidden = (torch.zeros((2, 4), dtype=torch.float32), torch.zeros((2, 4), dtype=torch.float32))

    decoded_acts, decoded_actions = _apply_recurrent_rollout_eval_decoding(
        cfg,
        object(),
        obs,
        logits,
        acts,
        actions,
        hidden,
        feedback,
        scan_state,
    )

    assert int(decoded_acts[0].item()) == env.ACTION_STAY
    assert decoded_actions[0]["message_tokens"] == [26, int(target_pos[0]), int(target_pos[1])]
    send, tokens, lengths = _comm_tensors_from_actions(decoded_actions, cfg, torch.device("cpu"))
    assert send.tolist() == [1.0, 0.0]
    assert lengths.tolist() == [3, 0]
    assert tokens[0, :3].tolist() == [26, int(target_pos[0]), int(target_pos[1])]


def test_recurrent_dagger_caps_and_weights_failed_rollouts():
    from syncorsink.train.mappo import resolve_device
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        collect_episode_demos,
        collect_recurrent_dagger_episodes,
        train_recurrent_bc,
    )

    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=30,
        oracle_type="signal_hint_comm",
        obs_exploration_memory=True,
        obs_feedback=True,
        obs_signal_sync_feedback=True,
        obs_signal_scan_state=True,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=8,
        demo_episodes=2,
        bc_epochs=1,
        bc_seq_len=8,
        bc_comm_loss_weight=0.1,
        bc_comm_send_pos_weight=-1,
        dagger_episodes=1,
        dagger_max_steps_per_episode=2,
        dagger_failed_episode_weight=0.125,
        hidden_dim=32,
        eval_episodes=1,
        device="cpu",
    )

    device = resolve_device(cfg.device)
    episodes = collect_episode_demos(cfg)
    model = train_recurrent_bc(cfg, episodes, device)
    dagger_episodes, summary = collect_recurrent_dagger_episodes(
        cfg,
        model,
        device,
        round_idx=0,
    )

    assert summary["episodes"] == 1
    assert summary["avg_stored_steps"] <= 2
    assert summary["transitions"] <= 4
    assert summary["effective_transitions"] <= summary["transitions"]
    assert summary["oracle_message_rollin_rate"] == 0.0
    assert summary["oracle_message_rollin_steps"] == 0
    assert summary["oracle_message_rollin_agents"] == 0
    assert summary["oracle_message_rollin_tokens"] == 0
    assert summary["oracle_action_rollin_rate"] == 0.0
    assert summary["oracle_action_rollin_steps"] == 0
    assert summary["oracle_action_rollin_agents"] == 0
    assert summary["seed_base"] == 10000
    assert summary["seed_stride"] == 1000
    assert summary["seed_list"] == []
    assert dagger_episodes[0]["source"] == "dagger"
    assert dagger_episodes[0]["obs"].shape[0] <= 2
    if not dagger_episodes[0]["success"]:
        assert dagger_episodes[0]["capped"] is True
        assert dagger_episodes[0]["weight"] == 0.125


def test_recurrent_dagger_focus_step_weight_helpers():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _append_labeled_step,
        _episode_count_effective_transitions,
        _episode_map_size_diagnostics,
        _event_names_by_agent,
        _finalize_episode_sequence,
        _focus_replay_episodes,
        _new_episode_sequence,
        _scale_latest_agent_weights,
    )

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
        comm_token_limit=4,
        token_vocab_size=8,
    ))
    obs, _ = env.reset(seed=0)
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=8,
    )
    ep_data = _new_episode_sequence()
    actions = {
        0: {"action": env.ACTION_STAY, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": [1, 2]},
    }

    _append_labeled_step(ep_data, obs, actions, env, cfg, step_weight=np.array([1.0, 2.0]))
    _append_labeled_step(ep_data, obs, actions, env, cfg, step_weight=np.array([3.0, 1.0]))
    _append_labeled_step(ep_data, obs, actions, env, cfg, step_weight=np.array([1.0, 1.0]))
    scaled = _scale_latest_agent_weights(
        ep_data,
        num_agents=env.num_agents,
        agent_ids=[0],
        weight=4.0,
    )
    episode = _finalize_episode_sequence(
        ep_data,
        env,
        cfg,
        source="dagger",
        map_size=8,
        success=False,
        weight=0.5,
    )
    replay_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=8,
        dagger_focus_replay=True,
        dagger_replay_pre_steps=1,
        dagger_replay_post_steps=1,
        dagger_replay_weight=2.0,
        dagger_max_replay_snippets_per_episode=1,
    )
    replay = _focus_replay_episodes(
        episode,
        [{"event": "decoy_scan", "step": 1, "agents": [0]}],
        replay_cfg,
    )
    event_names = _event_names_by_agent(
        {"events": {0: [{"event": "decoy_scan"}], "1": [{"event": "clue_found"}]}},
        env.num_agents,
    )

    assert scaled == 1
    np.testing.assert_allclose(
        episode["step_weights"],
        np.array([[1.0, 2.0], [3.0, 1.0], [4.0, 1.0]], dtype=np.float32),
    )
    assert _episode_count_effective_transitions([episode]) == 6.0
    target_xy = np.asarray(env.scenario_state.data["target"], dtype=np.float32) / float(env.map_size - 1)
    np.testing.assert_allclose(
        episode["signal_target_aux_mask"],
        np.ones((3, 2), dtype=np.float32),
    )
    np.testing.assert_allclose(
        episode["signal_target_aux_xy"],
        np.broadcast_to(target_xy, (3, 2, 2)).astype(np.float32),
    )
    assert len(replay) == 1
    assert replay[0]["source"] == "dagger_focus_replay"
    assert replay[0]["map_size"] == 8
    assert replay[0]["trigger_event"] == "decoy_scan"
    assert replay[0]["trigger_agents"] == [0]
    assert replay[0]["obs"].shape[0] == 3
    np.testing.assert_allclose(replay[0]["signal_target_aux_xy"], episode["signal_target_aux_xy"])
    assert _episode_count_effective_transitions(replay) == 24.0
    diagnostics = _episode_map_size_diagnostics([episode, *replay])
    assert diagnostics["8"]["episodes"] == 2
    assert diagnostics["8"]["sources"] == {"dagger": 1, "dagger_focus_replay": 1}
    assert diagnostics["8"]["replay_episodes"] == 1
    assert diagnostics["8"]["replay_trigger_events"] == {"decoy_scan": 1}
    assert diagnostics["8"]["failed_episodes"] == 2
    controlled_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=8,
        dagger_focus_replay=True,
        dagger_replay_pre_steps=0,
        dagger_replay_post_steps=0,
        dagger_replay_weight=2.0,
        dagger_replay_event_weights="decoy_scan:0.5,joint_target_scan:3.0",
        dagger_replay_event_caps="decoy_scan:1",
        dagger_max_replay_snippets_per_episode=3,
    )
    controlled_replay = _focus_replay_episodes(
        episode,
        [
            {"event": "decoy_scan", "step": 1, "agents": [0], "kind": "focus"},
            {"event": "decoy_scan", "step": 2, "agents": [1], "kind": "focus"},
            {"event": "joint_target_scan", "step": 2, "agents": [0, 1], "kind": "positive"},
        ],
        controlled_cfg,
    )
    assert [snippet["trigger_event"] for snippet in controlled_replay] == [
        "decoy_scan",
        "joint_target_scan",
    ]
    assert [snippet["trigger_kind"] for snippet in controlled_replay] == ["focus", "positive"]
    assert [snippet["weight"] for snippet in controlled_replay] == [0.5, 3.0]
    assert [snippet["obs"].shape[0] for snippet in controlled_replay] == [1, 1]
    balanced_cfg = RecurrentConfig(
        **{
            **vars(controlled_cfg),
            "dagger_replay_event_weights": "",
            "dagger_replay_event_caps": "",
            "dagger_max_replay_snippets_per_episode": 3,
            "dagger_replay_balance_positive_events": "first_target_scan,joint_target_scan",
            "dagger_replay_balance_negative_events": "decoy_scan,rejected_target_scan",
            "dagger_replay_max_negative_per_positive": 0.5,
        }
    )
    balanced_replay = _focus_replay_episodes(
        episode,
        [
            {"event": "decoy_scan", "step": 0, "agents": [0], "kind": "focus"},
            {"event": "rejected_target_scan", "step": 1, "agents": [1], "kind": "focus"},
            {"event": "first_target_scan", "step": 1, "agents": [0], "kind": "positive"},
            {"event": "joint_target_scan", "step": 2, "agents": [0, 1], "kind": "positive"},
            {"event": "target_pursuit", "step": 2, "agents": [1], "kind": "positive"},
        ],
        balanced_cfg,
    )
    assert [snippet["trigger_event"] for snippet in balanced_replay] == [
        "first_target_scan",
        "joint_target_scan",
        "target_pursuit",
    ]
    priority_balanced_cfg = RecurrentConfig(
        **{
            **vars(balanced_cfg),
            "dagger_replay_priority_events": "movement_stall_miss",
        }
    )
    priority_balanced_replay = _focus_replay_episodes(
        episode,
        [
            {"event": "first_target_scan", "step": 0, "agents": [0], "kind": "positive"},
            {"event": "joint_target_scan", "step": 1, "agents": [0, 1], "kind": "positive"},
            {"event": "target_pursuit", "step": 1, "agents": [1], "kind": "positive"},
            {"event": "movement_stall_miss", "step": 2, "agents": [1], "kind": "focus"},
        ],
        priority_balanced_cfg,
    )
    assert [snippet["trigger_event"] for snippet in priority_balanced_replay] == [
        "movement_stall_miss",
        "first_target_scan",
        "joint_target_scan",
    ]
    roomy_balanced_cfg = RecurrentConfig(
        **{
            **vars(balanced_cfg),
            "dagger_max_replay_snippets_per_episode": 4,
        }
    )
    roomy_balanced_replay = _focus_replay_episodes(
        episode,
        [
            {"event": "decoy_scan", "step": 0, "agents": [0], "kind": "focus"},
            {"event": "rejected_target_scan", "step": 1, "agents": [1], "kind": "focus"},
            {"event": "first_target_scan", "step": 1, "agents": [0], "kind": "positive"},
            {"event": "joint_target_scan", "step": 2, "agents": [0, 1], "kind": "positive"},
            {"event": "target_pursuit", "step": 2, "agents": [1], "kind": "positive"},
        ],
        roomy_balanced_cfg,
    )
    assert [snippet["trigger_event"] for snippet in roomy_balanced_replay] == [
        "first_target_scan",
        "joint_target_scan",
        "target_pursuit",
        "decoy_scan",
    ]
    expert_capped_cfg = RecurrentConfig(
        **{
            **vars(roomy_balanced_cfg),
            "dagger_expert_max_replay_snippets_per_episode": 2,
        }
    )
    expert_capped_replay = _focus_replay_episodes(
        episode,
        [
            {"event": "decoy_scan", "step": 0, "agents": [0], "kind": "focus"},
            {"event": "rejected_target_scan", "step": 1, "agents": [1], "kind": "focus"},
            {"event": "first_target_scan", "step": 1, "agents": [0], "kind": "positive"},
            {"event": "joint_target_scan", "step": 2, "agents": [0, 1], "kind": "positive"},
            {"event": "target_pursuit", "step": 2, "agents": [1], "kind": "positive"},
        ],
        expert_capped_cfg,
        source="expert_positive_replay",
    )
    assert [snippet["trigger_event"] for snippet in expert_capped_replay] == [
        "decoy_scan",
        "first_target_scan",
    ]
    expert_balanced_replay = _focus_replay_episodes(
        episode,
        [
            {"event": "decoy_scan", "step": 0, "agents": [0], "kind": "focus"},
            {"event": "rejected_target_scan", "step": 1, "agents": [1], "kind": "focus"},
            {"event": "first_target_scan", "step": 1, "agents": [0], "kind": "positive"},
            {"event": "joint_target_scan", "step": 2, "agents": [0, 1], "kind": "positive"},
            {"event": "target_pursuit", "step": 2, "agents": [1], "kind": "positive"},
        ],
        balanced_cfg,
        source="expert_positive_replay",
    )
    assert [snippet["trigger_event"] for snippet in expert_balanced_replay] == [
        "decoy_scan",
        "first_target_scan",
        "rejected_target_scan",
    ]
    success_only_cfg = RecurrentConfig(
        **{
            **vars(controlled_cfg),
            "dagger_replay_success_only_events": "joint_target_scan",
        }
    )
    filtered_replay = _focus_replay_episodes(
        episode,
        [
            {"event": "decoy_scan", "step": 1, "agents": [0], "kind": "focus"},
            {"event": "joint_target_scan", "step": 2, "agents": [0, 1], "kind": "positive"},
        ],
        success_only_cfg,
    )
    successful_episode = dict(episode)
    successful_episode["success"] = True
    successful_replay = _focus_replay_episodes(
        successful_episode,
        [
            {"event": "decoy_scan", "step": 1, "agents": [0], "kind": "focus"},
            {"event": "joint_target_scan", "step": 2, "agents": [0, 1], "kind": "positive"},
        ],
        success_only_cfg,
    )
    assert [snippet["trigger_event"] for snippet in filtered_replay] == ["decoy_scan"]
    assert [snippet["trigger_event"] for snippet in successful_replay] == [
        "decoy_scan",
        "joint_target_scan",
    ]
    conservative_failed_parent_cfg = RecurrentConfig(
        **{
            **vars(controlled_cfg),
            "dagger_replay_event_weights": "",
            "dagger_replay_event_caps": "",
            "dagger_max_replay_snippets_per_episode": 3,
            "dagger_max_failed_parent_replay_snippets_per_episode": 1,
            "dagger_failed_parent_replay_weight_scale": 0.25,
        }
    )
    conservative_failed_parent_replay = _focus_replay_episodes(
        episode,
        [
            {"event": "target_discovery_miss", "step": 0, "agents": [0], "kind": "focus"},
            {"event": "target_handoff", "step": 1, "agents": [1], "kind": "positive"},
            {"event": "target_pursuit_miss", "step": 2, "agents": [0], "kind": "focus"},
        ],
        conservative_failed_parent_cfg,
    )
    assert [snippet["trigger_event"] for snippet in conservative_failed_parent_replay] == [
        "target_discovery_miss",
    ]
    assert conservative_failed_parent_replay[0]["weight"] == pytest.approx(0.5)
    successful_parent_replay = _focus_replay_episodes(
        successful_episode,
        [
            {"event": "target_discovery_miss", "step": 0, "agents": [0], "kind": "focus"},
            {"event": "target_handoff", "step": 1, "agents": [1], "kind": "positive"},
            {"event": "target_pursuit_miss", "step": 2, "agents": [0], "kind": "focus"},
        ],
        conservative_failed_parent_cfg,
    )
    assert [snippet["trigger_event"] for snippet in successful_parent_replay] == [
        "target_discovery_miss",
        "target_handoff",
        "target_pursuit_miss",
    ]
    assert [snippet["weight"] for snippet in successful_parent_replay] == [2.0, 2.0, 2.0]
    expert_replay = _focus_replay_episodes(
        episode,
        [{"event": "joint_target_scan", "step": 2, "agents": [0, 1], "kind": "positive"}],
        controlled_cfg,
        source="expert_positive_replay",
    )
    assert expert_replay[0]["source"] == "expert_positive_replay"
    expert_diagnostics = _episode_map_size_diagnostics(expert_replay)
    assert expert_diagnostics["8"]["replay_episodes"] == 1
    assert expert_diagnostics["8"]["sources"] == {"expert_positive_replay": 1}
    assert expert_diagnostics["8"]["replay_trigger_events"] == {"joint_target_scan": 1}
    assert event_names[0] == {"decoy_scan"}
    assert event_names[1] == {"clue_found"}


def test_recurrent_signal_target_interact_label_weighting():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_CLUE, TILE_TARGET
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _SIGNAL_TARGET_SCAN_KIND_FIRST,
        _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION,
        _SIGNAL_TARGET_SCAN_KIND_REDUNDANT_ACTIVE,
        _SIGNAL_TARGET_SCAN_KIND_REFRESH,
        _apply_deferred_solo_target_team_weights,
        _apply_redundant_target_scan_penalty,
        _apply_signal_constraint_message_copy_assist,
        _apply_signal_exact_target_message_guard,
        _apply_signal_scan_broadcast_assist,
        _apply_signal_scan_gate_decoding,
        _apply_signal_scan_refresh_decoding,
        _apply_signal_scan_sync_decoding,
        _apply_signal_negative_memory_scan_guard,
        _apply_signal_target_probe_assist,
        _apply_signal_target_decision_decoding,
        _apply_signal_target_validity_decoding,
        _apply_signal_target_scan_decoding,
        _apply_signal_redundant_target_wait_overrides,
        _apply_wrong_target_scan_penalty,
        _append_labeled_step,
        _feedback_dim,
        _new_episode_sequence,
        _redundant_target_scan_agents,
        _apply_signal_compatible_target_scan_assist,
        _signal_ambiguous_target_search_candidate,
        _signal_center_target_scan_decoding_candidate,
        _signal_center_compatible_target_scan_decoding_candidate,
        _signal_center_ambiguous_target_scan_candidate,
        _signal_center_target_observation_bucket,
        _signal_frontier_exploration_action_label_mask,
        _signal_bad_redundant_target_interact_agents,
        _signal_bad_redundant_target_interact_loss,
        _signal_bad_redundant_target_mask,
        _signal_target_interact_agents,
        _scale_solo_target_team_weights,
        _split_solo_target_scan_agents,
        _episode_signal_target_hypothesis_stats,
        _signal_scan_decision_loss,
        _signal_scan_gate_loss,
        _signal_redundant_target_wait_action_label_mask,
        _signal_target_decision_label_mask,
        _signal_target_decision_loss,
        _signal_target_hypothesis_label,
        _signal_target_hypothesis_loss,
        _signal_target_interact_miss_agents,
        _signal_target_scan_action_loss,
        _signal_target_scan_kind,
        _signal_target_scan_opportunity_label_mask,
        _signal_target_validity_label,
        _signal_target_validity_loss,
        _signal_visible_clue_action_label_mask,
        _wrong_target_scan_agents,
    )

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
    ))
    obs, _ = env.reset(seed=0)
    env.scenario_state.data["target"] = tuple(env.agent_positions[0])
    env.grid[env.agent_positions[0][1], env.agent_positions[0][0]] = TILE_TARGET
    obs[0]["local_grid"][obs[0]["local_grid"].shape[0] // 2, obs[0]["local_grid"].shape[1] // 2] = TILE_TARGET
    obs[0]["action_mask"] = np.ones((8,), dtype=np.float32)
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        bc_signal_target_interact_weight=4.0,
        bc_signal_redundant_target_interact_weight=1.5,
    )
    actions = {
        0: {"action": env.ACTION_INTERACT, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    }
    ep_data = _new_episode_sequence()

    assert _signal_target_interact_agents(env, actions) == [0]
    assert _signal_target_interact_miss_agents(
        env,
        actions,
        {
            0: {"action": env.ACTION_STAY, "message_tokens": []},
            1: {"action": env.ACTION_STAY, "message_tokens": []},
        },
    ) == [0]
    assert _signal_target_interact_miss_agents(env, actions, actions) == []
    _append_labeled_step(
        ep_data,
        obs,
        actions,
        env,
        cfg,
        step_weight=np.array([1.0, 2.0], dtype=np.float32),
    )

    assert ep_data["step_weights"] == [4.0, 2.0]
    assert ep_data["signal_target_scan_action_mask"] == [1.0, 0.0]
    assert ep_data["signal_target_scan_kind_id"] == [_SIGNAL_TARGET_SCAN_KIND_FIRST, -1]
    assert ep_data["signal_target_decision_mask"] == [1.0, 0.0]
    assert ep_data["signal_target_decision_label"] == [1.0, 0.0]

    valid_hold, bad_loop = _split_solo_target_scan_agents(env, obs, actions)
    assert valid_hold == [0]
    assert bad_loop == []
    team_ep_data = _new_episode_sequence()
    team_ep_data["step_weights"] = [1.0, 1.0]
    team_updates, teammate_agents = _scale_solo_target_team_weights(
        team_ep_data,
        num_agents=2,
        solo_target_agents=valid_hold,
        weight=2.5,
    )
    assert teammate_agents == [1]
    assert team_updates == 1
    assert team_ep_data["step_weights"] == [1.0, 2.5]
    deferred_ep_data = _new_episode_sequence()
    deferred_ep_data["step_weights"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    deferred_updates = _apply_deferred_solo_target_team_weights(
        deferred_ep_data,
        [{"step": 1, "agents": [1], "weight": 3.0}],
        num_agents=2,
        focus_window=1,
    )
    assert deferred_updates == 2
    assert deferred_ep_data["step_weights"] == [1.0, 1.0, 1.0, 3.0, 1.0, 3.0]

    env.steps = 2
    env.scenario_state.data["scan_log"] = {0: 2}
    env.scenario_state.data["scan_window"] = 3
    target = tuple(env.scenario_state.data["target"])
    env.agent_positions[1] = (
        min(env.map_size - 1, target[0] + 1),
        target[1],
    )
    assert _redundant_target_scan_agents(env, actions) == [0]
    valid_hold, bad_loop = _split_solo_target_scan_agents(env, obs, actions)
    assert valid_hold == [0]
    assert bad_loop == []
    np.testing.assert_allclose(_signal_bad_redundant_target_mask(env, obs), np.array([0.0, 0.0]))
    _append_labeled_step(
        ep_data,
        obs,
        actions,
        env,
        cfg,
        step_weight=np.array([1.0, 2.0], dtype=np.float32),
    )
    assert ep_data["step_weights"] == [4.0, 2.0, 1.5, 2.0]
    assert ep_data["signal_target_scan_kind_id"][-2:] == [
        _SIGNAL_TARGET_SCAN_KIND_REDUNDANT_ACTIVE,
        -1,
    ]
    assert ep_data["signal_target_decision_mask"][-2:] == [1.0, 0.0]
    assert ep_data["signal_target_decision_label"][-2:] == [0.0, 0.0]

    env.scenario_state.data["scan_window"] = 1
    env.agent_positions[1] = (
        env.map_size - 1 if target[0] < env.map_size - 1 else 0,
        env.map_size - 1 if target[1] < env.map_size - 1 else 0,
    )
    assert _redundant_target_scan_agents(env, actions) == [0]
    rewards = {0: 1.0, 1: 2.0}
    count, penalty_sum = _apply_redundant_target_scan_penalty(rewards, [0], 0.25)
    assert count == 1
    assert penalty_sum == pytest.approx(0.25)
    assert rewards == {0: 0.75, 1: 2.0}
    count, penalty_sum = _apply_redundant_target_scan_penalty(rewards, [0], 0.0)
    assert count == 0
    assert penalty_sum == 0.0
    assert rewards == {0: 0.75, 1: 2.0}
    assert _wrong_target_scan_agents(
        {"events": {0: [{"event": "decoy_scan"}], "1": [{"event": "target_scan"}]}},
        num_agents=2,
    ) == [0]
    count, penalty_sum = _apply_wrong_target_scan_penalty(rewards, [0], 0.5)
    assert count == 1
    assert penalty_sum == pytest.approx(0.5)
    assert rewards == {0: 0.25, 1: 2.0}
    count, penalty_sum = _apply_wrong_target_scan_penalty(rewards, [1], 0.0)
    assert count == 0
    assert penalty_sum == 0.0
    assert rewards == {0: 0.25, 1: 2.0}
    valid_hold, bad_loop = _split_solo_target_scan_agents(env, obs, actions)
    assert valid_hold == []
    assert bad_loop == [0]
    assert _signal_bad_redundant_target_interact_agents(env, obs, actions) == [0]
    np.testing.assert_allclose(_signal_bad_redundant_target_mask(env, obs), np.array([1.0, 0.0]))
    env.scenario_state.data["scan_window"] = 3
    corrected, corrected_agents = _apply_signal_redundant_target_wait_overrides(
        env,
        {
            0: {"action": env.ACTION_INTERACT, "message_tokens": [3]},
            1: {"action": env.ACTION_STAY, "message_tokens": [4]},
        },
    )
    assert corrected_agents == [0]
    assert corrected[0]["action"] == env.ACTION_STAY
    assert corrected[0]["message_tokens"] == [3]
    wait_mask, wait_action_id = _signal_redundant_target_wait_action_label_mask(env, obs, corrected)
    np.testing.assert_allclose(wait_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(wait_action_id, np.array([env.ACTION_STAY, -1], dtype=np.int64))
    wait_ep_data = _new_episode_sequence()
    _append_labeled_step(wait_ep_data, obs, corrected, env, cfg)
    assert wait_ep_data["signal_redundant_target_wait_action_mask"] == [1.0, 0.0]
    assert wait_ep_data["signal_redundant_target_wait_action_id"] == [env.ACTION_STAY, -1]
    env.scenario_state.data["scan_window"] = 1
    edge_corrected, edge_agents = _apply_signal_redundant_target_wait_overrides(
        env,
        {
            0: {"action": env.ACTION_INTERACT, "message_tokens": [3]},
            1: {"action": env.ACTION_STAY, "message_tokens": [4]},
        },
    )
    assert edge_agents == []
    assert edge_corrected[0]["action"] == env.ACTION_INTERACT
    _append_labeled_step(
        ep_data,
        obs,
        actions,
        env,
        cfg,
        step_weight=np.array([1.0, 2.0], dtype=np.float32),
    )
    assert ep_data["signal_bad_redundant_target_mask"] == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert ep_data["signal_target_scan_action_mask"] == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert ep_data["signal_target_scan_kind_id"] == [
        _SIGNAL_TARGET_SCAN_KIND_FIRST,
        -1,
        _SIGNAL_TARGET_SCAN_KIND_REDUNDANT_ACTIVE,
        -1,
        _SIGNAL_TARGET_SCAN_KIND_REFRESH,
        -1,
    ]
    env.scenario_state.data["scan_window"] = 3
    env.scenario_state.data["scan_log"] = {1: 2}
    assert _signal_target_scan_kind(env, 0) == _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION

    good_scan_logits = torch.zeros((4, 8), dtype=torch.float32)
    bad_scan_logits = torch.zeros((4, 8), dtype=torch.float32)
    good_scan_logits[0, env.ACTION_INTERACT] = 4.0
    good_scan_logits[1, env.ACTION_INTERACT] = 4.0
    bad_scan_logits[0, env.ACTION_INTERACT] = -4.0
    bad_scan_logits[1, env.ACTION_INTERACT] = -4.0
    kind_ids = torch.tensor([
        _SIGNAL_TARGET_SCAN_KIND_FIRST,
        _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION,
        _SIGNAL_TARGET_SCAN_KIND_REDUNDANT_ACTIVE,
        _SIGNAL_TARGET_SCAN_KIND_REFRESH,
    ], dtype=torch.long)
    scan_mask = torch.ones((4,), dtype=torch.float32)
    assert _signal_target_scan_action_loss(
        good_scan_logits,
        scan_mask,
        kind_ids,
        first_weight=1.0,
        joint_weight=2.0,
    ).item() < _signal_target_scan_action_loss(
        bad_scan_logits,
        scan_mask,
        kind_ids,
        first_weight=1.0,
        joint_weight=2.0,
    ).item()
    redundant_changed = good_scan_logits.clone()
    redundant_changed[2, env.ACTION_INTERACT] = -9.0
    redundant_changed[3, env.ACTION_INTERACT] = 9.0
    assert _signal_target_scan_action_loss(
        good_scan_logits,
        scan_mask,
        kind_ids,
        first_weight=1.0,
        joint_weight=2.0,
    ).item() == pytest.approx(_signal_target_scan_action_loss(
        redundant_changed,
        scan_mask,
        kind_ids,
        first_weight=1.0,
        joint_weight=2.0,
    ).item())
    bad_refresh_changed = good_scan_logits.clone()
    bad_refresh_changed[3, env.ACTION_INTERACT] = -9.0
    assert _signal_target_scan_action_loss(
        good_scan_logits,
        scan_mask,
        kind_ids,
        first_weight=1.0,
        refresh_weight=3.0,
        joint_weight=2.0,
    ).item() < _signal_target_scan_action_loss(
        bad_refresh_changed,
        scan_mask,
        kind_ids,
        first_weight=1.0,
        refresh_weight=3.0,
        joint_weight=2.0,
    ).item()
    scan_decision_good = torch.zeros((4, 8), dtype=torch.float32)
    scan_decision_bad = torch.zeros((4, 8), dtype=torch.float32)
    scan_decision_good[:2, env.ACTION_INTERACT] = 4.0
    scan_decision_good[2:, env.ACTION_INTERACT] = -4.0
    scan_decision_bad[:2, env.ACTION_INTERACT] = -4.0
    scan_decision_bad[2:, env.ACTION_INTERACT] = 4.0
    assert _signal_scan_decision_loss(
        scan_decision_good,
        torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
        positive_weight=2.0,
        negative_weight=1.0,
    ).item() < _signal_scan_decision_loss(
        scan_decision_bad,
        torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
        positive_weight=2.0,
        negative_weight=1.0,
    ).item()
    assert _signal_scan_gate_loss(
        torch.tensor([4.0, 4.0, -4.0, -4.0], dtype=torch.float32),
        torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
        positive_weight=2.0,
        negative_weight=1.0,
    ).item() < _signal_scan_gate_loss(
        torch.tensor([-4.0, -4.0, 4.0, 4.0], dtype=torch.float32),
        torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
        positive_weight=2.0,
        negative_weight=1.0,
    ).item()
    assert _signal_target_validity_loss(
        torch.tensor([4.0, -4.0], dtype=torch.float32),
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        positive_weight=2.0,
        negative_weight=1.0,
    ).item() < _signal_target_validity_loss(
        torch.tensor([-4.0, 4.0], dtype=torch.float32),
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        positive_weight=2.0,
        negative_weight=1.0,
    ).item()
    assert _signal_target_decision_loss(
        torch.tensor([4.0, -4.0], dtype=torch.float32),
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        positive_weight=2.0,
        negative_weight=1.0,
    ).item() < _signal_target_decision_loss(
        torch.tensor([-4.0, 4.0], dtype=torch.float32),
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        positive_weight=2.0,
        negative_weight=1.0,
    ).item()
    hypothesis_xy = torch.tensor([[0.25, 0.75], [0.50, 0.25]], dtype=torch.float32)
    hypothesis_logits = torch.logit(hypothesis_xy.clamp(0.01, 0.99))
    good_hypothesis_pred = torch.cat(
        [
            torch.tensor([[4.0, -4.0], [-4.0, 2.0]], dtype=torch.float32),
            hypothesis_logits,
        ],
        dim=1,
    )
    bad_hypothesis_pred = torch.cat(
        [
            torch.tensor([[-4.0, 4.0], [4.0, -2.0]], dtype=torch.float32),
            torch.flip(hypothesis_logits, dims=[0]),
        ],
        dim=1,
    )
    assert _signal_target_hypothesis_loss(
        good_hypothesis_pred,
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        hypothesis_xy,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.75], dtype=torch.float32),
    ).item() < _signal_target_hypothesis_loss(
        bad_hypothesis_pred,
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        hypothesis_xy,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.75], dtype=torch.float32),
    ).item()
    assert _signal_target_hypothesis_loss(
        good_hypothesis_pred,
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        hypothesis_xy,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.75], dtype=torch.float32),
        commit_loss_weight=0.0,
        ambiguity_loss_weight=0.0,
        xy_loss_weight=0.0,
    ).item() == pytest.approx(0.0)
    assert _signal_target_hypothesis_loss(
        good_hypothesis_pred,
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        hypothesis_xy,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.75], dtype=torch.float32),
        commit_loss_weight=0.0,
        ambiguity_loss_weight=0.0,
        xy_loss_weight=1.0,
    ).item() < _signal_target_hypothesis_loss(
        bad_hypothesis_pred,
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        hypothesis_xy,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.75], dtype=torch.float32),
        commit_loss_weight=0.0,
        ambiguity_loss_weight=0.0,
        xy_loss_weight=1.0,
    ).item()

    high_bad_logits = torch.zeros((2, 8), dtype=torch.float32)
    low_bad_logits = torch.zeros((2, 8), dtype=torch.float32)
    high_bad_logits[0, env.ACTION_INTERACT] = 4.0
    high_bad_logits[1, env.ACTION_INTERACT] = 8.0
    low_bad_logits[0, env.ACTION_INTERACT] = -4.0
    low_bad_logits[1, env.ACTION_INTERACT] = 8.0
    high_bad_loss = _signal_bad_redundant_target_interact_loss(
        high_bad_logits,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    low_bad_loss = _signal_bad_redundant_target_interact_loss(
        low_bad_logits,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    assert high_bad_loss.item() > low_bad_loss.item()
    assert high_bad_loss.item() == pytest.approx(
        torch.nn.functional.softplus(torch.tensor(4.0)).item()
    )

    env.steps = 5
    assert _redundant_target_scan_agents(env, actions) == []

    scan_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        eval_signal_target_scan_threshold=0.25,
    )
    decode_obs, _ = env.reset(seed=1)
    target_pos = tuple(int(v) for v in decode_obs[0]["self_pos"])
    rejected_pos = tuple(int(v) for v in decode_obs[1]["self_pos"])
    allowed_pos = ((rejected_pos[0] + 1) % env.map_size, rejected_pos[1])
    if allowed_pos == rejected_pos:
        allowed_pos = ((rejected_pos[0] - 1) % env.map_size, rejected_pos[1])
    for aid, center_pos in ((0, target_pos), (1, rejected_pos)):
        local_grid = np.zeros_like(decode_obs[aid]["local_grid"])
        local_grid[local_grid.shape[0] // 2, local_grid.shape[1] // 2] = TILE_TARGET
        decode_obs[aid]["local_grid"] = local_grid
        decode_obs[aid]["self_pos"] = np.array(center_pos, dtype=np.int16)
        decode_obs[aid]["action_mask"] = np.ones((8,), dtype=np.float32)
    decode_obs[0]["goal_hint"] = np.array(
        [26, target_pos[0], target_pos[1], -1, -1, -1, -1, -1],
        dtype=np.int16,
    )
    decode_obs[1]["goal_hint"] = np.array(
        [26, allowed_pos[0], allowed_pos[1], -1, -1, -1, -1, -1],
        dtype=np.int16,
    )
    env.scenario_state.data["target"] = target_pos
    env.agent_positions[0] = target_pos
    env.agent_positions[1] = rejected_pos
    env.steps = 0
    env.scenario_state.data["scan_log"] = {}
    env.scenario_state.data["scan_window"] = 3
    validity_mask, validity_label = _signal_target_validity_label(env, decode_obs)
    np.testing.assert_allclose(validity_mask, np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(validity_label, np.array([1.0, 0.0], dtype=np.float32))
    opportunity_mask, opportunity_kind = _signal_target_scan_opportunity_label_mask(
        env,
        decode_obs,
        scan_cfg,
    )
    np.testing.assert_allclose(opportunity_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        opportunity_kind,
        np.array([_SIGNAL_TARGET_SCAN_KIND_FIRST, -1], dtype=np.int64),
    )
    decision_mask, decision_label = _signal_target_decision_label_mask(
        env,
        decode_obs,
        scan_cfg,
        actions,
        target_opportunity_mask=opportunity_mask,
        target_opportunity_kind_id=opportunity_kind,
    )
    np.testing.assert_allclose(decision_mask, np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(decision_label, np.array([1.0, 0.0], dtype=np.float32))
    hypothesis_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        bc_signal_target_hypothesis_loss_weight=1.0,
        bc_signal_target_hypothesis_min_map_size=8,
    )
    hypothesis_mask, hypothesis_xy, hypothesis_commit, hypothesis_ambiguity = (
        _signal_target_hypothesis_label(env, decode_obs, hypothesis_cfg)
    )
    expected_hypothesis_xy = np.asarray(target_pos, dtype=np.float32) / float(env.map_size - 1)
    np.testing.assert_allclose(hypothesis_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(hypothesis_xy[0], expected_hypothesis_xy)
    np.testing.assert_allclose(hypothesis_commit, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(hypothesis_ambiguity, np.array([0.0, 0.0], dtype=np.float32))
    hypothesis_ep_data = _new_episode_sequence()
    _append_labeled_step(hypothesis_ep_data, decode_obs, actions, env, hypothesis_cfg)
    assert hypothesis_ep_data["signal_target_hypothesis_mask"] == [1.0, 0.0]
    np.testing.assert_allclose(
        hypothesis_ep_data["signal_target_hypothesis_xy"][0],
        expected_hypothesis_xy,
    )
    assert hypothesis_ep_data["signal_target_hypothesis_commit_label"] == [1.0, 0.0]
    assert hypothesis_ep_data["signal_target_hypothesis_ambiguity_label"] == [0.0, 0.0]
    exact_hypothesis_stats = _episode_signal_target_hypothesis_stats([hypothesis_ep_data])
    assert exact_hypothesis_stats["labels"] == 1
    assert exact_hypothesis_stats["commit_labels"] == 1
    assert exact_hypothesis_stats["ambiguous_labels"] == 0
    assert exact_hypothesis_stats["commit_rate"] == pytest.approx(1.0)
    assert exact_hypothesis_stats["ambiguity_mean"] == pytest.approx(0.0)
    env.steps = 2
    env.scenario_state.data["scan_log"] = {1: 2}
    opportunity_mask, opportunity_kind = _signal_target_scan_opportunity_label_mask(
        env,
        decode_obs,
        scan_cfg,
    )
    np.testing.assert_allclose(opportunity_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        opportunity_kind,
        np.array([_SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION, -1], dtype=np.int64),
    )
    uncertain_joint_obs = {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in decode_obs[0].items()
    }
    uncertain_joint_obs["goal_hint"] = np.full((8,), -1, dtype=np.int16)
    uncertain_joint_obs["messages_tokens"] = np.full_like(
        uncertain_joint_obs.get("messages_tokens", np.full((2, 8), -1, dtype=np.int16)),
        -1,
    )
    assert not _signal_center_target_scan_decoding_candidate(uncertain_joint_obs, scan_cfg)
    uncertain_obs = {0: uncertain_joint_obs, 1: decode_obs[1]}
    opportunity_mask, opportunity_kind = _signal_target_scan_opportunity_label_mask(
        env,
        uncertain_obs,
        scan_cfg,
    )
    np.testing.assert_allclose(opportunity_mask, np.array([0.0, 0.0], dtype=np.float32))
    no_info_hypothesis_mask, _, _, _ = _signal_target_hypothesis_label(
        env,
        uncertain_obs,
        hypothesis_cfg,
    )
    np.testing.assert_allclose(no_info_hypothesis_mask, np.array([0.0, 0.0], dtype=np.float32))
    assert _signal_center_target_observation_bucket(uncertain_joint_obs, scan_cfg) == "no_info"
    feedback_label_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_feedback=True,
        obs_signal_scan_state=True,
    )
    scan_state_feedback = np.zeros((2, _feedback_dim(feedback_label_cfg)), dtype=np.float32)
    scan_state_feedback[0, 12 + 1] = 1.0
    opportunity_mask, opportunity_kind = _signal_target_scan_opportunity_label_mask(
        env,
        uncertain_obs,
        feedback_label_cfg,
        feedback=scan_state_feedback,
    )
    np.testing.assert_allclose(opportunity_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        opportunity_kind,
        np.array([_SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION, -1], dtype=np.int64),
    )
    env.scenario_state.data["scan_log"] = {0: 2}
    opportunity_mask, opportunity_kind = _signal_target_scan_opportunity_label_mask(
        env,
        decode_obs,
        scan_cfg,
    )
    np.testing.assert_allclose(opportunity_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(opportunity_kind, np.array([-1, -1], dtype=np.int64))
    env.steps = 0
    env.scenario_state.data["scan_log"] = {}
    logits = torch.full((2, 8), -3.0, dtype=torch.float32)
    logits[:, env.ACTION_STAY] = 1.0
    logits[:, env.ACTION_INTERACT] = 0.2
    stay_actions = torch.full((2,), env.ACTION_STAY, dtype=torch.long)

    assert _signal_center_target_scan_decoding_candidate(decode_obs[0], scan_cfg)
    assert _signal_center_target_observation_bucket(decode_obs[0], scan_cfg) == "candidate"
    assert not _signal_center_target_scan_decoding_candidate(decode_obs[1], scan_cfg)
    decoded_actions = _apply_signal_target_scan_decoding(scan_cfg, decode_obs, logits, stay_actions)
    assert decoded_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    scan_cfg.eval_signal_target_scan_threshold = -1.0
    disabled_actions = _apply_signal_target_scan_decoding(scan_cfg, decode_obs, logits, stay_actions)
    assert disabled_actions.tolist() == [env.ACTION_STAY, env.ACTION_STAY]
    scan_cfg.eval_signal_scan_gate_threshold = 0.5
    gated_actions = _apply_signal_scan_gate_decoding(
        scan_cfg,
        decode_obs,
        stay_actions,
        torch.tensor([2.0, 2.0], dtype=torch.float32),
    )
    assert gated_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    interact_actions = torch.full((2,), env.ACTION_INTERACT, dtype=torch.long)
    scan_cfg.eval_signal_scan_gate_suppress = True
    rejected_actions = _apply_signal_scan_gate_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([2.0, 2.0], dtype=torch.float32),
    )
    assert rejected_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    suppressed_actions = _apply_signal_scan_gate_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([-2.0, -2.0], dtype=torch.float32),
    )
    assert suppressed_actions.tolist() == [env.ACTION_STAY, env.ACTION_STAY]
    ambiguous_target_obs = {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in decode_obs[0].items()
    }
    tx, ty = target_pos
    quadrant = (
        0 if tx < env.map_size / 2 and ty < env.map_size / 2 else
        1 if tx >= env.map_size / 2 and ty < env.map_size / 2 else
        2 if tx < env.map_size / 2 and ty >= env.map_size / 2 else 3
    )
    ambiguous_target_obs["goal_hint"] = np.array(
        [
            23,
            (tx + ty) % 2,
            quadrant,
            env.map_size,
            24,
            tx % 2,
            25,
            ty % 2,
        ],
        dtype=np.int16,
    )
    ambiguous_target_obs["messages_tokens"] = np.full_like(
        ambiguous_target_obs.get("messages_tokens", np.full((2, 8), -1, dtype=np.int16)),
        -1,
    )
    ambiguous_grid = np.zeros_like(ambiguous_target_obs["local_grid"])
    cy, cx = ambiguous_grid.shape[0] // 2, ambiguous_grid.shape[1] // 2
    ambiguous_grid[cy, cx] = TILE_TARGET
    half = env.map_size // 2
    alt_pos = None
    for candidate in ((tx + 2, ty), (tx - 2, ty), (tx, ty + 2), (tx, ty - 2)):
        ax, ay = candidate
        if not (0 <= ax < env.map_size and 0 <= ay < env.map_size):
            continue
        if (ax + ay) % 2 != (tx + ty) % 2 or ax % 2 != tx % 2 or ay % 2 != ty % 2:
            continue
        same_quadrant = (
            (quadrant == 0 and ax < half and ay < half)
            or (quadrant == 1 and ax >= half and ay < half)
            or (quadrant == 2 and ax < half and ay >= half)
            or (quadrant == 3 and ax >= half and ay >= half)
        )
        lx, ly = cx + (ax - tx), cy + (ay - ty)
        if same_quadrant and 0 <= lx < ambiguous_grid.shape[1] and 0 <= ly < ambiguous_grid.shape[0]:
            alt_pos = (ax, ay)
            ambiguous_grid[ly, lx] = TILE_TARGET
            break
    assert alt_pos is not None
    ambiguous_target_obs["local_grid"] = ambiguous_grid
    assert not _signal_center_target_scan_decoding_candidate(ambiguous_target_obs, scan_cfg)
    assert _signal_center_compatible_target_scan_decoding_candidate(ambiguous_target_obs, scan_cfg)
    assert _signal_center_ambiguous_target_scan_candidate(ambiguous_target_obs, scan_cfg)
    ambiguous_obs = {0: ambiguous_target_obs, 1: decode_obs[1]}
    ambiguous_opportunity_mask, ambiguous_opportunity_kind = (
        _signal_target_scan_opportunity_label_mask(
            env,
            ambiguous_obs,
            scan_cfg,
        )
    )
    default_decision_mask, default_decision_label = _signal_target_decision_label_mask(
        env,
        ambiguous_obs,
        scan_cfg,
        actions,
        target_opportunity_mask=ambiguous_opportunity_mask,
        target_opportunity_kind_id=ambiguous_opportunity_kind,
    )
    assert default_decision_mask[0] == pytest.approx(1.0)
    assert default_decision_label[0] == pytest.approx(1.0)
    strict_decision_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        bc_signal_ambiguous_target_decision_negatives=True,
        bc_signal_ambiguous_target_decision_min_map_size=8,
    )
    strict_decision_mask, strict_decision_label = _signal_target_decision_label_mask(
        env,
        ambiguous_obs,
        strict_decision_cfg,
        actions,
        target_opportunity_mask=ambiguous_opportunity_mask,
        target_opportunity_kind_id=ambiguous_opportunity_kind,
    )
    assert strict_decision_mask[0] == pytest.approx(1.0)
    assert strict_decision_label[0] == pytest.approx(0.0)
    ambiguous_hypothesis_mask, ambiguous_hypothesis_xy, ambiguous_hypothesis_commit, ambiguous_hypothesis_ambiguity = (
        _signal_target_hypothesis_label(env, ambiguous_obs, hypothesis_cfg)
    )
    np.testing.assert_allclose(ambiguous_hypothesis_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(ambiguous_hypothesis_xy[0], expected_hypothesis_xy)
    assert ambiguous_hypothesis_commit[0] == pytest.approx(0.0)
    assert ambiguous_hypothesis_ambiguity[0] > 0.0
    ambiguous_hypothesis_ep_data = _new_episode_sequence()
    _append_labeled_step(ambiguous_hypothesis_ep_data, ambiguous_obs, actions, env, hypothesis_cfg)
    ambiguous_hypothesis_stats = _episode_signal_target_hypothesis_stats(
        [ambiguous_hypothesis_ep_data]
    )
    assert ambiguous_hypothesis_stats["labels"] == 1
    assert ambiguous_hypothesis_stats["commit_labels"] == 0
    assert ambiguous_hypothesis_stats["ambiguous_labels"] == 1
    assert ambiguous_hypothesis_stats["commit_rate"] == pytest.approx(0.0)
    assert ambiguous_hypothesis_stats["ambiguity_mean"] > 0.0
    default_visible_clue_mask, default_visible_clue_action_id = _signal_visible_clue_action_label_mask(
        env,
        ambiguous_obs,
        scan_cfg,
    )
    np.testing.assert_allclose(default_visible_clue_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(default_visible_clue_action_id, np.array([-1, -1], dtype=np.int64))
    search_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_exploration_memory=True,
        bc_signal_ambiguous_target_search_labels=True,
        bc_signal_ambiguous_target_search_min_map_size=8,
        bc_signal_visible_clue_min_map_size=8,
        bc_signal_frontier_exploration_min_map_size=8,
    )
    assert _signal_ambiguous_target_search_candidate(ambiguous_target_obs, search_cfg)
    ambiguous_clue_obs = {
        0: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in ambiguous_target_obs.items()
        },
        1: decode_obs[1],
    }
    ambiguous_clue_grid = ambiguous_clue_obs[0]["local_grid"].copy()
    clue_lx = min(ambiguous_clue_grid.shape[1] - 1, cx + 1)
    ambiguous_clue_grid[cy, clue_lx] = TILE_CLUE
    ambiguous_clue_obs[0]["local_grid"] = ambiguous_clue_grid
    search_visible_clue_mask, search_visible_clue_action_id = _signal_visible_clue_action_label_mask(
        env,
        ambiguous_clue_obs,
        search_cfg,
    )
    np.testing.assert_allclose(search_visible_clue_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        search_visible_clue_action_id,
        np.array([env.ACTION_RIGHT, -1], dtype=np.int64),
    )
    search_clue_frontier_mask, search_clue_frontier_action_id = (
        _signal_frontier_exploration_action_label_mask(
            env,
            ambiguous_clue_obs,
            search_cfg,
        )
    )
    np.testing.assert_allclose(search_clue_frontier_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(search_clue_frontier_action_id, np.array([-1, -1], dtype=np.int64))
    ambiguous_frontier_obs = {
        0: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in ambiguous_target_obs.items()
        },
        1: decode_obs[1],
    }
    explored = np.ones((env.map_size, env.map_size), dtype=np.int8)
    fx = min(env.map_size - 1, tx + 1)
    if fx == tx:
        fx = max(0, tx - 1)
    explored[ty, fx] = 0
    ambiguous_frontier_obs[0]["explored_mask"] = explored
    ambiguous_frontier_obs[1]["explored_mask"] = np.ones_like(explored)
    default_frontier_mask, default_frontier_action_id = _signal_frontier_exploration_action_label_mask(
        env,
        ambiguous_frontier_obs,
        RecurrentConfig(
            scenario="signal_hunt",
            map_size=8,
            agents=2,
            obs_exploration_memory=True,
            bc_signal_frontier_exploration_min_map_size=8,
        ),
    )
    np.testing.assert_allclose(default_frontier_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(default_frontier_action_id, np.array([-1, -1], dtype=np.int64))
    search_frontier_mask, search_frontier_action_id = _signal_frontier_exploration_action_label_mask(
        env,
        ambiguous_frontier_obs,
        search_cfg,
    )
    np.testing.assert_allclose(search_frontier_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        search_frontier_action_id,
        np.array([
            env.ACTION_RIGHT if fx > tx else env.ACTION_LEFT,
            -1,
        ], dtype=np.int64),
    )
    scan_cfg.eval_signal_compatible_target_scan_assist = True
    compatible_actions = _apply_signal_compatible_target_scan_assist(
        scan_cfg,
        {0: ambiguous_target_obs, 1: decode_obs[1]},
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        scan_state={"step": 0, "scan_log": {}, "scan_pos": {}, "scan_window": 3},
    )
    assert compatible_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    compatible_wait_actions = _apply_signal_compatible_target_scan_assist(
        scan_cfg,
        {0: ambiguous_target_obs, 1: decode_obs[1]},
        torch.full((2,), env.ACTION_INTERACT, dtype=torch.long),
        scan_state={
            "step": 2,
            "scan_log": {0: 2},
            "scan_pos": {0: target_pos},
            "scan_window": 3,
        },
    )
    assert compatible_wait_actions.tolist() == [env.ACTION_STAY, env.ACTION_INTERACT]
    scan_cfg.eval_signal_compatible_target_scan_assist = False
    negative_guard_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_signal_negative_memory=True,
        obs_signal_negative_memory_window=4,
        eval_signal_negative_memory_scan_guard=True,
    )
    negative_guard_actions = _apply_signal_negative_memory_scan_guard(
        negative_guard_cfg,
        decode_obs,
        torch.full((2,), env.ACTION_INTERACT, dtype=torch.long),
        scan_state={
            "step": 5,
            "negative_target_log": [{"agent_id": 0, "pos": list(target_pos), "step": 4}],
        },
    )
    assert negative_guard_actions.tolist() == [env.ACTION_STAY, env.ACTION_INTERACT]
    stale_negative_guard_actions = _apply_signal_negative_memory_scan_guard(
        negative_guard_cfg,
        decode_obs,
        torch.full((2,), env.ACTION_INTERACT, dtype=torch.long),
        scan_state={
            "step": 10,
            "negative_target_log": [{"agent_id": 0, "pos": list(target_pos), "step": 4}],
        },
    )
    assert stale_negative_guard_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_INTERACT]
    probe_obs = {
        0: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in decode_obs[0].items()
        },
        1: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in decode_obs[1].items()
        },
    }
    for aid in probe_obs:
        probe_obs[aid]["goal_hint"] = np.full((8,), -1, dtype=np.int16)
        probe_obs[aid]["messages_tokens"] = np.full((2, 8), -1, dtype=np.int16)
    probe_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_signal_negative_memory=True,
        obs_signal_negative_memory_window=4,
        eval_signal_target_probe_assist=True,
    )
    probe_actions = _apply_signal_target_probe_assist(
        probe_cfg,
        probe_obs,
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        scan_state={"step": 5, "negative_target_log": []},
    )
    assert probe_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_INTERACT]
    negative_probe_actions = _apply_signal_target_probe_assist(
        probe_cfg,
        probe_obs,
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        scan_state={
            "step": 5,
            "negative_target_log": [{"agent_id": 0, "pos": list(target_pos), "step": 4}],
        },
    )
    assert negative_probe_actions.tolist() == [env.ACTION_STAY, env.ACTION_INTERACT]
    active_probe_actions = _apply_signal_target_probe_assist(
        probe_cfg,
        probe_obs,
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        scan_state={
            "step": 5,
            "scan_log": {0: 5},
            "scan_pos": {0: target_pos},
            "scan_window": 3,
            "negative_target_log": [],
        },
    )
    assert active_probe_actions.tolist() == [env.ACTION_STAY, env.ACTION_INTERACT]
    constraint_copy_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        eval_signal_constraint_message_copy_assist=True,
    )
    constraint_copy_obs = {
        0: {
            "goal_hint": np.array([24, 1, -1, -1, -1, -1, -1, -1], dtype=np.int16),
            "messages_tokens": np.full((2, 8), -1, dtype=np.int16),
        },
        1: {
            "goal_hint": np.full((8,), -1, dtype=np.int16),
            "messages_tokens": np.full((2, 8), -1, dtype=np.int16),
        },
    }
    copied_messages = _apply_signal_constraint_message_copy_assist(
        constraint_copy_cfg,
        constraint_copy_obs,
        {
            0: {"action": env.ACTION_STAY, "message_tokens": [24, 0]},
            1: {"action": env.ACTION_STAY, "message_tokens": [21, 1, 2, 3, 4]},
        },
    )
    assert copied_messages[0]["message_tokens"] == [24, 1]
    assert copied_messages[1]["message_tokens"] == [21, 1, 2, 3, 4]
    scan_cfg.eval_signal_exact_target_scan_lock = True
    exact_locked_gate_actions = _apply_signal_scan_gate_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([-2.0, -2.0], dtype=torch.float32),
        scan_state={"step": 0},
    )
    assert exact_locked_gate_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    scan_cfg.eval_signal_target_validity_threshold = 0.5
    exact_locked_validity_actions = _apply_signal_target_validity_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([-2.0, -2.0], dtype=torch.float32),
        scan_state={"step": 0},
    )
    assert exact_locked_validity_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    scan_cfg.eval_signal_target_decision_threshold = 0.5
    exact_locked_decision_actions = _apply_signal_target_decision_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([-2.0, -2.0], dtype=torch.float32),
        scan_state={"step": 0},
    )
    assert exact_locked_decision_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    scan_cfg.eval_signal_target_validity_threshold = -1.0
    scan_cfg.eval_signal_target_decision_threshold = -1.0
    scan_cfg.eval_signal_exact_target_scan_lock = False
    scan_cfg.eval_signal_target_scan_lock = True
    locked_gate_actions = _apply_signal_scan_gate_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([-2.0, -2.0], dtype=torch.float32),
    )
    assert locked_gate_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    scan_cfg.eval_signal_target_validity_threshold = 0.5
    validity_actions = _apply_signal_target_validity_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([2.0, -2.0], dtype=torch.float32),
    )
    assert validity_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    locked_validity_actions = _apply_signal_target_validity_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([-2.0, -2.0], dtype=torch.float32),
    )
    assert locked_validity_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    scan_cfg.eval_signal_target_validity_threshold = -1.0
    validity_disabled_actions = _apply_signal_target_validity_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([-2.0, -2.0], dtype=torch.float32),
    )
    assert validity_disabled_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_INTERACT]
    scan_cfg.eval_signal_target_decision_threshold = 0.5
    decision_actions = _apply_signal_target_decision_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([2.0, -2.0], dtype=torch.float32),
    )
    assert decision_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    locked_decision_actions = _apply_signal_target_decision_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([-2.0, -2.0], dtype=torch.float32),
    )
    assert locked_decision_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    scan_cfg.eval_signal_target_decision_suppress = False
    decision_force_actions = _apply_signal_target_decision_decoding(
        scan_cfg,
        decode_obs,
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        torch.tensor([2.0, -2.0], dtype=torch.float32),
    )
    assert decision_force_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    scan_cfg.eval_signal_target_decision_threshold = -1.0

    sync_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_feedback=True,
        obs_signal_scan_state=True,
        eval_signal_scan_sync_assist=True,
    )
    sync_feedback = np.zeros((2, _feedback_dim(sync_cfg)), dtype=np.float32)
    sync_offset = 12
    sync_feedback[0, sync_offset + 1] = 1.0  # teammate target scan is active: join it
    sync_feedback[1, sync_offset] = 1.0  # own scan is active: wait for teammate
    sync_obs = {}
    for aid, center_pos in ((0, target_pos), (1, rejected_pos)):
        obs_agent = {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in decode_obs[aid].items()
        }
        obs_agent["self_pos"] = np.array(center_pos, dtype=np.int16)
        obs_agent["goal_hint"] = np.array(
            [26, center_pos[0], center_pos[1], -1, -1, -1, -1, -1],
            dtype=np.int16,
        )
        sync_obs[aid] = obs_agent
    sync_actions = _apply_signal_scan_sync_decoding(
        sync_cfg,
        sync_obs,
        torch.tensor([env.ACTION_STAY, env.ACTION_INTERACT], dtype=torch.long),
        sync_feedback,
    )
    assert sync_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    relaxed_sync_obs = {
        0: uncertain_joint_obs,
        1: sync_obs[1],
    }
    relaxed_sync_actions = _apply_signal_scan_sync_decoding(
        sync_cfg,
        relaxed_sync_obs,
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        sync_feedback,
    )
    assert relaxed_sync_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    rejected_inactive_actions = _apply_signal_scan_sync_decoding(
        sync_cfg,
        decode_obs,
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        np.zeros_like(sync_feedback),
    )
    assert rejected_inactive_actions.tolist() == [env.ACTION_STAY, env.ACTION_STAY]
    rejected_sync_feedback = np.zeros_like(sync_feedback)
    rejected_sync_feedback[1, sync_offset + 1] = 1.0
    rejected_sync_actions = _apply_signal_scan_sync_decoding(
        sync_cfg,
        decode_obs,
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        rejected_sync_feedback,
    )
    assert rejected_sync_actions.tolist() == [env.ACTION_STAY, env.ACTION_STAY]
    mismatched_scan_state = {
        "scan_log": {0: 2},
        "scan_pos": {0: target_pos},
        "scan_window": 3,
        "step": 2,
    }
    mismatched_rejected_actions = _apply_signal_scan_sync_decoding(
        sync_cfg,
        decode_obs,
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        rejected_sync_feedback,
        scan_state=mismatched_scan_state,
    )
    assert mismatched_rejected_actions.tolist() == [env.ACTION_STAY, env.ACTION_STAY]
    matched_scan_state = {
        "scan_log": {0: 2},
        "scan_pos": {0: rejected_pos},
        "scan_window": 3,
        "step": 2,
    }
    matched_rejected_actions = _apply_signal_scan_sync_decoding(
        sync_cfg,
        decode_obs,
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        rejected_sync_feedback,
        scan_state=matched_scan_state,
    )
    assert matched_rejected_actions.tolist() == [env.ACTION_STAY, env.ACTION_INTERACT]
    broadcast_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        obs_signal_scan_state=True,
        eval_signal_scan_broadcast_assist=True,
    )
    broadcast_state = {
        "scan_log": {0: 2},
        "scan_pos": {0: (4, 1)},
        "scan_window": 3,
        "step": 2,
    }
    broadcast_actions = _apply_signal_scan_broadcast_assist(
        broadcast_cfg,
        {
            0: {"action": env.ACTION_STAY, "message_tokens": [26, 7, 3]},
            1: {"action": env.ACTION_STAY, "message_tokens": [9]},
        },
        broadcast_state,
    )
    assert broadcast_actions[0]["message_tokens"] == [26, 4, 1]
    assert broadcast_actions[1]["message_tokens"] == [9]
    repeat_broadcast_actions = _apply_signal_scan_broadcast_assist(
        broadcast_cfg,
        {
            0: {"action": env.ACTION_STAY, "message_tokens": [26, 7, 3]},
            1: {"action": env.ACTION_STAY, "message_tokens": [9]},
        },
        broadcast_state,
    )
    assert repeat_broadcast_actions[0]["message_tokens"] == [26, 7, 3]
    broadcast_state["scan_log"][0] = 3
    broadcast_state["step"] = 3
    refreshed_broadcast_actions = _apply_signal_scan_broadcast_assist(
        broadcast_cfg,
        {
            0: {"action": env.ACTION_STAY, "message_tokens": [26, 7, 3]},
            1: {"action": env.ACTION_STAY, "message_tokens": [9]},
        },
        broadcast_state,
    )
    assert refreshed_broadcast_actions[0]["message_tokens"] == [26, 4, 1]
    expired_broadcast_actions = _apply_signal_scan_broadcast_assist(
        broadcast_cfg,
        {
            0: {"action": env.ACTION_STAY, "message_tokens": [26, 7, 3]},
            1: {"action": env.ACTION_STAY, "message_tokens": []},
        },
        {
            "scan_log": {0: 2},
            "scan_pos": {0: (4, 1)},
            "scan_window": 3,
            "step": 9,
        },
    )
    assert expired_broadcast_actions[0]["message_tokens"] == [26, 7, 3]
    guard_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        eval_signal_exact_target_message_guard=True,
    )
    guard_obs = {
        0: {
            "self_pos": np.array([0, 0], dtype=np.int16),
            "goal_hint": np.full((8,), -1, dtype=np.int16),
            "messages_tokens": np.full((2, 8), -1, dtype=np.int16),
        },
        1: {
            "self_pos": np.array([1, 0], dtype=np.int16),
            "goal_hint": np.array([26, 4, 3, -1, -1, -1, -1, -1], dtype=np.int16),
            "messages_tokens": np.full((2, 8), -1, dtype=np.int16),
        },
    }
    guarded_actions = _apply_signal_exact_target_message_guard(
        guard_cfg,
        guard_obs,
        {
            0: {"action": env.ACTION_STAY, "message_tokens": [26, 4, 3]},
            1: {"action": env.ACTION_STAY, "message_tokens": [26, 4, 3]},
        },
        None,
    )
    assert guarded_actions[0]["message_tokens"] == []
    assert guarded_actions[1]["message_tokens"] == [26, 4, 3]
    relative_guard_obs = {
        0: {
            "self_pos": np.array([0, 0], dtype=np.int16),
            "goal_hint": np.array([22, 8, 2, 2, 2, 1, -1, -1], dtype=np.int16),
            "messages_tokens": np.full((2, 8), -1, dtype=np.int16),
        },
    }
    relative_guarded_actions = _apply_signal_exact_target_message_guard(
        guard_cfg,
        relative_guard_obs,
        {0: {"action": env.ACTION_STAY, "message_tokens": [26, 4, 3]}},
        None,
    )
    assert relative_guarded_actions[0]["message_tokens"] == [26, 4, 3]
    scan_trusted_actions = _apply_signal_exact_target_message_guard(
        guard_cfg,
        guard_obs,
        {0: {"action": env.ACTION_STAY, "message_tokens": [26, 4, 3]}},
        {"scan_log": {1: 2}, "scan_pos": {1: (4, 3)}, "scan_window": 3, "step": 2},
    )
    assert scan_trusted_actions[0]["message_tokens"] == [26, 4, 3]
    sync_cfg.eval_signal_scan_sync_force_first = True
    first_actions = _apply_signal_scan_sync_decoding(
        sync_cfg,
        sync_obs,
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        np.zeros_like(sync_feedback),
    )
    assert first_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_INTERACT]
    refresh_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_feedback=True,
        obs_signal_scan_state=True,
        eval_signal_scan_refresh_assist=True,
        eval_signal_scan_refresh_threshold=0.5,
    )
    refresh_feedback = np.zeros((2, _feedback_dim(refresh_cfg)), dtype=np.float32)
    refresh_feedback[0, sync_offset] = 1.0
    refresh_feedback[0, sync_offset + 2] = 0.5
    refresh_feedback[1, sync_offset] = 1.0
    refresh_feedback[1, sync_offset + 1] = 1.0
    refresh_feedback[1, sync_offset + 2] = 0.5
    refresh_actions = _apply_signal_scan_refresh_decoding(
        refresh_cfg,
        sync_obs,
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        refresh_feedback,
    )
    assert refresh_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
    refresh_memory_obs = {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in sync_obs[0].items()
    }
    refresh_memory_obs["goal_hint"] = np.full((8,), -1, dtype=np.int16)
    assert not _signal_center_target_scan_decoding_candidate(refresh_memory_obs, refresh_cfg)
    memory_refresh_actions = _apply_signal_scan_refresh_decoding(
        refresh_cfg,
        {0: refresh_memory_obs, 1: sync_obs[1]},
        torch.full((2,), env.ACTION_STAY, dtype=torch.long),
        refresh_feedback,
    )
    assert memory_refresh_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]


def test_recurrent_signal_target_pursuit_label_weighting():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.envs.maps import (
        TILE_BEACON,
        TILE_CLUE,
        TILE_EMPTY,
        TILE_TARGET,
        TILE_UNKNOWN,
        TILE_WALL,
        TILE_WATER,
    )
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _append_labeled_step,
        _apply_signal_exact_target_navigation_assist,
        _apply_signal_frontier_exploration_assist,
        _apply_signal_initial_exact_message_copy_assist,
        _apply_signal_initial_message_weight,
        _apply_signal_target_rendezvous_overrides,
        _apply_signal_initial_target_broadcast_overrides,
        _apply_signal_target_handoff_overrides,
        _apply_signal_target_scan_broadcast_overrides,
        _feedback_matrix,
        _finalize_episode_sequence,
        _label_latest_signal_decoy_drift_actions,
        _label_latest_signal_decoy_scan_actions,
        _label_latest_signal_rejected_target_drift_actions,
        _new_episode_sequence,
        _signal_clue_interact_action_label_mask,
        _signal_clue_interact_miss_agents,
        _signal_decoy_pursuit_agents,
        _signal_decoy_drift_action_loss,
        _signal_exact_target_handoff_candidate,
        _signal_constraint_frontier_targets,
        _signal_evidence_sweep_action_label_mask,
        _signal_evidence_sweep_miss_agents,
        _signal_frontier_exploration_action_label_mask,
        _signal_frontier_exploration_miss_agents,
        _signal_nearest_frontier_cell,
        _signal_movement_stall_miss_agents,
        _signal_navigation_action_from_obs,
        _signal_observation_allows_target,
        _signal_positive_target_pursuit_agents,
        _signal_rejected_target_drift_agents,
        _signal_assigned_frontier_cell,
        _signal_constraint_message_label,
        _signal_target_decoy_drift_miss_agents,
        _signal_target_discovery_miss_agents,
        _signal_target_handoff_miss_agents,
        _signal_target_match_action_label_mask,
        _signal_target_match_action_loss,
        _signal_target_pursuit_action_label_mask,
        _signal_target_pursuit_miss_agents,
        _signal_target_pursuit_agents,
        _signal_target_rendezvous_wait_agents,
        _signal_target_scan_opportunity_label_mask,
        _signal_target_scan_broadcaster_agents,
        _signal_visible_clue_action_label_mask,
        _signal_visible_clue_miss_agents,
        _signal_visible_target_match_features,
        _slice_recurrent_episode,
    )

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
    ))
    obs, _ = env.reset(seed=0)
    env.grid[:, :] = 0
    x, y = env.agent_positions[0]
    if x < env.map_size - 1:
        target = (x + 1, y)
        action = env.ACTION_RIGHT
    else:
        target = (x - 1, y)
        action = env.ACTION_LEFT
    env.scenario_state.data["target"] = target
    obs[0]["goal_hint"] = np.array([26, target[0], target[1], -1, -1, -1, -1, -1], dtype=np.int16)
    obs[1]["goal_hint"] = np.full_like(obs[1]["goal_hint"], -1)
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        bc_signal_target_pursuit_weight=3.0,
    )
    actions = {
        0: {"action": action, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    }
    ep_data = _new_episode_sequence()

    assert _signal_target_pursuit_agents(env, obs, actions) == [0]
    assert _signal_positive_target_pursuit_agents(env, obs, actions, min_map_size=16) == []
    assert _signal_observation_allows_target(obs[0], target, observed_map_size=8)
    assert _signal_target_pursuit_miss_agents(
        env,
        obs,
        actions,
        {
            0: {"action": env.ACTION_STAY, "message_tokens": []},
            1: {"action": env.ACTION_STAY, "message_tokens": []},
        },
    ) == [0]
    assert _signal_target_pursuit_miss_agents(env, obs, actions, actions) == []
    _append_labeled_step(ep_data, obs, actions, env, cfg)

    assert ep_data["step_weights"] == [3.0, 1.0]

    large_env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=16,
        num_agents=2,
        fov_preset="easy",
        max_steps=40,
    ))
    large_obs, _ = large_env.reset(seed=0)
    large_env.grid[:, :] = 0
    large_env.agent_positions[0] = (8, 8)
    large_target = (9, 8)
    large_decoy = (7, 8)
    large_env.scenario_state.data["target"] = large_target
    large_env.scenario_state.data["decoys"] = [large_decoy]
    large_env.grid[large_target[1], large_target[0]] = TILE_TARGET
    large_env.grid[large_decoy[1], large_decoy[0]] = TILE_TARGET
    large_obs = large_env._build_observations()
    large_obs[0]["goal_hint"] = np.array([
        26,
        large_target[0],
        large_target[1],
        -1, -1, -1, -1, -1,
    ], dtype=np.int16)
    large_obs[1]["goal_hint"] = np.full_like(large_obs[0]["goal_hint"], -1)
    large_oracle = {
        0: {"action": large_env.ACTION_RIGHT, "message_tokens": []},
        1: {"action": large_env.ACTION_STAY, "message_tokens": []},
    }
    large_model_decoy = {
        0: {"action": large_env.ACTION_LEFT, "message_tokens": []},
        1: {"action": large_env.ACTION_STAY, "message_tokens": []},
    }
    large_model_decoy_scan = {
        0: {"action": large_env.ACTION_INTERACT, "message_tokens": []},
        1: {"action": large_env.ACTION_STAY, "message_tokens": []},
    }
    large_model_stay = {
        0: {"action": large_env.ACTION_STAY, "message_tokens": []},
        1: {"action": large_env.ACTION_STAY, "message_tokens": []},
    }
    assert _signal_decoy_pursuit_agents(large_env, large_model_decoy) == [0]
    assert _signal_rejected_target_drift_agents(large_env, large_obs, large_model_decoy) == [0]
    assert _signal_rejected_target_drift_agents(large_env, large_obs, large_oracle) == []
    target_match_mask, target_match_action_id = _signal_target_match_action_label_mask(
        large_env,
        large_obs,
        large_oracle,
    )
    np.testing.assert_allclose(target_match_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        target_match_action_id,
        np.array([large_env.ACTION_RIGHT, -1], dtype=np.int64),
    )
    pursuit_action_mask, pursuit_action_id = _signal_target_pursuit_action_label_mask(
        large_env,
        large_obs,
        RecurrentConfig(scenario="signal_hunt", map_size=16, agents=2),
    )
    np.testing.assert_allclose(pursuit_action_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        pursuit_action_id,
        np.array([large_env.ACTION_RIGHT, -1], dtype=np.int64),
    )
    frontier_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=16,
        agents=2,
        obs_exploration_memory=True,
        bc_signal_frontier_exploration_action_weight=0.5,
        bc_signal_frontier_exploration_min_map_size=16,
    )
    frontier_obs = {
        aid: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in large_obs[aid].items()
        }
        for aid in range(2)
    }
    frontier_obs[0]["self_pos"] = np.array([8, 8], dtype=np.int16)
    frontier_obs[0]["goal_hint"] = np.full_like(frontier_obs[0]["goal_hint"], -1)
    frontier_obs[0]["messages_tokens"] = np.full_like(frontier_obs[0]["messages_tokens"], -1)
    frontier_obs[0]["local_grid"] = np.zeros_like(frontier_obs[0]["local_grid"], dtype=np.int16)
    frontier_obs[0]["action_mask"] = np.ones_like(frontier_obs[0]["action_mask"], dtype=np.float32)
    explored = np.ones((16, 16), dtype=np.int8)
    explored[8, 9] = 0
    frontier_obs[0]["explored_mask"] = explored
    frontier_obs[1]["explored_mask"] = np.ones((16, 16), dtype=np.int8)
    frontier_mask, frontier_action_id = _signal_frontier_exploration_action_label_mask(
        large_env,
        frontier_obs,
        frontier_cfg,
    )
    np.testing.assert_allclose(frontier_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        frontier_action_id,
        np.array([large_env.ACTION_RIGHT, -1], dtype=np.int64),
    )
    nearest_frontier = _signal_nearest_frontier_cell(
        np.array(
            [
                [1, 1, 0, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
            ],
            dtype=np.int8,
        ),
        (0, 0),
    )
    assert nearest_frontier == (1, 0)
    assigned_frontier_mask = np.ones((7, 7), dtype=np.int8)
    assigned_frontier_mask[1, 1] = 0
    assigned_frontier_mask[3, 5] = 0
    anchored_frontier = _signal_assigned_frontier_cell(
        assigned_frontier_mask,
        (3, 3),
        agent_id=1,
        num_agents=4,
    )
    constrained_frontier = _signal_assigned_frontier_cell(
        assigned_frontier_mask,
        (3, 3),
        agent_id=1,
        num_agents=4,
        constraint_targets=((1, 1),),
    )
    assert anchored_frontier in {(4, 3), (5, 2), (5, 4), (6, 3)}
    assert constrained_frontier in {(0, 1), (1, 0), (1, 2), (2, 1)}
    weak_constraint_hint = np.array([24, 1, 25, 1, -1, -1, -1, -1], dtype=np.int16)
    strong_constraint_hint = np.array([24, 1, 25, 1, 23, 0, 0, 7, -1, -1], dtype=np.int16)
    exact_constraint_hint = np.array([26, 1, 1, -1, -1], dtype=np.int16)
    constraint_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=7,
        bc_signal_constraint_frontier_bias=True,
    )
    empty_messages = np.full((1, 8), -1, dtype=np.int16)
    assert _signal_constraint_frontier_targets(
        {"goal_hint": weak_constraint_hint, "messages_tokens": empty_messages},
        constraint_cfg,
        7,
    ) == ()
    strong_targets = _signal_constraint_frontier_targets(
        {"goal_hint": strong_constraint_hint, "messages_tokens": empty_messages},
        constraint_cfg,
        7,
    )
    assert set(strong_targets) == {(1, 1), (3, 1), (1, 3), (3, 3)}
    assert _signal_constraint_frontier_targets(
        {"goal_hint": exact_constraint_hint, "messages_tokens": empty_messages},
        constraint_cfg,
        7,
    ) == ()
    default_frontier_assist = _apply_signal_frontier_exploration_assist(
        frontier_cfg,
        frontier_obs,
        torch.tensor([large_env.ACTION_STAY, large_env.ACTION_STAY], dtype=torch.long),
    )
    assert default_frontier_assist.tolist() == [large_env.ACTION_STAY, large_env.ACTION_STAY]
    frontier_assist_cfg = RecurrentConfig(
        **{
            **vars(frontier_cfg),
            "eval_signal_frontier_exploration_assist": True,
        }
    )
    assisted_frontier = _apply_signal_frontier_exploration_assist(
        frontier_assist_cfg,
        frontier_obs,
        torch.tensor([large_env.ACTION_STAY, large_env.ACTION_STAY], dtype=torch.long),
    )
    assert assisted_frontier.tolist() == [large_env.ACTION_RIGHT, large_env.ACTION_STAY]
    assert _signal_frontier_exploration_miss_agents(
        large_env,
        frontier_obs,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": []},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        frontier_cfg,
    ) == [0]
    assert _signal_frontier_exploration_miss_agents(
        large_env,
        frontier_obs,
        {
            0: {"action": large_env.ACTION_RIGHT, "message_tokens": []},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        frontier_cfg,
    ) == []
    spread_env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=16,
        num_agents=4,
        fov_preset="medium",
        max_steps=40,
    ))
    spread_obs, _ = spread_env.reset(seed=0)
    spread_env.grid[:, :] = 0
    spread_env.scenario_state.data["target"] = (15, 15)
    spread_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=16,
        agents=4,
        obs_exploration_memory=True,
        bc_signal_frontier_exploration_action_weight=0.5,
        bc_signal_frontier_exploration_min_map_size=16,
    )
    spread_explored = np.ones((16, 16), dtype=np.int8)
    spread_explored[7, 8] = 0
    spread_explored[8, 9] = 0
    spread_explored[9, 8] = 0
    spread_explored[8, 7] = 0
    for aid in range(4):
        spread_obs[aid]["self_pos"] = np.array([8, 8], dtype=np.int16)
        spread_obs[aid]["goal_hint"] = np.full_like(spread_obs[aid]["goal_hint"], -1)
        spread_obs[aid]["messages_tokens"] = np.full_like(spread_obs[aid]["messages_tokens"], -1)
        spread_obs[aid]["local_grid"] = np.zeros_like(spread_obs[aid]["local_grid"], dtype=np.int16)
        spread_obs[aid]["action_mask"] = np.ones_like(spread_obs[aid]["action_mask"], dtype=np.float32)
        spread_obs[aid]["explored_mask"] = spread_explored.copy()
    spread_mask, spread_action_id = _signal_frontier_exploration_action_label_mask(
        spread_env,
        spread_obs,
        spread_cfg,
    )
    np.testing.assert_allclose(spread_mask, np.ones((4,), dtype=np.float32))
    np.testing.assert_array_equal(
        spread_action_id,
        np.array([
            spread_env.ACTION_UP,
            spread_env.ACTION_RIGHT,
            spread_env.ACTION_DOWN,
            spread_env.ACTION_LEFT,
        ], dtype=np.int64),
    )
    sweep_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=16,
        agents=2,
        obs_exploration_memory=True,
        bc_signal_evidence_sweep_action_weight=0.5,
        bc_signal_evidence_sweep_min_map_size=16,
    )
    sweep_obs = {
        aid: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in frontier_obs[aid].items()
        }
        for aid in range(2)
    }
    sweep_explored = np.ones((16, 16), dtype=np.int8)
    sweep_explored[8, 2] = 0
    sweep_explored[8, 13] = 0
    for aid in range(2):
        sweep_obs[aid]["self_pos"] = np.array([8, 8], dtype=np.int16)
        sweep_obs[aid]["goal_hint"] = np.full_like(sweep_obs[aid]["goal_hint"], -1)
        sweep_obs[aid]["messages_tokens"] = np.full_like(sweep_obs[aid]["messages_tokens"], -1)
        sweep_obs[aid]["local_grid"] = np.zeros_like(sweep_obs[aid]["local_grid"], dtype=np.int16)
        sweep_obs[aid]["action_mask"] = np.ones_like(sweep_obs[aid]["action_mask"], dtype=np.float32)
        sweep_obs[aid]["explored_mask"] = sweep_explored.copy()
    default_sweep_mask, default_sweep_action_id = _signal_evidence_sweep_action_label_mask(
        large_env,
        sweep_obs,
        RecurrentConfig(
            scenario="signal_hunt",
            map_size=16,
            agents=2,
            obs_exploration_memory=True,
        ),
    )
    np.testing.assert_allclose(default_sweep_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(default_sweep_action_id, np.array([-1, -1], dtype=np.int64))
    sweep_mask, sweep_action_id = _signal_evidence_sweep_action_label_mask(
        large_env,
        sweep_obs,
        sweep_cfg,
    )
    np.testing.assert_allclose(sweep_mask, np.ones((2,), dtype=np.float32))
    np.testing.assert_array_equal(
        sweep_action_id,
        np.array([large_env.ACTION_LEFT, large_env.ACTION_RIGHT], dtype=np.int64),
    )
    assert _signal_evidence_sweep_miss_agents(
        large_env,
        sweep_obs,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": []},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        sweep_cfg,
    ) == [0, 1]
    assert _signal_evidence_sweep_miss_agents(
        large_env,
        sweep_obs,
        {
            0: {"action": large_env.ACTION_LEFT, "message_tokens": []},
            1: {"action": large_env.ACTION_RIGHT, "message_tokens": []},
        },
        sweep_cfg,
    ) == []
    target_frontier_obs = {
        aid: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in frontier_obs[aid].items()
        }
        for aid in range(2)
    }
    target_frontier_obs[0]["goal_hint"] = np.array([
        26,
        large_target[0],
        large_target[1],
        26,
        large_decoy[0],
        large_decoy[1],
        -1,
        -1,
    ], dtype=np.int16)
    target_frontier_pursuit_mask, target_frontier_pursuit_action_id = (
        _signal_target_pursuit_action_label_mask(
            large_env,
            target_frontier_obs,
            frontier_cfg,
        )
    )
    np.testing.assert_allclose(
        target_frontier_pursuit_mask,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        target_frontier_pursuit_action_id,
        np.array([large_env.ACTION_RIGHT, -1], dtype=np.int64),
    )
    target_frontier_mask, target_frontier_action_id = (
        _signal_frontier_exploration_action_label_mask(
            large_env,
            target_frontier_obs,
            frontier_cfg,
        )
    )
    np.testing.assert_allclose(
        target_frontier_mask,
        np.array([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        target_frontier_action_id,
        np.array([-1, -1], dtype=np.int64),
    )
    clue_obs = {
        aid: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in frontier_obs[aid].items()
        }
        for aid in range(2)
    }
    clue_obs[0]["local_grid"] = np.zeros_like(frontier_obs[0]["local_grid"], dtype=np.int16)
    cy, cx = clue_obs[0]["local_grid"].shape[0] // 2, clue_obs[0]["local_grid"].shape[1] // 2
    clue_obs[0]["local_grid"][cy, cx] = TILE_CLUE
    clue_mask, clue_action_id = _signal_frontier_exploration_action_label_mask(
        large_env,
        clue_obs,
        frontier_cfg,
    )
    np.testing.assert_allclose(clue_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(clue_action_id, np.array([-1, -1], dtype=np.int64))
    visible_clue_mask, visible_clue_action_id = _signal_visible_clue_action_label_mask(
        large_env,
        clue_obs,
        frontier_cfg,
    )
    np.testing.assert_allclose(
        visible_clue_mask,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        visible_clue_action_id,
        np.array([large_env.ACTION_INTERACT, -1], dtype=np.int64),
    )
    clue_interact_mask, clue_interact_action_id = _signal_clue_interact_action_label_mask(
        large_env,
        clue_obs,
        frontier_cfg,
    )
    np.testing.assert_allclose(
        clue_interact_mask,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        clue_interact_action_id,
        np.array([large_env.ACTION_INTERACT, -1], dtype=np.int64),
    )
    clue_sweep_mask, clue_sweep_action_id = _signal_evidence_sweep_action_label_mask(
        large_env,
        clue_obs,
        sweep_cfg,
    )
    np.testing.assert_allclose(clue_sweep_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(clue_sweep_action_id, np.array([-1, -1], dtype=np.int64))
    assert _signal_visible_clue_miss_agents(
        large_env,
        clue_obs,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": []},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        frontier_cfg,
    ) == [0]
    assert _signal_visible_clue_miss_agents(
        large_env,
        clue_obs,
        {
            0: {"action": large_env.ACTION_INTERACT, "message_tokens": []},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        frontier_cfg,
    ) == []
    assert _signal_clue_interact_miss_agents(
        large_env,
        clue_obs,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": []},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        frontier_cfg,
    ) == [0]
    assert _signal_clue_interact_miss_agents(
        large_env,
        clue_obs,
        {
            0: {"action": large_env.ACTION_INTERACT, "message_tokens": []},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        frontier_cfg,
    ) == []
    large_env.scenario_state.data["clue_claimed"] = {(8, 8)}
    claimed_clue_interact_mask, claimed_clue_interact_action_id = (
        _signal_clue_interact_action_label_mask(
            large_env,
            clue_obs,
            frontier_cfg,
        )
    )
    np.testing.assert_allclose(
        claimed_clue_interact_mask,
        np.array([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        claimed_clue_interact_action_id,
        np.array([-1, -1], dtype=np.int64),
    )
    claimed_clue_mask, claimed_clue_action_id = _signal_visible_clue_action_label_mask(
        large_env,
        clue_obs,
        frontier_cfg,
    )
    np.testing.assert_allclose(
        claimed_clue_mask,
        np.array([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        claimed_clue_action_id,
        np.array([-1, -1], dtype=np.int64),
    )
    large_env.scenario_state.data["clue_claimed"] = set()
    decoy_match_mask, decoy_match_action_id = _signal_target_match_action_label_mask(
        large_env,
        large_obs,
        large_model_decoy,
    )
    np.testing.assert_allclose(decoy_match_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(decoy_match_action_id, np.array([-1, -1], dtype=np.int64))
    assert _signal_positive_target_pursuit_agents(
        large_env,
        large_obs,
        large_oracle,
        min_map_size=16,
    ) == [0]
    assert _signal_positive_target_pursuit_agents(
        large_env,
        large_obs,
        large_oracle,
        min_map_size=32,
    ) == []
    assert _signal_target_decoy_drift_miss_agents(
        large_env,
        large_obs,
        large_oracle,
        large_model_decoy,
        min_map_size=16,
    ) == [0]
    assert _signal_target_discovery_miss_agents(
        large_env,
        large_obs,
        large_oracle,
        large_model_decoy,
        min_map_size=16,
    ) == []
    assert _signal_target_discovery_miss_agents(
        large_env,
        large_obs,
        large_oracle,
        large_model_stay,
        min_map_size=16,
    ) == [0]
    stall_history = {
        0: [(8, 8), (7, 8), (8, 8)],
        1: [(0, 0), (1, 0), (2, 0)],
    }
    assert _signal_movement_stall_miss_agents(
        large_env,
        large_obs,
        large_oracle,
        large_model_decoy,
        stall_history,
        min_map_size=16,
        window=4,
    ) == [0]
    assert _signal_movement_stall_miss_agents(
        large_env,
        large_obs,
        large_oracle,
        large_oracle,
        stall_history,
        min_map_size=16,
        window=4,
    ) == []
    assert _signal_movement_stall_miss_agents(
        large_env,
        large_obs,
        large_oracle,
        large_model_decoy,
        stall_history,
        min_map_size=32,
        window=4,
    ) == []
    assert _signal_target_discovery_miss_agents(
        large_env,
        large_obs,
        large_oracle,
        large_oracle,
        min_map_size=16,
    ) == []
    assert _signal_target_discovery_miss_agents(
        large_env,
        large_obs,
        large_oracle,
        large_model_decoy,
        min_map_size=32,
    ) == []
    assert _signal_target_decoy_drift_miss_agents(
        large_env,
        large_obs,
        large_oracle,
        large_model_decoy,
        min_map_size=32,
    ) == []

    large_env.agent_positions[0] = large_target
    large_env.agent_positions[1] = (large_target[0] - 1, large_target[1])
    handoff_obs = {
        0: dict(large_obs[0]),
        1: dict(large_obs[1]),
    }
    handoff_obs[0]["goal_hint"] = np.array([
        26,
        large_target[0],
        large_target[1],
        -1, -1, -1, -1, -1,
    ], dtype=np.int16)
    handoff_obs[1]["goal_hint"] = handoff_obs[0]["goal_hint"].copy()
    handoff_oracle = {
        0: {"action": large_env.ACTION_INTERACT, "message_tokens": []},
        1: {"action": large_env.ACTION_RIGHT, "message_tokens": []},
    }
    handoff_model_idle = {
        0: {"action": large_env.ACTION_INTERACT, "message_tokens": []},
        1: {"action": large_env.ACTION_STAY, "message_tokens": []},
    }
    handoff_model_join = {
        0: {"action": large_env.ACTION_INTERACT, "message_tokens": []},
        1: {"action": large_env.ACTION_RIGHT, "message_tokens": []},
    }
    assert _signal_target_handoff_miss_agents(
        large_env,
        handoff_obs,
        handoff_oracle,
        handoff_model_idle,
        feedback=None,
    ) == [1]
    assert _signal_target_handoff_miss_agents(
        large_env,
        handoff_obs,
        handoff_oracle,
        handoff_model_join,
        feedback=None,
    ) == []
    handoff_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=16,
        agents=2,
        obs_feedback=True,
        obs_signal_sync_feedback=True,
        obs_signal_scan_state=True,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        dagger_target_scan_broadcast_labels=True,
    )
    exact_handoff_cfg = RecurrentConfig(
        **{
            **vars(handoff_cfg),
            "dagger_target_handoff_requires_exact_target": True,
        }
    )
    rendezvous_cfg = RecurrentConfig(
        **{
            **vars(handoff_cfg),
            "dagger_signal_target_rendezvous_labels": True,
            "dagger_signal_target_rendezvous_min_map_size": 16,
            "dagger_signal_target_rendezvous_max_agents": 2,
        }
    )
    large_env.steps = 0
    large_env.scenario_state.data["scan_log"] = {}
    large_env.agent_positions[0] = large_target
    large_env.agent_positions[1] = (large_target[0] - 1, large_target[1])
    rendezvous_obs = large_env._build_observations()
    rendezvous_obs[0]["goal_hint"] = np.array([
        26,
        large_target[0],
        large_target[1],
        -1, -1, -1, -1, -1,
    ], dtype=np.int16)
    rendezvous_obs[1]["goal_hint"] = rendezvous_obs[0]["goal_hint"].copy()
    rendezvous_actions, rendezvous_agents, rendezvous_wait = _apply_signal_target_rendezvous_overrides(
        rendezvous_cfg,
        large_env,
        rendezvous_obs,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": [1]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": [2]},
        },
    )
    assert rendezvous_agents == [0, 1]
    assert rendezvous_wait == []
    assert rendezvous_actions[0]["action"] == large_env.ACTION_INTERACT
    assert rendezvous_actions[1]["action"] == large_env.ACTION_RIGHT
    large_env.agent_positions[1] = (max(0, large_target[0] - 6), large_target[1])
    far_rendezvous_obs = large_env._build_observations()
    far_rendezvous_obs[0]["goal_hint"] = rendezvous_obs[0]["goal_hint"].copy()
    far_rendezvous_obs[1]["goal_hint"] = rendezvous_obs[1]["goal_hint"].copy()
    far_actions, far_agents, far_wait = _apply_signal_target_rendezvous_overrides(
        rendezvous_cfg,
        large_env,
        far_rendezvous_obs,
        {
            0: {"action": large_env.ACTION_INTERACT, "message_tokens": [1]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": [2]},
        },
    )
    assert far_agents == [0, 1]
    assert far_wait == [0]
    assert far_actions[0]["action"] == large_env.ACTION_STAY
    assert far_actions[1]["action"] == large_env.ACTION_RIGHT
    assert _signal_target_rendezvous_wait_agents(
        rendezvous_cfg,
        large_env,
        far_rendezvous_obs,
    ) == [0]
    wait_opportunity_mask, wait_opportunity_kind = _signal_target_scan_opportunity_label_mask(
        large_env,
        far_rendezvous_obs,
        rendezvous_cfg,
    )
    np.testing.assert_allclose(wait_opportunity_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        wait_opportunity_kind,
        np.array([-1, -1], dtype=np.int64),
    )
    large_env.agent_positions[0] = large_target
    large_env.agent_positions[1] = (large_target[0] - 1, large_target[1])
    large_env.steps = 0
    large_env.scenario_state.data["scan_log"] = {}
    initial_broadcast_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=16,
        agents=2,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        bc_signal_initial_message_weight=4.0,
        bc_signal_initial_message_loss_weight=4.0,
        dagger_initial_target_broadcast_labels=True,
        eval_signal_initial_exact_message_copy_assist=True,
    )
    initial_weights = _apply_signal_initial_message_weight(
        np.ones((2,), dtype=np.float32),
        cfg=initial_broadcast_cfg,
        actions={
            0: {"action": large_env.ACTION_STAY, "message_tokens": [21, 7, 4, 3, 2]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        current_step=0,
    )
    np.testing.assert_allclose(initial_weights, np.array([4.0, 1.0], dtype=np.float32))
    late_weights = _apply_signal_initial_message_weight(
        np.ones((2,), dtype=np.float32),
        cfg=initial_broadcast_cfg,
        actions={
            0: {"action": large_env.ACTION_STAY, "message_tokens": [21, 7, 4, 3, 2]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        current_step=1,
    )
    np.testing.assert_allclose(late_weights, np.ones((2,), dtype=np.float32))
    initial_broadcasted, initial_broadcast_agents = _apply_signal_initial_target_broadcast_overrides(
        initial_broadcast_cfg,
        handoff_obs,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": [1]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        current_step=0,
    )
    assert initial_broadcast_agents == [0, 1]
    assert initial_broadcasted[0]["message_tokens"] == [26, large_target[0], large_target[1]]
    assert initial_broadcasted[1]["message_tokens"] == [26, large_target[0], large_target[1]]
    initial_msg_ep_data = _new_episode_sequence()
    _append_labeled_step(
        initial_msg_ep_data,
        handoff_obs,
        initial_broadcasted,
        large_env,
        initial_broadcast_cfg,
    )
    _append_labeled_step(
        initial_msg_ep_data,
        handoff_obs,
        initial_broadcasted,
        large_env,
        initial_broadcast_cfg,
    )
    assert initial_msg_ep_data["signal_initial_message_mask"] == [1.0, 1.0, 0.0, 0.0]
    assert initial_msg_ep_data["signal_constraint_message_mask"] == [0.0, 0.0, 0.0, 0.0]
    initial_msg_episode = _finalize_episode_sequence(
        initial_msg_ep_data,
        large_env,
        initial_broadcast_cfg,
    )
    np.testing.assert_allclose(
        initial_msg_episode["signal_initial_message_mask"],
        np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        initial_msg_episode["signal_constraint_message_mask"],
        np.zeros((2, 2), dtype=np.float32),
    )
    assert not _signal_constraint_message_label([24, 1], current_step=0)
    assert _signal_constraint_message_label([24, 1], current_step=1)
    assert _signal_constraint_message_label([22, 2, 3, 4, -1, 1], current_step=1)
    assert not _signal_constraint_message_label([26, large_target[0], large_target[1]], current_step=1)
    constraint_msg_ep_data = _new_episode_sequence()
    constraint_msg_actions = {
        0: {"action": large_env.ACTION_STAY, "message_tokens": [24, 1]},
        1: {"action": large_env.ACTION_STAY, "message_tokens": []},
    }
    _append_labeled_step(
        constraint_msg_ep_data,
        handoff_obs,
        constraint_msg_actions,
        large_env,
        initial_broadcast_cfg,
    )
    _append_labeled_step(
        constraint_msg_ep_data,
        handoff_obs,
        constraint_msg_actions,
        large_env,
        initial_broadcast_cfg,
    )
    assert constraint_msg_ep_data["signal_initial_message_mask"] == [1.0, 0.0, 0.0, 0.0]
    assert constraint_msg_ep_data["signal_constraint_message_mask"] == [0.0, 0.0, 1.0, 0.0]
    copied_initial_messages = _apply_signal_initial_exact_message_copy_assist(
        initial_broadcast_cfg,
        handoff_obs,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": [1, 2, 3]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        current_step=0,
    )
    assert copied_initial_messages[0]["message_tokens"] == [
        26,
        large_target[0],
        large_target[1],
    ]
    assert copied_initial_messages[1]["message_tokens"] == []
    copied_late_messages = _apply_signal_initial_exact_message_copy_assist(
        initial_broadcast_cfg,
        handoff_obs,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": [1, 2, 3]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        current_step=1,
    )
    assert copied_late_messages[0]["message_tokens"] == [1, 2, 3]
    late_broadcasted, late_broadcast_agents = _apply_signal_initial_target_broadcast_overrides(
        initial_broadcast_cfg,
        handoff_obs,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": [1]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        current_step=1,
    )
    assert late_broadcast_agents == []
    assert late_broadcasted[0]["message_tokens"] == [1]
    large_env.steps = 6
    large_env.scenario_state.data["scan_log"] = {0: 5}
    feedback = _feedback_matrix(handoff_cfg, 2, info={}, env=large_env)
    corrected, handoff_agents = _apply_signal_target_handoff_overrides(
        handoff_cfg,
        large_env,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": [1]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": [2]},
        },
        feedback,
    )
    assert handoff_agents == [1]
    assert corrected[1]["action"] == large_env.ACTION_RIGHT
    assert corrected[1]["message_tokens"] == [2]
    no_target_info_obs = {
        0: dict(handoff_obs[0]),
        1: dict(handoff_obs[1]),
    }
    no_target_info_obs[1]["goal_hint"] = np.full_like(handoff_obs[1]["goal_hint"], -1)
    no_target_info_obs[1]["messages_tokens"] = np.full_like(
        handoff_obs[1].get("messages_tokens", np.zeros((1, 8), dtype=np.int16)),
        -1,
    )
    gated_corrected, gated_handoff_agents = _apply_signal_target_handoff_overrides(
        handoff_cfg,
        large_env,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": [1]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": [2]},
        },
        feedback,
        obs=no_target_info_obs,
    )
    assert gated_handoff_agents == []
    assert gated_corrected[1]["action"] == large_env.ACTION_STAY
    informed_corrected, informed_handoff_agents = _apply_signal_target_handoff_overrides(
        handoff_cfg,
        large_env,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": [1]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": [2]},
        },
        feedback,
        obs=handoff_obs,
    )
    assert informed_handoff_agents == [1]
    assert informed_corrected[1]["action"] == large_env.ACTION_RIGHT
    weak_handoff_obs = {
        0: dict(handoff_obs[0]),
        1: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in handoff_obs[1].items()
        },
    }
    weak_handoff_obs[1]["self_pos"] = np.array(
        [large_target[0] - 1, large_target[1]],
        dtype=np.int16,
    )
    weak_grid = np.zeros_like(handoff_obs[1]["local_grid"], dtype=np.int16)
    cy, cx = weak_grid.shape[0] // 2, weak_grid.shape[1] // 2
    weak_grid[cy, min(cx + 1, weak_grid.shape[1] - 1)] = TILE_TARGET
    weak_handoff_obs[1]["local_grid"] = weak_grid
    weak_handoff_obs[1]["goal_hint"] = np.array([
        21,
        0,
        large_target[0],
        large_target[1],
        1,
        -1,
        -1,
        -1,
    ], dtype=np.int16)
    weak_handoff_obs[1]["messages_tokens"] = np.full_like(
        handoff_obs[1].get("messages_tokens", np.zeros((2, 8), dtype=np.int16)),
        -1,
    )
    assert not _signal_exact_target_handoff_candidate(
        weak_handoff_obs[1],
        exact_handoff_cfg,
        large_target,
    )
    weak_corrected, weak_handoff_agents = _apply_signal_target_handoff_overrides(
        handoff_cfg,
        large_env,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": [1]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": [2]},
        },
        feedback,
        obs=weak_handoff_obs,
    )
    assert weak_handoff_agents == [1]
    assert weak_corrected[1]["action"] == large_env.ACTION_RIGHT
    exact_gated_corrected, exact_gated_handoff_agents = _apply_signal_target_handoff_overrides(
        exact_handoff_cfg,
        large_env,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": [1]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": [2]},
        },
        feedback,
        obs=weak_handoff_obs,
    )
    assert exact_gated_handoff_agents == []
    assert exact_gated_corrected[1]["action"] == large_env.ACTION_STAY
    exact_message_obs = {
        0: weak_handoff_obs[0],
        1: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in weak_handoff_obs[1].items()
        },
    }
    exact_message_obs[1]["messages_tokens"] = np.array([
        [26, large_target[0], large_target[1], -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ], dtype=np.int16)
    trusted_scan_state = {
        "step": 6,
        "scan_window": 3,
        "scan_log": {0: 5},
        "scan_pos": {0: [large_target[0], large_target[1]]},
    }
    assert _signal_exact_target_handoff_candidate(
        exact_message_obs[1],
        exact_handoff_cfg,
        large_target,
        scan_state=trusted_scan_state,
        agent_id=1,
    )
    exact_corrected, exact_handoff_agents = _apply_signal_target_handoff_overrides(
        exact_handoff_cfg,
        large_env,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": [1]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": [2]},
        },
        feedback,
        obs=exact_message_obs,
        scan_state=trusted_scan_state,
    )
    assert exact_handoff_agents == [1]
    assert exact_corrected[1]["action"] == large_env.ACTION_RIGHT
    assert _signal_target_scan_broadcaster_agents(handoff_cfg, large_env, feedback, info={}) == []
    broadcast_feedback = _feedback_matrix(
        handoff_cfg,
        2,
        info={"events": {0: [{"event": "target_scan"}, {"event": "first_target_scan"}], 1: []}},
        env=large_env,
    )
    broadcast_info = {"events": {0: [{"event": "target_scan"}, {"event": "first_target_scan"}], 1: []}}
    assert _signal_target_scan_broadcaster_agents(
        handoff_cfg,
        large_env,
        broadcast_feedback,
        info=broadcast_info,
    ) == [0]
    broadcasted, broadcast_agents = _apply_signal_target_scan_broadcast_overrides(
        handoff_cfg,
        large_env,
        {
            0: {"action": large_env.ACTION_INTERACT, "message_tokens": [1]},
            1: {"action": large_env.ACTION_STAY, "message_tokens": [2]},
        },
        broadcast_feedback,
        info=broadcast_info,
    )
    assert broadcast_agents == [0]
    assert broadcasted[0]["message_tokens"] == [26, large_target[0], large_target[1]]
    assert broadcasted[1]["message_tokens"] == [2]

    nav_cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=16,
        agents=2,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        eval_signal_exact_target_navigation_assist=True,
        eval_signal_exact_target_memory_steps=10,
    )
    nav_obs = {
        0: dict(handoff_obs[0]),
        1: dict(handoff_obs[1]),
    }
    nav_obs[0]["goal_hint"] = np.full_like(handoff_obs[0]["goal_hint"], -1)
    nav_obs[1]["goal_hint"] = np.full_like(handoff_obs[1]["goal_hint"], -1)
    nav_obs[1]["self_pos"] = np.array([large_target[0] - 1, large_target[1]], dtype=np.int16)
    nav_obs[1]["messages_tokens"] = np.array(
        [[26, large_target[0], large_target[1], -1, -1, -1, -1, -1]],
        dtype=np.int16,
    )
    idle_acts = torch.tensor([large_env.ACTION_STAY, large_env.ACTION_STAY], dtype=torch.long)
    untrusted_nav = _apply_signal_exact_target_navigation_assist(
        nav_cfg,
        nav_obs,
        idle_acts,
        scan_state=None,
    )
    assert untrusted_nav.tolist() == [large_env.ACTION_STAY, large_env.ACTION_STAY]
    trusted_scan_state = {
        "step": 6,
        "scan_window": 3,
        "scan_log": {0: 5},
        "scan_pos": {0: [large_target[0], large_target[1]]},
    }
    trusted_nav = _apply_signal_exact_target_navigation_assist(
        nav_cfg,
        nav_obs,
        idle_acts,
        scan_state=trusted_scan_state,
    )
    assert trusted_nav.tolist() == [large_env.ACTION_STAY, large_env.ACTION_RIGHT]
    remembered_obs = {
        0: dict(nav_obs[0]),
        1: dict(nav_obs[1]),
    }
    remembered_obs[1]["messages_tokens"] = np.full_like(nav_obs[1]["messages_tokens"], -1)
    remembered_scan_state = {
        "step": 12,
        "scan_window": 3,
        "scan_log": {0: 5},
        "scan_pos": {0: [large_target[0], large_target[1]]},
        "exact_target_memory": trusted_scan_state["exact_target_memory"],
    }
    remembered_nav = _apply_signal_exact_target_navigation_assist(
        nav_cfg,
        remembered_obs,
        idle_acts,
        scan_state=remembered_scan_state,
    )
    assert remembered_nav.tolist() == [large_env.ACTION_STAY, large_env.ACTION_RIGHT]
    default_pursuit_mask, default_pursuit_action_id = (
        _signal_target_pursuit_action_label_mask(
            large_env,
            remembered_obs,
            nav_cfg,
            scan_state=remembered_scan_state,
        )
    )
    np.testing.assert_allclose(
        default_pursuit_mask,
        np.array([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        default_pursuit_action_id,
        np.array([-1, -1], dtype=np.int64),
    )
    memory_label_cfg = RecurrentConfig(
        **{
            **vars(nav_cfg),
            "bc_signal_target_pursuit_trust_exact_memory": True,
        }
    )
    remembered_pursuit_mask, remembered_pursuit_action_id = (
        _signal_target_pursuit_action_label_mask(
            large_env,
            remembered_obs,
            memory_label_cfg,
            scan_state=remembered_scan_state,
        )
    )
    np.testing.assert_allclose(
        remembered_pursuit_mask,
        np.array([0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        remembered_pursuit_action_id,
        np.array([-1, large_env.ACTION_RIGHT], dtype=np.int64),
    )
    capped_memory_obs = {
        0: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in remembered_obs[0].items()
        },
        1: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in remembered_obs[1].items()
        },
    }
    capped_memory_obs[0]["self_pos"] = np.array(
        [large_target[0] - 3, large_target[1]],
        dtype=np.int16,
    )
    capped_memory_obs[1]["self_pos"] = np.array(
        [large_target[0] - 1, large_target[1]],
        dtype=np.int16,
    )
    capped_memory_obs[0]["goal_hint"] = np.full_like(capped_memory_obs[0]["goal_hint"], -1)
    capped_memory_obs[1]["goal_hint"] = np.full_like(capped_memory_obs[1]["goal_hint"], -1)
    capped_memory_obs[0]["messages_tokens"] = np.full_like(
        capped_memory_obs[0]["messages_tokens"],
        -1,
    )
    capped_memory_obs[1]["messages_tokens"] = np.full_like(
        capped_memory_obs[1]["messages_tokens"],
        -1,
    )
    capped_memory_obs[0]["action_mask"] = np.ones_like(
        capped_memory_obs[0]["action_mask"],
        dtype=np.float32,
    )
    capped_memory_obs[1]["action_mask"] = np.ones_like(
        capped_memory_obs[1]["action_mask"],
        dtype=np.float32,
    )
    capped_memory_scan_state = {
        **remembered_scan_state,
        "exact_target_memory": {
            0: {"pos": [large_target[0], large_target[1]], "step": 10},
            1: {"pos": [large_target[0], large_target[1]], "step": 10},
        },
    }
    uncapped_memory_mask, uncapped_memory_action_id = (
        _signal_target_pursuit_action_label_mask(
            large_env,
            capped_memory_obs,
            memory_label_cfg,
            scan_state=capped_memory_scan_state,
        )
    )
    np.testing.assert_allclose(
        uncapped_memory_mask,
        np.array([1.0, 1.0], dtype=np.float32),
    )
    assert uncapped_memory_action_id[0] >= 0
    assert uncapped_memory_action_id[1] == large_env.ACTION_RIGHT
    capped_memory_label_cfg = RecurrentConfig(
        **{
            **vars(memory_label_cfg),
            "bc_signal_target_pursuit_max_agents": 1,
        }
    )
    capped_memory_mask, capped_memory_action_id = (
        _signal_target_pursuit_action_label_mask(
            large_env,
            capped_memory_obs,
            capped_memory_label_cfg,
            scan_state=capped_memory_scan_state,
        )
    )
    np.testing.assert_allclose(
        capped_memory_mask,
        np.array([0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        capped_memory_action_id,
        np.array([-1, large_env.ACTION_RIGHT], dtype=np.int64),
    )
    remembered_ep_data = _new_episode_sequence()
    _append_labeled_step(
        remembered_ep_data,
        remembered_obs,
        {
            0: {"action": large_env.ACTION_STAY, "message_tokens": []},
            1: {"action": large_env.ACTION_RIGHT, "message_tokens": []},
        },
        large_env,
        memory_label_cfg,
        scan_state=remembered_scan_state,
    )
    assert remembered_ep_data["signal_target_pursuit_action_mask"] == [0.0, 1.0]
    assert remembered_ep_data["signal_target_pursuit_action_id"] == [-1, large_env.ACTION_RIGHT]
    detour_obs = {
        "self_pos": np.array([9, 7], dtype=np.int16),
        "local_grid": np.array(
            [
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_CLUE, TILE_EMPTY, TILE_EMPTY, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_EMPTY, TILE_UNKNOWN, TILE_UNKNOWN, TILE_WALL, TILE_UNKNOWN],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_WALL, TILE_TARGET, TILE_WALL, TILE_WATER, TILE_WALL, TILE_WALL],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_WALL, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_WALL],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_WALL, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_WALL],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_WALL, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_WALL],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_WALL, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_WALL],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_WALL, TILE_WALL, TILE_WALL, TILE_WALL, TILE_WALL],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN],
            ],
            dtype=np.int16,
        ),
        "action_mask": np.array([1, 1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
    }
    assert _signal_navigation_action_from_obs(detour_obs, (6, 7)) == large_env.ACTION_UP
    reverse_detour_obs = {
        "self_pos": np.array([9, 5], dtype=np.int16),
        "local_grid": np.array(
            [
                [TILE_UNKNOWN, TILE_WALL, TILE_WALL, TILE_WALL, TILE_WALL, TILE_WALL, TILE_WALL, TILE_UNKNOWN, TILE_UNKNOWN],
                [TILE_EMPTY, TILE_WALL, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_WALL, TILE_WALL, TILE_UNKNOWN],
                [TILE_UNKNOWN, TILE_EMPTY, TILE_EMPTY, TILE_CLUE, TILE_EMPTY, TILE_EMPTY, TILE_CLUE, TILE_WALL, TILE_UNKNOWN],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_WALL, TILE_TARGET, TILE_WALL, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_WALL, TILE_EMPTY, TILE_EMPTY, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_WALL, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_UNKNOWN],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_WALL],
                [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_EMPTY, TILE_WALL],
            ],
            dtype=np.int16,
        ),
        "action_mask": np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.float32),
        "goal_hint": np.array([26, 6, 7, -1, -1, -1, -1, -1], dtype=np.int16),
        "messages_tokens": np.array([[26, 6, 7, -1, -1, -1, -1, -1]], dtype=np.int16),
    }
    reverse_scan_state = {
        "step": 5,
        "scan_window": 3,
        "scan_log": {1: 4},
        "scan_pos": {1: [6, 7]},
        "prev_positions": {0: [9, 6]},
    }
    reverse_detour_nav = _apply_signal_exact_target_navigation_assist(
        nav_cfg,
        {0: reverse_detour_obs, 1: {}},
        torch.tensor([large_env.ACTION_STAY, large_env.ACTION_STAY], dtype=torch.long),
        scan_state=reverse_scan_state,
    )
    assert reverse_detour_nav.tolist() == [large_env.ACTION_UP, large_env.ACTION_STAY]
    nav_obs[1]["self_pos"] = np.array([large_target[0], large_target[1]], dtype=np.int16)
    nav_obs[1]["action_mask"] = np.asarray(nav_obs[1]["action_mask"], dtype=np.float32).copy()
    nav_obs[1]["action_mask"][large_env.ACTION_INTERACT] = 1.0
    trusted_interact = _apply_signal_exact_target_navigation_assist(
        nav_cfg,
        nav_obs,
        idle_acts,
        scan_state=trusted_scan_state,
    )
    assert trusted_interact.tolist() == [large_env.ACTION_STAY, large_env.ACTION_INTERACT]
    scanner_obs = {
        0: dict(nav_obs[0]),
        1: dict(nav_obs[1]),
    }
    scanner_obs[0]["self_pos"] = np.array([large_target[0], large_target[1]], dtype=np.int16)
    scanner_obs[0]["goal_hint"] = np.array(
        [26, large_target[0], large_target[1], -1, -1, -1, -1, -1],
        dtype=np.int16,
    )
    scanner_obs[0]["messages_tokens"] = np.array(
        [[26, large_target[0], large_target[1], -1, -1, -1, -1, -1]],
        dtype=np.int16,
    )
    scanner_obs[0]["action_mask"] = np.asarray(scanner_obs[0]["action_mask"], dtype=np.float32).copy()
    scanner_obs[0]["action_mask"][large_env.ACTION_STAY] = 1.0
    scanner_obs[0]["action_mask"][large_env.ACTION_INTERACT] = 1.0
    active_scan_nav = _apply_signal_exact_target_navigation_assist(
        nav_cfg,
        scanner_obs,
        torch.tensor([large_env.ACTION_INTERACT, large_env.ACTION_STAY], dtype=torch.long),
        scan_state=trusted_scan_state,
    )
    assert active_scan_nav.tolist() == [large_env.ACTION_STAY, large_env.ACTION_INTERACT]
    joint_scan_state = {
        "step": 6,
        "scan_window": 3,
        "scan_log": {0: 5, 1: 6},
        "scan_pos": {
            0: [large_target[0], large_target[1]],
            1: [large_target[0], large_target[1]],
        },
    }
    joint_scan_nav = _apply_signal_exact_target_navigation_assist(
        nav_cfg,
        scanner_obs,
        torch.tensor([large_env.ACTION_STAY, large_env.ACTION_STAY], dtype=torch.long),
        scan_state=joint_scan_state,
    )
    assert joint_scan_nav.tolist() == [large_env.ACTION_INTERACT, large_env.ACTION_INTERACT]

    large_env.agent_positions[0] = (8, 8)
    large_env.agent_positions[1] = tuple(int(v) for v in large_env.agent_positions[1])
    large_env.steps = 0
    large_env.scenario_state.data["scan_log"] = {}
    large_cfg = RecurrentConfig(scenario="signal_hunt", map_size=16, agents=2)
    large_ep_data = _new_episode_sequence()
    _append_labeled_step(large_ep_data, large_obs, large_oracle, large_env, large_cfg)
    frontier_ep_data = _new_episode_sequence()
    _append_labeled_step(
        frontier_ep_data,
        frontier_obs,
        {
            0: {"action": large_env.ACTION_RIGHT, "message_tokens": []},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        large_env,
        frontier_cfg,
    )
    sweep_ep_data = _new_episode_sequence()
    _append_labeled_step(
        sweep_ep_data,
        sweep_obs,
        {
            0: {"action": large_env.ACTION_LEFT, "message_tokens": []},
            1: {"action": large_env.ACTION_RIGHT, "message_tokens": []},
        },
        large_env,
        sweep_cfg,
    )
    clue_ep_data = _new_episode_sequence()
    _append_labeled_step(
        clue_ep_data,
        clue_obs,
        {
            0: {"action": large_env.ACTION_INTERACT, "message_tokens": []},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        large_env,
        frontier_cfg,
    )
    target_frontier_ep_data = _new_episode_sequence()
    _append_labeled_step(
        target_frontier_ep_data,
        target_frontier_obs,
        {
            0: {"action": large_env.ACTION_RIGHT, "message_tokens": []},
            1: {"action": large_env.ACTION_STAY, "message_tokens": []},
        },
        large_env,
        frontier_cfg,
    )
    assert target_frontier_ep_data["signal_target_pursuit_action_mask"] == [1.0, 0.0]
    assert target_frontier_ep_data["signal_frontier_exploration_action_mask"] == [0.0, 0.0]
    assert _label_latest_signal_decoy_drift_actions(
        large_ep_data,
        num_agents=2,
        agent_ids=[0],
        model_actions=large_model_decoy,
    ) == 1
    assert _label_latest_signal_decoy_scan_actions(
        large_ep_data,
        num_agents=2,
        agent_ids=[0],
        model_actions=large_model_decoy_scan,
    ) == 1
    assert _label_latest_signal_rejected_target_drift_actions(
        large_ep_data,
        num_agents=2,
        agent_ids=[0],
        model_actions=large_model_decoy,
    ) == 1
    large_episode = _finalize_episode_sequence(large_ep_data, large_env, large_cfg)
    np.testing.assert_allclose(
        large_episode["signal_decoy_drift_action_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        large_episode["signal_decoy_drift_action_id"],
        np.array([[large_env.ACTION_LEFT, -1]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        large_episode["signal_decoy_scan_action_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        large_episode["signal_decoy_scan_action_id"],
        np.array([[large_env.ACTION_INTERACT, -1]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        large_episode["signal_target_match_action_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        large_episode["signal_target_match_action_id"],
        np.array([[large_env.ACTION_RIGHT, -1]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        large_episode["signal_target_opportunity_action_mask"],
        np.array([[0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        large_episode["signal_target_opportunity_kind_id"],
        np.array([[-1, -1]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        large_episode["signal_target_pursuit_action_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        large_episode["signal_target_pursuit_action_id"],
        np.array([[large_env.ACTION_RIGHT, -1]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        large_episode["signal_rejected_target_drift_action_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        large_episode["signal_rejected_target_drift_action_id"],
        np.array([[large_env.ACTION_LEFT, -1]], dtype=np.int64),
    )
    frontier_episode = _finalize_episode_sequence(frontier_ep_data, large_env, frontier_cfg)
    np.testing.assert_allclose(
        frontier_episode["signal_frontier_exploration_action_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        frontier_episode["signal_frontier_exploration_action_id"],
        np.array([[large_env.ACTION_RIGHT, -1]], dtype=np.int64),
    )
    sweep_episode = _finalize_episode_sequence(sweep_ep_data, large_env, sweep_cfg)
    np.testing.assert_allclose(
        sweep_episode["signal_evidence_sweep_action_mask"],
        np.array([[1.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        sweep_episode["signal_evidence_sweep_action_id"],
        np.array([[large_env.ACTION_LEFT, large_env.ACTION_RIGHT]], dtype=np.int64),
    )
    clue_episode = _finalize_episode_sequence(clue_ep_data, large_env, frontier_cfg)
    np.testing.assert_allclose(
        clue_episode["signal_clue_interact_action_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        clue_episode["signal_clue_interact_action_id"],
        np.array([[large_env.ACTION_INTERACT, -1]], dtype=np.int64),
    )
    large_replay = _slice_recurrent_episode(large_episode, 0, 1)
    np.testing.assert_allclose(
        large_replay["signal_decoy_drift_action_mask"],
        large_episode["signal_decoy_drift_action_mask"],
    )
    np.testing.assert_array_equal(
        large_replay["signal_decoy_drift_action_id"],
        large_episode["signal_decoy_drift_action_id"],
    )
    np.testing.assert_allclose(
        large_replay["signal_decoy_scan_action_mask"],
        large_episode["signal_decoy_scan_action_mask"],
    )
    np.testing.assert_array_equal(
        large_replay["signal_decoy_scan_action_id"],
        large_episode["signal_decoy_scan_action_id"],
    )
    np.testing.assert_allclose(
        large_replay["signal_target_match_action_mask"],
        large_episode["signal_target_match_action_mask"],
    )
    np.testing.assert_array_equal(
        large_replay["signal_target_match_action_id"],
        large_episode["signal_target_match_action_id"],
    )
    np.testing.assert_allclose(
        large_replay["signal_target_opportunity_action_mask"],
        large_episode["signal_target_opportunity_action_mask"],
    )
    np.testing.assert_array_equal(
        large_replay["signal_target_opportunity_kind_id"],
        large_episode["signal_target_opportunity_kind_id"],
    )
    np.testing.assert_allclose(
        large_replay["signal_target_pursuit_action_mask"],
        large_episode["signal_target_pursuit_action_mask"],
    )
    np.testing.assert_array_equal(
        large_replay["signal_target_pursuit_action_id"],
        large_episode["signal_target_pursuit_action_id"],
    )
    np.testing.assert_allclose(
        large_replay["signal_rejected_target_drift_action_mask"],
        large_episode["signal_rejected_target_drift_action_mask"],
    )
    np.testing.assert_array_equal(
        large_replay["signal_rejected_target_drift_action_id"],
        large_episode["signal_rejected_target_drift_action_id"],
    )
    frontier_replay = _slice_recurrent_episode(frontier_episode, 0, 1)
    np.testing.assert_allclose(
        frontier_replay["signal_frontier_exploration_action_mask"],
        frontier_episode["signal_frontier_exploration_action_mask"],
    )
    np.testing.assert_array_equal(
        frontier_replay["signal_frontier_exploration_action_id"],
        frontier_episode["signal_frontier_exploration_action_id"],
    )
    sweep_replay = _slice_recurrent_episode(sweep_episode, 0, 1)
    np.testing.assert_allclose(
        sweep_replay["signal_evidence_sweep_action_mask"],
        sweep_episode["signal_evidence_sweep_action_mask"],
    )
    np.testing.assert_array_equal(
        sweep_replay["signal_evidence_sweep_action_id"],
        sweep_episode["signal_evidence_sweep_action_id"],
    )
    clue_replay = _slice_recurrent_episode(clue_episode, 0, 1)
    np.testing.assert_allclose(
        clue_replay["signal_clue_interact_action_mask"],
        clue_episode["signal_clue_interact_action_mask"],
    )
    np.testing.assert_array_equal(
        clue_replay["signal_clue_interact_action_id"],
        clue_episode["signal_clue_interact_action_id"],
    )
    bad_high_logits = torch.zeros((2, 8), dtype=torch.float32)
    bad_low_logits = torch.zeros((2, 8), dtype=torch.float32)
    bad_high_logits[0, large_env.ACTION_LEFT] = 4.0
    bad_low_logits[0, large_env.ACTION_LEFT] = -4.0
    assert _signal_decoy_drift_action_loss(
        bad_high_logits,
        torch.tensor([large_env.ACTION_LEFT, -1], dtype=torch.long),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    ).item() > _signal_decoy_drift_action_loss(
        bad_low_logits,
        torch.tensor([large_env.ACTION_LEFT, -1], dtype=torch.long),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    ).item()
    bad_scan_high_logits = torch.zeros((2, 8), dtype=torch.float32)
    bad_scan_low_logits = torch.zeros((2, 8), dtype=torch.float32)
    bad_scan_high_logits[0, large_env.ACTION_INTERACT] = 4.0
    bad_scan_low_logits[0, large_env.ACTION_INTERACT] = -4.0
    assert _signal_decoy_drift_action_loss(
        bad_scan_high_logits,
        torch.tensor([large_env.ACTION_INTERACT, -1], dtype=torch.long),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    ).item() > _signal_decoy_drift_action_loss(
        bad_scan_low_logits,
        torch.tensor([large_env.ACTION_INTERACT, -1], dtype=torch.long),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    ).item()
    target_match_high_logits = torch.zeros((2, 8), dtype=torch.float32)
    target_match_low_logits = torch.zeros((2, 8), dtype=torch.float32)
    target_match_high_logits[0, large_env.ACTION_RIGHT] = 4.0
    target_match_low_logits[0, large_env.ACTION_RIGHT] = -4.0
    assert _signal_target_match_action_loss(
        target_match_high_logits,
        torch.tensor([large_env.ACTION_RIGHT, -1], dtype=torch.long),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    ).item() < _signal_target_match_action_loss(
        target_match_low_logits,
        torch.tensor([large_env.ACTION_RIGHT, -1], dtype=torch.long),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    ).item()

    quadrant = 0
    if target[0] >= 4 and target[1] < 4:
        quadrant = 1
    elif target[0] < 4 and target[1] >= 4:
        quadrant = 2
    elif target[0] >= 4 and target[1] >= 4:
        quadrant = 3
    obs[0]["goal_hint"] = np.array([
        21, TILE_BEACON, target[0], target[1], 0,
        23, (target[0] + target[1]) % 2, quadrant, 8,
        -1, -1, -1, -1, -1, -1, -1,
    ], dtype=np.int16)
    assert _signal_observation_allows_target(obs[0], target, observed_map_size=8)
    assert _signal_target_pursuit_agents(env, obs, actions) == [0]

    local_grid = np.zeros((3, 3), dtype=np.int16)
    local_grid[1, 1] = TILE_TARGET
    if target[0] < env.map_size - 1:
        local_grid[1, 2] = TILE_TARGET
        rejected_direction = np.array([1.0, 1.0 / 7.0, 0.0, 1.0 / 7.0], dtype=np.float32)
    else:
        local_grid[1, 0] = TILE_TARGET
        rejected_direction = np.array([1.0, -1.0 / 7.0, 0.0, 1.0 / 7.0], dtype=np.float32)
    obs[0]["local_grid"] = local_grid
    obs[0]["self_pos"] = np.array(target, dtype=np.int16)
    match_features = _signal_visible_target_match_features(
        obs[0],
        obs[0]["self_pos"],
        observed_map_size=8,
    )
    assert match_features.shape == (14,)
    np.testing.assert_allclose(match_features[:4], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(match_features[4:8], rejected_direction)
    assert match_features[8] == 1.0
    assert match_features[9] == 0.0
    assert match_features[10] == pytest.approx(0.5)
    assert match_features[11] == pytest.approx(0.5)
    assert match_features[12] == pytest.approx(0.5)
    assert match_features[13] == 1.0


def test_recurrent_signal_rejected_target_interact_auxiliary_labels():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_TARGET
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _append_labeled_step,
        _clear_true_target_rejected_mask,
        _finalize_episode_sequence,
        _new_episode_sequence,
        _signal_center_rejected_target,
        _signal_rejected_target_interact_action_loss,
        _signal_rejected_target_interact_agents,
        _signal_rejected_target_interact_loss,
        _slice_recurrent_episode,
    )

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
    ))
    obs, _ = env.reset(seed=0)
    rejected_pos = tuple(int(v) for v in env.agent_positions[0])
    allowed_target = ((rejected_pos[0] + 1) % env.map_size, rejected_pos[1])
    if allowed_target == rejected_pos:
        allowed_target = ((rejected_pos[0] - 1) % env.map_size, rejected_pos[1])
    env.scenario_state.data["target"] = allowed_target

    local_grid = np.asarray(obs[0]["local_grid"]).copy()
    local_grid[:] = 0
    local_grid[local_grid.shape[0] // 2, local_grid.shape[1] // 2] = TILE_TARGET
    obs[0]["local_grid"] = local_grid
    obs[0]["self_pos"] = np.array(rejected_pos, dtype=np.int16)
    obs[0]["goal_hint"] = np.array(
        [26, allowed_target[0], allowed_target[1], -1, -1, -1, -1, -1],
        dtype=np.int16,
    )
    obs[1]["local_grid"] = np.zeros_like(obs[1]["local_grid"])
    obs[1]["goal_hint"] = np.full_like(obs[0]["goal_hint"], -1)

    actions = {
        0: {"action": env.ACTION_INTERACT, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    }
    cfg = RecurrentConfig(scenario="signal_hunt", map_size=8, agents=2)
    ep_data = _new_episode_sequence()

    assert _signal_center_rejected_target(obs[0], observed_map_size=8)
    assert _signal_rejected_target_interact_agents(env, obs, actions) == [0]
    _append_labeled_step(ep_data, obs, actions, env, cfg)
    episode = _finalize_episode_sequence(ep_data, env, cfg)
    np.testing.assert_allclose(
        episode["signal_rejected_target_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    replay = _slice_recurrent_episode(episode, 0, 1)
    np.testing.assert_allclose(replay["signal_rejected_target_mask"], episode["signal_rejected_target_mask"])

    logits = torch.zeros((2, 8), dtype=torch.float32)
    logits[0, env.ACTION_INTERACT] = 2.0
    logits[1, env.ACTION_INTERACT] = 5.0
    loss = _signal_rejected_target_interact_loss(
        logits,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    )

    assert loss.item() == pytest.approx(torch.nn.functional.softplus(torch.tensor(2.0)).item())

    high_bad_action_logits = torch.zeros((2, 8), dtype=torch.float32)
    low_bad_action_logits = torch.zeros((2, 8), dtype=torch.float32)
    high_bad_action_logits[0, env.ACTION_INTERACT] = 4.0
    low_bad_action_logits[0, env.ACTION_INTERACT] = -4.0
    high_bad_action_logits[0, env.ACTION_STAY] = 0.5
    low_bad_action_logits[0, env.ACTION_STAY] = 0.5
    assert _signal_rejected_target_interact_action_loss(
        high_bad_action_logits,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    ).item() > _signal_rejected_target_interact_action_loss(
        low_bad_action_logits,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    ).item()

    env.scenario_state.data["target"] = rejected_pos
    env.agent_positions[0] = rejected_pos
    assert _signal_center_rejected_target(obs[0], observed_map_size=8)
    assert _signal_rejected_target_interact_agents(env, obs, actions) == []
    np.testing.assert_allclose(
        _clear_true_target_rejected_mask(
            env,
            np.array([1.0, 0.0], dtype=np.float32),
        ),
        np.array([0.0, 0.0], dtype=np.float32),
    )
    true_target_ep = _new_episode_sequence()
    _append_labeled_step(true_target_ep, obs, actions, env, cfg)
    true_target_episode = _finalize_episode_sequence(true_target_ep, env, cfg)
    np.testing.assert_allclose(
        true_target_episode["signal_rejected_target_mask"],
        np.array([[0.0, 0.0]], dtype=np.float32),
    )


def test_recurrent_signal_sync_feedback_from_target_scan_event():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _feedback_matrix

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
    ))
    env.reset(seed=0)
    env.scenario_state.data["target"] = tuple(env.agent_positions[0])

    _obs, _rewards, done, _truncated, info = env.step({
        0: {"action": env.ACTION_INTERACT, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    })
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_feedback=True,
        obs_signal_sync_feedback=True,
    )
    feedback = _feedback_matrix(cfg, 2, info=info)

    assert done is False
    assert {event["event"] for event in info["events"][0]} == {"target_scan", "first_target_scan"}
    assert feedback.shape == (2, 16)
    np.testing.assert_allclose(feedback[0, 12:16], np.array([1.0, 0.0, 0.5, 0.0], dtype=np.float32))
    np.testing.assert_allclose(feedback[1, 12:16], np.array([0.0, 1.0, 0.5, 0.0], dtype=np.float32))

    joint_feedback = _feedback_matrix(
        cfg,
        2,
        info={
            "events": {
                0: [{"event": "target_scan"}, {"event": "joint_target_scan"}],
                1: [{"event": "target_scan"}, {"event": "joint_target_scan"}],
            }
        },
    )
    np.testing.assert_allclose(joint_feedback[:, 12:16], np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float32))


def test_recurrent_signal_scan_state_feedback_persists_until_window_expires():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _feedback_matrix

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
        scan_window=2,
    ))
    env.reset(seed=0)
    env.scenario_state.data["target"] = tuple(env.agent_positions[0])
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        scan_window=2,
        obs_feedback=True,
        obs_signal_sync_feedback=True,
        obs_signal_scan_state=True,
    )

    reset_feedback = _feedback_matrix(cfg, 2, env=env)
    assert reset_feedback.shape == (2, 20)
    np.testing.assert_allclose(reset_feedback[:, 16:20], np.zeros((2, 4), dtype=np.float32))

    _obs, _rewards, done, _truncated, info = env.step({
        0: {"action": env.ACTION_INTERACT, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    })
    assert done is False
    feedback = _feedback_matrix(cfg, 2, info=info, env=env)
    np.testing.assert_allclose(feedback[0, 16:20], np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(feedback[1, 16:20], np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))

    env.steps += 2
    edge_feedback = _feedback_matrix(cfg, 2, env=env)
    np.testing.assert_allclose(
        edge_feedback[0, 16:20],
        np.array([1.0, 0.0, 1.0 / 3.0, 0.0], dtype=np.float32),
    )

    env.steps += 1
    expired_feedback = _feedback_matrix(cfg, 2, env=env)
    np.testing.assert_allclose(expired_feedback[:, 16:20], np.zeros((2, 4), dtype=np.float32))

    tracked_feedback = _feedback_matrix(
        cfg,
        2,
        scan_state={"scan_log": {0: 1}, "scan_window": 2, "step": 2},
    )
    np.testing.assert_allclose(
        tracked_feedback[:, 16:20],
        np.array([
            [1.0, 0.0, 2.0 / 3.0, 0.0],
            [0.0, 1.0, 0.0, 2.0 / 3.0],
        ], dtype=np.float32),
    )


def test_recurrent_signal_negative_memory_feedback_tracks_decoy_scans():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_TARGET
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _feedback_matrix

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
    ))
    obs, _ = env.reset(seed=0)
    decoy_pos = tuple(env.agent_positions[0])
    true_target = tuple(env.agent_positions[1])
    env.grid[decoy_pos[1], decoy_pos[0]] = TILE_TARGET
    env.scenario_state.data["target"] = true_target
    env.scenario_state.data["decoys"] = [decoy_pos]

    obs, _rewards, done, _truncated, info = env.step({
        0: {"action": env.ACTION_INTERACT, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    })
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_feedback=True,
        obs_signal_sync_feedback=True,
        obs_signal_scan_state=True,
        obs_signal_negative_memory=True,
        obs_signal_negative_memory_window=4,
    )
    feedback = _feedback_matrix(cfg, 2, info=info, env=env, obs=obs)

    assert done is False
    assert env.scenario_state.data["negative_target_log"] == [
        {"agent_id": 0, "pos": decoy_pos, "step": 1}
    ]
    assert feedback.shape == (2, 28)
    np.testing.assert_allclose(
        feedback[0, 20:28],
        np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert feedback[1, 21] == 1.0
    assert feedback[1, 27] == pytest.approx(1.0)

    tracked_feedback = _feedback_matrix(
        cfg,
        2,
        scan_state={"negative_target_log": [{"agent_id": 0, "pos": decoy_pos, "step": 1}], "step": 3},
        obs=obs,
    )
    np.testing.assert_allclose(
        tracked_feedback[0, 20:28],
        np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 3.0 / 5.0], dtype=np.float32),
    )

    expired_feedback = _feedback_matrix(
        cfg,
        2,
        scan_state={"negative_target_log": [{"agent_id": 0, "pos": decoy_pos, "step": 1}], "step": 6},
        obs=obs,
    )
    np.testing.assert_allclose(expired_feedback[:, 20:28], np.zeros((2, 8), dtype=np.float32))


def test_recurrent_signal_sync_response_label_weighting():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_TARGET
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _append_labeled_step,
        _feedback_matrix,
        _new_episode_sequence,
        _signal_active_scan_response_action_label_mask,
        _signal_scan_bridge_action_label_mask,
        _signal_sync_response_agents,
        _signal_sync_response_action_label_mask,
        _signal_target_handoff_miss_agents,
    )

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
    ))
    obs, _ = env.reset(seed=0)
    env.grid[:, :] = 0
    env.scenario_state.data["target"] = tuple(env.agent_positions[1])
    obs[1]["goal_hint"] = np.array([
        26,
        env.agent_positions[1][0],
        env.agent_positions[1][1],
        -1, -1, -1, -1, -1,
    ], dtype=np.int16)
    obs[1]["action_mask"] = np.ones((8,), dtype=np.float32)
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_feedback=True,
        obs_signal_sync_feedback=True,
        obs_signal_scan_state=True,
        bc_signal_sync_response_weight=5.0,
    )
    feedback = _feedback_matrix(
        cfg,
        2,
        info={"events": {0: [{"event": "target_scan"}], 1: []}},
    )
    actions = {
        0: {"action": env.ACTION_STAY, "message_tokens": []},
        1: {"action": env.ACTION_INTERACT, "message_tokens": []},
    }
    ep_data = _new_episode_sequence()

    assert _signal_sync_response_agents(env, obs, actions, feedback) == [1]
    assert _signal_sync_response_agents(env, obs, actions, feedback, cfg=cfg) == [1]
    sync_mask, sync_action_id = _signal_sync_response_action_label_mask(
        env,
        obs,
        actions,
        feedback,
        cfg=cfg,
    )
    np.testing.assert_allclose(sync_mask, np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(sync_action_id, np.array([-1, env.ACTION_INTERACT], dtype=np.int64))
    model_actions = {
        0: {"action": env.ACTION_STAY, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    }
    assert _signal_target_handoff_miss_agents(
        env,
        obs,
        actions,
        model_actions,
        feedback,
        cfg=cfg,
    ) == [1]
    assert _signal_target_handoff_miss_agents(env, obs, actions, actions, feedback) == []
    _append_labeled_step(ep_data, obs, actions, env, cfg, feedback=feedback)
    assert ep_data["step_weights"] == [1.0, 5.0]
    assert ep_data["signal_sync_response_action_mask"] == [0.0, 1.0]
    assert ep_data["signal_sync_response_action_id"] == [-1, env.ACTION_INTERACT]

    x, y = env.agent_positions[1]
    if x < env.map_size - 1:
        target = (x + 1, y)
        action = env.ACTION_RIGHT
    else:
        target = (x - 1, y)
        action = env.ACTION_LEFT
    env.scenario_state.data["target"] = target
    obs[1]["goal_hint"] = np.array([26, target[0], target[1], -1, -1, -1, -1, -1], dtype=np.int16)
    actions[1] = {"action": action, "message_tokens": []}
    ep_data = _new_episode_sequence()

    assert _signal_sync_response_agents(env, obs, actions, feedback) == [1]
    sync_move_mask, sync_move_action_id = _signal_sync_response_action_label_mask(
        env,
        obs,
        actions,
        feedback,
        cfg=cfg,
    )
    np.testing.assert_allclose(sync_move_mask, np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(sync_move_action_id, np.array([-1, action], dtype=np.int64))
    _append_labeled_step(ep_data, obs, actions, env, cfg, feedback=feedback)
    assert ep_data["step_weights"] == [1.0, 5.0]
    assert ep_data["signal_sync_response_action_mask"] == [0.0, 1.0]
    assert ep_data["signal_sync_response_action_id"] == [-1, action]

    env.steps = 4
    env.scenario_state.data["scan_log"] = {0: 4}
    env.scenario_state.data["scan_window"] = 3
    active_scan_feedback = _feedback_matrix(
        cfg,
        2,
        info={},
        env=env,
        obs=obs,
    )
    active_scan_ep_data = _new_episode_sequence()

    assert _signal_sync_response_agents(env, obs, actions, active_scan_feedback) == []
    assert _signal_sync_response_agents(env, obs, actions, active_scan_feedback, cfg=cfg) == [1]
    active_sync_mask, active_sync_action_id = _signal_sync_response_action_label_mask(
        env,
        obs,
        actions,
        active_scan_feedback,
        cfg=cfg,
    )
    np.testing.assert_allclose(active_sync_mask, np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(active_sync_action_id, np.array([-1, action], dtype=np.int64))
    assert _signal_target_handoff_miss_agents(
        env,
        obs,
        actions,
        model_actions,
        active_scan_feedback,
        cfg=cfg,
    ) == [1]
    _append_labeled_step(active_scan_ep_data, obs, actions, env, cfg, feedback=active_scan_feedback)
    assert active_scan_ep_data["step_weights"] == [1.0, 5.0]
    assert active_scan_ep_data["signal_sync_response_action_mask"] == [0.0, 1.0]
    assert active_scan_ep_data["signal_sync_response_action_id"] == [-1, action]

    active_scan_response_cfg = RecurrentConfig(
        **{
            **vars(cfg),
            "bc_signal_active_scan_response_action_weight": 1.1,
            "bc_signal_active_scan_response_min_map_size": 8,
            "bc_signal_active_scan_response_max_agents": 1,
        }
    )
    active_response_mask, active_response_action_id = (
        _signal_active_scan_response_action_label_mask(
            env,
            obs,
            active_scan_response_cfg,
            active_scan_feedback,
        )
    )
    np.testing.assert_allclose(active_response_mask, np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(
        active_response_action_id,
        np.array([-1, action], dtype=np.int64),
    )
    no_exact_obs = {0: dict(obs[0]), 1: dict(obs[1])}
    no_exact_obs[1]["goal_hint"] = np.full((8,), -1, dtype=np.int16)
    no_exact_mask, no_exact_action_id = _signal_active_scan_response_action_label_mask(
        env,
        no_exact_obs,
        active_scan_response_cfg,
        active_scan_feedback,
    )
    np.testing.assert_allclose(no_exact_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(no_exact_action_id, np.array([-1, -1], dtype=np.int64))
    active_response_ep_data = _new_episode_sequence()
    _append_labeled_step(
        active_response_ep_data,
        obs,
        actions,
        env,
        active_scan_response_cfg,
        feedback=active_scan_feedback,
    )
    assert active_response_ep_data["signal_active_scan_response_action_mask"] == [0.0, 1.0]
    assert active_response_ep_data["signal_active_scan_response_action_id"] == [-1, action]

    bridge_cfg = RecurrentConfig(
        **{
            **vars(cfg),
            "bc_signal_scan_bridge_action_weight": 1.2,
            "bc_signal_scan_bridge_min_map_size": 8,
            "bc_signal_scan_bridge_remaining_threshold": 0.5,
            "bc_signal_scan_bridge_max_teammate_distance": 2,
            "eval_signal_exact_target_memory_steps": 8,
        }
    )
    bridge_target = (2, 2)
    bridge_teammate_pos = (3, 2)
    env.grid[:, :] = 0
    env.grid[bridge_target[1], bridge_target[0]] = TILE_TARGET
    env.agent_positions = [bridge_target, bridge_teammate_pos]
    env.steps = 2
    env.scenario_state.data["target"] = bridge_target
    env.scenario_state.data["scan_window"] = 3
    env.scenario_state.data["scan_log"] = {0: 0}
    env.scenario_state.data["scan_pos"] = {0: bridge_target}
    bridge_obs = env._build_observations()
    exact_hint = np.array(
        [26, bridge_target[0], bridge_target[1], -1, -1, -1, -1, -1],
        dtype=np.int16,
    )
    bridge_obs[0]["goal_hint"] = exact_hint.copy()
    bridge_obs[1]["goal_hint"] = exact_hint.copy()
    bridge_feedback = _feedback_matrix(
        bridge_cfg,
        2,
        info={},
        env=env,
        obs=bridge_obs,
    )
    bridge_scan_state = {
        "step": 2,
        "scan_window": 3,
        "scan_log": {0: 0},
        "scan_pos": {0: bridge_target},
    }

    bridge_mask, bridge_action_id = _signal_scan_bridge_action_label_mask(
        env,
        bridge_obs,
        bridge_cfg,
        bridge_feedback,
        scan_state=bridge_scan_state,
    )
    np.testing.assert_allclose(bridge_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        bridge_action_id,
        np.array([env.ACTION_INTERACT, -1], dtype=np.int64),
    )

    far_bridge_cfg = RecurrentConfig(
        **{
            **vars(bridge_cfg),
            "bc_signal_scan_bridge_max_teammate_distance": 0,
        }
    )
    far_bridge_mask, far_bridge_action_id = _signal_scan_bridge_action_label_mask(
        env,
        bridge_obs,
        far_bridge_cfg,
        bridge_feedback,
        scan_state=bridge_scan_state,
    )
    np.testing.assert_allclose(far_bridge_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(far_bridge_action_id, np.array([-1, -1], dtype=np.int64))

    bridge_ep_data = _new_episode_sequence()
    _append_labeled_step(
        bridge_ep_data,
        bridge_obs,
        {
            0: {"action": env.ACTION_INTERACT, "message_tokens": []},
            1: {"action": env.ACTION_STAY, "message_tokens": []},
        },
        env,
        bridge_cfg,
        feedback=bridge_feedback,
        scan_state=bridge_scan_state,
    )
    assert bridge_ep_data["signal_scan_bridge_action_mask"] == [1.0, 0.0]
    assert bridge_ep_data["signal_scan_bridge_action_id"] == [env.ACTION_INTERACT, -1]


def test_recurrent_build_env_passes_signal_shaping_config():
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _build_env

    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        signal_decoy_count=3,
        decoy_penalty=2.5,
        scan_window=4,
        signal_shaping=True,
        signal_shaping_scale=0.05,
        signal_scan_bonus=0.1,
        signal_joint_scan_bonus=2.0,
        signal_colocation_bonus=0.3,
        signal_colocation_radius=3,
        signal_comm_utility=0.2,
        signal_target_visit_bonus=0.4,
        signal_decoy_visit_penalty=0.5,
        signal_unique_target_scan_bonus=0.6,
        comm_token_limit=4,
        comm_vocab_size=8,
        comm_max_messages=5,
        comm_len_cost=0.02,
        comm_cost=0.03,
    )

    env = _build_env(cfg)

    assert env.config.signal_decoy_count == 3
    assert env.config.decoy_penalty == 2.5
    assert env.config.scan_window == 4
    assert env.config.signal_shaping is True
    assert env.config.signal_shaping_scale == 0.05
    assert env.config.signal_scan_bonus == 0.1
    assert env.config.signal_joint_scan_bonus == 2.0
    assert env.config.signal_colocation_bonus == 0.3
    assert env.config.signal_colocation_radius == 3
    assert env.config.signal_comm_utility == 0.2
    assert env.config.signal_target_visit_bonus == 0.4
    assert env.config.signal_decoy_visit_penalty == 0.5
    assert env.config.signal_unique_target_scan_bonus == 0.6
    assert env.config.max_messages == 5
    assert env.config.comm_len_cost == 0.02
    assert env.config.comm_cost == 0.03


def test_recurrent_eval_score_prefers_fewer_decoys():
    from syncorsink.train.recurrent_bc_rl import _recurrent_eval_score

    same_success_many_decoys = {
        "success_rate": 0.3,
        "avg_return": 10.0,
        "avg_steps": 20.0,
        "signal": {"avg_decoy_scans": 5.0},
    }
    same_success_few_decoys = {
        "success_rate": 0.3,
        "avg_return": 0.0,
        "avg_steps": 60.0,
        "signal": {"avg_decoy_scans": 1.0},
    }

    assert _recurrent_eval_score(same_success_few_decoys) > _recurrent_eval_score(same_success_many_decoys)

    same_success_decoys_many_redundant = {
        "success_rate": 0.3,
        "avg_return": 10.0,
        "avg_steps": 20.0,
        "signal": {"avg_decoy_scans": 1.0, "avg_redundant_target_scans": 8.0},
    }
    same_success_decoys_few_redundant = {
        "success_rate": 0.3,
        "avg_return": 0.0,
        "avg_steps": 60.0,
        "signal": {"avg_decoy_scans": 1.0, "avg_redundant_target_scans": 1.0},
    }

    assert (
        _recurrent_eval_score(same_success_decoys_few_redundant)
        > _recurrent_eval_score(same_success_decoys_many_redundant)
    )


def test_recurrent_signal_eval_summary_includes_target_failure_modes():
    from syncorsink.train.recurrent_bc_rl import _summarize_signal_eval_rows

    summary = _summarize_signal_eval_rows([
        {
            "target_scans": 0.0,
            "true_target_visits": 3.0,
            "true_target_unscanned_visits": 3.0,
            "true_target_unscanned_obs_candidate": 1.0,
            "true_target_unscanned_obs_compatible": 1.0,
            "target_first_scan_misses": 2.0,
            "target_first_scan_miss_obs_candidate": 1.0,
            "target_first_scan_miss_obs_no_info": 1.0,
            "reached_any_target": 1.0,
            "reached_true_target": 1.0,
            "no_target_reached": 0.0,
            "true_target_reached_without_scan": 1.0,
        },
        {
            "target_scans": 2.0,
            "decoy_target_visits": 1.0,
            "wrong_target_scans": 1.0,
            "reached_any_target": 1.0,
            "reached_decoy_target": 1.0,
            "wrong_target_scanned": 1.0,
        },
        {
            "no_target_reached": 1.0,
        },
    ])

    assert summary["avg_target_scans"] == pytest.approx(2.0 / 3.0)
    assert summary["avg_true_target_visits"] == pytest.approx(1.0)
    assert summary["avg_true_target_unscanned_visits"] == pytest.approx(1.0)
    assert summary["avg_true_target_unscanned_obs_candidate"] == pytest.approx(1.0 / 3.0)
    assert summary["avg_true_target_unscanned_obs_compatible"] == pytest.approx(1.0 / 3.0)
    assert summary["avg_target_first_scan_misses"] == pytest.approx(2.0 / 3.0)
    assert summary["avg_target_first_scan_miss_obs_candidate"] == pytest.approx(1.0 / 3.0)
    assert summary["avg_target_first_scan_miss_obs_no_info"] == pytest.approx(1.0 / 3.0)
    assert summary["avg_reached_any_target"] == pytest.approx(2.0 / 3.0)
    assert summary["avg_no_target_reached"] == pytest.approx(1.0 / 3.0)
    assert summary["avg_true_target_reached_without_scan"] == pytest.approx(1.0 / 3.0)
    assert summary["avg_wrong_target_scanned"] == pytest.approx(1.0 / 3.0)


def test_recurrent_dagger_best_round_uses_eval_score(monkeypatch):
    import syncorsink.train.recurrent_bc_rl as recurrent
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig

    train_calls = {"count": 0}
    eval_calls = {"count": 0}
    seen_seed_counts = []

    def fake_train_recurrent_bc(cfg, episodes, device, model=None):
        cfg.eval_send_threshold = 0.25 + float(train_calls["count"])
        round_model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            round_model.weight.fill_(float(train_calls["count"]))
        train_calls["count"] += 1
        return round_model

    eval_results = [
        {
            "episodes": 2,
            "success_rate": 0.25,
            "avg_return": 1.0,
            "avg_steps": 20.0,
            "signal": {"avg_decoy_scans": 20.0},
        },
        {
            "episodes": 2,
            "success_rate": 0.25,
            "avg_return": -5.0,
            "avg_steps": 60.0,
            "signal": {"avg_decoy_scans": 4.0},
        },
    ]

    def fake_evaluate_recurrent_policy_multi_seed(
        cfg,
        model,
        device,
        *,
        seed_count,
        seed_list="",
        seed_list_field_name="rl_eval_seed_list",
    ):
        del cfg, model, device
        assert seed_list == ""
        assert seed_list_field_name == "eval_seed_list"
        seen_seed_counts.append(seed_count)
        result = eval_results[eval_calls["count"]]
        eval_calls["count"] += 1
        return result

    def fake_collect_recurrent_dagger_episodes(cfg, model, device, round_idx):
        episode = {
            "obs": np.zeros((1, 1, 1), dtype=np.float32),
            "source": "dagger",
        }
        return [episode], {"episodes": 1}

    monkeypatch.setattr(recurrent, "train_recurrent_bc", fake_train_recurrent_bc)
    monkeypatch.setattr(recurrent, "evaluate_recurrent_policy_multi_seed", fake_evaluate_recurrent_policy_multi_seed)
    monkeypatch.setattr(recurrent, "collect_recurrent_dagger_episodes", fake_collect_recurrent_dagger_episodes)

    initial_episode = {
        "obs": np.zeros((1, 1, 1), dtype=np.float32),
        "source": "expert",
    }
    cfg = RecurrentConfig(dagger_rounds=1, eval_seed_count=3)
    model, history, all_episodes, best_round = recurrent.train_recurrent_bc_dagger(
        cfg,
        [initial_episode],
        torch.device("cpu"),
    )

    assert seen_seed_counts == [3, 3]
    assert best_round["round"] == 1
    assert best_round["eval_send_threshold"] == pytest.approx(1.25)
    assert cfg.eval_send_threshold == pytest.approx(1.25)
    assert history[1]["eval_score"] > history[0]["eval_score"]
    assert len(all_episodes) == 2
    assert float(next(model.parameters()).item()) == pytest.approx(1.0)


def test_recurrent_dagger_restore_best_can_keep_latest_model(monkeypatch):
    import syncorsink.train.recurrent_bc_rl as recurrent
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig

    def fake_train_recurrent_bc(cfg, episodes, device, model=None):
        del device, model
        round_model = torch.nn.Linear(1, 1, bias=False)
        weight = float(len(episodes) - 1)
        cfg.eval_send_threshold = 0.25 + weight
        with torch.no_grad():
            round_model.weight.fill_(weight)
        return round_model

    def fake_evaluate_recurrent_policy_multi_seed(
        cfg,
        model,
        device,
        *,
        seed_count,
        seed_list="",
        seed_list_field_name="rl_eval_seed_list",
    ):
        del cfg, device, seed_count, seed_list, seed_list_field_name
        weight = float(next(model.parameters()).item())
        return {
            "episodes": 1,
            "success_rate": 1.0 if weight == 0.0 else 0.0,
            "avg_return": 10.0 if weight == 0.0 else -10.0,
            "avg_steps": 10.0,
        }

    def fake_collect_recurrent_dagger_episodes(cfg, model, device, round_idx):
        del cfg, model, device, round_idx
        episode = {
            "obs": np.zeros((1, 1, 1), dtype=np.float32),
            "source": "dagger",
        }
        return [episode], {"episodes": 1}

    monkeypatch.setattr(recurrent, "train_recurrent_bc", fake_train_recurrent_bc)
    monkeypatch.setattr(recurrent, "evaluate_recurrent_policy_multi_seed", fake_evaluate_recurrent_policy_multi_seed)
    monkeypatch.setattr(recurrent, "collect_recurrent_dagger_episodes", fake_collect_recurrent_dagger_episodes)

    initial_episode = {
        "obs": np.zeros((1, 1, 1), dtype=np.float32),
        "source": "expert",
    }

    restore_cfg = RecurrentConfig(dagger_rounds=1, dagger_restore_best=True)
    model, _history, _all_episodes, best_round = recurrent.train_recurrent_bc_dagger(
        restore_cfg,
        [initial_episode],
        torch.device("cpu"),
    )
    assert best_round["round"] == 0
    assert float(next(model.parameters()).item()) == pytest.approx(0.0)
    assert restore_cfg.eval_send_threshold == pytest.approx(0.25)

    latest_cfg = RecurrentConfig(dagger_rounds=1, dagger_restore_best=False)
    model, _history, _all_episodes, best_round = recurrent.train_recurrent_bc_dagger(
        latest_cfg,
        [initial_episode],
        torch.device("cpu"),
    )
    assert best_round["round"] == 0
    assert float(next(model.parameters()).item()) == pytest.approx(1.0)
    assert latest_cfg.eval_send_threshold == pytest.approx(1.25)


def test_recurrent_dagger_can_start_from_initial_model(monkeypatch):
    import syncorsink.train.recurrent_bc_rl as recurrent
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig

    initial_model = torch.nn.Linear(1, 1, bias=False)
    seen_start_models = []

    def fake_train_recurrent_bc(cfg, episodes, device, model=None):
        del cfg, episodes, device
        seen_start_models.append(model)
        if model is not None:
            return model
        return torch.nn.Linear(1, 1, bias=False)

    def fake_evaluate_recurrent_policy_multi_seed(
        cfg,
        model,
        device,
        *,
        seed_count,
        seed_list="",
        seed_list_field_name="rl_eval_seed_list",
    ):
        del cfg, model, device
        assert seed_list == ""
        assert seed_list_field_name == "eval_seed_list"
        assert seed_count == 2
        return {
            "episodes": 2,
            "success_rate": 0.5,
            "avg_return": 1.0,
            "avg_steps": 20.0,
            "signal": {"avg_decoy_scans": 1.0},
        }

    def fake_collect_recurrent_dagger_episodes(cfg, model, device, round_idx):
        del cfg, model, device, round_idx
        return [{"obs": np.zeros((1, 1, 1), dtype=np.float32), "source": "dagger"}], {"episodes": 1}

    monkeypatch.setattr(recurrent, "train_recurrent_bc", fake_train_recurrent_bc)
    monkeypatch.setattr(recurrent, "evaluate_recurrent_policy_multi_seed", fake_evaluate_recurrent_policy_multi_seed)
    monkeypatch.setattr(recurrent, "collect_recurrent_dagger_episodes", fake_collect_recurrent_dagger_episodes)

    initial_episode = {
        "obs": np.zeros((1, 1, 1), dtype=np.float32),
        "source": "expert",
    }
    model, history, _all_episodes, _best_round = recurrent.train_recurrent_bc_dagger(
        RecurrentConfig(dagger_rounds=1, dagger_retrain_from_scratch=False, eval_seed_count=2),
        [initial_episode],
        torch.device("cpu"),
        initial_model=initial_model,
    )

    assert seen_start_models[0] is initial_model
    assert seen_start_models[1] is initial_model
    assert model is initial_model
    assert history[0]["started_from_recurrent_init"] is True
    assert history[0]["retrain_from_scratch"] is False


def test_recurrent_dagger_early_stop_skips_extra_collection(monkeypatch):
    import syncorsink.train.recurrent_bc_rl as recurrent
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig

    train_calls = {"count": 0}
    eval_calls = {"count": 0}
    collect_calls = {"count": 0}
    seen_seed_counts = []

    def fake_train_recurrent_bc(cfg, episodes, device, model=None):
        cfg.eval_send_threshold = 0.25 + float(train_calls["count"])
        round_model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            round_model.weight.fill_(float(train_calls["count"]))
        train_calls["count"] += 1
        return round_model

    eval_results = [
        {
            "episodes": 2,
            "success_rate": 0.5,
            "avg_return": 1.0,
            "avg_steps": 20.0,
            "signal": {"avg_decoy_scans": 1.0},
        },
        {
            "episodes": 2,
            "success_rate": 0.25,
            "avg_return": 0.0,
            "avg_steps": 30.0,
            "signal": {"avg_decoy_scans": 0.0},
        },
        {
            "episodes": 2,
            "success_rate": 0.75,
            "avg_return": 2.0,
            "avg_steps": 10.0,
            "signal": {"avg_decoy_scans": 0.0},
        },
    ]

    def fake_evaluate_recurrent_policy_multi_seed(
        cfg,
        model,
        device,
        *,
        seed_count,
        seed_list="",
        seed_list_field_name="rl_eval_seed_list",
    ):
        del cfg, model, device
        assert seed_list == ""
        assert seed_list_field_name == "eval_seed_list"
        seen_seed_counts.append(seed_count)
        result = eval_results[eval_calls["count"]]
        eval_calls["count"] += 1
        return result

    def fake_collect_recurrent_dagger_episodes(cfg, model, device, round_idx):
        collect_calls["count"] += 1
        episode = {
            "obs": np.zeros((1, 1, 1), dtype=np.float32),
            "source": "dagger",
        }
        return [episode], {"episodes": 1}

    monkeypatch.setattr(recurrent, "train_recurrent_bc", fake_train_recurrent_bc)
    monkeypatch.setattr(recurrent, "evaluate_recurrent_policy_multi_seed", fake_evaluate_recurrent_policy_multi_seed)
    monkeypatch.setattr(recurrent, "collect_recurrent_dagger_episodes", fake_collect_recurrent_dagger_episodes)

    initial_episode = {
        "obs": np.zeros((1, 1, 1), dtype=np.float32),
        "source": "expert",
    }
    cfg = RecurrentConfig(dagger_rounds=3, dagger_early_stop_patience=1, eval_seed_count=4)
    model, history, all_episodes, best_round = recurrent.train_recurrent_bc_dagger(
        cfg,
        [initial_episode],
        torch.device("cpu"),
    )

    assert seen_seed_counts == [4, 4]
    assert len(history) == 2
    assert collect_calls["count"] == 1
    assert eval_calls["count"] == 2
    assert train_calls["count"] == 2
    assert history[1]["early_stop"] is True
    assert history[1]["non_improving_rounds"] == 1
    assert best_round["round"] == 0
    assert best_round["eval_send_threshold"] == pytest.approx(0.25)
    assert cfg.eval_send_threshold == pytest.approx(0.25)
    assert len(all_episodes) == 2
    assert float(next(model.parameters()).item()) == pytest.approx(0.0)


def test_recurrent_bc_wandb_logs_learning_rate():
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, train_recurrent_bc

    class FakeWandbRun:
        def __init__(self):
            self.payloads = []

        def log(self, payload):
            self.payloads.append(dict(payload))

    episode = {
        "obs": np.zeros((1, 1, 8), dtype=np.float32),
        "actions": np.zeros((1, 1), dtype=np.int64),
        "msg_tokens": np.zeros((1, 1, 1), dtype=np.int64),
        "msg_lens": np.zeros((1, 1), dtype=np.int64),
    }
    run = FakeWandbRun()

    train_recurrent_bc(
        RecurrentConfig(
            bc_epochs=1,
            bc_lr=0.123,
            bc_seq_len=1,
            hidden_dim=8,
            comm=False,
        ),
        [episode],
        torch.device("cpu"),
        wandb_run=run,
    )

    assert run.payloads
    assert run.payloads[0]["bc/lr"] == pytest.approx(0.123)


def test_recurrent_feedback_obs_keeps_action_mask_tail():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.train.mappo import action_mask_from_flat_obs
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _build_recurrent_obs_batch,
        _flatten_recurrent_obs,
    )

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
        obs_exploration_memory=True,
    ))
    obs, _ = env.reset(seed=0)
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_exploration_memory=True,
        obs_feedback=True,
    )
    feedback = np.ones((env.num_agents, 12), dtype=np.float32)

    flat = _flatten_recurrent_obs(obs[0], cfg, feedback=feedback[0])
    batch = _build_recurrent_obs_batch(obs, env.num_agents, cfg, feedback=feedback)

    expected_mask = torch.tensor(obs[0]["action_mask"], dtype=torch.float32)
    assert torch.equal(action_mask_from_flat_obs(torch.tensor(flat).unsqueeze(0))[0], expected_mask)
    assert torch.equal(action_mask_from_flat_obs(torch.tensor(batch))[0], expected_mask)
    assert flat.shape[0] == batch.shape[1]


def test_recurrent_obs_normalize_tokens_preserves_action_mask():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.train.mappo import action_mask_from_flat_obs
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _flatten_recurrent_obs,
        _normalize_recurrent_obs_agent,
    )

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
        comm_token_limit=4,
        token_vocab_size=32,
        max_messages=2,
    ))
    obs, _ = env.reset(seed=0)
    obs_agent = dict(obs[0])
    obs_agent["messages_tokens"] = np.array([[26, 4, 2, -1], [-1, -1, -1, -1]], dtype=np.int16)
    obs_agent["message_from"] = np.array([1, -1], dtype=np.int16)
    obs_agent["goal_hint"] = np.array([21, 7, 3, 4, 2] + [-1] * 11, dtype=np.int16)
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=32,
        comm_max_messages=2,
        obs_feedback=True,
        obs_normalize_tokens=True,
    )

    normalized = _normalize_recurrent_obs_agent(obs_agent, cfg)
    flat = _flatten_recurrent_obs(obs_agent, cfg, feedback=np.ones((12,), dtype=np.float32))

    assert normalized["messages_tokens"][0, 0] == pytest.approx(26 / 31)
    assert normalized["messages_tokens"][0, 3] == -1.0
    assert normalized["message_from"][0] == 1.0
    assert normalized["message_from"][1] == -1.0
    assert normalized["goal_hint"][0] == pytest.approx(21 / 31)
    assert normalized["goal_hint"][-1] == -1.0
    expected_mask = torch.tensor(obs_agent["action_mask"], dtype=torch.float32)
    assert torch.equal(action_mask_from_flat_obs(torch.tensor(flat).unsqueeze(0))[0], expected_mask)


def test_recurrent_egocentric_memory_flatten_cross_map_size():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.train.mappo import action_mask_from_flat_obs
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _flatten_recurrent_obs,
        _normalize_recurrent_obs_agent,
        _project_recurrent_memory,
    )

    env8 = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
        obs_exploration_memory=True,
        obs_exploration_age=True,
    ))
    env16 = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=16,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
        obs_exploration_memory=True,
        obs_exploration_age=True,
    ))
    obs8, _ = env8.reset(seed=0)
    obs16, _ = env16.reset(seed=0)
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_exploration_memory=True,
        obs_exploration_age=True,
        obs_feedback=True,
        obs_normalize_tokens=True,
        obs_memory_mode="egocentric",
        obs_memory_radius=2,
    )

    projected16 = _project_recurrent_memory(obs16[0], cfg)
    normalized16 = _normalize_recurrent_obs_agent(obs16[0], cfg)
    flat8 = _flatten_recurrent_obs(obs8[0], cfg, feedback=np.zeros((12,), dtype=np.float32))
    flat16 = _flatten_recurrent_obs(obs16[0], cfg, feedback=np.zeros((12,), dtype=np.float32))

    assert projected16["explored_mask"].shape == (5, 5)
    assert projected16["explored_age"].shape == (5, 5)
    assert normalized16["self_pos"].max() <= 1.0
    assert flat8.shape == flat16.shape
    expected_mask = torch.tensor(obs16[0]["action_mask"], dtype=torch.float32)
    assert torch.equal(action_mask_from_flat_obs(torch.tensor(flat16).unsqueeze(0))[0], expected_mask)


def test_recurrent_navigation_features_are_fixed_width_and_mask_safe():
    from syncorsink.envs.maps import TILE_CLUE, TILE_TARGET
    from syncorsink.train.mappo import action_mask_from_flat_obs
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _flatten_recurrent_obs,
        _navigation_features,
    )

    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 3] = TILE_TARGET
    local_grid[1, 2] = TILE_CLUE
    explored8 = np.zeros((8, 8), dtype=np.int8)
    explored8[3, 3] = 1
    obs_agent = {
        "local_grid": local_grid,
        "self_pos": np.array([3, 3], dtype=np.int16),
        "explored_mask": explored8,
        "action_mask": np.array([1, 1, 0, 1, 1, 0, 0, 0], dtype=np.float32),
    }
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_exploration_memory=True,
        obs_feedback=True,
        obs_navigation_features=True,
        obs_memory_mode="egocentric",
        obs_memory_radius=1,
    )

    features = _navigation_features(obs_agent, cfg, observed_map_size=8)
    flat8 = _flatten_recurrent_obs(obs_agent, cfg, feedback=np.zeros((12,), dtype=np.float32))
    obs_agent_16 = dict(obs_agent)
    explored16 = np.zeros((16, 16), dtype=np.int8)
    explored16[3, 3] = 1
    obs_agent_16["explored_mask"] = explored16
    flat16 = _flatten_recurrent_obs(obs_agent_16, cfg, feedback=np.zeros((12,), dtype=np.float32))

    assert features.shape == (25,)
    assert features[0] == 1.0  # visible clue group present
    assert features[2] < 0.0
    assert features[4] == 1.0  # visible target group present
    assert features[5] > 0.0
    assert features[-5] == 1.0  # frontier group present
    assert features[-1] > 0.0
    assert flat8.shape == flat16.shape
    expected_mask = torch.tensor(obs_agent["action_mask"], dtype=torch.float32)
    assert torch.equal(action_mask_from_flat_obs(torch.tensor(flat8).unsqueeze(0))[0], expected_mask)


def test_recurrent_pipeline_features_decode_message_and_keep_mask_safe():
    from syncorsink.envs.maps import TILE_RESOURCE, TILE_STATION
    from syncorsink.train.mappo import action_mask_from_flat_obs
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _flatten_recurrent_obs,
        _pipeline_coordination_features,
    )

    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 2] = TILE_STATION
    local_grid[2, 3] = TILE_RESOURCE
    local_resource_types = np.zeros((5, 5), dtype=np.int16)
    local_resource_types[2, 3] = 2
    obs_agent = {
        "local_grid": local_grid,
        "inventory": np.array([2], dtype=np.int16),
        "self_pos": np.array([4, 3], dtype=np.int16),
        "local_resource_types": local_resource_types,
        "messages_tokens": np.array(
            [[12, 0, 4, 3, 1, 2, -1, -1]],
            dtype=np.int16,
        ),
        "goal_hint": np.full((16,), -1, dtype=np.int16),
        "action_mask": np.array([1, 1, 0, 1, 1, 1, 0, 0], dtype=np.float32),
    }
    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=3,
        obs_pipeline_features=True,
    )

    features = _pipeline_coordination_features(obs_agent, cfg, observed_map_size=8)
    flat = _flatten_recurrent_obs(obs_agent, cfg)

    assert features.shape == (38,)
    assert features[0] == 1.0  # decoded a plan
    assert features[1] == 1.0  # from message
    assert features[12:16].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert features[16:20].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert features[20] == 1.0  # carrying a resource
    assert features[21] == 1.0  # held resource is needed
    assert features[25] == 1.0  # at active station
    assert features[28] == 1.0  # should deliver now
    assert features[29] == 0.0  # not an unsafe station interact
    assert features[31:35].tolist() == [1.0, 0.0, 0.0, 0.0]  # held target is here
    assert features[35] == 0.0  # held needed, but not at the wrong station
    assert features[36] == 0.0  # held resource belongs to the decoded plan
    assert features[37] == 0.0  # center resource is not an unneeded pickup target
    expected_mask = torch.tensor(obs_agent["action_mask"], dtype=torch.float32)
    assert torch.equal(action_mask_from_flat_obs(torch.tensor(flat).unsqueeze(0))[0], expected_mask)


def test_recurrent_pipeline_progress_features_track_stage_state():
    from syncorsink.envs.maps import TILE_STATION
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _PIPELINE_PROGRESS_FEATURE_SIZE,
        _pipeline_coordination_features,
    )

    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 2] = TILE_STATION
    hint = np.full((64,), -1, dtype=np.int16)
    hint[:9] = np.array([0, 1, 1, 1, 2, -1, 0, -1, 0], dtype=np.int16)
    hint[9:18] = np.array([1, 5, 6, 2, 2, 3, 1, 0, 1], dtype=np.int16)
    obs_agent = {
        "local_grid": local_grid,
        "inventory": np.array([0], dtype=np.int16),
        "self_pos": np.array([5, 6], dtype=np.int16),
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "goal_hint": hint,
    }
    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=3,
        obs_pipeline_progress_features=True,
    )

    features = _pipeline_coordination_features(
        obs_agent,
        cfg,
        observed_map_size=8,
        pipeline_state={
            "completed_stages": {0},
            "delivered_counts": {1: 1},
            "delivered_resources": {1: [2]},
            "sync_wait_stages": {1},
            "sync_wait_stations": {1: (5, 6)},
        },
    )

    assert features.shape == (_PIPELINE_PROGRESS_FEATURE_SIZE,)
    assert features[:18].tolist() == pytest.approx([
        1.0,
        1.0,
        0.5,
        1.0 / 7.0,
        1.0 / 7.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ])
    assert features[18:22].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert features[22:26].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_pipeline_goal_hint_keeps_multiple_stage_chunks():
    from syncorsink.envs.base import SyncOrSinkConfig, SyncOrSinkEnv

    env = SyncOrSinkEnv(
        SyncOrSinkConfig(
            scenario="pipeline_assembly",
            map_size=16,
            num_agents=3,
            fov_preset="easy",
            pipeline_stage_count=6,
            pipeline_required_per_stage_min=1,
            pipeline_required_per_stage_max=1,
            pipeline_sync_probability=0.0,
            pipeline_dependency_probability=0.0,
            goal_hint_size=64,
        )
    )
    obs, _ = env.reset(seed=0)

    assert env.observation_space.spaces["goal_hint"].shape == (64,)
    assert obs[0]["goal_hint"].shape == (64,)
    first_chunk = obs[0]["goal_hint"][:9].tolist()
    second_chunk = obs[0]["goal_hint"][9:18].tolist()
    assert first_chunk[0] == 0
    assert second_chunk[0] == 3
    assert second_chunk[1] >= 0
    assert second_chunk[2] >= 0
    assert second_chunk[4] > 0


def test_pipeline_trusted_plan_for_label_matches_later_hint_chunk():
    from syncorsink.train.recurrent_bc_rl import (
        _pipeline_plans_from_goal_hint,
        _pipeline_trusted_plan_for_label,
    )

    hint = np.full((64,), -1, dtype=np.int16)
    hint[:9] = np.array([0, 1, 1, 1, 2, -1, 0, -1, 0], dtype=np.int16)
    hint[9:18] = np.array([3, 5, 6, 1, 4, -1, 1, 0, 0], dtype=np.int16)
    obs_agent = {
        "goal_hint": hint,
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
    }
    active_stage = {
        "stage": 3,
        "station": (5, 6),
        "required": [4],
        "delivered": [],
        "deps": [0],
        "done": False,
    }

    plans = _pipeline_plans_from_goal_hint(hint, observed_map_size=8)
    matched = _pipeline_trusted_plan_for_label(obs_agent, observed_map_size=8, stage=active_stage)

    assert [plan["stage"] for plan in plans] == [0, 3]
    assert matched is not None
    assert matched["stage"] == 3
    assert matched["station"] == (5, 6)
    assert matched["required"] == [4]
    assert matched["deps"] == [0]


def test_pipeline_goal_hint_plan_selection_skips_completed_stage():
    from syncorsink.train.recurrent_bc_rl import _pipeline_plan_from_goal_hint

    hint = np.full((64,), -1, dtype=np.int16)
    hint[:9] = np.array([0, 1, 1, 1, 2, -1, 0, -1, 0], dtype=np.int16)
    hint[9:18] = np.array([3, 5, 6, 1, 4, -1, 1, 0, 0], dtype=np.int16)

    first = _pipeline_plan_from_goal_hint(hint, observed_map_size=8)
    active = _pipeline_plan_from_goal_hint(
        hint,
        observed_map_size=8,
        completed_stages={0},
    )
    blocked_preferred = _pipeline_plan_from_goal_hint(
        hint,
        observed_map_size=8,
        completed_stages=set(),
        preferred_resource=4,
    )
    completed_preferred = _pipeline_plan_from_goal_hint(
        hint,
        observed_map_size=8,
        completed_stages={0},
        preferred_resource=4,
    )

    assert first is not None and first["stage"] == 0
    assert active is not None and active["stage"] == 3
    assert blocked_preferred is not None and blocked_preferred["stage"] == 0
    assert completed_preferred is not None and completed_preferred["stage"] == 3


def test_pipeline_navigation_assist_sync_interact_uses_progress_state():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_STATION
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _pipeline_local_assist_action

    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 2] = TILE_STATION
    hint = np.full((64,), -1, dtype=np.int16)
    hint[:9] = np.array([0, 2, 2, 1, 3, -1, 0, -1, 1], dtype=np.int16)
    obs_agent = {
        "local_grid": local_grid,
        "inventory": np.array([0], dtype=np.int16),
        "self_pos": np.array([2, 2], dtype=np.int16),
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "goal_hint": hint,
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    cfg = RecurrentConfig(scenario="pipeline_assembly", map_size=8, agents=3)

    without_progress = _pipeline_local_assist_action(cfg, obs_agent)
    with_progress = _pipeline_local_assist_action(
        cfg,
        obs_agent,
        pipeline_state={
            "completed_stages": set(),
            "delivered_counts": {0: 1},
            "sync_wait_stages": {0},
        },
    )
    with_stage = _pipeline_local_assist_action(
        cfg,
        obs_agent,
        stage={
            "stage": 0,
            "station": (2, 2),
            "required": [3],
            "delivered": [3],
            "deps": [],
            "sync": True,
            "done": False,
        },
    )
    off_station_obs = {
        **obs_agent,
        "self_pos": np.array([1, 2], dtype=np.int16),
        "local_grid": np.zeros((5, 5), dtype=np.int16),
    }
    sync_rendezvous = _pipeline_local_assist_action(
        cfg,
        off_station_obs,
        pipeline_state={
            "completed_stages": set(),
            "delivered_counts": {0: 1},
            "delivered_resources": {0: [3]},
            "sync_wait_stages": set(),
        },
    )
    no_hint_sync = _pipeline_local_assist_action(
        cfg,
        {
            **off_station_obs,
            "goal_hint": np.full((64,), -1, dtype=np.int16),
        },
        pipeline_state={
            "completed_stages": set(),
            "delivered_counts": {},
            "delivered_resources": {},
            "sync_wait_stages": {7},
            "sync_wait_stations": {7: (2, 2)},
        },
    )

    assert without_progress is None
    assert with_progress == SyncOrSinkEnv.ACTION_INTERACT
    assert with_stage == SyncOrSinkEnv.ACTION_INTERACT
    assert sync_rendezvous == SyncOrSinkEnv.ACTION_RIGHT
    assert no_hint_sync == SyncOrSinkEnv.ACTION_RIGHT


def test_pipeline_state_tracks_delivered_resource_types():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _initial_pipeline_state,
        _pipeline_completed_stages,
        _pipeline_carry_target_plan,
        _pipeline_local_assist_action,
        _pipeline_sync_wait_plan,
        _pipeline_sync_wait_station,
        _pipeline_stage_delivered_count,
        _pipeline_stage_delivered_resources,
        _update_pipeline_state_from_info,
    )

    cfg = RecurrentConfig(scenario="pipeline_assembly", map_size=8, agents=2)
    state = _initial_pipeline_state(cfg)
    updated = _update_pipeline_state_from_info(
        cfg,
        state,
        {
            "events": {
                0: [{"event": "delivered", "stage": 2, "resource_type": 4}],
                1: [{"event": "stage_completed", "stage": 2}],
            }
        },
        num_agents=2,
    )

    assert _pipeline_stage_delivered_count(updated, 2) == 1
    assert _pipeline_stage_delivered_resources(updated, 2) == [4]
    assert 2 in _pipeline_completed_stages(updated)

    duplicate_carry = _update_pipeline_state_from_info(
        cfg,
        _initial_pipeline_state(cfg),
        {
            "events": {
                0: [{"event": "delivered", "stage": 1, "resource_type": 1}],
                1: [{
                    "event": "picked_resource",
                    "stage": 1,
                    "station": [3, 3],
                    "resource_type": 1,
                    "required": [1, 1],
                }],
            }
        },
        num_agents=2,
    )
    carry_plan = _pipeline_carry_target_plan(duplicate_carry, agent_id=1, held_type=1)
    assert carry_plan is not None
    assert carry_plan["required"] == [1, 1]
    compact_carry = _initial_pipeline_state(cfg)
    compact_carry["observed_map_size"] = 8
    compact_carry = _update_pipeline_state_from_info(
        cfg,
        compact_carry,
        {
            "events": {
                1: [{
                    "event": "picked_resource",
                    "stage": 1,
                    "station": [3, 3],
                    "resource_type": 1,
                    "required": [1],
                }],
            }
        },
        num_agents=2,
    )
    compact_carry = _update_pipeline_state_from_info(
        cfg,
        compact_carry,
        {"events": {0: [{"event": "stage_completed", "stage": 1}]}},
        num_agents=2,
    )
    assert _pipeline_carry_target_plan(compact_carry, agent_id=1, held_type=1) is None
    large_carry = _initial_pipeline_state(cfg)
    large_carry["observed_map_size"] = 16
    large_carry = _update_pipeline_state_from_info(
        cfg,
        large_carry,
        {
            "events": {
                1: [{
                    "event": "picked_resource",
                    "stage": 1,
                    "station": [3, 3],
                    "resource_type": 1,
                    "required": [1],
                }],
            }
        },
        num_agents=2,
    )
    large_carry = _update_pipeline_state_from_info(
        cfg,
        large_carry,
        {"events": {0: [{"event": "stage_completed", "stage": 1}]}},
        num_agents=2,
    )
    assert _pipeline_carry_target_plan(large_carry, agent_id=1, held_type=1) is not None
    carry_obs = {
        "local_grid": np.zeros((5, 5), dtype=np.int16),
        "inventory": np.array([1], dtype=np.int16),
        "self_pos": np.array([3, 4], dtype=np.int16),
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "goal_hint": np.full((16,), -1, dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    assist_action = _pipeline_local_assist_action(
        cfg,
        carry_obs,
        agent_id=1,
        current_action_id=SyncOrSinkEnv.ACTION_DROP,
        pipeline_state=duplicate_carry,
    )
    assert assist_action in {
        SyncOrSinkEnv.ACTION_UP,
        SyncOrSinkEnv.ACTION_DOWN,
        SyncOrSinkEnv.ACTION_LEFT,
        SyncOrSinkEnv.ACTION_RIGHT,
        SyncOrSinkEnv.ACTION_INTERACT,
        SyncOrSinkEnv.ACTION_STAY,
    }
    assert assist_action != SyncOrSinkEnv.ACTION_DROP

    waiting = _update_pipeline_state_from_info(
        cfg,
        _initial_pipeline_state(cfg),
        {
            "events": {
                0: [{"event": "pipeline_sync_wait", "stage": 3, "station": [5, 6]}],
            }
        },
        num_agents=2,
    )
    assert _pipeline_sync_wait_station(waiting, 3) == (5, 6)
    assert _pipeline_sync_wait_plan(waiting) == {
        "source": "state_sync_wait",
        "stage": 3,
        "station": (5, 6),
        "required": [],
        "sync": True,
    }
    cleared = _update_pipeline_state_from_info(
        cfg,
        waiting,
        {"events": {0: [{"event": "sync_complete", "stage": 3, "station": [5, 6]}]}},
        num_agents=2,
    )
    assert _pipeline_sync_wait_station(cleared, 3) is None
    assert _pipeline_sync_wait_plan(cleared) is None


def test_recurrent_pipeline_retargets_held_resource_after_stage_completed():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY, TILE_STATION
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _pipeline_local_assist_action

    local_grid = np.full((5, 5), TILE_EMPTY, dtype=np.int16)
    local_grid[2, 2] = TILE_STATION
    obs_agent = {
        "local_grid": local_grid,
        "inventory": np.array([1], dtype=np.int16),
        "self_pos": np.array([2, 3], dtype=np.int16),
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "goal_hint": np.array(
            [
                0, 2, 3, 1, 1, -1, 0, -1, 1,
                1, 2, 4, 1, 1, -1, 1, 0, 0,
                -1, -1, -1, -1, -1, -1, -1, -1, -1,
            ],
            dtype=np.int16,
        ),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    pipeline_state = {
        "completed_stages": {0},
        "delivered_counts": {0: 1},
        "delivered_resources": {0: [1]},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
        "carry_targets": {
            1: {
                "source": "state_carry_target",
                "stage": 0,
                "station": (2, 3),
                "resource_type": 1,
                "required": [1],
            }
        },
    }

    action_id = _pipeline_local_assist_action(
        RecurrentConfig(scenario="pipeline_assembly", map_size=8, agents=3),
        obs_agent,
        agent_id=1,
        current_action_id=SyncOrSinkEnv.ACTION_INTERACT,
        pipeline_state=pipeline_state,
    )

    assert action_id == SyncOrSinkEnv.ACTION_DOWN


def test_pipeline_navigation_assist_avoids_already_satisfied_resource():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_STATION
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _pipeline_local_assist_action

    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 2] = TILE_STATION
    action_mask = np.ones((8,), dtype=np.float32)
    action_mask[SyncOrSinkEnv.ACTION_DROP] = 0.0
    obs_agent = {
        "local_grid": local_grid,
        "inventory": np.array([2], dtype=np.int16),
        "self_pos": np.array([4, 3], dtype=np.int16),
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "goal_hint": np.array(
            [0, 4, 3, 2, 2, 3, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            dtype=np.int16,
        ),
        "action_mask": action_mask,
    }
    cfg = RecurrentConfig(scenario="pipeline_assembly", map_size=8, agents=2)

    action_id = _pipeline_local_assist_action(
        cfg,
        obs_agent,
        current_action_id=SyncOrSinkEnv.ACTION_STAY,
        pipeline_state={
            "completed_stages": set(),
            "delivered_counts": {0: 1},
            "delivered_resources": {0: [2]},
            "sync_wait_stages": set(),
        },
    )

    assert action_id != SyncOrSinkEnv.ACTION_INTERACT
    assert action_id in {
        SyncOrSinkEnv.ACTION_UP,
        SyncOrSinkEnv.ACTION_DOWN,
        SyncOrSinkEnv.ACTION_LEFT,
        SyncOrSinkEnv.ACTION_RIGHT,
    }


def test_recurrent_pipeline_features_fallback_to_hint_marks_wrong_station():
    from syncorsink.envs.maps import TILE_STATION
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _pipeline_coordination_features

    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 2] = TILE_STATION
    hint = np.full((16,), -1, dtype=np.int16)
    hint[:9] = np.array([0, 2, 2, 1, 3, -1, 0, -1, 0], dtype=np.int16)
    obs_agent = {
        "local_grid": local_grid,
        "inventory": np.array([2], dtype=np.int16),
        "self_pos": np.array([2, 3], dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "goal_hint": hint,
    }
    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=3,
        obs_pipeline_features=True,
    )

    features = _pipeline_coordination_features(obs_agent, cfg, observed_map_size=8)

    assert features[0] == 1.0
    assert features[1] == 0.0
    assert features[2] == 1.0
    assert features[21] == 0.0  # held resource is not needed by the decoded plan
    assert features[24] == 1.0  # standing on a station tile
    assert features[25] == 0.0  # not the active station
    assert features[26] == 1.0  # wrong-station context
    assert features[29] == 1.0  # unsafe station interact if the policy presses interact
    assert features[31] == 0.0  # held item has no target in this decoded plan
    assert features[35] == 0.0  # it is wrong inventory, not needed-at-wrong-station
    assert features[36] == 1.0  # carried resource does not belong to the decoded plan
    assert features[37] == 0.0


def test_recurrent_pipeline_navigation_assist_corrects_delivery_and_wrong_station():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_STATION
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_navigation_assist,
        _apply_recurrent_rollout_eval_decoding,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
        eval_pipeline_navigation_assist=True,
    )
    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 2] = TILE_STATION
    deliver_obs = {
        0: {
            "local_grid": local_grid,
            "inventory": np.array([2], dtype=np.int16),
            "self_pos": np.array([4, 3], dtype=np.int16),
            "local_resource_types": np.zeros((5, 5), dtype=np.int16),
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "goal_hint": np.array(
                [0, 4, 3, 1, 2, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.int16,
            ),
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }

    corrected = _apply_pipeline_navigation_assist(
        cfg,
        deliver_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
    )

    assert int(corrected[0].item()) == SyncOrSinkEnv.ACTION_INTERACT

    rollout_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        rl_rollout_eval_decoding=True,
        rl_rollout_pipeline_navigation_assist=True,
        eval_pipeline_navigation_assist=False,
    )
    decoded_acts, decoded_actions = _apply_recurrent_rollout_eval_decoding(
        rollout_cfg,
        object(),
        deliver_obs,
        torch.zeros((1, 8), dtype=torch.float32),
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        {0: {"action": SyncOrSinkEnv.ACTION_STAY, "message_tokens": []}},
        (torch.zeros((1, 4), dtype=torch.float32), torch.zeros((1, 4), dtype=torch.float32)),
        None,
        None,
    )

    assert rollout_cfg.eval_pipeline_navigation_assist is False
    assert int(decoded_acts[0].item()) == SyncOrSinkEnv.ACTION_INTERACT
    assert decoded_actions[0]["action"] == SyncOrSinkEnv.ACTION_INTERACT

    wrong_resource_obs = {
        0: {
            **deliver_obs[0],
            "inventory": np.array([3], dtype=np.int16),
        }
    }
    rollout_guard_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        rl_rollout_eval_decoding=True,
        rl_rollout_pipeline_station_interact_guard=True,
        eval_pipeline_station_interact_guard=False,
    )
    guarded_acts, guarded_actions = _apply_recurrent_rollout_eval_decoding(
        rollout_guard_cfg,
        object(),
        wrong_resource_obs,
        torch.zeros((1, 8), dtype=torch.float32),
        torch.tensor([SyncOrSinkEnv.ACTION_INTERACT], dtype=torch.long),
        {0: {"action": SyncOrSinkEnv.ACTION_INTERACT, "message_tokens": []}},
        (torch.zeros((1, 4), dtype=torch.float32), torch.zeros((1, 4), dtype=torch.float32)),
        None,
        None,
    )

    assert rollout_guard_cfg.eval_pipeline_station_interact_guard is False
    assert int(guarded_acts[0].item()) == SyncOrSinkEnv.ACTION_DROP
    assert guarded_actions[0]["action"] == SyncOrSinkEnv.ACTION_DROP

    class LowPipelineInteractGate:
        def pipeline_interact_gate(self, hidden_state):
            return hidden_state.new_full((hidden_state.shape[0], 1), -10.0)

    gated_rollout_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        rl_rollout_eval_decoding=True,
        rl_rollout_pipeline_navigation_assist=True,
        eval_pipeline_interact_gate_threshold=0.5,
    )
    gated_acts, gated_actions = _apply_recurrent_rollout_eval_decoding(
        gated_rollout_cfg,
        LowPipelineInteractGate(),
        deliver_obs,
        torch.zeros((1, 8), dtype=torch.float32),
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        {0: {"action": SyncOrSinkEnv.ACTION_STAY, "message_tokens": []}},
        (torch.zeros((1, 4), dtype=torch.float32), torch.zeros((1, 4), dtype=torch.float32)),
        None,
        None,
    )

    assert int(gated_acts[0].item()) == SyncOrSinkEnv.ACTION_INTERACT
    assert gated_actions[0]["action"] == SyncOrSinkEnv.ACTION_INTERACT

    wrong_station_obs = {
        0: {
            **deliver_obs[0],
            "self_pos": np.array([4, 3], dtype=np.int16),
            "goal_hint": np.array(
                [0, 3, 3, 1, 2, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.int16,
            ),
        }
    }
    corrected = _apply_pipeline_navigation_assist(
        cfg,
        wrong_station_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_INTERACT], dtype=torch.long),
    )

    assert int(corrected[0].item()) == SyncOrSinkEnv.ACTION_LEFT


def test_recurrent_pipeline_navigation_assist_coordinates_sync_rendezvous():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY, TILE_STATION
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_navigation_assist,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
        eval_pipeline_navigation_assist=True,
    )
    station = (4, 3)
    hint = np.array(
        [0, station[0], station[1], 1, 2, -1, 0, -1, 1, -1, -1, -1, -1, -1, -1, -1],
        dtype=np.int16,
    )
    empty_messages = np.full((1, 8), -1, dtype=np.int16)
    action_mask = np.ones((8,), dtype=np.float32)
    station_grid = np.full((3, 3), TILE_EMPTY, dtype=np.int16)
    station_grid[1, 1] = TILE_STATION
    open_grid = np.full((3, 3), TILE_EMPTY, dtype=np.int16)
    pipeline_state = {
        "completed_stages": set(),
        "delivered_counts": {0: 1},
        "delivered_resources": {0: [2]},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
    }
    one_ready_obs = {
        0: {
            "self_pos": np.array(station, dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": station_grid,
            "local_resource_types": np.zeros((3, 3), dtype=np.int16),
            "goal_hint": hint,
            "messages_tokens": empty_messages,
            "action_mask": action_mask,
        },
        1: {
            "self_pos": np.array([3, 3], dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": open_grid,
            "local_resource_types": np.zeros((3, 3), dtype=np.int16),
            "goal_hint": hint,
            "messages_tokens": empty_messages,
            "action_mask": action_mask,
        },
    }

    waiting = _apply_pipeline_navigation_assist(
        cfg,
        one_ready_obs,
        torch.tensor(
            [SyncOrSinkEnv.ACTION_INTERACT, SyncOrSinkEnv.ACTION_STAY],
            dtype=torch.long,
        ),
        pipeline_state=pipeline_state,
    )

    assert waiting.tolist() == [SyncOrSinkEnv.ACTION_INTERACT, SyncOrSinkEnv.ACTION_RIGHT]

    both_ready_obs = {
        **one_ready_obs,
        1: {
            **one_ready_obs[1],
            "self_pos": np.array(station, dtype=np.int16),
            "local_grid": station_grid,
        },
    }
    synced = _apply_pipeline_navigation_assist(
        cfg,
        both_ready_obs,
        torch.tensor(
            [SyncOrSinkEnv.ACTION_STAY, SyncOrSinkEnv.ACTION_STAY],
            dtype=torch.long,
        ),
        pipeline_state=pipeline_state,
    )

    assert synced.tolist() == [SyncOrSinkEnv.ACTION_INTERACT, SyncOrSinkEnv.ACTION_INTERACT]

    completed_state = {
        **pipeline_state,
        "completed_stages": {0},
    }
    completed = _apply_pipeline_navigation_assist(
        cfg,
        both_ready_obs,
        torch.tensor(
            [SyncOrSinkEnv.ACTION_STAY, SyncOrSinkEnv.ACTION_STAY],
            dtype=torch.long,
        ),
        pipeline_state=completed_state,
    )

    assert completed.tolist() == [SyncOrSinkEnv.ACTION_STAY, SyncOrSinkEnv.ACTION_STAY]


def test_recurrent_pipeline_sync_rendezvous_uses_navigation_memory_to_break_loop():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY, TILE_STATION
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_sync_rendezvous_assist,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=32,
        agents=2,
        eval_pipeline_navigation_assist=True,
    )
    station = (29, 8)
    station_grid = np.full((3, 3), TILE_EMPTY, dtype=np.int16)
    station_grid[1, 1] = TILE_STATION
    open_grid = np.full((3, 3), TILE_EMPTY, dtype=np.int16)
    obs = {
        0: {
            "self_pos": np.array(station, dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": station_grid,
            "local_resource_types": np.zeros((3, 3), dtype=np.int16),
            "goal_hint": np.full((16,), -1, dtype=np.int16),
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "action_mask": np.ones((8,), dtype=np.float32),
        },
        1: {
            "self_pos": np.array([20, 8], dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": open_grid,
            "local_resource_types": np.zeros((3, 3), dtype=np.int16),
            "goal_hint": np.full((16,), -1, dtype=np.int16),
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "action_mask": np.array([0, 1, 1, 0, 1, 1, 0, 0], dtype=np.float32),
        },
    }
    pipeline_state = {
        "sync_wait_stages": {1},
        "sync_wait_stations": {1: station},
        "navigation_memory": {
            1: {
                "target": [station[0], station[1]],
                "pos": [20, 9],
                "action": SyncOrSinkEnv.ACTION_UP,
                "context": "stage:1:sync",
            }
        },
    }

    corrected = _apply_pipeline_sync_rendezvous_assist(
        cfg,
        obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY, SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        pipeline_state=pipeline_state,
    )

    assert corrected.tolist() == [SyncOrSinkEnv.ACTION_INTERACT, SyncOrSinkEnv.ACTION_LEFT]
    assert pipeline_state["navigation_memory"][1]["action"] == SyncOrSinkEnv.ACTION_LEFT


def test_recurrent_pipeline_sync_message_routes_rendezvous_without_hint():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY, TILE_STATION
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_navigation_assist,
        _pipeline_plan_from_message_tokens,
        _pipeline_plan_message_tokens,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        eval_pipeline_navigation_assist=True,
        eval_pipeline_navigation_assist_trust_messages=True,
        eval_pipeline_plan_broadcast_assist=True,
    )
    station = (4, 3)
    plan = {
        "stage": 0,
        "station": station,
        "required": [2],
        "deps": [],
        "sync": True,
    }
    tokens = _pipeline_plan_message_tokens(cfg, plan, observed_map_size=8)
    assert tokens == [12, 0, station[0], station[1], 5, 2]
    parsed = _pipeline_plan_from_message_tokens(np.array([tokens], dtype=np.int16), 8)
    assert parsed["sync"] is True
    assert parsed["required"] == [2]

    station_grid = np.full((3, 3), TILE_EMPTY, dtype=np.int16)
    station_grid[1, 1] = TILE_STATION
    open_grid = np.full((3, 3), TILE_EMPTY, dtype=np.int16)
    action_mask = np.ones((8,), dtype=np.float32)
    messages = np.full((1, 8), -1, dtype=np.int16)
    messages[0, : len(tokens)] = tokens
    no_hint = np.full((16,), -1, dtype=np.int16)
    pipeline_state = {
        "completed_stages": set(),
        "delivered_counts": {0: 1},
        "delivered_resources": {0: [2]},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
    }
    obs = {
        0: {
            "self_pos": np.array(station, dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": station_grid,
            "local_resource_types": np.zeros((3, 3), dtype=np.int16),
            "goal_hint": no_hint,
            "messages_tokens": messages,
            "action_mask": action_mask,
        },
        1: {
            "self_pos": np.array([3, 3], dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": open_grid,
            "local_resource_types": np.zeros((3, 3), dtype=np.int16),
            "goal_hint": no_hint,
            "messages_tokens": messages,
            "action_mask": action_mask,
        },
    }

    corrected = _apply_pipeline_navigation_assist(
        cfg,
        obs,
        torch.tensor(
            [SyncOrSinkEnv.ACTION_STAY, SyncOrSinkEnv.ACTION_STAY],
            dtype=torch.long,
        ),
        pipeline_state=pipeline_state,
    )

    assert corrected.tolist() == [SyncOrSinkEnv.ACTION_INTERACT, SyncOrSinkEnv.ACTION_RIGHT]


def test_recurrent_pipeline_pickup_gate_decoding_suppresses_unneeded_resource():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY, TILE_RESOURCE
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_pickup_gate_decoding,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_pickup_gate_suppress=True,
    )
    hint = np.array(
        [0, 4, 3, 1, 2, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        dtype=np.int16,
    )
    local_grid = np.full((3, 3), TILE_EMPTY, dtype=np.int16)
    local_grid[1, 1] = TILE_RESOURCE
    local_resource = np.zeros((3, 3), dtype=np.int16)
    action_mask = np.ones((8,), dtype=np.float32)
    base_obs = {
        "self_pos": np.array([1, 1], dtype=np.int16),
        "inventory": np.array([0], dtype=np.int16),
        "local_grid": local_grid,
        "local_resource_types": local_resource,
        "goal_hint": hint,
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "action_mask": action_mask,
    }

    unneeded_obs = {0: {**base_obs, "local_resource_types": local_resource.copy()}}
    unneeded_obs[0]["local_resource_types"][1, 1] = 3
    suppressed = _apply_pipeline_pickup_gate_decoding(
        cfg,
        unneeded_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_PICKUP], dtype=torch.long),
    )
    assert suppressed.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    needed_obs = {0: {**base_obs, "local_resource_types": local_resource.copy()}}
    needed_obs[0]["local_resource_types"][1, 1] = 2
    allowed = _apply_pipeline_pickup_gate_decoding(
        cfg,
        needed_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_PICKUP], dtype=torch.long),
    )
    assert allowed.tolist() == [SyncOrSinkEnv.ACTION_PICKUP]

    structured_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_navigation_assist=True,
        eval_pipeline_navigation_assist_trust_messages=True,
        eval_pipeline_plan_broadcast_assist=True,
        eval_pipeline_pickup_gate_suppress=True,
    )
    later_message_obs = {0: {**base_obs, "local_resource_types": local_resource.copy()}}
    later_message_obs[0]["local_resource_types"][1, 1] = 2
    later_message_obs[0]["messages_tokens"] = np.array(
        [[12, 1, 5, 5, 1, 3, -1, -1]],
        dtype=np.int16,
    )
    lower_hint_allowed = _apply_pipeline_pickup_gate_decoding(
        structured_cfg,
        later_message_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_PICKUP], dtype=torch.long),
    )
    assert lower_hint_allowed.tolist() == [SyncOrSinkEnv.ACTION_PICKUP]

    no_plan_obs = {
        0: {
            **unneeded_obs[0],
            "goal_hint": np.full((16,), -1, dtype=np.int16),
        }
    }
    no_plan = _apply_pipeline_pickup_gate_decoding(
        cfg,
        no_plan_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_PICKUP], dtype=torch.long),
    )
    assert no_plan.tolist() == [SyncOrSinkEnv.ACTION_PICKUP]


def test_recurrent_pipeline_interact_gate_decoding_suppresses_low_confidence_station_interact():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_STATION
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_interact_gate_decoding,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
        eval_pipeline_interact_gate_threshold=0.5,
    )
    station_grid = np.zeros((5, 5), dtype=np.int16)
    station_grid[2, 2] = TILE_STATION
    empty_grid = np.zeros((5, 5), dtype=np.int16)
    obs = {
        0: {
            "local_grid": station_grid,
            "action_mask": np.ones((8,), dtype=np.float32),
        },
        1: {
            "local_grid": empty_grid,
            "action_mask": np.ones((8,), dtype=np.float32),
        },
    }
    acts = torch.tensor(
        [SyncOrSinkEnv.ACTION_INTERACT, SyncOrSinkEnv.ACTION_INTERACT],
        dtype=torch.long,
    )

    suppressed = _apply_pipeline_interact_gate_decoding(
        cfg,
        obs,
        acts,
        torch.tensor([[-3.0], [-3.0]], dtype=torch.float32),
    )
    assert suppressed.tolist() == [SyncOrSinkEnv.ACTION_STAY, SyncOrSinkEnv.ACTION_INTERACT]

    allowed = _apply_pipeline_interact_gate_decoding(
        cfg,
        obs,
        acts,
        torch.tensor([[3.0], [-3.0]], dtype=torch.float32),
    )
    assert allowed.tolist() == [SyncOrSinkEnv.ACTION_INTERACT, SyncOrSinkEnv.ACTION_INTERACT]

    promote_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
        eval_pipeline_interact_gate_threshold=0.5,
        eval_pipeline_interact_gate_promote=True,
    )
    promoted = _apply_pipeline_interact_gate_decoding(
        promote_cfg,
        obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY, SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        torch.tensor([[3.0], [3.0]], dtype=torch.float32),
    )
    assert promoted.tolist() == [SyncOrSinkEnv.ACTION_INTERACT, SyncOrSinkEnv.ACTION_STAY]


def test_recurrent_pipeline_plan_head_decoding_uses_visible_plan_threshold():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_plan_head_decoding,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_plan_head_threshold=0.5,
    )
    obs = {
        0: {
            "goal_hint": np.array(
                [0, 4, 3, 1, 2, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.int16,
            ),
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }
    logits = torch.full((1, 8), -4.0, dtype=torch.float32)
    logits[0, SyncOrSinkEnv.ACTION_RIGHT] = 4.0

    corrected = _apply_pipeline_plan_head_decoding(
        cfg,
        obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        logits,
    )

    assert corrected.tolist() == [SyncOrSinkEnv.ACTION_RIGHT]

    no_plan_obs = {
        0: {
            **obs[0],
            "goal_hint": np.full((16,), -1, dtype=np.int16),
        }
    }
    unchanged = _apply_pipeline_plan_head_decoding(
        cfg,
        no_plan_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        logits,
    )
    assert unchanged.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    masked_obs = {
        0: {
            **obs[0],
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }
    masked_obs[0]["action_mask"][SyncOrSinkEnv.ACTION_RIGHT] = 0.0
    masked = _apply_pipeline_plan_head_decoding(
        cfg,
        masked_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        logits,
    )
    assert masked.tolist() == [SyncOrSinkEnv.ACTION_STAY]


def test_recurrent_pipeline_navigation_head_decoding_requires_visible_plan_and_movement():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_navigation_head_decoding,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_navigation_head_threshold=0.5,
    )
    obs = {
        0: {
            "goal_hint": np.array(
                [0, 4, 3, 1, 2, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.int16,
            ),
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }
    logits = torch.full((1, 8), -4.0, dtype=torch.float32)
    logits[0, SyncOrSinkEnv.ACTION_RIGHT] = 4.0

    corrected = _apply_pipeline_navigation_head_decoding(
        cfg,
        obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        logits,
    )

    assert corrected.tolist() == [SyncOrSinkEnv.ACTION_RIGHT]

    no_plan_obs = {
        0: {
            **obs[0],
            "goal_hint": np.full((16,), -1, dtype=np.int16),
        }
    }
    unchanged = _apply_pipeline_navigation_head_decoding(
        cfg,
        no_plan_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        logits,
    )
    assert unchanged.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    masked_obs = {
        0: {
            **obs[0],
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }
    masked_obs[0]["action_mask"][SyncOrSinkEnv.ACTION_RIGHT] = 0.0
    masked = _apply_pipeline_navigation_head_decoding(
        cfg,
        masked_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        logits,
    )
    assert masked.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    non_movement_logits = torch.full((1, 8), -4.0, dtype=torch.float32)
    non_movement_logits[0, SyncOrSinkEnv.ACTION_INTERACT] = 4.0
    non_movement = _apply_pipeline_navigation_head_decoding(
        cfg,
        obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        non_movement_logits,
    )
    assert non_movement.tolist() == [SyncOrSinkEnv.ACTION_STAY]


def test_recurrent_pipeline_event_head_decoding_guards_pickup_and_interact():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY, TILE_RESOURCE, TILE_STATION
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_event_head_decoding,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_event_head_threshold=0.5,
    )
    hint = np.array(
        [0, 4, 3, 1, 2, -1, 0, -1, 0],
        dtype=np.int16,
    )
    empty_messages = np.full((1, 8), -1, dtype=np.int16)
    action_mask = np.ones((8,), dtype=np.float32)
    resource_grid = np.zeros((3, 3), dtype=np.int16)
    resource_grid[1, 1] = 2
    pickup_grid = np.full((3, 3), TILE_EMPTY, dtype=np.int16)
    pickup_grid[1, 1] = TILE_RESOURCE
    pickup_obs = {
        0: {
            "self_pos": np.array([1, 1], dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": pickup_grid,
            "local_resource_types": resource_grid,
            "goal_hint": hint,
            "messages_tokens": empty_messages,
            "action_mask": action_mask,
        }
    }
    pickup_logits = torch.full((1, 8), -4.0, dtype=torch.float32)
    pickup_logits[0, SyncOrSinkEnv.ACTION_PICKUP] = 4.0
    corrected = _apply_pipeline_event_head_decoding(
        cfg,
        pickup_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        pickup_logits,
    )
    assert corrected.tolist() == [SyncOrSinkEnv.ACTION_PICKUP]

    no_plan_obs = {0: {**pickup_obs[0], "goal_hint": np.full((9,), -1, dtype=np.int16)}}
    unchanged = _apply_pipeline_event_head_decoding(
        cfg,
        no_plan_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        pickup_logits,
    )
    assert unchanged.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    wrong_resource = resource_grid.copy()
    wrong_resource[1, 1] = 3
    wrong_resource_obs = {0: {**pickup_obs[0], "local_resource_types": wrong_resource}}
    unchanged = _apply_pipeline_event_head_decoding(
        cfg,
        wrong_resource_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        pickup_logits,
    )
    assert unchanged.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    masked_pickup_obs = {0: {**pickup_obs[0], "action_mask": action_mask.copy()}}
    masked_pickup_obs[0]["action_mask"][SyncOrSinkEnv.ACTION_PICKUP] = 0.0
    unchanged = _apply_pipeline_event_head_decoding(
        cfg,
        masked_pickup_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        pickup_logits,
    )
    assert unchanged.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    station_grid = np.full((3, 3), TILE_EMPTY, dtype=np.int16)
    station_grid[1, 1] = TILE_STATION
    station_obs = {
        0: {
            **pickup_obs[0],
            "self_pos": np.array([4, 3], dtype=np.int16),
            "inventory": np.array([2], dtype=np.int16),
            "local_grid": station_grid,
            "local_resource_types": np.zeros((3, 3), dtype=np.int16),
            "action_mask": action_mask.copy(),
        }
    }
    interact_logits = torch.full((1, 8), -4.0, dtype=torch.float32)
    interact_logits[0, SyncOrSinkEnv.ACTION_INTERACT] = 4.0
    corrected = _apply_pipeline_event_head_decoding(
        cfg,
        station_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        interact_logits,
    )
    assert corrected.tolist() == [SyncOrSinkEnv.ACTION_INTERACT]

    pickup_biased_event_logits = torch.full((1, 8), -6.0, dtype=torch.float32)
    pickup_biased_event_logits[0, SyncOrSinkEnv.ACTION_PICKUP] = 5.0
    pickup_biased_event_logits[0, SyncOrSinkEnv.ACTION_INTERACT] = 4.0
    corrected = _apply_pipeline_event_head_decoding(
        cfg,
        station_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        pickup_biased_event_logits,
    )
    assert corrected.tolist() == [SyncOrSinkEnv.ACTION_INTERACT]

    wrong_station_obs = {
        0: {
            **station_obs[0],
            "self_pos": np.array([5, 3], dtype=np.int16),
        }
    }
    unchanged = _apply_pipeline_event_head_decoding(
        cfg,
        wrong_station_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        interact_logits,
    )
    assert unchanged.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    carry_target_state = {
        "completed_stages": set(),
        "delivered_counts": {},
        "delivered_resources": {},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
        "carry_targets": {
            0: {"stage": 1, "station": (5, 3), "resource_type": 2},
        },
    }
    carry_target_suppressed = _apply_pipeline_event_head_decoding(
        cfg,
        station_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        interact_logits,
        pipeline_state=carry_target_state,
    )
    assert carry_target_suppressed.tolist() == [SyncOrSinkEnv.ACTION_STAY]
    carry_target_obs = {
        0: {
            **station_obs[0],
            "self_pos": np.array([5, 3], dtype=np.int16),
        }
    }
    carry_target_promoted = _apply_pipeline_event_head_decoding(
        cfg,
        carry_target_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        interact_logits,
        pipeline_state=carry_target_state,
    )
    assert carry_target_promoted.tolist() == [SyncOrSinkEnv.ACTION_INTERACT]

    wrong_held_obs = {0: {**station_obs[0], "inventory": np.array([3], dtype=np.int16)}}
    unchanged = _apply_pipeline_event_head_decoding(
        cfg,
        wrong_held_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        interact_logits,
    )
    assert unchanged.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    sync_hint = np.array(
        [0, 4, 3, 1, 2, -1, 0, -1, 1],
        dtype=np.int16,
    )
    sync_obs = {
        0: {
            **station_obs[0],
            "inventory": np.array([0], dtype=np.int16),
            "goal_hint": sync_hint,
        }
    }
    sync_state = {
        "completed_stages": set(),
        "sync_wait_stages": {0},
        "sync_wait_stations": {0: (4, 3)},
        "delivered_resources": {0: [2]},
    }
    premature_sync = _apply_pipeline_event_head_decoding(
        cfg,
        sync_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        interact_logits,
        pipeline_state={
            "completed_stages": set(),
            "sync_wait_stages": set(),
            "sync_wait_stations": {},
            "delivered_resources": {},
        },
    )
    corrected = _apply_pipeline_event_head_decoding(
        cfg,
        sync_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        interact_logits,
        pipeline_state=sync_state,
    )
    assert premature_sync.tolist() == [SyncOrSinkEnv.ACTION_STAY]
    assert corrected.tolist() == [SyncOrSinkEnv.ACTION_INTERACT]


def test_recurrent_pipeline_option_decoding_uses_visible_plan_threshold():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY, TILE_STATION
    from syncorsink.train.recurrent_bc_rl import (
        PIPELINE_OPTION_DELIVER,
        PIPELINE_OPTION_NAV_STATION,
        PIPELINE_OPTION_PICKUP,
        PIPELINE_OPTION_SYNC,
        RecurrentConfig,
        _apply_pipeline_option_decoding,
        _apply_pipeline_station_interact_guard,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_option_threshold=0.5,
    )
    local_grid = np.full((3, 3), TILE_EMPTY, dtype=np.int16)
    local_resource = np.zeros((3, 3), dtype=np.int16)
    local_resource[1, 1] = 2
    obs = {
        0: {
            "self_pos": np.array([1, 1], dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": local_grid,
            "local_resource_types": local_resource,
            "goal_hint": np.array(
                [0, 4, 3, 1, 2, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.int16,
            ),
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }
    option_logits = torch.full((1, 8), -4.0, dtype=torch.float32)
    option_logits[0, PIPELINE_OPTION_PICKUP] = 4.0

    corrected = _apply_pipeline_option_decoding(
        cfg,
        obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        option_logits,
    )
    assert corrected.tolist() == [SyncOrSinkEnv.ACTION_PICKUP]

    no_plan_obs = {
        0: {
            **obs[0],
            "goal_hint": np.full((16,), -1, dtype=np.int16),
        }
    }
    unchanged = _apply_pipeline_option_decoding(
        cfg,
        no_plan_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        option_logits,
    )
    assert unchanged.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    masked_obs = {0: {**obs[0], "action_mask": np.ones((8,), dtype=np.float32)}}
    masked_obs[0]["action_mask"][SyncOrSinkEnv.ACTION_PICKUP] = 0.0
    masked = _apply_pipeline_option_decoding(
        cfg,
        masked_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        option_logits,
    )
    assert masked.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    nav_station_logits = torch.full((1, 8), -4.0, dtype=torch.float32)
    nav_station_logits[0, PIPELINE_OPTION_NAV_STATION] = 4.0
    already_at_station_obs = {
        0: {
            **obs[0],
            "self_pos": np.array([4, 3], dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
        }
    }
    nav_station = _apply_pipeline_option_decoding(
        cfg,
        already_at_station_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        nav_station_logits,
    )
    assert nav_station.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    station_grid = np.full((3, 3), TILE_EMPTY, dtype=np.int16)
    station_grid[1, 1] = TILE_STATION
    deliver_logits = torch.full((1, 8), -4.0, dtype=torch.float32)
    deliver_logits[0, PIPELINE_OPTION_DELIVER] = 4.0
    completed_stage_obs = {
        0: {
            **obs[0],
            "self_pos": np.array([4, 3], dtype=np.int16),
            "inventory": np.array([2], dtype=np.int16),
            "local_grid": station_grid,
        }
    }
    completed_stage = _apply_pipeline_option_decoding(
        cfg,
        completed_stage_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        deliver_logits,
        pipeline_state={"completed_stages": {0}},
    )
    assert completed_stage.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    guard_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_station_interact_guard=True,
    )
    guarded_wrong_resource = _apply_pipeline_station_interact_guard(
        guard_cfg,
        {
            0: {
                **completed_stage_obs[0],
                "inventory": np.array([3], dtype=np.int16),
            }
        },
        torch.tensor([SyncOrSinkEnv.ACTION_INTERACT], dtype=torch.long),
        pipeline_state={"completed_stages": set()},
    )
    guarded_valid_delivery = _apply_pipeline_station_interact_guard(
        guard_cfg,
        completed_stage_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_INTERACT], dtype=torch.long),
        pipeline_state={"completed_stages": set()},
    )
    assert guarded_wrong_resource.tolist() == [SyncOrSinkEnv.ACTION_DROP]
    assert guarded_valid_delivery.tolist() == [SyncOrSinkEnv.ACTION_INTERACT]

    carry_target_state = {
        "completed_stages": set(),
        "delivered_counts": {},
        "delivered_resources": {},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
        "carry_targets": {
            0: {"stage": 1, "station": (5, 3), "resource_type": 2},
        },
    }
    guarded_carry_target_elsewhere = _apply_pipeline_station_interact_guard(
        guard_cfg,
        completed_stage_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_INTERACT], dtype=torch.long),
        pipeline_state=carry_target_state,
    )
    carry_target_station_obs = {
        0: {
            **completed_stage_obs[0],
            "self_pos": np.array([5, 3], dtype=np.int16),
        }
    }
    guarded_carry_target_station = _apply_pipeline_station_interact_guard(
        guard_cfg,
        carry_target_station_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_INTERACT], dtype=torch.long),
        pipeline_state=carry_target_state,
    )
    assert guarded_carry_target_elsewhere.tolist() != [SyncOrSinkEnv.ACTION_INTERACT]
    assert guarded_carry_target_station.tolist() == [SyncOrSinkEnv.ACTION_INTERACT]

    blocked_stage_obs = {
        0: {
            **completed_stage_obs[0],
            "goal_hint": np.array(
                [3, 4, 3, 1, 2, -1, 1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.int16,
            ),
        }
    }
    dependency_blocked_stage = _apply_pipeline_option_decoding(
        cfg,
        blocked_stage_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        deliver_logits,
        pipeline_state={"completed_stages": set()},
    )
    dependency_ready_stage = _apply_pipeline_option_decoding(
        cfg,
        blocked_stage_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        deliver_logits,
        pipeline_state={"completed_stages": {0}},
    )
    guarded_blocked_delivery = _apply_pipeline_station_interact_guard(
        guard_cfg,
        blocked_stage_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_INTERACT], dtype=torch.long),
        pipeline_state={"completed_stages": set()},
    )
    assert dependency_blocked_stage.tolist() == [SyncOrSinkEnv.ACTION_STAY]
    assert dependency_ready_stage.tolist() == [SyncOrSinkEnv.ACTION_STAY]
    assert guarded_blocked_delivery.tolist() == [SyncOrSinkEnv.ACTION_STAY]

    interact_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_option_threshold=0.5,
        eval_pipeline_option_allow_interact=True,
    )
    dependency_ready_stage_interact = _apply_pipeline_option_decoding(
        interact_cfg,
        blocked_stage_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        deliver_logits,
        pipeline_state={"completed_stages": {0}},
    )
    assert dependency_ready_stage_interact.tolist() == [SyncOrSinkEnv.ACTION_INTERACT]

    sync_logits = torch.full((1, 8), -4.0, dtype=torch.float32)
    sync_logits[0, PIPELINE_OPTION_SYNC] = 4.0
    sync_stage_obs = {
        0: {
            **completed_stage_obs[0],
            "inventory": np.array([0], dtype=np.int16),
            "goal_hint": np.array(
                [0, 4, 3, 1, 2, -1, 0, -1, 1, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.int16,
            ),
        }
    }
    premature_sync_state = {
        "completed_stages": set(),
        "delivered_counts": {},
        "delivered_resources": {},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
    }
    ready_sync_state = {
        "completed_stages": set(),
        "delivered_counts": {0: 1},
        "delivered_resources": {0: [2]},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
    }
    premature_option_sync = _apply_pipeline_option_decoding(
        interact_cfg,
        sync_stage_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        sync_logits,
        pipeline_state=premature_sync_state,
    )
    ready_option_sync = _apply_pipeline_option_decoding(
        interact_cfg,
        sync_stage_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        sync_logits,
        pipeline_state=ready_sync_state,
    )
    guarded_premature_sync = _apply_pipeline_station_interact_guard(
        guard_cfg,
        sync_stage_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_INTERACT], dtype=torch.long),
        pipeline_state=premature_sync_state,
    )
    guarded_ready_sync = _apply_pipeline_station_interact_guard(
        guard_cfg,
        sync_stage_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_INTERACT], dtype=torch.long),
        pipeline_state=ready_sync_state,
    )
    assert premature_option_sync.tolist() == [SyncOrSinkEnv.ACTION_STAY]
    assert ready_option_sync.tolist() == [SyncOrSinkEnv.ACTION_INTERACT]
    assert guarded_premature_sync.tolist() != [SyncOrSinkEnv.ACTION_INTERACT]
    assert guarded_ready_sync.tolist() == [SyncOrSinkEnv.ACTION_INTERACT]

    unknown_station_sync_obs = {
        0: {
            **sync_stage_obs[0],
            "goal_hint": np.full((16,), -1, dtype=np.int16),
        }
    }
    guarded_unknown_empty_station = _apply_pipeline_station_interact_guard(
        guard_cfg,
        unknown_station_sync_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_INTERACT], dtype=torch.long),
        pipeline_state=premature_sync_state,
    )
    guarded_sync_wait_station = _apply_pipeline_station_interact_guard(
        guard_cfg,
        unknown_station_sync_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_INTERACT], dtype=torch.long),
        pipeline_state={
            "completed_stages": set(),
            "delivered_counts": {},
            "delivered_resources": {},
            "sync_wait_stages": {0},
            "sync_wait_stations": {0: (4, 3)},
        },
    )
    assert guarded_unknown_empty_station.tolist() != [SyncOrSinkEnv.ACTION_INTERACT]
    assert guarded_sync_wait_station.tolist() == [SyncOrSinkEnv.ACTION_INTERACT]

    gated_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_option_threshold=0.5,
        eval_pipeline_option_allow_interact=True,
        eval_pipeline_interact_gate_threshold=0.5,
    )
    ready_delivery_obs = {
        0: {
            **obs[0],
            "self_pos": np.array([4, 3], dtype=np.int16),
            "inventory": np.array([2], dtype=np.int16),
            "local_grid": station_grid,
        }
    }
    low_gate_delivery = _apply_pipeline_option_decoding(
        gated_cfg,
        ready_delivery_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        deliver_logits,
        torch.tensor([[-3.0]], dtype=torch.float32),
    )
    assert low_gate_delivery.tolist() == [SyncOrSinkEnv.ACTION_STAY]
    high_gate_delivery = _apply_pipeline_option_decoding(
        gated_cfg,
        ready_delivery_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
        deliver_logits,
        torch.tensor([[3.0]], dtype=torch.float32),
    )
    assert high_gate_delivery.tolist() == [SyncOrSinkEnv.ACTION_INTERACT]


def test_recurrent_pipeline_feedback_flags_are_self_local():
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _ensure_feedback_parent_enabled,
        _feedback_dim,
        _feedback_matrix,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        obs_feedback=True,
        obs_pipeline_feedback=True,
    )
    info = {
        "events": {
            0: [
                {"event": "picked_resource", "resource_type": 2},
                {"event": "delivered", "stage": 1, "resource_type": 2, "station": [4, 5]},
                {"event": "pipeline_sync_wait", "stage": 1, "station": [4, 5]},
            ],
            1: [
                {"event": "stage_completed"},
                {"event": "sync_complete"},
                {"event": "pipeline_wrong_delivery"},
            ],
        }
    }
    obs = {
        0: {"self_pos": np.array([3, 5], dtype=np.int16)},
        1: {"self_pos": np.array([5, 5], dtype=np.int16)},
    }

    feedback = _feedback_matrix(cfg, 2, info=info, obs=obs)

    assert _feedback_dim(cfg) == 26
    assert feedback.shape == (2, 26)
    assert feedback[0, 12:20].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert feedback[1, 12:20].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert feedback[0, 20] == pytest.approx(2.0 / 8.0)
    assert feedback[0, 21] == pytest.approx(2.0 / 4.0)
    assert feedback[0, 22:26].tolist() == pytest.approx([1.0, 1.0 / 7.0, 0.0, 1.0 / 7.0])
    assert feedback[1, 20:26].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    shared_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        obs_feedback=True,
        obs_pipeline_feedback=True,
        obs_pipeline_shared_feedback=True,
    )
    shared_feedback = _feedback_matrix(shared_cfg, 2, info=info, obs=obs)

    assert _feedback_dim(shared_cfg) == 26
    assert shared_feedback.shape == (2, 26)
    assert shared_feedback[0, 12:20].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    assert shared_feedback[1, 12:20].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    assert shared_feedback[0, 20] == pytest.approx(2.0 / 8.0)
    assert shared_feedback[0, 21] == pytest.approx(2.0 / 4.0)
    assert shared_feedback[0, 22:26].tolist() == pytest.approx([1.0, 1.0 / 7.0, 0.0, 1.0 / 7.0])
    assert shared_feedback[1, 20] == pytest.approx(2.0 / 8.0)
    assert shared_feedback[1, 21] == pytest.approx(2.0 / 4.0)
    assert shared_feedback[1, 22:26].tolist() == pytest.approx([1.0, -1.0 / 7.0, 0.0, 1.0 / 7.0])

    implied_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        obs_pipeline_shared_feedback=True,
    )
    assert _ensure_feedback_parent_enabled(implied_cfg) is True
    assert implied_cfg.obs_pipeline_feedback is True
    assert implied_cfg.obs_feedback is True


def test_recurrent_pipeline_navigation_assist_pickup_resource_and_latest_message():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_RESOURCE
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_navigation_assist,
        _pipeline_plan_from_message_tokens,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
        eval_pipeline_navigation_assist=True,
        eval_pipeline_navigation_assist_trust_messages=True,
    )
    messages = np.array(
        [
            [12, 0, 1, 1, 1, 1, -1, -1],
            [12, 1, 5, 4, 1, 3, -1, -1],
        ],
        dtype=np.int16,
    )
    assert _pipeline_plan_from_message_tokens(messages, 8)["station"] == (5, 4)

    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 2] = TILE_RESOURCE
    local_resource_types = np.zeros((5, 5), dtype=np.int16)
    local_resource_types[2, 2] = 3
    pickup_obs = {
        0: {
            "local_grid": local_grid,
            "inventory": np.array([0], dtype=np.int16),
            "self_pos": np.array([4, 4], dtype=np.int16),
            "local_resource_types": local_resource_types,
            "messages_tokens": messages,
            "goal_hint": np.full((16,), -1, dtype=np.int16),
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }

    corrected = _apply_pipeline_navigation_assist(
        cfg,
        pickup_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
    )

    assert int(corrected[0].item()) == SyncOrSinkEnv.ACTION_PICKUP

    local_grid[2, 2] = 0
    local_resource_types[2, 2] = 0
    local_resource_types[2, 3] = 3
    resource_obs = {
        0: {
            **pickup_obs[0],
            "local_grid": local_grid,
            "local_resource_types": local_resource_types,
        }
    }
    corrected = _apply_pipeline_navigation_assist(
        cfg,
        resource_obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
    )

    assert int(corrected[0].item()) == SyncOrSinkEnv.ACTION_RIGHT


def test_recurrent_pipeline_structured_broadcast_assist_prefers_lower_message_stage():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_RESOURCE
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_navigation_assist,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_navigation_assist=True,
        eval_pipeline_navigation_assist_trust_messages=True,
        eval_pipeline_plan_broadcast_assist=True,
    )
    messages = np.array(
        [
            [12, 0, 1, 1, 1, 1, -1, -1],
            [12, 1, 5, 4, 1, 3, -1, -1],
        ],
        dtype=np.int16,
    )
    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 2] = TILE_RESOURCE
    local_resource_types = np.zeros((5, 5), dtype=np.int16)
    local_resource_types[2, 2] = 1
    obs = {
        0: {
            "local_grid": local_grid,
            "inventory": np.array([0], dtype=np.int16),
            "self_pos": np.array([4, 4], dtype=np.int16),
            "local_resource_types": local_resource_types,
            "messages_tokens": messages,
            "goal_hint": np.full((16,), -1, dtype=np.int16),
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }

    corrected = _apply_pipeline_navigation_assist(
        cfg,
        obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
    )

    assert int(corrected[0].item()) == SyncOrSinkEnv.ACTION_PICKUP


def test_recurrent_pipeline_navigation_memory_breaks_two_cell_loop():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _pipeline_local_assist_action,
        _update_pipeline_resource_memory_from_obs,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_navigation_assist=True,
    )
    obs_agent = {
        "self_pos": np.array([9, 5], dtype=np.int16),
        "inventory": np.array([1], dtype=np.int16),
        "local_grid": np.array(
            [
                [10, 10, 10, 1, 0, 0, 3, 10, 10],
                [10, 10, 10, 10, 0, 0, 10, 10, 10],
                [10, 10, 10, 1, 2, 0, 10, 10, 10],
                [10, 10, 10, 1, 0, 1, 10, 10, 10],
                [10, 10, 10, 1, 0, 1, 10, 10, 10],
                [10, 10, 10, 0, 0, 0, 10, 10, 10],
                [10, 0, 0, 0, 0, 0, 1, 10, 10],
                [0, 1, 0, 3, 0, 0, 1, 10, 10],
                [10, 1, 9, 1, 0, 1, 1, 10, 10],
            ],
            dtype=np.int16,
        ),
        "local_resource_types": np.zeros((9, 9), dtype=np.int16),
        "goal_hint": np.full((16,), -1, dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "action_mask": np.array([1, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32),
    }
    pipeline_state = {
        "completed_stages": {0, 1, 2},
        "delivered_counts": {},
        "delivered_resources": {},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
        "carry_targets": {
            0: {
                "source": "state_carry_target",
                "stage": 3,
                "station": (11, 1),
                "resource_type": 1,
                "required": [1],
            }
        },
        "navigation_memory": {
            0: {
                "target": [11, 1],
                "pos": [9, 4],
                "action": SyncOrSinkEnv.ACTION_DOWN,
                "context": "stage:3:deliver:1",
            }
        },
    }

    action = _pipeline_local_assist_action(
        cfg,
        obs_agent,
        agent_id=0,
        current_action_id=SyncOrSinkEnv.ACTION_UP,
        pipeline_state=pipeline_state,
    )

    assert action == SyncOrSinkEnv.ACTION_DOWN


def test_recurrent_pipeline_navigation_assist_opens_doors_en_route():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_DOOR, TILE_EMPTY
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _pipeline_local_assist_action

    local_grid = np.full((5, 5), TILE_EMPTY, dtype=np.int16)
    local_grid[2, 3] = TILE_DOOR
    action_mask = np.ones((8,), dtype=np.float32)
    action_mask[SyncOrSinkEnv.ACTION_RIGHT] = 0.0
    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=32,
        agents=3,
        eval_pipeline_navigation_assist=True,
    )
    carry_obs = {
        "self_pos": np.array([16, 4], dtype=np.int16),
        "inventory": np.array([1], dtype=np.int16),
        "local_grid": local_grid,
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "goal_hint": np.full((16,), -1, dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "action_mask": action_mask,
    }
    pipeline_state = {
        "completed_stages": set(),
        "delivered_counts": {},
        "delivered_resources": {},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
        "carry_targets": {
            0: {
                "source": "state_carry_target",
                "stage": 0,
                "station": (18, 7),
                "resource_type": 1,
                "required": [1, 3],
            }
        },
        "terrain_memory": {
            "passable": {(16, 4)},
            "blocked": {(17, 4)},
            "doors": {(17, 4)},
            "map_size": 32,
        },
    }

    carry_action = _pipeline_local_assist_action(
        cfg,
        carry_obs,
        agent_id=0,
        current_action_id=SyncOrSinkEnv.ACTION_LEFT,
        pipeline_state=pipeline_state,
    )

    assert carry_action == SyncOrSinkEnv.ACTION_INTERACT

    search_obs = {
        **carry_obs,
        "inventory": np.array([0], dtype=np.int16),
        "goal_hint": np.array(
            [0, 18, 7, 1, 1, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            dtype=np.int16,
        ),
    }
    search_state = {
        **pipeline_state,
        "carry_targets": {},
        "resource_memory": {1: {(18, 7)}},
    }
    search_action = _pipeline_local_assist_action(
        cfg,
        search_obs,
        agent_id=0,
        current_action_id=SyncOrSinkEnv.ACTION_LEFT,
        pipeline_state=search_state,
    )

    assert search_action == SyncOrSinkEnv.ACTION_INTERACT


def test_recurrent_pipeline_duplicate_carriers_use_compact_navigation_memory(monkeypatch):
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY
    import syncorsink.train.recurrent_bc_rl as recurrent

    cfg = recurrent.RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=32,
        agents=3,
        eval_pipeline_navigation_assist=True,
    )
    obs_agent = {
        "self_pos": np.array([10, 6], dtype=np.int16),
        "inventory": np.array([4], dtype=np.int16),
        "local_grid": np.full((5, 5), TILE_EMPTY, dtype=np.int16),
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "goal_hint": np.full((16,), -1, dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    duplicate_state = {
        "completed_stages": set(),
        "delivered_counts": {},
        "delivered_resources": {},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
        "carry_targets": {
            0: {
                "stage": 1,
                "station": (7, 6),
                "resource_type": 4,
                "required": [4, 4],
            },
            2: {
                "stage": 1,
                "station": (7, 6),
                "resource_type": 4,
                "required": [4, 4],
            },
        },
    }
    calls = []

    def fake_memory_adjusted(*args, **kwargs):
        calls.append(kwargs)
        return SyncOrSinkEnv.ACTION_UP

    monkeypatch.setattr(recurrent, "_pipeline_memory_adjusted_navigation_action", fake_memory_adjusted)

    duplicate_action = recurrent._pipeline_local_assist_action(
        cfg,
        obs_agent,
        agent_id=0,
        current_action_id=SyncOrSinkEnv.ACTION_STAY,
        pipeline_state=duplicate_state,
    )

    assert duplicate_action == SyncOrSinkEnv.ACTION_UP
    assert calls[-1]["prefer_known_route"] is False
    assert calls[-1]["compact_avoid_positions"] is True
    assert calls[-1]["avoid_previous_backtrack_position"] is True

    single_state = {
        **duplicate_state,
        "carry_targets": {0: duplicate_state["carry_targets"][0]},
    }
    single_action = recurrent._pipeline_local_assist_action(
        cfg,
        obs_agent,
        agent_id=0,
        current_action_id=SyncOrSinkEnv.ACTION_STAY,
        pipeline_state=single_state,
    )

    assert single_action == SyncOrSinkEnv.ACTION_UP
    assert calls[-1]["prefer_known_route"] is False
    assert calls[-1]["compact_avoid_positions"] is False
    assert calls[-1]["avoid_previous_backtrack_position"] is False


def test_recurrent_pipeline_partial_non_sync_delivery_prefers_known_route(monkeypatch):
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY
    import syncorsink.train.recurrent_bc_rl as recurrent

    cfg = recurrent.RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=32,
        agents=3,
        eval_pipeline_navigation_assist=True,
    )
    obs_agent = {
        "self_pos": np.array([10, 6], dtype=np.int16),
        "inventory": np.array([4], dtype=np.int16),
        "local_grid": np.full((5, 5), TILE_EMPTY, dtype=np.int16),
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "goal_hint": np.full((16,), -1, dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    pipeline_state = {
        "completed_stages": set(),
        "delivered_counts": {6: 1},
        "delivered_resources": {6: [2]},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
        "carry_targets": {
            0: {
                "stage": 6,
                "station": (7, 6),
                "resource_type": 4,
                "required": [2, 4],
            },
        },
    }
    calls = []

    def fake_memory_adjusted(*args, **kwargs):
        calls.append(kwargs)
        return SyncOrSinkEnv.ACTION_UP

    monkeypatch.setattr(recurrent, "_pipeline_memory_adjusted_navigation_action", fake_memory_adjusted)

    action = recurrent._pipeline_local_assist_action(
        cfg,
        obs_agent,
        agent_id=0,
        current_action_id=SyncOrSinkEnv.ACTION_STAY,
        pipeline_state=pipeline_state,
    )

    assert action == SyncOrSinkEnv.ACTION_UP
    assert calls[-1]["prefer_known_route"] is True
    assert calls[-1]["compact_avoid_positions"] is False


def test_recurrent_pipeline_carry_target_uses_hint_sync_for_route_trust(monkeypatch):
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY
    import syncorsink.train.recurrent_bc_rl as recurrent

    cfg = recurrent.RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=32,
        agents=3,
        eval_pipeline_navigation_assist=True,
    )
    obs_agent = {
        "self_pos": np.array([10, 6], dtype=np.int16),
        "inventory": np.array([4], dtype=np.int16),
        "local_grid": np.full((5, 5), TILE_EMPTY, dtype=np.int16),
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "goal_hint": np.array(
            [1, 7, 6, 2, 4, 2, 0, -1, 1, -1, -1, -1, -1, -1, -1, -1],
            dtype=np.int16,
        ),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    pipeline_state = {
        "completed_stages": set(),
        "delivered_counts": {},
        "delivered_resources": {},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
        "carry_targets": {
            0: {
                "stage": 1,
                "station": (7, 6),
                "resource_type": 4,
                "required": [4, 2],
            },
        },
    }
    calls = []

    def fake_memory_adjusted(*args, **kwargs):
        calls.append(kwargs)
        return SyncOrSinkEnv.ACTION_UP

    monkeypatch.setattr(recurrent, "_pipeline_memory_adjusted_navigation_action", fake_memory_adjusted)

    action = recurrent._pipeline_local_assist_action(
        cfg,
        obs_agent,
        agent_id=0,
        current_action_id=SyncOrSinkEnv.ACTION_STAY,
        pipeline_state=pipeline_state,
    )

    assert action == SyncOrSinkEnv.ACTION_UP
    assert calls[-1]["prefer_known_route"] is False


def test_recurrent_pipeline_navigation_memory_breaks_repeated_cell_cycle():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY
    from syncorsink.train.recurrent_bc_rl import _pipeline_memory_adjusted_navigation_action

    obs_agent = {
        "self_pos": np.array([12, 7], dtype=np.int16),
        "inventory": np.array([3], dtype=np.int16),
        "local_grid": np.full((3, 3), TILE_EMPTY, dtype=np.int16),
        "local_resource_types": np.zeros((3, 3), dtype=np.int16),
        "goal_hint": np.full((16,), -1, dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    pipeline_state = {
        "navigation_memory": {
            0: {
                "target": [22, 6],
                "pos": [13, 8],
                "action": SyncOrSinkEnv.ACTION_LEFT,
                "recent": [
                    {
                        "target": [22, 6],
                        "pos": [12, 7],
                        "action": SyncOrSinkEnv.ACTION_RIGHT,
                    }
                ],
            }
        }
    }

    action = _pipeline_memory_adjusted_navigation_action(
        obs_agent,
        (22, 6),
        SyncOrSinkEnv.ACTION_RIGHT,
        pipeline_state,
        0,
    )

    assert action != SyncOrSinkEnv.ACTION_RIGHT
    assert action in {
        SyncOrSinkEnv.ACTION_UP,
        SyncOrSinkEnv.ACTION_DOWN,
        SyncOrSinkEnv.ACTION_LEFT,
    }
    assert pipeline_state["navigation_memory"][0]["recent"][-1]["action"] == action


def test_recurrent_pipeline_navigation_memory_ignores_different_route_context():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY
    from syncorsink.train.recurrent_bc_rl import _pipeline_memory_adjusted_navigation_action

    obs_agent = {
        "self_pos": np.array([4, 14], dtype=np.int16),
        "inventory": np.array([4], dtype=np.int16),
        "local_grid": np.full((3, 3), TILE_EMPTY, dtype=np.int16),
        "local_resource_types": np.zeros((3, 3), dtype=np.int16),
        "goal_hint": np.full((16,), -1, dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    pipeline_state = {
        "navigation_memory": {
            0: {
                "target": [2, 7],
                "pos": [4, 15],
                "action": SyncOrSinkEnv.ACTION_UP,
                "context": "stage:1:deliver:3",
                "recent": [
                    {
                        "target": [2, 7],
                        "pos": [4, 14],
                        "action": SyncOrSinkEnv.ACTION_UP,
                        "context": "stage:1:deliver:3",
                    }
                ],
            }
        }
    }

    action = _pipeline_memory_adjusted_navigation_action(
        obs_agent,
        (2, 7),
        SyncOrSinkEnv.ACTION_UP,
        pipeline_state,
        0,
        context_key="stage:6:deliver:4",
    )

    assert action == SyncOrSinkEnv.ACTION_UP
    assert pipeline_state["navigation_memory"][0]["context"] == "stage:6:deliver:4"
    assert (
        pipeline_state["navigation_memory"][0]["recent"][-1]["context"]
        == "stage:6:deliver:4"
    )


def test_recurrent_pipeline_navigation_memory_prefers_known_route():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY
    from syncorsink.train.recurrent_bc_rl import _pipeline_memory_adjusted_navigation_action

    obs_agent = {
        "self_pos": np.array([4, 14], dtype=np.int16),
        "inventory": np.array([4], dtype=np.int16),
        "local_grid": np.full((3, 3), TILE_EMPTY, dtype=np.int16),
        "local_resource_types": np.zeros((3, 3), dtype=np.int16),
        "goal_hint": np.full((16,), -1, dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    corridor = {(4, y) for y in range(7, 15)} | {(3, 7), (2, 7)}
    pipeline_state = {
        "terrain_memory": {
            "passable": corridor,
            "blocked": set(),
            "doors": set(),
            "map_size": 32,
        },
        "navigation_memory": {
            0: {
                "target": [2, 7],
                "pos": [4, 15],
                "action": SyncOrSinkEnv.ACTION_UP,
                "context": "stage:6:deliver:4",
                "recent": [
                    {
                        "target": [2, 7],
                        "pos": [4, 14],
                        "action": SyncOrSinkEnv.ACTION_UP,
                        "context": "stage:6:deliver:4",
                    }
                ],
            }
        },
    }

    action = _pipeline_memory_adjusted_navigation_action(
        obs_agent,
        (2, 7),
        SyncOrSinkEnv.ACTION_UP,
        pipeline_state,
        0,
        context_key="stage:6:deliver:4",
        prefer_known_route=True,
    )

    assert action == SyncOrSinkEnv.ACTION_UP


def test_recurrent_pipeline_navigation_routes_to_door_frontier_when_target_path_unknown():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_DOOR, TILE_EMPTY
    from syncorsink.train.recurrent_bc_rl import _pipeline_navigation_action_from_obs

    target = (22, 6)
    loop_positions = {
        (12, 5),
        (13, 5),
        (12, 6),
        (13, 6),
        (12, 7),
        (13, 7),
        (12, 8),
        (13, 8),
    }
    pipeline_state = {
        "terrain_memory": {
            "passable": {
                *loop_positions,
                (13, 9),
                (13, 10),
                (13, 11),
            },
            "blocked": {(13, 12)},
            "doors": {(13, 12)},
            "map_size": 32,
        }
    }
    obs_agent = {
        "self_pos": np.array([13, 8], dtype=np.int16),
        "inventory": np.array([3], dtype=np.int16),
        "local_grid": np.full((3, 3), TILE_EMPTY, dtype=np.int16),
        "local_resource_types": np.zeros((3, 3), dtype=np.int16),
        "goal_hint": np.full((16,), -1, dtype=np.int16),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }

    detour = _pipeline_navigation_action_from_obs(
        obs_agent,
        target,
        pipeline_state,
        avoid_positions=loop_positions,
    )

    assert detour == SyncOrSinkEnv.ACTION_DOWN

    door_obs = {
        **obs_agent,
        "self_pos": np.array([13, 11], dtype=np.int16),
        "local_grid": np.array(
            [
                [TILE_EMPTY, TILE_EMPTY, TILE_EMPTY],
                [TILE_EMPTY, TILE_EMPTY, TILE_EMPTY],
                [TILE_EMPTY, TILE_DOOR, TILE_EMPTY],
            ],
            dtype=np.int16,
        ),
    }

    door_action = _pipeline_navigation_action_from_obs(
        door_obs,
        target,
        pipeline_state,
    )

    assert door_action == SyncOrSinkEnv.ACTION_INTERACT


def test_recurrent_pipeline_navigation_uses_terrain_memory_to_route_to_station():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _pipeline_local_assist_action

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_navigation_assist=True,
    )
    obs_agent = {
        "self_pos": np.array([1, 1], dtype=np.int16),
        "inventory": np.array([2], dtype=np.int16),
        "local_grid": np.full((3, 3), TILE_EMPTY, dtype=np.int16),
        "local_resource_types": np.zeros((3, 3), dtype=np.int16),
        "goal_hint": np.array(
            [0, 3, 3, 1, 2, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            dtype=np.int16,
        ),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    pipeline_state = {
        "completed_stages": set(),
        "delivered_counts": {},
        "delivered_resources": {},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
        "carry_targets": {},
        "terrain_memory": {
            "passable": {(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)},
            "blocked": {(2, 1), (2, 2)},
            "map_size": 8,
        },
    }

    action = _pipeline_local_assist_action(
        cfg,
        obs_agent,
        agent_id=0,
        current_action_id=SyncOrSinkEnv.ACTION_STAY,
        pipeline_state=pipeline_state,
    )

    assert action == SyncOrSinkEnv.ACTION_DOWN


def test_recurrent_pipeline_terrain_memory_skips_unknown_cells():
    from syncorsink.envs.maps import TILE_EMPTY, TILE_UNKNOWN, TILE_WALL
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _update_pipeline_resource_memory_from_obs,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=16,
        agents=1,
    )
    obs = {
        0: {
            "self_pos": np.array([2, 2], dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": np.array(
                [
                    [TILE_UNKNOWN, TILE_UNKNOWN, TILE_UNKNOWN],
                    [TILE_UNKNOWN, TILE_EMPTY, TILE_WALL],
                    [TILE_UNKNOWN, TILE_EMPTY, TILE_UNKNOWN],
                ],
                dtype=np.int16,
            ),
            "local_resource_types": np.zeros((3, 3), dtype=np.int16),
            "goal_hint": np.full((16,), -1, dtype=np.int16),
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }
    pipeline_state = {}

    _update_pipeline_resource_memory_from_obs(cfg, pipeline_state, obs)

    terrain = pipeline_state["terrain_memory"]
    assert (3, 2) in terrain["blocked"]
    assert (2, 2) in terrain["passable"]
    assert (2, 3) in terrain["passable"]
    assert (1, 1) not in terrain["passable"]


def test_recurrent_pipeline_resource_memory_ignores_unknown_cells_when_pruning():
    from syncorsink.envs.maps import TILE_EMPTY, TILE_UNKNOWN
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _update_pipeline_resource_memory_from_obs,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=32,
        agents=1,
    )
    local_grid = np.full((5, 5), TILE_EMPTY, dtype=np.int16)
    local_grid[0, 0] = TILE_UNKNOWN
    obs = {
        0: {
            "self_pos": np.array([8, 24], dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": local_grid,
            "local_resource_types": np.zeros((5, 5), dtype=np.int16),
            "goal_hint": np.full((16,), -1, dtype=np.int16),
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }
    pipeline_state = {"resource_memory": {2: {(6, 22), (6, 23)}}}

    _update_pipeline_resource_memory_from_obs(cfg, pipeline_state, obs)

    assert (6, 22) in pipeline_state["resource_memory"][2]
    assert (6, 23) not in pipeline_state["resource_memory"][2]


def test_recurrent_pipeline_frontier_exploration_assist_searches_for_missing_resource():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_frontier_exploration_assist,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        obs_exploration_memory=True,
        eval_pipeline_frontier_exploration_assist=True,
    )
    explored = np.ones((8, 8), dtype=np.int8)
    explored[4, 5] = 0
    obs = {
        0: {
            "self_pos": np.array([4, 4], dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": np.full((3, 3), TILE_EMPTY, dtype=np.int16),
            "local_resource_types": np.zeros((3, 3), dtype=np.int16),
            "goal_hint": np.array(
                [0, 6, 6, 1, 2, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.int16,
            ),
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "explored_mask": explored,
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }

    corrected = _apply_pipeline_frontier_exploration_assist(
        cfg,
        obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
    )

    assert int(corrected[0].item()) == SyncOrSinkEnv.ACTION_RIGHT


def test_recurrent_pipeline_frontier_exploration_assist_opens_search_doors():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_DOOR, TILE_EMPTY
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_frontier_exploration_assist,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=32,
        agents=1,
        obs_exploration_memory=True,
        eval_pipeline_frontier_exploration_assist=True,
    )
    local_grid = np.full((5, 5), TILE_EMPTY, dtype=np.int16)
    local_grid[2, 3] = TILE_DOOR
    explored = np.ones((32, 32), dtype=np.int8)
    obs = {
        0: {
            "self_pos": np.array([20, 16], dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": local_grid,
            "local_resource_types": np.zeros((5, 5), dtype=np.int16),
            "goal_hint": np.array(
                [0, 21, 23, 1, 4, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.int16,
            ),
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "explored_mask": explored,
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }

    corrected = _apply_pipeline_frontier_exploration_assist(
        cfg,
        obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY], dtype=torch.long),
    )

    assert int(corrected[0].item()) == SyncOrSinkEnv.ACTION_INTERACT


def test_recurrent_pipeline_frontier_exploration_assist_spreads_agents():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_pipeline_frontier_exploration_assist,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=16,
        agents=3,
        obs_exploration_memory=True,
        eval_pipeline_frontier_exploration_assist=True,
    )
    explored = np.ones((16, 16), dtype=np.int8)
    explored[0, 1] = 0
    explored[0, 14] = 0
    explored[15, 7] = 0
    base_obs = {
        "self_pos": np.array([7, 7], dtype=np.int16),
        "inventory": np.array([0], dtype=np.int16),
        "local_grid": np.full((5, 5), TILE_EMPTY, dtype=np.int16),
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "goal_hint": np.array(
            [0, 6, 6, 1, 2, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            dtype=np.int16,
        ),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "explored_mask": explored,
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    obs = {aid: {**base_obs, "explored_mask": explored.copy()} for aid in range(3)}

    corrected = _apply_pipeline_frontier_exploration_assist(
        cfg,
        obs,
        torch.tensor([SyncOrSinkEnv.ACTION_STAY] * 3, dtype=torch.long),
    )

    np.testing.assert_array_equal(
        corrected.numpy(),
        np.array(
            [
                SyncOrSinkEnv.ACTION_LEFT,
                SyncOrSinkEnv.ACTION_RIGHT,
                SyncOrSinkEnv.ACTION_DOWN,
            ],
            dtype=np.int64,
        ),
    )


def test_recurrent_pipeline_navigation_uses_remembered_required_resource():
    from syncorsink.envs import SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _pipeline_local_assist_action,
        _update_pipeline_resource_memory_from_obs,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        eval_pipeline_navigation_assist=True,
    )
    obs_agent = {
        "self_pos": np.array([6, 11], dtype=np.int16),
        "inventory": np.array([0], dtype=np.int16),
        "local_grid": np.full((5, 5), TILE_EMPTY, dtype=np.int16),
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "goal_hint": np.array(
            [1, 7, 11, 2, 1, 2, 0, -1, 1, -1, -1, -1, -1, -1, -1, -1],
            dtype=np.int16,
        ),
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "explored_mask": np.ones((16, 16), dtype=np.int8),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    visible_resource_obs = {
        0: {
            **obs_agent,
            "self_pos": np.array([3, 12], dtype=np.int16),
            "local_resource_types": np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.int16,
            ),
        }
    }
    memory_state = {"resource_memory": {}}
    _update_pipeline_resource_memory_from_obs(cfg, memory_state, visible_resource_obs)
    assert (2, 13) in memory_state["resource_memory"][2]

    pipeline_state = {
        "completed_stages": {0},
        "delivered_counts": {1: 1},
        "delivered_resources": {1: [1]},
        "sync_wait_stages": set(),
        "sync_wait_stations": {},
        "carry_targets": {},
        "resource_memory": {2: {(2, 13)}},
    }

    action = _pipeline_local_assist_action(
        cfg,
        obs_agent,
        agent_id=0,
        current_action_id=SyncOrSinkEnv.ACTION_STAY,
        pipeline_state=pipeline_state,
    )

    assert action == SyncOrSinkEnv.ACTION_LEFT


def test_recurrent_checkpoint_policy_updates_pipeline_resource_memory_from_obs():
    from syncorsink.envs.maps import TILE_EMPTY
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentCheckpointPolicy,
        RecurrentConfig,
        _build_recurrent_obs_batch,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=16,
        agents=1,
        fov_preset="easy",
        obs_exploration_memory=True,
        obs_pipeline_features=True,
        hidden_dim=16,
        comm=False,
        eval_pipeline_navigation_assist=True,
    )
    obs = {
        0: {
            "self_pos": np.array([8, 8], dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "local_grid": np.full((5, 5), TILE_EMPTY, dtype=np.int16),
            "local_resource_types": np.zeros((5, 5), dtype=np.int16),
            "local_node_types": np.zeros((5, 5), dtype=np.int16),
            "local_node_energy": np.zeros((5, 5), dtype=np.int16),
            "goal_hint": np.array(
                [0, 7, 7, 1, 2, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
                dtype=np.int16,
            ),
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "message_from": np.full((1,), -1, dtype=np.int16),
            "explored_mask": np.ones((16, 16), dtype=np.int8),
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }
    obs[0]["local_resource_types"][2, 3] = 2
    obs_dim = _build_recurrent_obs_batch(obs, 1, cfg).shape[1]
    model = MAPPORecurrentActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    )
    policy = RecurrentCheckpointPolicy(model, cfg, torch.device("cpu"))

    policy(obs, {}, {"step": 0})

    assert (9, 8) in policy.pipeline_state["resource_memory"][2]


def test_pipeline_frontier_exploration_labels_search_for_required_resource():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _pipeline_frontier_exploration_action_label_mask,
    )

    env = SyncOrSinkEnv(
        SyncOrSinkConfig(
            scenario="pipeline_assembly",
            map_size=8,
            num_agents=1,
            fov_preset="easy",
            obs_exploration_memory=True,
            pipeline_stage_count=1,
            pipeline_required_per_stage_min=1,
            pipeline_required_per_stage_max=1,
            pipeline_sync_probability=0.0,
            pipeline_dependency_probability=0.0,
        )
    )
    env.reset(seed=3)
    stage = env.scenario_state.data["stages"][0]
    station = tuple(int(v) for v in stage["station"])
    needed_type = int(stage["required"][0])
    hint = np.full((64,), -1, dtype=np.int16)
    hint[:9] = np.array(
        [int(stage["stage"]), station[0], station[1], 1, needed_type, -1, 0, -1, 0],
        dtype=np.int16,
    )
    explored_mask = np.ones((8, 8), dtype=np.int8)
    explored_mask[2, 3] = 0
    obs = {
        0: {
            "local_grid": np.zeros((5, 5), dtype=np.int16),
            "local_resource_types": np.zeros((5, 5), dtype=np.int16),
            "inventory": np.array([0], dtype=np.int16),
            "self_pos": np.array([3, 3], dtype=np.int16),
            "goal_hint": hint,
            "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
            "explored_mask": explored_mask,
            "action_mask": np.ones((8,), dtype=np.float32),
        }
    }
    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=1,
        obs_exploration_memory=True,
        bc_pipeline_frontier_exploration_action_loss_weight=0.5,
    )

    mask, action_id = _pipeline_frontier_exploration_action_label_mask(env, obs, cfg)

    np.testing.assert_array_equal(mask, np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(action_id, np.array([env.ACTION_UP], dtype=np.int64))

    visible_obs = {0: {**obs[0], "local_resource_types": obs[0]["local_resource_types"].copy()}}
    visible_obs[0]["local_resource_types"][2, 2] = needed_type
    mask, action_id = _pipeline_frontier_exploration_action_label_mask(env, visible_obs, cfg)

    np.testing.assert_array_equal(mask, np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(action_id, np.array([-1], dtype=np.int64))


def test_pipeline_frontier_exploration_labels_spread_agents():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _pipeline_frontier_exploration_action_label_mask,
    )

    env = SyncOrSinkEnv(
        SyncOrSinkConfig(
            scenario="pipeline_assembly",
            map_size=16,
            num_agents=3,
            fov_preset="easy",
            obs_exploration_memory=True,
            pipeline_stage_count=1,
            pipeline_required_per_stage_min=1,
            pipeline_required_per_stage_max=1,
            pipeline_sync_probability=0.0,
            pipeline_dependency_probability=0.0,
        )
    )
    env.reset(seed=3)
    stage = env.scenario_state.data["stages"][0]
    station = tuple(int(v) for v in stage["station"])
    needed_type = int(stage["required"][0])
    hint = np.full((64,), -1, dtype=np.int16)
    hint[:9] = np.array(
        [int(stage["stage"]), station[0], station[1], 1, needed_type, -1, 0, -1, 0],
        dtype=np.int16,
    )
    explored = np.ones((16, 16), dtype=np.int8)
    explored[0, 1] = 0
    explored[0, 14] = 0
    explored[15, 7] = 0
    base_obs = {
        "local_grid": np.zeros((5, 5), dtype=np.int16),
        "local_resource_types": np.zeros((5, 5), dtype=np.int16),
        "inventory": np.array([0], dtype=np.int16),
        "self_pos": np.array([7, 7], dtype=np.int16),
        "goal_hint": hint,
        "messages_tokens": np.full((1, 8), -1, dtype=np.int16),
        "explored_mask": explored,
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    obs = {aid: {**base_obs, "explored_mask": explored.copy()} for aid in range(3)}
    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=16,
        agents=3,
        obs_exploration_memory=True,
        bc_pipeline_frontier_exploration_action_loss_weight=0.5,
    )

    mask, action_id = _pipeline_frontier_exploration_action_label_mask(env, obs, cfg)

    np.testing.assert_array_equal(mask, np.ones((3,), dtype=np.float32))
    np.testing.assert_array_equal(
        action_id,
        np.array([env.ACTION_LEFT, env.ACTION_RIGHT, env.ACTION_DOWN], dtype=np.int64),
    )


def test_pipeline_assisted_action_label_mask_unions_corrections_and_trusted_labels():
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _pipeline_assisted_action_label_mask,
    )

    mask, action_id = _pipeline_assisted_action_label_mask(
        RecurrentConfig(scenario="pipeline_assembly"),
        np.array([1, 2, 3], dtype=np.int64),
        np.array([False, True, False]),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(mask, np.array([1.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(action_id, np.array([1, 2, 3], dtype=np.int64))

    non_pipeline_mask, non_pipeline_action_id = _pipeline_assisted_action_label_mask(
        RecurrentConfig(scenario="signal_hunt"),
        np.array([1, 2, 3], dtype=np.int64),
        np.array([True, True, True]),
    )

    np.testing.assert_array_equal(
        non_pipeline_mask,
        np.zeros((3,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        non_pipeline_action_id,
        np.array([-1, -1, -1], dtype=np.int64),
    )


def test_recurrent_pipeline_plan_action_and_message_labels_follow_trusted_plan():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.envs.maps import TILE_EMPTY
    from syncorsink.train.recurrent_bc_rl import (
        PIPELINE_OPTION_DELIVER,
        RecurrentConfig,
        _apply_pipeline_plan_broadcast_assist,
        _apply_pipeline_plan_broadcast_overrides,
        _append_labeled_step,
        _finalize_episode_sequence,
        _initial_pipeline_state,
        _new_episode_sequence,
        _pipeline_bad_action_label_masks,
        _pipeline_interact_gate_label_mask,
        _pipeline_option_label_mask,
        _pipeline_plan_action_label_mask,
        _pipeline_rollout_plan_action_label_mask,
        _pipeline_rollout_station_guard_action_label_mask,
        _pipeline_rollout_wrong_station_recovery_action_label_mask,
        _pipeline_station_interact_guard_action,
        _pipeline_station_guard_action_label_mask,
    )

    env = SyncOrSinkEnv(
        SyncOrSinkConfig(
            scenario="pipeline_assembly",
            map_size=8,
            num_agents=2,
            fov_preset="easy",
            pipeline_stage_count=1,
            pipeline_required_per_stage_min=1,
            pipeline_required_per_stage_max=1,
            pipeline_sync_probability=0.0,
            pipeline_dependency_probability=0.0,
            comm_token_limit=8,
            token_vocab_size=32,
        )
    )
    env.reset(seed=0)
    stage = env.scenario_state.data["stages"][0]
    station = tuple(int(v) for v in stage["station"])
    needed_type = int(stage["required"][0])
    env.agent_positions[0] = station
    env.inventories[0] = needed_type
    obs = env._build_observations()
    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        obs_pipeline_features=True,
        bc_pipeline_proactive_bad_action_labels=True,
    )
    silent_actions = {
        0: {"action": env.ACTION_STAY, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    }
    pipeline_state = _initial_pipeline_state(cfg)
    broadcasted_actions, broadcasters = _apply_pipeline_plan_broadcast_overrides(
        cfg,
        obs,
        silent_actions,
        pipeline_state=pipeline_state,
        current_step=0,
    )
    assert broadcasters == [0]
    assert broadcasted_actions[0]["message_tokens"] == [
        12,
        int(stage["stage"]),
        station[0],
        station[1],
        1,
        needed_type,
    ]
    assert broadcasted_actions[1]["message_tokens"] == []
    rebroadcasted_actions, rebroadcasters = _apply_pipeline_plan_broadcast_overrides(
        cfg,
        obs,
        silent_actions,
        pipeline_state=pipeline_state,
        current_step=1,
    )
    assert rebroadcasters == []
    assert rebroadcasted_actions[0]["message_tokens"] == []

    noisy_actions = {
        0: {"action": env.ACTION_STAY, "message_tokens": [12, 0, 0, 0, 1, 2]},
        1: {"action": env.ACTION_STAY, "message_tokens": [12, 0, 0, 0, 1, 2]},
    }
    assisted_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        eval_pipeline_plan_broadcast_assist=True,
    )
    assisted_actions = _apply_pipeline_plan_broadcast_assist(
        assisted_cfg,
        obs,
        noisy_actions,
        pipeline_state=_initial_pipeline_state(assisted_cfg),
        current_step=0,
    )
    assert assisted_actions[0]["message_tokens"] == [
        12,
        int(stage["stage"]),
        station[0],
        station[1],
        1,
        needed_type,
    ]
    assert assisted_actions[1]["message_tokens"] == []

    message_tokens = [12, int(stage["stage"]), station[0], station[1], 1, needed_type]
    actions = {
        0: {"action": env.ACTION_INTERACT, "message_tokens": message_tokens},
        1: {"action": env.ACTION_STAY},
    }

    mask, action_id = _pipeline_plan_action_label_mask(env, obs, actions, cfg)

    np.testing.assert_array_equal(mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(action_id, np.array([env.ACTION_INTERACT, -1], dtype=np.int64))
    rollout_plan_mask, rollout_plan_action_id = _pipeline_rollout_plan_action_label_mask(
        env,
        obs,
        cfg,
        pipeline_state={
            "completed_stages": set(),
            "delivered_counts": {},
            "delivered_resources": {},
            "sync_wait_stages": set(),
        },
    )
    np.testing.assert_array_equal(
        rollout_plan_mask,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        rollout_plan_action_id,
        np.array([env.ACTION_INTERACT, -1], dtype=np.int64),
    )
    option_mask, option_id = _pipeline_option_label_mask(env, obs, actions, cfg)
    np.testing.assert_array_equal(option_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        option_id,
        np.array([PIPELINE_OPTION_DELIVER, 0], dtype=np.int64),
    )

    message_obs = dict(obs[0])
    message_obs["goal_hint"] = np.full_like(message_obs["goal_hint"], -1)
    padded_message = np.full((1, cfg.comm_token_limit), -1, dtype=np.int16)
    padded_message[0, : len(message_tokens)] = np.array(message_tokens, dtype=np.int16)
    message_obs["messages_tokens"] = padded_message
    message_only_obs = {**obs, 0: message_obs}
    mask, action_id = _pipeline_plan_action_label_mask(env, message_only_obs, actions, cfg)

    np.testing.assert_array_equal(mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(action_id, np.array([env.ACTION_INTERACT, -1], dtype=np.int64))

    ep_data = _new_episode_sequence()
    _append_labeled_step(ep_data, obs, actions, env, cfg)
    assert ep_data["pipeline_plan_action_mask"] == [1.0, 0.0]
    assert ep_data["pipeline_plan_action_id"] == [env.ACTION_INTERACT, -1]
    assert ep_data["pipeline_option_mask"] == [1.0, 0.0]
    assert ep_data["pipeline_option_id"] == [PIPELINE_OPTION_DELIVER, 0]
    assert ep_data["pipeline_message_mask"] == [1.0, 0.0]
    assert ep_data["pipeline_send_gate_mask"] == [1.0, 1.0]
    assert ep_data["pipeline_send_gate_label"] == [1.0, 0.0]
    assert ep_data["pipeline_interact_gate_mask"] == [1.0, 0.0]
    assert ep_data["pipeline_interact_gate_label"] == [1.0, 0.0]
    assert ep_data["pipeline_sync_action_mask"] == [0.0, 0.0]
    assert ep_data["pipeline_sync_action_id"] == [-1, -1]
    assert ep_data["pipeline_station_guard_action_mask"] == [0.0, 0.0]
    assert ep_data["pipeline_station_guard_action_id"] == [-1, -1]

    episode = _finalize_episode_sequence(ep_data, env, cfg)
    np.testing.assert_array_equal(
        episode["pipeline_plan_action_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        episode["pipeline_plan_action_id"],
        np.array([[env.ACTION_INTERACT, -1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        episode["pipeline_option_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        episode["pipeline_option_id"],
        np.array([[PIPELINE_OPTION_DELIVER, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        episode["pipeline_message_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        episode["pipeline_send_gate_mask"],
        np.array([[1.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        episode["pipeline_send_gate_label"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        episode["pipeline_interact_gate_mask"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        episode["pipeline_interact_gate_label"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        episode["pipeline_sync_action_mask"],
        np.array([[0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        episode["pipeline_sync_action_id"],
        np.array([[-1, -1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        episode["pipeline_station_guard_action_mask"],
        np.array([[0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        episode["pipeline_station_guard_action_id"],
        np.array([[-1, -1]], dtype=np.int64),
    )

    wrong_station = next(pos for pos in env.meta["stations"] if tuple(pos) != station)
    env.agent_positions[0] = tuple(wrong_station)
    env.inventories[0] = needed_type
    wrong_station_obs = env._build_observations()
    stay_actions = {
        0: {"action": env.ACTION_STAY, "message_tokens": message_tokens},
        1: {"action": env.ACTION_STAY},
    }
    (
        bad_pickup_mask,
        bad_pickup_id,
        bad_drop_mask,
        bad_drop_id,
        bad_interact_mask,
        bad_interact_id,
    ) = (
        _pipeline_bad_action_label_masks(env, wrong_station_obs, stay_actions, cfg)
    )

    np.testing.assert_array_equal(bad_pickup_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_pickup_id, np.array([-1, -1], dtype=np.int64))
    np.testing.assert_array_equal(bad_drop_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_drop_id, np.array([-1, -1], dtype=np.int64))
    np.testing.assert_array_equal(bad_interact_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        bad_interact_id,
        np.array([env.ACTION_INTERACT, -1], dtype=np.int64),
    )
    guard_action_id = _pipeline_station_interact_guard_action(
        cfg,
        wrong_station_obs[0],
        pipeline_state=env.scenario_state.data,
    )
    assert guard_action_id is not None
    assert int(guard_action_id) in {
        env.ACTION_UP,
        env.ACTION_DOWN,
        env.ACTION_LEFT,
        env.ACTION_RIGHT,
    }
    assert int(wrong_station_obs[0]["action_mask"][int(guard_action_id)]) == 1
    (
        rollout_wrong_station_mask,
        rollout_wrong_station_action_id,
    ) = _pipeline_rollout_wrong_station_recovery_action_label_mask(
        cfg,
        wrong_station_obs,
        pipeline_state=env.scenario_state.data,
    )
    np.testing.assert_array_equal(
        rollout_wrong_station_mask,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert int(rollout_wrong_station_action_id[0]) in {
        env.ACTION_UP,
        env.ACTION_DOWN,
        env.ACTION_LEFT,
        env.ACTION_RIGHT,
    }
    assert int(wrong_station_obs[0]["action_mask"][int(rollout_wrong_station_action_id[0])]) == 1

    unneeded_resource_pos, unneeded_type = next(
        (pos, int(resource_type))
        for pos, resource_type in env.scenario_state.data["resource_types"].items()
        if int(resource_type) != needed_type
    )
    env.agent_positions[0] = tuple(unneeded_resource_pos)
    env.inventories[0] = 0
    unneeded_obs = env._build_observations()
    (
        bad_pickup_mask,
        bad_pickup_id,
        bad_drop_mask,
        bad_drop_id,
        bad_interact_mask,
        bad_interact_id,
    ) = _pipeline_bad_action_label_masks(env, unneeded_obs, stay_actions, cfg)

    assert unneeded_type != needed_type
    np.testing.assert_array_equal(bad_pickup_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_pickup_id, np.array([env.ACTION_PICKUP, -1], dtype=np.int64))
    np.testing.assert_array_equal(bad_drop_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_drop_id, np.array([-1, -1], dtype=np.int64))
    np.testing.assert_array_equal(bad_interact_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_interact_id, np.array([-1, -1], dtype=np.int64))

    env.agent_positions[0] = station
    env.inventories[0] = unneeded_type
    station_recovery_obs = env._build_observations()
    mask, action_id = _pipeline_plan_action_label_mask(env, station_recovery_obs, stay_actions, cfg)

    np.testing.assert_array_equal(mask, np.array([1.0, 0.0], dtype=np.float32))
    assert int(action_id[0]) in {
        env.ACTION_UP,
        env.ACTION_DOWN,
        env.ACTION_LEFT,
        env.ACTION_RIGHT,
    }
    assert int(station_recovery_obs[0]["action_mask"][int(action_id[0])]) == 1
    assert int(action_id[1]) == -1
    recovery_action_id = int(action_id[0])

    conservative_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
        comm_token_limit=8,
        comm_vocab_size=32,
        obs_pipeline_features=True,
    )
    mask, action_id = _pipeline_plan_action_label_mask(
        env,
        station_recovery_obs,
        stay_actions,
        conservative_cfg,
    )

    np.testing.assert_array_equal(mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(action_id, np.array([-1, -1], dtype=np.int64))

    (
        bad_pickup_mask,
        bad_pickup_id,
        bad_drop_mask,
        bad_drop_id,
        bad_interact_mask,
        bad_interact_id,
    ) = _pipeline_bad_action_label_masks(env, station_recovery_obs, stay_actions, cfg)

    np.testing.assert_array_equal(bad_pickup_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_pickup_id, np.array([-1, -1], dtype=np.int64))
    np.testing.assert_array_equal(bad_drop_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_drop_id, np.array([-1, -1], dtype=np.int64))
    np.testing.assert_array_equal(bad_interact_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        bad_interact_id,
        np.array([env.ACTION_INTERACT, -1], dtype=np.int64),
    )
    station_guard_mask, station_guard_action_id = _pipeline_station_guard_action_label_mask(
        env,
        station_recovery_obs,
        cfg,
    )
    np.testing.assert_array_equal(station_guard_mask, np.array([1.0, 0.0], dtype=np.float32))
    assert int(station_guard_action_id[0]) == recovery_action_id

    env.agent_positions[0] = station
    env.inventories[0] = 0
    empty_station_obs = env._build_observations()
    gate_mask, gate_label = _pipeline_interact_gate_label_mask(env, empty_station_obs)
    np.testing.assert_array_equal(gate_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(gate_label, np.array([0.0, 0.0], dtype=np.float32))
    station_guard_mask, station_guard_action_id = _pipeline_station_guard_action_label_mask(
        env,
        empty_station_obs,
        cfg,
    )
    assert station_guard_mask[1] == 0.0
    if float(station_guard_mask[0]) > 0.0:
        assert int(station_guard_action_id[0]) not in {env.ACTION_INTERACT, env.ACTION_STAY}
        assert int(empty_station_obs[0]["action_mask"][int(station_guard_action_id[0])]) == 1
    rollout_guard_mask, rollout_guard_action_id = (
        _pipeline_rollout_station_guard_action_label_mask(
            cfg,
            empty_station_obs,
            pipeline_state={
                "completed_stages": set(),
                "delivered_counts": {},
                "delivered_resources": {},
                "sync_wait_stages": set(),
            },
        )
    )
    np.testing.assert_array_equal(
        rollout_guard_mask,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert int(rollout_guard_action_id[0]) in {
        env.ACTION_UP,
        env.ACTION_DOWN,
        env.ACTION_LEFT,
        env.ACTION_RIGHT,
        env.ACTION_DROP,
        env.ACTION_STAY,
    }
    assert int(rollout_guard_action_id[0]) != env.ACTION_INTERACT
    (
        bad_pickup_mask,
        bad_pickup_id,
        bad_drop_mask,
        bad_drop_id,
        bad_interact_mask,
        bad_interact_id,
    ) = _pipeline_bad_action_label_masks(env, empty_station_obs, stay_actions, cfg)

    np.testing.assert_array_equal(bad_pickup_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_pickup_id, np.array([-1, -1], dtype=np.int64))
    np.testing.assert_array_equal(bad_drop_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_drop_id, np.array([-1, -1], dtype=np.int64))
    np.testing.assert_array_equal(bad_interact_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        bad_interact_id,
        np.array([env.ACTION_INTERACT, -1], dtype=np.int64),
    )

    empty_pos = (0, 0)
    env.grid[empty_pos[1], empty_pos[0]] = TILE_EMPTY
    env.scenario_state.data["resource_types"].pop(empty_pos, None)
    env.agent_positions[0] = empty_pos
    env.inventories[0] = unneeded_type
    recovery_obs = env._build_observations()
    drop_actions = {
        0: {"action": env.ACTION_DROP, "message_tokens": message_tokens},
        1: {"action": env.ACTION_STAY},
    }
    mask, action_id = _pipeline_plan_action_label_mask(env, recovery_obs, drop_actions, cfg)

    np.testing.assert_array_equal(mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(action_id, np.array([env.ACTION_DROP, -1], dtype=np.int64))

    env.agent_positions[0] = empty_pos
    env.inventories[0] = needed_type
    empty_obs = env._build_observations()
    (
        bad_pickup_mask,
        bad_pickup_id,
        bad_drop_mask,
        bad_drop_id,
        bad_interact_mask,
        bad_interact_id,
    ) = (
        _pipeline_bad_action_label_masks(env, empty_obs, stay_actions, cfg)
    )

    np.testing.assert_array_equal(bad_pickup_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_pickup_id, np.array([-1, -1], dtype=np.int64))
    np.testing.assert_array_equal(bad_drop_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_drop_id, np.array([env.ACTION_DROP, -1], dtype=np.int64))
    np.testing.assert_array_equal(bad_interact_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(bad_interact_id, np.array([-1, -1], dtype=np.int64))

    ep_data = _new_episode_sequence()
    _append_labeled_step(ep_data, unneeded_obs, stay_actions, env, cfg)
    assert ep_data["pipeline_bad_pickup_action_mask"] == [1.0, 0.0]
    assert ep_data["pipeline_bad_pickup_action_id"] == [env.ACTION_PICKUP, -1]

    ep_data = _new_episode_sequence()
    _append_labeled_step(ep_data, empty_obs, stay_actions, env, cfg)
    assert ep_data["pipeline_bad_drop_action_mask"] == [1.0, 0.0]
    assert ep_data["pipeline_bad_drop_action_id"] == [env.ACTION_DROP, -1]


def test_recurrent_pipeline_ppo_bad_pickup_reward_shaping():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import (
        _apply_pipeline_bad_interact_reward_shaping,
        _apply_pipeline_bad_pickup_reward_shaping,
        _pipeline_bad_interact_agents,
        _pipeline_bad_pickup_agents,
        _pipeline_resource_need_status,
        _pipeline_unneeded_drop_agents,
    )

    env = SyncOrSinkEnv(
        SyncOrSinkConfig(
            scenario="pipeline_assembly",
            map_size=8,
            num_agents=2,
            fov_preset="easy",
            pipeline_stage_count=1,
            pipeline_required_per_stage_min=1,
            pipeline_required_per_stage_max=1,
            pipeline_sync_probability=0.0,
            pipeline_dependency_probability=0.0,
        )
    )
    env.reset(seed=0)
    stage = env.scenario_state.data["stages"][0]
    needed_type = int(stage["required"][0])
    unneeded_pos, unneeded_type = next(
        (pos, int(resource_type))
        for pos, resource_type in env.scenario_state.data["resource_types"].items()
        if int(resource_type) != needed_type
    )

    env.agent_positions[0] = tuple(unneeded_pos)
    env.inventories[0] = 0
    pickup_actions = {
        0: {"action": env.ACTION_PICKUP, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    }

    assert _pipeline_resource_need_status(env, unneeded_type) == "not_required"
    assert _pipeline_bad_pickup_agents(env, pickup_actions) == [0]

    _obs, rewards, _done, _truncated, info = env.step(pickup_actions)
    count, penalty_sum, drop_count, bonus_sum = _apply_pipeline_bad_pickup_reward_shaping(
        rewards,
        info=info,
        num_agents=2,
        bad_pickup_candidates=[0],
        unneeded_drop_candidates=[],
        bad_pickup_penalty=0.2,
        unneeded_drop_bonus=0.1,
    )

    assert count == 1
    assert penalty_sum == pytest.approx(0.2)
    assert drop_count == 0
    assert bonus_sum == 0.0
    assert rewards[0] == pytest.approx(-0.2)
    assert rewards[1] == pytest.approx(0.0)

    drop_actions = {
        0: {"action": env.ACTION_DROP, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    }
    assert _pipeline_unneeded_drop_agents(env, drop_actions) == [0]

    _obs, rewards, _done, _truncated, info = env.step(drop_actions)
    count, penalty_sum, drop_count, bonus_sum = _apply_pipeline_bad_pickup_reward_shaping(
        rewards,
        info=info,
        num_agents=2,
        bad_pickup_candidates=[],
        unneeded_drop_candidates=[0],
        bad_pickup_penalty=0.2,
        unneeded_drop_bonus=0.1,
    )

    assert count == 0
    assert penalty_sum == 0.0
    assert drop_count == 1
    assert bonus_sum == pytest.approx(0.1)
    assert rewards[0] == pytest.approx(0.1)
    assert rewards[1] == pytest.approx(0.0)

    station = tuple(stage["station"])
    empty_interact_actions = {
        0: {"action": env.ACTION_INTERACT, "message_tokens": []},
        1: {"action": env.ACTION_STAY, "message_tokens": []},
    }
    env.agent_positions[0] = station
    env.inventories[0] = 0
    assert _pipeline_bad_interact_agents(env, empty_interact_actions) == [0]
    rewards = {0: 0.0, 1: 0.0}
    count, penalty_sum = _apply_pipeline_bad_interact_reward_shaping(
        rewards,
        bad_interact_candidates=[0],
        bad_interact_penalty=0.15,
    )
    assert count == 1
    assert penalty_sum == pytest.approx(0.15)
    assert rewards[0] == pytest.approx(-0.15)

    env.agent_positions[0] = station
    env.inventories[0] = unneeded_type
    assert _pipeline_bad_interact_agents(env, empty_interact_actions) == [0]

    env.agent_positions[0] = station
    env.inventories[0] = needed_type
    assert _pipeline_bad_interact_agents(env, empty_interact_actions) == []


def test_recurrent_signal_features_decode_targets_and_keep_mask_safe():
    from syncorsink.envs.maps import TILE_TARGET
    from syncorsink.train.mappo import action_mask_from_flat_obs
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _flatten_recurrent_obs,
        _signal_coordination_features,
        _signal_targets_from_tokens,
    )

    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 2] = TILE_TARGET
    explored8 = np.zeros((8, 8), dtype=np.int8)
    explored8[3, 3] = 1
    obs_agent = {
        "local_grid": local_grid,
        "self_pos": np.array([3, 3], dtype=np.int16),
        "explored_mask": explored8,
        "goal_hint": np.array([26, 6, 3, -1, -1, -1, -1, -1], dtype=np.int16),
        "messages_tokens": np.array([
            [26, 5, 4, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ], dtype=np.int16),
        "message_from": np.array([1, -1], dtype=np.int16),
        "action_mask": np.array([1, 0, 1, 0, 1, 0, 0, 0], dtype=np.float32),
    }
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_exploration_memory=True,
        obs_feedback=True,
        obs_signal_features=True,
        obs_memory_mode="egocentric",
        obs_memory_radius=1,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
    )

    features = _signal_coordination_features(obs_agent, cfg, observed_map_size=8)
    flat8 = _flatten_recurrent_obs(obs_agent, cfg, feedback=np.zeros((12,), dtype=np.float32))
    obs_agent_16 = dict(obs_agent)
    explored16 = np.zeros((16, 16), dtype=np.int8)
    explored16[3, 3] = 1
    obs_agent_16["explored_mask"] = explored16
    flat16 = _flatten_recurrent_obs(obs_agent_16, cfg, feedback=np.zeros((12,), dtype=np.float32))

    assert _signal_targets_from_tokens([22, 0, 5, 5, -1, -2, -1], observed_map_size=8) == [(4, 3)]
    assert features.shape == (38,)
    assert features[0] == 1.0
    assert features[1] == pytest.approx(3 / 7)
    assert features[4] == 1.0
    assert features[5] == pytest.approx(2 / 7)
    assert features[6] == pytest.approx(1 / 7)
    assert features[12] == 1.0
    assert features[13] == 0.0
    assert features[14] == 1.0
    assert features[15] == 1.0
    expected_constraint_tail = np.zeros((22,), dtype=np.float32)
    expected_constraint_tail[3] = 1.0
    np.testing.assert_allclose(features[16:], expected_constraint_tail)
    assert flat8.shape == flat16.shape
    expected_mask = torch.tensor(obs_agent["action_mask"], dtype=torch.float32)
    assert torch.equal(action_mask_from_flat_obs(torch.tensor(flat8).unsqueeze(0))[0], expected_mask)


def test_recurrent_signal_features_decode_constraint_grammar():
    from syncorsink.envs.maps import TILE_BEACON, TILE_TARGET, TILE_WATER
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _signal_coordination_features

    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 2] = TILE_TARGET
    obs_agent = {
        "local_grid": local_grid,
        "self_pos": np.array([3, 3], dtype=np.int16),
        "goal_hint": np.array([
            21, TILE_BEACON, 6, 5, 4,
            23, 1, 3, 8,
            24, 0,
            25, 1,
            -1, -1, -1,
        ], dtype=np.int16),
        "messages_tokens": np.array([
            [21, TILE_WATER, 2, 3, 2, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ], dtype=np.int16),
        "message_from": np.array([1, -1], dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_signal_features=True,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
    )

    features = _signal_coordination_features(obs_agent, cfg, observed_map_size=8)
    constraint = features[16:]

    assert features.shape == (38,)
    assert constraint.shape == (22,)
    assert constraint[0] == 1.0  # nearest near constraint present
    assert constraint[1] == pytest.approx(-1 / 7)
    assert constraint[2] == pytest.approx(0.0)
    assert constraint[4] == pytest.approx(2 / 7)
    assert constraint[5] == 1.0  # water object from nearest message constraint
    assert constraint[6] == 0.0
    assert constraint[8] == 1.0  # parity present
    assert constraint[9] == 1.0
    assert constraint[10] == 1.0  # quadrant present
    assert constraint[14] == 1.0  # SE one-hot
    assert constraint[15] == 1.0  # quadrant size normalized to map size
    assert constraint[16] == 1.0
    assert constraint[17] == 0.0
    assert constraint[18] == 1.0
    assert constraint[19] == 1.0
    assert constraint[20] == 1.0
    assert constraint[21] == 1.0


def test_recurrent_signal_features_include_inferred_target_candidates():
    from syncorsink.envs.maps import TILE_WATER
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _signal_coordination_features,
        _signal_inferred_constraint_targets,
    )

    obs_agent = {
        "local_grid": np.zeros((5, 5), dtype=np.int16),
        "self_pos": np.array([0, 0], dtype=np.int16),
        "goal_hint": np.array([21, TILE_WATER, 2, 2, 0, -1, -1, -1], dtype=np.int16),
        "messages_tokens": np.array([
            [23, 0, 0, 8, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ], dtype=np.int16),
        "message_from": np.array([1, -1], dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_signal_features=True,
        obs_signal_inferred_target_features=True,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
    )

    assert _signal_inferred_constraint_targets(obs_agent, observed_map_size=8) == [(2, 2)]
    features = _signal_coordination_features(obs_agent, cfg, observed_map_size=8)
    inferred = features[38:44]

    assert features.shape == (44,)
    np.testing.assert_allclose(
        inferred,
        np.array([1.0, 2 / 7, 2 / 7, 4 / 7, 1 / 32, 0.0], dtype=np.float32),
    )


def test_recurrent_signal_features_include_target_confidence_state():
    from syncorsink.envs.maps import TILE_TARGET
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _signal_coordination_features

    local_grid = np.zeros((5, 5), dtype=np.int16)
    local_grid[2, 2] = TILE_TARGET
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        obs_signal_features=True,
        obs_signal_confidence_features=True,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
    )

    exact_obs = {
        "local_grid": local_grid,
        "self_pos": np.array([2, 2], dtype=np.int16),
        "goal_hint": np.array([26, 2, 2, -1, -1, -1, -1, -1], dtype=np.int16),
        "messages_tokens": np.full((2, 8), -1, dtype=np.int16),
        "message_from": np.array([-1, -1], dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    exact_features = _signal_coordination_features(exact_obs, cfg, observed_map_size=8)
    exact_confidence = exact_features[38:52]

    assert exact_features.shape == (52,)
    np.testing.assert_allclose(
        exact_confidence,
        np.array([
            1.0,  # center is a target tile
            1.0,  # center matches an exact target hint
            1.0,  # center is safe to scan from exact/unique evidence
            1.0,  # center is compatible with the current evidence
            0.0,  # not merely ambiguous
            0.0,  # not rejected
            0.0,  # not unknown
            1.0,  # target information exists
            0.0,  # exact coordinates are not a constraint grammar
            0.0,
            0.25,
            0.125,
            0.125,
            0.0,
        ], dtype=np.float32),
    )

    ambiguous_obs = {
        "local_grid": local_grid,
        "self_pos": np.array([2, 2], dtype=np.int16),
        "goal_hint": np.array([23, 0, 0, 8, -1, -1, -1, -1], dtype=np.int16),
        "messages_tokens": np.full((2, 8), -1, dtype=np.int16),
        "message_from": np.array([-1, -1], dtype=np.int16),
        "action_mask": np.ones((8,), dtype=np.float32),
    }
    ambiguous_features = _signal_coordination_features(ambiguous_obs, cfg, observed_map_size=8)
    ambiguous_confidence = ambiguous_features[38:52]

    assert ambiguous_features.shape == (52,)
    np.testing.assert_allclose(
        ambiguous_confidence,
        np.array([
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.4,
            0.0,
            0.125,
            0.125,
            0.0,
        ], dtype=np.float32),
    )


def test_recurrent_agent_role_features_include_signal_search_anchor():
    from syncorsink.train.mappo import action_mask_from_flat_obs
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _agent_role_features,
        _flatten_recurrent_obs,
    )

    obs_agent = {
        "local_grid": np.zeros((5, 5), dtype=np.int16),
        "self_pos": np.array([7, 7], dtype=np.int16),
        "action_mask": np.array([1, 0, 1, 0, 1, 0, 0, 0], dtype=np.float32),
    }
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=16,
        agents=4,
        obs_agent_id_features=True,
    )

    features = _agent_role_features(obs_agent, cfg, observed_map_size=16, agent_id=2)
    flat = _flatten_recurrent_obs(obs_agent, cfg, agent_id=2)

    assert features.shape == (9,)
    np.testing.assert_allclose(features[:4], np.array([0.0, 0.0, 1.0, 0.0]))
    assert features[4] == pytest.approx(2 / 3)
    np.testing.assert_allclose(
        features[5:],
        np.array([1.0, 0.0, 8 / 15, 8 / 15], dtype=np.float32),
    )
    expected_mask = torch.tensor(obs_agent["action_mask"], dtype=torch.float32)
    assert torch.equal(action_mask_from_flat_obs(torch.tensor(flat).unsqueeze(0))[0], expected_mask)


def test_recurrent_signal_sector_features_track_assigned_frontier_progress():
    from syncorsink.train.mappo import action_mask_from_flat_obs
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _flatten_recurrent_obs,
        _signal_cell_in_agent_sector,
        _signal_sector_search_features,
    )

    explored = np.ones((16, 16), dtype=np.int8)
    explored[:, 10:] = 0
    obs_agent = {
        "local_grid": np.zeros((5, 5), dtype=np.int16),
        "self_pos": np.array([7, 7], dtype=np.int16),
        "explored_mask": explored,
        "action_mask": np.array([1, 0, 1, 0, 1, 0, 0, 0], dtype=np.float32),
    }
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=16,
        agents=4,
        obs_exploration_memory=True,
        obs_memory_mode="egocentric",
        obs_memory_radius=2,
        obs_signal_sector_features=True,
    )

    features = _signal_sector_search_features(obs_agent, cfg, observed_map_size=16, agent_id=1)
    flat = _flatten_recurrent_obs(obs_agent, cfg, agent_id=1)

    assert features.shape == (10,)
    np.testing.assert_allclose(
        features[:4],
        np.array([1.0, 2 / 15, 0.0, 2 / 15], dtype=np.float32),
    )
    assert features[4] == pytest.approx(10 / 16)
    assert features[5] == pytest.approx(0.25)
    assert features[6] == pytest.approx(1.0)
    assert features[7] == pytest.approx(0.0)
    assert features[8] == pytest.approx(8 / 15)
    assert features[9] == pytest.approx(2 / 15)
    assert _signal_cell_in_agent_sector((12, 4), agent_id=1, num_agents=4, width=16, height=16)
    assert not _signal_cell_in_agent_sector((4, 12), agent_id=1, num_agents=4, width=16, height=16)

    expected_mask = torch.tensor(obs_agent["action_mask"], dtype=torch.float32)
    assert torch.equal(action_mask_from_flat_obs(torch.tensor(flat).unsqueeze(0))[0], expected_mask)


def test_recurrent_labeled_steps_preserve_agent_role_features():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _append_labeled_step,
        _flatten_recurrent_obs,
        _new_episode_sequence,
    )

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=16,
        num_agents=4,
        fov_preset="medium",
        max_steps=40,
    ))
    obs, _ = env.reset(seed=123)
    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=16,
        agents=4,
        fov_preset="medium",
        obs_agent_id_features=True,
    )
    actions = {
        aid: {"action": env.ACTION_STAY, "message_tokens": []}
        for aid in range(env.num_agents)
    }
    ep_data = _new_episode_sequence()

    _append_labeled_step(ep_data, obs, actions, env, cfg)

    for aid in range(env.num_agents):
        expected = _flatten_recurrent_obs(obs[aid], cfg, agent_id=aid)
        no_agent_id = _flatten_recurrent_obs(obs[aid], cfg)
        np.testing.assert_allclose(ep_data["obs"][aid], expected)
        if aid > 0:
            assert not np.allclose(ep_data["obs"][aid], no_agent_id)


def test_recurrent_checkpoint_policy_egocentric_memory_cross_map_size(tmp_path):
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.recurrent_bc_rl import (
        PIPELINE_OPTION_NONE,
        RecurrentConfig,
        _build_env,
        _build_recurrent_obs_batch,
        _feedback_matrix,
        load_recurrent_checkpoint_policy,
    )

    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=20,
        obs_exploration_memory=True,
        obs_feedback=True,
        obs_normalize_tokens=True,
        obs_memory_mode="egocentric",
        obs_memory_radius=2,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=8,
        comm_max_messages=4,
        hidden_dim=16,
        eval_send_threshold=0.25,
    )
    env8 = _build_env(cfg)
    obs8, _ = env8.reset(seed=0)
    obs_dim = _build_recurrent_obs_batch(
        obs8,
        env8.num_agents,
        cfg,
        feedback=_feedback_matrix(cfg, env8.num_agents),
    ).shape[1]
    model = MAPPORecurrentActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    )
    checkpoint = tmp_path / "recurrent_egocentric.pt"
    torch.save({"model": model.state_dict(), "config": vars(cfg)}, checkpoint)

    policy = load_recurrent_checkpoint_policy(checkpoint, device="cpu")
    env16 = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=16,
        num_agents=2,
        fov_preset="easy",
        max_steps=20,
        obs_exploration_memory=True,
        comm_token_limit=4,
        token_vocab_size=8,
        max_messages=4,
    ))
    obs16, info16 = env16.reset(seed=1)
    actions = policy(obs16, info16, {"step": 0})

    assert sorted(actions) == [0, 1]
    assert all(0 <= int(action["action"]) < 8 for action in actions.values())


def test_recurrent_pipeline_feedback_legacy_checkpoint_auto_downgrades_metadata(tmp_path):
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.recurrent_bc_rl import (
        PIPELINE_OPTION_NONE,
        RecurrentConfig,
        _build_env,
        _build_recurrent_obs_batch,
        _feedback_matrix,
        load_recurrent_checkpoint_policy,
    )

    legacy_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=40,
        obs_feedback=True,
        obs_pipeline_feedback=True,
        obs_pipeline_feedback_metadata=False,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=16,
        comm_max_messages=4,
        hidden_dim=16,
    )
    env = _build_env(legacy_cfg)
    obs, info = env.reset(seed=0)
    obs_dim = _build_recurrent_obs_batch(
        obs,
        env.num_agents,
        legacy_cfg,
        feedback=_feedback_matrix(legacy_cfg, env.num_agents),
    ).shape[1]
    model = MAPPORecurrentActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=legacy_cfg.hidden_dim,
        comm_enabled=legacy_cfg.comm,
        comm_token_limit=legacy_cfg.comm_token_limit,
        comm_vocab_size=legacy_cfg.comm_vocab_size,
    )
    raw_cfg = vars(legacy_cfg).copy()
    raw_cfg.pop("obs_pipeline_feedback_metadata", None)
    checkpoint = tmp_path / "legacy_pipeline_feedback.pt"
    torch.save({"model": model.state_dict(), "config": raw_cfg}, checkpoint)

    policy = load_recurrent_checkpoint_policy(checkpoint, device="cpu")
    actions = policy(obs, info, {"step": 0})

    assert policy.cfg.obs_pipeline_feedback_metadata is False
    assert sorted(actions) == [0, 1]


def test_recurrent_training_map_sizes_helpers():
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _cfg_for_map_size,
        _cfg_for_training_episode,
        _eval_map_sizes,
        _training_map_schedule,
        _training_map_sizes,
    )

    cfg = RecurrentConfig(
        map_size=8,
        max_steps=60,
        train_map_sizes="8, 16,32",
        eval_map_sizes="16, 32",
        map_max_steps="16:120,32:240",
    )

    assert _training_map_sizes(cfg) == [8, 16, 32]
    assert _training_map_schedule(cfg) == [8, 16, 32]
    assert _eval_map_sizes(cfg) == [16, 32]
    assert _cfg_for_training_episode(cfg, 0).map_size == 8
    assert _cfg_for_training_episode(cfg, 0).max_steps == 60
    assert _cfg_for_training_episode(cfg, 1).map_size == 16
    assert _cfg_for_training_episode(cfg, 1).max_steps == 120
    assert _cfg_for_training_episode(cfg, 2).map_size == 32
    assert _cfg_for_training_episode(cfg, 2).max_steps == 240
    assert _cfg_for_training_episode(cfg, 3).map_size == 8
    assert _cfg_for_map_size(cfg, 16).max_steps == 120

    weighted_cfg = RecurrentConfig(**{**vars(cfg), "train_map_sampling_weights": "8:1,16:1,32:3"})
    assert _training_map_schedule(weighted_cfg) == [8, 16, 32, 32, 32]
    assert [_cfg_for_training_episode(weighted_cfg, idx).map_size for idx in range(7)] == [
        8,
        16,
        32,
        32,
        32,
        8,
        16,
    ]
    assert _cfg_for_training_episode(weighted_cfg, 3).max_steps == 240

    bad_cfg = RecurrentConfig(**{**vars(cfg), "train_map_sampling_weights": "64:2"})
    with pytest.raises(ValueError, match="not present in train_map_sizes"):
        _training_map_schedule(bad_cfg)


def test_signal_hint_comm_channel_warning_for_clipped_protocol():
    import warnings as py_warnings

    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _warn_if_signal_hint_comm_channel_is_too_small,
    )

    small = RecurrentConfig(
        scenario="signal_hunt",
        oracle_type="signal_hint_comm",
        train_map_sizes="8,16,32",
        eval_map_sizes="8,16,32",
        comm_token_limit=4,
        comm_vocab_size=8,
    )
    ok = RecurrentConfig(
        scenario="signal_hunt",
        oracle_type="signal_hint_comm",
        train_map_sizes="8,16,32",
        eval_map_sizes="8,16,32",
        comm_token_limit=8,
        comm_vocab_size=32,
    )

    with pytest.warns(UserWarning, match="clip or alias oracle messages"):
        _warn_if_signal_hint_comm_channel_is_too_small(small)
    with py_warnings.catch_warnings(record=True) as caught:
        py_warnings.simplefilter("always")
        _warn_if_signal_hint_comm_channel_is_too_small(ok)
    assert not caught


def test_recurrent_eval_map_sizes_aggregate_smoke():
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _build_recurrent_obs_batch,
        _build_training_env,
        _feedback_matrix,
        evaluate_recurrent_policy,
    )

    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=10,
        obs_exploration_memory=True,
        obs_feedback=True,
        obs_normalize_tokens=True,
        obs_memory_mode="egocentric",
        obs_memory_radius=2,
        obs_navigation_features=True,
        comm=False,
        hidden_dim=16,
        eval_episodes=1,
        eval_seed=123,
        eval_map_sizes="8,16",
    )
    env, active_cfg = _build_training_env(cfg, 0)
    obs, _ = env.reset(seed=0)
    obs_dim = _build_recurrent_obs_batch(
        obs,
        env.num_agents,
        active_cfg,
        feedback=_feedback_matrix(active_cfg, env.num_agents),
    ).shape[1]
    model = MAPPORecurrentActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    )

    result = evaluate_recurrent_policy(cfg, model, torch.device("cpu"))

    assert result["episodes"] == 2
    assert set(result["eval_map_sizes"]) == {"8", "16"}
    assert result["eval_map_sizes"]["8"]["episodes"] == 1
    assert result["eval_map_sizes"]["16"]["episodes"] == 1
    assert "success_rate" in result


def test_recurrent_eval_wandb_payload_includes_per_map_metrics():
    from syncorsink.train.recurrent_bc_rl import _recurrent_eval_wandb_payload

    result = {
        "episodes": 4,
        "success_rate": 0.25,
        "avg_return": 1.5,
        "avg_steps": 42.0,
        "avg_comm_tokens": 3.0,
        "eval_seed_count": 2,
        "signal": {"avg_decoy_scans": 2.0},
        "eval_map_sizes": {
            "8": {
                "success_rate": 0.5,
                "avg_return": 4.0,
                "avg_steps": 20.0,
                "avg_comm_tokens": 2.0,
                "signal": {"avg_target_scans": 7.0},
            },
            "16": {
                "success_rate": 0.0,
                "avg_return": -1.0,
                "avg_steps": 64.0,
                "avg_comm_tokens": 4.0,
                "signal": {"avg_target_scans": 1.0},
            },
        },
    }

    payload = _recurrent_eval_wandb_payload(
        result,
        update=9,
        is_best=False,
        best_eval={"success_rate": 0.75, "avg_return": 5.0, "update": 3},
    )

    assert payload["eval/success_rate"] == 0.25
    assert payload["eval/mean_comm_tokens"] == 3.0
    assert payload["eval/episodes"] == 4
    assert payload["eval/seed_count"] == 2
    assert payload["eval/best_success_rate"] == 0.75
    assert payload["eval/best_update"] == 3
    assert payload["eval/signal/avg_decoy_scans"] == 2.0
    assert payload["eval/map_8/success_rate"] == 0.5
    assert payload["eval/map_8/signal/avg_target_scans"] == 7.0
    assert payload["eval/map_16/mean_steps"] == 64.0


def test_recurrent_eval_multi_seed_aggregates_per_map_metrics(monkeypatch):
    import syncorsink.train.recurrent_bc_rl as recurrent
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig

    rows_by_seed = {
        123: {
            "episodes": 4,
            "success_rate": 0.25,
            "avg_return": 1.0,
            "avg_steps": 10.0,
            "avg_comm_tokens": 2.0,
            "signal": {"avg_decoy_scans": 4.0},
            "eval_map_sizes": {
                "8": {
                    "episodes": 2,
                    "success_rate": 0.5,
                    "avg_return": 2.0,
                    "avg_steps": 8.0,
                    "avg_comm_tokens": 1.0,
                    "signal": {"avg_decoy_scans": 2.0},
                },
                "16": {
                    "episodes": 2,
                    "success_rate": 0.0,
                    "avg_return": 0.0,
                    "avg_steps": 12.0,
                    "avg_comm_tokens": 3.0,
                    "signal": {"avg_decoy_scans": 6.0},
                },
            },
        },
        10123: {
            "episodes": 4,
            "success_rate": 0.75,
            "avg_return": 5.0,
            "avg_steps": 20.0,
            "avg_comm_tokens": 4.0,
            "signal": {"avg_decoy_scans": 0.0},
            "eval_map_sizes": {
                "8": {
                    "episodes": 2,
                    "success_rate": 1.0,
                    "avg_return": 6.0,
                    "avg_steps": 16.0,
                    "avg_comm_tokens": 5.0,
                    "signal": {"avg_decoy_scans": 0.0},
                },
                "16": {
                    "episodes": 2,
                    "success_rate": 0.5,
                    "avg_return": 4.0,
                    "avg_steps": 24.0,
                    "avg_comm_tokens": 3.0,
                    "signal": {"avg_decoy_scans": 0.0},
                },
            },
        },
    }
    seen_seeds = []

    def fake_evaluate_recurrent_policy(cfg, model, device):
        del model, device
        seen_seeds.append(cfg.eval_seed)
        return rows_by_seed[cfg.eval_seed]

    monkeypatch.setattr(recurrent, "evaluate_recurrent_policy", fake_evaluate_recurrent_policy)

    result = recurrent.evaluate_recurrent_policy_multi_seed(
        RecurrentConfig(eval_seed=123, eval_episodes=2, eval_map_sizes="8,16"),
        model=object(),
        device=torch.device("cpu"),
        seed_count=2,
    )

    assert seen_seeds == [123, 10123]
    assert result["eval_seed_count"] == 2
    assert result["eval_seeds"] == [123, 10123]
    assert result["episodes"] == 8
    assert result["success_rate"] == pytest.approx(0.5)
    assert result["avg_return"] == pytest.approx(3.0)
    assert result["signal"]["avg_decoy_scans"] == pytest.approx(2.0)
    assert result["eval_map_sizes"]["8"]["episodes"] == 4
    assert result["eval_map_sizes"]["8"]["success_rate"] == pytest.approx(0.75)
    assert result["eval_map_sizes"]["16"]["avg_return"] == pytest.approx(2.0)


def test_recurrent_eval_multi_seed_accepts_map_seed_schedule(monkeypatch):
    import syncorsink.train.recurrent_bc_rl as recurrent
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig

    seen = []

    def fake_single_map_eval(cfg, model, device):
        del model, device
        seen.append((int(cfg.map_size), int(cfg.eval_seed)))
        success_rate = 1.0 if (int(cfg.map_size), int(cfg.eval_seed)) in {(16, 3000), (32, 7000)} else 0.0
        return {
            "episodes": int(cfg.eval_episodes),
            "success_rate": success_rate,
            "avg_return": float(cfg.map_size),
            "avg_steps": float(cfg.eval_seed % 100),
            "avg_comm_tokens": 0.0,
        }

    monkeypatch.setattr(recurrent, "_evaluate_recurrent_policy_single_map", fake_single_map_eval)

    result = recurrent.evaluate_recurrent_policy_multi_seed(
        RecurrentConfig(eval_map_sizes="16,32", eval_episodes=2),
        model=object(),
        device=torch.device("cpu"),
        seed_count=99,
        seed_list="16:3000,13000+32:7000",
    )

    assert seen == [(16, 3000), (16, 13000), (32, 7000)]
    assert result["episodes"] == 6
    assert result["eval_seed_count"] == 3
    assert result["eval_seeds"] == [3000, 7000, 13000]
    assert result["eval_seed_lists"] == {"16": [3000, 13000], "32": [7000]}
    assert result["success_rate"] == pytest.approx(2 / 3)
    assert result["eval_map_sizes"]["16"]["episodes"] == 4
    assert result["eval_map_sizes"]["16"]["success_rate"] == pytest.approx(0.5)
    assert result["eval_map_sizes"]["32"]["episodes"] == 2
    assert result["eval_map_sizes"]["32"]["eval_seeds"] == [7000]

    with pytest.raises(ValueError, match="missing seed lists"):
        recurrent.evaluate_recurrent_policy_multi_seed(
            RecurrentConfig(eval_map_sizes="16,32", eval_episodes=1),
            model=object(),
            device=torch.device("cpu"),
            seed_count=1,
            seed_list="16:3000",
            seed_list_field_name="eval_seed_list",
        )


def test_recurrent_rl_balanced_rollout_collects_each_train_map_size():
    from syncorsink.policies.mappo_models import MAPPOCritic, MAPPORecurrentActor
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _balanced_rollout_step_counts_for_maps,
        _balanced_step_counts,
        _build_recurrent_obs_batch,
        _build_training_env,
        _collect_recurrent_rl_rollout,
        _feedback_matrix,
    )

    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        train_map_sizes="8,16,32",
        map_max_steps="8:20,16:20,32:20",
        agents=2,
        fov_preset="easy",
        max_steps=20,
        obs_exploration_memory=True,
        obs_feedback=True,
        obs_normalize_tokens=True,
        obs_memory_mode="egocentric",
        obs_memory_radius=2,
        comm=False,
        hidden_dim=16,
        rollout_steps=6,
        rl_balanced_rollouts=True,
        rl_rollout_eval_decoding=True,
    )
    env, active_cfg = _build_training_env(cfg, 0)
    obs, _ = env.reset(seed=0)
    obs_dim = _build_recurrent_obs_batch(
        obs,
        env.num_agents,
        active_cfg,
        feedback=_feedback_matrix(active_cfg, env.num_agents),
    ).shape[1]
    model = MAPPORecurrentActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    )
    critic = MAPPOCritic(obs_dim, hidden_dim=cfg.hidden_dim)

    rollout = _collect_recurrent_rl_rollout(
        cfg,
        model,
        critic,
        torch.device("cpu"),
        update=0,
        num_agents=env.num_agents,
    )

    assert _balanced_step_counts(7, 3) == [3, 2, 2]
    assert _balanced_rollout_step_counts_for_maps(cfg, [8, 16, 32]) == [2, 2, 2]
    assert len(rollout["obs_buf"]) == 6
    assert rollout["balanced"] is True
    assert rollout["map_step_counts"] == {"8": 2, "16": 2, "32": 2}
    assert rollout["eval_decoding_action_correction_count"] == 0
    assert rollout["eval_decoding_action_opportunities"] == 12
    assert len(rollout["eval_decoding_action_correction_mask_buf"]) == 6
    assert sum(int(mask.sum().item()) for mask in rollout["eval_decoding_action_correction_mask_buf"]) == 0
    assert rollout["pipeline_assisted_action_label_count"] == 0
    assert len(rollout["pipeline_assisted_action_mask_buf"]) == 6
    assert len(rollout["pipeline_assisted_action_id_buf"]) == 6
    assert sum(int(mask.sum().item()) for mask in rollout["pipeline_assisted_action_mask_buf"]) == 0
    assert rollout["pipeline_interact_gate_label_count"] == 0
    assert rollout["pipeline_interact_gate_positive_label_count"] == 0
    assert rollout["pipeline_interact_gate_negative_label_count"] == 0
    assert len(rollout["pipeline_interact_gate_mask_buf"]) == 6
    assert len(rollout["pipeline_interact_gate_label_buf"]) == 6
    assert sum(int(mask.sum().item()) for mask in rollout["pipeline_interact_gate_mask_buf"]) == 0
    assert rollout["pipeline_pickup_gate_label_count"] == 0
    assert rollout["pipeline_pickup_gate_positive_label_count"] == 0
    assert rollout["pipeline_pickup_gate_negative_label_count"] == 0
    assert len(rollout["pipeline_pickup_gate_mask_buf"]) == 6
    assert len(rollout["pipeline_pickup_gate_label_buf"]) == 6
    assert sum(int(mask.sum().item()) for mask in rollout["pipeline_pickup_gate_mask_buf"]) == 0
    assert rollout["pipeline_delivery_progress_action_label_count"] == 0
    assert len(rollout["pipeline_delivery_progress_action_mask_buf"]) == 6
    assert len(rollout["pipeline_delivery_progress_action_id_buf"]) == 6
    assert (
        sum(
            int(mask.sum().item())
            for mask in rollout["pipeline_delivery_progress_action_mask_buf"]
        )
        == 0
    )
    assert rollout["pipeline_navigation_action_label_count"] == 0
    assert len(rollout["pipeline_navigation_action_mask_buf"]) == 6
    assert len(rollout["pipeline_navigation_action_id_buf"]) == 6
    assert sum(int(mask.sum().item()) for mask in rollout["pipeline_navigation_action_mask_buf"]) == 0
    assert rollout["pipeline_sync_action_label_count"] == 0
    assert len(rollout["pipeline_sync_action_mask_buf"]) == 6
    assert len(rollout["pipeline_sync_action_id_buf"]) == 6
    assert sum(int(mask.sum().item()) for mask in rollout["pipeline_sync_action_mask_buf"]) == 0
    assert rollout["pipeline_ready_interact_action_label_count"] == 0
    assert len(rollout["pipeline_ready_interact_action_mask_buf"]) == 6
    assert len(rollout["pipeline_ready_interact_action_id_buf"]) == 6
    assert (
        sum(int(mask.sum().item()) for mask in rollout["pipeline_ready_interact_action_mask_buf"])
        == 0
    )
    assert rollout["pipeline_station_guard_action_label_count"] == 0
    assert len(rollout["pipeline_station_guard_action_mask_buf"]) == 6
    assert len(rollout["pipeline_station_guard_action_id_buf"]) == 6
    assert sum(int(mask.sum().item()) for mask in rollout["pipeline_station_guard_action_mask_buf"]) == 0
    assert rollout["pipeline_wrong_station_recovery_action_label_count"] == 0
    assert len(rollout["pipeline_wrong_station_recovery_action_mask_buf"]) == 6
    assert len(rollout["pipeline_wrong_station_recovery_action_id_buf"]) == 6
    assert (
        sum(
            int(mask.sum().item())
            for mask in rollout["pipeline_wrong_station_recovery_action_mask_buf"]
        )
        == 0
    )
    assert rollout["pipeline_plan_action_label_count"] == 0
    assert len(rollout["pipeline_plan_action_mask_buf"]) == 6
    assert len(rollout["pipeline_plan_action_id_buf"]) == 6
    assert sum(int(mask.sum().item()) for mask in rollout["pipeline_plan_action_mask_buf"]) == 0
    assert rollout["reset_after_buf"][1] is True
    assert rollout["reset_after_buf"][3] is True
    assert rollout["reset_after_buf"][5] is True
    assert rollout["ep_returns"] == []
    assert rollout["partial_ep_steps"] == [2, 2, 2]
    assert len(rollout["partial_ep_returns"]) == 3
    assert len(rollout["partial_ep_comm"]) == 3

    weighted_cfg = RecurrentConfig(**{**vars(cfg), "rl_rollout_map_steps": "8:2,16:3,32:4"})
    assert _balanced_rollout_step_counts_for_maps(weighted_cfg, [8, 16, 32]) == [2, 3, 4]
    weighted_rollout = _collect_recurrent_rl_rollout(
        weighted_cfg,
        model,
        critic,
        torch.device("cpu"),
        update=0,
        num_agents=env.num_agents,
    )

    assert len(weighted_rollout["obs_buf"]) == 9
    assert weighted_rollout["balanced"] is True
    assert weighted_rollout["map_step_counts"] == {"8": 2, "16": 3, "32": 4}
    assert weighted_rollout["reset_after_buf"][1] is True
    assert weighted_rollout["reset_after_buf"][4] is True
    assert weighted_rollout["reset_after_buf"][8] is True
    assert weighted_rollout["ep_returns"] == []
    assert weighted_rollout["partial_ep_steps"] == [2, 3, 4]

    bad_cfg = RecurrentConfig(**{**vars(cfg), "rl_rollout_map_steps": "64:5"})
    with pytest.raises(ValueError, match="not present in train_map_sizes"):
        _balanced_rollout_step_counts_for_maps(bad_cfg, [8, 16, 32])


def test_recurrent_comm_reference_kl_tracks_all_comm_heads():
    from syncorsink.train.recurrent_bc_rl import _recurrent_comm_reference_kl

    torch.manual_seed(0)
    send_logits = torch.randn(3, 1)
    token_logits = torch.randn(3, 4, 7)
    len_logits = torch.randn(3, 5)

    same = _recurrent_comm_reference_kl(
        send_logits,
        token_logits,
        len_logits,
        send_logits,
        token_logits,
        len_logits,
    )

    shifted_send_logits = send_logits + torch.tensor([[0.5], [-0.25], [0.75]])
    shifted_token_logits = token_logits.clone()
    shifted_token_logits[..., 0] += 0.5
    shifted_len_logits = len_logits.clone()
    shifted_len_logits[:, 1] -= 0.5
    shifted = _recurrent_comm_reference_kl(
        shifted_send_logits,
        shifted_token_logits,
        shifted_len_logits,
        send_logits,
        token_logits,
        len_logits,
    )

    assert same.item() == pytest.approx(0.0, abs=1e-7)
    assert shifted.item() > 0.0


def test_recurrent_action_class_balance_weights_rare_actions():
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _recurrent_action_class_weights

    episodes = [{
        "actions": np.array(
            [
                [0, 0, 0, 5],
                [0, 6, 7, 7],
            ],
            dtype=np.int64,
        )
    }]

    disabled = _recurrent_action_class_weights(
        episodes,
        RecurrentConfig(bc_action_class_balance=False),
        torch.device("cpu"),
    )
    weights = _recurrent_action_class_weights(
        episodes,
        RecurrentConfig(
            bc_action_class_balance=True,
            bc_action_class_balance_max_weight=1.5,
        ),
        torch.device("cpu"),
    )

    assert disabled is None
    assert weights is not None
    assert weights[5].item() > weights[0].item()
    assert weights[6].item() > weights[0].item()
    assert weights.max().item() <= 1.5 + 1e-6
    assert weights[1].item() == pytest.approx(1.0)


def test_recurrent_bc_event_action_weights_latest_event_agents():
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _scale_latest_bc_event_action_weights,
    )

    ep_data = {"step_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]}
    info = {
        "events": {
            0: [{"event": "delivered"}],
            "1": [{"event": "ignored"}],
            2: [{"event": "recharged"}, {"event": "sync_complete"}],
        }
    }
    cfg = RecurrentConfig(
        bc_event_action_weight=4.0,
        bc_event_action_events="delivered,recharged,sync_complete",
    )

    scaled, event_counts = _scale_latest_bc_event_action_weights(
        ep_data,
        info,
        cfg,
        num_agents=3,
    )
    disabled_scaled, disabled_counts = _scale_latest_bc_event_action_weights(
        {"step_weights": [1.0, 1.0, 1.0]},
        info,
        RecurrentConfig(bc_event_action_weight=1.0),
        num_agents=3,
    )

    assert scaled == 2
    assert event_counts == {"delivered": 1, "recharged": 1, "sync_complete": 1}
    assert ep_data["step_weights"] == [1.0, 1.0, 1.0, 4.0, 1.0, 4.0]
    assert disabled_scaled == 0
    assert disabled_counts == {}


def test_recurrent_dagger_failed_effective_ratio_cap_scales_failed_rollouts():
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _apply_dagger_failed_effective_ratio_cap,
        _episode_count_effective_transitions,
    )

    expert = {
        "source": "expert",
        "success": True,
        "weight": 1.0,
        "obs": np.zeros((2, 2, 1), dtype=np.float32),
        "step_weights": np.ones((2, 2), dtype=np.float32),
    }
    failed = {
        "source": "dagger",
        "success": False,
        "weight": 0.25,
        "obs": np.zeros((10, 2, 1), dtype=np.float32),
        "step_weights": np.ones((10, 2), dtype=np.float32),
    }
    episodes = [expert, failed]

    row = _apply_dagger_failed_effective_ratio_cap(
        episodes,
        RecurrentConfig(dagger_failed_effective_ratio_cap=0.25),
    )
    disabled = _apply_dagger_failed_effective_ratio_cap(
        [dict(expert), dict(failed)],
        RecurrentConfig(dagger_failed_effective_ratio_cap=-1.0),
    )

    assert row["applied"] is True
    assert row["reference_effective_transitions"] == pytest.approx(4.0)
    assert row["failed_dagger_effective_transitions_before"] == pytest.approx(5.0)
    assert row["failed_dagger_effective_transitions_after"] == pytest.approx(1.0)
    assert row["scale"] == pytest.approx(0.2)
    assert failed["weight"] == pytest.approx(0.05)
    assert _episode_count_effective_transitions(episodes) == pytest.approx(5.0)
    assert disabled["enabled"] is False
    assert disabled["applied"] is False


def test_recurrent_comm_length_loss_ignores_no_message_examples():
    from syncorsink.train.recurrent_bc_rl import (
        _mix_oracle_rollin_actions,
        _mix_oracle_rollin_messages,
        _recurrent_comm_loss,
        _recurrent_comm_loss_components,
        _send_threshold_for_target_rate,
    )

    model_actions = {
        0: {"action": 1, "message_tokens": [1, 2]},
        1: {"action": 2, "message_tokens": []},
    }
    oracle_actions = {
        0: {"action": 3, "message_tokens": [7]},
        1: {"action": 4, "message_tokens": [8, 9]},
    }
    mixed, replaced_agents, replaced_tokens = _mix_oracle_rollin_messages(
        model_actions,
        oracle_actions,
        1.0,
        np.random.default_rng(0),
    )
    assert {aid: action["action"] for aid, action in mixed.items()} == {0: 1, 1: 2}
    assert mixed[0]["message_tokens"] == [7]
    assert mixed[1]["message_tokens"] == [8, 9]
    assert replaced_agents == 2
    assert replaced_tokens == 3
    unchanged, replaced_agents, replaced_tokens = _mix_oracle_rollin_messages(
        model_actions,
        oracle_actions,
        0.0,
        np.random.default_rng(0),
    )
    assert unchanged[0]["message_tokens"] == [1, 2]
    assert unchanged[1]["message_tokens"] == []
    assert replaced_agents == 0
    assert replaced_tokens == 0

    action_mixed, action_replaced_agents = _mix_oracle_rollin_actions(
        model_actions,
        oracle_actions,
        1.0,
        np.random.default_rng(0),
    )
    assert {aid: action["action"] for aid, action in action_mixed.items()} == {0: 3, 1: 4}
    assert action_mixed[0]["message_tokens"] == [1, 2]
    assert action_mixed[1]["message_tokens"] == []
    assert action_replaced_agents == 2
    action_unchanged, action_replaced_agents = _mix_oracle_rollin_actions(
        model_actions,
        oracle_actions,
        0.0,
        np.random.default_rng(0),
    )
    assert {aid: action["action"] for aid, action in action_unchanged.items()} == {0: 1, 1: 2}
    assert action_replaced_agents == 0

    send_logits = torch.zeros((2, 1), requires_grad=True)
    token_logits = torch.zeros((2, 4, 8), requires_grad=True)
    len_logits = torch.zeros((2, 5), requires_grad=True)
    msg_tokens = torch.zeros((2, 4), dtype=torch.long)
    msg_lens = torch.tensor([0, 3], dtype=torch.long)

    components = _recurrent_comm_loss_components(
        send_logits,
        token_logits,
        len_logits,
        msg_tokens,
        msg_lens,
    )
    loss = _recurrent_comm_loss(
        send_logits,
        token_logits,
        len_logits,
        msg_tokens,
        msg_lens,
    )
    assert components["total"].item() == pytest.approx(
        (components["send"] + components["length"] + components["token"]).item()
    )
    assert loss.item() == pytest.approx(components["total"].item())
    weighted_components = _recurrent_comm_loss_components(
        send_logits,
        token_logits,
        len_logits,
        msg_tokens,
        msg_lens,
        send_loss_weight=2.0,
        length_loss_weight=0.5,
        token_loss_weight=0.25,
        send_rate_penalty_weight=4.0,
        send_rate_target=0.25,
    )
    expected_weighted = (
        2.0 * weighted_components["send"]
        + 0.5 * weighted_components["length"]
        + 0.25 * weighted_components["token"]
        + 4.0 * weighted_components["send_rate"]
    )
    assert weighted_components["send_rate"].item() == pytest.approx(0.0625)
    assert weighted_components["total"].item() == pytest.approx(expected_weighted.item())
    content_only_components = _recurrent_comm_loss_components(
        send_logits,
        token_logits,
        len_logits,
        msg_tokens,
        msg_lens,
        sample_weight=torch.tensor([0.0, 1.0]),
        send_loss_weight=0.0,
        length_loss_weight=1.0,
        token_loss_weight=1.0,
    )
    assert content_only_components["send"].item() > 0.0
    assert content_only_components["total"].item() == pytest.approx(
        (content_only_components["length"] + content_only_components["token"]).item()
    )
    assert _send_threshold_for_target_rate([0.1, 0.2, 0.8, 0.9], 0.5) == pytest.approx(0.5)
    assert _send_threshold_for_target_rate([0.1, 0.2, 0.8, 0.9], 0.0) == pytest.approx(1.0)
    assert _send_threshold_for_target_rate([0.1, 0.2, 0.8, 0.9], 1.0) == pytest.approx(0.0)
    loss.backward()

    assert torch.allclose(len_logits.grad[0], torch.zeros_like(len_logits.grad[0]))
    assert len_logits.grad[1].abs().sum().item() > 0.0


def test_recurrent_pipeline_interact_gate_threshold_calibration_sets_eval_threshold():
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _calibrate_pipeline_interact_gate_threshold,
    )

    class FakeModel:
        def __init__(self):
            self.step_logits = torch.logit(torch.tensor([
                [0.1, 0.2],
                [0.8, 0.9],
            ]))
            self.step = 0

        def eval(self):
            return self

        def init_hidden(self, batch, device):
            return (
                torch.zeros((batch, 1), dtype=torch.float32, device=device),
                torch.zeros((batch, 1), dtype=torch.float32, device=device),
            )

        def __call__(self, obs_t, hidden):
            del obs_t
            h = self.step_logits[self.step].reshape(-1, 1).to(hidden[0].device)
            self.step += 1
            return None, None, None, None, (h, hidden[1])

        def pipeline_interact_gate(self, hidden_state):
            return hidden_state

    episodes = [
        {
            "obs": np.zeros((2, 2, 1), dtype=np.float32),
            "pipeline_interact_gate_mask": np.ones((2, 2), dtype=np.float32),
            "pipeline_interact_gate_label": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        }
    ]
    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        eval_pipeline_interact_gate_threshold=-1.0,
        bc_pipeline_interact_gate_threshold_target_rate=0.5,
    )

    calibration = _calibrate_pipeline_interact_gate_threshold(cfg, FakeModel(), episodes, "cpu")

    assert calibration["old_threshold"] == pytest.approx(-1.0)
    assert calibration["threshold"] == pytest.approx(0.5)
    assert calibration["label_rate"] == pytest.approx(0.5)
    assert calibration["pred_rate"] == pytest.approx(0.5)
    assert cfg.eval_pipeline_interact_gate_threshold == pytest.approx(0.5)


def test_recurrent_rl_train_map_sizes_smoke():
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _build_recurrent_obs_batch,
        _build_training_env,
        _feedback_matrix,
        train_recurrent_rl,
    )

    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        train_map_sizes="8,16",
        agents=2,
        fov_preset="easy",
        max_steps=12,
        obs_exploration_memory=True,
        obs_feedback=True,
        obs_normalize_tokens=True,
        obs_memory_mode="egocentric",
        obs_memory_radius=2,
        obs_navigation_features=True,
        comm=False,
        hidden_dim=16,
        rl_updates=2,
        rollout_steps=2,
        rl_balanced_rollouts=True,
        rl_epochs=1,
        rl_eval_every=1,
        rl_eval_episodes=1,
        eval_episodes=1,
        eval_map_sizes="8,16",
        device="cpu",
    )
    env, active_cfg = _build_training_env(cfg, 0)
    obs, _ = env.reset(seed=0)
    obs_dim = _build_recurrent_obs_batch(
        obs,
        env.num_agents,
        active_cfg,
        feedback=_feedback_matrix(active_cfg, env.num_agents),
    ).shape[1]
    model = MAPPORecurrentActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    )

    trained = train_recurrent_rl(cfg, model, torch.device("cpu"))

    assert trained is model


def test_recurrent_signal_hint_comm_bc_smoke(tmp_path):
    from syncorsink.envs import SyncOrSinkConfig
    from syncorsink.eval.trajectory_audit import (
        AuditPolicySpec,
        make_recurrent_checkpoint_policy_factory,
        recurrent_checkpoint_env_config,
        run_trajectory_audit,
    )
    from syncorsink.train.mappo import resolve_device
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        collect_episode_demos,
        train_recurrent_bc_dagger,
    )

    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=60,
        oracle_type="signal_hint_comm",
        obs_exploration_memory=True,
        obs_feedback=True,
        obs_signal_features=True,
        obs_signal_sync_feedback=True,
        obs_signal_scan_state=True,
        obs_signal_target_match_features=True,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        demo_episodes=4,
        bc_epochs=1,
        bc_eval_every_epochs=1,
        bc_eval_episodes=1,
        bc_eval_seed_count=1,
        bc_restore_best_eval_epoch=True,
        bc_seq_len=16,
        bc_comm_loss_weight=0.1,
        bc_comm_send_pos_weight=-1,
        bc_signal_rejected_target_interact_loss_weight=0.05,
        bc_signal_bad_redundant_target_interact_loss_weight=0.05,
        bc_signal_first_target_scan_action_weight=0.1,
        bc_signal_joint_target_scan_action_weight=0.1,
        bc_signal_target_aux_weight=0.1,
        bc_signal_target_hypothesis_loss_weight=0.05,
        bc_signal_target_hypothesis_min_map_size=8,
        dagger_rounds=1,
        dagger_episodes=1,
        dagger_retrain_from_scratch=False,
        dagger_max_steps_per_episode=8,
        dagger_failed_episode_weight=0.25,
        rl_updates=0,
        hidden_dim=32,
        eval_episodes=1,
        eval_seed=3000,
        save=str(tmp_path / "recurrent_signal.pt"),
        device="cpu",
    )

    device = resolve_device(cfg.device)
    episodes = collect_episode_demos(cfg)
    assert len(episodes) == 4
    model, history, all_episodes, best_round = train_recurrent_bc_dagger(cfg, episodes, device)
    result = best_round["eval"]
    assert model is not None
    assert len(history) == 2
    assert len(all_episodes) == 5
    assert history[-1]["retrain_from_scratch"] is False
    assert best_round["round"] in {0, 1}
    assert history[0]["collect"]["episodes"] == 1
    assert history[0]["collect"]["avg_stored_steps"] <= 8
    assert history[-1]["dataset_sources"]["expert"] == 4
    assert history[-1]["dataset_sources"]["dagger"] == 1
    assert history[-1]["dataset_effective_transitions"] <= history[-1]["dataset_transitions"]
    assert result["episodes"] == 1
    assert "success_rate" in result
    assert "avg_no_target_reached" in result["signal"]
    assert "avg_wrong_target_scans" in result["signal"]

    checkpoint = tmp_path / "recurrent_signal.pt"
    torch.save({"model": model.state_dict(), "config": vars(cfg)}, checkpoint)
    larger_audit_env = SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=16,
        num_agents=2,
        fov_preset="easy",
        max_steps=120,
        obs_exploration_memory=False,
        comm_token_limit=8,
        token_vocab_size=32,
        max_messages=8,
    )
    larger_recurrent_audit_env = recurrent_checkpoint_env_config(checkpoint, larger_audit_env)
    assert larger_recurrent_audit_env.map_size == 16
    assert larger_recurrent_audit_env.max_steps == 120
    assert larger_recurrent_audit_env.obs_exploration_memory is True
    base_audit_env = SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=60,
        obs_exploration_memory=False,
        comm_token_limit=8,
        token_vocab_size=32,
        max_messages=8,
    )
    recurrent_audit_env = recurrent_checkpoint_env_config(checkpoint, base_audit_env)
    assert recurrent_audit_env.obs_exploration_memory is True
    audit = run_trajectory_audit(
        base_audit_env,
        [
            AuditPolicySpec(
                label="recurrent",
                factory=make_recurrent_checkpoint_policy_factory(checkpoint, device="cpu"),
                env_config=recurrent_audit_env,
            )
        ],
        episodes=1,
        seed=3000,
    )
    assert audit["policies"][0]["summary"]["episodes"] == 1
    assert audit["policies"][0]["env_config"]["obs_exploration_memory"] is True


def test_recurrent_audit_factory_inherits_checkpoint_send_threshold(monkeypatch, tmp_path):
    from syncorsink.eval.trajectory_audit import make_recurrent_checkpoint_policy_factory

    calls = []

    def fake_loader(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return lambda obs, info, state: {}

    monkeypatch.setattr(
        "syncorsink.train.recurrent_bc_rl.load_recurrent_checkpoint_policy",
        fake_loader,
    )
    checkpoint = tmp_path / "recurrent.pt"

    make_recurrent_checkpoint_policy_factory(checkpoint, device="cpu")(None)
    make_recurrent_checkpoint_policy_factory(
        checkpoint,
        device="cpu",
        eval_send_threshold=0.42,
        eval_pipeline_station_interact_guard=True,
        eval_pipeline_plan_broadcast_assist=True,
        eval_signal_frontier_exploration_assist=True,
        eval_signal_exact_target_scan_lock=True,
        eval_signal_compatible_target_scan_assist=True,
        eval_signal_compatible_target_scan_min_strength=4,
        eval_signal_negative_memory_scan_guard=True,
        eval_signal_target_probe_assist=True,
        eval_signal_constraint_message_copy_assist=True,
        eval_pipeline_frontier_exploration_assist=True,
        eval_pipeline_interact_gate_threshold=0.37,
        eval_pipeline_event_head_threshold=0.62,
        eval_pipeline_navigation_head_threshold=0.63,
        eval_pipeline_option_threshold=0.64,
        eval_pipeline_option_allow_interact=True,
    )(None)

    assert calls[0][1]["eval_send_threshold"] is None
    assert calls[0][1]["eval_pipeline_station_interact_guard"] is None
    assert calls[0][1]["eval_pipeline_plan_broadcast_assist"] is None
    assert calls[0][1]["eval_signal_frontier_exploration_assist"] is None
    assert calls[0][1]["eval_signal_exact_target_scan_lock"] is None
    assert calls[0][1]["eval_signal_compatible_target_scan_assist"] is None
    assert calls[0][1]["eval_signal_compatible_target_scan_min_strength"] is None
    assert calls[0][1]["eval_signal_negative_memory_scan_guard"] is None
    assert calls[0][1]["eval_signal_target_probe_assist"] is None
    assert calls[0][1]["eval_signal_constraint_message_copy_assist"] is None
    assert calls[0][1]["eval_pipeline_frontier_exploration_assist"] is None
    assert calls[0][1]["eval_pipeline_event_head_threshold"] is None
    assert calls[0][1]["eval_pipeline_navigation_head_threshold"] is None
    assert calls[0][1]["eval_pipeline_option_threshold"] is None
    assert calls[0][1]["eval_pipeline_option_allow_interact"] is None
    assert calls[1][1]["eval_send_threshold"] == pytest.approx(0.42)
    assert calls[1][1]["eval_pipeline_station_interact_guard"] is True
    assert calls[1][1]["eval_pipeline_plan_broadcast_assist"] is True
    assert calls[1][1]["eval_signal_frontier_exploration_assist"] is True
    assert calls[1][1]["eval_signal_exact_target_scan_lock"] is True
    assert calls[1][1]["eval_signal_compatible_target_scan_assist"] is True
    assert calls[1][1]["eval_signal_compatible_target_scan_min_strength"] == 4
    assert calls[1][1]["eval_signal_negative_memory_scan_guard"] is True
    assert calls[1][1]["eval_signal_target_probe_assist"] is True
    assert calls[1][1]["eval_signal_constraint_message_copy_assist"] is True
    assert calls[1][1]["eval_pipeline_frontier_exploration_assist"] is True
    assert calls[1][1]["eval_pipeline_interact_gate_threshold"] == pytest.approx(0.37)
    assert calls[1][1]["eval_pipeline_event_head_threshold"] == pytest.approx(0.62)
    assert calls[1][1]["eval_pipeline_navigation_head_threshold"] == pytest.approx(0.63)
    assert calls[1][1]["eval_pipeline_option_threshold"] == pytest.approx(0.64)
    assert calls[1][1]["eval_pipeline_option_allow_interact"] is True


def test_recurrent_actor_checkpoint_init_for_rl_smoke(tmp_path):
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.mappo import resolve_device
    from syncorsink.train.recurrent_bc_rl import (
        PIPELINE_OPTION_NONE,
        RecurrentConfig,
        _build_env,
        _build_recurrent_obs_batch,
        _feedback_matrix,
        _inherit_recurrent_init_observation_config,
        load_recurrent_actor_checkpoint,
        train_recurrent_rl,
    )

    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=12,
        obs_exploration_memory=True,
        obs_feedback=True,
        obs_normalize_tokens=True,
        obs_memory_mode="egocentric",
        obs_memory_radius=2,
        comm=False,
        hidden_dim=16,
        rl_updates=1,
        rollout_steps=2,
        rl_epochs=1,
        rl_eval_every=1,
        rl_eval_episodes=1,
        rl_eval_seed=4000,
        save=str(tmp_path / "rl_from_init.pt"),
        device="cpu",
    )
    env = _build_env(cfg)
    obs, _ = env.reset(seed=0)
    obs_dim = _build_recurrent_obs_batch(
        obs,
        env.num_agents,
        cfg,
        feedback=_feedback_matrix(cfg, env.num_agents),
    ).shape[1]
    model = MAPPORecurrentActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    )
    checkpoint = tmp_path / "recurrent_init.pt"
    torch.save({"model": model.state_dict(), "config": vars(cfg)}, checkpoint)

    device = resolve_device(cfg.device)
    loaded = load_recurrent_actor_checkpoint(checkpoint, cfg, device)
    inherited_shape_cfg = RecurrentConfig(**{
        **vars(cfg),
        "recurrent_init": str(checkpoint),
        "obs_memory_mode": "full",
    })
    inherited_shape = _inherit_recurrent_init_observation_config(inherited_shape_cfg)
    assert inherited_shape == {"obs_memory_mode": "egocentric"}
    shape_loaded = load_recurrent_actor_checkpoint(checkpoint, inherited_shape_cfg, device)
    assert shape_loaded.encoder.net[0].weight.shape[1] == obs_dim
    legacy_checkpoint = tmp_path / "recurrent_init_legacy_no_scan_gate.pt"
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith((
            "signal_scan_gate.",
            "signal_target_validity.",
            "signal_target_decision.",
            "signal_target_aux.",
            "signal_target_hypothesis.",
            "pipeline_plan_policy.",
            "pipeline_event_policy.",
            "pipeline_navigation_policy.",
            "pipeline_option_policy.",
        ))
    }
    torch.save({"model": legacy_state, "config": vars(cfg)}, legacy_checkpoint)
    legacy_loaded = load_recurrent_actor_checkpoint(legacy_checkpoint, cfg, device)
    assert hasattr(legacy_loaded, "signal_scan_gate")
    assert hasattr(legacy_loaded, "signal_target_validity")
    assert hasattr(legacy_loaded, "signal_target_decision")
    assert hasattr(legacy_loaded, "signal_target_aux")
    assert hasattr(legacy_loaded, "signal_target_hypothesis")
    assert torch.allclose(
        legacy_loaded.pipeline_plan_policy.weight,
        legacy_loaded.policy.linear.weight,
    )
    assert torch.allclose(
        legacy_loaded.pipeline_plan_policy.bias,
        legacy_loaded.policy.linear.bias,
    )
    assert torch.allclose(
        legacy_loaded.pipeline_event_policy.weight,
        legacy_loaded.policy.linear.weight,
    )
    assert torch.allclose(
        legacy_loaded.pipeline_event_policy.bias,
        legacy_loaded.policy.linear.bias,
    )
    assert torch.allclose(
        legacy_loaded.pipeline_navigation_policy.weight,
        legacy_loaded.policy.linear.weight,
    )
    assert torch.allclose(
        legacy_loaded.pipeline_navigation_policy.bias,
        legacy_loaded.policy.linear.bias,
    )
    assert torch.allclose(
        legacy_loaded.pipeline_option_policy.weight,
        torch.zeros_like(legacy_loaded.pipeline_option_policy.weight),
    )
    expected_option_bias = torch.full_like(legacy_loaded.pipeline_option_policy.bias, -6.0)
    expected_option_bias[PIPELINE_OPTION_NONE] = 6.0
    assert torch.allclose(legacy_loaded.pipeline_option_policy.bias, expected_option_bias)
    train_recurrent_rl(cfg, loaded, device)

    saved = torch.load(tmp_path / "rl_from_init.pt", map_location="cpu")
    assert saved["algorithm"] == "recurrent_bc_rl"
    assert saved["best_eval"]["episodes"] == 1
    expanded_cfg = RecurrentConfig(**{
        **vars(cfg),
        "obs_signal_negative_memory": True,
        "recurrent_init_allow_obs_dim_mismatch": True,
    })
    expanded = load_recurrent_actor_checkpoint(checkpoint, expanded_cfg, device)
    old_weight = model.state_dict()["encoder.net.0.weight"]
    expanded_weight = expanded.state_dict()["encoder.net.0.weight"]
    assert expanded_weight.shape[1] == obs_dim + 8
    torch.testing.assert_close(expanded_weight[:, :obs_dim - 8], old_weight[:, :obs_dim - 8])
    torch.testing.assert_close(expanded_weight[:, obs_dim - 8:obs_dim], torch.zeros_like(old_weight[:, -8:]))
    torch.testing.assert_close(expanded_weight[:, obs_dim:], old_weight[:, -8:])
    with pytest.raises(ValueError, match="hidden_dim"):
        load_recurrent_actor_checkpoint(checkpoint, RecurrentConfig(**{**vars(cfg), "hidden_dim": 32}), device)


def test_recurrent_checkpoint_expands_memory_before_pipeline_aux_features(tmp_path):
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _build_env,
        _build_recurrent_obs_batch,
        _recurrent_obs_layout_widths,
        load_recurrent_actor_checkpoint,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=3,
        fov_preset="easy",
        max_steps=20,
        obs_pipeline_features=True,
        obs_exploration_memory=False,
        comm=True,
        hidden_dim=16,
        device="cpu",
    )
    env = _build_env(cfg)
    obs, _ = env.reset(seed=0)
    obs_dim = _build_recurrent_obs_batch(obs, env.num_agents, cfg).shape[1]
    model = MAPPORecurrentActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    )
    with torch.no_grad():
        weight = model.encoder.net[0].weight
        pattern = torch.arange(weight.shape[1], dtype=weight.dtype).unsqueeze(0)
        weight.copy_(pattern.repeat(weight.shape[0], 1))
    checkpoint = tmp_path / "pipeline_aux_recurrent.pt"
    torch.save({"model": model.state_dict(), "config": vars(cfg), "obs_dim": obs_dim}, checkpoint)

    expanded_cfg = RecurrentConfig(**{
        **vars(cfg),
        "obs_exploration_memory": True,
        "obs_memory_mode": "egocentric",
        "obs_memory_radius": 1,
        "recurrent_init_allow_obs_dim_mismatch": True,
    })
    old_layout = _recurrent_obs_layout_widths(cfg)
    new_layout = _recurrent_obs_layout_widths(expanded_cfg)
    loaded = load_recurrent_actor_checkpoint(checkpoint, expanded_cfg, "cpu")
    old_weight = model.encoder.net[0].weight.detach()
    expanded_weight = loaded.encoder.net[0].weight.detach()

    base_dim = old_layout["base"]
    aux_dim = old_layout["aux"]
    memory_delta = new_layout["memory"] - old_layout["memory"]
    assert old_layout["memory"] == 0
    assert memory_delta > 0

    torch.testing.assert_close(expanded_weight[:, :base_dim], old_weight[:, :base_dim])
    torch.testing.assert_close(
        expanded_weight[:, base_dim:base_dim + memory_delta],
        torch.zeros_like(expanded_weight[:, base_dim:base_dim + memory_delta]),
    )
    torch.testing.assert_close(
        expanded_weight[:, base_dim + memory_delta:base_dim + memory_delta + aux_dim],
        old_weight[:, base_dim:base_dim + aux_dim],
    )
    torch.testing.assert_close(expanded_weight[:, -8:], old_weight[:, -8:])


def test_recurrent_residual_backbone_checkpoint_round_trip(tmp_path):
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _build_env,
        _build_recurrent_obs_batch,
        _feedback_matrix,
        load_recurrent_actor_checkpoint,
        load_recurrent_checkpoint_policy,
    )

    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=12,
        obs_exploration_memory=True,
        obs_feedback=True,
        obs_normalize_tokens=True,
        obs_memory_mode="egocentric",
        obs_memory_radius=2,
        hidden_dim=16,
        recurrent_backbone="residual_mlp",
        comm=False,
    )
    env = _build_env(cfg)
    obs, _info = env.reset(seed=0)
    obs_dim = _build_recurrent_obs_batch(
        obs,
        env.num_agents,
        cfg,
        feedback=_feedback_matrix(cfg, env.num_agents),
    ).shape[1]
    model = MAPPORecurrentActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        backbone=cfg.recurrent_backbone,
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    )
    logits, next_hidden = model(
        torch.zeros(env.num_agents, obs_dim),
        model.init_hidden(env.num_agents, torch.device("cpu")),
    )
    assert logits.shape == (env.num_agents, 8)
    assert next_hidden[0].shape == (env.num_agents, cfg.hidden_dim)

    checkpoint = tmp_path / "recurrent_residual.pt"
    torch.save({"model": model.state_dict(), "config": vars(cfg)}, checkpoint)
    loaded = load_recurrent_actor_checkpoint(checkpoint, cfg, torch.device("cpu"))
    assert loaded.backbone == "residual_mlp"
    assert "encoder.input.0.weight" in loaded.state_dict()

    policy = load_recurrent_checkpoint_policy(checkpoint, device="cpu")
    assert policy.metadata()["recurrent_backbone"] == "residual_mlp"

    with pytest.raises(ValueError, match="recurrent_backbone"):
        load_recurrent_actor_checkpoint(
            checkpoint,
            RecurrentConfig(**{**vars(cfg), "recurrent_backbone": "mlp"}),
            torch.device("cpu"),
        )


def test_recurrent_local_cnn_backbone_checkpoint_round_trip(tmp_path):
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _build_env,
        _build_recurrent_obs_batch,
        _feedback_matrix,
        _recurrent_fov_radius,
        load_recurrent_actor_checkpoint,
        load_recurrent_checkpoint_policy,
    )

    cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=3,
        fov_preset="easy",
        max_steps=20,
        obs_pipeline_features=True,
        hidden_dim=16,
        recurrent_backbone="local_cnn",
        comm=False,
    )
    env = _build_env(cfg)
    obs, _info = env.reset(seed=0)
    obs_dim = _build_recurrent_obs_batch(
        obs,
        env.num_agents,
        cfg,
        feedback=_feedback_matrix(cfg, env.num_agents),
    ).shape[1]
    model = MAPPORecurrentActor(
        obs_dim=obs_dim,
        action_dim=8,
        hidden_dim=cfg.hidden_dim,
        backbone=cfg.recurrent_backbone,
        fov_radius=_recurrent_fov_radius(cfg),
        comm_enabled=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
    )
    logits, next_hidden = model(
        torch.zeros(env.num_agents, obs_dim),
        model.init_hidden(env.num_agents, torch.device("cpu")),
    )
    assert logits.shape == (env.num_agents, 8)
    assert next_hidden[0].shape == (env.num_agents, cfg.hidden_dim)

    checkpoint = tmp_path / "recurrent_local_cnn.pt"
    torch.save({"model": model.state_dict(), "config": vars(cfg), "obs_dim": obs_dim}, checkpoint)
    loaded = load_recurrent_actor_checkpoint(checkpoint, cfg, torch.device("cpu"))
    state_keys = set(loaded.state_dict())
    assert loaded.backbone == "local_cnn"
    assert "encoder.spatial.0.weight" in state_keys
    assert "encoder.tail.0.weight" in state_keys

    policy = load_recurrent_checkpoint_policy(checkpoint, device="cpu")
    assert policy.metadata()["recurrent_backbone"] == "local_cnn"

    with pytest.raises(ValueError, match="fov_preset"):
        load_recurrent_actor_checkpoint(
            checkpoint,
            RecurrentConfig(**{**vars(cfg), "fov_preset": "medium"}),
            torch.device("cpu"),
        )


def test_recurrent_comm_feedback_ppo_smoke(tmp_path):
    from syncorsink.train.mappo import resolve_device
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        collect_episode_demos,
        train_recurrent_bc,
        train_recurrent_rl,
    )

    cfg = RecurrentConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=30,
        oracle_type="signal_hint_comm",
        obs_exploration_memory=True,
        obs_feedback=True,
        obs_signal_scan_state=True,
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        demo_episodes=2,
        bc_epochs=1,
        bc_seq_len=8,
        bc_comm_loss_weight=0.1,
        bc_comm_send_pos_weight=-1,
        rl_updates=1,
        rollout_steps=4,
        rl_rollout_eval_decoding=True,
        rl_eval_decoding_action_loss_weight=0.25,
        rl_epochs=1,
        rl_eval_every=1,
        rl_eval_episodes=1,
        rl_eval_seed=4000,
        rl_restore_best=True,
        rl_save_best=True,
        eval_signal_scan_sync_assist=True,
        eval_signal_scan_broadcast_assist=True,
        eval_signal_exact_target_message_guard=True,
        eval_signal_exact_target_navigation_assist=True,
        eval_signal_exact_target_memory_steps=8,
        hidden_dim=32,
        eval_episodes=1,
        save=str(tmp_path / "recurrent_rl.pt"),
        device="cpu",
    )

    device = resolve_device(cfg.device)
    episodes = collect_episode_demos(cfg)
    model = train_recurrent_bc(cfg, episodes, device)
    trained = train_recurrent_rl(cfg, model, device)
    checkpoint = torch.load(tmp_path / "recurrent_rl.pt", map_location="cpu")
    best_checkpoint = torch.load(tmp_path / "recurrent_rl_best.pt", map_location="cpu")

    assert trained is model
    assert checkpoint["algorithm"] == "recurrent_bc_rl"
    assert checkpoint["restored_best"] is True
    assert checkpoint["best_eval"]["episodes"] == 1
    assert checkpoint["final_eval"]["episodes"] == 1
    assert "signal" in checkpoint["best_eval"]
    assert best_checkpoint["best_eval"]["episodes"] == 1


def test_bc_rl_curriculum_tiny_smoke(tmp_path):
    from syncorsink.train.curriculum import BCRLCurriculumConfig, run_bc_rl_curriculum

    result = run_bc_rl_curriculum(BCRLCurriculumConfig(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=50,
        demo_episodes=5,
        dagger_rounds=0,
        bc_epochs=1,
        bc_batch_size=16,
        hidden_dim=32,
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=8,
        comm_max_messages=4,
        rl_updates=1,
        rl_rollout_steps=8,
        rl_epochs=1,
        rl_minibatch=8,
        train_eval_every=1,
        train_eval_episodes=1,
        eval_episodes=1,
        eval_stochastic=True,
        output_dir=str(tmp_path),
        run_name="tiny",
        wandb=True,
        wandb_mode="disabled",
        device="cpu",
    ))

    run_dir = tmp_path / "tiny"
    assert result["status"] == "complete"
    assert (run_dir / "demos" / "signal_hunt_oracle.npz").exists()
    assert (run_dir / "checkpoints" / "bc_dagger.pt").exists()
    assert (run_dir / "checkpoints" / "mappo_bc_rl.pt").exists()
    assert (run_dir / "checkpoints" / "mappo_bc_rl_best.pt").exists()
    assert result["demo"]["transitions"] > 0
    assert "send_rate" in result["demo"]
    assert "action_accuracy" in result["bc_diagnostics"]
    assert "pred_send_rate_threshold_0_50" in result["bc_diagnostics"]
    assert result["eval_bc"]["summary"]["episodes"] == 1
    assert result["eval_rl_deterministic"]["summary"]["episodes"] == 1
    assert result["eval_rl_stochastic"]["summary"]["episodes"] == 1
    assert result["eval_rl_stochastic"]["decode"]["action_mode"] == "sample"
    assert result["eval_rl_stochastic"]["decode"]["send_mode"] == "threshold"
    assert result["eval_rl_stochastic"]["decode"]["send_threshold"] == 0.25
    assert result["eval_rl_stochastic"]["decode"]["token_mode"] == "argmax"
    assert result["eval_rl_stochastic"]["decode"]["length_mode"] == "argmax"
    assert result["eval_rl_best_deterministic"]["summary"]["episodes"] == 1
    assert result["best_eval_checkpoint"]["path"].endswith("mappo_bc_rl_best.pt")
    assert result["wandb_summary"]["enabled"] is True
    assert result["wandb_summary"]["mode"] == "disabled"


def test_reward_model(tmp_path):
    from syncorsink.train.bc import train_reward_model, BCConfig
    model_path = str(tmp_path / "reward.pt")
    cfg = BCConfig(
        scenario="signal_hunt", map_size=8, agents=2, fov_preset="easy",
        demo_episodes=5, epochs=3, batch_size=16, lr=1e-3,
        hidden_dim=32, device="cpu", save=model_path, max_steps=50,
    )
    rnet = train_reward_model(cfg)
    assert rnet is not None


def test_signal_hunt_shaping_rewards():
    """Test that v4 coordination shaping rewards fire correctly."""
    from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
    config = SyncOrSinkConfig(
        scenario="signal_hunt", map_size=8, num_agents=2, fov_preset="easy",
        signal_shaping=True, signal_shaping_scale=0.1,
        signal_scan_bonus=0.2, signal_joint_scan_bonus=3.0,
        signal_colocation_bonus=0.5, signal_colocation_radius=2,
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)
    # Just verify it runs without error
    actions = {i: {"action": env.ACTION_STAY} for i in range(env.num_agents)}
    obs, rewards, done, truncated, info = env.step(actions)
    assert len(rewards) == env.num_agents


def test_signal_hunt_completion_shaping_rewards_fire_once():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv

    config = SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        signal_shaping=True,
        signal_target_visit_bonus=0.4,
        signal_decoy_visit_penalty=0.25,
        signal_unique_target_scan_bonus=0.7,
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)
    target = env.scenario_state.data["target"]
    decoy = env.scenario_state.data["decoys"][0]
    env.agent_positions[0] = target
    env.agent_positions[1] = decoy

    actions = {0: {"action": env.ACTION_STAY}, 1: {"action": env.ACTION_STAY}}
    _, rewards, done, _, info = env.step(actions)
    assert done is False
    assert rewards[0] == pytest.approx(0.4)
    assert rewards[1] == pytest.approx(-0.25)
    assert {"event": "target_visit"} in info["events"][0]
    assert {"event": "decoy_visit"} in info["events"][1]

    _, rewards, done, _, info = env.step(actions)
    assert done is False
    assert rewards[0] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(0.0)
    assert {"event": "target_visit"} not in info["events"][0]
    assert {"event": "decoy_visit"} not in info["events"][1]

    scan_actions = {0: {"action": env.ACTION_INTERACT}, 1: {"action": env.ACTION_STAY}}
    _, rewards, done, _, info = env.step(scan_actions)
    assert done is False
    assert rewards[0] == pytest.approx(0.7)
    assert {"event": "target_scan"} in info["events"][0]
    assert {"event": "unique_target_scan"} in info["events"][0]

    _, rewards, done, _, info = env.step(scan_actions)
    assert done is False
    assert rewards[0] == pytest.approx(0.0)
    assert {"event": "target_scan"} in info["events"][0]
    assert {"event": "unique_target_scan"} not in info["events"][0]


def test_energy_grid_node_critical_events():
    """Test that node_critical events fire when energy drops."""
    from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
    config = SyncOrSinkConfig(
        scenario="energy_grid", map_size=8, num_agents=2,
        fov_preset="easy", energy_preset="hard",
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)
    # Step until we get events or episode ends
    for _ in range(50):
        actions = {i: {"action": env.ACTION_STAY} for i in range(env.num_agents)}
        obs, rewards, done, truncated, info = env.step(actions)
        if done or truncated:
            break
    # Should have terminated (node depleted on hard preset)
    assert done or truncated


def test_energy_grid_private_monitor_masks_unassigned_node_energy():
    """Default energy grid observations hide unassigned node urgency."""
    from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig

    config = SyncOrSinkConfig(
        scenario="energy_grid", map_size=8, num_agents=2,
        fov_preset="easy", energy_preset="easy",
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)

    node_assignments = env.scenario_state.data["node_assignments"]
    node_energy = env.scenario_state.data["node_energy"]
    node_pos = next(pos for pos, assigned in node_assignments.items() if assigned != 0)
    assigned_agent = node_assignments[node_pos]
    env.agent_positions[0] = node_pos
    env.agent_positions[assigned_agent] = node_pos

    obs = env._build_observations()
    center = tuple(dim // 2 for dim in obs[0]["local_node_energy"].shape)

    assert env.config.energy_private_monitor is True
    assert int(obs[0]["local_node_energy"][center]) == 0
    assert int(obs[assigned_agent]["local_node_energy"][center]) == int(node_energy[node_pos])


def test_energy_grid_private_monitor_routes_node_critical_events_to_assigned_agent():
    """Default node_critical events must not leak private node state to every agent."""
    from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig

    config = SyncOrSinkConfig(
        scenario="energy_grid", map_size=8, num_agents=3,
        fov_preset="easy", energy_preset="easy",
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)
    node_pos, assigned_agent = next(iter(env.scenario_state.data["node_assignments"].items()))
    env.scenario_state.data["node_energy"][node_pos] = env.scenario_state.data["sync_threshold"]

    actions = {i: {"action": env.ACTION_STAY} for i in range(env.num_agents)}
    _, _, _, _, info = env.step(actions)

    for agent_id, events in info["events"].items():
        critical = [event for event in events if event.get("event") == "node_critical" and event.get("node") == node_pos]
        if agent_id == assigned_agent:
            assert len(critical) == 1
        else:
            assert critical == []


def test_energy_grid_easy_preset_sync_gates_initial_recharge():
    """The small core Energy Grid task requires paired recharge, not solo top-up."""
    from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig

    config = SyncOrSinkConfig(
        scenario="energy_grid", map_size=8, num_agents=2,
        fov_preset="easy", energy_preset="easy",
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)
    node_pos = next(iter(env.scenario_state.data["node_energy"]))
    node_type = env.scenario_state.data["node_types"][node_pos]

    assert env.scenario_state.data["node_energy"][node_pos] <= env.scenario_state.data["sync_threshold"]

    env.agent_positions[0] = node_pos
    env.inventories[0] = node_type
    actions = {
        0: {"action": env.ACTION_INTERACT},
        1: {"action": env.ACTION_STAY},
    }
    _, _, _, _, info = env.step(actions)

    assert info["recharge_count"] == 0
    assert env.inventories[0] == node_type

    env.agent_positions[0] = node_pos
    env.agent_positions[1] = node_pos
    env.inventories[0] = node_type
    env.inventories[1] = node_type
    actions = {
        0: {"action": env.ACTION_INTERACT},
        1: {"action": env.ACTION_INTERACT},
    }
    _, _, _, _, info = env.step(actions)

    assert info["recharge_count"] == 2
    assert env.inventories[0] == 0
    assert env.inventories[1] == 0


def test_energy_grid_symmetric_control_broadcasts_node_critical_events():
    """The legacy symmetric ablation remains explicit and observable."""
    from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig

    config = SyncOrSinkConfig(
        scenario="energy_grid", map_size=8, num_agents=3,
        fov_preset="easy", energy_preset="easy", energy_private_monitor=False,
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)
    node_pos = next(iter(env.scenario_state.data["node_energy"]))
    env.scenario_state.data["node_energy"][node_pos] = env.scenario_state.data["sync_threshold"]

    actions = {i: {"action": env.ACTION_STAY} for i in range(env.num_agents)}
    _, _, _, _, info = env.step(actions)

    assert all(
        any(event.get("event") == "node_critical" and event.get("node") == node_pos for event in events)
        for events in info["events"].values()
    )


def test_pipeline_dagger_focus_helpers_detect_pre_event_misses():
    from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
    from syncorsink.train.recurrent_bc_rl import (
        PIPELINE_WRONG_DELIVERY_ROOT_PICKUP_EVENT,
        RecurrentConfig,
        _label_latest_pipeline_bad_pickup_actions,
        _label_latest_pipeline_bad_interact_actions,
        _label_latest_pipeline_bad_drop_actions,
        _new_episode_sequence,
        _pipeline_bad_interact_agents,
        _pipeline_bad_pickup_miss_agents,
        _pipeline_delivery_action_label_mask,
        _pipeline_delivery_progress_action_label_mask,
        _pipeline_delivery_miss_agents,
        _pipeline_delivery_ready_agents,
        _pipeline_drop_miss_agents,
        _pipeline_navigation_action_label_mask,
        _pipeline_pickup_action_label_mask,
        _pipeline_pickup_gate_label_mask,
        _pipeline_ready_interact_action_label_mask,
        _pipeline_pickup_miss_agents,
        _pipeline_station_guard_action_label_mask,
        _pipeline_station_stall_miss_agents,
        _pipeline_sync_action_label_mask,
        _pipeline_wrong_station_recovery_action_label_mask,
        _pipeline_wrong_delivery_miss_agents,
        _update_pipeline_wrong_delivery_provenance,
    )

    config = SyncOrSinkConfig(
        scenario="pipeline_assembly",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        pipeline_stage_count=1,
        pipeline_required_per_stage_min=1,
        pipeline_required_per_stage_max=1,
        pipeline_sync_probability=0.0,
        pipeline_dependency_probability=0.0,
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)
    stage = env.scenario_state.data["stages"][0]
    needed_type = int(stage["required"][0])
    resource_pos = next(
        pos
        for pos, resource_type in env.scenario_state.data["resource_types"].items()
        if int(resource_type) == needed_type
    )

    env.agent_positions[0] = resource_pos
    env.inventories[0] = 0
    pipeline_label_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
    )
    pickup_mask, pickup_action_id = _pipeline_pickup_action_label_mask(
        env,
        {0: {"action": env.ACTION_PICKUP}},
    )
    np.testing.assert_array_equal(pickup_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(pickup_action_id, np.array([env.ACTION_PICKUP, -1], dtype=np.int64))
    pickup_gate_mask, pickup_gate_label = _pipeline_pickup_gate_label_mask(
        env,
        env._build_observations(),
        pipeline_label_cfg,
    )
    np.testing.assert_array_equal(pickup_gate_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(pickup_gate_label, np.array([1.0, 0.0], dtype=np.float32))
    assert _pipeline_pickup_miss_agents(
        env,
        {0: {"action": env.ACTION_PICKUP}},
        {0: {"action": env.ACTION_STAY}},
    ) == [0]
    resource_navigation_found = False
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        candidate_pos = (int(resource_pos[0]) + dx, int(resource_pos[1]) + dy)
        if not (0 <= candidate_pos[0] < env.map_size and 0 <= candidate_pos[1] < env.map_size):
            continue
        env.agent_positions[0] = candidate_pos
        env.inventories[0] = 0
        nav_obs = env._build_observations()
        navigation_mask, navigation_action_id = _pipeline_navigation_action_label_mask(
            env,
            nav_obs,
            pipeline_label_cfg,
        )
        if float(navigation_mask[0]) <= 0.0:
            continue
        resource_navigation_found = True
        assert int(navigation_action_id[0]) in {
            env.ACTION_UP,
            env.ACTION_DOWN,
            env.ACTION_LEFT,
            env.ACTION_RIGHT,
        }
        assert nav_obs[0]["action_mask"][int(navigation_action_id[0])] == 1
        break
    assert resource_navigation_found

    unneeded_resource_pos, unneeded_type = next(
        (pos, int(resource_type))
        for pos, resource_type in env.scenario_state.data["resource_types"].items()
        if int(resource_type) != needed_type
    )
    env.agent_positions[0] = tuple(unneeded_resource_pos)
    env.inventories[0] = 0
    bad_pickup_obs = env._build_observations()
    pickup_gate_mask, pickup_gate_label = _pipeline_pickup_gate_label_mask(
        env,
        bad_pickup_obs,
        pipeline_label_cfg,
    )
    np.testing.assert_array_equal(pickup_gate_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(pickup_gate_label, np.array([0.0, 0.0], dtype=np.float32))
    assert _pipeline_bad_pickup_miss_agents(
        env,
        bad_pickup_obs,
        {0: {"action": env.ACTION_STAY}},
        {0: {"action": env.ACTION_PICKUP}},
        pipeline_label_cfg,
    ) == [0]

    station = tuple(stage["station"])
    stage["delivered"] = []
    stage["done"] = False
    env.agent_positions[0] = station
    env.inventories[0] = needed_type
    delivery_mask, delivery_action_id = _pipeline_delivery_action_label_mask(
        env,
        {0: {"action": env.ACTION_INTERACT}},
    )
    np.testing.assert_array_equal(delivery_mask, np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(delivery_action_id, np.array([env.ACTION_INTERACT, -1], dtype=np.int64))
    delivery_progress_obs = env._build_observations()
    delivery_progress_mask, delivery_progress_action_id = _pipeline_delivery_progress_action_label_mask(
        env,
        delivery_progress_obs,
        pipeline_label_cfg,
    )
    np.testing.assert_array_equal(
        delivery_progress_mask,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        delivery_progress_action_id,
        np.array([env.ACTION_INTERACT, -1], dtype=np.int64),
    )
    ready_interact_mask, ready_interact_action_id = _pipeline_ready_interact_action_label_mask(
        env,
        delivery_progress_obs,
    )
    np.testing.assert_array_equal(
        ready_interact_mask,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        ready_interact_action_id,
        np.array([env.ACTION_INTERACT, -1], dtype=np.int64),
    )
    assert _pipeline_delivery_miss_agents(
        env,
        {0: {"action": env.ACTION_INTERACT}},
        {0: {"action": env.ACTION_STAY}},
    ) == [0]
    assert _pipeline_delivery_ready_agents(
        env,
        {0: {"action": env.ACTION_INTERACT}},
    ) == [0]
    assert _pipeline_delivery_ready_agents(
        env,
        {0: {"action": env.ACTION_STAY}},
    ) == []
    assert _pipeline_station_stall_miss_agents(
        env,
        {0: {"action": env.ACTION_INTERACT}},
        {0: {"action": env.ACTION_STAY}},
    ) == [0]
    assert _pipeline_drop_miss_agents(
        env,
        {0: {"action": env.ACTION_STAY}},
        {0: {"action": env.ACTION_DROP}},
    ) == [0]

    stage["sync"] = True
    stage["delivered"] = []
    env.agent_positions[0] = station
    env.inventories[0] = needed_type
    env.agent_positions[1] = station
    env.inventories[1] = 0
    pre_sync_obs = env._build_observations()
    sync_mask, sync_action_id = _pipeline_sync_action_label_mask(
        env,
        pre_sync_obs,
        pipeline_label_cfg,
    )
    np.testing.assert_array_equal(sync_mask, np.array([0.0, 1.0], dtype=np.float32))
    assert int(sync_action_id[1]) == env.ACTION_INTERACT
    station_guard_mask, station_guard_action_id = _pipeline_station_guard_action_label_mask(
        env,
        pre_sync_obs,
        pipeline_label_cfg,
    )
    np.testing.assert_array_equal(station_guard_mask, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(station_guard_action_id, np.array([-1, -1], dtype=np.int64))
    assert _pipeline_bad_interact_agents(
        env,
        {1: {"action": env.ACTION_INTERACT}},
    ) == []

    stage["delivered"] = [needed_type]
    env.agent_positions[0] = station
    env.inventories[0] = 0
    env.agent_positions[1] = station
    env.inventories[1] = 0
    sync_ready_obs = env._build_observations()
    sync_mask, sync_action_id = _pipeline_sync_action_label_mask(
        env,
        sync_ready_obs,
        pipeline_label_cfg,
    )
    assert float(sync_mask[0]) == pytest.approx(1.0)
    assert int(sync_action_id[0]) == env.ACTION_INTERACT
    sync_ready_interact_mask, sync_ready_interact_action_id = (
        _pipeline_ready_interact_action_label_mask(env, sync_ready_obs)
    )
    assert float(sync_ready_interact_mask[0]) == pytest.approx(1.0)
    assert int(sync_ready_interact_action_id[0]) == env.ACTION_INTERACT

    sync_navigation_found = False
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        candidate_pos = (int(station[0]) + dx, int(station[1]) + dy)
        if not (0 <= candidate_pos[0] < env.map_size and 0 <= candidate_pos[1] < env.map_size):
            continue
        env.agent_positions[0] = candidate_pos
        env.inventories[0] = 0
        sync_nav_obs = env._build_observations()
        sync_mask, sync_action_id = _pipeline_sync_action_label_mask(
            env,
            sync_nav_obs,
            pipeline_label_cfg,
        )
        if float(sync_mask[0]) <= 0.0:
            continue
        sync_navigation_found = True
        assert int(sync_action_id[0]) in {
            env.ACTION_UP,
            env.ACTION_DOWN,
            env.ACTION_LEFT,
            env.ACTION_RIGHT,
        }
        assert sync_nav_obs[0]["action_mask"][int(sync_action_id[0])] == 1
        break
    assert sync_navigation_found

    stage["sync"] = False
    stage["delivered"] = []

    wrong_station = next(pos for pos in env.meta["stations"] if tuple(pos) != station)
    env.agent_positions[0] = tuple(wrong_station)
    env.inventories[0] = needed_type
    wrong_station_obs = env._build_observations()
    navigation_mask, navigation_action_id = _pipeline_navigation_action_label_mask(
        env,
        wrong_station_obs,
        pipeline_label_cfg,
    )
    np.testing.assert_array_equal(
        navigation_mask,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert int(navigation_action_id[0]) in {
        env.ACTION_UP,
        env.ACTION_DOWN,
        env.ACTION_LEFT,
        env.ACTION_RIGHT,
    }
    assert wrong_station_obs[0]["action_mask"][int(navigation_action_id[0])] == 1
    (
        wrong_station_recovery_mask,
        wrong_station_recovery_action_id,
    ) = _pipeline_wrong_station_recovery_action_label_mask(
        env,
        wrong_station_obs,
        pipeline_label_cfg,
    )
    np.testing.assert_array_equal(
        wrong_station_recovery_mask,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert int(wrong_station_recovery_action_id[0]) in {
        env.ACTION_UP,
        env.ACTION_DOWN,
        env.ACTION_LEFT,
        env.ACTION_RIGHT,
    }
    assert wrong_station_obs[0]["action_mask"][int(wrong_station_recovery_action_id[0])] == 1
    assert _pipeline_wrong_delivery_miss_agents(
        env,
        {0: {"action": env.ACTION_STAY}},
        {0: {"action": env.ACTION_INTERACT}},
    ) == [0]
    assert _pipeline_delivery_ready_agents(
        env,
        {0: {"action": env.ACTION_INTERACT}},
    ) == []

    ep_data = _new_episode_sequence()
    ep_data["pipeline_bad_pickup_action_mask"].extend([0.0, 0.0])
    ep_data["pipeline_bad_pickup_action_id"].extend([-1, -1])
    assert _label_latest_pipeline_bad_pickup_actions(ep_data, num_agents=2, agent_ids=[0]) == 1
    assert ep_data["pipeline_bad_pickup_action_mask"] == [1.0, 0.0]
    assert ep_data["pipeline_bad_pickup_action_id"] == [env.ACTION_PICKUP, -1]
    ep_data["pipeline_bad_drop_action_mask"].extend([0.0, 0.0])
    ep_data["pipeline_bad_drop_action_id"].extend([-1, -1])
    assert _label_latest_pipeline_bad_drop_actions(ep_data, num_agents=2, agent_ids=[0]) == 1
    assert ep_data["pipeline_bad_drop_action_mask"] == [1.0, 0.0]
    assert ep_data["pipeline_bad_drop_action_id"] == [env.ACTION_DROP, -1]
    ep_data["pipeline_bad_interact_action_mask"].extend([0.0, 0.0])
    ep_data["pipeline_bad_interact_action_id"].extend([-1, -1])
    assert _label_latest_pipeline_bad_interact_actions(ep_data, num_agents=2, agent_ids=[0]) == 1
    assert ep_data["pipeline_bad_interact_action_mask"] == [1.0, 0.0]
    assert ep_data["pipeline_bad_interact_action_id"] == [env.ACTION_INTERACT, -1]

    provenance_cfg = RecurrentConfig(
        scenario="pipeline_assembly",
        map_size=8,
        agents=2,
        dagger_focus_error_weight=4.0,
        dagger_pipeline_wrong_delivery_provenance_labels=True,
        dagger_pipeline_wrong_delivery_provenance_weight=2.5,
    )
    ep_data = _new_episode_sequence()
    ep_data["pipeline_bad_pickup_action_mask"].extend([0.0, 0.0, 0.0, 0.0])
    ep_data["pipeline_bad_pickup_action_id"].extend([-1, -1, -1, -1])
    ep_data["step_weights"].extend([1.0, 1.0, 1.0, 1.0])
    inventory_sources = {0: None, 1: None}
    focus_records = []

    counts = _update_pipeline_wrong_delivery_provenance(
        ep_data,
        num_agents=2,
        step=0,
        info={"events": {0: [{"event": "picked_resource", "resource_type": unneeded_type}]}},
        model_actions={0: {"action": env.ACTION_PICKUP}, 1: {"action": env.ACTION_STAY}},
        oracle_actions={0: {"action": env.ACTION_STAY}, 1: {"action": env.ACTION_STAY}},
        rollout_actions={0: {"action": env.ACTION_PICKUP}, 1: {"action": env.ACTION_STAY}},
        inventory_sources=inventory_sources,
        cfg=provenance_cfg,
        focus_records=focus_records,
        focus_events={PIPELINE_WRONG_DELIVERY_ROOT_PICKUP_EVENT},
    )

    assert counts == {"bad_pickup_labels": 0, "focus_records": 0, "focus_weight_updates": 0}
    assert inventory_sources[0]["resource_type"] == unneeded_type
    assert inventory_sources[0]["labelable"] is True

    counts = _update_pipeline_wrong_delivery_provenance(
        ep_data,
        num_agents=2,
        step=1,
        info={"events": {0: [{"event": "pipeline_wrong_delivery", "resource_type": unneeded_type}]}},
        model_actions={0: {"action": env.ACTION_INTERACT}, 1: {"action": env.ACTION_STAY}},
        oracle_actions={0: {"action": env.ACTION_STAY}, 1: {"action": env.ACTION_STAY}},
        rollout_actions={0: {"action": env.ACTION_INTERACT}, 1: {"action": env.ACTION_STAY}},
        inventory_sources=inventory_sources,
        cfg=provenance_cfg,
        focus_records=focus_records,
        focus_events={PIPELINE_WRONG_DELIVERY_ROOT_PICKUP_EVENT},
    )

    assert counts == {"bad_pickup_labels": 1, "focus_records": 1, "focus_weight_updates": 1}
    assert ep_data["pipeline_bad_pickup_action_mask"] == [1.0, 0.0, 0.0, 0.0]
    assert ep_data["pipeline_bad_pickup_action_id"] == [env.ACTION_PICKUP, -1, -1, -1]
    assert ep_data["step_weights"] == [2.5, 1.0, 1.0, 1.0]
    assert focus_records == [{
        "event": PIPELINE_WRONG_DELIVERY_ROOT_PICKUP_EVENT,
        "step": 0,
        "agents": [0],
        "kind": "provenance",
        "root_event": "pipeline_wrong_delivery",
        "root_step": 1,
        "resource_type": unneeded_type,
    }]

    ep_data = _new_episode_sequence()
    ep_data["pipeline_bad_pickup_action_mask"].extend([0.0, 0.0, 0.0, 0.0])
    ep_data["pipeline_bad_pickup_action_id"].extend([-1, -1, -1, -1])
    ep_data["step_weights"].extend([1.0, 1.0, 1.0, 1.0])
    inventory_sources = {0: None, 1: None}
    _update_pipeline_wrong_delivery_provenance(
        ep_data,
        num_agents=2,
        step=0,
        info={"events": {0: [{"event": "picked_resource", "resource_type": needed_type}]}},
        model_actions={0: {"action": env.ACTION_PICKUP}, 1: {"action": env.ACTION_STAY}},
        oracle_actions={0: {"action": env.ACTION_PICKUP}, 1: {"action": env.ACTION_STAY}},
        rollout_actions={0: {"action": env.ACTION_PICKUP}, 1: {"action": env.ACTION_STAY}},
        inventory_sources=inventory_sources,
        cfg=provenance_cfg,
        focus_records=[],
        focus_events={PIPELINE_WRONG_DELIVERY_ROOT_PICKUP_EVENT},
    )
    counts = _update_pipeline_wrong_delivery_provenance(
        ep_data,
        num_agents=2,
        step=1,
        info={"events": {0: [{"event": "pipeline_wrong_delivery", "resource_type": needed_type}]}},
        model_actions={0: {"action": env.ACTION_INTERACT}, 1: {"action": env.ACTION_STAY}},
        oracle_actions={0: {"action": env.ACTION_STAY}, 1: {"action": env.ACTION_STAY}},
        rollout_actions={0: {"action": env.ACTION_INTERACT}, 1: {"action": env.ACTION_STAY}},
        inventory_sources=inventory_sources,
        cfg=provenance_cfg,
        focus_records=[],
        focus_events={PIPELINE_WRONG_DELIVERY_ROOT_PICKUP_EVENT},
    )

    assert counts == {"bad_pickup_labels": 0, "focus_records": 0, "focus_weight_updates": 0}
    assert ep_data["pipeline_bad_pickup_action_mask"] == [0.0, 0.0, 0.0, 0.0]


def test_oracle_policies_all_scenarios():
    """Verify all oracle policies run without error."""
    from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
    from syncorsink.policies.oracle import (
        signal_hunt_oracle_strong, energy_oracle_strong, pipeline_oracle_strong,
    )
    for scenario, oracle_fn in [
        ("signal_hunt", signal_hunt_oracle_strong),
        ("energy_grid", energy_oracle_strong),
        ("pipeline_assembly", pipeline_oracle_strong),
    ]:
        config = SyncOrSinkConfig(
            scenario=scenario, map_size=8, num_agents=2,
            fov_preset="easy", energy_preset="easy",
        )
        env = SyncOrSinkEnv(config)
        obs, info = env.reset(seed=0)
        policy = oracle_fn(env)
        for _ in range(5):
            actions = policy(obs, info, {"step": 0})
            obs, rewards, done, truncated, info = env.step(actions)
            if done or truncated:
                break
