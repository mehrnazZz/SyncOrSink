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
        updates=1,
        rollout_steps=8,
        epochs=1,
        minibatch=8,
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

    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["algorithm"] == "mappo"
    assert payload["config"]["comm"] is True
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
        obs_signal_negative_memory=True,
        obs_signal_negative_memory_window=12,
        obs_signal_inferred_target_features=True,
        bc_signal_redundant_target_interact_weight=1.5,
        hidden_dim=96,
        bc_eval_every_epochs=2,
        bc_eval_episodes=3,
        bc_eval_seed_count=2,
        bc_restore_best_eval_epoch=True,
        bc_signal_target_pursuit_weight=2.0,
        bc_signal_target_pursuit_action_weight=0.9,
        bc_signal_sync_response_weight=2.5,
        bc_signal_sync_response_action_loss_weight=1.25,
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
        eval_signal_target_validity_threshold=0.55,
        eval_signal_target_decision_threshold=0.6,
        eval_signal_target_decision_suppress=False,
        eval_signal_scan_broadcast_assist=True,
        eval_signal_exact_target_message_guard=True,
        eval_signal_exact_target_navigation_assist=True,
        eval_signal_exact_target_memory_steps=24,
        eval_signal_scan_refresh_assist=True,
        eval_signal_scan_refresh_threshold=0.5,
        dagger_focus_error_weight=4.0,
        dagger_focus_recovery_weight=2.5,
        dagger_focus_window=3,
        dagger_target_discovery_min_map_size=8,
        dagger_target_discovery_focus_weight=4.25,
        dagger_movement_stall_min_map_size=8,
        dagger_movement_stall_window=4,
        dagger_movement_stall_focus_weight=5.5,
        dagger_solo_target_team_weight=2.25,
        dagger_solo_target_team_success_only=True,
        dagger_positive_target_pursuit_min_map_size=8,
        dagger_replay_priority_events="movement_stall_miss",
        dagger_replay_balance_positive_events="first_target_scan,joint_target_scan",
        dagger_replay_balance_negative_events="decoy_scan,rejected_target_scan",
        dagger_replay_max_negative_per_positive=0.5,
        dagger_expert_max_replay_snippets_per_episode=3,
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
    assert result["config"]["bc_signal_sync_response_action_loss_weight"] == pytest.approx(1.25)
    assert result["config"]["bc_signal_redundant_target_wait_action_loss_weight"] == pytest.approx(1.4)
    assert result["config"]["bc_signal_rejected_target_interact_action_loss_weight"] == pytest.approx(0.7)
    assert result["config"]["bc_signal_target_validity_loss_weight"] == pytest.approx(0.6)
    assert result["config"]["bc_signal_target_decision_loss_weight"] == pytest.approx(0.4)
    assert result["config"]["eval_signal_target_validity_threshold"] == pytest.approx(0.55)
    assert result["config"]["eval_signal_target_decision_threshold"] == pytest.approx(0.6)
    assert result["config"]["eval_signal_target_decision_suppress"] is False
    assert result["config"]["eval_signal_scan_broadcast_assist"] is True
    assert result["config"]["eval_signal_exact_target_message_guard"] is True
    assert result["config"]["eval_signal_exact_target_navigation_assist"] is True
    assert result["config"]["eval_signal_exact_target_memory_steps"] == 24
    assert result["config"]["eval_signal_scan_refresh_assist"] is True
    assert result["config"]["eval_signal_scan_refresh_threshold"] == pytest.approx(0.5)
    assert result["config"]["initial_recurrent_checkpoint"] == "logs/recurrent_curriculum/example.pt"
    assert result["config"]["hidden_dim"] == 96
    assert result["config"]["bc_eval_every_epochs"] == 2
    assert result["config"]["bc_eval_episodes"] == 3
    assert result["config"]["bc_eval_seed_count"] == 2
    assert result["config"]["bc_restore_best_eval_epoch"] is True
    assert result["config"]["train_map_sampling_weights"] == "8:1,16:3"
    assert result["config"]["obs_exploration_age"] is True
    assert result["config"]["obs_signal_negative_memory"] is True
    assert result["config"]["dagger_solo_target_team_weight"] == pytest.approx(2.25)
    assert result["config"]["dagger_focus_error_weight"] == pytest.approx(4.0)
    assert result["config"]["dagger_focus_recovery_weight"] == pytest.approx(2.5)
    assert result["config"]["dagger_focus_window"] == 3
    assert result["config"]["dagger_movement_stall_min_map_size"] == 8
    assert result["config"]["dagger_movement_stall_window"] == 4
    assert result["config"]["dagger_movement_stall_focus_weight"] == pytest.approx(5.5)
    assert result["config"]["dagger_positive_target_pursuit_min_map_size"] == 8
    assert result["config"]["dagger_replay_priority_events"] == "movement_stall_miss"
    assert result["config"]["dagger_replay_balance_positive_events"] == (
        "first_target_scan,joint_target_scan"
    )
    assert result["config"]["dagger_replay_balance_negative_events"] == (
        "decoy_scan,rejected_target_scan"
    )
    assert result["config"]["dagger_replay_max_negative_per_positive"] == pytest.approx(0.5)
    assert result["config"]["dagger_expert_max_replay_snippets_per_episode"] == 3
    assert result["planned_stages"][0]["train_map_sizes"] == [8]
    assert result["planned_stages"][1]["train_map_sizes"] == [8, 16]
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
    assert stage_cfg.train_map_sampling_weights == "8:1,16:3"
    assert stage_cfg.obs_exploration_age is True
    assert stage_cfg.obs_signal_negative_memory_window == 12
    assert stage_cfg.obs_signal_inferred_target_features is True
    assert stage_cfg.hidden_dim == 96
    assert stage_cfg.bc_eval_every_epochs == 2
    assert stage_cfg.bc_eval_episodes == 3
    assert stage_cfg.bc_eval_seed_count == 2
    assert stage_cfg.bc_restore_best_eval_epoch is True
    assert stage_cfg.bc_signal_redundant_target_interact_weight == pytest.approx(1.5)
    assert stage_cfg.bc_signal_target_pursuit_weight == pytest.approx(2.0)
    assert stage_cfg.bc_signal_target_pursuit_action_weight == pytest.approx(0.9)
    assert stage_cfg.bc_signal_sync_response_weight == pytest.approx(2.5)
    assert stage_cfg.bc_signal_sync_response_action_loss_weight == pytest.approx(1.25)
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
    assert stage_cfg.eval_signal_target_validity_threshold == pytest.approx(0.55)
    assert stage_cfg.eval_signal_target_decision_threshold == pytest.approx(0.6)
    assert stage_cfg.eval_signal_target_decision_suppress is False
    assert stage_cfg.eval_signal_scan_broadcast_assist is True
    assert stage_cfg.eval_signal_exact_target_message_guard is True
    assert stage_cfg.eval_signal_exact_target_navigation_assist is True
    assert stage_cfg.eval_signal_exact_target_memory_steps == 24
    assert stage_cfg.eval_signal_scan_refresh_assist is True
    assert stage_cfg.eval_signal_scan_refresh_threshold == pytest.approx(0.5)
    assert stage_cfg.dagger_focus_error_weight == pytest.approx(4.0)
    assert stage_cfg.dagger_focus_recovery_weight == pytest.approx(2.5)
    assert stage_cfg.dagger_focus_window == 3
    assert stage_cfg.dagger_target_discovery_min_map_size == 8
    assert stage_cfg.dagger_target_discovery_focus_weight == pytest.approx(4.25)
    assert stage_cfg.dagger_movement_stall_min_map_size == 8
    assert stage_cfg.dagger_movement_stall_window == 4
    assert stage_cfg.dagger_movement_stall_focus_weight == pytest.approx(5.5)
    assert stage_cfg.dagger_solo_target_team_weight == pytest.approx(2.25)
    assert stage_cfg.dagger_solo_target_team_success_only is True
    assert stage_cfg.dagger_positive_target_pursuit_min_map_size == 8
    assert stage_cfg.dagger_replay_priority_events == "movement_stall_miss"
    assert stage_cfg.dagger_replay_balance_positive_events == "first_target_scan,joint_target_scan"
    assert stage_cfg.dagger_replay_balance_negative_events == "decoy_scan,rejected_target_scan"
    assert stage_cfg.dagger_replay_max_negative_per_positive == pytest.approx(0.5)
    assert stage_cfg.dagger_expert_max_replay_snippets_per_episode == 3

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
    from syncorsink.envs.maps import TILE_TARGET
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _SIGNAL_TARGET_SCAN_KIND_FIRST,
        _SIGNAL_TARGET_SCAN_KIND_JOINT_COMPLETION,
        _SIGNAL_TARGET_SCAN_KIND_REDUNDANT_ACTIVE,
        _SIGNAL_TARGET_SCAN_KIND_REFRESH,
        _apply_deferred_solo_target_team_weights,
        _apply_redundant_target_scan_penalty,
        _apply_signal_exact_target_message_guard,
        _apply_signal_scan_broadcast_assist,
        _apply_signal_scan_gate_decoding,
        _apply_signal_scan_refresh_decoding,
        _apply_signal_scan_sync_decoding,
        _apply_signal_target_decision_decoding,
        _apply_signal_target_validity_decoding,
        _apply_signal_target_scan_decoding,
        _apply_signal_redundant_target_wait_overrides,
        _apply_wrong_target_scan_penalty,
        _append_labeled_step,
        _feedback_dim,
        _new_episode_sequence,
        _redundant_target_scan_agents,
        _signal_center_target_scan_decoding_candidate,
        _signal_bad_redundant_target_interact_agents,
        _signal_bad_redundant_target_interact_loss,
        _signal_bad_redundant_target_mask,
        _signal_target_interact_agents,
        _scale_solo_target_team_weights,
        _split_solo_target_scan_agents,
        _signal_scan_decision_loss,
        _signal_scan_gate_loss,
        _signal_redundant_target_wait_action_label_mask,
        _signal_target_decision_label_mask,
        _signal_target_decision_loss,
        _signal_target_interact_miss_agents,
        _signal_target_scan_action_loss,
        _signal_target_scan_kind,
        _signal_target_scan_opportunity_label_mask,
        _signal_target_validity_label,
        _signal_target_validity_loss,
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
    scan_cfg.eval_signal_target_validity_threshold = 0.5
    validity_actions = _apply_signal_target_validity_decoding(
        scan_cfg,
        decode_obs,
        interact_actions,
        torch.tensor([2.0, -2.0], dtype=torch.float32),
    )
    assert validity_actions.tolist() == [env.ACTION_INTERACT, env.ACTION_STAY]
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
    from syncorsink.envs.maps import TILE_BEACON, TILE_TARGET
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _append_labeled_step,
        _apply_signal_exact_target_navigation_assist,
        _apply_signal_target_handoff_overrides,
        _apply_signal_target_scan_broadcast_overrides,
        _feedback_matrix,
        _finalize_episode_sequence,
        _label_latest_signal_decoy_drift_actions,
        _label_latest_signal_decoy_scan_actions,
        _label_latest_signal_rejected_target_drift_actions,
        _new_episode_sequence,
        _signal_decoy_pursuit_agents,
        _signal_decoy_drift_action_loss,
        _signal_movement_stall_miss_agents,
        _signal_observation_allows_target,
        _signal_positive_target_pursuit_agents,
        _signal_rejected_target_drift_agents,
        _signal_target_decoy_drift_miss_agents,
        _signal_target_discovery_miss_agents,
        _signal_target_handoff_miss_agents,
        _signal_target_match_action_label_mask,
        _signal_target_match_action_loss,
        _signal_target_pursuit_action_label_mask,
        _signal_target_pursuit_miss_agents,
        _signal_target_pursuit_agents,
        _signal_target_scan_broadcaster_agents,
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

    large_env.agent_positions[0] = (8, 8)
    large_env.agent_positions[1] = tuple(int(v) for v in large_env.agent_positions[1])
    large_env.steps = 0
    large_env.scenario_state.data["scan_log"] = {}
    large_cfg = RecurrentConfig(scenario="signal_hunt", map_size=16, agents=2)
    large_ep_data = _new_episode_sequence()
    _append_labeled_step(large_ep_data, large_obs, large_oracle, large_env, large_cfg)
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
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _append_labeled_step,
        _feedback_matrix,
        _new_episode_sequence,
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

    def fake_evaluate_recurrent_policy_multi_seed(cfg, model, device, *, seed_count):
        del cfg, model, device
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

    def fake_evaluate_recurrent_policy_multi_seed(cfg, model, device, *, seed_count):
        del cfg, model, device
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

    def fake_evaluate_recurrent_policy_multi_seed(cfg, model, device, *, seed_count):
        del cfg, model, device
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


def test_recurrent_checkpoint_policy_egocentric_memory_cross_map_size(tmp_path):
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.recurrent_bc_rl import (
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
    assert rollout["reset_after_buf"][1] is True
    assert rollout["reset_after_buf"][3] is True
    assert rollout["reset_after_buf"][5] is True

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


def test_recurrent_comm_length_loss_ignores_no_message_examples():
    from syncorsink.train.recurrent_bc_rl import (
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
    assert _send_threshold_for_target_rate([0.1, 0.2, 0.8, 0.9], 0.5) == pytest.approx(0.5)
    assert _send_threshold_for_target_rate([0.1, 0.2, 0.8, 0.9], 0.0) == pytest.approx(1.0)
    assert _send_threshold_for_target_rate([0.1, 0.2, 0.8, 0.9], 1.0) == pytest.approx(0.0)
    loss.backward()

    assert torch.allclose(len_logits.grad[0], torch.zeros_like(len_logits.grad[0]))
    assert len_logits.grad[1].abs().sum().item() > 0.0


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


def test_recurrent_actor_checkpoint_init_for_rl_smoke(tmp_path):
    from syncorsink.policies.mappo_models import MAPPORecurrentActor
    from syncorsink.train.mappo import resolve_device
    from syncorsink.train.recurrent_bc_rl import (
        RecurrentConfig,
        _build_env,
        _build_recurrent_obs_batch,
        _feedback_matrix,
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
    legacy_checkpoint = tmp_path / "recurrent_init_legacy_no_scan_gate.pt"
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith((
            "signal_scan_gate.",
            "signal_target_validity.",
            "signal_target_decision.",
            "signal_target_aux.",
        ))
    }
    torch.save({"model": legacy_state, "config": vars(cfg)}, legacy_checkpoint)
    legacy_loaded = load_recurrent_actor_checkpoint(legacy_checkpoint, cfg, device)
    assert hasattr(legacy_loaded, "signal_scan_gate")
    assert hasattr(legacy_loaded, "signal_target_validity")
    assert hasattr(legacy_loaded, "signal_target_decision")
    assert hasattr(legacy_loaded, "signal_target_aux")
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
        comm=True,
        comm_token_limit=4,
        comm_vocab_size=8,
        demo_episodes=2,
        bc_epochs=1,
        bc_seq_len=8,
        bc_comm_loss_weight=0.1,
        bc_comm_send_pos_weight=-1,
        rl_updates=1,
        rollout_steps=4,
        rl_epochs=1,
        rl_eval_every=1,
        rl_eval_episodes=1,
        rl_eval_seed=4000,
        rl_restore_best=True,
        rl_save_best=True,
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
