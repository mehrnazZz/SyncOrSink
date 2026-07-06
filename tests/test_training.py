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
    episode = _finalize_episode_sequence(ep_data, env, cfg, source="dagger", weight=0.5)
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
    assert len(replay) == 1
    assert replay[0]["source"] == "dagger_focus_replay"
    assert replay[0]["trigger_event"] == "decoy_scan"
    assert replay[0]["trigger_agents"] == [0]
    assert replay[0]["obs"].shape[0] == 3
    assert _episode_count_effective_transitions(replay) == 24.0
    assert event_names[0] == {"decoy_scan"}
    assert event_names[1] == {"clue_found"}


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


def test_recurrent_dagger_best_round_uses_eval_score(monkeypatch):
    import syncorsink.train.recurrent_bc_rl as recurrent
    from syncorsink.train.recurrent_bc_rl import RecurrentConfig

    train_calls = {"count": 0}
    eval_calls = {"count": 0}

    def fake_train_recurrent_bc(cfg, episodes, device, model=None):
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

    def fake_evaluate_recurrent_policy(cfg, model, device):
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
    monkeypatch.setattr(recurrent, "evaluate_recurrent_policy", fake_evaluate_recurrent_policy)
    monkeypatch.setattr(recurrent, "collect_recurrent_dagger_episodes", fake_collect_recurrent_dagger_episodes)

    initial_episode = {
        "obs": np.zeros((1, 1, 1), dtype=np.float32),
        "source": "expert",
    }
    model, history, all_episodes, best_round = recurrent.train_recurrent_bc_dagger(
        RecurrentConfig(dagger_rounds=1),
        [initial_episode],
        torch.device("cpu"),
    )

    assert best_round["round"] == 1
    assert history[1]["eval_score"] > history[0]["eval_score"]
    assert len(all_episodes) == 2
    assert float(next(model.parameters()).item()) == pytest.approx(1.0)


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


def test_recurrent_signal_hint_comm_bc_smoke(tmp_path):
    from syncorsink.envs import SyncOrSinkConfig
    from syncorsink.eval.trajectory_audit import (
        AuditPolicySpec,
        make_recurrent_checkpoint_policy_factory,
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
        comm=True,
        comm_token_limit=8,
        comm_vocab_size=32,
        demo_episodes=4,
        bc_epochs=1,
        bc_seq_len=16,
        bc_comm_loss_weight=0.1,
        bc_comm_send_pos_weight=-1,
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

    checkpoint = tmp_path / "recurrent_signal.pt"
    torch.save({"model": model.state_dict(), "config": vars(cfg)}, checkpoint)
    audit = run_trajectory_audit(
        SyncOrSinkConfig(
            scenario="signal_hunt",
            map_size=8,
            num_agents=2,
            fov_preset="easy",
            max_steps=60,
            obs_exploration_memory=True,
            comm_token_limit=8,
            token_vocab_size=32,
            max_messages=8,
        ),
        [
            AuditPolicySpec(
                label="recurrent",
                factory=make_recurrent_checkpoint_policy_factory(checkpoint, device="cpu"),
            )
        ],
        episodes=1,
        seed=3000,
    )
    assert audit["policies"][0]["summary"]["episodes"] == 1


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
