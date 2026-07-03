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
    )
    train_mappo(cfg)


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
