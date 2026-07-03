# Masked Learned-Policy Smoke Sweep

This note records a compact local smoke sweep after invalid-action masking was applied across learned-policy paths.

These are not benchmark-quality results. Budgets are intentionally tiny: the goal is to verify that masked rollout, update, and eval paths run across scenarios.

## Common Settings

- CPU device
- `map_size=8`
- `fov_preset=easy`
- `updates=3`
- `rollout_steps=64`
- `epochs=2`
- `minibatch=32`
- `eval_every=3`
- `eval_episodes=2`
- `seed=0`
- no W&B logging
- no checkpoint saved

## Commands

```bash
python examples/mappo_train.py --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy --updates 3 --rollout-steps 64 --epochs 2 --minibatch 32 --device cpu --seed 0 --eval-every 3 --eval-episodes 2 --max-steps 60
python examples/mappo_train.py --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy --energy-preset easy --updates 3 --rollout-steps 64 --epochs 2 --minibatch 32 --device cpu --seed 0 --eval-every 3 --eval-episodes 2 --max-steps 80
python examples/mappo_train.py --scenario pipeline_assembly --map-size 8 --agents 3 --fov-preset easy --updates 3 --rollout-steps 64 --epochs 2 --minibatch 32 --device cpu --seed 0 --eval-every 3 --eval-episodes 2 --max-steps 80

python examples/tarmac_train.py --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy --updates 3 --rollout-steps 64 --epochs 2 --minibatch 32 --device cpu --seed 0 --eval-every 3 --eval-episodes 2 --max-steps 60
python examples/tarmac_train.py --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy --energy-preset easy --updates 3 --rollout-steps 64 --epochs 2 --minibatch 32 --device cpu --seed 0 --eval-every 3 --eval-episodes 2 --max-steps 80

python examples/comm_mat_train.py --scenario signal_hunt --map-size 8 --agents 2 --fov-preset easy --updates 3 --rollout-steps 64 --epochs 2 --minibatch 32 --device cpu --seed 0 --eval-every 3 --eval-episodes 2 --max-steps 60
python examples/comm_mat_train.py --scenario energy_grid --map-size 8 --agents 3 --fov-preset easy --energy-preset easy --updates 3 --rollout-steps 64 --epochs 2 --minibatch 32 --device cpu --seed 0 --eval-every 3 --eval-episodes 2 --max-steps 80
python examples/comm_mat_train.py --scenario pipeline_assembly --map-size 8 --agents 3 --fov-preset easy --updates 3 --rollout-steps 64 --epochs 2 --minibatch 32 --device cpu --seed 0 --eval-every 3 --eval-episodes 2 --max-steps 80
```

## Final Eval Summary

| Method | Scenario | Eval Return | Eval Steps | Success |
|---|---|---:|---:|---:|
| MAPPO | signal_hunt | 0.00 | 60.0 | 0.00 |
| MAPPO | energy_grid | -14.50 | 36.0 | 0.00 |
| MAPPO | pipeline_assembly | 0.00 | 80.0 | 0.00 |
| TarMAC | signal_hunt | 0.00 | 60.0 | 0.00 |
| TarMAC | energy_grid | -14.50 | 36.0 | 0.00 |
| Comm-MAT | signal_hunt | -0.68 | 60.0 | 0.00 |
| Comm-MAT | energy_grid | -39.84 | 36.0 | 0.00 |
| Comm-MAT | pipeline_assembly | -14.40 | 80.0 | 0.00 |

## Takeaway

All masked learned-policy paths exercised here completed rollout, PPO update, and eval without invalid-action masking crashes or shape issues. The zero success rates are expected at this budget and should not be compared to the longer experiment-report baselines.

The next benchmark-quality step is to run longer multi-seed sweeps using explicit `--seed` values.
