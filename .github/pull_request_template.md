## Summary

Describe the change and the motivation.

## Type

- [ ] Environment/scenario change
- [ ] Baseline/model change
- [ ] Leaderboard/result submission
- [ ] Documentation
- [ ] Infrastructure/CI

## Validation

- [ ] `pytest tests`
- [ ] `python -m compileall syncorsink examples tests`
- [ ] `python examples/validate_leaderboard.py`

## Leaderboard Submissions

If this PR adds or updates result artifacts:

- [ ] Result JSON is under `results/syncorsink_v0_1/`
- [ ] `docs/leaderboard_results.md`, `.csv`, and `.json` were regenerated
- [ ] The submission declares the correct track
- [ ] The result artifact includes method name, method type, authors, and any repository/paper/checkpoint URI
- [ ] No checkpoints, traces, videos, demos, logs, W&B runs, or secrets are committed

## Notes

Add any caveats, known limitations, or follow-up work.
