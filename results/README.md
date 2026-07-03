# Result Registry

This directory stores lightweight leaderboard result artifacts.

Rules:

- Commit result JSON files only.
- Do not commit checkpoints, traces, videos, demos, logs, or W&B runs.
- Each result artifact must validate against `syncorsink.result.v0.1`.
- Official `syncorsink_v0_1` submissions belong under `results/syncorsink_v0_1/`.

Rebuild the public leaderboard table:

```bash
python examples/build_leaderboard.py \
  --results results/syncorsink_v0_1 \
  --benchmark benchmarks/syncorsink_v0_1.json \
  --out-md docs/leaderboard_results.md \
  --out-csv docs/leaderboard_results.csv \
  --out-json docs/leaderboard_results.json
```

If you are testing partial or smoke artifacts locally, use `--allow-partial` and
write outputs to `/tmp` instead of committing them.
