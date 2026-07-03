# SyncOrSink v0.1 Results

Place official `syncorsink_v0_1` result artifacts in this directory.

Recommended naming:

```text
<track>/<submission-slug>.json
```

Examples:

```text
symbolic_dtde/random.json
symbolic_dtde/oracle_strong.json
llm_text/gpt-4o-mini-tools.json
```

Generate a result artifact with:

```bash
python examples/benchmark_run.py \
  --spec benchmarks/syncorsink_v0_1.json \
  --policy-override random \
  --results-json results/syncorsink_v0_1/symbolic_dtde/random.json \
  --track symbolic_dtde \
  --submission-name random-v0.1 \
  --method-name Random \
  --method-type random \
  --authors "SyncOrSink Contributors"
```

Starter built-in baseline commands:

```bash
python examples/benchmark_run.py \
  --spec benchmarks/syncorsink_v0_1.json \
  --policy-override scripted \
  --results-json results/syncorsink_v0_1/symbolic_dtde/scripted.json \
  --track symbolic_dtde \
  --submission-name scripted-v0.1 \
  --method-name Scripted \
  --method-type scripted \
  --authors "SyncOrSink Contributors"

python examples/benchmark_run.py \
  --spec benchmarks/syncorsink_v0_1.json \
  --policy-override oracle_strong \
  --results-json results/syncorsink_v0_1/symbolic_ctde/oracle_strong.json \
  --track symbolic_ctde \
  --submission-name oracle-strong-v0.1 \
  --method-name Oracle Strong \
  --method-type oracle \
  --authors "SyncOrSink Contributors"
```

External submissions should use `--policy-entrypoint`.

Result files in this directory should be complete official-suite runs. Do not
commit partial smoke runs here.
