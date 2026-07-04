# External Policy Submissions

SyncOrSink can evaluate external policies without editing the repository. A
submission exposes a Python entrypoint:

```text
module.submodule:object
```

The object should be a factory, class, or callable that returns a policy. The
recommended factory signature is:

```python
def build_policy(env, spec):
    return MyPolicy(env, spec)
```

Run it with:

```bash
python examples/benchmark_run.py \
  --spec benchmarks/syncorsink_v0_1.json \
  --policy-entrypoint my_package.my_agent:build_policy \
  --results-json results/my_agent_v0_1.json \
  --track symbolic_dtde \
  --submission-name my-agent-v0.1 \
  --method-name "My Agent" \
  --method-type "Transformer MARL" \
  --authors "First Author,Second Author"
```

## Policy Interface

Leaderboard-style external submissions use decentralized execution by default.
The policy is called once per agent with only that agent's observation, that
agent's incoming messages/events, and public scalar info:

```python
class MyPolicy:
    def reset(self, episode=None, seed=None):
        pass

    def load_checkpoint(self, path):
        pass

    def metadata(self):
        return {
            "method_name": "My Agent",
            "method_type": "Transformer MARL",
        }

    def act_agent(self, agent_id, obs, info, state):
        return {
            "action": 4,
            "message_tokens": [],
            "message_text": "",
        }
```

`obs` is the per-agent observation for `agent_id`, not the full team
observation dict. `info` includes only that agent's `messages_text`,
`messages_with_sender`, `events`, `comm_tokens`, and public scalar fields.
It does not include `central_obs` during external policy execution.

An agent-level callable is also accepted:

```python
def policy(agent_id, obs, info, state):
    return {
        "action": 4,
        "message_tokens": [],
    }
```

For local debugging only, `examples/benchmark_run.py` supports
`--allow-centralized-external-policy`, which restores the older whole-team
interface:

```python
class MyPolicy:
    def act(self, obs, info, state):
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[int(agent_id)] = {
                "action": 4,
                "message_tokens": [],
                "message_text": "",
            }
        return actions
```

Do not use centralized external execution for `symbolic_dtde` or leaderboard
submissions. Built-in oracle/debug policies may still use centralized state and
are reported on separate tracks.

The required action payload is:

- `action`: integer action id from `0..7`
- `message_tokens`: list of integer communication tokens for token tracks
- `message_text`: optional string for text/LLM tracks

`reset` is called at the start of each episode when present. `load_checkpoint` is
called when a checkpoint is supplied and the factory did not receive a
`checkpoint` keyword argument.

## Factory Arguments

SyncOrSink passes only the keyword arguments accepted by the entrypoint
signature. Supported names:

- `env`: initialized `SyncOrSinkEnv` for the current case
- `spec`: case spec dictionary
- `checkpoint`: optional checkpoint path or URI
- any keys supplied through `--policy-kwargs`

Example with extra kwargs:

```bash
python examples/benchmark_run.py \
  --spec benchmarks/syncorsink_v0_1.json \
  --policy-entrypoint my_package.my_agent:build_policy \
  --policy-checkpoint /path/to/checkpoint.pt \
  --policy-kwargs '{"temperature": 0.0, "device": "cuda"}'
```

## Minimal Example

The repository includes a no-op example:

```bash
python examples/benchmark_run.py \
  --spec examples/benchmark_spec.json \
  --policy-entrypoint examples.external_policy:build_policy \
  --results-json /tmp/syncorsink_external_policy_smoke.json \
  --track symbolic_dtde \
  --submission-name external-policy-smoke \
  --method-name StayPolicy \
  --method-type example \
  --authors SyncOrSink
```

This policy always stays still, so it is only a loader smoke test.

## Checkpoint Policy

Do not commit checkpoints to git. Use `--policy-checkpoint` for local evaluation
and `--checkpoint-uri` in result artifacts for public submissions. Recommended
storage options are Hugging Face Hub, W&B Artifacts, GitHub Releases, or
institutional object storage.
