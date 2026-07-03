from __future__ import annotations


class StayPolicy:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents

    def reset(self, episode: int | None = None, seed: int | None = None):
        return None

    def metadata(self):
        return {
            "method_name": "StayPolicy",
            "method_type": "example",
        }

    def act(self, obs: dict, info: dict, state: dict):
        return {
            int(agent_id): {
                "action": 4,
                "message_tokens": [],
                "message_text": "",
            }
            for agent_id in obs
        }


def build_policy(env, spec):
    return StayPolicy(num_agents=env.num_agents)
