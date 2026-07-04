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

    def act_agent(self, agent_id: int, obs: dict, info: dict, state: dict):
        return {
            "action": 4,
            "message_tokens": [],
            "message_text": "",
        }


def build_policy(env, spec):
    return StayPolicy(num_agents=env.num_agents)
