from __future__ import annotations

import numpy as np

from .pathing import shortest_path, move_action_from_delta


def _blocked_positions(env, agent_id, allow_positions=None):
    blocked = set(env.agent_positions)
    blocked.discard(env.agent_positions[agent_id])
    if allow_positions:
        blocked.difference_update(allow_positions)
    return blocked


# Scripted policies with global pathing using full env state.


def pipeline_planner(env):
    def _policy(obs, info, state):
        actions = {}
        # choose next incomplete stage
        stages = env.scenario_state.data.get("stages", [])
        open_stages = [s for s in stages if not s["done"]]
        target_stage = open_stages[0] if open_stages else None

        for agent_id in range(env.num_agents):
            pos = env.agent_positions[agent_id]
            inv = env.inventories[agent_id]
            if target_stage is None:
                actions[agent_id] = {"action": env.ACTION_STAY, "message_tokens": []}
                continue

            # if at station and holding required item, interact
            if pos == target_stage["station"] and inv in target_stage["required"]:
                actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue

            # if not holding resource, go to closest required resource
            if inv == 0:
                candidates = [p for p, t in env.scenario_state.data["resource_types"].items() if t in target_stage["required"]]
                if candidates:
                    blocked = _blocked_positions(env, agent_id, set(candidates))
                    dx, dy, _ = shortest_path(env.grid, pos, set(candidates), blocked_positions=blocked)
                    act = move_action_from_delta(dx, dy, env)
                    actions[agent_id] = {"action": act, "message_tokens": []}
                else:
                    actions[agent_id] = {"action": env.ACTION_STAY, "message_tokens": []}
                continue

            # holding resource: go to station
            blocked = _blocked_positions(env, agent_id, {target_stage["station"]})
            dx, dy, _ = shortest_path(env.grid, pos, {target_stage["station"]}, blocked_positions=blocked)
            act = move_action_from_delta(dx, dy, env)
            actions[agent_id] = {"action": act, "message_tokens": []}

        return actions

    return _policy


def energy_planner(env):
    def _policy(obs, info, state):
        actions = {}
        node_energy = env.scenario_state.data.get("node_energy", {})
        node_types = env.scenario_state.data.get("node_types", {})
        # prioritize lowest-energy node
        nodes_sorted = sorted(node_energy.items(), key=lambda kv: kv[1])
        target_node = nodes_sorted[0][0] if nodes_sorted else None

        for agent_id in range(env.num_agents):
            pos = env.agent_positions[agent_id]
            inv = env.inventories[agent_id]
            if target_node is None:
                actions[agent_id] = {"action": env.ACTION_STAY, "message_tokens": []}
                continue

            # if at node with matching resource, interact
            if pos == target_node and inv == node_types.get(target_node, 0):
                actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue

            # if not holding resource, seek matching resource
            if inv == 0:
                needed = node_types.get(target_node, 0)
                candidates = [p for p, t in env.scenario_state.data["resource_types"].items() if t == needed]
                if candidates:
                    blocked = _blocked_positions(env, agent_id, set(candidates))
                    dx, dy, _ = shortest_path(env.grid, pos, set(candidates), blocked_positions=blocked)
                    act = move_action_from_delta(dx, dy, env)
                    actions[agent_id] = {"action": act, "message_tokens": []}
                else:
                    actions[agent_id] = {"action": env.ACTION_STAY, "message_tokens": []}
                continue

            # holding resource: go to node
            blocked = _blocked_positions(env, agent_id, {target_node})
            dx, dy, _ = shortest_path(env.grid, pos, {target_node}, blocked_positions=blocked)
            act = move_action_from_delta(dx, dy, env)
            actions[agent_id] = {"action": act, "message_tokens": []}

        return actions

    return _policy


def signal_hunt_planner(env):
    def _policy(obs, info, state):
        actions = {}
        target = env.scenario_state.data.get("target")
        clue_tiles = set(env.meta.get("clues", []))

        for agent_id in range(env.num_agents):
            pos = env.agent_positions[agent_id]
            # if at clue or target, interact
            if pos in clue_tiles or pos == target:
                actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue

            # otherwise go to nearest clue
            blocked = _blocked_positions(env, agent_id, set(clue_tiles))
            dx, dy, _ = shortest_path(env.grid, pos, clue_tiles, blocked_positions=blocked)
            act = move_action_from_delta(dx, dy, env)
            actions[agent_id] = {"action": act, "message_tokens": []}

        return actions

    return _policy
