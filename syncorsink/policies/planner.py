from __future__ import annotations

from collections import Counter
from typing import Dict

from .pathing import shortest_path, shortest_path_distance, move_action_from_delta
from .oracle import _move_or_open, _move_to_any_or_open


def _blocked_positions(env, agent_id, allow_positions=None):
    blocked = set(env.agent_positions)
    blocked.discard(env.agent_positions[agent_id])
    if allow_positions:
        blocked.difference_update(allow_positions)
    return blocked


def pipeline_central_planner(env):
    """
    Centralized planner for pipeline assembly. Uses full state, explicit assignment.
    """
    def _policy(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        stages = env.scenario_state.data.get("stages", [])
        open_stages = [s for s in stages if not s["done"]]
        available = []
        for s in open_stages:
            deps_done = all(stages[d]["done"] for d in s.get("deps", []))
            if deps_done:
                available.append(s)
        open_stages = available if available else open_stages
        if not open_stages:
            return {i: {"action": env.ACTION_STAY, "message_tokens": []} for i in range(env.num_agents)}
        # choose stage with minimal remaining needs
        def _need_count(s):
            return sum((Counter(s["required"]) - Counter(s.get("delivered", []))).values())
        stage = sorted(open_stages, key=_need_count)[0]
        station = stage["station"]
        required = stage["required"]
        delivered = stage.get("delivered", [])

        # Compute needs as a multiset. ``remaining_needs`` are not delivered
        # yet; ``pickup_needs`` also subtract resources already being carried so
        # empty agents do not fetch duplicates and block carriers.
        need_counts = Counter(required) - Counter(delivered)
        remaining_needs = []
        for k, v in need_counts.items():
            remaining_needs.extend([k] * v)

        pickup_counts = need_counts.copy()
        for inv in env.inventories:
            if pickup_counts.get(inv, 0) > 0:
                pickup_counts[inv] -= 1
        pickup_needs = []
        for k, v in pickup_counts.items():
            pickup_needs.extend([k] * v)

        # if no needs and sync required, converge & sync
        if stage["sync"] and not remaining_needs:
            for aid in range(env.num_agents):
                pos = env.agent_positions[aid]
                if pos == station:
                    actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                else:
                    actions[aid] = {"action": _move_or_open(env, aid, pos, station), "message_tokens": []}
            return actions

        # assign agents to needs (greedy by distance)
        free_agents = [aid for aid in range(env.num_agents) if env.inventories[aid] == 0]
        assigned = {}
        for need in pickup_needs:
            best = None
            best_cost = None
            best_pos = None
            for aid in free_agents:
                pos = env.agent_positions[aid]
                candidates = [p for p, t in env.scenario_state.data["resource_types"].items() if t == need]
                if not candidates:
                    continue
                dist = shortest_path_distance(env.grid, pos, set(candidates))
                if dist is None:
                    continue
                if best_cost is None or dist < best_cost:
                    best = aid
                    best_cost = dist
                    best_pos = candidates
            if best is not None:
                assigned[best] = (need, best_pos)
                free_agents.remove(best)

        # decide actions
        for aid in range(env.num_agents):
            pos = env.agent_positions[aid]
            inv = env.inventories[aid]

            # if holding required resource, deliver
            if inv in required:
                if pos == station:
                    actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                else:
                    actions[aid] = {"action": _move_or_open(env, aid, pos, station), "message_tokens": []}
                continue

            # drop useless inventory to free hands
            if inv != 0 and inv not in required:
                if env.grid[pos[1], pos[0]] == 0:
                    actions[aid] = {"action": env.ACTION_DROP, "message_tokens": []}
                    continue

            if inv == 0 and aid in assigned:
                need, candidates = assigned[aid]
                current = env.scenario_state.data["resource_types"].get(pos)
                if current == need:
                    actions[aid] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                    continue
                actions[aid] = {"action": _move_to_any_or_open(env, aid, pos, candidates), "message_tokens": []}
                continue

            if not pickup_needs and stage["sync"]:
                if pos == station:
                    actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                else:
                    actions[aid] = {"action": _move_or_open(env, aid, pos, station), "message_tokens": []}
                continue

            # fallback: move toward station
            actions[aid] = {"action": _move_or_open(env, aid, pos, station), "message_tokens": []}

        return actions

    return _policy


def energy_central_planner(env):
    """
    Centralized planner for energy grid. Uses full state, prioritizes lowest-energy nodes.
    """
    def _policy(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        node_energy = env.scenario_state.data.get("node_energy", {})
        node_types = env.scenario_state.data.get("node_types", {})
        if not node_energy:
            return {i: {"action": env.ACTION_STAY, "message_tokens": []} for i in range(env.num_agents)}

        # pick lowest-energy node (critical)
        target_node = sorted(node_energy.items(), key=lambda kv: kv[1])[0][0]
        target_type = node_types.get(target_node, 0)

        for aid in range(env.num_agents):
            pos = env.agent_positions[aid]
            inv = env.inventories[aid]

            # if carrying a resource, deliver to matching node with lowest energy
            if inv != 0:
                matching_nodes = [n for n, t in node_types.items() if t == inv]
                if matching_nodes:
                    target_node = sorted(matching_nodes, key=lambda n: node_energy.get(n, 0))[0]
                    target_type = inv
                if pos == target_node and inv == target_type:
                    actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                else:
                    dx, dy, _ = shortest_path(env.grid, pos, {target_node})
                    actions[aid] = {"action": move_action_from_delta(dx, dy, env), "message_tokens": []}
                continue

            # pick up matching resource for lowest-energy node
            current = env.scenario_state.data["resource_types"].get(pos)
            if current == target_type:
                actions[aid] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                continue
            candidates = [p for p, t in env.scenario_state.data["resource_types"].items() if t == target_type]
            if candidates:
                dx, dy, _ = shortest_path(env.grid, pos, set(candidates))
                actions[aid] = {"action": move_action_from_delta(dx, dy, env), "message_tokens": []}
                continue
            actions[aid] = {"action": env.ACTION_STAY, "message_tokens": []}

        return actions

    return _policy


def signal_hunt_central_planner(env):
    """
    Centralized planner for signal hunt. Uses true target and synchronizes scan.
    """
    def _policy(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        target = env.scenario_state.data.get("target")
        if target is None:
            return {i: {"action": env.ACTION_STAY, "message_tokens": []} for i in range(env.num_agents)}
        for aid in range(env.num_agents):
            pos = env.agent_positions[aid]
            if pos == target:
                actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
            else:
                dx, dy, _ = shortest_path(env.grid, pos, {target})
                actions[aid] = {"action": move_action_from_delta(dx, dy, env), "message_tokens": []}
        return actions

    return _policy
