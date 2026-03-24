from __future__ import annotations

from typing import Dict

from .pathing import shortest_path, move_action_from_delta


def _adjacent_to_doors(grid):
    doors = []
    size = grid.shape[0]
    for y in range(size):
        for x in range(size):
            if grid[y, x] == 9:
                doors.append((x, y))
    adj = set()
    for (x, y) in doors:
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and grid[ny, nx] == 0:
                adj.add((nx, ny))
    return doors, adj


def _blocked_positions(env, agent_id, allow_positions=None):
    blocked = set(env.agent_positions)
    blocked.discard(env.agent_positions[agent_id])
    if allow_positions:
        blocked.difference_update(allow_positions)
    return blocked


def _move_or_open(env, agent_id, pos, target):
    blocked_positions = _blocked_positions(env, agent_id, {target})
    dx, dy, tgt = shortest_path(env.grid, pos, {target}, blocked_positions=blocked_positions)
    if tgt is not None:
        return move_action_from_delta(dx, dy, env)
    doors, adj = _adjacent_to_doors(env.grid)
    if adj:
        dx, dy, _ = shortest_path(env.grid, pos, adj, blocked=(1, 9), blocked_positions=blocked_positions)
        if pos in adj:
            return env.ACTION_INTERACT
        return move_action_from_delta(dx, dy, env)
    return env.ACTION_STAY


def _move_to_any_or_open(env, agent_id, pos, targets):
    blocked_positions = _blocked_positions(env, agent_id, set(targets))
    dx, dy, tgt = shortest_path(env.grid, pos, set(targets), blocked_positions=blocked_positions)
    if tgt is not None:
        return move_action_from_delta(dx, dy, env)
    # try to open doors
    doors, adj = _adjacent_to_doors(env.grid)
    if adj:
        dx, dy, _ = shortest_path(env.grid, pos, adj, blocked=(1, 9), blocked_positions=blocked_positions)
        if pos in adj:
            return env.ACTION_INTERACT
        return move_action_from_delta(dx, dy, env)
    return env.ACTION_STAY


def pipeline_oracle(env):
    """
    Greedy oracle: uses full state, minimal sync coordination.
    """
    def _policy(obs, info, state):
        actions = {}
        stages = env.scenario_state.data.get("stages", [])
        open_stages = [s for s in stages if not s["done"]]
        # choose a stage whose deps are satisfied
        available = []
        for s in open_stages:
            deps_done = all(stages[d]["done"] for d in s.get("deps", []))
            if deps_done:
                available.append(s)
        open_stages = available if available else open_stages
        if not open_stages:
            return {i: {"action": env.ACTION_STAY, "message_tokens": []} for i in range(env.num_agents)}
        stage = open_stages[0]
        station = stage["station"]
        required = stage["required"]
        delivered_list = stage.get("delivered", [])
        needs = []
        for r in required:
            if delivered_list.count(r) < required.count(r):
                needs.append(r)

        for agent_id in range(env.num_agents):
            pos = env.agent_positions[agent_id]
            inv = env.inventories[agent_id]
            if stage["sync"] and not needs:
                others_at_station = any(env.agent_positions[i] == station for i in range(env.num_agents) if i != agent_id)
                if pos == station and others_at_station:
                    actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                else:
                    actions[agent_id] = {"action": _move_or_open(env, agent_id, pos, station), "message_tokens": []}
                continue
            if inv in required:
                if pos == station:
                    actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                else:
                    actions[agent_id] = {"action": _move_or_open(env, agent_id, pos, station), "message_tokens": []}
                continue
            if inv == 0:
                current = env.scenario_state.data["resource_types"].get(pos)
                if current is not None and (current in needs or current in required):
                    actions[agent_id] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                    continue
                candidates = [p for p, t in env.scenario_state.data["resource_types"].items() if t in needs or t in required]
                if candidates:
                    actions[agent_id] = {"action": _move_to_any_or_open(env, agent_id, pos, candidates), "message_tokens": []}
                    continue
            actions[agent_id] = {"action": env.ACTION_STAY, "message_tokens": []}
        return actions

    return _policy


def pipeline_oracle_strong(env):
    """
    Strong oracle: explicit assignment of agents to required resources + sync staging.
    """
    def _policy(obs, info, state):
        actions = {}
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
        stage = open_stages[0]
        station = stage["station"]
        required = stage["required"]
        delivered_list = stage.get("delivered", [])
        needs = []
        for r in required:
            if delivered_list.count(r) < required.count(r):
                needs.append(r)

        # assign each agent a target requirement (if any)
        assignments = {}
        for i, aid in enumerate(range(env.num_agents)):
            assignments[aid] = needs[i % len(needs)] if needs else None

        for agent_id in range(env.num_agents):
            pos = env.agent_positions[agent_id]
            inv = env.inventories[agent_id]
            assigned = assignments.get(agent_id)

            if stage["sync"] and not needs:
                # all delivered: converge and sync (no gating on others)
                if pos == station:
                    actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                else:
                    actions[agent_id] = {"action": _move_or_open(env, agent_id, pos, station), "message_tokens": []}
                continue

            # holding assigned resource -> deliver
            if assigned is not None and inv == assigned:
                if pos == station:
                    actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                else:
                    actions[agent_id] = {"action": _move_or_open(env, agent_id, pos, station), "message_tokens": []}
                continue

            # holding any required resource -> deliver
            if inv in required:
                if pos == station:
                    actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                else:
                    actions[agent_id] = {"action": _move_or_open(env, agent_id, pos, station), "message_tokens": []}
                continue

            # acquire assigned resource
            if inv == 0 and assigned is not None:
                current = env.scenario_state.data["resource_types"].get(pos)
                if current == assigned:
                    actions[agent_id] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                    continue
                candidates = [p for p, t in env.scenario_state.data["resource_types"].items() if t == assigned]
                if candidates:
                    actions[agent_id] = {"action": _move_to_any_or_open(env, agent_id, pos, candidates), "message_tokens": []}
                    continue

            # fallback: if assigned type missing, grab any needed resource
            if inv == 0 and needs:
                candidates = [p for p, t in env.scenario_state.data["resource_types"].items() if t in needs]
                if candidates:
                    actions[agent_id] = {"action": _move_to_any_or_open(env, agent_id, pos, candidates), "message_tokens": []}
                    continue

            # fallback: roam toward station
            actions[agent_id] = {"action": _move_or_open(env, agent_id, pos, station), "message_tokens": []}
        return actions

    return _policy


def energy_oracle(env):
    """
    Oracle: always serve lowest-energy node with matching resource, sync if needed.
    """
    def _policy(obs, info, state):
        actions = {}
        node_energy = env.scenario_state.data.get("node_energy", {})
        node_types = env.scenario_state.data.get("node_types", {})
        if not node_energy:
            return {i: {"action": env.ACTION_STAY, "message_tokens": []} for i in range(env.num_agents)}
        target_node = sorted(node_energy.items(), key=lambda kv: kv[1])[0][0]
        target_type = node_types.get(target_node, 0)

        for agent_id in range(env.num_agents):
            pos = env.agent_positions[agent_id]
            inv = env.inventories[agent_id]
            if inv != 0:
                # choose lowest-energy node matching resource type
                matching = [n for n, t in node_types.items() if t == inv]
                if matching:
                    target_node = sorted(matching, key=lambda n: node_energy.get(n, 0))[0]
                    target_type = inv
            if pos == target_node and inv == target_type:
                actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue
            if inv == 0:
                current = env.scenario_state.data["resource_types"].get(pos)
                if current is not None and current in node_types.values():
                    actions[agent_id] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                    continue
                candidates = [p for p, t in env.scenario_state.data["resource_types"].items() if t == target_type]
                if candidates:
                    dx, dy, _ = shortest_path(env.grid, pos, set(candidates))
                    actions[agent_id] = {"action": _move_to_any_or_open(env, agent_id, pos, candidates), "message_tokens": []}
                    continue
            actions[agent_id] = {"action": _move_or_open(env, agent_id, pos, target_node), "message_tokens": []}
        return actions

    return _policy


def energy_oracle_strong(env):
    """
    Strong oracle: assigns agents to different lowest-energy nodes if possible.
    """
    def _policy(obs, info, state):
        actions = {}
        node_energy = env.scenario_state.data.get("node_energy", {})
        node_types = env.scenario_state.data.get("node_types", {})
        nodes_sorted = sorted(node_energy.items(), key=lambda kv: kv[1])
        if not nodes_sorted:
            return {i: {"action": env.ACTION_STAY, "message_tokens": []} for i in range(env.num_agents)}

        critical_node, critical_energy = nodes_sorted[0]
        sync_threshold = env.scenario_state.data.get("sync_threshold", 3)
        use_sync_focus = critical_energy <= sync_threshold

        for agent_id in range(env.num_agents):
            # If holding a resource, prioritize lowest-energy node (or critical sync node).
            if use_sync_focus:
                target_node = critical_node
            else:
                target_node = nodes_sorted[min(agent_id, len(nodes_sorted)-1)][0]
            target_type = node_types.get(target_node, 0)
            pos = env.agent_positions[agent_id]
            inv = env.inventories[agent_id]
            if inv != 0:
                matching = [n for n, t in node_types.items() if t == inv]
                if matching:
                    target_node = sorted(matching, key=lambda n: node_energy.get(n, 0))[0]
                    target_type = inv
            if pos == target_node and inv == target_type:
                actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue
            if inv == 0:
                current = env.scenario_state.data["resource_types"].get(pos)
                if current is not None and current in node_types.values():
                    actions[agent_id] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                    continue
                # grab nearest available resource (any node-matching type)
                candidates = list(env.scenario_state.data["resource_types"].keys())
                if candidates:
                    actions[agent_id] = {"action": _move_to_any_or_open(env, agent_id, pos, candidates), "message_tokens": []}
                    continue
            actions[agent_id] = {"action": _move_or_open(env, agent_id, pos, target_node), "message_tokens": []}
        return actions

    return _policy


def energy_oracle_planner(env):
    """
    Planning oracle for energy_grid: optimal agent-to-node assignment based on
    distance and urgency. Handles many-nodes-few-agents via priority scheduling.

    Strategy:
      1. Rank nodes by urgency (lowest energy first)
      2. Assign each agent to the best (closest + most urgent) node it can serve
      3. If carrying a resource, deliver to the matching lowest-energy node
      4. If not carrying, pick up the resource type needed by the most urgent reachable node
      5. Avoid duplicate assignments — spread agents across different nodes
    """
    from collections import deque

    def _bfs_dist(grid, start, goal, blocked_positions=None):
        """BFS distance from start to goal."""
        size = grid.shape[0]
        q = deque([start])
        dist = {start: 0}
        bp = blocked_positions or set()
        while q:
            x, y = q.popleft()
            if (x, y) == goal:
                return dist[(x, y)]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in dist:
                    if (nx, ny) in bp:
                        continue
                    if grid[ny, nx] in (1, 9):
                        continue
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
        return 999  # unreachable

    def _policy(obs, info, state):
        actions = {}
        node_energy = env.scenario_state.data.get("node_energy", {})
        node_types = env.scenario_state.data.get("node_types", {})
        resource_types = env.scenario_state.data.get("resource_types", {})
        sync_threshold = env.scenario_state.data.get("sync_threshold", 3)

        if not node_energy:
            return {i: {"action": env.ACTION_STAY, "message_tokens": []} for i in range(env.num_agents)}

        # Sort nodes by urgency (lowest energy first)
        nodes_by_urgency = sorted(node_energy.items(), key=lambda kv: kv[1])

        # Track which nodes are claimed this step
        claimed_nodes = set()

        # First pass: agents carrying resources → deliver to best matching node
        for agent_id in range(env.num_agents):
            pos = env.agent_positions[agent_id]
            inv = env.inventories[agent_id]
            if inv == 0:
                continue

            # Find lowest-energy node matching this resource type
            matching = [
                (npos, e) for npos, e in nodes_by_urgency
                if node_types.get(npos) == inv and npos not in claimed_nodes
            ]
            if not matching:
                # No unclaimed matching node — try any matching
                matching = [
                    (npos, e) for npos, e in nodes_by_urgency
                    if node_types.get(npos) == inv
                ]
            if not matching:
                actions[agent_id] = {"action": env.ACTION_STAY, "message_tokens": []}
                continue

            target_node = matching[0][0]
            claimed_nodes.add(target_node)

            if pos == target_node:
                # Check sync requirement
                energy = node_energy.get(target_node, 999)
                if energy <= sync_threshold:
                    # Need 2 agents — check if another is here
                    others_here = [
                        a for a in range(env.num_agents)
                        if a != agent_id and env.agent_positions[a] == target_node
                        and env.inventories[a] == inv
                    ]
                    if others_here:
                        actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                    else:
                        actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                else:
                    actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
            else:
                actions[agent_id] = {"action": _move_or_open(env, agent_id, pos, target_node), "message_tokens": []}

        # Second pass: agents without resources → pick up what's needed most
        for agent_id in range(env.num_agents):
            if agent_id in actions:
                continue
            pos = env.agent_positions[agent_id]

            # Check if standing on a useful resource
            current_res = resource_types.get(pos)
            if current_res is not None and current_res in node_types.values():
                actions[agent_id] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                continue

            # Find the most urgent unclaimed node, then find a resource of its type
            best_score = 999999
            best_resource = None
            for npos, energy in nodes_by_urgency:
                if npos in claimed_nodes:
                    continue
                ntype = node_types.get(npos)
                # Find nearest resource of this type
                for rpos, rtype in resource_types.items():
                    if rtype == ntype:
                        dist_to_res = _bfs_dist(env.grid, pos, rpos, _blocked_positions(env, agent_id))
                        dist_res_to_node = _bfs_dist(env.grid, rpos, npos)
                        # Score: urgency (lower energy = higher priority) + distance
                        score = energy * 0.5 + (dist_to_res + dist_res_to_node)
                        if score < best_score:
                            best_score = score
                            best_resource = rpos
                            best_node = npos

            if best_resource is not None:
                claimed_nodes.add(best_node)
                actions[agent_id] = {"action": _move_to_any_or_open(env, agent_id, pos, [best_resource]), "message_tokens": []}
            else:
                # No useful resource found — explore toward unclaimed nodes
                unclaimed = [npos for npos, _ in nodes_by_urgency if npos not in claimed_nodes]
                if unclaimed:
                    actions[agent_id] = {"action": _move_to_any_or_open(env, agent_id, pos, unclaimed), "message_tokens": []}
                else:
                    actions[agent_id] = {"action": env.ACTION_STAY, "message_tokens": []}

        return actions

    return _policy


def signal_hunt_oracle(env):
    """
    Oracle: goes directly to true target and synchronizes scan.
    """
    def _policy(obs, info, state):
        actions = {}
        target = env.scenario_state.data.get("target")
        for agent_id in range(env.num_agents):
            pos = env.agent_positions[agent_id]
            if pos == target:
                actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue
            blocked_positions = _blocked_positions(env, agent_id, {target})
            dx, dy, _ = shortest_path(env.grid, pos, {target}, blocked_positions=blocked_positions)
            actions[agent_id] = {"action": move_action_from_delta(dx, dy, env), "message_tokens": []}
        return actions

    return _policy


def signal_hunt_oracle_strong(env):
    """
    Strong oracle: first converge near target, then synchronize scan together.
    """
    def _policy(obs, info, state):
        actions = {}
        target = env.scenario_state.data.get("target")
        # pick a rendezvous adjacent to target to ensure simultaneous scan
        rendezvous = target
        for agent_id in range(env.num_agents):
            pos = env.agent_positions[agent_id]
            if pos == target:
                actions[agent_id] = {"action": env.ACTION_INTERACT, "message_tokens": []}
            else:
                blocked_positions = _blocked_positions(env, agent_id, {rendezvous})
                dx, dy, _ = shortest_path(env.grid, pos, {rendezvous}, blocked_positions=blocked_positions)
                actions[agent_id] = {"action": move_action_from_delta(dx, dy, env), "message_tokens": []}
        return actions

    return _policy
