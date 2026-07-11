from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .maps import (
    TILE_BEACON,
    TILE_CLUE,
    TILE_NODE,
    TILE_RESOURCE,
    TILE_STATION,
    TILE_TARGET,
    TILE_WATER,
    TILE_EMPTY,
    MapSpec,
    generate_base_map,
    generate_room_map,
    largest_component_positions,
    place_random_positions,
    place_tiles,
    place_tiles_in_positions,
    place_tiles_with_constraints,
)
from .utils import manhattan


@dataclass
class ScenarioState:
    data: dict


class ScenarioBase:
    name = "base"

    def build_map(self, size: int, rng: np.random.Generator, config) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def reset(self, env) -> ScenarioState:
        raise NotImplementedError

    def step(self, env, actions: dict[int, dict]) -> tuple[dict[int, float], bool, dict]:
        raise NotImplementedError


class PipelineAssembly(ScenarioBase):
    name = "pipeline_assembly"

    def build_map(self, size: int, rng: np.random.Generator, config) -> tuple[np.ndarray, dict]:
        station_count = max(3, min(5, size // 6))
        resource_count = station_count * 3
        min_starts = max(6, int(getattr(config, "num_agents", 3)))
        min_component = max(station_count + resource_count + min_starts + 4, int(size * size * 0.25))

        for _ in range(20):
            if config.use_rooms:
                grid, rooms = generate_room_map(MapSpec(size=size), rng)
                if not config.use_doors:
                    grid[grid == 9] = 0
            else:
                grid = generate_base_map(MapSpec(size=size), rng)
                rooms = []

            component = largest_component_positions(grid)
            if not component:
                component = [(x, y) for y in range(size) for x in range(size) if grid[y, x] == 0]
            if len(component) < min_component:
                continue

            stations = place_tiles_in_positions(grid, rng, TILE_STATION, count=station_count, positions=component)
            resources = place_tiles_in_positions(grid, rng, TILE_RESOURCE, count=resource_count, positions=component)
            agent_starts = place_tiles_in_positions(grid, rng, TILE_EMPTY, count=min_starts, positions=component)
            if len(stations) == station_count and len(resources) == resource_count and len(agent_starts) >= min_starts:
                break
        else:
            if config.use_rooms:
                grid, rooms = generate_room_map(MapSpec(size=size), rng)
                if not config.use_doors:
                    grid[grid == 9] = 0
            else:
                grid = generate_base_map(MapSpec(size=size), rng)
                rooms = []
            component = largest_component_positions(grid)
            if not component:
                component = [(x, y) for y in range(size) for x in range(size) if grid[y, x] == 0]
            stations = place_tiles_in_positions(grid, rng, TILE_STATION, count=station_count, positions=component)
            resources = place_tiles_in_positions(grid, rng, TILE_RESOURCE, count=resource_count, positions=component)
            agent_starts = place_tiles_in_positions(grid, rng, TILE_EMPTY, count=min_starts, positions=component)
        meta = {
            "stations": stations,
            "resources": resources,
            "agent_starts": agent_starts,
        }
        return grid, meta

    def reset(self, env) -> ScenarioState:
        rng = env.rng
        stations = env.meta["stations"]
        stage_count = max(4, min(7, len(stations) + 2))
        resource_type_count = 4

        # ensure total required resources do not exceed available resources
        resource_types = {pos: (idx % resource_type_count) + 1 for idx, pos in enumerate(env.meta["resources"])}
        resource_pool = list(resource_types.values())
        if resource_pool:
            stage_count = min(stage_count, max(1, len(resource_pool) // 2))

        stages = []
        for stage_id in range(stage_count):
            station = stations[stage_id % len(stations)]
            # reserve at least 1 resource per remaining stage
            remaining_stages = stage_count - stage_id
            max_req = max(1, len(resource_pool) - (remaining_stages - 1))
            req_count = int(rng.integers(1, min(2, max_req) + 1))
            required = []
            for _ in range(req_count):
                if resource_pool:
                    idx = int(rng.integers(0, len(resource_pool)))
                    required.append(resource_pool.pop(idx))
                else:
                    required.append(int(rng.integers(1, resource_type_count + 1)))
            sync_required = bool(rng.integers(0, 2))
            stages.append(
                {
                    "stage": stage_id,
                    "station": station,
                    "required": required,
                    "delivered": [],
                    "deps": [],
                    "sync": sync_required,
                    "done": False,
                }
            )

        # DAG dependencies (only backward edges)
        for stage in stages[1:]:
            if rng.random() < 0.7:
                dep = int(rng.integers(0, stage["stage"]))
                stage["deps"].append(dep)

        # partial blueprint hints per agent
        hints = {}
        for agent_id in range(env.num_agents):
            known = [s for idx, s in enumerate(stages) if idx % env.num_agents == agent_id]
            hints[agent_id] = known
        # full plan for local policies
        full_plan = stages

        data = {
            "stages": stages,
            "hints": hints,
            "full_plan": full_plan,
            "resource_types": resource_types,
            "resource_type_count": resource_type_count,
        }
        return ScenarioState(data=data)

    def step(self, env, actions: dict[int, dict]) -> tuple[dict[int, float], bool, dict]:
        rewards = {i: 0.0 for i in range(env.num_agents)}
        events = {i: [] for i in range(env.num_agents)}

        # check for completion
        if all(s["done"] for s in env.scenario_state.data["stages"]):
            for i in rewards:
                rewards[i] += env.reward_complete
            return rewards, True, {"events": events}

        # collect interactions per station
        station_interactors = {}
        for agent_id, action in actions.items():
            if action.get("action") == env.ACTION_INTERACT:
                pos = env.agent_positions[agent_id]
                station_interactors.setdefault(pos, []).append(agent_id)

        # process deliveries
        for agent_id, action in actions.items():
            if action.get("action") != env.ACTION_INTERACT:
                continue
            pos = env.agent_positions[agent_id]
            for stage in env.scenario_state.data["stages"]:
                if stage["done"]:
                    continue
                if pos != stage["station"]:
                    continue
                if env.inventories[agent_id] == 0:
                    continue
                if env.inventories[agent_id] in stage["required"]:
                    # allow duplicate deliveries if required multiple times
                    req_count = stage["required"].count(env.inventories[agent_id])
                    del_count = stage["delivered"].count(env.inventories[agent_id])
                    if del_count < req_count:
                        stage["delivered"].append(env.inventories[agent_id])
                        env.inventories[agent_id] = 0
                        rewards[agent_id] += env.reward_stage
                        events[agent_id].append({"event": "delivered", "stage": stage["stage"]})

        # finalize stages
        for stage in env.scenario_state.data["stages"]:
            if stage["done"]:
                continue
            deps_done = all(env.scenario_state.data["stages"][d]["done"] for d in stage["deps"])
            if not deps_done:
                continue
            # check multiset completion
            if any(stage["delivered"].count(req) < stage["required"].count(req) for req in stage["required"]):
                continue
            # require synchronized interaction if sync stage
            if stage["sync"]:
                interactors = station_interactors.get(stage["station"], [])
                if len(interactors) < 2:
                    continue
                for agent_id in interactors:
                    rewards[agent_id] += env.reward_stage * 0.5
                    events[agent_id].append({"event": "sync_complete", "stage": stage["stage"]})
            stage["done"] = True

        if env.config.pipeline_shaping:
            shaping = env.scenario_state.data.setdefault("shaping", {"last_target": {}, "last_dist": {}})

            def pick_target(agent_id: int):
                carry = env.inventories[agent_id]
                stages = env.scenario_state.data["stages"]
                if carry != 0:
                    candidates = []
                    for stage in stages:
                        if stage["done"]:
                            continue
                        if carry not in stage["required"]:
                            continue
                        if stage["delivered"].count(carry) >= stage["required"].count(carry):
                            continue
                        if not all(stages[d]["done"] for d in stage["deps"]):
                            continue
                        candidates.append(stage["station"])
                    if not candidates:
                        return None
                    pos = env.agent_positions[agent_id]
                    best = min(candidates, key=lambda p: manhattan(pos, p))
                    return ("station", best, carry)
                needed_types = []
                for stage in stages:
                    if stage["done"]:
                        continue
                    if not all(stages[d]["done"] for d in stage["deps"]):
                        continue
                    for req in stage["required"]:
                        if stage["delivered"].count(req) < stage["required"].count(req):
                            needed_types.append(req)
                if not needed_types:
                    return None
                pos = env.agent_positions[agent_id]
                candidates = [
                    p for p, t in env.scenario_state.data.get("resource_types", {}).items() if t in needed_types
                ]
                if not candidates:
                    return None
                best = min(candidates, key=lambda p: manhattan(pos, p))
                return ("resource", best, 0)

            for agent_id in range(env.num_agents):
                target = pick_target(agent_id)
                if target is None:
                    shaping["last_target"].pop(agent_id, None)
                    shaping["last_dist"].pop(agent_id, None)
                    continue
                pos = env.agent_positions[agent_id]
                dist = manhattan(pos, target[1])
                last_target = shaping["last_target"].get(agent_id)
                if last_target != target:
                    shaping["last_target"][agent_id] = target
                    shaping["last_dist"][agent_id] = dist
                    continue
                last_dist = shaping["last_dist"].get(agent_id, dist)
                rewards[agent_id] += (last_dist - dist) * env.config.pipeline_shaping_scale
                shaping["last_dist"][agent_id] = dist

        done = all(s["done"] for s in env.scenario_state.data["stages"])
        if done:
            for i in rewards:
                rewards[i] += env.reward_complete
        return rewards, done, {"events": events}


class EnergyGrid(ScenarioBase):
    name = "energy_grid"

    def build_map(self, size: int, rng: np.random.Generator, config) -> tuple[np.ndarray, dict]:
        n_agents = getattr(config, "num_agents", 3)
        # Node count: ~1.5-2x agents to force triage without being impossible
        node_count = max(3, int(n_agents * 1.5) + size // 16)
        resource_count = max(node_count * 2, node_count + 1)
        min_starts = max(6, int(getattr(config, "num_agents", 3)))
        min_component = max(node_count + resource_count + min_starts + 4, int(size * size * 0.30))

        # Retry map generation until we avoid cramped/isolated pockets.
        for _ in range(20):
            if config.use_rooms:
                grid, rooms = generate_room_map(MapSpec(size=size), rng)
                if not config.use_doors:
                    grid[grid == 9] = 0
            else:
                grid = generate_base_map(MapSpec(size=size), rng)

            component = largest_component_positions(grid)
            if not component:
                component = [(x, y) for y in range(size) for x in range(size) if grid[y, x] == 0]
            if len(component) < min_component:
                continue

            nodes = place_tiles_in_positions(grid, rng, TILE_NODE, count=node_count, positions=component)
            resources = place_tiles_in_positions(grid, rng, TILE_RESOURCE, count=resource_count, positions=component)
            agent_starts = place_tiles_in_positions(grid, rng, TILE_EMPTY, count=min_starts, positions=component)
            if len(nodes) < node_count or len(resources) < resource_count or len(agent_starts) < min_starts:
                continue
            return grid, {"nodes": nodes, "resources": resources, "agent_starts": agent_starts}

        # Fallback: keep previous behavior if retries fail.
        if config.use_rooms:
            grid, rooms = generate_room_map(MapSpec(size=size), rng)
            if not config.use_doors:
                grid[grid == 9] = 0
        else:
            grid = generate_base_map(MapSpec(size=size), rng)
        component = largest_component_positions(grid)
        if not component:
            component = [(x, y) for y in range(size) for x in range(size) if grid[y, x] == 0]
        nodes = place_tiles_in_positions(grid, rng, TILE_NODE, count=node_count, positions=component)
        resources = place_tiles_in_positions(grid, rng, TILE_RESOURCE, count=resource_count, positions=component)
        agent_starts = place_tiles_in_positions(grid, rng, TILE_EMPTY, count=min_starts, positions=component)
        return grid, {"nodes": nodes, "resources": resources, "agent_starts": agent_starts}

    def reset(self, env) -> ScenarioState:
        import math
        rng = env.rng
        preset = getattr(env.config, "energy_preset", "hard")
        # Energy scales with map size while staying tight enough to force
        # triage; more nodes than agents means agents must communicate.
        sqrt_s = math.sqrt(env.map_size)
        n_nodes = len(env.meta["nodes"])
        if preset == "easy":
            scaled_energy = max(15, int(sqrt_s * 10))
            grace = max(4, int(sqrt_s * 3))
            # The small core task should still require communication: every
            # recharge is sync-gated, so assigned monitors must bring partners
            # to the right typed node instead of solo-topping-up visible nodes.
            sync_threshold = scaled_energy
            success_recharges = n_nodes
        else:
            scaled_energy = max(12, int(sqrt_s * 8), int(env.map_size * 2.25))
            grace = max(2, int(sqrt_s * 2))
            sync_threshold = max(3, int(scaled_energy * 0.3))
            success_recharges = min(n_nodes, max(env.num_agents + 1, int(math.ceil(n_nodes * 0.7))))
        node_energy = {pos: scaled_energy for pos in env.meta["nodes"]}
        node_types = {pos: int(rng.integers(1, 3)) for pos in env.meta["nodes"]}
        # distribute resources to ensure coverage of node types
        resource_positions = list(env.meta["resources"])
        node_type_list = list(node_types.values())
        if node_type_list and resource_positions:
            pool = []
            while len(pool) < len(resource_positions):
                pool.extend(node_type_list)
            pool = pool[: len(resource_positions)]
            rng.shuffle(pool)
            resource_types = {pos: pool[i] for i, pos in enumerate(resource_positions)}
        else:
            resource_types = {pos: int(rng.integers(1, 3)) for pos in resource_positions}
        # refill: small — buys a few steps, not a full reset
        env.energy_refill = max(3, int(sqrt_s))
        # Private monitoring: assign each node to a specific agent.
        # Only the assigned agent can see that node's energy level.
        # Other agents see energy=0 (unknown) for unassigned nodes.
        node_list = list(node_energy.keys())
        node_assignments = {}  # node_pos -> agent_id
        agent_nodes = {aid: [] for aid in range(env.num_agents)}  # agent_id -> [node_pos]
        for i, node_pos in enumerate(node_list):
            assigned_agent = i % env.num_agents
            node_assignments[node_pos] = assigned_agent
            agent_nodes[assigned_agent].append(node_pos)

        data = {
            "node_energy": node_energy,
            "node_types": node_types,
            "resource_types": resource_types,
            "spawn_rate": 0.3 if preset == "easy" else 0.15,
            "sync_threshold": sync_threshold,
            "recharge_count": 0,
            "success_recharges": success_recharges,
            "grace_steps": grace,
            "drain_period": 1,  # always drain every step — no free passes
            "node_assignments": node_assignments,
            "agent_nodes": agent_nodes,
        }
        return ScenarioState(data=data)

    def step(self, env, actions: dict[int, dict]) -> tuple[dict[int, float], bool, dict]:
        rewards = {i: 0.0 for i in range(env.num_agents)}
        events = {i: [] for i in range(env.num_agents)}

        # drain nodes
        grace_steps = env.scenario_state.data.get("grace_steps", 0)
        drain_period = env.scenario_state.data.get("drain_period", 1)
        if env.steps > grace_steps and env.steps % drain_period == 0:
            for pos in env.scenario_state.data["node_energy"]:
                env.scenario_state.data["node_energy"][pos] -= 1

        # emit node_critical events when energy drops below sync threshold
        sync_thresh = env.scenario_state.data.get("sync_threshold", 3)
        node_assignments = env.scenario_state.data.get("node_assignments", {})
        private_monitor = bool(getattr(env.config, "energy_private_monitor", False))
        for pos, energy in env.scenario_state.data["node_energy"].items():
            if energy == sync_thresh or energy == sync_thresh // 2:
                if private_monitor:
                    assigned_agent = node_assignments.get(pos)
                    recipients = [assigned_agent] if assigned_agent is not None else []
                else:
                    recipients = range(env.num_agents)
                for agent_id in recipients:
                    events[agent_id].append({
                        "event": "node_critical",
                        "node": pos,
                        "energy": energy,
                    })

        # collect interactors per node
        node_interactors = {}
        for agent_id, action in actions.items():
            if action.get("action") == env.ACTION_INTERACT:
                pos = env.agent_positions[agent_id]
                if pos in env.scenario_state.data["node_energy"]:
                    node_interactors.setdefault(pos, []).append(agent_id)

        # interactions: deliver fuel
        for node_pos, interactors in node_interactors.items():
            node_type = env.scenario_state.data["node_types"][node_pos]
            energy = env.scenario_state.data["node_energy"][node_pos]
            require_sync = energy <= env.scenario_state.data["sync_threshold"]

            valid = [a for a in interactors if env.inventories[a] == node_type]
            if require_sync and len(valid) < 2:
                continue
            for agent_id in valid[:2 if require_sync else 1]:
                env.inventories[agent_id] = 0
                env.scenario_state.data["node_energy"][node_pos] += env.energy_refill
                rewards[agent_id] += env.reward_stage
                env.scenario_state.data["recharge_count"] += 1
                events[agent_id].append({"event": "recharged", "node": node_pos})

        # spawn new resources stochastically
        if env.rng.random() < env.scenario_state.data["spawn_rate"]:
            empty_positions = []
            for y in range(env.map_size):
                for x in range(env.map_size):
                    if env.grid[y, x] == 0:
                        empty_positions.append((x, y))
            if empty_positions:
                pos = empty_positions[int(env.rng.integers(0, len(empty_positions)))]
                env.grid[pos[1], pos[0]] = TILE_RESOURCE
                env.scenario_state.data["resource_types"][pos] = int(env.rng.integers(1, 3))

        # success if enough recharges achieved
        recharge_count = env.scenario_state.data.get("recharge_count", 0)
        success_recharges = env.scenario_state.data.get("success_recharges", 0)
        success = recharge_count >= success_recharges
        # fail if any node depleted (unless already successful)
        depleted = False if success else any(v <= 0 for v in env.scenario_state.data["node_energy"].values())
        done = success or depleted
        if done:
            if depleted and not success:
                for i in rewards:
                    rewards[i] -= env.reward_fail
        info = {
            "depleted": depleted,
            "success": success,
            "events": events,
            "recharge_count": recharge_count,
            "success_recharges": success_recharges,
        }
        if env.config.energy_shaping:
            shaping = env.scenario_state.data.setdefault("shaping", {"last_target": {}, "last_dist": {}})

            def pick_target(agent_id: int):
                carry = env.inventories[agent_id]
                if carry != 0:
                    candidates = [
                        pos
                        for pos, ntype in env.scenario_state.data.get("node_types", {}).items()
                        if ntype == carry
                    ]
                    if not candidates:
                        return None
                    pos = env.agent_positions[agent_id]
                    best = min(candidates, key=lambda p: manhattan(pos, p))
                    return ("node", best, carry)
                candidates = list(env.scenario_state.data.get("resource_types", {}).keys())
                if not candidates:
                    return None
                pos = env.agent_positions[agent_id]
                best = min(candidates, key=lambda p: manhattan(pos, p))
                return ("resource", best, 0)

            for agent_id in range(env.num_agents):
                target = pick_target(agent_id)
                if target is None:
                    shaping["last_target"].pop(agent_id, None)
                    shaping["last_dist"].pop(agent_id, None)
                    continue
                pos = env.agent_positions[agent_id]
                dist = manhattan(pos, target[1])
                last_target = shaping["last_target"].get(agent_id)
                if last_target != target:
                    shaping["last_target"][agent_id] = target
                    shaping["last_dist"][agent_id] = dist
                    continue
                last_dist = shaping["last_dist"].get(agent_id, dist)
                rewards[agent_id] += (last_dist - dist) * env.config.energy_shaping_scale
                shaping["last_dist"][agent_id] = dist
        return rewards, done, info


class SignalHunt(ScenarioBase):
    name = "signal_hunt"

    def build_map(self, size: int, rng: np.random.Generator, config) -> tuple[np.ndarray, dict]:
        rooms = []
        if config.use_rooms:
            grid, rooms = generate_room_map(MapSpec(size=size), rng)
            if not config.use_doors:
                grid[grid == 9] = 0  # convert doors to empty if disabled
        else:
            grid = generate_base_map(MapSpec(size=size), rng)
        clue_count = max(4, size // 6)
        decoy_count = config.signal_decoy_count if config.signal_decoy_count is not None else max(2, size // 8)
        # place true target first
        targets = place_tiles(grid, rng, TILE_TARGET, count=1 + decoy_count)
        true_target = targets[0] if targets else None
        avoid = [true_target] if true_target else []
        # clues farther from target
        clues = place_tiles_with_constraints(grid, rng, TILE_CLUE, count=clue_count, min_dist=4, avoid=avoid)
        # water moderately near target, but not adjacent
        water = place_tiles_with_constraints(grid, rng, TILE_WATER, count=max(2, size // 6), min_dist=2, avoid=[])
        beacons = place_tiles_with_constraints(grid, rng, TILE_BEACON, count=1, min_dist=3, avoid=avoid)

        # Ensure at least one water tile near target when possible
        if targets:
            tx, ty = targets[0]
            candidate = (tx + 1, ty)
            if 0 <= candidate[0] < size and 0 <= candidate[1] < size and grid[candidate[1], candidate[0]] == 0:
                grid[candidate[1], candidate[0]] = TILE_WATER
                water.append(candidate)

        # Ensure at least one beacon with a fixed relation when possible
        if targets:
            tx, ty = targets[0]
            beacon_pos = (tx - 2, ty)
            if 0 <= beacon_pos[0] < size and 0 <= beacon_pos[1] < size and grid[beacon_pos[1], beacon_pos[0]] == 0:
                grid[beacon_pos[1], beacon_pos[0]] = TILE_BEACON
                beacons.append(beacon_pos)
        agent_starts = place_random_positions(grid, rng, count=6)
        meta = {
            "clues": clues,
            "targets": targets,
            "water": water,
            "beacons": beacons,
            "rooms": rooms,
            "agent_starts": agent_starts,
        }
        return grid, meta

    def reset(self, env) -> ScenarioState:
        rng = env.rng
        target = env.meta["targets"][0]
        decoys = env.meta["targets"][1:]
        size = env.map_size
        beacons = env.meta["beacons"]
        water = env.meta["water"]

        clues = []
        constraints = []

        # attribute + object clue
        close_water = [p for p in water if manhattan(p, target) <= 2]
        if close_water:
            nearest_water = min(close_water, key=lambda p: manhattan(p, target))
            clues.append("target near water")
            constraints.append({"type": "near", "object": "water", "pos": nearest_water, "dist": 2})

        # relational clue
        valid_beacons = [b for b in beacons if (b[0] + 2, b[1]) == target]
        if valid_beacons:
            beacon = valid_beacons[0]
            clues.append(f"target two tiles east of beacon at {beacon[0]},{beacon[1]}")
            constraints.append({"type": "offset", "object": "beacon", "pos": beacon, "dx": 2, "dy": 0})

        # riddle-like clue (still deterministic constraint)
        quadrant = (
            "NW" if target[0] < size / 2 and target[1] < size / 2 else
            "NE" if target[0] >= size / 2 and target[1] < size / 2 else
            "SW" if target[0] < size / 2 and target[1] >= size / 2 else "SE"
        )
        parity = (target[0] + target[1]) % 2
        clues.append(f"I rest where x+y is {parity}, in the {quadrant} quadrant.")
        constraints.append({"type": "parity_quadrant", "parity": parity, "quadrant": quadrant, "size": size})

        # extra constraints for decoys
        clues.append(f"target x parity {target[0] % 2}")
        constraints.append({"type": "x_parity", "value": target[0] % 2})
        clues.append(f"target y parity {target[1] % 2}")
        constraints.append({"type": "y_parity", "value": target[1] % 2})

        agent_hints = {}
        agent_hint_specs = {}
        for agent_id in range(env.num_agents):
            hint_idx = agent_id % len(clues)
            agent_hints[agent_id] = clues[hint_idx]
            agent_hint_specs[agent_id] = constraints[hint_idx]

        data = {
            "target": target,
            "clues": clues,
            "clue_specs": constraints,
            "agent_hints": agent_hints,
            "agent_hint_specs": agent_hint_specs,
            "constraints": constraints,
            "decoys": decoys,
            "clue_claimed": set(),
            "agent_clues": {i: [] for i in range(env.num_agents)},
            "agent_clue_specs": {i: [] for i in range(env.num_agents)},
            "scan_log": {},
            "scan_window": env.config.scan_window,
        }
        return ScenarioState(data=data)

    def step(self, env, actions: dict[int, dict]) -> tuple[dict[int, float], bool, dict]:
        rewards = {i: 0.0 for i in range(env.num_agents)}
        events = {i: [] for i in range(env.num_agents)}
        target = env.scenario_state.data["target"]

        # clue collection
        for agent_id, action in actions.items():
            if action.get("action") == env.ACTION_INTERACT:
                pos = env.agent_positions[agent_id]
                if pos in env.meta["clues"] and pos not in env.scenario_state.data["clue_claimed"]:
                    clue_idx = len(env.scenario_state.data["clue_claimed"]) % len(env.scenario_state.data["clues"])
                    clue = env.scenario_state.data["clues"][clue_idx]
                    env.scenario_state.data["agent_clues"][agent_id].append(clue)
                    env.scenario_state.data["agent_clue_specs"][agent_id].append(
                        env.scenario_state.data["clue_specs"][clue_idx]
                    )
                    env.scenario_state.data["clue_claimed"].add(pos)
                    events[agent_id].append({"event": "clue_found"})

        # scans on target
        scanners_this_step = set()
        for agent_id, action in actions.items():
            if action.get("action") == env.ACTION_INTERACT and env.agent_positions[agent_id] == target:
                env.scenario_state.data["scan_log"][agent_id] = env.steps
                scanners_this_step.add(agent_id)
                events[agent_id].append({"event": "target_scan"})
            elif action.get("action") == env.ACTION_INTERACT and env.agent_positions[agent_id] in env.scenario_state.data.get("decoys", []):
                rewards[agent_id] -= env.reward_stage * env.config.decoy_penalty
                events[agent_id].append({"event": "decoy_scan"})

        window = env.scenario_state.data["scan_window"]
        recent = [a for a, t in env.scenario_state.data["scan_log"].items() if env.steps - t <= window]
        if len(recent) >= 2:
            for aid in recent:
                events[aid].append({"event": "joint_target_scan"})
            for i in rewards:
                rewards[i] += env.reward_complete
            return rewards, True, {"events": events}

        if env.config.signal_shaping:
            shaping = env.scenario_state.data.setdefault("shaping", {
                "last_target": {}, "last_dist": {},
                "last_msgs": {},  # track last message step per agent for comm utility
            })

            # --- Proximity shaping (existing): guide agents toward clues then target ---
            def pick_target(agent_id: int):
                has_clue = len(env.scenario_state.data.get("agent_clues", {}).get(agent_id, [])) > 0
                if not has_clue:
                    remaining = [c for c in env.meta.get("clues", []) if c not in env.scenario_state.data["clue_claimed"]]
                    if not remaining:
                        return None
                    pos = env.agent_positions[agent_id]
                    best = min(remaining, key=lambda p: manhattan(pos, p))
                    return ("clue", best, 0)
                return ("target", target, 0)

            for agent_id in range(env.num_agents):
                target_pos = pick_target(agent_id)
                if target_pos is None:
                    shaping["last_target"].pop(agent_id, None)
                    shaping["last_dist"].pop(agent_id, None)
                    continue
                pos = env.agent_positions[agent_id]
                dist = manhattan(pos, target_pos[1])
                last_target = shaping["last_target"].get(agent_id)
                if last_target != target_pos:
                    shaping["last_target"][agent_id] = target_pos
                    shaping["last_dist"][agent_id] = dist
                    continue
                last_dist = shaping["last_dist"].get(agent_id, dist)
                rewards[agent_id] += (last_dist - dist) * env.config.signal_shaping_scale
                shaping["last_dist"][agent_id] = dist

            # --- Part 1a: Scan bonuses (solo + joint near-miss) ---
            # Solo scan: small bonus for interacting on target (teaches "go + interact").
            # Joint scan: large bonus when a partner ALSO scanned within the window.
            # The joint bonus is the key gradient toward synchronized behavior.
            scan_log = env.scenario_state.data["scan_log"]
            window = env.scenario_state.data["scan_window"]
            if scanners_this_step:
                # Check how many agents have scanned recently (including this step)
                recent_scanners = {a for a, t in scan_log.items() if env.steps - t <= window}
                recent_scanners |= scanners_this_step

                if len(recent_scanners) >= 2 and env.config.signal_joint_scan_bonus > 0:
                    # Near-miss: 2+ agents scanned within the window but didn't
                    # get the full success (which would have returned above).
                    # This is the critical gradient toward synchronized scanning.
                    for aid in recent_scanners:
                        rewards[aid] += env.config.signal_joint_scan_bonus
                elif env.config.signal_scan_bonus > 0:
                    # Solo scan: only give the small bonus if no joint bonus was triggered.
                    # This prevents farming the solo bonus when joint is available.
                    for aid in scanners_this_step:
                        rewards[aid] += env.config.signal_scan_bonus

            # --- Part 1b: Co-location bonus ---
            # Only fires when 2+ agents are near the target AND at least one interacted.
            # Pure proximity without action is not rewarded (prevents hovering).
            if env.config.signal_colocation_bonus > 0 and scanners_this_step:
                radius = env.config.signal_colocation_radius
                near_target = [
                    aid for aid in range(env.num_agents)
                    if manhattan(env.agent_positions[aid], target) <= radius
                ]
                if len(near_target) >= 2:
                    for aid in near_target:
                        rewards[aid] += env.config.signal_colocation_bonus

            # --- Part 2: Communication utility bonus ---
            # Reward agent A if: A sent a message in the last few steps, and a teammate
            # subsequently moved closer to the target or interacted on it.
            # This gives gradient signal to the comm channel by connecting messages to outcomes.
            if env.config.signal_comm_utility > 0:
                utility_window = 3  # how many steps back to credit a message
                # Track who sent messages this step
                for agent_id, action in actions.items():
                    msg = action.get("message_tokens") or []
                    if len(msg) > 0:
                        shaping["last_msgs"][agent_id] = env.steps

                # Check if any agent took a "useful" action (moved closer to target or interacted on it)
                for agent_id in range(env.num_agents):
                    pos = env.agent_positions[agent_id]
                    did_useful = False
                    # Useful: interacted on target
                    if actions.get(agent_id, {}).get("action") == env.ACTION_INTERACT and pos == target:
                        did_useful = True
                    # Useful: moved closer to target than last step
                    elif agent_id in shaping.get("last_dist", {}):
                        curr_dist = manhattan(pos, target)
                        prev_dist = shaping["last_dist"].get(agent_id, curr_dist)
                        if curr_dist < prev_dist:
                            did_useful = True

                    if did_useful:
                        # Credit teammates who sent messages recently
                        for sender_id in range(env.num_agents):
                            if sender_id == agent_id:
                                continue
                            last_msg_step = shaping["last_msgs"].get(sender_id, -999)
                            if env.steps - last_msg_step <= utility_window:
                                rewards[sender_id] += env.config.signal_comm_utility

        return rewards, False, {"events": events}


SCENARIOS = {
    PipelineAssembly.name: PipelineAssembly(),
    EnergyGrid.name: EnergyGrid(),
    SignalHunt.name: SignalHunt(),
}
