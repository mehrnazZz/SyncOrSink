from __future__ import annotations

from typing import Dict

from .planner import pipeline_central_planner



def _clip_tokens(tokens: list[int], vocab_size: int, limit: int) -> list[int]:
    clipped = [max(0, min(vocab_size - 1, int(t))) for t in tokens]
    return clipped[:limit]



def _planner_message(env) -> list[int]:
    stages = env.scenario_state.data.get("stages", [])
    open_stages = [s for s in stages if not s["done"]]
    stage = open_stages[0] if open_stages else None
    if stage is None:
        return [12, 0]
    sx, sy = stage["station"]
    required = stage["required"]
    tokens = [12, stage["stage"], sx, sy, len(required)]
    tokens.extend(required)
    return tokens



def pipeline_planner_comm(env):
    base = pipeline_central_planner(env)
    passable = {0, 2, 3}
    mem = {aid: {} for aid in range(env.num_agents)}
    last_tokens: list[int] | None = None

    def _policy(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        nonlocal last_tokens
        actions = base(obs, info, state)
        tokens = _planner_message(env)
        tokens = _clip_tokens(tokens, env.token_vocab_size, env.comm_token_limit)
        send = tokens != (last_tokens or [])
        if send:
            last_tokens = tokens
        for aid in range(env.num_agents):
            act = actions.get(aid, {"action": env.ACTION_STAY})
            act["message_tokens"] = tokens if send else []
            actions[aid] = act
        return actions

    return _policy


def pipeline_planner_follower(env):
    """
    Decentralized follower: uses only local obs + received planner messages.
    Message format: [12, stage_id, sx, sy, req_len, req1, req2, ...]
    """
    def _decode_message(ob):
        tokens = ob.get("messages_tokens")
        if tokens is None:
            return None
        for msg in tokens:
            if len(msg) < 5:
                continue
            if int(msg[0]) != 12:
                continue
            stage_id = int(msg[1])
            sx, sy = int(msg[2]), int(msg[3])
            req_len = int(msg[4])
            reqs = []
            for i in range(req_len):
                if 5 + i < len(msg):
                    val = int(msg[5 + i])
                    if val >= 0:
                        reqs.append(val)
            return {"stage": stage_id, "station": (sx, sy), "reqs": reqs}
        return None

    def _local_grid_ids(ob):
        local = ob["local_grid"]
        if local.ndim == 3:
            return np.argmax(local, axis=0).astype(np.int16)
        return local.astype(np.int16)

    from .local_oracle import local_oracle_plus, _bfs_next_step, _frontier_goals
    explorer = local_oracle_plus(env)
    passable = {0, 2, 3}
    mem = {aid: {} for aid in range(env.num_agents)}

    def _policy(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        explore_actions = explorer(obs, info, state)
        for aid, ob in obs.items():
            msg = _decode_message(ob)
            local_ids = _local_grid_ids(ob)
            local_resource = ob.get("local_resource_types")
            inv = int(ob["inventory"][0])
            px, py = int(ob["self_pos"][0]), int(ob["self_pos"][1])
            # update memory map
            h, w = local_ids.shape
            cx, cy = w // 2, h // 2
            for y in range(h):
                for x in range(w):
                    gx, gy = px + (x - cx), py + (y - cy)
                    mem[aid][(gx, gy)] = int(local_ids[y, x])

            if msg is None:
                actions[aid] = explore_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                actions[aid]["message_tokens"] = []
                continue

            station = msg["station"]
            reqs = msg["reqs"]

            # interact if at station with any resource
            if (px, py) == station and inv != 0:
                actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue

            # pickup required resource if standing on it
            if inv == 0 and local_resource is not None:
                cy, cx = local_resource.shape[0] // 2, local_resource.shape[1] // 2
                center_type = int(local_resource[cy, cx])
                if center_type in reqs:
                    actions[aid] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                    continue

            # move toward required resource (local view)
            if inv == 0 and local_resource is not None:
                h, w = local_resource.shape
                cx, cy = w // 2, h // 2
                best = None
                best_dist = 1e9
                for y in range(h):
                    for x in range(w):
                        if int(local_resource[y, x]) in reqs:
                            dist = abs(x - cx) + abs(y - cy)
                            if dist < best_dist:
                                best_dist = dist
                                best = (x - cx, y - cy)
                if best is not None:
                    dx, dy = best
                    actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
                    continue

            # if holding resource, move toward station
            if inv != 0:
                dx = station[0] - px
                dy = station[1] - py
                actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
                continue

            # fallback: explore within local grid
            actions[aid] = explore_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
            actions[aid]["message_tokens"] = []
        return actions

    from .local_oracle import _move_from_delta  # reuse helper
    import numpy as np

    return _policy


def energy_planner_comm(env):
    """
    Centralized energy planner + comm broadcast.
    Message format: [16, node_x, node_y, node_type]
    """
    from .planner import energy_central_planner
    base = energy_central_planner(env)
    last_tokens: list[int] | None = None

    def _message():
        node_energy = env.scenario_state.data.get("node_energy", {})
        node_types = env.scenario_state.data.get("node_types", {})
        if not node_energy:
            return [16, 0, 0, 0]
        target_node = sorted(node_energy.items(), key=lambda kv: kv[1])[0][0]
        tx, ty = target_node
        ttype = node_types.get(target_node, 0)
        return [16, tx, ty, ttype]

    def _policy(obs: dict, info: dict, state: dict):
        nonlocal last_tokens
        actions = base(obs, info, state)
        tokens = _message()
        tokens = _clip_tokens(tokens, env.token_vocab_size, env.comm_token_limit)
        send = tokens != (last_tokens or [])
        if send:
            last_tokens = tokens
        for aid in range(env.num_agents):
            act = actions.get(aid, {"action": env.ACTION_STAY})
            act["message_tokens"] = tokens if send else []
            actions[aid] = act
        return actions

    return _policy


def signal_hunt_planner_comm(env):
    """
    Centralized signal hunt planner + comm broadcast.
    Message format: [17, target_x, target_y]
    """
    from .planner import signal_hunt_central_planner
    base = signal_hunt_central_planner(env)
    last_tokens: list[int] | None = None

    def _message():
        target = env.scenario_state.data.get("target")
        if target is None:
            return [17, 0, 0]
        return [17, target[0], target[1]]

    def _policy(obs: dict, info: dict, state: dict):
        nonlocal last_tokens
        actions = base(obs, info, state)
        tokens = _message()
        tokens = _clip_tokens(tokens, env.token_vocab_size, env.comm_token_limit)
        send = tokens != (last_tokens or [])
        if send:
            last_tokens = tokens
        for aid in range(env.num_agents):
            act = actions.get(aid, {"action": env.ACTION_STAY})
            act["message_tokens"] = tokens if send else []
            actions[aid] = act
        return actions

    return _policy


def pipeline_planner_comm_followers(env):
    """
    Composite: centralized planner emits messages, decentralized followers act on them.
    """
    planner = pipeline_planner_comm(env)
    follower = pipeline_planner_follower(env)

    def _policy(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        # planner emits messages (stored in actions)
        planner_actions = planner(obs, info, state)
        # follower decides actions using local obs + inbox messages
        follower_actions = follower(obs, info, state)
        # merge: follower actions for movement + planner messages
        actions: Dict[int, dict] = {}
        for aid in range(env.num_agents):
            act = follower_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
            msg = planner_actions.get(aid, {}).get("message_tokens", [])
            # only agent 0 broadcasts planner message to avoid inbox flooding
            act["message_tokens"] = msg if aid == 0 else []
            # sightings go via aux channel
            act["message_tokens_aux"] = act.get("message_tokens_aux", [])
            actions[aid] = act
        return actions

    return _policy


def pipeline_planner_comm_followers_regions(env):
    """
    Composite: planner broadcasts stage + assigns each agent a search region (quadrant).
    Follower explores region and picks required resources when seen.
    """
    base = pipeline_central_planner(env)
    last_tokens: dict[int, list[int]] = {}

    def _region_for_agent(aid: int) -> int:
        return aid % 4

    def _planner_message(aid: int) -> list[int]:
        stages = env.scenario_state.data.get("stages", [])
        open_stages = [s for s in stages if not s["done"]]
        stage = open_stages[0] if open_stages else None
        if stage is None:
            return [13, 0, 0, 0, 0, _region_for_agent(aid)]
        sx, sy = stage["station"]
        reqs = stage["required"]
        tokens = [13, stage["stage"], sx, sy, len(reqs)]
        tokens.extend(reqs[:2])
        tokens.append(_region_for_agent(aid))
        return tokens

    def _in_region(x: int, y: int, region: int) -> bool:
        half = env.map_size / 2
        if region == 0:
            return x < half and y < half  # NW
        if region == 1:
            return x >= half and y < half  # NE
        if region == 2:
            return x < half and y >= half  # SW
        return x >= half and y >= half  # SE

    def _policy(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions = base(obs, info, state)
        for aid, ob in obs.items():
            msg = _planner_message(aid)
            msg = _clip_tokens(msg, env.token_vocab_size, env.comm_token_limit)
            if msg != last_tokens.get(aid, []):
                actions.setdefault(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                actions[aid]["message_tokens"] = msg
                last_tokens[aid] = msg
            else:
                actions.setdefault(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                actions[aid]["message_tokens"] = []
        return actions

    def _follower(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        for aid, ob in obs.items():
            local_ids = ob["local_grid"]
            if local_ids.ndim == 3:
                local_ids = np.argmax(local_ids, axis=0).astype(np.int16)
            inv = int(ob["inventory"][0])
            px, py = int(ob["self_pos"][0]), int(ob["self_pos"][1])
            local_resource = ob.get("local_resource_types")
            sight_msg = None

            # decode last message from inbox
            msg = None
            tokens = ob.get("messages_tokens")
            if tokens is not None:
                for m in tokens:
                    if len(m) >= 6 and int(m[0]) == 13:
                        msg = m
                        break
            if msg is None:
                actions[aid] = {"action": env.ACTION_STAY, "message_tokens": []}
                continue
            req_len = int(msg[4])
            reqs = []
            if req_len > 0 and len(msg) > 5:
                reqs.append(int(msg[5]))
            if req_len > 1 and len(msg) > 6:
                reqs.append(int(msg[6]))
            region = int(msg[-1])

            # send resource sightings
            if local_resource is not None:
                h, w = local_resource.shape
                cx, cy = w // 2, h // 2
                for y in range(h):
                    for x in range(w):
                        rtype = int(local_resource[y, x])
                        if rtype <= 0:
                            continue
                        gx, gy = px + (x - cx), py + (y - cy)
                        sight = [14, rtype, gx, gy]
                        actions.setdefault(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                        actions[aid]["message_tokens"] = sight
                        break

            # pickup if on resource
            if inv == 0 and local_resource is not None:
                cy, cx = local_resource.shape[0] // 2, local_resource.shape[1] // 2
                center_type = int(local_resource[cy, cx])
                if center_type in reqs:
                    actions[aid] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                    continue

            # move toward any sighted resource of required type
            sight_targets = []
            if tokens is not None:
                for m in tokens:
                    if len(m) >= 4 and int(m[0]) == 14:
                        rtype = int(m[1])
                        if rtype in reqs:
                            gx, gy = int(m[2]), int(m[3])
                            sight_targets.append((gx, gy))
            if inv == 0 and sight_targets:
                tx, ty = sight_targets[0]
                dx = tx - px
                dy = ty - py
                actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
                continue

            # move toward nearest required resource in region (local view)
            if inv == 0 and local_resource is not None:
                h, w = local_resource.shape
                cx, cy = w // 2, h // 2
                best = None
                best_dist = 1e9
                for y in range(h):
                    for x in range(w):
                        gx, gy = px + (x - cx), py + (y - cy)
                        if not _in_region(gx, gy, region):
                            continue
                        if int(local_resource[y, x]) in reqs:
                            dist = abs(x - cx) + abs(y - cy)
                            if dist < best_dist:
                                best_dist = dist
                                best = (x - cx, y - cy)
                if best is not None:
                    dx, dy = best
                    actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
                    continue

            # explore within region: bias movement toward region center
            half = env.map_size / 2
            if region == 0:
                target = (int(half * 0.25), int(half * 0.25))
            elif region == 1:
                target = (int(half * 1.25), int(half * 0.25))
            elif region == 2:
                target = (int(half * 0.25), int(half * 1.25))
            else:
                target = (int(half * 1.25), int(half * 1.25))
            dx = target[0] - px
            dy = target[1] - py
            actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
        return actions

    from .local_oracle import _move_from_delta  # reuse helper
    import numpy as np

    def _composite(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        planner_actions = _policy(obs, info, state)
        follower_actions = _follower(obs, info, state)
        actions: Dict[int, dict] = {}
        for aid in range(env.num_agents):
            act = follower_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
            msg = planner_actions.get(aid, {}).get("message_tokens", [])
            act["message_tokens"] = msg
            actions[aid] = act
        return actions

    return _composite


def pipeline_planner_dispatcher(env):
    """
    Planner collects resource sightings and dispatches agents to exact coords.
    Message format: [15, sx, sy, need, rx, ry] where rx,ry = -1 if unknown.
    """
    from .planner import pipeline_central_planner
    from .local_oracle import _move_from_delta, _bfs_next_step, _frontier_goals
    import numpy as np

    base = pipeline_central_planner(env)
    passable = {0, 2, 3}
    mem = {aid: {} for aid in range(env.num_agents)}
    last_tokens: dict[int, list[int]] = {}
    sightings: dict[int, tuple[int, int]] = {}

    def _planner_message(need: int, station: tuple[int, int], sight: tuple[int, int] | None):
        sx, sy = station
        if sight is None:
            return [15, sx, sy, need, -1, -1]
        rx, ry = sight
        return [15, sx, sy, need, rx, ry]

    def _planner(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions = base(obs, info, state)
        # collect sighting messages from inboxes with sender info
        for ob in obs.values():
            tokens = ob.get("messages_tokens")
            senders = ob.get("message_from")
            if tokens is None or senders is None:
                continue
            for idx, m in enumerate(tokens):
                if len(m) >= 4 and int(m[0]) == 14:
                    rtype = int(m[1])
                    rx, ry = int(m[2]), int(m[3])
                    sender = int(senders[idx])
                    if sender >= 0:
                        sightings[rtype] = (rx, ry, sender)
        # get current stage needs
        stages = env.scenario_state.data.get("stages", [])
        open_stages = [s for s in stages if not s["done"]]
        stage = open_stages[0] if open_stages else None
        if stage is None:
            return actions
        station = stage["station"]
        required = stage["required"]
        # broadcast assignment per agent (prefer assigning sighting agent)
        assigned = {}
        for rtype, sight in sightings.items():
            rx, ry, sender = sight
            if sender not in assigned:
                assigned[sender] = (rtype, (rx, ry))
        for aid in range(env.num_agents):
            if aid in assigned:
                need, sight = assigned[aid]
            else:
                need = required[aid % len(required)] if required else -1
                sight = sightings.get(need)
                if sight is not None and len(sight) == 3:
                    sight = (sight[0], sight[1])
            msg = _planner_message(need, station, sight if isinstance(sight, tuple) else None)
            msg = _clip_tokens(msg, env.token_vocab_size, env.comm_token_limit)
            if msg != last_tokens.get(aid, []):
                actions.setdefault(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                actions[aid]["message_tokens"] = msg
                last_tokens[aid] = msg
            else:
                actions.setdefault(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                actions[aid]["message_tokens"] = []
        return actions

    from .local_oracle import local_oracle_plus
    explorer = local_oracle_plus(env)

    def _follower(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        explore_actions = explorer(obs, info, state)
        for aid, ob in obs.items():
            local_ids = ob["local_grid"]
            if local_ids.ndim == 3:
                local_ids = np.argmax(local_ids, axis=0).astype(np.int16)
            inv = int(ob["inventory"][0])
            px, py = int(ob["self_pos"][0]), int(ob["self_pos"][1])
            local_resource = ob.get("local_resource_types")

            # decode dispatcher message
            msg = None
            tokens = ob.get("messages_tokens")
            if tokens is not None:
                for m in tokens:
                    if len(m) >= 6 and int(m[0]) == 15:
                        msg = m
                        break
            if msg is None:
                actions[aid] = explore_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                actions[aid]["message_tokens"] = []
                continue
            station = (int(msg[1]), int(msg[2]))
            need = int(msg[3])
            rx, ry = int(msg[4]), int(msg[5])

            sight_msg = None
            # send sightings
            if local_resource is not None:
                h, w = local_resource.shape
                cx, cy = w // 2, h // 2
                for y in range(h):
                    for x in range(w):
                        rtype = int(local_resource[y, x])
                        if rtype <= 0:
                            continue
                        gx, gy = px + (x - cx), py + (y - cy)
                        sight_msg = [14, rtype, gx, gy]
                        break

            # pickup if on needed resource
            if inv == 0 and local_resource is not None:
                cy, cx = local_resource.shape[0] // 2, local_resource.shape[1] // 2
                center_type = int(local_resource[cy, cx])
                if center_type == need:
                    actions[aid] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                    if sight_msg is not None:
                        actions[aid]["message_tokens_aux"] = sight_msg
                    continue

            # if have resource, go to station
            if inv != 0:
                step = _bfs_next_step((px, py), {station}, passable, mem[aid])
                if step is None:
                    actions[aid] = explore_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                else:
                    dx, dy = step
                    actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
                if sight_msg is not None:
                    actions[aid]["message_tokens_aux"] = sight_msg
                continue

            # if sighting exists, go there
            if rx >= 0 and ry >= 0:
                step = _bfs_next_step((px, py), {(rx, ry)}, passable, mem[aid])
                if step is None:
                    actions[aid] = explore_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                else:
                    dx, dy = step
                    actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
                if sight_msg is not None:
                    actions[aid]["message_tokens_aux"] = sight_msg
                continue

            frontier = _frontier_goals(mem[aid], passable)
            step = _bfs_next_step((px, py), frontier, passable, mem[aid]) if frontier else None
            if step is None:
                actions[aid] = explore_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
            else:
                dx, dy = step
                actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
            actions[aid]["message_tokens_aux"] = sight_msg or []
        return actions

    def _composite(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        planner_actions = _planner(obs, info, state)
        # inject planner messages into follower obs for same-step reaction
        obs_with_msgs = {}
        for aid, ob in obs.items():
            msg = planner_actions.get(aid, {}).get("message_tokens", [])
            ob2 = dict(ob)
            # prepend planner message
            if msg:
                tokens = ob.get("messages_tokens")
                if tokens is not None:
                    tokens = tokens.copy()
                    tokens[0] = -1
                    tokens[0, : len(msg)] = msg
                    ob2["messages_tokens"] = tokens
            obs_with_msgs[aid] = ob2
        follower_actions = _follower(obs_with_msgs, info, state)
        actions: Dict[int, dict] = {}
        for aid in range(env.num_agents):
            act = follower_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
            msg = planner_actions.get(aid, {}).get("message_tokens", [])
            act["message_tokens"] = msg
            actions[aid] = act
        return actions

    return _composite


def pipeline_planner_semidec(env):
    """
    Semi-decentralized expert:
    - Agents send local resource/station sightings via aux channel.
    - Planner builds shared resource map from messages only.
    - Planner broadcasts assignments; followers execute locally.
    """
    from .planner import pipeline_central_planner
    from .local_oracle import local_oracle_plus, _move_from_delta
    import numpy as np

    base = pipeline_central_planner(env)
    shared_resource_types: dict[tuple[int, int], int] = {}
    last_tokens: dict[int, list[int]] = {}
    explorer = local_oracle_plus(env)

    def _planner(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions = base(obs, info, state)
        # collect sightings from inbox
        for ob in obs.values():
            tokens = ob.get("messages_tokens")
            senders = ob.get("message_from")
            if tokens is None or senders is None:
                continue
            for idx, m in enumerate(tokens):
                if len(m) >= 5 and int(m[0]) == 20:
                    tile = int(m[1])
                    subtype = int(m[2])
                    rx, ry = int(m[3]), int(m[4])
                    if tile == 2:  # resource
                        shared_resource_types[(rx, ry)] = subtype
        # compute stage needs
        stages = env.scenario_state.data.get("stages", [])
        open_stages = [s for s in stages if not s["done"]]
        stage = open_stages[0] if open_stages else None
        if stage is None:
            return actions
        station = stage["station"]
        required = stage["required"]
        # broadcast target per agent (resource type + station)
        for aid in range(env.num_agents):
            need = required[aid % len(required)] if required else -1
            msg = [21, station[0], station[1], need]
            msg = _clip_tokens(msg, env.token_vocab_size, env.comm_token_limit)
            if msg != last_tokens.get(aid, []):
                actions.setdefault(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                actions[aid]["message_tokens"] = msg
                last_tokens[aid] = msg
            else:
                actions.setdefault(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                actions[aid]["message_tokens"] = []
        return actions

    def _follower(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        explore_actions = explorer(obs, info, state)
        for aid, ob in obs.items():
            local_ids = ob["local_grid"]
            if local_ids.ndim == 3:
                local_ids = np.argmax(local_ids, axis=0).astype(np.int16)
            inv = int(ob["inventory"][0])
            px, py = int(ob["self_pos"][0]), int(ob["self_pos"][1])
            local_resource = ob.get("local_resource_types")

            # send local sightings (resource, station)
            sight = None
            if local_resource is not None:
                h, w = local_resource.shape
                cx, cy = w // 2, h // 2
                for y in range(h):
                    for x in range(w):
                        rtype = int(local_resource[y, x])
                        if rtype > 0:
                            gx, gy = px + (x - cx), py + (y - cy)
                            sight = [20, 2, rtype, gx, gy]
                            break
            if sight is None:
                # station sighting
                cy, cx = local_ids.shape[0] // 2, local_ids.shape[1] // 2
                if int(local_ids[cy, cx]) == 3:
                    sight = [20, 3, 0, px, py]

            # decode planner message
            msg = None
            tokens = ob.get("messages_tokens")
            if tokens is not None:
                for m in tokens:
                    if len(m) >= 4 and int(m[0]) == 21:
                        msg = m
                        break
            if msg is None:
                actions[aid] = explore_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                actions[aid]["message_tokens_aux"] = sight or []
                continue

            station = (int(msg[1]), int(msg[2]))
            need = int(msg[3])

            # pickup if on needed resource
            if inv == 0 and local_resource is not None:
                cy, cx = local_resource.shape[0] // 2, local_resource.shape[1] // 2
                if int(local_resource[cy, cx]) == need:
                    actions[aid] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                    actions[aid]["message_tokens_aux"] = sight or []
                    continue

            # if holding resource, go to station
            if inv != 0:
                dx = station[0] - px
                dy = station[1] - py
                actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
                actions[aid]["message_tokens_aux"] = sight or []
                continue

            # if planner-known resource sighted in shared map (approx)
            target = None
            for (rx, ry), rtype in shared_resource_types.items():
                if rtype == need:
                    target = (rx, ry)
                    break
            if target is not None:
                dx = target[0] - px
                dy = target[1] - py
                actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
                actions[aid]["message_tokens_aux"] = sight or []
                continue

            actions[aid] = explore_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
            actions[aid]["message_tokens_aux"] = sight or []
        return actions

    def _composite(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        planner_actions = _planner(obs, info, state)
        # inject planner messages into follower obs for same-step reaction
        obs_with_msgs = {}
        for aid, ob in obs.items():
            msg = planner_actions.get(aid, {}).get("message_tokens", [])
            ob2 = dict(ob)
            if msg:
                tokens = ob.get("messages_tokens")
                if tokens is not None:
                    tokens = tokens.copy()
                    tokens[0] = -1
                    tokens[0, : len(msg)] = msg
                    ob2["messages_tokens"] = tokens
            obs_with_msgs[aid] = ob2
        follower_actions = _follower(obs_with_msgs, info, state)
        actions: Dict[int, dict] = {}
        for aid in range(env.num_agents):
            act = follower_actions.get(aid, {"action": env.ACTION_STAY, "message_tokens": []})
            msg = planner_actions.get(aid, {}).get("message_tokens", [])
            act["message_tokens"] = msg if aid == 0 else []
            act["message_tokens_aux"] = act.get("message_tokens_aux", [])
            actions[aid] = act
        return actions

    return _composite
