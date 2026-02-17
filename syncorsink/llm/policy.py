from __future__ import annotations

import json
import re
from typing import Callable, Dict, Any, List

from .batch import batch_llm_call
from .cache import PromptCache


ACTION_NAMES = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "stay": 4,
    "interact": 5,
    "pickup": 6,
    "drop": 7,
}
ACTION_ID_TO_NAME = {v: k for k, v in ACTION_NAMES.items()}
MOVE_ACTIONS = {0, 1, 2, 3}
TILE_NAMES = {
    -1: "unknown",
    0: "empty",
    1: "wall",
    2: "resource",
    3: "station",
    4: "node",
    5: "clue",
    6: "target",
    7: "water",
    8: "beacon",
    9: "door",
    10: "unknown",
}


def parse_jsonish(text: str) -> dict | None:
    def _try(raw: str):
        try:
            return json.loads(raw)
        except Exception:
            return None

    s = text if isinstance(text, str) else str(text)
    data = _try(s)
    if isinstance(data, dict):
        return data
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3:
            s2 = "\n".join(lines[1:-1]).strip()
            data = _try(s2)
            if isinstance(data, dict):
                return data
    start = s.find("{")
    end = s.rfind("}")
    if 0 <= start < end:
        data = _try(s[start : end + 1])
        if isinstance(data, dict):
            return data
    return None


def assigned_sector(agent_id: int, num_agents: int) -> str:
    sectors = ["NW", "NE", "SW", "SE"]
    n = max(1, num_agents)
    return sectors[int(agent_id) % min(4, n)]


def normalize_message_text(msg: str | None) -> str | None:
    if not isinstance(msg, str):
        return None
    text = " ".join(msg.strip().split())
    if not text:
        return None
    if "|" in text:
        return text[:120]
    short = " ".join(text.split()[:10])
    return f"status={short}"[:120]


def _is_structured_message(msg: str | None) -> bool:
    if not isinstance(msg, str):
        return False
    text = msg.strip()
    if not text:
        return False
    kv = parse_kv_message(text)
    status = str(kv.get("status", "")).strip()
    if not status:
        return False
    # Require at least one informative field besides status.
    for k in ("target", "task", "intent", "why"):
        if str(kv.get(k, "")).strip():
            return True
    return False


def parse_kv_message(text: str) -> dict:
    out = {}
    if not isinstance(text, str):
        return out
    raw = text.strip()
    if not raw:
        return out
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip().lower()] = v.strip()
        elif "status" not in out:
            out["status"] = p
    return out


def _decode_local_tile(local, y: int, x: int) -> int:
    if getattr(local, "ndim", 0) == 3:
        return int(local[:, y, x].argmax())
    return int(local[y, x])


def _local_to_global(self_pos, y: int, x: int, h: int, w: int) -> tuple[int, int]:
    cx, cy = int(self_pos[0]), int(self_pos[1])
    dx = int(x - (w // 2))
    dy = int(y - (h // 2))
    return (cx + dx, cy + dy)


_NODE_TARGET_RE = re.compile(r"node:(?P<t>\d+)@(?P<x>-?\d+),(?P<y>-?\d+)")
_TYPE_TARGET_RE = re.compile(r"type:(?P<t>\d+)@(?P<x>-?\d+),(?P<y>-?\d+)")
_REGION_TARGET_RE = re.compile(r"(?:region|scan)@(?P<x>-?\d+),(?P<y>-?\d+)")
_STAGE_TARGET_RE = re.compile(r"stage:(?P<s>-?\d+)")


def _parse_node_target(target: str) -> tuple[int, int, int] | None:
    if not isinstance(target, str):
        return None
    m = _NODE_TARGET_RE.search(target) or _TYPE_TARGET_RE.search(target)
    if not m:
        return None
    return (int(m.group("x")), int(m.group("y")), int(m.group("t")))


def _parse_region_target(target: str) -> tuple[int, int] | None:
    if not isinstance(target, str):
        return None
    m = _REGION_TARGET_RE.search(target)
    if not m:
        return None
    return (int(m.group("x")), int(m.group("y")))


def _parse_stage_target(text: str) -> int | None:
    if not isinstance(text, str):
        return None
    m = _STAGE_TARGET_RE.search(text)
    if not m:
        return None
    return int(m.group("s"))


def grid_to_ascii(local_grid) -> str:
    mapping = {
        0: ".",
        1: "#",
        2: "R",
        3: "S",
        4: "N",
        5: "C",
        6: "T",
        7: "W",
        8: "B",
        9: "D",
        10: "?",
    }
    lines = []
    for row in local_grid:
        lines.append("".join(mapping.get(int(v), "?") for v in row))
    return "\n".join(lines)


def int_grid_to_ascii(grid) -> str:
    lines = []
    for row in grid:
        lines.append(" ".join(str(int(v)) for v in row))
    return "\n".join(lines)


def center_tile_id(agent_obs: dict) -> int:
    local = agent_obs.get("local_grid")
    if local is None:
        return -1
    # If one-hot encoded (C,H,W), decode tile id at center.
    if getattr(local, "ndim", 0) == 3:
        h = int(local.shape[1] // 2)
        w = int(local.shape[2] // 2)
        return int(local[:, h, w].argmax())
    if getattr(local, "ndim", 0) == 2:
        h = int(local.shape[0] // 2)
        w = int(local.shape[1] // 2)
        return int(local[h, w])
    return -1


def exploration_summary(agent_obs: dict) -> tuple[float, str]:
    explored = agent_obs.get("explored_mask")
    if explored is None:
        return 0.0, "unknown"
    explored_bool = explored.astype(bool)
    unseen = ~explored_bool
    h, w = int(explored.shape[0]), int(explored.shape[1])
    total = max(1, h * w)
    seen = int(explored_bool.sum())
    ratio = seen / total
    pos = agent_obs.get("self_pos")
    if pos is None:
        return ratio, "unknown"
    x, y = int(pos[0]), int(pos[1])
    north = int(unseen[: y + 1, :].sum()) if y >= 0 else 0
    south = int(unseen[y:, :].sum()) if y < h else 0
    west = int(unseen[:, : x + 1].sum()) if x >= 0 else 0
    east = int(unseen[:, x:].sum()) if x < w else 0
    scores = {"up": north, "down": south, "left": west, "right": east}
    best_dir = max(scores, key=scores.get)
    if scores[best_dir] <= 0:
        best_dir = "unknown"
    return ratio, best_dir


def allowed_actions(agent_obs: dict) -> list[str]:
    mask = agent_obs.get("action_mask")
    if mask is not None:
        out = []
        for i in range(min(len(mask), 8)):
            if int(mask[i]) == 1:
                out.append(ACTION_ID_TO_NAME.get(i, str(i)))
        if out:
            return out

    local = agent_obs.get("local_grid")
    if local is None:
        return ["up", "down", "left", "right", "stay", "interact", "pickup", "drop"]
    if getattr(local, "ndim", 0) == 3:
        h, w = int(local.shape[1]), int(local.shape[2])
    else:
        h, w = int(local.shape[0]), int(local.shape[1])
    cy, cx = h // 2, w // 2
    blocked = {1, 9}  # wall, door
    out = ["stay"]
    neighbors = [("up", cy - 1, cx), ("down", cy + 1, cx), ("left", cy, cx - 1), ("right", cy, cx + 1)]
    for name, y, x in neighbors:
        if 0 <= y < h and 0 <= x < w:
            t = _decode_local_tile(local, y, x)
            if t not in blocked:
                out.append(name)
    # deterministic ordering for prompt stability
    order = ["up", "down", "left", "right", "stay"]
    allowed = [a for a in order if a in out]
    # conservative fallback for non-movement actions when no mask exists
    allowed.extend(["interact", "pickup", "drop"])
    return allowed


def energy_local_summary(agent_obs: dict) -> list[str]:
    lines = []
    node_energy = agent_obs.get("local_node_energy")
    node_types = agent_obs.get("local_node_types")
    res_types = agent_obs.get("local_resource_types")
    if node_energy is None or node_types is None or res_types is None:
        return lines

    h = int(node_energy.shape[0] // 2)
    w = int(node_energy.shape[1] // 2)
    center = (h, w)
    carry = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0

    # Find low-energy node and nearest resource in local FOV.
    low_nodes = []
    resources = []
    matching_nodes = []
    for y in range(node_energy.shape[0]):
        for x in range(node_energy.shape[1]):
            e = int(node_energy[y, x])
            nt = int(node_types[y, x])
            rt = int(res_types[y, x])
            # Ignore unknown/occluded sentinel entries (10).
            if nt > 0 and nt != 10 and e > 0 and e != 10:
                dist = abs(y - center[0]) + abs(x - center[1])
                low_nodes.append((e, dist, nt, x, y))
                if carry > 0 and nt == carry:
                    matching_nodes.append((dist, x, y))
            if rt > 0 and rt != 10:
                dist = abs(y - center[0]) + abs(x - center[1])
                resources.append((dist, rt, x, y))

    if low_nodes:
        low_nodes.sort(key=lambda t: (t[0], t[1]))
        e, dist, nt, x, y = low_nodes[0]
        lines.append(f"Lowest visible node: type={nt} energy={e} local=({x},{y}) dist={dist}")
    if resources:
        resources.sort(key=lambda t: t[0])
        dist, rt, x, y = resources[0]
        lines.append(f"Nearest visible resource: type={rt} local=({x},{y}) dist={dist}")
    if matching_nodes:
        matching_nodes.sort(key=lambda t: t[0])
        dist, x, y = matching_nodes[0]
        lines.append(f"Nearest matching node for carried type={carry}: local=({x},{y}) dist={dist}")
    lines.append(f"Carried resource type: {carry}")
    return lines


def _semantic_mem(state: dict, agent_id: int) -> dict:
    world = state.setdefault("_agent_semantic", {})
    return world.setdefault(int(agent_id), {"nodes": {}, "resources": {}, "landmarks": {}})


def update_agent_commitment(action: dict, agent_obs: dict, state: dict | None, agent_id: int):
    if not isinstance(state, dict):
        return
    if str(state.get("_scenario", "")) != "energy_grid":
        return
    runtime = state.setdefault("_agent_runtime", {})
    hist = runtime.setdefault(int(agent_id), {})
    commit = hist.get("commitment")
    step = int(state.get("step", 0))
    inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
    tile = center_tile_id(agent_obs)

    # First, parse explicit planner state fields if present.
    a_intent = str(action.get("intent", "")).lower().strip()
    a_target = _parse_node_target(str(action.get("target", "")))
    if a_target is not None and a_intent in {"deliver", "gather", "sync", "assist", "switch", "explore"}:
        x, y, t = a_target
        hist["commitment"] = {"intent": a_intent, "target": (x, y), "type": t, "step": step}
        return

    # Parse intent from outgoing message if available.
    txt = str(action.get("message_text") or "").strip()
    if txt:
        kv = parse_kv_message(txt)
        intent = str(kv.get("intent", "")).lower()
        target = _parse_node_target(str(kv.get("target", "")))
        if target is not None and intent in {"deliver", "switch", "assist"}:
            x, y, t = target
            hist["commitment"] = {"intent": intent, "target": (x, y), "type": t, "step": step}
            return

    # Auto-close stale commitment after a delivery-like interaction.
    if commit and int(action.get("action", 4)) == 5 and tile == 4 and inv == 0:
        hist["commitment"] = None
        return

    # TTL: force refresh if commitment is too old.
    if commit is not None:
        age = step - int(commit.get("step", step))
        if age > 12:
            hist["commitment"] = None
            commit = None

    # If still no commitment, derive one from semantic memory so decisions stay consistent.
    if commit is None:
        mem = state.get("_agent_semantic", {}).get(int(agent_id), {})
        nodes = mem.get("nodes", {})
        resources = mem.get("resources", {})
        pos = agent_obs.get("self_pos")
        if pos is None:
            return
        px, py = int(pos[0]), int(pos[1])

        def dist(p):
            return abs(int(p[0]) - px) + abs(int(p[1]) - py)

        # Carrying: commit to matching node type with lowest known energy.
        if inv > 0 and nodes:
            matches = []
            for (x, y), d in nodes.items():
                dtype = int(d.get("type", -1))
                if dtype == inv and dtype != 10:
                    energy = int(d.get("energy", 999))
                    age = step - int(d.get("step", step))
                    matches.append((energy, age, dist((x, y)), x, y))
            if matches:
                matches.sort()
                _, _, _, tx, ty = matches[0]
                hist["commitment"] = {"intent": "deliver", "target": (tx, ty), "type": inv, "step": step}
                return

        # Not carrying: commit to nearest resource, preferring type needed by lowest-energy known node.
        if inv == 0 and resources:
            needed_type = None
            if nodes:
                crit = sorted(
                    [
                        (int(d.get("energy", 999)), int(d.get("type", -1)))
                        for d in nodes.values()
                        if int(d.get("type", -1)) > 0 and int(d.get("type", -1)) != 10
                    ]
                )
                if crit:
                    needed_type = crit[0][1]
            cand = []
            for (x, y), d in resources.items():
                rtype = int(d.get("type", -1))
                if rtype <= 0 or rtype == 10:
                    continue
                pri = 0 if needed_type is not None and rtype == needed_type else 1
                age = step - int(d.get("step", step))
                cand.append((pri, dist((x, y)), age, x, y, rtype))
            if cand:
                cand.sort()
                _, _, _, tx, ty, rt = cand[0]
                hist["commitment"] = {"intent": "gather", "target": (tx, ty), "type": rt, "step": step}
                return


def commitment_prompt_lines(agent_id: int, state: dict | None) -> list[str]:
    if not isinstance(state, dict):
        return []
    runtime = state.get("_agent_runtime", {})
    hist = runtime.get(int(agent_id), {})
    c = hist.get("commitment")
    if not c:
        return []
    step = int(state.get("step", 0))
    age = max(0, step - int(c.get("step", step)))
    t = c.get("target")
    target = f"({int(t[0])},{int(t[1])})" if isinstance(t, tuple) and len(t) == 2 else "unknown"
    lines = [
        f"Current commitment: intent={c.get('intent', 'unknown')} target={target} type={c.get('type', 'unknown')} age={age}",
        "Commitment policy: continue current commitment for 4-8 steps unless blocked/no-progress or a new CRITICAL alert arrives.",
    ]
    return lines


def update_semantic_memory_from_obs(agent_obs: dict, state: dict | None, agent_id: int):
    if not isinstance(state, dict):
        return
    if "self_pos" not in agent_obs or "local_grid" not in agent_obs:
        return
    step = int(state.get("step", 0))
    mem = _semantic_mem(state, agent_id)
    nodes = mem["nodes"]
    resources = mem["resources"]
    landmarks = mem["landmarks"]
    local = agent_obs["local_grid"]
    h, w = (
        (int(local.shape[1]), int(local.shape[2]))
        if getattr(local, "ndim", 0) == 3
        else (int(local.shape[0]), int(local.shape[1]))
    )
    local_node_types = agent_obs.get("local_node_types")
    local_node_energy = agent_obs.get("local_node_energy")
    local_resource_types = agent_obs.get("local_resource_types")

    for y in range(h):
        for x in range(w):
            tile = _decode_local_tile(local, y, x)
            if tile == 10:  # unknown/occluded
                continue
            gx, gy = _local_to_global(agent_obs["self_pos"], y, x, h, w)
            if gx < 0 or gy < 0:
                continue
            if tile == 4 and local_node_types is not None and local_node_energy is not None:
                ntype = int(local_node_types[y, x])
                nenergy = int(local_node_energy[y, x])
                if ntype > 0 and ntype != 10 and nenergy > 0 and nenergy != 10:
                    nodes[(gx, gy)] = {
                        "type": ntype,
                        "energy": nenergy,
                        "step": step,
                        "source": "self",
                        "confidence": 1.0,
                    }
            elif tile == 2 and local_resource_types is not None:
                rtype = int(local_resource_types[y, x])
                if rtype > 0 and rtype != 10:
                    resources[(gx, gy)] = {
                        "type": rtype,
                        "step": step,
                        "source": "self",
                        "confidence": 1.0,
                    }
            elif tile in {3, 5, 6, 7, 8, 9}:
                kind = {
                    3: "station",
                    5: "clue",
                    6: "target_or_decoy",
                    7: "water",
                    8: "beacon",
                    9: "door",
                }.get(tile, "landmark")
                landmarks[(gx, gy)] = {
                    "kind": kind,
                    "step": step,
                    "source": "self",
                }
            else:
                # If currently visible and not a resource tile, clear stale resource belief.
                if (gx, gy) in resources:
                    resources.pop((gx, gy), None)


def update_semantic_memory_from_messages(info: dict, state: dict | None, agent_id: int):
    if not isinstance(state, dict):
        return
    step = int(state.get("step", 0))
    mem = _semantic_mem(state, agent_id)
    nodes = mem["nodes"]
    incoming = info.get("messages_with_sender", {}).get(agent_id, [])
    for item in incoming:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        kv = parse_kv_message(text)
        target = _parse_node_target(str(kv.get("target", "")))
        if target is None:
            continue
        x, y, t = target
        why = str(kv.get("why", "")).lower()
        est = None
        if "critical" in why:
            est = 2
        elif "low" in why:
            est = 5
        prev = nodes.get((x, y))
        # Never override a newer self observation with older message belief.
        if prev and prev.get("source") == "self" and int(prev.get("step", -1)) >= step - 2:
            continue
        nodes[(x, y)] = {
            "type": t,
            "energy": int(est) if est is not None else int(prev.get("energy", -1)) if prev else -1,
            "step": step,
            "source": "message",
            "confidence": 0.6,
        }


def semantic_prompt_lines(agent_id: int, state: dict | None, scenario: str) -> list[str]:
    if not isinstance(state, dict):
        return []
    mem = state.get("_agent_semantic", {}).get(int(agent_id))
    if not mem:
        return []
    step = int(state.get("step", 0))
    lines = []
    if scenario == "energy_grid":
        nodes = mem.get("nodes", {})
        if nodes:
            ranked = []
            for (x, y), d in nodes.items():
                age = max(0, step - int(d.get("step", step)))
                energy = int(d.get("energy", -1))
                ntype = int(d.get("type", -1))
                src = str(d.get("source", "unknown"))
                priority = 0
                if energy >= 0 and energy <= 2:
                    priority = -2
                elif energy >= 0 and energy <= 5:
                    priority = -1
                ranked.append((priority, age, energy if energy >= 0 else 999, x, y, ntype, src))
            ranked.sort()
            lines.append("Semantic memory (best-known nodes):")
            for pr, age, energy, x, y, ntype, src in ranked[:6]:
                e_txt = "unknown" if energy == 999 else str(energy)
                lines.append(f"- node type={ntype} at ({x},{y}) energy={e_txt} age={age} src={src}")
        resources = mem.get("resources", {})
        if resources:
            lines.append("Semantic memory (known resources):")
            items = sorted(
                [(int(v.get("step", step)), int(v.get("type", -1)), x, y) for (x, y), v in resources.items()],
                reverse=True,
            )
            for s, t, x, y in items[:5]:
                age = max(0, step - s)
                lines.append(f"- resource type={t} at ({x},{y}) age={age}")
    elif scenario == "pipeline_assembly":
        resources = mem.get("resources", {})
        if resources:
            lines.append("Semantic memory (known resources):")
            items = sorted(
                [(int(v.get("step", step)), int(v.get("type", -1)), x, y) for (x, y), v in resources.items()],
                reverse=True,
            )
            for s, t, x, y in items[:8]:
                age = max(0, step - s)
                lines.append(f"- resource type={t} at ({x},{y}) age={age}")
        landmarks = mem.get("landmarks", {})
        stations = [
            (int(v.get("step", step)), x, y)
            for (x, y), v in landmarks.items()
            if str(v.get("kind", "")) == "station"
        ]
        if stations:
            lines.append("Semantic memory (known stations):")
            for s, x, y in sorted(stations, reverse=True)[:6]:
                age = max(0, step - s)
                lines.append(f"- station at ({x},{y}) age={age}")
    elif scenario == "signal_hunt":
        landmarks = mem.get("landmarks", {})
        def _pick(kind: str):
            out = []
            for (x, y), v in landmarks.items():
                if str(v.get("kind", "")) == kind:
                    out.append((int(v.get("step", step)), x, y))
            return sorted(out, reverse=True)

        clues = _pick("clue")
        tars = _pick("target_or_decoy")
        water = _pick("water")
        beacons = _pick("beacon")
        if clues:
            lines.append("Semantic memory (known clues):")
            for s, x, y in clues[:8]:
                age = max(0, step - s)
                lines.append(f"- clue at ({x},{y}) age={age}")
        if tars:
            lines.append("Semantic memory (known target_or_decoy markers):")
            for s, x, y in tars[:8]:
                age = max(0, step - s)
                lines.append(f"- target_or_decoy marker at ({x},{y}) age={age}")
        if water:
            lines.append("Semantic memory (known water tiles):")
            for s, x, y in water[:6]:
                age = max(0, step - s)
                lines.append(f"- water at ({x},{y}) age={age}")
        if beacons:
            lines.append("Semantic memory (known beacons):")
            for s, x, y in beacons[:6]:
                age = max(0, step - s)
                lines.append(f"- beacon at ({x},{y}) age={age}")
    return lines


def voi_prompt_lines(agent_obs: dict, info: dict, agent_id: int, state: dict | None) -> list[str]:
    tile = center_tile_id(agent_obs)
    scenario = str(info.get("scenario", ""))
    events = info.get("events", {}).get(agent_id, [])
    incoming = info.get("messages_with_sender", {}).get(agent_id, [])
    runtime = state.get("_agent_runtime", {}) if isinstance(state, dict) else {}
    hist = runtime.get(int(agent_id), {})
    mem = hist.get("memory", [])

    event_tags = []
    if tile == 5:
        event_tags.append("on_clue_tile")
    if tile == 6:
        event_tags.append("on_target_tile")
    if tile == 2:
        event_tags.append("on_resource_tile")
    if tile == 3:
        event_tags.append("on_station_tile")
    if tile == 4:
        event_tags.append("on_node_tile")
    if events:
        event_tags.append("local_event_emitted")
    if incoming:
        event_tags.append("received_teammate_message")
    if any("changed" in str(x) for x in mem[-3:]):
        event_tags.append("local_observation_changed")

    runtime = state.get("_agent_runtime", {}) if isinstance(state, dict) else {}
    hist = runtime.get(int(agent_id), {})
    sent_events = hist.get("sent_events", {})
    recent_sent = []
    if isinstance(sent_events, dict) and isinstance(state, dict):
        now = int(state.get("step", 0))
        for k, s in sent_events.items():
            age = max(0, now - int(s))
            if age <= 20:
                recent_sent.append(f"{k} age={age}")
    lines = [
        "policy for communication:",
        "Message only for valuable state updates: new discovery, task start/change, completion, low-energy alert, or stale-info correction.",
        "Do not repeat the same event (same status+target+intent) unless information changed or message age is old.",
        "Treat information age as uncertainty: rebroadcast only when older reports may be stale or contradicted by new observation.",
        "If no high-value update: set message_text to empty string.",
        "If sending: use compact template status=...|intent=...|target=...|why=...",
        f"Current VoI event tags: {', '.join(event_tags) if event_tags else 'none'}",
    ]
    if recent_sent:
        lines.append("Recently broadcast events:")
        for item in sorted(recent_sent)[:8]:
            lines.append(f"- {item}")
    if scenario == "signal_hunt":
        lines.append("Signal Hunt VoI: send new clue constraints, target evidence, and scan sync status.")
    elif scenario == "pipeline_assembly":
        lines.append("Pipeline VoI: send stage requirements, dependency blockers, and station sync timing changes.")
    elif scenario == "energy_grid":
        lines.append("Energy VoI: send new node/resource discoveries, low-energy alerts, task starts/changes, and recharge completions.")
    return lines


def _remember(state: dict, agent_id: int, text: str):
    runtime = state.setdefault("_agent_runtime", {})
    hist = runtime.setdefault(int(agent_id), {"memory": []})
    memory = hist.setdefault("memory", [])
    memory.append(text)
    if len(memory) > 6:
        del memory[0]


def _event_key_from_message(text: str) -> str | None:
    kv = parse_kv_message(text)
    status = str(kv.get("status", "")).strip().lower()
    intent = str(kv.get("intent", "")).strip().lower()
    target = str(kv.get("target", "")).strip().lower()
    if not (status or intent or target):
        return None
    return f"{status}|{intent}|{target}"


def remember_sent_message(state: dict | None, agent_id: int, message_text: str | None):
    if not isinstance(state, dict):
        return
    text = str(message_text or "").strip()
    if not text:
        return
    key = _event_key_from_message(text)
    if key is None:
        return
    runtime = state.setdefault("_agent_runtime", {})
    hist = runtime.setdefault(int(agent_id), {})
    sent = hist.setdefault("sent_events", {})
    sent[key] = int(state.get("step", 0))


def _recently_sent_same_event(state: dict | None, agent_id: int, message_text: str | None, ttl: int = 6) -> bool:
    if not isinstance(state, dict):
        return False
    text = str(message_text or "").strip()
    if not text:
        return False
    key = _event_key_from_message(text)
    if key is None:
        return False
    runtime = state.setdefault("_agent_runtime", {})
    hist = runtime.setdefault(int(agent_id), {})
    sent = hist.setdefault("sent_events", {})
    if not isinstance(sent, dict):
        return False
    last = sent.get(key)
    if last is None:
        return False
    now = int(state.get("step", 0))
    return (now - int(last)) <= int(ttl)


def update_agent_memory(agent_obs: dict, info: dict, agent_id: int, state: dict | None):
    if not isinstance(state, dict):
        return
    runtime = state.setdefault("_agent_runtime", {})
    hist = runtime.setdefault(
        int(agent_id),
        {
            "last_pos": None,
            "last_issued_action": None,
            "last_inventory_type": None,
            "last_inventory_event": None,
            "memory": [],
            "last_seen_messages": [],
            "queued_actions": [],
            "last_center_tile": None,
            "sent_events": {},
        },
    )
    pos = agent_obs.get("self_pos")
    if pos is not None:
        pos_t = (int(pos[0]), int(pos[1]))
        last_pos = hist.get("last_pos")
        last_act = hist.get("last_issued_action")
        if last_pos is not None and last_act is not None:
            if pos_t == last_pos and int(last_act) in MOVE_ACTIONS:
                _remember(state, agent_id, f"last move action {int(last_act)} made no progress")
            elif pos_t != last_pos and int(last_act) in MOVE_ACTIONS:
                _remember(state, agent_id, f"last move action {int(last_act)} changed position")
        hist["last_pos"] = pos_t
    center = center_tile_id(agent_obs)
    last_center = hist.get("last_center_tile")
    if last_center is not None and center != last_center:
        _remember(state, agent_id, f"center tile changed {TILE_NAMES.get(int(last_center), 'unknown')} -> {TILE_NAMES.get(int(center), 'unknown')}")
    hist["last_center_tile"] = center

    inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
    prev_inv = hist.get("last_inventory_type")
    if prev_inv is not None and int(prev_inv) != inv:
        _remember(state, agent_id, f"inventory changed {int(prev_inv)} -> {inv}")
        if int(prev_inv) == 0 and inv > 0:
            evt = f"picked_up_type_{inv}"
            _remember(state, agent_id, f"picked resource type {inv}")
        elif int(prev_inv) > 0 and inv == 0:
            evt = f"consumed_or_delivered_type_{int(prev_inv)}"
            _remember(state, agent_id, f"inventory cleared after using type {int(prev_inv)}")
        else:
            evt = f"swapped_type_{int(prev_inv)}_to_{inv}"
            _remember(state, agent_id, f"swapped carried type {int(prev_inv)} -> {inv}")
        hist["last_inventory_event"] = evt
    hist["last_inventory_type"] = inv

    incoming = info.get("messages_with_sender", {}).get(agent_id, [])
    compact = [f"a{int(m.get('from', -1))}:{str(m.get('text', '')).strip()}" for m in incoming if str(m.get("text", "")).strip()]
    old = hist.get("last_seen_messages", [])
    if compact and compact != old:
        _remember(state, agent_id, "new teammate messages observed")
        _remember(state, agent_id, f"latest_msg {compact[-1]}")
    hist["last_seen_messages"] = compact


def memory_prompt_lines(agent_id: int, state: dict | None) -> list[str]:
    if not isinstance(state, dict):
        return []
    runtime = state.get("_agent_runtime", {})
    hist = runtime.get(int(agent_id), {})
    memory = hist.get("memory", [])
    lines = []
    if memory:
        lines.append("Recent memory:")
        for item in memory[-6:]:
            lines.append(f"- {item}")
    stuck_steps = int(hist.get("stuck_steps", 0))
    repeat_action = int(hist.get("repeat_action", 0))
    if stuck_steps >= 2 or repeat_action >= 2:
        lines.append("Reminder: repeated non-progress actions detected; change direction or interact only on relevant tiles.")
    lines.extend(commitment_prompt_lines(agent_id, state))
    team_commitments = state.get("_team_commitments", {}) if isinstance(state, dict) else {}
    if team_commitments:
        now = int(state.get("step", 0))
        lines.append("Known team commitments:")
        for aid in sorted(team_commitments.keys()):
            c = team_commitments[aid]
            age = max(0, now - int(c.get("step", now)))
            intent = str(c.get("intent", "unknown"))
            target = str(c.get("target", "unknown"))
            lines.append(f"- agent_{int(aid)} intent={intent} target={target} age={age}")
    return lines


def refresh_team_commitments(info: dict, state: dict | None):
    if not isinstance(state, dict):
        return
    commits = state.setdefault("_team_commitments", {})
    now = int(state.get("step", 0))
    msg_by_agent = info.get("messages_with_sender", {})
    for _recv, items in msg_by_agent.items():
        for item in items:
            sender = int(item.get("from", -1))
            text = str(item.get("text", "")).strip()
            if sender < 0 or not text:
                continue
            kv = parse_kv_message(text)
            status = str(kv.get("status", "")).lower()
            intent = str(kv.get("intent", "")).lower()
            target = str(kv.get("target", "")).lower()
            if intent or target or status in {"assign", "claim", "cover", "assist"}:
                commits[sender] = {
                    "intent": intent or status or "unknown",
                    "target": target or kv.get("pos", "unknown"),
                    "step": now,
                }


def _common_prompt_header(agent_obs: dict, info: dict, agent_id: int, state: dict | None) -> list[str]:
    local = agent_obs["local_grid"]
    inventory = int(agent_obs["inventory"][0])
    pos = agent_obs.get("self_pos")
    hints = info.get("goal_hint_texts", {}).get(agent_id)
    messages = info.get("messages_text", {}).get(agent_id, [])
    step = None if state is None else state.get("step")
    center_tile = center_tile_id(agent_obs)
    center_tile_name = TILE_NAMES.get(center_tile, "unknown")
    explore_ratio, frontier_dir = exploration_summary(agent_obs)
    local_events = info.get("events", {}).get(agent_id, [])
    scenario = str(info.get("scenario", ""))
    n_agents = int(state.get("_num_agents", 0)) if isinstance(state, dict) else 0
    sector = assigned_sector(agent_id, n_agents if n_agents > 0 else 4)

    lead_line = "You are one agent in a cooperative multi-agent POMDP."
    if scenario == "energy_grid":
        lead_line = (
            "You are the planner for an agent in a cooperative multi-agent POMDP energy grid game. "
            "The goal of the game is to find low-energy nodes and coordinate typed resource deliveries needed to recharge them before they run out."
        )

    prompt = [
        lead_line,
        "Return only JSON with keys: action, message_text, plan_actions.",
        "JSON schema: {\"action\": int|name, \"message_text\": string, \"plan_actions\": [up to 5]}.",
        "message_text is your decision variable for whether this agent should send a message to other agents this step.",
        "Valid actions: up, down, left, right, stay, interact, pickup, drop.",
        "You are the local planner for this agent. Decide from local observation, semantic memory, recent events, and teammate messages.",
        "Keep communication concise and task-relevant.",
        "Do not repeat the same message unless there is new information.",
        "If the same movement fails repeatedly, pick a different direction.",
        "Plan cooperatively: explore systematically, share important discoveries, and avoid duplicate work.",
        "Use compact event messages: status=...|intent=...|target=...|why=... .",
        f"Inventory: {inventory}",
        f"Default assigned sector: {sector}",
        f"Center tile: {center_tile_name} ({center_tile})",
        "Allowed actions now: " + ", ".join(allowed_actions(agent_obs)),
        f"Exploration coverage: {explore_ratio:.2f}",
        f"Suggested unexplored direction: {frontier_dir}",
    ]
    if pos is not None:
        prompt.append(f"Self position (local map coords): ({int(pos[0])},{int(pos[1])})")
    if step is not None:
        prompt.append(f"Step: {step}")
    prompt.append("Local tile view:")
    prompt.append(grid_to_ascii(local))
    if hints:
        prompt.append(f"Agent hint: {hints}")
    if local_events:
        prompt.append("Recent local events:")
        for ev in local_events[-4:]:
            prompt.append(str(ev))
    messages_with_sender = info.get("messages_with_sender", {}).get(agent_id, [])
    if messages_with_sender:
        prompt.append("Incoming messages with sender:")
        for item in messages_with_sender[-8:]:
            sender = int(item.get("from", -1))
            text = str(item.get("text", "")).strip()
            if text:
                prompt.append(f"agent_{sender}: {text}")
    elif messages:
        prompt.append("Incoming messages: " + " | ".join(m for m in messages if m))
    prompt.extend(voi_prompt_lines(agent_obs, info, agent_id, state))
    return prompt


def default_prompt(obs: dict, info: dict, agent_id: int, state: dict | None = None) -> str:
    agent_obs = obs[agent_id]
    scenario = info.get("scenario", "unknown")
    prompt = _common_prompt_header(agent_obs, info, agent_id, state)

    if "goal_hint" in agent_obs:
        hint_tokens = [int(v) for v in agent_obs["goal_hint"].tolist() if int(v) >= 0]
        if hint_tokens:
            prompt.append("Structured goal hint tokens: " + " ".join(str(v) for v in hint_tokens))

    if scenario == "pipeline_assembly":
        prompt.append("You are the planner for one agent in Pipeline Assembly.")
        prompt.append("Goal: finish all pipeline stages by collecting required resources and completing station interactions with dependencies.")
        prompt.append("Map interpretation for this scenario:")
        prompt.append("- local_grid is your local FOV window, agent at center.")
        prompt.append("- '?' means currently unknown/not visible due to line-of-sight occlusion or bounds.")
        prompt.append("- Tile legend used here: .=empty, #=wall, R=resource, S=station, D=door, ?=unknown.")
        prompt.append("Scenario: pipeline_assembly (task planning with stage dependencies).")
        if "local_resource_types" in agent_obs:
            prompt.append("Local resource types grid (0 means none):")
            prompt.append(int_grid_to_ascii(agent_obs["local_resource_types"]))
        prompt.append("Plan: gather required resources, deliver at stations, coordinate sync interactions.")
        prompt.append("Checklist: pickup required resources, move to station, interact at station, report stage progress once.")
    elif scenario == "energy_grid":
        prompt.append("Map interpretation for this scenario:")
        prompt.append("- local_grid is your local FOV window, agent at center.")
        prompt.append("- '?' means currently unknown/not visible due to line-of-sight occlusion or bounds.")
        prompt.append("- Tile legend used here: .=empty, #=wall, R=resource, N=node, D=door, ?=unknown.")
        prompt.append("Scenario: energy_grid.")
        recharge_count = info.get("recharge_count")
        success_recharges = info.get("success_recharges")
        depleted = info.get("depleted")
        if recharge_count is not None and success_recharges is not None:
            prompt.append(f"Mission progress: recharges={int(recharge_count)}/{int(success_recharges)}")
        if depleted is not None:
            prompt.append(f"Depleted flag: {bool(depleted)}")
        prompt.append("Action policy (apply in order):")
        prompt.append("1) If carrying type t and on node type t -> interact now.")
        prompt.append("2) If carrying type t and a CRITICAL/LOW node type t is known -> move to it.")
        prompt.append("3) If carrying type t and no urgent node visible -> move to nearest known matching node.")
        prompt.append("4) If not carrying and on resource -> pickup now.")
        prompt.append("5) If not carrying -> move to nearest known useful resource (prefer type needed by low-energy nodes).")
        prompt.append("6) If no useful target known -> explore sector systematically until you discover nodes/resources.")
        prompt.append("If no valid target is known (visible or in semantic memory), explore your sector systematically until you discover nodes/resources.")
        prompt.append("Exploration policy: sweep in one direction for multiple steps before turning; announce only new useful discoveries.")
        prompt.append("Environment dynamics:")
        prompt.append("- Nodes lose energy over time (periodic drain).")
        prompt.append("- A valid typed delivery recharges the node immediately on that step.")
        prompt.append("- When a node is low energy, recharge may require synchronized interact by 2 agents.")
        prompt.append("- Episode fails if any node depletes before success.")
        prompt.append("- Episode succeeds when required recharge count is reached.")
        prompt.append("Emergency policy: CRITICAL node energy <=2, LOW <=5. Critical nodes take priority over exploration.")
        prompt.append("Stability policy: keep current commitment for multiple steps; switch only if blocked repeatedly or a new CRITICAL node appears.")
        prompt.append("Communication policy (message only on important change/event):")
        prompt.append("- New node discovered: status=node|target=node:t@x,y|why=found")
        prompt.append("- New resource discovered: status=resource|target=type:t@x,y|why=found")
        prompt.append("- Starting new task: status=start|intent=...|target=...|why=assignment")
        prompt.append("- Completion/update: status=done|target=...|why=recharged")
        prompt.append("- Do not rebroadcast same status+target unless changed or stale by age.")
        prompt.append("Deduplicate with teammate plans: if a teammate already started a target, pick another high-value target.")
        prompt.extend(energy_local_summary(agent_obs))
        prompt.extend(semantic_prompt_lines(agent_id, state, "energy_grid"))
    elif scenario == "signal_hunt":
        prompt.append("You are the planner for one agent in Signal Hunt.")
        prompt.append("Goal: discover clues, fuse constraints across agents, and synchronize target scan.")
        prompt.append("Map interpretation for this scenario:")
        prompt.append("- local_grid is your local FOV window, agent at center.")
        prompt.append("- '?' means currently unknown/not visible due to line-of-sight occlusion or bounds.")
        prompt.append("- Tile legend used here: .=empty, #=wall, C=clue, T=target, W=water, B=beacon, D=door, ?=unknown.")
        prompt.append("Scenario: signal_hunt (cooperative clue fusion and target scan).")
        prompt.append("Plan: collect distributed clues, communicate constraints, converge on likely target, synchronize scan.")
        prompt.append("If you are on a clue or target tile, prefer interact. Otherwise explore; do not spam interact.")
        prompt.append("Checklist:")
        prompt.append("- If center tile is clue/target -> interact.")
        prompt.append("- Else move toward unexplored direction unless teammate assigned a region.")
        prompt.append("- Send message only on new clue/constraint, region assignment, or target confirmation.")
    else:
        prompt.append("Scenario: generic cooperative task.")

    prompt.extend(memory_prompt_lines(agent_id, state))
    return "\n".join(prompt)


def stabilize_agent_action(action: dict, agent_obs: dict, agent_id: int, state: dict | None) -> dict:
    if not isinstance(state, dict):
        return action

    runtime = state.setdefault("_agent_runtime", {})
    hist = runtime.setdefault(
        int(agent_id),
        {
            "last_action": None,
            "repeat_action": 0,
            "last_pos": None,
            "last_positions": [],
            "stuck_steps": 0,
            "last_msg": None,
            "memory": [],
            "queued_actions": [],
            "last_issued_action": None,
            "last_seen_messages": [],
            "interact_history": {},
        },
    )

    out = dict(action)
    act = int(out.get("action", 4))
    msg = out.get("message_text")
    msg_clean = normalize_message_text(msg)
    if isinstance(msg_clean, str):
        if not msg_clean or not _is_structured_message(msg_clean):
            out["message_text"] = None
        else:
            if _recently_sent_same_event(state, agent_id, msg_clean, ttl=6):
                out["message_text"] = None
            else:
                out["message_text"] = msg_clean
                hist["last_msg"] = msg_clean.lower()
    else:
        out["message_text"] = None

    pos = agent_obs.get("self_pos")
    if pos is not None:
        pos_t = (int(pos[0]), int(pos[1]))
        pos_hist = hist.setdefault("last_positions", [])
        pos_hist.append(pos_t)
        if len(pos_hist) > 6:
            del pos_hist[:-6]
        last_pos = hist.get("last_pos")
        last_issued = hist.get("last_issued_action")
        # Count "stuck" only when the previous issued action was movement and made no progress.
        if last_pos is not None and last_issued in MOVE_ACTIONS:
            if last_pos == pos_t:
                hist["stuck_steps"] = int(hist.get("stuck_steps", 0)) + 1
            else:
                hist["stuck_steps"] = 0
        elif last_issued not in MOVE_ACTIONS:
            # Non-movement actions (stay/pickup/interact/drop) should not accumulate stuck.
            hist["stuck_steps"] = 0
        hist["last_pos"] = pos_t

    if hist.get("last_action") == act:
        hist["repeat_action"] = int(hist.get("repeat_action", 0)) + 1
    else:
        hist["repeat_action"] = 0

    repeat_action = int(hist.get("repeat_action", 0))
    stuck_steps = int(hist.get("stuck_steps", 0))
    force_alternate = False
    if act == 4 and repeat_action >= 2:
        force_alternate = True
    if act in MOVE_ACTIONS and repeat_action >= 2 and stuck_steps >= 3:
        force_alternate = True
    if force_alternate:
        step = int(state.get("step", 0))
        alternatives = [0, 1, 2, 3]
        idx = (step + int(agent_id)) % len(alternatives)
        alt = alternatives[idx]
        if alt == act:
            alt = alternatives[(idx + 1) % len(alternatives)]
        act = alt

    # Enforce env-provided valid action mask when available.
    mask = agent_obs.get("action_mask")
    if mask is not None:
        if act < 0 or act >= len(mask) or int(mask[act]) != 1:
            # Prefer valid non-idle actions; fallback to stay.
            valid = [i for i in range(min(len(mask), 8)) if int(mask[i]) == 1]
            preferred = [i for i in valid if i != 4]
            act = preferred[0] if preferred else (4 if 4 in valid else 4)

    # Break short movement oscillations (A-B-A-B patterns) by forcing a perpendicular detour move.
    pos_hist = hist.get("last_positions", [])
    if (
        isinstance(pos_hist, list)
        and len(pos_hist) >= 4
        and pos_hist[-1] == pos_hist[-3]
        and pos_hist[-2] == pos_hist[-4]
        and act in MOVE_ACTIONS
    ):
        mask = agent_obs.get("action_mask")
        valid_moves = []
        if mask is not None:
            valid_moves = [i for i in [0, 1, 2, 3] if i < len(mask) and int(mask[i]) == 1]
        else:
            valid_moves = [ACTION_NAMES[m] for m in allowed_actions(agent_obs) if m in {"up", "down", "left", "right"}]
        if valid_moves:
            reverse = {0: 1, 1: 0, 2: 3, 3: 2}
            # avoid repeating current/reverse direction when possible
            cand = [m for m in valid_moves if m not in {act, reverse.get(act, -1)}]
            if cand:
                act = cand[0]

    # Tile-aware safety gate to avoid pathological interact/pickup loops.
    tile = center_tile_id(agent_obs)
    can_interact = tile in {3, 4, 5, 6}  # station, node, clue, target
    can_pickup = tile == 2
    scenario = str(state.get("_scenario", "")) if isinstance(state, dict) else ""
    inventory = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
    pos_t = tuple(int(v) for v in agent_obs.get("self_pos", (0, 0)))
    interact_history = hist.setdefault("interact_history", {})
    last_interact_step = interact_history.get(pos_t, -10_000)
    interact_recent = (int(state.get("step", 0)) - int(last_interact_step)) <= 4

    if act == 5 and not can_interact:
        step = int(state.get("step", 0))
        alternatives = [0, 1, 2, 3]
        act = alternatives[(step + int(agent_id)) % len(alternatives)]
    elif scenario == "signal_hunt" and tile in {5, 6} and act != 5 and not interact_recent:
        act = 5
    elif act == 6 and not can_pickup:
        if can_interact:
            act = 5
        else:
            step = int(state.get("step", 0))
            alternatives = [0, 1, 2, 3]
            act = alternatives[(step + int(agent_id) + 1) % len(alternatives)]
    elif act == 7 and inventory == 0:
        act = 4

    out["action"] = act
    hist["last_action"] = act
    hist["last_issued_action"] = act
    if act == 5:
        interact_history[pos_t] = int(state.get("step", 0))
    remember_sent_message(state, agent_id, out.get("message_text"))
    return out


class LLMPolicy:
    def __init__(
        self,
        llm_call: Callable[[str], str],
        prompt_fn: Callable[..., str] | None = None,
        default_action: int = 4,
        postprocess: Callable[[dict], dict] | None = None,
        cache: PromptCache | None = None,
        batch_fn: Callable[[List[str], Callable[[str], str]], List[str]] | None = None,
    ):
        self.llm_call = llm_call
        self.prompt_fn = prompt_fn or default_prompt
        self.default_action = default_action
        self.postprocess = postprocess
        self.cache = cache
        self.batch_fn = batch_fn or batch_llm_call

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        if isinstance(state, dict):
            state["_scenario"] = info.get("scenario", "")
            state["_num_agents"] = len(obs)
            refresh_team_commitments(info, state)
            state["task_events"] = []
        scenario = str(state.get("_scenario", "")) if isinstance(state, dict) else str(info.get("scenario", ""))

        def incoming_sig(agent_id: int) -> tuple[str, ...]:
            items = info.get("messages_with_sender", {}).get(agent_id, [])
            sig = []
            for m in items[-8:]:
                sender = int(m.get("from", -1))
                txt = str(m.get("text", "")).strip()
                if txt:
                    sig.append(f"{sender}:{txt}")
            return tuple(sig)

        def should_replan(agent_id: int, agent_obs: dict, queued_actions: list[int]) -> tuple[bool, str]:
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            ctx = hist.get("plan_ctx", {})
            if not queued_actions:
                return True, "no_plan"
            if int(hist.get("stuck_steps", 0)) >= 3:
                return True, "stuck"
            # Replan on new local events from env (e.g., recharge, clue interactions)
            if info.get("events", {}).get(agent_id):
                return True, "local_event"
            # Replan when communication context changes (new teammate message)
            sig = incoming_sig(agent_id)
            if sig and sig != tuple(ctx.get("incoming_sig", ())):
                return True, "new_message"
            # Replan on actionable discovery at center tile.
            center = center_tile_id(agent_obs)
            old_center = int(ctx.get("center_tile", -1))
            if center in {2, 3, 4, 5, 6} and center != old_center:
                return True, "new_center_object"
            # Replan on inventory change (picked/dropped/consumed).
            inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
            if int(ctx.get("inventory", inv)) != inv:
                return True, "inventory_change"
            # Replan when teammate requests sync/assist explicitly.
            for token in sig:
                low = token.lower()
                if ("sync" in low) or ("assist" in low) or ("help" in low):
                    return True, "coord_request"
            return False, "continue_plan"

        def save_plan_ctx(agent_id: int, agent_obs: dict, task: dict | None = None):
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
            target = ""
            if isinstance(task, dict):
                target = str(task.get("target", "") or "")
                if not target:
                    target = str(task.get("message_text", "") or "")
            hist["plan_ctx"] = {
                "step": int(state.get("step", 0)) if isinstance(state, dict) else 0,
                "incoming_sig": incoming_sig(agent_id),
                "center_tile": int(center_tile_id(agent_obs)),
                "inventory": inv,
                "target": target,
            }

        def _pipeline_plan_init(agent_id: int, agent_obs: dict):
            if scenario != "pipeline_assembly":
                return
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            if hist.get("pipeline_plan_initialized"):
                return
            hints = _decode_pipeline_goal_hint(agent_obs)
            plan = []
            for h in hints:
                need = {}
                for t in h.get("required", []):
                    need[int(t)] = int(need.get(int(t), 0)) + 1
                plan.append(
                    {
                        "stage": int(h.get("stage", -1)),
                        "station": tuple(h.get("station", (0, 0))),
                        "need": need,
                        "delivered": {},
                        "sync": bool(h.get("sync", False)),
                        "sync_done": False,
                        "deps": [int(x) for x in h.get("deps", []) if int(x) >= 0],
                        "done": False,
                    }
                )
            hist["pipeline_plan"] = plan
            hist["pipeline_plan_initialized"] = True
            hist["pipeline_stage_cursor"] = int(hist.get("pipeline_stage_cursor", 0))

        def _pipeline_update_progress(agent_id: int, agent_obs: dict):
            if scenario != "pipeline_assembly":
                return
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            _pipeline_plan_init(agent_id, agent_obs)
            plan = hist.get("pipeline_plan", [])
            if not plan:
                return

            by_stage = {int(s["stage"]): s for s in plan}
            for ev in info.get("events", {}).get(agent_id, []):
                evn = str((ev or {}).get("event", "")).strip().lower()
                sid = int((ev or {}).get("stage", -1))
                if sid not in by_stage:
                    continue
                st = by_stage[sid]
                if evn == "delivered":
                    rtype = int((ev or {}).get("resource_type", 0))
                    if rtype <= 0:
                        rtype = int(hist.get("task_last_inventory_type", 0))
                    if rtype > 0:
                        st["delivered"][rtype] = int(st["delivered"].get(rtype, 0)) + 1
                elif evn == "sync_complete":
                    st["sync_done"] = True

            incoming = info.get("messages_with_sender", {}).get(agent_id, [])
            for m in incoming[-8:]:
                kv = parse_kv_message(str(m.get("text", "")).strip())
                if str(kv.get("status", "")).strip().lower() != "done":
                    continue
                sid = _parse_stage_target(str(kv.get("target", "")))
                if sid is None or sid not in by_stage:
                    continue
                by_stage[sid]["done"] = True

            done_stage_ids = {int(s["stage"]) for s in plan if bool(s.get("done", False))}
            for st in plan:
                if st.get("done"):
                    continue
                deps = [int(x) for x in st.get("deps", [])]
                if any(d not in done_stage_ids for d in deps):
                    continue
                need = st.get("need", {})
                delivered = st.get("delivered", {})
                ready = True
                for t, n in need.items():
                    if int(delivered.get(int(t), 0)) < int(n):
                        ready = False
                        break
                if ready and (not st.get("sync", False) or st.get("sync_done", False)):
                    st["done"] = True
                    done_stage_ids.add(int(st["stage"]))

            for i, st in enumerate(plan):
                if not bool(st.get("done", False)):
                    hist["pipeline_stage_cursor"] = int(i)
                    return

        prompts = []
        order = []
        for agent_id in obs.keys():
            update_semantic_memory_from_obs(obs[agent_id], state, agent_id)
            update_semantic_memory_from_messages(info, state, agent_id)
            update_agent_memory(obs[agent_id], info, agent_id, state)
            runtime = state.setdefault("_agent_runtime", {})
            hist = runtime.setdefault(int(agent_id), {"queued_actions": []})
            queued_actions = hist.get("queued_actions", [])
            replan, replan_reason = should_replan(agent_id, obs[agent_id], queued_actions)
            if queued_actions and not replan:
                action_id = int(queued_actions.pop(0))
                model_action = {"action": action_id, "message_text": None}
                action = model_action
                if self.postprocess:
                    action = self.postprocess(action)
                action = stabilize_agent_action(action, obs[agent_id], agent_id, state)
                update_agent_commitment(action, obs[agent_id], state, agent_id)
                actions[agent_id] = action
                if isinstance(state, dict):
                    trace = state.setdefault("llm_calls", [])
                    trace.append(
                        {
                            "agent_id": int(agent_id),
                            "mode": "text",
                            "from_chunk": True,
                            "queued_remaining": len(queued_actions),
                            "replan": False,
                            "replan_reason": replan_reason,
                            "model_action": model_action,
                            "parsed_action": action,
                        }
                    )
                continue
            if replan and queued_actions:
                hist["queued_actions"] = []
            try:
                prompt = self.prompt_fn(obs, info, agent_id, state)
            except TypeError:
                prompt = self.prompt_fn(obs, info, agent_id)
            hist["pending_replan_reason"] = replan_reason
            order.append(agent_id)
            prompts.append(prompt)

        responses = []
        for p in prompts:
            if self.cache:
                cached = self.cache.get(p)
                if cached is not None:
                    responses.append(cached)
                    continue
            responses.append(None)

        missing_idx = [i for i, r in enumerate(responses) if r is None]
        if missing_idx:
            to_call = [prompts[i] for i in missing_idx]
            called = self.batch_fn(to_call, self.llm_call)
            for idx, resp in zip(missing_idx, called):
                responses[idx] = resp
                if self.cache:
                    self.cache.set(prompts[idx], resp)

        for idx, (agent_id, raw) in enumerate(zip(order, responses)):
            model_action = self._parse_response(raw)
            if isinstance(state, dict):
                runtime = state.setdefault("_agent_runtime", {})
                hist = runtime.setdefault(int(agent_id), {"queued_actions": []})
                plan = model_action.get("plan_actions", [])
                if isinstance(plan, list):
                    clean = []
                    for a in plan[:5]:
                        clean.append(int(self._action_to_id(a)))
                    hist["queued_actions"] = clean
                save_plan_ctx(agent_id, obs[agent_id])
            action = model_action
            if self.postprocess:
                action = self.postprocess(action)
            action = stabilize_agent_action(action, obs[agent_id], agent_id, state)
            update_agent_commitment(action, obs[agent_id], state, agent_id)
            actions[agent_id] = action
            if isinstance(state, dict):
                trace = state.setdefault("llm_calls", [])
                queued = state.get("_agent_runtime", {}).get(int(agent_id), {}).get("queued_actions", [])
                reason = state.get("_agent_runtime", {}).get(int(agent_id), {}).get("pending_replan_reason")
                trace.append(
                    {
                        "agent_id": int(agent_id),
                        "mode": "text",
                        "from_chunk": False,
                        "replan": True,
                        "replan_reason": reason,
                        "prompt": prompts[idx],
                        "raw_response": raw,
                        "model_action": model_action,
                        "parsed_action": action,
                        "queued_actions": queued,
                    }
                )
                state.get("_agent_runtime", {}).get(int(agent_id), {}).pop("pending_replan_reason", None)
        return actions

    def _parse_response(self, text: str) -> dict:
        # JSON path
        data = parse_jsonish(text if isinstance(text, str) else str(text))
        if isinstance(data, dict):
            act = data.get("action", self.default_action)
            action_id = self._action_to_id(act)
            message_text = data.get("message_text")
            plan_actions = data.get("plan_actions")
            if plan_actions is None:
                plan_actions = data.get("action_plan")
            out = {"action": action_id, "message_text": message_text}
            if isinstance(plan_actions, list):
                out["plan_actions"] = plan_actions
            return out
        # heuristic parse
        for token in str(text).replace("{", " ").replace("}", " ").split():
            token = token.strip().lower()
            if token.isdigit():
                return {"action": int(token), "message_text": None}
            if token in ACTION_NAMES:
                return {"action": ACTION_NAMES[token], "message_text": None}
        return {"action": self.default_action, "message_text": None}

    def _action_to_id(self, act) -> int:
        if isinstance(act, int):
            return act
        if isinstance(act, str):
            return ACTION_NAMES.get(act.lower(), self.default_action)
        return self.default_action


def _nearest_local_pos(values, valid_fn):
    h, w = values.shape
    cy, cx = h // 2, w // 2
    best = None
    for y in range(h):
        for x in range(w):
            v = int(values[y, x])
            if not valid_fn(v):
                continue
            d = abs(y - cy) + abs(x - cx)
            cand = (d, x, y, v)
            if best is None or cand < best:
                best = cand
    return best


def _move_toward_local(x: int, y: int, values, allowed: set[str]) -> int:
    h, w = values.shape if values.ndim == 2 else (values.shape[1], values.shape[2])
    cy, cx = h // 2, w // 2
    dx = x - cx
    dy = y - cy
    # Manhattan-greedy with mask fallback.
    prefs = []
    if abs(dx) >= abs(dy):
        if dx < 0:
            prefs.append("left")
        elif dx > 0:
            prefs.append("right")
        if dy < 0:
            prefs.append("up")
        elif dy > 0:
            prefs.append("down")
    else:
        if dy < 0:
            prefs.append("up")
        elif dy > 0:
            prefs.append("down")
        if dx < 0:
            prefs.append("left")
        elif dx > 0:
            prefs.append("right")
    prefs.extend(["up", "down", "left", "right", "stay"])
    for p in prefs:
        if p in allowed:
            return ACTION_NAMES[p]
    return ACTION_NAMES["stay"]


def _canonical_executor_task(name: str) -> str:
    raw = str(name or "").strip().lower()
    aliases = {
        # generic aliases
        "pickup_visible": "pickup_visible_resource",
        "deliver_visible": "deliver_to_matching_node",
        "explore": "explore_sector",
        "wait": "hold_position",
        "keep": "hold_position",
        # signal hunt aliases
        "inspect_visible_clue": "sync_interact",
        "verify_reported_target": "sync_interact",
        "explore_for_clues": "explore_sector",
        "coordinate_scan": "respond_to_teammate",
        # pipeline aliases
        "collect_required_part": "pickup_visible_resource",
        "deliver_part_to_station": "deliver_to_matching_node",
        "perform_stage_sync": "sync_interact",
        "explore_for_parts": "explore_sector",
        "coordinate_stage_plan": "respond_to_teammate",
    }
    return aliases.get(raw, raw if raw else "hold_position")


def _semantic_local_map_lines(agent_obs: dict, scenario: str) -> list[str]:
    local = agent_obs.get("local_grid")
    if local is None:
        return []
    h, w = (
        (int(local.shape[1]), int(local.shape[2]))
        if getattr(local, "ndim", 0) == 3
        else (int(local.shape[0]), int(local.shape[1]))
    )
    cx, cy = w // 2, h // 2
    pos = agent_obs.get("self_pos")
    node_types = agent_obs.get("local_node_types")
    node_energy = agent_obs.get("local_node_energy")
    res_types = agent_obs.get("local_resource_types")

    unknown = 0
    walls = 0
    entities = []
    for y in range(h):
        for x in range(w):
            tile = _decode_local_tile(local, y, x)
            if tile == 10:
                unknown += 1
                continue
            if tile == 1:
                walls += 1
                continue
            if x == cx and y == cy:
                continue
            dx, dy = int(x - cx), int(y - cy)
            gxy = None
            if pos is not None:
                gx, gy = _local_to_global(pos, y, x, h, w)
                gxy = (int(gx), int(gy))
            if tile == 2 and res_types is not None:
                rt = int(res_types[y, x])
                if rt > 0 and rt != 10:
                    entities.append(("resource", rt, dx, dy, gxy, None))
            elif tile == 4 and node_types is not None:
                nt = int(node_types[y, x])
                ne = int(node_energy[y, x]) if node_energy is not None else -1
                if nt > 0 and nt != 10:
                    entities.append(("node", nt, dx, dy, gxy, ne))
            elif tile == 5:
                entities.append(("clue", -1, dx, dy, gxy, None))
            elif tile == 6:
                entities.append(("target_or_decoy", -1, dx, dy, gxy, None))
            elif tile == 7:
                entities.append(("water", -1, dx, dy, gxy, None))
            elif tile == 8:
                entities.append(("beacon", -1, dx, dy, gxy, None))
            elif tile == 3:
                entities.append(("station", -1, dx, dy, gxy, None))
            elif tile == 9:
                entities.append(("door", -1, dx, dy, gxy, None))

    lines = [
        "Semantic local map:",
        f"- visible_cells={h*w-unknown} unknown_cells={unknown} walls={walls}",
    ]
    if not entities:
        lines.append("- no salient entities visible")
        return lines
    # Sort by Manhattan distance from center.
    entities.sort(key=lambda t: (abs(t[2]) + abs(t[3]), t[0]))
    for ent in entities[:10]:
        kind, etype, dx, dy, gxy, extra = ent
        gtxt = f" global=({gxy[0]},{gxy[1]})" if gxy is not None else ""
        if kind == "node":
            etxt = "unknown" if extra is None or int(extra) < 0 or int(extra) == 10 else str(int(extra))
            lines.append(f"- node type={etype} energy={etxt} rel=({dx},{dy}){gtxt}")
        elif kind == "resource":
            lines.append(f"- resource type={etype} rel=({dx},{dy}){gtxt}")
        else:
            lines.append(f"- {kind} rel=({dx},{dy}){gtxt}")
    if scenario == "signal_hunt":
        lines.append("- NOTE: target_or_decoy markers may include decoys; disambiguate with clue constraints before final scan.")
    return lines


def _pipeline_goal_hint_lines(agent_obs: dict, info: dict, agent_id: int) -> list[str]:
    lines = []
    raw = agent_obs.get("goal_hint")
    if raw is None:
        return lines
    toks = [int(v) for v in raw.tolist()]
    if not toks or all(t < 0 for t in toks):
        return lines
    lines.append("Stage hint (partial blueprint for this agent):")
    chunk = 9
    for i in range(0, len(toks), chunk):
        seg = toks[i : i + chunk]
        if len(seg) < 9:
            continue
        if seg[0] < 0:
            continue
        stage_id = seg[0]
        sx = seg[1]
        sy = seg[2]
        req_n = max(0, seg[3])
        req = []
        if seg[4] >= 0:
            req.append(seg[4])
        if seg[5] >= 0:
            req.append(seg[5])
        dep_n = seg[6]
        deps = [seg[7]] if (dep_n > 0 and seg[7] >= 0) else []
        sync = int(seg[8]) if seg[8] >= 0 else 0
        lines.append(
            f"- stage={stage_id} station=({sx},{sy}) required={req if req else '[]'} deps={deps if deps else '[]'} sync={sync}"
        )
    return lines


def _decode_pipeline_goal_hint(agent_obs: dict) -> list[dict]:
    out = []
    raw = agent_obs.get("goal_hint")
    if raw is None:
        return out
    toks = [int(v) for v in raw.tolist()]
    if not toks or all(t < 0 for t in toks):
        return out
    chunk = 9
    for i in range(0, len(toks), chunk):
        seg = toks[i : i + chunk]
        if len(seg) < 9 or seg[0] < 0:
            continue
        req = []
        if seg[4] >= 0:
            req.append(seg[4])
        if seg[5] >= 0:
            req.append(seg[5])
        dep_n = seg[6]
        deps = [seg[7]] if (dep_n > 0 and seg[7] >= 0) else []
        out.append(
            {
                "stage": int(seg[0]),
                "station": (int(seg[1]), int(seg[2])),
                "required": req,
                "deps": deps,
                "sync": bool(int(seg[8]) > 0),
            }
        )
    out.sort(key=lambda s: s.get("stage", 999))
    return out


def _pipeline_progress_prompt_lines(agent_id: int, state: dict | None) -> list[str]:
    if not isinstance(state, dict):
        return []
    runtime = state.get("_agent_runtime", {})
    hist = runtime.get(int(agent_id), {})
    plan = hist.get("pipeline_plan", [])
    if not isinstance(plan, list) or not plan:
        return []
    cursor = int(hist.get("pipeline_stage_cursor", 0))
    cursor = max(0, min(cursor, len(plan) - 1))
    lines = ["Pipeline progress (best-known):"]
    done_ids = [int(s.get("stage", -1)) for s in plan if bool(s.get("done", False))]
    if done_ids:
        lines.append(f"- completed_stages={sorted(done_ids)}")
    else:
        lines.append("- completed_stages=[]")
    st = plan[cursor]
    sid = int(st.get("stage", -1))
    deps = [int(x) for x in st.get("deps", [])]
    deps_done = all(int(d) in set(done_ids) for d in deps) if deps else True
    need = st.get("need", {})
    delivered = st.get("delivered", {})
    outstanding = []
    for t, n in sorted(need.items()):
        rem = int(n) - int(delivered.get(int(t), 0))
        if rem > 0:
            outstanding.append((int(t), int(rem)))
    lines.append(
        f"- current_stage={sid} deps={deps if deps else []} deps_done={int(deps_done)} station={tuple(st.get('station', (0, 0)))} sync={int(bool(st.get('sync', False)))} sync_done={int(bool(st.get('sync_done', False)))}"
    )
    if outstanding:
        out_txt = ", ".join([f"type {t} x{n}" for t, n in outstanding])
        lines.append(f"- outstanding_requirements: {out_txt}")
    else:
        lines.append("- outstanding_requirements: none")
    lines.append(
        "Stage-closing rule: if outstanding_requirements is none, stop explore_for_parts and prioritize deliver_part_to_station or perform_stage_sync until this stage completes."
    )
    lines.append(
        "Dependency rule: if deps_done=0, do not force-close this stage yet; communicate blocker and help finish upstream stage."
    )
    return lines


def executor_prompt(obs: dict, info: dict, agent_id: int, state: dict | None = None) -> str:
    agent_obs = obs[agent_id]
    scenario = str(info.get("scenario", "unknown"))
    inventory = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
    center = center_tile_id(agent_obs)
    center_name = TILE_NAMES.get(center, "unknown")
    pos = agent_obs.get("self_pos")
    explore_ratio, frontier = exploration_summary(agent_obs)

    lead_line = "You are a high-level planner for one agent in a cooperative multi-agent POMDP."
    if scenario == "energy_grid":
        lead_line = (
            "You are the planner for an agent in a cooperative multi-agent POMDP energy grid game. "
            "The goal of the game is to find low-energy nodes and coordinate typed resource deliveries needed to recharge them before they run out."
        )
    elif scenario == "signal_hunt":
        lead_line = (
            "You are the planner for an agent in a cooperative multi-agent POMDP signal hunt game. "
            "The goal is to discover and fuse distributed clues, then coordinate a target confirmation scan."
        )
    elif scenario == "pipeline_assembly":
        lead_line = (
            "You are the planner for an agent in a cooperative multi-agent POMDP pipeline assembly game. "
            "The goal is to complete dependent assembly stages by collecting resources and coordinating station interactions."
        )

    if scenario == "signal_hunt":
        allowed_task_lines = [
            "- inspect_visible_clue",
            "- verify_reported_target",
            "- explore_for_clues",
            "- coordinate_scan",
            "- hold_position",
            "- respond_to_teammate",
        ]
        skill_lines = [
            "- inspect_visible_clue: interact when on clue/target evidence tiles.",
            "- verify_reported_target: move/prepare to validate target claims with teammates.",
            "- explore_for_clues: systematically reveal unknown regions to gather constraints.",
            "- coordinate_scan: synchronize confirmation timing with teammates.",
            "- hold_position: wait while preserving readiness for sync/updates.",
            "- respond_to_teammate: adjust plan from incoming clue/target messages.",
        ]
        comm_lines = [
            "Communication policy (message only on important change):",
            "- New clue/constraint: status=clue|target=region@x,y|why=new_constraint",
            "- Candidate target evidence: status=target|target=region@x,y|why=evidence",
            "- Confirmation/sync request: status=sync|target=scan@x,y|why=confirm",
            "- Completion/update: status=done|task=verify_reported_target|target=...|why=completed",
            "- Stale correction: status=update|target=...|why=correcting_stale_info",
            "- Do not repeat the same status+target unless changed or stale by age.",
        ]
    elif scenario == "pipeline_assembly":
        allowed_task_lines = [
            "- collect_required_part",
            "- deliver_part_to_station",
            "- perform_stage_sync",
            "- explore_for_parts",
            "- coordinate_stage_plan",
            "- hold_position",
            "- respond_to_teammate",
        ]
        skill_lines = [
            "- collect_required_part: pick up required components for pending stages.",
            "- deliver_part_to_station: carry component to assembly station for current stage.",
            "- perform_stage_sync: coordinate simultaneous station interactions when required.",
            "- explore_for_parts: reveal unknown rooms to locate missing components.",
            "- coordinate_stage_plan: align stage ordering and assignments with teammates.",
            "- hold_position: wait while preserving readiness for sync/updates.",
            "- respond_to_teammate: adjust task from incoming stage/resource messages.",
        ]
        comm_lines = [
            "Communication policy (message only on important change):",
            "- New part found: status=part|target=type:t@x,y|why=found",
            "- Stage assignment/start: status=start|task=...|target=stage:s|why=assignment",
            "- Dependency/blocker update: status=update|target=stage:s|why=dependency_or_blocker",
            "- Sync rendezvous: status=start|task=perform_stage_sync|target=stage:s|why=rendezvous",
            "- Sync ready/execute: status=update|task=perform_stage_sync|target=stage:s|why=waiting_peer_or_sync_now",
            "- Stage completion: status=done|task=...|target=stage:s|why=completed",
            "- Do not repeat the same status+target unless changed or stale by age.",
        ]
    else:
        allowed_task_lines = [
            "- pickup_visible_resource",
            "- deliver_to_matching_node",
            "- sync_interact",
            "- explore_sector",
            "- hold_position",
            "- respond_to_teammate",
        ]
        skill_lines = [
            "- pickup_visible_resource: acquire a visible/known resource when inventory is empty.",
            "- deliver_to_matching_node: carry current resource type to a matching node and recharge.",
            "- sync_interact: coordinate timing for multi-agent interact when sync is needed.",
            "- explore_sector: systematically reveal unknown areas in assigned sector.",
            "- hold_position: wait while preserving readiness for sync/updates.",
            "- respond_to_teammate: adjust task based on incoming coordination message.",
        ]
        comm_lines = [
            "Communication policy (message only on important change):",
            "- New node detection: status=node|target=node:t@x,y|why=found",
            "- New resource detection: status=resource|target=type:t@x,y|why=found",
            "- Task start/change: status=start|task=...|target=...|why=assignment",
            "- Low-energy alert: status=alert|target=node:t@x,y|why=critical_or_low",
            "- Completion/update: status=done|task=...|target=...|why=completed",
            "- Stale correction: status=update|target=...|why=correcting_stale_info",
            "- Do not repeat same status+target unless changed or stale by age.",
        ]

    prompt = [
        lead_line,
        "Do not output primitive movement actions. The low-level executor handles navigation and interactions.",
        "Return only JSON with keys: task, task_plan, message_text.",
        "JSON schema: {\"task\": string, \"task_plan\": [up to 5 task names], \"message_text\": string}.",
        "Allowed task names:",
        *allowed_task_lines,
        "Skill semantics (executor handles pathing):",
        *skill_lines,
        "message_text is your decision variable for whether this agent should send a message to others this step.",
        *comm_lines,
        f"Scenario: {scenario}",
        f"Inventory: {inventory}",
        "Inventory semantics: 0 means empty. If >0, that number is the carried resource type id.",
        f"Center tile: {center_name} ({center})",
        f"Exploration coverage: {explore_ratio:.2f}",
        f"Suggested exploration direction: {frontier}",
    ]
    if isinstance(state, dict):
        hist = state.get("_agent_runtime", {}).get(int(agent_id), {})
        inv_evt = hist.get("last_inventory_event")
        if inv_evt:
            prompt.append(f"Recent inventory event: {inv_evt}")
    if pos is not None:
        prompt.append(f"Self position (global coords): ({int(pos[0])},{int(pos[1])})")

    if scenario == "energy_grid":
        recharge_count = info.get("recharge_count")
        success_recharges = info.get("success_recharges")
        if recharge_count is not None and success_recharges is not None:
            prompt.append(f"Mission progress: recharges={int(recharge_count)}/{int(success_recharges)}")
        prompt.extend(
            [
                "Energy dynamics:",
                "- Nodes drain periodically over time.",
                "- Valid typed delivery recharges immediately.",
                "- Low-energy nodes may require synchronized interact by 2 agents.",
                "- Fail if any node depletes before success threshold.",
                "Energy policy priority:",
                "1) If inventory empty -> pickup_visible_resource.",
                "2) If carrying type t -> deliver_to_matching_node.",
                "3) If sync likely/needed -> sync_interact.",
                "4) Else explore_sector.",
            ]
        )
    elif scenario == "signal_hunt":
        prompt.extend(
            [
                "Signal Hunt policy priority:",
                "1) If standing on clue/target relevant tile -> inspect_visible_clue.",
                "2) If teammate reports high-value clue/target evidence -> respond_to_teammate.",
                "3) If unexplored frontiers exist -> explore_for_clues.",
                "4) If coordinated confirmation needed -> coordinate_scan.",
                "5) Otherwise hold_position.",
                "Communication focus:",
                "- Broadcast only new clue constraints, target evidence, or confirmation state changes.",
                "- Prefer compact messages: status=clue|target=...|why=new_constraint",
                "Decoy rule: visible target-like markers can be decoys; only finalize verification after clue constraints agree.",
            ]
        )
    elif scenario == "pipeline_assembly":
        prompt.extend(
            [
                "Pipeline Assembly policy priority:",
                "Stage dependencies: stages must be completed in order of dependency; blocked stages require upstream completion first.",
                "Interpretation rule: if Inventory > 0, you are already carrying a part type and should not choose collect_required_part unless the carry was just cleared.",
                "1) If inventory empty (Inventory=0) -> collect_required_part for earliest unblocked stage.",
                "2) If Inventory>0 and type is needed by current unblocked stage -> deliver_part_to_station.",
                "3) If Inventory>0 but type is not needed -> coordinate_stage_plan or explore_for_parts (do not spam collect_required_part).",
                "4) If stage requires synchronized interaction -> perform_stage_sync.",
                "5) If no required part/station visible -> explore_for_parts.",
                "6) Otherwise hold_position.",
                "Task-switch rule: after Inventory changes from 0 to >0, next task should usually be deliver_part_to_station unless dependency is blocked.",
                "Anti-drift rule: avoid repeating collect_required_part across replans while Inventory>0.",
                "Stage-closure rule: once current stage has no outstanding requirements, stop explore_for_parts and close this stage before starting another stage.",
                "Sync protocol for sync=1 stages:",
                "a) Move to the current stage station and announce rendezvous.",
                "b) If teammate not yet confirmed, hold_position and send waiting_peer update.",
                "c) When both agents are ready at the same station, choose perform_stage_sync and send sync_now update.",
                "d) After sync completion, send done message and switch to next unblocked stage.",
                "Communication focus:",
                "- Broadcast stage dependencies, part finds, and stage progress only.",
                "- Prefer compact messages: status=start|task=...|target=stage:s|why=stage_progress",
            ]
        )

    msgs = info.get("messages_with_sender", {}).get(agent_id, [])
    if msgs:
        prompt.append("Incoming messages with sender:")
        for item in msgs[-8:]:
            txt = str(item.get("text", "")).strip()
            if txt:
                prompt.append(f"agent_{int(item.get('from', -1))}: {txt}")

    if scenario == "signal_hunt":
        hint_txt = info.get("goal_hint_texts", {}).get(agent_id)
        if hint_txt:
            prompt.append(f"Agent clue hint: {hint_txt}")
    if scenario == "pipeline_assembly":
        prompt.extend(_pipeline_goal_hint_lines(agent_obs, info, agent_id))
        prompt.extend(_pipeline_progress_prompt_lines(agent_id, state))

    prompt.append("Local tile view:")
    prompt.append(grid_to_ascii(agent_obs["local_grid"]))
    if scenario == "energy_grid":
        prompt.extend(energy_local_summary(agent_obs))
    sem_lines = semantic_prompt_lines(agent_id, state, "energy_grid" if scenario == "energy_grid" else scenario)
    if sem_lines:
        prompt.extend(sem_lines)
    else:
        prompt.append("Semantic memory: none yet (will populate as entities are observed or reported).")
    prompt.extend(memory_prompt_lines(agent_id, state))
    return "\n".join(prompt)


class LLMExecutorPolicy(LLMPolicy):
    """
    LLM planner outputs high-level task + optional action hints.
    A deterministic executor converts task into low-level env action each step.
    """

    def __init__(self, llm_call, prompt_fn=None, default_action: int = 4, postprocess=None, cache=None, batch_fn=None):
        super().__init__(
            llm_call=llm_call,
            prompt_fn=prompt_fn or executor_prompt,
            default_action=default_action,
            postprocess=postprocess,
            cache=cache,
            batch_fn=batch_fn,
        )

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        if isinstance(state, dict):
            state["_scenario"] = info.get("scenario", "")
            state["_num_agents"] = len(obs)
            refresh_team_commitments(info, state)
            state["task_events"] = []
        scenario = str(state.get("_scenario", "")) if isinstance(state, dict) else str(info.get("scenario", ""))

        def incoming_sig(agent_id: int) -> tuple[str, ...]:
            items = info.get("messages_with_sender", {}).get(agent_id, [])
            sig = []
            for m in items[-8:]:
                sender = int(m.get("from", -1))
                txt = str(m.get("text", "")).strip()
                if txt:
                    sig.append(f"{sender}:{txt}")
            return tuple(sig)

        def _message_replan_signal(agent_id: int, current_task: str) -> tuple[bool, str]:
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            items = info.get("messages_with_sender", {}).get(agent_id, [])
            if not items:
                return False, "none"
            pipeline_stage_id = None
            pipeline_relevant_stages = set()
            if scenario == "pipeline_assembly":
                plan = hist.get("pipeline_plan", [])
                if isinstance(plan, list) and plan:
                    cur = int(hist.get("pipeline_stage_cursor", 0))
                    cur = max(0, min(cur, len(plan) - 1))
                    cur_stage = plan[cur]
                    pipeline_stage_id = int(cur_stage.get("stage", -1))
                    pipeline_relevant_stages.add(pipeline_stage_id)
                    for d in cur_stage.get("deps", []):
                        pipeline_relevant_stages.add(int(d))
            now = int(state.get("step", 0)) if isinstance(state, dict) else 0
            seen = hist.setdefault("seen_message_keys", set())
            if not isinstance(seen, set):
                seen = set(seen)
                hist["seen_message_keys"] = seen
            last_msg_replan_step = int(hist.get("last_msg_replan_step", -10_000))
            cooldown = 3
            urgent_status = {"alert", "assist", "sync", "help"}
            material_status = {"alert", "assist", "sync", "help", "node", "resource", "update", "done"}
            if scenario == "signal_hunt":
                material_status |= {"clue", "target"}

            def _normalize_target_field(raw: str) -> str:
                txt2 = str(raw or "").strip().lower()
                if ("=" in txt2) and ("|" in txt2):
                    kv2 = parse_kv_message(txt2)
                    tgt2 = str(kv2.get("target", "")).strip().lower()
                    if tgt2:
                        return tgt2
                return txt2

            for m in items[-8:]:
                sender = int(m.get("from", -1))
                txt = str(m.get("text", "")).strip()
                if not txt:
                    continue
                kv = parse_kv_message(txt)
                status = str(kv.get("status", "")).strip().lower()
                target = str(kv.get("target", "")).strip().lower()
                task = _canonical_executor_task(str(kv.get("task", "")).strip().lower())
                if scenario == "pipeline_assembly" and status in {"start", "update", "done", "assist", "sync"}:
                    sid = _parse_stage_target(target)
                    if sid is not None and sid not in pipeline_relevant_stages:
                        continue
                key = f"{sender}|{status}|{target}|{task}"
                is_new = key not in seen
                if is_new:
                    seen.add(key)
                # Replan immediately for urgent events.
                if status in urgent_status and (is_new or (now - last_msg_replan_step) >= 1):
                    hist["last_msg_replan_step"] = now
                    return True, "urgent_message"
                # Only material + new updates can trigger replan, and not too frequently.
                eff_cooldown = 1 if scenario == "signal_hunt" else cooldown
                if status in material_status and is_new and (now - last_msg_replan_step) >= eff_cooldown:
                    hist["last_msg_replan_step"] = now
                    return True, "new_message"
                # "start" is usually noisy; only replan on same task family with a different target claim.
                current_target = _normalize_target_field(hist.get("plan_ctx", {}).get("target", ""))
                if (
                    status == "start"
                    and is_new
                    and task
                    and current_task
                    and task == current_task
                    and target
                    and current_target
                    and (now - last_msg_replan_step) >= cooldown
                ):
                    # Parallel assignments should not be treated as conflicts.
                    incoming_stage = _parse_stage_target(target)
                    current_stage = _parse_stage_target(current_target)
                    if incoming_stage is not None and current_stage is not None and incoming_stage != current_stage:
                        continue
                    incoming_t = _parse_node_target(target)
                    current_t = _parse_node_target(current_target)
                    if incoming_t is not None and current_t is not None and incoming_t != current_t:
                        continue
                    if target == current_target:
                        hist["last_msg_replan_step"] = now
                        return True, "task_conflict"
                    # For unknown target formats, only treat as conflict when one string contains the other.
                    if (target in current_target) or (current_target in target):
                        hist["last_msg_replan_step"] = now
                        return True, "task_conflict"
                    continue
            return False, "none"

        def _event_replan_signal(agent_id: int) -> tuple[bool, str]:
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            items = info.get("events", {}).get(agent_id, [])
            if not items:
                return False, "none"
            now = int(state.get("step", 0)) if isinstance(state, dict) else 0
            seen = hist.setdefault("seen_event_keys", set())
            if not isinstance(seen, set):
                seen = set(seen)
                hist["seen_event_keys"] = seen
            last_ev_step = int(hist.get("last_event_replan_step", -10_000))
            cooldown = 4
            for ev in items:
                ev_name = str((ev or {}).get("event", "")).strip().lower()
                key = f"{ev_name}|{str(ev)}"
                is_new = key not in seen
                if is_new:
                    seen.add(key)
                if not is_new:
                    continue
                # Scenario-specific material events only.
                if scenario == "signal_hunt":
                    # Signal Hunt benefits from fast adaptation to local evidence/penalties.
                    if ev_name not in {"clue_found", "decoy_scan"}:
                        continue
                elif scenario == "pipeline_assembly":
                    if ev_name not in {"delivered", "sync_complete"}:
                        continue
                elif scenario == "energy_grid":
                    if ev_name not in {"recharged"}:
                        continue
                # Keep signal-hunt event replanning responsive; others stay conservative.
                eff_cooldown = 1 if scenario == "signal_hunt" else cooldown
                if (now - last_ev_step) < eff_cooldown:
                    continue
                hist["last_event_replan_step"] = now
                return True, "local_event"
            return False, "none"

        def _is_center_actionable_change(agent_obs: dict, old_center: int, current_task: str) -> bool:
            center = center_tile_id(agent_obs)
            if center == old_center:
                return False
            # Avoid broad churn: only replan on changes relevant to current task and scenario.
            if scenario == "signal_hunt":
                # React mainly to clue/target evidence, not every landmark transition.
                if current_task in {"sync_interact", "respond_to_teammate"}:
                    return center in {5, 6}
                if current_task in {"explore_sector"}:
                    return center in {5}
                return center in {5, 6}
            if scenario == "pipeline_assembly":
                inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
                if current_task in {"pickup_visible_resource"} and inv == 0:
                    return center == 2
                if current_task in {"deliver_to_matching_node", "sync_interact"} and inv != 0:
                    return center == 3
                return False
            if scenario == "energy_grid":
                if current_task in {"pickup_visible_resource"}:
                    return center == 2
                if current_task in {"deliver_to_matching_node", "sync_interact"}:
                    return center == 4
                return center in {2, 4}
            return center in {2, 3, 4, 5, 6}

        def should_replan(agent_id: int, agent_obs: dict) -> tuple[bool, str]:
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            task = hist.get("executor_task")
            current_task = _canonical_executor_task((task or {}).get("task", ""))
            ctx = hist.get("plan_ctx", {})
            if task is None:
                return True, "no_task"
            if int(hist.get("stuck_steps", 0)) >= 3:
                active = hist.get("active_task_run") or {}
                trend = list(active.get("distance_trend", []))
                non_improving = False
                if len(trend) >= 4:
                    tail = trend[-4:]
                    # Non-improving if distance never decreases in the recent window.
                    non_improving = all(tail[i] >= tail[i - 1] for i in range(1, len(tail)))
                # If we lack distance signal, trust movement-failure criterion alone.
                if non_improving or not trend:
                    return True, "stuck"
            ev_replan, ev_reason = _event_replan_signal(agent_id)
            if ev_replan:
                return True, ev_reason
            sig = incoming_sig(agent_id)
            if sig and sig != tuple(ctx.get("incoming_sig", ())):
                msg_replan, msg_reason = _message_replan_signal(agent_id, current_task)
                if msg_replan:
                    return True, msg_reason
            center = center_tile_id(agent_obs)
            old_center = int(ctx.get("center_tile", -1))
            if _is_center_actionable_change(agent_obs, old_center, current_task):
                if scenario == "pipeline_assembly":
                    now = int(state.get("step", 0))
                    last_center = int(hist.get("last_center_replan_step", -10_000))
                    # Don't churn on center changes while executing delivery/sync lanes.
                    if current_task in {"deliver_to_matching_node", "sync_interact"}:
                        pass
                    elif (now - last_center) >= 6:
                        hist["last_center_replan_step"] = now
                        return True, "new_center_object"
                else:
                    return True, "new_center_object"
            inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
            if int(ctx.get("inventory", inv)) != inv:
                return True, "inventory_change"
            timeout_limit = 8
            if scenario == "pipeline_assembly" and current_task in {"deliver_to_matching_node", "sync_interact"}:
                timeout_limit = 14
            if int(state.get("step", 0)) - int(ctx.get("step", 0)) >= timeout_limit:
                return True, "task_timeout"
            for token in sig:
                low = token.lower()
                if ("sync" in low) or ("assist" in low) or ("help" in low):
                    return True, "coord_request"
            return False, "continue_task"

        def save_plan_ctx(agent_id: int, agent_obs: dict, task: dict | None = None):
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
            target = ""
            if isinstance(task, dict):
                target = str(task.get("target", "") or "")
                if not target:
                    target = str(task.get("message_text", "") or "")
            hist["plan_ctx"] = {
                "step": int(state.get("step", 0)) if isinstance(state, dict) else 0,
                "incoming_sig": incoming_sig(agent_id),
                "center_tile": int(center_tile_id(agent_obs)),
                "inventory": inv,
                "target": target,
            }

        def _pipeline_plan_init(agent_id: int, agent_obs: dict):
            if scenario != "pipeline_assembly":
                return
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            if hist.get("pipeline_plan_initialized"):
                return
            hints = _decode_pipeline_goal_hint(agent_obs)
            plan = []
            for h in hints:
                need = {}
                for t in h.get("required", []):
                    need[int(t)] = int(need.get(int(t), 0)) + 1
                plan.append(
                    {
                        "stage": int(h.get("stage", -1)),
                        "station": tuple(h.get("station", (0, 0))),
                        "need": need,
                        "delivered": {},
                        "sync": bool(h.get("sync", False)),
                        "sync_done": False,
                        "deps": [int(x) for x in h.get("deps", []) if int(x) >= 0],
                        "done": False,
                    }
                )
            hist["pipeline_plan"] = plan
            hist["pipeline_plan_initialized"] = True
            hist["pipeline_stage_cursor"] = int(hist.get("pipeline_stage_cursor", 0))

        def _pipeline_update_progress(agent_id: int, agent_obs: dict):
            if scenario != "pipeline_assembly":
                return
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            _pipeline_plan_init(agent_id, agent_obs)
            plan = hist.get("pipeline_plan", [])
            if not plan:
                return

            by_stage = {int(s["stage"]): s for s in plan}
            for ev in info.get("events", {}).get(agent_id, []):
                evn = str((ev or {}).get("event", "")).strip().lower()
                sid = int((ev or {}).get("stage", -1))
                if sid not in by_stage:
                    continue
                st = by_stage[sid]
                if evn == "delivered":
                    rtype = int((ev or {}).get("resource_type", 0))
                    if rtype <= 0:
                        rtype = int(hist.get("task_last_inventory_type", 0))
                    if rtype > 0:
                        st["delivered"][rtype] = int(st["delivered"].get(rtype, 0)) + 1
                elif evn == "sync_complete":
                    st["sync_done"] = True

            incoming = info.get("messages_with_sender", {}).get(agent_id, [])
            for m in incoming[-8:]:
                kv = parse_kv_message(str(m.get("text", "")).strip())
                if str(kv.get("status", "")).strip().lower() != "done":
                    continue
                sid = _parse_stage_target(str(kv.get("target", "")))
                if sid is None or sid not in by_stage:
                    continue
                by_stage[sid]["done"] = True

            done_stage_ids = {int(s["stage"]) for s in plan if bool(s.get("done", False))}
            for st in plan:
                if st.get("done"):
                    continue
                deps = [int(x) for x in st.get("deps", [])]
                if any(d not in done_stage_ids for d in deps):
                    continue
                need = st.get("need", {})
                delivered = st.get("delivered", {})
                ready = True
                for t, n in need.items():
                    if int(delivered.get(int(t), 0)) < int(n):
                        ready = False
                        break
                if ready and (not st.get("sync", False) or st.get("sync_done", False)):
                    st["done"] = True
                    done_stage_ids.add(int(st["stage"]))

            for i, st in enumerate(plan):
                if not bool(st.get("done", False)):
                    hist["pipeline_stage_cursor"] = int(i)
                    return

        def task_done(agent_obs: dict, task_name: str) -> bool:
            task_name = _canonical_executor_task(task_name)
            inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
            tile = center_tile_id(agent_obs)
            if task_name in {"pickup_visible_resource", "pickup_visible"}:
                return inv != 0
            if task_name in {"deliver_to_matching_node", "deliver_visible"}:
                return inv == 0
            if task_name == "sync_interact":
                return tile in {3, 4, 5, 6}
            return False

        def task_distance(agent_obs: dict, task_name: str) -> int | None:
            task_name = _canonical_executor_task(task_name)
            local = agent_obs.get("local_grid")
            if local is None:
                return None
            if task_name in {"pickup_visible_resource", "pickup_visible"}:
                res = agent_obs.get("local_resource_types")
                if res is None:
                    return None
                best = _nearest_local_pos(res, lambda v: v > 0 and v != 10)
                return int(best[0]) if best is not None else None
            if task_name in {"deliver_to_matching_node", "deliver_visible"}:
                inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
                if inv == 0:
                    return None
                nodes = agent_obs.get("local_node_types")
                if nodes is None:
                    return None
                best = _nearest_local_pos(nodes, lambda v: v == inv)
                return int(best[0]) if best is not None else None
            if task_name == "explore_sector":
                _, frontier = exploration_summary(agent_obs)
                return 0 if frontier in {"up", "down", "left", "right"} else None
            return None

        def finalize_task(agent_id: int, outcome: str, reason: str, agent_obs: dict):
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            active = hist.get("active_task_run")
            if not active:
                return
            step_now = int(state.get("step", 0)) if isinstance(state, dict) else 0
            inv_now = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
            summary = {
                "task": active.get("task"),
                "task_start_step": int(active.get("task_start_step", step_now)),
                "task_end_step": step_now,
                "task_outcome": outcome,
                "reason": reason,
                "steps": int(step_now - int(active.get("task_start_step", step_now)) + 1),
                "progress": {
                    "distance_trend": list(active.get("distance_trend", []))[-12:],
                    "inventory_change": int(inv_now != int(active.get("start_inventory", inv_now))),
                    "interact_success": int(active.get("interact_success", 0)),
                },
            }
            hist.setdefault("task_history", []).append(summary)
            hist["active_task_run"] = None
            if isinstance(state, dict):
                state.setdefault("task_events", []).append({"agent_id": int(agent_id), **summary})

        def start_task(agent_id: int, task: dict, reason: str, agent_obs: dict):
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            task_name = _canonical_executor_task(task.get("task", "hold_position"))
            step_now = int(state.get("step", 0)) if isinstance(state, dict) else 0
            inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
            hist["active_task_run"] = {
                "task": task_name,
                "task_start_step": step_now,
                "start_reason": reason,
                "start_inventory": inv,
                "distance_trend": [],
                "interact_success": 0,
            }
            if isinstance(state, dict):
                state.setdefault("task_events", []).append(
                    {
                        "agent_id": int(agent_id),
                        "task": task_name,
                        "task_start_step": step_now,
                        "task_outcome": "started",
                        "reason": reason,
                    }
                )

        def update_task_progress(agent_id: int, agent_obs: dict):
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
            active = hist.get("active_task_run")
            if active:
                tname = str(active.get("task", "hold_position"))
                dist = task_distance(agent_obs, tname)
                if dist is not None:
                    trend = active.setdefault("distance_trend", [])
                    trend.append(int(dist))
                    if len(trend) > 20:
                        del trend[:-20]
                prev_inv = int(hist.get("task_last_inventory", inv))
                last_act = hist.get("last_issued_action")
                if last_act == ACTION_NAMES["interact"] and prev_inv > inv:
                    active["interact_success"] = int(active.get("interact_success", 0)) + 1
            hist["task_last_inventory"] = inv
            hist["task_last_inventory_type"] = inv

        def maybe_advance_task(agent_obs: dict, task: dict) -> tuple[dict, bool]:
            name = _canonical_executor_task(task.get("task", "hold_position"))
            q = task.get("task_plan", [])
            if not isinstance(q, list):
                q = []
            done = task_done(agent_obs, name)
            if done and q:
                nxt = str(q.pop(0)).strip()
                task["task"] = _canonical_executor_task(nxt if nxt else "hold_position")
                task["task_plan"] = q
            return task, bool(done)

        def execute_task(agent_id: int, agent_obs: dict, task: dict) -> dict:
            task_name = _canonical_executor_task(task.get("task", "hold_position"))
            msg = task.get("message_text")
            allowed = set(allowed_actions(agent_obs))
            tile = center_tile_id(agent_obs)
            inv = int(agent_obs.get("inventory", [0])[0]) if "inventory" in agent_obs else 0
            local = agent_obs.get("local_grid")
            self_pos = agent_obs.get("self_pos")
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})

            def _move_toward_global(gx: int, gy: int) -> int:
                if self_pos is None:
                    for m in ["up", "down", "left", "right", "stay"]:
                        if m in allowed:
                            return ACTION_NAMES[m]
                    return ACTION_NAMES["stay"]
                sx, sy = int(self_pos[0]), int(self_pos[1])
                dx = int(gx - sx)
                dy = int(gy - sy)
                prefs = []
                if abs(dx) >= abs(dy):
                    if dx < 0:
                        prefs.append("left")
                    elif dx > 0:
                        prefs.append("right")
                    if dy < 0:
                        prefs.append("up")
                    elif dy > 0:
                        prefs.append("down")
                else:
                    if dy < 0:
                        prefs.append("up")
                    elif dy > 0:
                        prefs.append("down")
                    if dx < 0:
                        prefs.append("left")
                    elif dx > 0:
                        prefs.append("right")
                prefs.extend(["up", "down", "left", "right", "stay"])
                for p in prefs:
                    if p in allowed:
                        return ACTION_NAMES[p]
                return ACTION_NAMES["stay"]

            # Generic waypoint/detour fallback to break repeated local loops.
            if int(hist.get("stuck_steps", 0)) >= 4 and int(hist.get("detour_steps", 0)) <= 0:
                hist["detour_steps"] = 2
            detour_steps = int(hist.get("detour_steps", 0))
            if detour_steps > 0:
                hist["detour_steps"] = detour_steps - 1
                _, frontier = exploration_summary(agent_obs)
                if frontier in allowed:
                    return {"action": ACTION_NAMES[frontier], "message_text": msg}
                for m in ["up", "down", "left", "right", "stay"]:
                    if m in allowed:
                        return {"action": ACTION_NAMES[m], "message_text": msg}

            if scenario == "pipeline_assembly":
                _pipeline_plan_init(agent_id, agent_obs)
                plan = hist.get("pipeline_plan", [])
                stage = None
                stage_id = -1
                if plan:
                    cursor = int(hist.get("pipeline_stage_cursor", 0))
                    cursor = max(0, min(cursor, len(plan) - 1))
                    stage = plan[cursor]
                    stage_id = int(stage.get("stage", -1))
                req_types = set((stage or {}).get("need", {}).keys())
                station = (stage or {}).get("station")
                res = agent_obs.get("local_resource_types")
                need = (stage or {}).get("need", {})
                delivered = (stage or {}).get("delivered", {})
                needs_more = False
                for t, n in need.items():
                    if int(delivered.get(int(t), 0)) < int(n):
                        needs_more = True
                        break
                sync_required = bool((stage or {}).get("sync", False))
                sync_done = bool((stage or {}).get("sync_done", False))
                sync_ready = bool(stage) and (not needs_more) and sync_required and (not sync_done)

                incoming = info.get("messages_with_sender", {}).get(agent_id, [])
                teammate_sync_intent = False
                for item in incoming[-8:]:
                    kv = parse_kv_message(str(item.get("text", "")).strip())
                    tgt_stage = _parse_stage_target(str(kv.get("target", "")))
                    if tgt_stage != stage_id:
                        continue
                    status = str(kv.get("status", "")).strip().lower()
                    task_hint = _canonical_executor_task(str(kv.get("task", "")).strip())
                    if status in {"start", "update", "assist"} and task_hint in {
                        "sync_interact",
                        "deliver_to_matching_node",
                        "respond_to_teammate",
                    }:
                        teammate_sync_intent = True
                        break

                # Anti-loop detour: after repeated movement failures, explore briefly
                # before retrying direct station/resource approach.
                if int(hist.get("stuck_steps", 0)) >= 3 and int(hist.get("detour_steps", 0)) <= 0:
                    hist["detour_steps"] = 3

                # Sync stages require two agents interacting on the same station in the same step.
                # Use explicit rendezvous + wait + synchronized interact behavior.
                if sync_ready and station is not None:
                    sync_msg = f"status=start|task=perform_stage_sync|target=stage:{stage_id}|why=rendezvous"
                    if tile != 3:
                        hist["sync_wait_steps"] = 0
                        return {"action": _move_toward_global(int(station[0]), int(station[1])), "message_text": sync_msg}

                    wait_steps = int(hist.get("sync_wait_steps", 0)) + 1
                    hist["sync_wait_steps"] = wait_steps
                    step_now = int(state.get("step", 0)) if isinstance(state, dict) else 0
                    # Deterministic pulse keeps both agents aligned once they reach station.
                    pulse_interact = ((step_now + max(stage_id, 0)) % 2) == 0
                    if teammate_sync_intent or wait_steps >= 2:
                        pulse_interact = True
                    if pulse_interact and "interact" in allowed:
                        msg_now = f"status=update|task=perform_stage_sync|target=stage:{stage_id}|why=sync_now"
                        return {"action": ACTION_NAMES["interact"], "message_text": msg_now}
                    msg_wait = f"status=update|task=perform_stage_sync|target=stage:{stage_id}|why=waiting_peer"
                    return {"action": ACTION_NAMES["stay"], "message_text": msg_wait}
                hist["sync_wait_steps"] = 0

                if task_name == "pickup_visible_resource":
                    if tile == 2 and inv == 0 and "pickup" in allowed:
                        if res is not None:
                            h, w = res.shape
                            cy, cx = h // 2, w // 2
                            ctype = int(res[cy, cx])
                            if (not req_types) or (ctype in req_types):
                                return {"action": ACTION_NAMES["pickup"], "message_text": msg}
                        else:
                            return {"action": ACTION_NAMES["pickup"], "message_text": msg}
                    if res is not None and local is not None:
                        if req_types:
                            best = _nearest_local_pos(res, lambda v: v in req_types)
                        else:
                            best = _nearest_local_pos(res, lambda v: v > 0 and v != 10)
                        if best is not None:
                            _, x, y, _ = best
                            return {"action": _move_toward_local(x, y, local, allowed), "message_text": msg}
                    # no useful part visible: explore, do not camp station with empty inventory
                    _, frontier = exploration_summary(agent_obs)
                    if frontier in allowed:
                        return {"action": ACTION_NAMES[frontier], "message_text": msg}
                elif task_name == "deliver_to_matching_node":
                    if tile == 3 and inv != 0 and "interact" in allowed:
                        return {"action": ACTION_NAMES["interact"], "message_text": msg}
                    if station is not None:
                        return {"action": _move_toward_global(int(station[0]), int(station[1])), "message_text": msg}
                elif task_name == "sync_interact":
                    if tile == 3 and "interact" in allowed:
                        return {"action": ACTION_NAMES["interact"], "message_text": msg}
                    if station is not None:
                        return {"action": _move_toward_global(int(station[0]), int(station[1])), "message_text": msg}
                elif task_name in {"respond_to_teammate", "explore_sector"}:
                    _, frontier = exploration_summary(agent_obs)
                    if frontier in allowed:
                        return {"action": ACTION_NAMES[frontier], "message_text": msg}
                if "stay" in allowed:
                    return {"action": ACTION_NAMES["stay"], "message_text": msg}

            if scenario == "signal_hunt":
                land = state.get("_agent_semantic", {}).get(int(agent_id), {}).get("landmarks", {}) if isinstance(state, dict) else {}
                # Pull likely target from message_text when present.
                target_pos = _parse_region_target(str(task.get("message_text", "") or ""))
                if target_pos is None:
                    # fallback to nearest known target/decoy marker
                    tars = [pos for pos, meta in land.items() if str(meta.get("kind", "")) == "target_or_decoy"]
                    if tars and self_pos is not None:
                        sx, sy = int(self_pos[0]), int(self_pos[1])
                        target_pos = min(tars, key=lambda p: abs(int(p[0]) - sx) + abs(int(p[1]) - sy))
                if task_name == "sync_interact":
                    if tile in {5, 6} and "interact" in allowed:
                        return {"action": ACTION_NAMES["interact"], "message_text": msg}
                    if target_pos is not None:
                        return {"action": _move_toward_global(int(target_pos[0]), int(target_pos[1])), "message_text": msg}
                elif task_name == "respond_to_teammate":
                    if target_pos is not None:
                        return {"action": _move_toward_global(int(target_pos[0]), int(target_pos[1])), "message_text": msg}
                    _, frontier = exploration_summary(agent_obs)
                    if frontier in allowed:
                        return {"action": ACTION_NAMES[frontier], "message_text": msg}

            if task_name == "pickup_visible_resource":
                if tile == 2 and inv == 0 and "pickup" in allowed:
                    return {"action": ACTION_NAMES["pickup"], "message_text": msg}
                res = agent_obs.get("local_resource_types")
                if res is not None and local is not None:
                    best = _nearest_local_pos(res, lambda v: v > 0 and v != 10)
                    if best is not None:
                        _, x, y, _ = best
                        return {"action": _move_toward_local(x, y, local, allowed), "message_text": msg}
            elif task_name == "deliver_to_matching_node":
                if tile == 4 and inv != 0 and "interact" in allowed:
                    nodes = agent_obs.get("local_node_types")
                    if nodes is not None:
                        h, w = nodes.shape
                        cy, cx = h // 2, w // 2
                        if int(nodes[cy, cx]) == inv:
                            return {"action": ACTION_NAMES["interact"], "message_text": msg}
                nodes = agent_obs.get("local_node_types")
                if nodes is not None and local is not None and inv != 0:
                    best = _nearest_local_pos(nodes, lambda v: v == inv)
                    if best is not None:
                        _, x, y, _ = best
                        return {"action": _move_toward_local(x, y, local, allowed), "message_text": msg}
            elif task_name == "sync_interact":
                if "interact" in allowed and tile in {3, 4, 5, 6}:
                    return {"action": ACTION_NAMES["interact"], "message_text": msg}
                return {"action": ACTION_NAMES["stay"], "message_text": msg}
            elif task_name == "explore_sector":
                _, frontier = exploration_summary(agent_obs)
                if frontier in allowed:
                    return {"action": ACTION_NAMES[frontier], "message_text": msg}
                for m in ["up", "down", "left", "right", "stay"]:
                    if m in allowed:
                        return {"action": ACTION_NAMES[m], "message_text": msg}
            elif task_name == "respond_to_teammate":
                # Simple coordination fallback: stay/interact if possible, otherwise explore frontier.
                if "interact" in allowed and tile in {3, 4, 5, 6}:
                    return {"action": ACTION_NAMES["interact"], "message_text": msg}
                _, frontier = exploration_summary(agent_obs)
                if frontier in allowed:
                    return {"action": ACTION_NAMES[frontier], "message_text": msg}

            # keep / wait / fallback
            if "stay" in allowed:
                return {"action": ACTION_NAMES["stay"], "message_text": msg}
            for m in ["up", "down", "left", "right"]:
                if m in allowed:
                    return {"action": ACTION_NAMES[m], "message_text": msg}
            return {"action": self.default_action, "message_text": msg}

        prompts = []
        order = []
        for agent_id in obs.keys():
            update_semantic_memory_from_obs(obs[agent_id], state, agent_id)
            update_semantic_memory_from_messages(info, state, agent_id)
            update_agent_memory(obs[agent_id], info, agent_id, state)
            update_task_progress(agent_id, obs[agent_id])
            _pipeline_update_progress(agent_id, obs[agent_id])
            runtime = state.setdefault("_agent_runtime", {})
            hist = runtime.setdefault(int(agent_id), {})
            replan, reason = should_replan(agent_id, obs[agent_id])
            hist["pending_replan_reason"] = reason
            if not replan:
                task = hist.get("executor_task", {"task": "keep"})
                prev_name = _canonical_executor_task(task.get("task", "hold_position"))
                task, done_prev = maybe_advance_task(obs[agent_id], task)
                hist["executor_task"] = task
                active = hist.get("active_task_run")
                if done_prev:
                    finalize_task(agent_id, "success", "predicate_met", obs[agent_id])
                new_name = _canonical_executor_task(task.get("task", "hold_position"))
                if active is None or str(active.get("task", "")) != new_name:
                    start_task(agent_id, task, "continue_task", obs[agent_id])
                save_plan_ctx(agent_id, obs[agent_id], task)
                action = execute_task(agent_id, obs[agent_id], task)
                if self.postprocess:
                    action = self.postprocess(action)
                action = stabilize_agent_action(action, obs[agent_id], agent_id, state)
                update_agent_commitment(action, obs[agent_id], state, agent_id)
                actions[agent_id] = action
                if isinstance(state, dict):
                    state.setdefault("llm_calls", []).append(
                        {
                            "agent_id": int(agent_id),
                            "mode": "text",
                            "from_chunk": True,
                            "replan": False,
                            "replan_reason": reason,
                            "executor_task": task,
                            "parsed_action": action,
                        }
                    )
                continue
            prev_task = hist.get("executor_task")
            if prev_task is not None:
                if reason == "stuck":
                    out = "stuck"
                elif reason == "task_timeout":
                    out = "timeout"
                else:
                    out = "interrupted"
                finalize_task(agent_id, out, reason, obs[agent_id])
            prompt = self.prompt_fn(obs, info, agent_id, state)
            order.append(agent_id)
            prompts.append(prompt)

        responses = []
        for p in prompts:
            if self.cache:
                cached = self.cache.get(p)
                if cached is not None:
                    responses.append(cached)
                    continue
            responses.append(None)
        missing_idx = [i for i, r in enumerate(responses) if r is None]
        if missing_idx:
            to_call = [prompts[i] for i in missing_idx]
            called = self.batch_fn(to_call, self.llm_call)
            for idx, resp in zip(missing_idx, called):
                responses[idx] = resp
                if self.cache:
                    self.cache.set(prompts[idx], resp)

        for idx, (agent_id, raw) in enumerate(zip(order, responses)):
            data = parse_jsonish(raw) or {}
            plan = data.get("task_plan", [])
            if not isinstance(plan, list):
                plan = []
            task_name = str(data.get("task", "")).strip() or (str(plan[0]).strip() if plan else "hold_position")
            task_name = _canonical_executor_task(task_name)
            if plan and task_name == str(plan[0]).strip():
                plan_queue = [str(x).strip() for x in plan[1:5] if str(x).strip()]
            else:
                plan_queue = [str(x).strip() for x in plan[:5] if str(x).strip()]
            task = {
                "task": task_name,
                "task_plan": plan_queue,
                "message_text": data.get("message_text"),
            }
            runtime = state.setdefault("_agent_runtime", {}) if isinstance(state, dict) else {}
            hist = runtime.setdefault(int(agent_id), {})
            hist["executor_task"] = task
            start_task(agent_id, task, str(hist.get("pending_replan_reason", "replan")), obs[agent_id])
            save_plan_ctx(agent_id, obs[agent_id], task)
            action = execute_task(agent_id, obs[agent_id], task)
            if self.postprocess:
                action = self.postprocess(action)
            action = stabilize_agent_action(action, obs[agent_id], agent_id, state)
            update_agent_commitment(action, obs[agent_id], state, agent_id)
            actions[agent_id] = action
            if isinstance(state, dict):
                reason = hist.get("pending_replan_reason")
                state.setdefault("llm_calls", []).append(
                    {
                        "agent_id": int(agent_id),
                        "mode": "text",
                        "from_chunk": False,
                        "replan": True,
                        "replan_reason": reason,
                        "prompt": prompts[idx],
                        "raw_response": raw,
                        "executor_task": task,
                        "parsed_action": action,
                    }
                )
                hist.pop("pending_replan_reason", None)
        if isinstance(state, dict):
            runtime = state.setdefault("_agent_runtime", {})
            summary = {}
            for agent_id in obs.keys():
                hist = runtime.setdefault(int(agent_id), {})
                completed = hist.get("task_history", [])
                outcomes = {"success": 0, "timeout": 0, "stuck": 0, "interrupted": 0}
                for item in completed:
                    out = str(item.get("task_outcome", ""))
                    if out in outcomes:
                        outcomes[out] += 1
                summary[int(agent_id)] = {
                    "active_task": hist.get("active_task_run"),
                    "last_completed": completed[-1] if completed else None,
                    "completed_counts": outcomes,
                }
            state["task_metrics"] = summary
        return actions
