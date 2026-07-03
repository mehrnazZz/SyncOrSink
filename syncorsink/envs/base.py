from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # fallback for environments without gymnasium
    import gym
    from gym import spaces

from .communication import Message, count_tokens
from .maps import (
    TILE_EMPTY,
    TILE_RESOURCE,
    TILE_STATION,
    TILE_NODE,
    TILE_CLUE,
    TILE_TARGET,
    TILE_WATER,
    TILE_BEACON,
    TILE_DOOR,
    TILE_UNKNOWN,
)
from .observation import extract_local, visible_mask
from .scenarios import SCENARIOS
from .utils import FOV_PRESETS, get_rng


@dataclass
class SyncOrSinkConfig:
    scenario: str = "pipeline_assembly"
    map_size: int = 16
    num_agents: int = 3
    fov_preset: str = "medium"
    comm_mode: str = "tokens"  # "tokens" or "text"
    comm_token_limit: int = 24
    max_messages: int = 8
    token_vocab_size: int = 256
    max_steps: int = 300
    comm_radius: int | None = None
    comm_cost: float = 0.01
    comm_len_cost: float = 0.0
    # map / rendering options
    use_rooms: bool = True
    use_doors: bool = True
    enable_fog_of_war: bool = True
    # signal_hunt options
    signal_decoy_count: int | None = None
    decoy_penalty: float = 0.5
    scan_window: int = 3
    # pipeline shaping
    pipeline_shaping: bool = False
    pipeline_shaping_scale: float = 0.01
    # energy grid shaping
    energy_shaping: bool = False
    energy_shaping_scale: float = 0.01
    # signal hunt shaping
    signal_shaping: bool = False
    signal_shaping_scale: float = 0.01
    signal_scan_bonus: float = 0.0       # small bonus for solo scan on target (keep low to avoid farming)
    signal_joint_scan_bonus: float = 0.0  # large bonus when partner also scanned within window (near-miss reward)
    signal_colocation_bonus: float = 0.0  # reward when 2+ agents interact near target simultaneously
    signal_colocation_radius: int = 2     # manhattan distance threshold for co-location
    signal_comm_utility: float = 0.0      # reward for sending a message that precedes teammate's useful action
    # energy grid difficulty presets
    energy_preset: str = "hard"  # "easy" or "hard"
    energy_private_monitor: bool = True  # each agent only sees energy of assigned nodes
    # deterministic map control
    map_seed: int | None = None
    map_variant: int = 0
    split: str | None = None
    # benchmark track
    track: str = "dtde"  # "dtde" or "ctde"
    # rendering options
    render_god_view: bool = False
    render_split_view: bool = False
    render_style: str = "arcade_flat"  # "arcade_flat" or "sprite"
    obs_onehot: bool = False
    # optional per-agent exploration memory in observations
    obs_exploration_memory: bool = True
    obs_exploration_age: bool = False
    # include valid-action mask per agent in observations
    obs_action_mask: bool = True


class SyncOrSinkEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human", "rgb_array"]}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4
    ACTION_INTERACT = 5
    ACTION_PICKUP = 6
    ACTION_DROP = 7

    def __init__(self, config: SyncOrSinkConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or SyncOrSinkConfig()
        self.render_mode = render_mode
        self._pg_renderer = None
        if self.config.fov_preset not in FOV_PRESETS:
            raise ValueError(f"Unknown fov preset: {self.config.fov_preset}")
        if self.config.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.config.scenario}")

        self.num_agents = self.config.num_agents
        self.map_size = self.config.map_size
        self.fov_radius = FOV_PRESETS[self.config.fov_preset].radius
        self.comm_mode = self.config.comm_mode
        self.comm_token_limit = self.config.comm_token_limit
        self.max_messages = self.config.max_messages
        self.token_vocab_size = self.config.token_vocab_size
        self.max_steps = self.config.max_steps
        self.comm_radius = self.config.comm_radius
        self.obs_onehot = self.config.obs_onehot
        self.obs_exploration_memory = self.config.obs_exploration_memory
        self.obs_exploration_age = self.config.obs_exploration_age
        self.obs_action_mask = self.config.obs_action_mask

        self.scenario = SCENARIOS[self.config.scenario]
        self.rng = get_rng(None)
        self.steps = 0

        # rewards
        self.reward_stage = 1.0
        self.reward_complete = 10.0
        self.reward_fail = 5.0
        self.energy_start = 6
        self.energy_refill = 4
        self.comm_cost = self.config.comm_cost

        self.grid = None
        self.meta = {}
        self.agent_positions: list[tuple[int, int]] = []
        self.inventories: list[int] = []
        self.scenario_state = None
        self.inboxes: list[list[Message]] = []
        self.agent_explored: list[np.ndarray] = []
        self.agent_last_seen_step: list[np.ndarray] = []

        self.action_space = spaces.Dict({
            "action": spaces.Discrete(8),
            "message_tokens": spaces.Box(
                low=-1,
                high=self.token_vocab_size - 1,
                shape=(self.comm_token_limit,),
                dtype=np.int16,
            ),
            "message_tokens_aux": spaces.Box(
                low=-1,
                high=self.token_vocab_size - 1,
                shape=(self.comm_token_limit,),
                dtype=np.int16,
            ),
        })

        local_shape = (self.fov_radius * 2 + 1, self.fov_radius * 2 + 1)
        self._tile_channels = max(TILES.values()) + 1
        if self.obs_onehot:
            local_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._tile_channels, local_shape[0], local_shape[1]),
                dtype=np.float32,
            )
        else:
            local_space = spaces.Box(
                low=0,
                high=12,
                shape=local_shape,
                dtype=np.int16,
            )
        self.observation_space = spaces.Dict({
            "local_grid": local_space,
            "inventory": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int16),
            "self_pos": spaces.Box(low=0, high=self.map_size - 1, shape=(2,), dtype=np.int16),
            "local_resource_types": spaces.Box(
                low=0, high=10, shape=local_shape, dtype=np.int16
            ),
            "local_node_types": spaces.Box(
                low=0, high=10, shape=local_shape, dtype=np.int16
            ),
            "local_node_energy": spaces.Box(
                low=0, high=50, shape=local_shape, dtype=np.int16
            ),
            "messages_tokens": spaces.Box(
                low=-1,
                high=self.token_vocab_size - 1,
                shape=(self.max_messages, self.comm_token_limit),
                dtype=np.int16,
            ),
            "message_from": spaces.Box(low=-1, high=self.num_agents - 1, shape=(self.max_messages,), dtype=np.int16),
            "goal_hint": spaces.Box(low=-1, high=1024, shape=(16,), dtype=np.int16),
        })
        if self.obs_action_mask:
            self.observation_space.spaces["action_mask"] = spaces.Box(
                low=0,
                high=1,
                shape=(8,),
                dtype=np.int8,
            )
        if self.obs_exploration_memory:
            self.observation_space.spaces["explored_mask"] = spaces.Box(
                low=0,
                high=1,
                shape=(self.map_size, self.map_size),
                dtype=np.int8,
            )
            if self.obs_exploration_age:
                self.observation_space.spaces["explored_age"] = spaces.Box(
                    low=-1,
                    high=max(1024, self.max_steps),
                    shape=(self.map_size, self.map_size),
                    dtype=np.int16,
                )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.rng = get_rng(seed)
        self.steps = 0
        map_seed = self.config.map_seed if self.config.map_seed is not None else seed
        if self.config.split is not None and map_seed is None:
            from syncorsink.eval.splits import seed_for_variant
            map_seed = seed_for_variant(self.config.split, self.config.map_variant)
        if map_seed is not None:
            map_rng = get_rng(map_seed + self.config.map_variant)
        else:
            map_rng = self.rng
        self.grid, self.meta = self.scenario.build_map(self.map_size, map_rng, self.config)
        starts = self.meta.get("agent_starts", [])
        if len(starts) < self.num_agents:
            # fallback: sample additional valid empty cells (never walls/doors/objects).
            used = set(starts)
            empties = []
            for y in range(self.map_size):
                for x in range(self.map_size):
                    if self.grid[y, x] == TILE_EMPTY and (x, y) not in used:
                        empties.append((x, y))
            self.rng.shuffle(empties)
            need = self.num_agents - len(starts)
            starts = starts + empties[:need]
            # Last resort if map is pathological: reuse existing empty cells.
            if len(starts) < self.num_agents:
                fallback_pool = empties if empties else list(used)
                while len(starts) < self.num_agents and fallback_pool:
                    starts.append(fallback_pool[int(self.rng.integers(0, len(fallback_pool)))])
            if len(starts) < self.num_agents:
                # Absolute safety: fill from any non-wall/non-door cell to keep reset robust.
                walkable = []
                for y in range(self.map_size):
                    for x in range(self.map_size):
                        if int(self.grid[y, x]) not in (1, TILE_DOOR):
                            walkable.append((x, y))
                if not walkable:
                    walkable = [(0, 0)]
                while len(starts) < self.num_agents:
                    starts.append(walkable[int(self.rng.integers(0, len(walkable)))])
        self.agent_positions = starts[: self.num_agents]
        self.inventories = [0 for _ in range(self.num_agents)]
        self.inboxes = [[] for _ in range(self.num_agents)]
        self.scenario_state = self.scenario.reset(self)
        self._init_exploration_memory()
        self._update_exploration_memory()

        obs = self._build_observations()
        info = self._build_info()
        if self.config.track == "ctde":
            from .central_obs import build_central_obs
            info["central_obs"] = build_central_obs(self)
        return obs, info

    def step(self, actions: dict[int, Any]):
        self.steps += 1
        parsed_actions = self._parse_actions(actions)

        # movement phase
        occupied = {pos: idx for idx, pos in enumerate(self.agent_positions)}
        allow_multi = {TILE_STATION, TILE_NODE, TILE_TARGET}
        for agent_id, action in parsed_actions.items():
            if action["action"] in (self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT):
                x, y = self.agent_positions[agent_id]
                nx, ny = x, y
                if action["action"] == self.ACTION_UP:
                    ny -= 1
                elif action["action"] == self.ACTION_DOWN:
                    ny += 1
                elif action["action"] == self.ACTION_LEFT:
                    nx -= 1
                elif action["action"] == self.ACTION_RIGHT:
                    nx += 1
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if self.grid[ny, nx] not in (1, TILE_DOOR):
                        if (nx, ny) in occupied and self.grid[ny, nx] not in allow_multi:
                            continue
                        occupied.pop(self.agent_positions[agent_id], None)
                        self.agent_positions[agent_id] = (nx, ny)
                        occupied[(nx, ny)] = agent_id

        # interaction phase: pickup/drop
        for agent_id, action in parsed_actions.items():
            pos = self.agent_positions[agent_id]
            if action["action"] == self.ACTION_PICKUP:
                if self.inventories[agent_id] == 0:
                    if pos in self._resource_positions():
                        resource_type = self._resource_positions()[pos]
                        self.inventories[agent_id] = resource_type
                        self._remove_resource(pos)
            elif action["action"] == self.ACTION_DROP:
                if self.inventories[agent_id] != 0 and self.grid[pos[1], pos[0]] == TILE_EMPTY:
                    self._add_resource(pos, self.inventories[agent_id])
                    self.inventories[agent_id] = 0
            elif action["action"] == self.ACTION_INTERACT and self.config.use_doors:
                # open adjacent door if present
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        if self.grid[ny, nx] == TILE_DOOR:
                            self.grid[ny, nx] = TILE_EMPTY
                            break

        # communication phase
        comm_tokens = {i: 0 for i in range(self.num_agents)}
        for agent_id, action in parsed_actions.items():
            msg_tokens = action.get("message_tokens")
            msg_tokens_aux = action.get("message_tokens_aux")
            msg_text = action.get("message_text")
            token_count = count_tokens(msg_tokens, msg_text)
            aux_count = count_tokens(msg_tokens_aux, None)
            if token_count == 0 and aux_count == 0:
                continue
            if token_count > self.comm_token_limit:
                if msg_tokens is not None:
                    msg_tokens = msg_tokens[: self.comm_token_limit]
                if msg_text is not None:
                    msg_text = " ".join(msg_text.split()[: self.comm_token_limit])
                token_count = self.comm_token_limit
            if aux_count > self.comm_token_limit:
                if msg_tokens_aux is not None:
                    msg_tokens_aux = msg_tokens_aux[: self.comm_token_limit]
                aux_count = self.comm_token_limit
            comm_tokens[agent_id] = token_count + aux_count
            messages = []
            if token_count > 0:
                messages.append(Message(sender=agent_id, tokens=msg_tokens or [], text=msg_text))
            if aux_count > 0:
                messages.append(Message(sender=agent_id, tokens=msg_tokens_aux or [], text=None))
            for recv_id in range(self.num_agents):
                if recv_id == agent_id:
                    continue
                if self.comm_radius is not None:
                    ax, ay = self.agent_positions[agent_id]
                    bx, by = self.agent_positions[recv_id]
                    if abs(ax - bx) + abs(ay - by) > self.comm_radius:
                        continue
                for message in messages:
                    self.inboxes[recv_id].append(message)
                if len(self.inboxes[recv_id]) > self.max_messages:
                    self.inboxes[recv_id] = self.inboxes[recv_id][-self.max_messages :]

        # scenario-specific step
        rewards, done, scenario_info = self.scenario.step(self, parsed_actions)
        self._update_exploration_memory()

        # communication penalty
        for agent_id, count in comm_tokens.items():
            rewards[agent_id] -= self.comm_cost * count
            if self.config.comm_len_cost > 0:
                rewards[agent_id] -= self.config.comm_len_cost * count

        truncated = self.steps >= self.max_steps
        obs = self._build_observations()
        info = self._build_info()
        if self.config.track == "ctde":
            from .central_obs import build_central_obs
            info["central_obs"] = build_central_obs(self)
        info["comm_tokens"] = comm_tokens
        info.update(scenario_info)
        return obs, rewards, done, truncated, info

    def render(self):
        if self.render_mode in (None, "ansi"):
            return self._render_ansi()
        if self.render_mode == "human":
            return self._render_human()
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        raise NotImplementedError("Only ansi and human render are supported.")

    def _render_ansi(self) -> str:
        glyphs = {
            TILE_EMPTY: ".",
            1: "#",
            TILE_RESOURCE: "R",
            TILE_STATION: "S",
            TILE_NODE: "N",
            TILE_CLUE: "C",
            TILE_TARGET: "T",
            TILE_WATER: "W",
            TILE_BEACON: "B",
            TILE_DOOR: "D",
            TILE_UNKNOWN: "?",
        }
        grid = [[glyphs.get(int(self.grid[y, x]), "?") for x in range(self.map_size)] for y in range(self.map_size)]
        for agent_id, (x, y) in enumerate(self.agent_positions):
            grid[y][x] = str(agent_id % 10)
        lines = ["".join(row) for row in grid]
        lines.append(f"steps={self.steps} scenario={self.config.scenario} agents={self.num_agents}")
        return "\n".join(lines)

    def _render_human(self):
        if self._pg_renderer is None:
            from .pygame_render import SyncOrSinkPygameRenderer
            self._pg_renderer = SyncOrSinkPygameRenderer(
                map_size=self.map_size,
                god_view=self.config.render_god_view,
                split_view=self.config.render_split_view,
                style=self.config.render_style,
            )
        info_text = f"steps={self.steps} scenario={self.config.scenario} agents={self.num_agents}"
        mask = None
        if self.config.enable_fog_of_war:
            mask = visible_mask(self.grid, self.agent_positions, self.fov_radius)
        resource_types = self.scenario_state.data.get("resource_types")
        node_energy = self.scenario_state.data.get("node_energy")
        node_types = self.scenario_state.data.get("node_types")
        decoys = self.scenario_state.data.get("decoys")
        self._pg_renderer.render(
            self.grid,
            self.agent_positions,
            info_text=info_text,
            fov_radius=self.fov_radius,
            visible_mask=mask,
            inventories=self.inventories,
            resource_types=resource_types,
            node_energy=node_energy,
            node_types=node_types,
            decoys=decoys,
            scenario=self.config.scenario,
        )
        return None

    def _render_rgb(self):
        if self._pg_renderer is None:
            from .pygame_render import SyncOrSinkPygameRenderer
            self._pg_renderer = SyncOrSinkPygameRenderer(map_size=self.map_size)
        info_text = f"steps={self.steps} scenario={self.config.scenario} agents={self.num_agents}"
        mask = None
        if self.config.enable_fog_of_war:
            mask = visible_mask(self.grid, self.agent_positions, self.fov_radius)
        resource_types = self.scenario_state.data.get("resource_types")
        node_energy = self.scenario_state.data.get("node_energy")
        node_types = self.scenario_state.data.get("node_types")
        decoys = self.scenario_state.data.get("decoys")
        surface = self._pg_renderer.render_rgb(
            self.grid,
            self.agent_positions,
            info_text=info_text,
            fov_radius=self.fov_radius,
            visible_mask=mask,
            inventories=self.inventories,
            resource_types=resource_types,
            node_energy=node_energy,
            node_types=node_types,
            decoys=decoys,
            scenario=self.config.scenario,
        )
        return surface

    def close(self):
        if self._pg_renderer is not None:
            self._pg_renderer.close()
            self._pg_renderer = None

    def _parse_actions(self, actions: dict[int, Any]) -> dict[int, dict]:
        parsed = {}
        for agent_id in range(self.num_agents):
            act = actions.get(agent_id, {"action": self.ACTION_STAY})
            if isinstance(act, int):
                act = {"action": int(act)}
            action_id = int(act.get("action", self.ACTION_STAY))
            msg_tokens = act.get("message_tokens")
            if msg_tokens is not None:
                msg_tokens = [int(t) for t in list(msg_tokens) if int(t) >= 0][: self.comm_token_limit]
            msg_tokens_aux = act.get("message_tokens_aux")
            if msg_tokens_aux is not None:
                msg_tokens_aux = [int(t) for t in list(msg_tokens_aux) if int(t) >= 0][: self.comm_token_limit]
            msg_text = act.get("message_text")
            parsed[agent_id] = {
                "action": action_id,
                "message_tokens": msg_tokens,
                "message_tokens_aux": msg_tokens_aux,
                "message_text": msg_text if self.comm_mode == "text" else None,
            }
        return parsed

    def _resource_positions(self) -> dict[tuple[int, int], int]:
        return self.scenario_state.data.get("resource_types", {})

    def _remove_resource(self, pos: tuple[int, int]):
        if pos in self.scenario_state.data.get("resource_types", {}):
            self.scenario_state.data["resource_types"].pop(pos, None)
        if self.grid[pos[1], pos[0]] == TILE_RESOURCE:
            self.grid[pos[1], pos[0]] = TILE_EMPTY

    def _add_resource(self, pos: tuple[int, int], resource_type: int):
        self.scenario_state.data.setdefault("resource_types", {})[pos] = resource_type
        self.grid[pos[1], pos[0]] = TILE_RESOURCE

    def _build_observations(self) -> dict[int, dict]:
        obs = {}
        resource_type_grid = np.zeros((self.map_size, self.map_size), dtype=np.int16)
        for pos, rtype in self._resource_positions().items():
            resource_type_grid[pos[1], pos[0]] = int(rtype)
        node_type_grid = np.zeros((self.map_size, self.map_size), dtype=np.int16)
        for pos, ntype in self.scenario_state.data.get("node_types", {}).items():
            node_type_grid[pos[1], pos[0]] = int(ntype)
        node_energy_grid = np.zeros((self.map_size, self.map_size), dtype=np.int16)
        for pos, energy in self.scenario_state.data.get("node_energy", {}).items():
            node_energy_grid[pos[1], pos[0]] = int(energy)

        # Private monitoring: build per-agent energy grids
        private_monitor = (
            self.config.energy_private_monitor
            and self.config.scenario == "energy_grid"
            and "node_assignments" in self.scenario_state.data
        )
        if private_monitor:
            agent_energy_grids = {}
            node_assignments = self.scenario_state.data["node_assignments"]
            for agent_id in range(self.num_agents):
                masked = np.zeros((self.map_size, self.map_size), dtype=np.int16)
                for pos, energy in self.scenario_state.data.get("node_energy", {}).items():
                    if node_assignments.get(pos) == agent_id:
                        masked[pos[1], pos[0]] = int(energy)
                agent_energy_grids[agent_id] = masked

        for agent_id in range(self.num_agents):
            local = extract_local(self.grid, self.agent_positions[agent_id], self.fov_radius)
            if self.obs_onehot:
                local = self._onehot(local)
            local_resource_types = extract_local(resource_type_grid, self.agent_positions[agent_id], self.fov_radius)
            local_node_types = extract_local(node_type_grid, self.agent_positions[agent_id], self.fov_radius)
            if private_monitor:
                local_node_energy = extract_local(agent_energy_grids[agent_id], self.agent_positions[agent_id], self.fov_radius)
            else:
                local_node_energy = extract_local(node_energy_grid, self.agent_positions[agent_id], self.fov_radius)
            messages_tokens, message_from = self._encode_messages(self.inboxes[agent_id])
            hint_tokens = self._encode_hint(agent_id)
            obs[agent_id] = {
                "local_grid": local,
                "inventory": np.array([self.inventories[agent_id]], dtype=np.int16),
                "self_pos": np.array(self.agent_positions[agent_id], dtype=np.int16),
                "local_resource_types": local_resource_types,
                "local_node_types": local_node_types,
                "local_node_energy": local_node_energy,
                "messages_tokens": messages_tokens,
                "message_from": message_from,
                "goal_hint": hint_tokens,
            }
            if self.obs_action_mask:
                obs[agent_id]["action_mask"] = self._valid_action_mask(agent_id)
            if self.obs_exploration_memory:
                obs[agent_id]["explored_mask"] = self.agent_explored[agent_id].astype(np.int8, copy=True)
                if self.obs_exploration_age:
                    last_seen = self.agent_last_seen_step[agent_id]
                    age = np.where(last_seen >= 0, self.steps - last_seen, -1).astype(np.int16)
                    obs[agent_id]["explored_age"] = age
        return obs

    def _valid_action_mask(self, agent_id: int) -> np.ndarray:
        mask = np.zeros((8,), dtype=np.int8)
        x, y = self.agent_positions[agent_id]
        inv = self.inventories[agent_id]

        # stay is always valid
        mask[self.ACTION_STAY] = 1

        # movement validity (bounds + static blockers + occupancy unless multi-occupancy tile)
        occupied = {pos: idx for idx, pos in enumerate(self.agent_positions)}
        allow_multi = {TILE_STATION, TILE_NODE, TILE_TARGET}
        move_deltas = {
            self.ACTION_UP: (0, -1),
            self.ACTION_DOWN: (0, 1),
            self.ACTION_LEFT: (-1, 0),
            self.ACTION_RIGHT: (1, 0),
        }
        for act, (dx, dy) in move_deltas.items():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.map_size and 0 <= ny < self.map_size):
                continue
            tile = int(self.grid[ny, nx])
            if tile in (1, TILE_DOOR):
                continue
            if (nx, ny) in occupied and occupied[(nx, ny)] != agent_id and tile not in allow_multi:
                continue
            mask[act] = 1

        pos = (x, y)
        center_tile = int(self.grid[y, x])

        # interact validity: useful interaction tile or adjacent door (if enabled)
        can_interact = center_tile in {TILE_STATION, TILE_NODE, TILE_CLUE, TILE_TARGET}
        if self.config.use_doors and not can_interact:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size and int(self.grid[ny, nx]) == TILE_DOOR:
                    can_interact = True
                    break
        if can_interact:
            mask[self.ACTION_INTERACT] = 1

        # pickup/drop validity
        if inv == 0 and pos in self._resource_positions():
            mask[self.ACTION_PICKUP] = 1
        if inv != 0 and center_tile == TILE_EMPTY:
            mask[self.ACTION_DROP] = 1

        return mask

    def _init_exploration_memory(self):
        self.agent_explored = [np.zeros((self.map_size, self.map_size), dtype=bool) for _ in range(self.num_agents)]
        self.agent_last_seen_step = [np.full((self.map_size, self.map_size), -1, dtype=np.int16) for _ in range(self.num_agents)]

    def _update_exploration_memory(self):
        if not self.obs_exploration_memory:
            return
        for aid, pos in enumerate(self.agent_positions):
            mask = visible_mask(self.grid, [pos], self.fov_radius)
            self.agent_explored[aid] |= mask
            self.agent_last_seen_step[aid][mask] = int(self.steps)

    def _onehot(self, local: np.ndarray) -> np.ndarray:
        # One-hot encode tiles 0..max into shape (C, H, W)
        channels = self._tile_channels
        h, w = local.shape
        out = np.zeros((channels, h, w), dtype=np.float32)
        for c in range(channels):
            out[c] = (local == c).astype(np.float32)
        return out

    def _encode_messages(self, inbox: list[Message]):
        tokens = np.full((self.max_messages, self.comm_token_limit), -1, dtype=np.int16)
        senders = np.full((self.max_messages,), -1, dtype=np.int16)
        for idx, msg in enumerate(inbox[-self.max_messages :]):
            senders[idx] = msg.sender
            for j, t in enumerate(msg.tokens[: self.comm_token_limit]):
                tokens[idx, j] = int(t)
        return tokens, senders

    def _encode_hint(self, agent_id: int) -> np.ndarray:
        hint = []
        if self.config.scenario == "pipeline_assembly":
            stages = self.scenario_state.data.get("hints", {}).get(agent_id, [])
            for stage in stages:
                required = stage.get("required", [])
                deps = stage.get("deps", [])
                hint.extend(
                    [
                        stage["stage"],
                        stage["station"][0],
                        stage["station"][1],
                        len(required),
                        required[0] if len(required) > 0 else -1,
                        required[1] if len(required) > 1 else -1,
                        len(deps),
                        deps[0] if deps else -1,
                        1 if stage.get("sync") else 0,
                    ]
                )
        elif self.config.scenario == "energy_grid" and self.config.energy_private_monitor:
            # Encode assigned node positions and types so agent knows its responsibility
            agent_nodes = self.scenario_state.data.get("agent_nodes", {}).get(agent_id, [])
            node_types = self.scenario_state.data.get("node_types", {})
            for node_pos in agent_nodes:
                ntype = node_types.get(node_pos, 0)
                hint.extend([node_pos[0], node_pos[1], ntype])
        elif self.config.scenario == "signal_hunt":
            seen_specs = set()
            specs = []
            initial = self.scenario_state.data.get("agent_hint_specs", {}).get(agent_id)
            if initial:
                specs.append(initial)
            specs.extend(self.scenario_state.data.get("agent_clue_specs", {}).get(agent_id, []))
            for spec in specs:
                encoded = self._encode_signal_constraint(spec)
                if not encoded:
                    continue
                key = tuple(encoded)
                if key in seen_specs:
                    continue
                seen_specs.add(key)
                hint.extend(encoded)
                if len(hint) >= 16:
                    break
        hint = hint[:16]
        padded = hint + [-1] * (16 - len(hint))
        return np.array(padded, dtype=np.int16)

    def _encode_signal_constraint(self, constraint: dict) -> list[int]:
        ctype = constraint.get("type")
        if ctype == "near":
            pos = constraint.get("pos", (-1, -1))
            return [21, self._signal_object_code(constraint.get("object")), int(pos[0]), int(pos[1]), int(constraint.get("dist", 0))]
        if ctype == "offset":
            pos = constraint.get("pos", (-1, -1))
            return [
                22,
                self._signal_object_code(constraint.get("object")),
                int(pos[0]),
                int(pos[1]),
                int(constraint.get("dx", 0)),
                int(constraint.get("dy", 0)),
            ]
        if ctype == "parity_quadrant":
            return [
                23,
                int(constraint.get("parity", 0)),
                self._quadrant_code(constraint.get("quadrant")),
                int(constraint.get("size", self.map_size)),
            ]
        if ctype == "x_parity":
            return [24, int(constraint.get("value", 0))]
        if ctype == "y_parity":
            return [25, int(constraint.get("value", 0))]
        return []

    @staticmethod
    def _signal_object_code(name: Any) -> int:
        return {"water": TILE_WATER, "beacon": TILE_BEACON}.get(str(name), 0)

    @staticmethod
    def _quadrant_code(name: Any) -> int:
        return {"NW": 0, "NE": 1, "SW": 2, "SE": 3}.get(str(name), -1)

    def _build_info(self) -> dict:
        info = {
            "messages_text": {},
            "messages_with_sender": {},
            "scenario": self.config.scenario,
        }
        for agent_id in range(self.num_agents):
            txt = [m.text for m in self.inboxes[agent_id] if m.text]
            info["messages_text"][agent_id] = txt
            info["messages_with_sender"][agent_id] = [
                {"from": int(m.sender), "text": m.text}
                for m in self.inboxes[agent_id]
                if m.text
            ]
        return info


# Expose tiles for users
TILES = {
    "empty": TILE_EMPTY,
    "resource": TILE_RESOURCE,
    "station": TILE_STATION,
    "node": TILE_NODE,
    "clue": TILE_CLUE,
    "target": TILE_TARGET,
    "water": TILE_WATER,
    "beacon": TILE_BEACON,
    "door": TILE_DOOR,
    "unknown": TILE_UNKNOWN,
}
