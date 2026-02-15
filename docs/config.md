# Configuration Reference

All configuration options are defined in `syncorsink/envs/base.py` under `SyncOrSinkConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `scenario` | str | `"pipeline_assembly"` | Scenario name: `pipeline_assembly`, `energy_grid`, `signal_hunt`. |
| `map_size` | int | `16` | Grid size (supports 8/16/32). |
| `num_agents` | int | `3` | Number of agents. |
| `fov_preset` | str | `"medium"` | Partial observability radius: `hard`, `medium`, `easy`. |
| `comm_mode` | str | `"tokens"` | Communication mode: `tokens` or `text`. |
| `comm_token_limit` | int | `24` | Max tokens per step per agent. |
| `max_messages` | int | `8` | Max messages stored per agent. |
| `token_vocab_size` | int | `256` | Token vocab size for token mode. |
| `max_steps` | int | `300` | Episode truncation length. |
| `comm_radius` | int? | `None` | Message radius (None = broadcast). |
| `use_rooms` | bool | `True` | Use room/corridor map generator. |
| `use_doors` | bool | `True` | Enable doors in rooms. |
| `enable_fog_of_war` | bool | `True` | Mask tiles outside FOV in render. |
| `signal_decoy_count` | int? | `None` | Override decoy target count in Signal Hunt. |
| `decoy_penalty` | float | `0.5` | Penalty multiplier for decoy scan. |
| `scan_window` | int | `3` | Steps window for joint scan. |
| `map_seed` | int? | `None` | Fixed map seed. |
| `map_variant` | int | `0` | Variant offset for seed. |
| `split` | str? | `None` | Split name: `train`, `val`, `test`. |
| `track` | str | `"dtde"` | Training track: `dtde` or `ctde`. |
| `render_god_view` | bool | `False` | Render full map (no fog). |
| `render_split_view` | bool | `False` | Render agent view + god view side‑by‑side. |
| `render_style` | str | `"arcade_flat"` | Render style: `arcade_flat` or `sprite`. |
| `obs_onehot` | bool | `False` | If true, `local_grid` becomes one‑hot channels `(C,H,W)` instead of integer ids. |
