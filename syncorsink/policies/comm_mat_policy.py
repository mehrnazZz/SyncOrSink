from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from syncorsink.models.comm_mat import CommMATConfig, CommMATModel

from .base import BasePolicy
from .registry import register


@dataclass
class CommMATPolicyConfig:
    action_dim: int = 8
    tile_vocab_size: int = 16
    comm_vocab_size: int = 256
    comm_token_limit: int = 24
    max_messages: int = 8
    max_agents: int = 16
    goal_hint_dim: int = 32
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    deterministic: bool = True
    send_threshold: float = 0.5


def _to_grid_ids(local_grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(local_grid)
    if arr.ndim == 3:
        # one-hot channels -> tile id
        return arr.argmax(axis=0).astype(np.int64)
    return arr.astype(np.int64)


def _get_recv_from(obs_agent: dict, max_messages: int) -> np.ndarray:
    recv_from = obs_agent.get("messages_from")
    if recv_from is None:
        recv_from = obs_agent.get("message_from")
    if recv_from is None:
        return np.full((max_messages,), -1, dtype=np.int64)
    recv_from = np.asarray(recv_from, dtype=np.int64).reshape(-1)
    if recv_from.shape[0] < max_messages:
        pad = np.full((max_messages - recv_from.shape[0],), -1, dtype=np.int64)
        recv_from = np.concatenate([recv_from, pad], axis=0)
    return recv_from[:max_messages]


def _get_recv_tokens(obs_agent: dict, max_messages: int, token_limit: int) -> np.ndarray:
    toks = np.asarray(
        obs_agent.get("messages_tokens", np.zeros((max_messages, token_limit), dtype=np.int64)),
        dtype=np.int64,
    )
    if toks.ndim == 1:
        toks = toks.reshape(1, -1)
    out = np.zeros((max_messages, token_limit), dtype=np.int64)
    h = min(max_messages, toks.shape[0])
    w = min(token_limit, toks.shape[1])
    out[:h, :w] = toks[:h, :w]
    return out


@register("comm_mat")
class CommMATPolicy(BasePolicy):
    """
    DTDE Comm-MAT policy adapter for SyncOrSink.
    Returns per-agent env action and optional message_tokens.
    """

    def __init__(
        self,
        model: CommMATModel | None = None,
        config: CommMATPolicyConfig | None = None,
        device: str = "cpu",
        checkpoint: str | None = None,
    ):
        self.cfg = config or CommMATPolicyConfig()
        self.device = torch.device(device)
        self.model = model
        self._built = model is not None
        self._checkpoint_path = checkpoint if checkpoint is not None and model is None else None
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        if checkpoint is not None:
            if self.model is not None:
                self._load_checkpoint(checkpoint)

    def _load_checkpoint(self, path: str):
        if self.model is None:
            raise ValueError("Model must be initialized before loading checkpoint")
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()

    def _build_if_needed(self, obs: dict):
        if self._built:
            return
        sample = obs[next(iter(obs.keys()))]
        goal_hint = np.asarray(sample.get("goal_hint", np.zeros((self.cfg.goal_hint_dim,), dtype=np.float32)))
        msgs = np.asarray(sample.get("messages_tokens", np.zeros((self.cfg.max_messages, self.cfg.comm_token_limit))))
        max_messages = int(msgs.shape[0]) if msgs.ndim >= 2 else self.cfg.max_messages
        token_limit = int(msgs.shape[1]) if msgs.ndim >= 2 else self.cfg.comm_token_limit
        model_cfg = CommMATConfig(
            action_dim=self.cfg.action_dim,
            tile_vocab_size=self.cfg.tile_vocab_size,
            comm_vocab_size=self.cfg.comm_vocab_size,
            comm_token_limit=token_limit,
            max_messages=max_messages,
            max_agents=self.cfg.max_agents,
            goal_hint_dim=max(self.cfg.goal_hint_dim, int(goal_hint.reshape(-1).shape[0])),
            hidden_dim=self.cfg.hidden_dim,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            dropout=self.cfg.dropout,
        )
        self.model = CommMATModel(model_cfg).to(self.device)
        self.model.eval()
        self._built = True
        if self._checkpoint_path is not None:
            self._load_checkpoint(self._checkpoint_path)
            self._checkpoint_path = None

    def reset(self):
        return None

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        self._build_if_needed(obs)
        assert self.model is not None
        model_cfg = self.model.cfg

        agent_ids = sorted(obs.keys())
        grid_ids = []
        inventory = []
        self_pos = []
        goal_hint = []
        recv_tokens = []
        recv_from = []
        action_masks = []

        for aid in agent_ids:
            oa = obs[aid]
            grid_ids.append(_to_grid_ids(oa["local_grid"]))
            inventory.append(np.asarray(oa.get("inventory", np.array([0], dtype=np.float32)), dtype=np.float32).reshape(1))
            self_pos.append(np.asarray(oa.get("self_pos", np.array([0, 0], dtype=np.float32)), dtype=np.float32).reshape(2))
            gh = np.asarray(oa.get("goal_hint", np.zeros((model_cfg.goal_hint_dim,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            goal_hint.append(gh)
            recv_tokens.append(_get_recv_tokens(oa, model_cfg.max_messages, model_cfg.comm_token_limit))
            recv_from.append(_get_recv_from(oa, model_cfg.max_messages))
            action_masks.append(np.asarray(oa.get("action_mask", np.ones((self.cfg.action_dim,), dtype=np.float32)), dtype=np.float32))

        grid_t = torch.tensor(np.stack(grid_ids), dtype=torch.long, device=self.device)
        inv_t = torch.tensor(np.stack(inventory), dtype=torch.float32, device=self.device)
        pos_t = torch.tensor(np.stack(self_pos), dtype=torch.float32, device=self.device)
        # Pad goal_hint to batch max width.
        gh_max = max(x.shape[0] for x in goal_hint)
        gh_arr = np.zeros((len(goal_hint), gh_max), dtype=np.float32)
        for i, g in enumerate(goal_hint):
            gh_arr[i, : g.shape[0]] = g
        gh_t = torch.tensor(gh_arr, dtype=torch.float32, device=self.device)
        recv_tok_t = torch.tensor(np.stack(recv_tokens), dtype=torch.long, device=self.device)
        recv_from_t = torch.tensor(np.stack(recv_from), dtype=torch.long, device=self.device)
        mask_t = torch.tensor(np.stack(action_masks), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            out = self.model(
                grid_ids=grid_t,
                inventory=inv_t,
                self_pos=pos_t,
                goal_hint=gh_t,
                recv_tokens=recv_tok_t,
                recv_from=recv_from_t,
            )

        logits = out["action_logits"]
        invalid = (mask_t <= 0).bool()
        logits = logits.masked_fill(invalid, -1e9)

        if self.cfg.deterministic:
            actions = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()

        send_prob = torch.sigmoid(out["send_logit"])
        send_mask = send_prob >= float(self.cfg.send_threshold)
        if self.cfg.deterministic:
            msg_lens = out["msg_len_logits"].argmax(dim=-1)
            msg_toks = out["msg_token_logits"].argmax(dim=-1)
        else:
            len_dist = torch.distributions.Categorical(logits=out["msg_len_logits"])
            msg_lens = len_dist.sample()
            tok_dist = torch.distributions.Categorical(logits=out["msg_token_logits"])
            msg_toks = tok_dist.sample()

        out_actions: Dict[int, dict] = {}
        for i, aid in enumerate(agent_ids):
            if bool(send_mask[i].item()) and int(msg_lens[i].item()) > 0:
                L = min(int(msg_lens[i].item()), model_cfg.comm_token_limit)
                msg = msg_toks[i, :L].detach().cpu().tolist()
            else:
                msg = []
            out_actions[int(aid)] = {
                "action": int(actions[i].item()),
                "message_tokens": [int(t) for t in msg],
            }
        return out_actions
