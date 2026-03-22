from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CommMATConfig:
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
    comm_disabled: bool = False  # ablation: zero out message inputs, disable comm heads


class CommMATModel(nn.Module):
    """
    DTDE communication-aware multi-agent transformer backbone.

    It encodes per-agent local observation tokens, received message tokens,
    and lightweight self/hint state into a single transformer sequence.
    """

    def __init__(self, cfg: CommMATConfig):
        super().__init__()
        self.cfg = cfg

        h = cfg.hidden_dim
        self.tile_embed = nn.Embedding(cfg.tile_vocab_size, h)
        self.comm_token_embed = nn.Embedding(cfg.comm_vocab_size, h)
        self.sender_embed = nn.Embedding(cfg.max_agents + 1, h)
        self.goal_hint_proj = nn.Linear(cfg.goal_hint_dim, h)
        self.self_proj = nn.Linear(3, h)  # inventory, x, y
        self.pos_embed = nn.Embedding(4096, h)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, h))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        self.action_head = nn.Linear(h, cfg.action_dim)
        self.send_head = nn.Linear(h, 1)
        self.msg_len_head = nn.Linear(h, cfg.comm_token_limit + 1)
        self.msg_token_head = nn.Linear(h, cfg.comm_token_limit * cfg.comm_vocab_size)
        self.value_head = nn.Linear(h, 1)

        nn.init.normal_(self.cls_token, std=0.02)

    def _add_pos(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        idx = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
        return x + self.pos_embed(idx)

    def forward(
        self,
        grid_ids: torch.Tensor,
        inventory: torch.Tensor,
        self_pos: torch.Tensor,
        goal_hint: torch.Tensor,
        recv_tokens: torch.Tensor,
        recv_from: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            grid_ids: (B, H, W) tile ids
            inventory: (B, 1)
            self_pos: (B, 2)
            goal_hint: (B, G)
            recv_tokens: (B, M, L)
            recv_from: (B, M)
        """
        b = grid_ids.shape[0]
        h = self.cfg.hidden_dim

        safe_grid = grid_ids.long().clamp(min=0, max=self.cfg.tile_vocab_size - 1)
        obs_tok = self.tile_embed(safe_grid.view(b, -1))  # (B, H*W, D)

        # Per-message token: average embedded message tokens + sender embedding.
        if self.cfg.comm_disabled:
            # Ablation: zero out message tokens so transformer sees no comm input
            msg_tok = torch.zeros(b, self.cfg.max_messages, h, device=grid_ids.device)
        else:
            safe_recv = recv_tokens.long().clamp(min=0, max=self.cfg.comm_vocab_size - 1)
            msg_emb = self.comm_token_embed(safe_recv)  # (B, M, L, D)
            msg_tok = msg_emb.mean(dim=2)  # (B, M, D)
            sender_ids = recv_from.long().clamp(min=-1, max=self.cfg.max_agents) + 1
            msg_tok = msg_tok + self.sender_embed(sender_ids)

        self_vec = torch.cat([inventory.float(), self_pos.float()], dim=-1)  # (B,3)
        self_tok = self.self_proj(self_vec).unsqueeze(1)  # (B,1,D)

        if goal_hint.shape[-1] < self.cfg.goal_hint_dim:
            pad = torch.zeros(
                (b, self.cfg.goal_hint_dim - goal_hint.shape[-1]),
                device=goal_hint.device,
                dtype=goal_hint.dtype,
            )
            goal_hint = torch.cat([goal_hint, pad], dim=-1)
        elif goal_hint.shape[-1] > self.cfg.goal_hint_dim:
            goal_hint = goal_hint[:, : self.cfg.goal_hint_dim]
        hint_tok = self.goal_hint_proj(goal_hint.float()).unsqueeze(1)  # (B,1,D)

        cls = self.cls_token.expand(b, -1, -1)  # (B,1,D)
        seq = torch.cat([cls, self_tok, hint_tok, obs_tok, msg_tok], dim=1)  # (B,N,D)
        seq = self._add_pos(seq)
        z = self.encoder(seq)
        pooled = z[:, 0]  # CLS

        action_logits = self.action_head(pooled)
        send_logit = self.send_head(pooled).squeeze(-1)
        msg_len_logits = self.msg_len_head(pooled)
        msg_token_logits = self.msg_token_head(pooled).view(
            b, self.cfg.comm_token_limit, self.cfg.comm_vocab_size
        )
        value = self.value_head(pooled).squeeze(-1)
        return {
            "action_logits": action_logits,
            "send_logit": send_logit,
            "msg_len_logits": msg_len_logits,
            "msg_token_logits": msg_token_logits,
            "value": value,
        }
