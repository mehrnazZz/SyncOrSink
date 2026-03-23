"""TarMAC: Targeted Multi-Agent Communication.

Each agent produces a message vector + attention key from its observation.
Messages are aggregated via soft attention (receiver query × sender keys)
so agents learn WHO to communicate with, not just WHAT to say.

Reference: Das et al., "TarMAC: Targeted Multi-Agent Communication", ICML 2019.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TarMACConfig:
    obs_dim: int = 423
    action_dim: int = 8
    hidden_dim: int = 128
    msg_dim: int = 32       # dimension of message vectors
    key_dim: int = 32       # dimension of attention key/query
    n_rounds: int = 1       # number of communication rounds per step
    value_head: bool = True  # include value head for PPO


class TarMACAgent(nn.Module):
    """Single-agent TarMAC module.

    Forward pass:
      1. Encode observation → hidden state
      2. Produce message vector + attention key
      3. Produce query for receiving messages
      4. After communication round: (hidden + aggregated msg) → action + value
    """

    def __init__(self, cfg: TarMACConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )

        # Message generation: hidden → message vector + attention key
        self.msg_head = nn.Linear(h, cfg.msg_dim)
        self.key_head = nn.Linear(h, cfg.key_dim)
        self.query_head = nn.Linear(h, cfg.key_dim)

        # Post-communication layers
        self.post_comm = nn.Sequential(
            nn.Linear(h + cfg.msg_dim, h),
            nn.ReLU(),
        )

        # Action head
        self.action_head = nn.Linear(h, cfg.action_dim)

        # Value head
        if cfg.value_head:
            self.value_fn = nn.Linear(h, 1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation → hidden state. Shape: (B, obs_dim) → (B, hidden_dim)."""
        return self.encoder(obs)

    def generate_message(self, hidden: torch.Tensor):
        """Generate message vector and attention key from hidden state.

        Returns:
            msg: (B, msg_dim) — message content
            key: (B, key_dim) — attention key for receivers to attend to
            query: (B, key_dim) — attention query for receiving messages
        """
        msg = self.msg_head(hidden)
        key = self.key_head(hidden)
        query = self.query_head(hidden)
        return msg, key, query

    def act(self, hidden: torch.Tensor, agg_msg: torch.Tensor):
        """Produce action logits and value from hidden state + aggregated message.

        Args:
            hidden: (B, hidden_dim)
            agg_msg: (B, msg_dim) — attention-weighted sum of received messages
        """
        combined = torch.cat([hidden, agg_msg], dim=-1)
        post = self.post_comm(combined)
        action_logits = self.action_head(post)
        value = self.value_fn(post).squeeze(-1) if self.cfg.value_head else None
        return action_logits, value


class TarMACModel(nn.Module):
    """Multi-agent TarMAC with shared parameters.

    All agents share the same TarMACAgent network. Communication happens
    via attention-weighted message passing between agents.
    """

    def __init__(self, cfg: TarMACConfig):
        super().__init__()
        self.cfg = cfg
        self.agent = TarMACAgent(cfg)
        self.scale = cfg.key_dim ** -0.5

    def forward(self, obs_all: torch.Tensor):
        """Forward pass for all agents simultaneously.

        Args:
            obs_all: (N, obs_dim) — observations for all N agents

        Returns:
            dict with:
                action_logits: (N, action_dim)
                value: (N,) if value_head enabled
                messages: (N, msg_dim) — final message vectors (for logging)
                attention: (N, N) — attention weights (who talked to whom)
        """
        N = obs_all.shape[0]

        # Step 1: Encode all agents' observations
        hidden = self.agent.encode(obs_all)  # (N, hidden_dim)

        # Communication rounds
        attn_weights = None
        for _ in range(self.cfg.n_rounds):
            # Step 2: Generate messages and attention keys/queries
            msgs, keys, queries = self.agent.generate_message(hidden)
            # msgs: (N, msg_dim), keys: (N, key_dim), queries: (N, key_dim)

            # Step 3: Compute attention weights
            # Each agent i attends to all other agents j
            # attn[i,j] = softmax(query_i · key_j / sqrt(d))
            attn_logits = torch.matmul(queries, keys.T) * self.scale  # (N, N)

            # Mask self-attention (agent shouldn't attend to own message)
            mask = torch.eye(N, device=obs_all.device).bool()
            attn_logits = attn_logits.masked_fill(mask, -1e9)

            attn_weights = F.softmax(attn_logits, dim=-1)  # (N, N)

            # Step 4: Aggregate messages via attention
            agg_msg = torch.matmul(attn_weights, msgs)  # (N, msg_dim)

            # Update hidden for next round (if multiple rounds)
            if self.cfg.n_rounds > 1:
                hidden = self.agent.post_comm(torch.cat([hidden, agg_msg], dim=-1))

        # Step 5: Produce actions and values
        action_logits, value = self.agent.act(hidden, agg_msg)

        result = {
            "action_logits": action_logits,
            "messages": msgs,
            "attention": attn_weights,
        }
        if value is not None:
            result["value"] = value
        return result
