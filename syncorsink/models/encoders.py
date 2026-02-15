from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, depth: int = 2):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class TransformerEncoder(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(token_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj(tokens)
        x = self.encoder(x, src_key_padding_mask=attn_mask)
        return x[:, 0] if x.dim() == 3 else x
