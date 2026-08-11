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


class _ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class ResidualMLPEncoder(nn.Module):
    """Flat-observation encoder with residual blocks and normalization."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[_ResidualMLPBlock(hidden_dim) for _ in range(max(0, depth))]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        h = self.blocks(h)
        return self.output_norm(h)


class LocalCNNFlatEncoder(nn.Module):
    """Encoder for flattened observations with local FOV planes at the front."""

    def __init__(self, input_dim: int, fov_radius: int, hidden_dim: int = 128):
        super().__init__()
        self.fov_radius = int(fov_radius)
        self.window = self.fov_radius * 2 + 1
        self.area = self.window * self.window
        spatial_width = self.area * 4 + 3
        if input_dim <= spatial_width:
            raise ValueError(
                "local_cnn backbone needs flat obs with local grid/resource/node/energy "
                f"planes plus tail features; got input_dim={input_dim}, fov_radius={fov_radius}"
            )
        self.tail_dim = int(input_dim) - (self.area * 4)
        self.spatial = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, hidden_dim),
            nn.ReLU(),
        )
        self.tail = nn.Sequential(
            nn.Linear(self.tail_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.area
        grid = x[:, :n] / 12.0
        inventory_and_pos = x[:, n : n + 3]
        resources = x[:, n + 3 : n + 3 + n]
        nodes = x[:, n + 3 + n : n + 3 + (2 * n)]
        energy = x[:, n + 3 + (2 * n) : n + 3 + (3 * n)]
        remaining = x[:, n + 3 + (3 * n) :]
        spatial = torch.stack([grid, resources, nodes, energy], dim=1).reshape(
            x.shape[0],
            4,
            self.window,
            self.window,
        )
        tail = torch.cat([inventory_and_pos, remaining], dim=-1)
        return self.fuse(torch.cat([self.spatial(spatial), self.tail(tail)], dim=-1))


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
