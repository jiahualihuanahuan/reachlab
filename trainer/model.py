"""Recurrent actor-critic sized to sit beside a Q4 9B model on a 10 GB card."""

from __future__ import annotations

import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x + self.fc2(torch.tanh(self.fc1(x))))


class Policy(nn.Module):
    def __init__(self, width: int, blocks: int = 4, obs: int = 32, actions: int = 8):
        super().__init__()
        self.width = width
        self.obs = obs
        self.actions = actions
        self.stem = nn.Sequential(nn.Linear(obs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.blocks = nn.ModuleList([Residual(width) for _ in range(blocks)])
        self.gru = nn.GRU(width, width, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, actions))
        self.critic = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def forward_step(self, obs: torch.Tensor, hidden: torch.Tensor | None):
        x = self.stem(obs)
        for block in self.blocks:
            x = block(x)
        if hidden is None:
            hidden = torch.zeros(1, obs.shape[0], self.width, device=obs.device, dtype=obs.dtype)
        out, hidden = self.gru(x.unsqueeze(1), hidden)
        z = out.squeeze(1)
        return self.actor(z), self.critic(z).squeeze(-1), hidden


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
