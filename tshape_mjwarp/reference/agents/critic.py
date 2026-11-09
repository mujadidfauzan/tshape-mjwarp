"""Centralized scalar value per team state, used only during training."""

from torch import nn

from .actor import mlp


class CentralizedCritic(nn.Module):
    def __init__(self, state_size, hidden_size=256):
        super().__init__()
        self.network = mlp(state_size, hidden_size, 1)

    def forward(self, states):
        return self.network(states).squeeze(-1)
