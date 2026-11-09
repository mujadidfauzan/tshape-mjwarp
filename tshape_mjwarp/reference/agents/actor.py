"""Shared continuous actor with consistent tanh-squashed Gaussian likelihoods."""

import math

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


def mlp(input_size, hidden_size, output_size, output_gain=1.0):
    layers = []
    for in_size in (input_size, hidden_size):
        layer = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal_(layer.weight, math.sqrt(2))
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.Tanh()))
    output = nn.Linear(hidden_size, output_size)
    nn.init.orthogonal_(output.weight, output_gain)
    nn.init.zeros_(output.bias)
    return nn.Sequential(*layers, output)


class SharedActor(nn.Module):
    def __init__(self, observation_size, action_size=2, hidden_size=128, initial_log_std=-0.5):
        super().__init__()
        self.mean = mlp(observation_size, hidden_size, action_size, output_gain=0.01)
        self.log_std = nn.Parameter(torch.full((action_size,), float(initial_log_std)))

    def distribution(self, observations):
        mean = self.mean(observations)
        std = self.log_std.clamp(-5, 2).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def squashed_log_prob(distribution, pre_tanh):
        # log(1 - tanh(z)^2), evaluated without cancellation at saturated actions.
        log_jacobian = 2 * (math.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))
        return (distribution.log_prob(pre_tanh) - log_jacobian).sum(dim=-1)

    def sample(self, observations, deterministic=False):
        distribution = self.distribution(observations)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        actions = pre_tanh.tanh()
        log_prob = self.squashed_log_prob(distribution, pre_tanh)
        return actions, pre_tanh, log_prob

    def evaluate(self, observations, pre_tanh):
        distribution = self.distribution(observations)
        log_prob = self.squashed_log_prob(distribution, pre_tanh)
        # Squashed entropy has no simple analytic form. Use a fresh reparameterized
        # sample for its Monte Carlo estimate, rather than the base Normal entropy.
        entropy = -self.squashed_log_prob(distribution, distribution.rsample())
        return log_prob, entropy
