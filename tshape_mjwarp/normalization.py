"""Frozen-on-rollout normalization without per-forward host scalar reads."""
import torch
from torch import nn


class RunningNormalizer(nn.Module):
    def __init__(self, size, clip=10., enabled=True):
        super().__init__()
        self.clip, self.enabled = clip, enabled
        self.register_buffer("mean", torch.zeros(size, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(size, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0., dtype=torch.float64))

    def forward(self, values):
        if not self.enabled:
            return values
        normalized = ((values-self.mean.to(values.dtype))/
                      (self.variance.to(values.dtype)+1e-6).sqrt()).clamp(-self.clip, self.clip)
        return torch.where(self.count > 0, normalized, values)

    @torch.no_grad()
    def update(self, values):
        if not self.enabled:
            return
        values = values.reshape(-1, self.mean.numel()).to(torch.float64)
        n = len(values)
        if not n:
            return
        delta = values.mean(0)-self.mean
        total = self.count+n
        variance = (self.variance*self.count + values.var(0, unbiased=False)*n
                    + delta.square()*self.count*n/total)/total
        self.mean.add_(delta*n/total)
        self.variance.copy_(variance.clamp_min(1e-6))
        self.count.copy_(total)
