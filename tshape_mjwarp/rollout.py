"""Device-resident team rollouts; GAE never crosses worlds or reset episodes."""
import torch


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    if any(v.shape != rewards.shape for v in (values, next_values, terminated, truncated)) or rewards.ndim != 2:
        raise ValueError("GAE requires matching [time, world] tensors")
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma*(~terminated[t])*next_values[t]-values[t]
        carry = delta + gamma*lam*(~(terminated[t] | truncated[t]))*carry
        advantages[t] = carry
    return advantages, advantages+values


class RolloutBuffer:
    def __init__(self, capacity, num_envs, dimensions, device):
        self.capacity, self.num_envs, self.size = capacity, num_envs, 0
        n, o, s, a = (dimensions[k] for k in ("num_agents", "observation_size", "state_size", "action_size"))
        shapes = dict(observations=(n, o), states=(s,), pre_tanh=(n, a), old_log_probs=(n,),
                      rewards=(), old_values=(), next_values=(), terminated=(), truncated=())
        self.data = {key: torch.empty((capacity, num_envs, *shape), device=device,
                                     dtype=torch.bool if key in ("terminated", "truncated") else torch.float32)
                     for key, shape in shapes.items()}

    def add(self, **items):
        if self.size >= self.capacity or set(items) != set(self.data):
            raise ValueError("rollout full or transition fields incorrect")
        for key, value in items.items():
            if value.shape != self.data[key].shape[1:]:
                raise ValueError(f"invalid transition shape: {key}")
            self.data[key][self.size].copy_(value.detach())
        self.size += 1

    def batch(self, gamma, gae_lambda):
        if not self.size:
            raise ValueError("empty rollout")
        d = {k: v[:self.size] for k, v in self.data.items()}
        advantage, returns = compute_gae(d["rewards"], d["old_values"], d["next_values"],
                                         d["terminated"], d["truncated"], gamma, gae_lambda)
        if advantage.numel() > 1:
            advantage = (advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
        keys = ("observations", "states", "pre_tanh", "old_log_probs", "old_values")
        batch = {k: d[k].flatten(0, 1) for k in keys}
        batch.update(advantages=advantage.flatten(), returns=returns.flatten())
        if not all(torch.isfinite(v).all() for v in batch.values()):
            raise FloatingPointError("non-finite rollout")
        return batch
