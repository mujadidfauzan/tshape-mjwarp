"""Shared-policy MAPPO with a scalar centralized team critic."""

import numpy as np
import torch
from torch import nn

from .reference.agents.actor import SharedActor
from .reference.agents.critic import CentralizedCritic
from .training_config import load_mappo_config
from .normalization import RunningNormalizer

def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return device


class MAPPO(nn.Module):
    def __init__(self, observation_size, state_size, action_size=2, num_agents=4, config=None):
        super().__init__()
        self.config = cfg = load_mappo_config(config)
        self.device = resolve_device(cfg.device)
        self.dimensions = {"observation_size": observation_size, "state_size": state_size,
                           "action_size": action_size, "num_agents": num_agents}
        self.actor = SharedActor(observation_size, action_size, cfg.actor_hidden_size, cfg.initial_log_std)
        self.critic = CentralizedCritic(state_size, cfg.critic_hidden_size)
        self.obs_normalizer = RunningNormalizer(observation_size, cfg.normalization_clip, cfg.normalize_inputs)
        self.state_normalizer = RunningNormalizer(state_size, cfg.normalization_clip, cfg.normalize_inputs)
        self.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, eps=1e-5)

    def tensor(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, observations, deterministic=False):
        # Actor execution uses only its input; the centralized critic is not called.
        inputs = self.obs_normalizer(self.tensor(observations))
        actions, latent, log_probs = self.actor.sample(inputs, deterministic)
        return actions, latent, log_probs

    @torch.no_grad()
    def value(self, state):
        return self.critic(self.state_normalizer(self.tensor(state)))

    def update(self, buffer, rng):
        cfg = self.config
        batch = {name: self.tensor(value) for name, value in buffer.batch(cfg.gamma, cfg.gae_lambda).items()}
        count = buffer.size * buffer.num_envs
        # Input statistics stay fixed across BOTH rollout collection and all epochs.
        observations = self.obs_normalizer(batch["observations"])
        states = self.state_normalizer(batch["states"])
        rows = []
        early_stop = False
        for epoch in range(cfg.ppo_epochs):
            for indices in torch.tensor_split(torch.randperm(count, device=self.device, generator=rng), min(cfg.num_minibatches, count)):
                ids = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                new_log_probs, entropy = self.actor.evaluate(observations[ids], batch["pre_tanh"][ids])
                log_ratio = new_log_probs - batch["old_log_probs"][ids]
                ratio = log_ratio.exp()
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                if not torch.isfinite(approx_kl):
                    raise FloatingPointError("non-finite policy KL; inspect training settings")
                if approx_kl.item() > cfg.target_kl:
                    early_stop = True
                    break
                advantage = batch["advantages"][ids, None]
                surrogate = torch.minimum(ratio * advantage,
                                          ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * advantage)
                policy_loss = -surrogate.mean()
                actor_loss = policy_loss - cfg.entropy_coef * entropy.mean()
                predicted = self.critic(states[ids])
                squared_error = (predicted - batch["returns"][ids]).square()
                if cfg.value_clip > 0:
                    old = batch["old_values"][ids]
                    clipped = old + (predicted - old).clamp(-cfg.value_clip, cfg.value_clip)
                    squared_error = torch.maximum(squared_error, (clipped - batch["returns"][ids]).square())
                value_loss = 0.5 * squared_error.mean()
                if not torch.isfinite(actor_loss) or not torch.isfinite(value_loss):
                    raise FloatingPointError("non-finite PPO loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm,
                                                     error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (cfg.value_coef * value_loss).backward()
                critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm,
                                                      error_if_nonfinite=True)
                self.critic_optimizer.step()
                rows.append({"policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                             "entropy": float(entropy.mean().detach()), "approx_kl": float(approx_kl.detach()),
                             "clip_fraction": float(((ratio - 1).abs() > cfg.clip_ratio).float().mean().detach()),
                             "actor_grad_norm": float(actor_grad), "critic_grad_norm": float(critic_grad)})
            if early_stop:
                break
        if not rows:
            raise RuntimeError("PPO performed no updates; rollout likelihoods or normalization changed")
        with torch.no_grad():
            predicted = self.critic(states)
            variance = batch["returns"].var(unbiased=False)
            explained = (1 - (batch["returns"] - predicted).var(unbiased=False) / variance).item() if variance > 1e-8 else None
            # Only after PPO finishes: the next rollout will use the new statistics.
            self.obs_normalizer.update(batch["observations"])
            self.state_normalizer.update(batch["states"])
        return {**{name: float(np.mean([row[name] for row in rows])) for name in rows[0]},
                "optimizer_updates": len(rows), "kl_early_stop": early_stop,
                "explained_variance": explained, "team_transitions": count,
                "actor_samples": count * self.dimensions["num_agents"]}

