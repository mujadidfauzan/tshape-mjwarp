import unittest
import torch
from tshape_mjwarp.rollout import compute_gae, RolloutBuffer
from tshape_mjwarp.mappo import MAPPO
from tshape_mjwarp.training_config import MAPPOConfig, load_mappo_config


class RolloutTests(unittest.TestCase):
    def test_default_agent_config_needs_no_yaml_file(self):
        torch.set_num_threads(1)
        self.assertEqual(load_mappo_config(), MAPPOConfig())
        self.assertEqual(load_mappo_config(actor_hidden_size=8).actor_hidden_size, 8)
        agent = MAPPO(3, 5, num_agents=4)
        actions, _, _ = agent.act(torch.zeros(2, 4, 3))
        self.assertEqual(actions.shape, (2, 4, 2))

    def test_gae_separates_worlds_and_timeouts(self):
        rewards = torch.tensor([[1., 10.], [100., 20.]])
        values = torch.zeros_like(rewards)
        nxt = torch.tensor([[5., 0.], [0., 0.]])
        term = torch.tensor([[False, False], [True, True]])
        trunc = torch.tensor([[True, False], [False, False]])
        adv, returns = compute_gae(rewards, values, nxt, term, trunc, .9, 1.)
        torch.testing.assert_close(adv, torch.tensor([[5.5, 28.], [100., 20.]]))

    def test_single_sample_drives_actor_and_normalization_is_frozen(self):
        torch.set_num_threads(1)
        cfg = MAPPOConfig(rollout_steps=1, ppo_epochs=1, entropy_coef=0.,
                          actor_hidden_size=8, critic_hidden_size=8)
        agent = MAPPO(3, 5, num_agents=4, config=cfg)
        obs, state = torch.ones(1, 4, 3), torch.zeros(1, 5)
        action, latent, logp = agent.act(obs)
        normalizer_count = agent.obs_normalizer.count.clone()
        buffer = RolloutBuffer(1, 1, agent.dimensions, "cpu")
        buffer.add(observations=obs, states=state, pre_tanh=latent, old_log_probs=logp,
                   rewards=torch.tensor([5.]), old_values=agent.value(state), next_values=torch.zeros(1),
                   terminated=torch.ones(1, dtype=torch.bool), truncated=torch.zeros(1, dtype=torch.bool))
        before = [p.clone() for p in agent.actor.parameters()]
        torch.testing.assert_close(agent.obs_normalizer.count, normalizer_count)
        metrics = agent.update(buffer, torch.Generator().manual_seed(42))
        self.assertGreater(metrics["actor_grad_norm"], 0.)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(before, agent.actor.parameters())))
        self.assertEqual(agent.obs_normalizer.count.item(), 4.)
