"""Dense batched Medium API. Terminal features are read before explicit reset."""
import torch

from .physics import oracle, ReferencePhysics, WarpPhysics
from .task import MediumTask
from .spawning import LayoutSampler


class BatchedEnvironment:
    def __init__(self, config, simulation, seed=42):
        self.config, self.simulation = config, simulation
        self.device = torch.device(simulation.device)
        self.oracle = oracle(config)
        self.task = MediumTask(self.oracle, config, self.device)
        self.physics = (ReferencePhysics if simulation.backend == "reference" else WarpPhysics)(config, simulation, self.task)
        self.sampler = LayoutSampler(self.oracle, self.task, simulation.num_envs, seed)
        e = simulation.num_envs
        self.steps = torch.zeros(e, device=self.device, dtype=torch.long)
        self.returns = torch.zeros(e, device=self.device)
        self.obstacle_events = torch.zeros_like(self.steps)
        self.path_length = torch.zeros_like(self.returns)
        self.max_penetration = torch.zeros_like(self.returns)
        self.done = torch.zeros(e, device=self.device, dtype=torch.bool)
        self.initial_qpos = self.physics.qpos.clone()
        self.dimensions = dict(observation_size=self.task.observation_size, state_size=self.task.state_size,
                               num_agents=config.num_agents, action_size=2)
        self.observations = torch.zeros((e, config.num_agents, self.task.observation_size), device=self.device)
        self.states = torch.zeros((e, self.task.state_size), device=self.device)
        self.reset()

    @torch.no_grad()
    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones_like(self.done)
        else:
            mask = mask.clone()
        qpos = self.sampler.sample(mask)
        self.physics.reset(mask, qpos)
        self.initial_qpos.copy_(torch.where(mask[:, None], qpos, self.initial_qpos))
        self.steps.masked_fill_(mask, 0)
        self.returns.masked_fill_(mask, 0.)
        self.obstacle_events.masked_fill_(mask, 0)
        self.path_length.masked_fill_(mask, 0.)
        self.max_penetration.masked_fill_(mask, 0.)
        self.done.masked_fill_(mask, False)
        obs, state, _ = self.task.features(self.physics.qpos, self.physics.qvel, self.physics.contacts,
                                           self.steps, torch.zeros_like(mask))
        self.observations.copy_(torch.where(mask[:, None, None], obs, self.observations))
        self.states.copy_(torch.where(mask[:, None], state, self.states))
        return self.observations, self.states

    @torch.no_grad()
    def step(self, actions):
        if actions.shape != (len(self.done), self.config.num_agents, 2):
            raise ValueError("actions must have shape [num_envs, num_agents, 2]")
        torch._assert_async(torch.isfinite(actions).all(), "non-finite actions")
        torch._assert_async((~self.done).all(), "reset completed worlds before stepping again")
        previous = self.task.distance(self.task.geometry(self.physics.qpos, self.physics.qvel)[2])
        previous = torch.nan_to_num(previous, nan=0., posinf=0., neginf=0.)
        report = self.physics.step(actions.clamp(-1, 1))
        self.steps += 1
        failed = report["failed"]
        _, _, _, _, _, corners = self.task.geometry(self.physics.qpos, self.physics.qvel)
        success = (corners[..., 0].amin(1) >= self.task.door.x_max+self.config.crossing_margin-1e-7) & ~failed
        term = failed | success
        trunc = (self.steps >= self.config.max_steps) & ~term
        distance = self.task.distance(self.task.geometry(self.physics.qpos, self.physics.qvel)[2])
        distance = torch.where(failed, previous, distance)
        reward, parts = self.task.reward(previous, distance, success, failed)
        self.returns += reward
        self.obstacle_events += report["obstacle_events"]
        self.path_length += report["path_length"]
        self.max_penetration = torch.maximum(self.max_penetration, report["penetration"])
        obs, state, clearance = self.task.features(self.physics.qpos, self.physics.qvel, self.physics.contacts,
                                                   self.steps, success)
        self.observations.copy_(torch.where(failed[:, None, None], self.observations, obs))
        self.states.copy_(torch.where(failed[:, None], self.states, state))
        self.done.copy_(term | trunc)
        info = dict(success=success, stable=~failed, failure_code=report["failure_code"],
                    episode_return=self.returns.clone(), episode_steps=self.steps.clone(),
                    completion_time_s=torch.where(success, self.steps*self.oracle.control_dt, torch.nan),
                    crossing_clearance_m=clearance, distance_to_goal=distance,
                    obstacle_collision_events=self.obstacle_events.clone(),
                    object_path_length_m=self.path_length.clone(), max_penetration_m=self.max_penetration.clone(),
                    reward_components=parts)
        return self.observations, self.states, reward, term, trunc, info

    def close(self):
        # No global CUDA cleanup: training/evaluation may coexist in one process.
        pass
