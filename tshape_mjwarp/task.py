"""Medium observations, diagnostics and reward, vectorized over worlds with Torch."""
import numpy as np
import torch


class MediumTask:
    def __init__(self, oracle, config, device):
        self.config, self.door = config, oracle.doorway
        self.device = torch.device(device)
        m = oracle.model
        if not np.allclose(m.body_ipos[oracle.object_body, :2], 0):
            raise ValueError("current tensor geometry requires the baseline centered object COM")
        if not np.allclose(m.geom_quat[oracle.object_geom_ids], [1, 0, 0, 0]):
            raise ValueError("object boxes must align with the body frame")
        self.n = config.num_agents
        tensor = lambda v, dtype=torch.float32: torch.as_tensor(v, dtype=dtype, device=device)
        self.agent_qpos = tensor(oracle.agent_qpos, torch.long)
        self.agent_dofs = tensor(oracle.agent_dofs, torch.long)
        self.object_qpos = tensor(oracle.object_qpos, torch.long)
        self.object_dofs = tensor(oracle.object_dofs, torch.long)
        self.motor_ids = tensor(oracle.motor_ids, torch.long)
        self.motor_limits = tensor(oracle.motor_limits)
        self.agent_origin = tensor(m.body_pos[oracle.agent_bodies, :2])
        self.object_origin = tensor(m.body_pos[oracle.object_body, :2])
        self.lower, self.upper, self.radii = tensor(oracle.lower), tensor(oracle.upper), tensor(oracle.radii)
        self.goal, self.goal_half = tensor(config.goal_center), tensor(config.goal_half_size)
        points = []
        for g in oracle.object_geom_ids:
            x, y = m.geom_size[g, :2]
            points.extend(np.array([[-x, -y], [-x, y], [x, -y], [x, y]]) + m.geom_pos[g, :2])
        self.local_corners = tensor(np.asarray(points))
        self.portal_corners = tensor([[x, y] for x in (self.door.x_min, self.door.x_max)
                                     for y in (self.door.y_min, self.door.y_max)])
        self.portal_to_goal = tensor([self.door.route_distance(p, config.goal_center)
                                     for p in self.portal_corners.cpu().numpy()])
        self.other_agents = torch.tensor([[j for j in range(self.n) if i != j]
                                         for i in range(self.n)], device=device, dtype=torch.long).reshape(self.n, self.n-1)
        self.door_features = tensor(self.door.features(np.zeros(2), config.crossing_margin))
        self.observation_size, self.state_size = 23 + 4*self.n, 22 + 4*self.n

    def geometry(self, qpos, qvel):
        positions = qpos[:, self.agent_qpos] + self.agent_origin
        velocities = qvel[:, self.agent_dofs]
        obj = qpos[:, self.object_qpos[:2]] + self.object_origin
        yaw = qpos[:, self.object_qpos[2]]
        local = self.local_corners
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        corners = torch.stack((c*local[:, 0] - s*local[:, 1],
                               s*local[:, 0] + c*local[:, 1]), dim=-1) + obj[:, None]
        return positions, velocities, obj, yaw, qvel[:, self.object_dofs], corners

    def visible(self, start, end):
        dx = end[..., 0] - start[..., 0]
        vertical = dx.abs() < 1e-12
        safe_dx = torch.where(vertical, torch.ones_like(dx), dx)
        a, b = (self.door.x_min-start[..., 0])/safe_dx, (self.door.x_max-start[..., 0])/safe_dx
        lo, hi = torch.minimum(a, b).clamp_min(0), torch.maximum(a, b).clamp_max(1)
        y0 = start[..., 1] + lo*(end[..., 1]-start[..., 1])
        y1 = start[..., 1] + hi*(end[..., 1]-start[..., 1])
        ordinary = (lo > hi) | ((torch.minimum(y0, y1) >= self.door.y_min-1e-7)
                               & (torch.maximum(y0, y1) <= self.door.y_max+1e-7))
        parallel = ((start[..., 0] < self.door.x_min) | (start[..., 0] > self.door.x_max)
                    | ((torch.minimum(start[..., 1], end[..., 1]) >= self.door.y_min-1e-7)
                       & (torch.maximum(start[..., 1], end[..., 1]) <= self.door.y_max+1e-7)))
        return torch.where(vertical, parallel, ordinary)

    def distance(self, obj):
        direct = (obj-self.goal).norm(dim=-1)
        candidates = (obj[:, None]-self.portal_corners).norm(dim=-1) + self.portal_to_goal
        candidates = candidates.masked_fill(~self.visible(obj[:, None], self.portal_corners), torch.inf)
        route = candidates.amin(dim=-1)
        return torch.where(self.visible(obj, self.goal) | ~torch.isfinite(route), direct, route)

    def boundary_penetration(self, qpos, qvel):
        pos, _, _, _, _, corners = self.geometry(qpos, qvel)
        return torch.stack(((self.lower-corners).flatten(1).amax(1),
                            (corners-self.upper).flatten(1).amax(1),
                            (self.lower-(pos-self.radii[None, :, None])).flatten(1).amax(1),
                            ((pos+self.radii[None, :, None])-self.upper).flatten(1).amax(1)), dim=1).amax(1).clamp_min(0)

    def features(self, qpos, qvel, contacts, steps, success):
        pos, vel, obj, yaw, obj_vel, corners = self.geometry(qpos, qvel)
        e = len(qpos)
        remaining = (1-steps.float()/self.config.max_steps).clamp_min(0)
        hold = success.float()
        pose_features = torch.stack((yaw.sin(), yaw.cos()), -1)
        expand = lambda x: x[:, None].expand(-1, self.n, -1)
        clearance = torch.cat((pos-self.lower-self.radii[None, :, None],
                                self.upper-pos-self.radii[None, :, None]), -1)
        others = pos[:, self.other_agents] - pos[:, :, None]
        door = self.door_features.expand(e, self.n, -1).clone()
        door[..., :2] -= pos
        door[..., 4] -= pos[..., 0]
        obs = torch.cat((pos, vel, obj[:, None]-pos, expand(pose_features), expand(obj_vel[:, :2]),
                         expand(obj_vel[:, 2:3]), self.goal-pos, self.goal_half.expand(e, self.n, -1),
                         others.flatten(2), vel[:, self.other_agents].flatten(2), clearance,
                         contacts.float()[..., None], remaining[:, None, None].expand(-1, self.n, -1),
                         hold[:, None, None].expand(-1, self.n, -1), door), -1)
        state = torch.cat((obj, pose_features, obj_vel[:, :2], obj_vel[:, 2:3], self.goal.expand(e, -1),
                           self.goal_half.expand(e, -1), pos.flatten(1), vel.flatten(1),
                           self.lower.expand(e, -1), self.upper.expand(e, -1), remaining[:, None],
                           hold[:, None], self.door_features.expand(e, -1)), -1)
        crossing = corners[..., 0].amin(1) - self.door.x_max-self.config.crossing_margin
        return obs, state, crossing

    def reward(self, previous, distance, success, failed):
        cfg = self.config
        parts = torch.stack((cfg.distance_weight*(1-torch.tanh(distance/cfg.distance_scale)),
                             cfg.progress_weight*(previous-distance), cfg.success_reward*success.float(),
                             torch.zeros_like(distance)), -1)
        parts = torch.where(failed[:, None], torch.zeros_like(parts), parts)
        parts[:, 3] = -cfg.failure_penalty*failed.float()
        return parts.sum(-1), parts
