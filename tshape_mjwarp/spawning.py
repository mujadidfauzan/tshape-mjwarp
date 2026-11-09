"""Device-side bounded sampling with independent, checkpointable world RNGs."""
import torch


class LayoutSampler:
    def __init__(self, oracle, task, num_envs, seed):
        self.task, self.cfg = task, task.config
        self.device = task.device
        self.rng = (torch.arange(num_envs, device=self.device, dtype=torch.int64)*1000003 + seed) % 2**32
        self.base_qpos = torch.as_tensor(oracle.data.qpos.copy(), device=self.device, dtype=torch.float32)
        m = oracle.model
        self.box_centers = torch.as_tensor(m.geom_pos[oracle.object_geom_ids, :2], device=self.device, dtype=torch.float32)
        self.box_half = torch.as_tensor(m.geom_size[oracle.object_geom_ids, :2], device=self.device, dtype=torch.float32)
        from .reference.envs.tshape_env import TShapeParallelEnv
        reference = TShapeParallelEnv({**self.cfg.to_dict(), "randomize_object_position": False,
                                      "randomize_object_orientation": False})
        try:
            reference.reset(seed=0)
            self.reference_positions = torch.as_tensor(reference.physics.get_agents_state()[0], device=self.device)
        finally:
            reference.close()
        self.reference_positions = self.reference_positions.float()
        self.obstacles = [(torch.as_tensor(m.geom_pos[g, :2], device=self.device),
                           torch.as_tensor(m.geom_size[g, :2], device=self.device)) for g in oracle.obstacles]

    def uniform(self, mask, shape):
        count = 1
        for d in shape:
            count *= d
        x = (self.rng[:, None] + torch.arange(1, count+1, device=self.device)*0x9e3779b9) % 2**32
        x = ((x ^ (x >> 16))*0x45d9f3b) % 2**32
        x = ((x ^ (x >> 16))*0x45d9f3b) % 2**32
        x = x ^ (x >> 16)
        self.rng.copy_(torch.where(mask, (self.rng+count) % 2**32, self.rng))
        return (x.float()/2**32).reshape(len(mask), *shape)

    def gap(self, points, xy, yaw):
        delta = points-xy[:, None]
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        local = torch.stack((c*delta[..., 0]+s*delta[..., 1], -s*delta[..., 0]+c*delta[..., 1]), -1)
        outside = (local[:, :, None]-self.box_centers).abs()-self.box_half
        return outside.clamp_min(0).norm(dim=-1).amin(-1)

    def free(self, points, radius):
        valid = ((points-radius >= self.task.lower) & (points+radius <= self.task.upper)).all(-1)
        for center, half in self.obstacles:
            valid &= ((points-center).abs()-half).clamp_min(0).norm(dim=-1) >= (radius+.01).squeeze(-1)
        return valid

    def sample(self, mask):
        cfg, task = self.cfg, self.task
        e, k = len(mask), 128
        pose = torch.tensor(cfg.object_pose, device=self.device, dtype=torch.float32).expand(e, k, 3).clone()
        if cfg.randomize_object_position:
            pose[..., :2] += (2*self.uniform(mask, (k, 2))-1)*torch.tensor(cfg.object_position_range, device=self.device)
        if cfg.randomize_object_orientation:
            lo, hi = cfg.object_yaw_range
            pose[..., 2] = lo+(hi-lo)*self.uniform(mask, (k,))
        q = self.base_qpos.expand(e, k, -1).clone()
        q[..., task.object_qpos[:2]] = pose[..., :2]-task.object_origin
        q[..., task.object_qpos[2]] = pose[..., 2]
        flat = q.flatten(0, 1)
        corners = task.geometry(flat, torch.zeros_like(flat))[-1].reshape(e, k, -1, 2)
        valid = ((corners >= task.lower-1e-7) & (corners <= task.upper+1e-7)).all((-1, -2))
        valid &= corners[..., 0].amax(-1) < task.door.x_min
        reference_delta = self.reference_positions-torch.tensor(cfg.object_pose[:2], device=self.device)
        formation = None
        if not cfg.randomize_agents:
            yaw = pose[..., 2]-cfg.object_pose[2]
            c, s = yaw.cos()[..., None], yaw.sin()[..., None]
            formation = torch.stack((c*reference_delta[:, 0]-s*reference_delta[:, 1],
                                     s*reference_delta[:, 0]+c*reference_delta[:, 1]), -1)+pose[..., None, :2]
            valid &= self.free(formation, task.radii[None, None, :, None]).all(-1)
            gap = self.gap(formation.flatten(0, 1), pose[..., :2].flatten(0, 1), pose[..., 2].flatten())
            valid &= (gap.reshape(e, k, task.n) >= task.radii).all(-1)
        torch._assert_async((valid.any(-1) | ~mask).all(), "No valid object pose in 128 candidates; adjust spawn ranges")
        choice = valid.int().argmax(-1)
        row = torch.arange(e, device=self.device)
        q = q[row, choice]
        selected = pose[row, choice]
        if formation is not None:
            positions = formation[row, choice]
        else:
            corners = corners[row, choice]
            positions = torch.zeros((e, task.n, 2), device=self.device)
            for i in range(task.n):
                radius = task.radii[i]
                low = torch.maximum(corners.amin(1)-radius-cfg.spawn_gap_max, task.lower+radius+.01)
                high = torch.minimum(corners.amax(1)+radius+cfg.spawn_gap_max, task.upper-radius-.01)
                candidates = low[:, None]+self.uniform(mask, (512, 2))*(high-low)[:, None]
                gap = self.gap(candidates, selected[:, :2], selected[:, 2])-radius
                okay = (gap >= cfg.spawn_gap_min) & (gap <= cfg.spawn_gap_max) & self.free(candidates, radius)
                if i:
                    separation = (candidates[:, :, None]-positions[:, None, :i]).norm(dim=-1)
                    okay &= (separation >= radius+task.radii[None, None, :i]+.01).all(-1)
                torch._assert_async((okay.any(-1) | ~mask).all(), "Agent placement exhausted 512 candidates; widen spawn band")
                positions[:, i] = candidates[row, okay.int().argmax(-1)]
        q[:, task.agent_qpos] = positions-task.agent_origin
        return q
