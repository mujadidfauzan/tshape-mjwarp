"""Explicit CPU oracle and MJWarp backend. No silent fallback when CUDA is absent."""
from pathlib import Path
import numpy as np
import torch
import mujoco

from .reference.envs.physics import TShapePhysics
from .kernels import reduce_contacts


def oracle(config):
    return TShapePhysics(config.frame_skip, config.resolved_model_path, config.num_agents, task="medium")


class ReferencePhysics:
    def __init__(self, config, simulation, task):
        self.config, self.task = config, task
        self.worlds = [oracle(config) for _ in range(simulation.num_envs)]
        self.qpos = torch.zeros((simulation.num_envs, self.worlds[0].model.nq))
        self.qvel = torch.zeros_like(self.qpos)
        self.contacts = torch.zeros((simulation.num_envs, config.num_agents), dtype=torch.bool)

    def reset(self, mask, qpos):
        for i in mask.nonzero().flatten().tolist():
            p = self.worlds[i]
            p.reset()
            p.data.qpos[:] = qpos[i].numpy()
            mujoco.mj_forward(p.model, p.data)
        self._read()

    def _read(self):
        self.qpos.copy_(torch.tensor(np.stack([p.data.qpos for p in self.worlds]), dtype=torch.float32))
        self.qvel.copy_(torch.tensor(np.stack([p.data.qvel for p in self.worlds]), dtype=torch.float32))
        self.contacts.copy_(torch.tensor(np.stack([p.contact_state()[0] for p in self.worlds])))

    def step(self, actions):
        reports = [p.step(a.numpy(), max_penetration=self.config.max_penetration,
                          max_linear_speed=self.config.max_linear_speed,
                          max_angular_speed=self.config.max_angular_speed) for p, a in zip(self.worlds, actions)]
        self._read()
        codes = {None: 0, "non_finite_state": 1, "excessive_penetration": 2,
                 "excessive_speed": 3, "automatic_reset": 4, "mujoco_warning": 5}
        return dict(failed=torch.tensor([not r["stable"] for r in reports]),
                    failure_code=torch.tensor([codes[r["failure_reason"]] for r in reports]),
                    penetration=torch.tensor([r["max_penetration_m"] for r in reports]),
                    obstacle_events=torch.tensor([r["obstacle_collision_events"] for r in reports]),
                    path_length=torch.tensor([r["object_path_length_m"] for r in reports]))


class WarpPhysics:
    def __init__(self, config, simulation, task):
        import warp as wp
        import mujoco_warp as mjw
        from importlib.metadata import version
        if (version("mujoco-warp") != "3.5.0" or mujoco.__version__ != "3.5.0"
                or version("warp-lang") != "1.11.0"):
            raise RuntimeError("Install pinned requirements: mujoco==3.5.0, mujoco-warp==3.5.0, warp-lang==1.11.0")
        self.wp, self.mjw, self.task = wp, mjw, task
        if wp.config.kernel_cache_dir is None:
            wp.config.kernel_cache_dir = str(Path(__file__).resolve().parents[1]/".cache/warp")
        self.config, self.simulation = config, simulation
        self.device = torch.device(simulation.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run check on the remote GPU PC or explicitly use reference/cpu")
        wp.init()
        self.cpu = oracle(config)
        with self.scope():
            self.model = mjw.put_model(self.cpu.model)
            self.data = mjw.make_data(self.cpu.model, nworld=simulation.num_envs,
                                      nconmax=simulation.nconmax, njmax=simulation.njmax)
        self.qpos, self.qvel = wp.to_torch(self.data.qpos), wp.to_torch(self.data.qvel)
        self.ctrl, self.time = wp.to_torch(self.data.ctrl), wp.to_torch(self.data.time)
        n, g = simulation.num_envs, self.cpu.model.ngeom
        self.agent_map = torch.full((g,), -1, device=self.device, dtype=torch.long)
        self.agent_map[torch.tensor(self.cpu.agent_geom_ids, device=self.device)] = torch.arange(config.num_agents, device=self.device)
        self.object_map = torch.zeros(g, device=self.device, dtype=torch.bool)
        self.object_map[list(self.cpu.object_geoms)] = True
        self.obstacle_map = torch.zeros_like(self.object_map)
        self.obstacle_map[list(self.cpu.obstacles)] = True
        self.previous_pairs = torch.zeros((n, g*g), device=self.device, dtype=torch.bool)
        self.contacts = torch.zeros((n, config.num_agents), device=self.device, dtype=torch.bool)

    def scope(self):
        if self.device.type == "cuda":
            return self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device)))
        return self.wp.ScopedDevice("cpu")

    def contact_state(self):
        wp, d = self.wp, self.data
        torch._assert_async((wp.to_torch(d.nacon) <= d.naconmax).all(), "Contact overflow: increase nconmax")
        torch._assert_async((wp.to_torch(d.ncollision) <= d.naconmax).all(), "Broadphase overflow: increase nconmax")
        torch._assert_async((wp.to_torch(d.nefc) <= d.njmax).all(), "Constraint overflow: increase njmax")
        flags, penetration, pairs = reduce_contacts(wp.to_torch(d.contact.geom), wp.to_torch(d.contact.worldid),
                                                    wp.to_torch(d.contact.dist), wp.to_torch(d.contact.efc_address),
                                                    wp.to_torch(d.nacon), d.nworld, self.cpu.model.ngeom,
                                                    self.agent_map, self.object_map, self.obstacle_map)
        self.contacts.copy_(flags[:, :self.config.num_agents])
        return penetration, pairs

    def reset(self, mask, qpos):
        self.qpos.copy_(torch.where(mask[:, None], qpos, self.qpos))
        for name in ("qvel", "qacc", "qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied", "time"):
            view = self.wp.to_torch(getattr(self.data, name))
            view.masked_fill_(mask.reshape(len(mask), *([1]*(view.ndim-1))), 0.)
        self.previous_pairs.masked_fill_(mask[:, None], False)
        with self.scope():
            self.mjw.forward(self.model, self.data)
        self.contact_state()
        self.safe_qpos, self.safe_qvel = self.qpos.clone(), self.qvel.clone()

    def diagnostics(self):
        penetration, pairs = self.contact_state()
        penetration = torch.maximum(penetration, self.task.boundary_penetration(self.qpos, self.qvel))
        finite = (torch.isfinite(self.qpos).all(1) & torch.isfinite(self.qvel).all(1)
                  & torch.isfinite(self.wp.to_torch(self.data.qacc)).all(1) & torch.isfinite(self.time))
        vel = self.qvel[:, self.task.agent_dofs].norm(dim=-1).amax(1)
        ov = self.qvel[:, self.task.object_dofs]
        speed = (torch.maximum(vel, ov[:, :2].norm(dim=-1)) > self.config.max_linear_speed
                 ) | (ov[:, 2].abs() > self.config.max_angular_speed)
        code = torch.zeros(len(self.qpos), device=self.device, dtype=torch.long)
        code = torch.where(speed, 3, code)
        code = torch.where(penetration > self.config.max_penetration, 2, code)
        code = torch.where(~finite, 1, code)
        return code, penetration, pairs

    def step(self, actions):
        low, high = self.task.motor_limits[..., 0], self.task.motor_limits[..., 1]
        self.ctrl[:, self.task.motor_ids] = torch.where(actions >= 0, actions*high, -actions*low)
        code, maximum, _ = self.diagnostics()
        failed = code != 0
        maximum = torch.nan_to_num(maximum, nan=0., posinf=0.)
        self.qpos.copy_(torch.where(failed[:, None], self.safe_qpos, self.qpos))
        self.qvel.copy_(torch.where(failed[:, None], self.safe_qvel, self.qvel))
        events = torch.zeros_like(code)
        length = torch.zeros_like(maximum)
        # Failed worlds are restored every substep; healthy worlds continue normally.
        for _ in range(self.config.frame_skip):
            old_q, old_v, old_t = self.qpos.clone(), self.qvel.clone(), self.time.clone()
            old_obj = self.task.geometry(old_q, old_v)[2]
            with self.scope():
                self.mjw.step(self.model, self.data)
                self.mjw.forward(self.model, self.data)
            new_code, pen, pairs = self.diagnostics()
            new_code = torch.where((self.time-old_t-self.cpu.model.opt.timestep).abs() > 1e-5, 4, new_code)
            code = torch.where(~failed, new_code, code)
            failed |= new_code != 0
            maximum = torch.maximum(maximum, torch.nan_to_num(pen, nan=0.))
            events += ((pairs & ~self.previous_pairs).sum(1)*(~failed)).long()
            self.previous_pairs.copy_(pairs)
            displacement = (self.task.geometry(self.qpos, self.qvel)[2]-old_obj).norm(dim=-1)
            length += torch.where(failed, 0., displacement)
            self.qpos.copy_(torch.where(failed[:, None], old_q, self.qpos))
            self.qvel.copy_(torch.where(failed[:, None], old_v, self.qvel))
            self.time.copy_(torch.where(failed, old_t, self.time))
            self.wp.to_torch(self.data.qacc_warmstart).masked_fill_(failed[:, None], 0.)
        with self.scope():
            self.mjw.forward(self.model, self.data)
        self.contact_state()
        self.safe_qpos.copy_(self.qpos)
        self.safe_qvel.copy_(self.qvel)
        return dict(failed=failed, failure_code=code, penetration=maximum,
                    obstacle_events=events, path_length=length)
