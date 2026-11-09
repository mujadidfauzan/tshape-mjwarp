"""Planar MuJoCo mechanics for Easy and the single-doorway Medium arena."""

from pathlib import Path
import re

import mujoco
import numpy as np

from .model import build_model_xml
from .doorway import Doorway

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "assets" / "mujoco" / "arena.xml"


def finite_array(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite numbers")
    return value


class TShapePhysics:
    def __init__(self, frame_skip=50, model_path=MODEL_PATH, num_agents=None, task="easy"):
        if isinstance(frame_skip, bool) or not isinstance(frame_skip, (int, np.integer)) or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        self.model_path = Path(model_path).resolve()
        xml = self.model_path.read_text()
        self.model_xml = xml if num_agents is None else build_model_xml(xml, num_agents)
        self.model = m = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(m)
        self.frame_skip = int(frame_skip)
        self.agent_names = sorted([
            m.body(i).name for i in range(m.nbody)
            if re.fullmatch(r"agent_\d+", m.body(i).name)
            ], key=lambda name: int(name.split("_")[1])
        )
        self.num_agents = n = len(self.agent_names)
        if not 1 <= n <= 16 or self.agent_names != [f"agent_{i}" for i in range(n)]:
            raise ValueError("Easy requires 1 to 16 consecutively named agents starting at agent_0")
        self.agent_bodies = np.array([m.body(n).id for n in self.agent_names])
        self.agent_geom_ids = np.array([m.geom(f"{n}_geom").id for n in self.agent_names])
        self.agent_geoms = set(self.agent_geom_ids.tolist())
        joints = np.array([[m.joint(f"{n}_{axis}").id for axis in ("x", "y")]
                           for n in self.agent_names])
        self.agent_dofs = m.jnt_dofadr[joints]
        self.agent_qpos = m.jnt_qposadr[joints]
        self.motor_ids = np.array([[m.actuator(f"{n}_f{axis}").id for axis in ("x", "y")]
                                   for n in self.agent_names])
        if m.nu != 2*n or m.nq != 2*n+3 or m.nv != 2*n+3:
            raise ValueError("Expected two motors/slide joints per agent and three object DOFs")
        if not np.array_equal(m.actuator_trnid[self.motor_ids, 0], joints):
            raise ValueError("Motor names must match their agent slide joints")
        if not np.all(m.actuator_ctrllimited[self.motor_ids]):
            raise ValueError("Agent motors must have explicit control limits")
        if not np.allclose(m.actuator_gear[self.motor_ids, 0], 1.0):
            raise ValueError("Agent motors must use gear=1 so controls represent force in N")
        self.motor_limits = m.actuator_ctrlrange[self.motor_ids].copy()
        if np.any(self.motor_limits[..., 0] >= 0) or np.any(self.motor_limits[..., 1] <= 0):
            raise ValueError("Agent force limits must straddle zero")
        self.object_body = m.body("t_shape").id
        object_joints = [m.joint(n).id for n in ("t_x", "t_y", "t_yaw")]
        self.object_qpos = m.jnt_qposadr[object_joints]
        self.object_dofs = m.jnt_dofadr[object_joints]
        self.object_geom_ids = np.flatnonzero(m.geom_bodyid == self.object_body).tolist()
        if not self.object_geom_ids or any(m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX
                                           for g in self.object_geom_ids):
            raise ValueError("Transported object must consist of collision boxes in one body")
        self.object_geoms = set(self.object_geom_ids)
        self._moving_geom_set = self.agent_geoms | self.object_geoms
        self.moving_geoms = np.array(sorted(self._moving_geom_set))
        self.radii = m.geom_size[self.agent_geom_ids, 0].copy()
        self.walls = {m.geom(f"wall_{side}").id for side in ("left", "right", "bottom", "top")}
        self.obstacles = {i for i in range(m.ngeom) if m.geom(i).name.startswith("obstacle_")}
        if self.obstacles and task != "medium":
            raise ValueError("The Easy baseline has no internal obstacles")
        self.lower = np.array([m.geom("wall_left").pos[0] + m.geom("wall_left").size[0],
                               m.geom("wall_bottom").pos[1] + m.geom("wall_bottom").size[1]])
        self.upper = np.array([m.geom("wall_right").pos[0] - m.geom("wall_right").size[0],
                               m.geom("wall_top").pos[1] - m.geom("wall_top").size[1]])
        self.doorway = Doorway.from_physics(self) if task == "medium" else None
        self._previous_obstacle_pairs = set()
        self.reset()

    @property
    def control_dt(self):
        return self.model.opt.timestep * self.frame_skip

    def reset(self, object_pose=None, agent_positions=None):
        """Reset motion/forces; optional world body pose and world agent positions."""
        if object_pose is not None:
            object_pose = finite_array(object_pose, (3,), "object_pose")
        if agent_positions is not None:
            agent_positions = finite_array(agent_positions, (self.num_agents, 2), "agent_positions")
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if object_pose is not None:
            d.qpos[self.object_qpos[:2]] = object_pose[:2] - m.body_pos[self.object_body, :2]
            d.qpos[self.object_qpos[2]] = object_pose[2]
        if agent_positions is not None:
            d.qpos[self.agent_qpos] = agent_positions - m.body_pos[self.agent_bodies, :2]
        self._previous_wall_pairs = set()
        self._previous_obstacle_pairs = set()
        mujoco.mj_forward(m, d)

    def rotation(self):
        return self.data.xmat[self.object_body].reshape(3, 3)[:2, :2].copy()

    def get_object_state(self):
        d = self.data
        v = d.qvel[self.object_dofs]
        offset = d.xipos[self.object_body, :2] - d.xpos[self.object_body, :2]
        return {
            "position": d.xipos[self.object_body, :2].copy(),
            "yaw": float(d.qpos[self.object_qpos[2]]),
            "velocity": v[:2] + v[2] * np.array([-offset[1], offset[0]]),
            "angular_velocity": float(v[2]),
        }

    def get_agents_state(self):
        return (self.data.xpos[self.agent_bodies, :2].copy(),
                self.data.qvel[self.agent_dofs].copy())

    def object_corners(self):
        """World XY corners of every box, including the short end bar."""
        corners = []
        for geom_id in self.object_geom_ids:
            half = self.model.geom_size[geom_id, :2]
            local = np.array([[-half[0], -half[1]], [-half[0], half[1]],
                              [half[0], -half[1]], [half[0], half[1]]])
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)[:2, :2]
            corners.append(local @ rotation.T + self.data.geom_xpos[geom_id, :2])
        return np.vstack(corners)

    def boundary_penetration(self):
        extents = []
        for geom in self.moving_geoms:
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                extents.append(np.repeat(self.model.geom_size[geom, 0], 2))
            else:
                rotation = self.data.geom_xmat[geom].reshape(3, 3)
                extents.append((np.abs(rotation) @ self.model.geom_size[geom])[:2])
        centers = self.data.geom_xpos[self.moving_geoms, :2]
        return float(max(0.0, np.max(self.lower - (centers - extents)),
                         np.max(centers + extents - self.upper)))

    def contact_state(self):
        flags = np.zeros(self.num_agents, dtype=bool)
        wall_pairs = set()
        penetration = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.efc_address < 0:
                continue
            pair = {int(contact.geom1), int(contact.geom2)}
            penetration = max(penetration, -float(contact.dist))
            if pair & self.object_geoms:
                for agent, geom in enumerate(self.agent_geom_ids):
                    flags[agent] |= int(geom) in pair
            if pair & (self.walls | self.obstacles) and pair & self._moving_geom_set:
                wall_pairs.add(tuple(sorted(pair)))
        return flags, wall_pairs, penetration

    def diagnostics(self, max_penetration=0.005, max_linear_speed=5.0, max_angular_speed=10.0):
        d = self.data
        result = {"stable": False, "failure_reason": None, "penetration_m": 0.0,
                  "linear_speed_m_s": 0.0, "angular_speed_rad_s": 0.0}
        if not np.isfinite(d.time) or not all(np.isfinite(a).all() for a in (d.qpos, d.qvel, d.qacc)):
            result["failure_reason"] = "non_finite_state"
            return result
        if np.any(d.warning.number):
            result["failure_reason"] = "mujoco_warning"
            return result
        obj = self.get_object_state()
        _, velocities = self.get_agents_state()
        _, _, contact_penetration = self.contact_state()
        result["penetration_m"] = max(self.boundary_penetration(), contact_penetration)
        result["linear_speed_m_s"] = float(max(np.linalg.norm(obj["velocity"]),
                                                  np.max(np.linalg.norm(velocities, axis=1))))
        result["angular_speed_rad_s"] = float(abs(obj["angular_velocity"]))
        if result["penetration_m"] > max_penetration:
            result["failure_reason"] = "excessive_penetration"
        elif result["linear_speed_m_s"] > max_linear_speed or result["angular_speed_rad_s"] > max_angular_speed:
            result["failure_reason"] = "excessive_speed"
        result["stable"] = result["failure_reason"] is None
        return result

    def failure_details(self):
        """Capture the actual failing state before the environment substitutes
        its last valid terminal observation. Only used on physics failure.
        """
        def finite_list(array):
            values = np.asarray(array, dtype=np.float64)
            return values.tolist() if np.isfinite(values).all() else None

        d = self.data
        finite_state = all(np.isfinite(a).all() for a in (d.qpos, d.qvel, d.qacc))
        details = {
            "time_s": float(d.time) if np.isfinite(d.time) else None,
            "qpos": finite_list(d.qpos), "qvel": finite_list(d.qvel),
            "agent_motor_forces_N": finite_list(d.ctrl[self.motor_ids]),
            "boundary_penetration_m": None, "contacts": [],
            "warnings": {mujoco.mjtWarning(i).name: int(warning.number)
                         for i, warning in enumerate(d.warning) if warning.number},
        }
        if not finite_state:
            return details
        details["boundary_penetration_m"] = self.boundary_penetration()
        force = np.zeros(6)
        for index in range(d.ncon):
            contact = d.contact[index]
            if contact.efc_address < 0 or not np.isfinite(contact.dist):
                continue
            ids = (int(contact.geom1), int(contact.geom2))
            mujoco.mj_contactForce(self.model, d, index, force)
            details["contacts"].append({
                "geoms": [self.model.geom(g).name for g in ids],
                "penetration_m": max(0.0, -float(contact.dist)),
                "normal_force_N": float(force[0]) if np.isfinite(force[0]) else None,
            })
        details["contacts"].sort(key=lambda item: item["penetration_m"], reverse=True)
        return details

    def step(self, actions, **safety_limits):
        actions = np.clip(finite_array(actions, (self.num_agents, 2), "actions"), -1.0, 1.0)
        low, high = self.motor_limits[..., 0], self.motor_limits[..., 1]
        self.data.ctrl[self.motor_ids] = np.where(actions >= 0, actions * high, -actions * low)
        report = {"stable": True, "failure_reason": None, "physics_steps": 0,
                  "object_path_length_m": 0.0, "control_effort_N2_s": 0.0,
                  "wall_collision_events": 0, "wall_contact_physics_steps": 0,
                  "max_penetration_m": 0.0, "failure_details": None}
        report["obstacle_collision_events"] = 0
        report["agent_object_contact_time_s"] = np.zeros(self.num_agents)
        check = self.diagnostics(**safety_limits)
        if not check["stable"]:
            report.update(stable=False, failure_reason=check["failure_reason"],
                          max_penetration_m=check["penetration_m"],
                          failure_details=self.failure_details())
            return report
        previous_position = self.get_object_state()["position"]
        for _ in range(self.frame_skip):
            old_time = self.data.time
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            report["physics_steps"] += 1
            report["control_effort_N2_s"] += float(np.sum(self.data.ctrl ** 2) * self.model.opt.timestep)
            check = self.diagnostics(**safety_limits)
            report["max_penetration_m"] = max(report["max_penetration_m"], check["penetration_m"])
            if not np.isclose(self.data.time, old_time + self.model.opt.timestep, rtol=0, atol=1e-9):
                check.update(stable=False, failure_reason="automatic_reset")
            if not check["stable"]:
                report.update(stable=False, failure_reason=check["failure_reason"],
                              failure_details=self.failure_details())
                break
            position = self.get_object_state()["position"]
            report["object_path_length_m"] += float(np.linalg.norm(position - previous_position))
            previous_position = position
            flags, pairs, _ = self.contact_state()
            obstacle_pairs = {pair for pair in pairs if set(pair) & self.obstacles}
            report["obstacle_collision_events"] += len(obstacle_pairs - self._previous_obstacle_pairs)
            self._previous_obstacle_pairs = obstacle_pairs
            report["agent_object_contact_time_s"] += flags * self.model.opt.timestep
            wall_pairs = {pair for pair in pairs if set(pair) & self.walls}
            report["wall_collision_events"] += len(wall_pairs - self._previous_wall_pairs)
            report["wall_contact_physics_steps"] += int(bool(wall_pairs))
            self._previous_wall_pairs = wall_pairs
        return report
