"""Geometry of one static, axis-aligned doorway, crossed in the +X direction."""

from dataclasses import dataclass
from itertools import permutations

import mujoco
import numpy as np


@dataclass(frozen=True)
class Doorway:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @classmethod
    def from_physics(cls, physics):
        model = physics.model
        names = {model.geom(i).name for i in physics.obstacles}
        if names != {"obstacle_lower", "obstacle_upper"}:
            raise ValueError("Medium requires exactly obstacle_lower and obstacle_upper")
        lower, upper = (model.geom(name) for name in ("obstacle_lower", "obstacle_upper"))
        for geom in (lower, upper):
            if (geom.type != mujoco.mjtGeom.mjGEOM_BOX or geom.bodyid != 0
                    or not np.allclose(geom.quat, [1, 0, 0, 0])
                    or not geom.contype or not geom.conaffinity
                    or geom.pos[2] - geom.size[2] > .01
                    or geom.pos[2] + geom.size[2] < .21):
                raise ValueError("doorway walls must be static, collidable, axis-aligned boxes covering object height")
        if not np.allclose([lower.pos[0], lower.size[0]], [upper.pos[0], upper.size[0]]):
            raise ValueError("doorway walls must share X position and thickness")
        if (lower.pos[1] - lower.size[1] > physics.lower[1]
                or upper.pos[1] + upper.size[1] < physics.upper[1]):
            raise ValueError("doorway walls must reach both arena boundaries")
        result = cls(float(lower.pos[0] - lower.size[0]), float(lower.pos[0] + lower.size[0]),
                     float(lower.pos[1] + lower.size[1]), float(upper.pos[1] - upper.size[1]))
        if not (physics.lower[0] < result.x_min < result.x_max < physics.upper[0]
                and physics.lower[1] < result.y_min < result.y_max < physics.upper[1]):
            raise ValueError("doorway must have one open gap inside the arena")
        return result

    def features(self, position, margin):
        center = np.array([(self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2])
        return np.r_[center - position, self.x_max - self.x_min,
                     self.y_max - self.y_min, self.x_max + margin - position[0]].astype(np.float32)

    def clearance(self, corners, margin):
        return float(np.min(corners[:, 0]) - self.x_max - margin)

    def crossed(self, corners, margin):
        return bool(np.isfinite(corners).all() and self.clearance(corners, margin) >= -1e-9)

    def _visible(self, start, end):
        """Whether a point segment stays inside the opening while in the wall slab."""
        dx = end[0] - start[0]
        if abs(dx) < 1e-12:
            if start[0] < self.x_min or start[0] > self.x_max:
                return True
            ys = (start[1], end[1])
        else:
            bounds = sorted(((self.x_min - start[0]) / dx, (self.x_max - start[0]) / dx))
            lo, hi = max(0., bounds[0]), min(1., bounds[1])
            if lo > hi:
                return True
            ys = (start[1] + lo * (end[1] - start[1]), start[1] + hi * (end[1] - start[1]))
        return min(ys) >= self.y_min - 1e-9 and max(ys) <= self.y_max + 1e-9

    def route_distance(self, position, target):
        """Continuous shortest point-path distance through the opening.

        This reward heuristic does not certify clearance of the full rigid body.
        A valid path needs at most two aperture corners; no phase switch is used.
        """
        start, target = np.asarray(position), np.asarray(target)
        if self._visible(start, target):
            return float(np.linalg.norm(target - start))
        corners = [np.array([x, y]) for x in (self.x_min, self.x_max)
                   for y in (self.y_min, self.y_max)]
        distances = []
        for count in (1, 2):
            for middle in permutations(corners, count):
                points = (start, *middle, target)
                if all(self._visible(a, b) for a, b in zip(points, points[1:])):
                    distances.append(sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:])))
        # A penetrated center has no free point path. Keep failure diagnostics
        # finite; the physics safety checks remain responsible for terminating it.
        return float(min(distances)) if distances else float(np.linalg.norm(target - start))
