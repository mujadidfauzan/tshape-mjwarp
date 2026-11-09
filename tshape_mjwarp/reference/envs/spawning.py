"""Seeded agent placement in a narrow band around the actual object footprint."""

import numpy as np


def sample_object_pose(config, rng):
    """Sample enabled components only; disabled components keep their base pose."""
    pose = np.array(config.object_pose, dtype=float)
    if config.randomize_object_position:
        extent = np.asarray(config.object_position_range)
        pose[:2] += rng.uniform(-extent, extent)
    if config.randomize_object_orientation:
        pose[2] = rng.uniform(*config.object_yaw_range)
    return pose


def sample_agent_positions(physics, rng, gap_min, gap_max):
    corners = physics.object_corners()
    positions = []
    for i, radius in enumerate(physics.radii):
        lower = np.maximum(corners.min(axis=0) - radius - gap_max, physics.lower + radius + .01)
        upper = np.minimum(corners.max(axis=0) + radius + gap_max, physics.upper - radius - .01)
        if np.any(lower >= upper):
            raise ValueError("no free spawn band around the object inside the arena")
        for _ in range(10000):
            point = rng.uniform(lower, upper)
            distances = []
            for g in physics.object_geom_ids:
                rotation = physics.data.geom_xmat[g].reshape(3, 3)[:2, :2]
                local = rotation.T @ (point - physics.data.geom_xpos[g, :2])
                outside = np.maximum(np.abs(local) - physics.model.geom_size[g, :2], 0.)
                distances.append(np.linalg.norm(outside))
            gap = min(distances) - radius
            if not gap_min <= gap <= gap_max:
                continue
            if any(np.linalg.norm(np.maximum(
                    np.abs(point - physics.model.geom_pos[g, :2]) - physics.model.geom_size[g, :2], 0.))
                   < radius + .01 for g in physics.obstacles):
                continue
            if positions and np.any(np.linalg.norm(np.asarray(positions) - point, axis=1)
                                    < physics.radii[:i] + radius + .01):
                continue
            positions.append(point)
            break
        else:
            raise ValueError(f"cannot place agent_{i} without overlap; widen the spawn band or reduce num_agents")
    return np.asarray(positions)
