"""Easy observations: 18+4*N actor features and 17+4*N critic features.

Vectors use world-aligned XY axes. This is NOT a limited-range local sensor model.
Other agents appear in ascending agent index, excluding the observer.
Medium appends five doorway features without changing existing Easy dimensions.
"""

import numpy as np

def feature_sizes(num_agents, doorway=False):
    # Actor: 22 own/object/goal features plus four for each other agent.
    # Critic: 17 object/goal/arena features plus four for every agent.
    extra = 5 if doorway else 0
    return 18 + 4 * num_agents + extra, 17 + 4 * num_agents + extra


def build_state(physics, goal_center, goal_half_size, remaining_time, goal_hold, crossing_margin=.05):
    obj = physics.get_object_state()
    positions, velocities = physics.get_agents_state()
    return np.concatenate([
        obj["position"], [np.sin(obj["yaw"]), np.cos(obj["yaw"])],
        obj["velocity"], [obj["angular_velocity"]], goal_center, goal_half_size,
        positions.ravel(), velocities.ravel(), physics.lower, physics.upper,
        [remaining_time, goal_hold],
        *([physics.doorway.features(np.zeros(2), crossing_margin)] if physics.doorway else []),
    ]).astype(np.float32)


def build_observations(physics, goal_center, goal_half_size, remaining_time, goal_hold, crossing_margin=.05):
    obj = physics.get_object_state()
    positions, velocities = physics.get_agents_state()
    contacts, _, _ = physics.contact_state()
    result = {}
    for i, name in enumerate(physics.agent_names):
        others = [j for j in range(physics.num_agents) if j != i]
        wall_clearances = np.r_[positions[i] - physics.lower - physics.radii[i],
                                physics.upper - positions[i] - physics.radii[i]]
        result[name] = np.concatenate([
            positions[i], velocities[i], obj["position"] - positions[i],
            [np.sin(obj["yaw"]), np.cos(obj["yaw"])], obj["velocity"],
            [obj["angular_velocity"]], goal_center - positions[i], goal_half_size,
            (positions[others] - positions[i]).ravel(), velocities[others].ravel(),
            wall_clearances, [float(contacts[i]), remaining_time, goal_hold],
            *([physics.doorway.features(positions[i], crossing_margin)] if physics.doorway else []),
        ]).astype(np.float32)
    return result
