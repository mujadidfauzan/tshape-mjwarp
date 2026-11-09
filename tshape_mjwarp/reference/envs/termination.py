"""Full-object containment and settling for the fixed target rectangle."""

import numpy as np


def object_inside_goal(corners, goal_center, goal_half_size):
    lower = np.asarray(goal_center) - goal_half_size
    upper = np.asarray(goal_center) + goal_half_size
    return bool(np.isfinite(corners).all() and np.all(corners >= lower - 1e-9)
                and np.all(corners <= upper + 1e-9))


def goal_is_settled(physics, goal_center, goal_half_size, linear_speed, angular_speed):
    obj = physics.get_object_state()
    return (object_inside_goal(physics.object_corners(), goal_center, goal_half_size)
            and np.linalg.norm(obj["velocity"]) <= linear_speed
            and abs(obj["angular_velocity"]) <= angular_speed)
