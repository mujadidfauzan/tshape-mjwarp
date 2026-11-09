"""Easy team reward and positive tanh distance reward for Medium."""

import math


def wrapped_yaw_error(yaw, goal_yaw):
    delta = yaw - goal_yaw
    return math.atan2(math.sin(delta), math.cos(delta))


def team_reward(previous_distance, distance, yaw, success, failed, config):
    parts = dict(distance=0.0, orientation=0.0, progress=0.0, success=0.0, failure=0.0)
    if failed:
        parts["failure"] = -config.failure_penalty
    else:
        if not all(math.isfinite(v) for v in (previous_distance, distance, yaw)):
            raise ValueError("reward inputs must be finite for stable physics")
        error = abs(wrapped_yaw_error(yaw, config.goal_yaw)) / math.pi
        parts.update(
            distance=(config.distance_weight * (1 - math.tanh(distance / config.distance_scale))
                      if config.task == "medium" else
                      -config.distance_weight * distance / (distance + config.distance_scale)),
            orientation=(0.0 if config.task == "medium" else
                         -config.orientation_weight * math.exp(-distance / config.orientation_distance_scale) * error),
            progress=config.progress_weight * (previous_distance - distance),
            success=config.success_reward if success else 0.0,
        )
    parts = {key: float(value) for key, value in parts.items()}
    return float(sum(parts.values())), parts
