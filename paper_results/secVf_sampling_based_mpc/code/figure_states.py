"""Select recorded states that illustrate progress toward one unchanged goal."""

import numpy as np


def validate_future_steps(object_pose, goals, current_step, future_steps):
    """Reject goal transitions, backward time, and poses moving away from the target."""
    steps = list(future_steps)
    if not steps or steps != sorted(set(steps)):
        raise ValueError("Future steps must be distinct and in chronological order")
    if current_step < 0 or current_step >= len(object_pose):
        raise ValueError("Current step is outside the recorded trajectory")
    if steps[0] <= current_step or steps[-1] >= len(object_pose):
        raise ValueError("Future steps must be later recorded rows")
    active_goal = goals[min(current_step, len(goals) - 1)]
    indices = np.minimum(np.arange(current_step, steps[-1] + 1), len(goals) - 1)
    if not np.allclose(goals[indices], active_goal):
        raise ValueError("Future overlay crosses a goal change")
    distances = np.linalg.norm(object_pose[[current_step, *steps], :2] - active_goal, axis=1)
    if not np.all(np.isfinite(distances)) or not np.all(np.diff(distances) < 0):
        raise ValueError("Future poses must make progress toward the highlighted goal")
    return distances


def transform_mesh(vertices, normals, pose, scale=(1, 1, 1)):
    """Apply mesh scale and a saved xyz/xyzw pose, preserving contact tilts."""
    x, y, z, w = np.asarray(pose[3:7]) / np.linalg.norm(pose[3:7])
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    points = (np.asarray(vertices) * scale) @ rotation.T + pose[:3]
    transformed_normals = None
    if normals is not None:
        transformed_normals = (np.asarray(normals) / scale) @ rotation.T
        transformed_normals /= np.maximum(np.linalg.norm(transformed_normals, axis=1, keepdims=True), 1e-12)
        transformed_normals = transformed_normals.astype(np.float32)
    return points.astype(np.float32), transformed_normals
