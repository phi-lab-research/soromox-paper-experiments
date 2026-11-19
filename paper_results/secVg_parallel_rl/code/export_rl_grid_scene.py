"""Export saved trained-policy grid states without running a policy or simulation."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import trimesh
from soromox.rendering.base_geometry import base_plate_mesh
from soromox.rendering.config import GeometryConfig

from render_rl_video import build_rl_robot, load_rollout, make_grid_offsets
from rl_render_style import TARGET_SPHERE_RADIUS

CASE = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tube_mesh(poses, radius, sides=32):
    angles = np.arange(sides)*2*np.pi/sides
    radial = np.stack((np.zeros(sides), np.cos(angles), np.sin(angles)), axis=1)
    normals = np.einsum('nij,kj->nki', poses[:, :3, :3], radial)
    rings = poses[:, None, :3, 3]+radius*normals
    points, normals = rings.reshape(-1, 3), normals.reshape(-1, 3)
    faces = []
    for i in range(len(poses)-1):
        for j in range(sides):
            a, b = i*sides+j, i*sides+(j+1) % sides
            faces.extend(((a, b, b+sides), (a, b+sides, a+sides)))
    for end, sign in ((0, -1), (-1, 1)):
        start = len(points)
        points = np.vstack((points, rings[end], poses[end, :3, 3]))
        normals = np.vstack((normals, np.tile(sign*poses[end, :3, 0], (sides+1, 1))))
        for j in range(sides):
            tri = (start+sides, start+j, start+(j+1) % sides)
            faces.append(tri if sign == 1 else tri[::-1])
    return points, np.asarray(faces), normals


def combine_meshes(meshes):
    points, faces, normals, offset = [], [], [], 0
    for vertices, indices, ns in meshes:
        points.append(vertices)
        faces.append(indices+offset)
        normals.append(ns)
        offset += len(vertices)
    return np.vstack(points), np.vstack(faces), np.vstack(normals)


def camera_coordinates(points, eye, target):
    forward = np.asarray(target)-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return (points-eye) @ np.stack((right, up, forward), axis=1)


def fit_camera(points, aspect=4/3, fov=45., margin=.05):
    target = (points.min(axis=0)+points.max(axis=0))/2
    direction = np.array([.16, -1., 1.15])
    direction /= np.linalg.norm(direction)
    low, high = .1, 15.
    for _ in range(40):
        distance = (low+high)/2
        eye = target+distance*direction
        xyz = camera_coordinates(points, eye, target)
        half_height = xyz[:, 2]*np.tan(np.deg2rad(fov)/2)
        fits = (xyz[:, 2].min() > 0 and np.max(np.abs(xyz[:, 0])/half_height/aspect) <= 1-2*margin
                and np.max(np.abs(xyz[:, 1])/half_height) <= 1-2*margin)
        if fits:
            high = distance
        else:
            low = distance
    return (target+high*direction).tolist(), target.tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=CASE/'data/traj/rl_rollout_trained_64_envs.npz')
    parser.add_argument('--times', type=float, nargs=2, default=(5., 15.))
    parser.add_argument('--output-dir', type=Path, default=CASE/'outputs/paper_snapshots')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    outputs = [args.output_dir/f'grid_{i+1}_geometry.npz' for i in range(2)]
    if any(p.exists() for p in outputs) and not args.force:
        raise FileExistsError('Geometry exists; pass --force to replace it')
    rollout = load_rollout(args.data)
    if rollout.policy != 'trained' or rollout.num_envs != 64:
        raise ValueError('Expected the complete saved 64-environment trained-policy rollout')
    if not np.isfinite(rollout.q_ts).all() or not np.isfinite(rollout.ball_ts).all():
        raise ValueError('Nonfinite trajectory')
    indices = []
    for t in args.times:
        index = int(np.argmin(abs(rollout.ts-t)))
        if not np.isclose(rollout.ts[index], t):
            raise ValueError('Requested time is not a saved rollout instant')
        indices.append(index)
    robot = build_rl_robot(rollout.arm_length, rollout.arm_radius)
    s = np.linspace(0, rollout.arm_length, 65)
    fk = jax.jit(jax.vmap(robot.forward_kinematics_abscissa_batched, in_axes=(0, None)))
    offsets = make_grid_offsets(64, 8, 8, .28)
    # Mounts sit on the shared studio floor at z=-0.05 m.
    geom = GeometryConfig()
    offsets[:, 2] = geom.base_plate_thickness-.05
    mount_v, mount_f = base_plate_mesh(rollout.arm_radius*geom.base_plate_radius_scale,
                                      geom.base_plate_thickness, geom.base_plate_style)
    mount_v[:, 2] -= geom.base_plate_thickness/2
    mount_n = np.asarray(trimesh.Trimesh(mount_v, mount_f, process=False).vertex_normals)
    mount_mesh = combine_meshes([(mount_v+o, mount_f, mount_n) for o in offsets])
    scenes, bounds = [], []
    for index in indices:
        poses = np.asarray(fk(rollout.q_ts[:, index], s)).copy()
        np.testing.assert_allclose(poses[:, 0, :3, 0], np.tile([0, 0, 1.], (64, 1)), atol=1e-8)
        errors = np.linalg.norm(poses[:, -1, :3, 3]-rollout.ball_ts[:, index], axis=1)
        poses[:, :, :3, 3] += offsets[:, None]
        robot_mesh = combine_meshes([tube_mesh(p, rollout.arm_radius) for p in poses])
        targets = rollout.ball_ts[:, index]+offsets
        bounds.extend([robot_mesh[0], mount_mesh[0], targets-TARGET_SPHERE_RADIUS, targets+TARGET_SPHERE_RADIUS])
        scenes.append((robot_mesh, targets, errors))
    eye, target = fit_camera(np.vstack(bounds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for output, index, (mesh, targets, errors) in zip(outputs, indices, scenes, strict=True):
        metadata = dict(source_file=args.data.name, source_sha256=sha256(args.data),
                        exporter_sha256=sha256(Path(__file__)), policy='trained', num_envs=64,
                        environment_indices=list(range(64)), grid_shape=[8, 8], grid_spacing_m=.28,
                        base_offsets=offsets.tolist(), arm_length_m=rollout.arm_length,
                        arm_radius_m=rollout.arm_radius, target_radius_m=TARGET_SPHERE_RADIUS,
                        timestep=index, time_s=float(rollout.ts[index]), current_state_only=True,
                        camera_eye=eye, camera_target=target, width=1600, height=1200, vertical_fov_degrees=45.,
                        mean_tracking_error_mm=float(1000*errors.mean()),
                        tracking_error_mm=(1000*errors).tolist(), mount_style=geom.base_plate_style)
        arrays = {}
        for name, m in [('robot', mesh), ('mount', mount_mesh)]:
            for part, a in zip(('points', 'indices', 'normals'), m, strict=True):
                arrays[f'{name}_{part}'] = a.astype(np.int32 if part == 'indices' else np.float32)
        np.savez_compressed(output, **arrays, targets=targets, metadata=json.dumps(metadata))
        print(output.name, 't=', metadata['time_s'], 'mean error [mm]=', metadata['mean_tracking_error_mm'])


if __name__ == '__main__':
    main()
