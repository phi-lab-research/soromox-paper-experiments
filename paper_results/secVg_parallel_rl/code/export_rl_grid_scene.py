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

from render_rl_video import build_rl_robot, load_rollout
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


def path_mesh(points, radius, dashed=False):
    meshes = []
    for i, (a, b) in enumerate(zip(points[:-1], points[1:], strict=True)):
        if (dashed and (i//3) % 2) or np.linalg.norm(b-a) < 1e-8:
            continue
        mesh = trimesh.creation.cylinder(radius=radius, segment=np.stack((a, b)), sections=8)
        meshes.append((np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.vertex_normals)))
    return combine_meshes(meshes)


def extended_grid(eye, target, base_z):
    """Tile recorded environments beyond the visible frustum, not new simulations."""
    rows, cols = np.meshgrid(np.arange(80), np.arange(-128, 128), indexing='ij')
    offsets = np.column_stack(((cols.ravel()+.5)*.24, -.84+rows.ravel()*.24,
                               np.full(rows.size, base_z)))
    # Conservative bounds cover all saved shapes, targets, and mounts.
    corners = np.array([[x, y, z] for x in (-.16, .16) for y in (-.16, .16) for z in (-.06, .32)])
    forward = np.asarray(target)-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    xyz = (offsets[:, None]+corners-eye) @ np.stack((right, up, forward), axis=1)
    scale = 720/(2*xyz[:, :, 2]*np.tan(np.deg2rad(45)/2))
    x, y = 900+scale*xyz[:, :, 0], 360-scale*xyz[:, :, 1]
    visible = ((x.max(1) >= 0) & (x.min(1) <= 1800) & (y.max(1) >= 235) & (y.min(1) <= 720))
    ids = ((rows.ravel()*13+cols.ravel()*7) % 64)[visible]
    return offsets[visible], ids.tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=CASE/'data/traj/rl_rollout_trained_64_envs.npz')
    parser.add_argument('--output-dir', type=Path, default=CASE/'outputs/paper_snapshots')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    outputs = [args.output_dir/f'{name}_geometry.npz' for name in ('grid_close', 'tracking')]
    if any(p.exists() for p in outputs) and not args.force:
        raise FileExistsError('Geometry exists; pass --force to replace it')
    rollout = load_rollout(args.data)
    if rollout.policy != 'trained' or rollout.num_envs != 64:
        raise ValueError('Expected the complete saved 64-environment trained-policy rollout')
    if not np.isfinite(rollout.q_ts).all() or not np.isfinite(rollout.ball_ts).all():
        raise ValueError('Nonfinite trajectory')
    robot = build_rl_robot(rollout.arm_length, rollout.arm_radius)
    s = np.linspace(0, rollout.arm_length, 65)
    fk = jax.jit(jax.vmap(robot.forward_kinematics_abscissa_batched, in_axes=(0, None)))
    geom = GeometryConfig()
    mount_v, mount_f = base_plate_mesh(rollout.arm_radius*geom.base_plate_radius_scale,
                                      geom.base_plate_thickness, geom.base_plate_style)
    mount_v[:, 2] -= geom.base_plate_thickness/2
    mount_n = np.asarray(trimesh.Trimesh(mount_v, mount_f, process=False).vertex_normals)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind, output, time in zip(('grid', 'tracking'), outputs, (5., 10.), strict=True):
        index = int(np.argmin(abs(rollout.ts-time)))
        np.testing.assert_allclose(rollout.ts[index], time)
        if kind == 'grid':
            eye, target, width, height = [0., -1.5, .30], [0., -.2, .08], 1800, 720
            offsets, ids = extended_grid(np.asarray(eye), target, geom.base_plate_thickness-.05)
            source_poses = np.asarray(fk(rollout.q_ts[:, index], s)).copy()
            meshes, mounts = [], []
            cache = {}
            for env, offset in zip(ids, offsets, strict=True):
                distance = np.linalg.norm(offset-np.asarray(eye))
                stride, sides, mount_res = (1, 32, 48) if distance < 2 else ((4, 16, 24) if distance < 5 else (8, 12, 12))
                key = (env, stride)
                if key not in cache:
                    cache[key] = tube_mesh(source_poses[env, ::stride], rollout.arm_radius, sides=sides)
                v, f, n = cache[key]
                meshes.append((v+offset, f, n))
                mount_key = ('mount', mount_res)
                if mount_key not in cache:
                    mv, mf = base_plate_mesh(rollout.arm_radius*geom.base_plate_radius_scale, geom.base_plate_thickness, geom.base_plate_style, resolution=mount_res)
                    mv[:, 2] -= geom.base_plate_thickness/2
                    cache[mount_key] = (mv, mf, np.asarray(trimesh.Trimesh(mv, mf, process=False).vertex_normals))
                mv, mf, mn = cache[mount_key]
                mounts.append((mv+offset, mf, mn))
            mesh, mount_mesh = combine_meshes(meshes), combine_meshes(mounts)
            errors = np.linalg.norm(source_poses[:, -1, :3, 3]-rollout.ball_ts[:, index], axis=1)
        else:
            ids = [0]
            offsets = np.array([[0., 0., geom.base_plate_thickness-.05]])
            poses = np.asarray(fk(rollout.q_ts[ids, index], s)).copy()
            errors = np.linalg.norm(poses[:, -1, :3, 3]-rollout.ball_ts[ids, index], axis=1)
            poses[:, :, :3, 3] += offsets[:, None]
            mesh = combine_meshes([tube_mesh(p, rollout.arm_radius) for p in poses])
            mount_mesh = combine_meshes([(mount_v+o, mount_f, mount_n) for o in offsets])
            eye, target, width, height = [.23, -.37, .29], [-.01, 0., .105], 1200, 1000
        targets = rollout.ball_ts[ids, index]+offsets
        metadata = dict(source_file=args.data.name, source_sha256=sha256(args.data),
                        exporter_sha256=sha256(Path(__file__)), policy='trained', num_envs=64 if kind == 'grid' else 1,
                        display_instance_count=len(ids), display_tiling=kind == 'grid',
                        backdrop_start_m=24. if kind == 'grid' else 3.,
                        scene_kind=kind, environment_indices=ids, grid_spacing_m=.24 if kind == 'grid' else None,
                        base_offsets=offsets.tolist(), arm_length_m=rollout.arm_length,
                        arm_radius_m=rollout.arm_radius, target_radius_m=TARGET_SPHERE_RADIUS,
                        timestep=index, time_s=float(rollout.ts[index]), current_state_only=True,
                        camera_eye=eye, camera_target=target, width=width, height=height, vertical_fov_degrees=45.,
                        mean_tracking_error_mm=float(1000*errors.mean()),
                        mount_style=geom.base_plate_style, framing='close perspective' if kind == 'grid' else 'complete robot')
        arrays = {}
        for name, m in [('robot', mesh), ('mount', mount_mesh)]:
            for part, a in zip(('points', 'indices', 'normals'), m, strict=True):
                arrays[f'{name}_{part}'] = a.astype(np.int32 if part == 'indices' else np.float32)
        if kind == 'tracking':
            actual = np.asarray(fk(rollout.q_ts[0], np.array([rollout.arm_length])))[:, 0, :3, 3]+offsets[0]
            desired = rollout.ball_ts[0]+offsets[0]
            arrays.update(actual_path=actual, target_path=desired, path_times=rollout.ts)
            for name, points, radius, dashed in [('actual_trace', actual, .0009, False), ('target_trace', desired, .0016, True)]:
                for part, a in zip(('points', 'indices', 'normals'), path_mesh(points, radius, dashed), strict=True):
                    arrays[f'{name}_{part}'] = a.astype(np.int32 if part == 'indices' else np.float32)
            metadata.update(path_interval_s=[float(rollout.ts[0]), float(rollout.ts[-1])],
                            path_samples=len(rollout.ts), environment_selection='environment 0, fixed index',
                            mean_path_error_mm=float(1000*np.linalg.norm(actual-desired, axis=1).mean()))
        np.savez_compressed(output, **arrays, targets=targets, metadata=json.dumps(metadata))
        print(output.name, 'instances', len(ids), 'time', metadata['time_s'])


if __name__ == '__main__':
    main()
