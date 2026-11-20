"""Export matched current-state geometry from the saved safety rollouts.

Only forward kinematics is evaluated. A single rigid display transform rotates
the robot, obstacles, and target together so the positive-z gravity frame is
shown as a hanging robot in the shared z-up paper studio.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from pcs_cf_cbf_clf_common import build_simulation_setup
from soromox.rendering.base_geometry import base_plate_mesh
from soromox.rendering.config import GeometryConfig

CASE = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tube_mesh(poses, radius, sides=48):
    """Sweep the PCS body and close its tip with the contact-model hemisphere."""
    angles = np.arange(sides) * 2 * np.pi / sides
    radial = np.stack((np.zeros(sides), np.cos(angles), np.sin(angles)), axis=1)
    normals = np.einsum('nij,kj->nki', poses[:, :3, :3], radial)
    rings = poses[:, None, :3, 3] + radius * normals
    points, normals = rings.reshape(-1, 3), normals.reshape(-1, 3)
    faces = []
    for i in range(len(poses)-1):
        for j in range(sides):
            a, b = i*sides+j, i*sides+(j+1) % sides
            faces.extend(((a, b, b+sides), (a, b+sides, a+sides)))
    # The contact model surrounds the terminal backbone sample with a sphere.
    # Continue the tube with its forward hemisphere, not a flat terminal face.
    tip, axis = poses[-1, :3, 3], poses[-1, :3, 0]
    last_ring = (len(poses)-1)*sides
    for latitude in np.linspace(0, np.pi/2, 25)[1:-1]:
        cap_normals = np.cos(latitude)*normals[(len(poses)-1)*sides:len(poses)*sides] + np.sin(latitude)*axis
        start = len(points)
        points = np.vstack((points, tip+radius*cap_normals))
        normals = np.vstack((normals, cap_normals))
        for j in range(sides):
            a, b = last_ring+j, last_ring+(j+1) % sides
            c, d = start+j, start+(j+1) % sides
            faces.extend(((a, b, d), (a, d, c)))
        last_ring = start
    pole = len(points)
    points = np.vstack((points, tip+radius*axis))
    normals = np.vstack((normals, axis))
    for j in range(sides):
        faces.append((last_ring+j, last_ring+(j+1) % sides, pole))
    # The mounting plate hides the flat proximal closure.
    start = len(points)
    points = np.vstack((points, rings[0], poses[0, :3, 3]))
    normals = np.vstack((normals, np.tile(-poses[0, :3, 0], (sides+1, 1))))
    for j in range(sides):
        faces.append((start+(j+1) % sides, start+j, start+sides))
    return points, np.asarray(faces), normals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=CASE/'data')
    parser.add_argument('--output-dir', type=Path, default=CASE/'outputs/paper_snapshots')
    parser.add_argument('--time', type=float, default=7.999)
    args = parser.parse_args()
    robot = build_simulation_setup().robot
    params_path = args.data_dir/'system_parameters.npz'
    with np.load(params_path) as params:
        lengths, radii = params['segment_length'], params['backbone_radius']
        force_limit = float(params['force_limit'])
    if not np.allclose(radii, radii[0]):
        raise ValueError('This scene exporter requires the saved constant-radius PCS arm')
    display_rotation = Rotation.from_euler('z', 25, degrees=True).as_matrix() @ Rotation.from_euler('x', 180, degrees=True).as_matrix()
    display_translation = np.array([0., 0., .45])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = display_rotation, display_translation
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method in ('without_cbf', 'with_cbf'):
        source = args.data_dir/f'rollout_{method}.npz'
        with np.load(source) as data:
            ts = data['ts']
            if not ts[0] <= args.time <= ts[-1]:
                raise ValueError('Requested snapshot is outside the saved rollout')
            index = int(np.argmin(abs(ts-args.time)))
            poses = np.asarray(robot.forward_kinematics_abscissa_batched(data['q_ts'][index], np.linspace(0, lengths.sum(), 161)))
            # Detect model/frame drift before making a convincing but incorrect rendering.
            distance = np.linalg.norm(poses[-1, :3, 3]-data['target_center'])
            np.testing.assert_allclose(distance, data['distance'][index], atol=1e-8)
            contact_points = np.asarray(robot.forward_kinematics_abscissa_batched(data['q_ts'][index], data['s_ps']))[:, :3, 3]
            clearance = np.linalg.norm(contact_points[:, None]-data['obs_centers'][None], axis=-1) - data['obs_radii'][None] - data['robot_radius']
            reconstructed_force = max(0., -1000.*float(clearance.min()))
            np.testing.assert_allclose(reconstructed_force, data['force'][index], atol=1e-7)
            poses = transform @ poses
            points, faces, normals = tube_mesh(poses, float(radii[0]))
            geom = GeometryConfig()
            mount_v, mount_f = base_plate_mesh(float(radii[0])*geom.base_plate_radius_scale, geom.base_plate_thickness, geom.base_plate_style)
            axis = poses[0, :3, 0]
            mount_rotation = Rotation.align_vectors([axis], [[0, 0, 1]])[0].as_matrix()
            mount_v = mount_v @ mount_rotation.T + poses[0, :3, 3] - .5*geom.base_plate_thickness*axis
            mount_n = np.asarray(trimesh.Trimesh(mount_v, mount_f, process=False).vertex_normals)
            metadata = dict(method=method, source_file=source.name, source_sha256=sha256(source),
                            parameters_sha256=sha256(params_path), exporter_sha256=sha256(Path(__file__)),
                            timestep=index, time_s=float(ts[index]), selection='last recorded state in both rollouts',
                            force_n=float(data['force'][index]), goal_distance_m=float(distance), force_limit_n=force_limit,
                            display_transform=transform.tolist(), mount_style=geom.base_plate_style,
                            end_effector_world=poses[-1, :3, 3].tolist(),
                            target_world=(data['target_center'] @ display_rotation.T+display_translation).tolist(),
                            current_state_only=True, tip_geometry='contact-model hemisphere',
                            contact_radius_m=float(data['robot_radius']),
                            maximum_penetration_m=float(max(0., -clearance.min())))
            np.savez_compressed(args.output_dir/f'{method}_geometry.npz',
                                robot_points=points.astype(np.float32), robot_indices=faces.astype(np.int32), robot_normals=normals.astype(np.float32),
                                mount_points=mount_v.astype(np.float32), mount_indices=mount_f.astype(np.int32), mount_normals=mount_n.astype(np.float32),
                                obs_centers=data['obs_centers'] @ display_rotation.T+display_translation,
                                obs_radii=data['obs_radii'], target=data['target_center'] @ display_rotation.T+display_translation,
                                metadata=json.dumps(metadata))
            print(method, metadata['time_s'], metadata['force_n'], metadata['goal_distance_m'])


if __name__ == '__main__':
    main()
