"""Export recorded control states into the shared studio; no new simulation."""

import argparse
import json
from pathlib import Path
import sys

import jax
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from soromox.rendering.base_geometry import base_plate_mesh
from soromox.rendering.config import GeometryConfig

from control_tuning_data import CASE, PAPER, TRACKING, load_data, sha256
from operational_space_impedance_common import build_problem, load_run_npz
from secvd_case import build_sec_vd_robot

# Reuse the same flat-ended swept body used by the RL rendering.
sys.path.insert(0, str(PAPER / "secVg_parallel_rl/code"))
from export_rl_grid_scene import tube_mesh, combine_meshes, path_mesh


def fit_camera(points, direction, aspect, margin=0.06):
    target = (points.min(0) + points.max(0)) / 2
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    forward = -direction
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    low, high = 0.01, 2.0
    for _ in range(45):
        distance = (low + high) / 2
        eye = target + direction * distance
        xyz = (points - eye) @ np.stack((right, up, forward), axis=1)
        half = xyz[:, 2] * np.tan(np.deg2rad(45) / 2)
        fits = (
            xyz[:, 2].min() > 0
            and max(
                np.max(abs(xyz[:, 0]) / half / aspect), np.max(abs(xyz[:, 1]) / half)
            )
            < 1 - 2 * margin
        )
        if fits:
            high = distance
        else:
            low = distance
    return (target + direction * high).tolist(), target.tolist()


def mounting(poses):
    geom = GeometryConfig()
    v, f = base_plate_mesh(
        0.02 * geom.base_plate_radius_scale,
        geom.base_plate_thickness,
        geom.base_plate_style,
    )
    axis = poses[0, :3, 0]
    rotation = Rotation.align_vectors([axis], [[0, 0, 1.0]])[0].as_matrix()
    v = v @ rotation.T + poses[0, :3, 3] - 0.5 * geom.base_plate_thickness * axis
    return v, f, np.asarray(trimesh.Trimesh(v, f, process=False).vertex_normals)


def target_frame(pose):
    angles = np.linspace(0, 2 * np.pi, 81)
    points = pose[:3, 3] + 0.022 * (
        np.cos(angles)[:, None] * pose[:3, 1] + np.sin(angles)[:, None] * pose[:3, 2]
    )
    meshes = [path_mesh(points, 0.0006)]
    for axis, length in [(0, 0.02), (1, 0.022)]:
        meshes.append(
            path_mesh(
                np.stack((pose[:3, 3], pose[:3, 3] + length * pose[:3, axis])), 0.0006
            )
        )
    return combine_meshes(meshes)


def export(output, motion=False, snapshot_time=5.0):
    tracking, cases, audit = load_data()
    run = load_run_npz(TRACKING)
    tracking_robot = build_problem(run.trajectory_config).robot
    gain_robot, _, length = build_sec_vd_robot()
    assert tracking_robot.num_internal_dofs == 12 and gain_robot.num_internal_dofs == 6
    fk_tracking = jax.jit(
        jax.vmap(tracking_robot.forward_kinematics_abscissa_batched, in_axes=(0, None))
    )
    fk_gain = jax.jit(
        jax.vmap(gain_robot.forward_kinematics_abscissa_batched, in_axes=(0, None))
    )
    # Verify saved poses against their real model, including orientation.
    for q, x, fk, length in [
        (tracking["q_traj"], tracking["x_traj_full"], fk_tracking, 0.2)
    ] + [
        (
            cases["synergistic"][f"q_ts_{tag}"][batch],
            cases["synergistic"][f"x_ts_{tag}"][batch],
            fk_gain,
            0.1,
        )
        for tag, batch in [
            ("init", cases["synergistic"]["initial_batch"]),
            ("best", int(cases["synergistic"]["best_batch"])),
        ]
    ]:
        poses = np.asarray(fk(q, np.array([length])))[:, 0]
        np.testing.assert_allclose(poses[:, :3, 3], x[:, 3:], atol=1e-8)
        np.testing.assert_allclose(
            poses[:, :3, :3], Rotation.from_rotvec(x[:, :3]).as_matrix(), atol=1e-7
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    scenes = []
    transform = np.eye(4)
    transform[2, 3] = 0.01
    index = int(np.argmin(abs(tracking["t_traj"] - 6.25)))
    poses = (
        transform
        @ np.asarray(
            fk_tracking(tracking["q_traj"][index : index + 1], np.linspace(0, 0.2, 129))
        )[0]
    )
    desired = tracking["x_des_traj_full"][index]
    target_pose = np.eye(4)
    target_pose[:3, :3] = Rotation.from_rotvec(desired[:3]).as_matrix()
    target_pose[:3, 3] = desired[3:]
    target_pose = transform @ target_pose
    period = (tracking["t_traj"] >= 4) & (tracking["t_traj"] <= 8)
    reference = tracking["x_des_traj_full"][period, 3:] + transform[:3, 3]
    meshes = dict(
        robot=tube_mesh(poses, 0.02, sides=48),
        mount=mounting(poses),
        target_path=path_mesh(reference, 0.0008, dashed=True),
        target_frame=target_frame(target_pose),
    )
    ghosts = []
    for time in (4.0, 4.75, 7.25):
        ghost_index = int(np.argmin(abs(tracking["t_traj"] - time)))
        ghost_poses = (
            transform
            @ np.asarray(
                fk_tracking(
                    tracking["q_traj"][ghost_index : ghost_index + 1],
                    np.linspace(0, 0.2, 129),
                )
            )[0]
        )
        name = f"recorded_pose_{ghost_index}"
        # Near-coincident proximal surfaces cause RTX transparency artifacts.
        # This omitted portion lies inside the solid body in the displayed view.
        vertices, faces, normals = tube_mesh(ghost_poses[10:], 0.02, sides=48)
        # Omit the coincident base cap; the single shared solid mount stays visible.
        faces = np.delete(faces, np.s_[-96:-48], axis=0)
        # Ghost surfaces embedded in the opaque body must not tint its surface.
        a, b = poses[:-1, :3, 3], poses[1:, :3, 3]
        ab = b - a
        fraction = np.clip(
            np.einsum("vsi,si->vs", vertices[:, None] - a, ab)
            / np.sum(ab * ab, axis=1),
            0,
            1,
        )
        distance = np.linalg.norm(
            vertices[:, None] - (a + fraction[..., None] * ab), axis=2
        ).min(axis=1)
        faces = faces[(distance[faces] > 0.0202).all(axis=1)]
        meshes[name] = vertices, faces, normals
        ghosts.append(
            dict(
                mesh_name=name,
                timestep=ghost_index,
                time_s=float(tracking["t_traj"][ghost_index]),
                opacity=0.20,
                omitted_proximal_length_m=0.2 * 10 / 128,
                embedded_faces_omitted=True,
                end_effector_world=ghost_poses[-1, :3, 3].tolist(),
            )
        )
    eye, target = fit_camera(
        np.vstack([m[0] for m in meshes.values()]), [0.15, -1.0, 1.3], 1640 / 1400
    )
    # Keep the solid state in the foreground of this motion illustration.
    # Cull ghost triangles crossing its projected silhouette; transparent RTX
    # surfaces otherwise tint the opaque body at near intersections.
    from PIL import Image, ImageDraw
    from plot_control_tuning_summary import project_point

    projection = dict(
        camera_eye=eye,
        camera_target=target,
        width=1640,
        height=1400,
        vertical_fov_degrees=45.0,
    )
    silhouette = Image.new("1", (1640, 1400))
    painter = ImageDraw.Draw(silhouette)
    solid_pixels = np.array([project_point(v, projection) for v in meshes["robot"][0]])
    for tri in meshes["robot"][1]:
        painter.polygon([tuple(p) for p in solid_pixels[tri]], fill=1)
    mask = np.asarray(silhouette)
    for ghost in ghosts:
        name = ghost["mesh_name"]
        vertices, faces, normals = meshes[name]
        pixels = np.array([project_point(v, projection) for v in vertices])
        samples = np.concatenate(
            (pixels[faces], pixels[faces].mean(axis=1, keepdims=True)), axis=1
        )
        xy = np.rint(samples).astype(int)
        x = np.clip(xy[..., 0], 0, 1639)
        y = np.clip(xy[..., 1], 0, 1399)
        faces = faces[~mask[y, x].any(axis=1)]
        meshes[name] = vertices, faces, normals
        ghost["opaque_state_foreground"] = True
    scenes.append(
        (
            "tracking",
            meshes,
            dict(
                kind="tracking",
                num_segments=2,
                source_sha256=audit["tracking"]["source_sha256"],
                timestep=index,
                time_s=float(tracking["t_traj"][index]),
                camera_eye=eye,
                camera_target=target,
                width=1640,
                height=1400,
                end_effector_world=poses[-1, :3, 3].tolist(),
                target_world=target_pose[:3, 3].tolist(),
                display_transform=transform.tolist(),
                path_interval_s=[4.0, 8.0],
                recorded_pose_overlays=ghosts,
                overlay_semantics="other recorded states along the same tracking rollout",
                tip_facing_camera_cosine=float(
                    poses[-1, :3, 0]
                    @ (
                        (np.asarray(eye) - poses[-1, :3, 3])
                        / np.linalg.norm(np.asarray(eye) - poses[-1, :3, 3])
                    )
                ),
            ),
        )
    )
    if motion:
        scenes.clear()
    d = cases["synergistic"]
    transform = np.eye(4)
    transform[:3, :3] = (
        Rotation.from_euler("z", -90, degrees=True).as_matrix()
        @ Rotation.from_euler("x", 180, degrees=True).as_matrix()
    )
    transform[2, 3] = 0.28
    sampled = np.unique(np.rint(np.linspace(0, len(d["t_ts"]) - 1, 101)).astype(int))
    state_bounds = []
    for tag, batch in [("init", d["initial_batch"]), ("best", int(d["best_batch"]))]:
        p = transform @ np.asarray(
            fk_gain(d[f"q_ts_{tag}"][batch, sampled], np.linspace(0, 0.1, 65))
        )
        # Entire recorded motion determines one common camera for both runs.
        centerlines = p[:, :, :3, 3].reshape(-1, 3)
        state_bounds.extend((centerlines - 0.02, centerlines + 0.02, mounting(p[0])[0]))
    eye, target = fit_camera(
        np.vstack(state_bounds), [-0.25, -1.0, 0.16], 0.72, margin=0.08
    )
    if not 0 <= snapshot_time <= d["t_ts"][-1]:
        raise ValueError("Snapshot time must lie within the recorded rollout")
    times = np.linspace(0, 5, 61) if motion else [snapshot_time]
    for frame, time in enumerate(times):
        index = int(np.argmin(abs(d["t_ts"] - time)))
        for name, tag, batch in [
            ("initial", "init", d["initial_batch"]),
            ("optimized", "best", int(d["best_batch"])),
        ]:
            poses = (
                transform
                @ np.asarray(
                    fk_gain(
                        d[f"q_ts_{tag}"][batch, index : index + 1],
                        np.linspace(0, 0.1, 97),
                    )
                )[0]
            )
            meshes = dict(robot=tube_mesh(poses, 0.02, sides=48), mount=mounting(poses))
            target_point = (
                transform[:3, :3] @ d["x_des_ts"][index, 3:] + transform[:3, 3]
            )
            meta = dict(
                kind=name,
                num_segments=1,
                source_sha256=audit["synergistic"]["source_sha256"],
                batch_zero_based=int(batch),
                timestep=index,
                time_s=float(d["t_ts"][index]),
                camera_eye=eye,
                camera_target=target,
                width=648 if motion else 1008,
                height=900 if motion else 1400,
                end_effector_world=poses[-1, :3, 3].tolist(),
                target_world=target_point.tolist(),
                display_transform=transform.tolist(),
            )
            scenes.append((f"{name}_{frame:03d}" if motion else name, meshes, meta))
    for name, meshes, meta in scenes:
        meta.update(
            exporter_sha256=sha256(Path(__file__)),
            vertical_fov_degrees=45.0,
            current_state_only=not bool(meta.get("recorded_pose_overlays")),
            solid_state_is_recorded=True,
            tip_geometry="flat-ended swept PCS surface",
            mesh_names=list(meshes),
        )
        arrays = {}
        for key, mesh in meshes.items():
            for part, a in zip(("points", "indices", "normals"), mesh, strict=True):
                arrays[f"{key}_{part}"] = a.astype(
                    np.int32 if part == "indices" else np.float32
                )
        np.savez_compressed(
            output / f"{name}_geometry.npz", **arrays, metadata=json.dumps(meta)
        )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=CASE / "outputs/paper_snapshots"
    )
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--snapshot-time", type=float, default=5.0,
                        help="Time of matched tuning snapshots in seconds")
    args = parser.parse_args()
    export(args.output_dir, args.motion, args.snapshot_time)
