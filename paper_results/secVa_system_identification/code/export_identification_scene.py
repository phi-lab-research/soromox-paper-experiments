"""Export measured markers and model shapes into the shared paper studio."""

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import trimesh
from soromox.rendering.base_geometry import base_plate_mesh
from soromox.rendering.config import GeometryConfig

from identification_data import (
    CASE,
    CSV_MARKERS,
    build_robot,
    representative_indices,
    sha256,
)

PAPER = CASE.parent
sys.path.insert(0, str(PAPER))
from paper_render_style import PAPER_STUDIO


def fit_camera(points, direction, aspect, margin=0.055):
    target = (points.min(0) + points.max(0)) / 2
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    forward = -direction
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    low, high = 0.01, 3.0
    for _ in range(48):
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


def combine(meshes):
    return trimesh.util.concatenate(meshes)


def body_mesh(poses, s):
    radius = np.interp(s, [0, 0.305, 0.36], [0.01541, 0.00642, 0.0048])
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    radial = np.stack((np.zeros(64), np.cos(angles), np.sin(angles)), axis=-1)
    rings = poses[:, None, :3, 3] + radius[:, None, None] * np.einsum(
        "nij,kj->nki", poses[:, :3, :3], radial
    )
    points = rings.reshape(-1, 3)
    faces = []
    for i in range(len(s) - 1):
        for j in range(64):
            a, b = i * 64 + j, i * 64 + (j + 1) % 64
            faces.extend([(a, b, b + 64), (a, b + 64, a + 64)])
    for end, sign in [(0, -1), (-1, 1)]:
        start = len(points)
        points = np.vstack((points, rings[end], poses[end, :3, 3]))
        for j in range(64):
            triangle = (start + 64, start + j, start + (j + 1) % 64)
            faces.append(triangle if sign == 1 else triangle[::-1])
    return trimesh.Trimesh(points, np.array(faces), process=False)


def mounting(base):
    config = GeometryConfig()
    v, f = base_plate_mesh(
        0.01541 * config.base_plate_radius_scale,
        config.base_plate_thickness,
        config.base_plate_style,
    )
    # Hanging installation: the mount sits above the fixed end.
    v[:, 2] = -v[:, 2] + config.base_plate_thickness / 2
    v += base
    return trimesh.Trimesh(v, f, process=False)


def export(evaluation, output, force=False):
    evaluation_sha256 = sha256(evaluation / "identification_evaluation.npz")
    if output.exists() and list(output.glob("*_geometry.npz")) and not force:
        raise FileExistsError("Geometry exists; pass --force")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(evaluation / "identification_evaluation.npz") as d:
        selected = representative_indices(d["identified_errors"])
        robot = build_robot()
        marker_s = jnp.array([0.1291, 0.2195, 0.2800, 0.36])
        offsets = np.array(
            [
                [0.0, 0.0, 0.025],
                [0.0, 0.0, -0.021],
                [0.0, 0.0, 0.020],
                [0.008, 0.0, 0.0],
            ]
        )
        marker_fk = jax.jit(
            jax.vmap(lambda q: robot.forward_kinematics_abscissa_batched(q, marker_s))
        )
        marker_geometry = {}
        for stage in ("initial", "identified", "learned", "fitted_residual"):
            frames = np.asarray(marker_fk(d[stage + "_q"]))
            anchors = frames[:, :, :3, 3]
            predicted = anchors + np.einsum(
                "nmij,mj->nmi", frames[:, :, :3, :3], offsets
            )
            np.testing.assert_allclose(predicted, d[stage + "_markers"], atol=1e-12)
            marker_geometry[stage + "_anchors"] = anchors
        marker_geometry_path = output / "marker_geometry.npz"
        np.savez_compressed(
            marker_geometry_path,
            **marker_geometry,
            abscissae=np.asarray(marker_s),
            local_offsets=offsets,
        )

        # Raise the hanging system into the shared floor scene. No rotation,
        # deformation, or registration is applied to measurements or predictions.
        shift = np.array([0.0, 0.0, 0.43])
        s = d["s"]
        all_points = []
        for index in selected:
            for stage in ["initial", "identified", "learned"]:
                all_points.extend(
                    [
                        d[stage + "_poses"][index, :, :3, 3] + shift,
                        d[stage + "_markers"][index] + shift,
                    ]
                )
            all_points.append(d["measured"][index] + shift)
        all_points = np.vstack(all_points)
        all_points = np.vstack(
            (all_points - 0.017, all_points + 0.017, mounting(shift).vertices)
        )
        # Oblique front view exposes the two tendon bending directions.
        direction = [0.0, -1.0, 0.25]
        common_eye, common_target = fit_camera(all_points, direction, 34 / 57)
        for kind, index in [("overview", selected[2])] + [
            (f"comparison_{i + 1}", idx) for i, idx in enumerate(selected)
        ]:
            poses = d["learned_poses"][index].copy()
            poses[:, :3, 3] += shift
            body = body_mesh(poses, s)
            mount = mounting(shift)
            meshes = {"robot": body, "mount": mount}
            if kind == "overview":
                balls = []
                for point in d["measured"][index] + shift:
                    ball = trimesh.creation.icosphere(subdivisions=3, radius=0.004)
                    ball.apply_translation(point)
                    balls.append(ball)
                meshes["measurements"] = combine(balls)
                bounds = np.vstack([m.vertices for m in meshes.values()])
                eye, target = fit_camera(bounds, direction, 64 / 109, margin=0.055)
                width, height = 1152, 1962
            else:
                eye, target = common_eye, common_target
                width, height = 680, 1140
            meta = dict(
                kind=kind,
                configuration_index_zero_based=index,
                csv=CSV_MARKERS[index][0],
                geometry_source="learned residual static equilibrium; no measured continuous surface is available",
                mounting="schematic mount from the shared model renderer; not hardware CAD",
                display_translation=shift.tolist(),
                width=width,
                height=height,
                camera_eye=eye,
                camera_target=target,
                vertical_fov_degrees=45.0,
                mesh_names=list(meshes),
                mesh_colors={
                    "robot": PAPER_STUDIO.robot,
                    "mount": PAPER_STUDIO.mount,
                    "measurements": "#FFBC42",
                },
                mesh_opacity={"robot": 1.0 if kind == "overview" else 0.32},
                evaluation_sha256=evaluation_sha256,
                marker_geometry_sha256=sha256(marker_geometry_path),
                exporter_sha256=sha256(Path(__file__)),
            )
            arrays = {}
            for name, mesh in meshes.items():
                for field, value in [
                    ("points", mesh.vertices),
                    ("indices", mesh.faces),
                    ("normals", mesh.vertex_normals),
                ]:
                    arrays[name + "_" + field] = np.asarray(
                        value, dtype=np.int32 if field == "indices" else np.float32
                    )
            np.savez_compressed(
                output / f"{kind}_geometry.npz", **arrays, metadata=json.dumps(meta)
            )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir", type=Path, default=CASE / "outputs/identification"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=CASE / "outputs/paper_snapshots"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    export(args.evaluation_dir, args.output_dir, args.force)
