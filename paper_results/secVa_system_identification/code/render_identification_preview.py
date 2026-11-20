"""Render a local Open3D preview of the audited identification scenes.

The RTX renderer remains the final studio backend. This local option allows
layout review without transferring geometry to the GPU workstation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER))
from paper_render_style import PAPER_STUDIO, rgb


def render(geometry, output, force=False):
    if output.exists() and not force:
        raise FileExistsError("Preview exists; pass --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(geometry) as d:
        meta = json.loads(str(d["metadata"]))
        w, h = meta["width"], meta["height"]
        renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
        scene = renderer.scene
        scene.set_background(np.array([*rgb(PAPER_STUDIO.backdrop), 1.0]))
        scene.set_lighting(scene.SOFT_SHADOWS, (0, 0, -1))
        scene.scene.enable_sun_light(False)
        scene.scene.set_indirect_light_intensity(30000)
        scene.view.set_antialiasing(True)
        scene.view.set_ambient_occlusion(True)
        scene.view.set_shadowing(True, o3d.visualization.rendering.View.ShadowType.VSM)
        grading = o3d.visualization.rendering.ColorGrading
        scene.view.set_color_grading(
            grading(grading.Quality.ULTRA, grading.ToneMapping.FILMIC)
        )
        direction = Rotation.from_euler(
            "xyz", PAPER_STUDIO.key_rotation_degrees, degrees=True
        ).apply([0, 0, -1])
        scene.scene.add_directional_light("key", [1, 1, 1], direction, 60000, True)
        for meshname in meta["mesh_names"]:
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(d[meshname + "_points"]),
                o3d.utility.Vector3iVector(d[meshname + "_indices"]),
            )
            mesh.vertex_normals = o3d.utility.Vector3dVector(d[meshname + "_normals"])
            mat = o3d.visualization.rendering.MaterialRecord()
            alpha = meta["mesh_opacity"].get(meshname, 1.0)
            mat.shader = "defaultLitTransparency" if alpha < 1 else "defaultLit"
            mat.base_color = [*rgb(meta["mesh_colors"][meshname], linear=True), alpha]
            mat.base_roughness = PAPER_STUDIO.robot_roughness
            mat.base_metallic = 0.0
            scene.add_geometry(meshname, mesh, mat)
        floor = o3d.geometry.TriangleMesh.create_box(20, 20, 0.01)
        floor.translate([-10, -10, -0.06])
        floor.compute_vertex_normals()
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = [*rgb(PAPER_STUDIO.backdrop, linear=True), 1.0]
        mat.base_roughness = 1.0
        scene.add_geometry("floor", floor, mat)
        renderer.setup_camera(
            45.0, meta["camera_target"], meta["camera_eye"], [0, 0, 1], 0.001, 20.0
        )
        image = renderer.render_to_image()
        o3d.io.write_image(str(output), image)
        meta.update(
            backend="Open3D local preview",
            studio=PAPER_STUDIO.manifest(),
            renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
            calibration={
                "ambient_intensity": 30000,
                "key_intensity": 60000,
                "tone_mapping": "FILMIC",
            },
        )
        (output.with_suffix(".json")).write_text(json.dumps(meta))
        print(output)
        del renderer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    render(args.geometry, args.output, args.force)
