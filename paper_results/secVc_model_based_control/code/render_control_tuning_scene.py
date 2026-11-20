"""Render exported control/tuning states with the shared paper RTX studio."""

import argparse
import json
from pathlib import Path
import sys
import hashlib

import numpy as np
import newton
import warp as wp

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER))
from paper_render_style import PAPER_STUDIO, rgb
from paper_rtx_studio import PaperViewerRTX, look_at


def render(geometry, output, samples=128):
    with np.load(geometry) as data:
        meta = json.loads(str(data["metadata"]))
        viewer = PaperViewerRTX(
            width=meta["width"],
            height=meta["height"],
            headless=True,
            fps=30,
            samples_per_pixel=samples,
        )
        model = newton.ModelBuilder().finalize(device="cuda")
        viewer.set_model(model)
        np.testing.assert_allclose(viewer.camera.fov, meta["vertical_fov_degrees"])
        look_at(viewer, meta["camera_eye"], meta["camera_target"])
        overlays = {m["mesh_name"]: m for m in meta.get("recorded_pose_overlays", [])}
        viewer.shadow_excluded_mesh_names = ("target_path", "target_frame", *overlays)
        for name in meta["mesh_names"]:
            color = (
                PAPER_STUDIO.robot
                if name == "robot" or name in overlays
                else (PAPER_STUDIO.mount if name == "mount" else PAPER_STUDIO.target)
            )
            viewer.log_mesh(
                name,
                wp.array(data[name + "_points"], dtype=wp.vec3),
                wp.array(data[name + "_indices"].flatten(), dtype=wp.int32),
                normals=wp.array(data[name + "_normals"], dtype=wp.vec3),
                color=rgb(color, linear=True),
                roughness=PAPER_STUDIO.robot_roughness,
                metallic=0.0,
                opacity=overlays[name]["opacity"] if name in overlays else None,
            )
        viewer.begin_frame(0.0)
        viewer.log_state(model.state())
        viewer.end_frame()
        output.parent.mkdir(parents=True, exist_ok=True)
        viewer.save_screenshot(str(output))
        viewer.close()
    meta.update(
        studio=PAPER_STUDIO.manifest(),
        samples_per_pixel=samples,
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        studio_renderer_sha256=hashlib.sha256(
            (PAPER / "paper_rtx_studio.py").read_bytes()
        ).hexdigest(),
        geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
    )
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError("Output exists; use --force")
    render(args.geometry, args.output, args.samples)
