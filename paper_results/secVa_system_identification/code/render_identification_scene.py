"""Render identification meshes using the shared paper RTX studio."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import newton
import numpy as np
import warp as wp

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER))
from paper_render_style import PAPER_STUDIO, rgb
from paper_rtx_studio import PaperViewerRTX, look_at


def render(geometry, output, samples=128, force=False):
    if output.exists() and not force:
        raise FileExistsError("Image exists; pass --force")
    with np.load(geometry) as d:
        meta = json.loads(str(d["metadata"]))
        viewer = PaperViewerRTX(
            width=meta["width"],
            height=meta["height"],
            headless=True,
            fps=30,
            samples_per_pixel=samples,
        )
        model = newton.ModelBuilder().finalize(device="cuda")
        viewer.set_model(model)
        look_at(viewer, meta["camera_eye"], meta["camera_target"])
        viewer.shadow_excluded_mesh_names = (
            ("measurements",) if meta["kind"] == "overview" else ("robot",)
        )
        for name in meta["mesh_names"]:
            viewer.log_mesh(
                name,
                wp.array(d[name + "_points"], dtype=wp.vec3),
                wp.array(d[name + "_indices"].flatten(), dtype=wp.int32),
                normals=wp.array(d[name + "_normals"], dtype=wp.vec3),
                color=rgb(meta["mesh_colors"][name], linear=True),
                roughness=PAPER_STUDIO.robot_roughness,
                metallic=0.0,
                opacity=meta["mesh_opacity"].get(name, 1.0),
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
    render(args.geometry, args.output, args.samples, args.force)
