"""Render a complete saved RL environment grid in the shared paper RTX studio."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('PXR_WORK_THREAD_LIMIT', '1')
import newton
import numpy as np
import warp as wp

PAPER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPER))
from paper_render_style import PAPER_STUDIO, rgb
from paper_rtx_studio import PaperViewerRTX, look_at


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-pixel', type=int, default=128)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists() and not args.force:
        raise FileExistsError('Snapshot exists; pass --force to replace it')
    with np.load(args.geometry) as data:
        meta = json.loads(str(data['metadata']))
        builder = newton.ModelBuilder()
        for i, center in enumerate(data['targets']):
            builder.add_shape_sphere(body=-1, xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                                     radius=meta['target_radius_m'], color=rgb(PAPER_STUDIO.target, linear=True),
                                     label=f'target_{i}')
        model = builder.finalize(device='cuda')
        viewer = PaperViewerRTX(width=meta['width'], height=meta['height'], headless=True, fps=30,
                                samples_per_pixel=args.samples_per_pixel)
        viewer.set_model(model)
        np.testing.assert_allclose(viewer.camera.fov, meta['vertical_fov_degrees'])
        look_at(viewer, meta['camera_eye'], meta['camera_target'])
        for name, color in [('robot', PAPER_STUDIO.robot), ('mount', PAPER_STUDIO.mount)]:
            viewer.log_mesh(name, wp.array(data[f'{name}_points'], dtype=wp.vec3),
                            wp.array(data[f'{name}_indices'].flatten(), dtype=wp.int32),
                            normals=wp.array(data[f'{name}_normals'], dtype=wp.vec3),
                            color=rgb(color, linear=True), roughness=PAPER_STUDIO.robot_roughness, metallic=0.)
        viewer.begin_frame(0.)
        viewer.log_state(model.state())
        viewer.end_frame()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        viewer.save_screenshot(str(args.output))
        viewer.close()
    meta.update(studio=PAPER_STUDIO.manifest(), samples_per_pixel=args.samples_per_pixel,
                renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                studio_renderer_sha256=hashlib.sha256((PAPER/'paper_rtx_studio.py').read_bytes()).hexdigest(),
                geometry_sha256=hashlib.sha256(args.geometry.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
