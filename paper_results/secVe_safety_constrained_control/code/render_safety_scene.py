"""Render an exported safety scene with the shared MPC paper studio on an RTX GPU."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eye', type=float, nargs=3, default=(.04, -.60, .4214))
    parser.add_argument('--target', type=float, nargs=3, default=(.04, -.01, .32))
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=1400)
    parser.add_argument('--samples-per-pixel', type=int, default=128)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError('Snapshot exists; pass --force to replace it')
    with np.load(args.geometry) as data:
        metadata = json.loads(str(data['metadata']))
        builder = newton.ModelBuilder()
        for i, (center, radius) in enumerate(zip(data['obs_centers'], data['obs_radii'], strict=True)):
            builder.add_shape_sphere(body=-1, xform=wp.transform(wp.vec3(*center), wp.quat_identity()), radius=float(radius),
                                     color=rgb(PAPER_STUDIO.obstacle, linear=True), label=f'obstacle_{i}')
        model = builder.finalize(device='cuda')
        viewer = PaperViewerRTX(width=args.width, height=args.height, headless=True, fps=30, samples_per_pixel=args.samples_per_pixel)
        viewer.set_model(model)
        look_at(viewer, args.eye, args.target)
        forward = np.asarray(args.target)-args.eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative_goal = data['target']-args.eye
        depth = float(relative_goal @ forward)
        half_height = depth*np.tan(np.deg2rad(viewer.camera.fov)/2)
        goal_pixel = [args.width/2 + (relative_goal @ right)*args.height/(2*half_height),
                      args.height/2 - (relative_goal @ up)*args.height/(2*half_height)]
        metadata.update(target_pixel_xy=goal_pixel, vertical_fov_degrees=float(viewer.camera.fov))
        for name, color in (('robot', PAPER_STUDIO.robot), ('mount', PAPER_STUDIO.mount)):
            viewer.log_mesh(name, wp.array(data[f'{name}_points'], dtype=wp.vec3), wp.array(data[f'{name}_indices'].flatten(), dtype=wp.int32),
                            normals=wp.array(data[f'{name}_normals'], dtype=wp.vec3), color=rgb(color, linear=True),
                            roughness=PAPER_STUDIO.robot_roughness, metallic=0.)
        viewer.begin_frame(0.)
        viewer.log_state(model.state())
        viewer.end_frame()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        viewer.save_screenshot(str(args.output))
        viewer.close()
    metadata.update(studio=PAPER_STUDIO.manifest(), camera_eye=args.eye, camera_target=args.target,
                    width=args.width, height=args.height, samples_per_pixel=args.samples_per_pixel,
                    renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    studio_renderer_sha256=hashlib.sha256((PAPER/'paper_rtx_studio.py').read_bytes()).hexdigest(),
                    geometry_sha256=hashlib.sha256(args.geometry.read_bytes()).hexdigest(), file=args.output.name)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
