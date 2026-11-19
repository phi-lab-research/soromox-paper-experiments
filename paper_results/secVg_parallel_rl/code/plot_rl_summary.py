"""Compose two complete parallel-rollout grids and the existing training evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CASE = Path(__file__).resolve().parents[1]
PAPER = CASE.parent
sys.path.insert(0, str(PAPER))
from paper_render_style import PAPER_STUDIO
from paper_style import PAPER_STYLE_PATH
import plot_rl


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshots(snapshot_dir, source):
    snapshots = []
    with np.load(source) as data:
        ts = data['ts']
        num_envs = data['q_ts'].shape[1]
    for i in (1, 2):
        path = snapshot_dir/f'grid_{i}.png'
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata['source_sha256'] != sha256(source):
            raise ValueError('Grid snapshot source differs from the saved rollout')
        if (metadata['policy'] != 'trained' or not metadata['current_state_only']
                or metadata['num_envs'] != num_envs or metadata['environment_indices'] != list(range(num_envs))):
            raise ValueError('Expected every saved environment from the trained-policy rollout')
        np.testing.assert_allclose(metadata['time_s'], ts[metadata['timestep']])
        if metadata['studio'] != json.loads(json.dumps(PAPER_STUDIO.manifest())):
            raise ValueError('Grid snapshot studio is stale')
        snapshots.append((path, dict(metadata, image_sha256=sha256(path))))
    for key in ('camera_eye', 'camera_target', 'width', 'height', 'vertical_fov_degrees', 'base_offsets', 'grid_shape'):
        if snapshots[0][1][key] != snapshots[1][1][key]:
            raise ValueError(f'Grid comparison differs in {key}')
    if snapshots[0][1]['time_s'] >= snapshots[1][1]['time_s']:
        raise ValueError('Snapshots must show increasing rollout times')
    return snapshots


def shared_grid_crop(snapshots):
    """Crop unused background using all geometry in both matched snapshots."""
    bounds = []
    for path, meta in snapshots:
        geometry = path.with_name(path.stem+'_geometry.npz')
        if sha256(geometry) != meta['geometry_sha256']:
            raise ValueError('Grid geometry differs from the rendered snapshot')
        with np.load(geometry) as data:
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            points = np.vstack((data['robot_points'], data['mount_points'],
                (data['targets'][:, None]+meta['target_radius_m']*corners).reshape(-1, 3)))
        eye = np.asarray(meta['camera_eye'])
        forward = np.asarray(meta['camera_target'])-eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        xyz = (points-eye) @ np.stack((right, up, forward), axis=1)
        scale = meta['height']/(2*xyz[:, 2]*np.tan(np.deg2rad(meta['vertical_fov_degrees'])/2))
        pixels = np.column_stack((meta['width']/2+scale*xyz[:, 0], meta['height']/2-scale*xyz[:, 1]))
        bounds.extend((pixels.min(axis=0), pixels.max(axis=0)))
    bounds = np.asarray(bounds)
    low, high = bounds.min(axis=0)-32, bounds.max(axis=0)+32
    center = (low+high)/2
    width = max(high[0]-low[0], (high[1]-low[1])*4/3)
    size = np.array([width, width*3/4])
    low, high = center-size/2, center+size/2
    if np.any(low < 0) or high[0] > meta['width'] or high[1] > meta['height']:
        raise ValueError('Crop would clip the matched scene; refit the camera')
    return [int(np.floor(low[0])), int(np.floor(low[1])), int(np.ceil(high[0])), int(np.ceil(high[1]))]


def build_figure(snapshots, groups):
    plt.style.use(PAPER_STYLE_PATH)
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 7, 'text.usetex': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.bbox': None,
                         'svg.hashsalt': 'soromox-parallel-rl'})
    width, height = 180, 134
    fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
    def rect(x, y, w, h):
        return [x/width, y/height, w/width, h/height]
    fig.text(2/width, 129/height, 'A', va='bottom', fontsize=11, fontweight='bold')
    fig.text(.5, 129/height, '64 parallel environments', va='bottom', ha='center', fontsize=8)
    left, top, right, bottom = shared_grid_crop(snapshots)
    for i, (path, metadata) in enumerate(snapshots):
        x = 2+i*90
        ax = fig.add_axes(rect(x, 58, 86, 64.5))
        ax.imshow(plt.imread(path)[top:bottom, left:right])
        ax.set_axis_off()
        fig.text((x+43)/width, 124/height, f"$t={metadata['time_s']:g}$ s", ha='center', fontsize=8)
    fig.text(2/width, 51/height, 'B', va='bottom', fontsize=11, fontweight='bold')
    ax = fig.add_axes(rect(15, 10, 162, 39))
    plot_rl.build_figure(groups, argparse.Namespace(points=500, episodes_per_point=0), ax=ax)
    for line in ax.lines:
        line.set_linewidth(1.25)
        if '(CPU,' in line.get_label():
            line.set_linestyle('--')
    ax.set(ylabel='Mean episode return', ylim=(-60, 130), yticks=(-50, 0, 50, 100))
    ax.grid(color='.9', linewidth=.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', ncol=2, frameon=False, handlelength=1.8, columnspacing=1.5, labelspacing=.35)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', type=Path, default=CASE/'outputs/paper_snapshots')
    parser.add_argument('--source', type=Path, default=CASE/'data/traj/rl_rollout_trained_64_envs.npz')
    parser.add_argument('--csv-dir', type=Path, default=CASE/'data/reward_logs')
    parser.add_argument('--output-dir', type=Path, default=PAPER/'final_outputs')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    stem = args.output_dir/'parallel_reinforcement_learning'
    outputs = [stem.with_suffix(ext) for ext in ('.pdf', '.png', '.svg', '.json')]
    if any(p.exists() for p in outputs) and not args.force:
        raise FileExistsError('Figure exists; pass --force to replace it')
    snapshots = load_snapshots(args.snapshot_dir, args.source)
    groups = plot_rl.discover_csv_groups(args.csv_dir)
    fig = build_figure(snapshots, groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for p in outputs[:3]:
        fig.savefig(p, dpi=300)
        if p.suffix == '.svg':
            p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)
    sources = {label: [dict(file=str(p.relative_to(args.csv_dir)), sha256=sha256(p)) for p in paths]
               for label, paths in groups.items()}
    outputs[3].write_text(json.dumps(dict(studio=PAPER_STUDIO.manifest(), snapshots=[m for _, m in snapshots],
        shared_crop_xyxy=shared_grid_crop(snapshots), reward_sources=sources, reward_processing=dict(interpolation_points=500, aggregation='mean and sample standard deviation',
            common_time_window=True, random_sample_points=plot_rl.WANDB_SAMPLE_POINTS,
            random_seed=plot_rl.WANDB_SAMPLE_SEED, gaussian_smoothing_records=plot_rl.WANDB_SMOOTHING)), indent=2)+'\n')
    print(outputs[0])


if __name__ == '__main__':
    main()
