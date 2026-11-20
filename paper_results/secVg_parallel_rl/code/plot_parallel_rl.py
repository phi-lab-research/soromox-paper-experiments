"""Compose a close parallel grid, trained-policy tracking, and learning curves."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

CASE = Path(__file__).resolve().parents[1]
PAPER = CASE.parent
sys.path.insert(0, str(PAPER))
from paper_render_style import PAPER_STUDIO
from paper_style import PAPER_STYLE_PATH
import rl_training_curves


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshots(snapshot_dir, source):
    snapshots = []
    with np.load(source) as data:
        ts = data['ts']
        num_envs = data['q_ts'].shape[1]
    for name, ids in [('grid_close', list(range(num_envs))), ('tracking', [0])]:
        path = snapshot_dir/f'{name}.png'
        geometry = snapshot_dir/f'{name}_geometry.npz'
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata['source_sha256'] != sha256(source):
            raise ValueError('Snapshot source differs from the saved rollout')
        if (metadata['policy'] != 'trained' or not metadata['current_state_only']
                or (name == 'tracking' and metadata['environment_indices'] != ids)
                or (name == 'grid_close' and set(metadata['environment_indices']) != set(ids))):
            raise ValueError('Unexpected policy or selected environments')
        np.testing.assert_allclose(metadata['time_s'], ts[metadata['timestep']])
        if metadata['studio'] != json.loads(json.dumps(PAPER_STUDIO.manifest())):
            raise ValueError('Snapshot studio is stale')
        if sha256(geometry) != metadata['geometry_sha256']:
            raise ValueError('Geometry differs from the rendered snapshot')
        snapshots.append((path, dict(metadata, image_sha256=sha256(path))))
    with np.load(snapshot_dir/'tracking_geometry.npz') as data:
        np.testing.assert_array_equal(data['path_times'], ts)
        if len(data['actual_path']) != len(ts) or len(data['target_path']) != len(ts):
            raise ValueError('Tracking paths must cover the complete saved rollout')
    return snapshots


def build_figure(snapshots, groups):
    plt.style.use(PAPER_STYLE_PATH)
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'legend.fontsize': 7, 'text.usetex': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.bbox': None,
                         'svg.hashsalt': 'soromox-parallel-rl'})
    width, height = 180, 119
    fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
    def rect(x, y, w, h):
        return [x/width, y/height, w/width, h/height]
    for label, x, y in [('B', 2, 59), ('C', 43, 59)]:
        fig.text(x/width, y/height, label, va='bottom', fontsize=11, fontweight='bold')
    ax_grid = fig.add_axes(rect(2, 64, 176, 176*540/1800))
    ax_grid.imshow(plt.imread(snapshots[0][0])[160:700])
    ax_grid.set_axis_off()
    ax_grid.text(.008, .978, 'A', transform=ax_grid.transAxes, va='top',
                 fontsize=11, fontweight='bold')
    ax_tracking = fig.add_axes(rect(2, 2, 640*54.17/1000, 54.17))
    ax_tracking.imshow(plt.imread(snapshots[1][0])[:, 240:880])
    ax_tracking.set_axis_off()
    fig.legend(handles=[
        Line2D([], [], color=PAPER_STUDIO.target, ls='--', label='Target'),
        Line2D([], [], color=rl_training_curves.COLORS['pre_opt_1'], label='End effector'),
    ], loc='center', bbox_to_anchor=(23/width, 60.1/height), ncol=1,
       frameon=False, fontsize=7.5, handlelength=1.5, labelspacing=.15)
    ax = fig.add_axes(rect(53, 11, 124, 47.5))
    rl_training_curves.build_figure(groups, argparse.Namespace(points=500, episodes_per_point=0), ax=ax)
    for line in ax.lines:
        line.set_linewidth(1.25)
        if '(CPU,' in line.get_label():
            line.set_linestyle('--')
    ax.set(ylabel='Mean episode return', ylim=(-60, 130), yticks=(-50, 0, 50, 100))
    ax.grid(color='.9', linewidth=.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, fontsize=8, handlelength=1.7, labelspacing=.3)
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
    groups = rl_training_curves.discover_csv_groups(args.csv_dir)
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
        grid_crop_xyxy=[0, 160, 1800, 700], tracking_crop_xyxy=[240, 0, 880, 1000], reward_sources=sources, reward_processing=dict(interpolation_points=500, aggregation='mean and sample standard deviation',
            common_time_window=True, random_sample_points=rl_training_curves.WANDB_SAMPLE_POINTS,
            random_seed=rl_training_curves.WANDB_SAMPLE_SEED, gaussian_smoothing_records=rl_training_curves.WANDB_SMOOTHING)), indent=2)+'\n')
    print(outputs[0])


if __name__ == '__main__':
    main()
