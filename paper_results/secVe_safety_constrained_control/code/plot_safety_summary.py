"""Compose the safety figure from saved traces and matched studio snapshots."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

CASE = Path(__file__).resolve().parents[1]
PAPER = CASE.parent
sys.path.insert(0, str(PAPER))
from paper_render_style import PAPER_STUDIO
from paper_style import PAPER_COLORS, PAPER_STYLE_PATH

METHODS = {
    'without_cbf': ('HOCLF', PAPER_COLORS['pre_opt_1'], '-'),
    'with_cbf': ('HOCLF + HOCBF', PAPER_COLORS['post_opt_1'], '--'),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(data_dir):
    with np.load(data_dir/'system_parameters.npz') as params:
        limit = float(params['force_limit'])
    results = {}
    for key in METHODS:
        path = data_dir/f'rollout_{key}.npz'
        with np.load(path) as data:
            results[key] = {name: data[name].copy() for name in ('ts', 'force', 'distance', 'q_ts', 'obs_centers', 'obs_radii', 'target_center')}
        r = results[key]
        if not all(np.isfinite(r[k]).all() for k in ('ts', 'force', 'distance')):
            raise ValueError('Nonfinite saved plot data')
        if not np.all(np.diff(r['ts']) > 0):
            raise ValueError('Saved times must increase')
        r['source_sha256'] = sha256(path)
    for name in ('ts', 'obs_centers', 'obs_radii', 'target_center'):
        np.testing.assert_allclose(results['without_cbf'][name], results['with_cbf'][name])
    return results, limit


def load_snapshots(root, results, parameters_path):
    snapshots = []
    for key in METHODS:
        path = root/f'{key}.png'
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['source_sha256'] != results[key]['source_sha256'] or meta['parameters_sha256'] != sha256(parameters_path):
            raise ValueError('Snapshot data differ from plotted data')
        if meta['studio'] != json.loads(json.dumps(PAPER_STUDIO.manifest())):
            raise ValueError('Snapshot studio is stale')
        if meta.get('tip_geometry') != 'contact-model hemisphere':
            raise ValueError('Snapshot tip does not represent the contact-model hemisphere')
        if not meta['current_state_only'] or meta['timestep'] != len(results[key]['ts'])-1:
            raise ValueError('Expected the final recorded current state')
        np.testing.assert_allclose(meta['time_s'], results[key]['ts'][-1])
        np.testing.assert_allclose(meta['force_n'], results[key]['force'][-1])
        np.testing.assert_allclose(meta['goal_distance_m'], results[key]['distance'][-1], atol=1e-8)
        snapshots.append((path, {**meta, 'sha256': sha256(path)}))
    for field in ('camera_eye', 'camera_target', 'display_transform', 'time_s', 'width', 'height'):
        if snapshots[0][1][field] != snapshots[1][1][field]:
            raise ValueError(f'Snapshot comparison differs in {field}')
    return snapshots


def build_figure(results, limit, snapshots):
    plt.style.use(PAPER_STYLE_PATH)
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'text.usetex': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.bbox': None,
                         'svg.hashsalt': 'soromox-safety'})
    width, height = 180, 130
    fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
    def rect(x, y, w, h):
        return [x/width, y/height, w/width, h/height]
    for label, x, y in (('A', 2, 125), ('B', 2, 37), ('C', 94, 37)):
        fig.text(x/width, y/height, label, va='bottom', fontsize=11, fontweight='bold')
    for i, (key, (path, meta)) in enumerate(zip(METHODS, snapshots, strict=True)):
        x = 2+i*90
        ax = fig.add_axes(rect(x, 43, 86, 75.25))
        ax.imshow(plt.imread(path))
        # A screen-space marker at the exact 3D target projection remains readable
        # when the current robot reaches and occludes the target.
        ax.scatter(*meta['target_pixel_xy'], s=48, facecolors='none', edgecolors=PAPER_STUDIO.target,
                   linewidths=1.2, path_effects=[pe.Stroke(linewidth=2, foreground='white'), pe.Normal()])
        ax.set_axis_off()
        fig.text((x+43)/width, 121/height, METHODS[key][0], color=METHODS[key][1], ha='center', fontsize=8)
    axes = [fig.add_axes(rect(x, 10, 72, 24)) for x in (16, 106)]
    for ax, metric, scale, ylabel in zip(axes, ('force', 'distance'), (1, 1000),
                                       ('Maximum pairwise\nnormal force [N]', 'Goal distance [mm]'), strict=True):
        for key, (label, color, style) in METHODS.items():
            r = results[key]
            ax.plot(r['ts'], scale*r[metric], color=color, ls=style, lw=1.3, label=label)
        ax.set(xlim=(0, 8), xticks=(0, 2, 4, 6, 8), xlabel='Time [s]', ylabel=ylabel)
        ax.grid(color='.9', lw=.5)
        ax.set_axisbelow(True)
    axes[0].axhline(limit, color='.3', ls=':', lw=1, label=f'{limit:g} N limit')
    axes[0].set(ylim=(0, 37), yticks=(0, 5, 15, 25, 35))
    axes[0].legend(loc='center right', frameon=False, handlelength=1.6, labelspacing=.25)
    axes[1].set(ylim=(0, 120), yticks=(0, 30, 60, 90, 120))
    axes[1].legend(loc='upper right', frameon=False, handlelength=1.6)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=CASE/'data')
    parser.add_argument('--snapshot-dir', type=Path, default=CASE/'outputs/paper_snapshots')
    parser.add_argument('--output-dir', type=Path, default=PAPER/'final_outputs')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    stem = args.output_dir/'safety_constrained_control'
    paths = [stem.with_suffix(e) for e in ('.pdf', '.png', '.svg', '.json')]
    if any(p.exists() for p in paths) and not args.force:
        raise FileExistsError('Figure exists; pass --force to replace it')
    results, limit = load_data(args.data_dir)
    snapshots = load_snapshots(args.snapshot_dir, results, args.data_dir/'system_parameters.npz')
    fig = build_figure(results, limit, snapshots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for p in paths[:3]:
        fig.savefig(p, dpi=300)
        if p.suffix == '.svg':
            p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)
    metrics = {key: dict(source_sha256=r['source_sha256'], maximum_force_n=float(r['force'].max()),
                        maximum_violation_n=float(np.maximum(r['force']-limit, 0).max()),
                        final_distance_mm=float(1000*r['distance'][-1]), final_time_s=float(r['ts'][-1]))
               for key, r in results.items()}
    paths[3].write_text(json.dumps(dict(studio=PAPER_STUDIO.manifest(), force_limit_n=limit,
                                       metrics=metrics, snapshots=[m for _, m in snapshots]), indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
