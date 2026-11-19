"""Verify the full saved grid, matched views, and retained training evidence."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT/'paper_results/secVg_parallel_rl'
sys.path.insert(0, str(CASE/'code'))
spec = importlib.util.spec_from_file_location('rl_summary', CASE/'code/plot_rl_summary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
SOURCE = CASE/'data/traj/rl_rollout_trained_64_envs.npz'
SNAPSHOTS = CASE/'outputs/paper_snapshots'


def test_complete_grid_preserves_saved_targets_and_time():
    snapshots = summary.load_snapshots(SNAPSHOTS, SOURCE)
    with np.load(SOURCE) as saved:
        for path, meta in snapshots:
            assert meta['num_envs'] == 64
            with np.load(path.with_name(path.stem+'_geometry.npz')) as scene:
                np.testing.assert_allclose(scene['targets']-meta['base_offsets'], saved['ball_ts'][meta['timestep']], atol=1e-12)
                assert len(scene['targets']) == 64
                assert meta['time_s'] in (5., 15.)
                assert np.isfinite(scene['robot_points']).all()
    left, top, right, bottom = summary.shared_grid_crop(snapshots)
    assert 0 <= left < right <= 1600 and 0 <= top < bottom <= 1200
    # The production crop includes every projected mesh/target bound plus margin.
    assert (right-left)/(bottom-top) == pytest.approx(4/3, abs=.003)


def test_reject_mismatched_grid_camera(tmp_path):
    for i in (1, 2):
        p = SNAPSHOTS/f'grid_{i}.json'
        meta = json.loads(p.read_text())
        if i == 2:
            meta['camera_eye'][0] += .1
        (tmp_path/p.name).write_text(json.dumps(meta))
        (tmp_path/f'grid_{i}.png').write_bytes(b'image')
    with pytest.raises(ValueError, match='camera_eye'):
        summary.load_snapshots(tmp_path, SOURCE)


def test_training_comparisons_and_uncertainty_fit_in_plot():
    snapshots = summary.load_snapshots(SNAPSHOTS, SOURCE)
    groups = summary.plot_rl.discover_csv_groups(CASE/'data/reward_logs')
    assert len(groups) == 6 and all(len(paths) == 2 for paths in groups.values())
    fig = summary.build_figure(snapshots, groups)
    try:
        ax = fig.axes[-1]
        assert len(ax.lines) == 6
        assert [line.get_label() for line in ax.lines] == list(summary.plot_rl.COLORS_RL)
        for band in ax.collections:
            for path in band.get_paths():
                assert path.vertices[:, 1].min() >= ax.get_ylim()[0]
                assert path.vertices[:, 1].max() <= ax.get_ylim()[1]
    finally:
        summary.plt.close(fig)


def test_existing_output_requires_force(tmp_path):
    p = tmp_path/'parallel_reinforcement_learning.pdf'
    p.write_bytes(b'original')
    with pytest.raises(FileExistsError, match='--force'):
        summary.main(['--output-dir', str(tmp_path)])
    assert p.read_bytes() == b'original'
