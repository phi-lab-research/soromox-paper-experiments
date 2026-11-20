"""Verify recorded tracking evidence and explicitly tiled display geometry."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT/'paper_results/secVg_parallel_rl'
sys.path.insert(0, str(CASE/'code'))
spec = importlib.util.spec_from_file_location('rl_summary', CASE/'code/plot_parallel_rl.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
SOURCE = CASE/'data/traj/rl_rollout_trained_64_envs.npz'
SNAPSHOTS = CASE/'outputs/paper_snapshots'


def test_grid_repetition_is_declared_and_preserves_targets():
    snapshots = summary.load_snapshots(SNAPSHOTS, SOURCE)
    _, meta = snapshots[0]
    assert meta['num_envs'] == 64 and meta['display_tiling']
    assert meta['display_instance_count'] > 64
    assert set(meta['environment_indices']) == set(range(64))
    with np.load(SOURCE) as saved, np.load(SNAPSHOTS/'grid_close_geometry.npz') as scene:
        np.testing.assert_allclose(scene['targets']-meta['base_offsets'],
                                   saved['ball_ts'][meta['timestep'], meta['environment_indices']], atol=1e-12)
    direction = np.asarray(meta['camera_target'])-meta['camera_eye']
    assert abs(np.rad2deg(np.arctan2(direction[2], np.linalg.norm(direction[:2])))) < 15
    offsets = np.asarray(meta['base_offsets'])
    assert np.ptp(offsets[:, 0]) > 20 and np.ptp(offsets[:, 1]) > 10


def test_tracking_paths_use_complete_recorded_rollout():
    import jax
    from render_rl_video import build_rl_robot

    snapshots = summary.load_snapshots(SNAPSHOTS, SOURCE)
    _, meta = snapshots[1]
    assert meta['environment_indices'] == [0] and not meta['display_tiling']
    with np.load(SOURCE) as saved, np.load(SNAPSHOTS/'tracking_geometry.npz') as scene:
        robot = build_rl_robot(float(saved['arm_length']), float(saved['arm_radius']))
        fk = jax.jit(jax.vmap(robot.forward_kinematics_abscissa_batched, in_axes=(0, None)))
        actual = np.asarray(fk(saved['q_ts'][:, 0], np.array([float(saved['arm_length'])])))[:, 0, :3, 3]
        offset = np.asarray(meta['base_offsets'])[0]
        np.testing.assert_allclose(scene['actual_path']-offset, actual, atol=1e-10)
        np.testing.assert_allclose(scene['target_path']-offset, saved['ball_ts'][:, 0], atol=1e-12)
        np.testing.assert_array_equal(scene['path_times'], saved['ts'])
        assert meta['path_interval_s'] == [0., 20.]


def test_reject_modified_geometry(tmp_path):
    import shutil
    for name in ('grid_close', 'tracking'):
        for suffix in ('.json', '.png', '_geometry.npz'):
            shutil.copy2(SNAPSHOTS/(name+suffix), tmp_path/(name+suffix))
    (tmp_path/'tracking_geometry.npz').write_bytes(b'unrelated geometry')
    with pytest.raises(ValueError, match='Geometry differs'):
        summary.load_snapshots(tmp_path, SOURCE)


def test_training_comparisons_and_uncertainty_fit_in_plot():
    snapshots = summary.load_snapshots(SNAPSHOTS, SOURCE)
    groups = summary.rl_training_curves.discover_csv_groups(CASE/'data/reward_logs')
    assert len(groups) == 6 and all(len(paths) == 2 for paths in groups.values())
    fig = summary.build_figure(snapshots, groups)
    try:
        ax = fig.axes[-1]
        assert [line.get_label() for line in ax.lines] == list(summary.rl_training_curves.COLORS_RL)
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
