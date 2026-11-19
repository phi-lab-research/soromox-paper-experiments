"""Preserve saved safety evidence and reject mismatched visual comparisons."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT/'paper_results/secVe_safety_constrained_control'
spec = importlib.util.spec_from_file_location('safety_figure', CASE/'code/plot_safety_summary.py')
figure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure)


def test_all_saved_trace_points_and_force_limit():
    results, limit = figure.load_data(CASE/'data')
    assert limit == 5
    assert results['without_cbf']['force'].max() == pytest.approx(33.61205838)
    assert results['with_cbf']['force'].max() < limit
    assert results['without_cbf']['distance'][-1]*1000 == pytest.approx(.04175686)
    assert results['with_cbf']['distance'][-1]*1000 == pytest.approx(30.0577294)
    snapshots = figure.load_snapshots(CASE/'outputs/paper_snapshots', results, CASE/'data/system_parameters.npz')
    fig = figure.build_figure(results, limit, snapshots)
    try:
        force_ax, distance_ax = fig.axes[-2:]
        for i, r in enumerate(results.values()):
            np.testing.assert_array_equal(force_ax.lines[i].get_ydata(), r['force'])
            np.testing.assert_array_equal(distance_ax.lines[i].get_ydata(), r['distance']*1000)
            np.testing.assert_array_equal(force_ax.lines[i].get_xdata(), r['ts'])
        assert [t.get_text() for t in force_ax.get_legend().get_texts()] == ['HOCLF', 'HOCLF + HOCBF', '5 N limit']
    finally:
        figure.plt.close(fig)


def test_reject_different_camera_or_saved_data(tmp_path):
    results, _ = figure.load_data(CASE/'data')
    for key in figure.METHODS:
        source = CASE/f'outputs/paper_snapshots/{key}.json'
        (tmp_path/source.name).write_text(source.read_text())
        (tmp_path/f'{key}.png').write_bytes(b'image')
    path = tmp_path/'with_cbf.json'
    meta = json.loads(path.read_text())
    meta['camera_eye'][0] += .1
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='camera_eye'):
        figure.load_snapshots(tmp_path, results, CASE/'data/system_parameters.npz')
    meta['source_sha256'] = 'unrelated trajectory'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='data differ'):
        figure.load_snapshots(tmp_path, results, CASE/'data/system_parameters.npz')


def test_output_requires_explicit_replacement(tmp_path):
    path = tmp_path/'safety_constrained_control.pdf'
    path.write_bytes(b'existing')
    with pytest.raises(FileExistsError, match='--force'):
        figure.main(['--output-dir', str(tmp_path)])
    assert path.read_bytes() == b'existing'


def test_rendered_tip_matches_saved_contact_penetration():
    """The visible surface must reproduce the force-producing tip contact."""
    import sys
    import trimesh

    sys.path.insert(0, str(CASE/'code'))
    from pcs_cf_cbf_clf_common import build_simulation_setup
    robot = build_simulation_setup().robot
    for key in figure.METHODS:
        with np.load(CASE/f'data/rollout_{key}.npz') as saved:
            tip_pose = np.asarray(robot.forward_kinematics_abscissa_batched(saved['q_ts'][-1], saved['s_ps']))[-1]
            with np.load(CASE/f'outputs/paper_snapshots/{key}_geometry.npz') as geometry:
                meta = json.loads(str(geometry['metadata']))
                transform = np.asarray(meta['display_transform'])
                tip = (transform @ tip_pose)[:3, 3]
                axis = (transform @ tip_pose)[:3, 0]
                mesh = trimesh.Trimesh(geometry['robot_points'], geometry['robot_indices'], process=True)
                assert mesh.is_watertight
                center = geometry['obs_centers'][1]
                # Both active contacts lie ahead of the terminal cross-section.
                assert (center-tip) @ axis > 0
                _, distance, _ = trimesh.proximity.closest_point_naive(mesh, center[None])
                inside_tip = np.linalg.norm(center-tip) < saved['robot_radius']
                signed_distance = -distance[0] if inside_tip else distance[0]
                visual_penetration = geometry['obs_radii'][1]-signed_distance
                assert visual_penetration == pytest.approx(saved['force'][-1]/1000, abs=1e-4)


def test_position_markers_use_camera_projection():
    results, _ = figure.load_data(CASE/'data')
    snapshots = figure.load_snapshots(CASE/'outputs/paper_snapshots', results, CASE/'data/system_parameters.npz')
    for _, meta in snapshots:
        np.testing.assert_allclose(figure.project_point(meta['camera_target'], meta),
                                   [meta['width']/2, meta['height']/2])
    nominal, constrained = [meta for _, meta in snapshots]
    assert np.linalg.norm(np.asarray(nominal['end_effector_pixel_xy'])-nominal['target_pixel_xy']) < 1
    assert np.linalg.norm(np.asarray(constrained['end_effector_pixel_xy'])-constrained['target_pixel_xy']) > 50
