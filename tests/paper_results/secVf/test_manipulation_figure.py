"""Checks that publication summaries retain the saved experimental evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "paper_results" / "secVf_sampling_based_mpc" / "code"
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location(
    "manipulation_figure", CODE / "plot_smpc_summary.py"
)
figure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure)


def test_published_outcomes_and_all_six_paths():
    data = CODE.parent / "data"
    with (data / "square_icem_seed0.npz").open("rb") as stream:
        if stream.read(7) == b"version":
            pytest.skip("Fetch the released trajectory with git lfs pull")
    methods, goals = figure.load_runs(data)
    assert goals.shape == (4, 2)
    assert methods["iCEM"]["success"] == 3
    assert methods["MPPI"]["success"] == 1
    assert methods["iCEM"]["planning_mean_s"] == pytest.approx(26.543102, abs=1e-5)
    assert methods["MPPI"]["planning_mean_s"] == pytest.approx(8.503319, abs=1e-5)
    fig, ax = figure.plt.subplots()
    figure.draw_paths(ax, methods, goals)
    # The first line is the desired square, followed by all six recorded paths.
    expected = [
        xy * 1000 for method in methods.values() for xy in method["trajectories"]
    ]
    assert len(ax.lines) == 1 + len(expected)
    for line, xy in zip(ax.lines[1:], expected, strict=True):
        np.testing.assert_allclose(line.get_xydata(), xy)
    figure.plt.close(fig)


def test_rejects_snapshots_from_different_data_or_style(tmp_path):
    source = tmp_path / "run.npz"
    source.write_bytes(b"saved trajectory")
    root = tmp_path / "snapshots"
    view = root / "overview"
    view.mkdir(parents=True)
    manifest = {
        "source_sha256": "wrong",
        "view": "overview",
        "studio": figure.PAPER_STUDIO.manifest(),
        "snapshots": [],
    }
    (view / "snapshots.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="trajectory differs"):
        figure.load_snapshot(root, "overview", 19, source)
    manifest["source_sha256"] = figure.sha256(source)
    manifest["studio"]["robot"] = "#000000"
    (view / "snapshots.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="style is stale"):
        figure.load_snapshot(root, "overview", 19, source)


def test_existing_figure_requires_force(tmp_path):
    (tmp_path / "sampling_based_mpc.pdf").write_bytes(b"existing figure")
    with pytest.raises(FileExistsError, match="--force"):
        figure.main(["--output-dir", str(tmp_path)])
    assert (tmp_path / "sampling_based_mpc.pdf").read_bytes() == b"existing figure"


def test_linear_material_colors():
    from paper_render_style import rgb

    assert rgb("#FFFFFF", linear=True) == (1.0, 1.0, 1.0)
    assert rgb("#000000", linear=True) == (0.0, 0.0, 0.0)
    assert rgb("#808080", linear=True) == pytest.approx((0.2158605,) * 3)
    assert rgb("#808080") == pytest.approx((128 / 255,) * 3)


def test_sampling_frames_require_matching_iteration_and_source(tmp_path):
    source = tmp_path / "run.npz"
    source.write_bytes(b"saved samples")
    root = tmp_path / "sampling"
    root.mkdir()
    path = root / "iteration_0.png"
    path.write_bytes(b"rendered frame")
    metadata = {
        "source_sha256": figure.sha256(source),
        "studio": figure.PAPER_STUDIO.manifest(),
        "timestep": 0, "iteration": 0, "horizon_step": 10,
        "solid_configuration": "current_recorded_state",
    }
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata))
    assert figure.load_sampling(tmp_path, 0, source)[0] == path
    metadata["solid_configuration"] = "lowest_cost_prediction"
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="must show the current state"):
        figure.load_sampling(tmp_path, 0, source)
    metadata["solid_configuration"] = "current_recorded_state"
    metadata["iteration"] = 1
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unexpected sampling iteration"):
        figure.load_sampling(tmp_path, 0, source)
    metadata["iteration"] = 0
    metadata["source_sha256"] = "different run"
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="trajectory differs"):
        figure.load_sampling(tmp_path, 0, source)


def test_tradeoff_preserves_every_scored_point_and_checks_sources(tmp_path):
    source = tmp_path / "sweep.npz"
    source.write_bytes(b"recorded sweep")
    scored = {
        "sources": [{"file": source.name, "sha256": figure.sha256(source)}],
        "points": [
            {"family": family, "mean_time_s": time, "mean_negative_cost": cost}
            for family, time, cost in (("mppi", 10, -1200), ("icem_fixed_iters", 30, -1000),
                                      ("icem_fixed_batch", 30, -1000))
        ],
    }
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(scored))
    fig, ax = figure.plt.subplots()
    figure.draw_tradeoff(ax, figure.load_tradeoff(path, tmp_path))
    actual = np.concatenate([artist.get_offsets() for artist in ax.collections])
    np.testing.assert_allclose(actual, [[10, -1200], [30, -1000], [30, -1000]])
    figure.plt.close(fig)
    source.write_bytes(b"changed sweep")
    with pytest.raises(ValueError, match="sweep data changed"):
        figure.load_tradeoff(path, tmp_path)


def test_composed_trajectory_plot_preserves_equal_distance_scales(tmp_path):
    image = tmp_path / "render.png"
    figure.plt.imsave(image, np.ones((20, 20, 3)))
    goals = np.array([[0.05, -0.1], [0.15, -0.1], [0.15, 0.0], [0.05, 0.0]])
    methods = {name: {"color": color, "trajectories": [goals] * 3}
               for name, (_, color) in figure.METHODS.items()}
    fig = figure.build_figure(methods, goals, [(image, {})], [(image, {})] * 3,
                              {"points": []})
    fig.canvas.draw()
    ax = next(ax for ax in fig.axes if ax.get_xlabel() == "$x$ [mm]")
    origin, x_step, y_step = ax.transData.transform([[50, -100], [150, -100], [50, 0]])
    assert np.linalg.norm(x_step - origin) == pytest.approx(np.linalg.norm(y_step - origin))
    assert ax.get_aspect() == 1.0
    figure.plt.close(fig)


def test_future_poses_approach_the_same_goal_and_reject_transition():
    from figure_states import validate_future_steps

    source = CODE.parent / "data/square_icem_seed0.npz"
    with source.open("rb") as stream:
        if stream.read(7) == b"version":
            pytest.skip("Fetch the released trajectory with git lfs pull")
    with np.load(source) as data:
        poses, goals = data["object_pose"], data["goal_xy"]
    distances = validate_future_steps(poses, goals, 19, [21, 23])
    np.testing.assert_allclose(distances * 1000, [83.201699, 40.950878, 14.842057], atol=1e-5)
    with pytest.raises(ValueError, match="goal change"):
        validate_future_steps(poses, goals, 19, [21, 24])
    with pytest.raises(ValueError, match="chronological"):
        validate_future_steps(poses, goals, 19, [23, 21])
    wrong = poses.copy()
    wrong[21, :2] = [1, 1]
    with pytest.raises(ValueError, match="make progress"):
        validate_future_steps(wrong, goals, 19, [21, 23])


def test_ghost_mesh_preserves_orientation_and_scale():
    from figure_states import transform_mesh

    half = np.sqrt(0.5)
    points, normals = transform_mesh(np.array([[1, 0, 0]]), np.array([[1, 0, 0]]),
                                    np.array([3, 4, 5, 0, 0, half, half]), (2, 1, 1))
    np.testing.assert_allclose(points, [[3, 6, 5]], atol=1e-6)
    np.testing.assert_allclose(normals, [[0, 1, 0]], atol=1e-6)
