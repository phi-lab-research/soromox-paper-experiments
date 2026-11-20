"""Check the scientific comparisons and provenance of the combined figure."""

import importlib.util
from pathlib import Path
import sys
import json
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "paper_results/secVc_model_based_control"
sys.path.insert(0, str(CASE / "code"))
spec = importlib.util.spec_from_file_location(
    "control_tuning_summary", CASE / "code/plot_control_tuning_summary.py"
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.fixture(scope="module")
def data():
    return summary.load_data()


def test_models_and_run_selection_are_audited(data):
    tracking, cases, audit = data
    assert tracking["q_traj"].shape[1] == 12
    for method in ("collocated", "synergistic"):
        assert cases[method]["q_ts_init"].shape[2] == 6
        assert cases[method]["u_ts_init"].shape[2] == 3
    assert audit["synergistic"]["initial_batch_zero_based"] == 4
    assert audit["synergistic"]["optimized_batch_zero_based"] == 1
    assert audit["synergistic"]["optimized_iteration_zero_based"] == 99
    assert (
        audit["synergistic"]["improvement_from_batch_median_percent"]
        != audit["synergistic"]["improvement_from_selected_initial_percent"]
    )
    # load_data rescored every initial and best rollout against recorded losses.
    np.testing.assert_allclose(audit["synergistic"]["optimized_settling_s"], 0.455)


def test_matched_views_preserve_physical_tip_target_distance(data):
    tracking, cases, audit = data
    snapshots = summary.load_snapshots(CASE / "outputs/paper_snapshots", *data)
    for name, tag in [("initial", "init"), ("optimized", "best")]:
        meta = snapshots[name][1]
        distance = 1000 * np.linalg.norm(
            np.asarray(meta["end_effector_world"]) - meta["target_world"]
        )
        np.testing.assert_allclose(
            distance,
            cases["synergistic"][f"position_error_{tag}_mm"][meta["timestep"]],
            atol=1e-7,
        )
        assert meta["num_segments"] == 1
        np.testing.assert_allclose(meta["time_s"], cases["synergistic"]["t_ts"][-1])
    assert snapshots["tracking"][1]["num_segments"] == 2


def test_reject_wrong_run_or_modified_geometry(data, tmp_path):
    source = CASE / "outputs/paper_snapshots"
    for name in ("tracking", "initial", "optimized"):
        for suffix in (".json", ".png", "_geometry.npz"):
            shutil.copy2(source / (name + suffix), tmp_path / (name + suffix))
    meta = json.loads((tmp_path / "initial.json").read_text())
    meta["batch_zero_based"] = 1
    (tmp_path / "initial.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="wrong run"):
        summary.load_snapshots(tmp_path, *data)
    shutil.copy2(source / "initial.json", tmp_path / "initial.json")
    (tmp_path / "optimized_geometry.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="geometry changed"):
        summary.load_snapshots(tmp_path, *data)


def test_plots_keep_complete_recorded_intervals(data):
    snapshots = summary.load_snapshots(CASE / "outputs/paper_snapshots", *data)
    fig = summary.build_figure(data[0], data[1], snapshots)
    try:
        for ax in fig.axes[3:5]:
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), data[0]["t_traj"])
        for line in fig.axes[-1].lines[::2]:
            np.testing.assert_array_equal(
                line.get_xdata(), data[1]["synergistic"]["t_ts"]
            )
        d = data[1]["synergistic"]
        errors = 1000 * np.linalg.norm(
            d["x_ts_init"][:, :, 3:] - d["x_des_ts"][None, :, 3:], axis=-1
        )
        vertices = fig.axes[-1].collections[0].get_paths()[0].vertices
        for i in [0, 500, 5000]:
            ys = vertices[np.isclose(vertices[:, 0], d["t_ts"][i]), 1]
            np.testing.assert_allclose(
                [ys.min(), ys.max()], [errors[:, i].min(), errors[:, i].max()]
            )
        assert fig.axes[-1].get_ylim()[1] >= errors.max()
    finally:
        summary.plt.close(fig)


def test_existing_output_requires_force(tmp_path):
    pdf = tmp_path / "control_and_gain_optimization.pdf"
    pdf.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        summary.main(["--output-dir", str(tmp_path)])
    assert pdf.read_bytes() == b"original"


def test_tracking_overlays_are_recorded_and_solid_tip_faces_camera(data):
    from scipy.spatial.transform import Rotation

    snapshots = summary.load_snapshots(CASE / "outputs/paper_snapshots", *data)
    meta = snapshots["tracking"][1]
    tracking = data[0]
    transform = np.asarray(meta["display_transform"])
    axis = (
        transform[:3, :3]
        @ Rotation.from_rotvec(
            tracking["x_traj_full"][meta["timestep"], :3]
        ).as_matrix()[:, 0]
    )
    view = np.asarray(meta["camera_eye"]) - meta["end_effector_world"]
    view /= np.linalg.norm(view)
    assert axis @ view > 0.8
    assert len(meta["recorded_pose_overlays"]) == 3
    assert meta["solid_state_is_recorded"] and not meta["current_state_only"]
    for pose in meta["recorded_pose_overlays"]:
        assert pose["timestep"] != meta["timestep"]
        assert 0 < pose["opacity"] < 0.5
        np.testing.assert_allclose(pose["time_s"], tracking["t_traj"][pose["timestep"]])
