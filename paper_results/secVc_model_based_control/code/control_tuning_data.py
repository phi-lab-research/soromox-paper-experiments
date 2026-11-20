"""Audit the saved control/tuning examples without repeating any simulation."""

import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

CASE = Path(__file__).resolve().parents[1]
PAPER = CASE.parent
GAIN = PAPER / "secVd_control_gain_optimization"
for directory in (GAIN / "code", CASE / "operational_space_impedance_control/code"):
    sys.path.insert(0, str(directory))
from secvd_results import load_results
from secvd_objective import recompute_archive_losses
from render_control_gain_optimization_animations import initial_median_batch

TRACKING = (
    CASE
    / "operational_space_impedance_control/data/control_pcs_with_impedance_trajectory.npz"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    with np.load(TRACKING) as archive:
        tracking = {k: archive[k].copy() for k in archive.files}
    x, desired = tracking["x_traj_full"], tracking["x_des_traj_full"]
    tracking["position_error_mm"] = 1000 * np.linalg.norm(
        x[:, 3:] - desired[:, 3:], axis=1
    )
    tracking["orientation_error_deg"] = np.rad2deg(
        (
            Rotation.from_rotvec(desired[:, :3]) * Rotation.from_rotvec(x[:, :3]).inv()
        ).magnitude()
    )
    np.testing.assert_allclose(
        np.sqrt(np.mean(tracking["position_error_mm"] ** 2)),
        1000 * tracking["metrics_position_rmse"],
    )
    np.testing.assert_allclose(
        np.sqrt(np.mean(tracking["orientation_error_deg"] ** 2)),
        np.rad2deg(tracking["metrics_orientation_rmse"]),
    )
    assert tracking["q_traj"].shape[1] == 12
    audit = {
        "tracking": dict(
            segments=2,
            length_m=0.2,
            source_sha256=sha256(TRACKING),
            position_rmse_mm=float(1000 * tracking["metrics_position_rmse"]),
            orientation_rmse_deg=float(
                np.rad2deg(tracking["metrics_orientation_rmse"])
            ),
        )
    }
    cases = {}
    for method in ("collocated", "synergistic"):
        source = GAIN / f"data/{method}/optimization_results.npz"
        d = load_results(source.parent, expected_method=method)
        assert d["q_ts_init"].shape[2] == 6 and d["u_ts_init"].shape[2] == 3
        scores = recompute_archive_losses(d)
        for phase in ("init", "best"):
            np.testing.assert_allclose(
                scores[f"recomputed_{phase}"],
                scores[f"stored_{phase}"],
                rtol=1e-7,
                atol=1e-12,
            )
        initial, best = initial_median_batch(d), int(d["best_batch"])
        d["initial_batch"] = initial
        for tag, batch in [("init", initial), ("best", best)]:
            d[f"position_error_{tag}_mm"] = 1000 * np.linalg.norm(
                d[f"x_ts_{tag}"][batch, :, 3:] - d["x_des_ts"][:, 3:], axis=1
            )
        initial_median = float(np.median(d["history_loss"][0]))
        optimum = float(scores["stored_best"][best])
        band = 0.05 * d["position_error_best_mm"][0]
        outside = np.flatnonzero(d["position_error_best_mm"] > band)
        settling = (
            None
            if len(outside) and outside[-1] == len(d["t_ts"]) - 1
            else float(d["t_ts"][outside[-1] + 1] if len(outside) else d["t_ts"][0])
        )
        audit[method] = dict(
            segments=1,
            length_m=0.1,
            tendons=3,
            source_sha256=sha256(source),
            initial_batch_zero_based=initial,
            optimized_batch_zero_based=best,
            optimized_iteration_zero_based=int(d["best_iteration"][best]),
            initial_loss_median=initial_median,
            selected_initial_loss=float(scores["stored_init"][initial]),
            best_loss=optimum,
            improvement_from_batch_median_percent=100 * (1 - optimum / initial_median),
            improvement_from_selected_initial_percent=100
            * (1 - optimum / scores["stored_init"][initial]),
            final_initial_position_error_mm=float(d["position_error_init_mm"][-1]),
            final_optimized_position_error_mm=float(d["position_error_best_mm"][-1]),
            five_percent_initial_distance_band_mm=float(band),
            optimized_settling_s=settling,
            initial_settles=bool(d["position_error_init_mm"][-1] <= band),
        )
        cases[method] = d
    return tracking, cases, audit
