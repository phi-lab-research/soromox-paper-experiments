"""Recheck legacy gain-performance claims with explicitly defined saved-data metrics."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "secVd_control_gain_optimization/data"


def report():
    results = {}
    for method, selected in (("collocated", 5), ("synergistic", 4)):
        source = DATA / method / "optimization_results.npz"
        with np.load(source) as d:
            t = d["t_ts"]
            best = int(d["best_batch"])
            prefix = "q" if method == "collocated" else "x"
            indices = slice(0, 3) if method == "collocated" else slice(3, 6)
            reference = d[f"{prefix}_des_ts"][:, indices]
            initial = d[f"{prefix}_ts_init"][:, :, indices]
            optimized = d[f"{prefix}_ts_best"][best, :, indices]
            # Angular-strain vector or Cartesian-position vector, respectively.
            scale = 1.0 if method == "collocated" else 1000.0
            initial_integrals = scale * np.trapezoid(
                np.linalg.norm(initial - reference, axis=-1), t, axis=-1
            )
            optimized_error = np.linalg.norm(optimized - reference, axis=-1)
            optimized_integral = scale * np.trapezoid(optimized_error, t)
            median_curve_integral = scale * np.trapezoid(
                np.linalg.norm(np.median(initial, axis=0) - reference, axis=-1), t
            )
            outside = np.flatnonzero(optimized_error > 0.05 * optimized_error[0])
            settled = None if outside[-1] == len(t) - 1 else float(t[outside[-1] + 1])
            coordinate_band = 0.05 * np.abs(reference[0] - optimized[0])
            coordinate_outside = np.flatnonzero(
                (np.abs(optimized - reference) > coordinate_band).any(axis=1)
            )
            coordinate_settled = (
                None if coordinate_outside[-1] == len(t) - 1
                else float(t[coordinate_outside[-1] + 1])
            )
            direction = np.sign(reference[0] - optimized[0])
            overshoot = np.maximum(0, (optimized - reference) * direction).max(0)
            history = d["history_loss"]
            results[method] = {
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "interval_s": [float(t[0]), float(t[-1])],
                "integral_units": "rad s / m" if method == "collocated" else "mm s",
                "integral_definition": "trapezoidal integral of Euclidean error norm",
                "initial_integrals": initial_integrals.tolist(),
                "initial_integral_median": float(np.median(initial_integrals)),
                "pointwise_median_curve_integral": float(median_curve_integral),
                "selected_initial_start_one_based": selected + 1,
                "optimized_start_one_based": best + 1,
                "optimized_integral": float(optimized_integral),
                "integral_reduction_from_initial_median_percent": float(
                    100 * (1 - optimized_integral / np.median(initial_integrals))
                ),
                "integral_reduction_from_selected_initial_percent": float(
                    100 * (1 - optimized_integral / initial_integrals[selected])
                ),
                "integral_reduction_from_pointwise_median_curve_percent": float(
                    100 * (1 - optimized_integral / median_curve_integral)
                ),
                "loss_variance_reduction_initial_to_final_percent": float(
                    100 * (1 - np.var(history[-1]) / np.var(history[0]))
                ),
                "optimized_sustained_settling_5percent_initial_norm_s": settled,
                "optimized_sustained_settling_5percent_each_coordinate_s": coordinate_settled,
                "optimized_coordinate_overshoot": (scale * overshoot).tolist(),
                "overshoot_units": "rad / m" if method == "collocated" else "mm",
            }
    return results


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
