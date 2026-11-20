"""Evaluate saved gain-optimization rollouts without running a simulator.

All times use the recorded sample grid. A missing settling time is represented
by JSON null; reductions with undefined or zero baselines are also null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]


def reduction_percent(initial, optimized):
    """Return a reduction only when both metrics and a positive baseline exist."""
    if initial is None or optimized is None or initial <= 0:
        return None
    return float(100 * (1 - optimized / initial))


def response_metrics(t, response, target, band_fraction=0.05):
    """Measure a constant-setpoint response, in the supplied coordinate units.

    Settling requires the Euclidean error to remain within band_fraction times
    its initial magnitude through the last saved sample. Overshoot is the
    largest excursion beyond a commanded coordinate's target, in the direction
    of its initial-to-target step. Zero-step coordinates have zero overshoot.
    First entry is reported separately because it need not imply settling.
    """
    t, response, target = (np.asarray(v, dtype=float) for v in (t, response, target))
    if (
        t.ndim != 1
        or len(t) < 2
        or np.any(np.diff(t) <= 0)
        or response.ndim != 2
        or response.shape[0] != len(t)
        or target.shape != response.shape
        or not all(np.isfinite(v).all() for v in (t, response, target))
    ):
        raise ValueError(
            "Expected finite (time, coordinate) arrays and increasing times"
        )
    if not np.allclose(target, target[0], rtol=0, atol=1e-12):
        raise ValueError("Step-response metrics require a constant target")
    if not 0 < band_fraction < 1:
        raise ValueError("band_fraction must lie between zero and one")
    error = np.linalg.norm(response - target, axis=-1)
    threshold = band_fraction * error[0]
    inside = error <= threshold
    entered = np.flatnonzero(inside)
    outside = np.flatnonzero(~inside)
    settling = (
        float(t[0])
        if not outside.size
        else None
        if outside[-1] == len(t) - 1
        else float(t[outside[-1] + 1])
    )
    direction = np.sign(target[0] - response[0])
    excursions = np.maximum(0, (response - target) * direction)
    coordinate_outside = np.flatnonzero(
        (
            np.abs(response - target) > band_fraction * np.abs(target[0] - response[0])
        ).any(axis=1)
    )
    coordinate_settling = (
        float(t[0])
        if not coordinate_outside.size
        else None
        if coordinate_outside[-1] == len(t) - 1
        else float(t[coordinate_outside[-1] + 1])
    )
    return {
        "integrated_error": float(np.trapezoid(error, t)),
        "initial_error": float(error[0]),
        "final_error": float(error[-1]),
        "settling_band": float(threshold),
        "first_entry_s": float(t[entered[0]]) if entered.size else None,
        "settling_s": settling,
        "coordinate_settling_s": coordinate_settling,
        "coordinate_overshoot": excursions.max(axis=0).tolist(),
        "maximum_coordinate_overshoot": float(excursions.max()),
        "peak_overshoot_vector_norm": float(np.linalg.norm(excursions, axis=-1).max()),
    }


def median_metric(values):
    """Median including nonsettled runs as censored above the observation window.

    In particular, never discard nonsettled runs to manufacture a finite median.
    """
    value = np.median([np.inf if v is None else v for v in values])
    return float(value) if np.isfinite(value) else None


def archive_metrics(path):
    """Report all initial runs and the best optimized run from one valid archive."""
    from secvd_results import load_results

    path = Path(path)
    data = load_results(path)
    if bool(data["is_placeholder"]):
        raise ValueError("Performance claims require a non-placeholder archive")
    method = str(data["method"].item())
    prefix, channels, scale, units = (
        ("q", slice(0, 3), 1.0, "rad/m")
        if method == "collocated"
        else ("x", slice(3, 6), 1000.0, "mm")
    )
    target = scale * data[f"{prefix}_des_ts"][:, channels]
    initial = scale * data[f"{prefix}_ts_init"][:, :, channels]
    best = int(data["best_batch"])
    optimized = scale * data[f"{prefix}_ts_best"][best, :, channels]
    history = data["history_loss"]
    valid = data["history_finite_mask"]
    if not valid.all():
        raise ValueError(
            "Batch performance comparisons require all recorded losses to be valid"
        )
    selected = int(np.argmin(abs(history[0] - np.median(history[0]))))
    t = data["t_ts"]
    runs = [response_metrics(t, y, target) for y in initial]
    final = response_metrics(t, optimized, target)
    keys = (
        "integrated_error",
        "first_entry_s",
        "settling_s",
        "maximum_coordinate_overshoot",
    )
    median = {key: median_metric([run[key] for run in runs]) for key in keys}
    median_curve = response_metrics(t, np.median(initial, axis=0), target)
    return {
        "method": method,
        "source": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "interval_s": [float(t[0]), float(t[-1])],
        "coordinate_units": units,
        "integrated_error_units": units + " s",
        "definitions": {
            "integrated_error": "Trapezoidal time integral of Euclidean error norm.",
            "settling_s": "First saved time after which Euclidean error stays within 5% of its initial magnitude through the last sample.",
            "first_entry_s": "First entry into that band, without requiring subsequent residence.",
            "maximum_coordinate_overshoot": "Largest positive coordinate excursion beyond the target in the initial step direction; zero-step coordinates contribute zero.",
            "initial_metric_median": "Median of metrics evaluated separately for all initial runs; nonsettled times are censored, never omitted.",
            "legacy_pointwise_median_response": "Metrics of the coordinatewise median response used in the old figure; this is generally not a simulated trajectory.",
        },
        "selected_initial_start_one_based": selected + 1,
        "optimized_start_one_based": best + 1,
        "initial_runs": runs,
        "selected_initial": runs[selected],
        "initial_metric_median": median,
        "legacy_pointwise_median_response": median_curve,
        "optimized": final,
        "reductions_from_initial_metric_median_percent": {
            key: reduction_percent(median[key], final[key]) for key in keys
        },
        "reductions_from_selected_initial_percent": {
            key: reduction_percent(runs[selected][key], final[key]) for key in keys
        },
        "reductions_from_legacy_median_response_percent": {
            key: reduction_percent(median_curve[key], final[key]) for key in keys
        },
        "loss": {
            "initial_median": float(np.median(history[0])),
            "optimized_best": float(history.min()),
            "reduction_percent": reduction_percent(
                np.median(history[0]), history.min()
            ),
            "initial_variance": float(np.var(history[0])),
            "final_variance": float(np.var(history[-1])),
            "variance_reduction_percent": reduction_percent(
                np.var(history[0]), np.var(history[-1])
            ),
        },
    }


def write_report(output, report):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def report(data_dir=CASE_DIR / "data"):
    return {
        method: archive_metrics(Path(data_dir) / method / "optimization_results.npz")
        for method in ("collocated", "synergistic")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CASE_DIR / "data")
    parser.add_argument(
        "--output", type=Path, default=CASE_DIR / "outputs/control_gain_metrics.json"
    )
    args = parser.parse_args()
    results = report(args.data_dir)
    write_report(args.output, results)
    print(f"Saved {args.output}")
    for method, result in results.items():
        print(
            method,
            json.dumps(
                {
                    "initial_medians": result["initial_metric_median"],
                    "optimized": result["optimized"],
                    "reductions_percent": result[
                        "reductions_from_initial_metric_median_percent"
                    ],
                }
            ),
        )


if __name__ == "__main__":
    main()
