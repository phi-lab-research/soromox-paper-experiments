"""Load and validate the archived GPU throughput measurements."""

import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CASE = Path(__file__).resolve().parents[1]
PAPER = CASE.parent
sys.path.insert(0, str(PAPER))
from paper_style import (
    PAPER_COLORS as PAPER_COLORS,
    PAPER_STYLE_PATH as PAPER_STYLE_PATH,
)

SYSTEMS = ("articulated_soft_robot", "planar_pcs", "pcs", "gvs")
LABELS = ("Articulated", "Planar PCS", "Spatial PCS", "GVS")
SIZES = (1, 2, 4, 8, 16, 32)
BATCHES = tuple(2**i for i in range(11))
MARKERS = ("o", "s", "^", "D", "p", "x")
STYLES = ("-", "--", "-.", ":", (0, (5, 2)), (0, (1, 1)))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_measurements(csv_path):
    d = pd.read_csv(csv_path)
    expected = {
        (model, size, batch) for model in SYSTEMS for size in SIZES for batch in BATCHES
    }
    keys = list(
        d[["system", "size_value", "batch_size"]].itertuples(index=False, name=None)
    )
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(
            "Expected exactly one measurement for every model/size/batch combination"
        )
    for field, value in {
        "device_kind": "NVIDIA GeForce RTX 5090",
        "solver": "tsit5",
        "solver_dt": 1e-4,
        "save_dt": 1e-2,
        "duration_s": 1.0,
        "warmup_runs": 1,
        "timing_repeats": 3,
        "x64_enabled": True,
        "rollout_finite": True,
        "device": "gpu",
    }.items():
        if not (d[field] == value).all():
            raise ValueError(f"Unexpected or mixed benchmark setting: {field}")
    if not np.isfinite(d.wall_time_s).all() or not (d.wall_time_s > 0).all():
        raise ValueError("Wall-clock measurements must be finite and positive")
    for column, expected_values in {
        "total_simulated_time_s": d.batch_size * d.simulated_time_s,
        "total_speed_ratio": d.batch_size * d.simulated_time_s / d.wall_time_s,
        "per_env_speed_ratio": d.simulated_time_s / d.wall_time_s,
    }.items():
        np.testing.assert_allclose(d[column], expected_values, rtol=1e-12, atol=1e-12)
    summaries = {}
    for system in SYSTEMS:
        model = d[d.system == system]
        expected_backend = "jax" if system == "articulated_soft_robot" else "warp"
        if set(model.resolved_backend) != {expected_backend}:
            raise ValueError(f"Unexpected backend for {system}")
        sizes = {}
        for size in SIZES:
            group = model[model.size_value == size].sort_values("batch_size")
            speed = group.total_speed_ratio.to_numpy()
            sizes[str(size)] = {
                "throughput_at_1": float(speed[0]),
                "throughput_at_1024": float(speed[-1]),
                "gain_1024_vs_1": float(speed[-1] / speed[0]),
                "peak_batch": int(group.iloc[np.argmax(speed)].batch_size),
                "drop_from_peak_percent": float(100 * (1 - speed[-1] / speed.max())),
                "monotone_throughput": bool((np.diff(speed) >= 0).all()),
            }
        summaries[system] = {
            "backend": expected_backend,
            "sizes": sizes,
            "max_abs_saved_state": float(model.max_abs_state.max()),
        }
    audit = {
        "rows": len(d),
        "csv_sha256": sha256(csv_path),
        "metric": "batch_size * simulated_time_s / wall_time_s",
        "units": "aggregate simulated seconds per wall-clock second",
        "timing": "mean of three synchronized executions after one warm-up; no repeat-level samples saved",
        "hardware": str(d.device_kind.iloc[0]),
        "source_revision": d.git_revision.unique().tolist(),
        "source_dirty": d.git_dirty.unique().tolist(),
        "all_rollouts_finite": True,
        "validity_scope": "Finite states are recorded; this timing dataset alone does not establish physical fidelity or numerical convergence.",
        "systems": summaries,
    }
    return d, audit
