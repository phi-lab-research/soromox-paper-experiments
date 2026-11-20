"""Compose four aligned GPU throughput panels from the archived paper measurements."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd

CASE = Path(__file__).resolve().parents[1]
PAPER = CASE.parent
sys.path.insert(0, str(PAPER))
from paper_style import PAPER_COLORS, PAPER_STYLE_PATH

SYSTEMS = ("articulated_soft_robot", "planar_pcs", "pcs", "gvs")
LABELS = ("Articulated", "Planar PCS", "Spatial PCS", "GVS")
SIZES = (1, 2, 4, 8, 16, 32)
BATCHES = tuple(2**i for i in range(11))
COLORS = tuple(
    PAPER_COLORS[key]
    for key in ("pre_opt_1", "pre_opt_2", "x_t", "y_t", "post_opt_1", "post_opt_3")
)
MARKERS = ("o", "s", "^", "D", "p", "x")
STYLES = ("-", "--", "-.", ":", (0, (5, 2)), (0, (1, 1)))
NAME = "gpu_simulation_scaling"


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


def compose(csv_path, output, force=False):
    if not force and any(
        (output / f"{NAME}.{ext}").exists() for ext in ("pdf", "png", "svg", "json")
    ):
        raise FileExistsError("Figure exists; pass --force")
    data, audit = audit_measurements(csv_path)
    plt.style.use(PAPER_STYLE_PATH)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "savefig.bbox": None,
        }
    )
    width, height = 180.0, 116.0
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    legend = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            ls=style,
            lw=1.15,
            ms=3,
            mfc="white",
            label=f"$N={size}$",
        )
        for size, color, marker, style in zip(
            SIZES, COLORS, MARKERS, STYLES, strict=True
        )
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.0),
        ncol=6,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.8,
    )
    for panel, (system, title) in enumerate(zip(SYSTEMS, LABELS, strict=True)):
        row, col = divmod(panel, 2)
        x, y, w, h = 18 + 81 * col, 62 - 50 * row, 77, 40
        ax = fig.add_axes([x / width, y / height, w / width, h / height])
        for size, color, marker, style in zip(
            SIZES, COLORS, MARKERS, STYLES, strict=True
        ):
            group = data[
                (data.system == system) & (data.size_value == size)
            ].sort_values("batch_size")
            ax.plot(
                group.batch_size,
                group.total_speed_ratio,
                color=color,
                ls=style,
                marker=marker,
                lw=1.15,
                ms=3,
                mfc="white",
                mew=0.8,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(0.8, 1250)
        ax.set_ylim(1e-3, 500)
        ax.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.axhline(1, color="#666666", ls=(0, (1.5, 2)), lw=0.85, zorder=0)
        ax.tick_params(labelleft=col == 0, labelbottom=row == 1, length=3)
        ax.tick_params(which="minor", length=1.5)
        fig.text(
            x / width, (y + h + 1.5) / height, chr(65 + panel), fontsize=11, va="bottom"
        )
        fig.text(
            (x + w / 2) / width,
            (y + h + 1.5) / height,
            title,
            fontsize=9,
            va="bottom",
            ha="center",
        )
    fig.text(
        0.54,
        1 / height,
        r"Parallel environments $n_\mathrm{b}$",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.text(
        1.5 / width,
        57 / height,
        r"Aggregate simulation throughput $\Gamma_\mathrm{sim}$",
        rotation=90,
        ha="left",
        va="center",
        fontsize=8.5,
    )
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = output / f"{NAME}.{ext}"
        fig.savefig(path, dpi=300)
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)
    audit.update(
        dimensions_mm=[width, height],
        composer_sha256=sha256(Path(__file__)),
        pdf_sha256=sha256(output / f"{NAME}.pdf"),
    )
    (output / f"{NAME}.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(output / f"{NAME}.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv", type=Path, default=CASE / "data/parallel_rollout_benchmark.csv"
    )
    p.add_argument("--output-dir", type=Path, default=PAPER / "final_outputs")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    compose(args.csv, args.output_dir, args.force)
