#!/usr/bin/env python3
"""Compose the manipulation figure from saved RTX snapshots and six push runs.

Only NumPy and Matplotlib are needed: plotting never constructs a simulator.
Render overview and detail snapshots with rendering.py first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
PAPER_RESULTS_DIR = CASE_DIR.parent
sys.path.insert(0, str(PAPER_RESULTS_DIR))

from paper_style import PAPER_COLORS, PAPER_STYLE_PATH
from paper_render_style import PAPER_STUDIO
from summarize_runs import summarize

METHODS = {
    "iCEM": ("square_icem_seed{seed}.npz", PAPER_COLORS["post_opt_1"]),
    "MPPI": ("square_mppi_k800_white_seed{seed}.npz", PAPER_COLORS["pre_opt_1"]),
}
SEED_STYLES = ("-", "--", ":")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runs(data_dir: Path) -> tuple[dict, np.ndarray]:
    """Compute annotations using the experiment's online goal-reach test."""
    methods = {}
    reference_goals = None
    for name, (pattern, color) in METHODS.items():
        paths = [data_dir / pattern.format(seed=i) for i in range(3)]
        trajectories, metrics = [], []
        for path in paths:
            with np.load(path) as data:
                xy = data["object_pose"][:, :2]
                goals = data["goal_xy"]
                if not np.isfinite(xy).all() or not np.isfinite(goals).all():
                    raise ValueError(f"Non-finite trajectory in {path}")
                trajectories.append(xy)
                if reference_goals is None:
                    changes = np.r_[True, np.any(goals[1:] != goals[:-1], axis=1)]
                    reference_goals = goals[changes]
            metrics.append(summarize(path))
        methods[name] = {
            "paths": paths,
            "trajectories": trajectories,
            "color": color,
            "success": int(sum(m[0] >= len(reference_goals) for m in metrics)),
            "count": len(paths),
            "planning_mean_s": float(np.mean([m[2] for m in metrics])),
            "planning_std_s": float(np.std([m[2] for m in metrics])),
        }
    return methods, reference_goals


def load_snapshot(root: Path, view: str, step: int, source: Path) -> tuple[Path, dict]:
    """Require traceable snapshots from the requested trajectory and camera view."""
    manifest_path = root / view / "snapshots.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_sha256"] != sha256(source):
        raise ValueError(
            f"Snapshot trajectory differs from the plotted run: {manifest_path}"
        )
    if manifest["view"] != view:
        raise ValueError(f"Unexpected camera view in {manifest_path}")
    if manifest["studio"] != json.loads(json.dumps(PAPER_STUDIO.manifest())):
        raise ValueError(f"Snapshot style is stale; regenerate {manifest_path.parent}")
    matches = [s for s in manifest["snapshots"] if s["step"] == step]
    if len(matches) != 1:
        raise ValueError(
            f"Snapshot step {step} missing or duplicated in {manifest_path}"
        )
    path = manifest_path.parent / matches[0]["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, {
        **matches[0],
        "view": view,
        "camera_eye": manifest["camera_eye"],
        "camera_target": manifest["camera_target"],
        "sha256": sha256(path),
    }


def image_panel(fig, rect, path):
    """Display the complete frame at its native aspect ratio; never auto-crop."""
    ax = fig.add_axes(rect)
    ax.imshow(plt.imread(path), aspect="equal", interpolation="antialiased")
    ax.set_axis_off()
    return ax


def heading(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", va="bottom")
    fig.text(x + 0.032, y, title, fontsize=9, va="bottom")


def draw_paths(ax, methods, goals):
    """Use method colors and seed line styles, with a two-entry external legend."""
    goals_mm = goals * 1000
    loop = np.vstack((goals_mm, goals_mm[:1]))
    ax.plot(*loop.T, color="0.65", ls="--", lw=0.8, zorder=0)
    for method in methods.values():
        for seed, xy in enumerate(method["trajectories"]):
            xy = xy * 1000
            ax.plot(*xy.T, color=method["color"], ls=SEED_STYLES[seed], lw=1.3)
            ax.scatter(
                *xy[-1],
                marker="x",
                color=method["color"],
                s=24,
                linewidths=1.1,
                zorder=5,
            )
    ax.scatter(
        *goals_mm.T,
        marker="s",
        color=PAPER_STUDIO.target,
        edgecolor="white",
        linewidth=0.6,
        s=24,
        zorder=4,
    )
    offsets = ((-10, -12), (6, -12), (6, 5), (-10, 5))
    for i, (goal, offset) in enumerate(zip(goals_mm, offsets, strict=True), start=1):
        ax.annotate(str(i), goal, xytext=offset, textcoords="offset points", fontsize=7)
    start = methods["iCEM"]["trajectories"][0][0] * 1000
    ax.scatter(
        *start,
        marker="o",
        s=27,
        facecolor="white",
        edgecolor="0.2",
        linewidth=1,
        zorder=6,
    )
    all_xy = np.vstack(
        [goals_mm] + [xy * 1000 for m in methods.values() for xy in m["trajectories"]]
    )
    lower, upper = all_xy.min(0) - 12, all_xy.max(0) + 12
    ax.set(
        xlim=(lower[0], upper[0]),
        ylim=(lower[1], upper[1]),
        xlabel="$x$ [mm]",
        ylabel="$y$ [mm]",
    )
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.grid(color="0.9", lw=0.5)
    handles = [
        Line2D([], [], color=m["color"], lw=1.5, label=name)
        for name, m in methods.items()
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1,
    )


def build_figure(methods, goals, snapshots):
    """Large system overview, two contact details, and the task evidence."""
    plt.style.use(PAPER_STYLE_PATH)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.bbox": None,
        }
    )
    fig = plt.figure(figsize=(18 / 2.54, 12.6 / 2.54), facecolor="white")
    heading(fig, 0.01, 0.962, "A", "Franka + soft tentacle")
    image_panel(fig, [0.01, 0.312, 0.60, 0.643], snapshots[0][0])
    heading(fig, 0.01, 0.284, "B", "Changing push direction")
    for rect, (path, metadata) in zip(
        ([0.01, 0.018, 0.295, 0.2634], [0.315, 0.018, 0.295, 0.2634]),
        snapshots[1:],
        strict=True,
    ):
        image_panel(fig, rect, path)
        fig.text(
            rect[0] + rect[2] / 2,
            0.007,
            f"t = {metadata['time_s']:.2f} s",
            ha="center",
            fontsize=6.5,
        )
    heading(fig, 0.66, 0.962, "C", "Object paths")
    ax = fig.add_axes([0.70, 0.502, 0.285, 0.407])
    draw_paths(ax, methods, goals)
    fig.text(0.68, 0.402, "Three seeds per method", fontsize=7, color="0.3")
    fig.text(
        0.68,
        0.389,
        "Start: circle; finish: cross\nGoals: numbered squares",
        va="top",
        fontsize=7,
        color="0.3",
    )
    fig.text(0.68, 0.292, "Completed square", fontsize=8, fontweight="bold")
    for i, (name, method) in enumerate(methods.items()):
        fig.text(
            0.68,
            0.253 - i * 0.035,
            f"{name}: {method['success']}/{method['count']} runs",
            color=method["color"],
            fontsize=8,
        )
    fig.text(0.68, 0.150, "Planning time / step", fontsize=8, fontweight="bold")
    for i, (name, method) in enumerate(methods.items()):
        fig.text(
            0.68,
            0.111 - i * 0.035,
            f"{name}: {method['planning_mean_s']:.1f} $\\pm$ {method['planning_std_s']:.1f} s",
            color=method["color"],
            fontsize=8,
        )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CASE_DIR / "data")
    parser.add_argument(
        "--snapshot-dir", type=Path, default=CASE_DIR / "outputs" / "paper_snapshots"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PAPER_RESULTS_DIR / "final_outputs"
    )
    parser.add_argument("--overview-step", type=int, default=19)
    parser.add_argument("--detail-steps", type=int, nargs=2, default=(12, 19))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    stem = args.output_dir / "sampling_based_mpc"
    outputs = [stem.with_suffix(ext) for ext in (".pdf", ".svg", ".png", ".json")]
    if not args.force and any(p.exists() for p in outputs):
        raise FileExistsError("Figure output exists; pass --force to replace it")
    methods, goals = load_runs(args.data_dir)
    source = args.data_dir / "square_icem_seed0.npz"
    snapshots = [
        load_snapshot(args.snapshot_dir, "overview", args.overview_step, source)
    ]
    snapshots += [
        load_snapshot(args.snapshot_dir, "detail", i, source) for i in args.detail_steps
    ]
    fig = build_figure(methods, goals, snapshots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in outputs[:3]:
        fig.savefig(path, dpi=300)
        if path.suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    provenance = {
        "studio": PAPER_STUDIO.manifest(),
        "snapshots": [metadata for _, metadata in snapshots],
        "runs": [
            {"file": p.name, "sha256": sha256(p)}
            for m in methods.values()
            for p in m["paths"]
        ],
        "metrics": {
            name: {
                k: v
                for k, v in m.items()
                if k in ("success", "count", "planning_mean_s", "planning_std_s")
            }
            for name, m in methods.items()
        },
        "planning_time": "Mean and population standard deviation across seed means, excluding the first MPC step in each run.",
    }
    outputs[3].write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved {stem}.{{pdf,svg,png,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
