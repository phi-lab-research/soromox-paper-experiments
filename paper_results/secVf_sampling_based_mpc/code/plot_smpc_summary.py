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
    "MPPI": ("square_mppi_k800_white_seed{seed}.npz", PAPER_COLORS["pre_opt_1"]),
    "iCEM": ("square_icem_seed{seed}.npz", PAPER_COLORS["post_opt_1"]),
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
    future = manifest.get("future_overlay")
    if future:
        from figure_states import validate_future_steps
        if future.get("kind") != "later_recorded_states_same_goal":
            raise ValueError("Future overlays must preserve the highlighted goal")
        with np.load(source) as data:
            validate_future_steps(data["object_pose"], data["goal_xy"], step,
                                  [entry["step"] for entry in future["states"]])
            if not np.allclose(future["goal_xy"], data["goal_xy"][step]):
                raise ValueError("Future overlay goal differs from the recorded active goal")
    return path, {
        **matches[0],
        "view": view,
        "camera_eye": manifest["camera_eye"],
        "camera_target": manifest["camera_target"],
        "future_overlay": future,
        "sha256": sha256(path),
    }


def image_panel(fig, rect, path, crop=None):
    """Preserve aspect ratio; use only explicitly specified, shared crop bounds."""
    ax = fig.add_axes(rect)
    pixels = plt.imread(path)
    if crop is not None:
        x0, x1, y0, y1 = crop
        height, width = pixels.shape[:2]
        pixels = pixels[round(y0 * height):round(y1 * height), round(x0 * width):round(x1 * width)]
    ax.imshow(pixels, aspect="equal", interpolation="antialiased")
    ax.set_axis_off()
    return ax


def panel_letter(fig, x, y, letter):
    fig.text(x, y, letter, fontsize=11, fontweight="bold", va="bottom")


def load_sampling(root: Path, iteration: int, source: Path) -> tuple[Path, dict]:
    path = root / "sampling" / f"iteration_{iteration}.png"
    manifest = json.loads(path.with_suffix(".json").read_text())
    if manifest["source_sha256"] != sha256(source):
        raise ValueError("Sampling trajectory differs from the plotted run")
    if manifest["studio"] != json.loads(json.dumps(PAPER_STUDIO.manifest())):
        raise ValueError("Sampling style is stale; regenerate the cloud frames")
    if (manifest["iteration"], manifest["timestep"], manifest["horizon_step"]) != (iteration, 0, 10):
        raise ValueError("Unexpected sampling iteration, planning step, or horizon step")
    if manifest.get("solid_configuration") != "current_recorded_state":
        raise ValueError("Solid sampling geometry must show the current state; regenerate the cloud frames")
    return path, {**manifest, "sha256": sha256(path)}


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


SAMPLING_CROP = (0.05, 0.95, 0.04, 0.88)
OVERVIEW_CROP = (0.25, 0.84, 0.10, 0.90)


def load_tradeoff(path, data_dir):
    tradeoff = json.loads(path.read_text())
    for source in tradeoff["sources"]:
        if sha256(data_dir / source["file"]) != source["sha256"]:
            raise ValueError("Planner sweep data changed; regenerate plot_smpc_diagnostics.py")
    return tradeoff


def draw_tradeoff(ax, tradeoff):
    """Identify sample budgets and iteration counts beside the sweep points."""
    markers = {"mppi": "^", "icem_fixed_iters": "x", "icem_fixed_batch": "o"}
    # Keep labels close to their own markers; split the two nearly coincident
    # MPPI budgets above/below and identify them with short leader lines.
    label_offsets = {
        ("mppi", "K=200"): (-4, 5, "right"),
        ("mppi", "K=400"): (4, -6, "left"),
        ("mppi", "K=800"): (0, 6, "center"),
        ("mppi", "K=1600"): (-4, 5, "right"),
        ("icem_fixed_iters", "K=50"): (0, 6, "center"),
        ("icem_fixed_iters", "K=100"): (3, -6, "left"),
        ("icem_fixed_iters", "K=200"): (4, 0, "left"),
        ("icem_fixed_iters", "K=400"): (0, -6, "center"),
        ("icem_fixed_batch", "it=1"): (4, 0, "left"),
        ("icem_fixed_batch", "it=2"): (-4, -3, "right"),
        ("icem_fixed_batch", "it=4"): (4, 0, "left"),
    }
    for point in tradeoff["points"]:
        family = point["family"]
        color = METHODS["MPPI" if family == "mppi" else "iCEM"][1]
        style = {"facecolors": "none", "edgecolors": color} if family == "icem_fixed_batch" else {"color": color}
        ax.scatter(point["mean_time_s"], point["mean_negative_cost"],
                   marker=markers[family], s=24, linewidths=0.8,
                   **style, zorder=3)
        label = point.get("label", "")
        # K=200, I=3 is the common point in the two iCEM sweeps.
        if label and (family, label) in label_offsets:
            dx, dy, ha = label_offsets[(family, label)]
            text = label.replace("it=", "I=")
            if family == "icem_fixed_iters" and label == "K=200":
                text += "\nI=3"
            leader = None
            if family == "mppi" and label in ("K=200", "K=400"):
                leader = dict(arrowstyle="-", color=color, lw=0.45,
                              shrinkA=0.5, shrinkB=2.5)
            ax.annotate(text, (point["mean_time_s"], point["mean_negative_cost"]),
                        xytext=(dx, dy), textcoords="offset points", ha=ha,
                        va="center", fontsize=6, color=color, zorder=4,
                        linespacing=0.9, arrowprops=leader)
    ax.set(xlim=(0, 49), ylim=(-1670, -780),
           xlabel="Mean planning time / step [s]", ylabel="Mean $-$push cost / step")
    ax.set_xticks((0, 10, 20, 30, 40))
    ax.set_yticks((-1600, -1400, -1200, -1000))
    ax.set_axisbelow(True)
    ax.grid(color="0.9", lw=0.5)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    handles = [
        Line2D([], [], marker="^", color=METHODS["MPPI"][1], ls="none", markersize=4,
               label="MPPI"),
        Line2D([], [], marker="x", color=METHODS["iCEM"][1], ls="none", markersize=4,
               label="iCEM (3 iterations)"),
        Line2D([], [], marker="o", color=METHODS["iCEM"][1], markerfacecolor="none",
               ls="none", markersize=4, label="iCEM ($K=200$)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False,
              handlelength=1, handletextpad=0.5, labelspacing=0.25, borderpad=0.3)


def build_figure(methods, goals, snapshots, sampling, tradeoff):
    """Whole system and two stacked plots above a full-width sampling row."""
    plt.style.use(PAPER_STYLE_PATH)
    plt.rcParams.update({
        "text.usetex": False, "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.bbox": None,
    })
    width, height = 180, 142
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor="white")
    def rect(x, y, w, h):
        return [x / width, y / height, w / width, h / height]
    def letter(x, y, label):
        panel_letter(fig, x / width, y / height, label)
    letter(2, 135.5, "A")
    image_panel(fig, rect(2, 49, 82, 83.39), snapshots[0][0], OVERVIEW_CROP)
    letter(94, 135.5, "B")
    ax = fig.add_axes(rect(115.5, 83, 49, 49))
    draw_paths(ax, methods, goals)
    ax.set(xlim=(32, 168), ylim=(-118, 18), aspect="equal")
    handles = [Line2D([], [], color=m["color"], lw=1.5, label=name)
               for name, m in methods.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, handlelength=1.3, handletextpad=0.5,
              columnspacing=1, borderaxespad=0.4)
    ax.set_xticks((50, 100, 150))
    ax.set_yticks((-100, -50, 0))
    for label, offset in zip(ax.texts, ((-7, -6), (5, -6), (5, 2), (-7, 2)), strict=True):
        label.set_position(offset)
    letter(2, 43.7, "C")
    for i, (path, _) in enumerate(sampling):
        image_panel(fig, rect(2 + i * 59, 2, 57, 39.9), path, SAMPLING_CROP)
        fig.text((4 + i * 59) / width, 39 / height, f"Iteration {i + 1}",
                 fontsize=6.5, va="top", color="0.2")
    letter(94, 76, "D")
    tradeoff_ax = fig.add_axes(rect(103, 52, 74, 21))
    draw_tradeoff(tradeoff_ax, tradeoff)
    tradeoff_ax.set_yticks((-1600, -1300, -1000))
    tradeoff_ax.set_ylabel("Mean $-$push cost\nper step", fontsize=7)
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
    parser.add_argument("--tradeoff-data", type=Path, default=CASE_DIR / "outputs" / "planner_tradeoff.json")
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
    sampling = [load_sampling(args.snapshot_dir, i, source) for i in range(3)]
    tradeoff = load_tradeoff(args.tradeoff_data, args.data_dir)
    fig = build_figure(methods, goals, snapshots, sampling, tradeoff)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in outputs[:3]:
        fig.savefig(path, dpi=300)
        if path.suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    provenance = {
        "studio": PAPER_STUDIO.manifest(),
        "snapshots": [metadata for _, metadata in snapshots],
        "overview_crop_xyxy_normalized": [OVERVIEW_CROP[0], OVERVIEW_CROP[2], OVERVIEW_CROP[1], OVERVIEW_CROP[3]],
        "sampling": [metadata for _, metadata in sampling],
        "sampling_crop_xyxy_normalized": [SAMPLING_CROP[0], SAMPLING_CROP[2], SAMPLING_CROP[1], SAMPLING_CROP[3]],
        "planner_tradeoff": tradeoff,
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
