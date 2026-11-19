#!/usr/bin/env python3
"""Appendix figure: contact details and the full planner cost/time sweep."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

CODE_DIR = Path(__file__).resolve().parent
CASE_DIR = CODE_DIR.parent
PAPER_RESULTS_DIR = CASE_DIR.parent
for path in (CODE_DIR, PAPER_RESULTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paper_style import PAPER_COLORS as COLORS
from paper_style import PAPER_STYLE_PATH as PAPER_STYLE
from four_tendon_soft_arm import FourTendonSoftArm
from smpc import SMPCParams, push_cost
from plot_smpc_summary import image_panel, load_snapshot, panel_letter

RESULTS_DIR = CASE_DIR / "data"
OUTPUT_DIR = CASE_DIR / "outputs"
OUTPUT_FILENAME = "sampling_based_mpc_diagnostics"
FINAL_OUTPUT_DIR = CASE_DIR.parent / "final_outputs"
TRADEOFF_RESULTS_DIRS = [RESULTS_DIR / "cmp_seeds" / f"seed{i}" for i in range(3)]

# Copies of run_experiment.py's values: importing it would pull in newton_cosserat just to plot.
OBJECT_FOOTPRINT = 0.05
TABLE_TOP_Z = 0.0
CONTACT_SEGMENT_INDEX = 2

@dataclasses.dataclass(frozen=True)
class RunSpec:
    filename: str
    family: str
    label: str


RUNS = [
    RunSpec("cmp_mppi_k200.npz", "mppi", "K=200"),
    RunSpec("cmp_mppi_k400.npz", "mppi", "K=400"),
    RunSpec("cmp_mppi_k800.npz", "mppi", "K=800"),
    RunSpec("cmp_mppi_k1600.npz", "mppi", "K=1600"),
    RunSpec("cmp_icem_it3_k50.npz", "icem_fixed_iters", "K=50"),
    RunSpec("cmp_icem_it3_k100.npz", "icem_fixed_iters", "K=100"),
    RunSpec("cmp_icem_k200_it3.npz", "icem_fixed_iters", "K=200"),
    RunSpec("cmp_icem_it3_k400.npz", "icem_fixed_iters", "K=400"),
    RunSpec("cmp_icem_k200_it1.npz", "icem_fixed_batch", "it=1"),
    RunSpec("cmp_icem_k200_it2.npz", "icem_fixed_batch", "it=2"),
    RunSpec("cmp_icem_k200_it3.npz", "icem_fixed_batch", "it=3"),
    RunSpec("cmp_icem_k200_it4.npz", "icem_fixed_batch", "it=4"),
]

FAMILY_STYLE = {
    "mppi": {
        "marker": "^",
        "color": COLORS["pre_opt_1"],
        "name": "MPPI",
        "offset": (-8, 0),
        "ha": "right",
        "va": "center",
    },
    "icem_fixed_iters": {
        "marker": "x",
        "color": COLORS["post_opt_1"],
        "name": "iCEM (fixed iters=3)",
        "offset": (0, 9),
        "ha": "center",
        "va": "bottom",
    },
    "icem_fixed_batch": {
        "marker": "o",
        "color": COLORS["post_opt_1"],
        "name": "iCEM (fixed K=200)",
        "offset": (0, -9),
        "ha": "center",
        "va": "top",
    },
}

LABEL_OVERRIDES = {
    ("cmp_mppi_k400.npz", "mppi"): {"offset": (8, 0), "ha": "left", "va": "center"},
    ("cmp_icem_k200_it3.npz", "icem_fixed_iters"): {
        "offset": (8, 7),
        "ha": "left",
        "va": "bottom",
    },
    ("cmp_mppi_k1600.npz", "mppi"): {"offset": (0, 9), "ha": "center", "va": "bottom"},
    ("cmp_icem_it3_k50.npz", "icem_fixed_iters"): {
        "offset": (-8, 0),
        "ha": "right",
        "va": "center",
    },
    ("cmp_icem_it3_k100.npz", "icem_fixed_iters"): {
        "offset": (0, -9),
        "ha": "center",
        "va": "top",
    },
}


def configure_matplotlib() -> None:
    """Configure a compact publication-style matplotlib theme."""
    plt.style.use(PAPER_STYLE)
    plt.rcParams.update(
        {
            "savefig.pad_inches": 0.1,
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 150,
        }
    )
    if shutil.which("latex") is not None:
        plt.rcParams.update({"text.usetex": True})


def build_params(robot) -> SMPCParams:
    """SMPCParams for re-scoring the cmp_ runs; score_run fills in each run's goal, contact_z and yaw."""
    return SMPCParams(
        robot=robot,
        control_std=jnp.ones(robot.num_actuators + 3),
        K=1,
        contact_segment_index=CONTACT_SEGMENT_INDEX,
        object_footprint=OBJECT_FOOTPRINT,
        table_z=TABLE_TOP_Z,
    )


score_push_cost = eqx.filter_jit(push_cost)


def score_run(path: Path, params: SMPCParams) -> tuple[float, float]:
    """Return (mean planning time per step [s], -push_cost of the executed trajectory).

    Planning time sums diagnostics_iter_time over iterations (rollouts included) and averages over
    steps 1.., skipping step 0's JIT warm-up. The cost uses the run's goal, starting object height
    and starting yaw, as run_experiment.py sets them.
    """
    with np.load(path, allow_pickle=True) as data:
        iter_time = data["diagnostics_iter_time"].astype(float)
        X_final = data["X_final"]
        U_best = data["U_best"]
        goal_xy = data["goal_xy"][0]
        initial_object_z = data["object_pose"][0, 2]

    yaw_index = 2 * params.robot.num_internal_dofs + 4
    params = dataclasses.replace(
        params,
        push_goal_xy=jnp.asarray(goal_xy),
        contact_z=float(initial_object_z),
        push_goal_yaw=jnp.asarray(X_final[0, yaw_index]),
    )
    cost = float(score_push_cost(params, jnp.asarray(X_final), jnp.asarray(U_best)))
    return float(iter_time[1:].sum(axis=1).mean()), -cost


def load_seed_mean_points(
    results_dirs: list[Path], params: SMPCParams
) -> list[tuple[RunSpec, float, float]]:
    """Score every RUNS configuration in each of results_dirs (one seed per directory)
    and return (spec, mean_time, mean_neg_cost) per configuration.
    """
    scores: dict[
        Path, tuple[float, float]
    ] = {}  # cmp_icem_k200_it3 appears in two families
    points = []
    for spec in RUNS:
        paths = [results_dir / spec.filename for results_dir in results_dirs]
        for path in paths:
            if path not in scores:
                scores[path] = score_run(path, params)
        times, costs = zip(*(scores[path] for path in paths), strict=True)
        points.append((spec, float(np.mean(times)), float(np.mean(costs))))
    return points


def plot_planner_tradeoff(
    points: list[tuple[RunSpec, float, float]], ax: plt.Axes
) -> None:
    """Scatter (mean planning time, mean -push_cost) per configuration, one marker style per family."""
    seen_families = []
    for spec, x, y in points:
        style = FAMILY_STYLE[spec.family]
        scatter_kwargs = {"color": style["color"]}
        if style["marker"] == "o":
            scatter_kwargs = {"facecolors": "none", "edgecolors": style["color"], "linewidth": 1.0}
        elif style["marker"] == "^":
            scatter_kwargs.update(edgecolor="black", linewidth=0.5)
        ax.scatter(
            x,
            y,
            marker=style["marker"],
            s=60,
            zorder=3,
            label=style["name"] if spec.family not in seen_families else None,
            **scatter_kwargs,
        )
        if spec.family not in seen_families:
            seen_families.append(spec.family)
        place = LABEL_OVERRIDES.get((spec.filename, spec.family), style)
        ax.annotate(
            spec.label,
            (x, y),
            textcoords="offset points",
            xytext=place["offset"],
            ha=place["ha"],
            va=place["va"],
            fontsize=6,
        )

    ax.set_xlabel(r"Mean planning wall-clock time per MPC step $\mathrm{[s]}$")
    ax.set_ylabel(r"Mean $-$push cost per MPC step")
    xs = [x for _, x, _ in points]
    ys = [y for _, _, y in points]
    x_span = (max(xs) - min(xs)) or 1.0
    y_span = (max(ys) - min(ys)) or 1.0
    ax.set_xlim(0.0, max(xs) + 0.08 * x_span)
    ax.set_ylim(min(ys) - 0.14 * y_span, max(ys) + 0.12 * y_span)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(loc="best", frameon=True)


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.pdf", dpi=300, pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    """Keep contact details and the full planner sweep in the appendix."""
    configure_matplotlib()
    robot = FourTendonSoftArm().model
    points = load_seed_mean_points(TRADEOFF_RESULTS_DIRS, build_params(robot))
    # Export scored data so the main figure remains reproducible without JAX/RTX.
    sources = sorted({directory / spec.filename for directory in TRADEOFF_RESULTS_DIRS for spec in RUNS})
    export = {
        "points": [{**dataclasses.asdict(spec), "mean_time_s": x, "mean_negative_cost": y}
                   for spec, x, y in points],
        "sources": [{"file": str(path.relative_to(RESULTS_DIR)),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources],
        "definition": "Three-seed mean of per-run planning time (excluding the first step) and negative executed-trajectory push cost.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "planner_tradeoff.json").write_text(json.dumps(export, indent=2) + "\n")
    fig = plt.figure(figsize=(18 / 2.54, 9 / 2.54))
    panel_letter(fig, 0.01, 0.950, "A")
    for step, y in ((12, 0.475), (19, 0.012)):
        path, _ = load_snapshot(CASE_DIR / "outputs" / "paper_snapshots", "detail", step,
                                RESULTS_DIR / "square_icem_seed0.npz")
        image_panel(fig, [0.01, y, 0.36, 0.450], path)
    panel_letter(fig, 0.44, 0.950, "B")
    plot_planner_tradeoff(points, fig.add_axes([0.47, 0.13, 0.51, 0.78]))
    save_figure(fig, FINAL_OUTPUT_DIR, OUTPUT_FILENAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
