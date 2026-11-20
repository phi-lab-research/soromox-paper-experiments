"""Appendix evidence for configuration-space control and actuation-space tuning."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from control_tuning_data import CASE, PAPER, GAIN, sha256
from secvd_results import load_results
from secvd_objective import recompute_archive_losses
from render_control_gain_optimization_animations import initial_median_batch

sys.path.insert(0, str(CASE / "configuration_space_comparison/code"))
sys.path.insert(0, str(PAPER))
from configuration_space_comparison_simulation import load_comparison_npz
from plot_configuration_space_comparison import (
    CANONICAL_INPUT,
    CONTROLLER_LINESTYLES,
    physical_strain_values,
)
from paper_style import PAPER_COLORS as COLORS, PAPER_STYLE_PATH


def load_appendix_data():
    run = load_comparison_npz(CANONICAL_INPUT)
    group = next(g for g in run.plot_groups if g.key == "combined")
    assert len(group.controller_names) == 5
    first = run.results[group.scenario_key][group.controller_names[0]]
    for name in group.controller_names:
        current = run.results[group.scenario_key][name]
        np.testing.assert_array_equal(current["t"], first["t"])
        np.testing.assert_allclose(
            current["metrics"]["q_des"], first["metrics"]["q_des"]
        )
        assert current["q"].shape[1] == 6
    source = GAIN / "data/collocated/optimization_results.npz"
    tuning = load_results(source.parent, expected_method="collocated")
    if (
        tuning["q_ts_init"].shape[0] != 6
        or tuning["q_ts_init"].shape[2] != 6
        or tuning["u_ts_init"].shape[2] != 3
    ):
        raise ValueError("Expected six starts of a one-segment, three-tendon model")
    scores = recompute_archive_losses(tuning)
    for phase in ("init", "best"):
        np.testing.assert_allclose(
            scores[f"stored_{phase}"],
            scores[f"recomputed_{phase}"],
            rtol=1e-7,
            atol=1e-12,
        )
    initial = initial_median_batch(tuning)
    best = int(tuning["best_batch"])
    audit = dict(
        configuration_source_sha256=sha256(CANONICAL_INPUT),
        tuning_source_sha256=sha256(source),
        configuration_segments=1,
        configuration_actuation="fully actuated",
        configuration_controller_order=list(group.controller_names),
        configuration_interval_s=[float(first["t"][0]), float(first["t"][-1])],
        tuning_segments=1,
        tuning_tendons=3,
        tuning_case="collocated actuation-space control",
        initial_batch_zero_based=initial,
        optimized_batch_zero_based=best,
        optimized_iteration_zero_based=int(tuning["best_iteration"][best]),
        initial_selection="recorded run nearest the median initial loss",
        initial_band="pointwise per-coordinate min-max across all six initial runs",
        loss_band="pointwise min-max across finite starts at each optimization iteration",
        loss_curve="minimum across finite starts at each iteration",
        improvement_from_initial_batch_median_percent=float(
            100 * (1 - scores["stored_best"][best] / np.median(scores["stored_init"]))
        ),
        tuning_interval_s=[float(tuning["t_ts"][0]), float(tuning["t_ts"][-1])],
        displayed_response_interval_s=[0.0, 2.0],
        displayed_loss_multiplier=1000,
    )
    return run, group, tuning, audit


def build_figure(run, group, tuning, audit):
    plt.style.use(PAPER_STYLE_PATH)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": None,
            "svg.hashsalt": "soromox-control-appendix",
        }
    )
    width, height = 180, 132
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))

    def axes(x, y, w, h):
        ax = fig.add_axes([x / width, y / height, w / width, h / height])
        ax.grid(color=".90", linewidth=0.5)
        ax.set_axisbelow(True)
        return ax

    first = run.results[group.scenario_key][group.controller_names[0]]
    config_axes = []
    labels = [r"$\kappa_y$ [rad/m]", r"$\kappa_z$ [rad/m]", r"$\sigma_x$ [-]"]
    for row, (index, strain, ylabel) in enumerate(
        zip(run.strain_indices, run.strain_names, labels, strict=True)
    ):
        ax = axes(15, 102 - row * 21, 161, 17)
        config_axes.append(ax)
        ax.plot(
            first["t"],
            physical_strain_values(first["metrics"]["q_des"][:, index], strain),
            color="black",
            ls="--",
            lw=1.35,
            label="Desired",
            zorder=5,
        )
        for name in group.controller_names:
            d = run.results[group.scenario_key][name]
            ax.plot(
                d["t"],
                physical_strain_values(d["q"][:, index], strain),
                color=group.colors[name],
                ls=CONTROLLER_LINESTYLES[name],
                lw=1.05,
                label=name,
            )
        ax.axvline(group.phase_boundary_time, color=".4", ls=":", lw=0.8)
        ax.set(xlim=(0, 30), ylabel=ylabel, xticks=np.arange(0, 31, 5))
        ax.tick_params(axis="y", pad=2)
        if row < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time [s]")
    handles, labels = config_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.993),
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.1,
        labelspacing=0.3,
    )

    loss_ax = axes(15, 12, 53, 36)
    losses = 1000 * np.where(
        tuning["history_finite_mask"], tuning["history_loss"], np.nan
    )
    if not np.isfinite(losses).any(axis=1).all() or np.nanmin(losses) <= 0:
        raise ValueError(
            "Loss plot requires positive finite evidence at every iteration"
        )
    low, high = np.nanmin(losses, axis=1), np.nanmax(losses, axis=1)
    iterations = np.arange(1, len(losses) + 1)
    loss_ax.fill_between(
        iterations,
        low,
        high,
        color=COLORS["pre_opt_1"],
        alpha=0.16,
        lw=0,
        label="Across starts",
    )
    loss_ax.plot(iterations, low, color=COLORS["post_opt_1"], lw=1.2, label="Minimum")
    loss_ax.set(
        yscale="linear",
        xlim=(1, len(losses)),
        xlabel="Iteration",
        ylabel=r"Loss [$10^{-3}$]",
        xticks=[1, 25, 50, 75, 100],
    )
    loss_ax.set_ylim(low.min() * 0.93, high.max() * 1.12)
    loss_ax.legend(
        handles=[
            Line2D([], [], color=COLORS["post_opt_1"], lw=1.2, label="Minimum"),
            Patch(facecolor=COLORS["pre_opt_1"], alpha=0.16, label="Across starts"),
        ],
        loc="upper right",
        frameon=False,
        handlelength=1.5,
        labelspacing=0.25,
    )

    response_ax = axes(81.5, 12, 94.5, 36)
    selected, best = (
        audit["initial_batch_zero_based"],
        audit["optimized_batch_zero_based"],
    )
    colors = [COLORS[c] for c in ("x_t", "y_t", "z_t")]
    for k, color in enumerate(colors):
        initial = tuning["q_ts_init"][:, :, k]
        response_ax.fill_between(
            tuning["t_ts"],
            initial.min(axis=0),
            initial.max(axis=0),
            color=color,
            alpha=0.15,
            lw=0,
        )
        response_ax.plot(
            tuning["t_ts"], initial[selected], color=color, ls="-.", lw=1.0
        )
        response_ax.plot(
            tuning["t_ts"], tuning["q_ts_best"][best, :, k], color=color, lw=1.1
        )
        response_ax.plot(
            tuning["t_ts"], tuning["q_des_ts"][:, k], color=color, ls="--", lw=0.9
        )
    response_ax.set(
        xlim=(0, 2),
        xticks=[0, 0.5, 1, 1.5, 2],
        xlabel="Time [s]",
        ylabel="Angular strain [rad/m]",
    )
    response_ax.legend(
        handles=[
            Line2D([], [], color=".25", ls="-.", lw=1, label="Initial (selected)"),
            Line2D([], [], color=".25", lw=1.1, label="Optimized (best)"),
            Line2D([], [], color=".25", ls="--", lw=0.9, label="Target"),
        ],
        loc="center right",
        frameon=False,
        handlelength=2,
        labelspacing=0.25,
    )
    fig.legend(
        handles=[
            Line2D(
                [], [], color=c, marker="o", ls="none", ms=4, label=rf"$\kappa_{axis}$"
            )
            for axis, c in zip("xyz", colors, strict=True)
        ],
        loc="lower center",
        bbox_to_anchor=(128.75 / width, 49 / height),
        ncol=3,
        frameon=False,
        columnspacing=1,
        handletextpad=0.2,
        borderaxespad=0,
    )
    for letter, x, y in [("A", 2, 129), ("B", 2, 54), ("C", 69, 54)]:
        fig.text(
            x / width, y / height, letter, fontsize=11, fontweight="bold", va="top"
        )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PAPER / "final_outputs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    stem = args.output_dir / "control_and_gain_optimization_appendix"
    outputs = [stem.with_suffix(ext) for ext in (".pdf", ".png", ".svg", ".json")]
    if any(p.exists() for p in outputs) and not args.force:
        raise FileExistsError("Appendix figure exists; use --force")
    data = load_appendix_data()
    fig = build_figure(*data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in outputs[:3]:
        fig.savefig(path, dpi=300)
        if path.suffix == ".svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)
    outputs[3].write_text(
        json.dumps(
            dict(
                audit=data[3],
                plotter_sha256=sha256(Path(__file__)),
                dimensions_mm=[180, 132],
                axis_label_pt=8.5,
                tick_pt=8,
            ),
            indent=2,
        )
        + "\n"
    )
    print(outputs[0])


if __name__ == "__main__":
    main()
