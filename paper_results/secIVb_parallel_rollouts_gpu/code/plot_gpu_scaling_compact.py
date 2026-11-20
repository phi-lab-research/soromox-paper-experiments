"""Compact main-paper GPU scaling plot; preserve original and four-panel versions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from plot_gpu_scaling_summary import (
    CASE,
    PAPER,
    PAPER_COLORS,
    PAPER_STYLE_PATH,
    SYSTEMS,
    LABELS,
    SIZES,
    MARKERS,
    STYLES,
    audit_measurements,
    sha256,
)

NAME = "gpu_simulation_scaling_compact"
MODEL_COLORS = tuple(
    PAPER_COLORS[k] for k in ("pre_opt_1", "x_t", "post_opt_1", "post_opt_3")
)


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
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.7,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.4,
            "savefig.bbox": None,
        }
    )
    width, height = 87.0, 67.0
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    ax = fig.add_axes([12 / width, 8 / height, 73 / width, 57 / height])
    for model, color in zip(SYSTEMS, MODEL_COLORS, strict=True):
        envelope = (
            data[data.system == model]
            .groupby("batch_size")
            .total_speed_ratio.agg(["min", "max"])
        )
        ax.fill_between(
            envelope.index,
            envelope["min"],
            envelope["max"],
            color=color,
            alpha=0.035,
            linewidth=0,
            zorder=0,
        )
        for size, marker, style in zip(SIZES, MARKERS, STYLES, strict=True):
            rows = data[(data.system == model) & (data.size_value == size)].sort_values(
                "batch_size"
            )
            # Curves stay fully opaque above the faint model-size envelope.
            ax.plot(
                rows.batch_size,
                rows.total_speed_ratio,
                color=color,
                ls=style,
                marker=marker,
                markevery=2,
                ms=2.4,
                mfc="white",
                mew=0.65,
                lw=0.85 if size in (1, 32) else 0.65,
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(0.8, 1250)
    ax.set_ylim(0.0017, 500)
    ax.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1, color="#777777", ls=(0, (1, 2)), lw=0.7, zorder=0)
    ax.set_xlabel(r"Parallel environments $n_\mathrm{b}$", labelpad=2)
    ax.set_ylabel(r"Aggregate throughput $\Gamma_\mathrm{sim}$", labelpad=2)
    ax.tick_params(length=2.5, pad=2)
    ax.tick_params(which="minor", length=1.3)
    model_handles = [
        Line2D([], [], color=color, lw=1.3, label=label)
        for color, label in zip(MODEL_COLORS, LABELS, strict=True)
    ]
    model_legend = ax.legend(
        handles=model_handles,
        loc="upper left",
        frameon=False,
        handlelength=1.8,
        handletextpad=0.55,
        borderaxespad=0.45,
        labelspacing=0.3,
    )
    ax.add_artist(model_legend)
    size_handles = [
        Line2D(
            [],
            [],
            color="#333333",
            ls=style,
            marker=marker,
            lw=0.75,
            ms=2.4,
            mfc="white",
            mew=0.65,
            label=f"$N={size}$",
        )
        for size, marker, style in zip(SIZES, MARKERS, STYLES, strict=True)
    ]
    ax.legend(
        handles=size_handles,
        loc="lower right",
        ncol=2,
        frameon=False,
        columnspacing=0.8,
        handlelength=2.1,
        handletextpad=0.5,
        borderaxespad=0.5,
        labelspacing=0.3,
    )
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = output / f"{NAME}.{ext}"
        fig.savefig(path, dpi=450)
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)
    audit.update(
        envelope_opacity=0.035,
        envelope_meaning="Minimum and maximum throughput across tested link/segment counts at each batch size; not uncertainty",
        dimensions_mm=[width, height],
        composer_sha256=sha256(Path(__file__)),
        audit_source_sha256=sha256(
            Path(__file__).with_name("plot_gpu_scaling_summary.py")
        ),
        markers="Every second measured batch is marked to reduce clutter; all 264 measurements define the lines.",
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
