"""Compose the audited identification figure and full configuration appendix."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

CASE = Path(__file__).resolve().parents[1]
PAPER = CASE.parent
sys.path.insert(0, str(PAPER))
from paper_style import PAPER_COLORS, PAPER_STYLE_PATH

STAGES = ("initial", "identified", "learned", "fitted_residual")
COLORS = (
    PAPER_COLORS["pre_opt_1"],
    PAPER_COLORS["post_opt_1"],
    PAPER_COLORS["post_opt_3"],
    "#777777",
)
LABELS = (
    "Initial parameters",
    "Identified parameters",
    "Learned residual",
    "Fitted residual (ref.)",
)
LINESTYLES = ("--", "-.", "-", ":")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure():
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
            "lines.linewidth": 1.2,
            "savefig.bbox": None,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
        }
    )


def project(points, meta):
    points = np.atleast_2d(points)
    eye = np.array(meta["camera_eye"])
    target = np.array(meta["camera_target"])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    xyz = (points - eye) @ np.stack((right, up, forward), axis=1)
    scale = meta["height"] / (
        2 * xyz[:, 2] * np.tan(np.deg2rad(meta["vertical_fov_degrees"]) / 2)
    )
    return np.column_stack(
        (meta["width"] / 2 + scale * xyz[:, 0], meta["height"] / 2 - scale * xyz[:, 1])
    )


def draw_comparison(
    ax,
    d,
    index,
    meta,
    include_reference=False,
    linewidth=1.55,
    markersize=3.5,
    marker_geometry=None,
):
    shift = np.array(meta["display_translation"])
    for stage, color, style in zip(STAGES, COLORS, LINESTYLES, strict=True):
        if stage == "fitted_residual" and not include_reference:
            continue
        xy = project(d[stage + "_poses"][index, :, :3, 3] + shift, meta)
        ax.plot(*xy.T, color=color, ls=style, lw=linewidth, zorder=3)
        if marker_geometry is not None:
            anchors = project(marker_geometry[stage + "_anchors"][index] + shift, meta)
            predictions = project(d[stage + "_markers"][index] + shift, meta)
            for anchor, prediction in zip(anchors, predictions, strict=True):
                ax.plot(
                    *np.stack((anchor, prediction)).T,
                    color=color,
                    lw=0.65,
                    ls="-",
                    alpha=0.9,
                    zorder=2,
                )
        xy = project(d[stage + "_markers"][index] + shift, meta)
        ax.plot(
            *xy.T, color=color, marker="x", ls="none", ms=markersize, mew=0.75, zorder=4
        )
    xy = project(d["measured"][index] + shift, meta)
    ax.plot(
        *xy.T,
        ls="none",
        marker="o",
        ms=markersize + 1,
        markerfacecolor="white",
        markeredgecolor=PAPER_COLORS["ground_truth"],
        markeredgewidth=1.1,
        zorder=5,
    )
    ax.set_xlim(0, meta["width"])
    ax.set_ylim(meta["height"], 0)
    ax.set_axis_off()
    ax.set_aspect("equal")


def legends(reference=False):
    count = 4 if reference else 3
    return [
        Line2D(
            [],
            [],
            color=COLORS[i],
            ls=LINESTYLES[i],
            label=LABELS[i],
            lw=1.55,
            marker="x",
            ms=3,
            mew=0.7,
        )
        for i in range(count)
    ] + [
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            mfc="white",
            mec=PAPER_COLORS["ground_truth"],
            ms=4,
            label="Measured markers",
        )
    ]


def axes_mm(fig, x, y, w, h, width, height):
    return fig.add_axes([x / width, y / height, w / width, h / height])


def save(fig, path):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(path.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def compose(evaluation, scenes, output, force=False):
    targets = [
        output / f"{name}.pdf"
        for name in [
            "model_identification_residual_learning",
            "identification_shape_comparisons",
            "identification_error_details",
        ]
    ]
    if any(p.exists() for p in targets) and not force:
        raise FileExistsError("Figure exists; pass --force")
    audit = json.loads((evaluation / "identification_audit.json").read_text())
    if (
        sha256(evaluation / "identification_evaluation.npz")
        != audit["evaluation_sha256"]
    ):
        raise ValueError("Audit and evaluation disagree")
    metadata = {
        name: json.loads((scenes / f"{name}.json").read_text())
        for name in ["overview", "comparison_1", "comparison_2", "comparison_3"]
    }
    for meta in metadata.values():
        if meta["evaluation_sha256"] != audit["evaluation_sha256"]:
            raise ValueError("Rendered scene is stale")
    marker_geometry_path = scenes / "marker_geometry.npz"
    marker_geometry = dict(np.load(marker_geometry_path))
    for meta in metadata.values():
        if sha256(marker_geometry_path) != meta["marker_geometry_sha256"]:
            raise ValueError("Marker geometry differs from rendered scene provenance")
    configure()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(evaluation / "identification_evaluation.npz") as d:
        width, height = 180.0, 90.0
        fig = plt.figure(figsize=(width / 25.4, height / 25.4))
        ax = axes_mm(fig, 1, 1.5, 64, 80, width, height)
        overview_ax = ax
        ax.imshow(plt.imread(scenes / "overview.png"))
        ax.set_axis_off()
        meta = metadata["overview"]
        ax.set_xlim(0, meta["width"])
        ax.set_ylim(1720, 280)
        overview_index = audit["selected_indices_zero_based"][2]
        shift = np.array(meta["display_translation"])
        anchors = project(
            marker_geometry["learned_anchors"][overview_index] + shift, meta
        )
        predictions = project(d["learned_markers"][overview_index] + shift, meta)
        for anchor, prediction in zip(anchors, predictions, strict=True):
            ax.plot(
                *np.stack((anchor, prediction)).T, color="#64676B", lw=0.85, zorder=3
            )
        ax.plot(
            *predictions.T,
            ls="none",
            marker="x",
            ms=4,
            mew=1.0,
            color=COLORS[2],
            zorder=4,
        )
        xy = project(
            d["measured"][audit["selected_indices_zero_based"][2]]
            + meta["display_translation"],
            meta,
        )
        for i, (x, y) in enumerate(xy):
            ax.annotate(
                f"$M_{i + 1}$",
                (x, y),
                xytext=(-6, -2) if i in (1, 3) else (6, 1),
                ha="right" if i in (1, 3) else "left",
                textcoords="offset points",
                fontsize=8,
                color="#333333",
            )
        for i, index in enumerate(audit["selected_indices_zero_based"]):
            ax = axes_mm(fig, 70 + i * 36, 34, 34, 47.5, width, height)
            name = f"comparison_{i + 1}"
            ax.imshow(plt.imread(scenes / f"{name}.png"))
            draw_comparison(
                ax, d, index, metadata[name], marker_geometry=marker_geometry
            )
            ax.set_xlim(0, 680)
            ax.set_ylim(1080, 130)
            ax.text(
                0.5,
                -0.035,
                ["Tendon 1", "Tendon 2", "Both tendons"][i],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
            )
        fig.legend(
            handles=legends(),
            loc="upper center",
            bbox_to_anchor=(123 / width, 89 / height),
            ncol=2,
            columnspacing=1.0,
            handlelength=2.0,
            labelspacing=0.3,
            frameon=False,
            borderaxespad=0.0,
        )
        ax = axes_mm(fig, 100, 0, 76, 27, width, height)
        values = [audit["rmse_mm"][stage]["all_22"] for stage in STAGES]
        ax.barh(
            np.arange(4),
            values,
            color=COLORS,
            height=0.58,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_yticks(np.arange(4), LABELS)
        ax.invert_yaxis()
        ax.set_xlim(0, 55)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_xlabel("Marker position RMSE [mm]", labelpad=2)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", visible=False)
        for letter, x, y in [("A", 2, 88), ("B", 68, 88), ("C", 68, 28.5)]:
            fig.text(x / width, y / height, letter, fontsize=11, va="top")
        # Align the entire C plot (including ticks and xlabel) to the bottom
        # of A, holding its top edge fixed and preserving the font sizes.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        plot_bbox = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        position = ax.get_position()
        bottom_shift = overview_ax.get_position().y0 - plot_bbox.y0
        ax.set_position(
            [
                position.x0,
                position.y0 + bottom_shift,
                position.width,
                position.height - bottom_shift,
            ]
        )
        fig.canvas.draw()
        final_bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted()
        )
        assert abs(final_bbox.y0 - overview_ax.get_position().y0) < 1e-6
        save(fig, targets[0].with_suffix(""))
        # Full configuration comparison, vector curves with identical geometry,
        # camera and scale in every tile. Appendix is deliberately data-dense.
        width, height = 180.0, 169.0
        fig = plt.figure(figsize=(width / 25.4, height / 25.4))
        fig.legend(
            handles=legends(True),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
        )
        meta = metadata["comparison_1"]
        for index in range(22):
            row, col = divmod(index, 6)
            ax = axes_mm(fig, 2 + col * 29.5, 4 + (3 - row) * 39, 28, 35, width, height)
            draw_comparison(
                ax,
                d,
                index,
                meta,
                True,
                linewidth=0.8,
                markersize=2.5,
                marker_geometry=marker_geometry,
            )
            ax.set_xlim(125, 565)
            ax.set_ylim(1055, 274)
            ax.text(
                0.5,
                1.01,
                str(index + 1),
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
            )
        save(fig, targets[1].with_suffix(""))
        width, height = 180.0, 76.0
        fig = plt.figure(figsize=(width / 25.4, height / 25.4))
        fig.legend(
            handles=legends(True)[:4],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=4,
            frameon=False,
            handlelength=1.6,
            columnspacing=1.0,
        )
        ax = axes_mm(fig, 16, 13, 65, 50, width, height)
        for stage, color, style in zip(STAGES, COLORS, LINESTYLES, strict=True):
            rms = 1000 * np.sqrt(
                np.mean(np.sum(d[stage + "_errors"] ** 2, axis=-1), axis=0)
            )
            ax.plot(np.arange(1, 5), rms, color=color, ls=style, marker="o", ms=3)
        ax.set_xticks(np.arange(1, 5), [f"$M_{i}$" for i in range(1, 5)])
        ax.set_xlabel("Marker")
        ax.set_ylabel("Position RMSE [mm]")
        ax.set_ylim(bottom=0)
        ax = axes_mm(fig, 105, 13, 70, 50, width, height)
        for i, (stage, color) in enumerate(zip(STAGES, COLORS, strict=True)):
            errors = 1000 * np.linalg.norm(d[stage + "_errors"], axis=-1).ravel()
            # Empirical distributions preserve every error without smoothing.
            ax.plot(
                np.sort(errors), np.arange(1, 89) / 88, color=color, ls=LINESTYLES[i]
            )
        ax.set_xlabel("Marker position error [mm]")
        ax.set_ylabel("Cumulative fraction")
        ax.set_ylim(0, 1)
        for letter, x in [("A", 2), ("B", 90)]:
            fig.text(x / width, 64 / height, letter, fontsize=11, va="top")
        save(fig, targets[2].with_suffix(""))
    manifest = dict(
        audit=audit,
        dimensions_mm=[180, 90],
        marker_geometry_sha256=sha256(marker_geometry_path),
        overview_crop_pixels=[0, 280, 1152, 1720],
        comparison_crop_pixels=[0, 130, 680, 1080],
        render_status="local preview"
        if any("preview" in m.get("backend", "") for m in metadata.values())
        else "RTX studio",
        composer_sha256=sha256(Path(__file__)),
        renders={
            name: dict(image_sha256=sha256(scenes / f"{name}.png"), metadata=meta)
            for name, meta in metadata.items()
        },
        pdf_sha256=sha256(targets[0]),
    )
    (output / "model_identification_residual_learning.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(targets[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir", type=Path, default=CASE / "outputs/identification"
    )
    parser.add_argument(
        "--scene-dir", type=Path, default=CASE / "outputs/paper_snapshots"
    )
    parser.add_argument("--output-dir", type=Path, default=PAPER / "final_outputs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    compose(args.evaluation_dir, args.scene_dir, args.output_dir, args.force)
