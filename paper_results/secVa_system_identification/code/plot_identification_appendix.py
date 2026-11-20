"""Compose a compact identification overlay, marker RMSE, and error distribution."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_identification_summary import (
    CASE,
    PAPER,
    STAGES,
    COLORS,
    LINESTYLES,
    axes_mm,
    configure,
    legends,
    sha256,
)

NAME = "model_identification_residual_learning_appendix"


def compose(evaluation, scenes, output, force=False):
    targets = [output / f"{NAME}.{ext}" for ext in ("pdf", "png", "svg", "json")]
    if any(path.exists() for path in targets) and not force:
        raise FileExistsError("Figure exists; pass --force")
    data_path = evaluation / "identification_evaluation.npz"
    marker_path = scenes / "marker_geometry.npz"
    audit = json.loads((evaluation / "identification_audit.json").read_text())
    meta = json.loads((scenes / "comparison_1.json").read_text())
    if sha256(data_path) != audit["evaluation_sha256"]:
        raise ValueError("Audit and evaluation disagree")
    if meta["evaluation_sha256"] != audit["evaluation_sha256"]:
        raise ValueError("Camera metadata and evaluation disagree")
    if sha256(marker_path) != meta["marker_geometry_sha256"]:
        raise ValueError("Marker geometry differs from scene provenance")
    configure()
    width, height = 158.0, 113.0
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    fig.legend(
        handles=legends(True),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
        columnspacing=1.7,
        handlelength=2.6,
    )
    per_marker = {}
    with np.load(data_path) as d:
        # Match the original paper's shared 3D view: the eight physical-fit
        # configurations, two parameter stages, and only M2/M4 for clarity.
        ax = fig.add_axes(
            [2 / width, 4 / height, 93 / width, 95 / height], projection="3d"
        )
        shown_markers = [1, 3]
        for stage, color, style in zip(
            STAGES[:2], COLORS[:2], LINESTYLES[:2], strict=True
        ):
            for index in range(8):
                curve = 1000 * d[stage + "_poses"][index, :, :3, 3]
                ax.plot(*curve.T, color=color, ls=style, lw=1.0)
                markers = 1000 * d[stage + "_markers"][index, shown_markers]
                ax.scatter(
                    *markers.T,
                    marker="x",
                    color=color,
                    s=12,
                    linewidths=0.8,
                    depthshade=False,
                )
        for index in range(8):
            measured = 1000 * d["measured"][index, shown_markers]
            ax.scatter(
                *measured.T,
                marker="o",
                facecolors="white",
                edgecolors="#FFBC42",
                s=14,
                linewidths=0.9,
                depthshade=False,
            )
        ax.view_init(elev=20, azim=15)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_zlim(-350, 0)
        ax.set_box_aspect([360, 360, 350], zoom=1.04)
        ax.set_xticks([-150, 0, 150])
        ax.set_yticks([-150, 0, 150])
        ax.set_zticks([-300, -200, -100, 0])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((0.95, 0.95, 0.95, 0.35))
            axis.pane.set_edgecolor("#DDDDDD")
            axis._axinfo["grid"].update(color=(0.75, 0.75, 0.75, 0.4), linewidth=0.45)
        ax.tick_params(pad=-1, labelsize=7.5)
        ax.set_xlabel("$x$ [mm]", labelpad=-3)
        ax.set_ylabel("$y$ [mm]", labelpad=-3)
        ax.set_zlabel("$z$ [mm]", labelpad=-3)

        # One common linear radial scale for all markers and model stages.
        ax = fig.add_axes(
            [107 / width, 56 / height, 44 / width, 44 / height], projection="polar"
        )
        theta = np.arange(4) * np.pi / 2
        closed_theta = np.r_[theta, theta[0]]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(theta, [f"$M_{i}$" for i in range(1, 5)])
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_rlabel_position(45)
        ax.tick_params(axis="x", pad=1)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["polar"].set_color("#B0B0B0")
        ax.grid(True, color="#A0A0A0", alpha=0.5)
        for stage, color, style in zip(STAGES, COLORS, LINESTYLES, strict=True):
            errors = d[stage + "_errors"]
            rms = 1000 * np.sqrt(np.mean(np.sum(errors**2, axis=-1), axis=0))
            per_marker[stage] = rms.tolist()
            # RMS across the four marker RMSEs must match the audited pooled score.
            np.testing.assert_allclose(
                np.sqrt(np.mean(rms**2)), audit["rmse_mm"][stage]["all_22"], rtol=1e-12
            )
            ax.plot(
                closed_theta,
                np.r_[rms, rms[0]],
                color=color,
                ls=style,
                lw=1.3,
                marker="o",
                ms=2.6,
            )
        fig.text(
            129 / width,
            47 / height,
            "Marker position RMSE [mm]",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

        ax = axes_mm(fig, 104, 13, 50, 30, width, height)
        for stage, color, style in zip(STAGES, COLORS, LINESTYLES, strict=True):
            errors = np.sort(
                1000 * np.linalg.norm(d[stage + "_errors"], axis=-1).ravel()
            )
            ax.step(
                np.r_[0, errors],
                np.r_[0, np.arange(1, len(errors) + 1) / len(errors)],
                where="post",
                color=color,
                ls=style,
                lw=1.3,
            )
        ax.set_xlabel("Marker position error [mm]")
        ax.set_ylabel("Cumulative fraction")
        ax.set_xlim(left=0)
        ax.set_xticks([0, 50, 100, 150])
        ax.set_ylim(0, 1.02)
        ax.set_yticks(np.arange(0, 1.01, 0.25))

    for letter, x, y in (("A", 1, 101), ("B", 94, 101), ("C", 94, 47)):
        fig.text(x / width, y / height, letter, fontsize=11, va="top")
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        path = output / f"{NAME}.{ext}"
        fig.savefig(path, dpi=300)
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)
    manifest = {
        "dimensions_mm": [width, height],
        "shape_configurations_one_based": list(range(1, 9)),
        "shape_stages": list(STAGES[:2]),
        "shape_markers_one_based": [2, 4],
        "quantitative_configurations_one_based": list(range(1, 23)),
        "evaluation_sha256": sha256(data_path),
        "marker_geometry_sha256": sha256(marker_path),
        "composer_sha256": sha256(Path(__file__)),
        "shared_composer_sha256": sha256(
            Path(__file__).with_name("plot_identification_summary.py")
        ),
        "per_marker_rmse_mm": per_marker,
        "metric": "sqrt(mean over 22 configurations of squared Euclidean marker error)",
        "distribution": "empirical cumulative distribution of all 88 Euclidean errors per stage",
        "generalization_scope": audit["generalization_scope"],
        "pdf_sha256": sha256(output / f"{NAME}.pdf"),
    }
    (output / f"{NAME}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / f"{NAME}.pdf")


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
