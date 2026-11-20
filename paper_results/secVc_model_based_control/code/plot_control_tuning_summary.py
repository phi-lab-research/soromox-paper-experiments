"""Compose the combined pose-control and gain-tuning paper figure."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from control_tuning_data import CASE, PAPER, load_data, sha256

sys.path.insert(0, str(PAPER))
from paper_render_style import PAPER_STUDIO
from paper_style import PAPER_COLORS, PAPER_STYLE_PATH


def project_point(point, metadata):
    eye = np.asarray(metadata["camera_eye"], float)
    forward = np.asarray(metadata["camera_target"]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(point) - eye
    scale = metadata["height"] / (
        2
        * (relative @ forward)
        * np.tan(np.deg2rad(metadata["vertical_fov_degrees"]) / 2)
    )
    return [
        metadata["width"] / 2 + scale * (relative @ right),
        metadata["height"] / 2 - scale * (relative @ up),
    ]


def load_snapshots(root, tracking, cases, audit):
    snapshots = {}
    for name in ("tracking", "initial", "optimized"):
        path = root / f"{name}.png"
        meta = json.loads(path.with_suffix(".json").read_text())
        source = audit["tracking" if name == "tracking" else "synergistic"][
            "source_sha256"
        ]
        if meta["source_sha256"] != source or (
            not meta["current_state_only"]
            and not (name == "tracking" and meta.get("solid_state_is_recorded"))
        ):
            raise ValueError(
                "Snapshot must show the current state of the plotted source"
            )
        if meta["geometry_sha256"] != sha256(root / f"{name}_geometry.npz"):
            raise ValueError("Snapshot geometry changed")
        if meta["studio"] != json.loads(json.dumps(PAPER_STUDIO.manifest())):
            raise ValueError("Snapshot studio is stale")
        if name != "tracking":
            d = cases["synergistic"]
            tag = "init" if name == "initial" else "best"
            batch = d["initial_batch"] if name == "initial" else int(d["best_batch"])
            if meta["batch_zero_based"] != batch:
                raise ValueError("Snapshot selected the wrong run")
            np.testing.assert_allclose(meta["time_s"], d["t_ts"][meta["timestep"]])
            transform = np.asarray(meta["display_transform"])
            expected = (
                transform[:3, :3] @ d[f"x_ts_{tag}"][batch, meta["timestep"], 3:]
                + transform[:3, 3]
            )
            np.testing.assert_allclose(meta["end_effector_world"], expected, atol=1e-8)
        else:
            np.testing.assert_allclose(
                meta["time_s"], tracking["t_traj"][meta["timestep"]]
            )
            for ghost in meta.get("recorded_pose_overlays", []):
                k = ghost["timestep"]
                np.testing.assert_allclose(ghost["time_s"], tracking["t_traj"][k])
                transform = np.asarray(meta["display_transform"])
                expected = (
                    transform[:3, :3] @ tracking["x_traj_full"][k, 3:]
                    + transform[:3, 3]
                )
                np.testing.assert_allclose(
                    ghost["end_effector_world"], expected, atol=1e-8
                )
        snapshots[name] = (path, dict(meta, image_sha256=sha256(path)))
    for key in (
        "camera_eye",
        "camera_target",
        "time_s",
        "display_transform",
        "width",
        "height",
    ):
        if snapshots["initial"][1][key] != snapshots["optimized"][1][key]:
            raise ValueError("Tuning views must have matched camera, scale, and time")
    return snapshots


def draw_snapshot(ax, path, meta):
    ax.imshow(plt.imread(path))
    if meta["kind"] != "tracking":
        ax.scatter(
            *project_point(meta["target_world"], meta),
            s=42,
            facecolors="none",
            edgecolors=PAPER_STUDIO.target,
            linewidths=1.1,
            zorder=4,
            path_effects=[pe.Stroke(linewidth=1.9, foreground="white"), pe.Normal()],
        )
        ax.scatter(
            *project_point(meta["end_effector_world"], meta),
            s=12,
            facecolors="white",
            edgecolors=PAPER_STUDIO.mount,
            linewidths=0.8,
            zorder=5,
        )
    ax.set_axis_off()


def build_figure(tracking, cases, snapshots):
    plt.style.use(PAPER_STYLE_PATH)
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "text.usetex": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": None,
            "svg.hashsalt": "soromox-control-tuning",
        }
    )
    width, height = 180, 118.5
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor="white")

    def rect(x, y, w, h):
        return [x / width, y / height, w / width, h / height]

    for name, x, y, w, h in [
        ("tracking", 2, 54, 82, 62.5),
        ("initial", 88, 54, 44.4, 62.5),
        ("optimized", 134, 54, 44.4, 62.5),
    ]:
        ax = fig.add_axes(rect(x, y, w, h))
        draw_snapshot(ax, *snapshots[name])
        ax.set_ylim(1250, 0)
        if name != "tracking":
            ax.set_xlim(60, 948)
        if name in ("tracking", "initial"):
            ax.text(
                0.02,
                0.98,
                "A" if name == "tracking" else "B",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
                fontweight="bold",
            )
        if name != "tracking":
            ax.text(
                0.55,
                0.975,
                "Initial (selected)" if name == "initial" else "Optimized (best)",
                transform=ax.transAxes,
                va="top",
                ha="center",
                fontsize=8.5,
                color=PAPER_COLORS["pre_opt_1" if name == "initial" else "post_opt_1"],
            )
    for key, y, label in [
        ("position_error_mm", 33, "Position error\n[mm]"),
        ("orientation_error_deg", 9, "Orientation error\n[deg]"),
    ]:
        ax = fig.add_axes(rect(15, y, 69, 18.5))
        ax.plot(
            tracking["t_traj"], tracking[key], color=PAPER_COLORS["pre_opt_1"], lw=1.2
        )
        ax.set(xlim=(0, 10), ylim=(0, 10.5), yticks=[0, 5, 10], ylabel=label)
        if key == "position_error_mm":
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time [s]")
        ax.grid(color=".9", lw=0.5)
        ax.set_axisbelow(True)
    ax = fig.add_axes(rect(97.5, 9, 78.5, 42.5))
    d = cases["synergistic"]
    initial_errors = 1000 * np.linalg.norm(
        d["x_ts_init"][:, :, 3:] - d["x_des_ts"][None, :, 3:], axis=-1
    )
    initial_band = ax.fill_between(
        d["t_ts"],
        initial_errors.min(axis=0),
        initial_errors.max(axis=0),
        color=PAPER_COLORS["pre_opt_1"],
        alpha=0.16,
        linewidth=0,
        label="Initial range",
        zorder=1,
    )
    for key, label, color in [
        ("init", "Initial (selected)", PAPER_COLORS["pre_opt_1"]),
        ("best", "Optimized (best)", PAPER_COLORS["post_opt_1"]),
    ]:
        ax.plot(
            d["t_ts"], d[f"position_error_{key}_mm"], label=label, color=color, lw=1.4
        )
        index = snapshots["initial"][1]["timestep"]
        ax.plot(
            d["t_ts"][index],
            d[f"position_error_{key}_mm"][index],
            marker="o",
            ms=4,
            color=color,
            mec="white",
            mew=0.5,
            clip_on=False,
            zorder=5,
        )
    ax.set(
        xlim=(0, 5),
        ylim=(0, 28),
        yticks=[0, 10, 20],
        xlabel="Time [s]",
        ylabel="Position error [mm]",
    )
    ax.grid(color=".9", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[ax.lines[0], initial_band, ax.lines[2]],
        frameon=False,
        loc="upper right",
    )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-dir", type=Path, default=CASE / "outputs/paper_snapshots"
    )
    parser.add_argument("--output-dir", type=Path, default=PAPER / "final_outputs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    stem = args.output_dir / "control_and_gain_optimization"
    outputs = [stem.with_suffix(ext) for ext in (".pdf", ".png", ".svg", ".json")]
    if any(p.exists() for p in outputs) and not args.force:
        raise FileExistsError("Figure exists; use --force")
    tracking, cases, audit = load_data()
    snapshots = load_snapshots(args.snapshot_dir, tracking, cases, audit)
    fig = build_figure(tracking, cases, snapshots)
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
                audit=audit,
                initial_envelope=dict(
                    statistic="pointwise min-max of Euclidean position-error norms",
                    num_initializations=int(cases["synergistic"]["x_ts_init"].shape[0]),
                ),
                snapshot_crop_y=[0, 1250],
                gain_snapshot_crop_x=[60, 948],
                snapshots={k: m for k, (_, m) in snapshots.items()},
                plotter_sha256=sha256(Path(__file__)),
            ),
            indent=2,
        )
        + "\n"
    )
    print(outputs[0])


if __name__ == "__main__":
    main()
