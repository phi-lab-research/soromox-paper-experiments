"""Render independent RTX frames and compose the recorded gain-tuning motion."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parent


def render_frames(geometry_dir, frame_dir, workers):
    frame_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(geometry_dir.glob("initial_*_geometry.npz")) + sorted(
        geometry_dir.glob("optimized_*_geometry.npz")
    )
    if not sources:
        raise ValueError("No motion geometry found")

    def one(path):
        stem = path.name.removesuffix("_geometry.npz")
        output = frame_dir / f"{stem}.png"
        # A fresh process per saved state avoids RTX accumulation across poses.
        with (frame_dir / f"{stem}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(CODE / "render_control_tuning_scene.py"),
                    "--geometry",
                    str(path),
                    "--output",
                    str(output),
                    "--samples",
                    "64",
                    "--force",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return stem

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, p) for p in sources]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"{index}/{len(sources)} {future.result()}", flush=True)


def compose(frame_dir, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_control_tuning_summary import draw_snapshot
    from control_tuning_data import sha256
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to compose the recorded frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    composite = frame_dir / "composite"
    composite.mkdir(exist_ok=True)
    sources = sorted(frame_dir.glob("initial_*.png"))
    if len(sources) != 61:
        raise ValueError("Expected 61 saved motion frames")
    metadata = []
    for index, path in enumerate(sources):
        plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 12})
        fig = plt.figure(figsize=(9, 6), dpi=120, facecolor="white")
        pair = []
        for col, kind in enumerate(("initial", "optimized")):
            image = frame_dir / f"{kind}_{index:03d}.png"
            meta = json.loads(image.with_suffix(".json").read_text())
            ax = fig.add_axes([0.035 + col * 0.5, 0.02, 0.43, 0.88])
            draw_snapshot(ax, image, meta)
            ax.set_title(
                "Initial gains" if col == 0 else "Optimized gains",
                color="#006BA6" if col == 0 else "#f1552e",
                pad=6,
            )
            pair.append(dict(metadata=meta, image_sha256=sha256(image)))
        if pair[0]["metadata"]["time_s"] != pair[1]["metadata"]["time_s"]:
            raise ValueError("Frame times differ")
        fig.text(
            0.5,
            0.975,
            f"t = {pair[0]['metadata']['time_s']:.2f} s",
            ha="center",
            va="top",
        )
        fig.savefig(composite / f"frame_{index:03d}.png")
        plt.close(fig)
        metadata.append(pair)
    output = output_dir / "control_gain_optimization_comparison.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "12",
            "-i",
            str(composite / "frame_%03d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(output),
            "-filter_complex",
            "fps=12,scale=900:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop",
            "0",
            str(output.with_suffix(".gif")),
        ],
        check=True,
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                fps=12,
                frames=metadata,
                description="Recorded initial-median-loss and optimized-best runs; same cameras and physical times.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.render_only:
        if args.geometry_dir is None:
            parser.error("--geometry-dir is required for rendering")
        render_frames(args.geometry_dir, args.frame_dir, args.workers)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required for composition")
        compose(args.frame_dir, args.output_dir)
