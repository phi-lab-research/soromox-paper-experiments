"""Checks that publication summaries retain the saved experimental evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "paper_results" / "secVf_sampling_based_mpc" / "code"
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location(
    "manipulation_figure", CODE / "plot_smpc_summary.py"
)
figure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure)


def test_published_outcomes_and_all_six_paths():
    data = CODE.parent / "data"
    with (data / "square_icem_seed0.npz").open("rb") as stream:
        if stream.read(7) == b"version":
            pytest.skip("Fetch the released trajectory with git lfs pull")
    methods, goals = figure.load_runs(data)
    assert goals.shape == (4, 2)
    assert methods["iCEM"]["success"] == 3
    assert methods["MPPI"]["success"] == 1
    assert methods["iCEM"]["planning_mean_s"] == pytest.approx(26.543102, abs=1e-5)
    assert methods["MPPI"]["planning_mean_s"] == pytest.approx(8.503319, abs=1e-5)
    fig, ax = figure.plt.subplots()
    figure.draw_paths(ax, methods, goals)
    # The first line is the desired square, followed by all six recorded paths.
    expected = [
        xy * 1000 for method in methods.values() for xy in method["trajectories"]
    ]
    assert len(ax.lines) == 1 + len(expected)
    for line, xy in zip(ax.lines[1:], expected, strict=True):
        np.testing.assert_allclose(line.get_xydata(), xy)
    figure.plt.close(fig)


def test_rejects_snapshots_from_different_data_or_style(tmp_path):
    source = tmp_path / "run.npz"
    source.write_bytes(b"saved trajectory")
    root = tmp_path / "snapshots"
    view = root / "overview"
    view.mkdir(parents=True)
    manifest = {
        "source_sha256": "wrong",
        "view": "overview",
        "studio": figure.PAPER_STUDIO.manifest(),
        "snapshots": [],
    }
    (view / "snapshots.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="trajectory differs"):
        figure.load_snapshot(root, "overview", 19, source)
    manifest["source_sha256"] = figure.sha256(source)
    manifest["studio"]["robot"] = "#000000"
    (view / "snapshots.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="style is stale"):
        figure.load_snapshot(root, "overview", 19, source)


def test_existing_figure_requires_force(tmp_path):
    (tmp_path / "sampling_based_mpc.pdf").write_bytes(b"existing figure")
    with pytest.raises(FileExistsError, match="--force"):
        figure.main(["--output-dir", str(tmp_path)])
    assert (tmp_path / "sampling_based_mpc.pdf").read_bytes() == b"existing figure"


def test_linear_material_colors():
    from paper_render_style import rgb

    assert rgb("#FFFFFF", linear=True) == (1.0, 1.0, 1.0)
    assert rgb("#000000", linear=True) == (0.0, 0.0, 0.0)
    assert rgb("#808080", linear=True) == pytest.approx((0.2158605,) * 3)
    assert rgb("#808080") == pytest.approx((128 / 255,) * 3)
