"""Check saved-run selection, units, envelopes, and appendix scope."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "paper_results/secVc_model_based_control/code"
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location(
    "control_tuning_appendix", CODE / "plot_control_tuning_appendix.py"
)
appendix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appendix)


@pytest.fixture(scope="module")
def data():
    return appendix.load_appendix_data()


def test_audit_has_correct_model_scope_and_real_runs(data):
    run, group, tuning, audit = data
    assert len(group.controller_names) == 5
    assert audit["configuration_segments"] == audit["tuning_segments"] == 1
    assert audit["tuning_tendons"] == 3
    assert audit["initial_batch_zero_based"] == 5
    assert audit["optimized_batch_zero_based"] == 1
    assert audit["optimized_iteration_zero_based"] == 99
    np.testing.assert_allclose(
        audit["improvement_from_initial_batch_median_percent"], 46.557604750837015
    )


def test_figure_preserves_units_run_selection_and_initial_range(data):
    run, group, tuning, audit = data
    fig = appendix.build_figure(*data)
    try:
        assert len(fig.axes) == 5
        first = run.results[group.scenario_key][group.controller_names[0]]
        # Physical axial strain has an undeformed offset, unlike saved displacement.
        np.testing.assert_allclose(
            fig.axes[2].lines[0].get_ydata(),
            1 + first["metrics"]["q_des"][:, run.strain_indices[2]],
        )
        for ax in fig.axes[:3]:
            assert ax.get_xlim() == (0, 30)
            assert len(ax.lines) == 7  # Desired, five controllers, phase boundary.
        np.testing.assert_allclose(
            fig.axes[3].lines[0].get_ydata(), 1000 * tuning["history_loss"].min(axis=1)
        )
        assert len(fig.axes[3].lines[0].get_xdata()) == 100
        ax = fig.axes[4]
        assert ax.get_xlim() == (0, 2)
        for k in range(3):
            np.testing.assert_allclose(
                ax.lines[3 * k].get_ydata(), tuning["q_ts_init"][5, :, k]
            )
            np.testing.assert_allclose(
                ax.lines[3 * k + 1].get_ydata(), tuning["q_ts_best"][1, :, k]
            )
            vertices = ax.collections[k].get_paths()[0].vertices
            for i in (0, 500, 2000):
                ys = vertices[np.isclose(vertices[:, 0], tuning["t_ts"][i]), 1]
                np.testing.assert_allclose(
                    [ys.min(), ys.max()],
                    [
                        tuning["q_ts_init"][:, i, k].min(),
                        tuning["q_ts_init"][:, i, k].max(),
                    ],
                )
    finally:
        appendix.plt.close(fig)


def test_refuses_to_overwrite_existing_figure(tmp_path):
    pdf = tmp_path / "control_and_gain_optimization_appendix.pdf"
    pdf.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        appendix.main(["--output-dir", str(tmp_path)])
    assert pdf.read_bytes() == b"existing"
