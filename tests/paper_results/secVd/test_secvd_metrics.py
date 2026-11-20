"""Analytic response cases and saved-data checks for reported performance."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

CASE = (
    Path(__file__).resolve().parents[3]
    / "paper_results/secVd_control_gain_optimization"
)
sys.path.insert(0, str(CASE / "code"))
from secvd_metrics import (
    response_metrics,
    median_metric,
    reduction_percent,
    report,
    write_report,
)


def test_entry_is_not_settling_and_overshoot_is_directional():
    t = np.arange(5.0)
    response = np.array([[0.0], [0.96], [1.2], [1.04], [1.0]])
    result = response_metrics(t, response, np.ones_like(response))
    assert result["first_entry_s"] == 1.0
    assert result["settling_s"] == 3.0
    assert result["maximum_coordinate_overshoot"] == pytest.approx(0.2)
    assert result["integrated_error"] == pytest.approx(0.78)
    reverse = response_metrics(t, -response, -np.ones_like(response))
    assert reverse == result


def test_no_settling_is_not_a_hundred_percent_reduction():
    response = np.array([[0.0], [1.0], [1.1]])
    result = response_metrics(np.arange(3.0), response, np.ones_like(response))
    assert result["first_entry_s"] == 1.0
    assert result["settling_s"] is None
    assert median_metric([1.0, None, None]) is None
    assert median_metric([1.0, 2.0, None]) == 2.0
    assert reduction_percent(None, 0.455) is None
    assert reduction_percent(0, 0) is None


def test_already_at_target_and_zero_step_coordinates():
    t = np.arange(3.0)
    y = np.zeros((3, 2))
    assert response_metrics(t, y, y)["settling_s"] == 0
    y[:, 0] = [0, 1, 1]
    y[:, 1] = [0, 0.2, 0]
    target = np.tile([1, 0], (3, 1))
    result = response_metrics(t, y, target)
    assert result["settling_s"] == 2
    assert result["coordinate_overshoot"] == [0, 0]


@pytest.mark.parametrize("kind", ["nonfinite", "time", "target"])
def test_invalid_responses_are_rejected(kind):
    t, y, target = np.arange(3.0), np.zeros((3, 1)), np.ones((3, 1))
    if kind == "nonfinite":
        y[1] = np.nan
    elif kind == "time":
        t[1] = t[0]
    else:
        target[-1] = 2
    with pytest.raises(ValueError):
        response_metrics(t, y, target)


def test_saved_data_metrics_and_json(tmp_path):
    result = report(CASE / "data")
    act, op = result["collocated"], result["synergistic"]
    assert act["initial_metric_median"]["settling_s"] == pytest.approx(0.3785)
    assert act["optimized"]["settling_s"] == pytest.approx(0.137)
    assert act["optimized"]["maximum_coordinate_overshoot"] == pytest.approx(0.8060905)
    assert act["reductions_from_initial_metric_median_percent"][
        "settling_s"
    ] == pytest.approx(63.8044914)
    assert op["initial_metric_median"]["settling_s"] is None
    assert all(run["settling_s"] is None for run in op["initial_runs"])
    assert op["optimized"]["settling_s"] == pytest.approx(0.455)
    assert op["optimized"]["maximum_coordinate_overshoot"] == pytest.approx(4.24445765)
    assert op["optimized"]["integrated_error"] == pytest.approx(2.911873, abs=1e-6)
    path = tmp_path / "metrics.json"
    write_report(path, result)
    assert json.loads(path.read_text()) == result
