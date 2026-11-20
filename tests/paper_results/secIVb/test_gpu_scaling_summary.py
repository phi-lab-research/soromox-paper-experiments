"""Protect the paper figure's measurement scope and throughput interpretation."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CASE = REPO / "paper_results/secIVb_parallel_rollouts_gpu"
spec = importlib.util.spec_from_file_location(
    "gpu_summary", CASE / "code/plot_gpu_scaling_summary.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CSV = CASE / "data/parallel_rollout_benchmark.csv"


def test_archived_measurements_and_reported_gains():
    _, audit = module.audit_measurements(CSV)
    assert audit["rows"] == 264
    gvs = audit["systems"]["gvs"]["sizes"]
    assert gvs["1"]["gain_1024_vs_1"] == pytest.approx(679.737397)
    assert gvs["32"]["gain_1024_vs_1"] == pytest.approx(133.614434)
    assert gvs["32"]["peak_batch"] == 512
    assert not gvs["32"]["monotone_throughput"]
    assert gvs["32"]["throughput_at_1024"] < 1


def test_duplicate_cannot_replace_a_missing_case(tmp_path):
    data, _ = module.audit_measurements(CSV)
    data.iloc[-1] = data.iloc[0]
    path = tmp_path / "incomplete.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly one measurement"):
        module.audit_measurements(path)


def test_per_environment_rate_cannot_be_presented_as_aggregate(tmp_path):
    data, _ = module.audit_measurements(CSV)
    data["total_speed_ratio"] = data["per_env_speed_ratio"]
    path = tmp_path / "wrong_normalization.csv"
    data.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        module.audit_measurements(path)
