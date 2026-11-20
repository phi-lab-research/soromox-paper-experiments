"""Check scientific meanings that were ambiguous in the original figure."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "paper_results/secVa_system_identification/code"
sys.path.insert(0, str(CODE))
from identification_data import marker_rmse, measured_from_csv, representative_indices
from identify_soft_tentacle_residual import TauNN, checkpoint_features
import jax
import equinox as eqx

DATA = REPO / "paper_results/secVa_system_identification/data"
EVALUATION = REPO / "paper_results/secVa_system_identification/outputs/identification"


def test_rmse_uses_squared_euclidean_errors():
    errors = np.zeros((1, 4, 3))
    errors[0, 0, :2] = [3, 4]
    assert marker_rmse(errors) == 2.5
    assert marker_rmse(errors) != np.linalg.norm(errors, axis=-1).mean()
    with pytest.raises(ValueError):
        marker_rmse(np.zeros((4, 2)))


def test_measurements_match_raw_csvs():
    np.testing.assert_allclose(
        measured_from_csv(DATA), np.load(DATA / "measured_pts.npy"), atol=1e-14
    )


def test_current_tension_api_reproduces_archived_checkpoint_predictions():
    model = TauNN(2, 9, width=512, depth=4, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(DATA / "tau_nn_model.eqx", model)
    with np.load(EVALUATION / "identification_evaluation.npz") as d:
        actual = np.asarray(jax.vmap(model)(checkpoint_features(d["u"])))
        np.testing.assert_allclose(actual, d["tau_learned"], atol=1e-12)
        # Positive features do not reproduce these trained weights' predictions.
        assert not np.allclose(np.asarray(jax.vmap(model)(d["u"])), actual)


def test_common_dataset_errors_and_representatives():
    with np.load(EVALUATION / "identification_evaluation.npz") as d:
        for stage, value in [
            ("initial", 49.8236242),
            ("identified", 19.2606062),
            ("learned", 6.8853304),
            ("fitted_residual", 6.5529055),
        ]:
            assert d[stage + "_errors"].shape == (22, 4, 3)
            np.testing.assert_allclose(
                d[stage + "_markers"] - d["measured"], d[stage + "_errors"]
            )
            assert 1000 * marker_rmse(d[stage + "_errors"]) == pytest.approx(
                value, abs=1e-5
            )
        # Keep the eight-input identification result separate from the pooled score.
        assert 1000 * marker_rmse(d["exact_identified_errors"][:8]) == pytest.approx(
            22.9285776, abs=1e-5
        )
        assert representative_indices(d["identified_errors"]) == [1, 5, 17]
    audit = json.loads((EVALUATION / "identification_audit.json").read_text())
    assert audit["declared_nn_test_indices_zero_based"] == [1, 16]
    assert "no training metadata" in audit["checkpoint_split_provenance"]
    assert max(audit["archive_checks_max_abs"].values()) < 5e-7


def test_rendered_marker_offsets_match_the_measurement_model():
    geometry_path = EVALUATION.parent / "paper_snapshots/marker_geometry.npz"
    with (
        np.load(geometry_path) as geometry,
        np.load(EVALUATION / "identification_evaluation.npz") as d,
    ):
        for stage in ("initial", "identified", "learned", "fitted_residual"):
            delta = d[stage + "_markers"] - geometry[stage + "_anchors"]
            np.testing.assert_allclose(
                np.linalg.norm(delta, axis=-1),
                np.tile([0.025, 0.021, 0.020, 0.008], (22, 1)),
                atol=1e-12,
            )
        # Displayed measured/predicted separation is the real remaining error.
        np.testing.assert_allclose(
            1000 * np.linalg.norm(d["learned_errors"][17], axis=-1),
            [9.137, 7.232, 5.020, 7.306],
            atol=0.001,
        )
