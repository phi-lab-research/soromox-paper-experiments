"""Marker error metrics and representative configuration selection."""

import numpy as np


def marker_rmse(errors):
    """RMS Euclidean marker error; do not average norms or divide by XYZ."""
    errors = np.asarray(errors)
    if errors.shape[-2:] != (4, 3) or not np.isfinite(errors).all():
        raise ValueError("Expected finite errors with trailing shape (4, 3)")
    return np.sqrt(np.mean(np.sum(errors**2, axis=-1)))


def representative_indices(errors):
    """One median-error case per tendon-input family, selected before residuals."""
    rms = np.sqrt(np.mean(np.sum(errors**2, axis=-1), axis=1))
    families = ([0, 1, 2, 3, 8, 9, 10], [4, 5, 6, 7, 11, 12, 13], list(range(14, 22)))
    return [
        int(ids[np.argmin(abs(rms[ids] - np.median(rms[ids])))]) for ids in families
    ]


def dataset_rmse_mm(errors):
    """Report pooled, physical-fitting, and source-declared network subset errors."""
    errors = np.asarray(errors)
    if errors.shape != (22, 4, 3):
        raise ValueError("Expected the 22 configurations and four markers")
    # The source training script holds out configurations 1 and 16 (zero-based).
    test = np.array([1, 16])
    train = np.setdiff1d(np.arange(22), test)
    return {
        name: float(1000 * marker_rmse(errors[indices]))
        for name, indices in (
            ("all_22", np.arange(22)),
            ("parameter_fit_8", np.arange(8)),
            ("declared_nn_train_20", train),
            ("declared_nn_test_2", test),
        )
    }
