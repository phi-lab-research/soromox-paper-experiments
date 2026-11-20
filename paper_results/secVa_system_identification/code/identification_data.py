"""Audit and re-evaluate the saved Section Va models on a common dataset.

No parameter fitting or neural-network training is performed here.
"""

import argparse
import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pandas as pd

from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting
from soromox.systems import (
    GVS,
    GVSSegment,
    JointSpec,
    LinearProfile,
    LinkSpec,
    RodConstitutiveLaw,
    StrainBasisSpec,
    TorsionRodModel,
    TransverseShearRodModel,
)
from identify_soft_tentacle_residual import TauNN, checkpoint_features, markers_from_q

CASE = Path(__file__).resolve().parents[1]
DATA = CASE / "data"
STAGES = ("initial", "identified", "learned", "fitted_residual")
# The checkpoint predates the positive-tension API. Its features remain negative.
CHECKPOINT_INPUT_SIGN = -1.0
DECLARED_TEST_INDICES = [1, 16]
LOAD_PAIRS = [
    (0.1, 0),
    (0.15, 0),
    (0.2, 0),
    (0.25, 0),
    (0, 0.1),
    (0, 0.15),
    (0, 0.2),
    (0, 0.25),
    (0.05, 0),
    (0.12, 0),
    (0.18, 0),
    (0, 0.05),
    (0, 0.12),
    (0, 0.18),
    (0.1, 0.1),
    (0.2, 0.2),
    (0.1, 0.2),
    (0.15, 0.2),
    (0.2, 0.1),
    (0.2, 0.15),
    (0.1, 0.05),
    (0.05, 0.1),
]
CSV_MARKERS = [
    ("m1_u01.csv", [4, 0, 3, 1, 2]),
    ("m1_u015.csv", [3, 2, 4, 1, 0]),
    ("m1_u02.csv", [3, 2, 4, 1, 0]),
    ("m1_u025.csv", [4, 3, 0, 2, 1]),
    ("m2_u01.csv", [2, 0, 4, 3, 1]),
    ("m2_u015.csv", [3, 2, 4, 1, 0]),
    ("m2_u02.csv", [3, 2, 4, 1, 0]),
    ("m2_u025.csv", [2, 1, 3, 0, 4]),
    ("m1_u005.csv", [3, 1, 2, 4, 0]),
    ("m1_u012.csv", [3, 2, 4, 1, 0]),
    ("m1_u018.csv", [2, 1, 3, 0, 4]),
    ("m2_u005.csv", [2, 1, 3, 0, 4]),
    ("m2_u012.csv", [2, 1, 3, 4, 0]),
    ("m2_u018.csv", [1, 0, 2, 3, 4]),
    ("m12_u001.csv", [2, 4, 3, 0, 1]),
    ("m12_u002.csv", [3, 2, 4, 1, 0]),
    ("m12_u001002.csv", [3, 2, 1, 0, 4]),
    ("m12_u0015002.csv", [4, 1, 3, 2, 0]),
    ("m12_u002001.csv", [4, 2, 3, 1, 0]),
    ("m12_u0020015.csv", [3, 1, 2, 0, 4]),
    ("m12_u00100005.csv", [3, 2, 4, 1, 0]),
    ("m12_u00050010.csv", [3, 2, 4, 1, 0]),
]


def build_robot(young=315000.0, poisson=0.45, density=1320.0):
    link1 = LinkSpec.circular(
        constitutive_law=RodConstitutiveLaw(
            transverse_shear=TransverseShearRodModel.UNIFORM,
            torsion=TorsionRodModel.POLAR_MOMENT,
        ),
        length=0.025 + 0.255 + 0.025,
        radius=LinearProfile(base=0.01541, tip=0.00642),
        density=density,
        young_modulus=young,
        shear_modulus=young / (2.0 * (1.0 + poisson)),
        material_damping_coefficient=10000.0,
        reference_strain=[0, 0, 0, 1, 0, 0],
    )
    link2 = LinkSpec.circular(
        constitutive_law=RodConstitutiveLaw(
            transverse_shear=TransverseShearRodModel.UNIFORM,
            torsion=TorsionRodModel.POLAR_MOMENT,
        ),
        length=0.055,
        radius=LinearProfile(base=0.00642, tip=0.0048),
        density=density,
        young_modulus=young,
        shear_modulus=young / (2.0 * (1.0 + poisson)),
        material_damping_coefficient=10000.0,
        reference_strain=[0, 0, 0, 1, 0, 0],
    )
    joint1 = JointSpec(type="fixed")
    joint2 = JointSpec(type="fixed")
    basis1 = StrainBasisSpec(
        type="monomial",
        strain_selector=[1, 1, 1, 1, 0, 0],
        basis_order=[0, 1, 1, 1, 0, 0],
    )
    basis2 = StrainBasisSpec(
        type="monomial",
        strain_selector=[0, 1, 1, 0, 0, 0],
        basis_order=[0, 0, 0, 0, 0, 0],
    )
    num_gauss_points = [8, 8]
    g = [0.0, 0.0, -9.81]
    tendon_angles = jnp.deg2rad(jnp.array([30.0, 150.0]))
    active_tendon_routing = ThreadlikeRouting.linear(
        intercept=0.0114
        * jnp.stack(
            (
                jnp.zeros_like(tendon_angles),
                jnp.cos(tendon_angles),
                jnp.sin(tendon_angles),
            ),
            axis=-1,
        ),
        slope=-0.0295
        * jnp.stack(
            (
                jnp.zeros_like(tendon_angles),
                jnp.cos(tendon_angles),
                jnp.sin(tendon_angles),
            ),
            axis=-1,
        ),
        start_segment_index=0,
        end_segment_index=(0, 0),
    )
    base_pose = jnp.array([0.0, jnp.sqrt(0.5), 0.0, jnp.sqrt(0.5), 0.0, 0.0, 0.0])
    robot = GVS.from_segments(
        [
            GVSSegment(
                link=link1,
                joint=joint1,
                basis=basis1,
                num_gauss_points=num_gauss_points[0],
            ),
            GVSSegment(
                link=link2,
                joint=joint2,
                basis=basis2,
                num_gauss_points=num_gauss_points[1],
            ),
        ],
        gravity=jnp.asarray(g),
        base_pose=base_pose,
        actuators=ThreadlikeActuator.tendons(active_tendon_routing),
        scale_rotational_basis_by_length=True,
    )
    return robot


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def marker_rmse(errors):
    """RMS Euclidean marker error; do not average norms or divide by XYZ."""
    errors = np.asarray(errors)
    if errors.shape[-2:] != (4, 3) or not np.isfinite(errors).all():
        raise ValueError("Expected finite errors with trailing shape (4, 3)")
    return np.sqrt(np.mean(np.sum(errors**2, axis=-1)))


def measured_from_csv(data):
    rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    offset = np.array([-0.03, -0.013, 0.0])
    measurements = []
    for filename, order in CSV_MARKERS:
        p = (
            pd.read_csv(data / filename, header=None)
            .iloc[:, 2:]
            .mean()
            .to_numpy()
            .reshape(5, 3)
        )
        measurements.append((p[order[1:]] - p[order[0]] - offset) @ rotation.T)
    return np.array(measurements)


def representative_indices(errors):
    """One median-error case per tendon-input family, selected before residuals."""
    rms = np.sqrt(np.mean(np.sum(errors**2, axis=-1), axis=1))
    families = ([0, 1, 2, 3, 8, 9, 10], [4, 5, 6, 7, 11, 12, 13], list(range(14, 22)))
    return [
        int(ids[np.argmin(abs(rms[ids] - np.median(rms[ids])))]) for ids in families
    ]


def evaluate(data, output, force=False):
    targets = [
        output / "identification_evaluation.npz",
        output / "identification_audit.json",
    ]
    if not force and any(p.exists() for p in targets):
        raise FileExistsError("Evaluation exists; pass --force")
    measured = measured_from_csv(data)
    np.testing.assert_allclose(measured, np.load(data / "measured_pts.npy"), atol=1e-14)
    u = np.asarray(LOAD_PAIRS) / 0.0325
    model = TauNN(2, 9, width=512, depth=4, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(data / "tau_nn_model.eqx", model)
    tau_nn = np.asarray(jax.vmap(model)(checkpoint_features(jnp.asarray(u))))
    tau_star = np.concatenate(
        [
            np.load(data / f"tau_star_batch_{part}.npy")
            for part in ["1actpt1", "1actpt2", "2act"]
        ],
        axis=1,
    ).T
    results, convergence = {}, {}
    s = np.linspace(0.0, 0.36, 161)
    rounded = build_robot()
    fk = jax.jit(jax.vmap(lambda q: rounded.forward_kinematics_abscissa_batched(q, s)))
    marker = jax.jit(jax.vmap(lambda q: markers_from_q(rounded, q).reshape(4, 3)))
    for stage in (*STAGES, "exact_identified"):
        robot = (
            build_robot(505000.0, 0.45, 1500.0)
            if stage == "initial"
            else build_robot(
                *(
                    float(np.load(data / f"{name}_hat.npy"))
                    for name in ["E", "nu", "rho"]
                )
            )
            if stage == "exact_identified"
            else rounded
        )
        taus = (
            tau_nn
            if stage == "learned"
            else tau_star
            if stage == "fitted_residual"
            else np.zeros((22, 9))
        )

        def equilibrium(q, args):
            ui, tau = args
            return (
                robot.stiffness_matrix() @ q
                + robot.gravitational_force(q)
                - robot.actuation_matrix(q) @ ui
                - tau
            )

        @jax.jit
        def solve(ui, tau):
            solution = optx.root_find(
                equilibrium,
                optx.Newton(rtol=1e-6, atol=1e-6),
                jnp.zeros(9),
                args=(ui, tau),
                max_steps=200,
            )
            return solution.value, equilibrium(solution.value, (ui, tau))

        solved = [solve(ui, tau) for ui, tau in zip(u, taus, strict=True)]
        q = np.asarray([a[0] for a in solved])
        residual = np.asarray([a[1] for a in solved])
        if not np.isfinite(q).all() or np.max(abs(residual)) > 1e-5:
            raise ValueError(f"Equilibrium did not converge: {stage}")
        markers = np.asarray(marker(q))
        errors = markers - measured
        results[stage] = dict(
            q=q, markers=markers, errors=errors, poses=np.asarray(fk(q))
        )
        convergence[stage] = float(np.max(abs(residual)))
        print(stage, "RMSE [mm]", 1000 * marker_rmse(errors), flush=True)
    # Compare against independent archived errors, curves, and NN error norms.
    checks = {}

    def check(name, actual, expected, atol=5e-7):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-6)
        checks[name] = float(np.max(abs(actual - expected)))

    check(
        "initial_eight_errors",
        results["initial"]["errors"][:8].reshape(8, 12),
        np.load(data / "errors_before.npy"),
    )
    check(
        "exact_fitted_eight_errors",
        results["exact_identified"]["errors"][:8].reshape(8, 12),
        np.load(data / "errors_after.npy"),
    )
    check(
        "rounded_identified_markers",
        results["identified"]["markers"],
        np.load(data / "markers_orig.npy"),
    )
    check(
        "fitted_residual_markers",
        results["fitted_residual"]["markers"],
        np.load(data / "markers_tau_star.npy"),
    )
    df = pd.read_csv(data / "violin_data.csv")
    for stage, condition in [
        ("identified", r"$\tau_{ext} = 0$"),
        ("learned", r"$\tau_{ext} = \tau_{ext,NN}$"),
        ("fitted_residual", r"$\tau_{ext} = \tau_{ext}^*$"),
    ]:
        archived = (
            df.loc[df["Tau Condition"] == condition, "Error"]
            .to_numpy()
            .reshape(4, 22)
            .T
        )
        check(
            stage + "_archived_error_norms",
            np.linalg.norm(results[stage]["errors"], axis=-1),
            archived,
        )
    check(
        "misnamed_old_rmse_is_mean_norm",
        np.load(data / "rmse_overall.npy"),
        np.array(
            [
                np.linalg.norm(
                    results[k]["errors"][:8]
                    if k == "initial"
                    else results[k]["errors"],
                    axis=-1,
                ).mean()
                for k in ["initial", "identified", "fitted_residual", "learned"]
            ]
        ),
    )
    chosen = representative_indices(results["identified"]["errors"])
    train = np.setdiff1d(np.arange(22), DECLARED_TEST_INDICES)
    summary = {
        stage: {
            name: float(1000 * marker_rmse(d["errors"][ids]))
            for name, ids in [
                ("all_22", np.arange(22)),
                ("parameter_fit_8", np.arange(8)),
                ("declared_nn_train_20", train),
                ("declared_nn_test_2", DECLARED_TEST_INDICES),
            ]
        }
        for stage, d in results.items()
    }
    audit = dict(
        schema_version=1,
        stages=list(STAGES),
        metric="sqrt(mean over configurations and markers of squared Euclidean position error)",
        units="mm",
        configurations=22,
        markers_per_configuration=4,
        num_segments=2,
        num_dofs=9,
        first_link_basis="constant torsion, linear bending in two axes, linear axial strain (7 DOF)",
        second_link_basis="constant bending in two axes (2 DOF)",
        exact_identified_parameters=[
            float(np.load(data / f"{n}_hat.npy")) for n in ["E", "nu", "rho"]
        ],
        residual_model_parameters=[315000.0, 0.45, 1320.0],
        checkpoint_input_sign=CHECKPOINT_INPUT_SIGN,
        checkpoint_split_provenance="20/2 indices declared by source; checkpoint contains weights but no training metadata",
        declared_nn_test_indices_zero_based=DECLARED_TEST_INDICES,
        generalization_scope="All-22 summary includes fitted and NN-training configurations; not a held-out score. First declared NN test case also belongs to the physical parameter fit set.",
        selected_indices_zero_based=chosen,
        selection_rule="nearest median identified-model per-configuration RMS error within each of tendon 1 only, tendon 2 only, and both tendons; first index breaks ties",
        selected_csvs=[CSV_MARKERS[i][0] for i in chosen],
        rmse_mm=summary,
        convergence_max_abs=convergence,
        archive_checks_max_abs=checks,
        source_sha256={
            p.name: sha256(p)
            for p in sorted(data.iterdir())
            if p.suffix in {".csv", ".npy", ".eqx"}
        },
        evaluator_sha256=sha256(Path(__file__)),
    )
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{stage}_{name}": value
        for stage, d in results.items()
        for name, value in d.items()
    }
    np.savez_compressed(
        targets[0],
        **arrays,
        measured=measured,
        u=u,
        tau_learned=tau_nn,
        tau_fitted=tau_star,
        s=s,
    )
    audit["evaluation_sha256"] = sha256(targets[0])
    targets[1].write_text(json.dumps(audit, indent=2) + "\n")
    print("Representative inputs:", audit["selected_csvs"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument(
        "--output-dir", type=Path, default=CASE / "outputs/identification"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    evaluate(args.data_dir, args.output_dir, args.force)
