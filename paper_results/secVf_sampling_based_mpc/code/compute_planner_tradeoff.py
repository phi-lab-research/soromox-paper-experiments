#!/usr/bin/env python3
"""Compute the executed-trajectory cost and planning time for the MPC figure."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

CODE_DIR = Path(__file__).resolve().parent
CASE_DIR = CODE_DIR.parent
PAPER_RESULTS_DIR = CASE_DIR.parent
for path in (CODE_DIR, PAPER_RESULTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from four_tendon_soft_arm import FourTendonSoftArm
from smpc import SMPCParams, push_cost

RESULTS_DIR = CASE_DIR / "data"
OUTPUT_DIR = CASE_DIR / "outputs"

# Copies of run_experiment.py's values: importing it would pull in newton_cosserat just to plot.
OBJECT_FOOTPRINT = 0.05
TABLE_TOP_Z = 0.0
CONTACT_SEGMENT_INDEX = 2


@dataclasses.dataclass(frozen=True)
class RunSpec:
    filename: str
    family: str
    label: str


RUNS = [
    RunSpec("cmp_mppi_k200.npz", "mppi", "K=200"),
    RunSpec("cmp_mppi_k400.npz", "mppi", "K=400"),
    RunSpec("cmp_mppi_k800.npz", "mppi", "K=800"),
    RunSpec("cmp_mppi_k1600.npz", "mppi", "K=1600"),
    RunSpec("cmp_icem_it3_k50.npz", "icem_fixed_iters", "K=50"),
    RunSpec("cmp_icem_it3_k100.npz", "icem_fixed_iters", "K=100"),
    RunSpec("cmp_icem_k200_it3.npz", "icem_fixed_iters", "K=200"),
    RunSpec("cmp_icem_it3_k400.npz", "icem_fixed_iters", "K=400"),
    RunSpec("cmp_icem_k200_it1.npz", "icem_fixed_batch", "it=1"),
    RunSpec("cmp_icem_k200_it2.npz", "icem_fixed_batch", "it=2"),
    RunSpec("cmp_icem_k200_it3.npz", "icem_fixed_batch", "it=3"),
    RunSpec("cmp_icem_k200_it4.npz", "icem_fixed_batch", "it=4"),
]


def build_params(robot) -> SMPCParams:
    """SMPCParams for re-scoring the cmp_ runs; score_run fills in each run's goal, contact_z and yaw."""
    return SMPCParams(
        robot=robot,
        control_std=jnp.ones(robot.num_actuators + 3),
        K=1,
        contact_segment_index=CONTACT_SEGMENT_INDEX,
        object_footprint=OBJECT_FOOTPRINT,
        table_z=TABLE_TOP_Z,
    )


score_push_cost = eqx.filter_jit(push_cost)


def score_run(path: Path, params: SMPCParams) -> tuple[float, float]:
    """Return (mean planning time per step [s], -push_cost of the executed trajectory).

    Planning time sums diagnostics_iter_time over iterations (rollouts included) and averages over
    steps 1.., skipping step 0's JIT warm-up. The cost uses the run's goal, starting object height
    and starting yaw, as run_experiment.py sets them.
    """
    with np.load(path, allow_pickle=True) as data:
        iter_time = data["diagnostics_iter_time"].astype(float)
        X_final = data["X_final"]
        U_best = data["U_best"]
        goal_xy = data["goal_xy"][0]
        initial_object_z = data["object_pose"][0, 2]

    yaw_index = 2 * params.robot.num_internal_dofs + 4
    params = dataclasses.replace(
        params,
        push_goal_xy=jnp.asarray(goal_xy),
        contact_z=float(initial_object_z),
        push_goal_yaw=jnp.asarray(X_final[0, yaw_index]),
    )
    cost = float(score_push_cost(params, jnp.asarray(X_final), jnp.asarray(U_best)))
    return float(iter_time[1:].sum(axis=1).mean()), -cost


def load_seed_mean_points(
    results_dirs: list[Path], params: SMPCParams
) -> list[tuple[RunSpec, float, float]]:
    """Score every RUNS configuration in each of results_dirs (one seed per directory)
    and return (spec, mean_time, mean_neg_cost) per configuration.
    """
    scores: dict[
        Path, tuple[float, float]
    ] = {}  # cmp_icem_k200_it3 appears in two families
    points = []
    for spec in RUNS:
        paths = [results_dir / spec.filename for results_dir in results_dirs]
        for path in paths:
            if path not in scores:
                scores[path] = score_run(path, params)
        times, costs = zip(*(scores[path] for path in paths), strict=True)
        points.append((spec, float(np.mean(times)), float(np.mean(costs))))
    return points


def main(argv=None) -> int:
    """Export planner scores for the main figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir / "planner_tradeoff.json"
    if output.exists() and not args.force:
        raise FileExistsError("Planner scores exist; pass --force")
    directories = [args.data_dir / "cmp_seeds" / f"seed{i}" for i in range(3)]
    robot = FourTendonSoftArm().model
    points = load_seed_mean_points(directories, build_params(robot))
    # Export scored data so the main figure remains reproducible without JAX/RTX.
    sources = sorted(
        {directory / spec.filename for directory in directories for spec in RUNS}
    )
    export = {
        "points": [
            {**dataclasses.asdict(spec), "mean_time_s": x, "mean_negative_cost": y}
            for spec, x, y in points
        ],
        "sources": [
            {
                "file": str(path.relative_to(args.data_dir)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sources
        ],
        "definition": "Three-seed mean of per-run planning time (excluding the first step) and negative executed-trajectory push cost.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(export, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
