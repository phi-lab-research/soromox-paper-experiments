# Paper Results

These experiments use the [SoRoMoX library](https://github.com/phi-lab-research/soromox).
Start with the [repository setup and provenance](../README.md).

This directory maps sections of the
[SoRoMoX preprint](https://arxiv.org/abs/2608.06650) to the code, data, and
canonical outputs used to produce the reported results. Each section keeps
scripts under `code/`, source or generated artifacts under `data/`, and
publication figures/videos under `outputs/`.

| Paper section | Directory | Reproduced artifacts |
| --- | --- | --- |
| IVa, sequential CPU benchmarks | [`secIVa_benchmarking_sequential_cpu/`](secIVa_benchmarking_sequential_cpu/README.md) | SoRoMoX, SoRoSim, and PyElastica rollout data and comparisons |
| IVb, parallel GPU rollouts | [`secIVb_parallel_rollouts_gpu/`](secIVb_parallel_rollouts_gpu/README.md) | GPU timing tables and batch-scaling figures |
| Va, system identification | [`secVa_system_identification/`](secVa_system_identification/README.md) | parameter/residual data and shape, error, and RMSE figures |
| Vc, model-based control | [`secVc_model_based_control/`](secVc_model_based_control/README.md) | standardized configuration- and operational-space figures |
| Vd, control-gain optimization | [`secVd_control_gain_optimization/`](secVd_control_gain_optimization/README.md) | collocated/synergistic optimization data and comparison |
| Ve, safety-constrained control | `secVe_safety_constrained_control/` | rollout data, plots, and videos |
| Vf, sampling-based MPC | [`secVf_sampling_based_mpc/`](secVf_sampling_based_mpc/README.md) | iCEM/MPPI push runs, sample-cloud renders, and summary figure |
| Vg, parallel reinforcement learning | [`secVg_parallel_rl/`](secVg_parallel_rl/README.md) | training, checkpoints, reward curve, rollout data, and media |

Section Vf has its own uv project with the Newton stack; its commands run from
that directory with `uv run --project .` instead of the `paper_results` extra.

`final_outputs/` contains composed figures assembled from one or more section
outputs. Trained model checkpoints are stored with Git LFS; section data and
paper figures remain regular Git objects unless their size requires otherwise.

The [rendering overview](rendering/README.md) compares the four core backends
and all six visual presets. Its captures, generator, and LaTeX inclusion snippet
live under `rendering/`; the composed PDF, SVG, and PNG are saved as
`final_outputs/renderers_and_presets.*`.

The main control/tuning figure combines the two-segment pose-tracking example
with a matched initial/optimized comparison for the single-segment,
three-tendon gain-tuning robot:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/code/plot_control_tuning_summary.py --force
```

It writes `final_outputs/control_and_gain_optimization.*`. The
[Section Vc README](secVc_model_based_control/README.md) records the verified
model and run selection, geometry export, shared studio rendering, and motion
comparison. Full configuration-space, pose-component, and gain-history figures
are retained for the appendix. The legacy `plot_model_based_control.py` entry
point remains available for reproducing the earlier Section Vc composite.

The safety and parallel-RL studies now have separate main figures:

```bash
uv run --extra paper_results python paper_results/secVe_safety_constrained_control/code/plot_safety_summary.py --force
uv run --extra paper_results python paper_results/secVg_parallel_rl/code/plot_rl_summary.py --force
```

These commands compose the committed matched studio snapshots and saved
numerical evidence into `final_outputs/safety_constrained_control.*` and
`final_outputs/parallel_reinforcement_learning.*`, including source manifests.
See the case READMEs for geometry export and RTX snapshot regeneration.
The RL figure combines a close grid view, a recorded single-robot tracking
trajectory, and the training comparison.

`assemble_safety_rl_figure.py` remains available for reproducing the legacy
combined safety/RL figure. Its `--rl-only` mode writes a separate
`parallel_reinforcement_learning_legacy.*` montage.

## PyElastica benchmark dependency

The Section IVa PyElastica benchmarks also require the pinned COOMM fork, which
is installed from the repository's `paper-benchmarks` dependency group:

```bash
uv sync --extra paper_results --group paper-benchmarks
```

Use the repository lockfile for a reproducible environment. The COOMM fork is
installed separately through the `paper-benchmarks` dependency group.

## Shared conventions

The standardized Matplotlib plotters use the base publication settings in
[`paper.mplstyle`](paper.mplstyle) and the semantic color palette in
[`paper_style.py`](paper_style.py). Plotters apply the style file first, then
keep figure-specific font sizes, DPI, padding, and layout settings next to the
figure code. This centralizes typography, colors, grids, legends, spines, and
line defaults without forcing differently sized paper panels to use one font
scale.

The shared [rendering style](RENDERING_STYLE.md) defines robot, target, obstacle,
and object materials, studio lighting, and framing conventions through
`paper_render_style.py`. Manipulation, safety, parallel RL, and control/tuning use the shared studio.
The manipulation figure combines a complete-system overview, planner comparisons,
and sampling snapshots; its separate diagnostic figure is excluded from the paper.

Artifact paths remain case-local. Entry points derive their case directory from
the script location and default to that case's `data/` and `outputs/`
directories, so commands work from any current working directory. Shared schemas
and filename rules within a case live in its common module.
