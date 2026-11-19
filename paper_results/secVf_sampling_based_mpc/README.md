# Section Vf: Sampling-Based MPC

This case runs closed-loop sampling-based MPC, with MPPI via `--mppi` and
iCEM by default, in which a tapered four-tendon soft arm on a kinematically driven
floating base pushes a box or cylinder around a 10 cm square. The arm is
defined in `code/four_tendon_soft_arm.py`, and the planner primitives and push
cost live in `code/smpc.py`. Each replan samples rollouts against the full
simulation, which runs on [newton-cosserat](https://github.com/phi-lab-research/newton-cosserat):
it couples the SoRoMoX Cosserat rod model of the arm to the rigid-body physics
of [Newton](https://github.com/newton-physics/newton), and resolves the
arm-object-table contacts with its SAP contact solver.

## Environment

This case has two uv projects, one for the experiments and one for the
figures. Run every command from this directory:

```bash
cd paper_results/secVf_sampling_based_mpc
uv sync --project env   # experiments
uv sync --project .              # renderers and summary figure
```

Both projects pin `soromox==0.6.0` through `override-dependencies` and use
fixed Newton-Cosserat revisions. Install with `uv sync --locked` to retain
the recorded dependency versions; neither project requires a library checkout.

The experiment project, [`env/pyproject.toml`](env/pyproject.toml),
pins the stack the paper data was generated on:

| Package | Version |
| --- | --- |
| `newton-cosserat` | [`phi-lab-research/newton-cosserat`](https://github.com/phi-lab-research/newton-cosserat) `8ac49ac9bdd5f411b1ed1a277d4979550c52501c` (Newton 1.7) |
| `newton` | `1.7.0` |
| `warp-lang` | `1.18.0` |
| `jax` | `0.11.1` |

The figure project, [`pyproject.toml`](pyproject.toml), stays on Newton 1.6
because Newton 1.7's RTX viewer requires `ovrtx` 0.5, which renders the sample
clouds black:

| Package | Version |
| --- | --- |
| `newton-cosserat` | [`tud-phi/newton-cosserat`](https://github.com/phi-lab-research/newton-cosserat) `05c705592370ad6ff95bc1504face194fcdd3c5e` |
| `newton` | `1.6.0.dev0`, git rev `7c677b95` (adds `log_mesh` opacity for the sample clouds) |
| `ovrtx` | `0.4.1` |
| `warp-lang` | `1.16.0` |

Experiments run on the CPU or a GPU (`--device cpu` or `--device cuda:0`).
Both renderers need an RTX GPU. On a machine without a GPU, set
`JAX_PLATFORMS=cpu`; otherwise the JAX CUDA 13 plugin fails at import.

## Experiments

Run one closed-loop push with `code/run_experiment.py`:

```bash
uv run --project env python code/run_experiment.py --device cuda:0 \
  --stop-after-one-square --object-shape box --k-samples 200 100 50 --num-iters 3 --seed 0 \
  --save-path /tmp/soromox-secVf/square_icem_seed0.npz
```

`--k-samples` takes one value or a per-iteration schedule. `--save-samples`
also stores every candidate rollout, which the sample-cloud renderer needs; the
files are large.

Regenerate the complete paper data set:

```bash
DEVICE=cuda:0 code/generate_paper_results.sh
```

The script runs three parts, each over seeds 0, 1 and 2:

- a 20-step planner comparison toward the single push goal (0.05, -0.10) m,
  written to
  `data/cmp_seeds/seed*/`: an MPPI sample-count sweep (`cmp_mppi_k*.npz`,
  K = 200 to 1600, white noise), an iCEM iteration sweep at K = 200
  (`cmp_icem_k200_it*.npz`, 1 to 4 iterations), and an iCEM sample-count sweep
  at 3 iterations (`cmp_icem_it3_k*.npz`, K = 50 to 400);
- full square pushes with MPPI at K = 800 (`data/square_mppi_k800_white_seed*.npz`)
  and iCEM (`data/square_icem_seed*.npz`); the seed-0 iCEM run is
  saved with `--save-samples`;
- two seed-0 iCEM square-push variants, a 100 g box
  (`data/square_icem_box100g.npz`) and a cylinder
  (`data/square_icem_cylinder.npz`).

The script replaces same-named files in `data/`. Set `RESULTS_DIR` to stage a
run elsewhere. Of the committed runs, only `data/square_icem_seed0.npz` keeps
the per-sample arrays; it is stored in Git LFS (`git lfs pull` fetches it).

## Key numbers

The tables below summarize the committed runs. Recompute them with:

```bash
uv run --project . python code/summarize_runs.py
```

A run succeeds when it reaches every goal (all four square corners, or the
single comparison goal) under the online reach test of `run_experiment.py`:
the object passes within 10 % of the leg length of the goal. The final error is
the distance from the object's last (x, y) position to the last active goal.
The time per MPC step is the wall-clock planning time of one replan, rollouts
included, averaged over steps 1 onward to skip the JIT warm-up of step 0.
Values are mean ± standard deviation over seeds 0, 1 and 2; the 100 g box and
cylinder runs use seed 0 only. Square pushes stop after the last corner or
after 100 MPC steps.

Square push:

| Configuration | Success | Final error [mm] | Time per MPC step [s] | MPC steps |
| --- | --- | --- | --- | --- |
| MPPI K = 800, box | 1/3 | 15.9 ± 9.3 | 8.50 ± 0.84 | 85 |
| iCEM, box | 3/3 | 8.0 ± 1.3 | 26.54 ± 0.20 | 51 |
| iCEM, 100 g box | 0/1 | 10.1 | 29.42 | 100 |
| iCEM, cylinder | 1/1 | 5.6 | 25.58 | 53 |

The failed MPPI seeds 1 and 2 reach only the first corner; the 100 g box
reaches three of four corners within 100 steps.

Planner comparison (20 steps toward the single goal):

| Configuration              | Success | Final error [mm] | Time per MPC step [s] |
| -------------------------- | ------- | ---------------- | --------------------- |
| MPPI K = 200               | 2/3     | 11.8 ± 7.6       | 8.39 ± 0.75           |
| MPPI K = 400               | 1/3     | 13.0 ± 4.9       | 8.72 ± 1.11           |
| MPPI K = 800               | 3/3     | 5.9 ± 0.8        | 12.93 ± 1.48          |
| MPPI K = 1600              | 3/3     | 3.6 ± 0.6        | 29.53 ± 0.70          |
| iCEM K = 200, 1 iteration  | 2/3     | 34.4 ± 46.4      | 8.31 ± 0.49           |
| iCEM K = 200, 2 iterations | 3/3     | 6.5 ± 1.6        | 19.32 ± 0.64          |
| iCEM K = 200, 3 iterations | 3/3     | 8.0 ± 1.5        | 31.62 ± 0.79          |
| iCEM K = 200, 4 iterations | 3/3     | 4.1 ± 0.4        | 44.82 ± 0.21          |
| iCEM 3 iterations, K = 50  | 3/3     | 4.2 ± 1.2        | 19.02 ± 0.55          |
| iCEM 3 iterations, K = 100 | 3/3     | 5.9 ± 1.2        | 23.07 ± 0.70          |
| iCEM 3 iterations, K = 200 | 3/3     | 8.0 ± 1.5        | 30.86 ± 0.58          |
| iCEM 3 iterations, K = 400 | 3/3     | 5.8 ± 2.4        | 41.05 ± 1.54          |

## Paper artifacts

The main figure gives the complete robot the largest view: **A**, a full
Franka–tentacle scene; **B**, object paths for all three seeds of each planner;
**C**, three iCEM sampling iterations across the full bottom row; **D**, the
planner tradeoff stacked beneath the centered trajectory plot in the upper
right. The trajectory legend sits above its plot; the tradeoff
legend identifies all three marker families inside unused plot space. The
diagnostic figure is excluded from the paper; no additional appendix material
is planned for this figure. Panel letters,
axis labels, and legends remain in the figures; explanations, seed counts,
marker meanings, success rates, and timing statistics belong in the captions.
See [the shared rendering style](../RENDERING_STYLE.md) and
[`figure_caption.tex`](figure_caption.tex) for the proposed main-figure caption.

### Main figure

Render only the required saved states on an RTX host, without rerunning the
experiment or encoding a video. Run from this section directory with the pinned
figure environment installed. Both JAX and Newton must use the GPU; do not set
`JAX_PLATFORMS=cpu` for rendering.

```bash
uv run --project . python code/rendering.py \
  --run-path data/square_icem_seed0.npz --view overview --snapshot-steps 19 \
  --future-steps 21 23 --future-opacity 0.3 \
  --width 2400 --height 1800 --samples-per-pixel 128

uv run --project . python code/rendering.py \
  --run-path data/square_icem_seed0.npz --view detail --snapshot-steps 12 19 \
  --width 1600 --height 1000 --samples-per-pixel 128
```

Images and provenance are written under `outputs/paper_snapshots/{overview,detail}/`.
Pass `--force` to replace existing snapshots. Steps are saved trajectory row
indices, including the initial row; physical time is `step * --control-dt`.
Step 12 is the first transition to the second corner goal, and step 19 is partway
along that push. The overview uses opaque step 19 (0.95 s), with faint steps
21 and 23 (1.05 and 1.15 s). The renderer verifies that all intervening states
keep the same active goal and that the selected cube poses approach it
(83.2 → 41.0 → 14.8 mm). The faint tentacle and cube use recorded configurations;
the matching Franka poses use the existing kinematic display model. They show
later executed motion, not predictions from the current MPC step. The active
goal stays fixed. Omit `--future-steps` for a single-state overview. These times are fixed across the
figure and are recorded in the manifests. Batches use a fresh render process
for each snapshot to preserve path-tracing accumulation in the pinned RTX
backend; the batch manifest is published after all selected states render.

Render the three sampling iterations with the same studio (from the run saved
with `--save-samples`). The shared camera is fixed across the three frames:

```bash
OUT_DIR=outputs/paper_snapshots/sampling code/render_sample_cloud.sh data/square_icem_seed0.npz 10 --force
```

Each PNG has a JSON sidecar with source and style provenance. The current
recorded tentacle and cube use the overview's native geometry, coral/gold
materials, and soft shadows. Their poses are identical across iterations,
including the cube's full recorded position and quaternion. The viewing
direction matches the overview. The lowest-cost predicted tentacle is faint
coral (opacity 0.3). Translucent blue tentacles show all other candidates, with
opacity decreasing from 0.12 to 0.012 by cost rank. Prediction overlays cast no
shadows. Predicted objects are omitted by default (`--max-object-samples 0`),
so only the current recorded cube is visible.

Export the planner sweep scores once with the section environment (the scored
JSON is also committed, so composing the main figure needs no JAX):

```bash
JAX_PLATFORMS=cpu uv run --project . python code/plot_smpc_diagnostics.py
```

This writes `outputs/planner_tradeoff.json`, including source hashes and all
12 plotted configurations (the two iCEM sweep families share one configuration).

Compose the figure from these snapshots and the saved square-push and sweep results:

```bash
uv run --project . python code/plot_smpc_summary.py --force
```

This writes `../final_outputs/sampling_based_mpc.{pdf,svg,png,json}`. The generator
needs only NumPy and Matplotlib and can also run on a CPU/macOS environment with
those packages, without installing the RTX dependencies. It verifies snapshot
trajectory and studio settings before composing images. A fixed overview crop
(normalized x: 0.25–0.84, y: 0.10–0.90) removes empty background around the
complete robot without stretching it. The trajectory plot uses equal axis
scaling; both plots are stacked closely in the upper right. The sampling views
use one fixed crop (normalized x: 0.05–0.95, y: 0.04–0.88), recorded in the figure
manifest; the original PNGs remain intact. The generator verifies the scored
sweep source hashes and computes caption statistics directly from the runs using the same reach test as
`code/summarize_runs.py`. Timing excludes the first MPC step and reports mean
and population standard deviation across the three seed means. `--data-dir`,
`--snapshot-dir`, `--tradeoff-data`, and `--output-dir` allow staged builds. Existing figure outputs
require `--force` before replacement.

### Internal diagnostics

```bash
uv run --project . python code/plot_smpc_diagnostics.py
```

This writes `outputs/sampling_based_mpc_diagnostics.pdf`, containing the
two contact-detail frames and the complete single-goal planner cost/time sweep.
It is an internal analysis artifact, excluded from the paper and its appendix.
The sweep requires the SoRoMoX/JAX dependencies for rescoring the saved runs;
it does not rerun planning. On a CPU-only host, set `JAX_PLATFORMS=cpu`.

### Videos and camera overrides

```bash
uv run --project . python code/rendering.py --run-path data/square_icem_cylinder.npz \
  --object-shape cylinder --view overview --output outputs/square_icem_cylinder.mp4 --gif
```

The renderer reads saved trajectories and adds a decorative Franka FR3 whose
end effector follows the tentacle base. `--franka-base` changes its placement.
The default `overview` camera includes the complete system; `detail` focuses on
the contact region, and `legacy` retains the previous camera. Explicit
`--camera-eye X Y Z` and `--camera-target X Y Z` override either preset.
All views use the shared studio materials and lighting.

Numbered video frames are kept under `outputs/<stem>_frames/frames/`. The
committed legacy videos remain available and are stored with Git LFS. Encoding
MP4/GIF requires ffmpeg; snapshot-only rendering does not. Sample-cloud scenes
use translucent blue candidate overlays with the shared coral/gold scene materials.

### Interpreting the sampling views

These are successive iterations of the first MPC planning step, evaluated at
prediction step 10. They are distinct renders of different candidate sets:

| Iteration | Candidates | Best whole-rollout cost | Selected object x, y at prediction step 10 [mm] |
| --- | ---: | ---: | --- |
| 1 | 200 | 1943.30 | 45.35, −5.98 |
| 2 | 100 | 1867.62 | 49.91, −0.11 |
| 3 | 50 | 1781.56 | 49.23, −1.12 |

The solid current tentacle and cube are intentionally identical in all three
views. The candidate sets and faint selected tentacle predictions differ, and
the best cost across the complete rollout improves by 4.61% in the last
iteration. That cost change is not an error metric for the single frame.
The predicted object positions in the table are diagnostics, omitted from the
renderings; the last two differ by 1.22 mm at this prediction step.
The source is `diagnostics_iter_X_samples[0]` and `diagnostics_iter_cost[0]` in
`data/square_icem_seed0.npz`; the renderer selects each iteration's minimum-cost
candidate. The iteration identifiers in the figure distinguish the views, while
the explanation and candidate counts remain in the caption.
