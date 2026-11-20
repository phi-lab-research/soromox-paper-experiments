# Section Vc: Model-Based Control

Section Vc is reproduced by two independent cases:

- [`configuration_space_comparison/`](configuration_space_comparison/README.md)
  compares setpoint regulation, trajectory tracking, and regulation followed by
  tracking.
- [`operational_space_impedance_control/`](operational_space_impedance_control/README.md)
  evaluates operational-space impedance tracking with partial feedback
  linearization.

Run commands from the repository root after installing paper dependencies:

```bash
uv sync --extra paper_results
```

Each generator refuses to overwrite data unless `--force` is passed. Plotters
and renderers consume saved NPZ data and write only to their case `outputs/`
directory by default. The case READMEs list complete reproduction commands and
artifact inventories.

Reproduce the legacy Section Vc application figure from the canonical comparison
data and operational snapshots with:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/code/plot_model_based_control.py
```

The command writes `outputs/model_based_control.pdf` and a matching SVG; pass
`--force` to replace existing outputs.

## Evaluation metrics

Generate the complete human-readable report and its machine-readable JSON from
the committed trajectory artifacts without rerunning either simulation:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/code/report_metrics.py --json-output
```

The configuration-space report contains component-wise RMSE for all six strain
coordinates in the regulation and tracking phases. For regulation it also
reports the mean absolute error over the final 10% of each post-change setpoint
interval, first per setpoint and then as an equal-weight mean over the four
setpoints. The initial zero-reference interval is excluded. The operational
report contains Euclidean position and geodesic orientation RMSE, the maximum
pairwise reference span for each task quantity, and RMSE normalized by that
span in percent. The command writes `data/metrics.json` by default; an explicit
path may be supplied after `--json-output`.

The committed Section Vc paper figures are:

- `outputs/model_based_control.pdf`
- `configuration_space_comparison/outputs/configuration_space_comparison.pdf`
- `configuration_space_comparison/outputs/regulation_tracking_rmse.pdf`
- `operational_space_impedance_control/outputs/operational_space_impedance_control.pdf`

All use the shared `paper_results/paper.mplstyle` publication settings and the
regenerated Section Vc trajectory data.

## Combined control and gain-tuning main figure

The main paper now combines pose tracking and gain tuning in
`../final_outputs/control_and_gain_optimization.pdf`. The dedicated appendix
composite below contains configuration-space control
and actuation-space tuning. The full componentwise pose histories remain a
separate appendix figure; operational-space tuning is shown in the main figure.

The model/run audit is recorded in [control_tuning_audit.md](control_tuning_audit.md).
The tracking robot has two fully actuated segments. Both gain-tuning cases use
one segment and three tendons. The main comparison uses the synergistic case:
initial start 5 (nearest the median initial loss) and optimized start 2 at
iteration 100, with numbers counted from one. These are different starts.
The pointwise coordinate median from the legacy plot is not rendered as a robot.

Compose the committed studio snapshots and complete recorded error histories:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/code/plot_control_tuning_summary.py --force
```

Outputs are PDF, SVG, PNG, and a JSON manifest in `paper_results/final_outputs/`.
The manifest records source hashes, selected runs/times, camera transforms,
rendering settings, and the fixed vertical crop. The three views keep pixels
0--1250 of their 1400-pixel height; the original PNGs remain intact. The final
figure is 180 by 118.5 mm; the gain views additionally crop pixels 60--948
horizontally. Axis labels and legends are 8.5 pt, ticks 8 pt, and the control
and tuning plots are 69 and 78.5 mm wide, respectively, with RMSEs, times, and selection rules in the
[caption](figure_caption.tex).

To regenerate the static geometry, using the original model builders and saved
states without rerunning simulation or optimization:

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/code/export_control_tuning_scene.py
```

This checks forward kinematics against the stored poses and recomputes the
objectives from all initial/best saved rollouts. `outputs/paper_snapshots/`
contains exported geometry, rendered PNGs, provenance JSON sidecars, and an
audit JSON. Render each geometry on an NVIDIA RTX host using the Section Vf
Newton environment, for example:

```bash
paper_results/secVf_sampling_based_mpc/.venv/bin/python paper_results/secVc_model_based_control/code/render_control_tuning_scene.py --geometry paper_results/secVc_model_based_control/outputs/paper_snapshots/tracking_geometry.npz --output paper_results/secVc_model_based_control/outputs/paper_snapshots/tracking.png --force
```

Repeat for `initial` and `optimized`. Each state uses a fresh renderer, the
shared studio, and 128 samples per pixel. The gain views share the same camera,
physical scale, time (5 s), and hanging mounting. Pose tracking is shown at
6.25 s with its upright mounting and the tip facing the camera; faint shapes
show recorded states at 4.0, 4.75, and 7.25 s. They cast no shadows, and their proximal
15.625 mm are omitted where they overlap the solid robot to avoid transparency
artifacts. Ghost faces embedded in the solid robot are also omitted. The same solid mount serves all three poses. Green graphics indicate desired quantities;
white tip dots and green target rings follow the safety figure's convention.

### Recorded motion comparison

The matching initial/optimized motion is saved under
`../secVd_control_gain_optimization/outputs/control_gain_optimization_comparison`
as MP4, GIF, and provenance JSON. It covers 0--5 s with 61 matched states at
approximately 12 frames per second, selected from the saved 1 ms trajectories.
Both cameras stay fixed. All plotted errors retain the original sample rate.

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/code/export_control_tuning_scene.py --motion --output-dir /tmp/control-tuning-motion/geometry
# Run on the RTX host with the exported geometry and shared studio:
paper_results/secVf_sampling_based_mpc/.venv/bin/python paper_results/secVc_model_based_control/code/render_control_tuning_motion.py --render-only --geometry-dir /tmp/control-tuning-motion/geometry --frame-dir /tmp/control-tuning-motion/frames --workers 2
# After retrieving the frames, compose locally (ffmpeg required):
uv run --extra paper_results python paper_results/secVc_model_based_control/code/render_control_tuning_motion.py --frame-dir /tmp/control-tuning-motion/frames --output-dir paper_results/secVd_control_gain_optimization/outputs
```

The MP4/GIF are tracked by the existing Section Vd Git LFS rules. Intermediate
motion geometry and individual frames are temporary reproduction artifacts.

For clarity, the solid recorded pose is kept in the foreground: ghost triangles
that cross its camera-projected silhouette are omitted. This display-only
occlusion is recorded as `opaque_state_foreground` in each overlay manifest.

The tuning plot includes a pointwise min--max envelope of the Euclidean
position-error norms across all six initial gain sets. The blue line is the
selected actual initial run, and the coral line is the best optimized run.
Labels explicitly identify selected/best runs; the band is an empirical range,
not a confidence interval. The gap between plot axes is reduced to 13.5 mm.

## Configuration-space control and actuation-space tuning appendix

```bash
uv run --extra paper_results python paper_results/secVc_model_based_control/code/plot_control_tuning_appendix.py --force
```

This writes `final_outputs/control_and_gain_optimization_appendix.{pdf,png,svg,json}`.
The 180 by 132 mm figure uses the same 8.5 pt axis labels, 8 pt ticks, semantic
palette, and vector plots as the main figure. The source data are unchanged.

- A retains all five controllers and all 30 s of configuration-space regulation
  and tracking. Physical axial strain is `1 + Delta sigma_x`; the boundary at
  15 s separates regulation from tracking.
- B shows all 100 iterations of actuation-space gain optimization. The curve
  is the minimum objective at each iteration; the band is the range across six
  starts. The loss is displayed in units of 0.001 on a linear axis.
- C shows the first 2 s of the stored 5 s angular-strain responses. The initial
  trajectory is actual start 6, nearest the initial loss median; the optimized
  trajectory is start 2 at iteration 100. Numbers are one-based. Dash-dot and
  solid curves distinguish these two runs; dashed curves mark the targets.
  Coordinate-colored bands show the min--max range across all six initial runs.

The [caption](appendix_figure_caption.tex) distinguishes the two robot/controller
experiments and explains the selections. Both robots have one PCS segment;
configuration-space control is fully actuated, while the tuning example has
three tendons and collocated potential-shaping control. The JSON manifest records
source hashes, intervals, selection rules, and loss scaling. The loader checks
common reference/time arrays and recomputes each saved tuning objective.

This composite replaces the separate configuration-space and gain-optimization
figures in the appendix. Legacy generators and their outputs remain available.
