# Shared paper rendering style

`paper_render_style.py` defines `PAPER_STUDIO`, the shared material palette and
studio-lighting conventions. Manipulation, safety, parallel RL, and control/tuning share the same
`paper_rtx_studio.py` lighting and material implementation. Other application
renderers can adopt these settings as their figures are revised.

| Role | sRGB color |
| --- | --- |
| Soft robot | Coral `#F1552E` |
| Mount | Dark neutral `#33363B` |
| Target and desired spatial path | Green `#0EAD69` |
| Obstacle | Purple `#7B2CBF` |
| Manipulated object | Gold `#E7AC42` |
| Support surface | Light neutral `#DEE1E5` |
| Backdrop | Light neutral `#C9CDD2` |

Keep robot materials fixed across controller comparisons. Distinguish methods
through labels and plot traces, using the existing `paper_style.py` palette.
Always list MPPI before iCEM in legends, tables, and comparative caption text.
Preserve platform geometry and mounting, including the Franka's original rigid
materials. Robot identity should come from geometry rather than renderer style.

All palette values are sRGB. Newton RTX material inputs require linear RGB;
use `rgb(color, linear=True)` for them. Other backends must use their own color
conventions. The studio records light direction, softness, surface roughness, and backdrop
geometry. Use a sufficiently wide and tall backdrop for oblique camera views
so its boundary does not appear in the image. Intensity fields prefixed `rtx_` are calibrated for Newton RTX and
must not be copied numerically into renderers with different lighting units.

## Composition

- Give the complete system the largest view, including its base and task.
- Use a fixed camera within comparisons and leave margins around the motion.
- Put close-ups in explicitly identified detail panels; never automatically
  crop the overview based only on saturated pixels, which excludes gray hardware.
- Do not stretch images to fill slots. Preserve their native aspect ratio.
- Keep panel text to letters, axis labels with units, legends, and identifiers
  needed to read the data. Put explanations, sample counts, start/goal semantics,
  timing statistics, and other prose in the actual figure caption.
- Use panel letters without generic titles. Add a short title only when it
  provides information that the visual and caption cannot convey clearly.
- Leave about 3 mm between panel letters and the top of the rendered content
  at final print size. Reserve additional room for plot legends and axis labels.
- Keep identical physical units at equal axis scaling in spatial plots.
  Change the panel dimensions or surrounding layout to fit them.
- Give each plot a legend for its own encodings. A legend can occupy unused
  plot space when it does not cover data; keep legends outside robot and
  contact regions.
- Use the shared `paper.mplstyle` for plots and check the composed PDF at its
  intended print size. The manipulation figure is 180 by 142 mm.

## Manipulation reference

The main manipulation figure uses one complete Franka–tentacle view, object
paths centered above the wider planner tradeoff in the upper right, and three matched
iCEM sampling views across the full bottom row. The separate contact-detail and
annotated-sweep diagnostic figure is excluded from the paper, with no additional
appendix material planned. Robot poses come from the saved iCEM seed-0 trajectory; the
Franka is a display model following the prescribed base pose, not a simulated
dynamic component of the experiment.

The renderer's `overview` and `detail` camera presets are fixed across saved
states. `legacy` retains the previous camera for existing uses. Sampling views
use the overview's viewing direction with a closer camera, fixed across iterations.
Solid rendering always denotes the current recorded configuration, using the
same native tentacle capsules, gold object, materials, and soft shadows as the
overview. The current tentacle and full cube pose are identical across iterations.
The lowest-cost predicted tentacle is faint coral (opacity 0.3); all other
predicted tentacles are translucent blue, with opacity decreasing from 0.12 to
0.012 by cost rank. Prediction overlays cast no shadows. Predicted cubes are
omitted to keep the current gold cube clearly visible.
`--camera-eye` and `--camera-target` can override a snapshot preset.
The overview uses an explicit fixed crop, keeping normalized x in [0.25, 0.84]
and y in [0.10, 0.90]. It removes empty background while preserving the entire
robot, base, object, and target with margins. The three sampling views share a
fixed crop with normalized x in [0.05, 0.95] and y in [0.04, 0.88], retaining
the entire candidate cloud and target. Both crops are recorded in the figure
manifest; original renders are retained unchanged.

Snapshot batches use independent static render sessions for each state because
the pinned RTX backend can lose sample accumulation after runtime pose updates.
The initial active-goal marker is authored for the selected state, including
when the capture starts after the first goal transition.

Snapshots record source hashes, selected steps and physical times, camera,
image resolution, sample count, and the full studio settings in `snapshots.json`.
The figure generator verifies trajectory and style provenance before composing
the PDF, SVG, and PNG, then records source hashes and computed statistics in a
figure manifest. Sample-cloud PNGs have matching JSON sidecars recording the
planning step, iteration, horizon step, candidate counts, opacity, source hash,
camera, and studio settings. Re-render snapshots after changing studio settings.

Reproduction commands and the panel captions are in
[`secVf_sampling_based_mpc/README.md`](secVf_sampling_based_mpc/README.md).

The trajectory legend sits above the centered square plot; the tradeoff legend identifies
all three marker families in unused plot space. Sampling views carry iteration
identifiers, with candidate counts and quantitative differences in the caption.

Use one opaque current state and at most two faint later states to illustrate
progress toward the currently highlighted goal. Select states before any goal
change, require decreasing cube-to-goal distance, and show the matching rigid
robot, soft robot, and object poses together. Keep the active goal unchanged.
Identify the faint states as later recorded motion in the caption, distinct from
the sampled MPC predictions in panel C. Fixed robot links need no duplicate
geometry. Exclude ghost geometry from shadows and record times, target, distances,
and rigid-arm transforms in the snapshot manifest.

## Safety reference

Use two matched views of the final recorded state, with the nominal HOCLF
controller first and HOCLF + HOCBF second. Both robots are coral with the same
dark mount. Show the complete system and all three purple obstacles. A small
green ring marks the exact projected target position. A white dot with a dark
border marks the terminal backbone point. Both markers remain visible through
the robot. Use a compact marker
legend; these are graphical overlays at the true camera projections.
The paired scenes dominate the figure. Separate force and goal-distance plots
below them share their time range and method encoding. Keep each legend clear
of data. The force quantity is maximum pairwise normal contact force, with
the prescribed 5 N reference retained in the main plot. HOCLF uses blue
(`pre_opt_1`) and HOCLF + HOCBF coral (`post_opt_1`). Both recorded controller
traces are solid; the force-limit reference is dotted.

Read all plotted values from the saved rollouts. Before rendering, verify
forward-kinematic goal distance and discretized contact force against the
stored metrics. The visible tip must include the terminal contact sphere's
forward hemisphere: a flat face omits the geometry responsible for contact
in this case. Check rendered mesh penetration against the saved force and
penalty stiffness, and identify this contact geometry in the caption.
Apply one rigid display transform to robot, obstacles, and
target together. Frame provenance includes the full transform, data hashes,
camera, common timestamp, and studio. No supplementary diagnostic figure is
required for this comparison.

## Parallel RL reference

Use one low, close perspective view whose grid extends beyond the image
boundaries. The distant scene uses repeated display instances of the 64 saved
environments; record this explicitly in metadata and the caption. Extend the
studio floor beyond the visible population while retaining the shared lighting
and materials. Spatial offsets move each robot and its target together. Reduce
mesh tessellation only for distant display instances.

Pair the population view with a complete single-robot tracking example. Show
the target path in green dashes and the recorded end-effector path in solid
blue, using the full saved rollout, including the initial approach. The solid
robot and green sphere show one actual instant. Path graphics cast no shadows.
Explain the temporal extent and the selected environment in the caption.

Keep the learning-curve data and processing unchanged. CPU baselines precede
GPU batch sizes in ascending order; CPU lines are dashed and GPU lines solid.
The grid and tracking detail depict execution of the trained policy, while the
learning curves show training progress. Display repetition is not an additional
training run or a measured environment count.

## Control and gain-tuning reference

Use one substantial pose-tracking scene beside matched initial/optimized gain
views, with complete error histories below them. Preserve the distinct physical
models and mounting: two segments upright for pose tracking; one segment with
three tendons hanging for tuning. Apply any display rotation to the body,
target, and recorded motion together. Fit the gain camera to both full motions
and use it unchanged for both runs and the animation.

The body and mount retain the shared coral/dark materials. A green dashed path
shows the pose reference; a green ring, normal, and radial spoke encode its
instantaneous target pose. Path/pose graphics cast no shadows. The tuning
comparison uses the safety figure's projected white tip dot and green target
ring. Use blue (`pre_opt_1`) for initial gains and coral (`post_opt_1`) for
optimized gains in labels and solid error traces, with the initial case first.
Show the initial min--max range as a pale blue band, and label the solid initial
trajectory as selected and the optimized trajectory as best. Define the range
and run-selection rules in the caption.

Render actual recorded runs, never a pointwise median configuration assembled
from multiple runs. State the run-selection rule and whether the two cases
come from different optimization starts in the caption. Keep timing, sample
counts, and reported improvements there as well. The tracking panel includes three faint recorded poses to show its motion
range. State their times in the caption, keep one solid state facing the reader,
and suppress ghost shadows and near-coincident proximal geometry. Keep the
solid pose in front of the faint shapes; record silhouette occlusion in metadata. The tuning
still comparison uses matched physical times; the full motion clip uses the same selected runs.

Retain original renders. The 180 by 118.5 mm composite applies the same fixed
vertical crop (pixels 0--1250 of 1400) to all three scenes, preserving robot,
mount, task graphics, and space for panel letters. The tuning views additionally
crop horizontal pixels 60--948 to balance the two groups. At full width, use
8.5 pt axis labels/legends and 8 pt ticks, with similarly wide quantitative plots. Before composition, check
source hashes, geometry provenance, shared studio settings, and matched
camera/time/scale for the gain comparison. Reproduction commands and the
model/run audit are in `secVc_model_based_control/README.md`.
