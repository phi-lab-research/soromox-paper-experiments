# Shared paper rendering style

`paper_render_style.py` defines `PAPER_STUDIO`, the shared material palette and
studio-lighting conventions. The manipulation renderer is the first consumer.
Other application renderers can adopt these settings as their figures are
revised; their existing outputs have not been restyled in this change.

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
paths and the planner tradeoff stacked with aligned left edges in the upper
right, and three matched
iCEM sampling views across the full bottom row. Two matched
contact details and the fully annotated planner sweep appear in the appendix. Robot poses come from the saved iCEM seed-0 trajectory; the
Franka is a display model following the prescribed base pose, not a simulated
dynamic component of the experiment.

The renderer's `overview` and `detail` camera presets are fixed across saved
states. `legacy` retains the previous camera for existing uses such as sample
cloud views. Sampling predictions use translucent blue as an overlay; the
physical robot and lowest-cost prediction stay coral, and their objects stay
gold. Candidate overlays share the studio, camera, and roughness across iterations. `--camera-eye` and `--camera-target` can override a preset.
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

The trajectory legend sits beside the square plot; the tradeoff legend identifies
all three marker families in unused plot space. Sampling views carry iteration
identifiers, with candidate counts and quantitative differences in the caption.
