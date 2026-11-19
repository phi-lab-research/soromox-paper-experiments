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
- Keep legends and callouts outside the robot and contact regions.
- Use the shared `paper.mplstyle` for plots and check the composed PDF at its
  intended print size. The manipulation figure is 180 by 140 mm.

## Manipulation reference

The manipulation figure uses one complete Franka–tentacle view and two matched
contact details. Robot poses come from the saved iCEM seed-0 trajectory; the
Franka is a display model following the prescribed base pose, not a simulated
dynamic component of the experiment.

The renderer's `overview` and `detail` camera presets are fixed across saved
states. `legacy` retains the previous camera for existing uses such as sample
cloud diagnostics. `--camera-eye` and `--camera-target` can override a preset.
The main figure includes every supplied image in full; any cropping is done by
the intentionally chosen detail camera, not by image postprocessing.

Snapshot batches use independent static render sessions for each state because
the pinned RTX backend can lose sample accumulation after runtime pose updates.
The initial active-goal marker is authored for the selected state, including
when the capture starts after the first goal transition.

Snapshots record source hashes, selected steps and physical times, camera,
image resolution, sample count, and the full studio settings in `snapshots.json`.
The figure generator verifies trajectory and style provenance before composing
the PDF, SVG, and PNG, then records source hashes and computed statistics in a
figure manifest. Re-render snapshots after changing studio settings.

Reproduction commands and the panel captions are in
[`secVf_sampling_based_mpc/README.md`](secVf_sampling_based_mpc/README.md).
