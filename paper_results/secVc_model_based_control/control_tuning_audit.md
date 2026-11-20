# Control and gain-tuning figure audit

## Models and data

The pose-tracking example builds **two PCS segments**, each 0.1 m long and
0.02 m in radius, with 12 generalized coordinates and full actuation. Its
saved archive contains 1,001 samples over 0--10 s.

Both gain-tuning cases build **one PCS segment**, 0.1 m long and 0.02 m in
radius, with six generalized coordinates and three tendons. Both archives
contain six starts and 5,001 saved samples per rollout over 0--5 s. The old
caption's two-segment description for the synergistic case was incorrect.

The archived initialization metadata specifies log-uniform gain factors with
spread 3: seeds 2 and 0 for the collocated and synergistic cases, respectively.
The first start retains the nominal gains; the remaining five are sampled.

No new simulation or optimization is used in this redesign. Forward kinematics
from the original model builders agrees with saved positions and rotation
matrices for every displayed run. The dimensionless objectives recomputed from
all initial and best stored trajectories agree with the recorded losses.

## Run selection

The old gain plot's initial curve is a **pointwise, per-coordinate median**;
it is not generally the trajectory of any actual robot. Its band is the
initial min--max envelope. The optimized solid curve is the globally best
stored run, not the median final run as the old caption stated.

The new main figure uses actual rollouts selected by the existing video rule:
choose the finite initial run nearest the median initial loss, breaking ties
by lower loss and then lower index. Compare it with the lowest-loss optimized
run over all starts and iterations. Indices below are zero-based:

| Case | Selected initial start | Optimized start | Optimized iteration | Initial loss median | Best loss | Reduction from batch median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Collocated | 5 | 1 | 99 | 0.001489665749 | 0.000796113057 | 46.5576% |
| Synergistic | 4 | 1 | 99 | 0.014890593490 | 0.000185082702 | 98.7570% |

The main figure retains the synergistic example. The caption uses one-based
start/iteration numbers: initial start 5; optimized start 2, iteration 100.
They are different optimization starts, not a paired before/after of one start.
The selected initial loss is 0.013613588051, so its reduction to the optimized
loss is 98.6405%; this must not be confused with the reported 98.7570% reduction
from the batch median.

## Verified main-figure evidence

- Pose-tracking position RMSE: 2.835925 mm over all 1,001 samples.
- Pose-tracking geodesic orientation RMSE: 2.290228 degrees, including startup.
- Tuning snapshots: 0.5 s in both selected runs; position errors are 15.429 mm
  initially and 1.041 mm after tuning.
- Final position errors: 9.153885 mm initially and 0.121556 mm after tuning.
- The optimized run enters and remains within 5% of the initial Euclidean
  position error (1.095163 mm) at 0.455 s. The selected initial run does not
  settle within this band during 5 s.

The old 0.9 s settling, overshoot, cumulative-error, and 99% variance statements
are not used as evidence for the new figure: their precise run/metric selection
was not traceable to the existing text. The manuscript uses the verified loss
and position-error quantities above, with an explicit sustained-settling rule.

## Rendering and layout

The upright pose-tracking robot preserves its physical gravity direction.
The gain-tuning robot is shown hanging, using one rigid display transform for
the body, target, and stored motion together. Initial and optimized views have
identical cameras, scale, lighting, material, and selected physical time.

The common studio uses coral bodies, dark mounts, and green references. The
gain-comparison markers follow the safety figure: white current-tip dots and
green target rings at their exact camera projections, without connectors.
The pose target ring and its normal show orientation without a translucent
sphere obscuring the robot. The three large views share a row above the
position/orientation errors and the tuning transient. Numerical statistics,
selection details, and timestamps belong in the caption.

Machine-readable values and source hashes are in `outputs/paper_snapshots/audit.json`.
The full gain histories, both gain cases, the configuration-space comparison,
and the componentwise pose histories remain available for the appendix.

The matching motion comparison renders 61 saved states from 0--5 s for each
selected run at 12 fps. It uses the same fixed cameras, materials, and marker
projections as the still views. No interpolation or new dynamics are introduced.
The fixed figure crop keeps pixels 0--1250 of the original 1400-pixel height.

The revised tracking view uses a solid recorded state at 6.25 s, with its distal
face toward the camera. Three faint recorded states at 4.0, 4.75, and 7.25 s illustrate
the motion range. Their shared proximal 15.625 mm are omitted to prevent
near-coincident translucent surfaces; they cast no shadows. Embedded ghost faces are omitted as well. All states use one
camera and one mounting. Ghost timestamps and tip positions are checked against
the same saved trajectory. The gain views additionally crop horizontal pixels
60--948. Plot widths are 69 mm (tracking) and 78.5 mm (tuning), with 8.5 pt axis
labels/legends and 8 pt ticks at 180 mm figure width.

For clarity, the solid recorded pose is kept in the foreground: ghost triangles
that cross its camera-projected silhouette are omitted. This display-only
occlusion is recorded as `opaque_state_foreground` in each overlay manifest.

The tuning plot includes a pointwise min--max envelope of the Euclidean
position-error norms across all six initial gain sets. The blue line is the
selected actual initial run, and the coral line is the best optimized run.
Labels explicitly identify selected/best runs; the band is an empirical range,
not a confidence interval. The gap between plot axes is reduced to 13.5 mm.
