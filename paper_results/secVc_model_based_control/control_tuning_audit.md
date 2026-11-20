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
- Tuning snapshots: 5 s in both selected runs; position errors are 9.154 mm
  initially and 0.122 mm after tuning.
- Final position errors: 9.153885 mm initially and 0.121556 mm after tuning.
- The optimized run enters and remains within 5% of the initial Euclidean
  position error (1.095163 mm) at 0.455 s. The selected initial run does not
  settle within this band during 5 s.

The initial redesign omitted the old settling, overshoot, cumulative-error,
and variance statements because their precise definitions and run selections
were not traced. This did not establish that all of those claims were false.
The further saved-data recheck below recovers several of them with explicit
definitions and identifies a baseline mismatch and missing time units.

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
The dedicated appendix figure contains the configuration-space comparison and
the actuation-space gain-optimization case. Componentwise pose histories remain
separate; the operational-space tuning result is shown in the main figure.
The original full gain plots remain available as legacy outputs.

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

## Dedicated appendix figure

The appendix combines configuration-space control with the collocated
actuation-space gain-optimization case. All five controllers, commanded strains,
and 30 s of the configuration comparison are retained. Axial strain includes
the undeformed offset. The lower panels show all 100 optimization iterations
and the first 2 s of the stored 5 s angular-strain responses.

The initial response is actual collocated start 6, selected by proximity to the
initial loss median; the optimized response is start 2 at iteration 100. This
uses the same selection rule as the main figure. Initial coordinate bands are
pointwise min--max envelopes, not confidence intervals. The loss curve is the
minimum at each iteration and its band spans the six starts. A linear loss axis
uses an explicit multiplier of 1000; no underlying metric is changed. The
46.5576% improvement remains relative to the initial batch median.

The separate JSON manifest records source hashes, model scope, selected runs,
plot intervals, and displayed loss units. The paper no longer uses the legacy
pointwise-median response curves or the duplicated operational-space tuning
panels in its appendix figure.

## Further check of the earlier performance claims

Run `python paper_results/secVc_model_based_control/code/audit_gain_performance.py`
to recompute these quantities from the six initial rollouts, best stored
rollouts, and loss histories. No new optimization is performed.

- Loss variance across the six starts falls by 99.8184% (actuation space) and
  99.9953% (operational space), comparing the first and last iterations. The
  earlier approximately 99% claim is supported, although this alone does not
  establish identical trajectories or gains.
- The time integral of the angular-strain error norm of the best actuation-space
  run is 0.203392 rad s/m. The analogous Cartesian-position integral for the
  best operational-space run is 2.911873 mm s. These reproduce the numerical
  scale of the earlier 0.20 and 3 claims, with time included in the units.
- The actuation-space integral decreases by 34.20% relative to selected initial
  start 6, 47.17% relative to the median of the six initial integrals, or 42.66%
  relative to the integral of the legacy pointwise-median curve. Thus the old
  34% claim does not describe a reduction from the initial median.
- The operational-space integral decreases by approximately 95.19% relative to
  the median initial integral (95.23% relative to the legacy median curve),
  supporting the earlier approximately 95% claim.
- The optimized operational-space trajectory overshoots its y-coordinate target
  by 4.24446 mm. This supplies a coordinate-specific interpretation of the old
  approximately 4 mm overshoot, whose metric was previously unspecified.
- The old 0.9 s settling time is not reproduced by the explicit sustained
  Euclidean-error rule (0.455 s). Requiring each position coordinate to remain
  within 5% of its initial displacement gives 1.782 s. Neither calculation
  establishes what definition produced the historical 0.9 s value.
- The earlier actuation-space 52% settling-time and 29% overshoot improvements
  are superseded by the explicitly computed values below.

## Explicit transient metrics and restored manuscript results

The canonical implementation now lives in Section Vd's `code/secvd_metrics.py`.
The old audit entry point delegates to it. The stored report is
`paper_results/secVd_control_gain_optimization/outputs/control_gain_metrics.json`.
Future optimization runs automatically export the same metrics.

Using the median of metrics evaluated separately for all six initial runs:

- Actuation-space sustained settling: 0.3785 to 0.137 s, a 63.8045% reduction.
- Actuation-space maximum coordinate overshoot: 1.421849 to 0.806090 rad/m,
  a 43.3069% reduction.
- Operational-space sustained settling: every initial run is outside the band
  at 5 s, while the optimized run settles at 0.455 s.
- Operational-space maximum coordinate overshoot: 10.028677 to 4.244458 mm.

Settling requires the Euclidean error to remain within 5% of its initial
magnitude through the last sample. Maximum coordinate overshoot is the largest
positive excursion beyond a target along that coordinate's commanded step.
The former first-entry wording instead gives 0.046 to 0.025 s (45.6522%) for
actuation space. Against the selected initial run, the sustained-settling and
maximum-coordinate-overshoot reductions are 53.0822% and 35.6539%.

The manuscript restores the earlier loss, cumulative-error, transient-response,
and batch-spread narrative with these explicit definitions, corrected time
units, and recomputed values. It reports cumulative-error reductions of 47.2%
and 95.2% relative to the initial metric medians and loss-variance reductions
of 99.8% and more than 99.99%.
