# Section Vd: Control-Gain Optimization

This case optimizes PID gains through closed-loop rollouts for collocated
actuation-space and synergistic operational-space control of a one-segment PCS
robot. Each method evaluates six gain initializations in parallel.

## Method

Both controllers minimize the dimensionless objective

```text
L = (1 / T) * integral_0^T ||e_hat(t)||^2 dt

collocated   e_hat = [L_seg * e_kappa ; e_sigma]
synergistic  e_hat = [e_theta ; e_x / L_seg]
```

The optimizer uses Yogi with per-gain-family learning-rate multipliers. Initial
gains are sampled from a log-uniform distribution around the nominal values.

| Setting | Collocated | Synergistic |
| --- | ---: | ---: |
| Learning rate | `1500` | `1547` |
| Initialization seed | `2` | `0` |
| P/I/D rate multipliers | `1.0 / 0.5 / 0.1` | `1.0 / 0.5 / 0.1` |
| Solver step | `1e-3` | `1e-3` |
| Integral-error saturation | `10 mm` | n/a |

The collocated controller applies `tanh(e / e_sat) * e_sat` to the error before
integration.

## Run

Install the paper-results dependencies:

```bash
uv sync --extra paper_results
```

Run either optimization:

```bash
G=paper_results/secVd_control_gain_optimization/code/optimize_control_gains.py
uv run python $G --method collocated --num-iters 100 --force
uv run python $G --method synergistic --num-iters 100 --force
```

`--device auto` defers to JAX; `--device cpu` and `--device gpu` are strict.
Use `--result-dir` to stage results outside the committed data directory.

The main and appendix figures combine these runs with model-based control:

```bash
uv run python paper_results/secVc_model_based_control/code/plot_control_tuning.py
uv run python paper_results/secVc_model_based_control/code/plot_control_tuning_appendix.py
```

Render the optimized rollouts:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/render_control_gain_optimization_animations.py \
  --method both --fps 24 --gif --force
```

Use `--trajectory initial-median` for the finite initial rollout closest to the
median initial loss; the default is `optimized-best`. Initial-median files use
the `_initial_median_tracking` suffix; best files use `_optimized_best_tracking`.
All renderings share one gravity-down camera.

Run a solver-step, learning-rate, or saturation study:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/secvd_tuning_study.py \
  --method collocated --stage learning-rate
```

Tuning summaries are written to the ignored `data/tuning/` directory.

## Result archive

Each method writes `optimization_results.npz` using schema version 3.

| Field | Shape |
| --- | --- |
| `history_loss`, `history_finite_mask` | `(iterations, B)` |
| `history_Kp`, `history_Ki`, `history_Kd` | `(iterations, B, m)` |
| `init_Kp`, `init_Ki`, `init_Kd` | `(B, m)` |
| `q_ts_init/best`, `qd_ts_init/best` | `(B, timesteps, dofs)` |
| `u_ts_init/best` | `(B, timesteps, actuators)` |
| `x_ts_init/best` | `(B, timesteps, 6)` |
| `t_ts`, `q_des_ts`, `x_des_ts` | shared time-series arrays |

Pose vectors contain rotation-vector xyz followed by Cartesian xyz. Archives
also record the initialization, objective scales, saturation, solver, material,
and optimizer metadata needed to validate and rescore the stored rollouts.

Optimization archives and rendered MP4/GIF files use Git LFS. Comparison
figures remain in normal Git.

## Control-performance metrics

Recompute the reported metrics directly from both saved archives:

```bash
uv run python paper_results/secVd_control_gain_optimization/code/secvd_metrics.py
```

The command writes `outputs/control_gain_metrics.json`, with source hashes,
all six initial-run metrics, the selected initial run, initial metric medians,
the best optimized run, and percentage reductions for each baseline. Future
non-placeholder optimization runs also write `performance_metrics.json` beside
their NPZ archive. No new simulation is required to evaluate existing archives.

The metrics use the three angular strains for actuation-space control and the
three Cartesian position coordinates for operational-space control, over the
complete saved 5 s interval:

- Cumulative tracking error is the trapezoidal time integral of the Euclidean
  error norm, with units rad s/m or mm s.
- Settling time is the first saved time after which the error norm remains
  within 5% of its initial magnitude through the last sample. First entry into
  the band and the stricter coordinatewise settling rule are also exported.
- Maximum coordinate overshoot is the largest excursion past a coordinate's
  target in the direction of its initial-to-target step. Coordinates with no
  commanded step contribute zero to this overshoot metric. The report also
  includes each coordinate's overshoot and the peak overshoot-vector norm.
- Initial metric medians are computed after evaluating each initial run.
  Nonsettled runs are censored above the observation horizon, never discarded.
  Undefined settling times and reductions are exported as JSON `null`.
- Loss variance is compared across the six starts at the first and last
  optimization iterations. A zero baseline has no defined relative reduction.

| Metric | Actuation initial median | Actuation optimized | Operational initial median | Operational optimized |
| --- | ---: | ---: | ---: | ---: |
| Cumulative error [rad s/m or mm s] | 0.384958 | 0.203392 | 60.516058 | 2.911873 |
| Settling time [s] | 0.3785 | 0.137 | Not settled within 5 s | 0.455 |
| Maximum coordinate overshoot [rad/m or mm] | 1.421849 | 0.806090 | 10.028677 | 4.244458 |

Relative to the initial metric medians, actuation-space cumulative error,
settling time, and overshoot decrease by 47.17%, 63.80%, and 43.31%.
Operational-space cumulative error decreases by 95.19%. Loss variance falls
by 99.8184% and 99.9953%, respectively.

The old figure used a coordinatewise median response curve. Metrics of that
curve are exported separately as `legacy_pointwise_median_response`; they are
not the median of metrics evaluated per run. For the earlier first-entry
wording, the actuation-space median is 0.046 s initially and 0.025 s optimized
(45.65% reduction). The response subsequently leaves the band, so the paper
uses sustained settling. Against the actual selected initial run shown in the
figure, actuation-space settling and overshoot reductions are 53.08% and
35.65%. These baseline-dependent values do not reproduce the old 52% and 29%
claims, or the old operational-space settling time of 0.9 s.
