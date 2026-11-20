# Section IVb: Parallel GPU Rollouts

This case measures per-environment simulation speed and aggregate throughput as
the model size and number of parallel environments increase. Timings depend on
the JAX backend and accelerator, so the committed CSV files record the paper
machine's results and should not be expected to match on different hardware.

The parallel-rollout generator reuses the model registry and timing helpers from
`paper_results/benchmark_common.py`; the publication plotter remains
case-local so it can apply the common paper style.

GPU execution is the default. If JAX cannot initialize a compatible GPU, the
generator warns and exits without writing CPU timings under a GPU configuration.
Pass `--device cpu` explicitly to benchmark the same workload on the CPU.

Every generator run writes `data/parallel_rollout_benchmark.csv` by default. Pass
`--csv PATH` to override the destination. The CSV records the source revision
and dirty state, software versions, FP64 mode, UTC timestamp, and
accelerator/runtime identity needed to interpret results across machines. Use
`--backend jax` or
`--backend warp` to select the dynamics implementation for GVS, PCS, and
PlanarPCS. Run each backend in a separate process for a controlled comparison;
JAX-only systems continue to use JAX. The CSV records the requested backend,
the backend resolved for each system, and whether backend selection applies.

Each `(system, size, Gauss-point setting, batch size)` case runs in a fresh
process. Successful rows are checkpointed to the results CSV immediately, so
an out-of-memory failure cannot discard earlier measurements. Failed cases are
skipped and summarized after the remaining cases finish; the machine-readable
report defaults to `data/parallel_rollout_benchmark_failures.csv`. Pass
`--failures-csv PATH` to choose another location. Expected OOM-only failures do
not make the generator exit unsuccessfully, while unexpected worker errors do.

Use `--solver` to select `euler`, `semi-implicit-euler`, `heun`, `bosh3`,
`tsit5`, or `dopri5`; the default remains `tsit5`. The selected solver applies
to every requested system and model size. The CSV records the solver together
with `rollout_finite` and `max_abs_state`, so throughput measurements can be
screened for numerical divergence rather than interpreted on timing alone.

Install dependencies and confirm that JAX sees the intended GPU:

```bash
uv sync --locked --extra cuda13
uv run python -c "import jax; print(jax.devices())"
```

Generate the segment-scaling CSV with the paper integration settings:

```bash
uv run python paper_results/secIVb_parallel_rollouts_gpu/code/generate_parallel_rollout_benchmark.py \
  --systems articulated_soft_robot planar_pcs pcs gvs \
  --gvs-scaling segments --segment-counts 1 2 4 8 16 32 \
  --batch-sizes 1 2 4 8 16 32 64 128 256 512 1024 \
  --duration 1 --solver tsit5 --solver-dt 1e-4 --save-dt 1e-2
```

Generate the GVS strain-basis-order sweep:

```bash
uv run python paper_results/secIVb_parallel_rollouts_gpu/code/generate_parallel_rollout_benchmark.py \
  --systems gvs --gvs-scaling basis-order --gvs-basis-orders 0 1 2 3 4 5 \
  --batch-sizes 1 2 4 8 16 32 64 128 256 \
  --duration 1 --solver-dt 1e-4 --save-dt 1e-2 \
  --csv paper_results/secIVb_parallel_rollouts_gpu/data/gvs_basis_order_1s.csv
```

Regenerate the canonical segment-scaling figures from the committed CSV:

```bash
uv run python paper_results/secIVb_parallel_rollouts_gpu/code/plot_parallel_rollout_benchmark.py \
  --csv paper_results/secIVb_parallel_rollouts_gpu/data/parallel_rollout_benchmark.csv \
  --systems articulated_soft_robot planar_pcs pcs gvs \
  --segment-counts 1 2 4 8 16 32
```

In the current FP64 RTX 5090 results, automatic backend selection resolves to
Warp for GVS. Increasing the batch size from 1 to 1,024 raises one-segment GVS
throughput by 679.7× and 32-segment GVS throughput by 133.6×.

Use an explicit temporary `--csv` path for validation runs to avoid replacing
the committed hardware measurements.

## Four-panel figure alternative

```bash
uv run python paper_results/secIVb_parallel_rollouts_gpu/code/plot_gpu_scaling_summary.py
```

This generates `final_outputs/gpu_simulation_scaling.{pdf,svg,png,json}`. Pass
`--force` to replace existing files. The 180 by 116 mm figure uses four aligned
panels with shared axis scales and a common link/segment-count legend. It reads
the committed segment sweep without rerunning benchmarks. The original plotter
and its outputs remain available for supplementary analyses.

The composer requires the complete 264-row model/size/batch grid, rejects
duplicates and mixed protocol settings, and verifies that aggregate throughput
is `batch_size * simulated_time_s / wall_time_s`. The JSON records input and
output hashes, backend assignments, endpoint gains, throughput peaks, and
provenance limitations. `four_panel_figure_caption.tex` contains its caption.
Use full text width in the paper so each panel retains its intended typography.

### Measurement audit

All four models have six sizes (1, 2, 4, 8, 16, 32) and eleven batch sizes
(1 through 1024, powers of two). All measurements use FP64 on an RTX 5090,
Tsit5 at 0.1 ms for one simulated second, one warm-up, and three timed runs.
The generator synchronizes device results before ending each timer. Articulated
models resolve to JAX; planar PCS, spatial PCS, and GVS resolve to Warp.

The GVS aggregate-throughput gains from batch 1 to 1024 are 679.737x (N=1)
and 133.614x (N=32). N=32 reaches its maximum at batch 512; its final measured
throughput is below aggregate parity. A throughput factor of one refers to
aggregate simulated seconds per wall-clock second, not per-environment real-time
execution. No uncertainty bands are drawn: the archive contains mean runtimes,
without the individual three timing samples.

The source CSV records revision `0242a70c5c895aed886a2a214f9379e67079b95a`
with a dirty working tree. All rollout-finiteness flags are true, but maximum
absolute state entries reach about 4916 for PCS cases; these diagnostics mix
state components and do not establish physical fidelity or numerical convergence.
No measurements are removed or reinterpreted as accuracy evidence.

### Preserved original single-panel figure

`outputs/parallel_rollout_benchmark_combined.pdf` is the exact original figure
used as `figures/benchmarking/batch_simulation_scaling/benchmark_gpu.pdf` in the
paper. Their SHA-256 hashes match:
`204b677da19f99234f6ba6304d7925070123e4be15210b0fc4de85b71e56f82d`.
Its generator is `code/plot_parallel_rollout_benchmark.py`, which remains
unchanged. Preserve this original and its generator while evaluating alternatives;
the new four-panel composer writes separate files. The four-panel alternative remains available for detailed comparisons. The
main paper uses the refined compact version described below.

## Refined compact main-paper figure

```bash
uv run python paper_results/secIVb_parallel_rollouts_gpu/code/plot_gpu_scaling_compact.py
```

This creates `final_outputs/gpu_simulation_scaling_compact.{pdf,png,svg,json}`
at 87 by 67 mm for single-column placement. It uses borderless 3.5%-opacity envelopes spanning the tested model sizes,
omits the generic title, uses four distinct shared-palette colors for model families, and
places frameless legends in unused parts of the plot. All 264 measurements
remain in the curves; symbols mark every second tested batch size to reduce
clutter. The size encoding matches the alternative's markers and line styles,
while color here distinguishes model families. Aggregate throughput is explicit
on the vertical axis. `figure_caption.tex` is the compact manuscript caption.

The compact composer reuses the four-panel composer's data audit. Neither
composer replaces the original plotter or its output. All three figure versions
are maintained separately, and `--force` applies only to the selected version.
