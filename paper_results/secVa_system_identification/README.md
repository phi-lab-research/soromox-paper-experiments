# Section Va: System Identification

This case identifies the soft-tentacle parameters, evaluates the learned
residual model, and produces the paper's shape and error summaries. The 22
motion-capture CSV inputs and canonical derived arrays are committed in `data/`.

Install the paper dependencies:

```bash
uv sync --extra paper_results
```

Run parameter identification into a temporary directory first:

```bash
uv run python paper_results/secVa_system_identification/code/identify_soft_tentacle_parameters.py \
  --output-data-dir /tmp/soromox-secVa-parameters --no-show
```

The residual workflow consumes the fitted model, force arrays, parameter result
arrays, and motion-capture CSVs from one data directory. To regenerate the
canonical case-local arrays explicitly:

```bash
uv run python paper_results/secVa_system_identification/code/identify_soft_tentacle_residual.py \
  --force --no-show
```

Plot the committed or regenerated arrays without rerunning identification:

```bash
uv run python paper_results/secVa_system_identification/code/plot_system_id_residual.py
```

The parameter and residual optimizations are computationally intensive. Their
CLIs refuse to overwrite generated data unless `--force` is supplied.

## Audited identification figure

See `identification_audit.md` for the corrected metric, common evaluation set,
checkpoint input sign, model provenance, and representative-selection rule.
The canonical historical arrays are preserved. The new figure uses explicit
re-evaluations of the saved models, without rerunning optimization or training.

```bash
uv run python paper_results/secVa_system_identification/code/identification_data.py
uv run python paper_results/secVa_system_identification/code/export_identification_scene.py
```

Render the four exported `*_geometry.npz` files with
`code/render_identification_scene.py --geometry PATH --output PATH.png` in the
same NVIDIA RTX environment used for the other paper figures. Geometry and
rendering are separate so model evaluation does not require an RTX GPU.

```bash
uv run python paper_results/secVa_system_identification/code/plot_identification_summary.py
```

The outputs are `model_identification_residual_learning`,
`identification_shape_comparisons`, and `identification_error_details` in
`paper_results/final_outputs/`, in PDF, SVG, and PNG formats. The main figure
also includes a JSON manifest. Each CLI refuses overwrites unless `--force`
is supplied. `figure_caption.tex` contains the corresponding manuscript caption.
The old combined CPU/identification figure and its plotting script are retained
as legacy assets; their aggregate bars must not be interpreted as true RMSE.

A local layout preview can be rendered with `code/render_identification_preview.py` using the same geometry arguments. Its metadata identifies Open3D preview rendering separately from the RTX studio output. The current figure uses the shared RTX studio with 128 samples per pixel; the local preview renderer remains available for layout work.

## Appendix figure

```bash
uv run python paper_results/secVa_system_identification/code/plot_identification_appendix.py
```

`final_outputs/model_identification_residual_learning_appendix.pdf` combines one
3D overlay of the eight physical-parameter fitting configurations, the per-marker
RMSE spiderweb plot, and the empirical distribution of all 88 Euclidean marker
errors per model stage. The shape overlay compares initial and identified models
and displays only markers M2 and M4, following the original paper figure.
Colors, line styles, and marker conventions match the main figure; both
quantitative plots use all 22 configurations and all four markers. The spiderweb
uses a shared linear scale and recomputes RMS Euclidean errors from the audited
arrays; the historical `errnorm_*.npy` means are not used as RMSE. A JSON manifest
records every plotted per-marker value and the input hashes.

The script checks that the RMS of the four marker RMSEs reproduces each audited
pooled RMSE. It generates PDF, SVG, and PNG files without a GPU render; use
`--force` to replace existing outputs. `appendix_figure_caption.tex` provides
the manuscript caption. This composite supersedes the two preliminary appendix
outputs `identification_shape_comparisons` and `identification_error_details`,
which remain available from the main composer.
