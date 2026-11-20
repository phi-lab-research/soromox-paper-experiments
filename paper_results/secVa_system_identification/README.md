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

## Paper figures

The figures use the saved models evaluated on all 22 configurations. The error
metric is the RMS Euclidean marker-position error across configurations and
markers. One representative shape per tendon-input family is selected nearest
the median identified-model error. The learned residual uses the saved network
checkpoint, with its original negative tendon-force features.

Compose the main and appendix figures from the committed arrays and studio renders:

```bash
uv run python paper_results/secVa_system_identification/code/plot_identification.py
uv run python paper_results/secVa_system_identification/code/plot_identification_appendix.py
```

These write `final_outputs/model_identification_residual_learning.*` and
`final_outputs/model_identification_residual_learning_appendix.*` in PDF, SVG,
PNG, and JSON formats. The appendix contains the eight fitting configurations,
per-marker RMSE spiderweb plot, and error distribution. Its numerical plots use
all 22 configurations and four markers. JSON sidecars record source hashes and
plotted values. The captions are in `figure_caption.tex` and
`appendix_figure_caption.tex`.

To re-evaluate the models and export scene geometry:

```bash
uv run python paper_results/secVa_system_identification/code/identification_data.py
uv run python paper_results/secVa_system_identification/code/export_identification_scene.py
```

The evaluator checks recomputed predictions against the saved arrays. Render
each exported `*_geometry.npz` with `code/render_identification_scene.py
--geometry PATH --output PATH.png` in the shared NVIDIA RTX environment.
The plotters verify that scene metadata matches the evaluation and marker
geometry. Each command requires `--force` to replace existing outputs.
