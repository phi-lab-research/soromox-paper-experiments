# SoRoMoX paper experiments

Code, datasets, checkpoints, and figures for reproducing the experiments in
**[SoRoMoX: Fast, Differentiable, and Parallelizable Soft Robot Models](https://arxiv.org/abs/2608.06650)**.

[Experiment index](paper_results/README.md) · [Library](https://github.com/phi-lab-research/soromox) ·
[API documentation](https://phi-lab-research.github.io/soromox) · [Citation](CITATION.cff)

## Setup

Install Git LFS and uv, then clone this repository with its data:

```bash
git clone https://github.com/phi-lab-research/soromox-paper-experiments.git
cd soromox-paper-experiments
git lfs install
git lfs pull
uv sync --locked --group test --group paper-benchmarks
```

The root environment pins **SoRoMoX 0.6.0** and supports Python 3.11–3.14.
It installs the library as a dependency; no separate SoRoMoX checkout is needed.
The bundled [Open3D build recipe](third_party/open3d/README.md) applies the
renderer patches used for the paper. Its first installation requires native
build tools and may take several minutes. Run commands with `uv run --no-sync`
after syncing to retain the chosen dependency groups and GPU extra.

For Linux CUDA 13, add `--extra cuda13` to the sync command. CPU tests can use
`JAX_PLATFORMS=cpu`. SoRoSim comparisons additionally require MATLAB and SoRoSim.
Publication plots may require LaTeX and dvipng; video assembly requires FFmpeg.

**Section Vf sampling-based MPC has separate environments** for Newton 1.7
experiments and Newton 1.6 rendering. Follow its
[setup instructions](paper_results/secVf_sampling_based_mpc/README.md);
RTX rendering requires a compatible GPU. Do not install those stacks into the
root environment.

## Reproduce results

The [experiment index](paper_results/README.md) maps the paper sections to code,
input data, and canonical outputs. Each case contains `code/`, `data/`, and
`outputs/`; composed figures are in `paper_results/final_outputs/`.

Use the saved data to regenerate plots before rerunning simulations or training.
Write new runs to another directory: some generators overwrite their default
data paths. Timings depend on hardware; stochastic training need not reproduce
the saved checkpoints byte for byte.

```bash
uv run --no-sync python tools/verify_artifacts.py
MPLBACKEND=Agg JAX_PLATFORMS=cpu uv run --no-sync python -m pytest -m "not memory_intensive"
```

## Reproducibility and citation

The [artifact manifest](provenance/artifacts.json) records source revisions and
checksums for the canonical data and outputs. Run `tools/verify_artifacts.py`
to verify downloaded files. Experiment settings are documented with each case.

Cite the paper using [CITATION.cff](CITATION.cff) and report the experiment
release or commit, SoRoMoX version, and environment lockfile used.
