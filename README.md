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
The bundled [Open3D build recipe](third_party/open3d/README.md) preserves the
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
input data, and canonical outputs. Each case retains `code/`, `data/`, and
`outputs/`; composed figures are in `paper_results/final_outputs/`.
The old `paper_results`, `rendering`, `rl`, and `test` extra names remain as
aliases for the commands in the section READMEs.

Use the saved data to regenerate plots before rerunning simulations or training.
Write new runs to another directory: some generators overwrite their default
data paths. Timings depend on hardware; stochastic training need not reproduce
the saved checkpoints byte for byte.

```bash
uv run --no-sync python tools/verify_artifacts.py
MPLBACKEND=Agg JAX_PLATFORMS=cpu uv run --no-sync python -m pytest -m "not memory_intensive"
```

## Provenance and versions

The migration was extracted from [SoRoMoX `b4856ec7`](https://github.com/phi-lab-research/soromox/tree/b4856ec7c4ca5c6509255945f004aa7516bbe114)
with the relevant file history and authorship retained. The library code at
that revision matches the [0.6.0 release commit](https://github.com/phi-lab-research/soromox/tree/61a5ebd9cc5e56786044b0c35d33dac7893a9f46).
This identifies the migration baseline, not the generation date or software
revision of every historical experiment. Original result metadata remains intact.

[Provenance](provenance/migration.json) records source revisions and artifact
checksums. The [commit map](provenance/source-commit-map.tsv) maps original
commits to the extracted history (zero hashes denote omitted commits).
Shared benchmark helpers, the tentacle rendering scene, the Open3D recipe,
and the archive tool were copied from the same source revision.

Each experiment release should identify its manuscript revision, SoRoMoX
version, and environment lockfiles. Cite the paper using [CITATION.cff](CITATION.cff)
and report the exact experiment release or commit and SoRoMoX version used.
The manuscript link currently identifies the preprint; this migration does not
assign an unverified arXiv revision number to the stored results.

## Anonymous submission archives

The archive tool operates on this repository's committed tree:

```bash
uv run --no-sync python tools/create_anonymized_archive.py --output /tmp/soromox-experiments-anonymous.zip
```

Review its generated report and retained figures before submission. The library
has its own archive tool; this experiment archive does not bundle the library.
Run `git lfs pull` first: the tool verifies and includes committed LFS contents
and stops if any required object is missing or corrupted.
