# Migration validation

This records checks performed at the v1.0.0 migration baseline. The anonymized
archive generator and its tests were subsequently retired; archive checks below
are historical validation results.

The migration baseline is SoRoMoX commit
`b4856ec7c4ca5c6509255945f004aa7516bbe114`. Validation used Python 3.12.11 on
macOS with the root lockfile and installed `soromox==0.6.0`, without importing
the library from its source checkout.

- All 264 canonical data and output files match the source bytes and SHA-256
  checksums recorded in `migration.json`.
- All 49 LFS objects referenced by the extracted history are present locally;
  the current tree passes `git lfs fsck`.
- The migrated test suite passed: 368 tests. After adding LFS materialization
  to the anonymous archive tool, all 17 archive tests passed, including the
  three new missing/corrupt/present object cases.
- The original and migrated control-metrics scripts produced identical JSON
  from the saved trajectories in the same environment.
- The original and migrated CPU runtime plotters produced pixel-identical PNGs
  in the same environment. PDF, SVG, and PNG regeneration all completed.
- The sampling-based MPC summary reproduced the success counts, errors, and
  timings documented in its section README.
- All relative Markdown links in the experiment READMEs resolve locally.
- The actual anonymous archive was generated successfully: 414 included files,
  17 sanitized text files, 47 files with sanitized metadata, and two excluded
  files. Its LFS contents were materialized and verified.
- On the library side, 37 relevant tests, the documentation build, and both
  source/wheel distribution checks passed.
- The root and both MPC environment lockfiles were regenerated with fixed
  SoRoMoX and Newton-Cosserat dependencies.

## Existing artifact discrepancies

The committed Section Vc `data/metrics.json` does not match the report computed
by the existing script from the committed trajectories. This occurs in the
original repository as well: the original and migrated scripts produce exactly
the same new report. The migration preserves the committed report unchanged.

The committed CPU benchmark PNG also differs from a regeneration in the
validation environment. The original and migrated plotters generate identical
pixels, and the archived original figure is preserved unchanged. This check
does not establish which historical plotting environment generated that figure.

## Validation limits

The local Apple Metal compiler is unavailable. Local tests used the existing
patched Open3D wheel `0.20.0+b6c5e19.soromox3`; the committed build recipe is
unchanged from the library. Linux CI installs the environment through that
source-build recipe. Native rendering, full GPU benchmarks, Newton/RTX runs,
MATLAB/SoRoSim, and full RL training were not rerun as part of the migration.
