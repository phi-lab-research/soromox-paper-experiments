# Reproducibility notes

## Reference environment and artifacts

The root environment pins `soromox==0.6.0`. The reference checks use Python
3.12.11 on macOS and the patched Open3D wheel `0.20.0+b6c5e19.soromox3`.
The two Section Vf environments specify their own Newton, Warp, and JAX versions.

The [artifact manifest](artifacts.json) contains SHA-256 checksums for 264
canonical data and output files. After fetching Git LFS files, verify them with:

```bash
python tools/verify_artifacts.py
```

Use the section READMEs to regenerate figures and recompute metrics from saved
data. Timings depend on hardware, and full simulation or training runs may
produce different results. Record the environment, hardware, solver settings,
and random seeds with new results.

## Known output discrepancies

The Section Vc `data/metrics.json` differs from the report computed by
`report_metrics.py` from the committed trajectories. Specify whether a comparison
uses the saved report or metrics recomputed from those trajectories.

The saved CPU benchmark PNG differs from regeneration in the reference
environment. Exact pixel reproduction of that figure has not been verified.
The artifact manifest verifies the saved figure's contents.

## Native rendering

The [Open3D build instructions](../third_party/open3d/README.md) describe the
required native toolchain. Native graphics tests need the relevant graphics
services; Newton RTX rendering requires a compatible GPU. MATLAB/SoRoSim,
full GPU benchmarks, and RL training have additional requirements documented
in their section READMEs.
