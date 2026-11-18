# Patched Open3D build

This recipe builds the Open3D release pinned in `OPEN3D_REVISION` with the
renderer patches in this directory. See the
[build instructions](https://github.com/phi-lab-research/soromox/blob/61a5ebd9cc5e56786044b0c35d33dac7893a9f46/third_party/open3d/README.md)
for platform prerequisites, cache settings, and renderer fixes.

From this experiments repository, build and install with:

```bash
uv sync --locked --group test --group paper-benchmarks
```

On macOS, Open3D and PyTorch may load different OpenMP libraries. If the
process reports `OMP: Error #15`, put PyTorch's library directory first:

```bash
soromox_torch_lib="$(uv run --no-sync python -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("torch").origin).parent / "lib")')"
DYLD_LIBRARY_PATH="$soromox_torch_lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
  uv run --no-sync python -m pytest tests/paper_results
```
