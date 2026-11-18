# Patched Open3D build

This is the Open3D recipe copied from [SoRoMoX `b4856ec7`](https://github.com/phi-lab-research/soromox/tree/b4856ec7c4ca5c6509255945f004aa7516bbe114/third_party/open3d).
The source revision, patches, and native build behavior are preserved.
See the [upstream build instructions](https://github.com/phi-lab-research/soromox/blob/b4856ec7c4ca5c6509255945f004aa7516bbe114/third_party/open3d/README.md)
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
