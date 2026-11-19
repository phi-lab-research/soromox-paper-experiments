# Section Vg: Parallel Reinforcement Learning

This case contains the complete parallel SoRoMoX PPO workflow: environment,
training, evaluation, rollout generation, reward plotting, and offline video
rendering. Released checkpoints and source data live in `data/`; canonical paper
outputs live in `outputs/`.

Install the paper and RL dependencies:

```bash
uv sync --extra paper_results --extra rl
```

## Training and evaluation

Train with the paper defaults. Checkpoints and reward records are written below
the case-local `data/` directory:

```bash
uv run python paper_results/secVg_parallel_rl/code/train.py \
  --num-envs 64 --total-timesteps 1000000 --n-steps 200 \
  --save-dir /tmp/soromox-secVg/checkpoints \
  --checkpoint-dir /tmp/soromox-secVg/checkpoints \
  --reward-csv /tmp/soromox-secVg/training_reward.csv
```

`--device` selects where the environment is simulated. `--device gpu` pins JAX
to CUDA and runs SoRoMoX on the Warp backend; `--device cpu` pins JAX to the CPU
and runs SoRoMoX on the pure JAX backend. The default `--device auto` keeps
JAX's default platform and picks Warp on a GPU and JAX otherwise. PPO itself
runs through PyTorch independently of this choice. Evaluation and rollout
generation always use the `auto` backend.

The reward CSV has the columns `wall_time` and `episode_reward_mean`. One record
is written for every `--reward-batch-episodes` completed episodes (default 8):
`episode_reward_mean` is the mean return of those episodes and `wall_time` the
mean time in seconds at which they finished. The clock starts after the
environment has been built, so environment construction is not counted.

The training CLI refuses to replace the released checkpoint or an existing
reward CSV unless `--force` is supplied.

Evaluate the released checkpoint:

```bash
uv run python paper_results/secVg_parallel_rl/code/evaluate_rl.py
```

Evaluation reports the component-wise end-effector error, mean Euclidean
tracking error, and success rate for a fixed rollout.

## Paper artifacts

Recreate the reward curve from the committed training logs:

```bash
uv run python paper_results/secVg_parallel_rl/code/plot_rl.py
```

The plot reads three subfolders of `data/reward_logs/`, each holding one
reward CSV per training run:

| Folder | Runs | Legend label |
|---|---|---|
| `PyElastica-CPU/` | PyElastica baseline, 2 envs on CPU | PyElastica (CPU, 2 envs) |
| `SoRoMoX-CPU/` | SoRoMoX with `--device cpu`, 2 envs pinned to 2 CPU cores | SoRoMoX (CPU, 2 envs) |
| `SoRoMoX-GPU-RTX4070/` | SoRoMoX with `--device gpu` on an RTX 4070, 64, 128, 256 and 512 envs | SoRoMoX (RTX 4070, N envs) |

The device tag comes from the folder name (CPU, or the GPU model of a
`SoRoMoX-GPU-<model>/` folder) and the environment count from
file names of the form `SoRoMoX_<N>_envs_<run-id>.csv`; files without a count in
their name (PyElastica and SoRoMoX CPU) are labeled with 2 envs. Runs with the
same label are averaged and the shaded band shows one standard deviation.
The default reproduces the existing W&B-style processing: keep up to 1,500
records per run with random seed 0, apply Gaussian smoothing with scale 10
records, merge equal timestamps, and interpolate each run onto 500 points
over the time interval common to the runs in its group. The band is one
sample standard deviation across the two runs. Passing `--episodes-per-point
512` instead selects equal-episode aggregation. A SoRoMoX record holds
8 episodes and a PyElastica record 2.

Rollout generation and rendering are separate steps. Generate the single-arm
trajectory data for both the uniformly random initialized-policy baseline and
the fully trained PPO policy with:

```bash
uv run python paper_results/secVg_parallel_rl/code/run_rl_policy.py \
  --policy random --num-envs 1 --game-time 20 --n-steps 300 \
  --trajectory-output paper_results/secVg_parallel_rl/data/traj/rl_rollout_initialized_1_env.npz \
  --force

uv run python paper_results/secVg_parallel_rl/code/run_rl_policy.py \
  --policy trained --num-envs 1 --game-time 20 --n-steps 300 \
  --trajectory-output paper_results/secVg_parallel_rl/data/traj/rl_rollout_trained_1_env.npz \
  --force
```

The random baseline samples reproducible uniform actions from the environment;
the trained mode loads the released PPO checkpoint and normalization state.
Render both saved trajectories without rerunning either policy:

```bash
uv run python paper_results/secVg_parallel_rl/code/render_rl_video.py \
  --data paper_results/secVg_parallel_rl/data/traj/rl_rollout_initialized_1_env.npz \
  --force

uv run python paper_results/secVg_parallel_rl/code/render_rl_video.py \
  --data paper_results/secVg_parallel_rl/data/traj/rl_rollout_trained_1_env.npz \
  --force
```

When `--output` is omitted, the renderer writes the MP4 and GIF to the case
study `outputs/` directory using the input NPZ stem. For example,
`rl_rollout_initialized_1_env.npz` produces
`rl_rollout_initialized_1_env.mp4` and `rl_rollout_initialized_1_env.gif`. Use
`--output` or `--gif-output` to override either path. A single selected
environment uses the fixed camera and visual style of the paper result.

To generate and render parallel rollouts for both policy states, save all
environments in one NPZ per policy and pass each file to the same renderer:

```bash
uv run python paper_results/secVg_parallel_rl/code/run_rl_policy.py \
  --policy random --num-envs 64 --game-time 20 --n-steps 300 \
  --trajectory-output paper_results/secVg_parallel_rl/data/traj/rl_rollout_initialized_64_envs.npz \
  --force

uv run python paper_results/secVg_parallel_rl/code/run_rl_policy.py \
  --policy trained --num-envs 64 --game-time 20 --n-steps 300 \
  --trajectory-output paper_results/secVg_parallel_rl/data/traj/rl_rollout_trained_64_envs.npz \
  --force

uv run python paper_results/secVg_parallel_rl/code/render_rl_video.py \
  --data paper_results/secVg_parallel_rl/data/traj/rl_rollout_initialized_64_envs.npz \
  --force

uv run python paper_results/secVg_parallel_rl/code/render_rl_video.py \
  --data paper_results/secVg_parallel_rl/data/traj/rl_rollout_trained_64_envs.npz \
  --force
```

The multi-arm renderer uses every stored environment by default, arranges the
arms in a centered near-square grid, and uses a low camera with the outer robots
crossing the image boundary. `--max-envs N` selects the first `N` environments;
in particular,
`--max-envs 1` provides the first-class single-arm view from a batched rollout.
Use `--rows` and `--cols` together to override the automatic grid, and
`--grid-spacing` to change the distance between robot bases. Dotted target paths
are enabled by default for a single arm and hidden for parallel grids to keep the
individual robots legible; `--show-trajectory` and `--no-show-trajectory`
override that choice. Robots are rendered as swept surfaces. Parallel grids use
40 backbone samples per robot while the single-arm view uses 80;
`--num-points` overrides either default.

Rollouts default to a 20 second horizon at 15 policy steps per second. When
`--n-steps` is omitted, it is derived from `--game-time` and `--control-fps` so
the target trajectory covers the complete saved rollout.

Generated trajectories use time-major arrays and retain every environment:
`q_ts` has shape `(T, N, D)` and `ball_ts` has shape `(T, N, 3)`. Existing
unbatched single-arm files remain supported. Trajectory NPZs belong under
`data/traj/`; generated plots, MP4s, and GIFs belong under `outputs/`. The names
follow `rl_rollout_<policy-state>_<environment-count>` so corresponding data and
media share a stem. Current case-study plots and rendered media in `outputs/`
are versioned; MP4 and GIF files use Git LFS. Both CLIs protect existing outputs;
pass `--force` when replacement is intentional. Rendering requires Open3D, a
working display or headless graphics setup, and ffmpeg. Pass `--visible` on
desktop graphics stacks that cannot create a hidden Open3D window. Training and
rollout results are stochastic and accelerator dependent; the committed
checkpoint, normalization state, reward CSVs, and rollout NPZ provide the paper
provenance.

## Redesigned main paper figure

`code/plot_rl_summary.py` writes the main PDF, PNG, SVG, and provenance JSON
under `../final_outputs/parallel_reinforcement_learning.*`:

- A is one low, close view into a dense population of robots. Recorded states
  from the 64-environment trained-policy rollout at 5 s are repeated as display
  instances beyond the image boundaries. This is explicitly an illustration;
  the extra instances do not represent additional independent simulations.
- B shows environment 0 at 10 s, with its complete recorded 0--20 s target
  (green dashed) and end-effector (blue solid) paths. The mean position error
  over all 301 saved instants is 6.548 mm, including the initial approach.
- C retains both CPU baselines and all four GPU batch sizes with the existing
  learning-curve processing. These are the measured training comparisons.

Reproduce from committed snapshots, from this directory:

```bash
python code/plot_rl_summary.py --force
```

Export geometry with the experiments environment, then render in the RTX
runtime pinned by `../secVf_sampling_based_mpc/`:

```bash
JAX_PLATFORMS=cpu python code/export_rl_grid_scene.py --force
python code/render_rl_grid_scene.py --geometry outputs/paper_snapshots/grid_close_geometry.npz --output outputs/paper_snapshots/grid_close.png --force
python code/render_rl_grid_scene.py --geometry outputs/paper_snapshots/tracking_geometry.npz --output outputs/paper_snapshots/tracking.png --force
python code/plot_rl_summary.py --force
```

The grid has 0.28 m spacing and includes 7,050 display instances after
conservative culling against the full camera view. Every instance uses one of the 64 saved
robot/target pairs; environment indices and base offsets are recorded. Distant
meshes use fewer cross-sections and sides, while foreground robots retain the
full mesh resolution. The shared studio floor extends to a backdrop bend at
24 m for this large scene; all lighting and materials match MPC and safety.
The final panoramic crop is [0, 185, 1800, 720], retaining the distant rows
and a small margin above them. Panel A is taller to preserve the foreground
scale, with the same low camera angle. The population reaches the horizon
and continues beyond the lateral frame boundaries.

The tracking detail uses the existing RL model for forward kinematics and
preserves every saved time sample for both paths. Only the display base offset
is applied. Its 6.548 mm whole-trajectory error for environment 0 is distinct
from the 128-env final-step evaluation in the main text. Path meshes exclude
shadows, and the complete robot and mounting remain visible. Its display crop
is [240, 0, 880, 1000], removing empty side margins while retaining both
complete paths. The narrower panel gives the learning plot 124 mm of width;
its legend is 8 pt, tick labels 8 pt, and axis labels 9 pt. The row gap is
reduced while retaining space for panel letters and the tracking legend.

Snapshot and figure manifests contain source/geometry hashes, display-instance
counts, original environment indices, cameras, crop, and trajectory time range.
The renderer uses `../paper_rtx_studio.py` directly. The suggested caption is
`figure_caption.tex`. Existing random/trained rollout videos remain available;
this revision introduces no appendix figure.

The legacy assembler's `--rl-only` mode writes
`parallel_reinforcement_learning_legacy.*` to avoid replacing the current figure.
