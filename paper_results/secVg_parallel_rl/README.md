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

`code/plot_rl_summary.py` creates the main PDF, PNG, SVG, and provenance JSON
under `../final_outputs/parallel_reinforcement_learning.*`. Panel A shows every
environment in the saved trained-policy rollout, arranged as an 8x8 display
grid at 5 s and 15 s. These are two times during execution of one fixed policy.
Panel B retains both CPU baselines and all four GPU batch sizes, using the
same data and processing as the preceding main figure. The displayed 64-env
grid and the training-curve batch sizes have separate meanings.

Reproduce the figure from the committed snapshots (from this directory):

```bash
python code/plot_rl_summary.py --force
```

To rebuild the snapshot geometry, use the experiments environment:

```bash
JAX_PLATFORMS=cpu python code/export_rl_grid_scene.py --force
```

The exporter reconstructs the existing RL robot and evaluates forward
kinematics from the saved states. It applies the same grid offset to each
robot, mount, and target. Grid spacing is 0.28 m; the mount rests on the shared
studio floor. Both cameras are fitted together, and the composer computes
one crop from all robot surfaces, mounts, and target bounds in both scenes.
All 64 environments remain in view. Mean end-effector errors in these two
snapshots are 5.204 and 4.511 mm, respectively; these are snapshot statistics
for the 64 shown environments, distinct from the 128-env evaluation reported
in the main text.

Render in the RTX environment pinned by `../secVf_sampling_based_mpc/`:

```bash
python code/render_rl_grid_scene.py --geometry outputs/paper_snapshots/grid_1_geometry.npz --output outputs/paper_snapshots/grid_1.png --force
python code/render_rl_grid_scene.py --geometry outputs/paper_snapshots/grid_2_geometry.npz --output outputs/paper_snapshots/grid_2.png --force
python code/plot_rl_summary.py --force
```

The renderer imports `../paper_rtx_studio.py` directly, matching the MPC and
safety materials, backdrop, lighting, and shadows. Geometry, snapshots, and
figure manifests record source hashes, actual environment indices, time,
camera, crop, and the saved data used for each curve. The composer rejects
mismatched sources, incomplete grids, camera changes, and stale studio values.
The suggested caption is `figure_caption.tex`.

The existing random/trained single-arm videos remain available as supporting
media. No additional appendix figure is introduced. The legacy assembler's
`--rl-only` mode now writes `parallel_reinforcement_learning_legacy.*` so it
cannot overwrite the redesigned canonical figure by default.
