# Safety-constrained control figure

The main figure gives one message: the contact-force constraint holds in the
saved rollout while the robot accepts a larger final goal error. Panel A is
a matched pair at the last recorded instant (7.999 s); panels B and C show
force and goal distance separately with identical time axes. Nominal HOCLF
always precedes HOCLF + HOCBF. Robot materials stay identical across methods.

Only saved data are used. The two rollouts have 8,000 samples each. The
maximum pairwise normal forces are 33.6121 and 4.98668 N; terminal distances
are 0.0417569 and 30.0577 mm, respectively. The prescribed limit is read from
`data/system_parameters.npz`. These values supersede the older manuscript
summary (3 mm and 32 mm terminal errors).

## Reproduction

Export geometry in the experiments environment, using the same PCS model as
the existing renderer. Both the end-effector distance and the discretized
contact-force calculation are checked against the saved data before export.

```bash
JAX_PLATFORMS=cpu python code/export_safety_scene.py
```

The saved physics frame has positive-z gravity. One rigid transform rotates
the whole scene into the studio's z-up frame and translates the base to
0.45 m. The robot, obstacles, and target receive the same transform. The
mount is SoRoMoX's existing default mounting geometry. The distal end is rounded with the 36 mm-radius hemisphere used by the
contact model at the final backbone sample. A flat terminal face would hide
the force-producing contact: both active contacts lie beyond that face.
The corresponding penetrations are 33.6121 mm (HOCLF) and 4.98668 mm
(HOCLF + HOCBF), using the saved 1000 N/m penalty stiffness. The geometry
regression check compares these values with distances to the rendered mesh.
No predicted states, decorative obstacles, or additional hardware are introduced.

Render each geometry file in a fresh process in the pinned RTX environment
documented in `../secVf_sampling_based_mpc/README.md`:

```bash
python code/render_safety_scene.py --geometry outputs/paper_snapshots/without_cbf_geometry.npz --output outputs/paper_snapshots/without_cbf.png
python code/render_safety_scene.py --geometry outputs/paper_snapshots/with_cbf_geometry.npz --output outputs/paper_snapshots/with_cbf.png
python code/plot_safety_control.py
```

Use `--force` to replace existing renders or figures. The renderer uses
`../paper_rtx_studio.py`, shared directly with MPC. The figure composer projects the recorded terminal backbone point as a white
dot and its target as a green ring. A compact
legend identifies the two markers. Both remain visible through the robot;
they coincide when the end effector reaches the target. The terminal backbone
point is the center of the rounded contact tip, not its outermost surface.
Panel C reports the full 3D distance between these points.
These overlays do not change scene geometry.
The two views have identical cameras, scales, lighting, and times.

Outputs are `../final_outputs/safety_constrained_control.{pdf,png,svg,json}`.
Snapshot metadata records data, parameter, geometry, and renderer hashes,
the selected timestep, display transform, camera, and studio. The composer
rejects mismatched data, times, camera settings, or obsolete studio settings.
All saved trace points are plotted, without smoothing or downsampling.
The suggested paper caption is in `figure_caption.tex`.

The standalone parallel-RL figure is built by
`../secVg_parallel_rl/code/plot_parallel_rl.py`. No extra diagnostic figure or
appendix material is introduced for this safety redesign.
