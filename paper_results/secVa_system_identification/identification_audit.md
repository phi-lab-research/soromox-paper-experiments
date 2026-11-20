# Identification and residual-learning figure audit

The redesigned figure evaluates the saved models on a **common set of 22 static configurations and four markers**. It does not fit new physical parameters or retrain the neural network. `identification_data.py` reconstructs measurements from the 22 raw motion-capture CSVs, evaluates the saved parameters and checkpoint, checks the equilibrium residuals, and compares predictions with the archived arrays. The evaluation JSON records file hashes, numerical checks, input order, and representative selection.

## Metric and evaluation sets

The metric is `sqrt(mean(sum((predicted - measured)**2, axis=xyz)))`, pooling configurations and markers. Units below are millimetres. It is neither the mean Euclidean error nor coordinatewise RMSE.

| Model | All 22 configurations | First 8 parameter-fitting configurations |
| --- | ---: | ---: |
| Initial parameters | 49.823624 | 56.916733 |
| Identified, rounded parameters used by residual model | 19.260606 | 22.927317 |
| Learned residual | 6.885330 | 7.148460 |
| Independently fitted residual reference | 6.552905 | 6.480296 |
| Identified, full-precision parameters (audit only) | 19.228520 | 22.928578 |

Identification reduces common-set RMSE by 61.34%; the learned residual reduces it by another 64.25%. Together they provide a factor of 7.24 reduction. The previous 56.9-to-19.3 mm comparison mixed an eight-configuration initial score with a 22-configuration identified score. The correct eight-configuration identification comparison is 56.9 to 22.9 mm (59.7%).

The original `rmse_overall.npy` contains **mean Euclidean norms**, despite the filename and plot label: 36.83, 15.76, 6.09, and 6.41 mm. Its initial value also uses only eight configurations. `errnorm_*.npy` are likewise per-marker mean norms. The new figure recomputes true RMSE from position errors and never uses these aggregates as RMSE.

## Model and checkpoint provenance

- Two GVS links, 9 DOF. The first has constant torsion, linear bending in both axes, and **linear axial strain**, totaling seven DOF. The second has two constant-bending DOF. The earlier manuscript's constant-axial description was inconsistent with the basis.
- Saved fitted parameters: E = 315093.770228 Pa, nu = 0.45000000000000445, rho = 1317.66070971 kg/m^3. The residual case used rounded E = 315000 Pa, nu = 0.45, rho = 1320 kg/m^3; keep that baseline to match its force targets and checkpoint.
- `curves_orig.npy` and `markers_orig.npy` were overwritten by the residual workflow. They represent the **identified** model on 22 configurations, not initial parameters. `curves_hat.npy`/`markers_hat.npy` represent the exact fitted model on eight configurations. Initial shapes are re-evaluated explicitly.
- The checkpoint expects **negative tendon-tension features**. The current robot API uses positive tendon tensions. Negating the inputs reproduces every archived NN marker-error norm within numerical tolerance; passing positive tensions directly yields 27.12 mm pooled RMSE. `checkpoint_features` now makes the conversion explicit in evaluation and compatible training.
- No new optimized residuals are generated. The three saved force arrays are concatenated in their original order.
- The source declares 20 training cases and two test cases (zero-based indices 1 and 16). The weight file does not contain training metadata, so this declaration cannot be independently confirmed from the checkpoint alone. The respective learned-residual RMSE values are 6.734185 and 8.245790 mm. Test index 1 also belongs to the physical-parameter fitting set. All-22 results must not be described as held-out performance.

## Selection and rendering

Select the configuration nearest the median identified-model per-configuration RMS error **within each input family**, breaking ties by first index. This selects zero-based indices 1, 5, and 17 (`m1_u015.csv`, `m2_u015.csv`, `m12_u0015002.csv`). Selection does not maximize residual improvement.

The whole-arm rendering uses the learned-residual equilibrium for index 17. Gold spheres mark measured positions; the body is a model prediction, not a reconstructed measured surface. A schematic mount uses the shared model-renderer geometry. No unsupported hardware details are added. The hanging orientation is preserved; a common vertical translation places the geometry in the studio. Comparisons use identical cameras/scales, faint learned-model surfaces, and vector backbones/marker symbols. The initial/identified/learned colors are blue/coral/wine, with line styles as a second encoding; the per-configuration fitted reference is gray. Rendered bodies retain the common coral material.

The appendix contains all 22 shape comparisons, a direct per-marker RMSE plot, and empirical error distributions. The old CPU comparison is moved to the computational-performance appendix. The shared studio, typography, cameras, and source hashes are recorded in the figure manifest.

## Marker-offset display correction

The gold measurements do not lie on the backbone. The model attaches markers
at abscissae 0.1291, 0.2195, 0.2800, and 0.3600 m, with local offsets
(0,0,0.025), (0,0,-0.021), (0,0,0.020), and (0.008,0,0) m. The overview now
shows gray offset guides ending at predicted marker centres (wine crosses),
so their separation from measured centres is distinguishable from the known
offset geometry. These guides are explanatory geometry, not a claim about
unrecorded marker-support hardware. In the displayed combined-tendon case,
marker errors are 9.137, 7.232, 5.020, and 7.306 mm. No measurements, states,
parameters, or aggregate errors were changed to improve visual agreement.

`marker_geometry.npz` contains exact forward-kinematic marker anchors for all
22 cases and all four model stages. The exporter verifies each anchor plus
its rotated local offset against the independently evaluated marker position.
The comparison spokes use these same anchors. Comparisons retain a common
camera, scale, and crop; surface opacity is 0.32 and the main backbone line
width is 1.55 pt. The main figure is 180 by 90 mm. The A and B renderings share a top edge; the full C axis bounding box, including its x-axis label, aligns with the bottom of A.
