# TRIDENT-Depth 2.0 --- Team T69 Project Specification

**Hackathon:** Computer Vision Hackathon 2026\
**Assigned problem:** P4 --- Edge-Preserving Depth Map Upsampling and
Refinement\
**Team:** T69\
**Project working name:** TRIDENT-Depth 2.0 --- Trust-Routed Intelligent
Depth Enhancement Network

> This document is the project source of truth for implementation. The
> official hackathon PDF takes precedence wherever there is any
> conflict. Do not silently omit mandatory requirements, invent results,
> or claim novelty without evidence.

------------------------------------------------------------------------

## 1. Project objective

Build a working, deployable application that takes:

-   an RGB guidance image (when available); and
-   a low-resolution, sparse, noisy, or incomplete depth map

and reconstructs a dense, high-resolution depth map while preserving
object boundaries and depth discontinuities, reducing noise, avoiding
oversmoothing, and preventing RGB texture-copying artifacts.

The project must combine:

1.  Classical computer vision;
2.  Deep learning; and
3.  Hybrid depth reconstruction.

The target upsampling factors are **2×, 4×, and 8×**.

### Research question

Can confidence-aware, geometry-validated fusion of classical depth
reconstruction and learned residual correction improve depth accuracy
and boundary preservation under both low-resolution and sparse/noisy
depth conditions?

### Research hypotheses to test

-   **H1:** Depth-confidence estimation can reduce errors caused by
    unreliable or missing measurements.
-   **H2:** Geometry-validated RGB guidance can reduce texture-copying
    artifacts.
-   **H3:** Learned residual correction can improve on classical
    reconstruction while retaining valid measured depth.
-   **H4:** Adaptive fusion can improve the trade-off between depth
    accuracy and boundary preservation.
-   **H5:** Scale-conditioned processing can provide consistent
    performance at 2×, 4×, and 8×.

These are hypotheses, not established findings. Test them and report
actual results. Do not state that they are proven unless experiments
support the claim.

------------------------------------------------------------------------

## 2. Source of truth: mandatory P4 requirements

The official Computer Vision Hackathon 2026 PDF specifies the following
for P4. Every item below must be implemented and demonstrable.

### 2.1 Classical computer vision --- mandatory

1.  Nearest-neighbour interpolation.
2.  Bilinear interpolation.
3.  Bicubic interpolation.
4.  Joint bilateral filtering.
5.  Guided filtering.
6.  Edge detection and boundary-aware refinement.
7.  Depth discontinuity detection.
8.  Noise removal.
9.  Hole filling.

Implement the methods as identifiable modules, not as claims in
documentation only. Provide baseline outputs and tests.

### 2.2 Deep learning --- mandatory

1.  Implement or adapt a depth completion or depth upsampling network.
2.  Use a suitable architecture, such as a guided depth super-resolution
    network, CNN-based refinement model, or transformer-based depth
    model.
3.  Fine-tune or adapt the model using a suitable depth dataset.
4.  Incorporate RGB guidance where available.

A randomly initialized but untrained network is not a completed
deep-learning solution. Record the training data, split, configuration,
checkpoint, and evaluation procedure.

### 2.3 Hybrid depth reconstruction --- mandatory

1.  Combine interpolation and edge-aware filtering with deep learning.
2.  Develop a depth-refinement mechanism that preserves object
    boundaries.
3.  Handle sparse, noisy, and incomplete depth maps.
4.  Prevent oversmoothing and texture-copying artifacts.

The application must expose or demonstrate classical, deep-learning, and
hybrid results separately.

### 2.4 Advanced challenge requirements --- target all

1.  Support 2×, 4×, and 8× upsampling.
2.  Handle missing depth values and noisy depth measurements.
3.  Demonstrate RGB-guided edge-aware reconstruction.
4.  Compare interpolation, classical refinement, deep learning, and
    hybrid reconstruction.
5.  Evaluate on unseen depth maps.
6.  Demonstrate performance around object boundaries and depth
    discontinuities.

### 2.5 Required evaluation metrics

-   RMSE and MAE for depth accuracy.
-   PSNR and SSIM where appropriate.
-   Edge preservation and boundary accuracy.
-   Performance across different upsampling factors.
-   Processing latency and memory consumption.

Define every metric precisely. State units, valid-pixel masks,
thresholds, depth range, image normalization, timing procedure, and
hardware. Do not compare numbers from incompatible protocols as if they
were directly comparable.

------------------------------------------------------------------------

## 3. Dataset plan

### 3.1 Primary development dataset: NYU Depth V2

Official page: https://cs.nyu.edu/\~fergus/datasets/nyu_depth_v2.html

Use NYU Depth V2 for the initial implementation, controlled
low-resolution experiments, and training/validation/testing of the
custom model.

Prepare aligned RGB, depth, and validity masks. Check the actual
downloaded data format and units rather than assuming them.

### 3.2 Cross-domain dataset: KITTI Depth Completion

Official page:
https://www.cvlibs.net/datasets/kitti/eval_depth_completion.php?benchmark=depth_completion

Use KITTI to assess generalization to outdoor scenes and sparse LiDAR
depth. Respect the official train/validation/test protocol and
licensing/usage conditions. Do not use hidden test ground truth.

### 3.3 Optional additional evaluation: Middlebury Stereo

Official page: https://vision.middlebury.edu/stereo/

Middlebury primarily provides stereo disparity. Do not treat disparity
as metric depth without the required camera calibration and conversion.
It is optional and lower priority than completing the NYU and KITTI
experiments.

### 3.4 Dataset protocols must remain separate

Implement two distinct degradation protocols:

**Protocol A --- low-resolution depth super-resolution** - Start from
valid dense ground-truth depth. - Downsample depth to 1/2, 1/4, and 1/8
spatial resolution for 2×, 4×, and 8× reconstruction. -
Upsample/reconstruct to the original target resolution. - Evaluate
against the aligned dense ground truth.

**Protocol B --- sparse/noisy depth completion** - Create sparse depth
using documented sampling patterns and sampling densities. - Add
controlled depth noise. - Include missing regions / holes. - Preserve a
validity mask. - Evaluate against valid dense ground truth.

Do not describe artificially downsampled dense depth as real LiDAR
sparsity. Report Protocol A and Protocol B separately.

### 3.5 Data splitting and leakage prevention

-   Use scene-aware or official recommended splits where available.
-   Keep near-duplicate frames or frames from the same scene from
    leaking across train, validation, and test sets.
-   Fit/tune model parameters on training/validation data only.
-   Keep final test data held out.
-   For KITTI, report **zero-shot NYU-to-KITTI generalization**
    separately from any KITTI fine-tuning/adaptation.
-   Do not claim cross-dataset generalization unless it is actually
    evaluated.

------------------------------------------------------------------------

## 4. Proposed architecture: TRIDENT-Depth 2.0

The name is a project label, not a claim of globally novel research. The
implementation should be compared with prior work and described
honestly.

### 4.1 Input

The model/application should accept:

-   RGB image, where available;
-   degraded depth map;
-   validity mask;
-   upsampling scale (2×, 4×, or 8×);
-   degradation condition (low-resolution or sparse/noisy).

Validate dimensions, channel counts, numeric ranges, invalid values, and
RGB-depth alignment.

### 4.2 Classical reconstruction branch

Implement and expose: - nearest-neighbour; - bilinear; - bicubic; -
joint bilateral filtering; - guided filtering; - edge detection and
boundary-aware refinement; - depth discontinuity detection; -
denoising; - hole filling.

Use validity-aware filtering. Do not let zero-filled missing pixels
contaminate filter averages.

Produce a classical depth candidate and relevant intermediate
visualizations.

### 4.3 Geometry-validation branch

-   Extract RGB edges and multi-scale image features.
-   Compute depth gradients/discontinuities from available depth.
-   Compare RGB edge evidence with depth/geometric evidence.
-   Estimate a boundary-confidence map.
-   Avoid assuming every RGB texture edge is a true depth boundary.

The mechanism should be testable through ablation. Do not claim texture
rejection works until evaluated on texture-heavy scenes.

### 4.4 Learned residual branch

Use an existing architecture as a documented baseline and
implement/train a lightweight custom residual CNN.

Suggested initial model: - Input channels: RGB (3), depth (1), validity
mask (1), plus scale/degradation conditioning as appropriate. -
Lightweight multi-scale encoder-decoder with skip connections. - Initial
channel plan: 32, 64, 128, then decoder back to 64/32. - Output
heads: 1. depth residual correction; 2. uncertainty/log-variance; 3.
optional boundary confidence.

These are starting design choices, not fixed requirements. Profile and
adjust them.

Residual formulation:

`D_residual = D_classical + Delta_D`

The network should predict a correction to a classical initial estimate,
rather than blindly replacing the entire reconstruction.

### 4.5 Baseline model

Investigate a suitable depth-super-resolution architecture such as DKN
or FDKN and assess whether its implementation, dependencies, weights,
and license can be used within the project. If it cannot be reproduced
in the available time, document the limitation and use a clearly named,
implemented lightweight CNN baseline. Never label a simplified model as
DKN/FDKN.

Deep Anisotropic Diffusion may inform the hybrid/refinement design, but
full reproduction is not required unless time and implementation
feasibility allow.

### 4.6 Trust-routed adaptive fusion

Generate candidate depth estimates: - `D_c`: classical estimate; -
`D_e`: edge-aware depth estimate; - `D_n`: neural residual-corrected
estimate.

Predict or calculate pixel-wise weights: - `w_c`: classical trust; -
`w_e`: geometry/boundary trust; - `w_n`: neural trust.

Normalize so `w_c + w_e + w_n = 1`.

Fuse:

`D_f = w_c * D_c + w_e * D_e + w_n * D_n`

The routing can use: - validity; - depth confidence; - boundary
confidence; - neural uncertainty; - local depth gradients; -
scale/degradation condition.

Do not define `D_e` as an RGB edge image: it must be a depth
estimate/candidate.

### 4.7 Measurement consistency

For valid, reliable observed measurements, preserve consistency with the
input. A hard projection can be used for clean measurements. For noisy
observations, use confidence-weighted/soft consistency instead of
blindly forcing every noisy value to remain unchanged.

Keep input validity masks distinct from ground-truth validity masks.

------------------------------------------------------------------------

## 5. Training losses

Implement losses with correct masks and documented normalization.

### 5.1 Masked depth L1

`L_depth = sum(M_gt * abs(D_pred - D_gt)) / (sum(M_gt) + epsilon)`

### 5.2 Gradient consistency

`L_gradient = L1(grad_x(D_pred), grad_x(D_gt)) + L1(grad_y(D_pred), grad_y(D_gt))`

Apply appropriate valid-neighbour masks.

### 5.3 Edge-weighted depth loss

`L_edge = mean(W_edge * abs(D_pred - D_gt))`

Derive `W_edge` primarily from ground-truth depth structure or a
carefully validated geometric signal. Do not use raw RGB edges alone as
depth-edge labels.

### 5.4 Uncertainty-aware loss

For predicted log variance `s` and error `e`, a possible objective is:

`L_uncertainty = mean(exp(-s) * abs(e) + 0.5 * s)`

Clamp numerical ranges as needed and document the parameterization.

### 5.5 Measurement consistency

`L_measurement = sum(M_input * abs(D_pred - D_input)) / (sum(M_input) + epsilon)`

Use a soft term for noisy measurements; optionally hard-project reliable
measurements after fusion.

### 5.6 Combined objective

`L_total = lambda1*L_depth + lambda2*L_gradient + lambda3*L_edge + lambda4*L_uncertainty + lambda5*L_measurement`

Initial exploratory weights only: - depth: 1.0 - gradient: 0.2 - edge:
0.1 - uncertainty: 0.05 - measurement: 0.2

Tune on validation data. Do not treat these weights as proven optimal.
Add unit tests for masks, empty-valid-pixel cases, finite losses, and
gradients.

------------------------------------------------------------------------

## 6. Evaluation and ablation plan

### 6.1 Methods to compare

At minimum, compare: 1. nearest; 2. bilinear; 3. bicubic; 4. classical
refinement (including guided and joint bilateral); 5. deep-learning
baseline; 6. custom residual CNN; 7. final hybrid TRIDENT-Depth.

Report low-resolution and sparse/noisy protocols separately, and report
each scale separately.

### 6.2 Metrics

Implement: - RMSE; - MAE; - PSNR; - SSIM; - boundary precision, recall,
and F1 at a documented depth-gradient threshold; - edge preservation
measure with a documented definition; - inference latency; - peak memory
consumption where measurable.

Use valid ground-truth pixels consistently. Specify depth units and PSNR
data range. Explain SSIM normalization. Warm up the model and time
repeated inference runs. Report hardware and batch size.

### 6.3 Ablations

Run ablations that isolate: - depth confidence; - geometry/texture
validation; - residual correction; - uncertainty prediction; - adaptive
fusion; - measurement consistency.

Use the same data split, input degradations, and evaluation settings for
fair comparison.

### 6.4 Failure cases

Document at least three real failure cases, such as: - thin structures
and depth bleeding; - texture-heavy surfaces causing false guidance; -
large missing-depth regions with insufficient evidence.

For each, save RGB, degraded depth, classical output, learned output,
hybrid output, ground truth (if available), metrics, and an honest
explanation.

**Never invent or manually fill experimental results.** Use `N/A` or
`Not run` until a measurement exists.

------------------------------------------------------------------------

## 7. Application and deployment

### Frontend

-   HTML5;
-   CSS3;
-   vanilla JavaScript;
-   Canvas or suitable image visualization for depth maps and overlays.

### Backend

-   Python FastAPI;
-   OpenCV;
-   NumPy;
-   PyTorch;
-   a documented metrics module.

### User workflow

1.  Upload RGB image (where available).
2.  Upload depth map.
3.  Validate pair alignment and input format.
4.  Choose 2×, 4×, or 8×.
5.  Choose low-resolution or sparse/noisy input mode.
6.  Select nearest/bilinear/bicubic, classical, deep-learning, or hybrid
    method.
7.  Run reconstruction.
8.  Display input and output, intermediate maps, metrics, and latency.
9.  Provide clear loading and error states.

The application must not claim ground-truth accuracy metrics for
arbitrary user uploads when no ground truth is supplied. In that case,
show only valid no-reference diagnostics and runtime information,
clearly labelled.

### Deployment constraints

-   Public hosted application URL.
-   Works on laptop and phone.
-   Target response around 10 seconds as stated in the rubric; measure
    and report actual performance.
-   Clear loading/error states.
-   Repository runnable with one command or Dockerfile.
-   No secrets or credentials committed.
-   Do not depend on a temporary Colab/Kaggle notebook session remaining
    alive for production inference.
-   Save checkpoints and document model provenance and license.

------------------------------------------------------------------------

## 8. Technology and compute plan

The team has local CPU access and expects to use Google Colab or Kaggle
GPU for training.

-   Local CPU: frontend, FastAPI, classical CV, tests, integration,
    small inference tests.
-   Colab/Kaggle GPU: dataset experiments, training, validation,
    checkpoints, model export.
-   Save checkpoints and experiment configs outside temporary notebook
    storage.
-   Record GPU type, runtime, package versions, random seeds, and
    training settings.
-   Check actual GPU availability before starting a long run.
-   Keep a CPU-compatible fallback if deployment GPU is unavailable.

Do not assume a particular Colab/Kaggle GPU quota or availability.

------------------------------------------------------------------------

## 9. Suggested repository structure

``` text
TRIDENT-Depth-T69/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── backend/
│   ├── main.py
│   ├── schemas.py
│   ├── routes/
│   ├── preprocessing.py
│   └── inference.py
├── classical/
│   ├── interpolation.py
│   ├── guided_filter.py
│   ├── joint_bilateral.py
│   ├── edge_detection.py
│   ├── discontinuity.py
│   ├── denoising.py
│   └── hole_filling.py
├── models/
│   ├── baseline.py
│   ├── residual_cnn.py
│   ├── confidence_heads.py
│   └── fusion.py
├── training/
│   ├── dataset.py
│   ├── degradation.py
│   ├── losses.py
│   ├── train.py
│   └── validate.py
├── evaluation/
│   ├── metrics.py
│   ├── benchmark.py
│   ├── ablation.py
│   └── failure_analysis.py
├── tests/
├── configs/
├── experiments/
├── docs/
│   ├── architecture.md
│   ├── design_document.md
│   └── architecture.png
├── eval_P4_T69.md
├── README.md
├── requirements.txt
└── Dockerfile
```

Adapt this structure if needed, but keep modules separated and
maintainable.

------------------------------------------------------------------------

## 10. Team responsibilities

  -----------------------------------------------------------------------
  Member                              Primary responsibility
  ----------------------------------- -----------------------------------
  Dhanush S --- 1RUA24CSE0125         Classical CV and filtering

  Dhruthi R Kulkarni ---              Deep-learning model, training and
  1RUA24CSE0129                       losses

  E Sasi Vaishnav --- 1RUA24CSE0140   Literature/research, fusion,
                                      metrics and ablations

  Jathin Kirani Nagaraj ---           Frontend, FastAPI, integration and
  1RUA24CSE0178                       deployment
  -----------------------------------------------------------------------

All members must understand the overall architecture and be able to
explain their own work in the viva.

------------------------------------------------------------------------

## 11. Required documentation and hackathon submission

The official PDF requires: - live hosted application URL; - GitHub
repository with complete source code; - system architecture diagram; -
design document; - comparative evaluation results; - ablation study; -
failure-case analysis.

The evaluation file must be named:

`eval_P4_T69.md`

Its first line must contain all USNs comma-separated:

`1RUA24CSE0125,1RUA24CSE0129,1RUA24CSE0140,1RUA24CSE0178`

The README must explain setup and reproduce results as far as the
available artifacts allow. The architecture diagram must match the
implemented code. State limitations and ethics honestly.

### Rubric priorities from the official PDF

-   Evaluation metrics: 25 points.
-   Implementation: 20 points.
-   UI/UX and deployment: 15 points.
-   Project documentation: 15 points.
-   Demo and viva: 25 points.

The rubric emphasizes unseen inputs, a proper train/validation split,
modular code, bad-input handling, a working public URL, reproducible
documentation, and a three-minute demo that every team member can
explain.

------------------------------------------------------------------------

## 12. Implementation phases and acceptance gates

**Important workflow instruction:** Build and verify the codebase first.
Do not start full model training until the data pipeline, classical
methods, interfaces, and tests are working.

### Phase 0 --- inspect environment and plan

-   Inspect the existing repository before changing files.
-   Check Python version, available packages, OS, CPU/GPU, and disk
    space.
-   Propose a dependency plan.
-   Do not overwrite existing work without asking.

### Phase 1 --- codebase and data pipeline

Implement: - repository structure; - config and logging; - dataset
loaders and input validation; - RGB/depth alignment checks; - validity
masks; - degradation generation for both protocols and all scales; -
deterministic train/validation/test split handling.

**Acceptance gate:** Load a real sample and produce aligned RGB,
degraded depth, mask, and ground truth with documented shapes, units,
and valid ranges.

### Phase 2 --- classical CV

Implement all nine classical items listed in Section 2.1.

**Acceptance gate:** Run every method on sample inputs; save outputs;
test dimensions, invalid depth, boundaries, and empty masks.

### Phase 3 --- metrics and evaluation harness

Implement all required metrics and a reproducible benchmark command.

**Acceptance gate:** Test metric implementations on small synthetic
arrays with known expected values; produce a baseline results file
without fabricated values.

### Phase 4 --- model implementation, before training

Implement: - documented learned baseline; - custom residual CNN; -
prediction heads; - loss functions; - fusion module; - checkpoint
save/load; - inference API.

**Acceptance gate:** Forward pass works on synthetic tensors; output
shapes are correct; losses are finite; gradients flow; checkpoint
round-trip works. At this stage, do not claim trained-model performance.

### Phase 5 --- training notebook/scripts

Only after Phases 1--4 pass: - train on NYU; - validate on held-out NYU
scenes; - save checkpoints/configs; - compare with classical and learned
baselines; - test KITTI generalization if data/time permit.

**Acceptance gate:** Training and validation logs, saved checkpoint,
actual metric results, and reproducible inference.

### Phase 6 --- hybrid, ablation and failure analysis

-   integrate confidence/geometry fusion;
-   run ablations;
-   record at least three actual failure cases;
-   compare all methods at all scales and both degradation protocols.

**Acceptance gate:** Actual result tables and saved visualizations.

### Phase 7 --- frontend, backend and deployment

-   build HTML/CSS/JS UI;
-   implement FastAPI endpoints;
-   integrate classical and trained model inference;
-   add loading/error states and input validation;
-   deploy and test from phone/laptop.

**Acceptance gate:** Public URL works and a judge can upload supported
inputs and obtain results.

### Phase 8 --- submission audit

-   verify all mandatory requirements;
-   ensure diagrams match code;
-   complete README and design document;
-   complete `eval_P4_T69.md`;
-   check GitHub for secrets and large accidental data uploads;
-   rehearse the three-minute demo.

------------------------------------------------------------------------

## 13. Instructions to the coding assistant / ChatGPT working in VS Code

Follow these rules strictly:

1.  Treat this file as the project specification and the official P4
    requirements as mandatory.
2.  First inspect the repository and environment. Summarize what exists
    before editing.
3.  Build the project in small phases, with working code and tests at
    each phase.
4.  Do not generate the entire repository as untested placeholder files.
5.  Do not start model training before the Phase 1--4 acceptance gates
    pass.
6.  Do not silently omit any mandatory classical method or P4
    requirement.
7.  Do not invent dataset files, pretrained weights, results, citations,
    or benchmark numbers.
8.  Do not claim a baseline architecture has been reproduced unless its
    actual implementation and protocol have been verified.
9.  Do not download huge datasets or model weights without first
    explaining storage/runtime requirements and getting confirmation.
10. Keep data paths, model settings, and experiment parameters in
    configuration files.
11. Add unit tests for preprocessing, masks, interpolation, filtering,
    metrics, losses, fusion, and API input validation.
12. Keep CPU execution possible for classical methods and small smoke
    tests.
13. Provide exact commands to install dependencies, run tests, launch
    the API, and launch the frontend.
14. After each phase, report: files created/modified, commands run,
    tests passed/failed, known issues, and the next acceptance gate.
15. If a design decision is uncertain or conflicts with the official
    specification, stop and ask rather than silently guessing.
16. Never fill result tables with fabricated values. Use `Not run` until
    experiments are complete.
17. Prioritize a correct, complete, reproducible implementation over
    unnecessary complexity or decorative features.

### First task to execute

**Only perform Phase 0 and Phase 1 initially.**

Inspect the current VS Code workspace, report the environment and
existing files, propose the repository structure and dependency plan,
then implement the data pipeline and its tests. Do not train a model
yet. Do not move to Phase 2 until Phase 1 is tested and accepted.

------------------------------------------------------------------------

## 14. Final project principle

The goal is not merely to produce an impressive architecture diagram.
The final application, code, model, experiments, metrics, documentation,
and viva explanation must all agree.

**Mandatory compliance first. Measured research contribution second. UI
polish third.**

A feature is complete only when it is implemented, tested, and
demonstrable.
