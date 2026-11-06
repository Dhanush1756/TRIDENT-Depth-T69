# TRIDENT-Depth 2.0 — T69 design

## Problem and scope

P4 requires reconstructing high-resolution depth while preserving boundaries,
handling sensor noise and missing values, and combining classical and learned
methods. The application accepts metric depth and optional registered RGB.
The research question concerns confidence/geometry-guided residual fusion;
its benefit is unproven until controlled experiments exist.

## Data and interfaces

See `data_pipeline.md` for arrays, units, splits and separate degradation
protocols. `DepthDataset` loads lazily. `TorchDepthDataset` prepares classical
initialization while retaining both observation and target masks. The official
labelled NYU MAT/HDF5 has a dedicated converter preserving its original split;
see `nyu_preparation.md`. KITTI native integration remains unverified. The
application accepts PNG/TIFF/NPY numeric maps and PNG/JPEG RGB. Validity is
explicit, not inferred from visualization colors. Shape checks alone cannot
prove image registration.

## Classical implementation

Interpolation normalizes by resized validity support. Cubic overshoot is
clamped to the observed range. Denoising uses a valid-only local lower median
and retains the original mask. Hole filling uses the nearest valid measurement
with an optional maximum distance; empty masks remain empty. Joint bilateral
weights combine spatial distance and normalized guidance similarity, with
invalid depth excluded from numerator and denominator. Guided filtering uses
RGB luminance in a mask-weighted local linear model; it is not a full multichannel
color guided-filter implementation. Boundary refinement applies guidance only
where depth-discontinuity evidence supports RGB edges. This simple gate does
not establish texture rejection on challenging real scenes.

## Models and losses

See `architecture.md`. Training uses a named lightweight CNN baseline and
custom residual encoder-decoder. Masked L1, valid-neighbour gradient loss,
ground-truth-depth edge weighting, Laplace NLL and measurement consistency are
implemented. Laplace uncertainty uses `b=exp(s)` and loss `abs(error)/b+s`,
with s clamped to [-6,6]; this is log **scale**, not log variance. This corrects
the ambiguous example parameterization in the specification. Initial loss
weights remain exploratory. Dense measurement loss compares the area-pooled
prediction with low-resolution input, matching the degradation operator.
No real training has been executed.

## Fusion and consistency

Depth confidence decays exponentially with distance from observed support.
Geometric confidence combines RGB gradient magnitude and local depth edges.
Uncertainty modulates neural trust; positive candidate weights normalize to one.
These are hand-defined rules for evaluation, not learned or validated routing.
For same-resolution completion, reliable observations are hard-projected;
noisy observations receive 0.25 soft consistency. Dense SR does not hard-project
LR values onto arbitrary HR pixel locations. Dense consistency is enforced by
the matching area-pooling training loss. Routing/soft-consistency parameters
must be tuned on validation data only.

## Evaluation and ablations

See `evaluation_protocol.md`. The harness exports per-sample results and arrays.
Classical/deep/hybrid comparisons use identical deterministic degradations.
Inference ablations cover confidence, geometry, residual, uncertainty, adaptive
fusion and measurement consistency. Training ablations need separate runs.
The diagnostic renderer does not invent failure explanations. Final failure
analysis must include distinct real examples and human-reviewed causes.

## Baseline investigation

The [official DKN repository](https://github.com/cvlab-yonsei/dkn) publishes DKN
and FDKN under GPL-3.0 and lists PyTorch 1.0/CUDA 10/OpenCV 3.4 dependencies.
Its provided dataset split, code and weights have not been reproduced on this
Windows CPU environment. No code or weights were copied. The implemented
fallback is explicitly `LightweightCNN`; it is not a DKN reproduction. An
established-model comparison remains a research limitation.

## Application and deployment

FastAPI serves the frontend and API on one origin. Uploads are processed in
memory after multipart spooling. A single-process lock rejects concurrent
reconstructions with HTTP 429. Each input is capped at 8 MiB and 1 megapixel;
target resolution is capped at 1 megapixel. NumPy headers are inspected before
allocating arrays, object arrays are rejected, and image dimensions checked
before decoding. Public hosting should add reverse-proxy request-size limits
and rate limits. Users receive shared-range visualizations, runtime and numeric
NPZ/16-bit-mm PNG downloads. Accuracy metrics require a supplied valid reference.
PNG downloads are omitted above the 16-bit millimetre range to avoid clipping.

The Docker image runs as an unprivileged user. Classical operation needs no
checkpoint. Optional CPU Torch installation and a mounted trained checkpoint
enable neural methods. A Python-capable host is required; Cloudflare Worker/Sites
hosting cannot execute this backend. No hosted URL has been created. Docker
execution and browser/phone QA remain unverified.

## Reproducibility, limitations and ethics

Configuration, seeds, manifest hashes, model names, epoch logs and checkpoint
provenance are recorded. Checkpoints cannot be presented as trained merely
because forward passes work. Windows CPU and GPU notebook workflows share
portable manifest paths. Dataset licenses and official splits must be audited
before use; sensor RGB may include people and private rooms. Do not upload
sensitive imagery to a public demo. Outputs are research estimates and are not
validated for navigation, robotics safety or clinical use. Nearest filling can
bleed across boundaries; geometry gating can reinforce an incorrect classical
edge; large unobserved regions are underdetermined. The complete pipeline has
not yet been scientifically evaluated.

## Submission template limitation

The PDF refers to a design-document template without including its sections.
This document covers the available specification; the organizer's template is
still needed for final compliance review.
