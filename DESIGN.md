# TRIDENT-Depth 2.0 design

## Problem and implemented system

P4 requires dense depth reconstruction from LR or sparse/noisy/incomplete depth with RGB-guided
boundary preservation. Classical methods and versioned learned models are reusable independently.
[The architecture diagram](docs/architecture.md) distinguishes application dispatch from frozen
scientific implementations. See [official requirement mapping](docs/pdf_requirements_audit.md).

The original residual encoder-decoder uses 32/64/128 encoder channels, nearest decoder interpolation,
skips and metric residual correction. B0 and extended_v1 retain 473,059 stored parameters. Extended
training freezes the unused uncertainty head and supervises depth-boundary logits. No confidence
calibration is claimed. Exact architecture/loss/training contracts are preserved in
[extended design](docs/extended_v1.md); this submission phase changes no model or training code.

## Application contract

`backend.engine.ApplicationEngine` dispatches classical/legacy inference or exact-hash extended
weights. Classical mode does not require Torch. Extended raw deep inference reproduces original
bicubic plus nearest-valid hole initialization and feature packing. Extended hybrid calls the
original `ExtendedInference` adapter, including depth-supported classical/geometry fusion and
protocol-specific consistency. There is no uncertainty-based fusion substitution.

Deep and hybrid remain distinct: raw deep output is not silently replaced by hard measurement
retention. Clean hybrid completion retains measured points; noisy completion does not hard-copy
noisy inputs. Dense hybrid preserves LR block means. Completion stays on the target grid.
Learned serving requires registered RGB; missing guidance can use classical fallback chosen by the
user. Unsupported B0 modes fail clearly instead of extrapolating the dense4 control.

## Validation, privacy and resource bounds

Units are explicit; invalid or masked measurements are zeroed by the existing data contract.
Uploads are in memory, limited to 8 MiB each and at most one megapixel. NPY headers are checked
before allocation and object arrays are rejected. Alignment and target dimensions are checked.
A single-process request gate returns 429 while busy. No request uploads or predictions are written
to the repository. Health output omits checkpoint filesystem paths and historical training metadata.
Accuracy requires supplied GT and uses the original metric implementation and masks.

The UI uses server capabilities rather than a blanket trained-model claim, reports network/backend
errors, shows a loading state and hides stale results during a new request. It displays depth and
optional RGB with numeric downloads. App latency excludes upload parsing and visualization export;
it is distinct from the repeated, synchronized scientific benchmark timing.

## Scientific evidence and failure analysis

All protocols use the same scene-safe NYU development partition (672/123). The official test set
is locked. The independent benchmark recovered 854/861 records: six tasks have 123 frames,
and sparse_noisy_holes has 116. The [partial audit](docs/partial_independent_recovery_20261107/report.md)
verifies saved sums separately from training-time validation; no missing records are fabricated.
Incomplete/combined completion remains much weaker than dense SR.
Nearest filling cannot establish missing geometry; measurement loss has no measurements inside
large holes. Local context and limited hole diversity are hypotheses, not proven causes.

The final report must inspect central-hole and observed/missing errors, distance profiles,
correction quality, RGB-texture artifacts and representative/worst scenes. High whole-image SSIM
can hide severe hole-interior failure. Populate at least three actual independent failure cases
from saved benchmark evidence; do not substitute aggregate statistics for cases.

## Deployment and remaining gates

The Docker recipe serves the same frontend/API, runs non-root, supports PORT and optional CPU
Torch, and excludes datasets, checkpoints, credentials and ZIP archives. A read-only checkpoint
mount enables learned serving. Docker's engine is unavailable, so local container execution
remains unverified. The existing Modal public endpoint passed real-checkpoint synthetic
HTTP/browser tests; automatic GitHub deployment is prepared but has not run.
Exact checkpoint state is now audited locally; synthetic app checks do not establish NYU accuracy.
The app is a reconstruction tool, not a safety-critical ranging system; arbitrary unregistered or
out-of-distribution uploads can fail even when the file format is valid.
