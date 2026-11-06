# Official NYU Depth V2 preparation

Neural training is not authorized and has not been started. This workflow only
downloads, inspects, converts and validates data, and runs classical methods.

## Source release and provenance

Use the labelled release from the [official NYU page](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).
The approved download and original split come directly from NYU:

- `https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat`
- `https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat`

The downloaded MAT is 2,972,037,809 bytes, SHA-256
`2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa`.
The split file is 2,626 bytes, SHA-256
`6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee`.
These are measured local hashes of files retrieved from the official host, not
independently published NYU checksums. Download URLs, sizes and hashes are
retained in `data/nyu/source/download_provenance.json`.

No full raw-video release or model weights were downloaded. Dataset citation
and source usage terms remain applicable; the project does not grant a new
license over the dataset or redistribute its full contents.

## Actual inspected format

The labelled file is MATLAB 7.3/HDF5, with compressed datasets. Inspection found:

| Field | HDF5 shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `images` | 1449 × 3 × 640 × 480 | uint8 | RGB values 0–255 |
| `depths` | 1449 × 640 × 480 | float32 | Official filled/inpainted metric depth |
| `rawDepths` | 1449 × 640 × 480 | float32 | Projected sensor depth, missing observations retained |
| `scenes`, `sceneTypes` | 1 × 1449 references | MATLAB cells | Exact source scene name and type per frame |
| `rawRgbFilenames`, `rawDepthFilenames` | 1 × 1449 references | MATLAB cells | Original RGB/depth filename provenance |

Decode RGB with C,W,H → H,W,C and depth with W,H → H,W. Every frame is
480×640 after this transpose. Both depth fields are documented as metres in
the official release. Values are not divided by 1000 or normalized per image.
Raw Kinect disparity conversion and additional camera reprojection are not
appropriate here: `rawDepths` has already been projected and converted by NYU.

The split is a separate MATLAB file read with `scipy.io.loadmat`.
`trainNdxs` and `testNdxs` are 1-based uint16 column vectors. Measured counts
are 795/654, covering all 1,449 frames with no duplicate or overlapping indices.
Their 249/215 source scenes have no overlap.

## Conversion and masks

The converter streams one frame at a time. It exports RGB to lossless PNG,
filled and raw depth to float32 NPY, and separate bool NPY masks. Positive
finite values are preserved exactly; invalid depth values become zero and are
reported before sanitization. Values above 10 m are counted and preserved, not
silently clipped. Experiment range policies remain in configuration files.
Every RGB and valid depth value is compared with its decoded source after
export. No resize, crop, hole filling or geometry transformation is introduced.

`manifest_official.json` retains official train/test membership.
`manifest_development.json` adds a development validation holdout exclusively
from official training scenes, preserving `official_split` in every record.
The original scene string is retained as `source_scene`; conservative room
grouping also protects against hypothetical a/b recording suffix leakage.

The validation selection operates on whole rooms within scene type: sort room
IDs, hold out approximately 10% using evenly spaced positions, at least one
where a type has two or more rooms, and never its only room. This is an explicit
derived development split, **not an official NYU validation partition**. There
is no random frame split. Official test frames remain test-only.

The frozen benchmark subset chooses one median-index frame per room,
round-robin over validation scene types ranked by sample prevalence. Selection
is completed before method scores are measured. The exact IDs and manifest
SHA-256 are saved. This small subset is a development check, not a substitute
for a full held-out test evaluation.

## Integrity and registration

The integrity report includes per-frame and aggregate depth/invalid-value
statistics, shape/dtype checks, split/scene counts, all referenced file checks,
exact RGB+depth duplicate checks, source hashes and manifest hashes. Exact
duplicates across official partitions fail the conversion. Near-duplicates
cannot be ruled out by hashes alone; source scene separation is retained.

RGB/depth axis correspondence is checked against the actual source for all
exports. Registration panels show RGB, filled depth, projected raw depth and
raw depth discontinuities over RGB on the preselected validation frames.
These preserve NYU's supplied registration. Visual checks do not provide a
new subpixel registration calibration or prove that all RGB texture edges are
depth boundaries.

## Reproduce on Windows

Use a new destination when repeating conversion or benchmarks; existing output
directories are never overwritten.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-nyu.txt
.\.venv\Scripts\python.exe scripts/download_nyu.py
.\.venv\Scripts\python.exe scripts/inspect_nyu.py
.\.venv\Scripts\python.exe -m trident_depth.data.nyu
.\.venv\Scripts\python.exe -m trident_depth.evaluation.nyu_preparation
.\.venv\Scripts\python.exe scripts/audit_nyu_benchmarks.py
.\.venv\Scripts\python.exe scripts/export_nyu_reports.py
.\.venv\Scripts\python.exe -m pytest -q
```

The downloaded source is reused when already present. The other commands
refuse to overwrite their completed output paths; `inspect_nyu.py`, converter
and benchmark accept `--output` for a new run. The benchmark uses one warmup
and three timed repetitions per method/sample/scale, saves the six classical
methods separately and keeps dense SR distinct from synthetic sparse/noisy
completion. Completion remains same-resolution; its scale is conditioning
metadata. See `evaluation_protocol.md` for metric and memory definitions.
The current validation comparison uses full 480×640 images without an Eigen
crop, fixed 10 m PSNR/SSIM range, and the official filled-reference validity mask.
Do not compare its values directly with papers using other crops, resizing,
depth ranges, depth encodings, degradations or metric conventions.
Inspect both registration sheets and record the outcome and limitations in
`registration_review.json` under the benchmark directory. If that review is
absent on a reproduced run, report export labels visual registration review as
pending rather than claiming it passed. The current run includes an actual
eight-frame visual review.

Full data and arrays stay in ignored `data/` and `artifacts/`. Small auditable
source/integrity summaries and measured JSON/CSV results belong in `docs/nyu/`.
There are no training commands in this workflow.
