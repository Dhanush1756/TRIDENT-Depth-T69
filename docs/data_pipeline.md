# Data contracts and protocols

Use a JSON manifest containing a nonempty list. Paths are relative to the
manifest's own directory, so the same tree works on Windows and Colab/Kaggle.
Absolute local paths also work but are less portable. No dataset download is
triggered by loading a manifest.

```json
[
  {
    "id": "scene001-frame001",
    "scene_id": "scene001",
    "split": "train",
    "rgb": "rgb/001.png",
    "depth": "depth/001.npy",
    "mask": "masks/001.png",
    "depth_scale_to_m": 1.0
  }
]
```

This is a schema example, not a claim that these dataset files exist. Supported
splits are `train`, `validation`, `test`. Optional `rgb` and `mask` may be omitted.
Every record requires an explicit positive `depth_scale_to_m`; no unit inference
or per-image min/max normalization is applied. Use the source dataset's documented
encoding. PNG/TIFF/NPY depth stays numeric; RGB must be an RGB-mode image.
Depth colorizations and disparity are not accepted as metric depth. Never convert
Middlebury disparity without calibration and an explicit documented conversion.

Canonical arrays are float32 H×W depth in metres, bool H×W validity and optional
uint8 H×W×3 RGB. Nonfinite, nonpositive, out-of-range or externally masked depth
is invalid and stored as zero. Binary external masks accept 0/1/255. Input masks
and reference masks remain distinct. All-invalid samples are supported for
completion but cannot supply evaluation metrics or dense SR supervision.
RGB shape equality checks do not prove camera registration: the dataset preparer
must ensure calibration, alignment and frame synchronization.

## Protocol A: dense depth super-resolution

Reference must be fully valid. Downsampling takes nonoverlapping area means at
2×, 4× or 8×; target height/width must be divisible by scale. No implicit crops
or reference-hole filling occur. Select a documented valid crop yourself when
appropriate. RGB stays at target resolution. For example, 480×640 target at 4×
produces 120×160 input. Reconstruction returns to 480×640.

## Protocol B: synthetic sparse/noisy completion

Input stays at reference resolution; scale is recorded as a conditioning factor,
not a claim of spatial upsampling in this protocol. Uniform Bernoulli sampling
uses requested probability `density`. Grid sampling uses stride
`round(density**-0.5)` with a seeded random phase, so its realized density is
approximate. Both intersect reference validity. Density zero yields no samples.
A central rectangular hole removes approximately `hole_fraction` of the image
area. Gaussian noise has standard deviation `noise_std_m`, only at retained
observations. Samples pushed outside the configured range become invalid;
they are not clipped into fake reliable measurements. Metadata records realized
valid fraction. These sampling patterns are **not real LiDAR simulation**.

## Reproducibility and leakage

Randomness uses a stable SHA-256 seed derived from `(global_seed, sample_id)`,
independent of dataset iteration order. Degradations are fixed across epochs and
methods for reproducible comparison. A future augmentation pipeline may vary an
explicit epoch seed; it must keep test degradations fixed.

Every manifest is audited before split selection. Duplicate IDs, scene overlap
and shared file paths across splits are rejected. `assign_scene_splits` makes a
deterministic scene-level partition for custom development. Small collections may
produce empty splits because proportions are floored: inspect counts. Official
NYU/KITTI partitions take precedence; don't generate a new partition for an
official test set. File-path checks cannot detect duplicate content under
different filenames or incorrectly labelled scene IDs. Audit exported data and
record official split provenance before training. Test data never enters training.

## Resizing

`target_size` is an explicit optional `[height,width]`. The same resize is applied
to RGB/depth/masks. RGB is bilinear; depth is bilinear numerator divided by
bilinear validity support, with nearest-neighbour output validity. This prevents
zero-valued holes from diluting valid depths. It is a preprocessing operation,
not edge-preserving reconstruction; avoid unreported aspect-ratio changes or
resizing reference depth in benchmark protocols. The official labelled NYU
MAT/HDF5 release has a dedicated converter; see `nyu_preparation.md`.
KITTI native directory integration remains unverified. Export aligned pairs with
known units and official scene labels first when using other source layouts.
