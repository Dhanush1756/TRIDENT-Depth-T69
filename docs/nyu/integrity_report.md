# Official NYU dataset integrity report

Samples: **1449**. Every decoded RGB and depth frame is **480x640**.

| Partition | Frames | Scenes |
| --- | ---: | ---: |
| Official train | 795 | 249 |
| Official test | 654 | 215 |
| Development train | 672 | 219 |
| Development validation | 123 | 30 |
| Untouched test | 654 | 215 |

The official and development manifests are separate. Development validation comes only from whole official training rooms. No scene overlap was detected.

| Depth field | Positive range (m) | Missing/invalid fraction | Nonfinite | Negative | Above 10m |
| --- | --- | ---: | ---: | ---: | ---: |
| Filled reference | 0.713299513 to 9.995470047 | 0.000000000% | 0 | 0 | 0 |
| Projected raw sensor | 0.713299513 to 10.000000000 | 31.433833903% | 0 | 0 | 0 |

Each field contains 445,132,800 source pixels. Raw zero/missing pixels: 139,922,305.
Checked exported files: 7,245; missing: 0; shape/mask errors: 0; exact RGB+depth duplicates: 0.
RGB round-trip: exact. Maximum valid depth round-trip error: 0.0 m.
Exported RGB/depth/mask bytes: 4,977,218,731.

The conversion only transposes MATLAB/HDF5 axes. It performs no interpolation, resizing, crop, warp or depth quantization.
Representative registration panels and visual-review limitations are in `registration_review.json`.
Source URLs and measured file hashes are in `download_provenance.json`; source layout is in `source_inspection.json`.
Manifest hashes, selected validation rooms, source types and aggregate checks are in `integrity_summary.json`.
Full per-frame statistics remain in `data/nyu/converted/integrity_report.json`.

Filled depth includes NYU's original inpainting. This report verifies data integrity, not ground-truth sensor accuracy. No training was run.
