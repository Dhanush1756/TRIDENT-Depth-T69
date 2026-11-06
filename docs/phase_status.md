# Phase audit — 6 November 2026

The latest user instruction authorizes official NYU dataset preparation and
classical validation benchmarks. The labelled release and split download were
explicitly approved. Neural training still requires separate explicit approval.
Code preparation is distinct from passing an experimental acceptance gate.

| Phase | Implementation and verification | Remaining gate |
| --- | --- | --- |
| 0: inspect | Repository, tools, RAM, disk, packages, PDF and Git remote inspected | GPU hardware identification unavailable through restricted WMI; CPU Torch reports no CUDA |
| 1: data | Official NYU MAT/HDF5 inspected and all 1,449 frames converted exactly; 7,245 exported files audited; official 795/654 split preserved; scene-disjoint 672/123/654 development partition; both protocols verified at 2/4/8 | KITTI native integration remains pending |
| 2: classical | All nine methods implemented and unit-tested; full 480×640 NYU validation outputs saved for both protocols and all scales | Full held-out test and KITTI evaluation remain pending |
| 3: evaluation | 288 real NYU development-validation records on eight frozen scenes; accuracy, SSIM, boundaries, latency and scoped memory saved; all saved RMSE/MAE independently audited | Final test-set and trained-model comparisons not run |
| 4: models | Explicit CNN baseline; residual encoder-decoder; uncertainty; losses; fusion; inference guard; gradient and checkpoint tests | Models remain untrained; DKN/FDKN not reproduced |
| 5: training | NYU split and baseline audit; 12 protocol/scale/model configs plus a two-epoch pilot; finite CPU gradient probes; epoch-boundary RNG/optimizer resume; validation-RMSE best-checkpoint selection; GPU budget plan prepared | **Not run**: awaiting explicit pilot approval and verified GPU runtime; GPU peaks and resume not measured |
| 6: study | Checkpoint-driven comparison, six inference ablations, ranked diagnostic panels implemented | Actual trained-model ablations and three independent real failure cases **not run** |
| 7: application | FastAPI + vanilla HTML/CSS/JS; uploads, alignment checks, 2/4/8, loading/errors, numeric downloads, optional GT metrics; API tests and actual localhost HTTP upload/download passed | No trained checkpoint; public deployment, phone/browser QA and Docker execution unverified |
| 8: submission | README, implemented architecture, design, eval file and requirements audit prepared | Final experiments, public URL, source publication and viva remain pending |

## Environment observed

- Windows build 26300, x64; Python 3.10.8, Git 2.46.0, Node 20.17.0.
- AMD64 processor, 16 logical CPUs; approximately 15.34 GiB RAM.
- Approximately 97.43 GiB free disk space during inspection; free RAM was about 1.24 GiB.
- Global Python has incompatible OpenCV/NumPy, googletrans/httpx and TensorFlow/protobuf requirements. A separate `.venv` avoids these conflicts.
- The Windows `py` launcher does not discover the installed Python; use `python` or the explicit `.venv` executable.
- CPU-only PyTorch is used for architecture/gradient tests. This does not prove that the physical machine lacks a GPU.
- Docker CLI is installed, but its daemon is not running; container build/run could not be verified.
- No existing project source was overwritten. `.gitignore` was read before appending project artifact exclusions. The existing `.vscode/extensions.json` was preserved.

## Real-data smoke check

`scripts/fetch_smoke_sample.py` retrieved a 3,772,513-byte Open3D sample archive,
verified its upstream MD5 and recorded SHA-256. Only one Redwood RGB-D pair and a
documented fully valid 64×64 crop were retained under ignored `data/`.
The crop starts at y=16, x=56, spans 1.357–2.702 metres and is converted from
16-bit millimetres with factor 0.001.

Dense 4× export: input float32 depth and bool mask 16×16; target float32 depth
and bool mask 64×64; RGB uint8 64×64×3. Sparse export: input and target 64×64;
observed fraction 0.045166015625 with the committed sparse configuration.

This is sensor-reference I/O testing, **not NYU/KITTI benchmark evidence**. The
same scene at multiple scales is not multiple independent failure cases.
Data, generated outputs, virtual environments and weights are Git-ignored.

Reference: [Open3D RGB-D format documentation](https://www.open3d.org/html/tutorial/geometry/rgbd_image.html).

## Verification recorded

- `.venv/Scripts/python.exe -m pytest -q`: **112 passed**, retaining the original
  99 tests and adding 13 NYU checks. One upstream Starlette
  warning recommends its newer optional `httpx2` test transport; current tests
  succeed with the pinned httpx client.
- `.venv/Scripts/python.exe -m pip check`: **no broken requirements**.
- `git diff --check`: passed.
- JavaScript syntax check: `Get-Content frontend/app.js -Raw | node --check` passed.
- Latest real-file benchmark artifacts: `artifacts/final_check/dense` and
  `artifacts/final_check/sparse`, 18 records each. Includes measured traced
  allocation peaks and actual OS lifetime peak RSS, clearly scoped.
- `artifacts/final_check/diagnostics`: three scale-specific panels of one scene;
  not three distinct failure cases.
- Actual Uvicorn HTTP check: frontend and API returned 200; a real Redwood crop
  reconstructed to finite 64×64 metric depth and its NPZ download round-tripped.
  Evidence is in ignored `artifacts/http_smoke.json`.
- The approved official NYU labelled dataset and split were downloaded and
  converted; source URLs, byte counts and SHA-256 hashes are recorded.
- No optimizer/training run, source publication or public deployment was
  performed. Backpropagation tests only validate gradients.

## Official NYU preparation

The native release is MATLAB 7.3/HDF5. All decoded RGB/depth pairs are 480×640.
RGB and valid depth exports round-trip exactly, with zero valid-depth error.
Official training/test partitions contain 249/215 disjoint scenes. The separate
development split contains 219 training and 30 validation scenes while leaving
all 215 test scenes untouched. Original scene strings and original file names
are preserved in every record.

Filled-reference depth ranges from 0.7132995 to 9.9954700 m, with no invalid
pixels. Projected raw sensor depth has 31.4338% missing pixels. Neither field was
resized, warped or quantized. Eight scene-diverse validation frames were fixed
before method scores were measured. Registration panels were visually inspected:
no gross axis/frame mismatch was observed; source sensor/occlusion errors remain.

See `nyu_preparation.md` for conversion, data semantics and reproducible commands.
Full per-frame integrity details remain in `data/nyu/converted/integrity_report.json`.
Auditable dataset summary: `nyu/integrity_report.md`. Measured validation tables:
`nyu/validation_results.md`; full per-image results and provenance are alongside
the tables as JSON/CSV. Both protocols completed 144 records. Independent saved
prediction audit checked all 288 files, source references and input masks, with
zero difference in recomputed RMSE/MAE and no validation/test leakage.

The current classical refinement has higher dense-SR macro-average RMSE than
bicubic at all three scales on this small subset. Its boundary F1 improves over
bicubic at 4× and 8×. This trade-off is recorded, with no parameter changes to
hide the outcome. Sparse completion is separately labelled synthetic; repeated
scale conditioning does not imply spatial upsampling of its input.
