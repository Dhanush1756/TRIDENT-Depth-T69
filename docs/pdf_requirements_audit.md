# Official P4 audit

Read source: `Computer Vision Hackathon 2026.pdf`, local Downloads copy,
16 pages. P4 is on pages 8–9; dataset links on page 14; general deliverables
and rubric on pages 15–16. The project specification supplies additional
research and workflow details and does not replace the PDF.

Audited PDF SHA-256:
`335d8d41f05054174ced7bfd254bdb06ef1002802e785bfe3962deae38d18015`.

| PDF requirement | Code/documentation | Verification status |
| --- | --- | --- |
| Nearest, bilinear, bicubic | `trident_depth/classical/interpolation.py` | Unit tests; real smoke outputs |
| Joint bilateral | `classical/joint_bilateral.py` | Unit tests; real smoke outputs |
| Guided filtering | `classical/guided_filter.py` | Unit tests; real smoke outputs |
| Edges and boundary refinement | `classical/edge_detection.py` | Unit tests; geometry gating only, benefits unproven |
| Depth discontinuities | `classical/discontinuity.py` | Known-step and invalid-neighbour tests |
| Denoising and hole filling | `classical/denoising.py`, `classical/hole_filling.py` | Outlier, distance-cap and empty-support tests |
| Network and RGB guidance | `models/baseline.py`, `models/residual_cnn.py` | Forward/backward only; untrained |
| Dataset adaptation | `training/train.py` | Runner implemented; training **not run** |
| Classical + learned fusion | `models/fusion.py`, `inference.py` | Formula, shape and consistency tests; trained behavior unverified |
| Sparse/noisy inputs | `data/degradation.py` | Separate protocol and numeric/mask tests |
| Texture copying / oversmoothing prevention | Geometry gate and residual formulation | Hypotheses; require real trained-model study |
| 2× / 4× / 8× | Config, degradation, reconstruction, benchmark, UI | Dense spatial upsampling tested; sparse completion remains same-resolution |
| Compare interpolation/classical/deep/hybrid | `evaluation/benchmark.py` | Real NYU classical validation measured; deep/hybrid pending checkpoints |
| Unseen maps and boundary performance | Held-out manifest + boundary metrics | Eight preselected NYU development-validation scenes measured; final test evaluation **not run** |
| RMSE/MAE/PSNR/SSIM | `evaluation/metrics.py` | Known-value and mask tests |
| Edge preservation / boundary accuracy | Gradient similarity and boundary P/R/F1 | Definitions and numeric tests |
| Scale performance / latency / memory | `evaluation/benchmark.py` | NYU dense and sparse protocols measured at all scales; memory/timing scopes documented |
| Hosted app | `backend/`, `frontend/`, `Dockerfile` | Local API tests; no public URL |
| GitHub complete source | Local source; existing origin remote | Publication not verified |
| Architecture and design | `docs/architecture.md`, `docs/design_document.md` | Match implemented modules; final trained design pending |
| Evaluation file and USNs | `eval_P4_T69.md` | Correct filename, first line and metric order; final results not run |
| Ablation and failure analysis | Benchmark `--ablations`, `failure_analysis.py` | Tools exist; final experiments not run |
| Three-minute viva | `docs/demo.md` | Rehearsal pending |

## Rubric ambiguity

The prose on page 15 says demo/viva has 25 points and the other five criteria
have 15 each. The table instead lists evaluation 25, implementation 20,
UI/deployment 15, documentation 15, demo/viva 25 (100 total). Bands for some rows
also top out at 15 despite larger row maxima. This is an internal PDF
inconsistency; no weighting is silently chosen for project decisions. Ask the
organizers to clarify before scoring a submission.

The PDF mentions a design-document template but does not include the template's
sections. The current design document cannot certify compliance with an unseen
template. The template is an outstanding submission input.
