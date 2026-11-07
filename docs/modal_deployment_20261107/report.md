# Verified Modal deployment - 2026-11-07

Live application: https://dhanush1756--trident-depth-t69-web.us-east.modal.run

The public FastAPI/frontend deployment passed HTTP and isolated Chrome browser checks.
The exact extended_v1 epoch-12 checkpoint was SHA256-verified locally, after private
Volume upload, and before application loading. The Volume is mounted read-only.
Six classical methods, learned reconstruction and hybrid reconstruction passed synthetic
upload, visualization and numeric-download checks. Fourteen small learned/hybrid protocol
checks and one 240x320 hybrid size check passed. See the accompanying JSON reports.

No scientific accuracy results are claimed. These checks used synthetic inputs only,
not NYU training/validation/test frames. No Colab runtime was accessed. All 104 protected
source/config/notebook files match pre-review hashes. Existing evidence remains intact.

Resources: 2 physical CPU cores, 4 GiB RAM, no GPU, maximum one container, one concurrent
request, scale-to-zero after 30 idle seconds. Estimated requested-resource compute cost
is about $0.1263 per active hour before credits, build/egress/extra usage. This is not a
measurement of actual account charges. Cold starts can delay the first request.

Public browser checks covered loading indicators, rendered results, NPZ downloads,
wrong-RGB errors, missing-RGB client validation, completion scale locking, keyboard focus,
named controls and 1440/390/360-pixel layouts without horizontal overflow. Full WCAG and
physical-device reviews remain pending. Public model-file URL probes returned 404.

The Linux Modal image built successfully. Docker remains statically reviewed only.
The earlier preparation report is historical and has not been overwritten.
Nothing was staged, committed or pushed.
