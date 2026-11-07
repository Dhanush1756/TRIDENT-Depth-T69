1RUA24CSE0125,1RUA24CSE0129,1RUA24CSE0140,1RUA24CSE0178

# P4 - TRIDENT-Depth 2.0 - Team T69

## Submission status and provenance

Extended_v1 completed twelve training epochs. The recovered independent benchmark contains **854/861** saved task/frame records: six tasks have 123/123; sparse_noisy_holes has **116/123**. The evaluator failed during output saving. This is not a completed seven-task benchmark. Official test samples remain locked.
Live application URL: https://dhanush1756--trident-depth-t69-web.us-east.modal.run
Public HTTP/browser functionality checks passed with the exact epoch-12 checkpoint on CPU;
these synthetic checks do not supply scientific evaluation metrics.
Repository origin: https://github.com/Dhanush1756/TRIDENT-Depth-T69 (pending local changes must
be reviewed and published; current remote synchronization is not verified).

Checkpoint: epoch_012.pt, SHA256
`09bb8fc5edfff40c7fe27a39d44eb571d02726e0bb22a78c28afcd7f0bd844bd`.
NYU development split: 672 training frames / 123 validation frames, disjoint scenes.
All seven tasks use the same 123 validation IDs; there are 861 protocol/frame pairs.
Completion remains target resolution, not a spatial 4x claim.

## Recovered partial independent validation results - 854/861

| Task | Frames | Method | RMSE m | MAE m | PSNR mean dB | SSIM | Boundary P | R | F1 | Mean frame-median ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense2 | 123/123 | nearest_fill | 0.044431854 | 0.009577075 | 50.773503935 | 0.995293653 | 0.900224865 | 0.961360011 | 0.929110502 | 36.411 |
| dense2 | 123/123 | bilinear_fill | 0.036764275 | 0.008713918 | 52.480214848 | 0.996715324 | 0.981319039 | 0.725155326 | 0.831493321 | 38.261 |
| dense2 | 123/123 | bicubic_fill | 0.031780778 | 0.007521698 | 53.771530558 | 0.997482420 | 0.982083819 | 0.786190597 | 0.871722304 | 39.457 |
| dense2 | 123/123 | joint_bilateral | 0.041818652 | 0.011808319 | 51.119293517 | 0.995772088 | 0.956544453 | 0.711169731 | 0.812779701 | 377.519 |
| dense2 | 123/123 | guided | 0.047499125 | 0.014065094 | 50.012629675 | 0.994675271 | 0.950677204 | 0.692771784 | 0.797842096 | 378.465 |
| dense2 | 123/123 | classical | 0.042236977 | 0.011886008 | 51.056091703 | 0.995720715 | 0.956221035 | 0.709599116 | 0.811603199 | 461.652 |
| dense2 | 123/123 | extended | 0.030476068 | 0.007234403 | 54.631507014 | 0.997878025 | 0.978888263 | 0.827293202 | 0.895720305 | 83.283 |
| dense2 | 123/123 | extended_hybrid | 0.025448731 | 0.006240453 | 55.823270581 | 0.998350124 | 0.983292566 | 0.866590140 | 0.920548231 | 791.188 |
| dense4 | 123/123 | nearest_fill | 0.081895846 | 0.020880860 | 45.388798216 | 0.984075813 | 0.593219229 | 0.939960702 | 0.722368611 | 36.526 |
| dense4 | 123/123 | bilinear_fill | 0.065244272 | 0.018388610 | 47.442397952 | 0.990069600 | 0.853716034 | 0.502175504 | 0.620871647 | 39.560 |
| dense4 | 123/123 | bicubic_fill | 0.058527213 | 0.016203008 | 48.406662801 | 0.991453539 | 0.870988401 | 0.566739228 | 0.677965943 | 41.045 |
| dense4 | 123/123 | joint_bilateral | 0.064084495 | 0.019736918 | 47.491926505 | 0.990771386 | 0.882584340 | 0.582700893 | 0.692649268 | 395.675 |
| dense4 | 123/123 | guided | 0.066778354 | 0.021389721 | 47.126640755 | 0.990363266 | 0.895495224 | 0.587041417 | 0.700300081 | 396.153 |
| dense4 | 123/123 | classical | 0.064046705 | 0.019732003 | 47.506123606 | 0.990831661 | 0.882805741 | 0.582975481 | 0.692922606 | 474.239 |
| dense4 | 123/123 | extended | 0.044344922 | 0.012268089 | 51.097929922 | 0.995062199 | 0.956275142 | 0.763903245 | 0.847532450 | 86.084 |
| dense4 | 123/123 | extended_hybrid | 0.040598366 | 0.011235681 | 51.938588078 | 0.995554675 | 0.945452576 | 0.783218050 | 0.854881297 | 691.095 |
| dense4 | 123/123 | b0 | 0.045509639 | 0.012129690 | 50.949093803 | 0.994701076 | 0.941035664 | 0.726589360 | 0.817865078 | 85.413 |
| dense8 | 123/123 | nearest_fill | 0.129016003 | 0.039518573 | 41.386001375 | 0.963876699 | 0.360115876 | 0.619350092 | 0.446928853 | 35.027 |
| dense8 | 123/123 | bilinear_fill | 0.106452056 | 0.035748134 | 43.163030054 | 0.978919185 | 0.677158125 | 0.314479144 | 0.409928240 | 36.521 |
| dense8 | 123/123 | bicubic_fill | 0.097015800 | 0.032097075 | 43.966633195 | 0.980095514 | 0.724097740 | 0.360694540 | 0.458332234 | 37.945 |
| dense8 | 123/123 | joint_bilateral | 0.104791637 | 0.036140591 | 43.263077498 | 0.980483129 | 0.769782526 | 0.411123547 | 0.513044235 | 392.909 |
| dense8 | 123/123 | guided | 0.105150667 | 0.036904170 | 43.233990407 | 0.981011736 | 0.794530885 | 0.432304570 | 0.538802084 | 408.095 |
| dense8 | 123/123 | classical | 0.104646600 | 0.036112974 | 43.275992308 | 0.980593882 | 0.771175271 | 0.412793564 | 0.514821840 | 469.264 |
| dense8 | 123/123 | extended | 0.073958546 | 0.023656117 | 46.395990921 | 0.987103732 | 0.868797261 | 0.643064690 | 0.736162027 | 81.166 |
| dense8 | 123/123 | extended_hybrid | 0.072035227 | 0.022782980 | 46.735873514 | 0.987447056 | 0.857041757 | 0.645811219 | 0.732919849 | 669.584 |
| sparse | 123/123 | nearest_fill | 0.122609329 | 0.027449367 | 41.857918438 | 0.972570030 | 0.513694812 | 0.818860356 | 0.624439652 | 45.258 |
| sparse | 123/123 | bilinear_fill | 0.122609329 | 0.027449367 | 41.857918438 | 0.972570030 | 0.513694812 | 0.818860356 | 0.624439652 | 44.492 |
| sparse | 123/123 | bicubic_fill | 0.122609329 | 0.027449367 | 41.857918438 | 0.972570030 | 0.513694812 | 0.818860356 | 0.624439652 | 43.643 |
| sparse | 123/123 | joint_bilateral | 0.090029737 | 0.022331643 | 44.563343458 | 0.985585817 | 0.818803058 | 0.680915748 | 0.740645541 | 401.611 |
| sparse | 123/123 | guided | 0.084891918 | 0.022652156 | 45.072201348 | 0.987352432 | 0.852768898 | 0.644268522 | 0.729196236 | 414.951 |
| sparse | 123/123 | classical | 0.088745702 | 0.022185674 | 44.673641854 | 0.985968687 | 0.822007015 | 0.680017931 | 0.741349612 | 485.534 |
| sparse | 123/123 | extended | 0.099850634 | 0.024908580 | 43.744504743 | 0.980390595 | 0.595394329 | 0.909438046 | 0.715352439 | 92.810 |
| sparse | 123/123 | extended_hybrid | 0.096252995 | 0.024151163 | 44.008905016 | 0.981593013 | 0.598725209 | 0.905328918 | 0.716488287 | 632.231 |
| noisy | 123/123 | nearest_fill | 0.010001045 | 0.007979808 | 59.999106010 | 0.998975266 | 0.854701807 | 0.993090479 | 0.917539141 | 38.408 |
| noisy | 123/123 | bilinear_fill | 0.010001045 | 0.007979808 | 59.999106010 | 0.998975266 | 0.854701807 | 0.993090479 | 0.917539141 | 37.915 |
| noisy | 123/123 | bicubic_fill | 0.010001045 | 0.007979808 | 59.999106010 | 0.998975266 | 0.854701807 | 0.993090479 | 0.917539141 | 37.973 |
| noisy | 123/123 | joint_bilateral | 0.031636808 | 0.008697699 | 53.340501188 | 0.997429553 | 0.981599110 | 0.767883284 | 0.859553501 | 374.956 |
| noisy | 123/123 | guided | 0.038953367 | 0.011155278 | 51.630470508 | 0.996186449 | 0.963798811 | 0.747357073 | 0.839375250 | 418.623 |
| noisy | 123/123 | classical | 0.032280328 | 0.008871803 | 53.192452129 | 0.997350132 | 0.981557380 | 0.764976402 | 0.857676121 | 455.847 |
| noisy | 123/123 | extended | 0.010944585 | 0.008095313 | 59.478580885 | 0.998965950 | 0.855518818 | 0.993020428 | 0.917995919 | 82.571 |
| noisy | 123/123 | extended_hybrid | 0.011270416 | 0.008182542 | 59.266382145 | 0.998966330 | 0.863810966 | 0.991592442 | 0.922223182 | 604.316 |
| incomplete | 123/123 | nearest_fill | 0.433587709 | 0.102640359 | 31.665055790 | 0.974809843 | 0.842138520 | 0.756422519 | 0.794074780 | 36.893 |
| incomplete | 123/123 | bilinear_fill | 0.433587709 | 0.102640359 | 31.665055790 | 0.974809843 | 0.842138520 | 0.756422519 | 0.794074780 | 37.359 |
| incomplete | 123/123 | bicubic_fill | 0.433587709 | 0.102640359 | 31.665055790 | 0.974809843 | 0.842138520 | 0.756422519 | 0.794074780 | 37.206 |
| incomplete | 123/123 | joint_bilateral | 0.433555696 | 0.108126215 | 31.677577960 | 0.974774363 | 0.801280070 | 0.600625657 | 0.682103218 | 384.341 |
| incomplete | 123/123 | guided | 0.431797354 | 0.109483755 | 31.717212368 | 0.974557646 | 0.797691499 | 0.589369736 | 0.673034778 | 407.143 |
| incomplete | 123/123 | classical | 0.433402602 | 0.108193943 | 31.680032953 | 0.974773745 | 0.801744392 | 0.598453619 | 0.680841330 | 457.576 |
| incomplete | 123/123 | extended | 0.408790095 | 0.097991622 | 32.204005143 | 0.976322674 | 0.751472075 | 0.781786452 | 0.763143411 | 83.020 |
| incomplete | 123/123 | extended_hybrid | 0.408922194 | 0.097501048 | 32.205886123 | 0.976481844 | 0.755691524 | 0.781122531 | 0.764984567 | 608.522 |
| sparse_noisy_holes | 116/123 | nearest_fill | 0.452584743 | 0.127484846 | 31.269759011 | 0.954672298 | 0.423097439 | 0.610782434 | 0.491072372 | 43.732 |
| sparse_noisy_holes | 116/123 | bilinear_fill | 0.452584743 | 0.127484846 | 31.269759011 | 0.954672298 | 0.423097439 | 0.610782434 | 0.491072372 | 41.674 |
| sparse_noisy_holes | 116/123 | bicubic_fill | 0.452584743 | 0.127484846 | 31.269759011 | 0.954672298 | 0.423097439 | 0.610782434 | 0.491072372 | 40.274 |
| sparse_noisy_holes | 116/123 | joint_bilateral | 0.443146440 | 0.122120514 | 31.552180788 | 0.966034444 | 0.658578653 | 0.518867324 | 0.576461874 | 386.401 |
| sparse_noisy_holes | 116/123 | guided | 0.440203808 | 0.121297189 | 31.635240711 | 0.968025137 | 0.689705799 | 0.494749428 | 0.570850664 | 389.265 |
| sparse_noisy_holes | 116/123 | classical | 0.442783012 | 0.121965555 | 31.561381325 | 0.966365913 | 0.662089596 | 0.518873294 | 0.577747848 | 464.431 |
| sparse_noisy_holes | 116/123 | extended | 0.425872499 | 0.120073536 | 31.895308872 | 0.962183280 | 0.483614126 | 0.714650714 | 0.570930442 | 87.180 |
| sparse_noisy_holes | 116/123 | extended_hybrid | 0.425327397 | 0.119664702 | 31.914373383 | 0.963143150 | 0.486356331 | 0.710481695 | 0.571463385 | 606.683 |

RMSE/MAE are pooled over valid pixels in metres. PSNR/SSIM/boundary scores are mean per frame; latency is the mean of per-frame medians. Global PSNR, median-of-medians latency, GPU memory, eligible counts, regional errors and all 57 method/task aggregates are retained in the [technical report and data](docs/partial_independent_recovery_20261107/report.md). The combined task has 116 frames and must not be presented as 123-frame results. No additional inference was performed. Original provenance, prediction archives, panels and final replay/isolation certification remain unverified.

## Existing evidence - separate from final tables

The [288-record classical NYU comparison](docs/nyu/validation_results.md) uses eight frozen,
scene-diverse validation frames, six methods, three factors and two protocols. Those records,
A1/B0 artifacts, verification reports and checkpoint files are preserved unchanged.
The [epoch-12 log review](docs/extended_v1_epoch012_evaluation.md) records poor incomplete and
combined completion accuracy; it is not a replacement for reproduced final benchmark results.

## Ablations

A1/B0 isolate the uncertainty objective under the approved paired contract. Keep the exact
sample sets and provenance with their results. Final hybrid component comparisons and any
additional ablations: **Pending evidence review**. Do not invent a complete ablation study.

## Failure analysis

Inspect central-hole versus observed-region errors, distance-from-measurement profiles, learned
correction versus filled bicubic, depth bleeding, thin structures and RGB texture transfer.
High global SSIM can conceal a poor hole interior. Clean hybrid observed-point retention can
improve global metrics without reconstructing the missing region.

| Independent real case | Evidence and scene ID | Cause assessment | Status |
| --- | --- | --- | --- |
| Large missing region | Pending saved panels and region metrics | Separate observations from hypotheses | Pending |
| Sparse/noisy/holes | Pending saved panels and distance profile | Separate observations from hypotheses | Pending |
| Boundary/texture/thin structure | Pending actual failure selection | Separate observations from hypotheses | Pending |

Do not label these candidate categories as three observed failures until actual independent
examples have been reviewed. Preserve both representative and labelled worst-case panels.

## Application, deployment and reproducibility

The FastAPI app serves HTML/CSS/JS, validates units/masks/alignment, provides supported method
selection, loading/error states, previews and numeric NPZ/PNG downloads. Learned methods require
a configured verified checkpoint. Exact-weight local app checks use synthetic uploads only;
Public Modal hosting is verified; Docker and full independent benchmark completion remain pending. Setup: [README](README.md); design: [DESIGN](DESIGN.md);
[architecture](docs/architecture.md); [deployment](docs/deployment.md).

## Remaining submission gates

Preserve the 854-record partial benchmark; verify original provenance before any seven-frame continuation; review real failure cases;
build/run Docker if required; test on physical phone/laptop beyond current browser emulation;
record the verified URL; review/publish Git changes; complete the official template and demo.
KITTI/external generalization is unverified and must not be claimed. The official test set stays locked.
