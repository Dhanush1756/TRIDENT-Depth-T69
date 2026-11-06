# Three-minute demo rehearsal

Do not rehearse claims that have not been measured. Until trained checkpoints
exist, label the demo classical-only and show the disabled neural options.

- 0:00–0:30: Explain P4, explicit metric units, optional RGB and the two input modes.
- 0:30–1:15: Upload an unseen supported map, select scale, reconstruct; show
  input/output, intermediate filters and numeric download. Demonstrate loading
  and a mismatched-RGB error.
- 1:15–2:00: Once training/evaluation is complete, compare interpolation,
  classical, CNN and hybrid results on the same held-out sample and protocol.
  Explain ground-truth-dependent metrics and measured runtime.
- 2:00–2:40: Show a real failure case and ablation. Explain why RGB texture is
  not necessarily a depth edge, and how confidence and consistency work.
- 2:40–3:00: State dataset split/checkpoint provenance, limitations and contributions.

Responsibilities follow the specification: Dhanush — classical methods;
Dhruthi — model/losses/training; Sasi Vaishnav — evaluation/fusion/research;
Jathin — frontend/API/deployment. Every member should understand all interfaces.
Live-URL, phone and laptop tests remain required before submission.
