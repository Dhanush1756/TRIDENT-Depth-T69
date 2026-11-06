"""Supplementary read-only probes: zero optimizer steps, no checkpoint/benchmark writes,
validation split only. Output goes to the scratchpad."""
import io, json, math, sys, zipfile, hashlib
from pathlib import Path
REPO = Path(sys.argv[1]); ARCHIVE = Path(sys.argv[2]); OUT = Path(sys.argv[3])
sys.path.insert(0, str(REPO))
import numpy as np, torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.stats import mannwhitneyu
from trident_depth.config import PipelineConfig
from trident_depth.models.residual_cnn import ResidualCNN
from trident_depth.models.baseline import LightweightCNN
from trident_depth.models.checkpoint import load_checkpoint
from trident_depth.training.dataset import TorchDepthDataset
from trident_depth.training.losses import depth_edge_weights
from trident_depth.training.state import seed_everything

torch.set_num_threads(4)
prefix = "pilot_dense4_residual_seed69/"
with zipfile.ZipFile(ARCHIVE) as z:
    raw = {e: z.read(prefix + f"epoch_{e:03d}.pt") for e in (1, 2)}
    pointer = json.loads(z.read(prefix + "best_checkpoint.json"))
    comparison = json.loads(z.read(prefix + "validation_comparison/comparison.json"))
states = {e: torch.load(io.BytesIO(b), map_location="cpu", weights_only=True) for e, b in raw.items()}
hashes = {e: hashlib.sha256(b).hexdigest() for e, b in raw.items()}
baseline_path = REPO / "artifacts/pilot_dense4_audit/epoch_002.pt"
baseline, _ = load_checkpoint(baseline_path, device="cpu")
bstate = torch.load(baseline_path, map_location="cpu", weights_only=True)
report = {}

# 1. Checkpoint selection
report["selection"] = dict(pointer=pointer, comparison_best_epoch=comparison["best_custom_epoch"],
    comparison_custom_sha=comparison["custom_checkpoint_sha256"], archive_sha=hashes,
    selected_matches=comparison["custom_checkpoint_sha256"] == hashes[pointer["epoch"]],
    history_rmse={e: s["training_state"]["history"][-1]["validation_rmse_m"] for e, s in states.items()})

# 2. Init fingerprint: params with provably zero gradient keep their Colab init values.
seed_everything(69); local = ResidualCNN(upsampling="nearest").state_dict()
seed_everything(69); local_b = LightweightCNN().state_dict()
m1, m2 = states[1]["model_state"], states[2]["model_state"]
fp = {}
for name, sl in (("enc1.0.weight[:,6] (protocol channel, input=0)", lambda d: d["enc1.0.weight"][:, 6]),
                 ("boundary_head.weight", lambda d: d["boundary_head.weight"]),
                 ("boundary_head.bias", lambda d: d["boundary_head.bias"])):
    fp[name] = dict(epoch1_equals_epoch2=bool(torch.equal(sl(m1), sl(m2))),
                    equals_local_seed69=bool(torch.equal(sl(m2), sl(local))),
                    max_abs_diff_vs_local=float((sl(m2) - sl(local)).abs().max()))
fp["baseline layers.0.weight[:,6] equals local seed69"] = bool(torch.equal(bstate["model_state"]["layers.0.weight"][:, 6], local_b["layers.0.weight"][:, 6]))
report["init_fingerprint"] = fp
report["adam_steps"] = {e: sorted({float(v["step"]) for v in s["optimizer_state"]["state"].values()}) for e, s in states.items()}

cfg = states[2]["metadata"]["config"]
pipeline = PipelineConfig(**cfg["pipeline"])
dataset = TorchDepthDataset(REPO / "data/nyu/converted/manifest_development.json", pipeline, "validation", initializer="bicubic")
assert len(dataset) == 123
models = {}
for e, s in states.items():
    m = ResidualCNN(upsampling="nearest"); m.load_state_dict(s["model_state"], strict=True); models[e] = m.eval()

# 3. Init offset across seeds (structural, forward-only, 3 frames)
probe_frames = [dataset[i] for i in (0, 40, 80)]
seed_rows = []
for seed in list(range(10)) + [69]:
    torch.manual_seed(seed); net = ResidualCNN(upsampling="nearest").eval()
    torch.manual_seed(seed); bnet = LightweightCNN().eval()
    with torch.inference_mode():
        r = [float(net(f["rgb"][None], f["initial"][None], f["observed_support"][None], 4, "dense_sr")["residual"].mean()) for f in probe_frames]
        rb = [float(bnet(f["rgb"][None], f["initial"][None], f["observed_support"][None], 4, "dense_sr")["residual"].mean()) for f in probe_frames]
        s0 = [float(net(f["rgb"][None], f["initial"][None], f["observed_support"][None], 4, "dense_sr")["log_scale"].mean()) for f in probe_frames]
    seed_rows.append(dict(seed=seed, custom_mean_residual_m=float(np.mean(r)), baseline_mean_residual_m=float(np.mean(rb)), custom_mean_log_scale=float(np.mean(s0))))
report["init_offset_by_seed"] = seed_rows

# 4. Full validation pass
def bp(pred, obs):  # GT-free hard back-projection onto area-mean constraint
    return pred + F.interpolate(obs - F.avg_pool2d(pred, 4, 4), scale_factor=4, mode="nearest")

keys = ["bicubic", "bicubic_bp", "baseline", "baseline_demeaned", "baseline_bp"] + [f"c{e}{suf}" for e in (1, 2) for suf in ("", "_demeaned", "_bp")]
acc = {k: dict(se=0., ae=0., se_edge=0., ae_edge=0., ae_non=0., wins_mae=0) for k in keys}
N = Ne = 0
frame_dc = {k: [] for k in ("baseline", "c1", "c2")}
corr = {k: dict(rr=0., ee=0., re=0., rr_e=0., ee_e=0., re_e=0.) for k in ("baseline", "c1", "c2")}
effw = {e: dict(edge=0., non=0.) for e in (1, 2)}
bias_e0 = 0.
const_channels = True
bnd_scores, bnd_labels = [], []
bnd_stats = dict(mean=0., min=1., max=0.)
for i, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)):
    t, mk, init, obs = batch["target"], batch["target_valid"], batch["initial"], batch["observations"]
    const_channels &= bool(batch["observed_support"].all()) and bool(mk.all()) and bool(batch["input_valid"].all())
    edge = (depth_edge_weights(t, mk) > 1) & mk
    n, ne = int(mk.sum()), int(edge.sum()); N += n; Ne += ne
    e0 = (init - t).double()
    bias_e0 += float(e0[mk].sum())
    preds = {"bicubic": init, "bicubic_bp": bp(init, obs)}
    with torch.inference_mode():
        ob = baseline(batch["rgb"], init, batch["observed_support"], 4, "dense_sr")
        outs = {e: models[e](batch["rgb"], init, batch["observed_support"], 4, "dense_sr") for e in (1, 2)}
    for tag, depth in [("baseline", ob["depth"])] + [(f"c{e}", outs[e]["depth"]) for e in (1, 2)]:
        r = (depth - init).double()
        dc = float(r[mk].mean()); frame_dc[tag].append(dc)
        preds[tag] = depth
        preds[tag + "_demeaned"] = (init.double() + r - dc).float()
        preds[tag + "_bp"] = bp(depth, obs)
        rf = r - dc  # spatially varying part
        ideal = -e0
        c = corr[tag]
        c["rr"] += float((rf[mk] ** 2).sum()); c["ee"] += float(((ideal[mk] - ideal[mk].mean()) ** 2).sum()); c["re"] += float((rf[mk] * (ideal[mk] - ideal[mk].mean())).sum())
        c["rr_e"] += float((rf[edge] ** 2).sum()); c["ee_e"] += float((ideal[edge] ** 2).sum()); c["re_e"] += float((rf[edge] * ideal[edge]).sum())
    for e in (1, 2):
        w = 0.05 / outs[e]["log_scale"].double().exp()
        effw[e]["edge"] += float(w[edge].sum()); effw[e]["non"] += float(w[mk & ~edge].sum())
    bc = outs[2]["boundary_confidence"]
    bnd_stats["mean"] += float(bc.mean()) / 123; bnd_stats["min"] = min(bnd_stats["min"], float(bc.min())); bnd_stats["max"] = max(bnd_stats["max"], float(bc.max()))
    sel = slice(i % 97, None, 97)
    bnd_scores.append(bc[mk][sel].numpy()); bnd_labels.append(edge[mk][sel].numpy())
    bic_mae = float((init - t).abs()[mk].mean())
    for k, p in preds.items():
        err = (p.double() - t.double())
        a = acc[k]
        a["se"] += float((err[mk] ** 2).sum()); a["ae"] += float(err[mk].abs().sum())
        a["se_edge"] += float((err[edge] ** 2).sum()); a["ae_edge"] += float(err[edge].abs().sum())
        a["ae_non"] += float(err[mk & ~edge].abs().sum())
        a["wins_mae"] += int(float(err[mk].abs().mean()) < bic_mae)
    if (i + 1) % 20 == 0: print(f"{i+1}/123", flush=True)

report["dense_inputs_constant"] = dict(observed_support_all_ones=const_channels,
    note="channels 4-7 (validity, scale/8, protocol, rgb_present) are spatially and frame-wise constant for dense 4x")
report["bicubic_mean_signed_error_m"] = bias_e0 / N
report["metrics"] = {k: dict(rmse_m=math.sqrt(a["se"] / N), mae_m=a["ae"] / N, edge_rmse_m=math.sqrt(a["se_edge"] / Ne),
    edge_mae_m=a["ae_edge"] / Ne, nonedge_mae_m=a["ae_non"] / (N - Ne), frames_beating_bicubic_mae=a["wins_mae"]) for k, a in acc.items()}
report["edge_pixel_fraction"] = Ne / N
report["per_frame_dc_residual_m"] = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)), min=float(np.min(v)), max=float(np.max(v)),
    frac_positive=float(np.mean(np.array(v) > 0))) for k, v in frame_dc.items()}
report["spatial_correction_vs_ideal"] = {k: dict(corr_all=c["re"] / math.sqrt(c["rr"] * c["ee"]), corr_edge=c["re_e"] / math.sqrt(c["rr_e"] * c["ee_e"]),
    spatial_part_rms_m=math.sqrt(c["rr"] / N)) for k, c in corr.items()}
report["uncertainty_effective_l1_weight_on_depth"] = {e: dict(edge=v["edge"] / Ne, nonedge=v["non"] / (N - Ne), depth_term_weight=1.0) for e, v in effw.items()}
s, l = np.concatenate(bnd_scores), np.concatenate(bnd_labels).astype(bool)
auc = float(mannwhitneyu(s[l], s[~l]).statistic / (l.sum() * (~l).sum()))
report["boundary_head_epoch2"] = dict(stats=bnd_stats, auc_vs_gt_depth_edges=auc, sample_pixels=int(len(l)))
OUT.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
