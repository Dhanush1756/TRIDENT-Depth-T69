import io, sys, zipfile, json
from pathlib import Path
REPO=Path(sys.argv[1]); sys.path.insert(0,str(REPO))
import numpy as np, torch
from scipy.stats import mannwhitneyu
from trident_depth.config import PipelineConfig
from trident_depth.models.residual_cnn import ResidualCNN
from trident_depth.training.dataset import TorchDepthDataset
from trident_depth.training.losses import depth_edge_weights
st=torch.load(io.BytesIO(zipfile.ZipFile(sys.argv[2]).read("pilot_dense4_residual_seed69/epoch_002.pt")),map_location="cpu",weights_only=True)
m=ResidualCNN(upsampling="nearest"); m.load_state_dict(st["model_state"]); m.eval()
ds=TorchDepthDataset(REPO/"data/nyu/converted/manifest_development.json",PipelineConfig(**st["metadata"]["config"]["pipeline"]),"validation","bicubic")
S,L,U,E=[],[],[],[]
for i in range(0,123,12):
    f=ds[i]
    with torch.inference_mode(): o=m(f["rgb"][None],f["initial"][None],f["observed_support"][None],4,"dense_sr")
    edge=(depth_edge_weights(f["target"][None],f["target_valid"][None])>1)[0,0].numpy()
    S.append(o["boundary_confidence"][0,0].numpy()[::5,::5].ravel()); L.append(edge[::5,::5].ravel())
    U.append(o["log_scale"][0,0].numpy()[::5,::5].ravel()); E.append(np.abs(o["depth"]-f["target"][None])[0,0].numpy()[::5,::5].ravel())
s,l,u,e=map(np.concatenate,(S,L,U,E)); l=l.astype(bool)
auc=lambda x: mannwhitneyu(x[l],x[~l]).statistic/(l.sum()*(~l).sum())
print(json.dumps(dict(frames=len(S),pixels=int(len(l)),edge_frac=float(l.mean()),boundary_head_auc_vs_gt_edges=float(auc(s)),log_scale_auc_vs_gt_edges=float(auc(u)),
   err_auc_vs_edges=float(auc(e)))))
