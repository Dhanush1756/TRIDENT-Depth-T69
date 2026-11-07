"""Original scene split and degradation functions, without resolving locked payloads."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from ..config import PipelineConfig
from ..data.io import load_sample
from ..data.degradation import degrade
from ..classical.interpolation import interpolate
from ..classical.hole_filling import fill_holes
from ..training.colab import sha256, NYU_MANIFEST_SHA256
from ..training.corrective_data import REFERENCE_SHA256


def absolute(path):
    return Path(os.path.abspath(os.fspath(path)))


class DevelopmentData:
    """Allow only original 672 train + 123 validation RGB/depth/mask payloads.

    Mixed manifest labels are inspected to prove scene disjointness; test paths
    are never resolved, statted, opened or enumerated. No all-split audit runs.
    """
    def __init__(self, manifest, reference):
        self.manifest, self.reference = absolute(manifest), absolute(reference)
        self.root = self.manifest.parent
        if self.manifest.name != "manifest_development.json":
            raise ValueError("requires original development manifest")
        if self.reference.is_relative_to(self.root):
            raise ValueError("reference must be outside dataset storage")
        if sha256(self.manifest) != NYU_MANIFEST_SHA256 or sha256(self.reference) != REFERENCE_SHA256:
            raise ValueError("original manifest/reference SHA256 mismatch")
        rows = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.rows = {s:[r for r in rows if r["split"]==s] for s in ("train","validation")}
        self.frames = json.loads(self.reference.read_text(encoding="utf-8"))
        if [len(self.rows[s]) for s in self.rows] != [672,123]:
            raise ValueError("requires exactly original 672/123 development frames")
        scenes = {s:{r["scene_id"] for r in rows if r["split"]==s} for s in ("train","validation","test")}
        if [len(scenes[s]) for s in scenes] != [219,30,215] or any(
                scenes[a]&scenes[b] for a,b in (("train","validation"),("train","test"),("validation","test"))):
            raise ValueError("original scene partition mismatch/leakage")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate sample identity")
        if [r["id"] for r in self.rows["validation"]] != [r["id"] for r in self.frames]:
            raise ValueError("validation identity/order mismatch")
        self.allowed = set()
        for subset in self.rows.values():
            for r in subset:
                if r["official_split"] != "train" or r["depth_scale_to_m"] != 1:
                    raise ValueError("development data must be official train in metres")
                for k in ("rgb","depth","mask"):
                    p=absolute(self.root/r[k])
                    if not p.is_relative_to(self.root):
                        raise ValueError("payload escapes dataset storage")
                    self.allowed.add(p)
        if len(self.allowed)!=2385:
            raise ValueError("requires distinct 795 RGB/depth/mask triplets")
        self.enabled=False
        self.opened=set()
        self.denied=[]

    def guard(self, event, args):
        if not self.enabled or event not in {"open","os.listdir","os.scandir"}:
            return
        if not isinstance(args[0], (str,bytes,os.PathLike)):
            return
        path=absolute(os.fsdecode(args[0]))
        if not path.is_relative_to(self.root):
            return
        mode=str(args[1] or "") if event=="open" else ""
        flags=(args[2] or 0) if event=="open" else 0
        write=any(c in mode for c in "wax+") or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
        if event!="open" or write or path not in self.allowed|{self.manifest}:
            self.denied.append(str(path))
            raise PermissionError("locked/non-development dataset access: "+str(path))
        self.opened.add(path)

    @contextmanager
    def isolated(self):
        sys.addaudithook(self.guard)
        self.enabled=True
        try:
            yield self
        finally:
            self.enabled=False

    def verify_files(self):
        if not self.enabled:
            raise RuntimeError("payload audit requires active isolation")
        hashes={}
        for path in sorted(self.allowed):
            current=self.root
            for part in ("", *path.relative_to(self.root).parts):
                current=current/part
                if current.is_symlink():
                    raise ValueError("dataset symlinks forbidden")
            if path.resolve()!=path:
                raise ValueError("dataset aliases forbidden")
            hashes[path.relative_to(self.root).as_posix()]=sha256(path)
        for row,frame in zip(self.rows["validation"],self.frames):
            if row["scene_id"]!=frame["scene_id"] or any(hashes[row[k]]!=frame["source_files_sha256"][k] for k in ("rgb","depth","mask")):
                raise ValueError("validation files differ from original reference")
        return hashes

    def audit(self):
        if self.denied:
            raise ValueError("dataset isolation denied access")
        return dict(train_frames=672,validation_frames=123,train_scenes=219,validation_scenes=30,
                    official_test_sample_files_opened=0,shared_split_metadata_read=True,
                    distinct_development_payloads_opened=len(self.opened & self.allowed),
                    forbidden_access_attempts=self.denied,
                    scope="Python file-open/enumeration guard; not OS-level tracing")


def pipeline(task, seed=69):
    return PipelineConfig(**task["pipeline"],seed=seed,min_depth=0,max_depth=10)


class ProtocolDataset(Dataset):
    def __init__(self, data, tasks, split):
        if split not in {"train","validation"}:
            raise ValueError("test data is locked")
        self.data,self.tasks,self.split,self.epoch=data,tasks,split,1
        self.rows=data.rows[split]

    def __len__(self):
        return len(self.rows) if self.split=="train" else len(self.rows)*len(self.tasks)

    def __getitem__(self,index):
        if not self.data.enabled:
            raise RuntimeError("dataset access requires isolation")
        if not 0<=index<len(self):
            raise IndexError(index)
        if self.split=="train":
            ri,ti=index,(index+self.epoch-1)%len(self.tasks)
        else:
            ti,ri=divmod(index,len(self.rows))
        row,task=self.rows[ri],self.tasks[ti]
        # Epoch augmentation changes ONLY training degradation; validation is fixed.
        cfg=pipeline(task,69+1009*(self.epoch-1) if self.split=="train" else 69)
        source=load_sample(row,self.data.root,min_depth=0,max_depth=10)
        sample=degrade(source,cfg)
        return tensor_sample(sample,task,row["id"])


def tensor_sample(sample,task,sample_id):
    if not sample.input_valid.any():
        raise ValueError("no measurements: reject rather than invent depth")
    size=sample.target.depth.shape
    initial,valid=interpolate(sample.input_depth,sample.input_valid,size,"bicubic")
    initial,filled=fill_holes(initial,valid)
    _,support=interpolate(sample.input_depth,sample.input_valid,size,"nearest")
    if not filled.all() or sample.target.rgb is None:
        raise ValueError("requires fillable measurements and registered RGB")
    channel=lambda a:torch.from_numpy(np.ascontiguousarray(a))[None]
    return dict(initial=channel(initial),observed_support=channel(support),
                rgb=torch.from_numpy(np.ascontiguousarray(sample.target.rgb.transpose(2,0,1))).float()/255,
                target=channel(sample.target.depth),target_valid=channel(sample.target.valid),
                observations=channel(sample.input_depth),input_valid=channel(sample.input_valid),
                sample_id=sample_id,task=task["name"])
