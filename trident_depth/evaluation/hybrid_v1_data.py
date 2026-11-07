"""Validation-only payload isolation for hybrid_v1_dense4; no all-split file audit."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from ..training.colab import sha256, NYU_MANIFEST_SHA256
from ..training.corrective_data import REFERENCE_SHA256, audit_validation_reference
from ..training.dataset import TorchDepthDataset


def lexical(path):
    return Path(os.path.abspath(os.fspath(path)))


class ValidationOnly:
    """Read mixed split labels only; never resolve/stat/open non-validation payloads."""
    def __init__(self, manifest, reference):
        self.manifest = lexical(manifest)
        self.root = self.manifest.parent
        self.reference_path = lexical(reference)
        if self.manifest.name != "manifest_development.json":
            raise ValueError("requires immutable original manifest_development.json, not a sample file")
        if self.reference_path.is_relative_to(self.root) or self.reference_path.suffix.lower() != ".json":
            raise ValueError("immutable reference JSON must be outside dataset payload storage")
        if sha256(self.manifest) != NYU_MANIFEST_SHA256 or sha256(self.reference_path) != REFERENCE_SHA256:
            raise ValueError("requires immutable original NYU manifest and 123-frame reference")
        self.reference = json.loads(self.reference_path.read_text(encoding="utf-8"))
        all_rows = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.rows = [r for r in all_rows if r["split"] == "validation"]
        if len(self.rows) != 123 or [r["id"] for r in self.rows] != [r["id"] for r in self.reference]:
            raise ValueError("validation must match exactly the original 123 IDs/order")
        scenes = {r["scene_id"] for r in self.rows}
        others = {r["scene_id"] for r in all_rows if r["split"] != "validation"}
        if len(scenes) != 30 or scenes & others:
            raise ValueError("validation scene partition differs or leaks")
        if any(r["official_split"] != "train" or r["depth_scale_to_m"] != 1 for r in self.rows):
            raise ValueError("validation requires official train membership and explicit metre units")
        if any(r["scene_id"] != f["scene_id"] for r,f in zip(self.rows,self.reference)):
            raise ValueError("validation scenes differ from reference")
        self.paths = {lexical(self.root/r[k]) for r in self.rows for k in ("rgb","depth","mask")}
        if len(self.paths) != 369 or any(not p.is_relative_to(self.root) for p in self.paths):
            raise ValueError("validation paths must be 369 distinct files within dataset storage")
        self.opened, self.denied, self.loaded = [], [], []
        self.enabled = False

    def _guard(self, event, args):
        if not self.enabled or event not in {"open","os.listdir","os.scandir"}:
            return
        if not isinstance(args[0], (str,bytes,os.PathLike)):
            return
        path = lexical(os.fsdecode(args[0]))
        if not path.is_relative_to(self.root):
            return
        if event != "open":
            self.denied.append(dict(event=event,path=str(path)))
            raise PermissionError("dataset enumeration forbidden")
        mode, flags = str(args[1] or ""), args[2] or 0
        writing = any(k in mode for k in "wax+") or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing or path not in self.paths | {self.manifest}:
            self.denied.append(dict(event=event,path=str(path)))
            raise PermissionError("non-validation dataset access forbidden: "+str(path))
        self.opened.append(str(path))

    @contextmanager
    def isolated(self):
        sys.addaudithook(self._guard)
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = False

    def prepare(self, directory, pipeline):
        if not self.enabled:
            raise RuntimeError("validation preparation requires active access isolation")
        # Reject links BEFORE file hashing, so an allowed validation name cannot
        # redirect a read into a locked payload. Check components without following links.
        for path in self.paths:
            current=self.root
            if current.is_symlink():
                raise ValueError("validation symlink aliases are not allowed")
            for part in path.relative_to(self.root).parts:
                current=current/part
                if current.is_symlink():
                    raise ValueError("validation symlink aliases are not allowed")
            if path.resolve() != path:
                raise ValueError("validation path aliases are not allowed")
        # Existing verifier hashes only validation RGB/depth/mask files.
        actual = audit_validation_reference(self.manifest)
        if actual != self.reference:
            raise ValueError("reference supplied differs from original evaluation reference")
        projected = []
        for row in self.rows:
            copy = dict(row)
            for key in ("rgb","depth","mask"):
                path = lexical(self.root/row[key])
                copy[key] = str(path)
            projected.append(copy)
        path = Path(directory)/"validation_only_manifest.json"
        with path.open("x",encoding="utf-8") as stream:
            json.dump(projected,stream,indent=2)
        dataset = TorchDepthDataset(path,pipeline,"validation",initializer="bicubic")
        if [r["id"] for r in dataset.source.records] != [r["id"] for r in self.rows]:
            raise ValueError("original loader altered validation order")
        return dataset, path

    def verify_batch(self, batch, index):
        row, reference = self.rows[index], self.reference[index]
        sample_id = batch["sample_id"][0]
        from hashlib import sha256 as digest
        mask = batch["target_valid"]
        if (sample_id != row["id"] or sample_id in self.loaded
                or digest(mask.contiguous().numpy().tobytes()).hexdigest() != reference["mask_sha256"]
                or int(mask.sum()) != reference["valid_pixels"]
                or list(batch["target"].shape) != [1,1,480,640]
                or list(batch["observations"].shape) != [1,1,120,160]
                or not batch["input_valid"].all() or not batch["rgb_present"].all()):
            raise ValueError("validation ID, mask, dimensions or dense protocol mismatch")
        self.loaded.append(sample_id)

    def audit(self):
        if self.denied or self.loaded != [r["id"] for r in self.rows]:
            raise ValueError("validation isolation failed or evaluation incomplete")
        return dict(validation_frames=len(self.loaded), official_test_sample_files_opened=0,
            training_sample_files_opened=0, shared_split_metadata_read=True,
            dataset_open_events=len(self.opened), distinct_validation_files_opened=len(set(self.opened)&{str(p) for p in self.paths}),
            forbidden_access_attempts=self.denied,
            scope="Python file-open/enumeration guard over inspected NumPy/Pillow loader; not OS tracing")
