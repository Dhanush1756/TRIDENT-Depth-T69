"""Inspect actual downloaded NYU MAT/HDF5 and official split files, without conversion."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.io import loadmat


def decode_cell(handle, reference):
    array = np.asarray(handle[reference])
    if array.dtype.kind not in "ui":
        raise ValueError("expected MATLAB character array")
    return "".join(chr(int(v)) for v in array.ravel(order="F") if v)


def inspect(mat_path, split_path):
    result = dict(source=str(Path(mat_path).resolve()))
    with h5py.File(mat_path, "r") as handle:
        result["hdf5"] = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
            compression=value.compression, chunks=value.chunks) for key, value in handle.items()
            if isinstance(value, h5py.Dataset)}
        for key in ("scenes", "sceneTypes", "rawRgbFilenames", "rawDepthFilenames"):
            if key in handle:
                refs = np.asarray(handle[key]).ravel()
                result[key] = dict(count=len(refs), first=[decode_cell(handle, r) for r in refs[:5]])
        for key in ("images", "depths", "rawDepths"):
            array = np.asarray(handle[key][0])
            finite = np.isfinite(array)
            positive = finite & (array > 0)
            result[key] = dict(first_sample_storage_shape=list(array.shape),
                               min_finite=float(array[finite].min()) if finite.any() else None,
                               max_finite=float(array[finite].max()) if finite.any() else None,
                               nonfinite=int((~finite).sum()), nonpositive=int((finite & (array <= 0)).sum()),
                               positive_fraction=float(positive.mean()))
    splits = loadmat(split_path)
    result["splits"] = {k: dict(shape=list(v.shape), dtype=str(v.dtype), count=int(v.size),
                                minimum=int(v.min()), maximum=int(v.max()))
                        for k, v in splits.items() if not k.startswith("__") and np.issubdtype(v.dtype, np.number)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=Path("data/nyu/source/nyu_depth_v2_labeled.mat"))
    parser.add_argument("--splits", type=Path, default=Path("data/nyu/source/splits.mat"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/nyu/source_inspection.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = inspect(args.mat, args.splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
