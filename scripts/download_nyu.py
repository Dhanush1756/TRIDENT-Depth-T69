"""Download the approved official labelled NYU release and split metadata.

Existing complete files are never overwritten. Partial files can be resumed.
No training is started. Source and redirect URLs and local SHA-256 are recorded.
"""
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

SOURCES = {
    "nyu_depth_v2_labeled.mat": [
        "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    ],
    "splits.mat": [
        "https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",
        "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",
    ],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target, limit):
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "TRIDENT-Depth-T69 dataset preparation"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        if offset and response.status != 206:
            raise ValueError("server did not honor resume range; partial file retained")
        remaining = response.headers.get("Content-Length")
        total = offset + int(remaining) if remaining else None
        if total is not None and total > limit:
            raise ValueError("response exceeds approved size bound")
        resolved = response.url
        last = time.monotonic()
        written = offset
        with partial.open("ab" if offset else "xb") as stream:
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise ValueError("download exceeds approved size bound")
                stream.write(chunk)
                if time.monotonic() - last >= 15:
                    print(f"{target.name}: {written / 1024**2:.1f} MiB / {total}", flush=True)
                    last = time.monotonic()
        if total is not None and written != total:
            raise ValueError("download incomplete; partial retained")
    with partial.open("rb") as stream:
        signature = stream.read(128)
    if not (signature.startswith(b"MATLAB") or signature.startswith(b"\x89HDF\r\n\x1a\n")):
        raise ValueError("response is not a MAT/HDF5 file; inspect retained partial")
    partial.rename(target)
    return dict(requested_url=url, resolved_url=resolved, bytes=written, sha256=sha256(target))


def main():
    root = Path("data/nyu/source")
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "download_provenance.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for name, urls in SOURCES.items():
        target = root / name
        if target.exists():
            print(f"Already present: {name}; SHA-256 {sha256(target)}", flush=True)
            continue
        errors = []
        for url in urls:
            print(f"Requesting {url}", flush=True)
            try:
                report[name] = download(url, target, 4 * 1024**3 if name.startswith("nyu") else 1024**2)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"Completed {name}: {report[name]}", flush=True)
                break
            except (OSError, ValueError, urllib.error.URLError) as exc:
                errors.append(f"{url}: {exc}")
                print(errors[-1], flush=True)
        else:
            raise SystemExit("Official source download failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
