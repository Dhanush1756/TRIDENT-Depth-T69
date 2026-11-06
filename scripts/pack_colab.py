"""Package reviewed source or converted NYU files for manual Drive upload."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["source", "dataset"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/nyu/converted"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1] if args.kind == "source" else args.dataset.resolve()
    source_roots = {"trident_depth", "scripts", "configs", "tests", "notebooks", "backend", "frontend", "docs"}
    candidates = root.rglob("*") if args.kind == "dataset" else (
        path for top in root.iterdir() if top.name in source_roots or top.is_file()
        for path in (top.rglob("*") if top.is_dir() else [top]))
    files = sorted(path for path in candidates if path.is_file()
                   and not any(part in {"__pycache__", ".pytest_cache"} for part in path.relative_to(root).parts)
                   and (args.kind == "dataset" or path.relative_to(root).parts[0] in source_roots
                        or (path.parent == root and (path.name.startswith("requirements") or path.name in {"pyproject.toml", "README.md"}))))
    if not files:
        parser.error("no files to package")
    output = args.output.resolve()
    if args.kind == "dataset" and output.is_relative_to(root):
        parser.error("archive must be outside the dataset directory")
    partial = output.with_suffix(output.suffix + ".part")
    if output.exists() or partial.exists():
        parser.error("output or partial archive already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in files:
                archive.write(path, path.relative_to(root).as_posix())
        partial.rename(output)
    finally:
        partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    print(f"{output}\nfiles={len(files)} bytes={output.stat().st_size}\nSHA256={digest.hexdigest()}")


if __name__ == "__main__":
    main()
