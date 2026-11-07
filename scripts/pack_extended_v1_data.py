"""Stage only the 795 development frames for the extended Colab run."""
import argparse
import json
from pathlib import Path
import zipfile
from trident_depth.extended_v1.data import DevelopmentData
from trident_depth.training.colab import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        p.error("use a new archive and provenance path")
    data=DevelopmentData(args.manifest,Path(__file__).resolve().parents[1]/"docs/pilot_dense4/per_frame.json")
    if args.output.resolve().is_relative_to(data.root):
        p.error("archive must be outside dataset storage")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with data.isolated():
        hashes=data.verify_files()
        with zipfile.ZipFile(args.output,"x",zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            archive.write(data.manifest,"manifest_development.json")
            for name in sorted(hashes):
                archive.write(data.root/name,name)
        audit=data.audit()
    with args.output.with_suffix(".json").open("x") as f:
        json.dump(dict(archive_sha256=sha256(args.output),payload_files_sha256=hashes,
                       manifest_sha256=sha256(data.manifest),split_audit=audit),f,indent=2)
    print(args.output,sha256(args.output),flush=True)


if __name__=="__main__":
    main()
