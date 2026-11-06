"""Export one validated sample as raw numeric arrays plus provenance."""
import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import numpy as np
from .config import PipelineConfig
from .data.dataset import DepthDataset
from .logging_utils import configure_logging


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--output", required=True, type=Path, help="new output directory (must not exist)")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        config = PipelineConfig.load(args.config)
        dataset = DepthDataset(args.manifest, config, args.split)
        if not 0 <= args.index < len(dataset):
            raise ValueError(f"index out of range; selected split contains {len(dataset)} records")
        sample = dataset[args.index]
        arrays = dict(input_depth=sample.input_depth, input_valid=sample.input_valid,
                      ground_truth=sample.target.depth, ground_truth_valid=sample.target.valid)
        if sample.target.rgb is not None:
            arrays["rgb"] = sample.target.rgb
        metadata = dict(sample.metadata, sample_id=sample.target.sample_id,
                        scene_id=sample.target.scene_id, split=args.split, config=asdict(config),
                        source_manifest=str(args.manifest.resolve()), source_record=dataset.records[args.index])
        args.output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(args.output / "sample.npz", **arrays)
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logging.info("Saved %s; input %s -> target %s; metres; valid fraction %.4f",
                     args.output, sample.input_depth.shape, sample.target.depth.shape,
                     sample.metadata["valid_fraction"])
    except (ValueError, TypeError, KeyError, OSError, IndexError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
