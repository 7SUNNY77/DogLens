"""Export the selected checkpoint of a finished run into artifacts/model_bundle.

Exports <run_dir>/best.pt (best validation Top-1 over the whole experiment), never last.pt
implicitly. The bundle is verified by loading it back on CPU and comparing logits with the
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from doglens.bundle import BundleError, export_bundle, load_bundle
from doglens.catalog import MappingError
from doglens.experiment import load_run_checkpoint
from doglens.model import BreedClassifier
from doglens.training import TrainingError

LOGIT_ATOL = 1e-5  # same CPU kernels, same weights: differences must be at float32 noise level


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a DogLens run checkpoint as a model bundle and verify it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True, help="artifacts/runs/<run_id>")
    p.add_argument(
        "--checkpoint",
        default="best.pt",
        help="checkpoint file inside --run-dir (default: global best)",
    )
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/model_bundle"))
    p.add_argument("--mapping", type=Path, default=Path("data/prepared/class_mapping.json"))
    p.add_argument("--pretrained-source", type=Path, default=Path("data/weights/SOURCE.json"))
    p.add_argument("--overwrite", action="store_true", help="replace a non-empty bundle dir")
    return p.parse_args(argv)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = args.run_dir
        final = run_dir / "final_report.json"
        if not final.is_file():
            raise TrainingError(f"{run_dir} has no final_report.json: the run is not completed")
        checkpoint = run_dir / args.checkpoint
        meta_path = run_dir / "run_metadata.json"
        run_meta = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
        )
        source = (
            json.loads(args.pretrained_source.read_text(encoding="utf-8"))
            if args.pretrained_source.is_file()
            else None
        )
        metadata = export_bundle(
            checkpoint,
            args.output_dir,
            args.mapping,
            run_metadata=run_meta,
            pretrained_source=source,
            overwrite=args.overwrite,
        )
        bundle = load_bundle(args.output_dir, "cpu")
        reference = BreedClassifier(metadata["num_classes"])
        reference.load_state_dict(load_run_checkpoint(checkpoint)["model_state"], strict=True)
        reference.eval()
        torch.manual_seed(0)
        probe = torch.randn(4, 3, bundle.spec.crop_size, bundle.spec.crop_size)
        with torch.inference_mode():
            diff = float((reference(probe) - bundle.model(probe)).abs().max())
        if diff > LOGIT_ATOL:
            raise BundleError(f"bundle logits differ from the checkpoint by {diff}")
    except (BundleError, MappingError, TrainingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"OK: exported {metadata['model_version']} to {args.output_dir} "
        f"(source {metadata['source_checkpoint']['stage']} epoch "
        f"{metadata['source_checkpoint']['stage_epoch']}); CPU reload verified, "
        f"max |logit diff| = {diff:.3g} (tolerance {LOGIT_ATOL})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
