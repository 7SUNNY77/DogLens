"""Prepare Stanford Dogs: audit, official split + stratified validation, mapping, catalog.

Outputs (written atomically):
  <output-dir>/manifest.csv, <output-dir>/class_mapping.json,
  <output-dir>/preparation_report.json, <catalog-path> (assets/breeds.json),
  and optionally a visual-check gallery (never committed to Git).

Source data are only read, never modified.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from doglens.catalog import (
    MappingError,
    build_catalog,
    load_ru_names,
    mapping_sha256,
    mapping_to_dict,
)
from doglens.config import NUM_CLASSES, load_training_config
from doglens.data_prep import (
    FULL_LIST_FILE,
    OFFICIAL_SPLIT_FILES,
    DatasetError,
    context_box,
    excluded_csv_text,
    load_dataset_image,
    manifest_csv_text,
    pick_visual_samples,
    prepare,
)
from doglens.errors import ConfigError
from doglens.fileutil import (
    atomic_write_json,
    atomic_write_text,
    sha256_bytes,
    sha256_file,
)
from doglens.visual_check import render_panel

VISUAL_REVIEW_RECORD = "reports/dataset_visual_check.md"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare the Stanford Dogs manifest, class mapping, catalog and report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/images/Images"),
        help="directory with <wnid>-<breed>/ image folders; relative_path is relative to it",
    )
    p.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("data/annotations/Annotation"),
        help="directory with <wnid>-<breed>/ XML annotation folders",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="directory with official train_list.mat and test_list.mat",
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    p.add_argument("--catalog-path", type=Path, default=Path("assets/breeds.json"))
    p.add_argument(
        "--ru-names",
        type=Path,
        default=Path("assets/breed_names_ru.json"),
        help="optional overlay with verified Russian names",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="source of seed, data.val_fraction and data.bbox_context",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="threads for image decoding/hashing",
    )
    p.add_argument(
        "--visual-sample",
        type=int,
        default=26,
        help="number of visual-check panels to render (0 disables)",
    )
    p.add_argument(
        "--visual-dir",
        type=Path,
        default=None,
        help="gallery directory (default: <output-dir>/visual_check)",
    )
    p.add_argument(
        "--allow-changes",
        action="store_true",
        help="allow overwriting an existing manifest/mapping whose content differs",
    )
    args = p.parse_args(argv)
    for name in ("images_dir", "annotations_dir", "splits_dir"):
        if not getattr(args, name).is_dir():
            p.error(f"--{name.replace('_', '-')} does not exist: {getattr(args, name)}")
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if not 0 <= args.visual_sample <= 200:
        p.error("--visual-sample must be in 0..200")
    return args


def _posix(path: Path) -> str:
    return path.as_posix()


def _check_stable(path: Path, new_text: str, allow: bool, what: str) -> str:
    """Return 'created' | 'unchanged' | 'changed'; refuse silent changes of the split/mapping."""
    if not path.is_file():
        return "created"
    old = path.read_text(encoding="utf-8")
    # The mapping is compared as data; the manifest byte-for-byte.
    same = json.loads(old) == json.loads(new_text) if what == "class mapping" else old == new_text
    if same:
        return "unchanged"
    if not allow:
        raise DatasetError(
            f"existing {what} at {path} differs from the newly prepared one; the split and "
            "mapping must stay stable. Re-run with --allow-changes only for an explicit new run."
        )
    return "changed"


def _package_versions() -> dict[str, str]:
    import numpy
    import PIL
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def render_gallery(result, args: argparse.Namespace, seed: int, context: float) -> dict:
    visual_dir = args.visual_dir or args.output_dir / "visual_check"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rows = {r["image_id"]: r for r in result.rows}
    entries = []
    for n, (image_id, category) in enumerate(
        pick_visual_samples(result, args.visual_sample, seed), start=1
    ):
        row = rows[image_id]
        image = load_dataset_image(args.images_dir / row["relative_path"])
        bbox = (row["bbox_xmin"], row["bbox_ymin"], row["bbox_xmax"], row["bbox_ymax"])
        ctx = context_box(bbox, image.size, context)
        caption = (
            f"{n:02d} [{category}] {image_id} {row['breed_key']} split={row['split']} "
            f"size={image.width}x{image.height} bbox={bbox} context={ctx} "
            f"objects={result.record_facts[image_id]['n_objects']}"
        )
        name = f"{n:02d}_{category}_{image_id}.jpg"
        render_panel(image, bbox, ctx, caption).save(visual_dir / name, quality=90)
        entries.append(
            {
                "file": name,
                "image_id": image_id,
                "category": category,
                "breed_key": row["breed_key"],
                "bbox": list(bbox),
                "context_box": list(ctx),
            }
        )
    atomic_write_json(visual_dir / "index.json", {"panels": entries})
    return {"dir": _posix(visual_dir), "panels": len(entries), "index": entries}


def main(argv: list[str] | None = None, *, num_classes: int = NUM_CLASSES) -> int:
    """CLI entry point; ``num_classes`` exists only so tests can use tiny fixtures."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_training_config(args.config)
        seed, val_fraction, context = cfg.seed, cfg.data.val_fraction, cfg.data.bbox_context
        result = prepare(
            args.images_dir,
            args.annotations_dir,
            args.splits_dir,
            val_fraction=val_fraction,
            seed=seed,
            bbox_context=context,
            workers=args.workers,
            num_classes=num_classes,
            progress=lambda m: print(m, flush=True),
        )
        mapping = mapping_to_dict(result.classes)
        mapping_text = json.dumps(mapping, ensure_ascii=False, indent=2) + "\n"
        manifest_text = manifest_csv_text(result.rows)
        excluded_text = excluded_csv_text(result.report["excluded"])
        catalog = build_catalog(result.classes, load_ru_names(args.ru_names, result.classes))

        manifest_path = args.output_dir / "manifest.csv"
        mapping_path = args.output_dir / "class_mapping.json"
        excluded_path = args.output_dir / "excluded.csv"
        state = {
            "manifest": _check_stable(manifest_path, manifest_text, args.allow_changes, "manifest"),
            "class_mapping": _check_stable(
                mapping_path, mapping_text, args.allow_changes, "class mapping"
            ),
        }

        source_info_path = args.splits_dir / "SOURCE.json"
        split_files = {
            name: sha256_file(args.splits_dir / name)
            for name in [*OFFICIAL_SPLIT_FILES.values(), FULL_LIST_FILE]
            if (args.splits_dir / name).is_file()
        }
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "command": "scripts/prepare_dataset.py "
            + " ".join(sys.argv[1:] if argv is None else argv),
            "inputs": {
                "images_dir": _posix(args.images_dir),
                "annotations_dir": _posix(args.annotations_dir),
                "splits_dir": _posix(args.splits_dir),
                "config": _posix(args.config),
                "relative_path_root": _posix(args.images_dir),
                "split_files_sha256": split_files,
                "source": json.loads(source_info_path.read_text(encoding="utf-8"))
                if source_info_path.is_file()
                else None,
            },
            "parameters": {"seed": seed, "val_fraction": val_fraction, "bbox_context": context},
            "outputs": {
                "manifest": _posix(manifest_path),
                "manifest_sha256": sha256_bytes(manifest_text.encode("utf-8")),
                "class_mapping": _posix(mapping_path),
                "mapping_sha256": mapping_sha256(result.classes),
                "excluded": _posix(excluded_path),
                "excluded_sha256": sha256_bytes(excluded_text.encode("utf-8")),
                "catalog": _posix(args.catalog_path),
                "names_ru_verified": sum(b["name_ru"] is not None for b in catalog["breeds"]),
                "write_state": state,
            },
            "package_versions": _package_versions(),
            **result.report,
        }

        if args.visual_sample:
            gallery = render_gallery(result, args, seed, context)
            report["visual_check"] = {
                "status": "generated; manual review result recorded separately",
                "review_record": VISUAL_REVIEW_RECORD,
                "gallery_dir": gallery["dir"],
                "panels": gallery["panels"],
                "legend": "red = normalized bbox, green = bbox + context, right = final crop",
                "samples": gallery["index"],
            }
        else:
            report["visual_check"] = {"status": "not_generated"}

        atomic_write_text(manifest_path, manifest_text)
        atomic_write_text(mapping_path, mapping_text)
        atomic_write_text(excluded_path, excluded_text)
        atomic_write_json(args.catalog_path, catalog)
        atomic_write_json(args.output_dir / "preparation_report.json", report)
    except (DatasetError, MappingError, ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    sc = result.report["split_counts"]
    print(
        f"OK: classes={result.report['num_classes']} train={sc['train']} val={sc['val']} "
        f"test={sc['test']} excluded={result.report['excluded_count']} "
        f"issues={result.report['issue_counts']} "
        f"duplicate_groups={result.report['duplicates']['group_count']} "
        f"manifest={state['manifest']} mapping={state['class_mapping']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
