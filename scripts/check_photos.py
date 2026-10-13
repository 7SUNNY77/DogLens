"""Run the predictor on a folder of the user's own photos (external check, local only).

Optional labels CSV (UTF-8) with header ``file,expected[,xmin,ymin,xmax,ymax]``:
  expected = a supported breed_key from assets/breeds.json, or one of
             mixed_breed | not_dog | unsupported_breed | unknown
  ROI columns (optional) are the user box in EXIF-oriented pixels; empty = whole image.

Accuracy (Top-1/Top-3) is reported only for photos whose expected value is a supported breed.
Mixed breeds, unsupported breeds and photos without dogs are listed separately: the model
always answers with supported breeds, so they are a limitation analysis, not an accuracy.
Do not use Stanford Dogs images here: they are not an external check.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

from doglens.errors import DogLensError
from doglens.fileutil import atomic_write_json, atomic_write_text
from doglens.image_io import load_image
from doglens.predictor import Predictor

SPECIAL = ("mixed_breed", "not_dog", "unsupported_breed", "unknown")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check the DogLens model on your own local photos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None, help="optional labels CSV (see --help)")
    p.add_argument("--bundle", type=Path, default=Path("artifacts/model_bundle"))
    p.add_argument("--catalog", type=Path, default=Path("assets/breeds.json"))
    p.add_argument("--output-dir", type=Path, default=Path("reports/external_check"))
    args = p.parse_args(argv)
    if not args.input_dir.is_dir():
        p.error(f"--input-dir not found: {args.input_dir}")
    if args.labels is not None and not args.labels.is_file():
        p.error(f"--labels not found: {args.labels}")
    return args


def read_labels(path: Path | None, breed_keys: set[str]) -> dict[str, dict]:
    if path is None:
        return {}
    labels = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            expected = (row.get("expected") or "").strip()
            if expected not in breed_keys and expected not in SPECIAL:
                raise ValueError(f"unknown expected value {expected!r} for {row.get('file')!r}")
            box = [row.get(k, "").strip() for k in ("xmin", "ymin", "xmax", "ymax")]
            roi = tuple(int(v) for v in box) if all(box) else None
            labels[row["file"].strip()] = {"expected": expected, "roi": roi}
    return labels


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        predictor = Predictor.from_bundle_dir(args.bundle, args.catalog)
        keys = {e.breed_key for e in predictor.catalog}
        labels = read_labels(args.labels, keys)
    except (DogLensError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    files = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        print("ERROR: no JPEG/PNG/WebP files in --input-dir", file=sys.stderr)
        return 1
    rows = []
    for path in files:
        label = labels.get(path.name, {"expected": "", "roi": None})
        row = {"file": path.name, "expected": label["expected"], "roi": label["roi"]}
        try:
            image = load_image(path.read_bytes()).image
            result = predictor.predict(image, label["roi"], top_k=3)
            row.update(
                status="ok",
                crop_box=list(result.crop_box),
                top3=[p.breed_key for p in result.predictions],
                scores=[round(p.score, 4) for p in result.predictions],
            )
        except DogLensError as exc:
            row.update(status=f"error: {exc.user_message}", crop_box=None, top3=[], scores=[])
        rows.append(row)

    supported = [r for r in rows if r["status"] == "ok" and r["expected"] in keys]
    summary = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": predictor.model_version,
        "photos": len(rows),
        "errors": sum(r["status"] != "ok" for r in rows),
        "supported_breed_photos": len(supported),
        "top1": (sum(r["top3"][0] == r["expected"] for r in supported) / len(supported))
        if supported
        else None,
        "top3": (sum(r["expected"] in r["top3"] for r in supported) / len(supported))
        if supported
        else None,
        "limitation_cases": {k: sum(r["expected"] == k for r in rows) for k in SPECIAL},
        "unlabelled": sum(not r["expected"] for r in rows),
        "note": "small local sample; not comparable with the Stanford Dogs test metrics",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["file,expected,status,roi,crop_box,top1,top2,top3,score1,score2,score3"]
    for r in rows:
        top = (r["top3"] + ["", "", ""])[:3]
        sc = ([str(s) for s in r["scores"]] + ["", "", ""])[:3]
        roi = " ".join(map(str, r["roi"])) if r["roi"] else "whole"
        crop = " ".join(map(str, r["crop_box"])) if r["crop_box"] else ""
        cells = [r["file"], r["expected"], r["status"], roi, crop, *top, *sc]
        lines.append(",".join(c.replace(",", ";") for c in cells))
    atomic_write_text(args.output_dir / "predictions.csv", "\n".join(lines) + "\n")
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(summary)
    print(f"OK: results in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
