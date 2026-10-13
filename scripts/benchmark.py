"""Inference benchmark for the DogLens predictor (batch size 1).

Measures separately: imports, cold bundle loading, warm image decoding/validation, warm
preprocessing (region crop + shared preprocessing), warm model forward, the full
``Predictor.predict`` and decode + predict. Cold loading is never included in warm figures.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark DogLens CPU inference (batch size 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bundle", type=Path, default=Path("artifacts/model_bundle"))
    p.add_argument("--catalog", type=Path, default=Path("assets/breeds.json"))
    p.add_argument("--image", type=Path, required=True, help="local JPEG/PNG/WebP used as input")
    p.add_argument(
        "--roi",
        default=None,
        help="user ROI 'xmin,ymin,xmax,ymax' in EXIF-oriented pixels (default: whole)",
    )
    p.add_argument(
        "--device", choices=["cpu"], default="cpu", help="the current app supports CPU inference"
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--report", type=Path, default=Path("reports/benchmark/inference.json"))
    args = p.parse_args(argv)
    if args.warmup < 0 or args.repeats < 3:
        p.error("--warmup must be >= 0 and --repeats >= 3")
    if not args.image.is_file():
        p.error(f"--image not found: {args.image}")
    if args.roi is not None:
        try:
            args.roi = tuple(int(v) for v in args.roi.split(","))
        except ValueError:
            p.error("--roi must be four integers 'xmin,ymin,xmax,ymax'")
        if len(args.roi) != 4:
            p.error("--roi must be four integers 'xmin,ymin,xmax,ymax'")
    return args  # fmt: skip


def _portable_path(path: Path) -> str:
    """Repository-relative path, or only the file name for inputs outside the repository
    (absolute local paths can contain user names and must not end up in reports)."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return f"<outside repository>/{path.name}"


def stats(samples_s: list[float]) -> dict[str, float]:
    ms = sorted(s * 1000 for s in samples_s)
    p95_index = max(0, min(len(ms) - 1, round(0.95 * (len(ms) - 1))))
    return {
        "median_ms": statistics.median(ms),
        "p95_ms": ms[p95_index],
        "mean_ms": statistics.fmean(ms),
        "min_ms": ms[0],
        "max_ms": ms[-1],
        "n": len(ms),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    import torch
    import torchvision

    from doglens.errors import DogLensError
    from doglens.fileutil import atomic_write_json, sha256_file
    from doglens.image_io import load_image
    from doglens.predictor import Predictor
    from doglens.preprocessing import preprocess

    import_s = time.perf_counter() - t0

    try:
        t1 = time.perf_counter()
        predictor = Predictor.from_bundle_dir(args.bundle, args.catalog, args.device)
        cold_load_s = time.perf_counter() - t1
        data = args.image.read_bytes()
        image = load_image(data).image
        crop, region = predictor.prepare(image, args.roi)
    except DogLensError as exc:
        print(f"ERROR: {exc.user_message} ({exc.detail or exc.code})", file=sys.stderr)
        return 1
    model = predictor._bundle.model  # read-only use for the isolated forward timing
    spec = predictor.spec
    tensor = preprocess(crop, spec).unsqueeze(0)

    def timed(fn, n: int) -> list[float]:
        out = []
        for _ in range(n):
            s = time.perf_counter()
            fn()
            out.append(time.perf_counter() - s)
        return out

    def forward() -> None:
        with torch.inference_mode():
            model(tensor)

    def prep() -> None:
        c, _ = predictor.prepare(image, args.roi)
        preprocess(c, spec)

    phases = {
        "decode_validate": lambda: load_image(data),
        "preprocessing": prep,
        "model_forward": forward,
        "predict_total": lambda: predictor.predict(image, args.roi, 3),
        "decode_plus_predict": lambda: predictor.predict(load_image(data).image, args.roi, 3),
    }
    results = {}
    for name, fn in phases.items():
        timed(fn, args.warmup)
        results[name] = stats(timed(fn, args.repeats))

    report = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "scripts/benchmark.py "
        + " ".join(
            _portable_path(Path(a)) if Path(a).is_absolute() else a
            for a in (sys.argv[1:] if argv is None else argv)
        ),
        "device": args.device,
        "torch_num_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
        },
        "bundle": {
            "dir": args.bundle.as_posix(),
            "model_version": predictor.model_version,
            "files_sha256": {
                f.name: sha256_file(f) for f in sorted(args.bundle.iterdir()) if f.is_file()
            },
        },
        "input": {
            "image": _portable_path(args.image),
            "bytes": len(data),
            "size_after_exif": list(image.size),
            "user_roi": list(region.user_roi) if region.user_roi else None,
            "crop_box": list(region.crop_box),
            "crop_size": list(crop.size),
            "model_input": [1, 3, spec.crop_size, spec.crop_size],
            "roi_policy": predictor.crop_policy(),
        },
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cold": {"imports_s": import_s, "bundle_load_s": cold_load_s},
        "warm": results,
        "target_note": "goal: warm inference <= 3 s (orientation, not a guarantee)",
    }
    atomic_write_json(args.report, report)
    print(json.dumps({"cold": report["cold"], "warm_median_p95_ms": {
        k: (round(v["median_ms"], 1), round(v["p95_ms"], 1)) for k, v in results.items()}},
        indent=2))  # fmt: skip
    print(f"OK: report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
