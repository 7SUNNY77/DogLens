"""Evaluate a model bundle on a manifest split; write metrics, slices and per-image predictions.

The model is selected on validation only; this script never feeds results back into training.
With --predictions, metrics are recomputed from saved logits without a new forward pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from doglens.bundle import BundleError, load_bundle
from doglens.data_prep import DatasetError
from doglens.dataset import ManifestDataset, load_manifest, make_loader
from doglens.evaluation import (
    SLICES,
    EvaluationError,
    compute_metrics,
    predict_logits,
    predictions_frame,
    slice_mask,
    softmax,
)
from doglens.fileutil import atomic_write_json, atomic_write_text, sha256_file


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a DogLens model bundle on a manifest split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bundle", type=Path, default=Path("artifacts/model_bundle"))
    p.add_argument("--manifest", type=Path, default=Path("data/prepared/manifest.csv"))
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--images-dir", type=Path, default=Path("data/images/Images"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--output-dir", type=Path, default=None, help="default: reports/evaluation/<split>"
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="reuse a saved logits.npz instead of running the model",
    )
    p.add_argument(
        "--update-bundle-metrics",
        action="store_true",
        help="store the summary of this split in <bundle>/metrics.json",
    )
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error("--batch-size must be >= 1")
    return args  # fmt: skip


def _summary_md(report: dict) -> str:
    lines = [
        f"# Evaluation: split `{report['split']}`",
        "",
        f"- Model: `{report['model_version']}` (weights sha256 `{report['weights_sha256'][:16]}…`)",
        f"- Manifest sha256 `{report['manifest_sha256'][:16]}…`; "
        f"generated {report['generated_at_utc']}",
        "- Scores are softmax over 120 supported classes, not calibrated probabilities.",
        "- Macro F1: `f1_score(average='macro', zero_division=0)`;",
        "  *all labels* = fixed 120 labels,",
        "  *present* = classes with support > 0 in the slice. Slices with fewer classes are not",
        "  directly comparable on macro metrics.",
        "",
        "| Slice | Rule | Images | Classes | Top-1 | Top-3 | Macro F1 (120) | Macro F1 (present) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in report["slices"].items():
        if s.get("status") == "empty":
            lines.append(f"| {name} | `{s['rule']}` | 0 | 0 | – | – | – | – |")
            continue
        m = s["metrics"]
        lines.append(
            f"| {name} | `{s['rule']}` | {m['n']} | {m['classes_present']} | {m['top1']:.4f} | "
            f"{m['top3']:.4f} | {m['macro_f1_all_labels']:.4f} | "
            f"{m['macro_f1_present_labels']:.4f} |"
        )
    full = report["slices"]["full"]["metrics"]
    lines += ["", "## Most confused pairs (full slice, true → predicted)", ""]
    lines += ["| True | Predicted | Count | Share of true support |", "|---|---|---|---|"]
    for p in full["most_confused_pairs"][:15]:
        lines.append(
            f"| {p['true_breed_key']} | {p['pred_breed_key']} | {p['count']} | "
            f"{p['share_of_true_support']:.2f} |"
        )
    worst = sorted(
        (c for c in full["per_class"] if c["recall"] is not None), key=lambda c: c["recall"]
    )[:10]
    lines += ["", "## Lowest per-class recall (full slice)", ""]
    lines += ["| Breed | Support | Recall |", "|---|---|---|"]
    lines += [f"| {c['breed_key']} | {c['support']} | {c['recall']:.3f} |" for c in worst]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.output_dir or Path("reports/evaluation") / args.split
    try:
        bundle = load_bundle(args.bundle, "cpu")
        records = load_manifest(args.manifest, [args.split]).sort_values("image_id")
        records = records.reset_index(drop=True)
        labels = [c.class_id for c in bundle.classes]
        keys = {c.class_id: c.breed_key for c in bundle.classes}
        context = bundle.metadata["crop_policy"]["training_and_evaluation"]["context"]
        if args.predictions is not None:
            saved = np.load(args.predictions, allow_pickle=False)
            if list(saved["image_ids"]) != records["image_id"].tolist():
                raise EvaluationError("saved predictions do not match the manifest split")
            if str(saved["weights_sha256"]) != bundle.metadata["weights_sha256"]:
                raise EvaluationError("saved predictions come from different weights")
            logits = saved["logits"]
        else:
            ds = ManifestDataset(records, args.images_dir, bundle.spec, context)
            loader = make_loader(ds, args.batch_size, shuffle=False, seed=0)
            print(f"evaluating {len(ds)} records of split {args.split} on CPU ...", flush=True)
            logits = predict_logits(bundle.model, loader, bundle.device)
        scores = softmax(logits.astype(np.float64))
        if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-6):
            raise EvaluationError("softmax rows do not sum to 1")
        y = records["class_id"].to_numpy()
        slices = {}
        for name, spec in SLICES.items():
            mask = slice_mask(records, name)
            entry = {"rule": spec["rule"], "n": int(mask.sum())}
            if mask.sum() == 0:
                entry["status"] = "empty"
            else:
                entry["metrics"] = compute_metrics(y[mask], scores[mask], labels, keys)
                entry["classes_present"] = entry["metrics"]["classes_present"]
            slices[name] = entry
        report = {
            "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "command": "scripts/evaluate.py " + " ".join(sys.argv[1:] if argv is None else argv),
            "split": args.split,
            "model_version": bundle.metadata["model_version"],
            "weights_sha256": bundle.metadata["weights_sha256"],
            "class_mapping_sha256": bundle.metadata["class_mapping_sha256"],
            "manifest_sha256": sha256_file(args.manifest),
            "preprocessing": bundle.metadata["preprocessing"],
            "crop_context": context,
            "labels": labels,
            "zero_division": 0.0,
            "selection_note": "model chosen on validation only; test never used for selection",
            "limitations": [
                "Stanford Dogs comes from ImageNet and the backbone was initialised with "
                "ImageNet-1k weights: test metrics are optimistic, including the 'clean' slice, "
                "which only removes exact SHA-256 duplicates and label conflicts.",
                "Near-duplicates were not checked.",
            ],
            "slices": slices,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.predictions is None:
            tmp = out_dir / ".logits.npz"
            np.savez_compressed(
                tmp,
                image_ids=np.array(records["image_id"].tolist()),
                labels=y,
                logits=logits.astype(np.float32),
                weights_sha256=np.array(bundle.metadata["weights_sha256"]),
            )
            tmp.replace(out_dir / "logits.npz")
        preds = predictions_frame(records, scores, keys)
        atomic_write_text(
            out_dir / "predictions.csv", preds.to_csv(index=False, lineterminator="\n")
        )
        full = slices["full"]["metrics"]
        cm = pd.DataFrame(full["confusion_matrix"], index=labels, columns=labels)
        atomic_write_text(out_dir / "confusion_matrix.csv", cm.to_csv(lineterminator="\n"))
        atomic_write_json(out_dir / "metrics.json", report)
        atomic_write_text(out_dir / "summary.md", _summary_md(report))
        if args.update_bundle_metrics:
            metrics_path = args.bundle / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            key = "test" if args.split == "test" else "validation_recomputed"
            metrics[key] = {
                "status": "evaluated",
                "evaluated_at_utc": report["generated_at_utc"],
                "manifest_sha256": report["manifest_sha256"],
                "report": (out_dir / "metrics.json").as_posix(),
                "slices": {
                    name: (
                        {"rule": s["rule"], "n": s["n"], "status": "empty"}
                        if s.get("status") == "empty"
                        else {
                            "rule": s["rule"],
                            "n": s["n"],
                            "classes_present": s["metrics"]["classes_present"],
                            "top1": s["metrics"]["top1"],
                            "top3": s["metrics"]["top3"],
                            "macro_f1_all_labels": s["metrics"]["macro_f1_all_labels"],
                            "macro_f1_present_labels": s["metrics"]["macro_f1_present_labels"],
                        }
                    )
                    for name, s in slices.items()
                },
            }
            if args.split == "test":
                metrics["status"] = "evaluated"
            atomic_write_json(metrics_path, metrics)
    except (BundleError, DatasetError, EvaluationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for name, s in slices.items():
        if s.get("status") == "empty":
            print(f"{name}: empty")
        else:
            m = s["metrics"]
            print(
                f"{name}: n={m['n']} classes={m['classes_present']} top1={m['top1']:.4f} "
                f"top3={m['top3']:.4f} macroF1(120)={m['macro_f1_all_labels']:.4f}"
            )
    print(f"OK: reports in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
