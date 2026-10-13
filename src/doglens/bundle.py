"""Model bundle: export of the selected training checkpoint and verified loading for inference.

artifacts/model_bundle/{weights.pt, metadata.json, class_mapping.json, metrics.json}

``weights.pt`` is the trained DogLens state_dict (not the ImageNet initialisation). Loading
creates the architecture with ``weights=None``, never downloads anything, verifies schema,
checksums, mapping and tensor shapes, loads the state_dict strictly and runs a smoke forward.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
import torchvision

from doglens.catalog import ClassInfo, load_class_mapping, mapping_sha256
from doglens.config import NUM_CLASSES
from doglens.data_prep import CROP_POLICY
from doglens.errors import DogLensError
from doglens.fileutil import atomic_write_json, sha256_file
from doglens.model import ARCHITECTURE, BreedClassifier
from doglens.preprocessing import PreprocessingSpec, spec_from_dict
from doglens.schemas import BUNDLE_FILES, BUNDLE_SCHEMA_VERSION

MISSING_BUNDLE_HINT = (
    "Модель не найдена или повреждена. Обучите модель (scripts/train.py), экспортируйте её "
    "(scripts/export_bundle.py) в artifacts/model_bundle и перезапустите приложение."
)


class BundleError(DogLensError):
    def __init__(self, detail: str) -> None:
        super().__init__("bundle_invalid", MISSING_BUNDLE_HINT, detail)


@dataclass
class LoadedBundle:
    model: BreedClassifier
    metadata: dict[str, Any]
    classes: list[ClassInfo]
    spec: PreprocessingSpec
    metrics: dict[str, Any]
    device: torch.device


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read {path.name}: {exc}") from exc


def package_versions() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }


def export_bundle(
    checkpoint: Path,
    out_dir: Path,
    mapping_path: Path,
    *,
    run_metadata: dict[str, Any] | None = None,
    pretrained_source: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a complete bundle atomically (temp dir + rename). Returns metadata."""
    # Training-checkpoint reader (incl. its legacy allow-list) is needed only for export;
    # load_bundle() below never uses it.
    from doglens.experiment import load_run_checkpoint

    payload = load_run_checkpoint(checkpoint)
    identity, state, config = payload["identity"], payload["state"], payload["config"]
    classes = load_class_mapping(mapping_path)
    digest = mapping_sha256(classes)
    if identity["mapping_sha256"] != digest:
        raise BundleError("checkpoint mapping_sha256 differs from the class mapping file")
    spec = spec_from_dict(identity["preprocessing"])
    model = BreedClassifier(len(classes))
    model.load_state_dict(payload["model_state"], strict=True)
    weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise BundleError(f"{out_dir} is not empty; pass overwrite to replace the bundle")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".bundle_", dir=out_dir.parent))
    try:
        torch.save(weights, tmp / "weights.pt")
        weights_sha = sha256_file(tmp / "weights.pt")
        shutil.copyfile(mapping_path, tmp / "class_mapping.json")
        best = state["global_best"]
        run_id = (run_metadata or {}).get("run_id") or Path(checkpoint).parent.name
        model_version = f"doglens-effb0-{run_id}-{weights_sha[:8]}"
        history = state["history"]
        best_record = next(
            h
            for h in history
            if h["stage"] == best["stage"] and h["stage_epoch"] == best["stage_epoch"]
        )
        metadata = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "model_version": model_version,
            "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "architecture": {
                "name": ARCHITECTURE,
                "builder": "torchvision.models.efficientnet_b0(weights=None)",
                "head": "classifier[1] replaced by Linear(1280, num_classes)",
                "state_dict_prefix": "net.",
                "wrapper": "doglens.model.BreedClassifier",
            },
            "num_classes": len(classes),
            "class_mapping_sha256": digest,
            "weights_sha256": weights_sha,
            "preprocessing": spec.to_dict(),
            "crop_policy": {
                "training_and_evaluation": {
                    **CROP_POLICY,
                    "context": config["data"]["bbox_context"],
                },
                "app": "decided in the predictor stage; must be documented there",
            },
            "pretrained_initialisation": {
                "weights_enum": config["pretrained"]["weights_enum"],
                "sha256": config["pretrained"]["sha256"],
                "source": pretrained_source,
                "note": "ImageNet-1k weights used only to initialise training; weights.pt is the "
                "trained DogLens model",
            },
            "source_checkpoint": {
                "file": Path(checkpoint).as_posix(),
                "sha256": sha256_file(checkpoint),
                "run_id": run_id,
                "stage": best["stage"],
                "stage_epoch": best["stage_epoch"],
                "global_epoch": best["global_epoch"],
                "selection_rule": "max validation Top-1; ties -> lower validation loss",
            },
            "training": {
                "seed": config["seed"],
                "manifest_sha256": identity["manifest_sha256"],
                "device": identity["device"],
                "git": (run_metadata or {}).get("git", {"commit": None, "dirty": None}),
                "stage_stop_reasons": state.get("stage_stop_reasons"),
                "epochs_run": len(history),
            },
            "package_versions_export": package_versions(),
            "package_versions_training": {
                "torch": identity.get("torch"),
                "torchvision": identity.get("torchvision"),
            },
        }
        metrics = {
            "status": "validation_only",
            "model_version": model_version,
            "validation": {
                "split": "val",
                "n": best_record["val_n"],
                "top1": best_record["val_top1"],
                "top3": best_record["val_top3"],
                "loss": best_record["val_loss"],
                "source": "training history of the selected epoch",
            },
            "test": {"status": "not_evaluated"},
        }
        atomic_write_json(tmp / "metadata.json", metadata)
        atomic_write_json(tmp / "metrics.json", metrics)
        if out_dir.exists():
            backup = out_dir.with_name(out_dir.name + ".old")
            if backup.exists():
                shutil.rmtree(backup)
            out_dir.rename(backup)
            tmp.rename(out_dir)
            shutil.rmtree(backup)
        else:
            tmp.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return metadata


def load_bundle(bundle_dir: str | Path, device: str | torch.device = "cpu") -> LoadedBundle:
    bundle_dir = Path(bundle_dir)
    missing = [f for f in BUNDLE_FILES if not (bundle_dir / f).is_file()]
    if missing:
        raise BundleError(f"bundle {bundle_dir} is missing {missing}")
    metadata = _read_json(bundle_dir / "metadata.json")
    if metadata.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleError(f"unsupported bundle schema {metadata.get('schema_version')}")
    if metadata.get("architecture", {}).get("name") != ARCHITECTURE:
        raise BundleError("unsupported architecture")
    num_classes = metadata.get("num_classes")
    if num_classes != NUM_CLASSES:
        raise BundleError(f"bundle has {num_classes} classes, expected {NUM_CLASSES}")
    if sha256_file(bundle_dir / "weights.pt") != metadata.get("weights_sha256"):
        raise BundleError("weights.pt checksum does not match metadata")
    try:
        classes = load_class_mapping(bundle_dir / "class_mapping.json")
    except DogLensError as exc:
        raise BundleError(f"invalid class mapping: {exc}") from exc
    if mapping_sha256(classes) != metadata.get("class_mapping_sha256"):
        raise BundleError("class mapping does not match metadata")
    try:
        spec = spec_from_dict(metadata["preprocessing"])
    except (DogLensError, KeyError) as exc:
        raise BundleError(f"invalid preprocessing: {exc}") from exc
    try:
        state = torch.load(bundle_dir / "weights.pt", map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise BundleError(f"cannot load weights.pt safely: {exc}") from exc
    model = BreedClassifier(num_classes)  # architecture only, weights=None
    expected = model.state_dict()
    if set(state) != set(expected):
        raise BundleError("weights.pt keys do not match the architecture")
    bad = [k for k in expected if tuple(state[k].shape) != tuple(expected[k].shape)]
    if bad:
        raise BundleError(f"tensor shape mismatch: {bad[:5]}")
    model.load_state_dict(state, strict=True)
    dev = torch.device(device)
    model.to(dev).eval()
    with torch.inference_mode():
        probe = model(torch.zeros(1, 3, spec.crop_size, spec.crop_size, device=dev))
    if tuple(probe.shape) != (1, num_classes) or not torch.isfinite(probe).all():
        raise BundleError("smoke forward failed")
    metrics = _read_json(bundle_dir / "metrics.json")
    return LoadedBundle(model, metadata, classes, spec, metrics, dev)
