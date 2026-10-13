"""Tiny random-init training run + exported bundle for tests. Test-only, no real photos."""

from __future__ import annotations

from pathlib import Path

import torch
from test_dataset_loader import _write_images
from test_experiment import SMALL_SPEC, _config

from doglens.bundle import export_bundle
from doglens.catalog import load_class_mapping, mapping_sha256
from doglens.dataset import ManifestDataset, make_loader, seed_everything
from doglens.experiment import Experiment
from doglens.fileutil import atomic_write_json, sha256_file
from doglens.model import BreedClassifier

REPO = Path(__file__).resolve().parents[1]
MAPPING = REPO / "data" / "prepared" / "class_mapping.json"
CPU = torch.device("cpu")


def make_tiny_run(root: Path) -> dict:
    df = _write_images(root / "img", n=12)
    df["split"] = "test"
    df["dup_with_test"] = [1, 0] * 6
    df["n_boxes"] = [1, 1, 2] * 4
    manifest = root / "manifest.csv"
    df.to_csv(manifest, index=False)
    classes = load_class_mapping(MAPPING)
    df["breed_key"] = [classes[c].breed_key for c in df["class_id"]]
    df.to_csv(manifest, index=False)
    seed_everything(42)
    torch.manual_seed(0)
    cfg = _config(stage_a={"epochs": 1, "lr": 0.01})
    cfg["stage_b"]["epochs"] = 0
    run_dir = root / "run"
    run_dir.mkdir()
    exp = Experiment(
        model=BreedClassifier(),
        train_loader=make_loader(ManifestDataset(df, root / "img", SMALL_SPEC, 0.1), 4, True, 42),
        val_loader=make_loader(ManifestDataset(df, root / "img", SMALL_SPEC, 0.1), 4, False, 42),
        device=CPU,
        run_dir=run_dir,
        config=cfg,
        identity={
            "mapping_sha256": mapping_sha256(classes),
            "manifest_sha256": sha256_file(manifest),
            "preprocessing": SMALL_SPEC.to_dict(),
            "pretrained_sha256": cfg["pretrained"]["sha256"],
            "device": "cpu",
            "torch": torch.__version__,
            "torchvision": "x",
        },
        log=lambda _m: None,
        log_every=0,
    )
    exp.run()
    atomic_write_json(run_dir / "final_report.json", {"status": "completed"})
    atomic_write_json(run_dir / "run_metadata.json", {"run_id": "tiny", "git": {"commit": None}})
    return {"root": root, "run_dir": run_dir, "manifest": manifest, "images": root / "img"}


def make_tiny_bundle(root: Path) -> dict:
    run = make_tiny_run(root)
    export_bundle(run["run_dir"] / "best.pt", root / "bundle", MAPPING,
                  run_metadata={"run_id": "tiny", "git": {"commit": None}})  # fmt: skip
    return {**run, "bundle": root / "bundle"}
