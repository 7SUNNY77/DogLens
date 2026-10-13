"""Bundle export/load and evaluate.py on a tiny random-init run (no network, no dataset)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import sys
from pathlib import Path

import pytest
import torch
from _tiny_bundle import make_tiny_run

from doglens.bundle import BundleError, export_bundle, load_bundle
from doglens.experiment import load_run_checkpoint
from doglens.fileutil import sha256_file
from doglens.model import BreedClassifier

REPO = Path(__file__).resolve().parents[1]
MAPPING = REPO / "data" / "prepared" / "class_mapping.json"
CPU = torch.device("cpu")

pytestmark = pytest.mark.skipif(not MAPPING.is_file(), reason="tracked class mapping missing")


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory) -> dict:
    return make_tiny_run(tmp_path_factory.mktemp("tiny"))


def _export(tiny_run, out: Path, **kw) -> dict:
    return export_bundle(tiny_run["run_dir"] / "best.pt", out, MAPPING, **kw)


def test_export_and_strict_cpu_load_without_network(tiny_run, tmp_path, monkeypatch) -> None:
    out = tmp_path / "bundle"
    meta = _export(tiny_run, out, run_metadata={"run_id": "tiny", "git": {"commit": None}})
    assert sorted(p.name for p in out.iterdir()) == [
        "class_mapping.json",
        "metadata.json",
        "metrics.json",
        "weights.pt",
    ]
    assert meta["num_classes"] == 120 and meta["architecture"]["name"] == "efficientnet_b0"
    assert meta["training"]["git"] == {"commit": None}
    assert meta["source_checkpoint"]["stage"] == "A"
    assert meta["model_version"].startswith("doglens-effb0-tiny-")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test"] == {"status": "not_evaluated"}

    def no_network(*_a, **_k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(torch.hub, "download_url_to_file", no_network)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", no_network)
    bundle = load_bundle(out, "cpu")
    assert not bundle.model.training
    reference = BreedClassifier()
    reference.load_state_dict(load_run_checkpoint(tiny_run["run_dir"] / "best.pt")["model_state"])
    reference.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        assert torch.equal(reference(x), bundle.model(x))
    assert [c.breed_key for c in bundle.classes][:2] == ["chihuahua", "japanese_spaniel"]


def test_export_refuses_non_empty_dir_unless_overwrite(tiny_run, tmp_path) -> None:
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(BundleError, match="not empty"):
        _export(tiny_run, out)
    assert (out / "keep.txt").is_file()
    _export(tiny_run, out, overwrite=True)
    assert not (out / "keep.txt").exists() and (out / "weights.pt").is_file()
    assert not list(tmp_path.glob(".bundle_*")) and not (tmp_path / "bundle.old").exists()


def test_export_rejects_mapping_mismatch(tiny_run, tmp_path) -> None:
    other = tmp_path / "mapping.json"
    raw = json.loads(MAPPING.read_text(encoding="utf-8"))
    raw["classes"][0]["name_en"] = "Changed"
    from doglens.catalog import ClassInfo, mapping_to_dict

    other.write_text(
        json.dumps(mapping_to_dict([ClassInfo(**c) for c in raw["classes"]])), encoding="utf-8"
    )
    with pytest.raises(BundleError, match="mapping"):
        export_bundle(tiny_run["run_dir"] / "best.pt", tmp_path / "b", other)


@pytest.mark.parametrize("damage", ["missing", "weights", "mapping", "schema", "keys"])
def test_load_rejects_damaged_bundles(tiny_run, tmp_path, damage) -> None:
    out = tmp_path / "bundle"
    _export(tiny_run, out)
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if damage == "missing":
        (out / "metrics.json").unlink()
    elif damage == "weights":
        with open(out / "weights.pt", "ab") as fh:
            fh.write(b"tampered")
    elif damage == "mapping":
        raw = json.loads((out / "class_mapping.json").read_text(encoding="utf-8"))
        raw["mapping_sha256"] = "0" * 64
        (out / "class_mapping.json").write_text(json.dumps(raw), encoding="utf-8")
    elif damage == "schema":
        meta["schema_version"] = 99
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    elif damage == "keys":
        torch.save({"fc.weight": torch.zeros(120, 1280)}, out / "weights.pt")
        meta["weights_sha256"] = sha256_file(out / "weights.pt")
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(BundleError) as info:
        load_bundle(out)
    assert "Модель не найдена" in info.value.user_message


def test_missing_bundle_has_clear_message(tmp_path) -> None:
    with pytest.raises(BundleError) as info:
        load_bundle(tmp_path / "absent")
    assert "scripts/train.py" in info.value.user_message


def _cli(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_and_evaluate_cli(tiny_run, tmp_path, capsys) -> None:
    out = tmp_path / "bundle"
    export_cli, eval_cli = _cli("export_bundle"), _cli("evaluate")
    args = ["--run-dir", str(tiny_run["run_dir"]), "--output-dir", str(out)]
    assert export_cli.main([*args, "--mapping", str(MAPPING)]) == 0
    assert "CPU reload verified" in capsys.readouterr().out
    assert export_cli.main([*args, "--mapping", str(MAPPING)]) == 1  # non-empty, no overwrite
    report_dir = tmp_path / "eval"
    common = ["--bundle", str(out), "--manifest", str(tiny_run["manifest"]), "--split", "test",
              "--images-dir", str(tiny_run["images"]), "--output-dir", str(report_dir)]  # fmt: skip
    assert eval_cli.main([*common, "--update-bundle-metrics"]) == 0
    report = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert report["labels"] == list(range(120)) and report["zero_division"] == 0.0
    assert report["slices"]["full"]["n"] == 12
    assert report["slices"]["clean"]["n"] == 6  # dup_with_test == 1 for half the records
    assert report["slices"]["multi_box"]["n"] == 4
    assert report["slices"]["single_box"]["n"] == 8
    assert report["slices"]["full"]["metrics"]["classes_present"] == 3
    for name in ("predictions.csv", "confusion_matrix.csv", "logits.npz", "summary.md"):
        assert (report_dir / name).is_file()
    bundle_metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert bundle_metrics["status"] == "evaluated"
    assert bundle_metrics["test"]["slices"]["full"]["n"] == 12
    # Recompute from saved logits without a forward pass: identical metrics.
    again = tmp_path / "eval2"
    common2 = [*common[:-1], str(again), "--predictions", str(report_dir / "logits.npz")]
    assert eval_cli.main(common2) == 0
    report2 = json.loads((again / "metrics.json").read_text(encoding="utf-8"))
    assert report2["slices"] == report["slices"]
    shutil.rmtree(again)
