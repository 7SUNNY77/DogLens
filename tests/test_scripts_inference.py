"""benchmark.py and check_photos.py on the tiny bundle with synthetic images (test-only)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def _cli(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_benchmark_cli(tiny_bundle, tmp_path) -> None:
    img = tmp_path / "x.jpg"
    Image.new("RGB", (300, 200), (50, 90, 130)).save(img, "JPEG")
    report = tmp_path / "bench.json"
    cli = _cli("benchmark")
    args = ["--bundle", str(tiny_bundle["bundle"]), "--image", str(img), "--roi", "20,20,200,180",
            "--warmup", "1", "--repeats", "3", "--report", str(report)]  # fmt: skip
    assert cli.main(args) == 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["batch_size"] == 1 and data["device"] == "cpu"
    assert data["input"]["user_roi"] == [20, 20, 200, 180]
    assert data["input"]["crop_box"] == [2, 4, 218, 196]
    for phase in ("preprocessing", "model_forward", "predict_total", "decode_plus_predict"):
        assert data["warm"][phase]["n"] == 3 and data["warm"][phase]["median_ms"] > 0
    assert "bundle_load_s" in data["cold"]
    assert "weights.pt" in data["bundle"]["files_sha256"]
    for bad in (["--image", str(tmp_path / "absent.jpg")], ["--image", str(img), "--roi", "1,2,3"]):
        with pytest.raises(SystemExit) as info:
            cli.main(bad)
        assert info.value.code == 2


def test_check_photos_cli(tiny_bundle, tmp_path) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    Image.new("RGB", (300, 200), (200, 10, 10)).save(photos / "a.jpg", "JPEG")
    Image.new("RGB", (300, 200), (10, 200, 10)).save(photos / "b.png", "PNG")
    (photos / "broken.webp").write_bytes(b"not an image")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "file,expected,xmin,ymin,xmax,ymax\na.jpg,beagle,10,10,200,190\nb.png,not_dog,,,,\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    cli = _cli("check_photos")
    args = ["--input-dir", str(photos), "--labels", str(labels), "--output-dir", str(out),
            "--bundle", str(tiny_bundle["bundle"])]  # fmt: skip
    assert cli.main(args) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["photos"] == 3 and summary["errors"] == 1
    assert summary["supported_breed_photos"] == 1  # not_dog is not counted in accuracy
    assert summary["limitation_cases"]["not_dog"] == 1
    lines = (out / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert lines[1].startswith("a.jpg,beagle,ok,10 10 200 190,")
    bad = tmp_path / "bad.csv"
    bad.write_text("file,expected\na.jpg,unicorn\n", encoding="utf-8")
    assert cli.main([*args[:2], "--labels", str(bad), *args[4:]]) == 1
