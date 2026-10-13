"""scripts/package_bundle.py on the tiny test bundle."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _cli():
    spec = importlib.util.spec_from_file_location(
        "package_bundle", REPO / "scripts" / "package_bundle.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["package_bundle"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pack_verify_and_extract(tiny_bundle, tmp_path) -> None:
    cli = _cli()
    dist = tmp_path / "dist"
    assert (
        cli.main(["pack", "--bundle", str(tiny_bundle["bundle"]), "--output-dir", str(dist)]) == 0
    )
    (archive,) = dist.glob("*.zip")
    names = sorted(zipfile.ZipFile(archive).namelist())
    assert names == [
        "model_bundle/class_mapping.json",
        "model_bundle/metadata.json",
        "model_bundle/metrics.json",
        "model_bundle/weights.pt",
    ]
    raw_checksum = (dist / (archive.name + ".sha256")).read_bytes()
    assert b"\r" not in raw_checksum  # `sha256sum -c` compatible on Linux/macOS
    checksum = raw_checksum.decode("utf-8").split()
    assert checksum[1] == archive.name and len(checksum[0]) == 64
    # Deterministic bytes: rebuilding gives the same archive and checksum.
    first = archive.read_bytes()
    assert (
        cli.main(["pack", "--bundle", str(tiny_bundle["bundle"]), "--output-dir", str(dist)]) == 1
    )
    assert cli.main(["pack", "--bundle", str(tiny_bundle["bundle"]), "--output-dir", str(dist),
                     "--overwrite"]) == 0  # fmt: skip
    assert archive.read_bytes() == first
    target = tmp_path / "artifacts"
    assert cli.main(["verify", str(archive), "--extract-to", str(target)]) == 0
    assert sorted(p.name for p in (target / "model_bundle").iterdir()) == [
        n.split("/")[1] for n in names
    ]
    assert cli.main(["verify", str(archive), "--extract-to", str(target)]) == 1  # not empty
    assert cli.main(["verify", str(archive), "--sha256", "0" * 64]) == 1


def test_pack_refuses_extra_files_and_verify_refuses_foreign_archives(
    tiny_bundle, tmp_path
) -> None:
    cli = _cli()
    dirty = tmp_path / "bundle"
    shutil.copytree(tiny_bundle["bundle"], dirty)
    (dirty / "notes.txt").write_text("x", encoding="utf-8")
    assert cli.main(["pack", "--bundle", str(dirty), "--output-dir", str(tmp_path / "d")]) == 1
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "x")
    from doglens.fileutil import sha256_file

    assert cli.main(["verify", str(evil), "--sha256", sha256_file(evil)]) == 1
    assert not (tmp_path.parent / "escape.txt").exists()
