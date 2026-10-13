"""Package the trained model bundle as a release archive, or verify/unpack such an archive.

  pack:   validate artifacts/model_bundle (schema, checksums, mapping, strict load), write
          dist/<model_version>.zip with exactly one directory model_bundle/ holding
          weights.pt, metadata.json, class_mapping.json, metrics.json, plus an external
          <archive>.sha256 file; then verify the archive in a temporary directory.
  verify: check the archive SHA-256, extract safely and load the bundle with an offline
          inference smoke test. With --extract-to, the bundle is unpacked to
          <extract-to>/model_bundle (e.g. --extract-to artifacts).

Weights, datasets and training checkpoints never go into Git; the archive is meant to be a
GitHub Release asset once the owner publishes a release.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import torch

from doglens.bundle import BundleError, load_bundle
from doglens.fileutil import sha256_file
from doglens.schemas import BUNDLE_FILES

ARCHIVE_DIR = "model_bundle"
FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)  # deterministic archive bytes


class PackageError(RuntimeError):
    pass


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package or verify the DogLens model bundle archive.")
    sub = p.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack", help="create dist/<model_version>.zip (+ .sha256)")
    pack.add_argument("--bundle", type=Path, default=Path("artifacts/model_bundle"))
    pack.add_argument("--output-dir", type=Path, default=Path("dist"))
    pack.add_argument("--overwrite", action="store_true")
    verify = sub.add_parser("verify", help="check SHA-256, unpack and load an archive")
    verify.add_argument("archive", type=Path)
    verify.add_argument(
        "--sha256", default=None, help="expected SHA-256 (default: read <archive>.sha256)"
    )
    verify.add_argument(
        "--extract-to",
        type=Path,
        default=None,
        help="parent directory for model_bundle/ (e.g. artifacts)",
    )
    verify.add_argument(
        "--overwrite", action="store_true", help="replace an existing <extract-to>/model_bundle"
    )
    return p.parse_args(argv)  # fmt: skip


def _no_network(*_a, **_k):
    raise PackageError("network access attempted during offline bundle check")


def offline_smoke(bundle_dir: Path) -> str:
    """Load the bundle and run one forward pass with outgoing connections disabled."""
    original = socket.socket.connect
    socket.socket.connect = _no_network  # type: ignore[method-assign]
    try:
        bundle = load_bundle(bundle_dir, "cpu")
        size = bundle.spec.crop_size
        with torch.inference_mode():
            probs = torch.softmax(bundle.model(torch.zeros(1, 3, size, size)), dim=1)
        if not torch.isfinite(probs).all() or abs(float(probs.sum()) - 1.0) > 1e-4:
            raise PackageError("offline smoke forward produced invalid scores")
        return bundle.metadata["model_version"]
    finally:
        socket.socket.connect = original  # type: ignore[method-assign]


def safe_extract(archive: Path, target: Path) -> Path:
    """Extract only model_bundle/<required file>; refuse anything else."""
    expected = {f"{ARCHIVE_DIR}/{name}" for name in BUNDLE_FILES}
    with zipfile.ZipFile(archive) as zf:
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise PackageError(f"unsafe path in archive: {name!r}")
        if set(names) != expected or len(names) != len(expected):
            raise PackageError(f"archive must contain exactly {sorted(expected)}, got {names}")
        for name in names:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    return target / ARCHIVE_DIR


def pack(bundle_dir: Path, output_dir: Path, overwrite: bool) -> tuple[Path, str]:
    present = sorted(p.name for p in bundle_dir.iterdir()) if bundle_dir.is_dir() else []
    extra = set(present) - set(BUNDLE_FILES)
    if extra:
        raise PackageError(f"unexpected files in {bundle_dir}: {sorted(extra)}")
    model_version = offline_smoke(bundle_dir)  # full validation of the source bundle
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{model_version}.zip"
    checksum_file = archive.with_name(archive.name + ".sha256")
    if (archive.exists() or checksum_file.exists()) and not overwrite:
        raise PackageError(f"{archive} already exists; pass --overwrite to rebuild it")
    tmp = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(BUNDLE_FILES):
            info = zipfile.ZipInfo(f"{ARCHIVE_DIR}/{name}", date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, (bundle_dir / name).read_bytes())
    tmp.replace(archive)
    digest = sha256_file(archive)
    # LF on every OS so that `sha256sum -c` also works on Linux/macOS.
    checksum_file.write_text(f"{digest}  {archive.name}\n", encoding="utf-8", newline="\n")
    return archive, digest


def verify(archive: Path, expected: str | None, extract_to: Path | None, overwrite: bool) -> str:
    if not archive.is_file():
        raise PackageError(f"archive not found: {archive}")
    if expected is None:
        checksum_file = archive.with_name(archive.name + ".sha256")
        if not checksum_file.is_file():
            raise PackageError(f"no --sha256 given and {checksum_file.name} not found")
        expected = checksum_file.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(archive)
    if actual != expected.strip().lower():
        raise PackageError(f"SHA-256 mismatch: expected {expected}, got {actual}")
    with tempfile.TemporaryDirectory(prefix="doglens_bundle_") as tmp:
        bundle_dir = safe_extract(archive, Path(tmp))
        model_version = offline_smoke(bundle_dir)
        if extract_to is not None:
            target = extract_to / ARCHIVE_DIR
            if target.exists() and any(target.iterdir()):
                if not overwrite:
                    raise PackageError(f"{target} is not empty; pass --overwrite to replace it")
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(bundle_dir, target, dirs_exist_ok=True)
    return model_version


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "pack":
            archive, digest = pack(args.bundle, args.output_dir, args.overwrite)
            model_version = verify(archive, digest, None, False)
            print(f"OK: {archive} ({archive.stat().st_size} bytes)")
            print(f"SHA-256: {digest}  (also in {archive.name}.sha256)")
            print(f"verified: unpacked to a temp dir, loaded {model_version}, offline inference ok")
        else:
            model_version = verify(args.archive, args.sha256, args.extract_to, args.overwrite)
            where = f"; unpacked to {args.extract_to / ARCHIVE_DIR}" if args.extract_to else ""
            print(f"OK: SHA-256 verified, {model_version} loads, offline inference ok{where}")
    except (PackageError, BundleError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
