"""Stanford Dogs preparation: official split lists, XML annotations, audit, mapping, manifest.

This module validates the *source dataset*. It deliberately does not apply the public upload
limits from ``image_io`` (10 MiB, 20 MP, ROI >= 64 px): those protect the app from user input
and must not silently drop training records. Decoding still refuses truncated files and keeps
Pillow's decompression-bomb protection.

Coordinate convention of the source XML (established on the full dataset, see
``reports/dataset_preparation.md``): origin (0, 0) at the top-left pixel of the image as
stored (no EXIF rotation), ``xmax``/``ymax`` are *inclusive* pixel indices. The project format
is (xmin, ymin, xmax, ymax) with exclusive xmax/ymax, so the conversion adds 1 to the maxima.
"""

from __future__ import annotations

import io
import logging
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import scipy.io
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

from doglens.catalog import ClassInfo, breed_key_from_source_dir, validate_classes
from doglens.config import NUM_CLASSES
from doglens.fileutil import sha256_bytes
from doglens.image_io import to_rgb
from doglens.roi import expand_box
from doglens.schemas import EXCLUDED_COLUMNS, MANIFEST_COLUMNS, Roi

logger = logging.getLogger(__name__)

OFFICIAL_SPLIT_FILES = {"train": "train_list.mat", "test": "test_list.mat"}
LEAKAGE_REASONS = ("dup_with_test", "label_conflict")
FULL_LIST_FILE = "file_list.mat"
COORDINATE_CONVENTION = {
    "source_xml": "origin (0,0) = top-left pixel of the stored image (no EXIF rotation); "
    "xmin/ymin/xmax/ymax are inclusive 0-based pixel indices",
    "project": "(xmin, ymin, xmax, ymax) with exclusive xmax/ymax, matching PIL.Image.crop",
    "conversion": "(xmin, ymin, xmax + 1, ymax + 1)",
    "out_of_bounds": "clipped to the image and reported per record as 'bbox_clipped'",
    "exif": "boxes refer to stored pixels; records with EXIF orientation 2..8 are excluded "
    "and reported (training images are not EXIF-transposed)",
}
CROP_POLICY = {
    "box_selection": "largest valid box by area; ties -> earliest <object> in XML order",
    "context": "expand by context*width on left and right, context*height on top and bottom; "
    "floor for min, ceil for max; clip to image bounds",
    "manifest_bbox": "normalized original box (no context); context is applied on the fly",
    "min_size": "no minimum size for dataset boxes (upload ROI limits do not apply)",
}


class DatasetError(Exception):
    """Fatal preparation error: inputs are missing or globally inconsistent."""


class AnnotationError(ValueError):
    """A single annotation file is malformed."""


# --- official split lists ----------------------------------------------------------------------


@dataclass(frozen=True)
class SplitEntry:
    relative_path: str  # relative to the images dir, POSIX separators, e.g. 'n0...-X/n0..._1.jpg'
    annotation_path: str  # relative to the annotations dir, no extension
    label: int  # official 1-based label


def mat_cell_to_str(cell: Any) -> str:
    """Unwrap scipy's nested cell arrays (``array([array(['x'])])``) into a plain string."""
    value = cell
    while isinstance(value, np.ndarray):
        if value.size != 1:
            raise DatasetError(f"expected a single string in MAT cell, got shape {value.shape}")
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise DatasetError(f"expected a non-empty string in MAT cell, got {type(value).__name__}")
    return str(value)


def safe_relative_path(raw: str) -> str:
    """Normalize to POSIX and refuse absolute paths, drive letters and '..' components."""
    path = PurePosixPath(raw.replace("\\", "/"))
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and ":" in path.parts[0])
        or str(path) in ("", ".")
    ):
        raise DatasetError(f"unsafe relative path in split list: {raw!r}")
    return str(path)


def read_split_mat(path: str | Path) -> list[SplitEntry]:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"official split file not found: {path}")
    try:
        mat = scipy.io.loadmat(str(path))
    except Exception as exc:  # noqa: BLE001 - any parser failure is fatal for this file
        raise DatasetError(f"cannot read MAT file {path}: {exc}") from exc
    missing = [k for k in ("file_list", "annotation_list", "labels") if k not in mat]
    if missing:
        raise DatasetError(f"{path.name}: missing variables {missing}")
    files = np.asarray(mat["file_list"], dtype=object).reshape(-1)
    annotations = np.asarray(mat["annotation_list"], dtype=object).reshape(-1)
    labels = np.asarray(mat["labels"]).reshape(-1)
    if not (len(files) == len(annotations) == len(labels)):
        raise DatasetError(f"{path.name}: list lengths differ")
    if labels.dtype.kind not in "iuf" or not np.all(np.isfinite(labels.astype(float))):
        raise DatasetError(f"{path.name}: labels must be numeric")
    if not np.all(labels == np.round(labels)) or labels.min(initial=1) < 1:
        raise DatasetError(f"{path.name}: labels must be positive integers")
    return [
        SplitEntry(
            relative_path=safe_relative_path(mat_cell_to_str(f)),
            annotation_path=safe_relative_path(mat_cell_to_str(a)),
            label=int(lab),
        )
        for f, a, lab in zip(files, annotations, labels, strict=True)
    ]


def read_official_splits(splits_dir: str | Path) -> dict[str, list[SplitEntry]]:
    splits_dir = Path(splits_dir)
    missing = [n for n in OFFICIAL_SPLIT_FILES.values() if not (splits_dir / n).is_file()]
    if missing:
        raise DatasetError(
            f"official split files not found in {splits_dir}: {missing}. Download lists.tar from "
            "http://vision.stanford.edu/aditya86/ImageNetDogs/ and extract it there. "
            "A non-official split is never substituted automatically."
        )
    return {
        split: read_split_mat(splits_dir / name) for split, name in OFFICIAL_SPLIT_FILES.items()
    }


# --- XML annotations -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RawBox:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class AnnotationRecord:
    filename: str
    width: int
    height: int
    objects: tuple[RawBox, ...]
    invalid_objects: int = 0


def _int_field(node: ET.Element, path: str) -> int:
    text = node.findtext(path)
    if text is None:
        raise AnnotationError(f"missing <{path}>")
    try:
        return int(text.strip())
    except ValueError:
        raise AnnotationError(f"<{path}> is not an integer: {text!r}") from None


def parse_annotation(data: bytes) -> AnnotationRecord:
    """Parse one Stanford Dogs (PASCAL VOC-like) XML file."""
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise AnnotationError("DTD/entities are not allowed in annotations")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise AnnotationError(f"invalid XML: {exc}") from None
    if root.tag != "annotation":
        raise AnnotationError(f"unexpected root element <{root.tag}>")
    width, height = _int_field(root, "size/width"), _int_field(root, "size/height")
    if width <= 0 or height <= 0:
        raise AnnotationError(f"non-positive size {width}x{height}")
    objects: list[RawBox] = []
    invalid = 0
    for obj in root.findall("object"):
        try:
            name = (obj.findtext("name") or "").strip()
            if not name:
                raise AnnotationError("object without <name>")
            objects.append(
                RawBox(
                    name=name,
                    xmin=_int_field(obj, "bndbox/xmin"),
                    ymin=_int_field(obj, "bndbox/ymin"),
                    xmax=_int_field(obj, "bndbox/xmax"),
                    ymax=_int_field(obj, "bndbox/ymax"),
                )
            )
        except AnnotationError:
            invalid += 1
    if not objects:
        raise AnnotationError("no valid <object> with <bndbox>")
    return AnnotationRecord(
        filename=(root.findtext("filename") or "").strip(),
        width=width,
        height=height,
        objects=tuple(objects),
        invalid_objects=invalid,
    )


def xml_box_to_internal(xmin: int, ymin: int, xmax: int, ymax: int) -> Roi:
    """Inclusive 0-based XML box -> project box with exclusive xmax/ymax."""
    return xmin, ymin, xmax + 1, ymax + 1


@dataclass(frozen=True)
class NormalizedBox:
    box: Roi | None  # None when the box is unusable
    issue: str | None  # 'bbox_clipped' | 'bbox_degenerate' | 'bbox_outside_image' | None
    unclipped: Roi


def normalize_box(raw: RawBox, size: tuple[int, int]) -> NormalizedBox:
    width, height = size
    box = xml_box_to_internal(raw.xmin, raw.ymin, raw.xmax, raw.ymax)
    x0, y0, x1, y1 = box
    if x0 >= x1 or y0 >= y1:
        return NormalizedBox(None, "bbox_degenerate", box)
    clipped = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
    if clipped == box:
        return NormalizedBox(box, None, box)
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return NormalizedBox(None, "bbox_outside_image", box)
    return NormalizedBox(clipped, "bbox_clipped", box)


def select_largest_box(boxes: Sequence[Roi]) -> int:
    """Index of the largest box by area; ties resolve to the earliest index (XML order)."""
    if not boxes:
        raise ValueError("no boxes to choose from")
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    return areas.index(max(areas))


def context_box(box: Roi, size: tuple[int, int], context: float) -> Roi:
    """Training/eval crop: the normalized box plus ``context`` on every side, clipped."""
    return expand_box(box, size, context)


# --- images ---------------------------------------------------------------------------------


def exif_orientation(img: Image.Image) -> int | None:
    value = img.getexif().get(0x0112)
    return int(value) if isinstance(value, int) else None


def decode_dataset_image(data: bytes) -> Image.Image:
    """Decode a dataset image by content into RGB (alpha on white), without EXIF transpose.

    Raises on truncated/corrupt files and on Pillow decompression bombs. Upload limits are not
    applied here on purpose.
    """
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise RuntimeError("PIL.ImageFile.LOAD_TRUNCATED_IMAGES must be False")
    img = Image.open(io.BytesIO(data))
    img.load()
    rgb, _ = to_rgb(img)
    return rgb


def load_dataset_image(path: str | Path) -> Image.Image:
    return decode_dataset_image(Path(path).read_bytes())


@dataclass(frozen=True)
class ImageAudit:
    ok: bool
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None
    mode: str | None = None
    orientation: int | None = None
    error: str | None = None


def audit_image(path: Path) -> ImageAudit:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return ImageAudit(ok=False, error=f"read_error: {type(exc).__name__}")
    digest = sha256_bytes(data)
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt, mode, orientation = img.format, img.mode, exif_orientation(img)
            frames = getattr(img, "n_frames", 1)
            img.load()
            width, height = img.size
    except Exception as exc:  # noqa: BLE001
        return ImageAudit(ok=False, sha256=digest, error=f"decode_error: {type(exc).__name__}")
    if frames != 1:
        return ImageAudit(ok=False, sha256=digest, error=f"multi_frame: {frames}")
    return ImageAudit(True, digest, width, height, fmt, mode, orientation)


# --- mapping -------------------------------------------------------------------------------------


def list_class_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        raise DatasetError(f"directory not found: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def build_class_mapping(
    splits: dict[str, list[SplitEntry]],
    image_class_dirs: Sequence[str],
    annotation_class_dirs: Sequence[str],
    num_classes: int = NUM_CLASSES,
) -> list[ClassInfo]:
    """Derive the mapping from the official labels and verify it against both directory trees.

    The directory order is never assumed to match the labels: every label must map to exactly
    one directory (and vice versa) according to the split lists themselves.
    """
    label_dirs: dict[int, set[str]] = defaultdict(set)
    dir_labels: dict[str, set[int]] = defaultdict(set)
    for entries in splits.values():
        for e in entries:
            for rel in (e.relative_path, e.annotation_path):
                class_dir = PurePosixPath(rel).parts[0]
                label_dirs[e.label].add(class_dir)
                dir_labels[class_dir].add(e.label)
    if set(label_dirs) != set(range(1, num_classes + 1)):
        raise DatasetError(
            f"official labels must be exactly 1..{num_classes}, got {sorted(label_dirs)[:5]}..."
        )
    ambiguous = {lab: sorted(d) for lab, d in label_dirs.items() if len(d) != 1}
    ambiguous_dirs = {d: sorted(lab) for d, lab in dir_labels.items() if len(lab) != 1}
    if ambiguous or ambiguous_dirs:
        raise DatasetError(f"label/directory mismatch: {ambiguous or ambiguous_dirs}")
    listed = set(dir_labels)
    for name, actual in (("images", image_class_dirs), ("annotations", annotation_class_dirs)):
        if set(actual) != listed:
            raise DatasetError(
                f"{name} class directories differ from the split lists: "
                f"missing={sorted(listed - set(actual))} extra={sorted(set(actual) - listed)}"
            )
    classes = []
    for label in range(1, num_classes + 1):
        (source_dir,) = label_dirs[label]
        wnid, key, name_en = breed_key_from_source_dir(source_dir)
        classes.append(ClassInfo(label - 1, key, name_en, wnid, source_dir, label))
    validate_classes(classes, num_classes)
    return classes


# --- validation split -------------------------------------------------------------------------


def assign_validation(
    image_ids: Sequence[str],
    class_ids: Sequence[int],
    val_fraction: float,
    seed: int,
    groups: Sequence[str] | None = None,
) -> set[str]:
    """Stratified hold-out from the official train split; independent of input order.

    ``groups`` (e.g. content hashes) keeps exact duplicates on the same side: the split is
    drawn over one representative per group (smallest image_id, its class as stratum) and the
    other members follow it. With duplicate groups the validation size can deviate slightly
    from ``val_fraction``.
    """
    if len(image_ids) != len(class_ids) or not image_ids:
        raise DatasetError("validation split needs a non-empty, aligned train list")
    if groups is not None and len(groups) != len(image_ids):
        raise DatasetError("groups must align with image_ids")
    if len(set(image_ids)) != len(image_ids):
        raise DatasetError("duplicate image_id in train list")
    group_of = list(groups) if groups is not None else list(image_ids)
    members: dict[str, list[str]] = defaultdict(list)
    for image_id, group in zip(image_ids, group_of, strict=True):
        members[group].append(image_id)
    label_of = dict(zip(image_ids, class_ids, strict=True))
    reps = sorted(min(m) for m in members.values())
    rep_members = {min(m): m for m in members.values()}
    try:
        _, val_reps = train_test_split(
            reps,
            test_size=val_fraction,
            stratify=[label_of[r] for r in reps],
            random_state=seed,
            shuffle=True,
        )
    except ValueError as exc:
        raise DatasetError(f"cannot build a stratified validation split: {exc}") from exc
    return {m for r in val_reps for m in rep_members[r]}


# --- full preparation -----------------------------------------------------------------------------


@dataclass
class PreparationResult:
    classes: list[ClassInfo]
    rows: list[dict[str, Any]]
    report: dict[str, Any]
    # Per-record facts used to pick visual-check samples (not written to the manifest).
    record_facts: dict[str, dict[str, Any]] = field(default_factory=dict)


def _issue(issues: list[dict[str, Any]], code: str, image_id: str, **details: Any) -> None:
    issues.append({"code": code, "image_id": image_id, **details})


def prepare(
    images_dir: Path,
    annotations_dir: Path,
    splits_dir: Path,
    *,
    val_fraction: float,
    seed: int,
    bbox_context: float,
    num_classes: int = NUM_CLASSES,
    workers: int = 8,
    progress: Callable[[str], None] | None = None,
) -> PreparationResult:
    say = progress or (lambda _msg: None)
    splits = read_official_splits(splits_dir)
    full_list_path = splits_dir / FULL_LIST_FILE
    image_dirs = list_class_dirs(images_dir)
    annotation_dirs = list_class_dirs(annotations_dir)
    classes = build_class_mapping(splits, image_dirs, annotation_dirs, num_classes)
    class_by_label = {c.source_label: c for c in classes}
    class_by_dir = {c.source_dir: c for c in classes}

    issues: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    fatal_checks: dict[str, Any] = {}

    # Official split integrity.
    official: list[tuple[str, SplitEntry]] = []
    for split in ("train", "test"):
        official.extend((split, e) for e in splits[split])
    by_path: dict[str, list[str]] = defaultdict(list)
    for split, e in official:
        by_path[e.relative_path].append(split)
    duplicated_paths = {p: s for p, s in by_path.items() if len(s) > 1}
    if duplicated_paths:
        raise DatasetError(f"image listed more than once in official splits: {duplicated_paths}")
    train_ids = {PurePosixPath(e.relative_path).stem for e in splits["train"]}
    test_ids = {PurePosixPath(e.relative_path).stem for e in splits["test"]}
    fatal_checks["train_test_id_overlap"] = len(train_ids & test_ids)
    if train_ids & test_ids:
        raise DatasetError(f"train/test image_id overlap: {sorted(train_ids & test_ids)[:10]}")

    if full_list_path.is_file():
        full = read_split_mat(full_list_path)
        union = {(e.relative_path, e.label) for _, e in official}
        fatal_checks["file_list_matches_train_plus_test"] = {
            (e.relative_path, e.label) for e in full
        } == union and len(full) == len(union)
    else:
        fatal_checks["file_list_matches_train_plus_test"] = None

    # Files on disk vs. lists.
    disk_images = {
        f"{d}/{p.name}" for d in image_dirs for p in (images_dir / d).iterdir() if p.is_file()
    }
    disk_annotations = {
        f"{d}/{p.name}" for d in annotation_dirs for p in (annotations_dir / d).iterdir()
    }
    listed_images = set(by_path)
    for rel in sorted(disk_images - listed_images):
        _issue(issues, "unlisted_image", PurePosixPath(rel).stem, relative_path=rel)
    listed_annotations = {e.annotation_path for _, e in official}
    for rel in sorted(disk_annotations - listed_annotations):
        _issue(issues, "unlisted_annotation", PurePosixPath(rel).name, annotation_path=rel)

    say(f"auditing {len(official)} images with {workers} threads ...")
    paths = [images_dir / e.relative_path for _, e in official]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        audits = list(pool.map(audit_image, paths))

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    facts: dict[str, dict[str, Any]] = {}
    hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)  # sha -> (id, split, key)
    stats: Counter[str] = Counter()

    for (split, entry), audit in zip(official, audits, strict=True):
        rel = entry.relative_path
        image_id = PurePosixPath(rel).stem
        cls = class_by_label[entry.label]
        class_dir = PurePosixPath(rel).parts[0]
        if class_by_dir.get(class_dir) is not cls:
            raise DatasetError(f"{rel}: directory does not match official label {entry.label}")

        def exclude(reason: str, _id: str = image_id, _split: str = split, **details: Any) -> None:
            excluded.append({"image_id": _id, "split": _split, "reason": reason, **details})

        if rel not in disk_images:
            exclude("image_missing", relative_path=rel)
            continue
        ann_path = annotations_dir / entry.annotation_path
        if not ann_path.is_file():
            exclude("annotation_missing", annotation_path=entry.annotation_path)
            continue
        if not audit.ok:
            exclude("image_unreadable", error=audit.error)
            continue
        assert audit.sha256 and audit.width and audit.height
        hashes[audit.sha256].append((image_id, split, cls.breed_key))
        stats[f"format_{audit.format}"] += 1
        if audit.format != "JPEG" and rel.lower().endswith((".jpg", ".jpeg")):
            _issue(issues, "extension_format_mismatch", image_id, actual_format=audit.format)
        if audit.mode not in ("RGB",):
            _issue(issues, "non_rgb_mode", image_id, mode=audit.mode, decision="converted to RGB")
        if audit.orientation is not None and audit.orientation != 1:
            if 2 <= audit.orientation <= 8:
                exclude("exif_rotation_unsupported", orientation=audit.orientation)
                continue
            _issue(
                issues,
                "invalid_exif_orientation_value",
                image_id,
                orientation=audit.orientation,
                decision="ignored (no rotation), same as PIL.ImageOps.exif_transpose",
            )
        if audit.orientation == 1:
            stats["exif_orientation_1"] += 1

        try:
            ann = parse_annotation(ann_path.read_bytes())
        except AnnotationError as exc:
            exclude("annotation_invalid", error=str(exc))
            continue
        if ann.filename == "%s":
            # Unfilled template value in the source XML; the file path identifies the record.
            stats["xml_filename_placeholder"] += 1
        elif ann.filename and ann.filename != image_id:
            _issue(issues, "xml_filename_mismatch", image_id, xml_filename=ann.filename)
        if (ann.width, ann.height) != (audit.width, audit.height):
            exclude(
                "xml_size_mismatch",
                xml_size=[ann.width, ann.height],
                image_size=[audit.width, audit.height],
            )
            continue
        if ann.invalid_objects:
            _issue(issues, "invalid_object_ignored", image_id, count=ann.invalid_objects)
        names = {o.name for o in ann.objects}
        expected_name = cls.source_dir.split("-", 1)[1]
        if names != {expected_name}:
            _issue(issues, "object_name_mismatch", image_id, names=sorted(names))

        size = (audit.width, audit.height)
        valid: list[Roi] = []
        for idx, raw in enumerate(ann.objects):
            norm = normalize_box(raw, size)
            if norm.issue:
                _issue(
                    issues,
                    norm.issue,
                    image_id,
                    object_index=idx,
                    xml_box=[raw.xmin, raw.ymin, raw.xmax, raw.ymax],
                    image_size=list(size),
                    result=list(norm.box) if norm.box else None,
                )
            if norm.box is not None:
                valid.append(norm.box)
        if not valid:
            exclude("no_valid_bbox")
            continue
        chosen_idx = select_largest_box(valid)
        bbox = valid[chosen_idx]
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in valid]
        if len(valid) > 1:
            stats["multi_object_records"] += 1
            if areas.count(max(areas)) > 1:
                _issue(issues, "largest_bbox_tie", image_id, chosen_index=chosen_idx)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if min(bw, bh) < 64:
            stats["bbox_min_side_lt_64"] += 1
        rows_by_split[split].append(
            {
                "image_id": image_id,
                "relative_path": rel,
                "class_id": cls.class_id,
                "breed_key": cls.breed_key,
                "split": split,
                "bbox_xmin": bbox[0],
                "bbox_ymin": bbox[1],
                "bbox_xmax": bbox[2],
                "bbox_ymax": bbox[3],
            }
        )
        facts[image_id] = {
            "size": size,
            "n_objects": len(ann.objects),
            "format": audit.format,
            "orientation": audit.orientation,
            "sha256": audit.sha256,
            "bbox": bbox,
            "all_boxes": valid,
        }

    # Validation split from official train only.
    train_rows = rows_by_split["train"]
    val_ids = assign_validation(
        [r["image_id"] for r in train_rows],
        [r["class_id"] for r in train_rows],
        val_fraction,
        seed,
        groups=[facts[r["image_id"]]["sha256"] for r in train_rows],
    )
    for r in train_rows:
        if r["image_id"] in val_ids:
            r["split"] = "val"
    rows = sorted(train_rows + rows_by_split["test"], key=lambda r: r["relative_path"])

    # Exact duplicates by content (splits as assigned, before leakage exclusion).
    duplicate_groups = []
    final_split = {r["image_id"]: r["split"] for r in rows}
    for digest, members in sorted(hashes.items()):
        if len(members) > 1:
            ids = sorted(m[0] for m in members)
            splits_of = sorted({final_split.get(i, "excluded") for i in ids})
            duplicate_groups.append(
                {
                    "sha256": digest,
                    "image_ids": ids,
                    "splits": splits_of,
                    "breed_keys": sorted({m[2] for m in members}),
                    "cross_split": len(splits_of) > 1,
                    "cross_class": len({m[2] for m in members}) > 1,
                }
            )

    # Leakage policy: train/val never contain a copy of an official test image, nor content
    # labelled with different breeds. Official test records are never removed, only flagged.
    leak_reasons: dict[str, list[str]] = defaultdict(list)
    dup_with_test: set[str] = set()
    label_conflict: set[str] = set()
    duplicate_of: dict[str, list[str]] = {}
    for g in duplicate_groups:
        ids = g["image_ids"]
        for i in ids:
            duplicate_of[i] = [j for j in ids if j != i]
        has_test = "test" in g["splits"]
        has_train_val = bool({"train", "val"} & set(g["splits"]))
        for i in ids:
            if g["cross_class"]:
                label_conflict.add(i)
            if final_split.get(i) == "test":
                if has_train_val:
                    dup_with_test.add(i)
                continue
            if has_test:
                leak_reasons[i].append("dup_with_test")
            if g["cross_class"]:
                leak_reasons[i].append("label_conflict")
    kept: list[dict[str, Any]] = []
    for r in rows:
        image_id = r["image_id"]
        r["dup_with_test"] = int(image_id in dup_with_test)
        r["label_conflict"] = int(image_id in label_conflict)
        r["n_boxes"] = facts[image_id]["n_objects"]
        if image_id in leak_reasons:
            excluded.append(
                {
                    "image_id": image_id,
                    "split": r["split"],
                    "reason": "+".join(leak_reasons[image_id]),
                    "relative_path": r["relative_path"],
                    "class_id": r["class_id"],
                    "breed_key": r["breed_key"],
                    "sha256": facts[image_id]["sha256"],
                    "duplicate_of": duplicate_of[image_id],
                }
            )
        else:
            kept.append(r)
    rows = kept

    # Content-level disjointness of the final splits (fatal if violated).
    content_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        content_splits[facts[r["image_id"]]["sha256"]].add(r["split"])
    straddling = sorted(d for d, sp in content_splits.items() if len(sp) > 1)
    fatal_checks["content_overlap_between_splits"] = len(straddling)
    if straddling:
        raise DatasetError(f"identical content in several splits after exclusion: {straddling}")

    leak_excluded = [e for e in excluded if e["reason"].split("+")[0] in LEAKAGE_REASONS]
    leakage = {
        "policy": "remove from train/val every record whose exact content (sha256) also occurs "
        "in official test, and every record whose content occurs with another breed label; "
        "official test is never modified, its records are flagged instead",
        "applied_after_validation_split": True,
        "removed_total": len(leak_excluded),
        "removed_by_reason": dict(sorted(Counter(e["reason"] for e in leak_excluded).items())),
        "removed_by_split": dict(sorted(Counter(e["split"] for e in leak_excluded).items())),
        "removed_by_class": dict(sorted(Counter(e["breed_key"] for e in leak_excluded).items())),
        "test_flagged_dup_with_test": sorted(dup_with_test),
        "flagged_label_conflict_kept": sorted(r["image_id"] for r in rows if r["label_conflict"]),
        "group_breakdown": {
            "cross_split_and_cross_class": sum(
                g["cross_split"] and g["cross_class"] for g in duplicate_groups
            ),
            "cross_split_only": sum(
                g["cross_split"] and not g["cross_class"] for g in duplicate_groups
            ),
            "cross_class_only": sum(
                g["cross_class"] and not g["cross_split"] for g in duplicate_groups
            ),
            "neither": sum(not g["cross_split"] and not g["cross_class"] for g in duplicate_groups),
        },
    }

    counts: dict[str, dict[str, int]] = {
        c.breed_key: {"train": 0, "val": 0, "test": 0} for c in classes
    }
    for r in rows:
        counts[r["breed_key"]][r["split"]] += 1
    split_totals = Counter(r["split"] for r in rows)
    issue_counts = Counter(i["code"] for i in issues)

    report: dict[str, Any] = {
        "official_split": {
            "train_entries": len(splits["train"]),
            "test_entries": len(splits["test"]),
            "checks": fatal_checks,
        },
        "disk": {
            "image_class_dirs": len(image_dirs),
            "annotation_class_dirs": len(annotation_dirs),
            "image_files": len(disk_images),
            "annotation_files": len(disk_annotations),
        },
        "num_classes": len(classes),
        "split_counts": {s: split_totals.get(s, 0) for s in ("train", "val", "test")},
        "total_records": len(rows),
        "class_counts": counts,
        "class_count_range": {
            s: [min(c[s] for c in counts.values()), max(c[s] for c in counts.values())]
            for s in ("train", "val", "test")
        },
        "excluded": excluded,
        "excluded_count": len(excluded),
        "issues": issues,
        "issue_counts": dict(sorted(issue_counts.items())),
        "stats": dict(sorted(stats.items())),
        "duplicates": {
            "method": "sha256 of file bytes",
            "groups": duplicate_groups,
            "group_count": len(duplicate_groups),
            "cross_split_groups": sum(g["cross_split"] for g in duplicate_groups),
            "cross_class_groups": sum(g["cross_class"] for g in duplicate_groups),
            "test_ids_with_train_or_val_duplicate": sorted(
                i
                for g in duplicate_groups
                if "test" in g["splits"] and len(g["splits"]) > 1
                for i in g["image_ids"]
                if final_split.get(i) == "test"
            ),
            "decision": "official test is not modified; duplicates inside official train are "
            "kept on one side of train/val; train/val copies of test images and label-conflict "
            "copies are removed from train/val (see 'leakage')",
            "impact": "train/val<->test copies leak test images into training (optimistic test "
            "metrics); cross-class copies are label conflicts that cap achievable accuracy. "
            "Evaluation should additionally report test metrics without the listed test ids.",
        },
        "leakage": leakage,
        "limitations": [
            "Stanford Dogs is built from ImageNet images and the backbone is initialised with "
            "ImageNet-1k weights, which may have seen some official test images; test metrics are "
            "optimistic and not an independent external evaluation.",
            "Largest-box policy: in multi-dog frames the crop may contain other dogs or, with "
            "noisy annotations, a dog of another breed (see reports/dataset_visual_check.md).",
            "Label-conflict copies remain inside the untouched official test "
            "(flag label_conflict).",
        ],
        "coordinate_convention": COORDINATE_CONVENTION,
        "crop_policy": {**CROP_POLICY, "context": bbox_context},
        "validation_split": {
            "source": "official train only",
            "method": "sklearn.model_selection.train_test_split(stratify=class_id, shuffle=True)"
            " over one representative per exact-duplicate group (smallest image_id), sorted;"
            " other group members follow their representative",
            "duplicate_policy": "exact duplicates inside official train never straddle train/val",
            "val_fraction": val_fraction,
            "seed": seed,
        },
    }
    return PreparationResult(classes=classes, rows=rows, report=report, record_facts=facts)


def excluded_csv_text(excluded: Sequence[dict[str, Any]]) -> str:
    """All excluded records with their reason; list-valued fields are ';'-joined."""
    lines = [",".join(EXCLUDED_COLUMNS)]
    for e in sorted(excluded, key=lambda x: (x["image_id"], x["reason"])):
        values = []
        for c in EXCLUDED_COLUMNS:
            v = e.get(c, "")
            v = ";".join(v) if isinstance(v, list) else str(v)
            if any(ch in v for ch in ',"\n'):
                raise DatasetError(f"unexpected separator in excluded row {e['image_id']}")
            values.append(v)
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def manifest_csv_text(rows: Sequence[dict[str, Any]]) -> str:
    """Deterministic CSV: fixed column order, '\\n' line endings, UTF-8."""
    lines = [",".join(MANIFEST_COLUMNS)]
    for r in rows:
        values = [str(r[c]) for c in MANIFEST_COLUMNS]
        if any("," in v or '"' in v or "\n" in v for v in values):
            raise DatasetError(f"unexpected separator in manifest row {r['image_id']}")
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"


def pick_visual_samples(result: PreparationResult, n: int, seed: int) -> list[tuple[str, str]]:
    """Deterministic (image_id, category) list: edge cases first, then random breeds."""
    rows = {r["image_id"]: r for r in result.rows}
    facts = result.record_facts
    ids = sorted(i for i in facts if i in rows)  # excluded records are not rendered
    chosen: list[tuple[str, str]] = []
    used: set[str] = set()

    def take(category: str, candidates: list[str], k: int = 1) -> None:
        for image_id in candidates:
            if k == 0 or len(chosen) >= n:
                return
            if image_id not in used:
                chosen.append((image_id, category))
                used.add(image_id)
                k -= 1

    def touches(i: str, side: str) -> bool:
        (w, h), b = facts[i]["size"], facts[i]["bbox"]
        return {"left": b[0] == 0, "top": b[1] == 0, "right": b[2] == w, "bottom": b[3] == h}[side]

    clipped = sorted(
        {i["image_id"] for i in result.report["issues"] if i["code"] == "bbox_clipped"}
    )
    take("bbox_clipped", clipped, k=len(clipped))
    take(
        "full_image_box",
        [i for i in ids if all(touches(i, s) for s in ("left", "top", "right", "bottom"))],
    )
    take("touches_left_top", [i for i in ids if touches(i, "left") and touches(i, "top")])
    take("touches_right", [i for i in ids if touches(i, "right") and not touches(i, "left")])
    take("touches_bottom", [i for i in ids if touches(i, "bottom") and not touches(i, "top")])
    multi = sorted(ids, key=lambda i: (-facts[i]["n_objects"], i))
    take("multiple_objects", multi, k=3)
    take("non_jpeg_content", [i for i in ids if facts[i]["format"] != "JPEG"])
    tagged = [i for i in ids if facts[i]["orientation"] is not None]
    # Unusual orientation values first (e.g. the invalid 0), then ordinary tag = 1.
    take(
        "exif_orientation_tag", sorted(tagged, key=lambda i: (facts[i]["orientation"] == 1, i)), k=2
    )
    smallest = sorted(
        ids,
        key=lambda i: (
            min(
                facts[i]["bbox"][2] - facts[i]["bbox"][0], facts[i]["bbox"][3] - facts[i]["bbox"][1]
            ),
            i,
        ),
    )
    take("small_bbox", smallest, k=2)
    rng = random.Random(seed)
    by_breed: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        by_breed[rows[i]["breed_key"]].append(i)
    breeds = sorted(by_breed)
    rng.shuffle(breeds)
    for breed in breeds:
        if len(chosen) >= n:
            break
        take("random_breed", [rng.choice(by_breed[breed])])
    return chosen
