"""Checks of the real prepared Stanford Dogs outputs. Run with: uv run pytest -m integration

Skipped (not passed) when the dataset or prepared outputs are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from doglens.catalog import check_catalog_matches_mapping, load_catalog, load_class_mapping
from doglens.data_prep import read_official_splits
from doglens.fileutil import sha256_file
from doglens.schemas import MANIFEST_COLUMNS

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[1]
PREPARED = REPO / "data" / "prepared"
IMAGES = REPO / "data" / "images" / "Images"
SPLITS = REPO / "data" / "splits"


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    path = PREPARED / "manifest.csv"
    if not path.is_file():
        pytest.skip("data/prepared/manifest.csv not present; run scripts/prepare_dataset.py")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def report() -> dict:
    path = PREPARED / "preparation_report.json"
    if not path.is_file():
        pytest.skip("preparation_report.json not present")
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_schema_and_counts(manifest: pd.DataFrame, report: dict) -> None:
    assert list(manifest.columns) == list(MANIFEST_COLUMNS)
    assert manifest["image_id"].is_unique and manifest["relative_path"].is_unique
    counts = manifest["split"].value_counts().to_dict()
    assert counts == report["split_counts"]
    assert counts["test"] == 8580
    assert counts["train"] + counts["val"] == 12000 - sum(
        e["split"] in ("train", "val") for e in report["excluded"]
    )
    # Duplicate grouping and leakage removal shift a few records; 4.0 removed 67 in total.
    assert abs(counts["val"] - 2400) <= 50
    assert manifest["class_id"].nunique() == 120
    assert set(manifest["class_id"]) == set(range(120))


def test_official_membership_is_preserved(manifest: pd.DataFrame) -> None:
    if not ((SPLITS / "train_list.mat").is_file() and (SPLITS / "test_list.mat").is_file()):
        pytest.skip("official split files not present")
    splits = read_official_splits(SPLITS)
    official_test = {e.relative_path for e in splits["test"]}
    official_train = {e.relative_path for e in splits["train"]}
    by_split = manifest.groupby("split")["relative_path"].apply(set).to_dict()
    assert by_split["test"] <= official_test
    assert (by_split["train"] | by_split["val"]) <= official_train
    assert not (by_split["train"] & by_split["val"])
    labels = {e.relative_path: e.label - 1 for es in splits.values() for e in es}
    assert all(
        labels[p] == c for p, c in zip(manifest["relative_path"], manifest["class_id"], strict=True)
    )


def test_boxes_inside_images_and_files_exist(manifest: pd.DataFrame) -> None:
    if not IMAGES.is_dir():
        pytest.skip("images not present")
    assert (manifest["bbox_xmin"] >= 0).all() and (manifest["bbox_ymin"] >= 0).all()
    assert (manifest["bbox_xmax"] > manifest["bbox_xmin"]).all()
    assert (manifest["bbox_ymax"] > manifest["bbox_ymin"]).all()
    sample = manifest.sample(200, random_state=0)
    from PIL import Image

    for row in sample.itertuples():
        path = IMAGES / row.relative_path
        assert path.is_file()
        with Image.open(path) as img:
            assert row.bbox_xmax <= img.width and row.bbox_ymax <= img.height


def test_mapping_catalog_and_report_consistency(manifest: pd.DataFrame, report: dict) -> None:
    classes = load_class_mapping(PREPARED / "class_mapping.json")
    digest, entries = load_catalog(REPO / "assets" / "breeds.json")
    check_catalog_matches_mapping(digest, entries, classes)
    key_of = {c.class_id: c.breed_key for c in classes}
    assert all(
        key_of[c] == k for c, k in zip(manifest["class_id"], manifest["breed_key"], strict=True)
    )
    assert report["outputs"]["mapping_sha256"] == digest
    assert report["outputs"]["manifest_sha256"] == sha256_file(PREPARED / "manifest.csv")
    if (SPLITS / "train_list.mat").is_file():
        assert report["inputs"]["split_files_sha256"]["train_list.mat"] == sha256_file(
            SPLITS / "train_list.mat"
        )


def test_duplicates_never_straddle_train_and_val(manifest: pd.DataFrame, report: dict) -> None:
    split = dict(zip(manifest["image_id"], manifest["split"], strict=True))
    for group in report["duplicates"]["groups"]:
        sides = {split[i] for i in group["image_ids"] if split.get(i) in ("train", "val")}
        assert len(sides) <= 1, group


def test_splits_are_disjoint_by_file_content(manifest: pd.DataFrame) -> None:
    """No SHA-256 of image bytes may occur in more than one split (train/val/test)."""
    if not IMAGES.is_dir():
        pytest.skip("images not present")
    from concurrent.futures import ThreadPoolExecutor

    paths = [IMAGES / p for p in manifest["relative_path"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        digests = list(pool.map(sha256_file, paths))
    splits_of: dict[str, set[str]] = {}
    labels_of: dict[str, set[int]] = {}
    for digest, split, cls in zip(digests, manifest["split"], manifest["class_id"], strict=True):
        splits_of.setdefault(digest, set()).add(split)
        labels_of.setdefault(digest, set()).add(int(cls))
    assert [d for d, s in splits_of.items() if len(s) > 1] == []
    # Label conflicts may only remain inside the untouched official test split.
    conflicts = {d for d, labels in labels_of.items() if len(labels) > 1}
    assert all(splits_of[d] == {"test"} for d in conflicts)
    flagged = manifest[manifest["label_conflict"] == 1]
    assert set(flagged["split"]) <= {"test"}


def test_excluded_file_matches_report(report: dict) -> None:
    path = PREPARED / "excluded.csv"
    assert path.is_file()
    df = pd.read_csv(path)
    assert len(df) == report["excluded_count"]
    assert report["outputs"]["excluded_sha256"] == sha256_file(path)
    assert set(df["split"]) <= {"train", "val"}  # nothing removed from official test
