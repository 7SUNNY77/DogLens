"""Unit tests for Stanford Dogs preparation on small synthetic fixtures (no real dataset)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
from _stanford_fixture import FIXTURE_CLASSES, make_fixture, write_mats, write_split_mat, xml_text
from PIL import Image

from doglens.catalog import load_catalog, load_class_mapping
from doglens.data_prep import (
    AnnotationError,
    DatasetError,
    RawBox,
    assign_validation,
    build_class_mapping,
    context_box,
    decode_dataset_image,
    excluded_csv_text,
    manifest_csv_text,
    mat_cell_to_str,
    normalize_box,
    parse_annotation,
    prepare,
    read_official_splits,
    read_split_mat,
    safe_relative_path,
    select_largest_box,
    xml_box_to_internal,
)
from doglens.schemas import EXCLUDED_COLUMNS, MANIFEST_COLUMNS

REPO = Path(__file__).resolve().parents[1]
N3 = len(FIXTURE_CLASSES)


def _prepare(fx, **kw):
    params = {"val_fraction": 0.2, "seed": 42, "bbox_context": 0.1, "num_classes": N3, "workers": 2}
    params.update(kw)
    return prepare(fx.images, fx.annotations, fx.splits, **params)


# --- MAT lists -------------------------------------------------------------------------------


def test_mat_cell_to_str_unwraps_scipy_nesting() -> None:
    assert mat_cell_to_str(np.array([np.array(["a/b.jpg"])], dtype=object)) == "a/b.jpg"
    assert mat_cell_to_str(np.array([[np.str_("x")]])) == "x"
    assert mat_cell_to_str(b"bytes") == "bytes"
    with pytest.raises(DatasetError):
        mat_cell_to_str(np.array(["a", "b"]))
    with pytest.raises(DatasetError):
        mat_cell_to_str(np.array([""]))
    with pytest.raises(DatasetError):
        mat_cell_to_str(np.array([5]))


def test_read_split_mat_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "train_list.mat"
    write_split_mat(path, ["d-x/a_1.jpg", "d-y/b_2.jpg"], [1, 120])
    entries = read_split_mat(path)
    assert [(e.relative_path, e.annotation_path, e.label) for e in entries] == [
        ("d-x/a_1.jpg", "d-x/a_1", 1),
        ("d-y/b_2.jpg", "d-y/b_2", 120),
    ]
    assert all(isinstance(e.label, int) for e in entries)


def test_read_split_mat_accepts_integral_float_labels(tmp_path: Path) -> None:
    path = tmp_path / "l.mat"
    scipy.io.savemat(
        str(path),
        {
            "file_list": np.array([[np.array(["d/a.jpg"])]], dtype=object),
            "annotation_list": np.array([[np.array(["d/a"])]], dtype=object),
            "labels": np.array([[3.0]]),
        },
    )
    assert read_split_mat(path)[0].label == 3


@pytest.mark.parametrize(
    "variables",
    [
        {"file_list": ["a"], "annotation_list": ["a"]},  # missing labels
        {"file_list": ["a", "b"], "annotation_list": ["a"], "labels": [1]},  # length mismatch
        {"file_list": ["a"], "annotation_list": ["a"], "labels": [0]},  # non-positive
        {"file_list": ["a"], "annotation_list": ["a"], "labels": [1.5]},  # non-integer
    ],
)
def test_read_split_mat_rejects_bad_content(tmp_path: Path, variables: dict) -> None:
    path = tmp_path / "bad.mat"
    data = {}
    for key, values in variables.items():
        if key == "labels":
            data[key] = np.array(values, dtype=float).reshape(-1, 1)
        else:
            arr = np.empty((len(values), 1), dtype=object)
            for i, v in enumerate(values):
                arr[i, 0] = np.array([v])
            data[key] = arr
    scipy.io.savemat(str(path), data)
    with pytest.raises(DatasetError):
        read_split_mat(path)


def test_read_split_mat_missing_and_corrupt(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        read_split_mat(tmp_path / "absent.mat")
    bad = tmp_path / "corrupt.mat"
    bad.write_bytes(b"not a mat file at all")
    with pytest.raises(DatasetError):
        read_split_mat(bad)


@pytest.mark.parametrize("raw", ["../x/a.jpg", "/abs/a.jpg", "C:/x/a.jpg", "a/../../b.jpg", ""])
def test_unsafe_relative_paths_are_refused(raw: str) -> None:
    with pytest.raises(DatasetError):
        safe_relative_path(raw)


def test_backslashes_are_normalized() -> None:
    assert safe_relative_path("d-x\\a.jpg") == "d-x/a.jpg"


def test_missing_official_splits_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(DatasetError) as info:
        read_official_splits(tmp_path)
    message = str(info.value)
    assert "train_list.mat" in message and "lists.tar" in message
    assert "never substituted" in message


# --- XML and coordinates -----------------------------------------------------------------------

REAL_FORMAT_XML = b"""<annotation>
\t<folder>02085620</folder>
\t<filename>n02085620_10074</filename>
\t<source>
\t\t<database>ImageNet database</database>
\t</source>
\t<size>
\t\t<width>333</width>
\t\t<height>500</height>
\t\t<depth>3</depth>
\t</size>
\t<segment>0</segment>
\t<object>
\t\t<name>Chihuahua</name>
\t\t<pose>Unspecified</pose>
\t\t<truncated>0</truncated>
\t\t<difficult>0</difficult>
\t\t<bndbox>
\t\t\t<xmin>25</xmin>
\t\t\t<ymin>10</ymin>
\t\t\t<xmax>276</xmax>
\t\t\t<ymax>498</ymax>
\t\t</bndbox>
\t</object>
</annotation>
"""


def test_parse_real_format_annotation() -> None:
    ann = parse_annotation(REAL_FORMAT_XML)
    assert (ann.filename, ann.width, ann.height) == ("n02085620_10074", 333, 500)
    assert ann.objects == (RawBox("Chihuahua", 25, 10, 276, 498),)
    assert ann.invalid_objects == 0


def test_parse_multiple_objects_and_bad_object() -> None:
    xml = xml_text("i", (100, 80), [(1, 2, 30, 40), (10, 10, 99, 79)], "dog").replace(
        "</annotation>",
        "<object><name>dog</name><bndbox><xmin>a</xmin></bndbox></object></annotation>",
    )
    ann = parse_annotation(xml.encode())
    assert len(ann.objects) == 2 and ann.invalid_objects == 1


@pytest.mark.parametrize(
    "data",
    [
        b"<annotation><size><width>10</width></size></annotation>",  # no height
        b"<annotation><size><width>x</width><height>1</height></size></annotation>",
        b"<annotation><size><width>0</width><height>5</height></size></annotation>",
        b"<annotation><size><width>10</width><height>10</height></size></annotation>",  # no objects
        b"<annotation><size>",  # broken XML
        b"<other/>",
        b'<!DOCTYPE a [<!ENTITY x "y">]><annotation/>',
    ],
)
def test_malformed_annotations_raise(data: bytes) -> None:
    with pytest.raises(AnnotationError):
        parse_annotation(data)


def test_xml_box_conversion_inclusive_to_exclusive() -> None:
    assert xml_box_to_internal(25, 10, 276, 498) == (25, 10, 277, 499)


def test_full_image_box_maps_to_full_extent() -> None:
    # Observed on the real data: 0..W-1 boxes exist, xmax == W never occurs.
    norm = normalize_box(RawBox("d", 0, 0, 332, 499), (333, 500))
    assert norm.box == (0, 0, 333, 500) and norm.issue is None


def test_single_pixel_box_is_valid() -> None:
    assert normalize_box(RawBox("d", 5, 5, 5, 5), (10, 10)).box == (5, 5, 6, 6)


def test_out_of_bounds_box_is_clipped_and_reported() -> None:
    norm = normalize_box(RawBox("d", 10, 10, 286, 380), (284, 360))
    assert norm.box == (10, 10, 284, 360)
    assert norm.issue == "bbox_clipped"
    assert norm.unclipped == (10, 10, 287, 381)
    neg = normalize_box(RawBox("d", -3, 0, 50, 50), (100, 100))
    assert neg.box == (0, 0, 51, 51) and neg.issue == "bbox_clipped"


def test_degenerate_and_outside_boxes_are_unusable() -> None:
    assert normalize_box(RawBox("d", 50, 10, 40, 60), (100, 100)).issue == "bbox_degenerate"
    outside = normalize_box(RawBox("d", 200, 10, 250, 60), (100, 100))
    assert outside.box is None and outside.issue == "bbox_outside_image"


def test_select_largest_box_and_tie_rule() -> None:
    assert select_largest_box([(0, 0, 10, 10), (0, 0, 20, 20), (5, 5, 15, 15)]) == 1
    # Equal areas: the earliest object in XML order wins.
    assert select_largest_box([(0, 0, 10, 20), (50, 50, 70, 60), (0, 0, 20, 10)]) == 0
    with pytest.raises(ValueError):
        select_largest_box([])


def test_context_box_adds_10_percent_per_axis_and_clips() -> None:
    # 100 wide, 50 tall: +10 px left/right, +5 px top/bottom.
    assert context_box((100, 100, 200, 150), (400, 300), 0.1) == (90, 95, 210, 155)
    assert context_box((5, 2, 105, 52), (110, 54), 0.1) == (0, 0, 110, 54)


def test_decode_dataset_image_has_no_upload_limits() -> None:
    import io

    buf = io.BytesIO()
    Image.new("RGBA", (30, 20), (0, 0, 0, 0)).save(buf, "PNG")
    img = decode_dataset_image(buf.getvalue())  # 30x20 would be below the ROI minimum
    assert img.mode == "RGB" and img.size == (30, 20)
    assert img.getpixel((1, 1)) == (255, 255, 255)
    with pytest.raises(Exception):  # noqa: B017 - any decoder error is acceptable
        decode_dataset_image(b"\xff\xd8\xff" + b"0" * 50)


# --- mapping -----------------------------------------------------------------------------------


def test_mapping_follows_official_labels_not_directory_order(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    splits = read_official_splits(fx.splits)
    dirs = sorted(d for _, d in FIXTURE_CLASSES)
    classes = build_class_mapping(splits, dirs, dirs, num_classes=N3)
    assert [(c.class_id, c.source_dir) for c in classes] == [
        (0, "n00000003-Gamma_terrier"),
        (1, "n00000001-alpha_dog"),
        (2, "n00000002-Beta-hound"),
    ]
    assert [c.breed_key for c in classes] == ["gamma_terrier", "alpha_dog", "beta-hound"]
    assert [c.name_en for c in classes] == ["Gamma terrier", "Alpha dog", "Beta-hound"]
    assert [c.source_label for c in classes] == [1, 2, 3]


def test_mapping_is_stable_across_calls(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    dirs = sorted(d for _, d in FIXTURE_CLASSES)
    a = build_class_mapping(read_official_splits(fx.splits), dirs, dirs, num_classes=N3)
    b = build_class_mapping(read_official_splits(fx.splits), list(reversed(dirs)), dirs, N3)
    assert a == b


def test_mapping_rejects_directory_mismatch(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    splits = read_official_splits(fx.splits)
    dirs = sorted(d for _, d in FIXTURE_CLASSES)
    with pytest.raises(DatasetError, match="images class directories"):
        build_class_mapping(splits, [*dirs, "n09999999-extra"], dirs, num_classes=N3)
    with pytest.raises(DatasetError, match="annotations class directories"):
        build_class_mapping(splits, dirs, dirs[:2], num_classes=N3)


def test_mapping_rejects_label_in_two_directories(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    fx.train[0] = (fx.train[0][0], 2)  # a Gamma image now claims label 2
    write_mats(fx)
    dirs = sorted(d for _, d in FIXTURE_CLASSES)
    with pytest.raises(DatasetError, match="mismatch"):
        build_class_mapping(read_official_splits(fx.splits), dirs, dirs, num_classes=N3)


def test_mapping_rejects_missing_label(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    dirs = sorted(d for _, d in FIXTURE_CLASSES)
    with pytest.raises(DatasetError, match="exactly 1..4"):
        build_class_mapping(read_official_splits(fx.splits), dirs, dirs, num_classes=4)


# --- validation split --------------------------------------------------------------------------


def _ids(n_per_class: int, n_classes: int) -> tuple[list[str], list[int]]:
    ids = [f"c{c}_{i:03d}" for c in range(n_classes) for i in range(n_per_class)]
    return ids, [int(i[1]) for i in ids]


def test_validation_split_is_stratified_and_reproducible() -> None:
    ids, labels = _ids(100, 5)
    val = assign_validation(ids, labels, 0.2, 42)
    assert len(val) == 100
    assert all(sum(1 for i in val if i.startswith(f"c{c}_")) == 20 for c in range(5))
    assert assign_validation(ids, labels, 0.2, 42) == val
    assert assign_validation(ids, labels, 0.2, 7) != val


def test_validation_split_does_not_depend_on_input_order() -> None:
    ids, labels = _ids(30, 3)
    pairs = list(zip(ids, labels, strict=True))[::-1]
    val_a = assign_validation(ids, labels, 0.2, 42)
    val_b = assign_validation([p[0] for p in pairs], [p[1] for p in pairs], 0.2, 42)
    assert val_a == val_b


def test_validation_split_keeps_duplicate_groups_together() -> None:
    ids, labels = _ids(50, 2)
    groups = list(ids)
    groups[ids.index("c0_001")] = groups[ids.index("c0_000")]  # c0_000 == c0_001 (same bytes)
    groups[ids.index("c1_049")] = groups[ids.index("c0_002")]  # cross-class duplicate
    for seed in range(20):
        val = assign_validation(ids, labels, 0.2, seed, groups=groups)
        assert ("c0_000" in val) == ("c0_001" in val)
        assert ("c0_002" in val) == ("c1_049" in val)


def test_validation_split_errors() -> None:
    with pytest.raises(DatasetError):
        assign_validation([], [], 0.2, 42)
    with pytest.raises(DatasetError):
        assign_validation(["a", "a"], [0, 0], 0.5, 42)
    with pytest.raises(DatasetError):  # a class with a single member cannot be stratified
        assign_validation(["a", "b", "c"], [0, 0, 1], 0.5, 42)


# --- full preparation on fixtures ---------------------------------------------------------------


def test_prepare_counts_paths_and_no_overlap(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path, n_train=10, n_test=4)
    result = _prepare(fx)
    assert result.report["split_counts"] == {"train": 24, "val": 6, "test": 12}
    assert result.report["excluded"] == []
    assert result.report["class_count_range"]["val"] == [2, 2]
    ids_by_split: dict[str, set[str]] = {}
    for r in result.rows:
        ids_by_split.setdefault(r["split"], set()).add(r["image_id"])
        assert (fx.images / r["relative_path"]).is_file()  # relative to --images-dir
        assert "\\" not in r["relative_path"]
        assert Path(r["relative_path"]).stem == r["image_id"]
        # XML (10,5,89,69) inclusive -> exclusive internal box.
        assert (r["bbox_xmin"], r["bbox_ymin"], r["bbox_xmax"], r["bbox_ymax"]) == (10, 5, 90, 70)
    assert not ids_by_split["train"] & ids_by_split["val"]
    assert not ids_by_split["train"] & ids_by_split["test"]
    assert not ids_by_split["val"] & ids_by_split["test"]
    official_test = {Path(r).stem for r, _ in fx.test}
    assert ids_by_split["test"] == official_test  # official test untouched


def test_prepare_is_deterministic(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    a, b = _prepare(fx), _prepare(fx, workers=1)
    assert manifest_csv_text(a.rows) == manifest_csv_text(b.rows)
    assert a.classes == b.classes


def test_problem_records_are_reported_not_silently_dropped(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    rel_missing_ann, rel_corrupt, rel_size, rel_clip, rel_multi, rel_tiny, rel_rot, rel_bad = (
        fx.train[i][0] for i in (0, 1, 2, 3, 4, 5, 6, 7)
    )
    fx.annotation_path(rel_missing_ann).unlink()
    fx.image_path(rel_corrupt).write_bytes(b"\xff\xd8\xff" + b"0" * 100)
    stem = lambda rel: Path(rel).stem  # noqa: E731
    name = lambda rel: rel.split("/")[0].split("-", 1)[1]  # noqa: E731
    fx.annotation_path(rel_size).write_text(
        xml_text(stem(rel_size), (200, 90), [(1, 1, 50, 50)], name(rel_size)), encoding="utf-8"
    )
    fx.annotation_path(rel_clip).write_text(
        xml_text(stem(rel_clip), (120, 90), [(10, 10, 130, 95)], name(rel_clip)), encoding="utf-8"
    )
    fx.annotation_path(rel_multi).write_text(
        xml_text(stem(rel_multi), (120, 90), [(0, 0, 9, 9), (20, 20, 79, 59)], name(rel_multi)),
        encoding="utf-8",
    )
    fx.annotation_path(rel_tiny).write_text(
        xml_text(stem(rel_tiny), (120, 90), [(3, 4, 12, 13)], name(rel_tiny)), encoding="utf-8"
    )
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (120, 90), (1, 2, 3)).save(fx.image_path(rel_rot), "JPEG", exif=exif.tobytes())
    fx.annotation_path(rel_bad).write_text("<annotation><size>", encoding="utf-8")

    result = _prepare(fx)
    reasons = {e["image_id"]: e["reason"] for e in result.report["excluded"]}
    assert reasons == {
        stem(rel_missing_ann): "annotation_missing",
        stem(rel_corrupt): "image_unreadable",
        stem(rel_size): "xml_size_mismatch",
        stem(rel_rot): "exif_rotation_unsupported",
        stem(rel_bad): "annotation_invalid",
    }
    assert result.report["excluded_count"] == 5
    rows = {r["image_id"]: r for r in result.rows}
    clip_row = rows[stem(rel_clip)]
    assert (clip_row["bbox_xmax"], clip_row["bbox_ymax"]) == (120, 90)
    assert any(
        i["code"] == "bbox_clipped" and i["image_id"] == stem(rel_clip)
        for i in result.report["issues"]
    )
    multi = rows[stem(rel_multi)]
    assert (multi["bbox_xmin"], multi["bbox_xmax"]) == (20, 80)  # largest box chosen
    tiny = rows[stem(rel_tiny)]  # 10x10 box: below the upload ROI minimum, still kept
    assert (tiny["bbox_xmin"], tiny["bbox_ymin"], tiny["bbox_xmax"], tiny["bbox_ymax"]) == (
        3,
        4,
        13,
        14,
    )
    assert result.report["stats"]["bbox_min_side_lt_64"] >= 1
    assert sum(result.report["split_counts"].values()) == 42 - 5


def test_missing_image_file_is_reported(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    fx.image_path(fx.test[0][0]).unlink()
    result = _prepare(fx)
    assert result.report["excluded"] == [
        {
            "image_id": Path(fx.test[0][0]).stem,
            "split": "test",
            "reason": "image_missing",
            "relative_path": fx.test[0][0],
        }
    ]


def test_duplicates_are_reported_and_kept_within_train_val(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    shutil.copyfile(fx.image_path(fx.train[0][0]), fx.image_path(fx.train[1][0]))
    shutil.copyfile(fx.image_path(fx.train[2][0]), fx.image_path(fx.test[0][0]))
    result = _prepare(fx)
    dup = result.report["duplicates"]
    assert dup["group_count"] == 2 and dup["cross_split_groups"] >= 1
    test_id, leaked_id = Path(fx.test[0][0]).stem, Path(fx.train[2][0]).stem
    assert dup["test_ids_with_train_or_val_duplicate"] == [test_id]
    rows = {r["image_id"]: r for r in result.rows}
    # Same-class duplicates inside official train stay, on one side of train/val.
    assert rows[Path(fx.train[0][0]).stem]["split"] == rows[Path(fx.train[1][0]).stem]["split"]
    # The train copy of a test image is removed; the test image stays and is flagged.
    assert leaked_id not in rows
    assert rows[test_id]["split"] == "test" and rows[test_id]["dup_with_test"] == 1
    assert rows[test_id]["label_conflict"] == 0
    (excl,) = [e for e in result.report["excluded"] if e["image_id"] == leaked_id]
    assert excl["reason"] == "dup_with_test" and excl["duplicate_of"] == [test_id]
    assert excl["split"] in ("train", "val")
    assert result.report["leakage"]["removed_total"] == 1
    assert result.report["official_split"]["checks"]["content_overlap_between_splits"] == 0


def test_label_conflicts_are_removed_from_train_val_and_flagged_in_test(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    by_label: dict[int, list[str]] = {}
    for rel, label in fx.train:
        by_label.setdefault(label, []).append(rel)
    test_by_label: dict[int, list[str]] = {}
    for rel, label in fx.test:
        test_by_label.setdefault(label, []).append(rel)
    # Conflict fully inside official train: both copies must go.
    a, b = by_label[1][0], by_label[2][0]
    shutil.copyfile(fx.image_path(a), fx.image_path(b))
    # Conflict between train and test: train copy goes, test stays flagged.
    c, t = by_label[3][0], test_by_label[1][0]
    shutil.copyfile(fx.image_path(t), fx.image_path(c))
    result = _prepare(fx)
    rows = {r["image_id"]: r for r in result.rows}
    reasons = {e["image_id"]: e["reason"] for e in result.report["excluded"]}
    stem = lambda rel: Path(rel).stem  # noqa: E731
    assert reasons == {
        stem(a): "label_conflict",
        stem(b): "label_conflict",
        stem(c): "dup_with_test+label_conflict",
    }
    assert stem(a) not in rows and stem(b) not in rows and stem(c) not in rows
    assert rows[stem(t)]["dup_with_test"] == 1 and rows[stem(t)]["label_conflict"] == 1
    leak = result.report["leakage"]
    assert leak["removed_by_reason"] == {"dup_with_test+label_conflict": 1, "label_conflict": 2}
    assert leak["flagged_label_conflict_kept"] == [stem(t)]
    assert leak["test_flagged_dup_with_test"] == [stem(t)]
    # Official test is never reduced.
    assert sum(r["split"] == "test" for r in result.rows) == len(fx.test)


def test_n_boxes_column(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    rel = fx.train[0][0]
    name = rel.split("/")[0].split("-", 1)[1]
    fx.annotation_path(rel).write_text(
        xml_text(Path(rel).stem, (120, 90), [(0, 0, 9, 9), (20, 20, 79, 59), (1, 1, 5, 5)], name),
        encoding="utf-8",
    )
    rows = {r["image_id"]: r for r in _prepare(fx).rows}
    assert rows[Path(rel).stem]["n_boxes"] == 3
    assert rows[Path(fx.train[1][0]).stem]["n_boxes"] == 1


def test_excluded_csv_lists_every_exclusion(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    shutil.copyfile(fx.image_path(fx.test[0][0]), fx.image_path(fx.train[0][0]))
    fx.annotation_path(fx.train[1][0]).unlink()
    report = _prepare(fx).report
    text = excluded_csv_text(report["excluded"])
    df = pd.read_csv(pd.io.common.StringIO(text))
    assert list(df.columns) == list(EXCLUDED_COLUMNS)
    assert set(df["reason"]) == {"dup_with_test", "annotation_missing"}
    assert len(df) == report["excluded_count"] == 2


def test_train_test_overlap_is_fatal(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    fx.test.append(fx.train[0])
    write_mats(fx)
    with pytest.raises(DatasetError):
        _prepare(fx)


def test_unlisted_files_are_reported(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    extra = fx.images / FIXTURE_CLASSES[0][1] / "n00000003_999.jpg"
    Image.new("RGB", (120, 90)).save(extra, "JPEG")
    result = _prepare(fx)
    assert any(i["code"] == "unlisted_image" for i in result.report["issues"])


def test_manifest_csv_format(tmp_path: Path) -> None:
    fx = make_fixture(tmp_path)
    text = manifest_csv_text(_prepare(fx).rows)
    assert text.splitlines()[0] == ",".join(MANIFEST_COLUMNS)
    assert "\r" not in text and text.endswith("\n")
    df = pd.read_csv(pd.io.common.StringIO(text))
    assert list(df.columns) == list(MANIFEST_COLUMNS)
    assert df["image_id"].is_unique
    assert set(df["split"]) == {"train", "val", "test"}
    assert (df["bbox_xmax"] > df["bbox_xmin"]).all() and (df["bbox_ymax"] > df["bbox_ymin"]).all()


# --- CLI ---------------------------------------------------------------------------------------


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "prepare_cli", REPO / "scripts" / "prepare_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cli_args(fx, out: Path, *extra: str) -> list[str]:
    return [
        "--images-dir", str(fx.images), "--annotations-dir", str(fx.annotations),
        "--splits-dir", str(fx.splits), "--output-dir", str(out),
        "--catalog-path", str(out / "breeds.json"), "--ru-names", str(out / "absent_ru.json"),
        "--config", str(out / "cfg.yaml"), "--workers", "2", *extra,
    ]  # fmt: skip


def test_cli_end_to_end_and_rerun_stability(tmp_path: Path, capsys) -> None:
    fx = make_fixture(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()
    (out / "cfg.yaml").write_text("model: {num_classes: 120}\n", encoding="utf-8")
    cli = _load_cli()
    assert cli.main(_cli_args(fx, out, "--visual-sample", "4"), num_classes=N3) == 0
    for name in ("manifest.csv", "class_mapping.json", "preparation_report.json", "breeds.json"):
        assert (out / name).is_file()
    assert not list(out.glob("*.tmp")) and not list(out.glob(".*.tmp"))
    report = json.loads((out / "preparation_report.json").read_text(encoding="utf-8"))
    assert report["inputs"]["relative_path_root"] == fx.images.as_posix()
    assert set(report["inputs"]["split_files_sha256"]) == {
        "train_list.mat",
        "test_list.mat",
        "file_list.mat",
    }
    assert report["visual_check"]["panels"] == 4
    assert len(list((out / "visual_check").glob("*.jpg"))) == 4
    classes = load_class_mapping(out / "class_mapping.json", num_classes=N3)
    digest, entries = load_catalog(out / "breeds.json", num_classes=N3)
    assert digest == report["outputs"]["mapping_sha256"]
    assert [e.breed_key for e in entries] == [c.breed_key for c in classes]

    first = (out / "manifest.csv").read_bytes()
    assert cli.main(_cli_args(fx, out, "--visual-sample", "0"), num_classes=N3) == 0
    assert (out / "manifest.csv").read_bytes() == first
    assert "manifest=unchanged" in capsys.readouterr().out

    # A changed split must not overwrite the existing manifest silently.
    (out / "manifest.csv").write_text(
        first.decode().replace(",val,", ",train,", 1), encoding="utf-8"
    )
    tampered = (out / "manifest.csv").read_bytes()
    assert cli.main(_cli_args(fx, out, "--visual-sample", "0"), num_classes=N3) == 1
    assert (out / "manifest.csv").read_bytes() == tampered
    assert "differs" in capsys.readouterr().err
    assert (
        cli.main(_cli_args(fx, out, "--visual-sample", "0", "--allow-changes"), num_classes=N3) == 0
    )
    assert (out / "manifest.csv").read_bytes() == first


def test_cli_missing_splits_and_bad_args(tmp_path: Path, capsys) -> None:
    fx = make_fixture(tmp_path / "src")
    for f in fx.splits.iterdir():
        f.unlink()
    out = tmp_path / "out"
    out.mkdir()
    (out / "cfg.yaml").write_text("{}\n", encoding="utf-8")
    cli = _load_cli()
    assert cli.main(_cli_args(fx, out, "--visual-sample", "0"), num_classes=N3) == 1
    assert "lists.tar" in capsys.readouterr().err
    assert not (out / "manifest.csv").exists()
    with pytest.raises(SystemExit) as info:
        cli.main(["--images-dir", str(tmp_path / "nope")])
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
