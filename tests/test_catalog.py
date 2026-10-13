"""Tests for the class mapping / breed catalog single source of truth."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from doglens.catalog import (
    CatalogEntry,
    ClassInfo,
    MappingError,
    breed_key_from_source_dir,
    build_catalog,
    catalog_from_dict,
    check_catalog_matches_mapping,
    load_catalog,
    load_ru_names,
    mapping_from_dict,
    mapping_sha256,
    mapping_to_dict,
)

REPO = Path(__file__).resolve().parents[1]


def _classes(n: int = 3) -> list[ClassInfo]:
    return [
        ClassInfo(i, f"breed_{i}", f"Breed {i}", f"n{i:08d}", f"n{i:08d}-Breed_{i}", i + 1)
        for i in range(n)
    ]


def test_breed_key_from_source_dir() -> None:
    assert breed_key_from_source_dir("n02085620-Chihuahua") == (
        "n02085620",
        "chihuahua",
        "Chihuahua",
    )
    assert breed_key_from_source_dir("n02089078-black-and-tan_coonhound") == (
        "n02089078",
        "black-and-tan_coonhound",
        "Black-and-tan coonhound",
    )
    with pytest.raises(MappingError):
        breed_key_from_source_dir("Chihuahua")


def test_mapping_round_trip_and_tamper_detection() -> None:
    classes = _classes()
    raw = mapping_to_dict(classes)
    assert mapping_from_dict(raw, 3) == classes
    tampered = json.loads(json.dumps(raw))
    tampered["classes"][0]["breed_key"] = "other"
    with pytest.raises(MappingError, match="mapping_sha256"):
        mapping_from_dict(tampered, 3)
    with pytest.raises(MappingError):
        mapping_from_dict(raw, 4)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cs: [replace(cs[0], class_id=1), *cs[1:]],  # not contiguous
        lambda cs: [cs[0], replace(cs[1], breed_key="breed_0"), cs[2]],  # duplicate key
        lambda cs: [cs[0], replace(cs[1], breed_key="Bad Key"), cs[2]],  # invalid key
        lambda cs: [cs[0], replace(cs[1], source_label=5), cs[2]],  # label rule broken
    ],
)
def test_invalid_mappings_are_rejected(mutate) -> None:
    with pytest.raises(MappingError):
        mapping_to_dict(mutate(_classes()))


def test_catalog_uses_english_fallback_and_matches_mapping() -> None:
    classes = _classes()
    catalog = build_catalog(classes, {"breed_1": "Порода 1"})
    digest, entries = catalog_from_dict(catalog, 3)
    assert digest == mapping_sha256(classes)
    assert entries[0].name_ru is None and entries[0].display_name_ru == "Breed 0"
    assert entries[1].display_name_ru == "Порода 1"
    check_catalog_matches_mapping(digest, entries, classes)


def test_catalog_mismatch_is_detected() -> None:
    classes = _classes()
    digest, entries = catalog_from_dict(build_catalog(classes, {}), 3)
    other = [*classes[:2], replace(classes[2], name_en="Changed")]
    with pytest.raises(MappingError):
        check_catalog_matches_mapping(digest, entries, other)
    swapped = [entries[1], entries[0], entries[2]]
    with pytest.raises(MappingError):
        check_catalog_matches_mapping(digest, swapped, classes)
    bad = [*entries[:2], CatalogEntry(2, "breed_x", "Breed 2", None)]
    with pytest.raises(MappingError):
        check_catalog_matches_mapping(digest, bad, classes)


def test_ru_names_overlay_validation(tmp_path: Path) -> None:
    classes = _classes()
    path = tmp_path / "ru.json"
    assert load_ru_names(path, classes) == {}  # missing file -> no translations

    def write(names: dict) -> None:
        path.write_text(json.dumps({"schema_version": 1, "names": names}), encoding="utf-8")

    write({"breed_0": {"name_ru": "Порода", "source": "standard page"}})
    assert load_ru_names(path, classes) == {"breed_0": "Порода"}
    write({"unknown": {"name_ru": "X", "source": "s"}})
    with pytest.raises(MappingError):
        load_ru_names(path, classes)
    write({"breed_0": {"name_ru": "Порода"}})  # no source -> not verified
    with pytest.raises(MappingError):
        load_ru_names(path, classes)


def test_repository_ru_overlay_is_valid_against_catalog() -> None:
    digest, entries = load_catalog(REPO / "assets" / "breeds.json")
    classes = [
        ClassInfo(e.class_id, e.breed_key, e.name_en, "n00000000", e.breed_key, e.class_id + 1)
        for e in entries
    ]
    names = load_ru_names(REPO / "assets" / "breed_names_ru.json", classes)
    assert all(
        entries[[e.breed_key for e in entries].index(k)].name_ru == v for k, v in names.items()
    )


def test_repository_catalog_structure() -> None:
    digest, entries = load_catalog(REPO / "assets" / "breeds.json")
    assert len(entries) == 120
    assert [e.class_id for e in entries] == list(range(120))
    assert len({e.breed_key for e in entries}) == 120
    assert len(digest) == 64
    assert all(e.name_en for e in entries)


@pytest.mark.integration
def test_repository_catalog_matches_prepared_mapping() -> None:
    mapping_path = REPO / "data" / "prepared" / "class_mapping.json"
    if not mapping_path.is_file():
        pytest.skip("data/prepared/class_mapping.json not present (run prepare_dataset.py)")
    from doglens.catalog import load_class_mapping

    classes = load_class_mapping(mapping_path)
    digest, entries = load_catalog(REPO / "assets" / "breeds.json")
    check_catalog_matches_mapping(digest, entries, classes)
