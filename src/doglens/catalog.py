"""Class mapping (single source of truth) and the breed catalog generated from it.

``class_mapping.json`` is produced once by dataset preparation from the official Stanford Dogs
labels and later copied into the model bundle; inference never rebuilds it.
``assets/breeds.json`` is generated from the same mapping plus an optional, hand-maintained
overlay of verified Russian names (``assets/breed_names_ru.json``). Both carry the same
``mapping_sha256`` so an inconsistent pair is detected.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from doglens.config import NUM_CLASSES
from doglens.errors import ConfigError
from doglens.fileutil import canonical_json, sha256_bytes

MAPPING_SCHEMA_VERSION = 1
CATALOG_SCHEMA_VERSION = 1
RU_NAMES_SCHEMA_VERSION = 1

INDEX_RULE = (
    "class_id = official Stanford Dogs label - 1; the official labels 1..120 are read from "
    "train_list.mat/test_list.mat and each label must map to exactly one source directory"
)
BREED_KEY_RULE = "breed part of the source directory name '<wnid>-<Breed_name>', lower-cased"
NAME_RU_POLICY = (
    "name_ru is null unless a verified translation is listed in assets/breed_names_ru.json; "
    "the UI then shows name_en"
)

BREED_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
SOURCE_DIR_RE = re.compile(r"^(n\d{8})-(.+)$")


class MappingError(ConfigError):
    """Mapping or catalog is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ClassInfo:
    class_id: int
    breed_key: str
    name_en: str
    wnid: str
    source_dir: str
    source_label: int


@dataclass(frozen=True)
class CatalogEntry:
    class_id: int
    breed_key: str
    name_en: str
    name_ru: str | None

    @property
    def display_name_ru(self) -> str:
        """Russian name when verified, otherwise the English fallback."""
        return self.name_ru or self.name_en


def breed_key_from_source_dir(source_dir: str) -> tuple[str, str, str]:
    """Return (wnid, breed_key, name_en) for a directory like 'n02085620-Chihuahua'."""
    match = SOURCE_DIR_RE.match(source_dir)
    if not match:
        raise MappingError(f"unexpected class directory name: {source_dir!r}")
    wnid, raw = match.groups()
    key = raw.lower()
    if not BREED_KEY_RE.match(key):
        raise MappingError(f"cannot derive a breed_key from {source_dir!r}")
    words = raw.replace("_", " ")
    name_en = words[:1].upper() + words[1:]
    return wnid, key, name_en


def validate_classes(classes: list[ClassInfo], num_classes: int = NUM_CLASSES) -> None:
    if len(classes) != num_classes:
        raise MappingError(f"expected {num_classes} classes, got {len(classes)}")
    if [c.class_id for c in classes] != list(range(num_classes)):
        raise MappingError("class_id must be contiguous 0..N-1 in ascending order")
    for field_name in ("breed_key", "source_dir", "wnid", "source_label"):
        values = [getattr(c, field_name) for c in classes]
        if len(set(values)) != len(values):
            raise MappingError(f"duplicate {field_name} in mapping")
    for c in classes:
        if not BREED_KEY_RE.match(c.breed_key):
            raise MappingError(f"invalid breed_key {c.breed_key!r}")
        if c.source_label != c.class_id + 1:
            raise MappingError(f"class {c.class_id}: source_label must equal class_id + 1")
        if not c.name_en:
            raise MappingError(f"class {c.class_id}: empty name_en")


def mapping_sha256(classes: list[ClassInfo]) -> str:
    return sha256_bytes(canonical_json([asdict(c) for c in classes]).encode("utf-8"))


def mapping_to_dict(classes: list[ClassInfo]) -> dict[str, Any]:
    validate_classes(classes, len(classes))
    return {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "num_classes": len(classes),
        "index_rule": INDEX_RULE,
        "breed_key_rule": BREED_KEY_RULE,
        "mapping_sha256": mapping_sha256(classes),
        "classes": [asdict(c) for c in classes],
    }


def mapping_from_dict(raw: Any, num_classes: int = NUM_CLASSES) -> list[ClassInfo]:
    if not isinstance(raw, dict) or raw.get("schema_version") != MAPPING_SCHEMA_VERSION:
        raise MappingError("unsupported class mapping schema")
    try:
        classes = [ClassInfo(**item) for item in raw["classes"]]
    except (KeyError, TypeError) as exc:
        raise MappingError(f"malformed class mapping: {exc}") from exc
    validate_classes(classes, num_classes)
    if raw.get("num_classes") != num_classes:
        raise MappingError("num_classes does not match the class list")
    if raw.get("mapping_sha256") != mapping_sha256(classes):
        raise MappingError("mapping_sha256 does not match the class list")
    return classes


def load_class_mapping(path: str | Path, num_classes: int = NUM_CLASSES) -> list[ClassInfo]:
    path = Path(path)
    if not path.is_file():
        raise MappingError(f"class mapping not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MappingError(f"invalid JSON in {path}: {exc}") from exc
    return mapping_from_dict(raw, num_classes)


def load_ru_names(path: str | Path, classes: list[ClassInfo]) -> dict[str, str]:
    """Verified Russian names keyed by breed_key. A missing file means no translations."""
    path = Path(path)
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != RU_NAMES_SCHEMA_VERSION:
        raise MappingError(f"unsupported schema in {path}")
    names = raw.get("names", {})
    if not isinstance(names, dict):
        raise MappingError(f"'names' must be a mapping in {path}")
    known = {c.breed_key for c in classes}
    result: dict[str, str] = {}
    for key, entry in names.items():
        if key not in known:
            raise MappingError(f"unknown breed_key {key!r} in {path}")
        if not isinstance(entry, dict) or not str(entry.get("name_ru", "")).strip():
            raise MappingError(f"entry {key!r} in {path} needs a non-empty 'name_ru'")
        if not str(entry.get("source", "")).strip():
            raise MappingError(f"entry {key!r} in {path} needs a 'source' for verification")
        result[key] = str(entry["name_ru"]).strip()
    return result


def build_catalog(classes: list[ClassInfo], ru_names: dict[str, str]) -> dict[str, Any]:
    validate_classes(classes, len(classes))
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "generated_from": "class_mapping.json (dataset preparation)",
        "mapping_sha256": mapping_sha256(classes),
        "name_ru_policy": NAME_RU_POLICY,
        "breeds": [
            {
                "class_id": c.class_id,
                "breed_key": c.breed_key,
                "name_en": c.name_en,
                "name_ru": ru_names.get(c.breed_key),
            }
            for c in classes
        ],
    }


def catalog_from_dict(raw: Any, num_classes: int = NUM_CLASSES) -> tuple[str, list[CatalogEntry]]:
    """Return (mapping_sha256, entries) after structural validation."""
    if not isinstance(raw, dict) or raw.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise MappingError("unsupported catalog schema")
    try:
        entries = [CatalogEntry(**item) for item in raw["breeds"]]
        digest = str(raw["mapping_sha256"])
    except (KeyError, TypeError) as exc:
        raise MappingError(f"malformed catalog: {exc}") from exc
    if [e.class_id for e in entries] != list(range(num_classes)):
        raise MappingError("catalog class_id must be contiguous 0..N-1")
    keys = [e.breed_key for e in entries]
    if len(set(keys)) != len(keys):
        raise MappingError("duplicate breed_key in catalog")
    return digest, entries


def load_catalog(
    path: str | Path, num_classes: int = NUM_CLASSES
) -> tuple[str, list[CatalogEntry]]:
    path = Path(path)
    if not path.is_file():
        raise MappingError(f"breed catalog not found: {path}")
    return catalog_from_dict(json.loads(path.read_text(encoding="utf-8")), num_classes)


def check_catalog_matches_mapping(
    catalog_digest: str, entries: list[CatalogEntry], classes: list[ClassInfo]
) -> None:
    """Raise MappingError unless the catalog was generated from exactly this mapping."""
    if catalog_digest != mapping_sha256(classes):
        raise MappingError("catalog mapping_sha256 differs from the class mapping")
    if len(entries) != len(classes):
        raise MappingError("catalog and mapping have different sizes")
    for e, c in zip(entries, classes, strict=True):
        if (e.class_id, e.breed_key, e.name_en) != (c.class_id, c.breed_key, c.name_en):
            raise MappingError(f"catalog entry {e.class_id} does not match the mapping")
