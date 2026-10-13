"""Tiny synthetic Stanford-Dogs-like tree for tests. Test-only; no real photos involved."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

# Official labels deliberately do NOT follow alphabetical directory order.
FIXTURE_CLASSES = [
    # (official label, source dir)
    (1, "n00000003-Gamma_terrier"),
    (2, "n00000001-alpha_dog"),
    (3, "n00000002-Beta-hound"),
]


def xml_text(
    image_id: str,
    size: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    name: str,
    filename: str | None = None,
) -> str:
    objects = "".join(
        f"<object><name>{name}</name><pose>Unspecified</pose><truncated>0</truncated>"
        f"<difficult>0</difficult><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>"
        f"<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>"
        for b in boxes
    )
    return (
        f"<annotation><folder>x</folder><filename>{filename or image_id}</filename>"
        f"<source><database>ImageNet database</database></source>"
        f"<size><width>{size[0]}</width><height>{size[1]}</height><depth>3</depth></size>"
        f"<segment>0</segment>{objects}</annotation>"
    )


def cell_array(strings: list[str]) -> np.ndarray:
    arr = np.empty((len(strings), 1), dtype=object)
    for i, s in enumerate(strings):
        arr[i, 0] = np.array([s])
    return arr


def write_split_mat(path: Path, rel_paths: list[str], labels: list[int]) -> None:
    scipy.io.savemat(
        str(path),
        {
            "file_list": cell_array(rel_paths),
            "annotation_list": cell_array([p.rsplit(".", 1)[0] for p in rel_paths]),
            "labels": np.array(labels, dtype=np.uint8).reshape(-1, 1),
        },
    )


@dataclass
class Fixture:
    root: Path
    images: Path
    annotations: Path
    splits: Path
    train: list[tuple[str, int]] = field(default_factory=list)
    test: list[tuple[str, int]] = field(default_factory=list)

    def image_path(self, rel: str) -> Path:
        return self.images / rel

    def annotation_path(self, rel: str) -> Path:
        return self.annotations / rel.rsplit(".", 1)[0]


def make_fixture(root: Path, n_train: int = 10, n_test: int = 4, size=(120, 90)) -> Fixture:
    fx = Fixture(root, root / "Images", root / "Annotation", root / "splits")
    fx.splits.mkdir(parents=True)
    counter = 0
    for label, source_dir in FIXTURE_CLASSES:
        (fx.images / source_dir).mkdir(parents=True)
        (fx.annotations / source_dir).mkdir(parents=True)
        wnid, name = source_dir.split("-", 1)
        for i in range(n_train + n_test):
            counter += 1
            image_id = f"{wnid}_{i}"
            rel = f"{source_dir}/{image_id}.jpg"
            # Unique content per image so that no accidental exact duplicates exist.
            img = Image.new("RGB", size, (counter % 256, (counter * 7) % 256, (counter * 13) % 256))
            img.putpixel((0, 0), (counter // 256, 0, 0))
            img.save(fx.image_path(rel), "JPEG", quality=95)
            fx.annotation_path(rel).write_text(
                xml_text(image_id, size, [(10, 5, 89, 69)], name), encoding="utf-8"
            )
            (fx.train if i < n_train else fx.test).append((rel, label))
    write_mats(fx)
    return fx


def write_mats(fx: Fixture) -> None:
    write_split_mat(
        fx.splits / "train_list.mat", [r for r, _ in fx.train], [lab for _, lab in fx.train]
    )
    write_split_mat(
        fx.splits / "test_list.mat", [r for r, _ in fx.test], [lab for _, lab in fx.test]
    )
    everything = fx.train + fx.test
    write_split_mat(
        fx.splits / "file_list.mat", [r for r, _ in everything], [lab for _, lab in everything]
    )
