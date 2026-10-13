"""Manifest Dataset and deterministic DataLoader on synthetic fixture images."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from doglens.data_prep import DatasetError, context_box
from doglens.dataset import (
    HorizontalFlip,
    ManifestDataset,
    load_manifest,
    make_loader,
    seed_everything,
)
from doglens.preprocessing import preprocess, spec_from_weights
from doglens.schemas import MANIFEST_COLUMNS

SPEC = spec_from_weights()


def _write_images(root: Path, n: int = 12) -> pd.DataFrame:
    rows = []
    (root / "c-x").mkdir(parents=True)
    for i in range(n):
        image_id = f"img_{i:02d}"
        rel = f"c-x/{image_id}.jpg"
        img = Image.new("RGB", (160, 120), (i * 20 % 256, 50, 100))
        img.paste((255, 255, 0), (40, 30, 100, 90))
        img.save(root / rel, "JPEG", quality=95)
        rows.append(
            {
                "image_id": image_id,
                "relative_path": rel,
                "class_id": i % 3,
                "breed_key": f"b{i % 3}",
                "split": "train",
                "bbox_xmin": 40,
                "bbox_ymin": 30,
                "bbox_xmax": 100,
                "bbox_ymax": 90,
                "dup_with_test": 0,
                "label_conflict": 0,
                "n_boxes": 1,
            }
        )
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))


def test_item_is_context_crop_through_shared_preprocessing(tmp_path: Path) -> None:
    df = _write_images(tmp_path)
    ds = ManifestDataset(df, tmp_path, SPEC, context=0.1)
    tensor, label = ds[4]
    assert tensor.shape == (3, 224, 224) and label == 1
    image = Image.open(tmp_path / df.loc[4, "relative_path"]).convert("RGB")
    ctx = context_box((40, 30, 100, 90), image.size, 0.1)
    assert ctx == (34, 24, 106, 96)
    assert ds.crop(4).size == (72, 72)
    assert torch.equal(tensor, preprocess(image.crop(ctx), SPEC))


def test_context_is_clipped_to_image(tmp_path: Path) -> None:
    df = _write_images(tmp_path, n=1)
    df.loc[0, ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]] = [0, 0, 160, 120]
    assert ManifestDataset(df, tmp_path, SPEC, context=0.1).crop(0).size == (160, 120)


def test_transparent_png_on_white(tmp_path: Path) -> None:
    df = _write_images(tmp_path, n=1)
    png = Image.new("RGBA", (160, 120), (0, 0, 0, 0))
    rel = "c-x/alpha.jpg"  # extension deliberately wrong, like n02105855_2933
    png.save(tmp_path / rel, "PNG")
    df.loc[0, "relative_path"] = rel
    crop = ManifestDataset(df, tmp_path, SPEC, context=0.0).crop(0)
    assert crop.mode == "RGB" and crop.getpixel((5, 5)) == (255, 255, 255)


def test_loader_order_is_reproducible(tmp_path: Path) -> None:
    df = _write_images(tmp_path)
    df["class_id"] = range(len(df))  # labels identify items
    ds = ManifestDataset(df, tmp_path, SPEC, context=0.1)

    def order(seed: int) -> list[int]:
        return torch.cat([y for _, y in make_loader(ds, 4, shuffle=True, seed=seed)]).tolist()

    assert order(42) == order(42)
    assert order(42) != order(7)
    assert sorted(order(42)) == list(range(len(df)))
    unshuffled = torch.cat([y for _, y in make_loader(ds, 5, shuffle=False, seed=1)]).tolist()
    assert unshuffled == list(range(len(df)))


def test_augmentation_is_reproducible_with_seed(tmp_path: Path) -> None:
    df = _write_images(tmp_path, n=6)
    ds = ManifestDataset(df, tmp_path, SPEC, context=0.1, augment=HorizontalFlip(p=0.5))

    def run(seed: int) -> torch.Tensor:
        seed_everything(seed)
        return torch.stack([ds[i][0] for i in range(len(ds))])

    assert torch.equal(run(42), run(42))
    img = Image.new("RGB", (4, 1))
    img.putpixel((0, 0), (255, 0, 0))
    assert HorizontalFlip(p=1.0)(img).getpixel((3, 0)) == (255, 0, 0)
    assert HorizontalFlip(p=0.0)(img).getpixel((0, 0)) == (255, 0, 0)


def test_load_manifest_validation(tmp_path: Path) -> None:
    df = _write_images(tmp_path, n=3)
    path = tmp_path / "m.csv"
    df.to_csv(path, index=False)
    assert len(load_manifest(path, ["train"])) == 3
    with pytest.raises(DatasetError):
        load_manifest(path, ["test"])
    df.drop(columns=["n_boxes"]).to_csv(path, index=False)
    with pytest.raises(DatasetError):
        load_manifest(path)
    with pytest.raises(DatasetError):
        load_manifest(tmp_path / "absent.csv")
