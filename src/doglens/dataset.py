"""torch Dataset over the prepared manifest: context crop on the fly, shared preprocessing.

Image decoding reuses ``data_prep.load_dataset_image`` (content-based, alpha on white, no EXIF
transpose: boxes refer to stored pixels and the preparation excludes EXIF-rotated records).
The crop is the manifest box (largest valid box, exclusive maxima) plus the configured context,
clipped to the image (``data_prep.context_box``).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from doglens.data_prep import DatasetError, context_box, load_dataset_image
from doglens.preprocessing import PreprocessingSpec, preprocess
from doglens.schemas import MANIFEST_COLUMNS, SPLITS

Augment = Callable[[Image.Image], Image.Image]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_manifest(path: str | Path, splits: Sequence[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"manifest not found: {path}")
    df = pd.read_csv(path)
    if list(df.columns) != list(MANIFEST_COLUMNS):
        raise DatasetError(f"manifest columns differ from schema: {list(df.columns)}")
    if not set(df["split"]) <= set(SPLITS):
        raise DatasetError("manifest contains unknown split values")
    if splits is not None:
        df = df[df["split"].isin(list(splits))]
    if df.empty:
        raise DatasetError(f"no manifest records for splits {splits}")
    return df.reset_index(drop=True)


class HorizontalFlip:
    """Train-only augmentation drawing from torch's global RNG (seeded by seed_everything)."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.mirror(image) if torch.rand(()) < self.p else image


class ManifestDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        images_dir: str | Path,
        spec: PreprocessingSpec,
        context: float,
        augment: Augment | None = None,
    ) -> None:
        if records.empty:
            raise DatasetError("empty dataset")
        self.records = records.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.spec = spec
        self.context = context
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def crop(self, index: int) -> Image.Image:
        row = self.records.iloc[index]
        image = load_dataset_image(self.images_dir / row["relative_path"])
        box = (
            int(row["bbox_xmin"]),
            int(row["bbox_ymin"]),
            int(row["bbox_xmax"]),
            int(row["bbox_ymax"]),
        )
        return image.crop(context_box(box, image.size, self.context))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        crop = self.crop(index)
        if self.augment is not None:
            crop = self.augment(crop)
        return preprocess(crop, self.spec), int(self.records.iloc[index]["class_id"])


def _seed_worker(worker_id: int) -> None:  # pragma: no cover - only with num_workers > 0
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """Deterministic order for a given seed (explicit generator, seeded workers)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=_seed_worker,
        drop_last=False,
    )
