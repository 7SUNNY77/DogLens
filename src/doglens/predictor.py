"""Production predictor: trusted bundle -> top-k breeds for one user-selected region.

Contract:
    predict(image, roi=None, top_k=3) -> PredictionResult
    image: validated RGB image after EXIF orientation (see image_io.load_image)
    roi:   user box (xmin, ymin, xmax, ymax), exclusive maxima, *before* context;
           None = whole image (no context expansion)

Context from the bundle crop policy is added exactly once, in ``roi.model_region`` - the same
function the UI uses for its crop preview. Scores are softmax over all 120 supported classes;
they are not calibrated probabilities of a correct answer, and no "not a dog" / "mixed breed"
verdict is derived from them.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from doglens.bundle import BundleError, LoadedBundle, load_bundle
from doglens.catalog import CatalogEntry, check_catalog_matches_mapping, load_catalog
from doglens.config import DEFAULT_LIMITS, ImageLimits
from doglens.errors import DogLensError
from doglens.preprocessing import preprocess
from doglens.roi import APP_CROP_POLICY, ModelRegion, crop_model_region
from doglens.schemas import Prediction, PredictionResult

logger = logging.getLogger(__name__)

DEFAULT_BUNDLE_DIR = Path("artifacts/model_bundle")
DEFAULT_CATALOG = Path("assets/breeds.json")
SMALL_CROP_SIDE = 112  # below half of the 224 px model input the crop is strongly upsampled


class PredictionError(DogLensError):
    """Inference could not produce a valid result."""


def rank_scores(scores: np.ndarray, top_k: int) -> list[int]:
    """Class indices sorted by score descending, class_id ascending on exact ties."""
    order = np.lexsort((np.arange(scores.size), -scores))
    return [int(i) for i in order[:top_k]]


class Predictor:
    def __init__(self, bundle: LoadedBundle, catalog: list[CatalogEntry], catalog_digest: str):
        check_catalog_matches_mapping(catalog_digest, catalog, bundle.classes)
        self._bundle = bundle
        self._catalog = catalog
        self._lock = threading.Lock()  # one forward at a time on the shared model
        policy = bundle.metadata.get("crop_policy", {}).get("training_and_evaluation", {})
        if "context" not in policy:
            raise BundleError("bundle metadata has no training crop context")
        self.context = float(policy["context"])
        self.model_version: str = bundle.metadata["model_version"]
        self.device = bundle.device
        self.num_classes = len(bundle.classes)

    @classmethod
    def from_bundle_dir(
        cls,
        bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
        catalog_path: str | Path = DEFAULT_CATALOG,
        device: str = "cpu",
    ) -> Predictor:
        bundle = load_bundle(bundle_dir, device)
        try:
            digest, catalog = load_catalog(catalog_path)
        except DogLensError as exc:
            raise BundleError(f"breed catalog unavailable: {exc.detail or exc}") from exc
        try:
            return cls(bundle, catalog, digest)
        except DogLensError as exc:
            if isinstance(exc, BundleError):
                raise
            raise BundleError(f"catalog does not match the bundle: {exc.detail or exc}") from exc

    @property
    def metadata(self) -> dict:
        return self._bundle.metadata

    @property
    def catalog(self) -> list[CatalogEntry]:
        return list(self._catalog)

    @property
    def spec(self):
        return self._bundle.spec

    def crop_policy(self) -> str:
        return f"{APP_CROP_POLICY} (context={self.context})"

    def prepare(
        self, image: Image.Image, roi=None, limits: ImageLimits = DEFAULT_LIMITS
    ) -> tuple[Image.Image, ModelRegion]:
        """The exact crop the model will see (also used for the UI preview)."""
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise PredictionError(
                "invalid_image", "Изображение не подготовлено для распознавания.", "need RGB PIL"
            )
        return crop_model_region(image, roi, self.context, limits)

    def predict(self, image: Image.Image, roi=None, top_k: int = 3) -> PredictionResult:
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= self.num_classes
        ):
            raise PredictionError(
                "invalid_top_k", "Некорректное число вариантов.", f"top_k={top_k!r}"
            )
        request_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        crop, region = self.prepare(image, roi)
        tensor = preprocess(crop, self._bundle.spec).unsqueeze(0)
        t1 = time.perf_counter()
        with self._lock, torch.inference_mode():
            model = self._bundle.model
            model.eval()
            logits = model(tensor.to(self.device)).float().cpu().numpy()[0]
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        t2 = time.perf_counter()
        if logits.shape != (self.num_classes,) or not np.isfinite(logits).all():
            raise PredictionError(
                "invalid_output", "Модель вернула некорректный результат.", "non-finite logits"
            )
        z = logits.astype(np.float64) - logits.max()
        scores = np.exp(z) / np.exp(z).sum()  # softmax over all supported classes
        if not np.isfinite(scores).all() or abs(scores.sum() - 1.0) > 1e-6:
            raise PredictionError("invalid_output", "Модель вернула некорректный результат.")
        predictions = tuple(
            Prediction(
                class_id=self._catalog[i].class_id,
                breed_key=self._catalog[i].breed_key,
                name_ru=self._catalog[i].display_name_ru,
                name_en=self._catalog[i].name_en,
                score=float(scores[i]),
            )
            for i in rank_scores(scores, top_k)
        )
        warnings = []
        if region.whole_image:
            warnings.append(
                "Использовано всё изображение. Если на фото несколько собак или собака мелкая, "
                "выделите одну собаку рамкой."
            )
        cw = region.crop_box[2] - region.crop_box[0]
        ch = region.crop_box[3] - region.crop_box[1]
        if min(cw, ch) < SMALL_CROP_SIDE:
            warnings.append(
                "Выбранная область маленькая: изображение сильно увеличивается, результат может "
                "быть менее надёжным."
            )
        t3 = time.perf_counter()
        timings = {
            "preprocessing": (t1 - t0) * 1000,
            "model": (t2 - t1) * 1000,
            "postprocessing": (t3 - t2) * 1000,
            "total": (t3 - t0) * 1000,
        }
        logger.info(
            "prediction request=%s model=%s crop=%s total_ms=%.1f",
            request_id,
            self.model_version,
            region.crop_box,
            timings["total"],
        )
        return PredictionResult(
            request_id=request_id,
            model_version=self.model_version,
            predictions=predictions,
            timings_ms=timings,
            warnings=tuple(warnings),
            user_roi=region.user_roi,
            crop_box=region.crop_box,
        )


_CACHE: dict[tuple[str, str, str], Predictor] = {}
_CACHE_LOCK = threading.Lock()


def get_predictor(
    bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
    catalog_path: str | Path = DEFAULT_CATALOG,
    device: str = "cpu",
) -> Predictor:
    """Process-wide cached predictor (the model only; no user data is cached)."""
    key = (str(Path(bundle_dir).resolve()), str(Path(catalog_path).resolve()), device)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = Predictor.from_bundle_dir(bundle_dir, catalog_path, device)
        return _CACHE[key]
