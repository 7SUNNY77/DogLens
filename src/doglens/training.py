"""Training primitives: device selection, steps, checkpoints, environment info, smoke test."""

from __future__ import annotations

import math
import platform
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn

from doglens.dataset import HorizontalFlip, ManifestDataset, make_loader, seed_everything
from doglens.model import BreedClassifier, count_trainable_parameters
from doglens.preprocessing import PreprocessingSpec

CHECKPOINT_FORMAT_VERSION = 1


class TrainingError(RuntimeError):
    """Expected, user-presentable training failure."""


def select_device(requested: str = "auto") -> torch.device:
    """auto: CUDA -> MPS -> CPU, decided once. An explicit device must work or fail."""
    mps = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps and mps.is_available())
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps") if mps_ok else torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not mps_ok:
            raise TrainingError("--device mps requested, but MPS is not available")
        return torch.device("mps")
    raise TrainingError(f"unknown device {requested!r}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def environment_info(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        # str(): torch.__version__ is a TorchVersion object, which weights_only loading rejects
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
    }


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower() or "not enough memory" in str(exc).lower()


OOM_ADVICE = "Недостаточно памяти: уменьшите loader.batch_size 16 → 8 → 4."


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise TrainingError("no trainable parameters")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    try:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise TrainingError(f"non-finite loss: {value}")
        loss.backward()
        optimizer.step()
    except TrainingError:
        raise
    except (RuntimeError, MemoryError) as exc:
        if _is_oom(exc):
            raise TrainingError(OOM_ADVICE) from exc
        raise
    return value


@torch.inference_mode()
def predict_logits(model: nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    return model(images.to(device)).float().cpu()


def rng_state() -> dict[str, Any]:
    py = random.getstate()
    np_state = np.random.get_state()
    return {
        "python": [py[0], list(py[1]), py[2]],
        "numpy": [
            np_state[0],
            np_state[1].tolist(),
            *[int(x) for x in np_state[2:4]],
            float(np_state[4]),
        ],
        "torch": torch.get_rng_state(),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    config: dict[str, Any],
    mapping_sha256: str,
    preprocessing: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """state_dict-only checkpoint (no pickled Python objects), loadable with weights_only=True."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "config": config,
        "mapping_sha256": mapping_sha256,
        "preprocessing": preprocessing,
        "rng": rng_state(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, expected_mapping_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise TrainingError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise TrainingError("unsupported checkpoint format")
    if expected_mapping_sha256 is not None and payload["mapping_sha256"] != expected_mapping_sha256:
        raise TrainingError("checkpoint was trained with a different class mapping")
    return payload


# --- smoke test ----------------------------------------------------------------------------------


@dataclass
class SmokeSettings:
    per_class: int
    train_batches: int
    batch_size: int
    overfit_batch_size: int
    overfit_steps: int
    overfit_lr: float
    overfit_max_loss_ratio: float
    finetune_steps: int
    head_lr: float
    fine_tune_lr: float
    weight_decay: float
    unfreeze_blocks: tuple[str, ...]
    bn_policy: str
    context: float
    seed: int
    horizontal_flip: bool


def smoke_subset(train_records: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """First ``per_class`` records of every class by image_id (deterministic)."""
    ordered = train_records.sort_values(["class_id", "image_id"])
    return ordered.groupby("class_id", sort=True).head(per_class).reset_index(drop=True)


def _bn_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: buf.detach().clone()
        for name, buf in module.named_buffers()
        if "running_" in name or "num_batches_tracked" in name
    }


def run_smoke_test(
    model: BreedClassifier,
    train_records: pd.DataFrame,
    images_dir: Path,
    spec: PreprocessingSpec,
    settings: SmokeSettings,
    device: torch.device,
    checkpoint_path: Path,
    checkpoint_meta: dict[str, Any],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Pipeline check on a tiny subset. Raises TrainingError when a check fails."""
    s = settings
    seed_everything(s.seed)
    checks: dict[str, Any] = {}
    t_start = time.perf_counter()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    subset = smoke_subset(train_records, s.per_class)
    augment = HorizontalFlip() if s.horizontal_flip else None
    train_ds = ManifestDataset(subset, images_dir, spec, s.context, augment=augment)
    loader = make_loader(train_ds, s.batch_size, shuffle=True, seed=s.seed)
    log(f"subset: {len(subset)} images, {subset['class_id'].nunique()} classes")

    # 1. Stage A (frozen backbone) on the subset.
    model.set_trainable("head", bn_policy="frozen")
    checks["trainable_params_head"] = count_trainable_parameters(model)
    bn_before = _bn_buffers(model.net.features)
    optimizer = build_optimizer(model, s.head_lr, s.weight_decay)
    head_losses: list[float] = []
    step_times: list[float] = []
    t_phase = time.perf_counter()
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= s.train_batches:
            break
        if batch_idx == 0:
            shape = tuple(predict_logits(model, images, device).shape)
            checks["output_shape"] = list(shape)
            if shape != (images.shape[0], model.num_classes):
                raise TrainingError(f"unexpected output shape {shape}")
        t0 = time.perf_counter()
        head_losses.append(train_step(model, images, labels, optimizer, loss_fn, device))
        synchronize(device)
        step_times.append(time.perf_counter() - t0)
        log(f"head step {batch_idx + 1}: loss={head_losses[-1]:.4f}")
    head_phase_s = time.perf_counter() - t_phase
    bn_after = _bn_buffers(model.net.features)
    checks["head_bn_stats_unchanged"] = all(
        torch.equal(bn_before[k], bn_after[k]) for k in bn_before
    )
    if not checks["head_bn_stats_unchanged"]:
        raise TrainingError("frozen backbone BatchNorm statistics changed in stage A")

    # 2. Overfit one fixed batch (no augmentation) -> loss must drop clearly.
    fixed_ds = ManifestDataset(subset, images_dir, spec, s.context, augment=None)
    gen = torch.Generator().manual_seed(s.seed)
    idx = torch.randperm(len(fixed_ds), generator=gen)[: s.overfit_batch_size].tolist()
    batch = [fixed_ds[i] for i in idx]
    fixed_images = torch.stack([b[0] for b in batch])
    fixed_labels = torch.tensor([b[1] for b in batch])
    optimizer = build_optimizer(model, s.overfit_lr, s.weight_decay)
    overfit_losses = [
        train_step(model, fixed_images, fixed_labels, optimizer, loss_fn, device)
        for _ in range(s.overfit_steps)
    ]
    log(f"overfit: first={overfit_losses[0]:.4f} last={overfit_losses[-1]:.4f}")
    ratio = overfit_losses[-1] / overfit_losses[0]
    checks["overfit_loss_ratio"] = ratio
    if not ratio < s.overfit_max_loss_ratio:
        raise TrainingError(f"overfit loss did not drop enough: ratio={ratio:.3f}")

    # 3. Stage B step(s): fresh optimizer, listed blocks unfrozen, BN per policy.
    finetune_losses: list[float] = []
    finetune_step_times: list[float] = []
    if s.finetune_steps:
        model.set_trainable("finetune", s.unfreeze_blocks, s.bn_policy)
        checks["trainable_params_finetune"] = count_trainable_parameters(model)
        frozen_before = {k: v.clone() for k, v in model.net.features[0].state_dict().items()}
        tuned = model.net.get_submodule(s.unfreeze_blocks[-1])
        tuned_before = [p.detach().clone() for p in tuned.parameters()]
        bn_before = _bn_buffers(model.net.features)
        optimizer = build_optimizer(model, s.fine_tune_lr, s.weight_decay)
        for _ in range(s.finetune_steps):
            t0 = time.perf_counter()
            finetune_losses.append(
                train_step(model, fixed_images, fixed_labels, optimizer, loss_fn, device)
            )
            synchronize(device)
            finetune_step_times.append(time.perf_counter() - t0)
        checks["finetune_frozen_block_unchanged"] = all(
            torch.equal(frozen_before[k], v) for k, v in model.net.features[0].state_dict().items()
        )
        checks["finetune_unfrozen_block_changed"] = any(
            not torch.equal(a, b) for a, b in zip(tuned_before, tuned.parameters(), strict=True)
        )
        bn_after = _bn_buffers(model.net.features)
        checks["finetune_bn_stats_unchanged"] = all(
            torch.equal(bn_before[k], bn_after[k]) for k in bn_before
        )
        if not (
            checks["finetune_frozen_block_unchanged"] and checks["finetune_unfrozen_block_changed"]
        ):
            raise TrainingError("stage B freezing check failed")
        if s.bn_policy == "frozen" and not checks["finetune_bn_stats_unchanged"]:
            raise TrainingError("BatchNorm statistics changed under the 'frozen' policy")

    # 4. Checkpoint round trip: identical predictions after reload into a fresh model.
    reference = predict_logits(model, fixed_images, device)
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=0,
        config=checkpoint_meta["config"],
        mapping_sha256=checkpoint_meta["mapping_sha256"],
        preprocessing=spec.to_dict(),
        extra={"kind": "smoke_test"},
    )
    payload = load_checkpoint(checkpoint_path, checkpoint_meta["mapping_sha256"])
    reloaded = BreedClassifier(model.num_classes)
    reloaded.load_state_dict(payload["model_state"], strict=True)
    reloaded.to(device)
    again = predict_logits(reloaded, fixed_images, device)
    max_diff = float((reference - again).abs().max())
    checks["reload_max_abs_logit_diff"] = max_diff
    checks["reload_predictions_equal"] = bool(
        torch.equal(reference.argmax(1), again.argmax(1)) and max_diff <= 1e-5
    )
    if not checks["reload_predictions_equal"]:
        raise TrainingError(f"predictions differ after checkpoint reload (max diff {max_diff})")
    checks["all_losses_finite"] = all(
        math.isfinite(v) for v in head_losses + overfit_losses + finetune_losses
    )

    median = lambda xs: float(np.median(xs)) if xs else None  # noqa: E731
    return {
        "checks": checks,
        "losses": {"head": head_losses, "overfit": overfit_losses, "finetune": finetune_losses},
        "timing_s": {
            "total": time.perf_counter() - t_start,
            "head_step_median": median(step_times),
            # wall time per batch including image decoding and preprocessing (num_workers=0)
            "head_batch_wall_mean": head_phase_s / max(1, len(head_losses)),
            "finetune_step_median": median(finetune_step_times),
        },
        "subset_size": len(subset),
    }
