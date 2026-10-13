"""Full two-stage training experiment with epoch-boundary checkpoints and resume.

Stage A trains the classifier on a frozen backbone; stage B starts from the best stage-A
weights, unfreezes the configured blocks and uses a fresh optimizer. The best checkpoint is
chosen by validation Top-1 (ties: lower validation loss) across the whole experiment.

Resume works from the boundary of the last *completed* epoch (``last.pt``). An epoch that was
interrupted (time limit, crash) is repeated from its start; mid-batch resume is not supported.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.torch_version import TorchVersion
from torch.utils.data import DataLoader

from doglens.fileutil import atomic_write_json, sha256_file
from doglens.model import BreedClassifier, count_trainable_parameters
from doglens.training import TrainingError, build_optimizer, rng_state, synchronize, train_step

RUN_CHECKPOINT_FORMAT = 2
# Config sections that must be identical to resume a run.
RESUME_CRITICAL_KEYS = (
    "seed",
    "precision",
    "data",
    "loader",
    "model",
    "pretrained",
    "augment",
    "optimizer",
    "stage_a",
    "stage_b",
    "early_stopping_patience",
)


class StopRequested(Exception):  # noqa: N818 - control-flow signal, not an error
    """Raised between batches when the time limit is reached."""


@dataclass(frozen=True)
class StagePlan:
    name: str
    mode: str
    epochs: int
    lr: float
    unfreeze_blocks: tuple[str, ...]
    bn_policy: str


def stage_plans(cfg_dict: dict[str, Any]) -> list[StagePlan]:
    a, b = cfg_dict["stage_a"], cfg_dict["stage_b"]
    plans = [StagePlan("A", "head", int(a["epochs"]), float(a["lr"]), (), "frozen")]
    if int(b["epochs"]) > 0:
        plans.append(
            StagePlan(
                "B",
                "finetune",
                int(b["epochs"]),
                float(b["lr"]),
                tuple(b["unfreeze_blocks"]),
                str(b["batchnorm_policy"]),
            )
        )
    return [p for p in plans if p.epochs > 0]


# --- metrics -----------------------------------------------------------------------------------


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Validation pass: eval mode, inference_mode, sample-weighted loss, Top-1/Top-3."""
    model.eval()
    total_loss, n, top1, top3 = 0.0, 0, 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).float()
            total_loss += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
            best3 = logits.topk(min(3, logits.shape[1]), dim=1).indices
            top1 += int((best3[:, 0] == labels).sum())
            top3 += int((best3 == labels[:, None]).any(dim=1).sum())
            n += int(labels.numel())
    if n == 0:
        raise TrainingError("validation loader is empty")
    loss = total_loss / n
    if not math.isfinite(loss):
        raise TrainingError(f"non-finite validation loss: {loss}")
    return {"loss": loss, "top1": top1 / n, "top3": top3 / n, "n": n}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    should_stop: Callable[[], bool],
    log: Callable[[str], None],
    log_every: int = 100,
) -> dict[str, float]:
    """One pass over the loader. Loss is averaged per sample (last batch may be smaller)."""
    loss_fn = nn.CrossEntropyLoss()  # batch mean; re-weighted by batch size below
    weighted, n, batches = 0.0, 0, 0
    t0 = time.perf_counter()
    for images, labels in loader:
        if should_stop():
            raise StopRequested
        batch_loss = train_step(model, images, labels, optimizer, loss_fn, device)
        weighted += batch_loss * int(labels.numel())
        n += int(labels.numel())
        batches += 1
        if log_every and batches % log_every == 0:
            log(f"  batch {batches}: running loss={weighted / n:.4f}")
    synchronize(device)
    if n == 0:
        raise TrainingError("train loader is empty")
    return {"loss": weighted / n, "n": n, "batches": batches, "seconds": time.perf_counter() - t0}


def is_better(candidate: dict[str, float], best: dict[str, Any] | None) -> bool:
    """Higher validation Top-1 wins; on equal Top-1 the lower validation loss wins."""
    if best is None:
        return True
    if candidate["top1"] != best["top1"]:
        return candidate["top1"] > best["top1"]
    return candidate["loss"] < best["loss"]


# --- RNG ---------------------------------------------------------------------------------------


def restore_rng_state(state: dict[str, Any]) -> None:
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    np_state = state["numpy"]
    np.random.set_state(
        (np_state[0], np.array(np_state[1], dtype=np.uint32), np_state[2], np_state[3], np_state[4])
    )
    torch.set_rng_state(state["torch"])


# --- experiment ---------------------------------------------------------------------------------


def initial_state() -> dict[str, Any]:
    return {
        "stage_index": 0,
        "stage": None,
        "stage_epochs_completed": 0,
        "stage_finished": False,
        "stage_stop_reason": None,
        "global_epochs_completed": 0,
        "stage_best": None,
        "patience_counter": 0,
        "stage_bests": {},
        "stage_stop_reasons": {},
        "global_best": None,
        "history": [],
        "finished": False,
    }


def config_differences(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    return [k for k in RESUME_CRITICAL_KEYS if saved.get(k) != current.get(k)]


class Experiment:
    def __init__(
        self,
        *,
        model: BreedClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        run_dir: Path,
        config: dict[str, Any],
        identity: dict[str, Any],
        log: Callable[[str], None] = print,
        should_stop: Callable[[], bool] = lambda: False,
        log_every: int = 100,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)
        # mapping/preprocessing/manifest hashes, versions, device: stored in every checkpoint.
        # JSON round trip guarantees plain Python types (e.g. TorchVersion -> str).
        self.identity = json.loads(json.dumps(identity))
        self.config = json.loads(json.dumps(config))
        self.log = log
        self.should_stop = should_stop
        self.log_every = log_every
        self.plans = stage_plans(config)
        if not self.plans:
            raise TrainingError("no training stages with epochs > 0")
        self.patience = int(config["early_stopping_patience"])
        self.weight_decay = float(config["optimizer"]["weight_decay"])
        self.state = initial_state()
        self.optimizer: torch.optim.Optimizer | None = None

    # --- checkpoint I/O ---

    def _generator(self) -> torch.Generator:
        gen = self.train_loader.generator
        if gen is None:
            raise TrainingError("train loader needs an explicit torch.Generator for resume")
        return gen

    def _payload(self) -> dict[str, Any]:
        return {
            "format_version": RUN_CHECKPOINT_FORMAT,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": None,  # no LR scheduler is used
            "state": json.loads(json.dumps(self.state)),
            "rng": rng_state(),
            "loader_generator_state": self._generator().get_state(),
            "config": self.config,
            "identity": self.identity,
            "saved_at_unix": time.time(),
        }

    def _save(self, name: str) -> Path:
        path = self.run_dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._payload(), tmp)
        tmp.replace(path)
        return path

    def resume_from(self, checkpoint: Path) -> None:
        payload = load_run_checkpoint(checkpoint)
        check_resume_compatibility(payload, self.config, self.identity)
        self.state = payload["state"]
        self.model.load_state_dict(payload["model_state"], strict=True)
        if not self.state["finished"] and not self.state["stage_finished"]:
            plan = self.plans[self.state["stage_index"]]
            self._configure_stage(plan)
            if payload["optimizer_state"] is None:
                raise TrainingError("checkpoint has no optimizer state for an unfinished stage")
            assert self.optimizer is not None
            self.optimizer.load_state_dict(payload["optimizer_state"])
        restore_rng_state(payload["rng"])
        self._generator().set_state(payload["loader_generator_state"])
        self.log(
            f"resumed from {checkpoint}: stage {self.state['stage']} epochs completed "
            f"{self.state['stage_epochs_completed']} "
            f"(global {self.state['global_epochs_completed']}); "
            "any interrupted epoch is repeated from its start"
        )

    # --- stages ---

    def _configure_stage(self, plan: StagePlan) -> None:
        self.model.set_trainable(plan.mode, plan.unfreeze_blocks, plan.bn_policy)
        self.optimizer = build_optimizer(self.model, plan.lr, self.weight_decay)
        self.state["stage"] = plan.name

    def _start_stage(self, index: int) -> None:
        plan = self.plans[index]
        if index > 0:
            prev = self.plans[index - 1].name
            best_file = self.run_dir / f"best_stage_{prev}.pt"
            payload = load_run_checkpoint(best_file)
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.log(f"stage {plan.name}: restored best stage-{prev} weights from {best_file.name}")
        self.state.update(
            stage_index=index,
            stage_epochs_completed=0,
            stage_finished=False,
            stage_stop_reason=None,
            stage_best=None,
            patience_counter=0,
        )
        self._configure_stage(plan)
        self.log(
            f"stage {plan.name}: mode={plan.mode} blocks={list(plan.unfreeze_blocks)} "
            f"bn_policy={plan.bn_policy} lr={plan.lr} max_epochs={plan.epochs} "
            f"trainable_params={count_trainable_parameters(self.model)}"
        )

    def run(self) -> dict[str, Any]:
        """Train until all stages finish. Raises StopRequested at a batch boundary on time limit."""
        st = self.state
        if st["finished"]:
            self.log("run already finished; nothing to do")
            return st
        if st["stage"] is None:
            self._start_stage(0)
            self._save("last.pt")  # epoch-0 resume point: an interrupted first epoch is resumable
        while True:
            if st["stage_finished"]:
                nxt = st["stage_index"] + 1
                if nxt >= len(self.plans):
                    st["finished"] = True
                    self._save("last.pt")
                    break
                self._start_stage(nxt)
            plan = self.plans[st["stage_index"]]
            epoch = st["stage_epochs_completed"] + 1
            self.log(f"stage {plan.name} epoch {epoch}/{plan.epochs}: training")
            assert self.optimizer is not None
            train = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.device,
                self.should_stop,
                self.log,
                self.log_every,
            )
            t0 = time.perf_counter()
            val = evaluate_loader(self.model, self.val_loader, self.device)
            val_seconds = time.perf_counter() - t0
            st["stage_epochs_completed"] = epoch
            st["global_epochs_completed"] += 1
            record = {
                "stage": plan.name,
                "stage_epoch": epoch,
                "global_epoch": st["global_epochs_completed"],
                "train_loss": train["loss"],
                "train_n": train["n"],
                "val_loss": val["loss"],
                "val_top1": val["top1"],
                "val_top3": val["top3"],
                "val_n": val["n"],
                "train_seconds": train["seconds"],
                "val_seconds": val_seconds,
                "lr": plan.lr,
            }
            metrics = {"top1": val["top1"], "loss": val["loss"]}
            stage_improved = is_better(metrics, st["stage_best"])
            global_improved = is_better(metrics, st["global_best"])
            if stage_improved:
                st["stage_best"] = {"stage_epoch": epoch, **metrics}
                st["stage_bests"][plan.name] = {"stage_epoch": epoch, **metrics}
                st["patience_counter"] = 0
            else:
                st["patience_counter"] += 1
            if global_improved:
                st["global_best"] = {
                    "stage": plan.name,
                    "stage_epoch": epoch,
                    "global_epoch": st["global_epochs_completed"],
                    **metrics,
                    "val_top3": val["top3"],
                }
            record.update(
                stage_best=stage_improved,
                global_best=global_improved,
                patience=st["patience_counter"],
            )
            st["history"].append(record)
            if epoch >= plan.epochs:
                st["stage_finished"], st["stage_stop_reason"] = True, "max_epochs"
            elif st["patience_counter"] >= self.patience:
                st["stage_finished"], st["stage_stop_reason"] = True, "early_stopping"
            self.log(
                f"stage {plan.name} epoch {epoch}: train_loss={train['loss']:.4f} "
                f"val_loss={val['loss']:.4f} val_top1={val['top1']:.4f} val_top3={val['top3']:.4f} "
                f"train={train['seconds']:.0f}s val={val_seconds:.0f}s"
                + (" [stage best]" if stage_improved else "")
                + (" [global best]" if global_improved else "")
                + (f" patience={st['patience_counter']}/{self.patience}")
            )
            # Best files first, then last.pt: last.pt is the resume point and must reference
            # best files that already exist.
            if stage_improved:
                self._save(f"best_stage_{plan.name}.pt")
            if global_improved:
                shutil.copyfile(
                    self.run_dir / f"best_stage_{plan.name}.pt", self.run_dir / "best.tmp"
                )
                (self.run_dir / "best.tmp").replace(self.run_dir / "best.pt")
            if st["stage_finished"]:
                st["stage_stop_reasons"][plan.name] = st["stage_stop_reason"]
                self.log(f"stage {plan.name} finished: {st['stage_stop_reason']}")
            self._save("last.pt")
            atomic_write_json(self.run_dir / "history.json", st["history"])
        gb = st["global_best"]
        self.log(
            f"experiment finished; global best: stage {gb['stage']} epoch {gb['stage_epoch']} "
            f"val_top1={gb['top1']:.4f} val_loss={gb['loss']:.4f}"
        )
        return st


def load_run_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise TrainingError(f"checkpoint not found: {path}")
    try:
        # TorchVersion (a str subclass) is allow-listed only for checkpoints written before
        # identity values were normalised to plain str (run effb0-seed42-exp1, stage A).
        with torch.serialization.safe_globals([TorchVersion]):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != RUN_CHECKPOINT_FORMAT:
        raise TrainingError(
            f"{path} is not a DogLens run checkpoint (format {RUN_CHECKPOINT_FORMAT})"
        )
    return payload


def check_resume_compatibility(
    payload: dict[str, Any], config: dict[str, Any], identity: dict[str, Any]
) -> None:
    saved = payload["identity"]
    for key in ("mapping_sha256", "manifest_sha256", "preprocessing", "pretrained_sha256"):
        if saved.get(key) != identity.get(key):
            raise TrainingError(
                f"cannot resume: {key} differs from the checkpoint. Start a new run instead."
            )
    if saved.get("device") != identity.get("device"):
        raise TrainingError(
            f"cannot resume: checkpoint device {saved.get('device')} differs from "
            f"{identity.get('device')}; use the same --device"
        )
    diffs = config_differences(payload["config"], config)
    if diffs:
        raise TrainingError(
            f"cannot resume: config sections differ from the checkpoint: {diffs}. "
            "Start a new run for a changed configuration."
        )


def best_checkpoint_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "best.pt"
    payload = load_run_checkpoint(path)
    return {
        "file": path.as_posix(),
        "sha256": sha256_file(path),
        "global_best": payload["state"]["global_best"],
    }
