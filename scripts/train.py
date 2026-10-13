"""Train the DogLens classifier (EfficientNet-B0, 120 Stanford Dogs classes).

Modes:
  --smoke-test   pipeline check on a tiny subset (not a quality measure)
  --benchmark    timing of real train/validation batches and a full-run estimate
  (default)      full two-stage experiment in artifacts/runs/<run_id>/
  --resume P     continue a run from its last completed epoch (checkpoint P, usually last.pt)

Resume works from epoch boundaries only: an interrupted epoch is repeated from its start.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from doglens.catalog import MappingError, load_class_mapping, mapping_sha256
from doglens.config import TrainingConfig, load_training_config
from doglens.data_prep import DatasetError
from doglens.dataset import (
    HorizontalFlip,
    ManifestDataset,
    load_manifest,
    make_loader,
    seed_everything,
)
from doglens.errors import ConfigError
from doglens.experiment import (
    Experiment,
    StopRequested,
    best_checkpoint_summary,
    load_run_checkpoint,
)
from doglens.fileutil import atomic_write_json, sha256_file
from doglens.model import BreedClassifier, PretrainedWeightsError, count_trainable_parameters
from doglens.preprocessing import PreprocessingSpec, load_spec, spec_from_weights
from doglens.training import (
    SmokeSettings,
    TrainingError,
    build_optimizer,
    environment_info,
    run_smoke_test,
    select_device,
    synchronize,
    train_step,
)

EXIT_INTERRUPTED = 3
EXPECTED_ERRORS = (
    TrainingError,
    PretrainedWeightsError,
    MappingError,
    ConfigError,
    DatasetError,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the DogLens EfficientNet-B0 classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true", help="run the pipeline smoke test")
    mode.add_argument("--benchmark", action="store_true", help="time real batches, estimate run")
    mode.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume a run from this checkpoint (normally <run_dir>/last.pt)",
    )
    p.add_argument(
        "--run-id", default=None, help="name of the new run directory (default: UTC timestamp)"
    )
    p.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="stop between batches after this wall time; exits with code 3 and keeps "
        "the last completed-epoch checkpoint",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="override config 'device'; an explicit device must be available",
    )
    p.add_argument("--mapping", type=Path, default=Path("data/prepared/class_mapping.json"))
    p.add_argument("--preprocessing", type=Path, default=Path("assets/preprocessing.json"))
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="summary JSON for --smoke-test/--benchmark (default under reports/)",
    )
    p.add_argument("--benchmark-batches", type=int, default=5)
    p.add_argument("--benchmark-warmup", type=int, default=2)
    p.add_argument("--log-every", type=int, default=100, help="batches between progress lines")
    args = p.parse_args(argv)
    if args.max_minutes is not None and args.max_minutes <= 0:
        p.error("--max-minutes must be > 0")
    if args.run_id is not None and (not args.run_id or any(c in args.run_id for c in "/\\:")):
        p.error("--run-id must be a plain directory name")
    if args.benchmark_batches < 1 or args.benchmark_warmup < 0:
        p.error("--benchmark-batches must be >= 1 and --benchmark-warmup >= 0")
    return args  # fmt: skip


# --- shared setup -------------------------------------------------------------------------


@dataclass
class Setup:
    cfg: TrainingConfig
    device: torch.device
    env: dict[str, Any]
    digest: str
    spec: PreprocessingSpec
    identity: dict[str, Any]


def common_setup(args: argparse.Namespace, log) -> Setup:
    cfg = load_training_config(args.config)
    device = select_device(args.device or cfg.device)
    env = environment_info(device)
    log(
        f"seed={cfg.seed} torch={env['torch']} torchvision={env['torchvision']} "
        f"device={env['device']} threads={env['torch_num_threads']} "
        f"batch_size={cfg.loader.batch_size} num_workers={cfg.loader.num_workers}"
    )
    classes = load_class_mapping(args.mapping)
    digest = mapping_sha256(classes)
    manifest = load_manifest(cfg.data.manifest)
    key_of = {c.class_id: c.breed_key for c in classes}
    pairs = zip(manifest["class_id"], manifest["breed_key"], strict=True)
    if any(key_of.get(c) != k for c, k in pairs):
        raise MappingError("manifest class_id/breed_key disagree with class_mapping.json")
    spec = spec_from_weights(cfg.pretrained.weights_enum)
    if load_spec(args.preprocessing) != spec:
        raise ConfigError(
            f"{args.preprocessing} differs from the weights metadata of the installed "
            "torchvision; regenerate it deliberately"
        )
    identity = {
        "mapping_sha256": digest,
        "manifest_sha256": sha256_file(cfg.data.manifest),
        "preprocessing": spec.to_dict(),
        "pretrained_sha256": cfg.pretrained.sha256,
        "device": str(device),
        "torch": env["torch"],
        "torchvision": env["torchvision"],
    }
    return Setup(cfg, device, env, digest, spec, identity)


def build_loaders(s: Setup):
    cfg = s.cfg
    train_df = load_manifest(cfg.data.manifest, ["train"])
    val_df = load_manifest(cfg.data.manifest, ["val"])
    augment = HorizontalFlip() if cfg.augment.horizontal_flip else None
    root = Path(cfg.data.image_root)
    train_ds = ManifestDataset(train_df, root, s.spec, cfg.data.bbox_context, augment=augment)
    val_ds = ManifestDataset(val_df, root, s.spec, cfg.data.bbox_context)
    bs, nw = cfg.loader.batch_size, cfg.loader.num_workers
    return (
        make_loader(train_ds, bs, shuffle=True, seed=cfg.seed, num_workers=nw),
        make_loader(val_ds, bs, shuffle=False, seed=cfg.seed, num_workers=nw),
    )


def git_commit() -> dict[str, Any]:
    def run(*cmd: str) -> str | None:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("git", "rev-parse", "--verify", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


# --- smoke test --------------------------------------------------------------------------------


def run_smoke(args: argparse.Namespace) -> int:
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    s = common_setup(args, log)
    cfg = s.cfg
    train = load_manifest(cfg.data.manifest, ["train"])
    log(f"preprocessing: {json.dumps(s.spec.to_dict(), ensure_ascii=False)}")
    model = BreedClassifier(cfg.model.num_classes)
    model.load_imagenet_backbone(cfg.pretrained.path, cfg.pretrained.sha256)
    log(f"pretrained: {cfg.pretrained.path} sha256={cfg.pretrained.sha256}")
    settings = SmokeSettings(
        per_class=cfg.smoke.per_class,
        train_batches=cfg.smoke.train_batches,
        batch_size=cfg.loader.batch_size,
        overfit_batch_size=cfg.smoke.overfit_batch_size,
        overfit_steps=cfg.smoke.overfit_steps,
        overfit_lr=cfg.smoke.overfit_lr,
        overfit_max_loss_ratio=cfg.smoke.overfit_max_loss_ratio,
        finetune_steps=cfg.smoke.finetune_steps,
        head_lr=cfg.stage_a.lr,
        fine_tune_lr=cfg.stage_b.lr,
        weight_decay=cfg.optimizer.weight_decay,
        unfreeze_blocks=cfg.stage_b.unfreeze_blocks,
        bn_policy=cfg.stage_b.batchnorm_policy,
        context=cfg.data.bbox_context,
        seed=cfg.seed,
        horizontal_flip=cfg.augment.horizontal_flip,
    )
    smoke_dir = Path(cfg.output.smoke_dir)
    checkpoint = smoke_dir / "smoke_checkpoint.pt"
    started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = run_smoke_test(
        model,
        train,
        Path(cfg.data.image_root),
        s.spec,
        settings,
        s.device,
        checkpoint,
        {"config": cfg.to_dict(), "mapping_sha256": s.digest},
        log=log,
    )
    batches = math.ceil(len(train) / cfg.loader.batch_size)
    timing = result["timing_s"]
    estimate = {
        "basis": "smoke-test medians on this machine; excludes validation passes; rough",
        "train_batches_per_epoch": batches,
        "stage_a_epoch_min": round(batches * timing["head_batch_wall_mean"] / 60, 1),
    }
    if timing["finetune_step_median"] is not None:
        load_per_batch = max(0.0, timing["head_batch_wall_mean"] - timing["head_step_median"])
        per_batch = timing["finetune_step_median"] + load_per_batch
        estimate["stage_b_epoch_min"] = round(batches * per_batch / 60, 1)
    report = args.report or Path("reports/train_smoke_test.json")
    summary = {
        "generated_at_utc": started,
        "command": "scripts/train.py " + " ".join(sys.argv[1:]),
        "status": "passed",
        "seed": cfg.seed,
        "environment": s.env,
        "config": str(args.config),
        "manifest_sha256": s.identity["manifest_sha256"],
        "mapping_sha256": s.digest,
        "pretrained_sha256": cfg.pretrained.sha256,
        "preprocessing": s.spec.to_dict(),
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "note": "pipeline check on a tiny subset; not a quality metric",
        "full_training_time_estimate": estimate,
        **result,
    }
    atomic_write_json(report, summary)
    (smoke_dir / "smoke_test.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"OK: smoke test passed in {timing['total']:.1f}s; report {report}")
    return 0


# --- benchmark --------------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> int:
    s = common_setup(args, print)
    cfg = s.cfg
    seed_everything(cfg.seed)
    train_loader, val_loader = build_loaders(s)
    warm, n = args.benchmark_warmup, args.benchmark_batches
    total = warm + n

    def fetch(loader) -> tuple[list, list[float]]:
        batches, times = [], []
        it = iter(loader)
        for _ in range(total):
            t0 = time.perf_counter()
            batches.append(next(it))
            times.append(time.perf_counter() - t0)
        return batches, times[warm:]

    train_batches, train_load = fetch(train_loader)
    val_batches, val_load = fetch(val_loader)
    model = BreedClassifier(cfg.model.num_classes)
    model.load_imagenet_backbone(cfg.pretrained.path, cfg.pretrained.sha256)
    model.to(s.device)
    loss_fn = torch.nn.CrossEntropyLoss()

    def time_steps(mode: str, lr: float) -> tuple[list[float], int]:
        model.set_trainable(mode, cfg.stage_b.unfreeze_blocks, cfg.stage_b.batchnorm_policy)
        opt = build_optimizer(model, lr, cfg.optimizer.weight_decay)
        times = []
        for i, (x, y) in enumerate(train_batches):
            t0 = time.perf_counter()
            train_step(model, x, y, opt, loss_fn, s.device)
            synchronize(s.device)
            if i >= warm:
                times.append(time.perf_counter() - t0)
        return times, count_trainable_parameters(model)

    step_a, params_a = time_steps("head", cfg.stage_a.lr)
    step_b, params_b = time_steps("finetune", cfg.stage_b.lr)
    model.eval()
    val_fwd = []
    with torch.inference_mode():
        for i, (x, _y) in enumerate(val_batches):
            t0 = time.perf_counter()
            model(x.to(s.device))
            synchronize(s.device)
            if i >= warm:
                val_fwd.append(time.perf_counter() - t0)

    n_train, n_val = len(train_loader.dataset), len(val_loader.dataset)
    tb, vb = len(train_loader), len(val_loader)
    med = {
        "train_load": statistics.median(train_load),
        "val_load": statistics.median(val_load),
        "stage_a_step": statistics.median(step_a),
        "stage_b_step": statistics.median(step_b),
        "val_forward": statistics.median(val_fwd),
    }
    val_epoch = vb * (med["val_load"] + med["val_forward"])
    epoch_a = tb * (med["train_load"] + med["stage_a_step"]) + val_epoch
    epoch_b = tb * (med["train_load"] + med["stage_b_step"]) + val_epoch
    total_max = cfg.stage_a.epochs * epoch_a + cfg.stage_b.epochs * epoch_b
    result = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "scripts/train.py " + " ".join(sys.argv[1:]),
        "environment": s.env,
        "batch_size": cfg.loader.batch_size,
        "num_workers": cfg.loader.num_workers,
        "warmup_batches": warm,
        "timed_batches": n,
        "train_records": n_train,
        "val_records": n_val,
        "train_batches_per_epoch": tb,
        "val_batches_per_epoch": vb,
        "trainable_params": {"stage_a": params_a, "stage_b": params_b},
        "median_seconds_per_batch": med,
        "raw_seconds": {
            "train_load": train_load,
            "val_load": val_load,
            "stage_a_step": step_a,
            "stage_b_step": step_b,
            "val_forward": val_fwd,
        },
        "estimate_minutes": {
            "stage_a_epoch": epoch_a / 60,
            "stage_b_epoch": epoch_b / 60,
            "validation_pass": val_epoch / 60,
            "full_experiment_max_epochs": total_max / 60,
            "epochs": {"stage_a": cfg.stage_a.epochs, "stage_b": cfg.stage_b.epochs},
            "basis": "medians of timed batches after warm-up; serial loading (num_workers=0); "
            "upper bound without early stopping; excludes checkpoint I/O (~seconds/epoch)",
        },
        "performance_settings": "unchanged defaults (torch threads as reported, num_workers=0)",
    }
    report = args.report or Path("reports/train_benchmark.json")
    atomic_write_json(report, result)
    print(json.dumps({"median_seconds_per_batch": med, **result["estimate_minutes"]}, indent=2))
    print(f"OK: benchmark written to {report}")
    return 0


# --- full run ----------------------------------------------------------------------------------


def _make_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"doglens.train.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):  # fmt: skip
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def run_full(args: argparse.Namespace) -> int:
    pre_log: list[str] = []
    s = common_setup(args, pre_log.append)
    cfg = s.cfg
    now = datetime.now(UTC)
    if args.resume is not None:
        run_dir = args.resume.parent
        load_run_checkpoint(args.resume)  # validates the format before touching the run
        meta_path = run_dir / "run_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    else:
        run_id = args.run_id or now.strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(cfg.output.run_dir) / run_id
        if run_dir.exists():
            raise TrainingError(
                f"run directory {run_dir} already exists; use --resume {run_dir}/last.pt "
                "or choose another --run-id"
            )
        run_dir.mkdir(parents=True)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        meta = {
            "run_id": run_id,
            "created_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config_file": args.config.as_posix(),
            "identity": s.identity,
            "environment": s.env,
            "git": git_commit(),
            "seed": cfg.seed,
            "pretrained": {
                "path": cfg.pretrained.path,
                "sha256": cfg.pretrained.sha256,
                "weights_enum": cfg.pretrained.weights_enum,
            },  # fmt: skip
            "events": [],
        }
    meta_path = run_dir / "run_metadata.json"
    logger = _make_logger(run_dir)
    log = logger.info
    for line in pre_log:
        log(line)
    command = "scripts/train.py " + " ".join(sys.argv[1:])
    meta.setdefault("events", []).append(
        {
            "event": "resume" if args.resume else "start",
            "at_utc": now.isoformat(),
            "command": command,
            "max_minutes": args.max_minutes,
        }  # fmt: skip
    )
    meta["status"] = "running"
    atomic_write_json(meta_path, meta)

    seed_everything(cfg.seed)
    train_loader, val_loader = build_loaders(s)
    log(
        f"run_dir={run_dir} train={len(train_loader.dataset)} ({len(train_loader)} batches) "
        f"val={len(val_loader.dataset)} ({len(val_loader)} batches)"
    )
    model = BreedClassifier(cfg.model.num_classes)
    if args.resume is None:
        model.load_imagenet_backbone(cfg.pretrained.path, cfg.pretrained.sha256)
        log(f"pretrained backbone: {cfg.pretrained.path} sha256={cfg.pretrained.sha256}")
    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
    exp = Experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=s.device,
        run_dir=run_dir,
        config=cfg.to_dict(),
        identity=s.identity,
        log=log,
        should_stop=(lambda: time.monotonic() > deadline) if deadline else (lambda: False),
        log_every=args.log_every,
    )
    if args.resume is not None:
        exp.resume_from(args.resume)
    t0 = time.perf_counter()
    try:
        state = exp.run()
    except (StopRequested, KeyboardInterrupt) as exc:
        last = load_run_checkpoint(run_dir / "last.pt")["state"]
        reason = "time limit" if isinstance(exc, StopRequested) else "keyboard interrupt"
        log(
            f"STOPPED ({reason}) at a batch boundary. Last complete checkpoint: {run_dir}/last.pt "
            f"(stage {last['stage']}, stage epochs completed {last['stage_epochs_completed']}). "
            f"The interrupted epoch is repeated from its start on --resume {run_dir}/last.pt"
        )
        meta["status"] = "interrupted"
        meta["events"].append({"event": "interrupted", "reason": reason,
                               "at_utc": datetime.now(UTC).isoformat()})  # fmt: skip
        atomic_write_json(meta_path, meta)
        return EXIT_INTERRUPTED if isinstance(exc, StopRequested) else 130
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        meta["status"] = "failed"
        meta["events"].append({"event": "failed", "error": f"{type(exc).__name__}: {exc}",
                               "at_utc": datetime.now(UTC).isoformat()})  # fmt: skip
        atomic_write_json(meta_path, meta)
        raise
    elapsed = time.perf_counter() - t0
    history = state["history"]
    final = {
        "status": "completed",
        "run_id": run_dir.name,
        "finished_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_seconds_this_invocation": elapsed,
        "epoch_seconds_total": sum(h["train_seconds"] + h["val_seconds"] for h in history),
        "epochs_run": {st: sum(h["stage"] == st for h in history) for st in ("A", "B")},
        "stage_bests": state["stage_bests"],
        "stage_stop_reasons": state["stage_stop_reasons"],
        "global_best": state["global_best"],
        "selection_rule": "max validation Top-1; ties -> lower validation loss; test not used",
        "best_checkpoint": best_checkpoint_summary(run_dir),
        "history": history,
        "identity": s.identity,
        "environment": s.env,
    }
    atomic_write_json(run_dir / "final_report.json", final)
    meta["status"] = "completed"
    meta["events"].append({"event": "completed", "at_utc": datetime.now(UTC).isoformat()})
    atomic_write_json(meta_path, meta)
    log(f"OK: run completed; final report {run_dir / 'final_report.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.smoke_test:
            return run_smoke(args)
        if args.benchmark:
            return run_benchmark(args)
        return run_full(args)
    except EXPECTED_ERRORS as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
