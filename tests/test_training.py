"""Training primitives and the smoke-test routine on tiny synthetic data (random init)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from test_dataset_loader import SPEC, _write_images
from torch import nn

from doglens.model import BreedClassifier
from doglens.training import (
    SmokeSettings,
    TrainingError,
    build_optimizer,
    load_checkpoint,
    predict_logits,
    run_smoke_test,
    save_checkpoint,
    select_device,
    smoke_subset,
    train_step,
)

REPO = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


def test_select_device() -> None:
    assert select_device("cpu") == CPU
    assert select_device("auto").type in {"cpu", "cuda", "mps"}
    if not torch.cuda.is_available():
        with pytest.raises(TrainingError, match="CUDA"):
            select_device("cuda")
    with pytest.raises(TrainingError):
        select_device("tpu")


def test_train_step_finite_and_nonfinite() -> None:
    model = BreedClassifier()
    model.set_trainable("head")
    opt = build_optimizer(model, 1e-3, 1e-4)
    x, y = torch.randn(2, 3, 64, 64), torch.tensor([1, 2])
    assert train_step(model, x, y, opt, nn.CrossEntropyLoss(), CPU) > 0

    class NanLoss(nn.Module):
        def forward(self, logits, labels):
            return logits.sum() * float("nan")

    with pytest.raises(TrainingError, match="non-finite"):
        train_step(model, x, y, opt, NanLoss(), CPU)


def test_optimizer_needs_trainable_parameters() -> None:
    model = BreedClassifier()
    for p in model.parameters():
        p.requires_grad = False
    with pytest.raises(TrainingError):
        build_optimizer(model, 1e-3, 0.0)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = BreedClassifier()
    opt = build_optimizer(model, 1e-3, 0.0)
    x = torch.randn(2, 3, 64, 64)
    path = tmp_path / "ck.pt"
    save_checkpoint(
        path, model, opt, epoch=3, config={"a": 1}, mapping_sha256="m" * 64, preprocessing={}
    )
    payload = load_checkpoint(path, "m" * 64)
    assert payload["epoch"] == 3 and payload["config"] == {"a": 1}
    fresh = BreedClassifier()
    fresh.load_state_dict(payload["model_state"], strict=True)
    assert torch.equal(predict_logits(model, x, CPU), predict_logits(fresh, x, CPU))
    with pytest.raises(TrainingError, match="different class mapping"):
        load_checkpoint(path, "x" * 64)
    with pytest.raises(TrainingError):
        load_checkpoint(tmp_path / "absent.pt")


def test_smoke_subset_is_deterministic(tmp_path: Path) -> None:
    df = _write_images(tmp_path).sample(frac=1.0, random_state=3)
    subset = smoke_subset(df, 2)
    assert subset.groupby("class_id").size().tolist() == [2, 2, 2]
    assert subset["image_id"].tolist() == smoke_subset(df.iloc[::-1], 2)["image_id"].tolist()


def test_run_smoke_test_on_tiny_data(tmp_path: Path) -> None:
    df = _write_images(tmp_path / "img", n=12)
    settings = SmokeSettings(
        per_class=4, train_batches=2, batch_size=4, overfit_batch_size=4, overfit_steps=8,
        overfit_lr=0.05, overfit_max_loss_ratio=0.9, finetune_steps=1, head_lr=1e-3,
        fine_tune_lr=1e-4, weight_decay=1e-4, unfreeze_blocks=("features.8",),
        bn_policy="frozen", context=0.1, seed=42, horizontal_flip=True,
    )  # fmt: skip
    result = run_smoke_test(
        BreedClassifier(),
        df,
        tmp_path / "img",
        SPEC,
        settings,
        CPU,
        tmp_path / "ck.pt",
        {"config": {}, "mapping_sha256": "0" * 64},
        log=lambda _m: None,
    )
    checks = result["checks"]
    assert checks["output_shape"] == [4, 120]
    assert checks["head_bn_stats_unchanged"] and checks["finetune_bn_stats_unchanged"]
    assert checks["reload_predictions_equal"] and checks["all_losses_finite"]
    assert checks["overfit_loss_ratio"] < 0.9
    assert (tmp_path / "ck.pt").is_file()


def _train_cli():
    spec = importlib.util.spec_from_file_location("train_cli", REPO / "scripts" / "train.py")
    cli = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["train_cli"] = cli  # dataclasses in the script resolve their module here
    spec.loader.exec_module(cli)
    return cli


def test_train_cli_arguments(capsys) -> None:
    cli = _train_cli()
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
    for bad in (
        ["--smoke-test", "--benchmark"],
        ["--smoke-test", "--resume", "x.pt"],
        ["--max-minutes", "0"],
        ["--run-id", "a/b"],
    ):
        with pytest.raises(SystemExit) as info:
            cli.main(bad)
        assert info.value.code == 2
    assert cli.main(["--smoke-test", "--config", str(REPO / "absent.yaml")]) == 1
    assert cli.main(["--config", str(REPO / "absent.yaml")]) == 1
    assert "config file not found" in capsys.readouterr().err
