"""Full training loop: metrics aggregation, stages, early stopping, best selection, resume.

Small deterministic experiments on synthetic images with random-init EfficientNet-B0.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from test_dataset_loader import _write_images
from torch import nn
from torch.utils.data import DataLoader, Dataset

from doglens.config import TrainingConfig
from doglens.dataset import HorizontalFlip, ManifestDataset, make_loader, seed_everything
from doglens.experiment import (
    Experiment,
    StopRequested,
    evaluate_loader,
    is_better,
    load_run_checkpoint,
    train_one_epoch,
)
from doglens.model import BreedClassifier
from doglens.preprocessing import PreprocessingSpec, spec_from_weights
from doglens.training import TrainingError

CPU = torch.device("cpu")
_BASE = spec_from_weights()
SMALL_SPEC = PreprocessingSpec(72, 64, _BASE.interpolation, True, _BASE.mean, _BASE.std, "test")
IDENTITY = {
    "mapping_sha256": "m" * 64,
    "manifest_sha256": "f" * 64,
    "preprocessing": SMALL_SPEC.to_dict(),
    "pretrained_sha256": "p" * 64,
    "device": "cpu",
}


class Recording(Dataset):
    """Wraps a dataset and records which indices were read (to compare data order)."""

    def __init__(self, inner: Dataset) -> None:
        self.inner, self.reads = inner, []

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int):
        self.reads.append(int(i))
        return self.inner[i]


def _config(**overrides) -> dict:
    cfg = TrainingConfig().to_dict()
    cfg["stage_a"] = {"epochs": 2, "lr": 0.01}
    cfg["stage_b"] = {
        "epochs": 2,
        "lr": 0.001,
        "unfreeze_blocks": ["features.8"],
        "batchnorm_policy": "frozen",
    }
    cfg["early_stopping_patience"] = 3
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


@pytest.fixture(scope="module")
def images(tmp_path_factory) -> tuple[Path, object]:
    root = tmp_path_factory.mktemp("img")
    return root, _write_images(root, n=12)


def _experiment(images, run_dir: Path, cfg: dict, should_stop=lambda: False):
    root, df = images
    seed_everything(42)
    torch.manual_seed(0)
    model = BreedClassifier()
    train_ds = Recording(ManifestDataset(df, root, SMALL_SPEC, 0.1, augment=HorizontalFlip()))
    val_ds = ManifestDataset(df.iloc[:6], root, SMALL_SPEC, 0.1)
    run_dir.mkdir(parents=True, exist_ok=True)
    exp = Experiment(
        model=model,
        train_loader=make_loader(train_ds, 5, shuffle=True, seed=42),
        val_loader=make_loader(val_ds, 4, shuffle=False, seed=42),
        device=CPU,
        run_dir=run_dir,
        config=cfg,
        identity=dict(IDENTITY),
        log=lambda _m: None,
        should_stop=should_stop,
        log_every=0,
    )
    return exp, train_ds


def _stop_after(n_checks: int):
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > n_checks

    return should_stop


def _without_timing(state: dict) -> dict:
    clean = dict(state)
    clean["history"] = [
        {k: v for k, v in h.items() if k not in ("train_seconds", "val_seconds")}
        for h in state["history"]
    ]
    return clean


def _assert_same_state_dict(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def _assert_same_optimizer(a: dict, b: dict) -> None:
    assert a["param_groups"] == b["param_groups"]
    assert a["state"].keys() == b["state"].keys()
    for k in a["state"]:
        for name, value in a["state"][k].items():
            other = b["state"][k][name]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other


# --- aggregation and validation ------------------------------------------------------------------


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 64 * 64, 120)

    def forward(self, x):
        return self.fc(x.flatten(1))


def test_losses_are_weighted_by_samples_with_incomplete_last_batch(images) -> None:
    root, df = images
    ds = ManifestDataset(df, root, SMALL_SPEC, 0.1)  # 12 samples
    loader = DataLoader(ds, batch_size=5)  # batches of 5, 5, 2
    torch.manual_seed(0)
    model = _Tiny()
    xs = torch.stack([ds[i][0] for i in range(len(ds))])
    ys = torch.tensor([ds[i][1] for i in range(len(ds))])
    with torch.no_grad():
        expected = float(nn.functional.cross_entropy(model(xs), ys))  # per-sample mean
    val = evaluate_loader(model, loader, CPU)
    assert val["n"] == 12 and val["loss"] == pytest.approx(expected, rel=1e-6)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # no change: losses comparable
    train = train_one_epoch(model, loader, opt, CPU, lambda: False, lambda _m: None, 0)
    assert train["n"] == 12 and train["batches"] == 3
    assert train["loss"] == pytest.approx(expected, rel=1e-6)
    batch_means = [
        float(nn.functional.cross_entropy(model(xs[i : i + 5]), ys[i : i + 5]).detach())
        for i in (0, 5, 10)
    ]
    assert train["loss"] != pytest.approx(sum(batch_means) / 3, rel=1e-6)


def test_validation_changes_no_parameters_or_bn_stats(images) -> None:
    root, df = images
    model = BreedClassifier()
    model.set_trainable("finetune", ("features.8",), bn_policy="train_unfrozen")
    model.train()
    before = copy.deepcopy(model.state_dict())
    loader = make_loader(ManifestDataset(df, root, SMALL_SPEC, 0.1), 4, False, 0)
    evaluate_loader(model, loader, CPU)
    _assert_same_state_dict(before, model.state_dict())
    assert not any(m.training for m in model.modules())
    assert all(p.grad is None for p in model.parameters())


def test_is_better_rule() -> None:
    assert is_better({"top1": 0.5, "loss": 2.0}, None)
    assert is_better({"top1": 0.6, "loss": 9.0}, {"top1": 0.5, "loss": 1.0})
    assert is_better({"top1": 0.5, "loss": 1.0}, {"top1": 0.5, "loss": 2.0})
    assert not is_better({"top1": 0.5, "loss": 2.0}, {"top1": 0.5, "loss": 2.0})
    assert not is_better({"top1": 0.4, "loss": 0.1}, {"top1": 0.5, "loss": 2.0})


# --- stages, early stopping, best -----------------------------------------------------------------


def test_full_run_stages_files_and_history(images, tmp_path: Path) -> None:
    exp, _ = _experiment(images, tmp_path / "run", _config())
    state = exp.run()
    assert state["finished"]
    assert [(h["stage"], h["stage_epoch"]) for h in state["history"]] == [
        ("A", 1),
        ("A", 2),
        ("B", 1),
        ("B", 2),
    ]
    for name in ("last.pt", "best.pt", "best_stage_A.pt", "best_stage_B.pt", "history.json"):
        assert (tmp_path / "run" / name).is_file()
    best = load_run_checkpoint(tmp_path / "run" / "best.pt")
    gb = state["global_best"]
    assert best["state"]["global_best"] == gb
    winner = max(state["history"], key=lambda h: (h["val_top1"], -h["val_loss"]))
    assert (gb["stage"], gb["stage_epoch"]) == (winner["stage"], winner["stage_epoch"])
    assert exp.model.mode == "finetune" and exp.model.trainable_blocks == ("features.8",)


def test_early_stopping_transition_and_head_kept_as_global_best(images, tmp_path: Path) -> None:
    # lr=0: validation never improves after the first epoch of each stage.
    cfg = _config(
        stage_a={"epochs": 3, "lr": 0.0},
        stage_b={
            "epochs": 3,
            "lr": 0.0,
            "unfreeze_blocks": ["features.8"],
            "batchnorm_policy": "frozen",
        },
        early_stopping_patience=1,
    )
    exp, _ = _experiment(images, tmp_path / "run", cfg)
    state = exp.run()
    stages = [(h["stage"], h["stage_epoch"], h["patience"]) for h in state["history"]]
    assert stages == [("A", 1, 0), ("A", 2, 1), ("B", 1, 0), ("B", 2, 1)]
    assert state["global_best"]["stage"] == "A"  # equal metrics in B are not "better"
    assert state["stage_bests"]["B"]["stage_epoch"] == 1
    best = load_run_checkpoint(tmp_path / "run" / "best.pt")["model_state"]
    head_best = load_run_checkpoint(tmp_path / "run" / "best_stage_A.pt")["model_state"]
    _assert_same_state_dict(best, head_best)


def test_stage_b_starts_from_best_stage_a_weights(images, tmp_path: Path) -> None:
    cfg = _config(stage_b={"epochs": 1, "lr": 0.0, "unfreeze_blocks": ["features.8"],
                           "batchnorm_policy": "frozen"})  # fmt: skip
    exp, _ = _experiment(images, tmp_path / "run", cfg)
    exp.run()
    a_best = load_run_checkpoint(tmp_path / "run" / "best_stage_A.pt")["model_state"]
    # lr=0 in stage B and frozen BN: weights after B equal the restored stage-A best.
    _assert_same_state_dict(a_best, exp.model.state_dict())


# --- resume -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stop_checks",
    [
        0,  # stopped before the first batch of stage A epoch 1 (epoch-0 checkpoint)
        4,  # mid stage A epoch 2 (3 batches per epoch)
        6,  # exactly at the first batch of stage B (stage boundary)
        10,  # mid stage B epoch 2
    ],
)
def test_resume_matches_uninterrupted_run(images, tmp_path: Path, stop_checks: int) -> None:
    cfg = _config()
    full, full_ds = _experiment(images, tmp_path / "full", cfg)
    full_state = full.run()

    part, part_ds = _experiment(images, tmp_path / "part", cfg, _stop_after(stop_checks))
    with pytest.raises(StopRequested):
        part.run()
    last = load_run_checkpoint(tmp_path / "part" / "last.pt")
    completed = last["state"]["global_epochs_completed"]
    reads_before = part_ds.reads[: completed * 12]  # reads of completed epochs only

    resumed, resumed_ds = _experiment(images, tmp_path / "part", cfg)
    resumed.resume_from(tmp_path / "part" / "last.pt")
    resumed_state = resumed.run()

    # Data order: completed epochs + post-resume epochs == uninterrupted order.
    assert reads_before + resumed_ds.reads == full_ds.reads
    # Counters, history (all metrics bit-identical) and best selection are identical;
    # only wall-clock durations naturally differ.
    assert _without_timing(resumed_state) == _without_timing(full_state)
    # Bit-identical weights and optimizer state.
    _assert_same_state_dict(full.model.state_dict(), resumed.model.state_dict())
    _assert_same_optimizer(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    _assert_same_state_dict(
        load_run_checkpoint(tmp_path / "full" / "best.pt")["model_state"],
        load_run_checkpoint(tmp_path / "part" / "best.pt")["model_state"],
    )


def test_resume_of_finished_run_is_noop(images, tmp_path: Path) -> None:
    exp, _ = _experiment(images, tmp_path / "run", _config())
    final = exp.run()
    again, ds = _experiment(images, tmp_path / "run", _config())
    again.resume_from(tmp_path / "run" / "last.pt")
    assert _without_timing(again.run()) == _without_timing(final) and ds.reads == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"identity": {"mapping_sha256": "x" * 64}}, "mapping_sha256"),
        ({"identity": {"manifest_sha256": "x" * 64}}, "manifest_sha256"),
        ({"identity": {"preprocessing": {"other": 1}}}, "preprocessing"),
        ({"identity": {"device": "cuda"}}, "device"),
        ({"config": {"stage_a": {"epochs": 2, "lr": 0.5}}}, "stage_a"),
        ({"config": {"seed": 7}}, "seed"),
    ],
)
def test_incompatible_resume_is_rejected(images, tmp_path: Path, change, message) -> None:
    exp, _ = _experiment(images, tmp_path / "run", _config(), _stop_after(4))
    with pytest.raises(StopRequested):
        exp.run()
    cfg = _config(**change.get("config", {}))
    other, _ = _experiment(images, tmp_path / "run", cfg)
    other.identity.update(change.get("identity", {}))
    with pytest.raises(TrainingError, match=message):
        other.resume_from(tmp_path / "run" / "last.pt")


def test_resume_rejects_foreign_files(images, tmp_path: Path) -> None:
    exp, _ = _experiment(images, tmp_path / "run", _config())
    bogus = tmp_path / "bogus.pt"
    torch.save({"weights": torch.zeros(1)}, bogus)
    with pytest.raises(TrainingError, match="not a DogLens run checkpoint"):
        exp.resume_from(bogus)
    with pytest.raises(TrainingError, match="not found"):
        exp.resume_from(tmp_path / "absent.pt")


def test_experiment_normalises_identity_types(images, tmp_path: Path) -> None:
    root, df = images
    ds = ManifestDataset(df, root, SMALL_SPEC, 0.1)
    exp = Experiment(
        model=BreedClassifier(),
        train_loader=make_loader(ds, 4, True, 0),
        val_loader=make_loader(ds, 4, False, 0),
        device=CPU,
        run_dir=tmp_path,
        config=_config(),
        identity={**IDENTITY, "torch": torch.__version__},
        log=lambda _m: None,
    )
    assert type(exp.identity["torch"]) is str
    exp.run()  # includes the A->B transition, which reloads best_stage_A.pt
    for name in ("last.pt", "best.pt", "best_stage_A.pt", "best_stage_B.pt"):
        torch.load(tmp_path / name, map_location="cpu", weights_only=True)


def test_legacy_checkpoint_with_torchversion_is_readable(images, tmp_path: Path) -> None:
    exp, _ = _experiment(images, tmp_path / "run", _config(), _stop_after(4))
    with pytest.raises(StopRequested):
        exp.run()
    payload = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=True)
    payload["identity"]["torch"] = torch.__version__  # as written by the first exp1 invocation
    torch.save(payload, tmp_path / "legacy.pt")
    with pytest.raises(Exception, match="TorchVersion"):
        torch.load(tmp_path / "legacy.pt", map_location="cpu", weights_only=True)
    assert load_run_checkpoint(tmp_path / "legacy.pt")["identity"]["torch"] == torch.__version__
