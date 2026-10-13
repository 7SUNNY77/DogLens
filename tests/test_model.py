"""Model architecture, freezing modes and BatchNorm policy (random init, no downloads)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torchvision.models import efficientnet_b0

from doglens.fileutil import sha256_file
from doglens.model import (
    BreedClassifier,
    PretrainedWeightsError,
    count_trainable_parameters,
)

BLOCKS = ("features.6", "features.7", "features.8")


def _bn_layers(module: nn.Module) -> list[nn.Module]:
    return [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def _bn_stats(module: nn.Module) -> list[torch.Tensor]:
    return [
        t.detach().clone()
        for m in _bn_layers(module)
        for t in (m.running_mean, m.running_var, m.num_batches_tracked)
    ]


def _step(model: BreedClassifier, lr: float = 0.1) -> None:
    torch.manual_seed(0)
    x, y = torch.randn(4, 3, 64, 64), torch.tensor([0, 1, 2, 3])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()


def test_forward_shape() -> None:
    model = BreedClassifier().eval()
    with torch.inference_mode():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 120)
    assert torch.isfinite(out).all()


def test_head_mode_trainable_parameters_and_bn_eval() -> None:
    model = BreedClassifier()
    model.set_trainable("head")
    assert count_trainable_parameters(model) == 1280 * 120 + 120 == 153_720
    assert all(not p.requires_grad for p in model.net.features.parameters())
    assert all(p.requires_grad for p in model.net.classifier.parameters())
    model.train()
    assert all(not m.training for m in model.net.features.modules())  # incl. stochastic depth
    assert model.net.classifier.training


def test_head_mode_step_keeps_backbone_and_bn_stats() -> None:
    model = BreedClassifier()
    model.set_trainable("head")
    stats = _bn_stats(model.net.features)
    weights = [p.detach().clone() for p in model.net.features.parameters()]
    head = model.net.classifier[1].weight.detach().clone()
    _step(model)
    assert all(torch.equal(a, b) for a, b in zip(stats, _bn_stats(model.net.features), strict=True))
    assert all(
        torch.equal(a, b) for a, b in zip(weights, model.net.features.parameters(), strict=True)
    )
    assert not torch.equal(head, model.net.classifier[1].weight)


def test_finetune_mode_counts_and_frozen_bn_policy() -> None:
    model = BreedClassifier()
    model.set_trainable("finetune", BLOCKS, bn_policy="frozen")
    expected = 153_720 + sum(
        p.numel() for b in BLOCKS for p in model.net.get_submodule(b).parameters()
    )
    assert count_trainable_parameters(model) == expected == 3_309_460
    for i, block in enumerate(model.net.features):
        assert all(p.requires_grad == (f"features.{i}" in BLOCKS) for p in block.parameters())
    model.train()
    assert all(not m.training for m in _bn_layers(model))
    assert model.net.get_submodule("features.7").training  # non-BN layers of unfrozen blocks train
    assert not model.net.get_submodule("features.5").training
    stats = _bn_stats(model)
    frozen = [p.detach().clone() for p in model.net.get_submodule("features.0").parameters()]
    tuned = [p.detach().clone() for p in model.net.get_submodule("features.8").parameters()]
    _step(model)
    assert all(torch.equal(a, b) for a, b in zip(stats, _bn_stats(model), strict=True))
    assert all(
        torch.equal(a, b)
        for a, b in zip(frozen, model.net.get_submodule("features.0").parameters(), strict=True)
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(tuned, model.net.get_submodule("features.8").parameters(), strict=True)
    )


def test_train_unfrozen_bn_policy_updates_only_unfrozen_stats() -> None:
    model = BreedClassifier()
    model.set_trainable("finetune", ("features.8",), bn_policy="train_unfrozen")
    model.train()
    assert all(m.training for m in _bn_layers(model.net.get_submodule("features.8")))
    assert all(not m.training for m in _bn_layers(model.net.get_submodule("features.7")))
    frozen_stats = _bn_stats(model.net.get_submodule("features.7"))
    tuned_stats = _bn_stats(model.net.get_submodule("features.8"))
    _step(model)
    after_frozen = _bn_stats(model.net.get_submodule("features.7"))
    assert all(torch.equal(a, b) for a, b in zip(frozen_stats, after_frozen, strict=True))
    after_tuned = _bn_stats(model.net.get_submodule("features.8"))
    assert any(not torch.equal(a, b) for a, b in zip(tuned_stats, after_tuned, strict=True))


def test_eval_and_train_toggle_respects_mode() -> None:
    model = BreedClassifier()
    model.set_trainable("head")
    model.eval()
    assert not any(m.training for m in model.modules())
    model.train()
    model.train()  # repeated calls must not re-activate frozen layers
    assert all(not m.training for m in model.net.features.modules())


def test_invalid_modes_and_blocks() -> None:
    model = BreedClassifier()
    with pytest.raises(ValueError):
        model.set_trainable("partial")
    with pytest.raises(ValueError):
        model.set_trainable("finetune", ("features.9",))
    with pytest.raises(ValueError):
        model.set_trainable("finetune", ())
    with pytest.raises(ValueError):
        model.set_trainable("head", bn_policy="sometimes")


def test_load_imagenet_backbone_from_local_file(tmp_path: Path) -> None:
    # A locally generated (random) ImageNet-shaped state_dict stands in for the real file.
    torch.manual_seed(0)
    source = efficientnet_b0(weights=None)
    path = tmp_path / "fake_imagenet.pth"
    torch.save(source.state_dict(), path)
    model = BreedClassifier()
    model.load_imagenet_backbone(path, sha256_file(path))
    assert torch.equal(model.net.features[0][0].weight, source.features[0][0].weight)
    assert model.net.classifier[1].out_features == 120
    with pytest.raises(PretrainedWeightsError, match="SHA-256"):
        model.load_imagenet_backbone(path, "0" * 64)
    with pytest.raises(PretrainedWeightsError, match="not found"):
        model.load_imagenet_backbone(tmp_path / "absent.pth", "0" * 64)
    bad = tmp_path / "bad.pth"
    torch.save({"something": torch.zeros(1)}, bad)
    with pytest.raises(PretrainedWeightsError, match="mismatch"):
        model.load_imagenet_backbone(bad, sha256_file(bad))
