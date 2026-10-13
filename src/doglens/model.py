"""EfficientNet-B0 with a new 120-class head and explicit freezing / BatchNorm policy.

The architecture is always created with ``weights=None``; ImageNet initialisation comes only
from a verified local file (training), and the app later loads its own trained state_dict.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import efficientnet_b0

from doglens.config import NUM_CLASSES
from doglens.fileutil import sha256_file

ARCHITECTURE = "efficientnet_b0"
NUM_FEATURE_BLOCKS = 9  # torchvision efficientnet_b0.features has children 0..8
HEAD_KEYS = ("net.classifier.1.weight", "net.classifier.1.bias")
MODES = ("head", "finetune", "full")
BN_POLICIES = ("frozen", "train_unfrozen")


class PretrainedWeightsError(RuntimeError):
    """Local pretrained weights are missing or do not match the expected checksum."""


def load_pretrained_state_dict(path: str | Path, expected_sha256: str) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise PretrainedWeightsError(
            f"pretrained weights not found: {path}. See data/weights/SOURCE.json for the "
            "official torchvision URL; they are never downloaded implicitly."
        )
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise PretrainedWeightsError(f"SHA-256 mismatch for {path}: {actual}")
    return torch.load(path, map_location="cpu", weights_only=True)


class BreedClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.net = efficientnet_b0(weights=None)
        head = self.net.classifier[1]
        assert isinstance(head, nn.Linear)
        self.net.classifier[1] = nn.Linear(head.in_features, num_classes)
        self.num_classes = num_classes
        self.mode = "full"
        self.trainable_blocks: tuple[str, ...] = tuple(
            f"features.{i}" for i in range(NUM_FEATURE_BLOCKS)
        )
        self.bn_policy = "train_unfrozen"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_imagenet_backbone(self, path: str | Path, expected_sha256: str) -> None:
        """Load ImageNet weights for everything except the new classifier."""
        state = load_pretrained_state_dict(path, expected_sha256)
        state = {f"net.{k}": v for k, v in state.items() if not k.startswith("classifier.1.")}
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != set(HEAD_KEYS) or result.unexpected_keys:
            raise PretrainedWeightsError(
                f"unexpected key mismatch: missing={result.missing_keys} "
                f"unexpected={result.unexpected_keys}"
            )

    def set_trainable(
        self, mode: str, unfreeze_blocks: tuple[str, ...] = (), bn_policy: str = "frozen"
    ) -> None:
        """head: classifier only; finetune: classifier + listed feature blocks; full: all."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if bn_policy not in BN_POLICIES:
            raise ValueError(f"bn_policy must be one of {BN_POLICIES}")
        valid = {f"features.{i}" for i in range(NUM_FEATURE_BLOCKS)}
        unknown = set(unfreeze_blocks) - valid
        if unknown:
            raise ValueError(f"unknown feature blocks: {sorted(unknown)}")
        if mode == "head":
            blocks: tuple[str, ...] = ()
        elif mode == "finetune":
            if not unfreeze_blocks:
                raise ValueError("finetune mode needs unfreeze_blocks")
            blocks = tuple(sorted(set(unfreeze_blocks), key=lambda b: int(b.split(".")[1])))
        else:
            blocks = tuple(sorted(valid, key=lambda b: int(b.split(".")[1])))
        for i, block in enumerate(self.net.features):
            trainable = f"features.{i}" in blocks
            for p in block.parameters():
                p.requires_grad = trainable
        for p in self.net.classifier.parameters():
            p.requires_grad = True
        self.mode, self.trainable_blocks, self.bn_policy = mode, blocks, bn_policy
        self.train(self.training)

    def train(self, mode: bool = True) -> BreedClassifier:
        """Like nn.Module.train, but frozen blocks (and BN under the 'frozen' policy) stay eval.

        Frozen blocks are fully in eval mode: BatchNorm uses and keeps its running statistics
        and stochastic depth is disabled, so ``model.train()`` cannot silently re-activate them.
        """
        super().train(mode)
        if mode:
            for i, block in enumerate(self.net.features):
                if f"features.{i}" not in self.trainable_blocks:
                    block.eval()
                elif self.bn_policy == "frozen":
                    for m in block.modules():
                        if isinstance(m, nn.modules.batchnorm._BatchNorm):
                            m.eval()
            if not self.trainable_blocks:
                # Container flag has no compute effect; keep it consistent with its children.
                self.net.features.training = False
        return self


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
