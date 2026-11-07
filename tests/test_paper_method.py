"""The CAIM block, contrastive loss and protocol, checked against Sec. 3-4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from caim.data.celeba_lrhr import CelebALRHR
from caim.eval import eval_cross_res
from caim.losses import contrastive_pairs
from caim.models.caim_net import CAIM, CAIMFaceNet

ROOT = Path(__file__).resolve().parents[1]


def aim(block: CAIM, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    """Eq. (9): scale and shift the instance-normalised map with values from the style."""
    normalised = (x - x.mean(dim=(2, 3), keepdim=True)) / x.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
    gamma, beta = block.fc(style).chunk(2, dim=1)
    return normalised * (1 + gamma[..., None, None]) + beta[..., None, None]


def test_source_modality_is_left_unchanged():
    """Eq. (10) with g = 0."""
    x = torch.randn(2, 4, 6, 6)
    assert torch.equal(CAIM(4, 8)(x, torch.randn(2, 8), enable=False), x)


def test_target_modality_adds_the_full_modulation():
    """Eq. (10) with g = 1: CAIM(F, 1) = AIM(F) + F."""
    torch.manual_seed(0)
    block = CAIM(4, 8)
    x, style = torch.randn(2, 4, 6, 6), torch.randn(2, 8)
    assert torch.allclose(block(x, style, enable=True), aim(block, x, style) + x, atol=1e-5)


def test_gate_is_a_signal_not_a_learned_parameter():
    """Sec. 3: g is 1 for the target modality and 0 for the source."""
    assert "gate" not in dict(CAIM(4, 8).named_parameters())


def test_contrastive_loss_follows_eq_2():
    torch.manual_seed(1)
    source, target = F.normalize(torch.randn(6, 8), dim=1), F.normalize(torch.randn(6, 8), dim=1)
    genuine = (source - target).norm(dim=1)
    impostor = (source - target.roll(1, 0)).norm(dim=1)
    expected = 0.5 * (genuine.pow(2) + F.relu(2.0 - impostor).pow(2)).mean()
    assert contrastive_pairs(source, target, margin=2.0).item() == pytest.approx(expected.item(), abs=1e-5)


def test_network_output_is_a_unit_embedding():
    model = CAIMFaceNet(emb_dim=16, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 64, 64) * 2 - 1, target_modality=True)
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize("key", ["margin", "lr"])
def test_training_settings_match_the_paper(default_cfg, targets, key):
    assert default_cfg["train"][key] == targets["training"][key]


def test_training_uses_adam():
    assert "optim.Adam" in (ROOT / "caim" / "train.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def celeba(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("celeba")
    images = root / "img_align_celeba"
    images.mkdir()
    rng = np.random.RandomState(0)
    lines = []
    for subject in range(1, 21):
        for k in range(3):
            name = f"{subject:03d}{k}.png"
            Image.fromarray(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(images / name)
            lines.append(f"{name} {subject}")
    (root / "identity_CelebA.txt").write_text("\n".join(lines) + "\n")
    return root


class RecordingEmbedder:
    def __init__(self):
        self.probes = []

    def embed(self, x: torch.Tensor, target_modality: bool = False) -> torch.Tensor:
        if target_modality:
            self.probes.append(x[0].clone())
        return F.normalize(x.flatten(1)[:, :16], dim=1)


def test_probes_are_not_degraded_copies_of_gallery_images(celeba):
    """A probe made from the gallery image itself is matched to its own source."""
    cfg = {
        "data": {"celeba_root": str(celeba), "max_images": None},
        "model": {"image_size": 32, "lr_scale": 4},
        "train": {"seed": 42},
        "eval": {"max_ids": None, "batch_size": 64, "probes_per_id": 2},
    }
    model = RecordingEmbedder()
    eval_cross_res(model, cfg, torch.device("cpu"))
    ds = CelebALRHR(celeba, image_size=32, lr_scale=4, seed=42, split="test")
    gallery_lr = [ds[indices[0]][0] for indices in ds.by_class.values()]
    assert sum(any(torch.equal(p, g) for g in gallery_lr) for p in model.probes) == 0
