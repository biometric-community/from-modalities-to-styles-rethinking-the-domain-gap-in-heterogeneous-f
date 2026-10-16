"""ResNet18 + CAIM blocks (AdaIN-like gated residual modulation for target modality)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CAIM(nn.Module):
    """Conditional Adaptive Instance Modulation (paper Sec. III; compact form).

    y = x + g * (AdaIN(x; style) - x), with g learned; for source modality g→0.
    """

    def __init__(self, channels: int, style_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(style_dim, channels * 2)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, style: torch.Tensor, enable: bool) -> torch.Tensor:
        if not enable:
            return x
        # instance norm
        mu = x.mean(dim=(2, 3), keepdim=True)
        sig = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        x_n = (x - mu) / sig
        gamma_beta = self.fc(style)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        adain = x_n * (1 + gamma) + beta
        g = torch.sigmoid(self.gate)
        return x + g * (adain - x)


class CAIMFaceNet(nn.Module):
    def __init__(self, emb_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        # freeze early layers lightly — paper freezes pretrained FR; we freeze stem+layer1
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        for p in list(self.stem.parameters()) + list(self.layer1.parameters()):
            p.requires_grad_(False)
        self.style_enc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        self.caim2 = CAIM(128, 128)
        self.caim3 = CAIM(256, 128)
        self.caim4 = CAIM(512, 128)
        self.fc = nn.Linear(512, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor, target_modality: bool = False) -> torch.Tensor:
        # x in [-1,1]
        x = (x + 1.0) * 0.5
        h = self.stem(x)
        style = self.style_enc(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.caim2(h, style, enable=target_modality)
        h = self.layer3(h)
        h = self.caim3(h, style, enable=target_modality)
        h = self.layer4(h)
        h = self.caim4(h, style, enable=target_modality)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        h = self.bn(self.fc(h))
        return F.normalize(h, p=2, dim=1)

    def embed(self, x: torch.Tensor, target_modality: bool = False) -> torch.Tensor:
        return self.forward(x, target_modality=target_modality)
