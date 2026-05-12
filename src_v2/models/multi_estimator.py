from __future__ import annotations
import torch
import torch.nn as nn
from src.models.homography_estimator import BasicBlock
from src.geometry.dlt import solve_homography_dlt, offsets_to_homography


class ResNet34MultiHomographyEstimator(nn.Module):
    """ResNet-34 style estimator that predicts K sets of 4-corner offsets.

    The returned H matrices follow the same convention as V1 Oneline:
    they are used by transform_official_patch as destination/target -> source
    sampling matrices. For evaluation, invert them to map source points to target.
    """
    def __init__(self, in_ch: int = 2, num_hypotheses: int = 4):
        super().__init__()
        self.num_hypotheses = int(num_hypotheses)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 8 * self.num_hypotheses)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, h4p: torch.Tensor | None = None):
        b, _, h, w = x.shape
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(self.avgpool(out), 1)
        offsets = self.fc(out).reshape(b, self.num_hypotheses, 8)

        if h4p is None:
            flat_H = []
            for k in range(self.num_hypotheses):
                flat_H.append(offsets_to_homography(offsets[:, k], h, w))
            Hs = torch.stack(flat_H, dim=1)
        else:
            src = h4p.reshape(b, 4, 2).to(device=offsets.device, dtype=offsets.dtype)
            H_list = []
            for k in range(self.num_hypotheses):
                dst = src + offsets[:, k].reshape(b, 4, 2)
                H_list.append(solve_homography_dlt(src, dst))
            Hs = torch.stack(H_list, dim=1)
        return Hs, offsets
