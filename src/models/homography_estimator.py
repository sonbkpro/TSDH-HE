from __future__ import annotations
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from src.geometry.dlt import offsets_to_homography, solve_homography_dlt


RESNET34_URL = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet34HomographyEstimator(nn.Module):
    """ResNet-34 style h(.) from Table 1(c), predicting 4 corner offsets = 8 values."""
    def __init__(self, in_ch: int = 2, pretrained_backbone: bool = False):
        super().__init__()
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
        self.fc = nn.Linear(512, 8)
        if pretrained_backbone:
            self.load_imagenet_backbone()

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1))
        return nn.Sequential(*layers)

    def load_imagenet_backbone(self):
        pretrained = model_zoo.load_url(RESNET34_URL)
        own = self.state_dict()
        pretrained = {k: v for k, v in pretrained.items() if k in own and k not in {'conv1.weight', 'fc.weight', 'fc.bias'}}
        own.update(pretrained)
        self.load_state_dict(own)

    def forward(self, x, h4p: torch.Tensor | None = None):
        _, _, h, w = x.shape
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(self.avgpool(out), 1)
        offsets = self.fc(out)
        if h4p is None:
            H = offsets_to_homography(offsets, h, w)
        else:
            src = h4p.reshape(-1, 4, 2).to(device=offsets.device, dtype=offsets.dtype)
            dst = src + offsets.reshape(-1, 4, 2)
            H = solve_homography_dlt(src, dst)
        return H, offsets
