# ResNet-50 + GSF backbone for the 16 frame SS-v1
#
# Reference:
# https://github.com/swathikirans/GSF/blob/9d4d80e33dc81f8de882b583b7062606411cb696/backbones/resnetGSFModels.py

import torch
from torch import nn
from .gsf import GSF


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = conv1x1(channels, channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # Match the 16 frame pretrained GSF backbone
        self.gsf = GSF(fPlane=channels * self.expansion, num_segments=16, gsf_ch_ratio=25)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.gsf(self.conv3(out)))

        if self.downsample is not None:
            identity = self.downsample(x)
        
        result = self.relu(out + identity)
        return result


class ResNet50GSF(nn.Module):

    def __init__(self):
        super().__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # 3, 4, 6, 3 are counts of residual blocks in the four stages
        self.layer1 = self.make_layer(64, blocks=3)
        self.layer2 = self.make_layer(128, blocks=4, stride=2)
        self.layer3 = self.make_layer(256, blocks=6, stride=2)
        self.layer4 = self.make_layer(512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # My RGBModel replaces this with dropout
        self.fc = nn.Identity()

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def make_layer(self, channels, blocks, stride=1):
        out_channels = channels * Bottleneck.expansion
        downsample = None

        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(conv1x1(self.inplanes, out_channels, stride), nn.BatchNorm2d(out_channels))
        layers = [Bottleneck(self.inplanes, channels, stride, downsample)]
        self.inplanes = out_channels

        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, channels))

        result = nn.Sequential(*layers)
        
        return result

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))
