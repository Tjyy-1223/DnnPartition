import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from collections import abc, OrderedDict
import torch.nn.functional as F
import time
from torchvision import models

# models.resnet18()

class SentenceIterator(abc.Iterator):
    def __init__(self,features, layer1, layer2, layer3, layer4, classifier):
        self.features = features
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.classifier = classifier

        self._index = 0

        self.len1 = len(features)
        self.len2 = len(layer1)
        self.len3 = len(layer2)
        self.len4 = len(layer3)
        self.len5 = len(layer4)
        self.len6 = len(classifier)

    def __next__(self):
        try:
            if self._index < self.len1:
                layer = self.features[self._index]

            elif self._index < (self.len1 + self.len2):
                len = self.len1
                layer = self.layer1[self._index - len]

            elif self._index < (self.len1 + self.len2 + self.len3):
                len = self.len1 + self.len2
                layer = self.layer2[self._index - len]

            elif self._index < (self.len1 + self.len2 + self.len3 + self.len4):
                len = self.len1 + self.len2 + self.len3
                layer = self.layer3[self._index - len]

            elif self._index < (self.len1 + self.len2 + self.len3 + self.len4 + self.len5):
                len = self.len1 + self.len2 + self.len3 + self.len4
                layer = self.layer4[self._index - len]

            else:
                len = self.len1 + self.len2 + self.len3 + self.len4 + self.len5
                layer = self.classifier[self._index - len]

        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer


def conv3x3(in_planes: int, out_planes: int, stride: int = 1,padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=padding,bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    expansion: int = 1
    """基本的resnet残差块"""
    def __init__(self,inplanes: int,planes: int,stride: int = 1,
            downsample: Optional[nn.Module] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        """1x1卷积核"""
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[BasicBlock],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.classifier = nn.Sequential(
            self.avgpool,
            self.flatten,
            self.fc
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    """ 创建一个resnet残差块 """

    def _make_layer(self,block: Type[BasicBlock],planes: int,blocks: int,stride: int = 1,) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample,norm_layer)
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,planes,norm_layer=norm_layer)
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.features(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def __len__(self):
        return len(self.features) + len(self.layer1) + len(self.layer2) + len(self.layer3) + len(self.layer4) + len(self.classifier)

    def __iter__(self):
        return SentenceIterator(self.features, self.layer1, self.layer2, self.layer3,self.layer4, self.classifier)


# resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])

def resnet18(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)