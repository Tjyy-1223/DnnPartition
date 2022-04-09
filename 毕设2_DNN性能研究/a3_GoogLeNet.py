import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List, Callable, Any
from collections import abc, OrderedDict
import torch.nn.functional as F
import time
from torchvision import models

models.GoogLeNet()

class SentenceIterator(abc.Iterator):
    def __init__(self,features, inception3,inception4,inception5, classifier):
        self.features = features
        self.inception3 = inception3
        self.inception4 = inception4
        self.inception5 = inception5
        self.classifier = classifier

        self._index = 0

        self.len1 = len(features)
        self.len2 = len(inception3)
        self.len3 = len(inception4)
        self.len4 = len(inception5)
        self.len5 = len(classifier)

    def __next__(self):
        try:
            if self._index < self.len1:
                layer = self.features[self._index]
            elif self._index < (self.len1 + self.len2):
                len = self.len1
                layer = self.inception3[self._index - len]
            elif self._index < (self.len1 + self.len2 + self.len3):
                len = self.len1 + self.len2
                layer = self.inception4[self._index - len]
            elif self._index < (self.len1 + self.len2 + self.len3 + self.len4):
                len = self.len1 + self.len2 + self.len3
                layer = self.inception5[self._index - len]
            else:
                len = self.len1 + self.len2 + self.len3 + self.len4
                layer = self.classifier[self._index - len]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer

class GoogLeNet(nn.Module):
    __constants__ = ["aux_logits", "transform_input"]

    def __init__(
        self,
        num_classes: int = 1000,
        # 是否运行中查看inception的结果
        aux_logits: bool = False,
        # 是否对输入进行转换
        transform_input: bool = False,
        # 是否对模型进行初始化
        init_weights: Optional[bool] = False,
        blocks: Optional[List[Callable[..., nn.Module]]] = None,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
    ) -> None:
        super().__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        # true or false
        self.aux_logits = aux_logits
        self.transform_input = transform_input


        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)
        self.features = nn.Sequential(
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.conv3,
            self.maxpool2
        )


        # inception 3
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2)
        self.inception3 = nn.Sequential(
            self.inception3a,
            self.inception3b,
            self.maxpool3,
        )

        # inception 4
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        self.inception4 = nn.Sequential(
            self.inception4a,
            self.inception4b,
            self.inception4c,
            self.inception4d,
            self.inception4e,
            self.maxpool4
        )

        # inception 5
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        self.inception5 = nn.Sequential(
            self.inception5a,
            self.inception5b,
        )

        # aux_inception
        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(528, num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)
        self.classifier = nn.Sequential(
            self.avgpool,
            nn.Flatten(),
            self.dropout,
            self.fc
        )


        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        self.len1 = len(self.features)
        self.len2 = len(self.inception3)
        self.len3 = len(self.inception4)
        self.len4 = len(self.inception5)
        self.len5 = len(self.classifier)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self,x):
        x = self.features(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.classifier(x)
        return x

    def forward2(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1

    def __len__(self):
        return len(self.features) + len(self.inception3) + len(self.inception4) + len(self.inception5) + len(self.classifier)

    def __iter__(self):
        return SentenceIterator(self.features, self.inception3,self.inception4,self.inception5, self.classifier)

    def __getitem__(self, item):
        try:
            if item < self.len1:
                layer = self.features[item]
            elif item < (self.len1 + self.len2):
                len = self.len1
                layer = self.inception3[item - len]
            elif item < (self.len1 + self.len2 + self.len3):
                len = self.len1 + self.len2
                layer = self.inception4[item - len]
            elif item < (self.len1 + self.len2 + self.len3 + self.len4):
                len = self.len1 + self.len2 + self.len3
                layer = self.inception5[item - len]
            else:
                len = self.len1 + self.len2 + self.len3 + self.len4
                layer = self.classifier[item - len]
        except IndexError:
            raise StopIteration()
        return layer


class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)