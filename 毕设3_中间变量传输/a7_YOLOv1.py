import torch
import torch.nn as nn
from collections import abc
import time
import torchvision.models as model

"""
修改 alexnet 使其 iterable:
    下面是alexnet网络的迭代参数调整
    将下面的设置传入到alexnet的__iter__中可以完成对于alexnet网络的层级遍历
"""
class SentenceIterator(abc.Iterator):
    def __init__(self,features,classifier):
        self.features = features
        self.classifier = classifier
        self._index = 0
        self.len1 = len(features)
        self.len2 = len(classifier)


    def __next__(self):
        try:
            if self._index < self.len1:
                layer = self.features[self._index]
            else:
                layer = self.classifier[self._index - self.len1]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1

        return layer





"""
    这是一个可以遍历的alexnet模型
"""
class LeNet(nn.Module):
    def __init__(self,input_layer = 3,num_classes: int = 1000) -> None:
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 54 * 54, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1000)
        )
        self.len1 = len(self.features)
        self.len2 = len(self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def __iter__(self,):
        return SentenceIterator(self.features,self.classifier)

    def __len__(self):
        return (len(self.features) + len(self.classifier))

    def __getitem__(self, item):
        try:
            if item < self.len1:
                layer = self.features[item]
            else:
                layer = self.classifier[item - self.len1]
        except IndexError:
            raise StopIteration()
        return layer



