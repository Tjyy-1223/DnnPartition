import torch
import torch.nn as nn
from collections import abc


"""
    修改 alexnet 使其 iterable
    下面是alexnet网络的迭代参数调整
    将下面的设置传入到alexnet的__iter__中可以完成对于alexnet网络的层级遍历
"""
class SentenceIterator(abc.Iterator):
    def __init__(self,features,avgpool,classifier):
        self.features = features
        self.avg_pool = avgpool
        self.classifier = classifier
        self._index = 0
        self.len1 = len(features)
        self.len2 = 1
        self.len3 = len(classifier)


    def __next__(self):
        try:
            if self._index < self.len1:
                layer = self.features[self._index]
            elif self._index < (self.len1 + self.len2):
                layer = self.avg_pool
            else:
                layer = self.classifier[self._index - self.len1 - self.len2]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1

        return layer





"""
    这是一个可以遍历的alexnet模型
"""
class AlexNet(nn.Module):
    def __init__(self,input_layer = 3,num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_layer, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def __iter__(self,):
        return SentenceIterator(self.features,self.avgpool,self.classifier)

    def __len__(self):
        return (len(self.features) + 1 + len(self.classifier))





"""
划分的大致思路：
如选定 第 index(下标从1开始) 层对alexnet进行划分 ，则代表在第index后对模型进行划分
则对alexnet网络进行 层级遍历
将index层包括第index层 包装给edge_model作为返回 意为边缘节点
后续的节点包装给 cloud_model 意为云端节点
"""

def model_partition(alexnet,index):
    edge_model = nn.Sequential()
    cloud_model = nn.Sequential()
    idx = 1

    for layer in alexnet:
        if(idx <= index):
            edge_model.add_module(f"{idx}-{layer.__class__.__name__}",layer)
        else:
            cloud_model.add_module(f"{idx}-{layer.__class__.__name__}",layer)
        idx += 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    edge_model = edge_model.to(device)
    cloud_model = cloud_model.to(device)
    return edge_model,cloud_model