import torch
import torch.nn as nn
from collections import abc
import time


"""
修改 alexnet 使其 iterable:
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
model_partition函数可以将一个整体的model 划分成两个部分
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


"""
一个观察每层情况的函数
    可以输出layer的各种性质 其输出如下：
    1-Conv2d computation time: 30.260000 s
    output shape: torch.Size([10000, 64, 55, 55]) 	 transport_num:1936000000 	 transport_size:61952.0M
    weight  :  parameters size torch.Size([64, 3, 11, 11]) 	 parameters number 23232
    bias  :  parameters size torch.Size([64]) 	 parameters number 64
    
    最后返回运行结果x
"""
def show_features(alexnet,x):
    if len(alexnet) > 0:
        idx = 1
        for layer in alexnet:
            start_time = int(round(time.time() * 1000))
            x = layer(x)
            end_time = int(round(time.time() * 1000))
            # print(x.device)

            # 计算分割点 中间传输占用大小为多少m  主要与网络传输时延相关
            total_num = 1
            for num in x.shape:
                total_num *= num
            type_size = 32
            size = total_num * type_size / 1000 / 1000

            print("------------------------------------------------------------------")
            print(f'{idx}-{layer} \n'
                  f'computation time: {(end_time - start_time) / 1000 :>3} s\n'
                  f'output shape: {x.shape}\t transport_num:{total_num}\t transport_size:{size:.3f}M')

            # 计算各层的结构所包含的参数量 主要与计算时延相关
            # para = parameters.numel()
            for name, parameters in layer.named_parameters():
                print(f"{name}  :  parameters size {parameters.size()} \t parameters number {parameters.numel()}")
            idx += 1
        return x
    else:
        print("this model is a empty model")
        return x



"""
    一个显示边缘模型和云端模型每层结构的函数
"""
def show_model(edge_model,cloud_model):
    print(f"------------- edge model -----------------")
    print(edge_model)
    print(f"------------- cloud model -----------------")
    print(cloud_model)


