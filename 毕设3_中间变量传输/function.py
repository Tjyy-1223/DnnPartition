import torch
import torch.nn as nn
import time
import threading

"""
model_partition函数可以将一个整体的model 划分成两个部分
    划分的大致思路：
    如选定 第 index(下标从1开始) 层对alexnet进行划分 ，则代表在第index后对模型进行划分
    则对alexnet网络进行 层级遍历
    将index层包括第index层 包装给edge_model作为返回 意为边缘节点
    后续的节点包装给 cloud_model 意为云端节点
"""


def model_partition(alexnet, index):
    edge_model = nn.Sequential()
    cloud_model = nn.Sequential()
    idx = 1

    for layer in alexnet:
        if (idx <= index):
            edge_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
        else:
            cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
        idx += 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    edge_model = edge_model.to(device)
    cloud_model = cloud_model.to(device)
    return edge_model, cloud_model


"""
一个观察每层情况的函数
    可以输出layer的各种性质 其输出如下：
    1-Conv2d computation time: 30.260000 s
    output shape: torch.Size([10000, 64, 55, 55]) 	 transport_num:1936000000 	 transport_size:61952.0M
    weight  :  parameters size torch.Size([64, 3, 11, 11]) 	 parameters number 23232
    bias  :  parameters size torch.Size([64]) 	 parameters number 64

    最后返回运行结果x
    修改：省略了 激活层 batchnormal层 以及 dropout层
"""
def show_features(alexnet, x ,filter = True,epoch = 3):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU warm-up and prevent it from going into power-saving mode
    dummy_input = torch.rand(x.shape).to(device)

    # init loggers
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    init_starter = torch.cuda.Event(enable_timing=True)
    init_ender = torch.cuda.Event(enable_timing=True)

    # GPU warm-up
    with torch.no_grad():
        for i in range(3):
            init_starter.record()
            _ = alexnet(dummy_input)
            init_ender.record()
            # wait for GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            print(f"GPU warm-up - {i+1}")
            print(f'computation time: {curr_time :.3f} ms\n')

    if len(alexnet) > 0:
        idx = 1
        for layer in alexnet:
            if filter is True:
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue

            all_time = 0
            temp_x = x
            with torch.no_grad():
                for i in range(epoch):
                    temp_x = torch.rand(temp_x.shape).to(device)

                    # start_time = time.time()

                    starter.record()
                    x = layer(temp_x)
                    ender.record()
                    # wait for GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)

                    # end_time = time.time()
                    # print(curr_time)
                    all_time += curr_time

            # 计算分割点 中间传输占用大小为多少m  主要与网络传输时延相关
            total_num = 1
            for num in x.shape:
                total_num *= num
            type_size = 4
            size = total_num * type_size / 1000 / 1000

            print("------------------------------------------------------------------")
            print(f'{idx}-{layer} \n'
                  f'computation time: {(all_time):.3f} ms\n'
                  f'output shape: {x.shape}\t transport_num:{total_num}    transport_size:{size:.3f}MB')

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


def show_2model(edge_model, cloud_model):
    print(f"------------- edge model -----------------")
    print(edge_model)
    print(f"------------- cloud model -----------------")
    print(cloud_model)


def show_1model(model,filter = True):
    if len(model) > 0:
        idx = 1
        for layer in model:
            if filter is True:
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue
            print(f'{idx}-{layer}')
            idx += 1
    else:
        print("this model is a empty model")

