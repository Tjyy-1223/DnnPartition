import a0_alexNet
import socket_function
import torch
import time

"""
    1 - 初始化一个alexnet网络模型
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
AlexNet = a0_alexNet.AlexNet(input_layer = 3,num_classes = 1000)
AlexNet = AlexNet.to(device)

"""
    2 - 选定一个层进行划分 在这里选定第 8 层 即第2次卷积层 + Relu之后
    得到edge_model 和 cloud_model 
"""
point_index = 16
edge_model,cloud_model = a0_alexNet.model_partition(AlexNet,point_index)

"""
    3.用cloud_model在云端进行计算并记录时间
"""
x = socket_function.get_data(8080)

start_time = int(round(time.time() * 1000))
x = cloud_model(x)
end_time = int(round(time.time() * 1000))
print(f"云端计算用时 : {(end_time - start_time) / 1000 :>3} s\n")