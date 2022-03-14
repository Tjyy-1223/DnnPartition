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
point_index = 7
edge_model,cloud_model = a0_alexNet.model_partition(AlexNet,point_index)

x = torch.rand(size=(1000,3,224,224))
x = x.to(device)
print(f"x device : {x.device}")


"""
    3.用edge_model在边缘端进行计算并记录时间
"""
start_time = int(round(time.time() * 1000))
edge_x = edge_model(x)
end_time = int(round(time.time() * 1000))
print(f"边缘端计算用时 : {(end_time - start_time) / 1000 :>3} s")

socket_function.send_data(8080,edge_x)


"""
    4.观察一下edge_model的运算过程
"""
a0_alexNet.show_features(edge_model,x)