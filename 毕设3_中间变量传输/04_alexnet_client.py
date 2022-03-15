import a0_alexNet
import socket_function
import torch
import time
import socket
import pickle

"""
    1 - 初始化一个alexnet网络模型
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
AlexNet = a0_alexNet.AlexNet(input_layer = 3,num_classes = 1000)
AlexNet = AlexNet.to(device)


x = torch.rand(size=(1000,3,224,224),requires_grad=False)
x = x.to(device)
print(f"x device : {x.device}")



for i in range(len(AlexNet) + 1):
    point_index = i
    edge_model, cloud_model = a0_alexNet.model_partition(AlexNet, point_index)

    start_alltime = int(round(time.time() * 1000))
    start_time = int(round(time.time() * 1000))
    edge_x = edge_model(x)
    end_time = int(round(time.time() * 1000))
    print(f"从第{point_index}层进行划分\t边缘端计算用时 : {(end_time - start_time) / 1000 :>3} s")

    # 创建socket
    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 请求连接
    p.connect(('127.0.0.1', 8080))
    edge_x = pickle.dumps(edge_x)



    start_time = int(round(time.time() * 1000))
    p.sendall(edge_x)
    # 收到第一次信号 说明已经接收到了传过去的edge_x数据
    data = p.recv(1024)
    # print(data)
    end_time = int(round(time.time() * 1000))
    print(f"传输时延 : {(end_time - start_time) / 1000 :>3} s")




    # 收到第二次信号 说明对面的云端数据已经计算完毕
    data2 = p.recv(1024)
    # print("get the second yes,the entire computation has completed",data2)
    end_alltime = int(round(time.time() * 1000))
    print(f"从第{point_index}层进行划分\t云边协同计算用时 : {(end_alltime - start_alltime) / 1000 :>3} s")
    print("====================================")
    p.close()