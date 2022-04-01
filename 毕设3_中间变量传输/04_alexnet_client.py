import a0_alexNet
import function
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


x = torch.rand(size=(128,3,224,224),requires_grad=False)
x = x.to(device)
print(f"x device : {x.device}")
x.requires_grad = False

# GPU warm-up and prevent it from going into power-saving mode
dummy_input = torch.rand(x.shape).to(device)

# init loggers
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

# GPU warm-up
with torch.no_grad():
    for i in range(3):
        _ = AlexNet(dummy_input)
        print(f"GPU warm-up - {i+1}")

for i in range(len(AlexNet) + 1):
    point_index = i
    edge_model, cloud_model = function.model_partition(AlexNet, point_index)

    start_alltime = time.time()

    # 开始记录时间
    starter.record()
    with torch.no_grad():
        edge_x = edge_model(x)
    ender.record()

    # wait for GPU SYNC
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)

    print(f"从第{point_index}层进行划分\t边缘端计算用时 : {curr_time :.3f} ms")

    # 创建socket
    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 请求连接
    p.connect(('127.0.0.1', 8080))
    edge_x = pickle.dumps(edge_x)

    start_time = time.time()
    p.sendall(edge_x)
    # 收到第一次信号 说明已经接收到了传过去的edge_x数据
    data = p.recv(1024)
    # print(data)
    end_time = time.time()

    """
        通过查看 starttime 和 endtime 的值
        解决了网络传输过程中 服务端显示传输时延 比 客户端显示传输时延 大的问题
    """
    # print(start_time)
    # print(end_time)
    print(f"传输时延 : {(end_time - start_time)*1000:.3f} ms")


    # 收到第二次信号 说明对面的云端数据已经计算完毕
    data2 = p.recv(1024)
    # print(data)
    # print("get the second yes,the entire computation has completed",data2)
    end_alltime = time.time()

    print(f"从第{point_index}层进行划分\t云边协同计算用时 : {(end_alltime - start_alltime)*1000:.3f} ms")
    print("====================================")
    p.close()