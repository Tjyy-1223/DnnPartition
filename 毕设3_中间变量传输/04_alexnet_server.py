import a1_alexNet
import a2_vggNet
import a3_GoogLeNet
import a4_ResNet
import a5_MobileNet

import torch
import time
import socket
import pickle
import function
import torch.nn as nn


def warmUpGpu(model,x,device):
    # GPU warm-up and prevent it from going into power-saving mode
    dummy_input = torch.rand(x.shape).to(device)

    # GPU warm-up
    with torch.no_grad():
        for i in range(3):
            _ = model(dummy_input)

        avg_time = 0.0

        for i in range(300):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()

            _ = model(dummy_input)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            avg_time += curr_time
        avg_time /= 300
        print(f"GPU warm-up: {curr_time:.3f}ms")
        print("==============================================")


def warmUpCpu(model,x,device):
    # GPU warm-up and prevent it from going into power-saving mode
    dummy_input = torch.rand(x.shape).to(device)

    # GPU warm-up
    with torch.no_grad():
        for i in range(3):
            _ = model(dummy_input)

        avg_time = 0.0

        for i in range(10):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()

            curr_time = end - start
            avg_time += curr_time
        avg_time /= 10
        print(f"CPU warm-up: {curr_time:.3f}ms")
        print("==============================================")


def recordTimeGpu(model, x, device, epoch):
    all_time = 0.0
    for i in range(epoch):
        x = torch.rand(x.shape).to(device)
        # init loggers
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            starter.record()
            res_x = model(x)
            ender.record()

        # wait for GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        all_time += curr_time
    all_time /= epoch
    return res_x, all_time


def recordTimeCpu(model, x, device, epoch):
    all_time = 0.0
    for i in range(epoch):
        x = torch.rand(x.shape).to(device)

        with torch.no_grad():
            start_time = time.perf_counter()
            res_x = model(x)
            end_time = time.perf_counter()

        curr_time = end_time - start_time
        all_time += curr_time
    all_time /= epoch
    return res_x, all_time * 1000


def getDnnModel(index):
    if index == 1:
        alexnet = a1_alexNet.AlexNet(input_layer=3, num_classes=1000)
        return alexnet
    elif index == 2:
        vgg16 = a2_vggNet.vgg16_bn()
        return vgg16
    elif index == 3:
        GoogLeNet = a3_GoogLeNet.GoogLeNet()
        return GoogLeNet
    elif index == 4:
        resnet18 = a4_ResNet.resnet18()
        return resnet18
    elif index == 5:
        mobileNet = a5_MobileNet.mobilenet_v2()
        return mobileNet
    else:
        print("no model")
        return None


def startServer(ip,port):
    # 初始化一个 socket连接
    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # p.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    p.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 绑定端口号
    p.bind((ip,port))
    # 打开监听
    p.listen(5)
    return p


def startListening(model,p,device,epoch):
    index = 0
    for point_index in range(len(model) + 1):
        layer = None
        if point_index > 0:
            layer = model[point_index-1]
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                continue


        _, cloud_model = function.model_partition(model, point_index)
        # print(next(cloud_model.parameters()).device)

        # 等待客户端链接
        conn, client = p.accept()
        print(f"successfully connection :{conn}\n")
        # 收发消息

        """
            step1 接收边缘端的计算时延
        """
        edge_time = pickle.loads(conn.recv(1024))
        print(layer)
        print(f"从第{index}层进行划分\t边缘端计算用时 : {edge_time :.3f} ms")
        edge_resp = "yes".encode("UTF-8")
        conn.send(edge_resp)

        """
            step2 接收边缘端的中间数据
        """
        # 接收客户端发送来的数据长度
        edge_x_length = pickle.loads(conn.recv(1024))
        print(f"client 即将发送的edge_x的数据长度 {edge_x_length}")
        resp_length = "getLength".encode("UTF-8")
        conn.sendall(resp_length)

        data = [conn.recv(4096)]  # 一开始的部分,用于等待传输开始,避免接收不到的情况.
        start_time = time.perf_counter()
        if data[0] in (0, -1):  # 返回0,-1代表出错
            return False
        while True:
            packet = conn.recv(4096)
            data.append(packet)
            if len(b"".join(data)) >= edge_x_length:
                break
        end_time = time.perf_counter()

        print(f'server data length {len(b"".join(data))}')
        parse_data = pickle.loads(b"".join(data))

        """
            step3 打印传输中间数据所用的时间
            通过查看 starttime 和 endtime 的值
            解决了网络传输过程中 服务端显示传输时延 比 客户端显示传输时延 大的问题
        """
        # print(start_time)
        # print(end_time)
        transport_time = (end_time - start_time) * 1000
        print(f"传输时延: {transport_time:.3f} ms")
        print(f"client data: {parse_data.shape}")
        parse_data.requires_grad = False

        """ 这里 parse data 不用加to device：因为传过来的数据默认在cuda0上了"""
        parse_data = parse_data.to(device)
        # print(parse_data.device)

        """
            step4 告诉服务端已经接收到了数据
        """
        data1 = "yes".encode("UTF-8")
        # print("get the client's data,return yes to client",data1)
        conn.sendall(data1)

        """
            step5 记录云端计算用时
        """
        if device == "cuda":
            _,server_time = recordTimeGpu(cloud_model,parse_data,device,epoch)
        elif device == "cpu":
            _,server_time = recordTimeCpu(cloud_model, parse_data, device, epoch)

        print(f"从第{index}层进行划分\t云端计算用时 : {server_time :.3f} ms")
        print(f"从第{index}层进行划分\t云边协同计算用时 : {(edge_time + transport_time + server_time):.3f} ms")
        print("==============================================")

        conn.sendall("yes".encode("UTF-8"))
        # 关闭连接
        conn.close()
        index += 1
    p.close()


if __name__ == '__main__':
    """
        Params:
        1 modelIndex 挑选第几个model
        2 ip 服务端的IP地址
        3 port 服务端绑定的端口地址
        4 epoch 测量GPU/CPU 计算epoch次取平均值
        5 device 目前使用的设备
    """
    modelIndex = 1
    # ip = "127.0.0.1"
    ip = "112.86.199.171"
    port = 8090
    epoch = 300
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """
        Step1 根据modelIndex取出myModel类型
    """
    myModel = getDnnModel(modelIndex)
    myModel = myModel.to(device)

    """
        Step2 准备Input数据 后续若有真实数据可以在这里替换
    """
    x = torch.rand(size=(1, 3, 224, 224), requires_grad=False)
    x = x.to(device)
    print(f"x device : {x.device}")

    """
        Step3 GPU/CPU预热
    """
    if device == "cuda":
        warmUpGpu(myModel, x, device)
    elif device == "cpu":
        warmUpCpu(myModel, x, device)

    """
        Step4 绑定的端口启动Socket 并且开始监听
    """
    p = startServer(ip,port)
    startListening(myModel,p,device,epoch)




