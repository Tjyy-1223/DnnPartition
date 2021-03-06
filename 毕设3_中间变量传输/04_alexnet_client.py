import torch
import time
import socket
import pickle
import function
import torch.nn as nn

def startClient(model,x,device,ip,port,epoch):
    index = 0
    for point_index in range(len(model) + 1):
        if point_index > 0:
            layer = model[point_index-1]
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                continue

        # 创建socket
        p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 请求连接
        p.connect((ip,port))

        edge_model, _ = function.model_partition(model, point_index)
        edge_model = edge_model.to(device)

        """
            step1 记录边缘端的计算用时
        """
        if device == "cuda":
            edge_x,edge_time = function.recordTimeGpu(edge_model,x,device,epoch)
        elif device == "cpu":
            edge_x,edge_time = function.recordTimeCpu(edge_model, x, device, epoch)

        print(f"从第{index}层进行划分\t边缘端计算用时 : {edge_time :.3f} ms")

        """
            step2 发送边缘端的计算时延
        """
        p.sendall(pickle.dumps(edge_time))
        edge_resp = p.recv(1024).decode("utf-8")
        print(f"get {edge_resp} 边缘端计算时延 发送成功")

        """
            step3 发送边缘端计算后的中间数据
        """
        start_time = time.perf_counter()
        edge_x = pickle.dumps(edge_x)
        end_time = time.perf_counter()
        curr = (end_time - start_time)*1000
        print(f"pickle dumps times {curr}  ms")

        edge_x_length = len(edge_x)
        print(f'client data length {edge_x_length}')

        # 发送数据长度 告诉服务端有多少数据需要发送
        p.sendall(pickle.dumps(edge_x_length))
        resp_length = p.recv(1024).decode("UTF-8")
        print(f"{resp_length} server 已经收到要发送的数据长度")


        p.sendall(edge_x)
        # 收到第一次信号 说明已经接收到了传过去的edge_x数据
        edge_resp2 = p.recv(1024).decode("utf-8")
        print(f"get {edge_resp2} 边缘端中间数据 发送成功\n")

        """
            通过查看 starttime 和 endtime 的值
            解决了网络传输过程中 服务端显示传输时延 比 客户端显示传输时延 大的问题
        """
        # print(start_time)
        # print(end_time)
        # print(f"传输时延 : {(end_time - start_time)*1000:.3f} ms")

        """
            step4 接收到对方计算完成后发送的数据
        """
        # 收到第二次信号 说明对面的云端数据已经计算完毕
        edge_resp3 = p.recv(1024).decode("utf-8")

        print("云边协同计算完成",edge_resp3)
        end_alltime = time.perf_counter()
        # print(f"从第{point_index}层进行划分\t云边协同计算用时 : {(end_alltime - start_alltime)*1000:.3f} ms")

        print("====================================")
        p.close()
        index += 1


if __name__ == '__main__':
    """
    Params:
        1 modelIndex 挑选第几个model
        2 ip 服务端的IP地址
        3 port 服务端绑定的端口地址
        4 epoch 测量GPU/CPU 计算epoch次取平均值
        5 device 目前使用的设备
    """
    modelIndex = 6
    # ip = "127.0.0.1"
    ip = "122.96.111.129"
    port = 8090
    epoch = 300
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"


    """
        Step1 根据modelIndex取出myModel类型
    """
    myModel = function.getDnnModel(modelIndex)
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
        function.warmUpGpu(myModel,x,device)
    elif device == "cpu":
        function.warmUpCpu(myModel,x,device)

    """
        Step4 绑定的端口启动Socket 并且开始监听
    """
    startClient(myModel,x,device,ip,port,epoch)



