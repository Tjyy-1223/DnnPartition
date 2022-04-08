import a1_alexNet
import socket_function
import torch
import time
import socket
import pickle
import function


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
    for point_index in range(len(AlexNet) + 1):
        _, cloud_model = function.model_partition(model, point_index)
        # print(next(cloud_model.parameters()).device)

        # 等待客户端链接
        conn, client = p.accept()
        print(f"successfully connection :{conn}")
        # 收发消息

        """
            step1 接收边缘端的计算时延
        """
        edge_time = pickle.loads(conn.recv(1024))
        print(f"从第{point_index}层进行划分\t边缘端计算用时 : {edge_time :.3f} ms")
        edge_resp = "yes".encode("UTF-8")
        conn.send(edge_resp)

        """
            step2 接收边缘端的中间数据
        """
        data = []
        idx = 0
        start_time = 0
        while True:
            packet = conn.recv(1024)
            if idx == 0:
                start_time = time.perf_counter()
            data.append(packet)
            idx += 1
            if len(packet) < 1024: break
        parse_data = pickle.loads(b"".join(data))
        end_time = time.perf_counter()


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

        print(f"从第{point_index}层进行划分\t云端计算用时 : {server_time :.3f} ms")
        print(f"从第{point_index}层进行划分\t云边协同计算用时 : {(edge_time + transport_time + server_time):.3f} ms")
        print("==============================================")

        conn.sendall("yes".encode("UTF-8"))
        # 关闭连接
        conn.close()
    p.close()


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AlexNet = a1_alexNet.AlexNet(input_layer=3, num_classes=1000)
    AlexNet = AlexNet.to(device)

    x = torch.rand(size=(1, 3, 224, 224), requires_grad=False)
    x = x.to(device)
    print(f"x device : {x.device}")

    if device == "cuda":
        warmUpGpu(AlexNet, x, device)
    elif device == "cpu":
        warmUpCpu(AlexNet, x, device)

    ip = "127.0.0.1"
    port = 8090
    p = startServer(ip,port)

    epoch = 300
    startListening(AlexNet,p,device,epoch)




