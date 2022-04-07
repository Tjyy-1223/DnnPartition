import a1_alexNet
import function
import torch
import time
import socket
import pickle


def warmUp(model,x,device):
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
        avg_time /= 10
        print(f"GPU warm-up: {curr_time:.3f}ms")
        print("==============================================")


def startClient(model,x,device,ip,port,epoch):
    for point_index in range(len(AlexNet) + 1):
        # 创建socket
        p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 请求连接
        p.connect((ip,port))

        edge_model, _ = function.model_partition(model, point_index)

        """
            step1 记录边缘端的计算用时
        """
        edge_time = 0.0
        for i in range(epoch):
            x = torch.rand(x.shape).to(device)
            # init loggers
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            starter.record()
            with torch.no_grad():
                edge_x = edge_model(x)
            ender.record()
            # wait for GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            edge_time += curr_time
        edge_time /= epoch
        print(f"从第{point_index}层进行划分\t边缘端计算用时 : {edge_time :.3f} ms")

        """
            step2 发送边缘端的计算时延
        """
        p.send(pickle.dumps(edge_time))
        edge_resp = p.recv(1024).decode("utf-8")
        print(f"get {edge_resp} 边缘端计算时延 发送成功")

        """
            step3 发送边缘端计算后的中间数据
        """
        edge_x = pickle.dumps(edge_x)
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


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    AlexNet = a1_alexNet.AlexNet(input_layer=3, num_classes=1000)
    AlexNet = AlexNet.to(device)

    x = torch.rand(size=(1, 3, 224, 224), requires_grad=False)
    x = x.to(device)
    print(f"x device : {x.device}")

    warmUp(AlexNet,x,device)

    ip = "127.0.0.1"
    port = 8090
    epoch = 300
    startClient(AlexNet,x,device,ip,port,epoch)



