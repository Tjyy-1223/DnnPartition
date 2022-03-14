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

# 初始化一个 socket连接
# 创建socket
p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
p.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
# 绑定端口号
p.bind(('127.0.0.1', 8080))
# 打开监听
p.listen(5)


for i in range(len(AlexNet) + 1):
    point_index = i
    edge_model, cloud_model = a0_alexNet.model_partition(AlexNet, point_index)


    # 等待客户端链接
    conn, client = p.accept()
    print(f"successfully connection :{conn}")
    # 收发消息
    data = []
    idx = 0
    start_time = int(round(time.time() * 1000))
    while True:
        packet = conn.recv(1024)
        data.append(packet)
        # 100mb 提醒一次
        # if idx % (1024 * 100) == 0:
            # print(f"{(idx / (1024 * 100)) :.0f} * 100MB")
        idx += 1
        if len(packet) < 1024: break
    parse_data = pickle.loads(b"".join(data))
    end_time = int(round(time.time() * 1000))

    print(f"client data: {parse_data.shape} \n传输时延: {(end_time - start_time) / 1000 :>3}s")
    parse_data.requires_grad = False
    conn.sendall("yes".encode("UTF-8"))


    time.sleep(2)
    start_time2 = int(round(time.time() * 1000))
    cloud_x = cloud_model(parse_data)
    end_time2 = int(round(time.time() * 1000))
    print(f"从第{point_index}层进行划分\t云端计算用时 : {(end_time2 - start_time2) / 1000 :>3} s")
    print("==============================================")

    conn.sendall("yes".encode("UTF-8"))
    # 关闭连接
    conn.close()
p.close()