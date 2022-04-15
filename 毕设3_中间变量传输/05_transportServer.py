import torch
import time
import socket
import pickle
import function
import torch.nn as nn



def startServer(ip,port):
    # 初始化一个 socket连接
    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # p.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    p.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 绑定端口号
    p.bind((ip,port))
    # 打开监听
    p.listen(24)
    return p



def startListening(model,p,device,epoch,save = False,model_name="model",path = None):
    # X = [1,3,224,224]
    wh_min = 1
    wh_max = 224
    channel_min = 1
    channel_max = 64

    # for channel in range(1,64):
    for channel in range(channel_min, channel_max + 1):
        for w in range(wh_min, wh_max + 1):
            for h in range(wh_min, wh_max + 1):
                # 等待客户端链接
                conn, client = p.accept()
                print(f"successfully connection :{conn}\n")

                # 接收客户端发送来的数据长度
                dumps_x_length = pickle.loads(conn.recv(1024))
                # print(f"client 即将发送的edge_x的数据长度 {dumps_x_length}")
                resp_length = "getLength".encode("UTF-8")
                conn.sendall(resp_length)

                """
                    接收对面发送的数据
                """
                data = [conn.recv(1)]  # 一开始的部分,用于等待传输开始,避免接收不到的情况.
                start_time = time.perf_counter()
                if data[0] in (0, -1):  # 返回0,-1代表出错
                    return False
                while True:
                    packet = conn.recv(4096)
                    data.append(packet)
                    if len(b"".join(data)) >= dumps_x_length:
                        break
                end_time = time.perf_counter()

                # print(f'server data length {len(b"".join(data))}')
                parse_data = pickle.loads(b"".join(data))




if __name__ == '__main__':
    x = torch.rand(1,64,224,224)
    x_shape = x.shape

    prod = 1
    for i in range(len(x_shape)):
        prod *= x_shape[i]
    print(prod)

    print(len(pickle.dumps(x)))

    x = torch.rand(1, 24, 24, 24)
    x_shape = x.shape

    prod = 1
    for i in range(len(x_shape)):
        prod *= x_shape[i]
    print(prod)

    print(len(pickle.dumps(x)))
