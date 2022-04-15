import torch
import time
import socket
import pickle
import function
import torch.nn as nn


def startClient(model,x,device,ip,port,epoch):
    # X = [1,3,224,224]
    wh_min = 1
    wh_max = 224
    channel_min = 1
    channel_max = 64

    # for channel in range(1,64):
    for channel in range(channel_min,channel_max + 1):
        for w in range(wh_min,wh_max + 1):
            for h in range(wh_min,wh_max +1):

                # 创建socket
                p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 请求连接
                p.connect((ip, port))

                x = torch.rand(1,channel,w,h)
                x = x.to(device)


                """
                    发送边缘端计算后的中间数据
                """
                dumps_x = pickle.dumps(x)
                dumps_x_length = len(x)
                # print(f'client data length {x}')

                # 发送数据长度 告诉服务端有多少数据需要发送
                p.sendall(pickle.dumps(dumps_x_length))
                resp_length = p.recv(1024).decode("UTF-8")
                # print(f"{resp_length} server 已经收到要发送的数据长度")

                p.sendall(dumps_x)
                # 收到第一次信号 说明已经接收到了传过去的edge_x数据
                edge_resp2 = p.recv(1024).decode("utf-8")
                print(f"{x.shape}发送完毕")

                # print("====================================")
                p.close()


if __name__ == '__main__':
    for i in range(1,10+1):
        print(i)