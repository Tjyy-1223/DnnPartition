import socket
import time
import torch
import pickle

# 创建socket
p = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

# 请求连接
p.connect(('127.0.0.1',8080))

# 发送消息
x = torch.rand((96,124,124))
p.sendall(pickle.dumps(x))


data = p.recv(4096)
print("yes")
print(data)

p.close()