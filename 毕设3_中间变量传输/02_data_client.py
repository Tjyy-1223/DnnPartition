import socket_function
import torch
import pickle

# 发送消息
x = torch.rand((100,64,112,112))
bytes_x = pickle.dumps(x)

total_num = 1
for num in x.shape:
    total_num *= num
type_size = 4
size = total_num * type_size / 1000 / 1000

print(f" transport_size:{size:.3f}MB")

socket_function.send_data(8080,x)