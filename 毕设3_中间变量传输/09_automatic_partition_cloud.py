import socket
import torch
import function
import torch.nn as nn
import pickle

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


def responseCloudTime(p,device):
    # 等待客户端链接
    conn, client = p.accept()
    print(f"successfully connection :{conn}\n")

    model_selection = pickle.loads(conn.recv(1024))
    print(f"get the parameter:model_selection = {model_selection}")
    model = function.getDnnModel(model_selection)
    model.to(device)

    # 进行每层的计算
    cloudTime = []
    x = torch.rand(size=(1,3,224,224))
    x = x.to(device)

    if device == "cuda":
        function.warmUpGpu(model,x,device)
    if device == "cpu":
        function.warmUpCpu(model,x,device)

    for i in range(len(model)):
        layer = model[i]
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
            x = layer(x)
            continue
        time = None
        if device == "cuda":
            x, time = function.recordTimeGpu(layer,x,device,300)
        if device == "cpu":
            x, time = function.recordTimeCpu(layer, x, device, 300)
        print(f"{layer} \t\t computation time :{time:.3f}")
        cloudTime.append(time)

    pickle_cloudTime = pickle.dumps(cloudTime)
    data_length = len(pickle_cloudTime)
    conn.sendall(pickle.dumps(data_length))

    conn.sendall(pickle_cloudTime)
    conn.close()


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ip = "127.0.0.1"
    # ip = "112.86.199.171"
    port = 8090


    p = startServer(ip,port)
    while True:
        responseCloudTime(p,device)
    p.close()

