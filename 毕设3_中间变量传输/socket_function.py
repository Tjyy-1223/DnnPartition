import socket
import time
import torch
import pickle


def get_data(port):
    # 创建socket
    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    p.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    # 绑定端口号
    p.bind(('127.0.0.1', port))
    # 打开监听
    p.listen(5)
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
        if idx % (1024 * 100) == 0:
            print(f"{(idx/(1024*100)) :.0f} * 100MB")
        idx += 1
        if len(packet) < 1024: break
    parse_data = pickle.loads(b"".join(data))
    end_time = int(round(time.time() * 1000))

    print(f"client data: {parse_data.shape}\ntime: {(end_time - start_time) / 1000 :>3}s")

    conn.send("yes".encode("UTF-8"))

    # 关闭连接
    conn.close()
    p.close()



def send_data(port,x):
    # 创建socket
    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 请求连接
    p.connect(('127.0.0.1', port))
    p.sendall(pickle.dumps(x))

    data = p.recv(1024)
    print(data)

    p.close()

