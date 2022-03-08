import socket
import time
import pickle

# 创建socket
p = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
p.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
# 绑定端口号
p.bind(('127.0.0.1',8080))

# 打开监听
p.listen(5)

# 等待客户端链接
conn,client = p.accept()

# 收发消息
data = []
idx = 0
start_time = int(round(time.time() * 1000))
while True:
    packet = conn.recv(1024)
    data.append(packet)
    # print(f"{idx} - {len(packet)}")
    # idx += 1
    if len(packet) < 1024: break
parse_data = pickle.loads(b"".join(data))
end_time = int(round(time.time() * 1000))


print(f"client data: {parse_data.shape}\ntime: {(end_time - start_time) / 1000 :>3}s")

conn.send("yes".encode("UTF-8"))

# 关闭连接
conn.close()
p.close()