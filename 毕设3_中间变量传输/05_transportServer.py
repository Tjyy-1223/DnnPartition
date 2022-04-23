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



def startListening(p,device,epoch,save_flag=False,path = None,sheet_name=None):
    # X = [1,3,224,224]
    wh_min = 0
    wh_max = 224
    channel_min = 0
    channel_max = 64
    step1 = 4
    step2 = 7

    index = 0
    # for channel in range(1,64):
    for channel in range(channel_min, channel_max + 1,step1):
        for wh in range(wh_min, wh_max + 1,step2):
                if wh == 0 or channel == 0:
                    continue

                # 等待客户端链接
                conn, client = p.accept()
                print(f"successfully connection :{conn}\n")

                # 接收客户端发送来的数据长度 即1 - 序列化之后的长度
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
                transport_time = (end_time - start_time) * 1000


                # print(f'server data length {len(b"".join(data))}')
                parse_data = pickle.loads(b"".join(data))
                parse_data = parse_data.to(device)

                # 2 - parse_data的形状
                parse_data_shape = parse_data.shape

                # 3 parse_data shape 的 乘积
                prod = 1
                for i in range(len(parse_data_shape)):
                    prod *= parse_data.shape[i]
                # print(prod)

                print(f"shape:{parse_data_shape}   shape's prod:{prod}   dumps'length {dumps_x_length}  transport time:{transport_time:.3f} ms")
                if save_flag:
                    # value = [["index", "shape", "shape's prod", "dumps length", "transport time"]]

                    value = [[index,f"{parse_data_shape}",prod,dumps_x_length,round(transport_time,3)]]
                    function.write_excel_xls_append(path, sheet_name, value)

                print("====================================")
                conn.close()
                index += 1
    p.close()



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 10

    ip = "127.0.0.1"
    # ip = "112.86.199.140"
    port = 8090

    path = "../res/transport_time.xls"
    sheet_name = "time2"
    value = [["index", "shape", "shape's prod", "dumps length", "transport time(ms)"]]

    save_flag = True
    if save_flag:
        function.create_excel_xsl(path, sheet_name, value)
    p = startServer(ip, port)
    startListening(p, device, epoch, save_flag,path=path,sheet_name=sheet_name)
