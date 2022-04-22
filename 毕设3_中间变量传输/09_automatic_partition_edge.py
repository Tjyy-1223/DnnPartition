import socket
import torch
import function
import functionImg
import pickle
import torch.nn as nn
import joblib


def requestCloudTime(ip,port,model_selection):
    # 创建socket
    p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 请求连接
    p.connect((ip, port))
    # 发送 model_selection 参数
    p.sendall(pickle.dumps(model_selection))
    # 接收 serverTime pickle 长度
    data_length = pickle.loads(p.recv(1024))

    data = [p.recv(128)]  # 一开始的部分,用于等待传输开始,避免接收不到的情况.
    if data[0] in (0, -1):  # 返回0,-1代表出错
        return False
    while True:
        packet = p.recv(4096)
        data.append(packet)
        if len(b"".join(data)) >= data_length:
            break

    cloudTime = pickle.loads(b"".join(data))
    return cloudTime


def requestEdgeTime(model_selection,device = "cpu"):
    model = function.getDnnModel(model_selection)
    model.to(device)

    # 进行每层的计算
    EdgeTime = []

    x = torch.rand(size=(1, 3, 224, 224))
    x = x.to(device)

    if device == "cuda":
        function.warmUpGpu(model, x, device)
    if device == "cpu":
        function.warmUpCpu(model, x, device)

    for i in range(len(model)):
        layer = model[i]
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
            x = layer(x)
            continue
        time = None
        if device == "cuda":
            x, time = function.recordTimeGpu(layer, x, device, 300)
        if device == "cpu":
            x, time = function.recordTimeCpu(layer, x, device, 30)
        print(f"{layer} \t\t computation time :{time:.3f}")
        EdgeTime.append(time)
    return EdgeTime


def requestTransportTime(model_selection):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_data = torch.rand(size=(1, 3, 224, 224))
    input_data = input_data.to(device)

    transportModel = joblib.load("../model/transformTime.m")
    model = function.getDnnModel(model_selection)
    model = model.to(device)

    x = input_data
    transportTime = []
    for point_index in range(len(model)):
        layer = model[point_index]
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
            x = layer(x)
            continue
        else:
            x = layer(x)
            transportTime.append(functionImg.predictTransportTime(transportModel, x))
    return transportTime


def getBestPartitionPoint(model_selection,cloud_time,edge_time,transport_time):
    model_names = ["alexnet", "vgg16", "googLeNet", "resnet18", "mobileNetv2"]

    transportModel = joblib.load("../model/transformTime.m")
    path = "../res/cpu_gpu.xls"
    sheet_name = model_names[model_selection-1]

    layerName = function.get_excel_data(path, sheet_name, "layerName")
    shape = function.get_excel_data(path, sheet_name, "shape")
    transport_latency = function.get_excel_data(path, sheet_name, "transport_latency")

    edge_latency = function.get_excel_data(path, sheet_name, "edge_latency")
    cloud_latency = function.get_excel_data(path, sheet_name, "cloud_latency")
    end_to_end_latency = function.get_excel_data(path, sheet_name, "end-to-end latency")
    length = len(layerName)

    input_data = torch.rand(size=(1, 3, 224, 224))
    input_data = input_data.to(device)

    edge_latency_predict = [0.0]
    edge_latency_predict.extend(edge_time)

    cloud_latency_predict = cloud_time
    cloud_latency_predict.append(0.0)

    transport_latency_predict = [functionImg.predictTransportTime(transportModel, input_data)]
    transport_latency_predict.extend(transport_time)

    edge_latency_predict = function.addList(edge_latency_predict)
    cloud_latency_predict = function.addListReverse(cloud_latency_predict)

    real_partition = 0
    real_min_time = end_to_end_latency[0]
    predict_partition = 0
    predict_min_time = edge_latency_predict[0] + transport_latency_predict[0] + cloud_latency_predict[0]
    for i in range(len(layerName)):
        print(layerName[i])
        print(
            f"real time:  shape:{shape[i]}\t\tedge latency:{edge_latency[i]:.3f} ms\t\ttransport time:{transport_latency[i]:.3f} ms\t\t"
            f"cloud latency{cloud_latency[i]:.3f} ms\t\tend to end latency{end_to_end_latency[i]:.3f} ms")
        print(
            f"predict  :  shape:{shape[i]}\t\tedge latency:{edge_latency_predict[i]:.3f} ms\t\ttransport time:{transport_latency_predict[i]:.3f} ms\t\t"
            f"cloud latency{cloud_latency_predict[i]:.3f} ms\t\tend to end latency:{edge_latency_predict[i] + transport_latency_predict[i] + cloud_latency_predict[i]:.3f} ms")
        if end_to_end_latency[i] < real_min_time:
            real_partition = i
            real_min_time = end_to_end_latency[i]
        if edge_latency_predict[i] + transport_latency_predict[i] + cloud_latency_predict[i] < predict_min_time:
            predict_partition = i
            predict_min_time = edge_latency_predict[i] + transport_latency_predict[i] + cloud_latency_predict[i]
        print("================================================================")

    print(f"real partition point : {real_partition}  layer name: {layerName[real_partition]}")
    print(f"predict partition point : {predict_partition}  layer name: {layerName[predict_partition]}")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_selection = 5
    ip = "127.0.0.1"
    # ip = "112.86.199.171"
    port = 8090

    cloud_time = requestCloudTime(ip, port, model_selection)
    edge_time = requestEdgeTime(model_selection)
    transport_time = requestTransportTime(model_selection)
    print("======================================================================================================================")
    getBestPartitionPoint(model_selection,cloud_time, edge_time, transport_time)

