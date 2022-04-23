import joblib
import torch
import function
import functionImg
import a2_vggNet
import torch.nn as nn
import a1_alexNet

def addList(mylist):
    for i in range(len(mylist)):
        if i == 0:
            continue
        else:
            mylist[i] = mylist[i-1] + mylist[i]
    return mylist


def addListReverse(mylist):
    length = len(mylist)
    for i in range(length-1,-1,-1):
        if i == length - 1:
            continue
        else:
            mylist[i] = mylist[i+1] + mylist[i]
    return mylist


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "../res/cpu_gpu.xls"
    sheet_name = "alexnet"
    # sheet_name = "vgg16"
    """
        查看xls数据
    """
    # function.read_excel_xls(path,sheet_name)

    """
        读取各个列中的数据
    """
    layerName_exl = function.get_excel_data(path,sheet_name,"layerName")
    shape = function.get_excel_data(path,sheet_name,"shape")
    transport_latency = function.get_excel_data(path,sheet_name,"transport_latency")

    edge_latency = function.get_excel_data(path,sheet_name,"edge_latency")
    cloud_latency = function.get_excel_data(path,sheet_name,"cloud_latency")
    end_to_end_latency = function.get_excel_data(path,sheet_name,"end-to-end latency")

    length = len(layerName_exl)

    """
        开始尝试自动划分的步骤 使用建立好的模型
    """
    input_data = torch.rand(size=(1,3,224,224))
    input_data = input_data.to(device)

    alexnet = a1_alexNet.AlexNet()
    # alexnet = a2_vggNet.vgg16()
    alexnet = alexnet.to(device)

    # function.show_features_gpu(alexnet,input_data)


    """
        通过模型预测每层的时间
    """
    transportModel = joblib.load("../model/transformTime.m")
    linearModel_cuda = joblib.load("../model/linearTime_cuda.m")
    linearModel_mac = joblib.load("../model/linearTime_mac.m")
    maxPool2dModel_cuda = joblib.load("../model/maxPool2dTime_cuda.m")
    maxPool2dModel_mac = joblib.load("../model/maxPool2dTime_mac.m")
    conv2dModel_cuda = joblib.load("../model/conv2d_cuda.m")
    # conv2dModel_mac = joblib.load("../model/conv2d_mac.m")
    conv2dModel_mac = joblib.load("../model/conv2d_mac2.m")


    # layerName = ["None"]
    layerName = []
    # shape_list = [f"{input_data.shape}"]
    shape_list = []
    # transport_list = [functionImg.predictTransportTime(transportModel,input_data)]
    transport_list = []
    edgeLatency_list = []
    cloudLatency_list = []

    x = input_data
    for point_index in range(len(alexnet)):
        layer = alexnet[point_index]
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
            x = layer(x)
            continue
        else:
            if isinstance(layer,nn.Linear):
                edgeLatency_list.append(functionImg.predictLinearTime(linearModel_mac,layer))
                cloudLatency_list.append(functionImg.predictLinearTime(linearModel_cuda,layer))
            elif isinstance(layer,nn.MaxPool2d):
                edgeLatency_list.append(functionImg.predictMaxPool2dTime(maxPool2dModel_mac,layer,x))
                cloudLatency_list.append(functionImg.predictMaxPool2dTime(maxPool2dModel_cuda,layer,x))
            elif isinstance(layer,nn.Conv2d):
                edgeLatency_list.append(functionImg.predictConv2dTime(conv2dModel_mac,layer,x))
                cloudLatency_list.append(functionImg.predictConv2dTime(conv2dModel_cuda,layer,x))
            else:
                edgeLatency_list.append(0.0)
                cloudLatency_list.append(0.0)
            x = layer(x)
            layerName.append(f"{layer}")
            shape_list.append(f"{x.shape}")
            transport_list.append(functionImg.predictTransportTime(transportModel,x))
    # cloud_latency.append(0.0)

    # for i in range(len(layerName)):
    #     print(layerName[i])
    #     print(f"real    : edge latency:{edge_latency[i+1]:.3f} ms\t\tcloud latency:{cloud_latency[i+1]:.3f} ms\t\tshape:{shape_list[i]}\t\ttransport time:{transport_latency[i+1]:.3f} ms")
    #     print(f"predict : edge latency:{edgeLatency_list[i]:.3f} ms\t\tcloud latency:{cloudLatency_list[i]:.3f} ms\t\tshape:{shape_list[i]}\t\ttransport time:{transport_list[i]:.3f} ms")
    #     print("================================================================")

    edge_latency_predict = [0.0]
    edge_latency_predict.extend(edgeLatency_list)
    cloud_latency_predict = cloudLatency_list
    cloud_latency_predict.append(0.0)
    transport_latency_predict = [functionImg.predictTransportTime(transportModel,input_data)]
    transport_latency_predict.extend(transport_list)


    dge_latency_predict = addList(edge_latency_predict)
    cloud_latency_predict = addListReverse(cloud_latency_predict)


    real_partition = 0
    real_min_time = end_to_end_latency[0]
    predict_partition = 0
    predict_min_time = edge_latency_predict[0] + transport_latency_predict[0] + cloud_latency_predict[0]
    for i in range(len(layerName_exl)):
        print(layerName_exl[i])
        print(f"real time:  shape:{shape[i]}\t\tedge latency:{edge_latency[i]:.3f} ms\t\ttransport time:{transport_latency[i]:.3f} ms\t\t"
              f"cloud latency{cloud_latency[i]:.3f} ms\t\tend to end latency{end_to_end_latency[i]:.3f} ms")
        print(f"predict  :  shape:{shape[i]}\t\tedge latency:{edge_latency_predict[i]:.3f} ms\t\ttransport time:{transport_latency_predict[i]:.3f} ms\t\t"
              f"cloud latency{cloud_latency_predict[i]:.3f} ms\t\tend to end latency:{edge_latency_predict[i] + transport_latency_predict[i] + cloud_latency_predict[i]:.3f} ms")
        if end_to_end_latency[i] < real_min_time:
            real_partition = i
            real_min_time = end_to_end_latency[i]
        if edge_latency_predict[i] + transport_latency_predict[i] + cloud_latency_predict[i] < predict_min_time:
            predict_partition = i
            predict_min_time = edge_latency_predict[i] + transport_latency_predict[i] + cloud_latency_predict[i]
        print("================================================================")

    print(f"real partition point : {real_partition}  layer name: {layerName_exl[real_partition]}")
    print(f"predict partition point : {predict_partition}  layer name: {layerName_exl[predict_partition]}")





