import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt

import function
import functionImg
import torch.nn as nn
import model_features


def get_model_flops_params(model,x):
    flops_sum = 0.0
    params_sum = 0.0
    for i in range(len(model)):
        layer = model[i]
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
            x = layer(x)
            continue
        else:
            flops = model_features.get_layer_FLOPs(layer, x)
            params = model_features.get_layer_Params(layer, x)

            x = layer(x)

            flops_sum += flops
            params_sum += params
    return flops_sum,params_sum


def showDnnPartition(model_index):
    model_names = ["alexnet", "vgg16", "googLeNet", "resnet18", "mobileNetv2"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "../res/cpu_gpu.xls"
    sheet_name = model_names[model_index - 1]
    # sheet_name = "vgg16"
    """
        查看xls数据
    """
    # function.read_excel_xls(path,sheet_name)

    """
        读取各个列中的数据
    """
    layerName_exl = function.get_excel_data(path, sheet_name, "layerName")
    shape = function.get_excel_data(path, sheet_name, "shape")
    transport_latency = function.get_excel_data(path, sheet_name, "transport_latency")

    edge_latency = function.get_excel_data(path, sheet_name, "edge_latency")
    cloud_latency = function.get_excel_data(path, sheet_name, "cloud_latency")
    end_to_end_latency = function.get_excel_data(path, sheet_name, "end-to-end latency")

    length = len(layerName_exl)

    """
        开始尝试自动划分的步骤 使用建立好的模型
    """
    input_data = torch.rand(size=(1, 3, 224, 224))
    input_data = input_data.to(device)

    model = function.getDnnModel(model_index)
    model = model.to(device)

    # function.show_features_gpu(alexnet,input_data)

    """
        通过模型预测每层的时间
    """
    transportModel = joblib.load("../model/transformTime.m")
    flops_params_model_mac = joblib.load("../model/flops_params_time_mac.m")
    flops_params_model_cuda = joblib.load("../model/flops_params_time_cuda.m")

    # layerName = ["None"]
    layerName = []
    # shape_list = [f"{input_data.shape}"]
    shape_list = []
    # transport_list = [functionImg.predictTransportTime(transportModel,input_data)]
    transport_list = []
    edgeLatency_list = []
    cloudLatency_list = []

    x = input_data
    flops_sum = 0.0
    params_sum = 0.0

    all_flops,all_params = get_model_flops_params(model,input_data)
    cloud_all_time = functionImg.predictFlopsParamsTime(flops_params_model_cuda, all_flops / 10000, all_params / 10000)
    print(f"cloud run time for entire model : {cloud_all_time:.3f} ms")
    print("==========================================================================================================")

    for point_index in range(len(model)):
        layer = model[point_index]
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
            x = layer(x)
            continue
        else:
            flops = model_features.get_layer_FLOPs(layer, x)
            params = model_features.get_layer_Params(layer, x)

            x = layer(x)
            layerName.append(f"{layer}")
            shape_list.append(f"{x.shape}")

            flops_sum += flops
            params_sum += params

            edgeLatency_list.append(functionImg.predictFlopsParamsTime(flops_params_model_mac, flops_sum / 10000, params_sum / 10000))
            cloudLatency_list.append(cloud_all_time - functionImg.predictFlopsParamsTime(flops_params_model_cuda, flops_sum / 10000, params_sum / 10000))
            transport_list.append(functionImg.predictTransportTime(transportModel, x))


    edge_latency_predict = [0.0]
    edge_latency_predict.extend(edgeLatency_list)

    cloud_latency_predict = [cloud_all_time]
    cloud_latency_predict.extend(cloudLatency_list)

    transport_latency_predict = [functionImg.predictTransportTime(transportModel, input_data)]
    transport_latency_predict.extend(transport_list)


    real_partition = 0
    real_min_time = end_to_end_latency[0]
    predict_partition = 0
    predict_min_time = edge_latency_predict[0] + transport_latency_predict[0] + cloud_latency_predict[0]

    end_to_end_predict = []
    for i in range(len(layerName_exl)):
        print(layerName_exl[i])
        print(
            f"real time:  shape:{shape[i]}\t\tedge latency:{edge_latency[i]:.3f} ms\t\ttransport time:{transport_latency[i]:.3f} ms\t\t"
            f"cloud latency{cloud_latency[i]:.3f} ms\t\tend to end latency{end_to_end_latency[i]:.3f} ms")
        print(
            f"predict  :  shape:{shape[i]}\t\tedge latency:{edge_latency_predict[i]:.3f} ms\t\ttransport time:{transport_latency_predict[i]:.3f} ms\t\t"
            f"cloud latency{cloud_latency_predict[i]:.3f} ms\t\tend to end latency:{edge_latency_predict[i] + transport_latency_predict[i] + cloud_latency_predict[i]:.3f} ms")
        end_to_end_predict.append(edge_latency_predict[i] + transport_latency_predict[i] + cloud_latency_predict[i])
        if end_to_end_latency[i] < real_min_time:
            real_partition = i
            real_min_time = end_to_end_latency[i]
        if edge_latency_predict[i] + transport_latency_predict[i] + cloud_latency_predict[i] < predict_min_time:
            predict_partition = i
            predict_min_time = edge_latency_predict[i] + transport_latency_predict[i] + cloud_latency_predict[i]
        print("==========================================================================================================")

    print(f"real partition point : {real_partition}  layer name: {layerName_exl[real_partition]}"
          f"   best time : {end_to_end_latency[real_partition]:.3f}")
    print(f"predict partition point : {predict_partition}  layer name: {layerName_exl[predict_partition]}"
          f"   best time : {edge_latency_predict[predict_partition] + transport_latency_predict[predict_partition] + cloud_latency_predict[predict_partition]:.3f}")
    return end_to_end_latency,end_to_end_predict


def getImg(true_time,predict_time):
    # true_time = [119.166, 148.398, 35.81, 105.005, 28.324, 41.525, 35.808, 54.218, 17.055, 8.159, 11.091, 6.177, 6.047,
    #              0.18]
    # predict_time = [85.930, 110.725, 30.219, 80.007, 23.026, 39.600, 28.492, 28.492, 11.475, 11.475, 11.475, 8.964,
    #                 8.964, 7.452]

    # print(index)
    # print(times)

    ind = np.arange(len(true_time))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig = plt.figure(figsize=(8, 3.5))
    ax1 = fig.add_subplot(111)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xticks(ind)
    # ax1.set_xticklabels(('input', 'conv1', 'maxPool2d1', 'conv2', 'maxPool2d2', 'conv3',
    #                      'conv4', 'conv5', 'maxPool2d3', 'avgPool2d', 'flatten',
    #                      'linear1', 'linear2', 'linear3'), rotation=-45)
    ax1.get_xaxis().set_visible(False)  # 隐藏x坐标轴

    lns1 = ax1.bar(ind - width / 2, true_time, width, color='olivedrab',label='real time')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores')

    lns2 = ax1.bar(ind + width / 2, predict_time, width, color='goldenrod', label='predict time')

    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    # ax1.title("")
    ax1.set_ylabel('End-to-end Latency(ms)')

    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.title("Vgg16Net")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model_index = 2
    end_to_end_latency,end_to_end_predict = showDnnPartition(model_index)
    getImg(end_to_end_latency,end_to_end_predict)








