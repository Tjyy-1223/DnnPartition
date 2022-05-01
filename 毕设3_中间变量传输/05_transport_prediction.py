import torch
import function
import functionImg
import numpy as np
import joblib
import torch.nn as nn
import a1_alexNet


def alexNetPrediction():
    lin_reg = joblib.load("../model/transformTime.m")
    # lin_reg = joblib.load("../model/tranportTime_fast.m")

    """
    x = [28224,44100,63504,86436,112896,142884,176400,213444,254016,298116,345744,396900,451584]
    x = np.array(x)
    X = myTransform(x,degree=3)
    print(lin_reg.predict(X))
    """
    true_time = [119.166,148.398,35.81,105.005,28.324,41.525,35.808,54.218,17.055,8.159,11.091,6.177,6.047,0.18]
    transport_time_list = []

    # 使用lin_reg计算alexnet每层的传输时延
    alexnet = a1_alexNet.AlexNet()

    x = torch.rand(size=(1, 3, 224, 224))
    tranport_time = functionImg.predictTransportTime(lin_reg, x)
    print(f"{x.shape}\t\treal transport time {true_time[0]}ms\t\tpredict transport time {tranport_time:.3f} ms")
    transport_time_list.append(tranport_time)
    print("======================================")

    index = 1
    for layer in alexnet:
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
            continue
        x = layer(x)
        tranport_time = functionImg.predictTransportTime(lin_reg, x)
        print(f"{layer}\n{x.shape}\t\treal transport time {true_time[index]}ms\t\tpredict transport time {tranport_time:.3f} ms")
        transport_time_list.append(tranport_time)
        print("======================================")
        index += 1
    print(transport_time_list)




if __name__ == '__main__':
    path = "../res/transport_time.xls"
    sheet_name = "time1"

    # function.read_excel_xls(path,sheet_name)

    shape_prod = function.get_excel_data(path,sheet_name,"shape's prod")
    data_length = function.get_excel_data(path,sheet_name,"dumps length")
    transport_time = function.get_excel_data(path,sheet_name,"transport time(ms)")

    shape_size = np.array(shape_prod)
    data_length = np.array(data_length)
    transport_time = np.array(transport_time)

    # 线性回归
    # functionImg.getScatterImg(shape_size,data_length,"Data Size","Transmission Latency(ms)")
    # functionImg.myLinearRegression(x,y,"data's length","transport time(ms)")

    # modelPath = "../model/tranportTime_fast.m"
    # 多项式回归 no Ridge
    # lin_reg = functionImg.myPolynomialRegression(shape_size,transport_time,"Data Size","Transmission Latency(ms)")
    # print(lin_reg.coef_)

    # 多项式回归 Ridge
    # lin_reg = myPolynomialRidgeRegression(x,y)


    alexNetPrediction()







