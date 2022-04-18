import torch
import function
import functionImg
import numpy as np
import joblib
import a1_alexNet



if __name__ == '__main__':
    path = "../res/transport_time.xls"
    sheet_name = "time1"

    # function.read_excel_xls(path,sheet_name)

    shape_prod = function.get_excel_data(path,sheet_name,"shape's prod")
    data_length = function.get_excel_data(path,sheet_name,"dumps length")
    transport_time = function.get_excel_data(path,sheet_name,"transport time(ms)")

    x = np.array(shape_prod)
    y = np.array(transport_time)

    # 线性回归
    # myLinearRegression(x,y)

    # 多项式回归 no Ridge
    # lin_reg = myPolynomialRegression(x,y)
    # print(lin_reg.coef_)

    # 多项式回归 Ridge
    # lin_reg = myPolynomialRidgeRegression(x,y)


    lin_reg = joblib.load("../model/transformTime.m")

    """
    x = [28224,44100,63504,86436,112896,142884,176400,213444,254016,298116,345744,396900,451584]
    x = np.array(x)
    X = myTransform(x,degree=3)
    print(lin_reg.predict(X))
    """

    # 使用lin_reg计算alexnet每层的传输时延
    alexnet = a1_alexNet.AlexNet()

    x = torch.rand(size=(1, 3, 224, 224))
    tranport_time = functionImg.predictTransportTime(lin_reg,x)
    print(f"{x.shape}:predict transport time {tranport_time:.3f} ms")
    print("======================================")

    for layer in alexnet:
        x = layer(x)
        tranport_time = functionImg.predictTransportTime(lin_reg,x)
        print(f"{layer}\n{x.shape}:predict transport time {tranport_time:.3f} ms")
        print("======================================")





