import joblib
import torch
import function
import functionImg
import numpy as np
import torch.nn as nn





if __name__ == '__main__':
    path = "../res/linear_time.xls"
    sheet_name = "mac"

    input_size = function.get_excel_data(path,sheet_name,"inputSize")
    output_size = function.get_excel_data(path,sheet_name,"outputSize")
    computation_time = function.get_excel_data(path,sheet_name,"computation time(ms)")


    """
        绘制一波散点图
    """
    x1 = np.array(input_size)
    x2 = np.array(output_size)
    x = x1 * x2
    y = np.array(computation_time)
    labelx = "input size * output size"
    labely = "computation time(ms)"


    save_flag = False
    modelPath = "../model/linearTime_mac.m"
    # functionImg.getScatterImg(x,y,labelx,labely)

    # functionImg.myLinearRegression(x,y,labelx,labely,save_flag,modelPath)

    # functionImg.myPolynomialRegression(x,y)

    lin_reg = joblib.load(modelPath)

    linear1 = nn.Linear(9216,4096)
    linear2 = nn.Linear(4096,4096)
    linear3 = nn.Linear(4096,1000)
    print("real time: 6.885 ms     predict:\t",functionImg.predictLinearTime(lin_reg,linear1)," ms")
    print("real time: 3.755 ms     predict:\t",functionImg.predictLinearTime(lin_reg,linear2)," ms")
    print("real time: 0.997 ms     predict:\t",functionImg.predictLinearTime(lin_reg,linear3)," ms")



