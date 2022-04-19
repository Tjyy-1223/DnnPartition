import joblib
import torch
import function
import functionImg
import numpy as np
import torch.nn as nn


if __name__ == '__main__':
    path = "../res/maxPool2d_time.xls"
    sheet_name = "cuda"

    kernel_size = function.get_excel_data(path,sheet_name,"kernel size")
    stride = function.get_excel_data(path,sheet_name,"stride")
    padding = function.get_excel_data(path,sheet_name,"padding")
    computation_number = function.get_excel_data(path,sheet_name,"computation number")
    computation_time = function.get_excel_data(path, sheet_name, "computation time")

    """
            绘制一波散点图
        """
    x = np.array(computation_number)
    y = np.array(computation_time)
    labelx = "output's shape * kernel_size^2"
    labely = "computation time(ms)"

    save_flag = False
    modelPath = "../model/maxPool2dTime_cuda.m"
    # functionImg.getScatterImg(x, y, labelx, labely)

    # functionImg.myLinearRegression(x, y, labelx, labely, save_flag, modelPath)

    maxPool2d_reg = joblib.load(modelPath)


    input1 = torch.rand(size=(1,64,55,55))
    maxPool2d1 = nn.MaxPool2d(kernel_size=3,stride=2)

    input2 = torch.rand(size=(1, 192, 27, 27))
    maxPool2d2 = nn.MaxPool2d(kernel_size=3, stride=2)

    input3 = torch.rand(size=(1, 256,13,13))
    maxPool2d3 = nn.MaxPool2d(kernel_size=3, stride=2)

    print("real time: 0.549 ms     predict:\t", functionImg.predictMaxPool2dTime(maxPool2d_reg,maxPool2d1,input1), " ms")
    print("real time: 0.386 ms     predict:\t", functionImg.predictMaxPool2dTime(maxPool2d_reg,maxPool2d2,input2), " ms")
    print("real time: 0.123 ms     predict:\t", functionImg.predictMaxPool2dTime(maxPool2d_reg,maxPool2d3,input3), " ms")