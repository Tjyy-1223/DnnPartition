import joblib
import numpy
import torch
import function
import functionImg
import numpy as np
import torch.nn as nn


def save_model(x,y):
    save_flag = True
    modelPath = "../model/conv2d_mac.m"
    labely = "computation time(ms)"

    devide_n = 3
    """
     cuda :
        linear test MSE: 12.583123707667841
        polynomial test MSE: 36.11452865734343

        paper polynomial : x1 - input channel       x2 - map * map * output channel 
        test MSE: 288.21331425726095 
    """
    functionImg.myLinearRegression(x, y, "in_channel * out_channel * kernel_size^2 * out_map^2", labely,devide_n,save_flag,modelPath)
    # functionImg.myPolynomialRegression(x, y, "in_channel * out_channel * kernel_size^2 * out_map^2", labely,devide_n,save_flag,modelPath)


def compareData_cuda():
    modelPath = "../model/conv2d_cuda.m"
    conv2d_reg = joblib.load(modelPath)

    input1 = torch.rand(size=(1, 3, 224, 224))
    myConv2d1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)

    input2 = torch.rand(size=(1, 64, 112, 112))
    myConv2d2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)

    input3 = torch.rand(size=(1, 256, 28, 28))
    myConv2d3 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)


    print("cuda :")
    print("real time: 0.328 ms     predict:\t",round(functionImg.predictConv2dTime(conv2d_reg, myConv2d1, input1), 3)," ms")
    print("real time: 0.707 ms     predict:\t",round(functionImg.predictConv2dTime(conv2d_reg, myConv2d2, input2), 3)," ms")
    print("real time: 1.703 ms     predict:\t",round(functionImg.predictConv2dTime(conv2d_reg, myConv2d3, input3), 3)," ms")



def compareData_mac():
    modelPath = "../model/conv2d_mac.m"
    conv2d_reg = joblib.load(modelPath)

    input1 = torch.rand(size=(1, 3, 224, 224))
    myConv2d1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)

    input2 = torch.rand(size=(1, 64, 224, 224))
    myConv2d2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)

    input3 = torch.rand(size=(1,64,112,112))
    myConv2d3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)

    input4 = torch.rand(size=(1, 128, 112, 112))
    myConv2d4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

    input5 = torch.rand(size=(1, 256,28,28))
    myConv2d5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

    print("cuda :")
    print("real time: 2.088 ms     predict:\t",round(functionImg.predictConv2dTime(conv2d_reg, myConv2d1, input1), 3)," ms")
    print("real time: 23.281 ms     predict:\t",round(functionImg.predictConv2dTime(conv2d_reg, myConv2d2, input2), 3)," ms")
    print("real time: 11.392 ms     predict:\t",round(functionImg.predictConv2dTime(conv2d_reg, myConv2d3, input3), 3)," ms")
    print("real time: 20.531 ms     predict:\t", round(functionImg.predictConv2dTime(conv2d_reg, myConv2d4, input4), 3)," ms")
    print("real time: 10.118 ms     predict:\t", round(functionImg.predictConv2dTime(conv2d_reg, myConv2d5, input5), 3)," ms")







if __name__ == '__main__':
    path = "../res/Conv2d_time.xls"
    sheet_name = "mac"
    # sheet_name = "kernel"

    in_channel = function.get_excel_data(path,sheet_name,"in_channel")
    in_map = function.get_excel_data(path,sheet_name,"in_map")
    kernel_size = function.get_excel_data(path,sheet_name,"kernel size")
    stride = function.get_excel_data(path,sheet_name,"stride")
    padding = function.get_excel_data(path,sheet_name,"padding")
    computation_number = function.get_excel_data(path,sheet_name,"computation number")
    out_channel = function.get_excel_data(path, sheet_name, "out_channel")
    out_map = function.get_excel_data(path, sheet_name, "out_map")
    computation_time = function.get_excel_data(path, sheet_name, "computation time")

    """
            绘制一波散点图
        """
    in_channel = np.array(in_channel)
    in_map = np.array(in_map)
    kernel_size = np.array(kernel_size)
    stride = np.array(stride)
    padding = np.array(padding)
    computation_number = np.array(computation_number)
    out_channel = np.array(out_channel)
    out_map = np.array(out_map)

    number = in_channel * out_channel * kernel_size * kernel_size * out_map * out_map
    y = np.array(computation_time)

    one = np.ones(len(in_channel))
    var1 = in_channel
    var2 = out_channel * out_map * out_map
    x = np.c_[one, var1, var2]

    # save_model(number,y)

    compareData_cuda()

    # compareData_mac()



