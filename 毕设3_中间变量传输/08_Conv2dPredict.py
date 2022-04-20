import joblib
import numpy
import torch
import function
import functionImg
import numpy as np
import torch.nn as nn


if __name__ == '__main__':
    path = "../res/Conv2d_time.xls"
    sheet_name = "kernel"

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

    y = np.array(computation_time)

    labelx = "output's shape * kernel_size^2"
    labely = "computation time(ms)"

    save_flag = False
    modelPath = "../model/conv2d_cuda.m"


    """
        cuda : 0.119349
    """
    # functionImg.getScatterImg(kernel_size, y, "kernel size", labely)
    # functionImg.getScatterImg(in_channel, y, "in_channel", labely)
    # functionImg.getScatterImg(out_channel, y, "out_channel", labely)


    # functionImg.getScatterImg(in_channel * out_channel * kernel_size * kernel_size * out_map * out_map, y, "in_channel * out_channel * kernel_size^2 * out_map^2", labely)
    # functionImg.getScatterImg(in_channel * out_channel * kernel_size * kernel_size * out_map, y, "in_channel * out_channel * kernel_size^2 * out_map", labely)


    number = in_channel * out_channel * kernel_size * kernel_size * out_map * out_map
    """
        线性回归 number - time : test MSE: 12.583123707667829
    """
    # functionImg.myLinearRegression(number, y, "in_channel * out_channel * kernel_size^2 * out_map^2", labely,save_flag,modelPath)
    # functionImg.myLinearRegression((kernel_size/stride) * (kernel_size/stride) * out_channel, y, "output's shape * kernel_size^2", labely, save_flag,modelPath)
    # functionImg.myLinearRegression(in_channel * (kernel_size/stride) * (kernel_size/stride) * out_channel, y, "output's shape * kernel_size^2", labely, save_flag,modelPath)


    one = np.ones(len(in_channel))
    var1 = in_channel * out_channel * kernel_size * kernel_size * out_map * out_map
    x = np.c_[one, var1]

    devide_n = 1
    number = number
    functionImg.myLinearRegression(number, y, "in_channel * out_channel * kernel_size^2 * out_map^2", labely,devide_n,save_flag,modelPath)
    # functionImg.myPolynomialRegression(number, y, "in_channel * out_channel * kernel_size^2 * out_map^2", labely,devide_n,save_flag,modelPath)

