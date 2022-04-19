import joblib
import torch
import function
import functionImg
import numpy as np
import torch.nn as nn


if __name__ == '__main__':
    path = "../res/Conv2d_time.xls"
    sheet_name = "cuda"

    kernel_size = function.get_excel_data(path,sheet_name,"kernel size")
    stride = function.get_excel_data(path,sheet_name,"stride")
    padding = function.get_excel_data(path,sheet_name,"padding")
    computation_number = function.get_excel_data(path,sheet_name,"computation number")
    computation_time = function.get_excel_data(path, sheet_name, "computation time")

    """
            绘制一波散点图
        """
    kernel_size = np.array(kernel_size)
    stride = np.array(stride)
    padding = np.array(padding)
    computation_number = np.array(computation_number)
    y = np.array(computation_time)

    labelx = "output's shape * kernel_size^2"
    labely = "computation time(ms)"

    save_flag = False
    modelPath = "../model/maxPool2dTime_mac.m"

    functionImg.getScatterImg(computation_number, y, "output's shape * kernel_size^2 * in_channel", labely)
    # functionImg.getScatterImg(computation_number/kernel_size, y, "output's shape * kernel_size * in_channel", labely)
    # functionImg.getScatterImg(computation_number / kernel_size /kernel_size, y,"output's shape * in_channel", labely)