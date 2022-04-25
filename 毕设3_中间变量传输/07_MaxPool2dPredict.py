import joblib
import torch
import function
import functionImg
import numpy as np
import torch.nn as nn



def compareData_cuda():
    modelPath = "../model/maxPool2dTime_cuda.m"
    maxPool2d_reg = joblib.load(modelPath)

    input1 = torch.rand(size=(1, 64, 55, 55))
    maxPool2d1 = nn.MaxPool2d(kernel_size=3, stride=2)

    input2 = torch.rand(size=(1, 192, 27, 27))
    maxPool2d2 = nn.MaxPool2d(kernel_size=3, stride=2)

    input3 = torch.rand(size=(1, 256, 13, 13))
    maxPool2d3 = nn.MaxPool2d(kernel_size=3, stride=2)

    print("cuda :")
    print("real time: 0.032 ms     predict:\t", round(functionImg.predictMaxPool2dTime(maxPool2d_reg, maxPool2d1, input1),3),
          " ms")
    print("real time: 0.024 ms     predict:\t", round(functionImg.predictMaxPool2dTime(maxPool2d_reg, maxPool2d2, input2),3),
          " ms")
    print("real time: 0.019 ms     predict:\t", round(functionImg.predictMaxPool2dTime(maxPool2d_reg, maxPool2d3, input3),3),
          " ms")


def compareData_mac():
    modelPath = "../model/maxPool2dTime_mac.m"
    maxPool2d_reg = joblib.load(modelPath)

    input1 = torch.rand(size=(1, 64, 55, 55))
    maxPool2d1 = nn.MaxPool2d(kernel_size=3, stride=2)

    input2 = torch.rand(size=(1, 192, 27, 27))
    maxPool2d2 = nn.MaxPool2d(kernel_size=3, stride=2)

    input3 = torch.rand(size=(1, 256, 13, 13))
    maxPool2d3 = nn.MaxPool2d(kernel_size=3, stride=2)

    print("mac:")
    print("real time: 0.514 ms     predict:\t", round(functionImg.predictMaxPool2dTime(maxPool2d_reg, maxPool2d1, input1),3),
          " ms")
    print("real time: 0.357 ms     predict:\t", round(functionImg.predictMaxPool2dTime(maxPool2d_reg, maxPool2d2, input2),3),
          " ms")
    print("real time: 0.107 ms     predict:\t", round(functionImg.predictMaxPool2dTime(maxPool2d_reg, maxPool2d3, input3),3),
          " ms")


if __name__ == '__main__':
    path = "../res/maxPool2d_time.xls"
    sheet_name = "mac2"

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

    functionImg.getScatterImg(computation_number, y, "output's shape * kernel_size^2", labely)
    # functionImg.getScatterImg(computation_number/kernel_size, y, "output's shape * kernel_size", labely)
    # functionImg.getScatterImg(computation_number / kernel_size /kernel_size, y,"output's shape", labely)



    """
        cuda  :test MSE: 0.33244975878841315
        cuda 2:test MSE: 0.032208524006268814
        mac   :test MSE: 0.47780505040711524
        mac 2 :test MSE: 0.06754588686307808
    """
    # functionImg.myLinearRegression(computation_number, y, "output's shape * kernel_size^2", labely, save_flag, modelPath)

    """
        cuda  :test MSE: 0.1886064618881174
        cuda 2:test MSE: 0.009826018871006745
        mac   :test MSE: 0.194717963898536
        mac 2 :test MSE: 0.020775376366165114
    """
    # functionImg.myLinearRegression(computation_number/kernel_size, y, "output's shape * kernel_size", labely, save_flag, modelPath)

    """
        cuda  :test MSE: 0.057864216993720886
        cuda 2:test MSE: 0.020728157308170698
        mac   :test MSE: 1.087570246911298
        mac 2 :test MSE: 0.0986114609574741
    """
    # functionImg.myLinearRegression(computation_number / kernel_size /kernel_size, y, "output's shape", labely,save_flag, modelPath)

    # compareData_cuda()


    # compareData_mac()

