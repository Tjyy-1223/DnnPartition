import joblib
import torch
import function
import functionImg
import numpy as np
import torch.nn as nn

if __name__ == '__main__':
    # one = np.ones(3)
    # x = [4,9,16]
    # y = [2,3,4]
    #
    # x = np.array(x)
    # y = np.array(y)
    #
    # data = np.c_[one,y,one]
    #
    # print(data)

    path = "../res/Conv2d_time.xls"
    sheet_name = "kernel"

    function.read_excel_xls(path,sheet_name)