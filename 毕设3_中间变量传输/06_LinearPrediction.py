import torch
import function
import functionImg
import numpy as np


if __name__ == '__main__':
    path = "../res/linear_time.xls"
    sheet_name = "cuda"

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
    functionImg.getScatterImg(x,y,labelx,labely)