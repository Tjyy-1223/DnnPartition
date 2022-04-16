import numpy
import torch

import function
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    path = "../res/transport_time.xls"
    sheet_name = "time1"

    # function.read_excel_xls(path,sheet_name)

    shape_prod = function.get_excel_data(path,sheet_name,"shape's prod")
    data_length = function.get_excel_data(path,sheet_name,"dumps length")
    transport_time = function.get_excel_data(path,sheet_name,"transport time(ms)")

    x = np.array(data_length)
    y = np.array(transport_time)
    s = 50

    fig = plt.figure(figsize=(8,5))

    plt.scatter(x, y, s, c="g", alpha=0.5)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    # 设置坐标轴范围
    max_X = np.max(x)
    max_Y = np.max(y)
    plt.xlim((0, max_X))
    plt.ylim((0, max_Y))

    plt.xlabel("shape's prod")
    plt.ylabel("data's length")

    # 设置坐标轴刻度
    # my_x_ticks = np.arange(0, max_X)
    # my_y_ticks = np.arange(0, max_Y)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)

    plt.show()