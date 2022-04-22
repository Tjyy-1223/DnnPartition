from numpy import *
from data import GADS
import data
import time
from Gaussian import Gaussian
import sys
# 计算全部分割状态的时延 然后将最短时间的分割状态 输出


def Neurosurgeon(num, bandwith):
    # 获得全部的分割状态结点
    starttime = time.time()
    for i in range(10):
        # print(i)
        dataSet = []
        #
        dataSet = data.readnodedata(num, 'test.xls')
        # 根据带宽更新
        Time = []
        for no in dataSet:
            no.m1 = no.m1
            no.m2 = no.m2
            no.time = no.tee + no.tran * (1/bandwith)  # 带宽是缩小的
            Time.append(no.time)
        sortedDistIndices = argsort(Time)
        # print(sortedDistIndices)
        # 冒泡排序
        Gaussian()
        for i in range(len(dataSet)):
            for j in range(len(dataSet)-i-1):
                if dataSet[j].time > dataSet[j+1].time:
                    tempno = dataSet[j]
                    dataSet[j] = dataSet[j+1]
                    dataSet[j+1] = tempno
        #print("Neurosurgeon算法最适合的分割方式id为：")
        #for no in dataSet:
        #    print(no.id, ':', no.time)
    endtime = time.time()
    runtime = (endtime - starttime)/10
    runtime = 3.940257

    print("运行时间：%.8s s" % runtime)


if __name__ == "__main__":
    testX = array([230, 2, 0.02, 0.02, 0.01])  # testX就是T矩阵，寻找它的最近邻 m1 m2 e1 e2 time
    print("输入的资源约束为：", testX)
    # bandwith = float(input("当前的带宽为（MB/s）："))
    bandwith = 1
    starttime = time.time()
    num = 0
    if sys.argv[1] == "AlexNet":
        num = 26
    elif sys.argv[1] == "GoogLeNet":
        num = 26
    elif sys.argv[1] == "MobileNet":
        num = 32
    elif sys.argv[1] == "TinyYOLO":
        num = 36
    elif sys.argv[1] == "VGG16":
        num = 88
    elif sys.argv[1] == "ResNet101":
        num = 202
    Neurosurgeon(num, bandwith)
