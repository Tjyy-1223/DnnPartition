from numpy import *
from data import GADS
import data
import time

# knn是不是应该讲所有的  不太对
# 将node中的各项指标整理成矩阵形式
def createDataSet():
    group = []
    # create a matrix: each row as a sample
    nodes = data.creatdataset_knn(202, 'test.xls')  # 调用data中knn子集函数，返回待重组的分割状态
    for no in nodes:
        # unit每次都要更新
        unit = []
        unit.append(no.m1)
        unit.append(no.m2)
        unit.append(no.e1)
        unit.append(no.e2)
        unit.append(no.time)
        group.append(unit)
    group = array(group)  # 转化为矩阵
    # group = array([[1.0,0.9,4.5,3.2,5.6], [1.0,1.0,2.3,3.4,4.5], [0.1,0.2,1.2,2.3,1.0], [0.0,0.1,0.3,0.5,0.6]])
    # labels = ['A', 'A', 'B', 'B']  # four samples and two classes
    return group, nodes


# classify using kNN (k Nearest Neighbors )
# Input:      newInput: 1 x N
#             dataSet:  M x N (M samples N, features)
#             labels:   1 x M
#             k: number of neighbors to use for comparison
# Output:     the most popular class label
def kNNClassify(newInput, dataSet):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row得出行数 矩阵的第一维维度
    # step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    diff = tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = sum(squaredDiff, axis=1)  # sum is performed by row 将每一行相加
    distance = squaredDist ** 0.5
    #print("各个分割状态结点距离T的Lp距离如下：")
    #print(distance)  # 将所有的结点输出
    # step 2: sort the distance 将计算出的距离排序
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = argsort(distance)
    # fitno = nodes[sortedDistIndices[0]]  # 距离最近的结点
    '''classCount = {}  # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    '''
    return sortedDistIndices


if __name__ == "__main__":
    starttime = time.time()
    dataSet, nodes = createDataSet()
    for j in range(10):
        testX = GADS()
        '''testX.m1 = 231.5
        testX.m2 = 1.3
        testX.e1 = 0.03
        testX.e2 = 0.009
        testX.time = 0.017'''

        testX.m1 = 223.6
        testX.m2 = 9
        testX.e1 = 0.02
        testX.e2 = 0.018
        testX.time = 0.011
        '''testX.m1 = 80
        testX.m2 = 160
        testX.e1 = 0.01
        testX.e2 = 0.04
        testX.time = 0.04'''
        group = []
        group.append(testX)
        group = data.min_maxnormalization(group)
        unit = []
        unit.append(group[0].m1)
        unit.append(group[0].m2)
        unit.append(group[0].e1)
        unit.append(group[0].e2)
        unit.append(group[0].time)
        # testX = array([231.5, 1.3, 0.03, 0.009, 0.011])  # testX就是T矩阵，寻找它的最近邻 m1 m2 e1 e2 time
        sortedDistIndices = kNNClassify(unit, dataSet)
        #print("输入的资源约束为：", unit, "knn搜索后的分割方式id为：")
        #for i in range(len(sortedDistIndices)):
        #    print(nodes[sortedDistIndices[i]].id)
    endtime = time.time()
    runtime = (endtime - starttime)/10
    print("运行时间：%s s" % runtime)

