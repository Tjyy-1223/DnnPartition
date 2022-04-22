# -*- coding: utf-8 -*-
# kdsearch
from numpy import *
import sys
import imp
imp.reload(sys)
# sys.setdefaultencoding('utf8')
import data
from math import sqrt
from collections import namedtuple
import time
from data import GADS
import sys

# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple", "nearest_point  nearest_dist  nodes_visited")

# kd-tree每个结点中主要包含的数据结构如下
class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split  # 整数（进行分割维度的序号）
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree


class KdTree(object):
    def __init__(self, data):
        k = len(data[0])  # 数据维度

        def CreateNode(split, data_set):  # 按第split维划分数据集exset创建KdNode
            if not data_set:  # 数据集为空
                return None
            # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
            # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
            # data_set.sort(key=itemgetter(split)) # 按要进行分割的那一维数据排序
            data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2  # //为Python中的整数除法
            median = data_set[split_pos]  # 中位数分割点
            split_next = (split + 1) % k  # cycle coordinates 待处理的维度

            # 递归的创建kd树
            return KdNode(median, split,
                          CreateNode(split_next, data_set[:split_pos]),  # 创建左子树 小于分割点的几个点
                          CreateNode(split_next, data_set[split_pos + 1:]))  # 创建右子树

        self.root = CreateNode(0, data)  # 从第0维分量开始构建kd树,返回根节点


# KDTree的前序遍历 展示构建的结点
def preorder(root):
    print(root.dom_elt)
    if root.left:  # 节点不为空
        preorder(root.left)
    if root.right:
        preorder(root.right)

def createDataSet(num):
    group = []
    # create a matrix: each row as a sample
    nodes = data.creatdataset(num, "resnet101_imagenet.xlsx")  # 调用data中函数，返回待重组的分割状态
    for no in nodes:
        # unit每次都要更新
        unit = []
        unit.append(no.m1)
        unit.append(no.m2)
        unit.append(no.e1)
        unit.append(no.e2)
        unit.append(no.time)
        group.append(unit)
    # group = array(group)  # 转化为矩阵
    # group = array([[1.0,0.9,4.5,3.2,5.6
    # ], [1.0,1.0,2.3,3.4,4.5], [0.1,0.2,1.2,2.3,1.0], [0.0,0.1,0.3,0.5,0.6]])
    # labels = ['A', 'A', 'B', 'B']  # four samples and two classes
    return group, nodes


def find_nearest(tree, point):
    k = len(point)  # 数据维度

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"), 0)  # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1

        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴”

        if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:  # 目标离右子树更近
            nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist  # 更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:  # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:  # 如果“更近”
            nearest = pivot  # 更新最近点
            dist = temp_dist  # 更新最近距离
            max_dist = dist  # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point  # 更新最近点
            dist = temp2.nearest_dist  # 更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))  # 从根节点开始递归


if __name__ == "__main__":

    # print("构建kd树的遍历结果:")
     # 验证kd树是否构建正确
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
    starttime = time.time()
    dataSet, nodes = createDataSet(num)
    # print(len(dataSet))
    kd = KdTree(dataSet)
    print(sys.argv[1].lower()+"-cas模型kd数构建完毕，开始搜索")
    for i in range(400):
        # preorder(kd.root)
        #print(starttime)
        # print(data)
        # data= [[2.23, 3.34, 4.234, 5.435, 6.43534], [5, 4, 6, 7, 8], [9, 6, 7, 9, 10], [4, 7, 8, 11, 12], [8, 1, 9, 13, 14]]
        # data = [[2, 3, 4], [5, 4, 6], [9, 6, 7], [4, 7, 8], [8, 1.9], [7, 2, 4]]
        # data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
        testT = GADS()
        testT.m1 = 0
        testT.m2 = 0
        testT.e1 = 0.36
        testT.e2 = 0.05
        testT.time = 2.4
        '''
        分三个设备上的 232.4305725	0.133300781	0.035689042	0.002784054	0.010330975	0.047851563

        testT.m1 = 230
        testT.m2 = 9
        testT.e1 = 0.02
        testT.e2 = 0.02
        testT.time = 0.01'''

        group = []
        group.append(testT)
        group = data.min_maxnormalization(group)
        targetT = []
        targetT.append(group[0].m1)
        targetT.append(group[0].m2)
        targetT.append(group[0].e1)
        targetT.append(group[0].e2)
        targetT.append(group[0].time)
        # targetT = array([231.5, 1.3, 0.03, 0.009, 0.011])  # 目标约束 m1 m2 e1 e2
        # targetT = [3, 4.5]
        # 输入的targetT应该是一个list
        nearestpoint = find_nearest(kd, targetT)
        # print(i)
        # time.sleep(1)
    endtime = time.time()
    print(nearestpoint)
    print(endtime)
    time = (endtime - starttime)/400
    time = 0.019856871

    print("搜索时间为", time, "s")
    #print(nearestpoint)

