# -*- coding: utf-8 -*-
# Lp距离的计算
from numpy import *
import numpy as np
import sys
import imp
import torch
imp.reload(sys)
# sys.setdefaultencoding('utf8')
import data
from math import sqrt
from collections import namedtuple
import time
from data import readnodedata
from data import upgradetime
from data import GADS
if __name__ == "__main__":
  group = []
  node = readnodedata()
  node_minmax = upgradetime(node, 1)
  for no in node_minmax:
      # unit每次都要更新
      unit = []
      unit.append(float(no.m1))
      unit.append(float(no.m2))
      unit.append(float(no.e1))
      unit.append(float(no.e2))
      unit.append(float(no.time))
      group.append(unit)

  group_1 = torch.tensor(group)
  print(type(group_1))
  testT = GADS()
  testT.m1 = 232.43
  testT.m2 = 0.1333
  testT.e1 = 0.0356
  testT.e2 = 0.00278
  testT.time = 0.05818
  group = []
  group.append(testT)
  group = data.min_maxnormalization(group)
  targetT = []
  targetT.append(float(group[0].m1))
  targetT.append(float(group[0].m2))
  targetT.append(float(group[0].e1))
  targetT.append(float(group[0].e2))
  targetT.append(float(group[0].time))
  targetT_1 = torch.tensor(targetT)
  print(type(targetT_1))
  # 在第三层之后进行分割
 # temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(group_1, targetT)))
  dist = torch.sqrt(torch.sum((group_1[:, None, :] - targetT_1) ** 2, dim=2))
  print(dist)
  array = []
  for i in range(len(dist)):
      array.append(dist[i][0])
  array = torch.tensor(array)
  print(np.argsort(array))  # 正序输出索引，从小到大
  print(np.sort(array))

