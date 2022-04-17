import torch
import function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import joblib

def getScatterImg(x,y):
    s = 50

    fig = plt.figure(figsize=(8, 5))

    plt.scatter(x, y, s, c="g", alpha=0.5)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    # 设置坐标轴范围
    max_X = np.max(x)
    max_Y = np.max(y)
    plt.xlim((0, max_X))
    plt.ylim((0, max_Y))

    plt.xlabel("data's length")
    plt.ylabel("transport latency(ms)")

    # 设置坐标轴刻度
    # my_x_ticks = np.arange(0, max_X)
    # my_y_ticks = np.arange(0, max_Y)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)

    plt.show()


def getLinearImg(x,y,coef,intercept):
    """
    :param coef:  16.789
    :param intercept: 425.863
    """
    s = 50
    fig = plt.figure(figsize=(8, 5))

    plt.scatter(x, y, s, c="g", alpha=0.5)
    plt.plot(x,coef[0] * x + intercept)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # 设置坐标轴范围
    max_X = np.max(x)
    max_Y = np.max(y)
    plt.xlim((0, max_X))
    plt.ylim((0, max_Y))

    plt.xlabel("data's length")
    plt.ylabel("transport latency(ms)")

    plt.show()


def myLinearRegression(x,y):
    """
       coef:  16.789
       intercept: 425.863
    """
    X = x.reshape(-1,1)

    linreg = LinearRegression()
    linreg.fit(X,y)

    y_pred = linreg.predict(X)
    print("test MSE:",metrics.mean_squared_error(y_pred,y))

    s = 50

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s, c="g", alpha=0.5)
    plt.plot(np.sort(x), y_pred[np.argsort(x)])

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # 设置坐标轴范围
    max_X = np.max(x)
    max_Y = np.max(y)
    plt.xlim((0, max_X))
    plt.ylim((0, max_Y))

    plt.xlabel("data's shape")
    plt.ylabel("data's length")

    plt.show()


def myTransform(x,degree = 1):
    X = x.reshape(-1, 1)

    poly = PolynomialFeatures(degree=degree)
    poly.fit(X)
    X2 = poly.transform(X)

    return X2


def myPolynomialRegression(x,y):
    s = 50

    X2 = myTransform(x,degree=3)

    lin_reg = LinearRegression()
    lin_reg.fit(X2, y)

    y_predict = lin_reg.predict(X2)
    print("test MSE:", metrics.mean_squared_error(y_predict, y))
    # print("斜率参数: ",lin_reg.coef_)
    # print("截距参数: ",lin_reg.intercept_)
    """
    degree = 2
    斜率参数: [0.00000000e+00 2.11090193e-04 5.44408375e-10]
    截距参数: 40.11139070205547
    
    degree = 3
    斜率参数:  [0.00000000e+00 4.87227671e-04 2.36099771e-10 8.03935911e-17]
    截距参数:  6.9646173709356844
    """

    plt.scatter(x, y, s, c="g", alpha=0.5)
    plt.scatter(x,y_predict)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # 设置坐标轴范围
    max_X = np.max(x)
    max_Y = np.max(y)
    plt.xlim((0, max_X))
    plt.ylim((0, max_Y))

    plt.xlabel("data's shape")
    plt.ylabel("transport latency(ms)")

    plt.show()
    joblib.dump(lin_reg,'../model/transformTime.m')
    return lin_reg


def myPolynomialRidgeRegression(x, y):
    s = 50
    X2 = myTransform(x, degree=3)

    lin_reg = Ridge()
    lin_reg.fit(X2, y)

    y_predict = lin_reg.predict(X2)
    print("test MSE:", metrics.mean_squared_error(y_predict, y))
    # print("斜率参数: ",lin_reg.coef_)
    # print("截距参数: ",lin_reg.intercept_)
    """
    degree = 2
    斜率参数: [0.00000000e+00 2.11090193e-04 5.44408375e-10]
    截距参数: 40.11139070205547

    degree = 3
    斜率参数:  [0.00000000e+00 4.87227671e-04 2.36099771e-10 8.03935911e-17]
    截距参数:  6.9646173709356844
    """

    plt.scatter(x, y, s, c="g", alpha=0.5)
    plt.scatter(x, y_predict)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # 设置坐标轴范围
    max_X = np.max(x)
    max_Y = np.max(y)
    plt.xlim((0, max_X))
    plt.ylim((0, max_Y))

    plt.xlabel("data's shape")
    plt.ylabel("transport latency(ms)")

    plt.show()
    return lin_reg


if __name__ == '__main__':
    path = "../res/transport_time.xls"
    sheet_name = "time1"

    # function.read_excel_xls(path,sheet_name)

    shape_prod = function.get_excel_data(path,sheet_name,"shape's prod")
    data_length = function.get_excel_data(path,sheet_name,"dumps length")
    transport_time = function.get_excel_data(path,sheet_name,"transport time(ms)")

    x = np.array(shape_prod)
    y = np.array(transport_time)

    # 线性回归
    # myLinearRegression(x,y)

    # 多项式回归 no Ridge
    # lin_reg = myPolynomialRegression(x,y)
    # print(lin_reg.coef_)

    # 多项式回归 Ridge
    # lin_reg = myPolynomialRidgeRegression(x,y)


    lin_reg = joblib.load("../model/transformTime.m")
    x = [28224,44100,63504,86436,112896,142884,176400,213444,254016,298116,345744,396900,451584]
    x = np.array(x)
    X = myTransform(x,degree=3)
    print(lin_reg.predict(X))



