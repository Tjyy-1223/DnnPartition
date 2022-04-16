import torch
import function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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


def getImgWithLine(x,y,coef,intercept):
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

def myLinearRegression(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
    # print(X_train.shape)
    # print(y_train.shape)

    linreg = LinearRegression()
    linreg.fit(X_train,y_train)

    y_pred = linreg.predict(X_test)
    print("test RMSE:",metrics.mean_squared_error(y_pred,y_test))

    return linreg.coef_,linreg.intercept_


if __name__ == '__main__':
    path = "../res/transport_time.xls"
    sheet_name = "time1"

    # function.read_excel_xls(path,sheet_name)

    shape_prod = function.get_excel_data(path,sheet_name,"shape's prod")
    data_length = function.get_excel_data(path,sheet_name,"dumps length")
    transport_time = function.get_excel_data(path,sheet_name,"transport time(ms)")

    x = np.array(shape_prod)
    y = np.array(data_length)


    x.resize((*x.shape,1))
    coef,intercept = myLinearRegression(x,y)
    getImgWithLine(x,y,coef,intercept)






