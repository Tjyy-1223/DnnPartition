import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import joblib



def getScatterImg(x,y,xlabel,ylabel):
    """
    输入 x 和 y numpy对象，获得 x，y对应的绘制散点图
    :param x: x numpy
    :param y: y numpy
    :param xlabel: x坐标标注
    :param ylabel: y坐标标注
    :return:
    """
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

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 设置坐标轴刻度
    # my_x_ticks = np.arange(0, max_X)
    # my_y_ticks = np.arange(0, max_Y)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)

    plt.show()


def getLinearImg(x,y,coef,intercept):
    """
    绘制 xy 的散点图，然后绘制一条拟合直线
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


def myLinearRegression(x,y,xlabel,ylabel,save = False,modelPath = None):
    """
       根据 x 和 y 的趋势拟合一条直线
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
    plt.scatter(x, y_pred)
    plt.plot(np.sort(x), y_pred[np.argsort(x)],c="r")

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # 设置坐标轴范围
    max_X = np.max(x)
    max_Y = np.max(y)
    plt.xlim((0, max_X))
    plt.ylim((0, max_Y))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save:
        joblib.dump(linreg,modelPath)
    plt.show()


def myTransform(x,degree = 1):
    """
    多项式回归 所需要用到的数据转换
    :param x:
    :param degree:
    :return:
    """
    X = x.reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    poly.fit(X)
    X2 = poly.transform(X)

    return X2


def myPolynomialRegression(x, y,save = False,modelPath = None):
    """
        根据多项式回归模拟图
    """
    s = 50

    X2 = myTransform(x, degree=3)

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

    if save:
        joblib.dump(lin_reg,modelPath)

    return lin_reg


def myPolynomialRidgeRegression(x, y):
    """
        Ridge 多项式回归
    """
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


def predictTransportTime(model,x):
    """
    预测变量的传输时间
    输入模型和 x
    """
    x_shape = x.shape
    prod = 1
    for i in range(len(x_shape)):
        prod *= x_shape[i]
    data_x = myTransform(np.array([prod]), degree=3)
    tranport_time = model.predict(data_x)[0]
    return tranport_time


def predictLinearTime(model,linear):
    in_features = linear.in_features
    out_features = linear.out_features

    input_features = in_features * out_features
    input_features = np.array([input_features]).reshape(-1,1)

    computation_time = model.predict(input_features)[0]
    return computation_time


