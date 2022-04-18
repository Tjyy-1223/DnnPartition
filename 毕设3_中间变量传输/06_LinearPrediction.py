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
import a1_alexNet


if __name__ == '__main__':
    path = "../res/linear_time.xls"
    sheet_name = "cuda"

    input_size = function.get_excel_data(path,sheet_name,"inputSize")
    output_size = function.get_excel_data(path,sheet_name,"outputSize")
    computation_time = function.get_excel_data(path,sheet_name,"computation time(ms)")
    print(len(input_size))
    print(len(output_size))
    print(len(computation_time))
