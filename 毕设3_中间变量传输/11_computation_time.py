import time
from sklearn.preprocessing import MinMaxScaler
import joblib

import function
import functionImg
import model_features
import a1_alexNet
import torch
import torch.nn as nn
import numpy as np


def show_FLOPs_features(model,x,epoch=300,save_flag=False,Path=None,sheetname=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_len = len(model)

    flops_list = []
    params_list = []
    time_list = []

    now_x = x
    index = 1
    flops_sum = 0.0
    params_sum = 0.0
    computation_time = 0.0

    for i in range(model_len):
        layer = model[i]
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
            continue

        flops = model_features.get_model_FLOPs(layer, now_x)
        params = model_features.get_model_Params(layer, now_x)

        if device == "cuda":
            now_x, myTime = function.recordTimeGpu(layer, now_x, device, epoch)
        elif device == "cpu":
            now_x, myTime = function.recordTimeCpu(layer, now_x, device, epoch)
        else:
            myTime = 0.0

        computation_time += myTime
        flops_sum += flops
        params_sum += params

        print( f"{index} - {layer}   computation number : {flops} \t params : {params} \t layer computation time : {myTime:.3f}")

        flops_list.append(flops_sum)
        params_list.append(params_sum)
        time_list.append(computation_time)

        if save_flag:
            value = [[flops_sum,params_sum,round(flops_sum/10000,3),round(params_sum/10000,3),computation_time]]
            function.write_excel_xls_append(Path, sheetname, value)
        print(f"flops: {flops_sum}  \t  params: {params_sum}  \t  computation time: {computation_time:.3f} (ms)")
        print("=============================================================")
        index += 1
    return flops_list,params_list,time_list



def show_FLOPs_features2(model,x,epoch=300,device ="cpu",save_flag=False,Path=None,sheetname=None):
    device = device
    model_len = len(model)

    index = 1
    for i in range(model_len):
        layer = model[i]
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
            continue

        edge_model,_ = function.model_partition(model,i+1)

        if device == "cuda":
            _, myTime = function.recordTimeGpu(edge_model, x, device, epoch)
        elif device == "cpu":
            _, myTime = function.recordTimeCpu(edge_model, x, device, epoch)
        else:
            myTime = 0.0

        flops = model_features.get_model_FLOPs(edge_model, x)
        params = model_features.get_model_Params(edge_model, x)

        print(f"{index} - {layer}   computation number : {flops} \t params : {params} \t layer computation time : {myTime:.3f}")

        if save_flag:
            value = [[flops,params,round(flops/10000,3),round(params/10000,3),myTime]]
            function.write_excel_xls_append(Path, sheetname, value)
        print(f"flops: {flops}  \t  params: {params}  \t  computation time: {myTime:.3f} (ms)")
        print("=============================================================")

        index += 1



def get_predict_data(save_flag = False):
    save_flag = save_flag
    path = "../res/computation_time.xls"
    sheet_name = "cuda_all"
    value = [["flops", "params","flops2","params2","times",]]
    if save_flag:
        function.create_excel_xsl(path, sheet_name, value)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    flops = []
    params = []
    times = []
    for i in range(1,3):
        for i in range(1,6):
            model = function.getDnnModel(i)
            model = model.to(device)

            x = torch.rand(size=(1, 3, 224, 224))
            x = x.to(device)

            if device == "cuda":
                function.warmUpGpu(model, x, device)
            if device == "cpu":
                function.warmUpCpu(model,x,device)

            # function.show_features_gpu(alexnet,x)

            """
                细粒度分层模式
            """
            # my_flops, my_params, my_times = show_FLOPs_features(model,x,save_flag=save_flag,Path=path,sheetname=sheet_name)
            # flops.extend(my_flops)
            # params.extend(my_params)
            # times.extend(my_times)


            """
                粗粒度分层模式
            """
            show_FLOPs_features2(model,x,save_flag=save_flag,device=device,Path=path,sheetname=sheet_name)




def get_predict_model():
    mm = MinMaxScaler()
    path = "../res/computation_time.xls"
    # sheet_name = "mac_one"
    sheet_name = "cuda_all"

    flops = function.get_excel_data(path,sheet_name,"flops")
    flops2 = function.get_excel_data(path,sheet_name,"flops2")
    params = function.get_excel_data(path,sheet_name,"params")
    params2 = function.get_excel_data(path,sheet_name,"params2")
    times = function.get_excel_data(path,sheet_name,"times")

    flops = np.array(flops)
    flops2 = np.array(flops2)
    params = np.array(params)
    params2 = np.array(params2)
    times = np.array(times)

    # functionImg.getScatterImg(flops,times,"flops","times(ms)")
    functionImg.getScatterImg(flops2,times,"FLOPs","Latency(ms)")
    # functionImg.getScatterImg(params,times,"params","times(ms)")
    functionImg.getScatterImg(params2,times,"Params","Latency(ms)")

    save = False
    # functionImg.myPolynomialRegression_single(flops2,times,"flops","times(ms)",degree=3)

    # functionImg.myPolynomialRegression_single(flops2, times, "flops", "times(ms)", degree=2, save=save,
    #                                           modelPath="../model/flops_all_time_cuda.m")
    # #
    # ones = torch.ones(len(flops2))
    # x = np.c_[ones,flops2,params2]
    # functionImg.myPolynomialRegression(x,times,"y_real","y_predict",save=save,modelPath="../model/flops_params_all_time_cuda.m")



def compare_alexnet():
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # flops_predict_model = joblib.load("../model/flops_time_cuda.m")
    flops_params_predict_model = joblib.load("../model/flops_params_time_cuda.m")

    # flops_params_predict_model = joblib.load("../model/flops_params_all_time_cuda.m")

    model = function.getDnnModel(1)
    model = model.to(device)

    x = torch.rand(size=(1, 3, 224, 224))
    x = x.to(device)

    if device == "cuda":
        now_x, myTime = function.recordTimeGpu(model, x, device, 3)
    elif device == "cpu":
        now_x, myTime = function.recordTimeCpu(model, x, device, 3)


    flops_sum = 0.0
    params_sum = 0.0
    computation_time = 0.0

    now_x = x
    epoch = 300
    index = 0
    for i in range(len(model)):
        layer = model[i]
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
            now_x = layer(now_x)
            continue

        flops = model_features.get_layer_FLOPs(layer, now_x)
        params = model_features.get_layer_Params(layer, now_x)

        if device == "cuda":
            now_x, myTime = function.recordTimeGpu(layer, now_x, device, epoch)
        elif device == "cpu":
            now_x, myTime = function.recordTimeCpu(layer, now_x, device, epoch)
        else:
            myTime = 0

        time.sleep(1)


        computation_time += myTime
        flops_sum += flops
        params_sum += params

        print(f"{index} - {layer}   computation number : {flops} \t params : {params} \t layer computation time : {myTime:.3f} (ms)")


        print(f"flops: {flops_sum}  \t  params: {params_sum}  \t  computation time: {computation_time:.3f} (ms) \t "
              # f"flops predict : {functionImg.predictFlopsTime(flops_predict_model,flops_sum/10000):.3f} \t "
              f"flops params predict : {functionImg.predictFlopsParamsTime(flops_params_predict_model,flops_sum/10000,params_sum/10000):.3f}")
        print("=============================================================")
        index += 1



def test_model(save_flag = False):
    save_flag = save_flag
    path = "../res/computation_time.xls"
    sheet_name = "cuda"
    value = [["flops", "params","flops2","params2","times",]]
    if save_flag:
        function.create_excel_xsl(path, sheet_name, value)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    flops = []
    params = []
    times = []
    for i in range(1,2):
        for i in range(3,4):
            model = function.getDnnModel(i)
            model = model.to(device)

            x = torch.rand(size=(1, 3, 224, 224))
            x = x.to(device)

            if device == "cuda":
                function.warmUpGpu(model, x, device)
            if device == "cpu":
                function.warmUpCpu(model,x,device)

            # function.show_features_gpu(alexnet,x)

            if device == "cuda":
                _,nowtime = function.recordTimeGpu(model,x,device,300)
            if device == "cpu":
                _,nowtime = function.recordTimeCpu(model,x,device,300)

            print("what the fuck !",nowtime)

            """
                细粒度分层模式
            """
            # my_flops, my_params, my_times = show_FLOPs_features(model,x,save_flag=save_flag,Path=path,sheetname=sheet_name)
            # flops.extend(my_flops)
            # params.extend(my_params)
            # times.extend(my_times)


            """
                粗粒度分层模式
            """
            show_FLOPs_features2(model,x,save_flag=save_flag,device=device,Path=path,sheetname=sheet_name)



if __name__ == '__main__':
    save_flag = True


    """ 获取数据 """
    # get_predict_data(save_flag)

    """ 构建模型 """
    get_predict_model()

    """ 数据比较 """
    # compare_alexnet()


    # test_model(False)







