import time

import function
import functionImg
import model_features
import a1_alexNet
import torch
import numpy as np


def show_FLOPs_features_gpu(model,x,device="cuda",epoch=300,save = False):
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
        now_x,myTime = function.recordTimeGpu(layer,now_x,device,epoch)
        time.sleep(1)
        flops = model_features.get_layer_FLOPs(layer,now_x)
        params = model_features.get_layer_Params(layer,now_x)

        computation_time += myTime
        flops_sum += flops
        params_sum += params

        print(f"{index} - {layer}   computation number : {flops} \t params : {params} \t layer computation time : {myTime:.3f}")


        flops_list.append(flops_sum)
        params_list.append(params_sum)
        time_list.append(computation_time)

        print(f"flops: {flops_sum}  \t  params: {params_sum}  \t  computation time: {computation_time:.3f} (ms)")
        print("=============================================================")
        index += 1
    return flops_list,params_list,time_list



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flops = []
    params = []
    times = []

    for i in range(2):
        for i in range(1,3):
            model = function.getDnnModel(i)
            model = model.to(device)

            x = torch.rand(size = (1,3,224,224))
            x = x.to(device)

            if i == 1:
                function.warmUpGpu(model,x,device)

            # function.show_features_gpu(alexnet,x)
            my_flops,my_params,my_times = show_FLOPs_features_gpu(model,x)
            flops.extend(my_flops)
            params.extend(my_params)
            times.extend(my_times)

    flops = np.array(flops)
    params = np.array(params)
    times = np.array(times)

    # functionImg.getScatterImg(flops,times,"flops","times(ms)")
    # functionImg.getScatterImg(params,times,"params","times(ms)")
    model_Path = "../model/flops_time_cuda.m"
    functionImg.myPolynomialRegression_single(flops,times,"flops","times(ms)",degree=2,save=True,modelPath="../model/flops_time_cuda.m")
    functionImg.myPolynomialRegression_single(flops,times,"flops","times(ms)",degree=3)

    ones = torch.ones(len(flops))
    x = np.c_[ones,flops,params]
    functionImg.myPolynomialRegression(x,times,"y_real","y_predict",save=True,modelPath="../model/flops_params_time_cuda.m")


