import function
import functionImg
import model_features
import a1_alexNet
import torch
import numpy as np


def show_FLOPs_features_gpu(model,x,device="cuda",epoch=300):
    model_len = len(model)

    flops_list = []
    params_list = []
    time_list = []

    for i in range(model_len):
        flops_sum = 0.0
        params_sum = 0.0
        computation_time = 0.0
        index = 1
        now_x = x
        for j in range(0,i+1):
            layer = model[j]
            now_x,myTime = function.recordTimeGpu(layer,now_x,device,epoch)
            flops = model_features.get_layer_FLOPs(layer,now_x)
            params = model_features.get_layer_Params(layer,now_x)

            computation_time += myTime
            flops_sum += flops
            params_sum += params

            # print(f"{index} - {layer}   computation number : {flops} \t params : {params} \t layer computation time : {myTime:.3f}")
            index += 1

        flops_list.append(flops_sum)
        params_list.append(params_sum)
        time_list.append(computation_time)

        print(f"flops: {flops_sum}  \t  params: {params_sum}  \t  computation time: {computation_time:.3f} (ms)")
        print("=============================================================")
    return flops_list,params_list,time_list


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alexnet = a1_alexNet.AlexNet()
    alexnet.to(device)

    x = torch.rand(size = (1,3,224,224))
    x = x.to(device)

    function.warmUpGpu(alexnet,x,device)
    # function.show_features_gpu(alexnet,x)
    flops,params,times = show_FLOPs_features_gpu(alexnet,x)

    flops = np.array(flops)
    params = np.array(params)
    times = np.array(times)

    functionImg.getScatterImg(flops,times,"flops","times(ms)")
    functionImg.getScatterImg(params,times,"params","times(ms)")

   

