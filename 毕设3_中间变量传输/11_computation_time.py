import function
import model_features
import a1_alexNet
import torch


def show_FLOPs_features_gpu(model,x,device="cuda",epoch=300):
    model_len = len(model)

    for i in range(model_len):
        computation_time = 0.0
        index = 1
        now_x = x
        for j in range(0,i+1):
            layer = model[j]
            now_x,myTime = function.recordTimeGpu(layer,now_x,device,epoch)
            computation_time += myTime
            print(f"{index} - {layer}   layer computation time : {myTime:.3f}")
            index += 1
        print(f"computation time: {computation_time:.3f} (ms)")
        print("=============================================================")




if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alexnet = a1_alexNet.AlexNet()
    alexnet.to(device)

    x = torch.rand(size = (1,3,224,224))
    x = x.to(device)

    function.warmUpGpu(alexnet,x,device)
    # function.show_features_gpu(alexnet,x)
    show_FLOPs_features_gpu(alexnet,x)