import torch
import function
import a3_GoogLeNet
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


x = torch.rand(size=(1,3,224,224))
x = x.to(device)
print(f"x device : {x.device}")


GoogLeNet = a3_GoogLeNet.GoogLeNet()
GoogLeNet = GoogLeNet.to(device)
print(len(GoogLeNet))

temp_x = x
if device == "cpu":
    x = function.show_features_cpu(GoogLeNet,x,epoch=3)
elif device == "cuda":
    x = function.show_features_gpu(GoogLeNet,x,epoch=10)