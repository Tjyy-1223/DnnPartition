import torch
import function
import a5_MobileNet
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


x = torch.rand(size=(1,3,224,224))
x = x.to(device)
print(f"x device : {x.device}")

mobileNet = a5_MobileNet.mobilenet_v2()
mobileNet = mobileNet.to(device)
print(len(mobileNet))

temp_x = x
if device == "cpu":
    x = function.show_features_cpu(mobileNet,x,epoch=3)
elif device == "cuda":
    x = function.show_features_gpu(mobileNet,x,epoch=10)