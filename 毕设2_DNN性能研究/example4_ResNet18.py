import torch
import function
import a4_ResNet
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


x = torch.rand(size=(1,3,224,224))
x = x.to(device)
print(f"x device : {x.device}")


resnet18 = a4_ResNet.resnet18()
resnet18 = resnet18.to(device)
print(len(resnet18))

temp_x = x
if device == "cpu":
    x = function.show_features_cpu(resnet18,x,epoch=3)
elif device == "cuda":
    x = function.show_features_gpu(resnet18,x,epoch=10)