import torch
import function
import a2_vggNet
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


x = torch.rand(size=(1,3,224,224))
x = x.to(device)
print(f"x device : {x.device}")

vgg16 = a2_vggNet.vgg16_bn()
vgg16 = vgg16.to(device)


temp_x = x
if device == "cpu":
    x = function.show_features_cpu(vgg16,x,epoch=3)
elif device == "cuda":
    x = function.show_features_gpu(vgg16,x,epoch=10)