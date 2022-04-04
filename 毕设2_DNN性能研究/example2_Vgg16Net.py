import torch
import function
import a2_vggNet
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


x = torch.rand(size=(32,3,224,224))
x = x.to(device)
print(f"x device : {x.device}")

vgg16 = a2_vggNet.vgg16_bn()
vgg16 = vgg16.to(device)


temp_x = x
temp_x = function.show_features(vgg16,temp_x)