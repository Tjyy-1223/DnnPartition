import a1_alexNet
import a2_vggNet
import a3_GoogLeNet
import a4_ResNet
import a5_MobileNet
import torch.nn as nn

model = a1_alexNet.AlexNet()
# model = a2_vggNet.vgg16()
# model  =a3_GoogLeNet.GoogLeNet()
# model = a4_ResNet.resnet18()
# model = a5_MobileNet.MobileNetV2()

layerIndex = 1
for i in range(len(model)):
    layer = model[i]
    if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
        continue
    print(f"{layerIndex}-{layer}")
    layerIndex += 1
