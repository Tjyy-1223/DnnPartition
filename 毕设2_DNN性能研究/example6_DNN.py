import torch
import function
import a1_alexNet
import a2_vggNet
import a3_GoogLeNet
import a4_ResNet
import a5_MobileNet


def getDnnModel(index):
    if index == 1:
        alexnet = a1_alexNet.AlexNet(input_layer=3, num_classes=1000)
        return alexnet
    elif index == 2:
        vgg16 = a2_vggNet.vgg16_bn()
        return vgg16
    elif index == 3:
        GoogLeNet = a3_GoogLeNet.GoogLeNet()
        return GoogLeNet
    elif index == 4:
        resnet18 = a4_ResNet.resnet18()
        return resnet18
    elif index == 5:
        mobileNet = a5_MobileNet.mobilenet_v2()
        return mobileNet
    else:
        print("no model")
        return None


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    x = torch.rand(size=(1, 3, 224, 224))
    x = x.to(device)
    print(f"x device : {x.device}")

    model = getDnnModel(1)
    model.to(device)
    print(len(model))

    temp_x = x
    if device == "cpu":
        x = function.show_features_cpu(model, x, epoch=3)
    elif device == "cuda":
        x = function.show_features_gpu(model, x, epoch=10)
