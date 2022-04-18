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
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(device)

    x = torch.rand(size=(1, 3, 224, 224))
    x = x.to(device)
    print(f"x device : {x.device}")

    modelIndex = 2
    model_names = ["alexnet","vgg16","googLeNet","resnet18","mobileNetv2"]
    model_name = model_names[modelIndex-1]
    model = getDnnModel(modelIndex)
    model.to(device)
    print(len(model))

    temp_x = x
    epoch = 300
    save_flag = False
    filter = False
    path = "../res/DnnLayer_mac_power_all.xls"
    if device == "cpu":
        x = function.show_features_cpu(model, x, filter=filter ,epoch=epoch,save=save_flag,model_name=model_name,path=path)
    elif device == "cuda":
        x = function.show_features_gpu(model, x, filter=filter ,epoch=epoch,save=save_flag,model_name=model_name,path=path)
