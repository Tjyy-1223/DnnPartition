import time
from sklearn.preprocessing import MinMaxScaler
import joblib

import function
import functionImg
import model_features
import a1_alexNet
import a5_MobileNet
import torch
import torch.nn as nn
import a3_GoogLeNet
import a4_ResNet
import numpy as np


def test_ConvNormActivation():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.rand(size=(1, 3, 224, 224))
    x = x.to(device)

    channel = 32
    norm_layer = nn.BatchNorm2d
    convnorm = a5_MobileNet.ConvNormActivation(3, channel, kernel_size=3, stride=2, norm_layer=norm_layer,
                                               activation_layer=nn.ReLU6)
    convnorm = convnorm.to(device)

    function.warmUpGpu(convnorm, x)

    _, time1 = function.recordTimeGpu(convnorm, x)
    flops = model_features.get_model_FLOPs(convnorm, x)
    params = model_features.get_model_Params(convnorm, x)

    print(
        f"{convnorm}   flops : {flops} \t params : {params} \t layer computation time : {time1:.3f}")
    print("=================================================================")

    conv2d = nn.Sequential(
        nn.Conv2d(3, channel, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(channel),
        nn.ReLU(inplace=True)
    )
    conv2d = conv2d.to(device)

    for layer in conv2d:
        _, time1 = function.recordTimeGpu(layer, x)
        flops = model_features.get_model_FLOPs(layer, x)
        params = model_features.get_model_Params(layer, x)
        print(f"{layer}   flops : {flops} \t params : {params} \t layer computation time : {time1:.3f}")
        x = layer(x)
    print("=================================================================")



def test_InvertedResidual():
    device = "cuda" if torch.cuda.is_available() else "cpu"



    inp = 64
    expand_ratio = 1
    oup = 64
    stride = 2

    x = torch.rand(size=(1, inp, 224, 224))
    x = x.to(device)

    norm_layer = nn.BatchNorm2d
    InvertedResidual = a5_MobileNet.InvertedResidual(inp=inp,expand_ratio=expand_ratio,oup=oup,stride=2)
    InvertedResidual = InvertedResidual.to(device)

    function.warmUpGpu(InvertedResidual, x)

    _, time1 = function.recordTimeGpu(InvertedResidual, x)
    flops = model_features.get_model_FLOPs(InvertedResidual, x)
    params = model_features.get_model_Params(InvertedResidual, x)

    print(
        f"{InvertedResidual}   flops : {flops} \t params : {params} \t layer computation time : {time1:.3f}")
    print("=================================================================")

    hidden_dim = inp * expand_ratio
    conv2d = nn.Sequential(
        a5_MobileNet.ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6),
        a5_MobileNet.ConvNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                           activation_layer=nn.ReLU6),
        # pw-linear
        nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
        norm_layer(oup),
    )
    conv2d = conv2d.to(device)

    for layer in conv2d:
        _, time1 = function.recordTimeGpu(layer, x)
        flops = model_features.get_model_FLOPs(layer, x)
        params = model_features.get_model_Params(layer, x)
        print(f"{layer}   flops : {flops} \t params : {params} \t layer computation time : {time1:.3f}")
        x = layer(x)
    print("=================================================================")


def test_BasicConv2d():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inp = 64
    oup = 128

    x = torch.rand(size=(1, inp, 224, 224))
    x = x.to(device)

    norm_layer = nn.BatchNorm2d
    BasicConv2d = a3_GoogLeNet.BasicConv2d(inp,oup,kernel_size=7,stride=2,padding=3)
    BasicConv2d = BasicConv2d.to(device)

    function.warmUpGpu(BasicConv2d, x)

    _, time1 = function.recordTimeGpu(BasicConv2d, x)
    flops = model_features.get_model_FLOPs(BasicConv2d, x)
    params = model_features.get_model_Params(BasicConv2d, x)

    print(
        f"{BasicConv2d}   flops : {flops} \t params : {params} \t layer computation time : {time1:.3f}")
    print("=================================================================")

    conv2d = nn.Sequential(
        nn.Conv2d(inp, oup, bias=False,kernel_size=7,stride=2,padding=3),
        nn.BatchNorm2d(oup, eps=0.001)
    )
    conv2d = conv2d.to(device)

    for layer in conv2d:
        _, time1 = function.recordTimeGpu(layer, x)
        flops = model_features.get_model_FLOPs(layer, x)
        params = model_features.get_model_Params(layer, x)
        print(f"{layer}   flops : {flops} \t params : {params} \t layer computation time : {time1:.3f}")
        x = layer(x)
    print("=================================================================")



def test_Inception():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.rand(size=(1, 480, 27,27))
    x = x.to(device)

    norm_layer = nn.BatchNorm2d
    BasicConv2d = a3_GoogLeNet.Inception(480, 192, 96, 208, 16, 48, 64)
    BasicConv2d = BasicConv2d.to(device)

    function.warmUpGpu(BasicConv2d, x)

    _, time1 = function.recordTimeGpu(BasicConv2d, x)
    flops = model_features.get_model_FLOPs(BasicConv2d, x)
    params = model_features.get_model_Params(BasicConv2d, x)

    print(
        f"{BasicConv2d}   flops : {flops} \t params : {params} \t layer computation time : {time1:.3f}")
    print("=================================================================")

    in_channels = 480
    ch1x1 = 192
    ch3x3red = 96
    ch3x3 = 208
    ch5x5red = 16
    ch5x5 = 48
    pool_proj = 64
    conv_block = a3_GoogLeNet.BasicConv2d
    branch1 = nn.Sequential(
        conv_block(in_channels, ch1x1, kernel_size=1)
    )

    branch2 = nn.Sequential(
        conv_block(in_channels, ch3x3red, kernel_size=1),
        conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
    )

    branch3 = nn.Sequential(
        conv_block(in_channels, ch5x5red, kernel_size=1),
        conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
    )

    branch4 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        conv_block(in_channels, pool_proj, kernel_size=1),
    )
    branch1 = branch1.to(device)
    branch2 = branch2.to(device)
    branch3 = branch3.to(device)
    branch4 = branch4.to(device)

    get_branch(branch1,x)
    get_branch(branch2,x)
    get_branch(branch3,x)
    get_branch(branch4,x)

    print("=================================================================")




def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)



def test_BasicBlock():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inp = 256
    oup = 512
    stride = 1

    x = torch.rand(size=(1, inp, 64,64))
    x = x.to(device)

    norm_layer = nn.BatchNorm2d


    downsample = nn.Sequential(
        conv1x1(inp, oup, stride),
        norm_layer(oup),
    )

    BasicConv2d = a4_ResNet.BasicBlock(inp, oup, stride, downsample,norm_layer)
    BasicConv2d = BasicConv2d.to(device)

    function.warmUpGpu(BasicConv2d, x)

    _, time1 = function.recordTimeGpu(BasicConv2d, x)
    flops = model_features.get_model_FLOPs(BasicConv2d, x)
    params = model_features.get_model_Params(BasicConv2d, x)

    print(
        f"{BasicConv2d}   flops : {flops} \t params : {params} \t layer computation time : {time1:.3f}")
    print("=================================================================")



    conv2d = nn.Sequential(
        conv3x3(inp, oup, stride),
        norm_layer(oup),
        nn.ReLU(inplace=True),
        conv3x3(oup, oup),
        norm_layer(oup)
    )
    downsample = downsample.to(device)
    conv2d = conv2d.to(device)

    for layer in conv2d:
        _, time1 = function.recordTimeGpu(layer, x)
        flops = model_features.get_model_FLOPs(layer, x)
        params = model_features.get_model_Params(layer, x)
        print(f"{layer}   flops : {flops} \t params : {params} \t layer computation time : {time1:.3f}")
        x = layer(x)

    x = torch.rand(size=(1, inp, 64, 64))
    x = x.to(device)
    _, time1 = function.recordTimeGpu(downsample, x)
    flops = model_features.get_model_FLOPs(downsample, x)
    params = model_features.get_model_Params(downsample, x)
    print(f"{downsample}   flops : {flops} \t params : {params} \t layer computation time : {time1:.3f}")
    print("=================================================================")


def get_branch(branch,x):
    _, time1 = function.recordTimeGpu(branch, x)
    flops = model_features.get_model_FLOPs(branch, x)
    params = model_features.get_model_Params(branch, x)
    print(f"{branch}   flops : {flops} \t params : {params} \t layer computation time : {time1:.3f}")




if __name__ == '__main__':
    # test_ConvNormActivation()

    # test_InvertedResidual()

    # test_BasicConv2d()

    # test_Inception()

    # test_BasicBlock()
    device = "cuda"

    x = torch.rand(size=(1,512,28,28))
    x = x.to(device)
    myc = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    myc = myc.to(device)

    function.warmUpGpu(myc,x)

    _, time1 = function.recordTimeGpu(myc, x)
    print(time1)
    print(model_features.get_model_FLOPs(myc,x))

    print("============================================================")

    x = torch.rand(size=(1, 256, 28, 28))
    x = x.to(device)
    Inception = a4_ResNet.BasicBlock(256,256)
    Inception = Inception.to(device)

    _, time1 = function.recordTimeGpu(Inception, x)
    print(Inception)
    print(time1)
    print(model_features.get_model_FLOPs(Inception, x))

    print("============================================================")

    x = torch.rand(size=(1, 256, 28, 28))
    x = x.to(device)

    model = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    )
    model = model.to(device)

    _, time1 = function.recordTimeGpu(model, x)
    print(model)
    print(time1)
    print(model_features.get_model_FLOPs(model, x))

    print("============================================================")

    for layer in model:
        _, time1 = function.recordTimeGpu(layer, x)
        print(layer)
        print(time1)
        print(model_features.get_model_FLOPs(layer, x))
        x = layer(x)
