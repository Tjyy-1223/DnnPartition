import torch.nn as nn
import a3_GoogLeNet
import a4_ResNet
import a5_MobileNet

def get_model_FLOPs(model,x):
    flops = 0.0

    if isinstance(model,nn.Sequential):
        for i in range(len(model)):
            layer = model[i]
            flops += get_layer_FLOPs(layer,x)
            x = layer(x)

    else:
        flops = get_layer_FLOPs(model,x)

    if isinstance(model, a5_MobileNet.ConvNormActivation):
        flops = 1.60 * flops

    return flops


def get_model_Params(model,x):
    params = 0.0

    if isinstance(model, nn.Sequential):
        for i in range(len(model)):
            layer = model[i]
            params += get_layer_Params(layer,x)
            x = layer(x)

    else:
        params = get_layer_Params(model,x)

    if isinstance(model,a5_MobileNet.ConvNormActivation):
        params = 1.60 * params

    return params




def get_layer_FLOPs(layer,x):
    if isinstance(layer, nn.Linear):
        flops = get_linear_FLOPs(layer)

    elif isinstance(layer, nn.Conv2d):
        if layer.groups == 1:
            flops = get_conv2d_FLOPs(layer,x)
        else:
            flops = get_depthwise_separable_conv2d_FLOPs(layer,x)

    elif isinstance(layer, nn.MaxPool2d):
        flops = get_maxpool2d_FLOPs(layer,x)

    elif isinstance(layer,nn.Dropout):
        flops = get_dropout_FLOPs(x)

    elif isinstance(layer, nn.ReLU) or isinstance(layer,nn.ReLU6):
        flops = get_relu_FLOPs(x)

    elif isinstance(layer, nn.Flatten):
        flops = get_flatten_FLOPs(x)

    elif isinstance(layer, nn.BatchNorm2d):
        flops = get_batchNorm2d_FLOPs(x)

    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        flops = get_adaptive_avg_pool2d_FLOPs(layer,x)

    elif isinstance(layer, a3_GoogLeNet.BasicConv2d):
        flops = get_BasicConv2d_FLOPs(layer,x)

    elif isinstance(layer, a3_GoogLeNet.Inception):
        flops = get_Inception_FLOPs(layer,x)

    elif isinstance(layer,a4_ResNet.BasicBlock):
        flops = get_BasicBlock_FLOPs(layer,x)

    elif isinstance(layer,a5_MobileNet.ConvNormActivation):
        flops = get_ConvNormActivation_FLOPs(layer,x)

    elif isinstance(layer,a5_MobileNet.InvertedResidual):
        flops = get_InvertedResidual_FLOPs(layer,x)

    else:
        flops = 0
        raise InterruptedError
    return flops


def get_layer_Params(layer,x):
    if isinstance(layer, nn.Linear):
        params = get_linear_Params(layer)

    elif isinstance(layer, nn.Conv2d):
        if layer.groups == 1:
            params = get_conv2d_Params(layer,x)
        else:
            params = get_depthwise_separable_conv2d_Params(layer,x)

    elif isinstance(layer, nn.MaxPool2d):
        params = get_maxpool2d_Params()

    elif isinstance(layer,nn.Dropout):
        params = get_dropout_Params()

    elif isinstance(layer, nn.ReLU) or isinstance(layer,nn.ReLU6):
        params = get_relu_Params()

    elif isinstance(layer, nn.Flatten):
        params = get_flatten_Params()

    elif isinstance(layer, nn.BatchNorm2d):
        params = get_batchNorm2d_Params()

    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        params = get_adaptive_avg_pool2d_Params(layer,x)

    elif isinstance(layer, a3_GoogLeNet.BasicConv2d):
        params = get_BasicConv2d_Params(layer,x)

    elif isinstance(layer, a3_GoogLeNet.Inception):
        params = get_Inception_Params(layer,x)

    elif isinstance(layer,a4_ResNet.BasicBlock):
        params = get_BasicBlock_Params(layer,x)

    elif isinstance(layer,a5_MobileNet.ConvNormActivation):
        params = get_ConvNormActivation_Params(layer,x)

    elif isinstance(layer,a5_MobileNet.InvertedResidual):
        params = get_InvertedResidual_Params(layer,x)

    else:
        params = 0
        raise InterruptedError
    return params


def get_linear_FLOPs(linear_layer):
    input_size = linear_layer.in_features
    output_size = linear_layer.out_features
    flops = (2 * input_size - 1) * output_size
    return flops


def get_linear_Params(linear_layer):
    input_size = linear_layer.in_features
    output_size = linear_layer.out_features
    params = (input_size * output_size + output_size)
    return params


def get_relu_FLOPs(x):
    x_shape = x.shape
    flops = 1
    for i in range(len(x_shape)):
        flops *= x_shape[i]
    return flops

def get_relu_Params():
    return 0


def get_batchNorm2d_FLOPs(x):
    x_shape = x.shape
    flops = 1
    for i in range(len(x_shape)):
        flops *= x_shape[i]
    return flops

def get_batchNorm2d_Params():
    return 0


def get_maxpool2d_FLOPs(maxpool2d_layer,x):
    input_map = x.shape[2]
    output_map = (input_map - maxpool2d_layer.kernel_size + maxpool2d_layer.padding + maxpool2d_layer.stride) / maxpool2d_layer.stride
    flops = output_map * output_map * x.shape[1] * maxpool2d_layer.kernel_size * maxpool2d_layer.kernel_size
    return flops


def get_maxpool2d_Params():
    return 0


def get_dropout_FLOPs(x):
    x_shape = x.shape
    flops = 1
    for i in range(len(x_shape)):
        flops *= x_shape[i]
    return flops


def get_dropout_Params():
    return 0



def get_flatten_FLOPs(x):
    x_shape = x.shape
    flops = 1
    for i in range(len(x_shape)):
        flops *= x_shape[i]
    return flops


def get_flatten_Params():
    return 0


def get_adaptive_avg_pool2d_FLOPs(layer,x):
    input_map = x.shape[2]
    in_channel = x.shape[1]
    output_map = layer.output_size
    flops = input_map * input_map * in_channel
    return flops


def get_adaptive_avg_pool2d_Params(layer,x):
    input_map = x.shape[2]
    in_channel = x.shape[1]
    output_map = layer.output_size
    params = input_map * input_map * in_channel
    return params


def get_conv2d_FLOPs(conv2d_layer,x):
    in_channel = conv2d_layer.in_channels
    out_channel = conv2d_layer.out_channels
    kernel_size = conv2d_layer.kernel_size[0]
    padding = conv2d_layer.padding[0]
    stride = conv2d_layer.stride[0]
    input_map = x.shape[2]
    output_map = (input_map - kernel_size + padding + stride) / stride

    macc = kernel_size * kernel_size * in_channel * out_channel * output_map * output_map
    flops = 2 * macc
    return flops


def get_conv2d_Params(conv2d_layer,x):
    in_channel = conv2d_layer.in_channels
    out_channel = conv2d_layer.out_channels
    kernel_size = conv2d_layer.kernel_size[0]

    params = kernel_size * kernel_size * in_channel * out_channel + out_channel
    return params


def get_depthwise_separable_conv2d_FLOPs(conv2d_layer,x):
    in_channel = conv2d_layer.in_channels
    out_channel = conv2d_layer.out_channels
    kernel_size = conv2d_layer.kernel_size[0]
    input_map = x.shape[2]
    output_map = (input_map - conv2d_layer.kernel_size[0] + conv2d_layer.padding[0] + conv2d_layer.stride[0]) / conv2d_layer.stride[0]

    depthwise_macc = kernel_size * kernel_size * in_channel * output_map * output_map
    pointwise_macc = output_map * output_map * in_channel * out_channel
    flops = 2 * (depthwise_macc + pointwise_macc)
    return flops


def get_depthwise_separable_conv2d_Params(conv2d_layer,x):
    in_channel = conv2d_layer.in_channels
    out_channel = conv2d_layer.out_channels
    kernel_size = conv2d_layer.kernel_size[0]

    depthwise_params = kernel_size * kernel_size * in_channel
    pointwise_params = 1 * 1 * in_channel * out_channel + out_channel
    params = depthwise_params + pointwise_params
    return params


def get_expansion_block_FLOPs(conv2d_layer,x,Cexp):
    in_channel = conv2d_layer.in_channels
    out_channel = conv2d_layer.out_channels
    kernel_size = conv2d_layer.kernel_size[0]
    input_map = x.shape[2]
    output_map = (input_map - conv2d_layer.kernel_size[0] + conv2d_layer.padding[0] + conv2d_layer.stride[0]) / conv2d_layer.stride[0]

    expansion_layer = in_channel * input_map  * input_map  * Cexp
    depthwise_layer = kernel_size * kernel_size * Cexp * output_map * output_map
    projection_layer = Cexp * output_map * output_map * out_channel
    flops = 2 * (expansion_layer + depthwise_layer + projection_layer)
    return flops


def get_BasicConv2d_FLOPs(layer, x):
    conv = layer.conv
    conv_flops = get_conv2d_FLOPs(conv,x)
    x = conv(x)

    return conv_flops



def get_BasicConv2d_Params(layer, x):
    conv = layer.conv
    conv_params = get_conv2d_Params(conv,x)
    x = conv(x)

    return conv_params



def get_Inception_FLOPs(block,x):
    branch1 = block.branch1
    branch2 = block.branch2
    branch3 = block.branch3
    branch4 = block.branch4
    return get_model_FLOPs(branch1, x) \
           + get_model_FLOPs(branch2, x) \
           + get_model_FLOPs(branch3, x) \
           + get_model_FLOPs(branch4, x)


def get_Inception_Params(block,x):
    branch1 = block.branch1
    branch2 = block.branch2
    branch3 = block.branch3
    branch4 = block.branch4
    return get_model_Params(branch1, x) \
           + get_model_Params(branch2, x) \
           + get_model_Params(branch3, x) \
           + get_model_Params(branch4, x)



def get_BasicBlock_FLOPs(block,x):
    conv1 = block.conv1
    bn1 = block.bn1
    relu = block.relu
    conv2 = block.conv2
    bn2 = block.bn2
    child_model1 = nn.Sequential(
        conv1,bn1,relu,conv2,bn2
    )

    """1x1卷积核"""
    downsample = block.downsample

    if downsample is not None:
        return get_model_FLOPs(child_model1,x) + get_model_FLOPs(downsample,x)
    else:
        return get_model_FLOPs(child_model1,x)



def get_BasicBlock_Params(block,x):
    conv1 = block.conv1
    bn1 = block.bn1
    relu = block.relu
    conv2 = block.conv2
    bn2 = block.bn2
    child_model1 = nn.Sequential(
        conv1, bn1, relu, conv2, bn2
    )

    """1x1卷积核"""
    downsample = block.downsample

    if downsample is not None:
        return get_model_Params(child_model1, x) + get_model_Params(downsample, x)
    else:
        return get_model_Params(child_model1, x)



def get_ConvNormActivation_FLOPs(block,x):
    print("..............................")
    return 2 * get_model_FLOPs(block,x)



def get_ConvNormActivation_Params(block, x):
    return 2 * get_model_Params(block,x)


def get_InvertedResidual_FLOPs(block,x):
    block = block.conv
    flops = 0.0
    for layer in block:
        flops += get_model_FLOPs(layer,x)
        x = layer(x)
    return 0.7 * flops


def get_InvertedResidual_Params(block,x):
    block = block.conv
    params = 0.0
    for layer in block:
        params += get_model_Params(layer,x)
        x = layer(x)
    return 0.7 * params



