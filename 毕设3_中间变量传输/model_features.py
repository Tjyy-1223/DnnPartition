import torch.nn as nn


def get_later_FLOPs(layer,x):
    x_shape = x.shape
    flops = None
    if isinstance(layer, nn.Linear):
        flops = get_linear_FLOPs(layer)

    elif isinstance(layer, nn.Conv2d):
        flops = get_conv2d_FLOPs(layer,x)

    elif isinstance(layer, nn.MaxPool2d):
        flops = get_maxpool2d_FLOPs(layer,x)

    elif isinstance(layer,nn.Dropout):
        flops = get_dropout_FLOPs(x)

    elif isinstance(layer, nn.ReLU) or isinstance(layer,nn.ReLU6):
        flops = get_relu_FLOPs(x)

    elif isinstance(layer, nn.Flatten):
        flops = get_flatten_FLOPs(x)

    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        flops = get_adaptive_avg_pool2d_FLOPs(layer,x)

    else:
        flops = 0
    return flops


def get_linear_FLOPs(linear_layer):
    input_size = linear_layer.in_features
    output_size = linear_layer.out_features
    flops = (2 * input_size - 1) * output_size
    return flops


def get_relu_FLOPs(x):
    x_shape = x.shape
    flops = 1
    for i in range(len(x_shape)):
        flops *= x_shape[i]
    return flops


def get_maxpool2d_FLOPs(maxpool2d_layer,x):
    input_map = x.shape[2]
    output_map = (input_map - maxpool2d_layer.kernel_size + maxpool2d_layer.padding + maxpool2d_layer.stride) / maxpool2d_layer.stride
    flops = output_map * output_map * x.shape[1] * maxpool2d_layer.kernel_size * maxpool2d_layer.kernel_size
    return flops


def get_dropout_FLOPs(x):
    x_shape = x.shape
    flops = 1
    for i in range(len(x_shape)):
        flops *= x_shape[i]
    return flops


def get_flatten_FLOPs(x):
    x_shape = x.shape
    flops = 1
    for i in range(len(x_shape)):
        flops *= x_shape[i]
    return flops


def get_adaptive_avg_pool2d_FLOPs(layer,x):
    input_map = x.shape[2]
    in_channel = x.shape[1]
    output_map = layer.output_size
    flops = input_map * input_map * in_channel
    return flops


def get_conv2d_FLOPs(conv2d_layer,x):
    in_channel = conv2d_layer.in_channels
    out_channel = conv2d_layer.out_channels
    kernel_size= conv2d_layer.kernel_size
    input_map = x.shape[2]
    output_map = (input_map - conv2d_layer.kernel_size + conv2d_layer.padding + conv2d_layer.stride)/conv2d_layer.stride

    macc = kernel_size * kernel_size * in_channel * out_channel * output_map * output_map
    flops = 2 * macc
    return flops



def get_depthwise_separable_conv2d_FLOPs(conv2d_layer,x):
    in_channel = conv2d_layer.in_channels
    out_channel = conv2d_layer.out_channels
    kernel_size = conv2d_layer.kernel_size
    input_map = x.shape[2]
    output_map = (input_map - conv2d_layer.kernel_size + conv2d_layer.padding + conv2d_layer.stride) / conv2d_layer.stride

    depthwise_macc = kernel_size * kernel_size * in_channel * output_map * output_map
    pointwise_macc = output_map * output_map * in_channel * out_channel
    flops = 2 * (depthwise_macc + pointwise_macc)
    return flops



def get_expansion_block_FLOPs(conv2d_layer,x,Cexp):
    in_channel = conv2d_layer.in_channels
    out_channel = conv2d_layer.out_channels
    kernel_size = conv2d_layer.kernel_size
    input_map = x.shape[2]
    output_map = (input_map - conv2d_layer.kernel_size + conv2d_layer.padding + conv2d_layer.stride) / conv2d_layer.stride

    expansion_layer = in_channel * input_map  * input_map  * Cexp
    depthwise_layer = kernel_size * kernel_size * Cexp * output_map * output_map
    projection_layer = Cexp * output_map * output_map * out_channel
    flops = 2 * (expansion_layer + depthwise_layer + projection_layer)
    return flops