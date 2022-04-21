import torch
import function
import torch.nn as nn


def startWriteData():
    """
            stride should not be zero
            pad should be smaller than or equal to half of kernel size
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 300

    save_flag = True
    path = "../res/conv2d_time.xls"
    sheet_name = "small"
    # sheet_name = "test"
    value = [["index", "in_channel","in_map", "kernel size", "stride", "padding", "computation number","out_channel","out_map",
              "computation time"]]
    if save_flag:
        function.create_excel_xsl(path, sheet_name, value)

    # x_WH = 224
    index = 0
    for x_WH in [7, 13, 27, 55, 112]:
    # for x_WH in [7,14,28,56,112,224]:
        for in_channel in range(0, 257, 64):
            for out_channel in range(0,257,64):
                if x_WH >= 55 and (in_channel > 128 or out_channel > 128):
                    continue

                if in_channel == 0 or out_channel == 0:
                    continue

                x = torch.rand(size=(1, in_channel, x_WH, x_WH))
                x = x.to(device)

                kernel_size_max = min(11, x_WH) + 1

                for kernel_size in range(1, kernel_size_max, 2):
                    padding_max = min(2, kernel_size // 2) + 1
                    stride_max = min(4, kernel_size) + 1

                    # for stride in range(1, stride_max):
                    for stride in range(1, 2):

                        # for padding in range(0, padding_max):
                        for padding in range(1, 2):
                            myConv2d = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size, stride=stride, padding=padding)
                            myConv2d = myConv2d.to(device)

                            output_x = None
                            if device == "cuda":
                                output_x, computation_time = function.recordTimeGpu(myConv2d, x, device, epoch)
                            if device == "cpu":
                                output_x, computation_time = function.recordTimeCpu(myConv2d, x, device, epoch)

                            output_shape = output_x.shape
                            prod = 1
                            for i in range(len(output_shape)):
                                prod *= output_x.shape[i]

                            computation_number = prod * kernel_size * kernel_size * in_channel
                            print(
                                f"input shape:{x.shape}\tkernel size:{kernel_size}\tstride:{stride}\tpadding:{padding}\t"
                                f"computation number:{computation_number}\toutput shape:{output_x.shape}\tcomputation time : {computation_time:.3f} ms")

                            if save_flag:
                                value = [[index,x.shape[1],x.shape[2], kernel_size, stride, padding, computation_number,
                                          output_x.shape[1],output_x.shape[2], round(computation_time, 3)]]
                                function.write_excel_xls_append(path, sheet_name, value)

                            index += 1

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 300

    x = torch.rand(size=(1, 3, 224, 224))
    x = x.to(device)

    kernel_size = 11
    stride = 4
    padding = 2

    myMaxPool2d = nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding)
    myMaxPool2d = myMaxPool2d.to(device)

    output_x = None
    if device == "cuda":
        output_x, computation_time = function.recordTimeGpu(myMaxPool2d, x, device, epoch)
    if device == "cpu":
        output_x, computation_time = function.recordTimeCpu(myMaxPool2d, x, device, epoch)

    output_shape = output_x.shape
    prod = 1
    for i in range(len(output_shape)):
        prod *= output_x.shape[i]

    computation_number = prod * kernel_size * kernel_size * x.shape[1]
    print(f"input shape:{x.shape}\tkernel size:{kernel_size}\tstride:{stride}\tpadding:{padding}\t"
          f"computation number:{computation_number}\toutput shape:{output_x.shape}\tcomputation time : {computation_time:.3f} ms")




if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 300

    x = torch.rand(size=(1, 3,224,224))
    x = x.to(device)

    warmModel = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=11,stride=4,padding=2)
    warmModel = warmModel.to(device)
    if device == "cuda":
        function.warmUpGpu(warmModel, x, device)
    if device == "cpu":
        function.warmUpCpu(warmModel, x, device)

    # startWriteData()
    # test()


