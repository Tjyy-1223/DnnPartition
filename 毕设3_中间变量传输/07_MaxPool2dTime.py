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

    save_flag = False
    path = "../res/maxPool2d_time.xls"
    # sheet_name = "mac"
    sheet_name = "cuda2"
    value = [["index", "input shape", "kernel size", "stride", "padding", "computation number", "output shape",
              "computation time"]]
    if save_flag:
        function.create_excel_xsl(path, sheet_name, value)

    # x_WH = 224
    index = 0
    for x_WH in [6, 7, 13, 14, 27, 28, 55, 56, 112, 224]:
        for channel in range(0, 1024, 64):
            if x_WH >= 55 and channel > 128:
                continue

            if channel == 0:
                continue

            x = torch.rand(size=(1, channel, x_WH, x_WH))
            x = x.to(device)

            kernel_size_max = min(3, x_WH) + 1

            for kernel_size in range(1, kernel_size_max, 2):
                padding_max = kernel_size // 2 + 1
                stride_max = min(4, kernel_size) + 1

                for stride in range(2,3):
                # for stride in range(1, stride_max):

                    # for padding in range(0, padding_max):
                    for padding in range(0,1):
                        myMaxPool2d = nn.MaxPool2d(kernel_size, stride, padding)
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

                        computation_number = prod * kernel_size * kernel_size
                        print(f"input shape:{x.shape}\tkernel size:{kernel_size}\tstride:{stride}\tpadding:{padding}\t"
                              f"computation number:{computation_number}\toutput shape:{output_x.shape}\tcomputation time : {computation_time:.3f} ms")

                        if save_flag:
                            value = [[index, f"{x.shape}", kernel_size, stride, padding, computation_number,
                                      f"{output_x.shape}", round(computation_time, 3)]]
                            function.write_excel_xls_append(path, sheet_name, value)

                        index += 1



def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 300

    x = torch.rand(size=(1, 960, 28, 28))
    x = x.to(device)

    kernel_size = 3
    stride = 2
    padding = 0

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

    computation_number = prod * kernel_size * kernel_size
    print(f"input shape:{x.shape}\tkernel size:{kernel_size}\tstride:{stride}\tpadding:{padding}\t"
          f"computation number:{computation_number}\toutput shape:{output_x.shape}\tcomputation time : {computation_time:.3f} ms")

def test2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 300

    x = torch.rand(size=(1, 128, 224, 224))
    x = x.to(device)

    kernel_size = 1
    stride = 2
    padding = 0

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

    computation_number = prod * kernel_size * kernel_size
    print(f"input shape:{x.shape}\tkernel size:{kernel_size}\tstride:{stride}\tpadding:{padding}\t"
          f"computation number:{computation_number}\toutput shape:{output_x.shape}\tcomputation time : {computation_time:.3f} ms")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 300

    x = torch.rand(size=(1,64,112,112))
    x = x.to(device)

    if device == "cuda":
        function.warmUpGpu(nn.MaxPool2d(3,2,0), x, device)
    if device == "cpu":
        function.warmUpCpu(nn.MaxPool2d(3,2,0), x, device)

    # startWriteData()


    test()

    test2()









