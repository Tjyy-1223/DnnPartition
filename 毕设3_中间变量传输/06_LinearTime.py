import torch
import time
import socket
import pickle
import function
import torch.nn as nn





if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 300

    input_size = 9217
    output_size = 4096
    x = torch.rand(size = (1,input_size))
    myLinear = nn.Linear(input_size,output_size)
    function.warmUpCpu(myLinear, x, device)

    save_flag = True
    path = "../res/linear_time.xls"
    sheet_name = "mac"
    value = [["index", "layerName", "inputSize", "outputSize", "computation time(ms)"]]
    if save_flag:
        function.create_excel_xsl(path, sheet_name, value)

    index = 0
    for input_size in range(0,51201,512):
        output_size = 1000
        if input_size == 0 or output_size == 0:
            continue

        x = torch.rand(size=(1, input_size))
        myLinear = nn.Linear(input_size, output_size)

        if device == "cuda":
            _, computation_time = function.recordTimeGpu(myLinear,x,device,epoch)
        if device == "cpu":
            _, computation_time = function.recordTimeCpu(myLinear, x, device, epoch)

        print(f"input:{input_size}  output:{output_size}    computation time : {computation_time:.3f}")

        if save_flag:
            value = [[index, f"{myLinear}", input_size, output_size, round(computation_time, 3)]]
            function.write_excel_xls_append(path, sheet_name, value)
        index += 1



    for input_size in range(0,25088,512):
        for output_size in range(0,10240,512):
            if input_size == 0 or output_size == 0:
                continue

            x = torch.rand(size=(1, input_size))
            myLinear = nn.Linear(input_size, output_size)

            if device == "cuda":
                _,computation_time = function.recordTimeGpu(myLinear,x,device,epoch)
            if device == "cpu":
                _,computation_time = function.recordTimeCpu(myLinear, x, device, epoch)

            print(f"input:{input_size}  output:{output_size}    computation time : {computation_time:.3f}")

            if save_flag:
                value = [[index, f"{myLinear}", input_size, output_size, round(computation_time, 3)]]
                function.write_excel_xls_append(path, sheet_name, value)
            index += 1