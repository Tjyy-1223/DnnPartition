import torch
import torch.nn as nn
import time
import threading
import xlrd
import xlwt
from xlutils.copy import copy

import a1_alexNet
import a2_vggNet
import a3_GoogLeNet
import a4_ResNet
import a5_MobileNet
import a6_LeNet


"""
model_partition函数可以将一个整体的model 划分成两个部分
    划分的大致思路：
    如选定 第 index(下标从1开始) 层对alexnet进行划分 ，则代表在第index后对模型进行划分
    则对alexnet网络进行 层级遍历
    将index层包括第index层 包装给edge_model作为返回 意为边缘节点
    后续的节点包装给 cloud_model 意为云端节点
"""


def model_partition(alexnet, index):
    edge_model = nn.Sequential()
    cloud_model = nn.Sequential()
    idx = 1

    for layer in alexnet:
        if (idx <= index):
            edge_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
        else:
            cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
        idx += 1
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # edge_model = edge_model.to(device)
    # cloud_model = cloud_model.to(device)
    return edge_model, cloud_model


"""
一个观察每层情况的函数
    可以输出layer的各种性质 其输出如下：
    1-Conv2d computation time: 30.260000 s
    output shape: torch.Size([10000, 64, 55, 55]) 	 transport_num:1936000000 	 transport_size:61952.0M
    weight  :  parameters size torch.Size([64, 3, 11, 11]) 	 parameters number 23232
    bias  :  parameters size torch.Size([64]) 	 parameters number 64

    最后返回运行结果x
    修改：省略了 激活层 batchnormal层 以及 dropout层
"""
def show_features_gpu(alexnet, x ,filter = True,epoch = 300,save = False,model_name = "model",path = None):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU warm-up and prevent it from going into power-saving mode
    dummy_input = torch.rand(x.shape).to(device)

    # init loggers
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    init_starter = torch.cuda.Event(enable_timing=True)
    init_ender = torch.cuda.Event(enable_timing=True)

    # GPU warm-up
    with torch.no_grad():
        for i in range(3):
            init_starter.record()
            _ = alexnet(dummy_input)
            init_ender.record()
            # wait for GPU SYNC
            torch.cuda.synchronize()
            curr_time = init_starter.elapsed_time(init_ender)

            print(f"GPU warm-up - {i+1}")
            print(f'computation time: {curr_time :.3f} ms\n')

    if save:
        sheet_name = model_name
        value = [["index", "layerName","computation_time(ms)","output_shape","transport_num","transport_size(MB)"]]
        create_excel_xsl(path,sheet_name,value)

    if len(alexnet) > 0:
        idx = 1
        for layer in alexnet:
            if filter is True:
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue

            all_time = 0
            temp_x = x
            with torch.no_grad():
                for i in range(epoch):
                    temp_x = torch.rand(temp_x.shape).to(device)

                    starter.record()
                    x = layer(temp_x)
                    ender.record()
                    # wait for GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)

                    all_time += curr_time

            # 计算分割点 中间传输占用大小为多少m  主要与网络传输时延相关
            total_num = 1
            for num in x.shape:
                total_num *= num
            type_size = 4
            size = total_num * type_size / 1000 / 1000

            print("------------------------------------------------------------------")
            print(f'{idx}-{layer} \n'
                  f'computation time: {(all_time/epoch):.3f} ms\n'
                  f'output shape: {x.shape}\t transport_num:{total_num}    transport_size:{size:.3f}MB')

            if save:
                sheet_name = model_name
                value = [[idx, f"{layer}", round((all_time / epoch), 3), f"{x.shape}", total_num,round(size, 3)]]
                write_excel_xls_append(path,sheet_name,value)
            # 计算各层的结构所包含的参数量 主要与计算时延相关
            # para = parameters.numel()
            # for name, parameters in layer.named_parameters():
            #     print(f"{name}  :  parameters size {parameters.size()} \t parameters number {parameters.numel()}")
            idx += 1
        return x
    else:
        print("this model is a empty model")
        return x



"""
一个观察每层情况的函数
    可以输出layer的各种性质 其输出如下：
    1-Conv2d computation time: 30.260000 s
    output shape: torch.Size([10000, 64, 55, 55]) 	 transport_num:1936000000 	 transport_size:61952.0M
    weight  :  parameters size torch.Size([64, 3, 11, 11]) 	 parameters number 23232
    bias  :  parameters size torch.Size([64]) 	 parameters number 64

    最后返回运行结果x
    修改：省略了 激活层 batchnormal层 以及 dropout层
"""
def show_features_cpu(alexnet, x ,filter = True,epoch = 3,save = False,model_name = "model",path = None):

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # GPU warm-up and prevent it from going into power-saving mode
    dummy_input = torch.rand(x.shape).to(device)

    # GPU warm-up
    with torch.no_grad():
        for i in range(3):
            start = time.perf_counter()
            _ = alexnet(dummy_input)
            end = time.perf_counter()
            curr_time = end - start

            print(f"CPU warm-up - {i+1}")
            print(f'computation time: {curr_time*1000 :.3f} ms\n')

    if save:
        sheet_name = model_name
        value = [["index", "layerName","computation_time(ms)","output_shape","transport_num","transport_size(MB)"]]
        create_excel_xsl(path,sheet_name,value)

    if len(alexnet) > 0:
        idx = 1
        for layer in alexnet:
            if filter is True:
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue

            all_time = 0
            temp_x = x
            with torch.no_grad():
                for i in range(epoch):
                    temp_x = torch.rand(temp_x.shape).to(device)

                    start_time = time.perf_counter()
                    x = layer(temp_x)
                    end_time = time.perf_counter()
                    curr_time = end_time - start_time
                    all_time += curr_time

            # 计算分割点 中间传输占用大小为多少m  主要与网络传输时延相关
            total_num = 1
            for num in x.shape:
                total_num *= num
            type_size = 4
            size = total_num * type_size / 1000 / 1000

            print("------------------------------------------------------------------")
            print(f'{idx}-{layer} \n'
                  f'computation time: {(all_time/epoch)*1000:.3f} ms\n'
                  f'output shape: {x.shape}\t transport_num:{total_num}    transport_size:{size:.3f}MB')

            if save:
                sheet_name = model_name
                value = [[idx,f"{layer}",round((all_time/epoch)*1000,3),f"{x.shape}",total_num,round(size,3)]]
                write_excel_xls_append(path,sheet_name,value)

            # 计算各层的结构所包含的参数量 主要与计算时延相关
            # para = parameters.numel()
            # for name, parameters in layer.named_parameters():
            #     print(f"{name}  :  parameters size {parameters.size()} \t parameters number {parameters.numel()}")
            idx += 1
        return x
    else:
        print("this model is a empty model")
        return x

"""
    一个显示边缘模型和云端模型每层结构的函数
"""


def show_2model(edge_model, cloud_model):
    print(f"------------- edge model -----------------")
    print(edge_model)
    print(f"------------- cloud model -----------------")
    print(cloud_model)


def show_1model(model,filter = True):
    if len(model) > 0:
        idx = 1
        for layer in model:
            if filter is True:
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue
            print(f'{idx}-{layer}')
            idx += 1
    else:
        print("this model is a empty model")



"""
    设计一个函数，将结果保存在excel表中，excel的名字需要自己能够命名

"""
def create_excel_xsl(path,sheet_name,value):
    index = len(value)
    try:
        with xlrd.open_workbook(path) as workbook:
            workbook = copy(workbook)
            # worksheet = workbook.sheet_by_name(sheet_name)
            worksheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
            for i in range(len(value[0])):
                worksheet.col(i).width = 256*30  # Set the column width
            for i in range(0, index):
                for j in range(0, len(value[i])):
                    worksheet.write(i, j, value[i][j])
            workbook.save(path)
            print("xls格式表格创建成功")
    except FileNotFoundError:
        workbook = xlwt.Workbook()  # 新建一个工作簿
        worksheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
        for i in range(len(value[0])):
            worksheet.col(i).width = 256 * 30  # Set the column width
        for i in range(0, index):
            for j in range(0, len(value[i])):
                worksheet.write(i, j, value[i][j])
        workbook.save(path)
        print("xls格式表格创建成功")


def write_excel_xls_append(path,sheet_name,value):
    index = len(value)
    workbook = xlrd.open_workbook(path)
    worksheet = workbook.sheet_by_name(sheet_name)

    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(sheet_name)

    for i in range(len(value[0])):
        new_worksheet.col(i).width = 256 * 30  # Set the column width

    for i in range(0,index):
        for j in range(0,len(value[i])):
            new_worksheet.write(i+rows_old, j, value[i][j])

    new_workbook.save(path)  # 保存工作簿
    print("xls格式表格【追加】写入数据成功！")


def read_excel_xls(path,sheet_name):
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    worksheet = workbook.sheet_by_name(sheet_name)  # 获取工作簿中的所有表格
    for i in range(0, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            print(worksheet.cell_value(i, j), "\t", end="")  # 逐行逐列读取数据
        print()


def get_excel_data(path,sheet_name,col_name):
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    worksheet = workbook.sheet_by_name(sheet_name)  # 获取工作簿中的所有表格

    col_index = -1
    for j in range(0,worksheet.ncols):
        if worksheet.cell_value(0, j) == col_name:
            col_index = j
    if col_index == -1:
        print("no matched col name")
        return None

    """
        开始取相应列的数据
    """
    data = []
    for i in range(1, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            if j == col_index:
                data.append(worksheet.cell_value(i,j))
    return data

"""
    GPU预热操作
"""
def warmUpGpu(model,x,device = "cuda"):
    # GPU warm-up and prevent it from going into power-saving mode
    dummy_input = torch.rand(x.shape).to(device)

    # GPU warm-up
    with torch.no_grad():
        for i in range(3):
            _ = model(dummy_input)

        avg_time = 0.0

        for i in range(300):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()

            _ = model(dummy_input)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            avg_time += curr_time
        avg_time /= 300
        print(f"GPU warm-up: {curr_time:.3f}ms")
        print("==============================================")


"""
    CPU预热操作
"""
def warmUpCpu(model,x,device):
    # GPU warm-up and prevent it from going into power-saving mode
    dummy_input = torch.rand(x.shape).to(device)

    # GPU warm-up
    with torch.no_grad():
        for i in range(3):
            _ = model(dummy_input)

        avg_time = 0.0

        for i in range(10):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            curr_time = end - start
            avg_time += curr_time
        avg_time /= 10
        print(f"CPU warm-up: {curr_time:.3f}ms")
        print("==============================================")


"""
    记录GPU的用时
"""
def recordTimeGpu(model, x, device = "cuda", epoch = 300):
    model = model.to(device)
    all_time = 0.0
    for i in range(epoch):
        x = torch.rand(x.shape).to(device)
        # init loggers
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            starter.record()
            res_x = model(x)
            ender.record()

        # wait for GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        all_time += curr_time
    all_time /= epoch
    return res_x, all_time


"""
    记录CPU的用时
"""

def recordTimeCpu(model, x, device, epoch):
    model = model.to(device)
    all_time = 0.0
    for i in range(epoch):
        x = torch.rand(x.shape).to(device)

        with torch.no_grad():
            start_time = time.perf_counter()
            res_x = model(x)
            end_time = time.perf_counter()

        curr_time = end_time - start_time
        all_time += curr_time
    all_time /= epoch
    return res_x, all_time * 1000



"""
   获取DNN模型
"""
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
    elif index == 6:
        LeNet = a6_LeNet.LeNet()
        return LeNet
    else:
        print("no model")
        return None


def addList(mylist):
    for i in range(len(mylist)):
        if i == 0:
            continue
        else:
            mylist[i] = mylist[i-1] + mylist[i]
    return mylist


def addListReverse(mylist):
    length = len(mylist)
    for i in range(length-1,-1,-1):
        if i == length - 1:
            continue
        else:
            mylist[i] = mylist[i+1] + mylist[i]
    return mylist