import function
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    model_index = 1
    model_names = ["alexnet", "vgg16", "googLeNet", "resnet18", "mobileNetv2"]
    path = "../res/DnnLayer_cuda_new.xls"
    sheet_name = model_names[model_index - 1]

    index = function.get_excel_data(path, sheet_name, "index")
    layerName = function.get_excel_data(path, sheet_name, "layerName")
    times = function.get_excel_data(path, sheet_name, "computation_time(ms)")
    transport_num = function.get_excel_data(path, sheet_name, "transport_num")
    params = function.get_excel_data(path, sheet_name, "params")

    # print(index)
    # print(times)


    plt.figure(figsize=(8,3.5))
    N = len(index)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    # p1 = plt.bar(ind, times, width)
    # plt.ylabel('Computation Latency(ms)')

    p1 = plt.bar(ind, params, width,color="cornflowerblue")
    plt.ylabel('Layer Parameters')

    # plt.title('Layer Latency of AlexNet')
    plt.xticks(ind, ('conv1', 'maxPool2d1', 'conv2', 'maxPool2d2', 'conv3',
                     'conv4', 'conv5', 'maxPool2d3', 'avgPool2d', 'flatten',
                     'linear1', 'linear2', 'linear3'))
    plt.xticks(rotation=-45)

    # plt.yticks(np.arange(0, 81, 10))
    # plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    for a, b in zip(ind,params):
        plt.text(a, b + 0.02, '%.3f' % b, ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.show()