import function
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    model_index = 1
    model_names = ["alexnet", "vgg16", "googLeNet", "resnet18", "mobileNetv2"]
    path = "../res/cpu_gpu.xls"
    sheet_name = model_names[model_index - 1]

    index = function.get_excel_data(path, sheet_name, "index")
    layerName = function.get_excel_data(path, sheet_name, "layerName")

    end_to_end_latency = function.get_excel_data(path, sheet_name, "end-to-end latency")
    transport_latency = function.get_excel_data(path, sheet_name, "transport_latency")
    edge_latency = function.get_excel_data(path, sheet_name, "edge_latency")
    cloud_latency = function.get_excel_data(path, sheet_name, "cloud_latency")


    # print(index)
    # print(times)


    plt.figure(figsize=(8,3.5))
    N = len(index)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence


    p1 = plt.bar(ind, cloud_latency, width,color="black")
    p2 = plt.bar(ind, transport_latency, width, bottom=cloud_latency, color="darkgreen")
    p3 = plt.bar(ind, edge_latency, width,bottom=transport_latency,color="royalblue")

    plt.ylabel('end-to-end latency (ms)')
    plt.title('Best Partition Point of AlexNet')


    plt.xticks(ind, ('None','conv1', 'maxPool2d1', 'conv2', 'maxPool2d2', 'conv3',
                     'conv4', 'conv5', 'maxPool2d3', 'avgPool2d', 'flatten',
                     'linear1', 'linear2', 'linear3'))
    plt.xticks(rotation=-45)



    # plt.yticks(np.arange(0, 81, 10))
    # plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    # for a, b in zip(ind,end_to_end_latency):
    #     plt.text(a, b + 0.02, '%.3f' % b, ha='center', va='bottom', fontsize=7)


    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.legend((p1[0],p2[0],p3[0]), ('cloud latency', 'transport latency','edge latency'))
    plt.tight_layout()
    plt.show()