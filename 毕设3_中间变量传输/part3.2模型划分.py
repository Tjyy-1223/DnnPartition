import function
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    model_index = 6
    model_names = ["alexnet", "vgg16", "googLeNet", "resnet18", "mobileNetv2","lenet"]
    path = "../res/cpu_gpu.xls"
    sheet_name = model_names[model_index - 1]

    list_2 = ['None','Conv1', 'Conv2', 'MaxPool1', 'Conv3', 'Conv4', 'MaxPool2', 'Conv5', 'Conv6', 'Conv7', 'MaxPool3',
              'Conv8', 'Conv9', 'Conv10', 'MaxPool4', 'Conv11', 'Conv12', 'Conv13', 'MaxPool5', 'AvgPool', 'Flatten',
              'Linear1',
              'Linear2', 'Linear3']

    list_3 = ['None','Conv1', 'MaxPool1', 'Conv2', 'Conv3', 'MaxPool2', 'Inception1', 'Inception2',
              'MaxPool3', 'Inception3', 'Inception4', 'Inception5', 'Inception6', 'Inception7',
              'MaxPool4', 'Inception8', 'Inception9', 'AvgPool', 'Flatten', 'Linear']

    list_4 = ['None','Conv1', 'MaxPool1', 'BasicBlock1', 'BasicBlock2', 'BasicBlock3', 'BasicBlock4',
              'BasicBlock5', 'BasicBlock6', 'BasicBlock7', 'BasicBlock8', 'AvgPool', 'Flatten', 'Linear']

    list_5 = ['None','ConvBlock1', 'ResidualBlock1', 'ResidualBlock2', 'ResidualBlock3', 'ResidualBlock4', 'ResidualBlock5',
              'ResidualBlock6', 'ResidualBlock7', 'ResidualBlock8', 'ResidualBlock9', 'ResidualBlock10',
              'ResidualBlock11', 'ResidualBlock12',
              'ResidualBlock13', 'ResidualBlock14', 'ResidualBlock15', 'ResidualBlock16', 'ResidualBlock17',
              'ConvBlock2', 'AvgPool', 'Flatten', 'Linear']

    list_6 = ['None','conv1', 'maxPool1', 'conv2', 'maxPool2', 'flatten', 'linear1', 'linear2', 'linear3']


    index = function.get_excel_data(path, sheet_name, "index")
    layerName = function.get_excel_data(path, sheet_name, "layerName")

    end_to_end_latency = function.get_excel_data(path, sheet_name, "end-to-end latency")
    transport_latency = function.get_excel_data(path, sheet_name, "transport_latency")
    edge_latency = function.get_excel_data(path, sheet_name, "edge_latency")
    cloud_latency = function.get_excel_data(path, sheet_name, "cloud_latency")

    temp_list = []
    for i in range(len(edge_latency)):
        temp_list.append(edge_latency[i] + transport_latency[i])
    # print(end_to_end_latency)

    # print(index)
    # print(times)


    plt.figure(figsize=(5,3.5))
    N = len(index)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, edge_latency, width, color='steelblue', alpha=0.8)
    p2 = plt.bar(ind, transport_latency, width, bottom=edge_latency,color='darkgoldenrod',alpha = 0.8)
    p3 = plt.bar(ind, cloud_latency, width, bottom=temp_list, color="darkolivegreen",alpha = 0.8)



    plt.ylabel('latency (ms)')
    # plt.title('Best Partition Point of AlexNet')


    plt.xticks(ind, list_6)
    plt.xticks(rotation=-45)



    # plt.yticks(np.arange(0, 81, 10))
    # plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    # for a, b in zip(ind,end_to_end_latency):
    #     plt.text(a, b + 0.02, '%.3f' % b, ha='center', va='bottom', fontsize=7)


    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.legend((p3[0],p2[0],p1[0]), ('cloud latency', 'transmission latency','edge latency'))
    plt.tight_layout()
    plt.show()