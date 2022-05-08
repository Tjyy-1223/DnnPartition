import function
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    model_index = 6
    model_names = ["alexnet", "vgg16", "googLeNet", "resnet18", "mobileNetv2","LeNet"]
    path = "../res/DnnLayer_cuda.xls"
    sheet_name = model_names[model_index - 1]

    list_2 = ['Conv1','Conv2','MaxPool1','Conv3','Conv4','MaxPool2','Conv5','Conv6','Conv7','MaxPool3',
              'Conv8','Conv9','Conv10','MaxPool4','Conv11','Conv12','Conv13','MaxPool5','AvgPool', 'Flatten','Linear1',
              'Linear2','Linear3']

    list_3 = ['Conv1','MaxPool1','Conv2','Conv3','MaxPool2','Inception1','Inception2',
              'MaxPool3','Inception3','Inception4','Inception5','Inception6','Inception7',
              'MaxPool4','Inception8','Inception9','AvgPool', 'Flatten','Linear']


    list_4 = ['Conv1','MaxPool1','BasicBlock1','BasicBlock2','BasicBlock3','BasicBlock4',
              'BasicBlock5','BasicBlock6','BasicBlock7','BasicBlock8','AvgPool', 'Flatten','Linear']


    list_5 = ['ConvBlock1','ResidualBlock1','ResidualBlock2','ResidualBlock3','ResidualBlock4','ResidualBlock5',
             'ResidualBlock6','ResidualBlock7','ResidualBlock8','ResidualBlock9','ResidualBlock10','ResidualBlock11','ResidualBlock12',
             'ResidualBlock13','ResidualBlock14','ResidualBlock15','ResidualBlock16','ResidualBlock17','ConvBlock2','AvgPool', 'Flatten','Linear']


    list_6 = ['conv1', 'maxPool1', 'conv2','maxPool2', 'flatten','linear1','linear2','linear3']


    index = function.get_excel_data(path, sheet_name, "index")
    layerName = function.get_excel_data(path, sheet_name, "layerName")
    times = function.get_excel_data(path, sheet_name, "computation_time(ms)")
    transport_num = function.get_excel_data(path, sheet_name, "transport_num")

    # print(index)
    # print(times)


    ind = np.arange(len(index))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig = plt.figure(figsize=(5,3.5))
    ax1 = fig.add_subplot(111)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xticks(ind)
    # ax1.set_xticklabels(('conv1', 'relu','maxPool1', 'conv2','relu','maxPool2', 'conv3','relu',
    #                  'conv4', 'relu','conv5', 'relu','maxPool3', 'avgPool', 'flatten',
    #                  'linear1','relu','linear2','relu','linear3'),rotation = -25)
    ax1.set_xticklabels(list_6, rotation=-25)


    lns1 = ax1.bar(ind - width / 2, times, width, label='Latency',color="darkgoldenrod",alpha = 0.9)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores')

    ax2 = ax1.twinx()
    ax2.set_xticks(ind)
    # ax2.set_xticklabels(('conv1', 'relu','maxPool1', 'conv2','relu','maxPool2', 'conv3','relu',
    #                  'conv4', 'relu','conv5', 'relu','maxPool3', 'avgPool', 'flatten',
    #                  'linear1','relu','linear2','relu','linear3'),rotation = -25)
    # ax2.set_xticklabels(('conv1', 'relu', 'maxPool1', 'conv2', 'relu', 'maxPool2', 'conv3', 'relu',
    #                      'conv4', 'relu', 'conv5', 'relu', 'maxPool3', 'avgPool', 'flatten',
    #                      'linear1', 'relu', 'linear2', 'relu', 'linear3'), rotation=-25)


    lns2 = ax2.bar(ind + width / 2, transport_num, width, color='royalblue', label='Output Size',alpha = 0.9)

    # ax2.set_title('Latency And Shape of AlexNet')

    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    ax1.set_ylabel('Latency (ms)')
    ax2.set_ylabel('Output Size')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.tight_layout()
    plt.show()