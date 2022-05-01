import function
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    model_index = 1
    model_names = ["alexnet", "vgg16", "googLeNet", "resnet18", "mobileNetv2"]
    path = "../res/DnnLayer_cuda.xls"
    sheet_name = model_names[model_index - 1]

    index = function.get_excel_data(path, sheet_name, "index")
    layerName = function.get_excel_data(path, sheet_name, "layerName")
    times = function.get_excel_data(path, sheet_name, "computation_time(ms)")
    transport_num = function.get_excel_data(path, sheet_name, "transport_num")

    # print(index)
    # print(times)


    ind = np.arange(len(index))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig = plt.figure(figsize=(8,3.5))
    ax1 = fig.add_subplot(111)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(('conv1', 'maxPool2d1', 'conv2', 'maxPool2d2', 'conv3',
                     'conv4', 'conv5', 'maxPool2d3', 'avgPool2d', 'flatten',
                     'linear1', 'linear2', 'linear3'),rotation = -45)
    lns1 = ax1.bar(ind - width / 2, times, width, label='latency')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores')

    ax2 = ax1.twinx()
    ax2.set_xticks(ind)
    ax2.set_xticklabels(('conv1', 'maxPool2d1', 'conv2', 'maxPool2d2', 'conv3',
                         'conv4', 'conv5', 'maxPool2d3', 'avgPool2d', 'flatten',
                         'linear1', 'linear2', 'linear3'), rotation=-45)
    lns2 = ax2.bar(ind + width / 2, transport_num, width, color='darkblue', label='shape')

    ax2.set_title('Latency And Shape of AlexNet')

    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    ax1.set_ylabel('Latency(ms)')
    ax2.set_ylabel('Shape')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.tight_layout()
    plt.show()