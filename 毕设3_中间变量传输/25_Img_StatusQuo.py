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


    ind = np.arange(6)  # the x locations for the groups
    width = 0.33  # the width of the bars

    # fig = plt.figure(figsize=(8,3.5))



    cloud_compute = [122.023,114.487,71.982,102.303,86.99,102.829]
    edge_compute = [31.272,210.779,55.959,46.905,47.033,17.167]
    end_to_end_compute = [23.269,114.487,53.909,46.905,41.963,15.070]


    fig, ax1 = plt.subplots(figsize=(7,3))
    # ax1 = fig.add_subplot(111)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(('AlexNet','Vgg-16','GoogLeNet','ResNet-18','MobileNet v2','LeNet'),rotation = -15)
    lns1 = ax1.bar(ind - width / 2, cloud_compute,width/2, color="darkolivegreen",label='Cloud-Only',alpha = 0.8)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores')

    # ax2 = ax1.twinx()
    # ax2.set_xticks(ind)
    # ax2.set_xticklabels(('conv1', 'maxPool2d1', 'conv2', 'maxPool2d2', 'conv3',
    #                      'conv4', 'conv5', 'maxPool2d3', 'avgPool2d', 'flatten',
    #                      'linear1', 'linear2', 'linear3'), rotation=-45)
    lns2 = ax1.bar(ind , edge_compute, width/2, color='steelblue', label='Edge-Only',alpha = 0.8)
    ax1.bar(ind+ width / 2,end_to_end_compute,width/2,color='darkgoldenrod',label='Edge-Cloud',alpha = 0.8)
    # ax2.set_title('Latency And Shape of AlexNet')

    # fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    ax1.set_ylabel('Latency(ms)')
    # ax2.set_ylabel('Shape')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.legend()
    plt.tight_layout()
    plt.show()