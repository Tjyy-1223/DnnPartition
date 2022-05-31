import function
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':

    true_time = [119.166, 148.398, 35.81, 105.005, 28.324, 41.525, 35.808, 54.218, 17.055, 8.159, 11.091, 6.177, 6.047,6.18]
    predict_time = [85.930,110.725,30.219,80.007,23.026,39.600,28.492,28.492,11.475,11.475,11.475,8.964,8.964,7.452]

    # print(index)
    # print(times)

    ind = np.arange(len(true_time))  # the x locations for the groups
    width = 0.35  # the width of the bars

    # fig = plt.figure(figsize=(8,3.5))
    # ax1 = fig.add_subplot(111)
    # ax1.xaxis.set_ticks_position('bottom')
    # ax1.set_xticks(ind)
    # ax1.set_xticklabels(('input','conv1', 'maxPool2d1', 'conv2', 'maxPool2d2', 'conv3',
    #                  'conv4', 'conv5', 'maxPool2d3', 'avgPool2d', 'flatten',
    #                  'linear1', 'linear2', 'linear3'),rotation = -25)


    # lns1 = ax1.bar(ind - width / 2, true_time, width, label='Real Latency')
    #
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # # ax.set_ylabel('Scores')
    #
    # lns2 = ax1.bar(ind + width / 2, predict_time, width, color='darkblue', label='Predict Latency')
    #
    #
    # fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    # # ax1.title("")
    # ax1.set_ylabel('Latency(ms)')

    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    # plt.tight_layout()
    # plt.show()

    name_list = ['input','conv1', 'maxPool1', 'conv2', 'maxPool2', 'conv3',
                 'conv4', 'conv5', 'maxPool3', 'avgPool', 'flatten',
                 'linear1', 'linear2', 'linear3']


    fig = plt.figure(figsize=(8, 3.5))
    ax = fig.add_subplot(111)


    ax.errorbar(ind, true_time,label='real',marker='^')
    ax.errorbar(ind, predict_time, label='predict',marker='o')
    # ax.scatter(ind, true_time)
    # ax.scatter(ind, predict_time)


    ax.set_xticks(ind)
    ax.set_xticklabels(name_list, rotation=-25)

    # y = np.sin(np.arange(10.0) / 20.0 * np.pi) + 1
    # plt.errorbar(x, y)
    #
    # for a, b in zip(ind, true_time):
    #     plt.text(a, b - 0.4, '%.3f' % b, ha='center', va='bottom', fontsize=7)

    for a, b in zip(ind, predict_time):
        plt.text(a, b + 0.05, '%.3f' % b, ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Transmission Latency (ms)')

    ax.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.tight_layout()
    plt.show()