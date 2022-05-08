import function
import numpy as np
import matplotlib.pyplot as plt


model_index = 1
model_names = ["alexnet", "vgg16", "googLeNet", "resnet18", "mobileNetv2","LeNet"]
path1 = "../res/DnnLayer_cuda.xls"
path2 = "../res/DnnLayer_mac.xls"
path3 = "../res/cpu_gpu.xls"
sheet_name = model_names[model_index - 1]

name_list = ['conv1','maxPool1', 'conv2','maxPool2', 'conv3',
                      'conv4','conv5', 'maxPool3', 'avgPool', 'flatten',
                      'linear1','linear2','linear3']


index1 = function.get_excel_data(path1, sheet_name, "index")
times1 = function.get_excel_data(path1, sheet_name, "computation_time(ms)")
transport_num1 = function.get_excel_data(path1, sheet_name, "transport_num")

index2 = function.get_excel_data(path2, sheet_name, "index")
times2 = function.get_excel_data(path2, sheet_name, "computation_time(ms)")
transport_num2 = function.get_excel_data(path2, sheet_name, "transport_num")

transport_latency = function.get_excel_data(path3, sheet_name, "transport_latency")[1:]


fig = plt.figure(figsize=(8,3.5))
ax1 = fig.add_subplot(111)
ax1.xaxis.set_ticks_position('bottom')

ind = np.arange(len(index1))


upperlimits = np.array([1, 0] * 2)
lowerlimits = np.array([0, 1] * 2)

ax2 = ax1.twinx()

ax1.plot(ind,transport_num1,label='Output Size',color='mediumturquoise',alpha = 0.9)
ax1.scatter(ind,transport_num1,color='mediumturquoise',alpha = 0.9)
ax2.plot(ind,transport_latency,label='Transmission Latency',color='goldenrod',alpha = 0.9)
ax2.scatter(ind,transport_latency,color='goldenrod',alpha = 0.9)
# ax2.plot((0,14),(15,15),color="red")

ax1.set_xticks(ind)
ax1.set_xticklabels(name_list, rotation=-25)



# y = np.sin(np.arange(10.0) / 20.0 * np.pi) + 1
# plt.errorbar(x, y)


# for a, b in zip(ind, times1):
#     plt.text(a, b - 0.4, '%.3f' % b, ha='center', va='bottom', fontsize=7)
#
# for a, b in zip(ind, times2):
#     plt.text(a, b + 0.05, '%.3f' % b, ha='center', va='bottom', fontsize=7)
#


ax1.set_ylabel('Output Size')
ax2.set_ylabel('Transmission Latency(ms)')
# ax2.ylim((0, 300))

fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.tight_layout()
plt.show()