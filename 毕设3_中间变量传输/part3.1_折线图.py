import function
import numpy as np
import matplotlib.pyplot as plt


model_index = 1
model_names = ["alexnet", "vgg16", "googLeNet", "resnet18", "mobileNetv2","LeNet"]
path1 = "../res/DnnLayer_cuda.xls"
path2 = "../res/DnnLayer_mac.xls"
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




fig = plt.figure(figsize=(8,3.5))
ax = fig.add_subplot(111)

ind = np.arange(len(index1))


upperlimits = np.array([1, 0] * 2)
lowerlimits = np.array([0, 1] * 2)



ax.errorbar(ind,times1, yerr=0.1,label='Cloud')
ax.errorbar(ind,times2, yerr=0.1,label='Edge',linestyle="--")


ax.set_xticks(ind)
ax.set_xticklabels(name_list, rotation=-25)



# y = np.sin(np.arange(10.0) / 20.0 * np.pi) + 1
# plt.errorbar(x, y)


for a, b in zip(ind, times1):
    plt.text(a, b - 0.4, '%.3f' % b, ha='center', va='bottom', fontsize=7)

for a, b in zip(ind, times2):
    plt.text(a, b + 0.05, '%.3f' % b, ha='center', va='bottom', fontsize=7)



ax.set_ylabel('Computation Latency (ms)')

ax.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.tight_layout()
plt.show()