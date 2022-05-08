import function
import numpy as np
import matplotlib.pyplot as plt

edge_value = [0.001,25.09,13.528]
transmission_value = [119.166,6.18,8.159]
end_value = [122.033,31.272,23.269]
ind = np.arange(len(edge_value))


# Now switch to a more OO interface to exercise more features.
fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True,sharey=True,figsize=(11.5,4))
name_list = ['Cloud-only','Edge-only','Edge-cloud']

width = 0.35
ax = axs[0]
ax.errorbar(ind, edge_value,color='steelblue')
ax.bar(ind,edge_value,width/2,color='steelblue',alpha = 0.8)
ax.set_title('edge processing')
ax.set_ylabel('Latency (ms)')
ax.set_xticks(ind)
ax.set_xticklabels(name_list, rotation=-25)
for a, b in zip(ind,edge_value):
    ax.text(a, b + 0.2, '%.3f' % b, ha='center', va='bottom', fontsize=7)


ax = axs[1]
ax.errorbar(ind, transmission_value,color='darkgoldenrod')
ax.bar(ind,transmission_value,width/2,color='darkgoldenrod',alpha = 0.8)
ax.set_title('data communication')
ax.set_xticks(ind)
ax.set_xticklabels(name_list, rotation=-25)
for a, b in zip(ind,transmission_value):
    ax.text(a, b + 0.5, '%.3f' % b, ha='center', va='bottom', fontsize=7)


ax = axs[2]
ax.errorbar(ind, end_value,color='darkolivegreen')
ax.bar(ind,end_value,width/2,color='darkolivegreen',alpha = 0.8)
ax.set_title('alexnet processing')
ax.set_xticks(ind)
ax.set_xticklabels(name_list, rotation=-25)
for a, b in zip(ind,end_value):
    ax.text(a, b + 0.2, '%.3f' % b, ha='center', va='bottom', fontsize=7)



# fig.suptitle('Errorbar subsampling for better appearance')
plt.tight_layout()
plt.show()