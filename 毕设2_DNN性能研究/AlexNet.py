import torch
import a1_alexNet
device = "cuda" if torch.cuda.is_available() else "cpu"


"""
    导入alexnet模型
"""
alexnet = a1_alexNet.AlexNet(input_layer=3,num_classes=1000)
alexnet = alexnet.to(device)

"""
    set_index 为分割点
"""
set_index = 0
edge_model,cloud_model = a1_alexNet.model_partition(alexnet,set_index)

print(f"alexnet model : {len(alexnet)}")



"""
    模拟数据输入 
"""
x = torch.rand(size=(1000,3,224,224))
x = x.to(device)
print(f"x device : {x.device}")


x = a1_alexNet.show_features(cloud_model,x)