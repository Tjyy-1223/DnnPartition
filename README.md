# DnnPartition
**本科毕业设计-基于云边协同的DNN分层实现模式**

+ 利用多种深度学习计算框架实现DNN模型分层推理机制。

+ 探究影响模型分层结果的因素。

+ 探究和改进DNN模型分层在现有主流深度学习框架下实现层面的模式。



#### 00_BasicKonwledge

+ 包含了对pytorch和tensorflow深度学习架构的学习过程
+ 学习参考资料：李沐 - 动手学深度学习 b站有视频



#### 01_Pytorch_Tutorials

+ 对于pytorch官方的一些学习资料



#### 毕设0_读取相关数据集

+ 包含了对fashion_mnist数据集的读取过程



#### 毕设1_六种深度学习网络架构

+ 选取六种不同的深度学习网络架构
+ 分别使用tensorflow和pytorch对选定的深度学习网络架构进行实现



#### 毕设2_DNN性能研究(重点1)

+ **function.py**

封装了DNN分层实现模式中可能用到的函数（模型分割、执行输出、录入到excel表等），包含详细的注释。

+ **a1_alexNet - a6_LeNet.py**：

分别对AlexNet、VggNet、GoogLeNet、ResNet、MobileNet以及LeNet六种深度学习网络架构进行改进，通过补充**iter**，**getitem**函数使其能够被循环和通过index直接取出某层。

+ **01_AlexNet_Prediction.ipynb**：对AlexNet模型的训练和分层执行进行测试
+ **02_AlexNet_Partition.ipynb**：对AlexNet的模型划分和云边端协同执行进行测试
+ **03_Clound_Edge_Run.ipynb**：演示AlexNet的云边协同执行过程进行测试
+ **04_Simplify_Cloud_Edge.ipynb**：简化后的AlexNet云边协同测试
+ **05_vgg16_Prediction.ipynb**：模拟vgg16的分层执行过程
+ **06_GoogLeNet_Predicition.ipynb**：模拟GoogLeNet的分层执行过程
+ **07_resnet18_Prediction.ipynb**：模拟ResNet-18的分层执行过程
+ **08_mobilev2_Prediction.ipynb**：模拟MobileNet的分层执行过程

+ **example1_AlexNet.py** - **example5_MobileNetV2.py** ：模拟DNN模型的 分割过程 和 执行过程的展示
+ **example6_DNN.py**：统计每层的执行特征，并将执行特征保存到excel表格中
+ **example7_DNNImg.py**：统计每层的执行特征，并将执行特征保存到excel表格中



#### 毕设3_中间变量传输研究（重点2）

中间变量传输 + 云边协同执行过程模拟 + 绘制图像
