# GADS
**简介：**

针对单智能体终端计算资源不足、情境动态变化问题，GADS算法探索由多个共存智能体组成动态协作群组中模型分割点的快速自适应搜索问题。GADS算法利用串行协同计算技术，根据动态情境（带宽、设备电量等）实现智能体终端自组织、自适应、可伸缩的高效感知计算。

**主要功能：**

探究模型配置与资源变化的适应性机理，在多维性能优化空间中研究资源状态的动态变化方向，提出最佳边/端计算配置的近邻搜索效应快速搜索到资源状态动态迁移后的深度模型最优边端分割点，实现情境实时自适应的深度模型协同计算。

**配置：**

| 算法名称     | 基于图的深度模型自适应手术刀算法（Graph-based Adaptive DNN Surgery, GADS） |
| ------ | ------ |
| 算法接口	| python gads.py --config_file config.yaml| 
| 输入	| 由深度模型及部署情境，构建的分割状态图G| 
| 输出	| 根据动态运行情境快速自适应调优的模型分割点R_op| 
| 支持数据集| 	CiFar-10、CiFar-100、ImageNet、BDD100K| 
| 依赖库| 	Python 3.6+、numpy、torch、tensorflow1.8-1.13| 
| 参考资源		| Kang Y, Hauswald J, Gao C, et al. Neurosurgeon: Collaborative intelligence between the cloud and mobile edge[J]. ACM SIGARCH Computer Architecture News, 2017, 45(1): 615-629.| 
