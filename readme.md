# 关于EM算法的模拟实验


对于来自两正态总体的样本，通过em算法估计每个样本属于的组别，进而估计每一组的参数：均值和方差。

目的：探索EM算法对于盲态数据参数估计的准确性。

探索的条件：

* 样本量(nsamples)
* 组件差异(delta)
* 是否估计先验分类的参数(estimated_gamma, freeze_gamma)
* 初始化的准确性(true_init, random_init)
* 估计方差的模式(equal_var, noequal_var, freeze_var)
