# tiny_CNN性能测试

*convolution, relu 以及pooling是卷积神经网络的基本单元，这个项目采用C++实现了这三个基本单元的Forward以及Backward接口。以及实现了将这三个单元组合为一体的接口以提高性能，组合后的接口有1.2倍的性能提升。该项目采用OpenMP支持多核并行。*


[![Downloads](https://img.shields.io/npm/dm/eslint-config-airbnb.svg)](https://www.npmjs.com/package/eslint-config-airbnb)
[![Downloads](https://img.shields.io/npm/dm/eslint-config-airbnb-base.svg)](https://www.npmjs.com/package/eslint-config-airbnb-base)


## 目录

  1. [Introduction](#introduction) 
  2. [Results](#results)
  3. [Objects](#objects)
  


## Introduction
### 1. 项目结构
./src 源码

./lib 开源第三方矩阵库Eigen(底层可调用MKL)

./data mnist数据集

./python numpy实现算法demo以及Pytorch 作benchmark

./main.cpp 测试性能代码

### 2. Build
```
mkdir build
cd build
cmake ../
make
```
### 3. 测试过程
使用mnist测试数据集输入，采用如下Pytorch定义的简化CNN网络模型(./python/mnist_cnn_pytorch.py)。
 ```python
 class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
   

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 12 * 12)
        return x
 ```


**[⬆ 回到顶部](#目录)**

## Results
测试使用mac系统本地测试CPU(2 GHz 四核Intel Core i5)以及南京大学HPC集群(2.5GHz 24核Core)。
后者还是用了Intel ICC编译器优化。

### 1. forward results
调用n次从网络输入到输出forward计时测试程序性能。
| n    | Pytorch  | serial  |fuse    | 24-core | 24-core-fuse|naive-numpy | fuse_numpy
|  :---- | :----    | :---    |:---    | :---    |   :---  |   :---         |   :---
| n=1  | 0.582s | 1.042s |0.995s | 0.452s | 0.431s |   332.457s|   64.329s
| n=10 | 4.467s | 7.357s |7.850s | 1.433s | 1.254s |      -      |   -   |
| n=100| 42.883s| 69.868s|77.798s| 10.548s| 9.938s|       -      |   -   |



**[⬆ back to top](#目录)**

## Objects



**[⬆ back to top](#目录)**

