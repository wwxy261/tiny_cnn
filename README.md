# tiny_CNN性能测试

*Convolution, Relu 以及Pooling是卷积神经网络的基本单元，这个项目采用C++实现了这三个基本单元的Forward以及Backward计算。并且实现了将这三个单元组合为一体的模块以提高性能，组合后有1.2倍的性能提升。该项目采用OpenMP支持多核并行。*

## 目录

  1. [Introduction](#introduction) 
  2. [Results](#results)


## Introduction
### 1. 项目结构
./src 源码

./lib 开源第三方矩阵库Eigen(底层可调用MKL)

./data mnist数据集

./python numpy实现算法demo以及Pytorch搭建网络作为benchmark

./main.cpp 测试性能代码

### 2. Build
```
mkdir build
cd build
cmake ../
make
```

### 3. 算法简介
1. Convolution Caffe中经典的im2col算法实现

2. Convolution和MaxPooling实现中对最外层的n_sample循环加入parallel for优化

3. 将Convolution,Relu,Max_Pooling融合为一个模块，同样在最外层加入parallel for优化


### 4. 测试过程
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
### 1. 测试环境
Mac CPU(2 GHz 四核Intel Core i5)

HPC (2.5GHz 24核Core) ICC编译优化


### 2. Forward results
调用n次从网络输入到输出的forward计算并计时测试程序性能，结果如下表所示
| n    | Pytorch  | serial    |fuse    | 24-core | 24-core-fuse|naive-numpy | fuse_numpy
|  :---- | :----    | :---    |:---    | :---    |   :---  |   :---         |   :---
| n=1  | 0.582s | 1.042s |0.995s | 0.452s | 0.431s |   332.457s|   64.329s
| n=10 | 4.467s | 7.357s |7.850s | 1.433s | 1.254s |      -      |   -   |
| n=100| 42.883s| 69.868s|77.798s| 10.548s| 9.938s|       -      |   -   |


### 3. Backward results
调用n次从网络输出到输入的backward计算并计时测试程序性能,结果如下表所示




| n    | serial | fuse   | 24-core | 24-core-fuse |
| :--- | :---   | :---   | :---    |   :---       |
| n=1  | 1.082s | 0.956s | 0.470s  |   0.363s     |
| n=10 | 9.861s | 9.351s | 4.496s  |   3.478s     |
| n=100| 97.247s| 90.616s| 41.811s |   33.760s    |

**[⬆ back to top](#目录)**



