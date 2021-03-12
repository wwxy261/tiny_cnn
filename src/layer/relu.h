//
// Created by xieyang on 2021/3/12.
//

#ifndef TINY_CNN_RELU_H
#define TINY_CNN_RELU_H

#include "../layer.h"

class Relu: public Layer{
public:
    void forward(const Matrix& data_input);
};


#endif //TINY_CNN_RELU_H
