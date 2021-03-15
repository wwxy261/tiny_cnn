//
// Created by xieyang on 2021/3/12.
//

#ifndef TINY_CNN_RELU_H
#define TINY_CNN_RELU_H

#include "../layer.h"

class Relu: public Layer{
public:
    void forward(const Matrix& data_input) override;
    void backward(const Matrix& data_input, const Matrix& grad_input) override;
};


#endif //TINY_CNN_RELU_H
