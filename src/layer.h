//
// Created by xieyang on 2021/3/12.
//

#ifndef TINY_CNN_LAYER_H
#define TINY_CNN_LAYER_H

#include "utils.h"

class Layer{
public:
    Matrix data_output;       // layer output
    Matrix grad_output;        // gradient w.r.t input

    virtual void forward(const Matrix& data_input) = 0;
    virtual void backward(const Matrix& data_input, const Matrix& grad_input) = 0 ;
    virtual const Matrix& output() { return data_output; }
    virtual const Matrix& back_gradient() { return grad_output; }

};

#endif //TINY_CNN_LAYER_H
