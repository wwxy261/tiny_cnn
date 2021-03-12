//
// Created by xieyang on 2021/3/12.
//

#ifndef TINY_CNN_LAYER_H
#define TINY_CNN_LAYER_H

#include "utils.h"

class Layer{
public:
    Matrix data_output;       // layer output
    Matrix grad_input;        // gradient w.r.t input

    virtual void forward(const Matrix& data_input) = 0;


};

#endif //TINY_CNN_LAYER_H
