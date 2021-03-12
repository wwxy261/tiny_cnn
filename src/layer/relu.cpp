//
// Created by xieyang on 2021/3/12.
//

#include "relu.h"

void Relu::forward(const Matrix &data_input) {
    data_output = data_input.cwiseMax(0.0);
}
