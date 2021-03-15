//
// Created by xieyang on 2021/3/12.
//

#include "relu.h"

void Relu::forward(const Matrix &data_input) {
    // a = z*(z>0)
    data_output = data_input.cwiseMax(0.0);
}

void Relu::backward(const Matrix &data_input, const Matrix &grad_input) {
    // d(L)/d(z_i) = d(L)/d(a_i) * d(a_i)/d(z_i)
    //             = d(L)/d(a_i) * 1*(z_i>0)
    Matrix positive = (data_input.array() > 0.0).cast<float>();
    grad_output = grad_input.cwiseProduct(positive);
}