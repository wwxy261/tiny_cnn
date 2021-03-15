//
// Created by xieyang on 2021/3/15.
//

#ifndef TINY_CNN_LOSS_H
#define TINY_CNN_LOSS_H

#include "./utils.h"

class Loss {
protected:
    float loss;             // value of loss
    Matrix grad_out;

public:
    virtual ~Loss() {}
    virtual void evaluate(const Matrix& pred, const Matrix& target) = 0;
    virtual float output() { return loss; }
    virtual const Matrix& back_gradient() { return grad_out; }
};
#endif //TINY_CNN_LOSS_H
