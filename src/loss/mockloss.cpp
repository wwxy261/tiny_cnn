//
// Created by xieyang on 2021/3/15.
//

#include "mockloss.h"

void MockLoss::evaluate(const Matrix &pred, const Matrix &target) {
    loss = 0.0;
    grad_out.resize(pred.rows(),pred.cols());
    set_normal_random(grad_out.data(), grad_out.size(), 1, 0.1);
}