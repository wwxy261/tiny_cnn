//
// Created by xieyang on 2021/3/15.
//

#ifndef TINY_CNN_MOCKLOSS_H
#define TINY_CNN_MOCKLOSS_H

#include "../loss.h"

class MockLoss:public Loss{
public:
    void evaluate(const Matrix& pred, const Matrix& target);
};

#endif //TINY_CNN_MOCKLOSS_H
