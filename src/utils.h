//
// Created by xieyang on 2021/3/12.
//

#ifndef TINY_CNN_UTILS_H
#define TINY_CNN_UTILS_H

#include <iostream>
#include <algorithm>
#include <random>

#include "../lib/Eigen/Core"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vector;
typedef Eigen::Array<float, 1, Eigen::Dynamic> RowVector;

static std::default_random_engine generator;

// Normal distribution: N(mu, sigma^2)
inline void set_normal_random(float* arr, int n, float mu, float sigma) {
    std::normal_distribution<float> distribution(mu, sigma);
    for (int i = 0; i < n; i ++) {
        arr[i] = distribution(generator);
    }
}

#endif //TINY_CNN_UTILS_H
