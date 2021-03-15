//
// Created by xieyang on 2021/3/12.
//

#ifndef TINY_CNN_NETWORK_H
#define TINY_CNN_NETWORK_H

#include "layer.h"
#include "loss.h"

class Network {
private:
    std::vector<Layer*> layers;
    Loss* loss;

public:
    Network():loss(nullptr){}
    void add_layer(Layer* layer) { layers.push_back(layer); }
    void add_loss(Loss* loss_in) { loss = loss_in; }
    void forward(const Matrix& input_data);
    void backward(const Matrix& input, const Matrix& target);
};
#endif //TINY_CNN_NETWORK_H
