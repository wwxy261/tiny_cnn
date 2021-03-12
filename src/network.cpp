//
// Created by xieyang on 2021/3/12.
//

#include "network.h"

void Network::forward(const Matrix& input_data) {
    if (layers.empty())
        return;
    layers[0]->forward(input_data);
    for (int i = 1; i < layers.size(); i++) {
        layers[i]->forward(layers[i-1]->data_output);
    }
}