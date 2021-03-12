//
// Created by xieyang on 2021/3/12.
//

#ifndef TINY_CNN_NETWORK_H
#define TINY_CNN_NETWORK_H

#include "layer.h"

class Network {
private:
    std::vector<Layer *> layers;  // layer pointers

public:
    Network(){}
    void add_layer(Layer *layer) { layers.push_back(layer); }
    void forward(const Matrix& input_data);

};
#endif //TINY_CNN_NETWORK_H
