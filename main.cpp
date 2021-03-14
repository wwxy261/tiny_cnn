#include <iostream>
#include <fstream>
#include <chrono>

#ifdef Intel_MKL
#define EIGEN_USE_MKL_ALL
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "src/mnist.h"
#include "src/layer.h"
#include "src/network.h"
#include "src/layer/conv.h"
#include "src/layer/relu.h"
#include "src/layer/max_pooling.h"
#include "src/layer/conv_relu_max_pooling.h"

const double eps = 0.0001;

bool check_flot_equal(float x, float y){
    return abs(x-y)<eps;
}

int main() {
    std::ofstream out;
    out.open("out.txt");
    MNIST dataset("../data/mnist/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int n_test = dataset.test_data.cols();
    int dim_in = dataset.test_data.rows();
    out << "mnist train number: " << n_train << std::endl;
    out << "mnist test number: " << n_test << std::endl;
    out << "mnist test size (28 * 28 * 1) = " << dim_in << std::endl;

#ifdef _OPENMP
    int numProcs = omp_get_num_procs();
    out << "omp_get_num_procs() = " << numProcs << std::endl;
#else
    out << "serial test" << std::endl;
#endif

    Network serial_net;
    Layer* conv1 = new Conv(1, 28, 28, 6, 5, 5, 1, 0, 0);
    Layer* rule1 = new Relu;
    Layer* pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    serial_net.add_layer(conv1);
    serial_net.add_layer(rule1);
    serial_net.add_layer(pool1);

    const int n_epoch = 100;

    auto start = std::chrono::system_clock::now();
    for(int i=0;i<n_epoch;i++){
        serial_net.forward(dataset.test_data);
    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    out << "seral_net time cost " << double(duration.count()) * std::chrono::microseconds::period::num /
    std::chrono::microseconds::period::den << " seconds." << std::endl;

    float result1 = (pool1->data_output.sum())/(pool1->data_output.rows())/(pool1->data_output.cols());


    Network fuse_net;
    Layer* conv_rule_pooling = new Conv_relu_max_pooling(1,28,28,
                                                         6,5,5,
                                                         2,2,1,2,0,0);
    start = std::chrono::system_clock::now();
    fuse_net.add_layer(conv_rule_pooling);
    for(int i=0;i<n_epoch;i++){
        fuse_net.forward(dataset.test_data);
    }
    end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    out << "fuse_net time cost " << double(duration.count()) * std::chrono::microseconds::period::num /
                                           std::chrono::microseconds::period::den << " seconds." << std::endl;

    float result2 = (conv_rule_pooling->data_output.sum())/(conv_rule_pooling->data_output.rows())/(conv_rule_pooling->data_output.cols());
    out << "output data size " << conv_rule_pooling->data_output.cols() << "x" << conv_rule_pooling->data_output.rows() << std::endl;
    if(check_flot_equal(result1,result2)){
        out << "test result pass" << std::endl;
    }else{
        out << "test result fail" << std::endl;
    }
}

