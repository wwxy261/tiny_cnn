//
// Created by xieyang on 2021/3/12.
//

#ifndef TINY_CNN_CONV_RELU_MAX_POOLING_H
#define TINY_CNN_CONV_RELU_MAX_POOLING_H


#include "../layer.h"
#include "./conv.h"
#include "./max_pooling.h"

class Conv_relu_max_pooling: public Layer{
public:
    const int dim_in;
    int dim_out;

    int channel_in;
    int height_in;
    int width_in;
    int channel_out;
    int height_kernel;
    int width_kernel;
    int stride;
    int pad_h;
    int pad_w;

    int height_out;
    int width_out;

    Matrix weight;       // weight param, size=channel_in*h_kernel*w_kernel*channel_out
    Vector bias;         // bias param, size = channel_out
    Matrix grad_weight;  // gradient w.r.t weight
    Vector grad_bias;    // gradient w.r.t bias

    std::vector<Matrix> data_cols;

    int pooling_channel_in;
    int pooling_height_in;
    int pooling_width_in;
    int pooling_dim_in;

    int height_pool;            // pooling kernel height
    int width_pool;             // pooling kernel width
    int pooling_stride;         // pooling stride

    int pooling_channel_out;
    int pooling_height_out;
    int pooling_width_out;
    int pooling_dim_out;

    Matrix conv_result;
    Matrix relu_result;

    std::vector<std::vector<int> > max_idxs;  // index of max values

    void init();

public:
    Conv_relu_max_pooling(int channel_in, int height_in, int width_in, int channel_out,
                          int height_kernel, int width_kernel, int height_pool, int width_pool, int stride = 1, int stride_pool=1, int pad_w = 0,
                          int pad_h = 0):            dim_in(channel_in * height_in * width_in),
                                                     channel_in(channel_in), height_in(height_in), width_in(width_in),
                                                     channel_out(channel_out), height_kernel(height_kernel),
                                                     width_kernel(width_kernel), stride(stride), pad_w(pad_w), pad_h(pad_h),
                                                     height_pool(height_pool),width_pool(width_pool),pooling_stride(stride_pool){
        init();

    }
    void im2col(const Vector& image, Matrix& data_col);
    void col2im(const Matrix& data_col, Vector& image);
    void forward(const Matrix &data_input) override;
    void backward(const Matrix& data_input, const Matrix& grad_output) override;
};

#endif //TINY_CNN_CONV_RELU_MAX_POOLING_H
