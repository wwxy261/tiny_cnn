//
// Created by xieyang on 2021/3/12.
//

#ifndef TINY_CNN_CONV_H
#define TINY_CNN_CONV_H

#include "../layer.h"

class Conv: public Layer{
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



    void init();

public:
    Conv(int channel_in, int height_in, int width_in, int channel_out,
         int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,
         int pad_h = 0) :
            dim_in(channel_in * height_in * width_in),
            channel_in(channel_in), height_in(height_in), width_in(width_in),
            channel_out(channel_out), height_kernel(height_kernel),
            width_kernel(width_kernel), stride(stride), pad_w(pad_w), pad_h(pad_h)
    { init(); }

    void im2col(const Vector& image, Matrix& data_col);
    void col2im(const Matrix& data_col, Vector& image);

    void forward(const Matrix& data_input) override;

    void backward(const Matrix& data_input, const Matrix& grad_input) override;

};

#endif //TINY_CNN_CONV_H
