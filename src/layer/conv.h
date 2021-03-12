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

    void forward(const Matrix& data_input) {
        int n_sample = data_input.cols();
        data_output.resize(height_out * width_out * channel_out, n_sample);
        data_cols.resize(n_sample);
        for (int i = 0; i < n_sample; i ++) {
            // im2col
            Matrix data_col;
            im2col(data_input.col(i), data_col);
            data_cols[i] = data_col;
            // conv by product
            Matrix result = data_col * weight;  // result: (hw_out, channel_out)
            result.rowwise() += bias.transpose();
            data_output.col(i) = Eigen::Map<Vector>(result.data(), result.size());
        }
    }

};

#endif //TINY_CNN_CONV_H
