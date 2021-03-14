//
// Created by xieyang on 2021/3/12.
//

#include "conv.h"

void Conv::init() {
    height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
    width_out =   (1 + (width_in - width_kernel + 2 * pad_w) / stride);
    dim_out = height_out * width_out * channel_out;

    weight.resize(channel_in * height_kernel * width_kernel, channel_out);
    bias.resize(channel_out);
    grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
    grad_bias.resize(channel_out);
    set_normal_random(weight.data(), weight.size(), 0, 0.01);
    set_normal_random(bias.data(), bias.size(), 0, 0.01);
    //std::cout << weight.colwise().sum() << std::endl;
    //std::cout << weight.colwise().sum() + bias.transpose() << std::endl;
}

// im2col, used for bottom
// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void Conv::im2col(const Vector& image, Matrix& data_col) {
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;
    // im2col
    data_col.resize(hw_out, hw_kernel * channel_in);
    for (int c = 0; c < channel_in; c ++) {
        Vector map = image.block(hw_in * c, 0, hw_in, 1);  // c-th channel map
        for (int i = 0; i < hw_out; i ++) {
            int step_h = i / width_out;
            int step_w = i % width_out;
            int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
            for (int j = 0; j < hw_kernel; j ++) {
                int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
                int cur_row = start_idx / width_in + j / width_kernel - pad_h;
                if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
                    cur_row >= height_in) {
                    data_col(i, c * hw_kernel + j) = 0;
                }
                else {
                    //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
                    int pick_idx = cur_row * width_in + cur_col;
                    data_col(i, c * hw_kernel + j) = map(pick_idx);  // pick which pixel
                }
            }
        }
    }
}

void Conv::forward(const Matrix& data_input) {
    int n_sample = data_input.cols();
    data_output.resize(height_out * width_out * channel_out, n_sample);
    data_cols.resize(n_sample);
    #pragma omp parallel for
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
