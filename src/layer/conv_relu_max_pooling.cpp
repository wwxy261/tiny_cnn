//
// Created by xieyang on 2021/3/12.
//

#include "conv_relu_max_pooling.h"


void Conv_relu_max_pooling::init(){
    height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
    width_out =   (1 + (width_in - width_kernel + 2 * pad_w) / stride);
    dim_out = height_out * width_out * channel_out;

    weight.resize(channel_in * height_kernel * width_kernel, channel_out);
    bias.resize(channel_out);
    grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
    grad_bias.resize(channel_out);
    set_normal_random(weight.data(), weight.size(), 0, 0.01);
    set_normal_random(bias.data(), bias.size(), 0, 0.01);

    pooling_channel_in = channel_out;
    pooling_height_in = height_out;
    pooling_width_in = width_out;

    pooling_channel_out = pooling_channel_in;
    pooling_height_out =  (1 + std::ceil((pooling_height_in - height_pool) * 1.0 / pooling_stride));
    pooling_width_out =   (1 + std::ceil((pooling_width_in - height_pool) * 1.0 / pooling_stride));
    pooling_dim_out = pooling_height_out * pooling_width_out * pooling_channel_out;
}

// im2col, used for bottom
// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void Conv_relu_max_pooling::im2col(const Vector& image, Matrix& data_col) {
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;
    // im2col
    data_col.resize(hw_out, hw_kernel * channel_in);
    #pragma omp parallel for
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

void Conv_relu_max_pooling::forward(const Matrix &data_input) {
    int n_sample = data_input.cols();
    conv_result.resize(height_out * width_out * channel_out, n_sample);
    relu_result.resize(height_out * width_out * channel_out, n_sample);
    data_cols.resize(n_sample);

    int hw_in = pooling_height_in * pooling_width_in;
    int hw_pool = height_pool * width_pool;
    int hw_out = pooling_height_out * pooling_width_out;

    data_output.resize(pooling_dim_out, n_sample);
    data_output.setZero();
    data_output.array() += std::numeric_limits<float>::lowest();
    max_idxs.resize(n_sample, std::vector<int>(pooling_dim_out, 0));

    for (int i = 0; i < n_sample; i ++) {
        // im2col
        Matrix data_col;
        im2col(data_input.col(i), data_col);
        data_cols[i] = data_col;
        // conv by product
        Matrix result = data_col * weight;  // result: (hw_out, channel_out)
        result.rowwise() += bias.transpose();
        conv_result.col(i) = Eigen::Map<Vector>(result.data(), result.size());
        relu_result.col(i) = conv_result.col(i).cwiseMax(0.0);
        Vector image = relu_result.col(i);
        for (int c = 0; c < pooling_channel_in; c ++) {
            for (int i_out = 0; i_out < hw_out; i_out ++) {
                int step_h = i_out / pooling_width_out;
                int step_w = i_out % pooling_width_out;
                // left-top idx of window in raw image
                int start_idx = step_h * pooling_width_in * pooling_stride + step_w * pooling_stride;
                for (int i_pool = 0; i_pool < hw_pool; i_pool ++) {
                    if (start_idx % pooling_width_in + i_pool % width_pool >= pooling_width_in ||
                        start_idx / pooling_width_in + i_pool / width_pool >= pooling_height_in) {
                        continue;  // out of range
                    }
                    int pick_idx = start_idx + (i_pool / width_pool) * pooling_width_in
                                   + i_pool % width_pool + c * hw_in;
                    if (image(pick_idx) >= data_output(c * hw_out + i_out, i)) {  // max pooling
                        data_output(c * hw_out + i_out, i) = image(pick_idx);
                        max_idxs[i][c * hw_out + i_out] = pick_idx;
                    }
                }
            }
        }
    }
}

void Conv_relu_max_pooling::backward(const Matrix &data_input, const Matrix &grad_input) {
    Matrix max_pooling_grad_output(relu_result.rows(), relu_result.cols());
    max_pooling_grad_output.setZero();

    int n_sample = data_input.cols();
    grad_weight.setZero();
    grad_bias.setZero();
    grad_output.resize(height_in * width_in * channel_in, n_sample);
    grad_output.setZero();

    #pragma omp parallel for
    for (int i = 0; i < max_idxs.size(); i ++) {  // i-th sample

        for (int j = 0; j < max_idxs[i].size(); j ++) {
            max_pooling_grad_output(max_idxs[i][j], i) += grad_input(j, i);
        }

        Matrix positive = (conv_result.col(i).array() > 0.0).cast<float>();
        //Matrix rule_grad_output = max_pooling_grad_output.col(i).cwiseProduct(positive);

        // im2col of grad_top
        Matrix grad_input_i = max_pooling_grad_output.col(i).cwiseProduct(positive);
        Matrix grad_input_i_col = Eigen::Map<Matrix>(grad_input_i.data(),
                                                     height_out * width_out, channel_out);
        // d(L)/d(w) = \sum{ d(L)/d(z_i) * d(z_i)/d(w) }
        grad_weight += data_cols[i].transpose() * grad_input_i_col;
        // d(L)/d(b) = \sum{ d(L)/d(z_i) * d(z_i)/d(b) }
        grad_bias += grad_input_i_col.colwise().sum().transpose();
        // d(L)/d(x) = \sum{ d(L)/d(z_i) * d(z_i)/d(x) } = d(L)/d(z)_col * w'
        Matrix grad_output_i_col = grad_input_i_col * weight.transpose();
        // col2im of grad_input
        Vector grad_output_i;
        col2im(grad_output_i_col, grad_output_i);
        grad_output.col(i) = grad_output_i;
    }

}

// col2im, used for grad backward
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
// image size: Vector (height_in * width_in * channel_in)
void Conv_relu_max_pooling::col2im(const Matrix& data_col, Vector& image) {
    int hw_in = height_in * width_in;
    int hw_kernel = height_kernel * width_kernel;
    int hw_out = height_out * width_out;
    // col2im
    image.resize(hw_in * channel_in);
    image.setZero();
    for (int c = 0; c < channel_in; c ++) {
        for (int i = 0; i < hw_out; i ++) {
            int step_h = i / width_out;
            int step_w = i % width_out;
            int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
            for (int j = 0; j < hw_kernel; j ++) {
                int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
                int cur_row = start_idx / width_in + j / width_kernel - pad_h;
                if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
                    cur_row >= height_in) {
                    continue;
                }
                else {
                    //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
                    int pick_idx = cur_row * width_in + cur_col;
                    image(c * hw_in + pick_idx) += data_col(i, c * hw_kernel + j);  // pick which pixel
                }
            }
        }
    }
}
