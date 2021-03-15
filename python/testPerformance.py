import time

import numpy as np
import torch

from mnist import Mnist
from mnist_cnn_pytorch import Net
from operatorDemo import *


mnist = Mnist("./data/mnist/")
n_epoch = 1

def test_pytorch():
    x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (mnist.tr_x, mnist.tr_y, mnist.te_x, mnist.te_y))
    model = Net()
    print(model)
    time_start = time.time()
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    for i in range(n_epoch):
        model.forward(x_valid)

    time_end = time.time()
    time_cost = time_end - time_start
    print("totally cost %.3f sec" %time_cost)

def test_numpy_serial():
    np.random.seed(1)
    test_x = mnist.te_x[:,].reshape(10000,28,28,1)
    time_start = time.time()

    w1 = np.random.randn(5, 5, 1, 6)
    b1 = np.random.randn(1, 1, 1, 6)

    A1, _ = conv_forward(test_x, w1, b1, {"pad":0,"stride":1})
    Z1, _ = pool_forward(relu(A1), {"stride":2, "f":2})
    print(Z1.shape)
    time_end = time.time()
    time_cost = time_end - time_start
    print("totally cost %.3f sec" %time_cost)


def test_conv_relu_pooling_forward():
    np.random.seed(1)
    test_x = mnist.te_x[:,].reshape(10000,28,28,1)
    time_start = time.time()

    w1 = np.random.randn(5, 5, 1, 6)
    b1 = np.random.randn(1, 1, 1, 6)

    A1 = conv_relu_pooling_forward(test_x, w1, b1, {"pad":0,"stride":1}, {"stride":2, "f":2})
    print(A1.shape)
    time_end = time.time()
    time_cost = time_end - time_start
    print("totally cost %.3f sec" %time_cost)


if __name__ == "__main__":
    test_pytorch()
    # test_numpy_serial()
    # test_conv_relu_pooling_forward()


