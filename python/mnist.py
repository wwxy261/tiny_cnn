import os
import struct

import numpy as np

class Mnist:
    def __init__(self, file_dir):
        """ initialization
        Keyword Arguments:
        file_dir -- (str) the file path of mnist data
        """
        names = [
            "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte"
        ]
        # train image
        file = os.path.join(file_dir, names[0])
        with open(file, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            tr_x = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            self.tr_x = tr_x.reshape((size, -1))
            self.tr_size = size

        # train label
        file = os.path.join(file_dir, names[1])
        with open(file, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            self.tr_y = np.fromfile(f,
                                    dtype=np.dtype(np.uint8).newbyteorder('>'))
        # test image
        file = os.path.join(file_dir, names[2])
        with open(file, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            te_x = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            self.te_x = te_x.reshape((size, -1))
            self.te_size = size
        # test label
        file = os.path.join(file_dir, names[3])
        with open(file, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            self.te_y = np.fromfile(f,
                                    dtype=np.dtype(np.uint8).newbyteorder('>'))

    def next_train_batch(self, size, one_hot=False):
        """ return next train data with size size
        Keyword Arguments:
        size -- (int) the size of the batch
        return: (tuple) tuple of (tr_x, tr_y) of size `size'
        """
        index = np.random.choice(self.tr_size, size, False)
        tr_x = self.tr_x[index]
        tr_y = self.tr_y[index] if not one_hot else np.eye(
            10, None, dtype=np.uint8)[self.tr_y[index]]
        return tr_x, tr_y

    def next_test_batch(self, size, one_hot=False):
        """ return next test data with size size
        Keyword Arguments:
        size -- (int) the size of the batch
        return: (tuple) tuple of (te_x, te_y) of size `size'
        """
        index = np.random.choice(self.te_size, size, False)
        te_x = self.te_x[index]
        te_y = self.te_y[index] if not one_hot else np.eye(
            10, None, dtype=np.uint8)[self.te_y[index]]
        return te_x, te_y