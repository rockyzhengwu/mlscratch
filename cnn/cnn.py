#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import gzip

class ConvLayer(object):

    def __init__(self,
                 input_width,
                 input_height,
                 channel_num,
                 filter_width,
                 filter_height,
                 filter_num,
                 zero_padding,
                 stride,
                 activator,
                 learning_rate
                 ):

        self.input_width = input_width
        self.input_height = input_height
        self.channel_num = channel_num
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_num = filter_num
        self.zero_padding = zero_padding
        self.stride = stride
        self.activator = activator
        self.learning_rate = learning_rate

        self.output_height = ConvLayer.calc_output_size(input_height,
                                                        filter_height,
                                                        stride,
                                                        zero_padding
                                                        )
        self.output_width = ConvLayer.calc_output_size(input_width,
                                                       filter_width,
                                                       stride,
                                                       zero_padding
                                                       )
        self.output_array = np.zeros((filter_num, self.output_width, self.output_height))
        self.filters = [Filter(filter_width, filter_height, channel_num) for _ in range(filter_num)]

    @staticmethod
    def calc_output_size(input_size, filter_size, stride, zero_padding ):
        return int((input_size + 2 * zero_padding - filter_size ) / stride) + 1

    def forward(self, input_array):
        # todo 检查输入
        self.input_array = input_array
        self.padding_array = padding(input_array, self.zero_padding)
        for i in range(self.filter_num):
            f = self.filters[i]
            conv(self.padding_array, f.get_weights(), self.output_array[i], self.stride, f.get_bias(), self.activator.forward)


def get_patch(input_array, i, j,width, height, stride):
    min_width = i*stride
    max_width = i*stride + width
    max_height = j*stride + height
    min_height = j*stride
    return input_array[min_width: max_width, min_height: max_height]

def conv(input_array, fil_weights, output_array, stride, bias, active_func):
    d, filter_width, filter_height = fil_weights.shape
    m, n = output_array.shape
    for i in range(m):
        for j in range(n):
            output_array[i][j] =  np.sum(get_patch(input_array, i, j, filter_width, filter_height, stride) * fil_weights) + bias
            output_array[i][j] = active_func(output_array[i][j])

def padding(input_array, zero_padding):
    if zero_padding == 0:
        return input_array
    dzp = 2*zero_padding

    if input_array.ndim == 3:
        d, m,n = input_array.shape
        padd_array = np.zeros((d, m + dzp, n + dzp))
        padd_array[:, zero_padding: zero_padding+m , zero_padding: zero_padding+n] = input_array
        return padd_array

    elif input_array.ndim == 2:
        m, n = input_array.shape
        padd_array = np.zeros((m + dzp, n + dzp))
        padd_array[zero_padding: zero_padding+m, zero_padding: zero_padding+n] = input_array
        return padd_array

    else:
        raise(Exception("input array size error"))


class Filter(object):
    def __init__(self, filter_width, filter_height, depth):
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.depth = depth
        self.weights = np.zeros((depth , filter_width, filter_height))
        self.bias = 0

    def get_weights(self,):
        return self.weights

    def get_bias(self):
        return self.bias


class MaxPooling(object):
    pass


def load_data(file_path):
    gf = gzip.open(file_path, 'rb' )
    data = []
    labels = []
    for line in gf.readlines():
        line = str(line, 'utf-8')
        line = line.strip("\n").strip()
        if not line:
            continue

        line = line.split()
        line = np.array(list(map(float, line[:-1])))
        data.append(line.reshape(28, 28))
        la = [0.1]*10
        la[int(line[-1])] = 0.9
        labels.append(la)
    return np.array(data), np.array(labels)

class ReluActivator(object):

    def forward(self, weight_input):
        return max(0, weight_input)


conv_layer = ConvLayer(
    input_width=28,
    input_height=28,
    channel_num=1,
    filter_width=3,
    filter_height = 3,
    filter_num = 2,
    zero_padding = 0,
    stride = 1,
    activator = ReluActivator(),
    learning_rate = 0.01
)


#train_set, train_labels = load_data("../data/mnist_train.txt.gz")
test_set, test_labels = load_data("../data/mnist_test.txt.gz")
print(test_set[0].shape)
conv_layer.forward(test_set[0])
"""
todo 待实现反向部分，梯度的计算，
"""
