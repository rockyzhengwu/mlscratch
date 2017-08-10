#!/usr/bin/env python
# encoding: utf-8

"""
net work unmpy 实现
"""

import numpy as np
import gzip


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(a):
    return a * (1-a)

class Layer(object):
    def __init__(self, layer_index, size):
        self.layer_index = layer_index
        self.size = size
        self.output = np.zeros((size,1))
        self.delta = np.zeros(size)

    def set_output(self, output):
        self.output = output
        self.output.reshape((output.shape[0], 1))

    def calc_output_delta(self, y):
        self.delta = sigmoid_prime(self.output)*(y-self.output)



class Connection(object):
    def __init__(self, up_layer, down_layer):
        self.up_layer = up_layer
        self.down_layer = down_layer
        self.weights = np.zeros((up_layer.size, down_layer.size))
        self.bias = np.random.random(down_layer.size)

    def calc_down_output(self):
        z = np.dot(self.up_layer.output,self.weights) + self.bias
        self.down_layer.output = sigmoid(z)


    def calc_up_delta(self):
        self.up_layer.delta = np.dot(self.weights, self.down_layer.delta)*sigmoid_prime(self.up_layer.output)

    def update_weights(self, rate):
        up_output = self.up_layer.output.reshape((self.up_layer.output.shape[0],1))
        down_delta = self.down_layer.delta.reshape((self.down_layer.delta.shape[0],1)).transpose()
        d = rate* np.dot(up_output, down_delta)
        self.weights += d
        self.bias += rate* self.down_layer.delta

class NetWork(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.layer_size = len(sizes)
        self.layers = []
        self.connections = []
        self.error = 0.0

        for i in range(self.layer_size):
            self.layers.append(Layer(i, sizes[i]))

        for i in range(self.layer_size-1):
            conn = Connection(self.layers[i], self.layers[i+1])
            self.connections.append(conn)

    def calc_error(self, y):
        err = np.sum((y-self.layers[-1].output)*(y-self.layers[-1].output))
        self.total_error += err
        return err


    def train(self, data_set, labels, rate, iteration):
        for step in range(iteration):
            self.total_error = 0.0
            total_count = 0
            for x, y in zip(data_set, labels):
                self.train_one(x, y, rate)
                total_count +=1
                if total_count % 10000 ==0:
                    print("count: %d err: %f"%(total_count, self.total_error/total_count))

    def train_one(self, x,  y, rate):
        self.forward(x)
        self.calc_delta(y)
        self.update_weights(rate)
        err = self.calc_error(y)


    def update_weights(self, rate):
        for conn in self.connections:
            conn.update_weights(rate)

    def calc_delta(self, y):
        self.layers[-1].calc_output_delta(y)

        for i in range(self.layer_size-1)[::-1]:
            conn = self.connections[i]
            conn.calc_up_delta()


    def forward(self, x):
        self.layers[0].set_output(x)
        for conn in self.connections:
            conn.calc_down_output()
        return self.layers[-1].output


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
        data.append(list(map(float, line[:-1])))
        la = [0.1]*10
        la[int(line[-1])] = 0.9
        labels.append(la)
    return np.array(data), np.array(labels)

def evalidation(network, test_set, test_labels):
    def is_right(vec_1, vec_2):
        if np.argmax(vec_1) == np.argmax(vec_2):
            return True
        return False
    right_count = 0
    for x, y in zip(test_set, test_labels):
        pre_y = network.forward(x)
        if is_right(pre_y, y):
            right_count += 1
    print("total acc on test data: %f"%(right_count/len(test_labels)))


train_set, train_labels = load_data("../data/mnist_train.txt.gz")
test_set, test_labels = load_data("../data/mnist_test.txt.gz")

network = NetWork([784, 100, 10])
network.train(train_set, train_labels, 0.3, 1)
evalidation(network, test_set, test_labels)

