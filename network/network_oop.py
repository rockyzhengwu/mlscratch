#!/usr/bin/env python
# encoding: utf-8

"""
面向对象的思路实现

参考实现
https://www.zybuluo.com/hanbingtao/note/476663

不足之处是，写死了目标函数和激活函数
"""

import random
import math
import gzip

def sigmoid(z):
    return 1.0 / (1 + math.exp(-z))


class Node(object):
    def __init__(self, layer_index, index):
        self.layer_index = layer_index
        self.index = index
        # 存放上下游链接
        self.up_conns= []
        self.down_conns= []
        self.z = 0.0
        self.output = 0.0
        self.delta = 0.0

    def add_down_conn(self, conn):
        self.down_conns.append(conn)

    def add_up_conn(self, conn):
        self.up_conns.append(conn)

    def set_output(self, output):
        self.output = output

    def calc_output(self):
        z = 0.0
        for up in self.up_conns:
            z += up.up_node.output*up.weight
        self.z = z
        self.output = sigmoid(z)

    def calc_output_delta(self, y):
        self.delta = (y - self.output)*self.output*(1.0-self.output)

    def calc_hidden_delta(self):
        delta = 0.0
        for conn in self.down_conns:
            delta += conn.weight * conn.down_node.delta
        self.delta = delta * self.output * (1-self.output)

class ConstNode(object):
    def __init__(self, layer_index, index):
        self.layer_index = 0
        self.up_conns= []
        self.down_conns = []
        self.output = 1

    def add_down_conn(self, conn):
        self.down_conns.append(conn)

    def add_up_conn(self, conn):
        self.up_conns.append(conn)

    def calc_hidden_delta(self ):
        delta = 0.0
        for conn in self.down_conns:
            delta += conn.weight * conn.down_node.delta
        self.delta = delta*self.output*(1-self.output)

class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []

        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()


class Connection(object):
    def __init__(self, up_node, down_node):
        self.up_node = up_node
        self.down_node = down_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        self.gradient = self.down_node.delta*self.up_node.output

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def get_gradient(self):
        return self.gradient

class Connections(object):
    def __init__(self):
        self.conns = []


class NetWork(object):
    def __init__(self, sizes):
        self.layers = []
        self.sizes = sizes
        self.layer_count = len(sizes)
        self.connections = []
        self.err = 0.0

        for i in range(self.layer_count):
            self.layers.append(Layer(i, sizes[i]))

        for i in range(self.layer_count-1):
            up_layer = self.layers[i]
            down_layer = self.layers[i+1]
            connections = []
            # create conn
            for up in up_layer.nodes :
                for down in down_layer.nodes[:-1] :
                    connections.append(Connection(up, down))

            # 建立node 和connection 的关系: connection 添加到其上游节点的下游集合， 同时添加到下游节点的上游集合,
            for conn in connections:
                conn.up_node.add_down_conn(conn)
                conn.down_node.add_up_conn(conn)
                self.connections.append(conn)

    def calc_error(self, label, output):
        err = 0.0
        for i in range(self.sizes[-1]):
            err += (label[i]-output[i])*(label[i]-output[i])
        err = 0.5 * err
        return err

    def reset_error(self):
        self.err = 0.0

    def train(self, data_set, labels, rate, iteration):
        sample_count = len(labels)
        for step in range(iteration):
            self.reset_error()
            tmp_count = 0
            for x, label in zip(data_set, labels):
                tmp_count += 1
                self.train_one_sample(x, label, rate)
                if tmp_count %1000==0:
                    print("smaple %d, error %f"%(tmp_count, self.err/tmp_count))
            print("step %d, error %f"%(step, self.err / sample_count))

    def train_one_sample(self, x, label, rate):
        o = self.forward(x)
        err = self.calc_error(o, label)
        self.err += err
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_gradient(self):
        for conn in self.connections:
            conn.calc_gradient()


    def calc_delta(self, label):
        output_nodes= self.layers[-1].nodes
        for i in range(self.sizes[-1]):
            output_nodes[i].calc_output_delta(label[i])

        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_delta()

    def update_weight(self, rate):
        for conn in self.connections:
            conn.update_weight(rate)

    def forward(self, x):
        self.layers[0].set_output(x)
        for i in range(1, self.layer_count):
            self.layers[i].calc_output()
        return [node.output for node in self.layers[-1].nodes[:-1]]

    def get_gradient(self, sample, label):
        self.forward(sample)
        self.calc_delta(label)
        self.calc_gradient()


def check_gradient(network, sample_feature, sample_label):
    def calc_error(vec1, vec2):
        err = sum([a-b for a, b in zip(vec1, vec2)])
        return err
    epsilon = 0.0001
    network.get_gradient(sample_feature, sample_label)
    for conn in network.connections:
        conn.weight += epsilon
        actual_gradient = conn.get_gradient()
        err1 = calc_error(network.forward(sample_feature), sample_label)
        conn.weight -= 2*epsilon
        err2 = calc_error(network.forward(sample_feature), sample_label)
        exp_gradient = (err1 - err2)/(2 * epsilon)
        print("exp_gradient : %f, actual gradient:%f"%(actual_gradient, exp_gradient))


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
    return data, labels



def get_maxindex(input_list):
    input_len= len(input_list)
    max_index = 0
    max_value = 0
    for i in range(input_len):
        if max_value <input_list[i]:
            max_value = input_list[i]
            max_index = i
    return max_index

def evalidate(network, test_data, test_labels):
    data_count = len(test_labels)
    a_count = 0
    for i in range(data_count):
        x = test_data[i]
        pre_output = network.forward(x)
        pre_label = get_maxindex(pre_output)
        t_label = get_maxindex(test_labels[i])
        if pre_label== t_label:
            a_count += 1
    print("acc: %f"%(1.0*a_count/data_count,))


test_data, test_labels = load_data("../data/mnist_test.txt.gz")
train_data, train_labels = load_data("../data/mnist_test.txt.gz")
net_work = NetWork([784, 300, 10])
#check_gradient(net_work,train_data[0], train_labels[0])
net_work.train(train_data, train_labels, 0.3, 1)
evalidate(net_work, test_data, test_labels)
