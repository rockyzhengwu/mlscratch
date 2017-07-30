#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import operator
import os
import matplotlib
import matplotlib.pyplot as plt


def load_data():
    x = [
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ]
    y = ["A", "A", "B", "B"]
    return x, y


def knn(in_x, x, y , topn):
    """
    x: np.array (sample, feature)
    y: label list of sample
    """
    dis_array = np.sqrt(np.sum((x - in_x) * (x - in_x), axis=1))
    dis_index = np.argsort(dis_array)
    label_count = {}
    for k in range(topn):
        label = y[dis_index[k]]
        label_count[label] = label_count.get(label, 0) + 1
    sorted_count = sorted(label_count.items(), key = operator.itemgetter(1), reverse=True)
    return sorted_count[0][0]


def load_data_file(file_path):
    if not os.path.exists(file_path):
        raise("file : %s isn't exist")
    f = open(file_path, 'r')
    x = []
    y = []
    for line in f.readlines():
        line = line.strip("\n")
        line = line.split("\t")
        x.append([float(a) for a in line[:-1]])
        y.append(line[-1])
    return x, y


def norm_data(data):
    """
    归一化data, data 是np.array 类型
    """
    min_values = data.min(0)
    max_values = data.max(0)
    norm_data = (data-min_values)/(max_values-min_values)
    return norm_data


def plot_data(x, y):
    label_unic = list(set(y))
    y = [label_unic.index(l) for l in y]
    x = np.array(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x[:,1],x[:, 2], 15*np.array(y),15*np.array(y) )
    plt.show()

def test_dateing():
    x, y = load_data_file("datingTestSet.txt")
    x = np.array(x)
    x = norm_data(x)
    train_count = int(len(x)*0.9)
    train_x = x[:train_count]
    train_y = y[:train_count]
    right_count = 0
    test_count = 0
    for i in range(train_count):
        test_count += 1
        in_x = x[i]
        predict_y = knn(in_x, train_x, train_y , 200)
        if (predict_y == y[i]):
            right_count += 1
        print("predict answer: %s real is %s" %(predict_y, y[i], ))
    print("acc: ", right_count / test_count)


def test():
    in_x = [1.0, 0.9 ]
    x, y = load_data()
    pre_y = knn(in_x, x, y , 1)
    print("input:", in_x)
    print("result : %s" % (pre_y))


if __name__ == "__main__":
    # test()
    test_dateing()
