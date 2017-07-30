#!/usr/bin/env python
# encoding: utf-8


import numpy as np

def load_data(file_path):
    data = []
    labels = []
    f = open(file_path, 'r')
    for line in f.readlines():
        line = line.strip("\n")
        if not line:
            continue
        line_list= line.split()
        data.append([float(a) for a in line_list[:-1]])
        labels.append(int(line[-1]))
    return data, labels

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def train(x, y, steps=100, rate=0.1):
    m, n = x.shape
    y= y.reshape(m, 1)
    w = np.ones((n, 1))
    b = np.ones(1)
    for s in range(steps):
        z = np.dot(x, w) + b
        h = sigmoid(z)
        error = (y - h)
        w = w + (rate * np.dot(x.transpose(), error))/n
        b = b + rate * np.sum(error, axis=0)/n
    return w, b


def predict(w, b, test_vec):
    z = np.dot(test_vec, w) + b
    h = sigmoid(z)
    if h > 0.5 :
        return 1
    else :
        return 0


data, labels = load_data("testSet.txt")
x = np.array(data)
y = np.array(labels)
w, b = train(x, y, 500, 0.001)
test_vec = np.array([0.9, 1.3])
pre = predict(w, b, test_vec)
print(pre)





