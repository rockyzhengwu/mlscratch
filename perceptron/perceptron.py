#!/usr/bin/env python
# encoding: utf-8


import random
import matplotlib.pyplot as plt


def load_data():
    x = [[1, 1], [0, 0], [1, 0], [0, 1]]
    y = [1, 0, 0, 0]
    return x, y

class Perceptron(object):
    def __init__(self, input_nums):
        self.weights = [random.random() for _ in range(input_nums)]
        self.bias = random.random()

        # 用0 初始化
        self.weights = [0.0 for _ in range(input_nums)]
        self.bias = 0.0

    def train(self, x, y , epochs, rate):
        for step in range(epochs):
            right_nums = self._itera_one(x, y, rate)
            acc = right_nums/len(x)
            print("step: %d, acc : %f"%(step, acc,))
            if acc == 1:
                return

    def activator(self, z):
        return 1 if z>0 else 0

    def _itera_one(self, x, y, rate):
        right_nums = 0
        for vec , label in zip(x, y):
            z = sum([a * w for (a, w) in zip(vec, self.weights) ]) + self.bias
            h = self.activator(z)
            if h == label:
                right_nums += 1
                continue
            err = label - h
            delta = rate * err
            self.weights = [w + delta*v for w, v in zip(self.weights, vec) ]
            self.bias = self.bias + err * rate
        return right_nums

x, y = load_data()
perceptron = Perceptron(len(x[0]))
perceptron.train(x, y, 100, 0.01)
plt.scatter([a[0] for a in x], [a[1] for a in x])
plt.plot([0, -perceptron.bias/perceptron.weights[1]], [-perceptron.bias/perceptron.weights[0], 0])
plt.show()
