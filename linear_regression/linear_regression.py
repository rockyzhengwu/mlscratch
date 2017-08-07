#!/usr/bin/env python
# encoding: utf-8


import random
import matplotlib.pyplot as plt

def load_data():
    x = [[5], [3], [8], [1.4], [10.1]]
    y = [5500, 2300, 7600, 1800, 11400]
    return x, y

class LinearRegression(object):
    def __init__(self):
        self.weights = None
        self.bias = None
        pass
    def _init_params(self, feature_num):
        if not self.weights:
            self.weights = [random.random() for _ in range(feature_num)]
            self.bias = random.random()

    def _iter_once(self, x, y, rate):
        total_err = 0.0
        for vec , label in zip(x, y):
            h = self.predict(vec)
            err = label - h
            # 更新权重
            self.weights = [w + rate*err*a for w,a in zip(self.weights, vec)]
            self.bias = self.bias + err*rate
            total_err += abs(err)
        return total_err / len(x)


    def train(self, x, y, steps, rate, plot_error=False):
        self._init_params(len(x[0]))
        errors = []
        for s in range(steps):
            err = self._iter_once(x, y, rate)
            errors.append(err)
            print("step: %d , err: %f"%(s, err))

        if plot_error:
            self.plot_errors(errors)

    def plot_errors(self, errors):
        plt.plot(errors)
        plt.show()


    def predict(self, vec_x):
        h = sum([ w*v for w,v in zip(self.weights, vec_x) ]) + self.bias
        return h


if __name__ == "__main__":
    x, y = load_data()
    li = LinearRegression()
    li.train(x, y, 300, 0.0001)
    print(li.predict([3.4]))
    print(li.predict([15]))
    print(li.predict([1.5]))
    print(li.predict([6.3]))


