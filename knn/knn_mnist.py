#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import knn

def load_data(file_path):
    X = []
    Y = []
    for file_name in os.listdir(file_path):
        label = file_name.split("_")[0]
        Y.append(int(label))
        f = open(os.path.join(file_path, file_name))
        lines = f.readlines()
        lines = [l.strip("\n") for l in lines]
        data_str = "".join(lines)
        img = [float(a) for a in data_str]
        img = np.array(img)
        X.append(img)
    return X, Y

def test_mnist():
    train_x, train_y = load_data("digits/trainingDigits")
    test_x, test_y = load_data("digits/testDigits")
    pre_list = []
    for x , y in zip(test_x , test_y):
        pre = knn.knn(x, train_x, train_y, 1)
        pre_list.append(pre)
        print("predict : %d real result : %d"%(pre, y,))
    valu_result(pre_list, test_y)

def valu_result(pre_y, real_y):
    """
    效果评估
    """
    pre_count = {}
    all_count = {}
    for pre, real in zip(pre_y, real_y):
        if real not in pre_count:
            pre_count[real] = {}
        all_count[real] = all_count.get(real, 0) + 1
        if pre == real:
            pre_count[real]["t"] =  pre_count[real].get("t", 0)+ 1
        else:
            pre_count[real]['f'] = pre_count[real].get("f", 0) + 1
    all_true = 0.0
    for k, v in all_count.items():
        all_true += pre_count[k].get("t")
        print("acc: %s, %f "%(k, pre_count[k]["t"]/v,))
    print("all acc : %f"%(all_true/ sum(all_count.values())))

test_mnist()
"""
最后的结果是k==1 反而是最好高的准确率？很奇怪
"""
