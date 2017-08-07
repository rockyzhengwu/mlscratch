#!/usr/bin/env python
# encoding: utf-8

from collections import Counter
import math
import copy

def load_data():
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calc_ent(data_set):
     total_count = len(data_set)
     label_list = [l[-1] for l in data_set]
     label_count = Counter(label_list)
     ent  = 0.0
     for k, v in label_count.items():
         p = float(v) / total_count
         ent -= p*math.log(p, 2)
     return ent


def split_data_set(data, column, value):
    result = []
    for vec in data:
        if vec[column] == value:
            tmp_vec = vec[: column]
            tmp_vec.extend(vec[column+1: ])
            result.append(tmp_vec)
    return result

def choose_feature(data):
    features = len(data[0]) - 1
    total_samples = len(data)
    base_ent = calc_ent(data)
    best_gain = 0.0
    best_feature = -1
    for i in range(features):
        values = list(set([s[i] for s in data]))
        new_ent = 0.0
        for v in values:
            sub_data = split_data_set(data, i, v)
            prob = len(sub_data)/ total_samples
            new_ent += prob * calc_ent(sub_data)
        info_gain = base_ent - new_ent
        if info_gain > best_gain:
            best_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    counter = Counter(class_list)
    sorted_count = sorted(counter.items(), key = operator.itemgeter(1), reverse = True)
    return sorted_count[0][0]



def create_tree(data, train_labels):
    labels = copy.deepcopy(train_labels)
    class_list = [s[-1] for s in data]
    if class_list.count(class_list[0]) == len(data):
        return class_list[0]
    if len(data[0]) == 1:
        return majority_cnt(class_list)
    best_feature = choose_feature(data)
    best_label= labels.pop(best_feature)
    tree = {best_label:{}}
    values = list(set([s[best_feature] for s in data]))
    for v in values:
        sub_data= split_data_set(data, best_feature, v)
        tree[best_label][v] = create_tree(sub_data, labels)
    return tree


def classify(tree, labels, test_vec):
    feature = list(tree.keys())[0]
    feature_index = labels.index(feature)
    value = test_vec[feature_index]
    sub_dict = tree[feature]
    for key in sub_dict.keys():
        if value == key:
            if(isinstance(sub_dict[key],dict)):
                class_label = classify(sub_dict[key], labels, test_vec)
            else:
                class_label = sub_dict[key]
    return class_label


if __name__ == "__main__":
    data, labels = load_data()
    tree = create_tree(data, labels)
    print(tree)
    res = classify(tree, labels, [1, 0])
    res = classify(tree, labels, [1, 1])
    print(res)

"""
这段代码说清楚了决策树的大致思想，还有一些变种和过拟合问题
"""

