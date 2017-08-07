#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import math
import operator

def load_data_set():
    posting_list=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0,1,0,1,0,1]
    return posting_list,class_vec

def create_vocab(data_set):
    vocab = set([])
    for doc in data_set:
        vocab = vocab | set(doc)
    return list(vocab)

def doc_to_vec(doc, vocab):
    data_vec = [0]*len(vocab)
    for word in vocab :
        if word in doc:
            data_vec[vocab.index(word)] = 1
    return data_vec

def train(x, y):
    """
    """
    ## 统计每个词在每个类中出现的频数，
    ## 统计每个类出现的频数
    vocab = create_vocab(x)
    x_vec = [doc_to_vec(d, vocab) for d in x]
    label_set = list(set(y))
    sample_len = len(y)
    prob_dict= {}
    class_count = {}
    print("vocab size:%d "%(len(vocab)) )
    for label in label_set:
        sample_index = [i if y[i]==label else -1 for i in range(sample_len)]

        sub_set = [x_vec[j] for j in sample_index if j>=0]
        class_count[label] = len(sub_set)
        count_list = np.sum(sub_set, axis=0)
        prob_dict[label] = count_list

    for k, v in prob_dict.items():
        prob_dict[k] = (v/np.sum(v)).tolist()

    for k, v in class_count.items():
        class_count[k] = v / sample_len
    return class_count, prob_dict, vocab


def predict(class_prob, word_prob, vocab, doc):
    doc_vec = doc_to_vec(doc, vocab)
    result_prob = {}
    vocab_len = len(vocab)
    for label, prob_list in word_prob.items():
        result_prob[label]= 0.0
        for i in range(vocab_len):
            if doc_vec[i] > 0 and prob_list[i] > 0 :
                result_prob[label] -= math.log(prob_list[i], 2)
    sorted_prob = sorted(result_prob.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_prob[0][0]

def test():
    x, y = load_data_set()
    class_prob, word_prob, vocab = train(x, y)
    print(word_prob)
    doc = ["love", "my", "dalmation"]
    doc_2 = ["stupid", "garbage"]
    pre = predict(class_prob, word_prob, vocab, doc)
    print("doc:", pre)
    pre_2 = predict(class_prob, word_prob, vocab,doc_2)
    print("doc_w", pre_2)

if __name__ == "__main__":
    test()

"""
实现考虑了多类别问题，
这个方法里面没有考虑词的多次出现，如果一个词出现多次也按照一次计算
没有考虑平滑，如果某个词在某个类别中的概率为0 会背直接忽略
"""
# todo 平滑 处理
