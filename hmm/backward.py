#!/usr/bin/env python
#-*-coding=utf-8-*-

"""
后向算法
"""

# 转移概率矩阵
A = [
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.3],
    [0.2, 0.3, 0.5]
]
# 状态到观测的概率
B = [
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
]

# 状态初始概率
pi = [0.2, 0.4, 0.4]

def backward(O=[]):
    T = len(O)
    N = len(pi)
    # t =  T
    beta_4 = [1] * N
    # t = 3
    beta_3 = [0] * N

    for i in range(N):
        for j in range(N):
            beta_3[i] += A[i][j] * beta_4[j] * B[j][O[3]]
    # t = 2
    beta_2 = [0] * N
    for i in  range(N):
        for j in range(N):
            beta_2[i] += A[i][j] * beta_3[j]*B[j][O[2]]

    beta_1 = [0] * N
    for i in range(N):
        for j in range(N):
            beta_1[i] += A[i][j] * beta_2[j]*B[j][O[1]]

    p = 0.0
    for i in range(N):
        p += pi[i] * B[i][O[0]] * beta_1[i]

    print (p)


if __name__ == '__main__':
    backward([0, 1, 0, 1])


