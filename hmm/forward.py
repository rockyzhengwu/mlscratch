#!/usr/bin/env python
#-*-coding=utf-8-*-

"""
前向算法
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


def forward(O=[]):
    N = len(pi) # 状态集合 大小

    # t = 1
    alpha_1 = [pi[i] * B[i][O[0]] for i in range(N)]
    # t = 2
    alpha_2 = [0] * N
    for j in range(N):
        for i in range(N):
            alpha_2[j] += alpha_1[i] * A[i][j] * B[j][O[1]]
    # t=3
    alpha_3 = [0] * N
    for j in range(N):
        for i in range(N):
            alpha_3[j] += alpha_2[i] * A[i][j] * B[j][O[2]]
    # t=4
    alpha_4 = [0] * N
    for j in range(N):
        for i in range(N):
            alpha_4[j] += alpha_3[i] * A[i][j] * B[j][O[3]]

    print (sum(alpha_4))

if __name__ == '__main__':
    forward([0, 1, 0, 1])

