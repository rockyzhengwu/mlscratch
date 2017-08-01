#!/usr/bin/env python
#-*-coding=utf-8-*-

"""
viterbi 算法演算
"""

A = [
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
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


def viterbi(O=[]):
    T = len(O)
    N = len(pi)
    # t = 1
    V = [0] * T
    delta_1 = [pi[i] * B[i][O[0]] for i in range(N)]
    V[0] = [0] * N
    delta_all = []
    delta_all.append(delta_1)
    for t in range(1, T, 1):
        delta_2 = [0] * N
        V[t] = [0] * N
        for i in range(N):
            f = [delta_all[-1][j] * A[j][i] for j in range(N)]
            max_v = max(f)
            max_index = f.index(max_v)
            V[t][i] = max_index
            delta_2[i] = max_v * B[i][O[t]]
        delta_all.append(delta_2)
        print(delta_2)

    path = [0]*T
    delta_3 = delta_all[-1]
    path[T-1] = delta_3.index(max(delta_3))
    print(V)
    for i in range(T-2, -1, -1 ):
        print (i)
        path[i] = V[i+1][path[i+1]]
    print(path)

if __name__ == '__main__':
    viterbi([0, 1 , 0])
