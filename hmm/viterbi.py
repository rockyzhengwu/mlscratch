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
    print(delta_1)
    # t = 2
    delta_2 = [0] * N
    V[1] = [0] * N
    for i in range(N):
        t = [delta_1[j] * A[j][i] for j in range(N)]
        max_v = max(t)
        max_index = t.index(max_v)
        V[1][i] = max_index
        delta_2[i] = max_v*B[i][O[1]]
    print(delta_2)
    # t = 3
    delta_3 = [0] * N
    V[2] = [0] * N
    for i in range(N):
        t = [delta_2[j] * A[j][i]for j in range(N)]
        max_v = max(t)
        max_index = t.index(max_v)
        V[2][i] = max_index
        delta_3[i] = max_v * B[i][O[2]]
    print(delta_3)

    # t = 4
    # delta_4 = [0]*N
    # V[3] = [0]*N
    # for i in range(N):
    #     t = [delta_3[j]*A[j][i]for j in range(N)]
    #     max_v = max(t)
    #     max_index = t.index(max_v)
    #     V[3][i] = max_index
    #     delta_4[i] = max_v*B[i][O[3]]

    path = [0]*T
    path[T-1] = delta_3.index(max(delta_3))
    print(V)
    for i in range(T-2, -1, -1 ):
        print (i)
        path[i] = V[i+1][path[i+1]]
    print(path)

if __name__ == '__main__':
    viterbi([0, 1 , 0])
