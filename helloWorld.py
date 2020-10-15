# -*- coding: utf-8 -*-
# @Time: 2020/10/15 20:24
# @Author: Rollbear
# @Filename: helloWorld.py

import numpy as np
from entity.network import Network


def main():
    # 连接权值初始化
    w = np.zeros(shape=(7, 7))
    w[1, 4] = 0.2
    w[1, 5] = -0.3
    w[2, 4] = 0.4
    w[2, 5] = 0.1
    w[3, 4] = -0.5
    w[3, 5] = 0.2
    w[4, 6] = -0.3
    w[5, 6] = -0.2

    # 设置神经元偏置
    theta = np.zeros(shape=7)
    theta[4] = -0.4
    theta[5] = 0.2
    theta[6] = 0.1

    samples = [(1, 0, 1)]
    target = [[1]]

    # 使用连接权重和偏置初始化bp网络
    bpnn = Network(w=w, theta=theta, learning_rate=0.9, max_iter=9,
                   id_of_input_nodes=(1, 2, 3), id_of_output_nodes=tuple([6]),
                   error_threshold=0.1)
    bpnn.fit(samples, target, debug=True)


if __name__ == '__main__':
    main()
