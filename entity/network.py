# -*- coding: utf-8 -*-
# @Time: 2020/10/15 20:23
# @Author: Rollbear
# @Filename: network.py

import numpy as np


# 自然对数的底
e = 2.71828182


def sigmoid(x):
    """计算sigmoid函数"""
    return (1 + e ** (-x)) ** -1


def get_next_layer(cur_layer, w):
    next_layer = set()
    for node in cur_layer:
        for index, desc in enumerate(w[node]):
            if desc != 0:
                next_layer.add(index)
    return next_layer


def get_parent_layer(cur_layer, w):
    parent_layer = set()
    for node in cur_layer:
        for index, parent in enumerate(w[:, node]):
            if parent != 0:
                parent_layer.add(index)
    return parent_layer


class Network:
    def __init__(self, learning_rate, max_iter, error_threshold,
                 w, theta, id_of_input_nodes, id_of_output_nodes):
        """
        init the BPNN
        :param learning_rate: 学习率
        :param max_iter: 单个样本的最大迭代次数
        :param error_threshold: 网络误差阈值
        :param w: 初始化网络连接权重
        :param theta: 初始化神经元偏置
        :param id_of_input_nodes: 输入层的节点id
        :param id_of_output_nodes: 输出层的节点id
        """
        self.w = w
        self.theta = theta
        self.id_of_input_nodes = id_of_input_nodes
        self.id_of_output_nodes = id_of_output_nodes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.error_threshold = error_threshold

    def fit(self, x, y, debug=False):
        """
        训练BP神经网络模型
        :param x: 样本序列（有序）
        :param y: 预测目标序列（有序）
        :param debug: 在控制台输出训练过程信息
        :return: None
        """
        o = np.zeros(shape=len(self.w))
        for sample_index, sample in enumerate(x):
            if debug:
                print("-------------------------------")
            num_iter = 0
            while num_iter < self.max_iter:
                num_iter += 1
                if debug:
                    print(f"样本{sample}，迭代次数{num_iter}")
                    print("向前传播")

                # 样本传入到输入层
                cur_layer = self.id_of_input_nodes

                for index, value in enumerate(sample):
                    if debug:
                        print(f"O[{self.id_of_input_nodes[index]}] = {value}")
                    o[self.id_of_input_nodes[index]] = value

                while len(cur_layer) != 0:
                    # 获取下一层的节点
                    cur_layer = get_next_layer(cur_layer, self.w)

                    # 样本前向传播
                    for node in cur_layer:
                        s = 0
                        # 取列（父节点）
                        for p_index, parent in enumerate(self.w[:, node]):
                            if parent != 0 and o[p_index] != 0:
                                s += parent * o[p_index]
                        s += self.theta[node]
                        o[node] = sigmoid(s)
                        if debug:
                            print(f"O[{node}] = {sigmoid(s)}")

                # 计算输出层误差
                error = np.zeros(shape=len(self.w))
                for index, value in enumerate(y[sample_index]):
                    error[self.id_of_output_nodes[index]] = value - o[self.id_of_output_nodes[index]]
                if debug:
                    print(f"输出层误差{list(error)}")

                # 网络总误差
                total_error = sum([(item**2)/2 for item in error if item != 0])
                if debug:
                    print(f"网络总误差{total_error}")

                # 误差反向传播
                if debug:
                    print("误差反向传播")
                cur_layer = self.id_of_output_nodes
                while set(cur_layer) != set(self.id_of_input_nodes):
                    cur_layer = get_parent_layer(cur_layer, self.w)
                    for node in cur_layer:
                        s = 0
                        for d_index, desc in enumerate(self.w[node]):
                            if desc != 0:
                                s += desc * error[d_index]
                        # 计算每个节点的误差
                        error[node] = o[node] * (1 - o[node]) * s
                        if debug:
                            print(f"节点{node}误差 = {o[node]} * {1 - o[node]} * {s} = {o[node] * (1 - o[node]) * s}")

                # 权重与偏置调整
                if debug:
                    print("权重与偏置调整")
                for i in range(1, len(self.w)):
                    for j in range(1, len(self.w)):
                        if self.w[i, j] != 0:
                            if debug:
                                print(f"w[{i}, {j}] = {self.w[i, j]} + {o[i] * error[j] * self.learning_rate} = "
                                      f"{self.w[i, j] + o[i] * error[j] * self.learning_rate}")
                            self.w[i, j] += o[i] * error[j] * self.learning_rate
                for j in range(1, len(self.w)):
                    if self.theta[j] != 0:
                        if debug:
                            print(f"θ[{j}] = {self.theta[j]} + {self.learning_rate * error[j]} "
                                  f"= {self.theta[j] +self.learning_rate * error[j]}")
                        self.theta[j] += self.learning_rate * error[j]

                if total_error < self.error_threshold:
                    if debug:
                        print(f"网络总误差{total_error}小于阈值{self.error_threshold}，结束样本{sample}的迭代\n")
                    break  # 跳出循环，选择下一个样本

                if debug:
                    print("\n")

    def predict(self, x):
        """
        使用已拟合的BP神经网络模型预测
        :param x: 样本序列
        :return: 预测的值序列
        """
        y = []
        for sample in x:
            o = np.zeros(shape=len(self.w))
            for index, value in enumerate(sample):
                o[self.id_of_input_nodes[index]] = value

            cur_layer = self.id_of_input_nodes
            while len(cur_layer) != 0:
                cur_layer = get_next_layer(cur_layer, self.w)
                for node in cur_layer:
                    s = 0
                    # 取列（父节点）
                    for p_index, parent in enumerate(self.w[:, node]):
                        if parent != 0 and o[p_index] != 0:
                            s += parent * o[p_index]
                    s += self.theta[node]
                    o[node] = sigmoid(s)
            y.append([o[output_node] for output_node in self.id_of_output_nodes])
        return y
