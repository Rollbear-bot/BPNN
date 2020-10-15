# -*- coding: utf-8 -*-
# @Time: 2020/10/15 21:58
# @Author: Rollbear
# @Filename: unit_test.py

from unittest import TestCase
from entity.network import *
import numpy as np
import unittest


w = np.zeros(shape=(7, 7))
w[1, 4] = 0.2
w[1, 5] = -0.3
w[2, 4] = 0.4
w[2, 5] = 0.1
w[3, 4] = -0.5
w[3, 5] = 0.2
w[4, 6] = -0.3
w[5, 6] = -0.2


class UnitTest(TestCase):
    def test_get_next_layer(self):
        cur_layer = [1, 2, 3]
        next_layer = get_next_layer(cur_layer, w)
        assert next_layer == {4, 5}
        assert get_next_layer(next_layer, w) == {6}

    def test_get_parent_layer(self):
        cur_layer = [6]
        parent_layer = get_parent_layer(cur_layer, w)
        assert parent_layer == {4, 5}
        assert get_parent_layer(parent_layer, w) == {1, 2, 3}


if __name__ == '__main__':
    unittest.main()
