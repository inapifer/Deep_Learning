#!/usr/bin/env python3

from main_functions import *
import unittest
import numpy as np

class Test_linear_activation(unittest.TestCase):

    def test_basic(self):
        W_test, A_prev_test, b_test = np.ones([2,3]), np.ones([3,10]),\
                                      np.zeros([2,1])
        expected = np.matmul(W_test, A_prev_test) + b_test
        self.assertEqual(linear_activation(W_test, A_prev_test,
         b_test), expected)

unittest.main()
