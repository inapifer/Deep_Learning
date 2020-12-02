#!/usr/bin/env python3

import numpy as np

def linear_activation(A, X, b):
    return np.matmul(A,X) + b
