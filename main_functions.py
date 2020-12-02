#!/usr/bin/env python3

import numpy as np

def linear_activation(W, A, b):
    
    """
    Receives as parameters the weights of the network W,
    the activation function X and the bias.
    """

    return np.matmul(W, A) + b
