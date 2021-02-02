#!/usr/bin/env python3

import numpy as np
from main_functions_classicNN import *

def get_n_H(n_H_prev, f, stride=1, padding='valid'):
    """Returns the height of the next layer after considering the height of the
    previous one, the size of the filter f, the pad applied to the input and the
    stride."""

    if padding == 'valid':
        pad = 0
    elif padding == 'same':
        pad = (f-1)//2
    else:
        raise Exception('The padding must be either valid or same')

    n_H = int((n_H_prev + 2*pad - f)/stride + 1)
    return n_H

def get_n_W(n_W_prev, f, stride=1, padding='valid'):
    """Returns the height of the next layer after considering the height of the
    previous one, the size of the filter f, the pad applied to the input and the
    stride."""

    if padding == 'valid':
        pad = 0
    elif padding == 'same':
        pad = (f-1)//2
    else:
        raise Exception('The padding must be either valid or same')

    n_W = int((n_W_prev + 2*pad - f)/stride + 1)
    return n_W



def basic_convolution(portion, filter):
     """It makes the basic operation of convolution in 2D in which
     it multiplies the matrixes and sums the whole result."""

     #First, we make sure that the dimensions fit.
     portion_dims = portion.shape
     filter_dims = filter.shape

     assert portion_dims == filter_dims, "The number dimensions of the portion \
     and filter must be the same and the same shape"

     conv_operation = (portion * filter).sum()
     return conv_operation

def linear_and_forward_activation(W, A_prev, b, activation_name):
    """Doing the linear activation and the forward activation in
    one step."""

    Z = linear_activation(W, A_prev, b)
    A = activation_functions(Z, activation_name)

    return Z, A
