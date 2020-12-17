#!/usr/bin/env python3

import numpy as np

def get_n_H(n_H_prev, f, pad, stride):
    """Returns the height of the next layer after considering the height of the
    previous one, the size of the filter f, the pad applied to the input and the
    stride."""

    n_H = int((n_H_prev + 2*pad - f)/stride + 1)
    return n_H

def get_n_W(n_W_prev, f, pad, stride):
    """Returns the height of the next layer after considering the height of the
    previous one, the size of the filter f, the pad applied to the input and the
    stride."""
    
    n_W = nt((n_W_prev + 2*pad - f)/stride + 1)
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
