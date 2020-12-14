#!/usr/bin/env python3

import numpy as np

def basic_convolution(portion, filter):
     """It makes the basic operation of convolution in 2D in which
     it multiplies the matrixes and sums the whole result."""

     portion_dims = portion.shape
     filter_dims = filter.shape

     assert portion_dims == filter_dims, "The number dimensions of the portion \
     and filter must be the same and the same shape"


     conv_operation = (portion * filter).sum()
     return conv_operation
