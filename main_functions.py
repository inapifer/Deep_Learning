#!/usr/bin/env python3

import numpy as np

def linear_activation(W, A_prev, b):

    """
    Receives as parameters the weights of the network W,
    the activation function X and the bias.
    """
    Z = np.matmul(W, A_prev) + b
    return Z

def relu(Z):
    """A function that returns max(z,0) of a number"""
    return np.maximum(0,Z)

def sigmoid(Z):
    """returns the value of a sigmoid function"""
    return (1/(1 + np.exp(Z)))


def activation_functions(Z, activation_name):
    """Doing the activation function of relu and sigmoid"""
    if activation_name.lower() == 'relu':
        return relu(Z)
    if activation_name.lower() == 'sigmoid':
        return sigmoid(Z)

def forward_activation(W, A_prev, b, activation_name):
    """Doing the linear activation and the forward activation in
    one step."""
    Z = linear_activation(W, A_prev, b)
    A = activation_functions(Z, activation_name)

    return A
