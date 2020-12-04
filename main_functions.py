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

def derivative_relu(Z):
    """A function that returns the derivative of the relu function"""
    Drelu = Z.copy()
    Drelu[Drelu>0] = 1
    Drelu[Drelu<0] = 0

    return Drelu

def sigmoid(Z):
    """returns the value of a sigmoid function in a matrix"""
    return (1/(1 + np.exp(Z)))

def derivative_sigmoid(Z):
    """Returns the derivatives of the sigmoid function in a matrix
    (Scalars)"""
    return sigmoid(Z)*(1-sigmoid(Z))

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
