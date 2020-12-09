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
    elif activation_name.lower() == 'sigmoid':
        return sigmoid(Z)
    else:
        raise Exception("The only activation functions available\
                         are relu and sigmoid")


def derivative_activation(Z, activation_name):
    """Doing the derivative of the activation functions."""
    if activation_name.lower() == 'relu':
        return derivative_relu(Z)
    elif activation_name.lower() == 'sigmoid':
        return derivative_sigmoid(Z)
    else:
        raise Exception("The only activation functions\
                         available are relu ans sigmoid")

def forward_activation(W, A_prev, b, activation_name):
    """Doing the linear activation and the forward activation in
    one step."""
    Z = linear_activation(W, A_prev, b)
    A = activation_functions(Z, activation_name)

    return A


# Once the values have gone throught the network forward, I calculate
# the loss function.
# I will start the network just considering a binary classification problem.
def binary_class_loss_function(Y_hat, Y):
    # I make sure first that the dimensions of n(Number of features) and m(number of samples) are the same
    # in the results and the labels
    n, m = np.shape(Y)
    n_hat, m_hat = np.shape(Y_hat)
    assert(n==n_hat and m==m_hat)

    L = -(Y*(np.log(Y_hat)) + (1-Y)*(np.log(1-Y_hat)))

    return L

# Now I develop the cost function which will consider the loss functions

def cost_function(Y_hat, Y, name_loss='binary_class'):
    """This function calculates the cost function of all the samples
    given its name_loss. By default I will say it's binary classification."""

    available_names = ['binary_class']
    #I consider the name of the loss function.
    if name_loss.lower() == 'binary_class':
        L = binary_class_loss_function(Y_hat, Y)
        n, m = np.shape(L)
        C = 1/m*np.sum(L)
        return C
    else:
        raise Exception("The only available functions are: " \
        + str(available_names))


def derivative_cost_logistic(Y_hat, Y):
    """It returns the value of the derivatives of the cost function
    respect to Y_hat, which is the same as A[L], i.e., the activation
    of the last layer considering we use the logistic regression
    function."""

    assert Y_hat.shape == Y.shape, "Dimensions of Y and Y_hat must be the same"

    dA_L = -Y/Y_hat + (1-Y)/(1-Y_hat)

    return dA_L

def derivatives_Z(dA, Z, activation_name):
    """It returns the value of the derivative of the cost function
    with respect to Z of certain layer."""
    assert dA.shape == Z.shape, "Dimensions of dA and Z must be the same"
    # I calculate the derivative of the activation function in this
    #layer
    d_activation = derivative_activation(Z, activation_name)
    dZ = dA*d_activation
    # Now I make the (non-matricial) product of the derivatives
    # of the activation and dA.
    return dZ

def derivatives_W(dZ, A_prev):
    """Returns the derivatives of the weights in the layer l"""
    assert dZ.shape[1] == A_prev.shape[1], \
    "Number of samples must be the same in dZ and A_prev"
    m = dZ.shape[1]
    dW = 1/m * np.matmul(dZ, A_prev.T)
    return dW

def derivatives_b(dZ):
    """Returns the derivatives of the biases in certain layers."""
    m = dZ.shape[1]
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    return db

def derivatives_dA_prev(W, dZ):
    """Returns the derivatives of the activations of the
    before this one."""
    dA_prev = np.matmul(W.T, dZ)
    return dA_prev

def backward_iteration(dA, Z, activation_name, A_prev, W):
    """It does an iteration of the backpropagation algorithm in certain
    layer. It get dZ, dW, db and dA_prev."""
    dZ = derivatives_Z(dA,Z, activation_name)
    dW = derivatives_W(dZ, A_prev)
    db = derivatives_b(dZ)
    dA_prev = derivatives_dA_prev(W, dZ)

    #Now it is necessary to return the four values, dW and db because with them
    #we can update the values of the weights and biases of the network to improve
    # performance, and dZ and dA_prev because we have to use them in the layer before

    return dZ, dW, db, dA_prev

def reverse_activation_names(activation_names):
    """This function takes the activation names of the network, reverse
    their order and returns a list with the names and the number of the layers
    in tuples."""
    reversed_activation_names = list(reversed(list(enumerate(activation_names))))
    return reversed_activation_names
