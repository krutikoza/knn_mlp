# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    return np.sqrt(np.sum(np.square(x1-x2)))
    # raise NotImplementedError('This function must be implemented by the student.')


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    return sum(abs(val1-val2) for val1, val2 in zip(x1,x2))
    # raise NotImplementedError('This function must be implemented by the student.')


def identity(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)

    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    
    
    if derivative == False:
        return x
    else:
        return np.ones([len(x),len(x[0])])

    # raise NotImplementedError('This function must be implemented by the student.')


def sigmoid(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)

    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    s = 1/(1+np.exp(-x))
    
    if derivative == False:
        return s
    else:
        return s*(1-s)

    # raise NotImplementedError('This function must be implemented by the student.')


def tanh(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)

    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    s = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    if derivative == False:
        return s
    else:
        
        return 1-(s*s)

    # raise NotImplementedError('This function must be implemented by the student.')


def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    

    x = np.clip(x, -1e100, 1e100)
    

    if derivative == False:
        return np.maximum(x,0)
    else:
        return np.where(x <= 0, 0, 1) #

    # raise NotImplementedError('This function must be implemented by the student.')


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """
    
    
    p = np.clip(p, 1e-15, 1-1e-15)    
    loss = -y * np.log(p) - (1-y) * np.log(1-p) 
    return loss

    # raise NotImplementedError('This function must be implemented by the student.')


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
        
    values = np.zeros([len(y),max(y)+1], dtype= int)
    for i in range(len(y)):
        values[i][y[i]] = 1
    return values

    # raise NotImplementedError('This function must be implemented by the student.')
