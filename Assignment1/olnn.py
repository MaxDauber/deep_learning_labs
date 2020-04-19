#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se
"""
This is the main file of Assignment 1 for DD2424 Deep Learning
This assignment implements a one-layer neural network.
"""

import numpy as np
# import matplotlib.pyplot as plt

class OLNN:
    """
    OLNN - One Layer Neural Net
    """

    def __init__(self, data, targets, **kwargs):
        """
        Initialize Neural Network with data and parameters
        """
        var_defaults = {
            "eta": 0.1,  # learning rate
            "mean_weights": 0,  # mean of the weights
            "var_weights": 0.01,  # variance of the weights
            "lamda": 0,  # regularization parameter
            "batch_size": 100,  #batch size
            "epochs": 40,  # number of epochs
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.d = data.shape[0]
        self.n = data.shape[1]
        self.K = targets.shape[0]
        self.W, self.b = self.InitializeWeightsBias()

    def InitializeWeightsBias(self):
        """
            Initialize weight matrix and bias

            Args:
                None

            Returns:
                b = bias K x 1
                W = weights K x d
        """

        W = np.random.normal(self.mean_weights, self.var_weights, (self.K, self.d))
        b = np.random.normal(self.mean_weights, self.var_weights, (self.K, 1))
        return W, b


    def Softmax(self, X):
        """
            Standard definition of the softmax function

            Args:
                X: data matrix

            Returns:
                Softmax transformed data

        """
        return np.exp(X) / np.sum(np.exp(X), axis=0)



    def EvaluateClassifier(self, X, W, b):
        """
            Output stable softmax probabilities of the classifier

            Args:
                X: data matrix
                W: weights
                b: bias

            Returns:
                Softmax matrix of output probabilities
            """
        return self.Softmax(np.matmul(W, X) + b)


    def ComputeAccuracy(self, X, y, W, b):
        """
           Compute accuracy of network's predictions

            Args:
                X: data matrix
                y: ground truth labels
                W: weights
                b: bias

            Returns:
                acc (float): Accuracy of
        """

        # calculate predictions
        preds = np.argmax(self.EvaluateClassifier(X, W, b), axis=0)

        # calculate num of correct predictions
        num_correct = np.count_nonzero((y - preds) == 0)

        all = np.size(preds)

        if all != 0:
            acc = num_correct / all
        else:
            raise (ZeroDivisionError("Zero Division Error!"))

        return acc

    def ComputeCost(self, X, Y, W, b, lamda):
        """
            Computes the cost function for the set of images using cross-entropy loss

            Args:
                X: data matrix
                Y: one-hot encoding labels matrix
                lamda: regularization term

            Returns:
                cost (float): the cross-entropy loss
        """
        # dimensions of data
        d = np.shape(X)[0]
        N = np.shape(X)[1]

        # L2 regularization term
        regularization_term = lamda * np.sum(np.power(W, 2))

        # cross-entropy loss term
        loss_term = 0 - np.log(np.sum(np.prod((np.array(Y), self.EvaluateClassifier(X, W, b)), axis=0), axis=0))

        # Cost Function Calculation
        J = (1/N) * np.sum(loss_term) + regularization_term

        return J

    def ComputeGradients(self, X, Y, P, W, lamda):
        """ Computes the gradients of the weight and bias parameters
                Args:
                    X: data batch matrix
                    Y: one-hot-encoding labels batch vector
                    P: evaluated classifier for the batch
                    W: weights
                    lamda: regularization term

                Returns:
                    grad_W: gradient of the weight parameter
                    grad_b: gradient of the bias parameter

                """

        grad_W = np.zeros(np.shape(W))
        grad_b = np.zeros(np.shape(b))

        N = np.shape(X)[1]
        for i in range(N):
            Y_i = Y[:, i].reshape((-1, 1))
            P_i = P[:, i].reshape((-1, 1))
            g = - (Y_i - P_i)
            grad_b = grad_b + g
            grad_W = grad_W + g * X[:, i]

        grad_b = np.divide(grad_b, N)
        grad_W = np.divide(grad_W, N) + 2 * lamda * W

        return grad_W, grad_b


    def ComputeGradsNum(self, X, Y, P, W, b, lamda, h):
        """ Converted from matlab code """
        no 	= 	W.shape[0]
        d 	= 	X.shape[0]

        grad_W = np.zeros(W.shape);
        grad_b = np.zeros((no, 1));

        c = self.ComputeCost(X, Y, W, b, lamda);

        for i in range(len(b)):
            b_try = np.array(b)
            b_try[i] += h
            c2 = self.ComputeCost(X, Y, W, b_try, lamda)
            grad_b[i] = (c2-c) / h

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.array(W)
                W_try[i,j] += h
                c2 = self.ComputeCost(X, Y, W_try, b, lamda)
                grad_W[i,j] = (c2-c) / h

        return [grad_W, grad_b]

    def ComputeGradsNumSlow(self, X, Y, P, W, b, lamda, h):
        """ Converted from matlab code """
        no 	= 	W.shape[0]
        d 	= 	X.shape[0]

        grad_W = np.zeros(W.shape);
        grad_b = np.zeros((no, 1));
        for i in range(len(b)):
            b_try = np.array(b)
            b_try[i] -= h
            c1 = self.ComputeCost(X, Y, W, b_try, lamda)

            b_try = np.array(b)
            b_try[i] += h
            c2 = self.ComputeCost(X, Y, W, b_try, lamda)

            grad_b[i] = (c2-c1) / (2*h)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.array(W)
                W_try[i,j] -= h
                c1 = self.ComputeCost(X, Y, W_try, b, lamda)
                W_try = np.array(W)
                W_try[i,j] += h
                c2 = self.ComputeCost(X, Y, W_try, b, lamda)

                grad_W[i,j] = (c2-c1) / (2*h)
        return [grad_W, grad_b]