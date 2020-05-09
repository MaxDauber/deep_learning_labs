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

    def __init__(self, extracted, data, targets, **kwargs):
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
        for k, v in extracted.items():
            setattr(self, k, v)

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

    def ComputeGradients(self, X, Y, W, b, lamda):
        """
            Computes the gradients of the weight and bias parameters

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

        P = self.EvaluateClassifier(X, W, b)

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

    def CheckGradients(self, X, Y, lamda=0, method="fast"):
        """
            Checks analytically computed gradients against numerically computed to compute error

            Args:
                X: data batch matrix
                Y: one-hot-encoding labels batch vector
                method: type of numerical gradient computation

            Returns:
                None
        """
        P = self.EvaluateClassifier(X, self.W, self.b)

        if method == 'fast':
            grad_b_num, grad_w_num = self.ComputeGradsNum(X, Y, P, self.W, self.b, lamda, .000001)
        elif method == 'slow':
            grad_b_num, grad_w_num = self.ComputeGradsNumSlow(X, Y, P, self.W, self.b, lamda, .000001)

        grad_b, grad_w = self.ComputeGradients(X, Y, self.W, self.b, lamda)


        grad_w_vec = grad_w.flatten()
        grad_w_num_vec = grad_w_num.flatten()
        grad_b_vec = grad_b.flatten()
        grad_b_num_vec = grad_b_num.flatten()
        print("* W gradients *")
        print("mean relative error: ", np.mean(abs(grad_w_vec / grad_w_num_vec - 1)))
        print("* Bias gradients *")
        print("mean relative error: ", np.mean(abs(grad_b_vec / grad_b_num_vec - 1)))

    def MiniBatchGD(self, X, Y, y, GDparams, W, b, verbose=True):
        """
            Trains OLNN using mini-batch gradient descent

            Args:
                X: data matrix
                Y: one-hot-encoding labels matrix
                GDparams: hyperparameter object
                    n_batch : batch size
                    eta : learning rate
                    n_epochs : number of training epochs
                lamda: regularization term
                verbose :
        Returns:
            acc_train (float): the accuracy on the training set
            acc_val   (float): the accuracy on the validation set
            acc_test  (float): the accuracy on the testing set
        """
        self.cost_hist_tr = []
        self.cost_hist_val = []
        self.acc_hist_tr = []
        self.acc_hist_val = []
        # print("X ", np.shape(X))
        # print("Y ", np.shape(Y))
        # print("y ", np.shape(y))
        # print("W ", np.shape(W))
        # print("b ", np.shape(b))
        # print("X_val ", np.shape(self.X_validation))
        # print("Y_val ", np.shape(self.Y_validation))
        # print("y_val ", np.shape(self.y_validation))



        # rounded to avoid non-integer number of datapoints per step
        num_batches = int(self.n / GDparams.n_batch)

        for epoch in range(1, GDparams.n_epochs+1):
            for step in range(num_batches):
                start_batch = step * GDparams.n_batch
                end_batch = start_batch + GDparams.n_batch
                X_batch = X[:, start_batch:end_batch]
                Y_batch = Y[:, start_batch:end_batch]
                grad_w, grad_b = self.ComputeGradients(X_batch, Y_batch, W, b, GDparams.lamda)
                W = W - GDparams.eta * grad_w
                b = b - GDparams.eta * grad_b
            if verbose:
                # Y_pred_train = self.EvaluateClassifier(X, W, b)
                # Y_pred_val = self.EvaluateClassifier(self.X_validation, W, b)
                cost_train = self.ComputeCost(X, Y, W, b, GDparams.lamda)
                acc_train = self.ComputeAccuracy(X, y, W, b)
                cost_val = self.ComputeCost(self.X_validation, self.Y_validation, W, b, GDparams.lamda)
                acc_val = self.ComputeAccuracy(self.X_validation, self.y_validation, W, b)
                print("Epoch ", epoch, " // Train accuracy: ", acc_train, " // Train cost: ", cost_train,
                      " // Validation accuracy: ", acc_val, " // Validation cost: ", cost_val)
# def ComputeCost(self, X, Y, W, b, lamda):
# def ComputeAccuracy(self, X, y, W, b):

class GDparams:
    """
    Class containing hyperparameters for MiniBatchGD

    """

    def __init__(self, n_batch, eta, n_epochs, lamda):
        # n_batch: Number of samples in each mini-batch.
        self.n_batch = n_batch

        # eta: Learning rate
        self.eta = eta

        # n_epochs: Maximum number of learning epochs.
        self.n_epochs = n_epochs

        self.lamda = lamda