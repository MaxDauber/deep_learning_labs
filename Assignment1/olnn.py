#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se

import numpy as np
import matplotlib.pyplot as plt

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

    def ComputeGradientsAnalytical(self, X, Y, W, lamda):
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
        grad_b = np.zeros(np.shape(self.b))
        P = self.EvaluateClassifier(X, self.W, self.b)
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

    def CheckGradients(self, X, Y, method="fast"):
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
            grad_b_num, grad_w_num = self.ComputeGradsNum(X, Y, P, self.W, self.b, self.lamda, .000001)
        elif method == 'slow':
            grad_b_num, grad_w_num = self.ComputeGradsNumSlow(X, Y, P, self.W, self.b, self.lamda, .000001)

        grad_b, grad_w = self.ComputeGradientsAnalytical(X, Y, self.W, self.lamda)


        grad_w_vec = grad_w.flatten()
        grad_w_num_vec = grad_w_num.flatten()
        grad_b_vec = grad_b.flatten()
        grad_b_num_vec = grad_b_num.flatten()
        print("* W gradients *")
        print("mean relative error: ", np.mean(abs(grad_w_vec / grad_w_num_vec - 1)))
        print("* Bias gradients *")
        print("mean relative error: ", np.mean(abs(grad_b_vec / grad_b_num_vec - 1)))

    def MiniBatchGD(self, X, Y, X_val, Y_val, GDparams, verbose=True):
        """
            Trains OLNN using mini-batch gradient descent

            Args:
                X: data matrix
                Y: one-hot-encoding labels matrix
                X: validation data matrix
                Y: validation one-hot-encoding labels matrix
                GDparams: hyperparameter object
                    n_batch : batch size
                    eta : learning rate
                    n_epochs : number of training epochs
                verbose : whether to print updates and plot
        Returns:
            acc_train (float): the accuracy on the training set
            acc_val   (float): the accuracy on the validation set
            acc_test  (float): the accuracy on the testing set
        """
        self.cost_hist_tr = []
        self.cost_hist_val = []
        self.acc_hist_tr = []
        self.acc_hist_val = []

        # rounded to avoid non-integer number of datapoints per step
        num_batches = int(self.n / GDparams.n_batch)

        for epoch in range(GDparams.n_epochs):
            for j in range(num_batches):
                j_start = j * GDparams.n_batch
                j_end = j * GDparams.n_batch + GDparams.n_batch
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]
                grad_b, grad_w = self.ComputeGradientsAnalytical(X_batch, Y_batch,  self.W, self.lamda)
                self.w = self.w - GDparams.eta * grad_w
                self.b = self.b - GDparams.eta * grad_b

                if verbose:
                    self.PerformanceUpdate(epoch, X, Y, self.X_val, self.Y_val)
            # self.plot_cost_and_acc()
            # self.show_w()

    def PerformanceUpdate(self, epoch, X_train, Y_train, X_val, Y_val):
        """
        Compute and store the performance (cost and accuracy) of the model after every epoch,
        so it can be used later to plot the evolution of the performance
        """
        Y_pred_train = self.EvaluateClassifier(X_train)
        Y_pred_val = self.evaluate(X_val)
        cost_train = self.compute_cost(X_train, Y_pred_train)
        acc_train = self.compute_accuracy(Y_pred_train, Y_train)
        cost_val = self.compute_cost(X_val, Y_pred_val)
        acc_val = self.compute_accuracy(Y_pred_val, Y_val)
        self.cost_hist_tr.append(cost_train)
        self.acc_hist_tr.append(acc_train)
        self.cost_hist_val.append(cost_val)
        self.acc_hist_val.append(acc_val)
        print("Epoch ", epoch, " // Train accuracy: ", acc_train, " // Train cost: ", cost_train)


class GDparams:
    """
    Class containing hyperparameters for MiniBatchGD

    """

    def __init__(self, n_batch, eta, n_epochs):
        # n_batch: Number of samples in each mini-batch.
        self.n_batch = n_batch

        # eta: Learning rate
        self.eta = eta

        # n_epochs: Maximum number of learning epochs.
        self.n_epochs = n_epochs