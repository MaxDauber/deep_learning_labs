#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se

import numpy as np
import matplotlib.pyplot as plt

class MLNN:
    """
    MLNN - Multi Layer Neural Net
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
            "batch_size": 100,  # batch size
            "epochs": 40,  # number of epochs
            "hidden_size": 50,  # number of nodes in the hidden layer
            "h_param": 1e-6,  # parameter h for numerical grad check
            "lr_max": 1e-1,  # maximum for cyclical learning rate
            "lr_min": 1e-5  # minimum for cyclical learning rate
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))
        for k, v in extracted.items():
            setattr(self, k, v)

        self.d = data.shape[0]
        self.n = data.shape[1]
        self.K = targets.shape[0]
        self.m = self.hidden_size #nodes in hidden layer
        self.W_1, self.b_1, self.W_2, self.b_2 = self.InitializeWeightsBias()

    def InitializeWeightsBias(self):
        """
            Initialize weight matrix and bias

            Args:
                None

            Returns:
                b = bias K x 1
                W = weights K x d
        """

        W_1 = np.random.normal(self.mean_weights, self.var_weights, (self.m, self.d))
        W_2 = np.random.normal(self.mean_weights, self.var_weights, (self.K, self.m))
        b_1 = np.random.normal(self.mean_weights, self.var_weights, (self.m, 1))
        b_2 = np.random.normal(self.mean_weights, self.var_weights, (self.K, 1))
        return W_1, b_1, W_2, b_2

    def Softmax(self, X):
        """
            Standard definition of the softmax function

            Args:
                X: data matrix

            Returns:
                Softmax transformed data

        """
        return np.exp(X) / np.sum(np.exp(X), axis=0)

    def ReLu(x):
        """Computes ReLU activation
        Args:
            x: input matrix
        Returns:
            x : relu matrix
        """
        return np.maximum(x, 0)

    def EvaluateClassifier(self, X, W_1, b_1, W_2, b_2):
        """
            Output stable softmax probabilities of the classifier

            Args:
                X: data matrix
                W_1: weights
                b_1: bias
                W_2: weights
                b_2: bias

            Returns:
                p : a stable softmax matrix
                h : intermediate ReLU values
            """
        h = self.ReLU(np.dot(W_1, X) + b_1)
        P = self.SoftMax(np.dot(W_2, h) + b_2)

        return P, h

    def ComputeAccuracy(self, X, y, W_1, b_1, W_2, b_2):
        """
           Compute accuracy of network's predictions

            Args:
                X: data matrix
                y: ground truth labels
                W: weights
                b: bias

            Returns:
                acc : Accuracy on provided sets
        """


        # calculate predictions
        P, h = self.EvaluateClassifier(X, W_1, b_1, W_2, b_2)
        preds = np.argmax(P, axis=0)

        # calculate num of correct predictions
        correct = np.count_nonzero((y - preds) == 0)

        all = np.size(preds)
        if all == 0:
            raise (ZeroDivisionError("Zero Division Error!"))

        return correct / all

    def ComputeCost(self, X, Y, W_1, b_1, W_2, b_2, lamda):
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

        # L2 regularization term adjusted for 2 layers
        regularization_term = lamda * (np.sum(np.power(W_1, 2)) + np.sum(np.power(W_2, 2)))

        # cross-entropy loss term
        P, _, _ = self.EvaluateClassifier(X, W_1, b_1, W_2, b_2)
        cross_entropy_loss = 0 - np.log(np.sum(np.prod((np.array(Y), P), axis=0), axis=0))


        return (1 / N) * np.sum(cross_entropy_loss) + regularization_term # J

    def X_Positive(x):
        above_zero_indices = x > 0
        below_zero_indices = x <= 0
        x[above_zero_indices] = 1
        x[below_zero_indices] = 0

        return x

    def ComputeGradients(self, X, Y, W_1, b_1, W_2, b_2, lamda):
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

        grad_W1 = np.zeros(np.shape(W_1))
        grad_b1 = np.zeros(np.shape(b_1))
        grad_W2 = np.zeros(np.shape(W_2))
        grad_b2 = np.zeros(np.shape(b_2))

        P, h = self.EvaluateClassifier(X, W_1, b_1, W_2, b_2)

        N = np.shape(X)[1]
        for i in range(N):
            Yi = Y[:, i].reshape((-1, 1))
            Pi = P[:, i].reshape((-1, 1))
            Xi = X[:, i].reshape((-1, 1))
            hi = h[:, i].reshape((-1, 1))
            si = s1[:, i]

            g = Pi - Yi
            grad_b2 = grad_b2 + g
            grad_W2 = grad_W2 + np.dot(g, np.transpose(hi))

            # propagate error backwards
            g = np.dot(np.transpose(W_2), g)
            g = np.dot(np.diag(self.X_Positive(si)), g)

            grad_b1 = grad_b1 + g
            grad_W1 = grad_W1 + np.dot(g, np.transpose(Xi))

        grad_b1 = np.divide(grad_b1, N)
        grad_W1 = np.divide(grad_W1, N) + 2 * lamda * W_1

        grad_b2 = np.divide(grad_b2, N)
        grad_W2 = np.divide(grad_W2, N) + 2 * lamda * W2

        return grad_W1, grad_b1, grad_W2, grad_b2

        # grad_W = np.zeros(np.shape(W))
        # grad_b = np.zeros(np.shape(b))
        #
        # P = self.EvaluateClassifier(X, W, b)
        #
        # N = np.shape(X)[1]
        # for i in range(N):
        #     Y_i = Y[:, i].reshape((-1, 1))
        #     P_i = P[:, i].reshape((-1, 1))
        #     g = - (Y_i - P_i)
        #     grad_b = grad_b + g
        #     grad_W = grad_W + g * X[:, i]
        #
        # grad_b = np.divide(grad_b, N)
        # grad_W = np.divide(grad_W, N) + 2 * lamda * W
        #
        # return grad_W, grad_b

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

        grad_b1_n, grad_b2_n, grad_w1_n, grad_w2_n = self.ComputeGradsNumSlow(X, Y, P, self.W, self.b, lamda, .000001)

        grad_b, grad_w = self.ComputeGradients(X, Y, self.W, self.b, lamda)

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
            training_accuracy (float): the accuracy on the training set
            validation_accuracy   (float): the accuracy on the validation set
            acc_test  (float): the accuracy on the testing set
        """
        self.history_training_cost = []
        self.history_validation_cost = []
        self.history_training_accuracy = []
        self.history_validation_accuracy = []

        # rounded to avoid non-integer number of datapoints per step
        num_batches = int(self.n / GDparams.n_batch)

        for epoch in range(GDparams.n_epochs):
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
                training_cost = self.ComputeCost(X, Y, W, b, GDparams.lamda)
                training_accuracy = self.ComputeAccuracy(X, y, W, b)
                validation_cost = self.ComputeCost(self.X_validation, self.Y_validation, W, b, GDparams.lamda)
                validation_accuracy = self.ComputeAccuracy(self.X_validation, self.y_validation, W, b)

                self.history_training_cost.append(training_cost)
                self.history_training_accuracy.append(training_accuracy)
                self.history_validation_cost.append(validation_cost)
                self.history_validation_accuracy.append(validation_accuracy)

                print("Epoch ", epoch,
                      " | Train accuracy: ", "{:.4f}".format(training_accuracy),
                      " | Train cost: ", "{:.10f}".format(training_cost),
                      " | Validation accuracy: ", "{:.4f}".format(validation_accuracy),
                      " | Validation cost: ", "{:.10f}".format(validation_cost))
        print("Test Accuracy: ", self.ComputeAccuracy(self.X_test, self.y_test, W, b))
        self.ShowWeights(W, GDparams)

    def GeneratePlots(self):
        x = list(range(1, len(self.history_training_cost) + 1))
        plt.plot(x, self.history_training_cost, label="Training Loss")
        plt.plot(x, self.history_validation_cost, label="Validation Loss")
        plt.title("Loss over epochs for ")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.plot(x, self.history_training_accuracy, label="Training Accuracy")
        plt.plot(x, self.history_validation_accuracy, label="Validation Accuracy")
        plt.title("Accuracy over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    def ShowWeights(self, W, params):
        """ Display the image for each label in W """
        cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        fig, ax = plt.subplots(2, 5)
        fig.suptitle(
            "Weights learned for n_batch=100, n_epochs=40, eta=" + str(params.eta) + ", lamda=" + str(params.lamda))
        for i in range(2):
            for j in range(5):
                img = W[i * 5 + j, :].reshape((32, 32, 3), order='F')
                img = ((img - img.min()) / (img.max() - img.min()))
                img = np.rot90(img, 3)
                ax[i][j].imshow(img, interpolation="nearest")
                ax[i][j].set_title(cifar_classes[i * 5 + j])
                ax[i][j].axis('off')
        plt.savefig("Plots/weight_visualization_eta=" + str(params.eta) + "_lamda=" + str(params.lamda) + ".png")
        plt.show()


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

        # lamda: regularization term used for the gradient descent
        self.lamda = lamda