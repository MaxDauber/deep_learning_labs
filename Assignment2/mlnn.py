#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

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
                b 1/2 = bias
                W 1/2 = weights
        """

        W_1 = np.random.normal(self.mean_weights, self.var_weights, (self.m, self.d))
        W_2 = np.random.normal(self.mean_weights, self.var_weights, (self.K, self.m))
        b_1 = np.random.normal(self.mean_weights, self.var_weights, (self.m, 1))
        b_2 = np.random.normal(self.mean_weights, self.var_weights, (self.K, 1))
        return W_1, b_1, W_2, b_2

    def SoftMax(self, X):
        """
            Standard definition of the softmax function

            Args:
                X: data matrix

            Returns:
                Softmax transformed data

        """
        return np.exp(X) / np.sum(np.exp(X), axis=0)

    def ReLu(self, x):
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
                s1: s node in computational graph
                p : a stable softmax matrix
                h1 : intermediate ReLU values

            """
        s1 = np.dot(W_1, X) + b_1

        h1 = self.ReLu(s1) #1st hidden layer

        P = self.SoftMax(np.dot(W_2, h1) + b_2)

        return s1, h1, P

    def ComputeAccuracy(self, X, y, W_1, b_1, W_2, b_2):
        """
           Computes accuracy of network's predictions

            Args:
                X: data matrix
                y: ground truth labels
                W_1: weights
                b_1: bias
                W_2: weights
                b_2: bias

            Returns:
                Accuracy on provided sets
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
        _, _, P = self.EvaluateClassifier(X, W_1, b_1, W_2, b_2)
        cross_entropy_loss = 0 - np.log(np.sum(np.prod((np.array(Y), P), axis=0), axis=0))

        return (1 / N) * np.sum(cross_entropy_loss) + regularization_term # J

    def ComputeGradients(self, X, Y, W_1, b_1, W_2, b_2, lamda):
        """
            Computes the gradients of the weight and bias parameters using analytical method

            Args:
                X: data batch matrix
                Y: one-hot-encoding labels batch vector
                P: evaluated classifier for the batch
                W: weights
                lamda: regularization term

            Returns:
                gradient_W1: gradient of the weight parameter
                gradient_b1: gradient of the bias parameter
                gradient_W2: gradient of the weight parameter
                gradient_b2: gradient of the bias parameter

        """

        gradient_W1 = np.zeros(np.shape(W_1))
        gradient_b1 = np.zeros(np.shape(b_1))
        gradient_W2 = np.zeros(np.shape(W_2))
        gradient_b2 = np.zeros(np.shape(b_2))

        # Forward Pass
        s1, h, P = self.EvaluateClassifier(X, W_1, b_1, W_2, b_2)

        # Backward Pass
        for i in range(np.shape(X)[1]):
            Y_i = Y[:, i].reshape((-1, 1))
            P_i = P[:, i].reshape((-1, 1))
            X_i = X[:, i].reshape((-1, 1))
            hidden_i = h[:, i].reshape((-1, 1))
            s_i = s1[:, i]

            temp_g = P_i - Y_i
            gradient_b2 = gradient_b2 + temp_g
            gradient_W2 = gradient_W2 + np.dot(temp_g, hidden_i.T)


            temp_g = np.dot(W_2.T, temp_g)
            temp_g = np.dot(np.diag(list(map(lambda num: num > 0, s_i))), temp_g)


            gradient_b1 = gradient_b1 + temp_g
            gradient_W1 = gradient_W1 + np.dot(temp_g, X_i.T)

        gradient_b1 = np.divide(gradient_b1, np.shape(X)[1])
        gradient_W1 = np.divide(gradient_W1, np.shape(X)[1]) + 2 * lamda * W_1

        gradient_b2 = np.divide(gradient_b2, np.shape(X)[1])
        gradient_W2 = np.divide(gradient_W2, np.shape(X)[1]) + 2 * lamda * W_2

        return gradient_W1, gradient_b1, gradient_W2, gradient_b2


    def ComputeGradsNumSlow(self, X, Y, W1, b1, W2, b2, lamda, h):
        """
            Computes the gradients of the weight and bias parameters using numerical computation method

            Args:
                X: data batch matrix
                Y: one-hot-encoding labels batch vector
                P: evaluated classifier for the batch
                W: weights
                lamda: regularization term

            Returns:
                gradient_W1: gradient of the weight parameter
                gradient_b1: gradient of the bias parameter
                gradient_W2: gradient of the weight parameter
                gradient_b2: gradient of the bias parameter
        """

        W = [W1, W2]
        b = [b1, b2]

        # initialize gradients
        grad_W = []
        grad_b = []

        for i in range(len(W)):
            grad_W.append(np.zeros(np.shape(W[i])))
            grad_b.append(np.zeros(np.shape(b[i])))

        cost = self.ComputeCost(X, Y, W[0], b[0], W[1], b[1], lamda)

        for k in range(len(W)):
            for i in range(len(b[k])):
                b_try = deepcopy(b)
                b_try[k][i] += h
                cost2 = self.ComputeCost(X, Y, W[0], b_try[0], W[1], b_try[1], lamda)
                grad_b[k][i] = (cost2 - cost) / h

            for i in range(W[k].shape[0]):
                for j in range(W[k].shape[1]):
                    W_try = deepcopy(W)
                    W_try[k][i, j] += h
                    cost2 = self.ComputeCost(X, Y, W_try[0], b[0], W_try[1], b[1], lamda)
                    grad_W[k][i, j] = (cost2 - cost) / h

        return grad_W[0], grad_b[0], grad_W[1], grad_b[1]

    def CheckGradients(self, X, Y, lamda=0, method="fast"):
        """
            Checks analytically computed gradients against numerically computed to determine margin of error

            Args:
                X: data batch matrix
                Y: one-hot-encoding labels batch vector
                method: type of numerical gradient computation

            Returns:
                None
        """
        grad_w1_numerical, grad_b1_numerical, grad_w2_numerical, grad_b2_numerical = \
            self.ComputeGradsNumSlow(X, Y, self.W_1, self.b_1, self.W_2, self.b_2, lamda, 1e-5)
        grad_w1_analytical, grad_b1_analytical, grad_w2_analytical, grad_b2_analytical = \
            self.ComputeGradients(X, Y, self.W_1, self.b_1, self.W_2, self.b_2, lamda)

        grad_w_vec = grad_w1_analytical.flatten()
        grad_w_num_vec = grad_w1_numerical.flatten()
        grad_b_vec = grad_b1_analytical.flatten()
        grad_b_num_vec = grad_b1_numerical.flatten()
        print("* W_1 gradients *")
        print("mean relative error: ", np.mean(abs(grad_w_vec / grad_w_num_vec - 1)))
        print("* Bias_1 gradients *")
        print("mean relative error: ", np.mean(abs(grad_b_vec / grad_b_num_vec - 1)))

        grad_w_vec = grad_w2_analytical.flatten()
        grad_w_num_vec = grad_w2_numerical.flatten()
        grad_b_vec = grad_b2_analytical.flatten()
        grad_b_num_vec = grad_b2_numerical.flatten()
        print("* W_1 gradients *")
        print("mean relative error: ", np.mean(abs(grad_w_vec / grad_w_num_vec - 1)))
        print("* Bias_1 gradients *")
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