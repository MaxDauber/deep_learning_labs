#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

class MLNN:
    """
    MLNN - Multi Layer Neural Net (2+)
    """

    def __init__(self, extracted, data, targets,**kwargs):
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
            "h_param": 1e-6,  # parameter h for numerical grad check
            "lr_max": 1e-1,  # default maximum for cyclical learning rate
            "lr_min": 1e-5  # default minimum for cyclical learning rate
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))
        for k, v in extracted.items():
            setattr(self, k, v)

        self.hidden_sizes = [50, 50]
        # self.hidden_sizes = [50, 30, 20, 20, 10, 10, 10, 10] # for testing 9 layer network
        self.num_hidden = len(self.hidden_sizes)
        self.d = data.shape[0]
        self.n = data.shape[1]
        self.K = targets.shape[0]
        self.W, self.b = self.InitializeWeightsBias()
        self.test_accuracy = 0
        self.test_cost = 0



    def InitializeWeightsBias(self):
        """
            Initialize weight matrix and bias

            Args:
                None

            Returns:
                b = bias
                W = weights for k layers
        """

        W = [np.random.normal(self.mean_weights, self.var_weights, (self.hidden_sizes[0], self.d))] # first layer
        for i in range(1, self.num_hidden): # add intermediary layers
            W.append(np.random.normal(self.mean_weights, self.var_weights, (self.hidden_sizes[i], self.hidden_sizes[i - 1])))
        W.append(np.random.normal(self.mean_weights, self.var_weights, (self.K, self.hidden_sizes[self.num_hidden - 1])))

        b = [np.zeros((self.hidden_sizes[idx], 1)) for idx in range(self.num_hidden)] # add intermediary layers
        b.append(np.zeros((self.K, 1))) # add last layer

        return W, b

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

    def LeakyReLu(self, x):
        """Computes ReLU activation
        Args:
            x: input matrix
        Returns:
            x : relu matrix
        """
        return np.maximum(x, 0.01*x)

    def EvaluateClassifier(self, X, W, b):
        """
            Output stable softmax probabilities of the classifier

            Args:
                X: data matrix
                W: weights
                b: bias

            Returns:
                s: s node in computational graph
                p : a stable softmax matrix
                h1 : intermediate ReLU values

            """

        h = [X]
        s = []
        for layer in range(len(W)):
            s_idx = np.dot(W[layer], h[-1]) + b[layer]
            s.append(s_idx)
            h.append(self.ReLu(s_idx))
        P = self.SoftMax(s[-1])
        h.pop(0)

        return s, h, P

    def EvaluateClassifierBatchNorm(self, X, W, b, training=False):
        """
            Output stable softmax probabilities of the classifier

            Args:
                X: data matrix
                W: weights
                b: bias
                epsilon:

            Returns:
                s: s node in computational graph
                p : a stable softmax matrix
                h1 : intermediate ReLU values
                P:
                scores:
                batch_norm_scores:
                batch_norm_relu_scores:
                mus:
                vars:


        """
        scores = []  # nonnormalized scores
        batch_norm_scores = []
        batch_norm_relu_scores = [X]
        mus = []  # means
        vars = []  # variabilitiess
        score_layer = None


        for layer in range(len(W)):
            score_layer = np.dot(W[layer], batch_norm_relu_scores[-1]) + b[layer]
            scores.append(score_layer)

            # mean and variance calculations for each layer
            mu = np.mean(score_layer, axis=1)
            var = np.var(score_layer, axis=1)
            mus.append(mu)
            vars.append(var)

            # batch normalization calculations
            batch_norm = np.zeros(np.shape(score_layer))
            a = np.diag(np.power((var + 1e-16), (-1 / 2))) # 1e-16 is to prevent zero division
            for i in range(np.shape(score_layer)[1]):
                batch_norm[:, i] = np.dot(a, (score_layer[:, i] - mu))

            batch_norm_scores.append(batch_norm)
            batch_norm_relu_scores.append(self.ReLu(batch_norm))

        # final layer computation and pop X
        P = self.SoftMax(score_layer)
        batch_norm_relu_scores.pop(0)

        return P, scores, batch_norm_scores, batch_norm_relu_scores, mus, vars



    def ComputeAccuracy(self, X, y, W, b):
        """
           Computes accuracy of network's predictions

            Args:
                X: data matrix
                y: ground truth labels
                W: weights
                b: bias

            Returns:
                Accuracy on provided sets
        """


        # calculate predictions
        _, _, P = self.EvaluateClassifier(X, W, b)
        preds = np.argmax(P, axis=0)

        # calculate num of correct predictions
        correct = np.count_nonzero((y - preds) == 0)

        all = np.size(preds)
        if all == 0:
            raise (ZeroDivisionError("Zero Division Error!"))

        return correct / all

    def ComputeCost(self, X, Y, W, b, lamda):
        """
            Computes the cost function for the set of images using cross-entropy loss

            Args:
                X: data matrix
                Y: one-hot encoding labels matrix
                W: weights
                b: bias
                lamda: regularization term

            Returns:
                cost (float): the cross-entropy loss
        """

        # dimensions of data
        d = np.shape(X)[0]
        N = np.shape(X)[1]

        # L2 regularization term adjusted for k-layers
        regularization_term = np.sum([lamda * np.sum(np.power(W[layer], 2)) for layer in range(len(W))])

        # cross-entropy loss term
        _, _, P = self.EvaluateClassifier(X, W, b)
        cross_entropy_loss = 0 - np.log(np.sum(np.prod((np.array(Y), P), axis=0), axis=0))

        return (1 / N) * np.sum(cross_entropy_loss) + regularization_term

    def ComputeGradients(self, X, Y, W, b, lamda):
        """
            Computes the gradients of the weight and bias parameters using analytical method

            Args:
                X: data batch matrix
                Y: one-hot-encoding labels batch vector
                P: evaluated classifier for the batch
                W: weights
                b: bias
                lamda: regularization term

            Returns:
                gradient_W: gradient of the weight parameter
                gradient_b: gradient of the bias parameter

        """

        gradient_W = [np.zeros(np.shape(W[i])) for i in range(len(W))]
        gradient_b = [np.zeros(np.shape(b[i])) for i in range(len(W))]

        # Forward Pass
        _, H, P = self.EvaluateClassifier(X, W, b)

        # Backward pass
        G_batch = - (Y - P)

        for layer in range(self.num_hidden, 0, -1):
            #nasty code, but will refactor later
            gradient_W[layer] = np.dot(np.multiply((1 / np.shape(X)[1]), G_batch), np.transpose(H[layer - 1])) \
                                + 2 * np.multiply(lamda, W[layer])
            gradient_b[layer] = np.reshape(np.dot(np.multiply((1 / np.shape(X)[1]), G_batch), np.ones(np.shape(X)[1])),
                                       (gradient_b[layer].shape[0], 1))

            G_batch = np.dot(np.transpose(W[layer]), G_batch)
            H[layer - 1][H[layer - 1] <= 0] = 0
            G_batch = np.multiply(G_batch, H[layer - 1] > 0)
            # np.dot(np.diag(list(map(lambda num: num > 0, H[layer - 1]))), G_batch)

        gradient_W[0] = np.dot(np.multiply((1 / np.shape(X)[1]), G_batch), np.transpose(X)) + np.multiply(lamda, W[0])
        gradient_b[0] = np.reshape(np.dot(np.multiply((1 / np.shape(X)[1]), G_batch), np.ones(np.shape(X)[1])), b[0].shape)

        return gradient_W, gradient_b

    def ComputeGradientsBatchNorm(self, X, Y, W, b, lamda):
        """
            Computes the gradients of the weight and bias parameters using analytical method and batch normalization

            Args:
                X: data batch matrix
                Y: one-hot-encoding labels batch vector
                P: evaluated classifier for the batch
                W: weights
                b: bias
                lamda: regularization term

            Returns:
                gradient_W: gradient of the weight parameter
                gradient_b: gradient of the bias parameter

        """
        layers = len(W)

        gradient_W = [np.zeros(np.shape(W[i])) for i in range(len(W))]
        gradient_b = [np.zeros(np.shape(b[i])) for i in range(len(W))]

        # Forward Pass
        P, scores, bn_scores, bn_relu_scores, mus, vars = self.EvaluateClassifierBatchNorm(X, W, b)

        G_batch = -(Y - P)
        layer = len(W) - 1

        # compute grads
        gradient_b.insert(0, (1 / np.shape(Y)[1]) * np.sum(G_batch, axis=1).reshape(-1, 1))
        gradient_W.insert(0, (1 / np.shape(Y)[1]) * np.dot(G_batch, bn_relu_scores[layer-1].T) + 2 * lamda * W[layer])

        # propogate to prev layer
        G_batch = np.dot(np.transpose(W[layer]), G_batch)
        G_batch = np.multiply(G_batch, list(map(lambda num: num > 0, bn_scores[layer - 1])))

        # previous layers
        for layer in range(layers - 2, -1, -1):

            n = np.shape(G_batch)[1]
            var_layer = vars[layer]
            mu_layer = mus[layer]
            score_layer = scores[layer]

            V12 = np.diag(np.power((var_layer + 1e-16), (-1 / 2)))
            V32 = np.diag(np.power((var_layer + 1e-16), (-3 / 2)))

            gradJvar = np.zeros(np.shape(mu_layer))
            for i in range(n):
                gradJvar += np.dot(g[:, i], (np.dot(V32, np.diag(score_layer[:, i] - mu_layer))))
            gradJvar = -(1 / 2) * gradJvar

            gradJmu = np.zeros(np.shape(mu_layer))
            for i in range(n):
                gradJmu += np.dot(g[:, i], V12)
            gradJmu = gradJmu * (-1)

            gnew = np.zeros(np.shape(g))
            # for each datapoint
            for i in range(n):
                gnew[:, i] = np.dot(g[:, i], V12) + (2 / n) * np.dot(gradJvar, np.diag(score_layer[:, i] - mu_layer)) + (
                            1 / n) * gradJmu


            # compute grads
            if layer > 0:
                bn_relu_scores_prev_layer = bn_relu_scores[layer - 1]
            else:
                bn_relu_scores_prev_layer = X

            gradient_b.insert(0, (1 / np.shape(Y)[1]) * np.sum(G_batch, axis=1).reshape(-1, 1))
            gradient_W.insert(0, (1 / np.shape(Y)[1]) * np.dot(G_batch, bn_relu_scores[layer].T) + 2 * lamda * W[layer])


            bgrad, Wgrad = ComputeBatchBackGrads(n, G, bn_relu_scores_prev_layer, lamda, W, layer)
            gradient_b.insert(0, bgrad)
            gradient_W.insert(0, Wgrad)
            # propagate to prev layer
            if layer > 0:
                G = PropagateBatchBackGrads(G, W, bn_scores, layer)

        return gradient_W, gradient_b

    # def IndXPositive(x):
    #     above_zero_indices = x > 0
    #     below_zero_indices = x <= 0
    #     x[above_zero_indices] = 1
    #     x[below_zero_indices] = 0
    #
    #     return x
    #
    # def ComputeBatchBackGrads(n, g, bn_relu_scores_prev_layer, lamda, W, layer):
    #     bgrad = (1 / n) * np.sum(g, axis=1)
    #     bgrad = bgrad.reshape(-1, 1)
    #     Wgrad = (1 / n) * np.dot(g, bn_relu_scores_prev_layer.T) + 2 * lamda * W[layer]
    #     return (bgrad, Wgrad)
    #
    # def PropagateBatchBackGrads(g, W, bn_scores, layer):
    #     g = np.dot(W[layer].T, g)
    #     g = np.multiply(g, IndXPositive(bn_scores[layer - 1]))
    #     # did the elementwise multiplication (above)
    #     # instead of this:
    #     # for i in range(n):
    #     #     g[:,i] = np.dot(g[:,i], np.diag(IndXPositive(bn_scores[-2][:,i])))
    #     return g
    #
    # def BatchNormBackPass(layer, g, scores, mus, vars, epsilon):
    #     n = np.shape(g)[1]
    #
    #     vl = vars[layer]
    #     mul = mus[layer]
    #     sl = scores[layer]
    #
    #     V12 = (vl + epsilon)
    #     V12 = np.power(V12, (-1 / 2))
    #     V12 = np.diag(V12)
    #
    #     V32 = (vl + epsilon)
    #     V32 = np.power(V32, (-3 / 2))
    #     V32 = np.diag(V32)
    #
    #     gradJvar = np.zeros(np.shape(mul))
    #     for i in range(n):
    #         gradJvar += np.dot(g[:, i], (np.dot(V32, np.diag(sl[:, i] - mul))))
    #     gradJvar = -(1 / 2) * gradJvar
    #
    #     gradJmu = np.zeros(np.shape(mul))
    #     for i in range(n):
    #         gradJmu += np.dot(g[:, i], V12)
    #     gradJmu = gradJmu * (-1)
    #
    #     gnew = np.zeros(np.shape(g))
    #     # for each datapoint
    #     for i in range(n):
    #         gnew[:, i] = np.dot(g[:, i], V12) + (2 / n) * np.dot(gradJvar, np.diag(sl[:, i] - mul)) + (1 / n) * gradJmu
    #
    #     return gnew
    #
    # def BackwardBatchNorm(X, Y, P, W, scores, bn_scores, bn_relu_scores, lamda, mus, vars, epsilon):
    #     n = np.shape(Y)[1]
    #     layers = len(W)
    #
    #     bgrads = []
    #     Wgrads = []
    #
    #     # last layer
    #     layer = layers - 1
    #     g = -(Y - P)
    #     # compute grads
    #     bgrad, Wgrad = ComputeBatchBackGrads(n, g, bn_relu_scores[layer - 1], lamda, W, layer)
    #     bgrads.insert(0, bgrad)
    #     Wgrads.insert(0, Wgrad)
    #     # propagate to prev layer
    #     g = PropagateBatchBackGrads(g, W, bn_scores, layer)
    #
    #     # previous layers
    #     for layer in range(layers - 2, -1, -1):
    #         g = BatchNormBackPass(layer, g, scores, mus, vars, epsilon)
    #         # compute grads
    #         if layer > 0:
    #             bn_relu_scores_prev_layer = bn_relu_scores[layer - 1]
    #         else:
    #             bn_relu_scores_prev_layer = X
    #         bgrad, Wgrad = ComputeBatchBackGrads(n, g, bn_relu_scores_prev_layer, lamda, W, layer)
    #         bgrads.insert(0, bgrad)
    #         Wgrads.insert(0, Wgrad)
    #         # propagate to prev layer
    #         if layer > 0:
    #             g = PropagateBatchBackGrads(g, W, bn_scores, layer)
    #
    #     return (Wgrads, bgrads)
































    def ComputeGradsNum(self, X, Y, W, b, lamda, h):
        """
            Computes the gradients of the weight and bias parameters using numerical computation method

            Args:
                X: data batch matrix
                Y: one-hot-encoding labels batch vector
                P: evaluated classifier for the batch
                W: weights
                lamda: regularization term

            Returns:
                gradient_W: gradient of the weight parameter
                gradient_b: gradient of the bias parameter
        """

        # initialize grads
        gradient_W = [np.zeros(np.shape(W[layer])) for layer in range(len(W))]
        gradient_b = [np.zeros(np.shape(b[layer])) for layer in range(len(W))]


        cost = self.ComputeCost(X, Y, W, b, lamda)

        for k in range(len(W)):
            for i in range(len(b[k])):
                temp = deepcopy(b)
                temp[k][i] += h
                cost_2 = self.ComputeCost(X, Y, W, temp, lamda)
                gradient_b[k][i] = (cost_2 - cost) / h

            for i in range(W[k].shape[0]):
                for j in range(W[k].shape[1]):
                    temp = deepcopy(W)
                    temp[k][i, j] += h
                    cost_2 = self.ComputeCost(X, Y, temp, b, lamda)
                    gradient_W[k][i, j] = (cost_2 - cost) / h

        return gradient_W, gradient_b

    def CheckGradients(self, X, Y, lamda=0):
        """
            Checks analytically computed gradients against numerically computed to determine margin of error

            Args:
                X: data batch matrix
                Y: one-hot-encoding labels batch vector
                method: type of numerical gradient computation

            Returns:
                None
        """
        grad_w_numerical, grad_b_numerical = self.ComputeGradsNum(X, Y, self.W, self.b, lamda, 1e-5)
        grad_w_analytical, grad_b_analytical = self.ComputeGradients(X, Y, self.W, self.b, lamda)
        print("******* Performing Gradient Checks *******")
        for layer in range(self.num_hidden+1):
            grad_w_vec = grad_w_analytical[layer].flatten()
            grad_w_num_vec = grad_w_numerical[layer].flatten()
            grad_b_vec = grad_b_analytical[layer].flatten()
            grad_b_num_vec = grad_b_numerical[layer].flatten()
            print("* W gradients for layer", self.num_hidden - layer, " *")
            print("mean relative error: ", np.mean(abs((grad_w_vec + self.h_param ** 2) /
                                                       (grad_w_num_vec + self.h_param ** 2) - 1)))
            print("* b gradients for layer", self.num_hidden - layer, " *")
            print("mean relative error: ", np.mean(abs((grad_b_vec + self.h_param ** 2) /
                                                       (grad_b_num_vec + self.h_param ** 2) - 1)))
            x_w = np.arange(1, grad_w_vec.shape[0] + 1)
            plt.bar(x_w, grad_w_vec, 0.35, label='Analytical gradient', color='blue')
            plt.bar(x_w + 0.35, grad_w_num_vec, 0.35, label="fast", color='red')
            plt.legend()
            plt.title(("Gradient check of w", layer, ", batch size = " + str(X.shape[1])))
            plt.show()

            x_b = np.arange(1, grad_b_analytical[layer].shape[0] + 1)
            plt.bar(x_b, grad_b_vec, 0.35, label='Analytical gradient', color='blue')
            plt.bar(x_b + 0.35, grad_b_num_vec, 0.35, label="fast", color='red')
            plt.legend()
            plt.title(("Gradient check of b", layer, ", batch size = " + str(X.shape[1])))
            plt.show()


    def MiniBatchGD(self, X, Y, y, GDparams, verbose=True):
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
        # histories for top level training metrics
        self.history_training_cost = []
        self.history_validation_cost = []
        self.history_training_accuracy = []
        self.history_validation_accuracy = []

        # history for cyclic training
        self.history_update = []


        if GDparams.cyclic:
            lr = GDparams.lr_min
            t = 0
        else:
            lr = GDparams.lr

        # rounded to avoid non-integer number of datapoints per step
        num_batches = int(self.n / GDparams.n_batch)
        update_step = 0
        # for epoch in tqdm(range(GDparams.n_epochs)):
        for epoch in range(GDparams.n_epochs):
            for step in range(num_batches):
                start_batch = step * GDparams.n_batch
                end_batch = start_batch + GDparams.n_batch
                X_batch = X[:, start_batch:end_batch]
                Y_batch = Y[:, start_batch:end_batch]
                grad_w, grad_b = self.ComputeGradients(X_batch, Y_batch, self.W, self.b, GDparams.lamda)
                for layer in range(self.num_hidden):
                    self.W[layer] = self.W[layer] - lr * grad_w[layer]
                    self.b[layer] = self.b[layer] - lr * grad_b[layer]
                update_step += 1

                # implementing cyclic learning rate
                if GDparams.cyclic:
                    if t <= GDparams.n_s:
                        lr = GDparams.lr_min + t / GDparams.n_s * (GDparams.lr_max - GDparams.lr_min)
                    elif t <= 2 * GDparams.n_s:
                        lr = GDparams.lr_max - (t - GDparams.n_s) / GDparams.n_s * (GDparams.lr_max - GDparams.lr_min)
                    t = (t + 1) % (2 * GDparams.n_s)





            if verbose:
                training_cost = self.ComputeCost(X, Y, self.W, self.b, GDparams.lamda)
                training_accuracy = self.ComputeAccuracy(X, y, self.W, self.b)
                validation_cost = self.ComputeCost(self.X_validation, self.Y_validation,
                                                   self.W, self.b, GDparams.lamda)
                validation_accuracy = self.ComputeAccuracy(self.X_validation, self.y_validation,
                                                           self.W, self.b)

                if GDparams.cyclic :
                    self.history_update.append(update_step)

                self.history_training_cost.append(training_cost)
                self.history_training_accuracy.append(training_accuracy)
                self.history_validation_cost.append(validation_cost)
                self.history_validation_accuracy.append(validation_accuracy)

                print("Epoch ", epoch,
                      " | Train accuracy: ", "{:.4f}".format(training_accuracy),
                      " | Train cost: ", "{:.10f}".format(training_cost),
                      " | Validation accuracy: ", "{:.4f}".format(validation_accuracy),
                      " | Validation cost: ", "{:.10f}".format(validation_cost))

        self.test_accuracy = self.ComputeAccuracy(self.X_test, self.y_test, self.W, self.b)
        self.test_cost = self.ComputeCost(self.X_test, self.Y_test, self.W, self.b,
                                          GDparams.lamda)
        print("Test Accuracy: ", self.test_accuracy)
        print("Test Cost: ", self.test_cost)
        # self.ShowWeights(W, GDparams)

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

    def __init__(self, n_batch, lr, lr_max, lr_min, n_s, cyclic, n_epochs, lamda):
        # n_batch: Number of samples in each mini-batch.
        self.n_batch = n_batch

        # eta: Learning rate
        self.lr = lr

        # min/max for cyclical learning rate
        self.lr_max = lr_max
        self.lr_min = lr_min

        self.n_s = n_s
        self.cyclic = cyclic

        # n_epochs: Maximum number of learning epochs.
        self.n_epochs = n_epochs

        # lamda: regularization term used for the gradient descent
        self.lamda = lamda

