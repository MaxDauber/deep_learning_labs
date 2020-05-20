#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se
"""
This is the main file of Assignment 1 for DD2424 Deep Learning
This assignment implements a one-layer neural network.
"""


import pickle
# import statistics
# import unittest
# import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mlnn import MLNN, GDparams

def LoadBatch(filename):
    """
        Load Batch for dataset

        Args:
            filename: relative filepath for dataset

        Returns:
            X: images
            Y: one-hot labels
            y: labels
    """
    with open(filename, 'rb') as f:
        dataDict = pickle.load(f, encoding='bytes')

        X = (dataDict[b"data"]).T
        y = dataDict[b"labels"]
        Y = (np.eye(10)[y]).T

        meanX = np.mean(X)
        stdX = np.std(X)
        X = (X - meanX) / stdX

    return X, Y, y



def GeneratePlots(neural_net, params):
    x = list(range(1, len(neural_net.history_training_cost) + 1))
    plt.plot(x, neural_net.history_training_cost, label="Training Loss")
    plt.plot(x, neural_net.history_validation_cost, label="Validation Loss")
    plt.title("Loss over epochs for n_batch=100, n_epochs=40, eta=" + str(params.eta) + ", lamda=" + str(params.lamda))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks([0, 10, 20, 30, 40])
    plt.legend()
    plt.savefig("Plots/loss_eta=" + str(params.eta) + "_lamda=" + str(params.lamda) + ".png", bbox_inches="tight")
    plt.show()

    plt.plot(x, neural_net.history_training_accuracy, label="Training Accuracy")
    plt.plot(x, neural_net.history_validation_accuracy, label="Validation Accuracy")
    plt.title("Accuracy over epochs for n_batch=100, n_epochs=40, eta=" + str(params.eta) + ", lamda=" + str(params.lamda))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks([0, 10, 20, 30, 40])
    plt.legend()
    plt.savefig("Plots/accuracy_eta=" + str(params.eta) + "_lamda=" + str(params.lamda) + ".png", bbox_inches="tight")
    plt.show()

def save_as_mat(data, name="model"):
    """ Used to transfer a python model to matlab """
    # import scipy.io as sio
    # sio.savemat(name'.mat',{name:b})


if __name__ == '__main__':
    np.random.seed(7)
    X_train, Y_train, y_train = LoadBatch("Datasets/cifar-10-batches-py/data_batch_1")
    X_validation, Y_validation, y_validation = LoadBatch("Datasets/cifar-10-batches-py/data_batch_2")
    X_test, Y_test, y_test = LoadBatch("Datasets/cifar-10-batches-py/test_batch")

    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'y_train': y_train,
        'X_validation': X_validation,
        'Y_validation': Y_validation,
        'y_validation': y_validation,
        'X_test': X_test,
        'Y_test': Y_test,
        'y_test': y_test
    }

    # Check Gradient Code is Correct
    neural_net = MLNN(data, X_train[:2, :100], Y_train[:, :100])
    neural_net.CheckGradients(X_train[:2, :100], Y_train[:, :100])

    # Test Neural Network is Running
    # neural_net = OLNN(data, X_train, Y_train)
    # params = GDparams(n_batch = 100, eta = 0.001, n_epochs = 20, lamda = 0)
    # neural_net.MiniBatchGD(X_train, Y_train, y_train, params, neural_net.W, neural_net.b)

    # Run training for all parameter Settings
    # lamdas = [0, 0, .1, 1]
    # etas = [.1, .001, .001, .001]
    # for iter in range(4):
    #     neural_net = MLNN(data, X_train, Y_train)
    #     params = GDparams(n_batch=100, eta=etas[iter], n_epochs=40, lamda = lamdas[iter])
    #     print("MiniBatch Training with n_batch=100, n_epochs=40, eta=", etas[iter], ", lamda=", lamdas[iter])
    #     neural_net.MiniBatchGD(X_train, Y_train, y_train, params, neural_net.W, neural_net.b)
    #     GeneratePlots(neural_net, params)



