#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se
"""
This is the main file of Assignment 1 for DD2424 Deep Learning
This assignment implements a one-layer neural network.
"""


import pickle
import statistics
import unittest
import random
# import matplotlib
import numpy as np
# import scipy
from olnn import OLNN, GDparams

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

        X = (dataDict[b"data"] / 255).T
        y = dataDict[b"labels"]
        Y = (np.eye(10)[y]).T

    return X, Y, y

def montage(W):
    """ Display the image for each label in W """
    fig, ax = matplotlib.plots.subplots(2,5)
    for i in range(2):
        for j in range(5):
            im  = W[i+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    matplotlib.plots.show()



if __name__ == '__main__':
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    np.random.seed(7)

    # N = num of input examples
    # d = dimension of each input example
    # X = images (d x N)
    # Y = one-hot labels (K x N)
    # y = labels (N)
    X_train, Y_train, y_train = LoadBatch("Datasets/cifar-10-batches-py/data_batch_1")
    X_validation, Y_validation, y_validation = LoadBatch("Datasets/cifar-10-batches-py/data_batch_2")
    X_test, Y_test, y_test = LoadBatch("Datasets/cifar-10-batches-py/test_batch")

    neural_net = OLNN(X_train[:2, :100], Y_train[:, :100])
    # ann1.check_gradients(X_train[:2, :100], Y_train[:, :100], method='finite_diff')
    neural_net.CheckGradients(X_train[:2, :100], Y_train[:, :100])
    params = GDparams()
    neural_net.MiniBatchGD(X_train[:2, :100], Y_train[:, :100], params, lamda=1)



    # meanx = np.mean(X_train, axis=0)
    # stdx = np.std(X_train, axis=0)
    #
    # X_train -= meanx
    # X_train /= stdx
    # X_validation -= meanx
    # X_validation /= stdx
    # X_test -= meanx
    # X_test /= stdx
