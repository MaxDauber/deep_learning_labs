import pickle
import statistics
import unittest
import random

import matplotlib
import numpy as np
# import scipy


def softmax(x):
    """
        Standard definition of the softmax function
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
    """
        Load Batch for dataset
    """
    with open(filename, 'rb') as f:
        dataDict = pickle.load(f, encoding='bytes')

        X = (dataDict[b"data"] / 255).T
        y = dataDict[b"labels"]
        Y = (np.eye(10)[y]).T

    return X, Y, y

def EvaluateClassifier(X, W, b):
    """
        output probabilities of the classifier
    """
    return softmax(np.matmul(W, X) + b)


def ComputeAccuracy(X, y, W, b):
    """
       Computes the accuracy of the network’s predictions
    """
    acc = 0
    return acc

def ComputeCost(X, Y, W, b, lamda):
    """
        Computes the cost function for the set of images using cross entropy loss
    """
    d = np.shape(X)[0]
    N = np.shape(X)[1]

    regularization = lamda * np.sum(np.power(W, 2))

    J = (1/N) * np.sum(loss_term) + regularization_term
    return J

def ComputeGradsNum(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    c = ComputeCost(X, Y, W, b, lamda);

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i,j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i,j] = (c2-c) / h

    return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2-c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i,j] -= h
            c1 = ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i,j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i,j] = (c2-c1) / (2*h)

    return [grad_W, grad_b]

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

# def save_as_mat(data, name="model"):
#     """ Used to transfer a python model to matlab """
#     import scipy.io as sio
#     sio.savemat(name'.mat', {name: b})


if __name__ == '__main__':

    # N = num of input examples
    # d = dimension of each input example
    # X = images (d x N)
    # Y = one-hot labels (K x N)
    # y = labels (N)
    X_train, Y_train, y_train = LoadBatch("Datasets/cifar-10-batches-py/data_batch_1")
    X_validation, Y_validation, y_validation = LoadBatch("Datasets/cifar-10-batches-py/data_batch_2")
    X_test, Y_test, y_test = LoadBatch("Datasets/cifar-10-batches-py/test_batch")

    # Hyperparameters -> change for each version of training

    # lamda = regularization parameter
    lamda = 0
    # eta = learning rate
    eta = 0.1
    n_epochs = 40



    # K =num of labels
    K = 10

    # dim of each image
    d = np.shape(X_train)[0]

    # num of images
    N = np.shape(X_train)[1]

    # b = bias K x 1
    b = np.random.normal(0, 0.01, (K, 1))
    print(b)

    # W = weights K x d
    W = np.random.normal(0, 0.01, (K, d))