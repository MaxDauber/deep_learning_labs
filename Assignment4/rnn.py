#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class RNN:

    def __init__(self, data):
        self.m = 100
        self.k = 80
        self.eta = 0.1
        self.seq_length = 25
        self.e = 0
        self.eps = 1e-8

        self.params , self.adagrad_params = self.InitParams()
        self.grads = {}
        self.data = data

    def InitParams(self):
        """
            Initializes parameters for network

            Args:
                None

            Returns:
                params
        """

        params = {}
        self.sig = 0.01
        params['b'] = np.zeros((self.m, 1))  # bias vector
        params['c'] = np.zeros((self.k, 1))  # another bias vector
        params['u'] = np.random.rand(self.m, self.k) * self.sig  # weight matrix 1
        params['w'] = np.random.rand(self.m, self.m) * self.sig  # weight matrix 2
        params['v'] = np.random.rand(self.k, self.m) * self.sig  # weight matrix 3

        adagrad_params = {}
        adagrad_params['b'] = np.zeros_like(params['b'])
        adagrad_params['c'] = np.zeros_like(params['c'])
        adagrad_params['u'] = np.zeros_like(params['u'])
        adagrad_params['w'] = np.zeros_like(params['w'])
        adagrad_params['v'] = np.zeros_like(params['v'])

        return params


    def TrainModel(self, data, epochs=5, seq_length=100):
        """
            Trains model using the book data

            Args:
                data:   DataObject containing book information
                epochs: Number of epochs to train for

            Returns:
                N/A
        """
        book_length = len(data.book_data)
        smooth_loss = None
        for i in range(epochs):
            X, Y = self.GetMatrices(data)
            p, a, h = self.Evaluate(X)
            loss = self.ComputeLoss(p, Y)
            smooth_loss = self.ComputeSmoothLoss(loss, smooth_loss)
            self.ComputeGradients(X, Y, p, a, h)
            self.AdaGrad()
            self.report_progress(smooth_loss)

            # Update the counter of where we are in the book (e).
            # If we are at the end of the book, we start at the beginning again.
            new_e = self.e + self.seq_length
            if new_e > (book_length - self.seq_length - 1):
                new_e = 0
            self.e = new_e

            if i % 1000 == 0:
                print("update ", i, " // smooth loss: ", round(smooth_loss, 3))
            if i % 10000 == 0:
                one_hot = self.GenerateSequence(data, seq_length=seq_length)
                print("-------------- generated text: -------------------------------")
                print(data.onehot_to_string(one_hot))
                print("--------------------------------------------------------------")

    def GetMatrices(self, data):
        """
        get X (input) and Y (labels) matrices from the list of characters
        """
        X_chars = data.book_data[self.e: self.e + self.seq_length]
        Y_chars = data.book_data[self.e + 1: self.e + self.seq_length + 1]
        X = data.chars_to_onehot(X_chars)
        Y = data.chars_to_onehot(Y_chars)
        print(np.shape(X))
        print(np.shape(Y))
        return X, Y

    def Evaluate(self, X):
        """
        evaluates a sequence of one-hot encoded characters X and outputs a
        probability vector at each X_t representing the predicted probs
        for the next character.
        Used as forward pass of the backpropagation through time (bptt)
        """

        p = np.zeros((X.shape[1], self.k))
        a = np.zeros((X.shape[1], self.m))
        h = np.zeros((X.shape[1], self.m))

        for t in range(X.shape[1]):
            xt = X[:, t].reshape((self.k, 1))  # reshape from (k,) to (k,1)
            a_curr = np.dot(self.params['w'], self.h_prev) + np.dot(self.params['u'], xt) + self.params['b']
            h_curr = np.tanh(a_curr)
            o_curr = np.dot(self.params['v'], h_curr) + self.params['c']
            p_curr = self.SoftMax(o_curr)

            a[t] = a_curr.reshape(self.m)  # reshape from (m,1) to (m,)
            h[t] = h_curr.reshape(self.m)  # reshape from (m,1) to (m,)
            p[t] = p_curr.reshape(self.k)  # reshape from (k,1) to (k,)

            self.h_prev = h_curr

        return p, a, h

    def ComputeLoss(self, p, Y):
        """
        Compute the cross entropy loss between:
        - a (seq_len x k) matrix of predicted probabilities
        - a (k x seq _len) matrix of true one_hot encoded characters

        """
        loss = 0
        for t in range(self.seq_length):
            yt = Y[:, t]
            loss += -np.log(np.dot(yt.T, p[t]))

        return loss

    def ComputeSmoothLoss(self, loss, smooth_loss):
        if smooth_loss == None:
            smooth_loss = loss
        else:
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        return smooth_loss

    def report_progress(self, smooth_loss):
        pass

    def SoftMax(self, Y_pred_lin):
        """
        compute softmax activation, used in evaluating the prediction of the model
        """
        ones = np.ones(Y_pred_lin.shape[0])
        return np.exp(Y_pred_lin) / np.dot(ones.T, np.exp(Y_pred_lin))


    def AdaGrad(self):
        """
            Updates according to AdaGrad method

            Args:
                None

            Returns:
                None
        """
        for key in self.params:
            param = self.params[key]
            grad = self.grads[key]
            Gt = self.cum_g2[key] + np.square(grad)
            eps = self.eps * np.ones(Gt.shape)
            updated_param = param - self.eta / (np.sqrt(Gt + eps)) * grad
            self.params[key] = updated_param
            self.cum_g2[key] = Gt

    def GenerateSequence(self, data):
        """
        generate a sequence of n one_hot encoded characters based on initial
        hidden state h0 and input character x0

        Return: n*k matrix of n generated chars encoded as one-hot vectors
        """
        rand = np.random.randint(10000)
        X_chars = data.book_data[rand: rand + 100]
        X = data.chars_to_onehot(X_chars)
        p, a, h = self.Evaluate(X)
        char_seq = np.array([self.SelectChar(pt) for pt in p])
        return char_seq

    def SelectChar(self, prob):
        """
        Use the conditional probabilities of a character
        to generate a one_hot character based on a prob-weighted random choice
        """
        # draw an int in [0,k]
        indices = list(range(self.k))
        int_draw = int(np.random.choice(indices, 1, p=prob))

        # convert int to one-hot
        one_hot_draw = np.zeros(self.k)
        one_hot_draw[int_draw] = 1
        return one_hot_draw

    # def CheckGradients(self, data, h_param=1e-7):
    #     X, Y = self.GetMatrices(data)
    #     p, a, h = self.Evaluate(X)
    #     self.ComputeGradients(X, Y, p, a, h)
    #     for key in self.grads:
    #         print("----------------------------------------------------------------")
    #         print("comparing numerical and own gradient for: " + str(key))
    #         print("----------------------------------------------------------------")
    #         num_grad = self.num_gradient(key, X, Y, h_param)
    #         own_grad = self.grads[key]
    #         error = np.sum(self.grads[key] - num_grad)
    #
    #         grad_w_vec = own_grad.flatten()
    #         grad_w_num_vec = num_grad.flatten()
    #         x_w = np.arange(1, grad_w_vec.shape[0] + 1)
    #         plt.figure(figsize=(9, 8), dpi=80, facecolor='w', edgecolor='k')
    #         plt.bar(x_w, grad_w_vec, 0.35, label='Analytical gradient', color='blue')
    #         plt.bar(x_w + 0.35, grad_w_num_vec, 0.35, label='numerical gradient', color='red')
    #         plt.legend()
    #         plt.title("Gradient check of: " + str(key))
    #         plt.show()
    #         rel_error = abs(grad_w_vec / grad_w_num_vec - 1)
    #         print("mean relative error: ", np.mean(rel_error))
    #
    # def CheckGradients(self, X, Y, lamda=0):
    #     """
    #         Checks analytically computed gradients against numerically computed to determine margin of error
    #
    #         Args:
    #             X: data batch matrix
    #             Y: one-hot-encoding labels batch vector
    #             method: type of numerical gradient computation
    #
    #         Returns:
    #             None
    #     """
    #     grad_w_numerical, grad_b_numerical = self.ComputeGradsNum(X, Y, self.W, self.b, lamda, 1e-5)
    #     grad_w_analytical, grad_b_analytical = self.ComputeGradients(X, Y, self.W, self.b, lamda)
    #     print("******* Performing Gradient Checks *******")
    #     for layer in range(self.num_hidden+1):
    #         grad_w_vec = grad_w_analytical[layer].flatten()
    #         grad_w_num_vec = grad_w_numerical[layer].flatten()
    #         grad_b_vec = grad_b_analytical[layer].flatten()
    #         grad_b_num_vec = grad_b_numerical[layer].flatten()
    #         print("* W gradients for layer", self.num_hidden - layer, " *")
    #         print("mean relative error: ", np.mean(abs((grad_w_vec + self.h_param ** 2) /
    #                                                    (grad_w_num_vec + self.h_param ** 2) - 1)))
    #         print("* b gradients for layer", self.num_hidden - layer, " *")
    #         print("mean relative error: ", np.mean(abs((grad_b_vec + self.h_param ** 2) /
    #                                                    (grad_b_num_vec + self.h_param ** 2) - 1)))
    #         x_w = np.arange(1, grad_w_vec.shape[0] + 1)
    #         plt.bar(x_w, grad_w_vec, 0.35, label='Analytical gradient', color='blue')
    #         plt.bar(x_w + 0.35, grad_w_num_vec, 0.35, label="fast", color='red')
    #         plt.legend()
    #         plt.title(("Gradient check of w", layer, ", batch size = " + str(X.shape[1])))
    #         plt.show()
    #
    #         x_b = np.arange(1, grad_b_analytical[layer].shape[0] + 1)
    #         plt.bar(x_b, grad_b_vec, 0.35, label='Analytical gradient', color='blue')
    #         plt.bar(x_b + 0.35, grad_b_num_vec, 0.35, label="fast", color='red')
    #         plt.legend()
    #         plt.title(("Gradient check of b", layer, ", batch size = " + str(X.shape[1])))
    #         plt.show()

    # def ComputeGradients(self, X, Y, p, a, h):
    #     """
    #         Computes the gradients of the weight and bias parameters using analytical method
    #
    #         Args:
    #             X: data batch matrix
    #             Y: one-hot-encoding labels batch vector
    #             P: evaluated classifier for the batch
    #             W: weights
    #             b: bias
    #             lamda: regularization term
    #
    #         Returns:
    #             gradient_W: gradient of the weight parameter
    #             gradient_b: gradient of the bias parameter
    #
    #     """
    #     grad_o = np.zeros((self.seq_length, self.k))
    #     for t in range(self.seq_length):
    #         yt = Y[:, t].reshape(self.k, 1)
    #         pt = p[t].reshape(self.k, 1)
    #         grad_o[t] = (-(yt - pt)).reshape(self.k)
    #
    #     grad_v = np.zeros((self.k, self.m))
    #     for t in range(self.seq_length):
    #         grad_v += np.dot(grad_o[t].reshape(self.k, 1), h[t].reshape(1, self.m))
    #
    #     grad_a = np.zeros((self.seq_length, self.m))
    #     grad_h = np.zeros((self.seq_length, self.m))
    #
    #     grad_h[-1] = np.dot(grad_o[-1], self.params['v'])
    #     grad_h_last = grad_h[-1]
    #     diag_part = np.diag(1 - np.tanh(a[-1]) ** 2)
    #     grad_a[-1] = np.dot(grad_h_last, diag_part)
    #
    #     for t in reversed(range(self.seq_length - 1)):
    #         grad_h[t] = np.dot(grad_o[t], self.params['v']) + np.dot(grad_a[t - 1], self.params['w'])
    #         grad_h_part = grad_h[t]
    #         diag_part = np.diag(1 - np.tanh(a[t]) ** 2)
    #         grad_a[t] = np.dot(grad_h_part, diag_part)
    #
    #     grad_c = grad_o.sum(axis=0).reshape(self.k, 1)
    #     grad_b = grad_a.sum(axis=0).reshape(self.m, 1)
    #
    #     grad_w = np.zeros((self.m, self.m))
    #     for t in range(self.seq_length):
    #         grad_w += np.outer(grad_a[t].reshape(self.m, 1), self.h_prev)
    #
    #     grad_u = np.zeros((self.m, self.k))
    #     for t in range(self.seq_length):
    #         xt = X[:, t].reshape(self.k, 1)
    #         grad_u += np.dot(grad_a[t].reshape(self.m, 1), xt.T)
    #
    #     self.grads['u'] = grad_u
    #     self.grads['v'] = grad_v
    #     self.grads['w'] = grad_w
    #     self.grads['b'] = grad_b
    #     self.grads['c'] = grad_c

    def compute_gradients_num(self, inputs, targets, hprev, h, num_comparisons=20):
        """
        Numerically computes the gradients of the weight and bias parameters
        """
        rnn_params = self.params
        num_grads = {key: np.zeros_like(self.params[key]) for key in self.params.keys()}

        for key in rnn_params:
            for i in range(num_comparisons):
                old_par = rnn_params[key].flat[i]  # store old parameter
                rnn_params[key].flat[i] = old_par + h
                _, l1, _ = self.compute_gradients(inputs, targets, hprev)
                rnn_params[key].flat[i] = old_par - h
                _, l2, _ = self.compute_gradients(inputs, targets, hprev)
                rnn_params[key].flat[i] = old_par  # reset parameter to old value
                num_grads[key].flat[i] = (l1 - l2) / (2 * h)

        return num_grads

    def num_gradient(self, key, X, Y, h_param):
        num_grad = np.zeros(self.grads[key].shape)
        if key == 'b' or key == 'c':  # need to loop over 1 dim
            for i in range(self.params[key].shape[0]):
                self.params[key][i] -= h_param
                p1, _, _ = self.Evaluate(X)
                l1 = self.ComputeLoss(p1, Y)
                self.params[key][i] += 2 * h_param
                p2, _, _ = self.Evaluate(X)
                l2 = self.ComputeLoss(p2, Y)
                num_grad[i] = (l2 - l1) / (2 * h_param)
                self.params[key][i] -= h_param
        else:  # need to loop over 2 dimensions
            for i in range(self.params[key].shape[0]):
                for j in range(self.params[key].shape[1]):
                    self.params[key][i][j] -= h_param
                    p1, _, _ = self.Evaluate(X)
                    l1 = self.ComputeLoss(p1, Y)
                    self.params[key][i][j] += 2 * h_param
                    p2, _, _ = self.Evaluate(X)
                    l2 = self.ComputeLoss(p2, Y)
                    num_grad[i][j] = (l2 - l1) / (2 * h_param)
                    self.params[key][i][j] -= h_param
        return num_grad
