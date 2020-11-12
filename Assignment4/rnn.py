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
        self.eta = 0.01
        self.seq_length = 25

        self.params, self.adagrad_params = self.InitParams()
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

        return params, adagrad_params



    def SoftMax(self, Y):
        """
            Standard definition of the softmax function

            Args:
                Y: data matrix

            Returns:
                Softmax transformed data

        """
        ones = np.ones(Y.shape[0])
        return np.exp(Y) / np.dot(ones.T, np.exp(Y))

    def Evaluate(self, h_curr, x_curr):
        """
        Output stable softmax probabilities of the classifier
            
        Args:
            h_curr: hidden state
            x_curr: sequence of input vectors

        Returns:
            a: linear transformation of w and u, bias b
            h: intermediate tanh values
            o: linear transformation of v,  bias c
            p: a stable softmax matrix

        """

        a = np.dot(self.params['w'], h_curr) + np.dot(self.params['u'], x_curr) + self.params['b']
        h = np.tanh(a)
        o = np.dot(self.params['v'], h) + self.params['c']
        p = self.SoftMax(o)

        return a, h, o, p

    def SynthesizeText(self, hidden_prev, input, n):
        """
        Synthesize text based on the hidden state sequence
        """
        xnext = np.zeros((self.data.vocab_length, 1))
        xnext[input] = 1
        txt = ''

        for t in range(n):
            _, h, _, p = self.Evaluate(hidden_prev, xnext)

            input = np.random.choice(range(self.data.vocab_length), p=p.flat)
            xnext = np.zeros((self.data.vocab_length, 1))
            xnext[input] = 1
            txt += self.data.ind_to_char[input]

        return txt


    def CheckGrads(self, inputs, targets, hidden_prev):
        """
        Checks analytically computed gradients against numerically computed to determine margin of error
        """
        print("Gradient checks:")

        analytical_gradients, _, _ = self.ComputeGrads(inputs, targets, hidden_prev)
        numerical_gradients = self.ComputeGradsNum(inputs, targets, hidden_prev)

        comparisons = 100
        for grad in analytical_gradients:
            numerator = abs(analytical_gradients[grad].flat[:comparisons] -
                      numerical_gradients[grad].flat[:comparisons])
            denominator = np.asarray([max(abs(a), abs(b)) + 1e-10 for a, b in
                                zip(analytical_gradients[grad].flat[:comparisons],
                                    numerical_gradients[grad].flat[:comparisons])
                                ])
            max_rel_error = max(numerator / denominator)

            print("The maximum relative error for the %s gradient is: %e." %
                  (grad, max_rel_error))
        print()

    def ComputeGrads(self, inputs, targets, hprev):
        """ Analytically computes the gradients of the weight and bias parameters
        Args:
            inputs: indices of the chars of the input sequence
            targets     (list): indices of the chars of the target sequence
            hprev (np.ndarray): previous learnt hidden state sequence
        Returns:
            grads (dict): the updated analytical gradients dU, dW, dV, db and dc
            loss (float): the current loss
            h (np.ndarray): newly learnt hidden state sequence
        """

        x_, a_, h_, o_, p_ = {}, {}, {}, {}, {}
        n = len(inputs)
        loss = 0
        h_[-1] = np.copy(hprev)

        # Forward pass
        for t in range(n):
            x_[t] = np.zeros((self.data.vocab_length, 1))
            x_[t][inputs[t]] = 1

            a_[t], h_[t], o_[t], p_[t] = self.Evaluate(h_[t - 1], x_[t])

            loss += -np.log(p_[t][targets[t]][0])  # update the loss

        grads = {key: np.zeros_like(self.params[key]) for key in self.params.keys()}
        o = np.zeros_like(p_[0])
        h = np.zeros_like(h_[0])
        h_next = np.zeros_like(h_[0])
        a = np.zeros_like(a_[0])

        # Backward pass
        for t in range(n-1, -1, -1):
            o = np.copy(p_[t])
            o[targets[t]] -= 1

            grads["v"] += np.dot(o, h_[t].T)
            grads["c"] += o

            h = np.dot(self.params['v'].T, o) + h_next
            a = np.multiply(h, (1 - h_[t] ** 2))

            grads["u"] += np.dot(a, x_[t].T)
            grads["w"] += np.dot(a, h_[t - 1].T)
            grads["b"] += a

            h_next = np.dot(self.params['w'].T,  a)

        for grad in grads:
            grads[grad] = np.clip(grads[grad], -5, 5)

        h = h_[n - 1]

        return grads, loss, h

    def ComputeGradsNum(self, inputs, targets, hprev):
        rnn_params = self.params
        num_grads = {key: np.zeros_like(self.params[key]) for key in self.params.keys()}

        for key in rnn_params:
            for i in range(20):
                prev_param = rnn_params[key].flat[i]
                rnn_params[key].flat[i] = prev_param + 1e-5
                _, l1, _ = self.ComputeGrads(inputs, targets, hprev)
                rnn_params[key].flat[i] = prev_param - 1e-5
                _, l2, _ = self.ComputeGrads(inputs, targets, hprev)
                rnn_params[key].flat[i] = prev_param
                num_grads[key].flat[i] = (l1 - l2) / (2 * 1e-5)

        return num_grads
