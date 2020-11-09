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
                X: data matrix

            Returns:
                Softmax transformed data

        """
        # ones = np.ones(Y.shape[0])
        # return np.exp(Y) / np.dot(ones.T, np.exp(Y))
        return np.exp(Y) / np.sum(np.exp(Y), axis=0)

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
        # The next input vector
        xnext = np.zeros((self.data.vocab_length, 1))
        # Use the index to set the net input vector
        xnext[input] = 1  # 1-hot-encoding
        txt = ''

        for t in range(n):
            _, h, _, p = self.Evaluate(hidden_prev, xnext)

            probs = np.ndarray.flatten(p)
            input = np.random.choice(range(self.data.vocab_length), p=probs)
            xnext = np.zeros((self.data.vocab_length, 1))
            xnext[input] = 1
            txt += self.data.ind_to_char[input]

        return txt


    def CheckGrads(self, inputs, targets, hidden_prev, num_comps=20):
        """
        Checks analytically computed gradients against numerically computed to determine margin of error
        """
        grads_ana, _, _ = self.ComputeGrads(inputs, targets, hidden_prev)
        grads_num = self.ComputeGradsNum(inputs, targets, hidden_prev, 1e-5)

        print("Gradient checks:")
        for grad in grads_ana:
            num = abs(grads_ana[grad].flat[:num_comps] -
                      grads_num[grad].flat[:num_comps])
            denom = np.asarray([max(abs(a), abs(b)) + 1e-10 for a, b in
                                zip(grads_ana[grad].flat[:num_comps],
                                    grads_num[grad].flat[:num_comps])
                                ])
            max_rel_error = max(num / denom)

            print("The maximum relative error for the %s gradient is: %e." %
                  (grad, max_rel_error))
        print()

    def ComputeGrads(self, inputs, targets, hprev):
        """ Analytically computes the gradients of the weight and bias parameters
        Args:
            inputs      (list): indices of the chars of the input sequence
            targets     (list): indices of the chars of the target sequence
            hprev (np.ndarray): previous learnt hidden state sequence
        Returns:
            grads (dict): the updated analytical gradients dU, dW, dV, db and dc
            loss (float): the current loss
            h (np.ndarray): newly learnt hidden state sequence
        """
        n = len(inputs)
        loss = 0

        # Dictionaries for storing values during the forward pass
        aa, xx, hh, oo, pp = {}, {}, {}, {}, {}
        hh[-1] = np.copy(hprev)

        # Forward pass
        for t in range(n):
            xx[t] = np.zeros((self.data.vocab_length, 1))
            xx[t][inputs[t]] = 1  # 1-hot-encoding

            aa[t], hh[t], oo[t], pp[t] = self.Evaluate(hh[t - 1], xx[t])

            loss += -np.log(pp[t][targets[t]][0])  # update the loss

        # Dictionary for storing the gradients
        grads = {key: np.zeros_like(self.params[key]) for key in self.params.keys()}
        o = np.zeros_like(pp[0])
        h = np.zeros_like(hh[0])
        h_next = np.zeros_like(hh[0])
        a =  np.zeros_like(aa[0])

        # Backward pass
        for t in reversed(range(n)):
            o = np.copy(pp[t])
            o[targets[t]] -= 1

            grads["v"] += o @ hh[t].T
            grads["c"] += o

            h = self.params['v'].T @ o + h_next
            a = np.multiply(h, (1 - np.square(hh[t])))

            grads["u"] += a @ xx[t].T
            grads["w"] += a @ hh[t - 1].T
            grads["b"] += a

            h_next = self.params['w'].T @ a

        # Clip the gradients
        for grad in grads:
            grads[grad] = np.clip(grads[grad], -5, 5)

        # Update the hidden state sequence
        h = hh[n - 1]

        return grads, loss, h

    def ComputeGradsNum(self, inputs, targets, hprev, h, num_comparisons=20):
        """
        Numerically computes the gradients of the weight and bias parameters
        """
        rnn_params = self.params
        num_grads = {key: np.zeros_like(self.params[key]) for key in self.params.keys()}

        for key in rnn_params:
            for i in range(num_comparisons):
                old_par = rnn_params[key].flat[i]  # store old parameter
                rnn_params[key].flat[i] = old_par + h
                _, l1, _ = self.ComputeGrads(inputs, targets, hprev)
                rnn_params[key].flat[i] = old_par - h
                _, l2, _ = self.ComputeGrads(inputs, targets, hprev)
                rnn_params[key].flat[i] = old_par  # reset parameter to old value
                num_grads[key].flat[i] = (l1 - l2) / (2 * h)

        return num_grads
