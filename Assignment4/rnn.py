#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se

import numpy as np

class RNN:

    def __init__(self, data):
        self.m = 100
        self.k = 80
        self.eta = 0.01
        self.seq_length = 25

        self.params, self.adagrad_params = self.InitParams()
        self.data = data

    def InitParams(self):
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
        ones = np.ones(Y.shape[0])
        return np.exp(Y) / np.dot(ones.T, np.exp(Y))

    def Evaluate(self, h_curr, x_curr):
        a = np.dot(self.params['w'], h_curr) + np.dot(self.params['u'], x_curr) + self.params['b']
        h = np.tanh(a)
        o = np.dot(self.params['v'], h) + self.params['c']
        p = self.SoftMax(o)
        return a, h, o, p

    def Generate(self, hidden_prev, input, n):
        char_next = np.zeros((self.data.num_chars, 1))
        char_next[input] = 1
        txt = ''

        for t in range(n):
            _, h, _, p = self.Evaluate(hidden_prev, char_next)

            input = np.random.choice(range(self.data.num_chars), p=p.flat)
            char_next = np.zeros((self.data.num_chars, 1))
            char_next[input] = 1
            txt += self.data.ind_to_char[input]

        return txt


    def ComputeGrads(self, inputs, targets, hidden_prev):
        x_, a_, h_, o_, p_ = {}, {}, {}, {}, {}
        n = len(inputs)
        loss = 0
        h_[-1] = np.copy(hidden_prev)

        # Forward pass
        for t in range(n):
            x_[t] = np.zeros((self.data.num_chars, 1))
            x_[t][inputs[t]] = 1

            a_[t], h_[t], o_[t], p_[t] = self.Evaluate(h_[t - 1], x_[t])

            loss += -np.log(p_[t][targets[t]][0])

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


    def CheckGrads(self, inputs, targets, hidden_prev):
        print("Running Gradient Checks")
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

            print("The max relative error for the %s gradient is: %e." %
                  (grad, max_rel_error))

    def ComputeGradsNum(self, inputs, targets, hprev):
        params = self.params
        numerical_grads = {key: np.zeros_like(self.params[key]) for key in self.params.keys()}
        for key in params:
            for i in range(20):
                prev_param = params[key].flat[i]
                params[key].flat[i] = prev_param + 1e-5
                _, loss1, _ = self.ComputeGrads(inputs, targets, hprev)
                params[key].flat[i] = prev_param - 1e-5
                _, loss2, _ = self.ComputeGrads(inputs, targets, hprev)
                params[key].flat[i] = prev_param
                numerical_grads[key].flat[i] = (loss1 - loss2) / (2 * 1e-5)
        return numerical_grads
