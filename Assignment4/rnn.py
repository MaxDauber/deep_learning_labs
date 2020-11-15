#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se

import numpy as np

class RNN:

    def __init__(self, data):
        self.data = data
        self.m = 100
        self.k = self.data.num_chars
        self.eta = 0.01
        self.seq_length = 25
        self.params, self.adagrad_params = self.InitParams()

    def InitParams(self):
        params = {}
        self.sig = 0.01
        params['b'] = np.zeros((self.m, 1))
        params['c'] = np.zeros((self.k, 1))
        params['U'] = np.random.normal(0, self.sig, size=(self.m, self.k))
        params['W'] = np.random.normal(0, self.sig, size=(self.m, self.m))
        params['V'] = np.random.normal(0, self.sig, size=(self.k, self.m))

        adagrad_params = {key: np.zeros_like(params[key]) for key in params.keys()}

        return params, adagrad_params



    def Evaluate(self, h_curr, x_curr):
        a = np.dot(self.params['W'], h_curr) + np.dot(self.params['U'], x_curr) + self.params['b']
        h = np.tanh(a)
        o = np.dot(self.params['V'], h) + self.params['c']
        p = np.exp(o) / np.sum(np.exp(o), axis=0) # softmax
        return a, h, o, p

    def Generate(self, hidden_prev, input, text_len):
        next_chars = np.zeros((self.data.num_chars, 1))
        next_chars[input] = 1
        txt = ''
        for t in range(text_len):
            _, _, _, p = self.Evaluate(hidden_prev, next_chars)
            input = np.random.choice(range(self.data.num_chars), p=p.flat)
            next_chars = np.zeros((self.data.num_chars, 1))
            next_chars[input] = 1
            txt += self.data.ind_to_char[input]

        return txt

    def ComputeGrads(self, inputs, targets, hidden_state):
        """Computes the gradients of the parameters of the network by performing a forward and backward pass.
        Arguments:
            inputs {ndarray} -- The one-hot encoded input sequence
            targets {ndarray} -- The target one-hot encoded sequence
            hidden_state {ndarray} -- The current hidden state of the network
        Returns:
            grads [dict] -- The gradients of the network parameters stored as a dictionary
            loss [float] -- The current loss
            h [ndarray] -- The updated state of the hidden layer
        """
        # forward pass
        n = inputs.shape[1]
        p = np.zeros((self.k, n))
        h = np.zeros((self.m, n + 1))

        h[:, 0] = hidden_state
        for t in range(n):
            _, h[:, t + 1], _, p[:, t] = self.Evaluate(h[:, t], inputs[:, t])
            loss = -sum(np.log(np.multiply(targets, p).sum(axis=0)))

            # backward pass
            grads = {'U': np.zeros(self.params['U'].shape), 'V': np.zeros(self.params['V'].shape),
                     'W': np.zeros(self.params['W'].shape), 'b': np.zeros(self.params['b'].shape),
                     'c': np.zeros(self.params['c'].shape)}

            do = -(targets - p).T

            grads['V'] = np.dot(do.T, h[:, 1:].T)
            grads['c'] = do.sum(axis=0)

            dh, da = np.zeros((self.m, n)), np.zeros((self.m, n))

            dh[:, -1] = np.dot(do.T[:, -1], self.params['V'])
            da[:, -1] = np.multiply(dh[:, -1], (1 - h[:, -1] ** 2))

            for t in reversed(range(n - 1)):
                dh[:, t] = np.dot(do[t, :], self.params['V']) + \
                           np.dot(da[:, t + 1], self.params['W'])
                da[:, t] = np.multiply(dh[:, t], (1 - h[:, t + 1] ** 2))

            grads['U'] = np.dot(da, inputs.T)
            grads['W'] = np.dot(da, h[:, :-1].T)
            grads['b'] = da.sum(axis=1)

        return grads, loss, h[:, -1]


    # def ComputeGrads(self, inputs, targets, hidden_prev):
    #     x_, a_, h_, o_, p_ = {}, {}, {}, {}, {}
    #     n = len(inputs)
    #     loss = 0
    #     h_[-1] = np.copy(hidden_prev)
    #
    #     # Forward pass
    #     for t in range(n):
    #         x_[t] = np.zeros((self.data.num_chars, 1))
    #         x_[t][inputs[t]] = 1
    #
    #         a_[t], h_[t], o_[t], p_[t] = self.Evaluate(h_[t - 1], x_[t])
    #
    #         loss += -np.log(p_[t][targets[t]][0])
    #
    #     grads = {key: np.zeros_like(self.params[key]) for key in self.params.keys()}
    #     o = np.zeros_like(p_[0])
    #     h = np.zeros_like(h_[0])
    #     h_next = np.zeros_like(h_[0])
    #     a = np.zeros_like(a_[0])
    #
    #     # Backward pass
    #     for t in range(n-1, -1, -1):
    #         o = np.copy(p_[t])
    #         o[targets[t]] -= 1
    #
    #         grads["v"] += np.dot(o, h_[t].T)
    #         grads["c"] += o
    #
    #         h = np.dot(self.params['v'].T, o) + h_next
    #         a = np.multiply(h, (1 - h_[t] ** 2))
    #
    #         grads["u"] += np.dot(a, x_[t].T)
    #         grads["w"] += np.dot(a, h_[t - 1].T)
    #         grads["b"] += a
    #
    #         h_next = np.dot(self.params['w'].T,  a)
    #
    #     for grad in grads:
    #         grads[grad] = np.clip(grads[grad], -5, 5)
    #
    #     h = h_[n - 1]
    #
    #     return grads, loss, h


    def CheckGrads(self, inputs, targets, hidden_prev):
        print("Running Gradient Checks")
        analytical_gradients, _, _ = self.ComputeGrads(inputs, targets, hidden_prev)
        numerical_gradients = self.ComputeGradsNum(inputs, targets, hidden_prev)

        comparisons = 20
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


