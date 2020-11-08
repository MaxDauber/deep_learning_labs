#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se
"""
This is the main file of Assignment 4 for DD2424 Deep Learning
This assignment implements a text-generating RNN based on the Harry Potter Books.
"""


import pickle
import statistics
import unittest
from math import pow
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rnn import RNN


class DataObject:

    def __init__(self, file):
        self.data_string = open(file, 'r').read()
        self.book_data = list(self.data_string)

        # unique characters in string of text
        self.book_chars = list(set(self.data_string))
        self.vocab_len = len(self.book_chars)


        # dict mapping characters to ints
        self.char_to_ind= dict()
        for char in self.book_chars:
            self.char_to_ind[char] = len(self.char_to_ind)

        # dict mapping ints back to characters
        self.ind_to_char = dict(zip(self.char_to_ind.values(), self.char_to_ind.keys()))

    def char_to_int(self, char_list):
        return [self.char_to_ind[char] for char in char_list]

    def int_to_char(self, int_list):
        return [self.ind_to_char[int] for int in int_list]

    def onehot_to_string(self, one_hot_seq):
        gen_ints = [np.where(r == 1)[0][0] for r in one_hot_seq]
        gen_char_list = self.int_to_char(gen_ints)
        return ''.join(gen_char_list)

    def chars_to_onehot(self, char_list):
        int_list = self.char_to_int(char_list)
        one_hot = np.zeros((len(self.book_chars), len(int_list)))
        for i, int_elem in enumerate(int_list):
            one_hot[int_elem, i] = 1
        return one_hot


if __name__ == '__main__':
    np.random.seed(0)
    data = DataObject("goblet_book.txt")

    e = 0 # Book position tracker
    n = 0 # Iteration number
    epoch = 0
    num_epochs = 10
    rnn = RNN(data)


    while epoch < num_epochs:
        if n == 0 or e >= (len(rnn.book_data) - rnn.seq_length - 1):
            if epoch != 0: print("Finished %i epochs." % epoch)
            hprev = np.zeros((rnn.m, 1))
            e = 0
            epoch += 1

        inputs = [rnn.char_to_ind[char] for char in rnn.book_data[e:e + rnn.seq_length]]
        targets = [rnn.char_to_ind[char] for char in rnn.book_data[e + 1:e + rnn.seq_length + 1]]

        gradients, loss, hprev = rnn.ComputeGradients(inputs, targets, hprev)

        # Compute smooth loss
        if n == 0 and epoch == 1: smooth_loss = loss
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss

        # Check gradients
        if n == 0: rnn.check_gradients(inputs, targets, hprev)

        # Print the loss
        if n % 100 == 0: print('Iteration %d, smooth loss: %f' % (n, smooth_loss))

        # Print synthesized text
        if n % 500 == 0:
            txt = rnn.synthesize_text(hprev, inputs[0], 200)
            print('\nSynthesized text after %i iterations:\n %s\n' % (n, txt))
            print('Smooth loss: %f' % smooth_loss)

        # AdaGrad Update
        for key in rnn.params:
            rnn.adagrad_params[key] += gradients[key] * gradients[key]
            rnn.params[key] -= rnn.eta / np.sqrt(rnn.adagrad_params[key] + np.finfo(float).eps) * gradients[key]

        e += rnn.N
        n += 1
    # rnn.CheckGradients(data)
    # rnn.TrainModel(data)






