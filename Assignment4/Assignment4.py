#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Max Dauber  dauber@kth.se
"""
This is the main file of Assignment 4 for DD2424 Deep Learning
This assignment implements a text-generating RNN based on the Harry Potter Books.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from rnn import RNN


class DataObject:

    def __init__(self, file):
        self.data_string = open(file, 'r').read()
        self.book_data = list(self.data_string)

        # unique characters in string of text
        self.book_chars = list(set(self.data_string))
        self.vocab_length = len(self.book_chars)


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
    np.random.seed(42)
    data = DataObject("goblet_book.txt")

    e = 0 # Book position tracker
    n = 0 # Iteration number
    epoch = 0
    num_epochs = 20
    rnn = RNN(data)
    loss_vals = []

    while epoch < num_epochs:
        if n == 0 or e >= (len(data.book_data) - rnn.seq_length - 1):
            if epoch != 0: print("Finished %i epochs." % epoch)
            hidden_prev = np.zeros((rnn.m, 1))
            e = 0
            epoch += 1

        inputs = [data.char_to_ind[char] for char in data.book_data[e:e + rnn.seq_length]]
        targets = [data.char_to_ind[char] for char in data.book_data[e + 1:e + rnn.seq_length + 1]]

        gradients, loss, hidden_prev = rnn.ComputeGrads(inputs, targets, hidden_prev)

        if epoch == 1 and n == 0 :
            smooth_loss = loss
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        loss_vals.append(smooth_loss)

        if n == 0:
            rnn.CheckGrads(inputs, targets, hidden_prev)

        if n % 10000 == 0:
            one_hot = rnn.SynthesizeText(hidden_prev, inputs[0], 200)
            print("-------------- generated text after %i iterations: ------------"% n)
            print(one_hot)
            print("--------------------------------------------------------------")
            print('Smooth loss: %f' % smooth_loss)
            # print(f"Progress: {'{:.2%}'.format(e/len(data.book_data))}")

        # AdaGrad Update
        for key in rnn.params:
            rnn.adagrad_params[key] += gradients[key] * gradients[key]
            rnn.params[key] -= rnn.eta / np.sqrt(rnn.adagrad_params[key] + np.finfo(float).eps) * gradients[key]

        e += rnn.seq_length
        n += 1

    txt = rnn.SynthesizeText(hidden_prev, inputs[0], 1000)
    print("-------------- generated text using best model ---------------")
    print(txt)
    print("--------------------------------------------------------------")

    fig = sns.lineplot(data=loss_vals, color='black')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title("loss over training run")
    plt.savefig("loss_plot.png")
    plt.show()











