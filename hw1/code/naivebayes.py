"""
NaiveBayes
Author: Emily Tseng (et397)

This implementation works with the accompanying test.py.

--

This implements Multinomial Naive Bayes per Wang & Manning 2012 (https://www.aclweb.org/anthology/P12-2018.pdf), section 2.1.
    y = wx + b
    w = r = log( (p/l1_norm(p)) / (q/l1_norm(q)) )
    alpha = smoothing parameter
    p = alpha + sum of all positive training cases
    q = alpha + sum of all negative training cases
    b = log(N+/N−), where:
        N+ = the number of positive cases in the training data
        N- = the number of negative cases in the training data
    Note this formulation featurizes using a binarized indicator:
        x(k) = f^(k) = 1 if f(k) > 0, 0 otherwise
        f(k) = feature count vector; each index is the number of occurrences of that feature in the given training case. e.g.
            "The cat in the hat."
            vocab = [the, cat, in, hat, banana]
            f = [ 2, 1, 1, 1, 0 ]
            f^ = [ 1, 1, 1, 1, 0 ]
"""

from tqdm import tqdm
import torch
import numpy as np

class NaiveBayes:
    def __init__(self, alpha):
        """
            Initializes with a given smoothing parameter
        """
        self.alpha = alpha


    def train(self, train_iter, val_iter, epochs):
        """
            Trains the model based on the provided train and val sets.
        """
        for epoch in tqdm(epochs):



    def test(self, batch):
        """
            Produces a distribution of probabilities for each sample in the provided batch.
            Note the output should have an attribute 'classes' to work with the provided scoring function.
        """


