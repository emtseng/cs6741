"""
NaiveBayes
Author: Emily Tseng (et397)

This implementation works with the accompanying test.py.

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
            Trains the model over the given number of epochs.
        """
        # for epoch in tqdm(epochs):


    def test(self, batched_text):
        """
            Produces a distribution of probabilities for each sample in the test batch.
            Note the output should have an attribute 'classes' to work with the provided scoring function.
        """


