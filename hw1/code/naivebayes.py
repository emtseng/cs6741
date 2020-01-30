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
    def __init__(self, alpha, TEXT, LABEL):
        """
            Initializes with a given smoothing parameter
        """
        self.alpha = alpha
        self.vocab = TEXT.vocab
        self.labels = LABEL.vocab
        # Initialize with zero seen pos or neg samples
        self.p = self.alpha + np.zeros((self.vocab,))
        self.q = self.alpha + np.zeros((self.vocab,))
        self.update()
    
    def update(self):
        self.r = np.log((self.p/self.p.sum()) / (self.q/self.q.sum()))
        self.b = np.log(len(self.p) / len(self.q))

    def featurize(self, x):
        """
            Input: <vec> x
            Output: <vec> fx, featurized using the vocabulary
        """
        output = np.zeros((self.vocab,))
        for word_idx in x:
            output[word_idx] = 1
        return output

    def train(self, train_iter, val_iter):
        """
            "Trains" the model based on the provided train and val sets.
        """
        for batch_idx, train_batch in enumerate(train_iter):
            # Update the count vectors...
            for i in range(len(train_batch)):
                x = train_batch.text[:, i]
                fx = self.featurize(x)
                y = train_batch.label[i]
                if self.labels.itos[y.item()] == 'positive':
                  self.p += fx
                else:
                  self.q += fx
            # And recalculate r & b
            self.update()
            # Let's hope to see improvement at each batch
            batch_acc = self.evaluate(val_iter)
            print('Batch {} acc: {}'.format(batch_idx, batch_acc))

    def evaluate(self, val_iter):
        """
            Evaluates against a batch of given data.
        """
        correct = 0
        total = 0
        for val_batch in val_iter:
            for i in range(len(val_batch)):
                x = val_batch.text[:, i]
                y = val_batch.label[i]
                fx = self.featurize(x)
                yhat = self.predict(fx)
                if y == yhat:
                    correct += 1
                total += 1
        return float(correct / total)

    def predict(self, x):
        val = np.matmul(self.r.T, x) + self.b
        if val > 0:
            return 1
        else:
            return -1

    # def test(self, test_batch):
    #     """
    #         Produces a distribution of probabilities for each sample in the provided batch.
    #         Note the output should have an attribute 'classes' to work with the provided scoring function.
    #     """
    #     for batch_idx, test_batch