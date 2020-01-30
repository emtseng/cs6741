"""
NaiveBayes
Author: Emily Tseng (et397)

--

This implements Multinomial Naive Bayes per Wang & Manning 2012 (https://www.aclweb.org/anthology/P12-2018.pdf), section 2.1.
    y = wx + b
    w = r = log( (p/l1_norm(p)) / (q/l1_norm(q)) )
    p = alpha + sum of all positive training cases
    q = alpha + sum of all negative training cases
    alpha = smoothing parameter (e.g. 1.0)
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
        # Store which label is which
        self.label_map = {
            self.labels.itos[0]: 0,
            self.labels.itos[1]: 1
        }
        # Initialize with zero seen pos or neg samples
        self.p = self.alpha + np.zeros((len(self.vocab),))
        self.q = self.alpha + np.zeros((len(self.vocab),))
        # Initialize counts at 1 here also to prevent div by 0 error
        self.nplus = 1
        self.nminus = 1
        self.update()
        print("Initialized NaiveBayes model with vocab size {}, label size {}".format(len(self.vocab), len(self.labels)))
        print('\tself.p: {}\n\tself.q: {}\n\tself.r: {}\n\tself.b: {}'.format(self.p, self.q, self.r, self.b))
    
    def update(self):
        self.r = np.log((self.p/np.linalg.norm(self.p, ord=1)) / (self.q/np.linalg.norm(self.q, ord=1)))
        self.b = np.log(self.nplus / self.nminus)

    def featurize(self, x):
        """
            Input: <vec> x
            Output: <vec> fx, featurized using the vocabulary
        """
        output = np.zeros((len(self.vocab),))
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
                  self.nplus += 1
                else:
                  self.q += fx
                  self.nminus += 1
            # And recalculate r & b
            self.update()
            # Let's hope to see improvement at every 10 batches
            if batch_idx % 10 == 0:
              batch_acc = self.evaluate(val_iter)
              print('val acc after training batch {}: {}'.format(batch_idx, batch_acc))
              # print('\tself.p: {}\n\tself.q: {}\n\tself.r: {}\n\tself.b: {}'.format(self.p, self.q, self.r, self.b))


    def evaluate(self, val_iter):
        """
            Evaluates against a batch of given data.
        """
        correct = 0
        total = 0

        for batch_idx, val_batch in enumerate(val_iter):
            for i in range(len(val_batch)):
                x = val_batch.text[:, i]
                y = val_batch.label[i]
                fx = self.featurize(x)
                yhat = self.predict(fx)
                if y == yhat:
                    correct += 1
                # else:
                    # print('\tgot this wrong: {} shouldve been {}'.format(yhat, y))
                total += 1

        return float(correct / total)

    def predict(self, x):
        val = np.matmul(self.r.T, x) + self.b
        if val >= 0:
            return self.label_map['positive']
        else:
            return self.label_map['negative']