"""
LogisticRegression
Author: Emily Tseng et397

--

This implements logistic regression.

"""

import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes, batch_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes, bias = True)
