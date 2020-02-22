#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, char_embed_dim, num_filters):
        super(CNN, self).__init__()
        
        self.conv1D = nn.Conv1d(char_embed_dim, num_filters, kernel_size=5)
    
    def forward(self, x_reshaped):
        x_conv = self.conv1D(x_reshaped)
        return F.relu(x_conv).max(dim=-1)[0]

### END YOUR CODE

