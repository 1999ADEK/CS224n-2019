#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, word_embed_dim):
        super(Highway, self).__init__()
        
        self.proj = nn.Linear(word_embed_dim, word_embed_dim)
        self.gate = nn.Linear(word_embed_dim, word_embed_dim)
    
    def forward(self, x_convout):
        x_proj = F.relu(self.proj(x_convout))
        x_gate = torch.sigmoid(self.gate(x_convout))
        return x_gate * x_proj + (1 - x_gate) * x_convout

### END YOUR CODE 

