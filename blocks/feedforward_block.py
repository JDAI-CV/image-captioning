import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils as utils

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, 
        relu_dropout, dropout):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.layer_norms = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.layer_norms(x)
        return x