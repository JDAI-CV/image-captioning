import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils

class BasicAtt(nn.Module):
    def __init__(self, mid_dims, mid_dropout):
        super(BasicAtt, self).__init__()

        sequential = []
        for i in range(1, len(mid_dims) - 1):
            sequential.append(nn.Linear(mid_dims[i - 1], mid_dims[i]))
            sequential.append(nn.ReLU())
            if mid_dropout > 0:
                sequential.append(nn.Dropout(mid_dropout))
        self.attention_basic = nn.Sequential(*sequential) if len(sequential) > 0 else None
        self.attention_last = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)
        attn_weights = self.attention_last(att_map)
        attn_weights = attn_weights.squeeze(-1)
        if att_mask is not None:
            attn_weights = attn_weights.masked_fill(att_mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn = torch.matmul(attn_weights.unsqueeze(-2), value2).squeeze(-2)
        return attn