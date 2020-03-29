import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils
from layers.basic_att import BasicAtt

class SCAtt(BasicAtt):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAtt, self).__init__(mid_dims, mid_dropout)
        self.attention_last = nn.Linear(mid_dims[-2], 1)
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att_mask_ext = att_mask.unsqueeze(-1)
            att_map_pool = (torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2))
        else:
            att_map_pool = att_map.mean(-2)

        alpha_spatial = self.attention_last(att_map)
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)

        alpha_spatial = alpha_spatial.squeeze(-1)
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)

        value2 = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)

        attn = value1 * value2 * alpha_channel
        return attn

    # att_map: batch_size * seq_num * att_num * dim
    # att_mask: batch_size * seq_num * att_num
    # value1: batch_size * seq_num * dim
    # value2: batch_size * att_num * dim
    def forward2(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att_mask_ext = att_mask.unsqueeze(-1)
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            att_map_pool = att_map.mean(-2)

        alpha_spatial = self.attention_last(att_map)        
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)  # batch_size * seq_num * dim

        alpha_spatial = alpha_spatial.squeeze(-1) # batch_size * seq_num * att_num
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)

        value2 = torch.matmul(alpha_spatial, value2)  # batch_size * seq_num * dim

        attn = value1 * value2 * alpha_channel
        return attn


