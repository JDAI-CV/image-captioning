import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.Wah = nn.Linear(cfg.MODEL.RNN_SIZE, cfg.MODEL.ATT_HIDDEN_SIZE, bias=False)
        self.alpha = nn.Linear(cfg.MODEL.ATT_HIDDEN_SIZE, 1, bias=False)
        self.dropout = nn.Dropout(cfg.MODEL.ATT_HIDDEN_DROP) if cfg.MODEL.ATT_HIDDEN_DROP > 0 else None

        if cfg.MODEL.ATT_ACT == 'RELU':
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()

    # h -- batch_size * cfg.MODEL.RNN_SIZE
    # att_feats -- batch_size * att_num * att_feats_dim
    # p_att_feats -- batch_size * att_num * cfg.ATT_HIDDEN_SIZE
    def forward(self, h, att_feats, p_att_feats):
        Wah = self.Wah(h).unsqueeze(1)
        alpha = self.act(Wah + p_att_feats)
        if self.dropout is not None:
            alpha = self.dropout(alpha)
        alpha = self.alpha(alpha).squeeze(-1)
        alpha = F.softmax(alpha, dim=-1)
        att = torch.bmm(alpha.unsqueeze(1), att_feats).squeeze(1)
        return att