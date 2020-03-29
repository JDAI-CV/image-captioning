import torch
import torch.nn as nn
import torch.nn.functional as F

from models.att_basic_model import AttBasicModel
from layers.attention import Attention
from lib.config import cfg
import lib.utils as utils

class UpDown(AttBasicModel):
    def __init__(self):
        super(UpDown, self).__init__()
        self.num_layers = 2

        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.WORD_EMBED_DIM + self.att_dim
        self.lstm1 = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        # Second LSTM Layer
        self.lstm2 = nn.LSTMCell(cfg.MODEL.RNN_SIZE + self.att_dim, cfg.MODEL.RNN_SIZE)
        self.att = Attention()

        if cfg.MODEL.BOTTOM_UP.DROPOUT_FIRST_INPUT > 0:
            self.dropout1 = nn.Dropout(cfg.MODEL.BOTTOM_UP.DROPOUT_FIRST_INPUT)
        else:
            self.dropout1 = None
            
        if cfg.MODEL.BOTTOM_UP.DROPOUT_SEC_INPUT > 0:
            self.dropout2 = nn.Dropout(cfg.MODEL.BOTTOM_UP.DROPOUT_SEC_INPUT)
        else:
            self.dropout2 = None

    # state[0] -- h, state[1] -- c
    def Forward(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        state = kwargs[cfg.PARAM.STATE]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.mean(att_feats, 1)
        xt = self.word_embed(wt)
        
        # lstm1
        h2_tm1 = state[0][-1]
        input1 = torch.cat([h2_tm1, gv_feat, xt], 1)
        if self.dropout1 is not None:
            input1 = self.dropout1(input1)
        h1_t, c1_t = self.lstm1(input1, (state[0][0], state[1][0]))
        att = self.att(h1_t, att_feats, p_att_feats)

        # lstm2
        input2 = torch.cat([att, h1_t], 1)
        if self.dropout2 is not None:
            input2 = self.dropout2(input2)
        h2_t, c2_t = self.lstm2(input2, (state[0][1], state[1][1]))

        state = (torch.stack([h1_t, h2_t]), torch.stack([c1_t, c2_t]))
        return h2_t, state



        

        
        