import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.low_rank import LowRank
from lib.config import cfg
from blocks.feedforward_block import FeedForwardBlock
import lib.utils as utils

class LowRankBilinearLayer(nn.Module):
    def __init__(
        self,         
        embed_dim, 
        att_type,
        att_heads,
        att_mid_dim,
        att_mid_drop,
        dropout
    ):
        super(LowRankBilinearLayer, self).__init__()
        self.encoder_attn = LowRank(
            embed_dim = embed_dim, 
            att_type = att_type, 
            att_heads = att_heads, 
            att_mid_dim = att_mid_dim, 
            att_mid_drop = att_mid_drop
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self, 
        x, 
        key=None, 
        mask=None, 
        value1=None, 
        value2=None, 
        precompute=False
    ):    
        x = self.encoder_attn(
            query=x,
            key=key if key is not None else x,
            mask=mask,
            value1=value1 if value1 is not None else x,
            value2=value2 if value2 is not None else x,
            precompute=precompute
        )
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def precompute(self, key, value2):
        return self.encoder_attn.precompute(key, value2)

class LowRankBilinearEncBlock(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        att_type,
        att_heads,
        att_mid_dim,
        att_mid_drop,
        dropout, 
        layer_num
    ):
        super(LowRankBilinearEncBlock, self).__init__()
        
        self.layers = nn.ModuleList([])
        self.bifeat_emb = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([]) 
        for _ in range(layer_num):
            sublayer = LowRankBilinearLayer( 
                embed_dim = embed_dim, 
                att_type = att_type,
                att_heads = att_heads,
                att_mid_dim = att_mid_dim,
                att_mid_drop = att_mid_drop,
                dropout = dropout
            )
            self.layers.append(sublayer)

            self.bifeat_emb.append(nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                utils.activation(cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT),
                nn.Dropout(cfg.MODEL.BILINEAR.ENCODE_BIFEAT_EMB_DROPOUT)
            ))

            self.layer_norms.append(torch.nn.LayerNorm(embed_dim))

        self.proj = nn.Linear(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(cfg.MODEL.BILINEAR.DIM)
        
    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None):
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
        
        feat_arr = [gv_feat]
        for i, layer in enumerate(self.layers):
            gv_feat = layer(gv_feat, att_feats, att_mask, gv_feat, att_feats)
            att_feats_cat = torch.cat([gv_feat.unsqueeze(1).expand_as(att_feats), att_feats], dim = -1)

            att_feats = self.bifeat_emb[i](att_feats_cat) + att_feats
            att_feats = self.layer_norms[i](att_feats)
            feat_arr.append(gv_feat)

        gv_feat = torch.cat(feat_arr, dim=-1)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        return gv_feat, att_feats

class LowRankBilinearDecBlock(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        att_type,
        att_heads,
        att_mid_dim,
        att_mid_drop,
        dropout, 
        layer_num
    ):
        super(LowRankBilinearDecBlock, self).__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(layer_num):
            sublayer = LowRankBilinearLayer( 
                embed_dim = embed_dim, 
                att_type = att_type,
                att_heads = att_heads,
                att_mid_dim = att_mid_dim,
                att_mid_drop = att_mid_drop,
                dropout = dropout
            )
            self.layers.append(sublayer)
        
        self.proj = nn.Linear(embed_dim * (layer_num + 1), embed_dim)
        self.layer_norm = torch.nn.LayerNorm(cfg.MODEL.BILINEAR.DIM)
        
    def precompute(self, key, value2):
        keys = []
        value2s = []
        for layer in self.layers:
            k, v = layer.precompute(key, value2)
            keys.append(k)
            value2s.append(v)
        return torch.cat(keys, dim=-1), torch.cat(value2s, dim=-1)

    def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None, precompute=False):
        if precompute == True:
            dim = p_att_feats.size()[-1]
            keys = p_att_feats.narrow(-1, 0, dim // 2)
            value2s = p_att_feats.narrow(-1, dim // 2, dim // 2)
            dim = keys.size()[-1] // len(self.layers)
    
        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = (torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1))
            else:
                gv_feat = torch.mean(att_feats, 1)

        feat_arr = [gv_feat]
        for i, layer in enumerate(self.layers):
            key = keys.narrow(-1, i * dim, dim) if precompute else att_feats
            value2 = value2s.narrow(-1, i * dim, dim) if precompute else att_feats
                            
            gv_feat = layer(gv_feat, key, att_mask, gv_feat, value2, precompute)
            feat_arr.append(gv_feat)

        gv_feat = torch.cat(feat_arr, dim=-1)
        gv_feat = self.proj(gv_feat)
        gv_feat = self.layer_norm(gv_feat)
        return gv_feat, att_feats
