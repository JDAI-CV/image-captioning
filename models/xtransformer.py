import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

from layers.low_rank import LowRank
import blocks
import lib.utils as utils
from models.basic_model import BasicModel
from layers.positional_encoding import PositionalEncoding
from .pretrained_models import ImageClassification
from mlclassifier import GCNClassifier
device = torch.device("cuda")

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class XTransformer(BasicModel):
    def __init__(self, args,submodel):
        super(XTransformer, self).__init__()
        self.vocab_size = cfg.MODEL.VOCAB_SIZE + 1
        # image pretrained
        self.image_pretrained_models, self.input_visual_feats = ImageClassification.image_features(
            'densenet', fixed_weight=False, pretrained_model=cfg.MODEL.PretrainedImageModel)
        if args.dataset_name == 'IUXRAY':
          num_images = 2
          self.get_visual_features = self.forward_iuxray
        else: # 'MIMICCXR'
          num_images = 1
          self.get_visual_features = self.forward_mimiccxr

        # att_feats encoder
        sequential = []
        sequential.append(nn.Linear(self.input_visual_feats * num_images, cfg.MODEL.ATT_FEATS_EMBED_DIM))
        sequential.append(utils.activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
        if cfg.MODEL.ATT_FEATS_NORM == True:
            sequential.append(nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
        if cfg.MODEL.DROPOUT_ATT_EMBED > 0:
            sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED))

        self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None

        self.encoder = Encoder(
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            dropout=cfg.MODEL.BILINEAR.ENCODE_DROPOUT,
            att_type=cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads=cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim=cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DIM,
            att_mid_drop=cfg.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT,
            bifeat_emb_act=cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT,
            bifeat_emb_drop=cfg.MODEL.BILINEAR.ENCODE_BIFEAT_EMB_DROPOUT,
            ff_dropout=cfg.MODEL.BILINEAR.ENCODE_FF_DROPOUT,
            layer_num=cfg.MODEL.BILINEAR.ENCODE_LAYERS)

        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            att_type=cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads=cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim=cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM,
            att_mid_drop=cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT,
            bifeat_emb_act=cfg.MODEL.BILINEAR.BIFEAT_EMB_ACT,
            bifeat_emb_drop=cfg.MODEL.BILINEAR.DECODE_BIFEAT_EMB_DROPOUT,
            ff_dropout=cfg.MODEL.BILINEAR.DECODE_FF_DROPOUT,
            layer_num=cfg.MODEL.BILINEAR.DECODE_LAYERS)
        self.submodel = submodel

    def forward(self, **kwargs):
        # forward entry
        
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        seq = kwargs[cfg.PARAM.INPUT_SENT]
#         att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        att_mask = torch.ones(16,70).to(device)
        att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)
        att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)

        ##############################################
        seq_mask = (seq > 0).type(torch.cuda.IntTensor)
        seq_mask[:, 0] += 1
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        seq_mask = seq_mask.type(torch.cuda.FloatTensor)
        ##############################################
        # print('att_feats.shape b4 pretrain',att_feats.shape)
        # if len(att_feats.shape) == 5:
        #   batch_size = att_feats.shape[0]
        #   img_size = att_feats.shape[1]
        #   channel_size = att_feats.shape[2]
        #   hidden_size1 = att_feats.shape[3]
        #   hidden_size2 = att_feats.shape[4]
        #   att_feats = att_feats.view(batch_size, -1, hidden_size1, hidden_size2)

        # FEED IUXRAY: two images
        # att_feats_0 = self.image_pretrained_models(att_feats[:, 0])
        # att_feats_1 = self.image_pretrained_models(att_feats[:, 1])
        # att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)  # shape (bs, 2048, 7, 7)
        att_feats = self.get_visual_features(att_feats)
        batch_size, feat_size, _ = att_feats.shape
        att_feats = att_feats.reshape(batch_size, feat_size, -1)
        # print('att_feats.shape after pretrain', att_feats.shape)
        att_feats = self.att_embed(att_feats)  # forward entry
 
        gx, encoder_out = self.encoder(att_feats, att_mask)

        decoder_out = self.decoder(gx, seq, encoder_out, att_mask, seq_mask)
        # print('decoder_out.shape',decoder_out.shape) # 4, 41, 761
        # raise Exception('lol')
        return decoder_out

    def forward_iuxray(self, att_feats):
        att_feats, node_feats, fc_feats = self.submodel(att_feats[:,0], att_feats[:,1])
        input_feats = torch.cat((att_feats, node_feats), dim = 1)
        
#         att_feats_0 = self.image_pretrained_models(att_feats[:, 0])
#         att_feats_1 = self.image_pretrained_models(att_feats[:, 1])
#         att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)  # shape (bs, 2048, 7, 7)
        return input_feats

#     def forward_mimiccxr(self, att_feats):
#         att_feats = self.image_pretrained_models(att_feats)
#         return att_feats

    def get_logprobs_state(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        state = kwargs[cfg.PARAM.STATE]
        encoder_out = kwargs[cfg.PARAM.ATT_FEATS]
#         att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        att_mask = torch.ones(16,70).to(device)
        gx = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        # if state is None:
        #     ys = wt.unsqueeze(1)
        # else:
        #     ys = torch.cat([state[0][0], wt.unsqueeze(1)], dim=1)
        ys = torch.zeros(16,1, dtype = int).to(device)
        seq_mask = subsequent_mask(ys.size(1)).to(encoder_out.device).type(torch.cuda.FloatTensor)[:, -1, :].unsqueeze(
            1)
        decoder_out = self.decoder(gx, ys[:, -1].unsqueeze(-1), encoder_out, att_mask, seq_mask, p_att_feats, True).squeeze(1)

        logprobs = F.log_softmax(decoder_out, dim=-1)
        return logprobs, [ys.unsqueeze(0)]

    def _expand_state(self, batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            beam = beam.long()
            s = torch.gather(s.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([batch_size, beam_size] + shape[1:])))

            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    # the beam search code is inspired by https://github.com/aimagelab/meshed-memory-transformer
    def decode_beam(self, **kwargs):
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = torch.ones(16,70).to(device)
#         att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        beam_size = kwargs['BEAM_SIZE']
        batch_size = att_feats.size(0)
        seq_logprob = torch.zeros((batch_size, 1, 1)).cuda()
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1)).cuda()

        # Modified
        att_feats = self.get_visual_features(att_feats)
        batch_size, feat_size, _ = att_feats.shape
        att_feats = att_feats.reshape(batch_size, feat_size, -1)

        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        p_att_feats = self.decoder.precompute(encoder_out)

        state = None
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats

        outputs = []
        self.decoder.init_buffer(batch_size)
        for t in range(cfg.MODEL.SEQ_LEN):
            cur_beam_size = 1 if t == 0 else beam_size

            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            word_logprob, state = self.get_logprobs_state(**kwargs)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != 0).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(batch_size, beam_size, t, candidate_logprob)
            selected_beam = selected_idx / candidate_logprob.shape[-1]
            selected_beam = selected_beam.long()
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            self.decoder.apply_to_states(self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))
            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                                             selected_beam.unsqueeze(-1).expand(batch_size, beam_size,
                                                                                word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            wt = selected_words.squeeze(-1)

            if t == 0:
                encoder_out = utils.expand_tensor(encoder_out, beam_size)
                gx = utils.expand_tensor(gx, beam_size)
                att_mask = utils.expand_tensor(att_mask, beam_size)
                state[0] = state[0].squeeze(0)
                state[0] = utils.expand_tensor(state[0], beam_size)
                state[0] = state[0].unsqueeze(0)

                p_att_feats_tmp = []
                for p_feat in p_att_feats:
                    p_key, p_value2 = p_feat
                    p_key = utils.expand_tensor(p_key, beam_size)
                    p_value2 = utils.expand_tensor(p_value2, beam_size)
                    p_att_feats_tmp.append((p_key, p_value2))

                kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
                kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
                kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
                kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats_tmp

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, cfg.MODEL.SEQ_LEN))

        outputs = outputs.contiguous()[:, 0]
        log_probs = log_probs.contiguous()[:, 0]

        self.decoder.clear_buffer()
        return outputs, log_probs

    def decode(self, **kwargs):
        beam_size = kwargs['BEAM_SIZE']
        greedy_decode = kwargs['GREEDY_DECODE']
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = torch.ones(16,70).to(device)
#         att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]

        batch_size = att_feats.size(0)
        att_feats = self.att_embed(att_feats)
        gx, encoder_out = self.encoder(att_feats, att_mask)
        p_att_feats = self.decoder.precompute(encoder_out)
        self.decoder.init_buffer(batch_size)

        state = None
        sents = Variable(torch.zeros((batch_size, cfg.MODEL.SEQ_LEN), dtype=torch.long).cuda())
        logprobs = Variable(torch.zeros(batch_size, cfg.MODEL.SEQ_LEN).cuda())
        wt = Variable(torch.zeros(batch_size, dtype=torch.long).cuda())
        unfinished = wt.eq(wt)
        kwargs[cfg.PARAM.ATT_FEATS] = encoder_out
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gx
        kwargs[cfg.PARAM.P_ATT_FEATS] = p_att_feats
        for t in range(cfg.MODEL.SEQ_LEN):
            kwargs[cfg.PARAM.WT] = wt
            kwargs[cfg.PARAM.STATE] = state
            logprobs_t, state = self.get_logprobs_state(**kwargs)

            if greedy_decode:
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt > 0)
            wt = wt * unfinished.type_as(wt)
            sents[:, t] = wt
            logprobs[:, t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break
        self.decoder.clear_buffer()
        return sents, logprobs


class Encoder(nn.Module):
    def __init__(
            self,
            embed_dim,
            dropout,
            att_type,
            att_heads,
            att_mid_dim,
            att_mid_drop,
            bifeat_emb_act,
            bifeat_emb_drop,
            ff_dropout,
            layer_num
    ):
        super(Encoder, self).__init__()
        self.att_heads = att_heads
        self.layers = nn.ModuleList([])
        for i in range(layer_num):
            sublayer = EncoderLayer(
                embed_dim=embed_dim,
                dropout=dropout,
                att_type=att_type,
                att_heads=att_heads,
                att_mid_dim=att_mid_dim,
                att_mid_drop=att_mid_drop,
                bifeat_emb_act=bifeat_emb_act,
                bifeat_emb_drop=bifeat_emb_drop,
                ff_dropout=ff_dropout)
            self.layers.append(sublayer)

        self.proj_norm = nn.Sequential(
            nn.Linear(embed_dim * (layer_num + 1), embed_dim),
            torch.nn.LayerNorm(embed_dim))

    def forward(self, x, mask):
        # drop mask
        att_masks = mask
        # if att_masks is None:
        #     att_masks = x.new_ones(x.shape[:2], dtype=torch.long)
        # att_masks = att_masks.unsqueeze(-2)
        gx = (torch.sum(x * att_masks.unsqueeze(-1), 1) / torch.sum(att_masks.unsqueeze(-1), 1))

        gx_arr = [gx]
        for layer in self.layers:
            gx, x = layer(gx, x, att_masks)  # modified
            gx_arr.append(gx)

        gx = torch.cat(gx_arr, dim=-1)
        gx = self.proj_norm(gx)

        return gx, x


class EncoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim,
            dropout,
            att_type,
            att_heads,
            att_mid_dim,
            att_mid_drop,
            bifeat_emb_act,
            bifeat_emb_drop,
            ff_dropout
    ):
        super(EncoderLayer, self).__init__()
        self.encoder_attn = LowRank(
            embed_dim=embed_dim,
            att_type=att_type,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop)
        self.dropout = nn.Dropout(dropout)

        self.bifeat_emb = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            utils.activation(bifeat_emb_act),
            nn.Dropout(bifeat_emb_drop)
        )
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        self.ff_layer = blocks.create(
            'FeedForward',
            embed_dim=embed_dim,
            ffn_embed_dim=embed_dim * 4,
            relu_dropout=ff_dropout,
            dropout=ff_dropout)

    def forward(self, gx, x, mask):
        """
        gx : torch.Size([4, 768])
        x : torch.Size([4, 49, 768])
        mask : torch.Size([4, 49])

        """
        # print('gx.shape {}, x.shape {}, mask.shape {}'.format(gx.shape, x.shape, mask.shape))
        # print('self.encoder_attn',self.encoder_attn)
        gx = self.encoder_attn(
            query=gx,
            key=x,
            mask=mask,
            value1=gx,
            value2=x
        )

        # print('gx.shape',gx.shape) # 4, 768
        # raise Exception('lol')
        gx = self.dropout(gx)

        x_ = torch.cat([gx.unsqueeze(1).expand_as(x), x], dim=-1)
        x = self.bifeat_emb(x_) + x
        x = self.layer_norm(x)

        if self.ff_layer is not None:
            x = self.ff_layer(x)

        return gx, x


class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            dropout,
            att_type,
            att_heads,
            att_mid_dim,
            att_mid_drop,
            bifeat_emb_act,
            bifeat_emb_drop,
            ff_dropout,
            layer_num
    ):
        super(Decoder, self).__init__()
        self.att_heads = att_heads
        self.layers = nn.ModuleList([])
        self.embed_dim = embed_dim
        for i in range(layer_num):
            sublayer = DecoderLayer(
                embed_dim=embed_dim,
                dropout=dropout,
                att_type=att_type,
                att_heads=att_heads,
                att_mid_dim=att_mid_dim,
                att_mid_drop=att_mid_drop,
                bifeat_emb_act=bifeat_emb_act,
                bifeat_emb_drop=bifeat_emb_drop,
                ff_dropout=ff_dropout,
                last_layer=(i == layer_num - 1))
            self.layers.append(sublayer)

        self.dropout = nn.Dropout(cfg.MODEL.DROPOUT_WORD_EMBED)
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEncoding(
            embed_dim, cfg.MODEL.TRANSFORMER.PE_MAX_LEN
        )

        self.layer_norm_word = torch.nn.LayerNorm(embed_dim)
        self.generator = nn.Linear(embed_dim, vocab_size)

        self.wbil1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            utils.activation(cfg.MODEL.BILINEAR.ACT),
            torch.nn.LayerNorm(embed_dim)
        )
        self.wbil2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            utils.activation(cfg.MODEL.BILINEAR.ACT),
            torch.nn.LayerNorm(embed_dim)
        )
        self.wbi_drop = nn.Dropout(cfg.MODEL.BILINEAR.DECODE_DROPOUT)
        self.dropout_lm = nn.Dropout(cfg.MODEL.DROPOUT_LM)

        self.proj_norm = nn.Sequential(
            nn.Linear(embed_dim * (layer_num + 1), 2 * embed_dim),
            nn.GLU(),
            torch.nn.LayerNorm(embed_dim))

        self.clear_buffer()

    def init_buffer(self, batch_size):
        self.seq_len = 0
        self.x = torch.zeros((batch_size, 1, self.embed_dim)).cuda()
        for layer in self.layers:
            layer.init_buffer(batch_size)

    def clear_buffer(self):
        self.seq_len = None
        self.x = None
        for layer in self.layers:
            layer.clear_buffer()

    def apply_to_states(self, fn):
        self.x = fn(self.x)
        for layer in self.layers:
            layer.apply_to_states(fn)

    def precompute(self, encoder_out):
        p_att_feats = []
        for layer in self.layers:
            key, value2 = layer.precompute(encoder_out)
            p_att_feats.append((key, value2))
        return p_att_feats

    def forward(self, gx, prev_output_tokens, encoder_out, att_mask, seq_mask=None, p_att_feats=None, precompute=False):
        # device = torch.device("cuda")
        att_mask = att_mask.unsqueeze(1)

        # embed positions
        seq_len = prev_output_tokens.size(1)
        if self.seq_len is not None:
            seq_len = self.seq_len + seq_len
            self.seq_len = seq_len
            positions = self.embed_positions(seq_len)[:, -1, :].unsqueeze(1)
        else:
            positions = self.embed_positions(seq_len)

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        x = x + positions
        x = self.layer_norm_word(x)
        if self.dropout is not None:
            x = self.dropout(x)

        # decoder layers
        gx = self.wbil1(gx)
        if self.x is None:
            x_gx = (torch.sum(x.unsqueeze(1) * seq_mask.unsqueeze(-1), -2) / torch.sum(seq_mask, -1).unsqueeze(-1))
        else:
            self.x = self.x + x
            x_gx = self.x / seq_len
        x_gx = self.wbil2(x_gx)
        gx = gx.unsqueeze(1)
        gx = gx * x_gx
        gx = self.wbi_drop(gx)

        # print(gx.shape)
        # print(type(gx))
        # print((gx[0, 0, 0].dtype))

        bs, att_nums, features_sizes = gx.shape

        # gx_arr = torch.zeros((len(self.layers), bs, att_nums, features_sizes), dtype = torch.float32, device = device)# .to(torch.device("cuda")) # HARD CODE
        # print(gx_arr.shape)
        # print(gx_arr.shape)
        # gx_arr = gx_arr.to(device)

        gx_arr = [gx]  # .to('cuda')
        # gx_arr =

        for layerid, layer in enumerate(self.layers):
            # print('layerid',layerid)
            if precompute == False:
                p_key = None
                p_value2 = None
            else:
                p_key, p_value2 = p_att_feats[layerid]
            # print('x.shape b4',x.shape)
            # print('precompute',precompute)
            # print('att_mask.shape',att_mask.shape)
            # print('seq_mask.shape',seq_mask.shape)
            gx, x = layer(gx, x, encoder_out, att_mask, seq_mask=seq_mask, p_key=p_key, p_value2=p_value2,
                          precompute=precompute)
            # print('gx.shape',gx.shape) # bs, 41, 768
            # print('x.shape after',x.shape)
            # print(type(gx_arr))
            gx_arr.append(gx)
            # print('test',gx_arr[layerid, :, :, :].shape)
            # print('layerid',layerid)
            # print('gx_arr.shape',gx_arr.shape)
            # print('gx.device',gx.device)
            # print('gx_arr.device',gx_arr.device)
            # gx_arr[layerid] = gx
        # raise Exception('decoder lol')
        gx = torch.cat(gx_arr, dim=-1)

        gx = self.proj_norm(gx)

        gx = self.dropout_lm(gx)
        out = self.generator(gx)
        # raise Exception('decoder lol')
        return out


class DecoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim,
            dropout,
            att_type,
            att_heads,
            att_mid_dim,
            att_mid_drop,
            bifeat_emb_act,
            bifeat_emb_drop,
            ff_dropout,
            last_layer=False
    ):
        super(DecoderLayer, self).__init__()
        self.last_layer = last_layer
        self.word_attn = LowRank(
            embed_dim=embed_dim,
            att_type=att_type,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop)
        self.word_dropout = nn.Dropout(dropout)

        self.cross_att = LowRank(
            embed_dim=embed_dim,
            att_type=att_type,
            att_heads=att_heads,
            att_mid_dim=att_mid_dim,
            att_mid_drop=att_mid_drop)
        self.cross_dropout = nn.Dropout(dropout)
        self.layer_norm_cross = torch.nn.LayerNorm(embed_dim)

        if self.last_layer == False:
            self.bifeat_emb = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                utils.activation(bifeat_emb_act),
                nn.Dropout(bifeat_emb_drop)
            )
            self.layer_norm_x = torch.nn.LayerNorm(embed_dim)

            self.ff_layer = blocks.create(
                'FeedForward',
                embed_dim=embed_dim,
                ffn_embed_dim=embed_dim * 4,
                relu_dropout=ff_dropout,
                dropout=ff_dropout)

        self.layer_norm_gx = torch.nn.LayerNorm(embed_dim)

    def apply_to_states(self, fn):
        self.word_attn.apply_to_states(fn)

    def init_buffer(self, batch_size):
        self.word_attn.init_buffer(batch_size)

    def clear_buffer(self):
        self.word_attn.clear_buffer()

    def precompute(self, encoder_out):
        key, value2 = self.cross_att.precompute(encoder_out, encoder_out)
        return key, value2

    def forward(
            self,
            gx,
            x,
            encoder_out,
            att_mask,
            seq_mask,
            p_key=None,
            p_value2=None,
            precompute=False
    ):
        word_x = x
        residual = x
        x = self.word_attn.forward2(
            query=gx,
            key=x,
            mask=seq_mask,
            value1=gx,
            value2=x)
        x = self.word_dropout(x)

        x = residual + x

        residual = x
        x = self.layer_norm_cross(x)
        x = self.cross_att.forward2(
            query=x,
            key=encoder_out if precompute == False else p_key,
            mask=att_mask,
            value1=x,
            value2=encoder_out if precompute == False else p_value2,
            precompute=precompute)
        x = self.cross_dropout(x)
        gx = residual + x
        gx = self.layer_norm_gx(gx)

        if self.last_layer == False:
            x_ = torch.cat([gx, word_x], dim=-1)
            x = self.bifeat_emb(x_) + word_x
            x = self.layer_norm_x(x)

            if self.ff_layer is not None:
                x = self.ff_layer(x)
        else:
            x = None

        return gx, x
