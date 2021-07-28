import os
import sys
import numpy as np
import torch
import tqdm
import json
import evaluation
import lib.utils as utils
import datasets.data_loader as data_loader
from lib.config import cfg

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from scorer.cider import Cider
device = torch.device("cuda")


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Cider(), 'CIDEr'),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


class Evaler(object):
    def __init__(
            self,
            dataset,
            tokenizer
    ):
        super(Evaler, self).__init__()
        self.tokenizer = tokenizer
        # self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB) # TODO

        # self.eval_ids = np.array(utils.load_ids(eval_ids))
        self.eval_loader = data_loader.load_val(dataset)
        # self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)
        self.evaler = compute_scores

    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        # att_mask_a = torch.ones(16,70).to(device)
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        # kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask_a
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs

    def __call__(self, model, rname):
        model.eval()

        results, golden_sents = {}, {}
        with torch.no_grad():
            for _, (indices, target_seq, gv_feat, att_feats, att_mask) in tqdm.tqdm(enumerate(self.eval_loader)):
                ids = indices
                gv_feat = gv_feat.cuda()
                att_feats = att_feats.cuda()
                att_mask = att_mask.cuda()
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask)
                if kwargs['BEAM_SIZE'] > 1:
                    seq, _ = model.decode_beam(**kwargs)  # modified
                else:
                    seq, _ = model.decode(**kwargs)
                sents = utils.decode_sequence(self.tokenizer.idx2token, seq.data)  # to check
                # sents: [sent (str), ... ]
                gold_sents = utils.decode_sequence(self.tokenizer.idx2token,
                                                   target_seq.data)  # to check target_seq callable


                # for sid, sent in enumerate(sents):
                #     result = {ids[sid]: [sent]}
                #     results.append(result)
                # for sid, sent in enumerate(gold_sents):
                #     g_sents = {ids[sid]: [sent]}
                #     golden_sents.append(g_sents)

                for sid, sent in enumerate(sents):
                    results[ids[sid]] = [sent] # sents: [sent (str), ... ]
                    # results.append(result)
                for sid, sent in enumerate(gold_sents):
                    golden_sents[ids[sid]] = [sent]
                    # golden_sents.append(g_sents)
        # golden_sents : {image_id, [sent (str) ]}
        eval_res = self.evaler(golden_sents, results)

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results,
                  open(os.path.join(result_folder, 'result_' + rname + '.json'), 'w'))  # store the generated sentences

        model.train()
        return eval_res
