import os
import sys
import numpy as np
import pickle
from lib.config import cfg

from scorer.cider import Cider

factory = {
    'CIDEr': Cider,
}

def get_sents(sent):
    words = []
    for word in sent:
        words.append(word)
        if word == 0:
            break
    return words

class Scorer(object):
    def __init__(self):
        super(Scorer, self).__init__()
        self.scorers = []
        self.weights = cfg.SCORER.WEIGHTS
        # self.gts = pickle.load(open(cfg.SCORER.GT_PATH, 'rb'), encoding='bytes')
        for name in cfg.SCORER.TYPES:
            self.scorers.append(factory[name]())

    def __call__(self, gts, res):
        hypo = [get_sents(r) for r in res]
        gts = gts

        rewards_info = {}
        rewards = np.zeros(len(gts))
        for i, scorer in enumerate(self.scorers):
            score, scores = scorer.compute_score(gts, hypo)
            rewards += self.weights[i] * scores
            rewards_info[cfg.SCORER.TYPES[i]] = score
        return rewards, rewards_info